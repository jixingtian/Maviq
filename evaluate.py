import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List

from loguru import logger
from tqdm import tqdm

from config import (
    JUDGE_DEVICE,
    LOGS_DIR,
    MAX_TURNS,
    PLANNER_BACKEND,
    PLANNER_DEVICE,
    PRIMARY_DATASET,
    PRIMARY_TARGET,
    RESULTS_DIR,
    SD_DEVICE,
    SD_MODEL_ID,
)
from data import AttackSample, load_dataset
from judge import GPT51Judge, LlamaGuardJudge


def _get_attack_cost(result: dict) -> float:
    if "queries_used" in result:
        return float(result["queries_used"])
    return float(result.get("turns", 0))


def compute_metrics(results: list) -> dict:
    total = len(results)
    if total == 0:
        return {"asr": 0.0, "atc": 0.0, "n_success": 0, "n_total": 0}

    successes = [r for r in results if r.get("success")]
    n_success = len(successes)
    asr = n_success / total
    atc = sum(_get_attack_cost(r) for r in successes) / n_success if successes else 0.0

    return {
        "asr": round(asr, 4),
        "atc": round(atc, 2),
        "n_success": n_success,
        "n_total": total,
    }


def atomic_write_json(output_path: Path, payload) -> None:
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def run_experiment(
    method_name: str,
    attack_obj,
    samples: List[AttackSample],
    output_path: Path,
    resume: bool = True,
) -> list:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = set()
    all_results = []
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            all_results = json.load(f)
        completed_ids = {r["query_id"] for r in all_results}
        logger.info(f"Resuming {method_name}: {len(completed_ids)} already done")

    pending = [sample for sample in samples if sample.query_id not in completed_ids]
    logger.info(f"Running {method_name} on {len(pending)} samples")

    for sample in tqdm(pending, desc=method_name):
        try:
            result = attack_obj.attack(sample)
        except Exception as exc:
            logger.error(f"Attack error on {sample.query_id}: {exc}")
            result = {
                "query_id": sample.query_id,
                "query": sample.query,
                "scenario": sample.scenario,
                "method": method_name,
                "success": False,
                "turns": 0,
                "queries_used": 0,
                "response": "",
                "error": str(exc),
            }
        if "queries_used" not in result:
            result["queries_used"] = result.get("turns", 0)
        all_results.append(result)
        atomic_write_json(output_path, all_results)

    metrics = compute_metrics(all_results)
    logger.info(f"{method_name} | ASR={metrics['asr']:.2%} ATC={metrics['atc']:.2f}")
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="MAViQ evaluation")
    parser.add_argument(
        "--target",
        default=PRIMARY_TARGET,
        choices=["qwen3vl", "llava_ov", "internvl25", "gpt4o", "gpt51", "gemini31pro"],
    )
    parser.add_argument("--method", default="maviq", choices=["maviq"])
    parser.add_argument("--dataset", default=PRIMARY_DATASET, choices=["mm_safety", "jailbreakv"])
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--sd-device", default=SD_DEVICE)
    parser.add_argument("--judge-device", default=JUDGE_DEVICE)
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--planner-device", default=PLANNER_DEVICE)
    parser.add_argument(
        "--planner-backend",
        default=PLANNER_BACKEND,
        choices=["qwen3vl", "gpt4o"],
        help="Planner/router/refusal backend for MAViQ",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Filter by scenario names.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run on 5 samples only.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--judge", default="gpt51", choices=["gpt51", "gpt4o", "llamaguard"])
    return parser.parse_args()


def build_scenario_output_suffix(scenarios: list[str] | None) -> str:
    if not scenarios:
        return ""
    normalized = sorted(scenarios)
    digest = hashlib.sha1("||".join(normalized).encode("utf-8")).hexdigest()[:8]
    return f"_top{len(normalized)}_{digest}"


def build_max_turns_output_suffix(max_turns: int, explicit_max_turns: bool) -> str:
    return f"_mt{max_turns}" if explicit_max_turns else ""


def build_smoke_test_output_suffix(smoke_test: bool) -> str:
    return "_smoke" if smoke_test else ""


def load_sd_pipeline(sd_device: str):
    from diffusers import StableDiffusion3Pipeline
    import torch

    if not sd_device.startswith("cuda:"):
        raise ValueError(f"Unsupported sd_device: {sd_device}")

    logger.info(f"Loading SD 3.5 Large from {SD_MODEL_ID} on {sd_device}")
    sd_pipe = StableDiffusion3Pipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float16,
    ).to(sd_device)
    sd_pipe.set_progress_bar_config(disable=True)
    return sd_pipe


def build_judge(name: str, judge_device: str):
    if name == "llamaguard":
        return LlamaGuardJudge(device=judge_device)
    if name == "gpt4o":
        return GPT51Judge(model="gpt-4o-2024-11-20")
    return GPT51Judge()


def build_maviq(args, target, judge, sd_pipe):
    from attack.maviq import MAViQ, Planner, RouterAgent

    planner = Planner(device=args.planner_device, backend=args.planner_backend)
    if args.planner_backend == "qwen3vl":
        router = RouterAgent(
            model_instance=planner.agent.model,
            processor=planner.agent.processor,
            backend=args.planner_backend,
        )
        router.agent.process_vision_info = planner.agent.process_vision_info
    else:
        router = RouterAgent(backend=args.planner_backend)

    return MAViQ(
        target_model=target,
        judge=judge,
        planner=planner,
        router=router,
        sd_pipeline=sd_pipe,
        image_mode="every_turn",
        max_turns=args.max_turns,
        planner_backend=args.planner_backend,
    )


def main():
    args = parse_args()
    explicit_max_turns = "--max-turns" in sys.argv

    logger.add(LOGS_DIR / f"{args.target}_maviq_{int(time.time())}.log")
    logger.info(f"Using explicit devices | target={args.target_device}, planner={args.planner_device}, sd={args.sd_device}, judge={args.judge_device}")

    judge = build_judge(args.judge, args.judge_device)

    from attack.models import load_target_model

    target = load_target_model(args.target, device=args.target_device)
    sd_pipe = load_sd_pipeline(args.sd_device)

    samples = load_dataset(args.dataset, include_reference_images=False)
    logger.info(f"Loaded dataset={args.dataset} samples={len(samples)}")
    if args.scenarios:
        samples = [sample for sample in samples if sample.scenario in args.scenarios]
        logger.info(f"Filtered to scenarios {args.scenarios}: {len(samples)} samples")
    if args.smoke_test:
        samples = samples[:5]
        logger.info("Smoke test mode: 5 samples only")

    output_suffix = (
        build_scenario_output_suffix(args.scenarios)
        + build_max_turns_output_suffix(args.max_turns, explicit_max_turns)
        + build_smoke_test_output_suffix(args.smoke_test)
    )
    output_path = RESULTS_DIR / f"{args.target}_maviq{output_suffix}_{args.dataset}.json"

    attack = build_maviq(args, target, judge, sd_pipe)
    results = run_experiment(
        method_name="maviq",
        attack_obj=attack,
        samples=samples,
        output_path=output_path,
        resume=not args.no_resume,
    )

    metrics = compute_metrics(results)
    summary_path = RESULTS_DIR / f"{args.target}_metrics_summary.json"
    summary_record = {
        "dataset": args.dataset,
        "smoke_test": args.smoke_test,
        "scenarios": sorted(args.scenarios) if args.scenarios else None,
        "methods": ["maviq"],
        "max_turns": args.max_turns,
        "metrics": {"maviq": metrics},
    }

    summary_history = []
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            existing_summary = json.load(f)
        if isinstance(existing_summary, list):
            summary_history = existing_summary
        elif isinstance(existing_summary, dict):
            summary_history = [existing_summary]

    summary_history.append(summary_record)
    atomic_write_json(summary_path, summary_history)

    print(f"\n{'Method':<20} {'ASR':>8} {'ATC':>8}")
    print("-" * 36)
    print(f"{'maviq':<20} {metrics['asr']:>8.2%} {metrics['atc']:>8.2f}")


if __name__ == "__main__":
    main()
