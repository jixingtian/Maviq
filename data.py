import hashlib
import random
from pathlib import Path
from typing import List, Optional
from loguru import logger

from config import DATASETS, PRIMARY_DATASET


class AttackSample:
    def __init__(self, query_id: str, query: str, scenario: str,
                 ref_image_path: Optional[str] = None, source: str = "",
                 instruction: Optional[str] = None):
        self.query_id       = query_id
        self.query          = query
        self.scenario       = scenario
        self.ref_image_path = ref_image_path
        self.source         = source
        self.instruction    = instruction

    def __repr__(self):
        return f"AttackSample(id={self.query_id}, scenario={self.scenario})"

MM_SAFETY_SCENARIOS = [
    "01_Illegal_Activity", "02_HateSpeech", "03_Malware_Generation",
    "04_Physical_Harm", "05_Economic_Harm", "06_Fraud",
    "07_Sex", "08_Political_Lobbying", "09_Privacy_Violence",
    "10_Legal_Opinion", "11_Financial_Advice", "12_Health_Consultation",
    "13_Gov_Decision",
]

def load_mm_safety(data_dir: Path, samples_per_scenario: Optional[int] = None,
                   image_type: str = "Text_only", seed: int = 42) -> List[AttackSample]:
    import pandas as pd
    import io
    random.seed(seed)
    samples = []

    parquet_name = {
        "Text_only": "Text_only.parquet",
        "SD": "SD.parquet",
        "TYPO": "TYPO.parquet",
        "SD_TYPO": "SD_TYPO.parquet",
    }.get(image_type, "Text_only.parquet")

    root = data_dir / "data" if (data_dir / "data").is_dir() else data_dir
    scenario_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not scenario_dirs:
        logger.warning(f"MM-SafetyBench: no scenario dirs found in {data_dir}")
        return []

    def _stable_suffix(row, query: str, raw_img) -> str:
        row_id = row.get("id")
        if row_id is not None and str(row_id).strip():
            return str(row_id).strip().replace("/", "_")

        payload = [scenario, image_type, query]
        if isinstance(raw_img, bytes):
            payload.append(raw_img.hex())
        elif isinstance(raw_img, dict) and "bytes" in raw_img:
            payload.append(bytes(raw_img["bytes"]).hex())
        digest = hashlib.sha1("||".join(payload).encode("utf-8")).hexdigest()
        return digest[:16]

    for scenario_dir in scenario_dirs:
        parquet_file = scenario_dir / parquet_name
        if not parquet_file.exists():
            parquets = list(scenario_dir.glob("*.parquet"))
            parquet_file = parquets[0] if parquets else None
        if parquet_file is None:
            logger.warning(f"No parquet in {scenario_dir}, skipping")
            continue

        df = pd.read_parquet(parquet_file)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        if samples_per_scenario is not None:
            df = df.head(samples_per_scenario)

        scenario = scenario_dir.name
        for i, row in df.iterrows():
            query = str(row.get("question", "") or "")
            instruction = str(
                row.get("instruction")
                or row.get("instructions")
                or row.get("Instruction")
                or row.get("prompt")
                or query
            )
            ref_img = None
            raw_img = row.get("image")
            sample_suffix = _stable_suffix(row, query, raw_img)
            if raw_img is not None:
                if isinstance(raw_img, bytes):
                    img_path = data_dir / f"_cache_{scenario}_{image_type}_{sample_suffix}.png"
                    if not img_path.exists():
                        from PIL import Image as PILImage
                        PILImage.open(io.BytesIO(raw_img)).save(img_path)
                    ref_img = str(img_path)
                elif isinstance(raw_img, dict) and "bytes" in raw_img:
                    img_path = data_dir / f"_cache_{scenario}_{image_type}_{sample_suffix}.png"
                    if not img_path.exists():
                        from PIL import Image as PILImage
                        PILImage.open(io.BytesIO(raw_img["bytes"])).save(img_path)
                    ref_img = str(img_path)

            samples.append(AttackSample(
                query_id=f"mmsafety_{scenario}_{sample_suffix}",
                query=query,
                scenario=scenario,
                ref_image_path=ref_img,
                source="MM-SafetyBench",
                instruction=instruction,
            ))

    logger.info(f"Loaded {len(samples)} samples from MM-SafetyBench")
    return samples

def load_jailbreakv(data_dir: Path, sample_size: Optional[int] = None,
                    seed: int = 42,
                    include_reference_images: bool = False,
                    csv_name: Optional[str] = None) -> List[AttackSample]:
    import pandas as pd
    random.seed(seed)

    csv_candidates = []
    if csv_name:
        csv_candidates.append(data_dir / "JailBreakV_28K" / csv_name)
    csv_candidates.append(data_dir / "JailBreakV_28K" / "JailBreakV_28K.csv")

    csv_file = None
    for candidate in csv_candidates:
        if candidate.exists():
            csv_file = candidate
            break
    if csv_file is None:
        csvs = sorted(data_dir.glob("**/*.csv"))
        csv_file = csvs[0] if csvs else None
    if csv_file is None:
        logger.warning(f"JailBreakV-28K: no CSV found in {data_dir}")
        return []

    df = pd.read_csv(csv_file)
    query_col = "redteam_query" if "redteam_query" in df.columns else "jailbreak_query"
    scenario_col = "policy" if "policy" in df.columns else "format"

    def _stable_suffix(row, scenario: str, query: str, image_path) -> str:
        row_id = row.get("id")
        if row_id is not None and str(row_id).strip():
            return str(row_id).strip().replace("/", "_")

        payload = [
            scenario,
            query,
            str(image_path or ""),
        ]
        digest = hashlib.sha1("||".join(payload).encode("utf-8")).hexdigest()
        return digest[:16]

    samples = []
    for _, row in df.iterrows():
        sc = str(row.get(scenario_col, "unknown"))
        query = str(row.get(query_col, "") or "")
        img_path = row.get("image_path")
        sample_suffix = _stable_suffix(row, sc, query, img_path)
        ref_img = None
        if include_reference_images and img_path and isinstance(img_path, str):
            ref_img = str(data_dir / "JailBreakV_28K" / img_path)
        samples.append(AttackSample(
            query_id=f"jbv_{sc}_{sample_suffix}",
            query=query,
            scenario=sc,
            ref_image_path=ref_img,
            source="JailBreakV-28K",
        ))

    random.shuffle(samples)
    if sample_size is not None:
        samples = samples[:sample_size]
    logger.info(f"Loaded {len(samples)} samples from JailBreakV-28K")
    return samples
def load_dataset(name: str = PRIMARY_DATASET, include_reference_images: bool = False) -> List[AttackSample]:
    cfg = DATASETS[name]
    path = cfg["path"]

    if name == "mm_safety":
        image_type = "SD" if include_reference_images else "Text_only"
        return load_mm_safety(path, cfg.get("samples_per_scenario"), image_type=image_type)
    elif name == "jailbreakv":
        return load_jailbreakv(
            path,
            cfg.get("sample_size"),
            include_reference_images=include_reference_images,
            csv_name=cfg.get("csv_name"),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
