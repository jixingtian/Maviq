import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WORKSPACE = Path(os.getenv("MAVIQ_WORKSPACE", Path(__file__).resolve().parent)).expanduser()
RESULTS_DIR = Path(os.getenv("MAVIQ_RESULTS_DIR", WORKSPACE / "results")).expanduser()
DATA_DIR = Path(os.getenv("MAVIQ_DATA_DIR", WORKSPACE / "data")).expanduser()
LOGS_DIR = Path(os.getenv("MAVIQ_LOGS_DIR", WORKSPACE / "logs")).expanduser()

for directory in (RESULTS_DIR, DATA_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL_BASE_ENV = os.getenv("MAVIQ_MODEL_BASE")
MODEL_BASE = Path(MODEL_BASE_ENV).expanduser() if MODEL_BASE_ENV else None


def model_path(env_name: str, hf_id: str) -> str:
    override = os.getenv(env_name)
    if override:
        return override
    if MODEL_BASE is not None:
        return str(MODEL_BASE / hf_id)
    return hf_id

PLANNER_MODEL = os.getenv(
    "MAVIQ_PLANNER_MODEL",
    model_path("MAVIQ_QWEN3VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
)
PLANNER_BACKEND = os.getenv("MAVIQ_PLANNER_BACKEND", "gpt4o")
PLANNER_OPENAI_MODEL = os.getenv("MAVIQ_PLANNER_OPENAI_MODEL", "gpt-4o-2024-11-20")
PLANNER_DEVICE = os.getenv("MAVIQ_PLANNER_DEVICE", "cuda:0")
PLANNER_TEMP = float(os.getenv("MAVIQ_PLANNER_TEMP", "1.0"))
PLANNER_MAX_TOKENS = int(os.getenv("MAVIQ_PLANNER_MAX_TOKENS", "1024"))
PLANNER_MAX_PARSE_RETRIES = int(os.getenv("MAVIQ_PLANNER_MAX_PARSE_RETRIES", "3"))

SD_MODEL_ID = os.getenv(
    "MAVIQ_SD_MODEL_ID",
    model_path("MAVIQ_SD_MODEL_ID", "stabilityai/stable-diffusion-3.5-large"),
)
SD_DEVICE = os.getenv("MAVIQ_SD_DEVICE", "cuda:0")
SD_INFERENCE_STEPS = int(os.getenv("MAVIQ_SD_INFERENCE_STEPS", "30"))
SD_GUIDANCE_SCALE = float(os.getenv("MAVIQ_SD_GUIDANCE_SCALE", "7.0"))
SD_HEIGHT = int(os.getenv("MAVIQ_SD_HEIGHT", "512"))
SD_WIDTH = int(os.getenv("MAVIQ_SD_WIDTH", "512"))
MAVIQ_IMAGE_NEGATIVE_PROMPT = (
    "text-heavy image, readable paragraphs, large readable labels, captions, subtitles, watermark, logo, "
    "infographic, flowchart, abstract diagram, presentation slide, wireframe UI, pure screenshot, "
    "spreadsheet, poster wall, oversized table, dense grid, blank image, black image"
)

JUDGE_OPENAI_MODEL = os.getenv("MAVIQ_JUDGE_OPENAI_MODEL", "gpt-5.1-2025-11-13")
JUDGE_MODEL_ID = os.getenv(
    "MAVIQ_LLAMA_GUARD_MODEL",
    model_path("MAVIQ_LLAMA_GUARD_MODEL", "meta-llama/Llama-Guard-3-8B"),
)
JUDGE_DEVICE = os.getenv("MAVIQ_JUDGE_DEVICE", "cuda:0")
JUDGE_MAX_NEW_TOKENS = int(os.getenv("MAVIQ_JUDGE_MAX_NEW_TOKENS", "20"))

TARGET_MODELS = {
    "qwen3vl": {
        "model_id": os.getenv(
            "MAVIQ_QWEN3VL_MODEL",
            model_path("MAVIQ_QWEN3VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        ),
        "device": os.getenv("MAVIQ_QWEN3VL_DEVICE", "cuda:0"),
        "dtype": "bfloat16",
        "attn": "sdpa",
        "max_new_tokens": int(os.getenv("MAVIQ_TARGET_MAX_NEW_TOKENS", "4096")),
    },
    "llava_ov": {
        "model_id": os.getenv(
            "MAVIQ_LLAVA_OV_MODEL",
            model_path("MAVIQ_LLAVA_OV_MODEL", "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"),
        ),
        "device": os.getenv("MAVIQ_LLAVA_OV_DEVICE", "cuda:0"),
        "dtype": "bfloat16",
        "max_new_tokens": int(os.getenv("MAVIQ_TARGET_MAX_NEW_TOKENS", "4096")),
    },
    "internvl25": {
        "model_id": os.getenv(
            "MAVIQ_INTERNVL25_MODEL",
            model_path("MAVIQ_INTERNVL25_MODEL", "OpenGVLab/InternVL2_5-8B"),
        ),
        "device": os.getenv("MAVIQ_INTERNVL25_DEVICE", "cuda:0"),
        "dtype": "bfloat16",
        "max_new_tokens": int(os.getenv("MAVIQ_TARGET_MAX_NEW_TOKENS", "4096")),
    },
    "gpt4o": {
        "model_id": os.getenv("MAVIQ_GPT4O_MODEL", "gpt-4o-2024-11-20"),
        "api": "openai",
        "max_tokens": int(os.getenv("MAVIQ_TARGET_MAX_TOKENS", "4096")),
    },
    "gpt51": {
        "model_id": os.getenv("MAVIQ_GPT51_MODEL", "gpt-5.1-2025-11-13"),
        "api": "openai",
        "max_tokens": int(os.getenv("MAVIQ_TARGET_MAX_TOKENS", "4096")),
    },
    "gemini31pro": {
        "model_id": os.getenv("MAVIQ_GEMINI31PRO_MODEL", "gemini-3.1-pro-preview"),
        "api": "gemini",
        "max_tokens": int(os.getenv("MAVIQ_TARGET_MAX_TOKENS", "4096")),
    },
}
PRIMARY_TARGET = os.getenv("MAVIQ_PRIMARY_TARGET", "llava_ov")

MAX_TURNS = int(os.getenv("MAVIQ_MAX_TURNS", "5"))

DATASETS = {
    "mm_safety": {
        "path": Path(os.getenv("MAVIQ_MM_SAFETY_PATH", DATA_DIR / "MM-SafetyBench")).expanduser(),
        "scenarios": 13,
        "samples_per_scenario": None,
        "total": 1680,
    },
    "jailbreakv": {
        "path": Path(os.getenv("MAVIQ_JAILBREAKV_PATH", DATA_DIR / "JailBreakV-28K")).expanduser(),
        "csv_name": os.getenv("MAVIQ_JAILBREAKV_CSV", "mini_JailBreakV_28K.csv"),
        "sample_size": None,
    },
}
PRIMARY_DATASET = os.getenv("MAVIQ_PRIMARY_DATASET", "jailbreakv")
