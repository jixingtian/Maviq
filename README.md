# MAViQ

This repository contains the main MAViQ implementation only. Baseline attack
methods, experiment logs, generated images, and result files are intentionally
excluded.

## Environment

Create an environment with Python 3.10 or newer, then install the dependencies:

```bash
conda create -n maviq python=3.10 -y
conda activate maviq
pip install -r requirements.txt
```

The listed PyTorch wheels are CUDA 12.1 builds. If your machine uses a different
CUDA version, install the matching PyTorch build first, then install the
remaining packages from `requirements.txt`.

## API and Model Paths

MAViQ can use OpenAI API models for the planner and judge, Gemini's official
OpenAI-compatible endpoint for the Gemini target, and local HuggingFace
checkpoints for open-source targets and Stable Diffusion 3.5 Large. Start from
the example environment file:

```bash
cp .env.example .env
```

Then fill in the fields you need. At minimum, API-based runs require:

```bash
export OPENAI_API_KEY=...
export OPENAI_API_BASE=...  # optional, only for OpenAI-compatible OpenAI proxies
```

To run `--target gemini31pro`, set a Gemini API key. The default Gemini base URL
uses Google's official OpenAI-compatible endpoint.

```bash
export GEMINI_API_KEY=...
export GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai/
```

By default, model identifiers use public HuggingFace IDs. If checkpoints are
mirrored locally, set `MAVIQ_MODEL_BASE` to the local root or override individual
paths with environment variables such as `MAVIQ_QWEN3VL_MODEL`,
`MAVIQ_LLAVA_OV_MODEL`, `MAVIQ_INTERNVL25_MODEL`, and `MAVIQ_SD_MODEL_ID`.

## Data

Place datasets under `data/` or set dataset paths explicitly:

```bash
export MAVIQ_MM_SAFETY_PATH=/path/to/MM-SafetyBench
export MAVIQ_JAILBREAKV_PATH=/path/to/JailBreakV-28K
```

The default JailbreakV CSV name is `mini_JailBreakV_28K.csv`; override it with
`MAVIQ_JAILBREAKV_CSV` if needed.

Expected layouts are:

```text
data/
  MM-SafetyBench/
    data/<scenario_name>/Text_only.parquet
  JailBreakV-28K/
    JailBreakV_28K/mini_JailBreakV_28K.csv
```

Supported dataset names are `jailbreakv` and `mm_safety`.

## Run MAViQ

The default paper-style configuration uses GPT-4o for attacker-side auxiliary
modules, GPT-5.1 as the reported judge, Stable Diffusion 3.5 Large for image
generation, and a maximum turn budget of 5.

Run on JailbreakV-mini with GPT-4o as the target:

```bash
python evaluate.py \
  --method maviq \
  --target gpt4o \
  --dataset jailbreakv \
  --planner-backend gpt4o \
  --sd-device cuda:0 \
  --target-device cuda:0 \
  --judge-device cuda:0
```

Run with a local open-source target:

```bash
python evaluate.py \
  --method maviq \
  --target qwen3vl \
  --dataset jailbreakv \
  --planner-backend gpt4o \
  --sd-device cuda:0 \
  --target-device cuda:1 \
  --judge-device cuda:0
```

Use Qwen3VL as the attacker-side planner/router instead of GPT-4o:

```bash
python evaluate.py \
  --method maviq \
  --target gpt4o \
  --dataset jailbreakv \
  --planner-backend qwen3vl \
  --planner-device cuda:1 \
  --sd-device cuda:0 \
  --target-device cuda:0 \
  --judge-device cuda:0
```

For a quick syntax and data-path check, run only five samples:

```bash
python evaluate.py \
  --method maviq \
  --target gpt4o \
  --dataset jailbreakv \
  --smoke-test
```

By default, results resume from an existing output file. Add `--no-resume` to
start a fresh run.

## Main Arguments

`--target` selects the target MLLM. Supported values are `qwen3vl`, `llava_ov`,
`internvl25`, `gpt4o`, `gpt51`, and `gemini31pro`.

`--dataset` selects the evaluation dataset. Supported values are `jailbreakv`
and `mm_safety`.

`--planner-backend` selects attacker-side auxiliary modules. Use `gpt4o` for the
default paper setting or `qwen3vl` for a local planner/router setting.

`--judge` selects the reported ASR judge. The default is `gpt51`; `gpt4o` and
`llamaguard` are also available.

`--max-turns` sets the turn budget. The default is 5.

`--scenarios` filters a dataset by scenario names, for example:

```bash
python evaluate.py \
  --method maviq \
  --target gpt4o \
  --dataset mm_safety \
  --scenarios 03_Malware_Generation 06_Fraud
```

## Outputs

Results are written to `results/` by default:

```text
results/<target>_maviq_<dataset>.json
results/<target>_metrics_summary.json
logs/<target>_maviq_<timestamp>.log
```

Override output locations with:

```bash
export MAVIQ_RESULTS_DIR=/path/to/results
export MAVIQ_LOGS_DIR=/path/to/logs
```

Each result item includes the original query, scenario, per-turn questions,
image paths, target responses, judge score, success label, and scenario-memory
metadata used by MAViQ.
