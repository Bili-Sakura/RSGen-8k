#!/usr/bin/env bash
# =============================================================================
# RSGen-8k — Single Model + Technique Benchmark
# =============================================================================
# Production-ready benchmark script for a single model/technique combination.
# All arguments have sensible defaults and can be overridden via environment
# variables or command-line flags.
#
# Usage:
#   # Run with defaults (Text2Earth + MegaFusion → 8K)
#   bash scripts/benchmark_single.sh
#
#   # Override via environment variables
#   MODEL_NAME=diffusionsat TECHNIQUE=multidiffusion bash scripts/benchmark_single.sh
#
#   # Override via command-line flags
#   bash scripts/benchmark_single.sh --model_name geosynth --technique fouriscale
#
# =============================================================================
set -euo pipefail

# ---- Default configuration (override via env vars) --------------------------
MODEL_NAME="${MODEL_NAME:-text2earth}"
TECHNIQUE="${TECHNIQUE:-megafusion}"
PROMPT="${PROMPT:-A high-resolution satellite image of an urban area with dense buildings, roads and vegetation.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/benchmark}"
STAGE_RESOLUTIONS="${STAGE_RESOLUTIONS:-512 1024 2048 4096 8192}"
STAGE_STEPS="${STAGE_STEPS:-40 3 3 2 2}"
ENABLE_RESCHEDULE="${ENABLE_RESCHEDULE:-false}"
ENABLE_DILATION="${ENABLE_DILATION:-false}"
DISABLE_XFORMERS="${DISABLE_XFORMERS:-false}"
DISABLE_VAE_TILING="${DISABLE_VAE_TILING:-false}"
CONFIG_FILE="${CONFIG_FILE:-}"

# ---- Parse command-line overrides -------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)       MODEL_NAME="$2"; shift 2 ;;
        --technique)        TECHNIQUE="$2"; shift 2 ;;
        --prompt)           PROMPT="$2"; shift 2 ;;
        --negative_prompt)  NEGATIVE_PROMPT="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --guidance_scale)   GUIDANCE_SCALE="$2"; shift 2 ;;
        --num_inference_steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --mixed_precision)  MIXED_PRECISION="$2"; shift 2 ;;
        --output_dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --stage_resolutions) STAGE_RESOLUTIONS="$2"; shift 2 ;;
        --stage_steps)      STAGE_STEPS="$2"; shift 2 ;;
        --if_reschedule)    ENABLE_RESCHEDULE=true; shift ;;
        --if_dilation)      ENABLE_DILATION=true; shift ;;
        --no_xformers)      DISABLE_XFORMERS=true; shift ;;
        --no_vae_tiling)    DISABLE_VAE_TILING=true; shift ;;
        --config)           CONFIG_FILE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash $0 [OPTIONS]"
            echo ""
            echo "Options (also settable via env vars):"
            echo "  --model_name NAME       Base model name (default: text2earth)"
            echo "  --technique NAME        Upscaling technique (default: megafusion)"
            echo "  --prompt TEXT           Generation prompt"
            echo "  --negative_prompt TEXT  Negative prompt"
            echo "  --seed INT             Random seed (default: 42)"
            echo "  --guidance_scale FLOAT  CFG scale (default: 7.0)"
            echo "  --num_inference_steps N Total denoising steps (default: 50)"
            echo "  --mixed_precision TYPE  fp16 | bf16 | no (default: bf16)"
            echo "  --output_dir DIR       Output directory (default: ./outputs/benchmark)"
            echo "  --stage_resolutions R  Space-separated resolutions (default: 512 1024 2048 4096 8192)"
            echo "  --stage_steps S        Space-separated steps per stage (default: 40 3 3 2 2)"
            echo "  --if_reschedule        Enable noise rescheduling"
            echo "  --if_dilation          Enable dilation"
            echo "  --no_xformers          Disable xformers"
            echo "  --no_vae_tiling        Disable VAE tiling"
            echo "  --config FILE          YAML config file (overrides other args)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Resolve project root ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Build command ----------------------------------------------------------
RUN_DIR="${OUTPUT_DIR}/${MODEL_NAME}_${TECHNIQUE}_seed${SEED}"
mkdir -p "${RUN_DIR}"

CMD=(
    python "${PROJECT_ROOT}/scripts/generate.py"
    --model_name "${MODEL_NAME}"
    --technique "${TECHNIQUE}"
    --prompt "${PROMPT}"
    --seed "${SEED}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --mixed_precision "${MIXED_PRECISION}"
    --output_dir "${RUN_DIR}"
    --stage_resolutions ${STAGE_RESOLUTIONS}
    --stage_steps ${STAGE_STEPS}
)

if [[ -n "${NEGATIVE_PROMPT}" ]]; then
    CMD+=(--negative_prompt "${NEGATIVE_PROMPT}")
fi
if [[ "${ENABLE_RESCHEDULE}" == "true" ]]; then
    CMD+=(--if_reschedule)
fi
if [[ "${ENABLE_DILATION}" == "true" ]]; then
    CMD+=(--if_dilation)
fi
if [[ "${DISABLE_XFORMERS}" == "true" ]]; then
    CMD+=(--no_xformers)
fi
if [[ "${DISABLE_VAE_TILING}" == "true" ]]; then
    CMD+=(--no_vae_tiling)
fi
if [[ -n "${CONFIG_FILE}" ]]; then
    CMD=(python "${PROJECT_ROOT}/scripts/generate.py" --config "${CONFIG_FILE}" --output_dir "${RUN_DIR}")
fi

# ---- Run --------------------------------------------------------------------
echo "============================================================"
echo "RSGen-8k Benchmark — Single Run"
echo "============================================================"
echo "Model:       ${MODEL_NAME}"
echo "Technique:   ${TECHNIQUE}"
echo "Seed:        ${SEED}"
echo "Resolutions: ${STAGE_RESOLUTIONS}"
echo "Steps:       ${STAGE_STEPS}"
echo "Precision:   ${MIXED_PRECISION}"
echo "Output:      ${RUN_DIR}"
echo "============================================================"

START_TIME=$(date +%s)
"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/benchmark.log"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Completed in ${ELAPSED}s"
echo "Output saved to: ${RUN_DIR}"
echo "============================================================"

# Save run metadata
PROMPT_ESCAPED=$(printf '%s' "${PROMPT}" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")
export PROMPT_ESCAPED
python3 -c "
import json, os
prompt_val = json.loads(os.environ.get('PROMPT_ESCAPED', '\"\"'))
data = {
    'model_name': '${MODEL_NAME}',
    'technique': '${TECHNIQUE}',
    'prompt': prompt_val,
    'seed': ${SEED},
    'guidance_scale': ${GUIDANCE_SCALE},
    'num_inference_steps': ${NUM_INFERENCE_STEPS},
    'mixed_precision': '${MIXED_PRECISION}',
    'stage_resolutions': [int(x) for x in '${STAGE_RESOLUTIONS}'.split()],
    'stage_steps': [int(x) for x in '${STAGE_STEPS}'.split()],
    'elapsed_seconds': ${ELAPSED}
}
with open('${RUN_DIR}/run_metadata.json', 'w') as f:
    json.dump(data, f, indent=2)
"
echo "Metadata saved to: ${RUN_DIR}/run_metadata.json"
