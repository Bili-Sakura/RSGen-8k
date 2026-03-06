#!/usr/bin/env bash
# =============================================================================
# RSGen-8k — Batch Generation from XLRS-Bench Dataset
# =============================================================================
# Generates images from XLRS-Bench prompts for a given model/technique pair.
# Useful for building evaluation datasets.
#
# Usage:
#   # Generate 50 images with defaults
#   bash scripts/generate_dataset.sh
#
#   # Override via env vars or flags
#   MODEL_NAME=geosynth NUM_PROMPTS=100 bash scripts/generate_dataset.sh
#   bash scripts/generate_dataset.sh --model_name diffusionsat --num_prompts 20
#
# =============================================================================
set -euo pipefail

# ---- Default configuration --------------------------------------------------
MODEL_NAME="${MODEL_NAME:-text2earth}"
TECHNIQUE="${TECHNIQUE:-megafusion}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
DATASET_SEED="${DATASET_SEED:-42}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/dataset}"
STAGE_RESOLUTIONS="${STAGE_RESOLUTIONS:-512 1024 2048 4096 8192}"
STAGE_STEPS="${STAGE_STEPS:-40 3 3 2 2}"
ENABLE_RESCHEDULE="${ENABLE_RESCHEDULE:-false}"
DISABLE_XFORMERS="${DISABLE_XFORMERS:-false}"
DISABLE_VAE_TILING="${DISABLE_VAE_TILING:-false}"

# ---- Parse command-line overrides -------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)       MODEL_NAME="$2"; shift 2 ;;
        --technique)        TECHNIQUE="$2"; shift 2 ;;
        --num_prompts)      NUM_PROMPTS="$2"; shift 2 ;;
        --dataset_seed)     DATASET_SEED="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --guidance_scale)   GUIDANCE_SCALE="$2"; shift 2 ;;
        --num_inference_steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --mixed_precision)  MIXED_PRECISION="$2"; shift 2 ;;
        --output_dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --stage_resolutions) STAGE_RESOLUTIONS="$2"; shift 2 ;;
        --stage_steps)      STAGE_STEPS="$2"; shift 2 ;;
        --if_reschedule)    ENABLE_RESCHEDULE=true; shift ;;
        --no_xformers)      DISABLE_XFORMERS=true; shift ;;
        --no_vae_tiling)    DISABLE_VAE_TILING=true; shift ;;
        --help|-h)
            echo "Usage: bash $0 [OPTIONS]"
            echo ""
            echo "Options (also settable via env vars):"
            echo "  --model_name NAME       Base model (default: text2earth)"
            echo "  --technique NAME        Technique (default: megafusion)"
            echo "  --num_prompts N         Prompts to generate (default: 50)"
            echo "  --dataset_seed INT      Seed for prompt sampling (default: 42)"
            echo "  --seed INT             Generation seed (default: 42)"
            echo "  --guidance_scale FLOAT  CFG scale (default: 7.0)"
            echo "  --num_inference_steps N Steps (default: 50)"
            echo "  --mixed_precision TYPE  fp16 | bf16 | no (default: bf16)"
            echo "  --output_dir DIR       Output directory"
            echo "  --stage_resolutions R  Space-separated resolutions"
            echo "  --stage_steps S        Space-separated steps per stage"
            echo "  --if_reschedule        Enable noise rescheduling"
            echo "  --no_xformers          Disable xformers"
            echo "  --no_vae_tiling        Disable VAE tiling"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Resolve project root ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="${OUTPUT_DIR}/${MODEL_NAME}_${TECHNIQUE}_n${NUM_PROMPTS}"
mkdir -p "${RUN_DIR}"

# ---- Build extra flags ------------------------------------------------------
EXTRA_FLAGS=""
if [[ "${ENABLE_RESCHEDULE}" == "true" ]]; then EXTRA_FLAGS+=" --if_reschedule"; fi
if [[ "${DISABLE_XFORMERS}" == "true" ]]; then EXTRA_FLAGS+=" --no_xformers"; fi
if [[ "${DISABLE_VAE_TILING}" == "true" ]]; then EXTRA_FLAGS+=" --no_vae_tiling"; fi

# ---- Run --------------------------------------------------------------------
echo "============================================================"
echo "RSGen-8k — Dataset Generation from XLRS-Bench"
echo "============================================================"
echo "Model:       ${MODEL_NAME}"
echo "Technique:   ${TECHNIQUE}"
echo "Prompts:     ${NUM_PROMPTS}"
echo "Seed:        ${SEED}"
echo "Resolutions: ${STAGE_RESOLUTIONS}"
echo "Precision:   ${MIXED_PRECISION}"
echo "Output:      ${RUN_DIR}"
echo "============================================================"

START_TIME=$(date +%s)

python "${PROJECT_ROOT}/scripts/generate.py" \
    --model_name "${MODEL_NAME}" \
    --technique "${TECHNIQUE}" \
    --from_dataset \
    --num_prompts "${NUM_PROMPTS}" \
    --dataset_seed "${DATASET_SEED}" \
    --seed "${SEED}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --output_dir "${RUN_DIR}" \
    --stage_resolutions ${STAGE_RESOLUTIONS} \
    --stage_steps ${STAGE_STEPS} \
    ${EXTRA_FLAGS} \
    2>&1 | tee "${RUN_DIR}/generation.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Generated ${NUM_PROMPTS} images in ${ELAPSED}s"
echo "Output:  ${RUN_DIR}"
echo "============================================================"

# Save generation metadata
python3 -c "
import json
data = {
    'model_name': '${MODEL_NAME}',
    'technique': '${TECHNIQUE}',
    'num_prompts': ${NUM_PROMPTS},
    'dataset_seed': ${DATASET_SEED},
    'generation_seed': ${SEED},
    'guidance_scale': ${GUIDANCE_SCALE},
    'num_inference_steps': ${NUM_INFERENCE_STEPS},
    'mixed_precision': '${MIXED_PRECISION}',
    'stage_resolutions': [int(x) for x in '${STAGE_RESOLUTIONS}'.split()],
    'stage_steps': [int(x) for x in '${STAGE_STEPS}'.split()],
    'elapsed_seconds': ${ELAPSED}
}
with open('${RUN_DIR}/generation_metadata.json', 'w') as f:
    json.dump(data, f, indent=2)
"
