#!/usr/bin/env bash
# =============================================================================
# RSGen-8k — Multi-Seed Reproducibility Benchmark
# =============================================================================
# Runs a model/technique across multiple seeds for variance analysis.
# Useful for measuring generation stability and reproducibility.
#
# Usage:
#   # Run with 5 seeds (default)
#   bash scripts/benchmark_seeds.sh
#
#   # Custom seeds and model
#   SEEDS="0 42 123 456 789" MODEL_NAME=diffusionsat bash scripts/benchmark_seeds.sh
#
# =============================================================================
set -euo pipefail

# ---- Default configuration --------------------------------------------------
MODEL_NAME="${MODEL_NAME:-text2earth}"
TECHNIQUE="${TECHNIQUE:-megafusion}"
SEEDS="${SEEDS:-42 123 456 789 2024}"
PROMPT="${PROMPT:-A high-resolution satellite image of an urban area with dense buildings, roads and vegetation.}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/seed_benchmark}"
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
        --seeds)            SEEDS="$2"; shift 2 ;;
        --prompt)           PROMPT="$2"; shift 2 ;;
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
            echo "  --seeds LIST           Space-separated seeds (default: 42 123 456 789 2024)"
            echo "  --prompt TEXT           Generation prompt"
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

RUN_ROOT="${OUTPUT_DIR}/${MODEL_NAME}_${TECHNIQUE}"
mkdir -p "${RUN_ROOT}"

SUMMARY_CSV="${RUN_ROOT}/seed_benchmark_summary.csv"
echo "model,technique,seed,elapsed_seconds,output_dir" > "${SUMMARY_CSV}"

# ---- Build extra flags ------------------------------------------------------
EXTRA_FLAGS=""
if [[ "${ENABLE_RESCHEDULE}" == "true" ]]; then EXTRA_FLAGS+=" --if_reschedule"; fi
if [[ "${DISABLE_XFORMERS}" == "true" ]]; then EXTRA_FLAGS+=" --no_xformers"; fi
if [[ "${DISABLE_VAE_TILING}" == "true" ]]; then EXTRA_FLAGS+=" --no_vae_tiling"; fi

# ---- Run across seeds -------------------------------------------------------
echo "============================================================"
echo "RSGen-8k — Multi-Seed Benchmark"
echo "============================================================"
echo "Model:     ${MODEL_NAME}"
echo "Technique: ${TECHNIQUE}"
echo "Seeds:     ${SEEDS}"
echo "Output:    ${RUN_ROOT}"
echo "============================================================"

TOTAL_START=$(date +%s)
SEED_INDEX=0
NUM_SEEDS=$(echo "${SEEDS}" | wc -w)

for SEED in ${SEEDS}; do
    SEED_INDEX=$((SEED_INDEX + 1))
    SEED_DIR="${RUN_ROOT}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"

    echo ""
    echo "[${SEED_INDEX}/${NUM_SEEDS}] Seed: ${SEED}"

    START_TIME=$(date +%s)

    python "${PROJECT_ROOT}/scripts/generate.py" \
        --model_name "${MODEL_NAME}" \
        --technique "${TECHNIQUE}" \
        --prompt "${PROMPT}" \
        --seed "${SEED}" \
        --guidance_scale "${GUIDANCE_SCALE}" \
        --num_inference_steps "${NUM_INFERENCE_STEPS}" \
        --mixed_precision "${MIXED_PRECISION}" \
        --output_dir "${SEED_DIR}" \
        --stage_resolutions ${STAGE_RESOLUTIONS} \
        --stage_steps ${STAGE_STEPS} \
        ${EXTRA_FLAGS} \
        2>&1 | tee "${SEED_DIR}/benchmark.log" || {
            echo "WARNING: Run failed for seed ${SEED}"
            echo "${MODEL_NAME},${TECHNIQUE},${SEED},FAILED,${SEED_DIR}" >> "${SUMMARY_CSV}"
            continue
        }

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "${MODEL_NAME},${TECHNIQUE},${SEED},${ELAPSED},${SEED_DIR}" >> "${SUMMARY_CSV}"
    echo "  → Completed in ${ELAPSED}s"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================================"
echo "Multi-seed benchmark completed in ${TOTAL_ELAPSED}s"
echo "Summary: ${SUMMARY_CSV}"
echo "============================================================"
