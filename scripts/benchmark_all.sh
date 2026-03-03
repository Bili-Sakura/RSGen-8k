#!/usr/bin/env bash
# =============================================================================
# RSGen-8k — Full Benchmark: All Models × All Techniques
# =============================================================================
# Runs generation for every model/technique combination (or a user-specified
# subset) and collects timing + output metadata into a summary CSV.
#
# Usage:
#   # Run all 3 models × 6 techniques = 18 combinations
#   bash scripts/benchmark_all.sh
#
#   # Restrict to specific models/techniques
#   MODELS="text2earth diffusionsat" TECHNIQUES="megafusion multidiffusion" \
#       bash scripts/benchmark_all.sh
#
#   # Adjust common parameters
#   SEED=123 MIXED_PRECISION=bf16 bash scripts/benchmark_all.sh
#
# =============================================================================
set -euo pipefail

# ---- Default configuration --------------------------------------------------
MODELS="${MODELS:-text2earth diffusionsat geosynth}"
TECHNIQUES="${TECHNIQUES:-megafusion elasticdiffusion multidiffusion freescale demofusion fouriscale}"
PROMPT="${PROMPT:-A high-resolution satellite image of an urban area with dense buildings, roads and vegetation.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/benchmark_all}"
STAGE_RESOLUTIONS="${STAGE_RESOLUTIONS:-512 1024 2048 4096 8192}"
STAGE_STEPS="${STAGE_STEPS:-40 3 3 2 2}"
ENABLE_RESCHEDULE="${ENABLE_RESCHEDULE:-false}"
ENABLE_DILATION="${ENABLE_DILATION:-false}"
DISABLE_XFORMERS="${DISABLE_XFORMERS:-false}"
DISABLE_VAE_TILING="${DISABLE_VAE_TILING:-false}"

# ---- Parse command-line overrides -------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)           MODELS="$2"; shift 2 ;;
        --techniques)       TECHNIQUES="$2"; shift 2 ;;
        --prompt)           PROMPT="$2"; shift 2 ;;
        --negative_prompt)  NEGATIVE_PROMPT="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --guidance_scale)   GUIDANCE_SCALE="$2"; shift 2 ;;
        --num_inference_steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --mixed_precision)  MIXED_PRECISION="$2"; shift 2 ;;
        --output_dir)       OUTPUT_ROOT="$2"; shift 2 ;;
        --stage_resolutions) STAGE_RESOLUTIONS="$2"; shift 2 ;;
        --stage_steps)      STAGE_STEPS="$2"; shift 2 ;;
        --if_reschedule)    ENABLE_RESCHEDULE=true; shift ;;
        --if_dilation)      ENABLE_DILATION=true; shift ;;
        --no_xformers)      DISABLE_XFORMERS=true; shift ;;
        --no_vae_tiling)    DISABLE_VAE_TILING=true; shift ;;
        --help|-h)
            echo "Usage: bash $0 [OPTIONS]"
            echo ""
            echo "Options (also settable via env vars):"
            echo "  --models NAMES          Space-separated model names (default: all 3)"
            echo "  --techniques NAMES      Space-separated technique names (default: all 6)"
            echo "  --prompt TEXT           Generation prompt"
            echo "  --seed INT             Random seed (default: 42)"
            echo "  --guidance_scale FLOAT  CFG scale (default: 7.0)"
            echo "  --output_dir DIR       Root output directory"
            echo "  --stage_resolutions R  Space-separated resolutions"
            echo "  --stage_steps S        Space-separated steps per stage"
            echo "  --mixed_precision TYPE  fp16 | bf16 | no (default: fp16)"
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

mkdir -p "${OUTPUT_ROOT}"
SUMMARY_CSV="${OUTPUT_ROOT}/benchmark_summary.csv"
echo "model,technique,seed,elapsed_seconds,output_dir" > "${SUMMARY_CSV}"

# ---- Build extra flags ------------------------------------------------------
EXTRA_FLAGS=""
if [[ "${ENABLE_RESCHEDULE}" == "true" ]]; then EXTRA_FLAGS+=" --if_reschedule"; fi
if [[ "${ENABLE_DILATION}" == "true" ]]; then EXTRA_FLAGS+=" --if_dilation"; fi
if [[ "${DISABLE_XFORMERS}" == "true" ]]; then EXTRA_FLAGS+=" --no_xformers"; fi
if [[ "${DISABLE_VAE_TILING}" == "true" ]]; then EXTRA_FLAGS+=" --no_vae_tiling"; fi

# ---- Run all combinations ---------------------------------------------------
echo "============================================================"
echo "RSGen-8k — Full Benchmark"
echo "============================================================"
echo "Models:      ${MODELS}"
echo "Techniques:  ${TECHNIQUES}"
echo "Seed:        ${SEED}"
echo "Resolutions: ${STAGE_RESOLUTIONS}"
echo "Precision:   ${MIXED_PRECISION}"
echo "Output root: ${OUTPUT_ROOT}"
echo "============================================================"
echo ""

TOTAL_START=$(date +%s)
RUN_INDEX=0
TOTAL_RUNS=0

# Count total
for _m in ${MODELS}; do for _t in ${TECHNIQUES}; do TOTAL_RUNS=$((TOTAL_RUNS + 1)); done; done

for MODEL in ${MODELS}; do
    for TECH in ${TECHNIQUES}; do
        RUN_INDEX=$((RUN_INDEX + 1))
        RUN_DIR="${OUTPUT_ROOT}/${MODEL}_${TECH}_seed${SEED}"
        mkdir -p "${RUN_DIR}"

        echo "------------------------------------------------------------"
        echo "[${RUN_INDEX}/${TOTAL_RUNS}] Model: ${MODEL} | Technique: ${TECH}"
        echo "------------------------------------------------------------"

        START_TIME=$(date +%s)

        python "${PROJECT_ROOT}/scripts/generate.py" \
            --model_name "${MODEL}" \
            --technique "${TECH}" \
            --prompt "${PROMPT}" \
            --seed "${SEED}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --num_inference_steps "${NUM_INFERENCE_STEPS}" \
            --mixed_precision "${MIXED_PRECISION}" \
            --output_dir "${RUN_DIR}" \
            --stage_resolutions ${STAGE_RESOLUTIONS} \
            --stage_steps ${STAGE_STEPS} \
            ${EXTRA_FLAGS} \
            2>&1 | tee "${RUN_DIR}/benchmark.log" || {
                echo "WARNING: Run failed for ${MODEL} + ${TECH}"
                echo "${MODEL},${TECH},${SEED},FAILED,${RUN_DIR}" >> "${SUMMARY_CSV}"
                continue
            }

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "${MODEL},${TECH},${SEED},${ELAPSED},${RUN_DIR}" >> "${SUMMARY_CSV}"

        echo "  → Completed in ${ELAPSED}s"
        echo ""
    done
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo "============================================================"
echo "Full benchmark completed in ${TOTAL_ELAPSED}s"
echo "Summary: ${SUMMARY_CSV}"
echo "============================================================"
