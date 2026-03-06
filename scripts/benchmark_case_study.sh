#!/usr/bin/env bash
# =============================================================================
# RSGen-8k — Resolution Case Study & Metric Benchmark
# =============================================================================
# Runs generation at progressively larger output resolutions (1K → 8K) and
# evaluates image quality metrics (CLIP-Score, FID when reference available).
# Produces a summary CSV comparing metrics across resolution tiers.
#
# Usage:
#   # Full case study (1024 → 2048 → 4096 → 8192)
#   bash scripts/benchmark_case_study.sh
#
#   # With reference images for FID/LPIPS/PSNR/SSIM
#   bash scripts/benchmark_case_study.sh --reference_dir ./data/reference
#
#   # Custom model/technique
#   MODEL_NAME=diffusionsat TECHNIQUE=multidiffusion bash scripts/benchmark_case_study.sh
#
# =============================================================================
set -euo pipefail

# ---- Default configuration --------------------------------------------------
MODEL_NAME="${MODEL_NAME:-text2earth}"
TECHNIQUE="${TECHNIQUE:-megafusion}"
PROMPT="${PROMPT:-A high-resolution satellite image of an urban area with dense buildings, roads and vegetation.}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/case_study}"
REFERENCE_DIR="${REFERENCE_DIR:-}"
STAGE_STEPS_BASE="${STAGE_STEPS_BASE:-40 3 3 2 2}"
ENABLE_RESCHEDULE="${ENABLE_RESCHEDULE:-false}"
METRICS="${METRICS:-clip_score fid}"

# Resolution tiers: each row is "max_res|stage_resolutions|stage_steps"
# Format: output_res|res1 res2 ...|steps1 steps2 ...
RESOLUTION_TIERS_FULL=(
    "1024|512 1024|40 10"
    "2048|512 1024 2048|40 3 7"
    "4096|512 1024 2048 4096|40 3 3 4"
    "8192|512 1024 2048 4096 8192|40 3 3 2 2"
)
RESOLUTION_TIERS_QUICK=(
    "1024|512 1024|40 10"
    "2048|512 1024 2048|40 3 7"
)
QUICK_MODE="${QUICK_MODE:-false}"

# ---- Parse command-line overrides -------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)       MODEL_NAME="$2"; shift 2 ;;
        --technique)        TECHNIQUE="$2"; shift 2 ;;
        --prompt)           PROMPT="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --output_root)      OUTPUT_ROOT="$2"; shift 2 ;;
        --reference_dir)    REFERENCE_DIR="$2"; shift 2 ;;
        --metrics)          METRICS="$2"; shift 2 ;;
        --if_reschedule)    ENABLE_RESCHEDULE=true; shift ;;
        --quick)            QUICK_MODE=true; shift ;;
        --help|-h)
            echo "Usage: bash $0 [OPTIONS]"
            echo ""
            echo "Runs generation at 1K, 2K, 4K, 8K and evaluates metrics."
            echo ""
            echo "Options (also settable via env vars):"
            echo "  --model_name NAME       Base model (default: text2earth)"
            echo "  --technique NAME        Upscaling technique (default: megafusion)"
            echo "  --prompt TEXT           Generation prompt"
            echo "  --seed INT              Random seed (default: 42)"
            echo "  --output_root DIR       Output root (default: ./outputs/case_study)"
            echo "  --reference_dir DIR     Reference images for FID/LPIPS/PSNR/SSIM"
            echo "  --metrics LIST          Space-separated (default: clip_score fid)"
            echo "  --if_reschedule          Enable noise rescheduling"
            echo "  --quick                  Run only 1024 and 2048 (faster)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Resolve project root ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${OUTPUT_ROOT}"
SUMMARY_CSV="${OUTPUT_ROOT}/case_study_summary.csv"
TIMING_JSON="${OUTPUT_ROOT}/case_study_timing.json"

# ---- Header for summary CSV ------------------------------------------------
echo "model,technique,output_res,stage_resolutions,elapsed_sec,clip_score,fid,kid,cmmd,dino_similarity,lpips,psnr,ssim" > "${SUMMARY_CSV}"

# ---- Select resolution tiers ------------------------------------------------
if [[ "${QUICK_MODE}" == "true" ]]; then
    RESOLUTION_TIERS=("${RESOLUTION_TIERS_QUICK[@]}")
    echo "Quick mode: 1024px and 2048px only"
else
    RESOLUTION_TIERS=("${RESOLUTION_TIERS_FULL[@]}")
fi

# ---- Run each resolution tier -----------------------------------------------
TOTAL_START=$(date +%s)
declare -a RUN_DIRS=()

for tier in "${RESOLUTION_TIERS[@]}"; do
    IFS='|' read -r output_res resolutions steps <<< "${tier}"
    RUN_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${TECHNIQUE}_${output_res}px_seed${SEED}"
    RUN_DIRS+=("${RUN_DIR}")
    mkdir -p "${RUN_DIR}"

    echo ""
    echo "============================================================"
    echo "Case Study — ${output_res}px output"
    echo "  Stages: ${resolutions}"
    echo "  Steps:  ${steps}"
    echo "  Output: ${RUN_DIR}"
    echo "============================================================"

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
        --stage_resolutions ${resolutions}
        --stage_steps ${steps}
    )
    if [[ "${ENABLE_RESCHEDULE}" == "true" ]]; then
        CMD+=(--if_reschedule)
    fi

    START_TIME=$(date +%s)
    "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/benchmark.log" || {
        echo "WARNING: Generation failed for ${output_res}px"
        continue
    }
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # Save run metadata (with prompt for CLIP-Score)
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
    'output_resolution': ${output_res},
    'stage_resolutions': [int(x) for x in '${resolutions}'.split()],
    'stage_steps': [int(x) for x in '${steps}'.split()],
    'elapsed_seconds': ${ELAPSED}
}
with open('${RUN_DIR}/run_metadata.json', 'w') as f:
    json.dump(data, f, indent=2)
"

    # Run evaluation
    EVAL_ARGS=(
        --generated_dir "${RUN_DIR}"
        --output_file "${RUN_DIR}/eval_results.json"
        --metrics "${METRICS}"
    )
    if [[ -n "${REFERENCE_DIR}" ]]; then
        EVAL_ARGS+=(--reference_dir "${REFERENCE_DIR}")
    fi

    bash "${PROJECT_ROOT}/scripts/evaluate.sh" "${EVAL_ARGS[@]}" 2>/dev/null || true

    # Append to summary CSV
    python3 -c "
import json
import os

run_dir = '${RUN_DIR}'
eval_path = os.path.join(run_dir, 'eval_results.json')
resolutions_str = '${resolutions}'.replace(' ', '_')
elapsed = ${ELAPSED}

clip_score = ''
fid = ''
kid = ''
cmmd = ''
dino_similarity = ''
lpips = ''
psnr = ''
ssim = ''

if os.path.exists(eval_path):
    with open(eval_path) as f:
        m = json.load(f).get('metrics', {})
    clip_score = m.get('clip_score') or m.get('clip_feature_mean_norm') or ''
    fid = m.get('frechet_inception_distance') or m.get('fid') or ''
    kid = m.get('kid') or ''
    cmmd = m.get('cmmd') or ''
    dino_similarity = m.get('dino_similarity') or ''
    lpips = m.get('lpips') or ''
    psnr = m.get('psnr') or ''
    ssim = m.get('ssim') or ''

def fmt(v):
    return f'{v:.4f}' if isinstance(v, (int, float)) else str(v)

row = ','.join([
    '${MODEL_NAME}', '${TECHNIQUE}', '${output_res}',
    resolutions_str, str(elapsed),
    fmt(clip_score), fmt(fid), fmt(kid), fmt(cmmd), fmt(dino_similarity),
    fmt(lpips), fmt(psnr), fmt(ssim)
])
with open('${SUMMARY_CSV}', 'a') as f:
    f.write(row + '\n')
"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

# ---- Final summary ----------------------------------------------------------
echo ""
echo "============================================================"
echo "RSGen-8k Case Study — Complete"
echo "============================================================"
echo "Total time: ${TOTAL_ELAPSED}s"
echo "Summary:    ${SUMMARY_CSV}"
echo ""
if [[ -f "${SUMMARY_CSV}" ]]; then
    echo "Results:"
    column -t -s',' "${SUMMARY_CSV}" 2>/dev/null || cat "${SUMMARY_CSV}"
fi
echo "============================================================"
