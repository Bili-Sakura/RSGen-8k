#!/usr/bin/env bash
# =============================================================================
# RSGen-8k — Image Quality Evaluation
# =============================================================================
# Evaluates generated images using standard metrics: FID, CLIP-Score, LPIPS,
# and PSNR/SSIM (when reference images are available).
#
# Prerequisites:
#   pip install torch-fidelity clip-benchmark lpips scikit-image
#
# Usage:
#   # Evaluate a single generation run
#   bash scripts/evaluate.sh --generated_dir ./outputs/benchmark/text2earth_megafusion_seed42
#
#   # Compare against reference images
#   bash scripts/evaluate.sh \
#       --generated_dir ./outputs/benchmark/text2earth_megafusion_seed42 \
#       --reference_dir ./data/reference_images
#
#   # Evaluate all runs under a benchmark root
#   bash scripts/evaluate.sh --benchmark_root ./outputs/benchmark_all
#
# =============================================================================
set -euo pipefail

# ---- Default configuration --------------------------------------------------
GENERATED_DIR="${GENERATED_DIR:-}"
REFERENCE_DIR="${REFERENCE_DIR:-}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B/32}"
CMMD_CLIP_MODEL="${CMMD_CLIP_MODEL:-}"
DINO_MODEL="${DINO_MODEL:-}"
METRICS="${METRICS:-fid clip_score}"  # Space-separated: fid kid cmmd dino_similarity clip_score lpips psnr ssim

# ---- Parse command-line overrides -------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --generated_dir)    GENERATED_DIR="$2"; shift 2 ;;
        --reference_dir)    REFERENCE_DIR="$2"; shift 2 ;;
        --benchmark_root)   BENCHMARK_ROOT="$2"; shift 2 ;;
        --output_file)      OUTPUT_FILE="$2"; shift 2 ;;
        --batch_size)       BATCH_SIZE="$2"; shift 2 ;;
        --device)           DEVICE="$2"; shift 2 ;;
        --clip_model)       CLIP_MODEL="$2"; shift 2 ;;
        --cmmd_clip_model)  CMMD_CLIP_MODEL="$2"; shift 2 ;;
        --dino_model)       DINO_MODEL="$2"; shift 2 ;;
        --metrics)          METRICS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash $0 [OPTIONS]"
            echo ""
            echo "Options (also settable via env vars):"
            echo "  --generated_dir DIR     Directory of generated images"
            echo "  --reference_dir DIR     Directory of reference images"
            echo "  --benchmark_root DIR    Evaluate all subdirectories under this root"
            echo "  --output_file FILE      Write results to this file"
            echo "  --batch_size N          Batch size (default: 8)"
            echo "  --device DEVICE         cuda | cpu (default: cuda)"
            echo "  --clip_model PATH       CLIP for CLIP-Score (HF ID or local path)"
            echo "  --cmmd_clip_model PATH  CLIP for CMMD (HF ID or local path)"
            echo "  --dino_model PATH       DINOv2 for DINO similarity (HF ID or local path)"
            echo "  --metrics LIST          fid kid cmmd dino_similarity clip_score lpips psnr ssim"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Resolve project root ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Helper: evaluate a single directory ------------------------------------
evaluate_directory() {
    local gen_dir="$1"
    local ref_dir="${2:-}"
    local out_file="${3:-${gen_dir}/eval_results.json}"

    echo "------------------------------------------------------------"
    echo "Evaluating: ${gen_dir}"
    echo "------------------------------------------------------------"

    # Count generated images
    local num_images
    num_images=$(find "${gen_dir}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
    echo "  Found ${num_images} images"

    if [[ ${num_images} -eq 0 ]]; then
        echo "  SKIP: No images found."
        return
    fi

    REF_ARGS=()
    [[ -n "${ref_dir}" ]] && REF_ARGS=(--reference_dir "${ref_dir}")
    CMMD_ARGS=()
    [[ -n "${CMMD_CLIP_MODEL}" ]] && CMMD_ARGS=(--cmmd_clip_model "${CMMD_CLIP_MODEL}")
    DINO_ARGS=()
    [[ -n "${DINO_MODEL}" ]] && DINO_ARGS=(--dino_model "${DINO_MODEL}")
    python -m rsgen8k.metrics \
        --generated_dir "${gen_dir}" \
        --output_file "${out_file}" \
        --metrics "${METRICS}" \
        --device "${DEVICE}" \
        --batch_size ${BATCH_SIZE} \
        --clip_model "${CLIP_MODEL}" \
        "${REF_ARGS[@]}" \
        "${CMMD_ARGS[@]}" \
        "${DINO_ARGS[@]}"
}

# ---- Main logic -------------------------------------------------------------
if [[ -n "${BENCHMARK_ROOT}" ]]; then
    # Evaluate all subdirectories
    echo "============================================================"
    echo "RSGen-8k — Batch Evaluation"
    echo "============================================================"
    echo "Root: ${BENCHMARK_ROOT}"
    echo "Metrics: ${METRICS}"
    echo "============================================================"

    for run_dir in "${BENCHMARK_ROOT}"/*/; do
        if [[ -d "${run_dir}" ]]; then
            evaluate_directory "${run_dir}" "${REFERENCE_DIR}" "${run_dir}/eval_results.json"
        fi
    done

    echo ""
    echo "============================================================"
    echo "Batch evaluation complete."
    echo "============================================================"

elif [[ -n "${GENERATED_DIR}" ]]; then
    # Evaluate a single directory
    OUT="${OUTPUT_FILE:-${GENERATED_DIR}/eval_results.json}"
    evaluate_directory "${GENERATED_DIR}" "${REFERENCE_DIR}" "${OUT}"

else
    echo "ERROR: Specify --generated_dir or --benchmark_root"
    echo "Run with --help for usage."
    exit 1
fi
