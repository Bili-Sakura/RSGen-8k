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
METRICS="${METRICS:-fid clip_score}"  # Space-separated: fid clip_score lpips psnr ssim

# ---- Parse command-line overrides -------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --generated_dir)  GENERATED_DIR="$2"; shift 2 ;;
        --reference_dir)  REFERENCE_DIR="$2"; shift 2 ;;
        --benchmark_root) BENCHMARK_ROOT="$2"; shift 2 ;;
        --output_file)    OUTPUT_FILE="$2"; shift 2 ;;
        --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --clip_model)     CLIP_MODEL="$2"; shift 2 ;;
        --metrics)        METRICS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash $0 [OPTIONS]"
            echo ""
            echo "Options (also settable via env vars):"
            echo "  --generated_dir DIR     Directory of generated images"
            echo "  --reference_dir DIR     Directory of reference images (for FID/LPIPS/PSNR/SSIM)"
            echo "  --benchmark_root DIR    Evaluate all subdirectories under this root"
            echo "  --output_file FILE      Write results to this file (default: <dir>/eval_results.json)"
            echo "  --batch_size N          Batch size for metric computation (default: 8)"
            echo "  --device DEVICE         cuda | cpu (default: cuda)"
            echo "  --clip_model MODEL      CLIP model for CLIP-Score (default: ViT-B/32)"
            echo "  --metrics LIST          Space-separated metrics (default: fid clip_score)"
            echo "                          Available: fid clip_score lpips psnr ssim"
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

    # Build Python evaluation inline
    python - <<PYEOF
import json
import os
import sys
import glob

gen_dir = "${gen_dir}"
ref_dir = "${ref_dir}" if "${ref_dir}" else None
out_file = "${out_file}"
metrics_str = "${METRICS}"
device = "${DEVICE}"
batch_size = ${BATCH_SIZE}

results = {"generated_dir": gen_dir, "num_images": ${num_images}, "metrics": {}}
requested_metrics = metrics_str.split()

image_paths = sorted(
    glob.glob(os.path.join(gen_dir, "*.png"))
    + glob.glob(os.path.join(gen_dir, "*.jpg"))
    + glob.glob(os.path.join(gen_dir, "*.jpeg"))
)

# ---- FID (requires reference images) ----
if "fid" in requested_metrics and ref_dir:
    try:
        from torch_fidelity import calculate_metrics
        fid_metrics = calculate_metrics(
            input1=gen_dir,
            input2=ref_dir,
            cuda=(device == "cuda"),
            fid=True,
            batch_size=batch_size,
        )
        results["metrics"]["fid"] = fid_metrics.get("frechet_inception_distance", None)
        print(f"  FID: {results['metrics']['fid']:.4f}")
    except ImportError:
        print("  SKIP FID: torch-fidelity not installed (pip install torch-fidelity)")
    except Exception as e:
        print(f"  FID error: {e}")

# ---- CLIP Score ----
if "clip_score" in requested_metrics:
    try:
        import torch
        import clip
        from PIL import Image

        clip_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("${CLIP_MODEL}", device=clip_device)

        # Try to load prompts from generation metadata for text-image scoring
        metadata_path = os.path.join(gen_dir, "generation_metadata.json")
        run_metadata_path = os.path.join(gen_dir, "run_metadata.json")
        prompts = []
        for mp in [metadata_path, run_metadata_path]:
            if os.path.exists(mp):
                with open(mp) as mf:
                    meta = json.load(mf)
                    if "prompt" in meta:
                        prompts = [meta["prompt"]] * len(image_paths)
                break

        scores = []
        for idx, img_path in enumerate(image_paths):
            img = preprocess(Image.open(img_path)).unsqueeze(0).to(clip_device)
            with torch.no_grad():
                image_features = model.encode_image(img)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                if prompts:
                    text_tokens = clip.tokenize([prompts[idx]], truncate=True).to(clip_device)
                    text_features = model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (image_features @ text_features.T).item()
                    scores.append(similarity)
                else:
                    scores.append(image_features.cpu().numpy().mean())

        metric_key = "clip_score" if prompts else "clip_feature_mean_norm"
        results["metrics"][metric_key] = float(sum(scores) / len(scores)) if scores else None
        label = "CLIP Score (text-image)" if prompts else "CLIP Feature Mean Norm"
        print(f"  {label}: {results['metrics'][metric_key]:.4f}")
    except ImportError:
        print("  SKIP CLIP-Score: clip not installed (pip install git+https://github.com/openai/CLIP.git)")
    except Exception as e:
        print(f"  CLIP-Score error: {e}")

# ---- LPIPS (requires reference images) ----
if "lpips" in requested_metrics and ref_dir:
    try:
        import torch
        import lpips
        from PIL import Image
        from torchvision import transforms

        loss_fn = lpips.LPIPS(net='alex')
        if device == "cuda" and torch.cuda.is_available():
            loss_fn = loss_fn.cuda()

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        ref_images = sorted(
            glob.glob(os.path.join(ref_dir, "*.png"))
            + glob.glob(os.path.join(ref_dir, "*.jpg"))
        )
        n = min(len(image_paths), len(ref_images))
        lpips_vals = []
        for i in range(n):
            img1 = transform(Image.open(image_paths[i]).convert("RGB")).unsqueeze(0)
            img2 = transform(Image.open(ref_images[i]).convert("RGB")).unsqueeze(0)
            if device == "cuda" and torch.cuda.is_available():
                img1, img2 = img1.cuda(), img2.cuda()
            with torch.no_grad():
                lpips_vals.append(loss_fn(img1, img2).item())

        results["metrics"]["lpips"] = sum(lpips_vals) / len(lpips_vals) if lpips_vals else None
        print(f"  LPIPS: {results['metrics']['lpips']:.4f}")
    except ImportError:
        print("  SKIP LPIPS: lpips not installed (pip install lpips)")
    except Exception as e:
        print(f"  LPIPS error: {e}")

# ---- PSNR / SSIM (requires reference images) ----
for metric_name in ["psnr", "ssim"]:
    if metric_name in requested_metrics and ref_dir:
        try:
            import numpy as np
            from PIL import Image
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity

            ref_images = sorted(
                glob.glob(os.path.join(ref_dir, "*.png"))
                + glob.glob(os.path.join(ref_dir, "*.jpg"))
            )
            n = min(len(image_paths), len(ref_images))
            vals = []
            for i in range(n):
                img1 = np.array(Image.open(image_paths[i]).convert("RGB").resize((256, 256)))
                img2 = np.array(Image.open(ref_images[i]).convert("RGB").resize((256, 256)))
                if metric_name == "psnr":
                    vals.append(peak_signal_noise_ratio(img2, img1))
                else:
                    vals.append(structural_similarity(img2, img1, channel_axis=2))

            results["metrics"][metric_name] = sum(vals) / len(vals) if vals else None
            print(f"  {metric_name.upper()}: {results['metrics'][metric_name]:.4f}")
        except ImportError:
            print(f"  SKIP {metric_name.upper()}: scikit-image not installed (pip install scikit-image)")
        except Exception as e:
            print(f"  {metric_name.upper()} error: {e}")

# ---- Write results ----
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to: {out_file}")
PYEOF
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
