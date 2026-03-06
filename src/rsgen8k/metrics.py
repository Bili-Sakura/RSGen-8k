"""Image quality metrics for generated remote sensing images.

Provides FID, KID, CLIP-Score, CMMD, DINO similarity, LPIPS, PSNR, and SSIM
evaluation with optional reference images.

Pre-trained models (CLIP, DINOv2) can be loaded from HuggingFace model IDs or
local paths. Use --clip_model, --cmmd_clip_model, --dino_model to specify.

Prerequisites (optional, for each metric):
  - FID/KID: pip install torch-fidelity
  - CLIP-Score/CMMD: pip install transformers (or openai CLIP for clip_score)
  - DINO: pip install transformers (DINOv2)
  - LPIPS: pip install lpips
  - PSNR/SSIM: pip install scikit-image
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


def compute_fid(
    gen_dir: str,
    ref_dir: str,
    device: str = "cuda",
    batch_size: int = 8,
) -> Optional[float]:
    """Compute Fréchet Inception Distance between generated and reference images.

    Args:
        gen_dir: Directory of generated images.
        ref_dir: Directory of reference images.
        device: cuda or cpu.
        batch_size: Batch size for Inception feature extraction.

    Returns:
        FID value or None if computation failed.
    """
    try:
        from torch_fidelity import calculate_metrics

        metrics = calculate_metrics(
            input1=gen_dir,
            input2=ref_dir,
            cuda=(device == "cuda"),
            fid=True,
            batch_size=batch_size,
        )
        return metrics.get("frechet_inception_distance")
    except ImportError:
        logger.warning(
            "SKIP FID: torch-fidelity not installed (pip install torch-fidelity)"
        )
        return None
    except Exception as e:
        logger.warning("FID error: %s", e)
        return None


def compute_kid(
    gen_dir: str,
    ref_dir: str,
    device: str = "cuda",
    batch_size: int = 8,
) -> Optional[float]:
    """Compute Kernel Inception Distance (KID) between generated and reference images.

    Returns kernel_inception_distance_mean (lower is better).
    """
    try:
        from torch_fidelity import calculate_metrics

        metrics = calculate_metrics(
            input1=gen_dir,
            input2=ref_dir,
            cuda=(device == "cuda"),
            kid=True,
            batch_size=batch_size,
        )
        return metrics.get("kernel_inception_distance_mean")
    except ImportError:
        logger.warning(
            "SKIP KID: torch-fidelity not installed (pip install torch-fidelity)"
        )
        return None
    except Exception as e:
        logger.warning("KID error: %s", e)
        return None


def compute_cmmd(
    image_paths: List[str],
    ref_dir: str,
    clip_model: str = "openai/clip-vit-large-patch14-336",
    device: str = "cuda",
    batch_size: int = 32,
    sigma: float = 10.0,
    scale: float = 1000.0,
) -> Optional[float]:
    """Compute CMMD (CLIP Maximum Mean Discrepancy, CVPR'24) between gen and ref.

    Uses CLIP embeddings and MMD with Gaussian RBF kernel. Lower is better.
    clip_model: HuggingFace model ID or path to local CLIP directory.

    Ref: Jayasumana et al., "Rethinking FID: Towards a Better Evaluation Metric
    for Image Generation", CVPR 2024.
    """
    try:
        import torch
        from PIL import Image
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        model_path = os.path.expanduser(clip_model)
        processor = CLIPImageProcessor.from_pretrained(model_path)
        model = CLIPVisionModelWithProjection.from_pretrained(model_path).eval().to(dev)

        ref_paths = sorted(
            glob.glob(os.path.join(ref_dir, "*.png"))
            + glob.glob(os.path.join(ref_dir, "*.jpg"))
        )
        n = min(len(image_paths), len(ref_paths))
        if n == 0:
            return None

        def embed_paths(paths: List[str]) -> torch.Tensor:
            embs = []
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i : i + batch_size]
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = processor(images=images, return_tensors="pt")
                if dev.type == "cuda":
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                    e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
                embs.append(e.cpu())
            return torch.cat(embs, dim=0)

        gen_embs = embed_paths(image_paths[:n])
        ref_embs = embed_paths(ref_paths[:n])

        gamma = 1.0 / (2.0 * sigma**2)
        x, y = gen_embs.float(), ref_embs.float()

        x_sq = (x * x).sum(dim=1)
        y_sq = (y * y).sum(dim=1)

        m, n_dim = x.size(0), y.size(0)
        k_xx_mat = torch.exp(-gamma * (-2 * x @ x.T + x_sq.unsqueeze(1) + x_sq.unsqueeze(0)))
        k_xx = (k_xx_mat.sum() - k_xx_mat.diag().sum()) / max(1, m * (m - 1))

        k_yy_mat = torch.exp(-gamma * (-2 * y @ y.T + y_sq.unsqueeze(1) + y_sq.unsqueeze(0)))
        k_yy = (k_yy_mat.sum() - k_yy_mat.diag().sum()) / max(1, n_dim * (n_dim - 1))

        k_xy = torch.exp(-gamma * (-2 * x @ y.T + x_sq.unsqueeze(1) + y_sq.unsqueeze(0))).mean()

        mmd_sq = k_xx + k_yy - 2 * k_xy
        return float(scale * mmd_sq.clamp(min=0))
    except ImportError as e:
        logger.warning(
            "SKIP CMMD: transformers not installed or missing CLIP "
            "(pip install transformers): %s",
            e,
        )
        return None
    except Exception as e:
        logger.warning("CMMD error: %s", e)
        return None


def compute_dino_similarity(
    image_paths: List[str],
    ref_dir: str,
    device: str = "cuda",
    model_name: str = "facebook/dinov2-base",
) -> Optional[float]:
    """Compute mean cosine similarity between generated and reference images using DINOv2.

    Higher is better. Uses global CLS token embeddings.
    model_name: HuggingFace model ID or path to local DINOv2 directory.
    """
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        from transformers import AutoImageProcessor, AutoModel

        dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        model_path = os.path.expanduser(model_name)
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).eval().to(dev)

        ref_paths = sorted(
            glob.glob(os.path.join(ref_dir, "*.png"))
            + glob.glob(os.path.join(ref_dir, "*.jpg"))
        )
        n = min(len(image_paths), len(ref_paths))
        if n == 0:
            return None

        sims = []
        for i in range(n):
            img_gen = Image.open(image_paths[i]).convert("RGB")
            img_ref = Image.open(ref_paths[i]).convert("RGB")
            inputs_gen = processor(images=img_gen, return_tensors="pt").to(dev)
            inputs_ref = processor(images=img_ref, return_tensors="pt").to(dev)
            with torch.no_grad():
                out_gen = model(**inputs_gen)
                out_ref = model(**inputs_ref)
            e_gen = out_gen.last_hidden_state[:, 0].squeeze(0)
            e_ref = out_ref.last_hidden_state[:, 0].squeeze(0)
            e_gen = e_gen / e_gen.norm()
            e_ref = e_ref / e_ref.norm()
            sim = (e_gen @ e_ref).item()
            sims.append(sim)

        return sum(sims) / len(sims)
    except ImportError:
        logger.warning(
            "SKIP DINO similarity: transformers not installed (pip install transformers)"
        )
        return None
    except Exception as e:
        logger.warning("DINO similarity error: %s", e)
        return None


def _is_local_model_path(model_id: str) -> bool:
    """Return True if model_id is a path to a local directory."""
    return os.path.isdir(os.path.expanduser(model_id))


def _compute_clip_score_transformers(
    image_paths: List[str],
    gen_dir: str,
    model_id: str,
    device_str: str,
) -> tuple[Optional[float], str]:
    """CLIP-Score using transformers (for local path or HuggingFace model ID)."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    dev = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_id).eval().to(dev)
    processor = CLIPProcessor.from_pretrained(model_id)

    prompts: List[str] = []
    for name in ["generation_metadata.json", "run_metadata.json"]:
        path = os.path.join(gen_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                meta = json.load(f)
                if "prompt" in meta:
                    prompts = [meta["prompt"]] * len(image_paths)
                    break

    scores = []
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        if prompts:
            inputs = processor(
                text=[prompts[idx]],
                images=img,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            inputs = processor(images=img, return_tensors="pt")
        if dev.type == "cuda":
            inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if prompts and outputs.text_embeds is not None:
                text_features = outputs.text_embeds / outputs.text_embeds.norm(
                    dim=-1, keepdim=True
                )
                sim = (image_features @ text_features.T).item()
            else:
                sim = float(image_features.cpu().numpy().mean())
            scores.append(sim)

    avg = sum(scores) / len(scores) if scores else None
    key = "clip_score" if prompts else "clip_feature_mean_norm"
    return avg, key


def compute_clip_score(
    image_paths: List[str],
    gen_dir: str,
    clip_model: str = "ViT-B/32",
    device: str = "cuda",
) -> tuple[Optional[float], str]:
    """Compute CLIP text-image similarity when prompts available, else feature norm.

    Args:
        image_paths: Paths to generated images.
        gen_dir: Directory containing run_metadata.json or generation_metadata.json.
        clip_model: HuggingFace model ID (e.g., openai/clip-vit-base-patch32) or
            path to local model directory.
        device: cuda or cpu.

    Returns:
        (score, metric_key) where metric_key is 'clip_score' or 'clip_feature_mean_norm'.
    """
    try:
        import torch
        from PIL import Image

        clip_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        model_id = os.path.expanduser(clip_model)

        if _is_local_model_path(model_id):
            return _compute_clip_score_transformers(
                image_paths, gen_dir, model_id, device
            )

        import clip

        model, preprocess = clip.load(clip_model, device=clip_device)

        prompts: List[str] = []
        for name in ["generation_metadata.json", "run_metadata.json"]:
            path = os.path.join(gen_dir, name)
            if os.path.exists(path):
                with open(path) as f:
                    meta = json.load(f)
                    if "prompt" in meta:
                        prompts = [meta["prompt"]] * len(image_paths)
                        break

        scores = []
        for idx, img_path in enumerate(image_paths):
            img = preprocess(Image.open(img_path)).unsqueeze(0).to(clip_device)
            with torch.no_grad():
                image_features = model.encode_image(img)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                if prompts:
                    text_tokens = clip.tokenize(
                        [prompts[idx]], truncate=True
                    ).to(clip_device)
                    text_features = model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )
                    similarity = (image_features @ text_features.T).item()
                    scores.append(similarity)
                else:
                    scores.append(float(image_features.cpu().numpy().mean()))

        avg = sum(scores) / len(scores) if scores else None
        metric_key = "clip_score" if prompts else "clip_feature_mean_norm"
        return avg, metric_key
    except ImportError:
        logger.warning(
            "SKIP CLIP-Score: clip not installed "
            "(pip install git+https://github.com/openai/CLIP.git)"
        )
        return None, "clip_score"
    except Exception as e:
        logger.warning("CLIP-Score error: %s", e)
        return None, "clip_score"


def compute_lpips(
    image_paths: List[str],
    ref_dir: str,
    device: str = "cuda",
) -> Optional[float]:
    """Compute LPIPS (Learned Perceptual Image Patch Similarity) vs references.

    Args:
        image_paths: Paths to generated images.
        ref_dir: Directory of reference images.
        device: cuda or cpu.

    Returns:
        Mean LPIPS value or None.
    """
    try:
        import torch
        import lpips
        from PIL import Image
        from torchvision import transforms

        loss_fn = lpips.LPIPS(net="alex")
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
        if n == 0:
            return None

        vals = []
        for i in range(n):
            img1 = transform(Image.open(image_paths[i]).convert("RGB")).unsqueeze(0)
            img2 = transform(Image.open(ref_images[i]).convert("RGB")).unsqueeze(0)
            if device == "cuda" and torch.cuda.is_available():
                img1, img2 = img1.cuda(), img2.cuda()
            with torch.no_grad():
                vals.append(loss_fn(img1, img2).item())

        return sum(vals) / len(vals) if vals else None
    except ImportError:
        logger.warning("SKIP LPIPS: lpips not installed (pip install lpips)")
        return None
    except Exception as e:
        logger.warning("LPIPS error: %s", e)
        return None


def compute_psnr(
    image_paths: List[str],
    ref_dir: str,
) -> Optional[float]:
    """Compute mean PSNR between generated and reference images."""
    return _compute_psnr_ssim(image_paths, ref_dir, "psnr")


def compute_ssim(
    image_paths: List[str],
    ref_dir: str,
) -> Optional[float]:
    """Compute mean SSIM between generated and reference images."""
    return _compute_psnr_ssim(image_paths, ref_dir, "ssim")


def _compute_psnr_ssim(
    image_paths: List[str],
    ref_dir: str,
    metric: str,
) -> Optional[float]:
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import (
            peak_signal_noise_ratio,
            structural_similarity,
        )

        ref_images = sorted(
            glob.glob(os.path.join(ref_dir, "*.png"))
            + glob.glob(os.path.join(ref_dir, "*.jpg"))
        )
        n = min(len(image_paths), len(ref_images))
        if n == 0:
            return None

        vals = []
        for i in range(n):
            img1 = np.array(
                Image.open(image_paths[i]).convert("RGB").resize((256, 256))
            )
            img2 = np.array(
                Image.open(ref_images[i]).convert("RGB").resize((256, 256))
            )
            if metric == "psnr":
                vals.append(peak_signal_noise_ratio(img2, img1))
            else:
                vals.append(structural_similarity(img2, img1, channel_axis=2))

        return sum(vals) / len(vals) if vals else None
    except ImportError:
        logger.warning(
            "SKIP %s: scikit-image not installed (pip install scikit-image)",
            metric.upper(),
        )
        return None
    except Exception as e:
        logger.warning("%s error: %s", metric.upper(), e)
        return None


# ---------------------------------------------------------------------------
# Main evaluation API
# ---------------------------------------------------------------------------


def evaluate_directory(
    gen_dir: str,
    ref_dir: Optional[str] = None,
    out_file: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 8,
    clip_model: str = "ViT-B/32",
    cmmd_clip_model: Optional[str] = None,
    dino_model: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a directory of generated images with the requested metrics.

    Args:
        gen_dir: Directory containing generated PNG/JPG images.
        ref_dir: Optional directory of reference images (required for FID, KID,
            CMMD, DINO, LPIPS, PSNR, SSIM).
        out_file: Optional path to write JSON results.
        metrics: List of metric names: fid, kid, cmmd, dino_similarity,
            clip_score, lpips, psnr, ssim.
        device: cuda or cpu.
        batch_size: Batch size for FID/KID/CMMD.
        clip_model: CLIP model for CLIP-Score (HF ID or local path).
        cmmd_clip_model: CLIP model for CMMD (HF ID or local path). Defaults to
            openai/clip-vit-large-patch14-336 when None.
        dino_model: DINOv2 model for DINO similarity (HF ID or local path).
            Defaults to facebook/dinov2-base when None.
        verbose: Whether to print metric values.

    Returns:
        Dict with keys: generated_dir, num_images, metrics.
    """
    metrics = metrics or ["fid", "clip_score"]
    requested = [m.lower() for m in metrics]

    image_paths = sorted(
        glob.glob(os.path.join(gen_dir, "*.png"))
        + glob.glob(os.path.join(gen_dir, "*.jpg"))
        + glob.glob(os.path.join(gen_dir, "*.jpeg"))
    )
    num_images = len(image_paths)

    results: Dict[str, Any] = {
        "generated_dir": gen_dir,
        "num_images": num_images,
        "metrics": {},
    }

    if num_images == 0:
        if out_file:
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
        return results

    # FID
    if "fid" in requested and ref_dir:
        fid_val = compute_fid(gen_dir, ref_dir, device, batch_size)
        results["metrics"]["fid"] = fid_val
        if verbose and fid_val is not None:
            print(f"  FID: {fid_val:.4f}")

    # KID
    if "kid" in requested and ref_dir:
        kid_val = compute_kid(gen_dir, ref_dir, device, batch_size)
        results["metrics"]["kid"] = kid_val
        if verbose and kid_val is not None:
            print(f"  KID: {kid_val:.4f}")

    # CMMD (CVPR'24, CLIP embedding distance)
    if "cmmd" in requested and ref_dir:
        cmmd_clip = cmmd_clip_model or "openai/clip-vit-large-patch14-336"
        cmmd_val = compute_cmmd(
            image_paths, ref_dir, cmmd_clip, device, batch_size
        )
        results["metrics"]["cmmd"] = cmmd_val
        if verbose and cmmd_val is not None:
            print(f"  CMMD: {cmmd_val:.4f}")

    # DINO similarity
    if "dino_similarity" in requested and ref_dir:
        dino_mod = dino_model or "facebook/dinov2-base"
        dino_val = compute_dino_similarity(image_paths, ref_dir, device, dino_mod)
        results["metrics"]["dino_similarity"] = dino_val
        if verbose and dino_val is not None:
            print(f"  DINO similarity: {dino_val:.4f}")

    # CLIP-Score
    if "clip_score" in requested:
        clip_val, key = compute_clip_score(
            image_paths, gen_dir, clip_model, device
        )
        results["metrics"][key] = clip_val
        if verbose and clip_val is not None:
            label = "CLIP Score (text-image)" if key == "clip_score" else "CLIP Feature Mean Norm"
            print(f"  {label}: {clip_val:.4f}")

    # LPIPS
    if "lpips" in requested and ref_dir:
        lpips_val = compute_lpips(image_paths, ref_dir, device)
        results["metrics"]["lpips"] = lpips_val
        if verbose and lpips_val is not None:
            print(f"  LPIPS: {lpips_val:.4f}")

    # PSNR / SSIM
    for name in ["psnr", "ssim"]:
        if name in requested and ref_dir:
            if name == "psnr":
                val = compute_psnr(image_paths, ref_dir)
            else:
                val = compute_ssim(image_paths, ref_dir)
            results["metrics"][name] = val
            if verbose and val is not None:
                print(f"  {name.upper()}: {val:.4f}")

    if out_file:
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"  Results saved to: {out_file}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSGen-8k image quality metrics "
        "(FID, KID, CMMD, DINO similarity, CLIP-Score, LPIPS, PSNR, SSIM)"
    )
    parser.add_argument("--generated_dir", required=True, help="Directory of generated images")
    parser.add_argument(
        "--reference_dir",
        default=None,
        help="Reference images (for FID, KID, CMMD, DINO, LPIPS, PSNR, SSIM)",
    )
    parser.add_argument("--output_file", default=None, help="Output JSON path")
    parser.add_argument(
        "--metrics",
        default="fid clip_score",
        help="Space-separated: fid kid cmmd dino_similarity clip_score lpips psnr ssim",
    )
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--clip_model",
        default="ViT-B/32",
        help="CLIP for CLIP-Score (HF ID or local path, e.g. /path/to/clip)",
    )
    parser.add_argument(
        "--cmmd_clip_model",
        default=None,
        help="CLIP for CMMD (HF ID or local path). Default: openai/clip-vit-large-patch14-336",
    )
    parser.add_argument(
        "--dino_model",
        default=None,
        help="DINOv2 for DINO similarity (HF ID or local path). Default: facebook/dinov2-base",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress metric prints")

    args = parser.parse_args()

    out_file = args.output_file or os.path.join(args.generated_dir, "eval_results.json")
    metrics_list = args.metrics.split()

    evaluate_directory(
        gen_dir=args.generated_dir,
        ref_dir=args.reference_dir,
        out_file=out_file,
        metrics=metrics_list,
        device=args.device,
        batch_size=args.batch_size,
        clip_model=args.clip_model,
        cmmd_clip_model=args.cmmd_clip_model,
        dino_model=args.dino_model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
