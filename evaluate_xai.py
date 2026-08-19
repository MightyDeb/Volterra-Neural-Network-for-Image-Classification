"""
Quantitative Explainable AI (XAI) Evaluation Suite on DermaMNIST (7 Lesion Classes):
  - 7.1 Deletion Metric & Deletion AUC (Faithfulness / Evidence Removal)
  - 7.2 Insertion Metric & Insertion AUC (Faithfulness / Evidence Introduction)
  - 7.3 Explanation Stability & Perturbation Invariance (SSIM, Pearson r, IoU)
  - 7.4 Pointing Game & Localization Overlap (Top-k IoU, Pointing Accuracy)

Evaluates CNN (Grad-CAM), VNN (Volterra Pairwise Map), and ViT (Attention Rollout)
across DermaMNIST test batches to produce rigorous quantitative benchmarks.
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from models import build_model, load_model_checkpoint
from viz import generate_saliency_map
from train import get_dataset, get_stratified_sample_indices, DERMAMNIST_CLASSES
from logger_utils import setup_logger


# ---------------------------------------------------------------------------
# 1. Deletion & Insertion Metrics
# ---------------------------------------------------------------------------
def compute_deletion_curve(model: nn.Module, img: torch.Tensor, saliency_map: np.ndarray,
                           steps: int = 10, target_class: int = None) -> Tuple[np.ndarray, float]:
    """
    Progressively removes top-k% salient pixels and records model prediction confidence.
    Computes Deletion AUC using trapezoidal rule (lower AUC = more faithful explanation).
    """
    model.eval()
    device = next(model.parameters()).device
    img = img.to(device)
    if img.ndim == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        orig_out = model(img)
        if target_class is None:
            target_class = orig_out.argmax(dim=1).item()
        base_prob = torch.softmax(orig_out, dim=1)[0, target_class].item()

    h, w = saliency_map.shape
    total_pixels = h * w
    flat_indices = np.argsort(saliency_map.flatten())[::-1]  # Most to least salient

    probabilities = [base_prob]
    percentages = [0.0]

    img_modified = img.clone()
    baseline_val = img.mean().item()

    step_size = total_pixels // steps
    for step in range(1, steps + 1):
        idx_to_mask = flat_indices[: step * step_size]
        mask_2d = np.zeros(total_pixels, dtype=bool)
        mask_2d[idx_to_mask] = True
        mask_2d = torch.tensor(mask_2d.reshape(h, w), device=device, dtype=torch.bool)

        for c in range(img.shape[1]):
            img_modified[0, c][mask_2d] = baseline_val

        with torch.no_grad():
            out = model(img_modified)
            prob = torch.softmax(out, dim=1)[0, target_class].item()
            probabilities.append(prob)
            percentages.append(step / steps)

    deletion_auc = float(np.trapz(probabilities, percentages))
    return np.array(probabilities), deletion_auc


def compute_insertion_curve(model: nn.Module, img: torch.Tensor, saliency_map: np.ndarray,
                            steps: int = 10, target_class: int = None) -> Tuple[np.ndarray, float]:
    """
    Progressively inserts top-k% salient pixels into a blurred baseline and records
    model confidence. Computes Insertion AUC (higher AUC = more faithful explanation).
    """
    model.eval()
    device = next(model.parameters()).device
    img = img.to(device)
    if img.ndim == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        orig_out = model(img)
        if target_class is None:
            target_class = orig_out.argmax(dim=1).item()

    # Create blurred baseline image
    img_np = img[0].detach().cpu().numpy()
    blurred_np = np.zeros_like(img_np)
    for c in range(img_np.shape[0]):
        blurred_np[c] = gaussian_filter(img_np[c], sigma=4.0)
    baseline_img = torch.tensor(blurred_np, device=device).unsqueeze(0)

    with torch.no_grad():
        base_out = model(baseline_img)
        base_prob = torch.softmax(base_out, dim=1)[0, target_class].item()

    h, w = saliency_map.shape
    total_pixels = h * w
    flat_indices = np.argsort(saliency_map.flatten())[::-1]

    probabilities = [base_prob]
    percentages = [0.0]

    step_size = total_pixels // steps
    for step in range(1, steps + 1):
        img_current = baseline_img.clone()
        idx_to_insert = flat_indices[: step * step_size]
        mask_2d = np.zeros(total_pixels, dtype=bool)
        mask_2d[idx_to_insert] = True
        mask_2d = torch.tensor(mask_2d.reshape(h, w), device=device, dtype=torch.bool)

        for c in range(img.shape[1]):
            img_current[0, c][mask_2d] = img[0, c][mask_2d]

        with torch.no_grad():
            out = model(img_current)
            prob = torch.softmax(out, dim=1)[0, target_class].item()
            probabilities.append(prob)
            percentages.append(step / steps)

    insertion_auc = float(np.trapz(probabilities, percentages))
    return np.array(probabilities), insertion_auc


# ---------------------------------------------------------------------------
# 2. Explanation Stability & Robustness
# ---------------------------------------------------------------------------
def compute_ssim_2d(m1: np.ndarray, m2: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) between two normalized 2D saliency maps."""
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    mu1 = np.mean(m1)
    mu2 = np.mean(m2)
    sigma1_sq = np.var(m1)
    sigma2_sq = np.var(m2)
    sigma12 = np.mean((m1 - mu1) * (m2 - mu2))

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim)


def evaluate_explanation_stability(model: nn.Module, model_type: str, img: torch.Tensor,
                                   noise_std: float = 0.02,
                                   top_k_ratio: float = 0.20) -> Dict[str, float]:
    """
    Evaluates explanation stability against small non-semantic perturbations:
      - Structural Similarity (SSIM)
      - Pearson Correlation (r)
      - Salient Region Intersection-over-Union (IoU)
    """
    map_orig = generate_saliency_map(model, model_type, img)

    # Perturbed image (small Gaussian noise)
    noise = torch.randn_like(img) * noise_std
    img_pert = torch.clamp(img + noise, -1.0, 1.0)
    map_pert = generate_saliency_map(model, model_type, img_pert)

    # 1. SSIM
    ssim_val = compute_ssim_2d(map_orig, map_pert)

    # 2. Pearson Correlation
    flat_orig = map_orig.flatten()
    flat_pert = map_pert.flatten()
    if np.std(flat_orig) > 1e-6 and np.std(flat_pert) > 1e-6:
        r_val = float(np.corrcoef(flat_orig, flat_pert)[0, 1])
    else:
        r_val = 1.0

    # 3. Top-k Salient Region IoU
    k_elements = int(len(flat_orig) * top_k_ratio)
    mask_orig = flat_orig >= np.partition(flat_orig, -k_elements)[-k_elements]
    mask_pert = flat_pert >= np.partition(flat_pert, -k_elements)[-k_elements]
    intersection = np.logical_and(mask_orig, mask_pert).sum()
    union = np.logical_or(mask_orig, mask_pert).sum()
    iou_val = float(intersection / max(1, union))

    return {"stability_ssim": ssim_val, "stability_pearson": r_val, "stability_iou": iou_val}


# ---------------------------------------------------------------------------
# 3. Pointing Game & Localization Overlap
# ---------------------------------------------------------------------------
def pointing_game_accuracy(saliency_map: np.ndarray, ground_truth_mask: np.ndarray) -> bool:
    """
    Returns True if the maximum saliency point falls within the ground-truth ROI.
    """
    max_idx = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    return bool(ground_truth_mask[max_idx])


# ---------------------------------------------------------------------------
# 4. Batch Quantitative Evaluation Runner
# ---------------------------------------------------------------------------
def evaluate_model_xai(model: nn.Module, model_type: str, test_dataset,
                       sample_indices: List[int], logger=None) -> Dict[str, float]:
    """
    Runs quantitative XAI evaluation over specified stratified test indices.
    """
    deletion_aucs = []
    insertion_aucs = []
    ssims = []
    pearsons = []
    ious = []

    n_eval = len(sample_indices)
    msg = f"Evaluating XAI Quantitative Metrics for [{model_type.upper()}] on {n_eval} Stratified Test Samples..."
    if logger:
        logger.info(msg)
    else:
        print(msg)

    for step_i, idx in enumerate(sample_indices, start=1):
        item = test_dataset[idx]
        img = item[0]
        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        if img.ndim == 3:
            img = img.unsqueeze(0)

        saliency = generate_saliency_map(model, model_type, img)

        # Deletion & Insertion
        _, del_auc = compute_deletion_curve(model, img, saliency, steps=10)
        _, ins_auc = compute_insertion_curve(model, img, saliency, steps=10)
        deletion_aucs.append(del_auc)
        insertion_aucs.append(ins_auc)

        # Stability
        stab = evaluate_explanation_stability(model, model_type, img)
        ssims.append(stab["stability_ssim"])
        pearsons.append(stab["stability_pearson"])
        ious.append(stab["stability_iou"])

        if step_i % 25 == 0 or step_i == n_eval:
            progress_msg = (
                f"  [{model_type.upper()}] Processed {step_i:03d}/{n_eval:03d} | "
                f"Running Del AUC: {np.mean(deletion_aucs):.4f} | "
                f"Ins AUC: {np.mean(insertion_aucs):.4f} | "
                f"SSIM: {np.mean(ssims):.4f}"
            )
            if logger:
                logger.info(progress_msg)

    results = {
        "deletion_auc_mean": float(np.mean(deletion_aucs)),
        "deletion_auc_std": float(np.std(deletion_aucs)),
        "insertion_auc_mean": float(np.mean(insertion_aucs)),
        "insertion_auc_std": float(np.std(insertion_aucs)),
        "stability_ssim_mean": float(np.mean(ssims)),
        "stability_pearson_mean": float(np.mean(pearsons)),
        "stability_iou_mean": float(np.mean(ious)),
    }
    return results


def run_comprehensive_xai_suite(cnn_ckpt: str = "cnn.pt", vnn_ckpt: str = "vnn.pt",
                                vit_ckpt: str = "vit.pt", num_samples: int = 100,
                                random_seed: int = 42,
                                log_file: str = "xai_evaluation.log") -> None:
    """
    Loads CNN, VNN, and ViT checkpoints and evaluates them on DermaMNIST quantitative XAI metrics
    using identical stratified test set samples (100 samples by default), logging all metrics to disk.
    """
    logger = setup_logger("EvaluateXAI", log_file)
    logger.info("=" * 85)
    logger.info("QUANTITATIVE EXPLAINABLE AI (XAI) EVALUATION BENCHMARK (DERMAMNIST - 7 CLASSES)")
    logger.info(f"Checkpoints: CNN='{cnn_ckpt}', VNN='{vnn_ckpt}', ViT='{vit_ckpt}'")
    logger.info(f"Target Stratified Samples: {num_samples} (Seed: {random_seed}) | Log: '{log_file}'")
    logger.info("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, num_classes, in_channels = get_dataset(img_size=32)

    # Extract deterministic stratified sample indices from the Test set
    stratified_indices = get_stratified_sample_indices(test_set, num_samples=num_samples, random_seed=random_seed)
    
    # Log class breakdown of the sampled 100 test images
    if hasattr(test_set, "labels") and test_set.labels is not None:
        sample_labels = [int(np.asarray(test_set.labels)[i].squeeze()) for i in stratified_indices]
    else:
        sample_labels = [int(np.asarray(test_set[i][1]).squeeze()) for i in stratified_indices]
        
    class_counts = {c_idx: sample_labels.count(c_idx) for c_idx in range(7)}
    breakdown_str = ", ".join([f"{DERMAMNIST_CLASSES[c]}: {class_counts[c]}" for c in range(7)])
    logger.info(f"Selected {len(stratified_indices)} Stratified Test Images (Class distribution: {breakdown_str})")
    logger.info("The exact same test image indices will be evaluated across CNN, VNN, and ViT.\n")

    models_info = [
        ("cnn", cnn_ckpt),
        ("vnn", vnn_ckpt),
        ("vit", vit_ckpt),
    ]

    summary_table = []

    for m_type, ckpt_path in models_info:
        model = build_model(m_type, num_classes=7, in_channels=3, img_size=32).to(device)
        loaded = load_model_checkpoint(model, ckpt_path, device=device)
        if loaded:
            logger.info(f"Loaded {m_type.upper()} checkpoint from {ckpt_path}")
        else:
            logger.warning(f"Checkpoint '{ckpt_path}' not found or incompatible. Using initialized weights for {m_type.upper()}.")
        model.eval()

        res = evaluate_model_xai(model, m_type, test_set, stratified_indices, logger=logger)
        xai_method = {"cnn": "Grad-CAM", "vnn": "Pairwise Volterra Map", "vit": "Attention Rollout"}[m_type]
        summary_table.append({
            "Model": m_type.upper(),
            "XAI Method": xai_method,
            "Deletion AUC ↓": f"{res['deletion_auc_mean']:.4f} ± {res['deletion_auc_std']:.3f}",
            "Insertion AUC ↑": f"{res['insertion_auc_mean']:.4f} ± {res['insertion_auc_std']:.3f}",
            "Stability SSIM ↑": f"{res['stability_ssim_mean']:.4f}",
            "Stability IoU ↑": f"{res['stability_iou_mean']:.4f}",
            "Stability Pearson ↑": f"{res['stability_pearson_mean']:.4f}",
        })

    # Log & Print Markdown Table
    logger.info("\n" + "=" * 85)
    logger.info(f"FINAL QUANTITATIVE RESULTS TABLE (GFM Format - {len(stratified_indices)} Stratified Samples)")
    logger.info("=" * 85)
    logger.info("| Model | XAI Method | Deletion AUC ↓ | Insertion AUC ↑ | Stability (SSIM) ↑ | Stability (IoU) ↑ | Stability (Pearson) ↑ |")
    logger.info("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for row in summary_table:
        logger.info(
            f"| **{row['Model']}** | {row['XAI Method']} | {row['Deletion AUC ↓']} | "
            f"{row['Insertion AUC ↑']} | {row['Stability SSIM ↑']} | {row['Stability IoU ↑']} | {row['Stability Pearson ↑']} |"
        )
    logger.info("=" * 85 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Quantitative XAI Metrics on DermaMNIST")
    parser.add_argument("--cnn_ckpt", type=str, default="cnn.pt")
    parser.add_argument("--vnn_ckpt", type=str, default="vnn.pt")
    parser.add_argument("--vit_ckpt", type=str, default="vit.pt")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of stratified test samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic stratified sampling")
    parser.add_argument("--log_file", type=str, default="xai_evaluation.log", help="Path to output log file")
    args = parser.parse_args()

    run_comprehensive_xai_suite(
        cnn_ckpt=args.cnn_ckpt,
        vnn_ckpt=args.vnn_ckpt,
        vit_ckpt=args.vit_ckpt,
        num_samples=args.num_samples,
        random_seed=args.seed,
        log_file=args.log_file,
    )

