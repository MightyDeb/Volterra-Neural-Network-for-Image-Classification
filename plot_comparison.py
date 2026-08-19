"""
Qualitative Multi-Model Explainability Comparison Visualizer on DermaMNIST:
Produces 4-panel side-by-side figures:
  1. Original DermaMNIST Skin Lesion Image with True Class Label
  2. CNN : Grad-CAM Class Activation Saliency Map Overlay
  3. ViT : Attention Rollout Spatial Influence Map Overlay
  4. VNN : Volterra Pairwise Quadratic Interaction Map Overlay
Supports generating 15 qualitative plots covering all 7 classes and logging to disk.
"""

import argparse
import os
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import build_model, load_model_checkpoint
from viz import generate_saliency_map
from train import get_dataset, DERMAMNIST_CLASSES
from logger_utils import setup_logger


def denormalize_image(img_tensor: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> np.ndarray:
    """Denormalizes PyTorch image tensor (1, C, H, W) to [0, 1] RGB numpy array (H, W, C)."""
    img = img_tensor.clone().detach().cpu().squeeze(0)
    for c in range(min(3, img.shape[0])):
        img[c] = img[c] * std[c] + mean[c]
    np_img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    return np_img


def select_high_confidence_multiclass_indices(
    cnn, vit, vnn, dataset, target_count: int = 15,
    min_conf: float = 0.85, correct_only: bool = True,
    device: torch.device = None, random_seed: int = 42,
    logger = None
) -> List[int]:
    """
    Scans the test dataset and selects 15 samples spanning all 7 classes where the model
    makes correct predictions with high confidence (>= min_conf, default 85%).
    If a rare class has no sample reaching 85%, it gracefully selects the highest-confidence
    available correct samples for that class.
    """
    if device is None:
        device = next(vnn.parameters()).device

    if hasattr(dataset, "labels") and dataset.labels is not None:
        all_labels = np.asarray(dataset.labels).squeeze()
    else:
        all_labels = np.array([int(np.asarray(dataset[i][1]).squeeze()) for i in range(len(dataset))])

    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)
    class_candidates = {c: [] for c in unique_classes}

    # Evaluate test dataset samples
    with torch.no_grad():
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            if isinstance(img, np.ndarray):
                img = torch.tensor(img)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = img.to(device)
            true_c = int(np.asarray(lbl).squeeze())

            # Evaluate with models (ensemble probability for robustness)
            out_cnn = torch.softmax(cnn(img), dim=1)[0]
            out_vnn = torch.softmax(vnn(img), dim=1)[0]
            out_vit = torch.softmax(vit(img), dim=1)[0]

            # Primary model confidence (VNN / ensemble)
            ens_prob = (out_cnn + out_vnn + out_vit) / 3.0
            pred_c = int(ens_prob.argmax().item())
            conf = float(ens_prob[true_c].item())

            if not correct_only or (pred_c == true_c):
                class_candidates[true_c].append((i, conf))

    # Sort each class's candidates by confidence descending
    for c in unique_classes:
        class_candidates[c].sort(key=lambda x: x[1], reverse=True)

    # Allocate target counts (at least 2 per class = 14 + remainder for dominant class)
    base_per_class = max(1, target_count // num_classes)
    counts = {c: base_per_class for c in unique_classes}
    remaining = target_count - sum(counts.values())

    sorted_by_size = sorted(unique_classes, key=lambda c: len(np.where(all_labels == c)[0]), reverse=True)
    idx = 0
    while remaining > 0 and idx < len(sorted_by_size):
        c = sorted_by_size[idx % len(sorted_by_size)]
        counts[c] += 1
        remaining -= 1
        idx += 1

    selected_indices = []
    for c in sorted(unique_classes):
        cands = class_candidates[c]
        class_name = DERMAMNIST_CLASSES[c] if 0 <= c < len(DERMAMNIST_CLASSES) else f"class_{c}"
        
        # Filter for candidates >= min_conf
        high_conf = [s_idx for s_idx, conf in cands if conf >= min_conf]
        if len(high_conf) >= counts[c]:
            chosen = high_conf[:counts[c]]
            if logger:
                confs_str = ", ".join([f"{conf*100:.1f}%" for _, conf in cands[:counts[c]]])
                logger.info(f"Class '{class_name.upper()}': Selected {len(chosen)} samples reaching >={min_conf*100:.0f}% confidence ({confs_str})")
        elif len(cands) >= counts[c]:
            chosen = [s_idx for s_idx, _ in cands[:counts[c]]]
            if logger:
                confs_str = ", ".join([f"{conf*100:.1f}%" for _, conf in cands[:counts[c]]])
                logger.info(f"Class '{class_name.upper()}': Selected top {len(chosen)} highest-confidence available samples ({confs_str})")
        else:
            chosen = list(np.where(all_labels == c)[0])[:counts[c]]
            if logger:
                logger.info(f"Class '{class_name.upper()}': Fallback to {len(chosen)} default class samples")

        selected_indices.extend(chosen)

    return selected_indices[:target_count]


def select_all_classes_sample_indices(dataset, target_count: int = 15, random_seed: int = 42) -> List[int]:
    """
    Selects 15 test sample indices ensuring all 7 skin lesion classes are represented
    (at least 2 samples per class, with the remainder from dominant classes).
    """
    if hasattr(dataset, "labels") and dataset.labels is not None:
        all_labels = np.asarray(dataset.labels).squeeze()
    else:
        all_labels = np.array([int(np.asarray(dataset[i][1]).squeeze()) for i in range(len(dataset))])

    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)
    rng = np.random.RandomState(random_seed)

    class_indices = {c: np.where(all_labels == c)[0] for c in unique_classes}
    for c in unique_classes:
        rng.shuffle(class_indices[c])

    base_per_class = max(1, target_count // num_classes)
    counts = {c: min(base_per_class, len(class_indices[c])) for c in unique_classes}
    remaining = target_count - sum(counts.values())

    sorted_by_size = sorted(unique_classes, key=lambda c: len(class_indices[c]), reverse=True)
    idx = 0
    while remaining > 0 and idx < len(sorted_by_size):
        c = sorted_by_size[idx % len(sorted_by_size)]
        if counts[c] < len(class_indices[c]):
            counts[c] += 1
            remaining -= 1
        idx += 1

    selected = []
    for c in sorted(unique_classes):
        selected.extend(class_indices[c][:counts[c]])

    return selected[:target_count]


def generate_single_plot(cnn, vit, vnn, dataset, image_index: int,
                         output_path: str, device: torch.device) -> Dict[str, any]:
    """
    Generates and saves a single 4-panel interpretability figure for test sample `image_index`.
    """
    item = dataset[image_index]
    img_raw, label_raw = item[0], item[1]

    if isinstance(img_raw, np.ndarray):
        img_tensor = torch.tensor(img_raw)
    elif isinstance(img_raw, torch.Tensor):
        img_tensor = img_raw.clone()
    else:
        img_tensor = torch.tensor(img_raw)

    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    true_label = int(np.asarray(label_raw).squeeze())
    true_class_name = DERMAMNIST_CLASSES[true_label] if 0 <= true_label < len(DERMAMNIST_CLASSES) else str(true_label)

    # Get predictions and probabilities
    with torch.no_grad():
        out_cnn = torch.softmax(cnn(img_tensor), dim=1)[0]
        out_vit = torch.softmax(vit(img_tensor), dim=1)[0]
        out_vnn = torch.softmax(vnn(img_tensor), dim=1)[0]

    pred_cnn = int(out_cnn.argmax().item())
    conf_cnn = float(out_cnn.max().item())
    pred_vit = int(out_vit.argmax().item())
    conf_vit = float(out_vit.max().item())
    pred_vnn = int(out_vnn.argmax().item())
    conf_vnn = float(out_vnn.max().item())

    name_cnn = DERMAMNIST_CLASSES[pred_cnn] if 0 <= pred_cnn < len(DERMAMNIST_CLASSES) else str(pred_cnn)
    name_vit = DERMAMNIST_CLASSES[pred_vit] if 0 <= pred_vit < len(DERMAMNIST_CLASSES) else str(pred_vit)
    name_vnn = DERMAMNIST_CLASSES[pred_vnn] if 0 <= pred_vnn < len(DERMAMNIST_CLASSES) else str(pred_vnn)

    # Generate Saliency Maps
    cam = generate_saliency_map(cnn, "cnn", img_tensor, target_class=pred_cnn)
    rollout = generate_saliency_map(vit, "vit", img_tensor)
    vmap = generate_saliency_map(vnn, "vnn", img_tensor)

    # Denormalize image for display
    orig_img = denormalize_image(img_tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # Plot 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Panel 1: Original Image
    axes[0].imshow(orig_img)
    axes[0].set_title(f"DermaMNIST Lesion\nTrue: {true_class_name} (#{true_label})", fontsize=12, fontweight="bold")

    # Panel 2: CNN Grad-CAM
    axes[1].imshow(orig_img)
    im1 = axes[1].imshow(cam, cmap="jet", alpha=0.5)
    axes[1].set_title(f"CNN: Grad-CAM\nPred: {name_cnn} ({conf_cnn*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: ViT Attention Rollout
    axes[2].imshow(orig_img)
    im2 = axes[2].imshow(rollout, cmap="viridis", alpha=0.55)
    axes[2].set_title(f"ViT: Attention Rollout\nPred: {name_vit} ({conf_vit*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: VNN Pairwise Interaction Map
    axes[3].imshow(orig_img)
    im3 = axes[3].imshow(vmap, cmap="inferno", alpha=0.55)
    axes[3].set_title(f"VNN: 2nd-Order Volterra Map\nPred: {name_vnn} ({conf_vnn*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"XAI Multi-Architecture Comparison | DermaMNIST #{image_index} (True: {true_class_name})",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "index": image_index,
        "true_label": true_label,
        "true_class": true_class_name,
        "pred_cnn": name_cnn,
        "conf_cnn": conf_cnn,
        "pred_vit": name_vit,
        "conf_vit": conf_vit,
        "pred_vnn": name_vnn,
        "conf_vnn": conf_vnn,
        "output_path": output_path,
    }


def generate_qualitative_plots(cnn_ckpt: str = "cnn.pt", vit_ckpt: str = "vit.pt",
                               vnn_ckpt: str = "vnn.pt", num_plots: int = 15,
                               specific_index: int = None,
                               min_conf: float = 0.85,
                               correct_only: bool = True,
                               out_dir: str = "interpretability_plots",
                               log_file: str = "plot_comparison.log") -> List[Dict[str, any]]:
    """
    Generates qualitative interpretability figures covering all 7 DermaMNIST classes
    with optional high-confidence filtering (>= min_conf, default 85%) and logs all statistics to disk.
    """
    logger = setup_logger("PlotComparison", log_file)
    logger.info("=" * 80)
    logger.info("STARTING QUALITATIVE XAI SALIENCY PLOT GENERATION")
    logger.info(f"Checkpoints: CNN='{cnn_ckpt}', ViT='{vit_ckpt}', VNN='{vnn_ckpt}'")
    logger.info(f"Target Output Directory: '{out_dir}', Total Plots Requested: {num_plots}")
    logger.info(f"High-Confidence Filter: >={min_conf*100:.0f}% (Correct Class Predictions Only: {correct_only})")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, num_classes, in_channels = get_dataset(img_size=32)

    # Build models
    cnn = build_model("cnn", num_classes=7, in_channels=3, img_size=32).to(device)
    vit = build_model("vit", num_classes=7, in_channels=3, img_size=32).to(device)
    vnn = build_model("vnn", num_classes=7, in_channels=3, img_size=32).to(device)

    for m, name, ckpt in [(cnn, "CNN", cnn_ckpt), (vit, "ViT", vit_ckpt), (vnn, "VNN", vnn_ckpt)]:
        loaded = load_model_checkpoint(m, ckpt, device=device)
        if loaded:
            logger.info(f"Loaded {name} weights from {ckpt}")
        else:
            logger.warning(f"Checkpoint '{ckpt}' not found or incompatible. Using initialized weights for {name}.")
        m.eval()

    # Determine indices to plot
    if specific_index is not None:
        indices = [min(specific_index, len(test_set) - 1)]
    elif min_conf > 0.0:
        logger.info(f"Filtering test set for high-confidence samples (>={min_conf*100:.0f}%)...")
        indices = select_high_confidence_multiclass_indices(
            cnn=cnn, vit=vit, vnn=vnn, dataset=test_set, target_count=num_plots,
            min_conf=min_conf, correct_only=correct_only, device=device, logger=logger
        )
    else:
        indices = select_all_classes_sample_indices(test_set, target_count=num_plots, random_seed=42)

    logger.info(f"Final selected {len(indices)} test sample indices: {indices}")

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for plot_no, idx in enumerate(indices, start=1):
        item = test_set[idx]
        true_lbl = int(np.asarray(item[1]).squeeze())
        class_name = DERMAMNIST_CLASSES[true_lbl] if 0 <= true_lbl < len(DERMAMNIST_CLASSES) else f"class_{true_lbl}"

        filename = f"interpretability_plot_{plot_no:02d}_sample_{idx}_class_{class_name}.png"
        out_path = os.path.join(out_dir, filename)

        res = generate_single_plot(cnn, vit, vnn, test_set, idx, out_path, device)
        results.append(res)

        logger.info(
            f"Plot {plot_no:02d}/{len(indices):02d} [Sample #{idx:04d} | Class: {class_name.upper()}]: "
            f"CNN={res['pred_cnn']} ({res['conf_cnn']*100:.1f}%), "
            f"ViT={res['pred_vit']} ({res['conf_vit']*100:.1f}%), "
            f"VNN={res['pred_vnn']} ({res['conf_vnn']*100:.1f}%) -> Saved: {out_path}"
        )

    # For top-level compatibility, also save default sample as interpretability_comparison.png
    if len(results) > 0:
        default_out = "interpretability_comparison.png"
        generate_single_plot(cnn, vit, vnn, test_set, indices[0], default_out, device)
        logger.info(f"Saved default overview comparison figure to: {default_out}")

    logger.info("=" * 80)
    logger.info(f"Successfully generated all {len(results)} qualitative interpretability plots.")
    logger.info("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Multi-Class Qualitative Explainability Comparison Plots")
    parser.add_argument("--cnn_ckpt", type=str, default="cnn.pt")
    parser.add_argument("--vit_ckpt", type=str, default="vit.pt")
    parser.add_argument("--vnn_ckpt", type=str, default="vnn.pt")
    parser.add_argument("--num_plots", type=int, default=15, help="Number of qualitative plots covering all classes")
    parser.add_argument("--image_index", type=int, default=None, help="Optional specific single image index")
    parser.add_argument("--min_conf", type=float, default=0.85, help="Minimum prediction confidence threshold (e.g. 0.85)")
    parser.add_argument("--correct_only", action="store_true", default=True, help="Only select correct predictions")
    parser.add_argument("--out_dir", type=str, default="interpretability_plots", help="Directory for saved plots")
    parser.add_argument("--log_file", type=str, default="plot_comparison.log", help="Path to output log file")
    args = parser.parse_args()

    generate_qualitative_plots(
        cnn_ckpt=args.cnn_ckpt,
        vit_ckpt=args.vit_ckpt,
        vnn_ckpt=args.vnn_ckpt,
        num_plots=args.num_plots,
        specific_index=args.image_index,
        min_conf=args.min_conf,
        correct_only=args.correct_only,
        out_dir=args.out_dir,
        log_file=args.log_file,
    )
