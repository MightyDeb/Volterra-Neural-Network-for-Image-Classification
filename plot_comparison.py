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

from models import build_model
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

    # Minimum 2 samples per class for all 7 classes = 14 samples
    base_per_class = max(1, target_count // num_classes)
    counts = {c: min(base_per_class, len(class_indices[c])) for c in unique_classes}
    remaining = target_count - sum(counts.values())

    # Distribute remainder to largest classes
    sorted_by_size = sorted(unique_classes, key=lambda c: len(class_indices[c]), reverse=True)
    idx = 0
    while remaining > 0 and idx < len(sorted_by_size):
        c = sorted_by_size[idx % len(sorted_by_size)]
        if counts[c] < len(class_indices[c]):
            counts[c] += 1
            remaining -= 1
        idx += 1

    selected = []
    # Collect deterministically class by class
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
                               out_dir: str = "interpretability_plots",
                               log_file: str = "plot_comparison.log") -> List[Dict[str, any]]:
    """
    Generates qualitative interpretability figures covering all 7 DermaMNIST classes
    and logs all prediction statistics to disk.
    """
    logger = setup_logger("PlotComparison", log_file)
    logger.info("=" * 80)
    logger.info("STARTING QUALITATIVE XAI SALIENCY PLOT GENERATION")
    logger.info(f"Checkpoints: CNN='{cnn_ckpt}', ViT='{vit_ckpt}', VNN='{vnn_ckpt}'")
    logger.info(f"Target Output Directory: '{out_dir}', Total Plots Requested: {num_plots}")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, num_classes, in_channels = get_dataset(img_size=32)

    # Build models
    cnn = build_model("cnn", num_classes=7, in_channels=3, img_size=32).to(device)
    vit = build_model("vit", num_classes=7, in_channels=3, img_size=32).to(device)
    vnn = build_model("vnn", num_classes=7, in_channels=3, img_size=32).to(device)

    for m, name, ckpt in [(cnn, "CNN", cnn_ckpt), (vit, "ViT", vit_ckpt), (vnn, "VNN", vnn_ckpt)]:
        if ckpt and os.path.exists(ckpt):
            m.load_state_dict(torch.load(ckpt, map_location=device))
            logger.info(f"Loaded {name} weights from {ckpt}")
        else:
            logger.warning(f"Checkpoint '{ckpt}' not found. Using initialized weights for {name}.")
        m.eval()

    # Determine indices to plot
    if specific_index is not None:
        indices = [min(specific_index, len(test_set) - 1)]
    else:
        indices = select_all_classes_sample_indices(test_set, target_count=num_plots, random_seed=42)

    logger.info(f"Selected {len(indices)} test samples spanning classes: {indices}")

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
    parser.add_argument("--out_dir", type=str, default="interpretability_plots", help="Directory for saved plots")
    parser.add_argument("--log_file", type=str, default="plot_comparison.log", help="Path to output log file")
    args = parser.parse_args()

    generate_qualitative_plots(
        cnn_ckpt=args.cnn_ckpt,
        vit_ckpt=args.vit_ckpt,
        vnn_ckpt=args.vnn_ckpt,
        num_plots=args.num_plots,
        specific_index=args.image_index,
        out_dir=args.out_dir,
        log_file=args.log_file,
    )
