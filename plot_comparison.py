"""
Qualitative Multi-Model Explainability Comparison Visualizer on DermaMNIST:
Produces a 4-panel side-by-side figure:
  1. Original DermaMNIST Skin Lesion Image with Prediction & Confidence
  2. CNN : Grad-CAM Class Activation Saliency Map Overlay
  3. ViT : Attention Rollout Spatial Influence Map Overlay
  4. VNN : Volterra Pairwise Quadratic Interaction Map Overlay
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import build_model
from viz import generate_saliency_map
from train import get_dataset, DERMAMNIST_CLASSES


def denormalize_image(img_tensor: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> np.ndarray:
    """Denormalizes PyTorch image tensor (1, C, H, W) to [0, 1] RGB numpy array (H, W, C)."""
    img = img_tensor.clone().detach().cpu().squeeze(0)
    for c in range(min(3, img.shape[0])):
        img[c] = img[c] * std[c] + mean[c]
    np_img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    return np_img


def generate_comparison_plot(cnn_ckpt: str = "cnn.pt", vit_ckpt: str = "vit.pt",
                             vnn_ckpt: str = "vnn.pt", image_index: int = 0,
                             output_path: str = "interpretability_comparison.png") -> None:
    """
    Generates and saves the 4-panel interpretability figure comparing CNN, ViT, and VNN on DermaMNIST.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, num_classes, in_channels = get_dataset(img_size=32)

    if image_index >= len(test_set):
        print(f"[Warning] Index {image_index} exceeds dataset size ({len(test_set)}). Using index 0.")
        image_index = 0

    item = test_set[image_index]
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

    # Safely convert true label to integer scalar
    true_label = int(np.asarray(label_raw).squeeze())
    true_class_name = DERMAMNIST_CLASSES[true_label] if 0 <= true_label < len(DERMAMNIST_CLASSES) else str(true_label)

    # Build models
    cnn = build_model("cnn", num_classes=7, in_channels=3, img_size=32).to(device)
    vit = build_model("vit", num_classes=7, in_channels=3, img_size=32).to(device)
    vnn = build_model("vnn", num_classes=7, in_channels=3, img_size=32).to(device)

    # Load weights if available
    for m, ckpt in [(cnn, cnn_ckpt), (vit, vit_ckpt), (vnn, vnn_ckpt)]:
        if ckpt and os.path.exists(ckpt):
            m.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print(f"[Notice] Checkpoint '{ckpt}' not found. Using initialized weights for visualization.")

    cnn.eval()
    vit.eval()
    vnn.eval()

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

    # Plot Figure
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

    plt.suptitle(f"XAI Multi-Architecture Comparison | DermaMNIST Skin Lesions | Sample #{image_index}",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved qualitative comparison figure to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 4-Panel Qualitative Explainability Comparison on DermaMNIST")
    parser.add_argument("--cnn_ckpt", type=str, default="cnn.pt")
    parser.add_argument("--vit_ckpt", type=str, default="vit.pt")
    parser.add_argument("--vnn_ckpt", type=str, default="vnn.pt")
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--out", type=str, default="interpretability_comparison.png")
    args = parser.parse_args()

    generate_comparison_plot(
        cnn_ckpt=args.cnn_ckpt,
        vit_ckpt=args.vit_ckpt,
        vnn_ckpt=args.vnn_ckpt,
        image_index=args.image_index,
        output_path=args.out,
    )
