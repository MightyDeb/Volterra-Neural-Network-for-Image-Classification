"""
Produces a single side-by-side figure: original image, CNN Grad-CAM,
ViT attention rollout, VNN pairwise interaction map.

Usage (after training all three models with train.py and saving checkpoints):
    python plot_comparison.py --cnn_ckpt cnn.pt --vit_ckpt vit.pt --vnn_ckpt vnn.pt

If no checkpoints are given, runs with randomly initialized weights just to
verify the figure layout (NOT meaningful results -- for real results, save
model.state_dict() at the end of train.py and load them here).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from models import SimpleCNN, SimpleViT, SimpleVNN
from viz import gradcam_cnn, attention_rollout_vit, volterra_pairwise_map


def denormalize(img_tensor, mean, std):
    img = img_tensor.clone().squeeze(0)
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


def main(cnn_ckpt=None, vit_ckpt=None, vnn_ckpt=None, image_index=0):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
    img, label = test_set[image_index]
    img = img.unsqueeze(0)

    cnn = SimpleCNN()
    vit = SimpleViT()
    vnn = SimpleVNN()
    if cnn_ckpt:
        cnn.load_state_dict(torch.load(cnn_ckpt, map_location="cpu"))
    if vit_ckpt:
        vit.load_state_dict(torch.load(vit_ckpt, map_location="cpu"))
    if vnn_ckpt:
        vnn.load_state_dict(torch.load(vnn_ckpt, map_location="cpu"))

    cam = gradcam_cnn(cnn, img)
    rollout = attention_rollout_vit(vit, img)
    vmap = volterra_pairwise_map(vnn.features[0], img)

    orig = denormalize(img, mean, std)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(orig)
    axes[0].set_title(f"Original (label={label})")

    axes[1].imshow(orig)
    axes[1].imshow(cam, cmap="jet", alpha=0.5)
    axes[1].set_title("CNN: Grad-CAM")

    axes[2].imshow(rollout, cmap="viridis")
    axes[2].set_title("ViT: Attention Rollout\n(8x8 patch grid)")

    axes[3].imshow(orig)
    axes[3].imshow(vmap, cmap="jet", alpha=0.5)
    axes[3].set_title("VNN: Pairwise Interaction Map\n(2nd-order Volterra term)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("interpretability_comparison.png", dpi=150)
    print("Saved interpretability_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn_ckpt", type=str, default=None)
    parser.add_argument("--vit_ckpt", type=str, default=None)
    parser.add_argument("--vnn_ckpt", type=str, default=None)
    parser.add_argument("--image_index", type=int, default=0)
    args = parser.parse_args()
    main(args.cnn_ckpt, args.vit_ckpt, args.vnn_ckpt, args.image_index)
