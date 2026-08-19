"""
VNN Feature Map Analysis:
Inspects learned representations in the 2nd-order Volterra filter layers of SimpleVNN.
1. Extracts spatial feature maps [N, C, H, W] from the final Volterra layer before pooling.
2. Identifies the strongest channels via global average spatial activation across the dataset.
3. For each top channel, discovers the top-10 maximally activating images and plots:
   - top_images.png: 2x5 grid of highest-activating skin lesion images
   - activation_maps.png: 2x5 grid of localized spatial activation heatmaps overlaid on the images
Outputs saved in feature_analysis/channel_{channel_id}/
"""

import argparse
import os
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model, load_model_checkpoint
from train import get_dataset, DERMAMNIST_CLASSES
from logger_utils import setup_logger


def denormalize_image(img_tensor: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> np.ndarray:
    """Denormalizes PyTorch image tensor (C, H, W) to [0, 1] RGB numpy array (H, W, C)."""
    img = img_tensor.clone().detach().cpu()
    if img.ndim == 4:
        img = img.squeeze(0)
    for c in range(min(3, img.shape[0])):
        img[c] = img[c] * std[c] + mean[c]
    np_img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    return np_img


def analyze_vnn_features(
    vnn_ckpt: str = "vnn.pt",
    top_k_channels: int = 5,
    top_n_images: int = 10,
    out_dir: str = "feature_analysis",
    log_file: str = "vnn_feature_analysis.log",
    batch_size: int = 128
) -> Dict[str, any]:
    """
    Executes feature analysis on trained SimpleVNN model.
    """
    logger = setup_logger("VNNFeatureAnalysis", log_file)
    logger.info("=" * 80)
    logger.info("STARTING VNN FEATURE MAP & CHANNEL SPECIALIZATION ANALYSIS")
    logger.info(f"VNN Checkpoint: '{vnn_ckpt}' | Top Channels: {top_k_channels} | Top Images/Channel: {top_n_images}")
    logger.info(f"Target Output Directory: '{out_dir}'")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, num_classes, in_channels = get_dataset(img_size=32)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 1. Build and Load VNN Model
    vnn = build_model("vnn", num_classes=7, in_channels=3, img_size=32).to(device)
    loaded = load_model_checkpoint(vnn, vnn_ckpt, device=device)
    if loaded:
        logger.info(f"Successfully loaded trained VNN weights from {vnn_ckpt}")
    else:
        logger.warning(f"Checkpoint '{vnn_ckpt}' not found or incompatible. Using initialized weights.")
    vnn.eval()

    # 2. Extract Feature Maps from Last Volterra Layer
    logger.info(f"\nStep 1: Extracting spatial feature maps from last Volterra layer across {len(test_set)} test images...")
    all_features = []
    all_images = []
    all_labels = []
    all_preds = []
    all_confs = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            if isinstance(y, torch.Tensor) and y.ndim > 1:
                y = y.squeeze(-1)

            logits, feat_maps = vnn.forward_with_features(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confs = probs.max(dim=1).values

            all_features.append(feat_maps.cpu())
            all_images.append(x.cpu())
            all_labels.append(y.cpu() if isinstance(y, torch.Tensor) else torch.tensor(y))
            all_preds.append(preds.cpu())
            all_confs.append(confs.cpu())

    features = torch.cat(all_features, dim=0)  # Shape: [N, C, H, W]
    images = torch.cat(all_images, dim=0)      # Shape: [N, 3, 32, 32]
    labels = torch.cat(all_labels, dim=0).flatten()
    preds = torch.cat(all_preds, dim=0).flatten()
    confs = torch.cat(all_confs, dim=0).flatten()

    N, C, H, W = features.shape
    logger.info(f"Extracted Feature Tensor Shape: [N={N}, C={C}, H={H}, W={W}]")

    # 3. Find Strongest Channels
    # Average each feature map spatially: [N, C]
    channel_activation = features.mean(dim=(2, 3))
    # Average across images: [C]
    mean_activation = channel_activation.mean(dim=0)

    # Sort channels descending by mean activation
    sorted_channel_indices = torch.argsort(mean_activation, descending=True)
    selected_channels = sorted_channel_indices[:top_k_channels].tolist()

    logger.info(f"\nStep 2: Ranked All {C} Feature Channels. Top {top_k_channels} Most Active Channels:")
    logger.info("-" * 80)
    logger.info(f"{'Rank':<6} | {'Channel ID':<12} | {'Mean Spatial Activation':<26} | {'Peak Activation':<18}")
    logger.info("-" * 80)
    for rank, ch_id in enumerate(selected_channels, start=1):
        mean_act = mean_activation[ch_id].item()
        max_act = channel_activation[:, ch_id].max().item()
        logger.info(f"#{rank:<5} | Channel {ch_id:<4} | {mean_act:<26.4f} | {max_act:<18.4f}")
    logger.info("-" * 80 + "\n")

    # 4. Process Each Selected Top Channel
    results = {}
    os.makedirs(out_dir, exist_ok=True)

    for rank, ch_id in enumerate(selected_channels, start=1):
        ch_dir = os.path.join(out_dir, f"channel_{ch_id}")
        os.makedirs(ch_dir, exist_ok=True)

        values = channel_activation[:, ch_id]
        top_indices = torch.argsort(values, descending=True)[:top_n_images].tolist()

        logger.info(f"Processing Top Channel #{rank} (Channel {ch_id}):")
        logger.info(f"  Saving artifacts to: {ch_dir}")

        # Collect metadata for top images
        top_meta = []
        for rank_img, img_idx in enumerate(top_indices, start=1):
            true_l = int(labels[img_idx].item())
            pred_l = int(preds[img_idx].item())
            conf_val = float(confs[img_idx].item())
            act_val = float(values[img_idx].item())
            true_name = DERMAMNIST_CLASSES[true_l] if 0 <= true_l < len(DERMAMNIST_CLASSES) else str(true_l)
            pred_name = DERMAMNIST_CLASSES[pred_l] if 0 <= pred_l < len(DERMAMNIST_CLASSES) else str(pred_l)
            top_meta.append({
                "index": img_idx,
                "true_label": true_name,
                "pred_label": pred_name,
                "confidence": conf_val,
                "activation": act_val,
            })
            logger.info(f"    Top #{rank_img:02d} | Sample #{img_idx:04d} | Class: {true_name.upper():<6} | "
                        f"Pred: {pred_name.upper():<6} ({conf_val*100:.1f}%) | Activation: {act_val:.4f}")

        # -------------------------------------------------------------------
        # Visualization 1: top_images.png (2 rows x 5 cols grid)
        # -------------------------------------------------------------------
        fig_imgs, axes_imgs = plt.subplots(2, 5, figsize=(16, 6.5))
        axes_imgs = axes_imgs.flatten()

        for i, img_idx in enumerate(top_indices):
            ax = axes_imgs[i]
            img_rgb = denormalize_image(images[img_idx])
            ax.imshow(img_rgb)
            meta = top_meta[i]
            title = f"#{i+1}: {meta['true_label'].upper()}\nAct: {meta['activation']:.2f} | Conf: {meta['confidence']*100:.0f}%"
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.suptitle(f"VNN Feature Channel {ch_id} (Rank #{rank}) | Top {top_n_images} Maximally Activating Images",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        top_images_path = os.path.join(ch_dir, "top_images.png")
        plt.savefig(top_images_path, dpi=200, bbox_inches="tight")
        plt.close(fig_imgs)

        # -------------------------------------------------------------------
        # Visualization 2: activation_maps.png (2 rows x 5 cols heatmap overlays)
        # -------------------------------------------------------------------
        fig_maps, axes_maps = plt.subplots(2, 5, figsize=(16, 6.5))
        axes_maps = axes_maps.flatten()

        for i, img_idx in enumerate(top_indices):
            ax = axes_maps[i]
            img_rgb = denormalize_image(images[img_idx])
            
            # Extract 2D activation map for this channel: [H, W]
            raw_map = features[img_idx, ch_id].detach().numpy()
            
            # Normalize map to [0, 1]
            map_min, map_max = raw_map.min(), raw_map.max()
            if map_max - map_min > 1e-8:
                norm_map = (raw_map - map_min) / (map_max - map_min)
            else:
                norm_map = np.zeros_like(raw_map)

            # Upsample map to 32x32 image size
            norm_map_tensor = torch.tensor(norm_map).unsqueeze(0).unsqueeze(0)
            upsampled_map = F.interpolate(norm_map_tensor, size=(32, 32), mode="bilinear", align_corners=False)[0, 0].numpy()

            ax.imshow(img_rgb)
            im = ax.imshow(upsampled_map, cmap="inferno", alpha=0.6)
            meta = top_meta[i]
            title = f"#{i+1}: {meta['true_label'].upper()} Map\nPeak: {map_max:.2f}"
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.suptitle(f"VNN Feature Channel {ch_id} (Rank #{rank}) | Localized Spatial 2nd-Order Activation Maps",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        act_maps_path = os.path.join(ch_dir, "activation_maps.png")
        plt.savefig(act_maps_path, dpi=200, bbox_inches="tight")
        plt.close(fig_maps)

        logger.info(f"  -> Generated: {top_images_path}")
        logger.info(f"  -> Generated: {act_maps_path}\n")

        results[f"channel_{ch_id}"] = {
            "rank": rank,
            "channel_id": ch_id,
            "mean_activation": mean_activation[ch_id].item(),
            "top_images_path": top_images_path,
            "activation_maps_path": act_maps_path,
            "top_samples": top_meta,
        }

    # Summary table in log
    logger.info("=" * 80)
    logger.info("FEATURE ANALYSIS SUMMARY TABLE:")
    logger.info("=" * 80)
    logger.info("| Rank | Channel ID | Mean Activation | Dominant Class Responded | Artifact Folder |")
    logger.info("|:---|:---|:---|:---|:---|")
    for ch_key, info in results.items():
        classes_in_top = [m["true_label"] for m in info["top_samples"][:5]]
        dominant_class = max(set(classes_in_top), key=classes_in_top.count)
        logger.info(f"| #{info['rank']} | Channel {info['channel_id']} | {info['mean_activation']:.4f} | {dominant_class.upper()} | `feature_analysis/channel_{info['channel_id']}/` |")
    logger.info("=" * 80)
    logger.info("VNN Feature Analysis Complete.\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze VNN Learned 2nd-Order Feature Maps & Channels")
    parser.add_argument("--vnn_ckpt", type=str, default="vnn.pt", help="Path to trained VNN checkpoint")
    parser.add_argument("--top_channels", type=int, default=5, help="Number of strongest channels to analyze (e.g. 3-5)")
    parser.add_argument("--top_images", type=int, default=10, help="Number of top activating images per channel")
    parser.add_argument("--out_dir", type=str, default="feature_analysis", help="Output directory for channel subfolders")
    parser.add_argument("--log_file", type=str, default="vnn_feature_analysis.log", help="Log file path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction")
    args = parser.parse_args()

    analyze_vnn_features(
        vnn_ckpt=args.vnn_ckpt,
        top_k_channels=args.top_channels,
        top_n_images=args.top_images,
        out_dir=args.out_dir,
        log_file=args.log_file,
        batch_size=args.batch_size,
    )
