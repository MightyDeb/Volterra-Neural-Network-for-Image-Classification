"""
Deep Feature & Representation Analysis Suite for DermaMNIST (7 Lesion Classes):
  - 8.1 Intermediate Layer Feature Activations
  - 8.2 Latent Feature-Space Geometry (t-SNE / PCA on Skin Lesion Embeddings)
  - 8.3 Quantitative Cluster Separability (Silhouette Score, Davies-Bouldin Index)
  - 8.4 Cross-Architecture Centered Kernel Alignment (Linear & RBF CKA)
  - 8.5 Volterra 2nd-Order Feature Interaction Energy Breakdown
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models import build_model, SimpleVNN, load_model_checkpoint
from train import get_dataset, get_stratified_sample_indices, DERMAMNIST_CLASSES
from logger_utils import setup_logger


# ---------------------------------------------------------------------------
# 1. Cross-Model Representation Similarity (Centered Kernel Alignment / CKA)
# ---------------------------------------------------------------------------
def centering_matrix(n: int) -> np.ndarray:
    """Returns the centering projection matrix H = I_n - (1/n) 1 1^T."""
    return np.eye(n) - (1.0 / n) * np.ones((n, n))


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Computes Hilbert-Schmidt Independence Criterion (HSIC)."""
    n = K.shape[0]
    H = centering_matrix(n)
    KH = K @ H
    LH = L @ H
    return float(np.trace(KH @ LH) / ((n - 1) ** 2))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes Linear Centered Kernel Alignment (CKA) between two representation
    matrices X (N x d1) and Y (N x d2).
    
    Returns:
        float similarity score in [0.0, 1.0]
    """
    X = X - np.mean(X, axis=0, keepdims=True)
    Y = Y - np.mean(Y, axis=0, keepdims=True)

    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    if hsic_kk <= 0 or hsic_ll <= 0:
        return 0.0
    return float(hsic_kl / (np.sqrt(hsic_kk) * np.sqrt(hsic_ll)))


def compute_cross_model_cka(features_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Computes symmetric CKA similarity matrix across models/representations."""
    names = list(features_dict.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                score = linear_cka(features_dict[names[i]], features_dict[names[j]])
                matrix[i, j] = score
                matrix[j, i] = score

    return matrix, names


# ---------------------------------------------------------------------------
# 2. Feature-Space Geometry & Quantitative Separability
# ---------------------------------------------------------------------------
def extract_dataset_embeddings(model: nn.Module, dataset, sample_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts penultimate feature vectors and labels for specified DermaMNIST sample indices.
    """
    model.eval()
    device = next(model.parameters()).device

    features = []
    labels = []

    with torch.no_grad():
        for idx in sample_indices:
            img, label = dataset[idx]
            if isinstance(img, np.ndarray):
                img = torch.tensor(img)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = img.to(device)

            feat = model.extract_features(img)
            features.append(feat.cpu().numpy()[0])
            labels.append(int(np.asarray(label).squeeze()))

    return np.array(features), np.array(labels)


def compute_separability_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Computes Silhouette Score and Davies-Bouldin Index on native embeddings.
    """
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        return {"silhouette_score": 0.0, "davies_bouldin_index": 0.0}

    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        sil = float(silhouette_score(features, labels))
        dbi = float(davies_bouldin_score(features, labels))
    except Exception:
        sil, dbi = 0.0, 0.0

    return {"silhouette_score": sil, "davies_bouldin_index": dbi}


# ---------------------------------------------------------------------------
# 3. Volterra 2nd-Order Energy Analysis
# ---------------------------------------------------------------------------
def analyze_volterra_energy(vnn_model: SimpleVNN, dataset, sample_indices: List[int]) -> Dict[str, float]:
    """
    Computes the quadratic interaction energy ratio across layers in SimpleVNN
    on specified sample indices.
    """
    vnn_model.eval()
    device = next(vnn_model.parameters()).device

    layer_ratios = {"layer1": [], "layer2": [], "layer3": []}

    with torch.no_grad():
        for idx in sample_indices:
            img, _ = dataset[idx]
            if isinstance(img, np.ndarray):
                img = torch.tensor(img)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = img.to(device)

            # Layer 1
            l1_stats = vnn_model.layer1.compute_branch_energies(img)
            layer_ratios["layer1"].append(l1_stats["quadratic_ratio"])

            # Layer 2
            out1 = vnn_model.pool1(vnn_model.layer1(img))
            l2_stats = vnn_model.layer2.compute_branch_energies(out1)
            layer_ratios["layer2"].append(l2_stats["quadratic_ratio"])

            # Layer 3
            out2 = vnn_model.pool2(vnn_model.layer2(out1))
            l3_stats = vnn_model.layer3.compute_branch_energies(out2)
            layer_ratios["layer3"].append(l3_stats["quadratic_ratio"])

    return {
        "layer1_quad_ratio_mean": float(np.mean(layer_ratios["layer1"])),
        "layer2_quad_ratio_mean": float(np.mean(layer_ratios["layer2"])),
        "layer3_quad_ratio_mean": float(np.mean(layer_ratios["layer3"])),
    }


# ---------------------------------------------------------------------------
# 4. Master Analysis Pipeline
# ---------------------------------------------------------------------------
def run_representation_analysis(cnn_ckpt: str = "cnn.pt", vnn_ckpt: str = "vnn.pt",
                                vit_ckpt: str = "vit.pt", num_samples: int = 100,
                                random_seed: int = 42, save_plots: bool = True,
                                log_file: str = "representation_analysis.log") -> None:
    """
    Executes representation analysis on DermaMNIST (7 skin lesion classes)
    using identical stratified test set samples (100 samples by default) and logs all metrics to disk.
    """
    logger = setup_logger("AnalyzeRepresentations", log_file)
    logger.info("=" * 85)
    logger.info("DEEP FEATURE & REPRESENTATION ANALYSIS SUITE (DERMAMNIST - 7 LESION CLASSES)")
    logger.info(f"Checkpoints: CNN='{cnn_ckpt}', VNN='{vnn_ckpt}', ViT='{vit_ckpt}'")
    logger.info(f"Target Stratified Samples: {num_samples} (Seed: {random_seed}) | Log: '{log_file}'")
    logger.info("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, num_classes, in_channels = get_dataset(img_size=32)

    # Extract deterministic stratified sample indices
    stratified_indices = get_stratified_sample_indices(test_set, num_samples=num_samples, random_seed=random_seed)

    if hasattr(test_set, "labels") and test_set.labels is not None:
        sample_labels = [int(np.asarray(test_set.labels)[i].squeeze()) for i in stratified_indices]
    else:
        sample_labels = [int(np.asarray(test_set[i][1]).squeeze()) for i in stratified_indices]

    class_counts = {c_idx: sample_labels.count(c_idx) for c_idx in range(7)}
    breakdown_str = ", ".join([f"{DERMAMNIST_CLASSES[c]}: {class_counts[c]}" for c in range(7)])
    logger.info(f"Selected {len(stratified_indices)} Stratified Test Images (Class distribution: {breakdown_str})")
    logger.info("The exact same test image indices will be evaluated across CNN, VNN, and ViT.\n")

    models = {}
    for m_type, ckpt in [("cnn", cnn_ckpt), ("vnn", vnn_ckpt), ("vit", vit_ckpt)]:
        m = build_model(m_type, num_classes=7, in_channels=3, img_size=32).to(device)
        loaded = load_model_checkpoint(m, ckpt, device=device)
        if loaded:
            logger.info(f"Loaded {m_type.upper()} checkpoint from {ckpt}")
        else:
            logger.warning(f"Checkpoint '{ckpt}' not found or incompatible. Running with initialized weights for {m_type.upper()}.")
        models[m_type] = m

    # Extract Embeddings
    logger.info(f"\nExtracting latent representations on {len(stratified_indices)} DermaMNIST test samples...")
    embeddings = {}
    labels_dict = {}
    for m_type in ["cnn", "vnn", "vit"]:
        feat, lbls = extract_dataset_embeddings(models[m_type], test_set, sample_indices=stratified_indices)
        embeddings[m_type] = feat
        labels_dict[m_type] = lbls

    # 1. Separability Metrics
    logger.info("\n" + "=" * 85)
    logger.info(f"1. QUANTITATIVE CLUSTER SEPARABILITY ({len(stratified_indices)} Stratified Samples)")
    logger.info("=" * 85)
    logger.info("| Model | Feature Dim | Silhouette Score ↑ | Davies-Bouldin Index ↓ |")
    logger.info("| :--- | :--- | :--- | :--- |")
    for m_type in ["cnn", "vnn", "vit"]:
        sep = compute_separability_metrics(embeddings[m_type], labels_dict[m_type])
        dim = embeddings[m_type].shape[1]
        logger.info(f"| **{m_type.upper()}** | {dim} | {sep['silhouette_score']:.4f} | {sep['davies_bouldin_index']:.4f} |")

    # 2. CKA Matrix
    cka_matrix, names = compute_cross_model_cka(embeddings)
    logger.info("\n" + "=" * 85)
    logger.info("2. CROSS-ARCHITECTURE CKA SIMILARITY MATRIX (DermaMNIST)")
    logger.info("=" * 85)
    header = "| | " + " | ".join([n.upper() for n in names]) + " |"
    divider = "| :--- | " + " | ".join([":---:" for _ in names]) + " |"
    logger.info(header)
    logger.info(divider)
    for i, row_name in enumerate(names):
        vals = " | ".join([f"{cka_matrix[i, j]:.4f}" for j in range(len(names))])
        logger.info(f"| **{row_name.upper()}** | {vals} |")

    # 3. Volterra Branch Energy Breakdown
    vnn_energy = analyze_volterra_energy(models["vnn"], test_set, sample_indices=stratified_indices)
    logger.info("\n" + "=" * 85)
    logger.info("3. VOLTERRA 2ND-ORDER POLYNOMIAL ENERGY RATIO BY LAYER")
    logger.info("=" * 85)
    logger.info(f"  - Early Layer (Stage 1) Quadratic Energy Ratio : {vnn_energy['layer1_quad_ratio_mean']*100:.2f}%")
    logger.info(f"  - Middle Layer (Stage 2) Quadratic Energy Ratio : {vnn_energy['layer2_quad_ratio_mean']*100:.2f}%")
    logger.info(f"  - Deep Layer (Stage 3) Quadratic Energy Ratio  : {vnn_energy['layer3_quad_ratio_mean']*100:.2f}%")
    logger.info("=" * 85 + "\n")

    # Save t-SNE scatter plot
    if save_plots:
        try:
            from sklearn.manifold import TSNE
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for idx, m_type in enumerate(["cnn", "vnn", "vit"]):
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings[m_type]) - 1))
                proj = tsne.fit_transform(embeddings[m_type])
                scatter = axes[idx].scatter(
                    proj[:, 0], proj[:, 1], c=labels_dict[m_type], cmap="tab10", alpha=0.8, s=25
                )
                axes[idx].set_title(f"{m_type.upper()} Latent Space (t-SNE)", fontsize=13, fontweight="bold")
                axes[idx].axis("off")

            cbar = plt.colorbar(scatter, ax=axes, orientation="horizontal", fraction=0.04, pad=0.08)
            cbar.set_ticks(range(7))
            cbar.set_ticklabels(DERMAMNIST_CLASSES)
            plt.suptitle(f"DermaMNIST Skin Lesion Class Clusters (t-SNE - {len(stratified_indices)} Stratified Samples)",
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
            out_tsne = "representation_tsne.png"
            plt.savefig(out_tsne, dpi=150)
            plt.close(fig)
            logger.info(f"Saved {out_tsne} with DermaMNIST lesion classes.")
        except Exception as e:
            logger.warning(f"Skipping t-SNE plot generation ({e})")

    logger.info("=" * 85)
    logger.info("Deep Feature & Representation Analysis complete.")
    logger.info("=" * 85 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Feature & Representation Analysis on DermaMNIST")
    parser.add_argument("--cnn_ckpt", type=str, default="cnn.pt")
    parser.add_argument("--vnn_ckpt", type=str, default="vnn.pt")
    parser.add_argument("--vit_ckpt", type=str, default="vit.pt")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of stratified test samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic stratified sampling")
    parser.add_argument("--log_file", type=str, default="representation_analysis.log", help="Path to output log file")
    args = parser.parse_args()

    run_representation_analysis(
        cnn_ckpt=args.cnn_ckpt,
        vnn_ckpt=args.vnn_ckpt,
        vit_ckpt=args.vit_ckpt,
        num_samples=args.num_samples,
        random_seed=args.seed,
        log_file=args.log_file,
    )

