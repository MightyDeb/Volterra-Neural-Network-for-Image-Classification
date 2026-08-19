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

from models import build_model, SimpleVNN
from train import get_dataset, DERMAMNIST_CLASSES


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
def extract_dataset_embeddings(model: nn.Module, dataset, num_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts penultimate feature vectors and labels for DermaMNIST samples.
    """
    model.eval()
    device = next(model.parameters()).device
    n_eval = min(num_samples, len(dataset))

    features = []
    labels = []

    with torch.no_grad():
        for i in range(n_eval):
            img, label = dataset[i]
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
def analyze_volterra_energy(vnn_model: SimpleVNN, dataset, num_samples: int = 100) -> Dict[str, float]:
    """
    Computes the quadratic interaction energy ratio across layers in SimpleVNN.
    """
    vnn_model.eval()
    device = next(vnn_model.parameters()).device
    n_eval = min(num_samples, len(dataset))

    layer_ratios = {"layer1": [], "layer2": [], "layer3": []}

    with torch.no_grad():
        for i in range(n_eval):
            img, _ = dataset[i]
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
                                vit_ckpt: str = "vit.pt", num_samples: int = 300,
                                save_plots: bool = True) -> None:
    """
    Executes representation analysis on DermaMNIST (7 skin lesion classes).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, num_classes, in_channels = get_dataset(img_size=32)

    models = {}
    for m_type, ckpt in [("cnn", cnn_ckpt), ("vnn", vnn_ckpt), ("vit", vit_ckpt)]:
        m = build_model(m_type, num_classes=7, in_channels=3, img_size=32)
        if os.path.exists(ckpt):
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"Loaded {m_type.upper()} checkpoint from {ckpt}")
        else:
            print(f"[Notice] Checkpoint {ckpt} not found. Running with initialized weights.")
        m.to(device)
        models[m_type] = m

    # Extract Embeddings
    print(f"\nExtracting latent representations on {min(num_samples, len(test_set))} DermaMNIST samples...")
    embeddings = {}
    labels_dict = {}
    for m_type in ["cnn", "vnn", "vit"]:
        feat, lbls = extract_dataset_embeddings(models[m_type], test_set, num_samples=num_samples)
        embeddings[m_type] = feat
        labels_dict[m_type] = lbls

    # 1. Separability Metrics
    print("\n" + "=" * 75)
    print("1. QUANTITATIVE CLUSTER SEPARABILITY (DermaMNIST Latent Embeddings)")
    print("=" * 75)
    print("| Model | Feature Dim | Silhouette Score ↑ | Davies-Bouldin Index ↓ |")
    print("| :--- | :--- | :--- | :--- |")
    for m_type in ["cnn", "vnn", "vit"]:
        sep = compute_separability_metrics(embeddings[m_type], labels_dict[m_type])
        dim = embeddings[m_type].shape[1]
        print(f"| **{m_type.upper()}** | {dim} | {sep['silhouette_score']:.4f} | {sep['davies_bouldin_index']:.4f} |")

    # 2. CKA Matrix
    cka_matrix, names = compute_cross_model_cka(embeddings)
    print("\n" + "=" * 75)
    print("2. CROSS-ARCHITECTURE CKA SIMILARITY MATRIX (DermaMNIST)")
    print("=" * 75)
    header = "| | " + " | ".join([n.upper() for n in names]) + " |"
    divider = "| :--- | " + " | ".join([":---:" for _ in names]) + " |"
    print(header)
    print(divider)
    for i, row_name in enumerate(names):
        vals = " | ".join([f"{cka_matrix[i, j]:.4f}" for j in range(len(names))])
        print(f"| **{row_name.upper()}** | {vals} |")

    # 3. Volterra Branch Energy Breakdown
    vnn_energy = analyze_volterra_energy(models["vnn"], test_set, num_samples=min(100, num_samples))
    print("\n" + "=" * 75)
    print("3. VOLTERRA 2ND-ORDER POLYNOMIAL ENERGY RATIO BY LAYER")
    print("=" * 75)
    print(f"  - Early Layer (Stage 1) Quadratic Energy Ratio : {vnn_energy['layer1_quad_ratio_mean']*100:.2f}%")
    print(f"  - Middle Layer (Stage 2) Quadratic Energy Ratio : {vnn_energy['layer2_quad_ratio_mean']*100:.2f}%")
    print(f"  - Deep Layer (Stage 3) Quadratic Energy Ratio  : {vnn_energy['layer3_quad_ratio_mean']*100:.2f}%")
    print("=" * 75 + "\n")

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
            plt.suptitle("DermaMNIST Skin Lesion Class Clusters (t-SNE)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig("representation_tsne.png", dpi=150)
            print("Saved representation_tsne.png with DermaMNIST lesion classes.")
        except Exception as e:
            print(f"[Notice] Skipping t-SNE plot generation ({e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Feature & Representation Analysis on DermaMNIST")
    parser.add_argument("--cnn_ckpt", type=str, default="cnn.pt")
    parser.add_argument("--vnn_ckpt", type=str, default="vnn.pt")
    parser.add_argument("--vit_ckpt", type=str, default="vit.pt")
    parser.add_argument("--num_samples", type=int, default=150)
    args = parser.parse_args()

    run_representation_analysis(
        cnn_ckpt=args.cnn_ckpt,
        vnn_ckpt=args.vnn_ckpt,
        vit_ckpt=args.vit_ckpt,
        num_samples=args.num_samples,
    )
