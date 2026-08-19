# Explainable Medical Image Classification: VNN vs. CNN vs. ViT

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?style=flat&logo=python)](https://python.org)
[![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging%20%26%20XAI-008080.svg)](https://github.com/MightyDeb/Volterra-Neural-Network-for-Image-Classification)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comparative research framework evaluating **Volterra Neural Networks (VNN)**, **Convolutional Neural Networks (CNN)**, and **Vision Transformers (ViT)** for medical image classification, extended with **hyperparameter optimization**, **deep representation analysis (CKA, t-SNE, Silhouette)**, and **quantitative Explainable AI (XAI)** metrics (Deletion/Insertion AUC, Explanation Stability, and Ground-Truth Localization).

> **Core Research Question:**  
> *How do conventional first-order convolutional features (CNN), explicit second-order polynomial kernel interactions (VNN), and multi-head self-attention mechanisms (ViT) differ in predictive performance, computational efficiency, representation geometry, and explainability for medical imaging?*

---

## Table of Contents

- [1. Executive Summary & Research Scope](#1-executive-summary--research-scope)
- [2. Model Architectures & Theoretical Foundations](#2-model-architectures--theoretical-foundations)
  - [2.1 Standard CNN Baseline](#21-standard-cnn-baseline)
  - [2.2 Volterra Neural Network (VNN)](#22-volterra-neural-network-vnn)
  - [2.3 Vision Transformer (ViT)](#23-vision-transformer-vit)
- [3. Medical Datasets & Preprocessing Pipeline](#3-medical-datasets--preprocessing-pipeline)
  - [3.1 Target Benchmarks](#31-target-benchmarks)
  - [3.2 Preprocessing & Data Augmentation](#32-preprocessing--data-augmentation)
- [4. Hyperparameter Optimization & Search Space](#4-hyperparameter-optimization--search-space)
- [5. Classification & Clinical Performance Metrics](#5-classification--clinical-performance-metrics)
- [6. Explainable AI (XAI) Methods](#6-explainable-ai-xai-methods)
  - [6.1 CNN: Grad-CAM](#61-cnn-grad-cam)
  - [6.2 ViT: Attention Rollout](#62-vit-attention-rollout)
  - [6.3 VNN: Volterra Pairwise Interaction Map](#63-vnn-volterra-pairwise-interaction-map)
- [7. Quantitative XAI Evaluation](#7-quantitative-xai-evaluation)
  - [7.1 Deletion Metric (Fidelity / Faithfulness)](#71-deletion-metric-fidelity--faithfulness)
  - [7.2 Insertion Metric (Fidelity / Early Salience)](#72-insertion-metric-fidelity--early-salience)
  - [7.3 Explanation Stability & Robustness](#73-explanation-stability--robustness)
  - [7.4 Ground-Truth Localization & Pointing Game](#74-ground-truth-localization--pointing-game)
- [8. Deep Representation & Latent Space Analysis](#8-deep-representation--latent-space-analysis)
  - [8.1 Intermediate Feature Activations](#81-intermediate-feature-activations)
  - [8.2 Feature-Space Geometry (t-SNE / UMAP / PCA)](#82-feature-space-geometry-t-sne--umap--pca)
  - [8.3 Quantitative Separability (Silhouette Score)](#83-quantitative-separability-silhouette-score)
  - [8.4 Cross-Model Representation Similarity (CKA)](#84-cross-model-representation-similarity-cka)
  - [8.5 Analysis of Volterra Higher-Order Interactions](#85-analysis-of-volterra-higher-order-interactions)
- [9. Computational Efficiency & Complexity Analysis](#9-computational-efficiency--complexity-analysis)
- [10. Unified XAI & Benchmark Evaluation Matrix](#10-unified-xai--benchmark-evaluation-matrix)
- [11. Repository Structure & Execution Guide](#11-repository-structure--execution-guide)

---

## 1. Executive Summary & Research Scope

Medical image classification demands high predictive accuracy alongside **faithful, verifiable explanations**. Deep neural networks often function as black boxes whose internal decision-making processes are difficult for clinicians to interpret.

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                          Comparative Evaluation Space                         │
├───────────────────────┬───────────────────────────────┬───────────────────────┤
│    Convolution (CNN)  │     Volterra Filters (VNN)    │   Self-Attention (ViT)│
│  - Local receptive    │   - Explicit 2nd-order        │   - Global receptive  │
│    fields             │     polynomial interactions   │     fields            │
│  - Translational      │   - Activation-free           │   - Dynamic patch-to- │
│    equivariance       │     nonlinearity              │     patch routing     │
│  - XAI: Grad-CAM      │   - XAI: Pairwise Heatmap     │   - XAI: Attn Rollout │
└───────────────────────┴───────────────────────────────┴───────────────────────┘
```

This project establishes a standardized evaluation protocol across three pillars:
1. **Predictive Performance & Calibration**: Standard and clinical classification metrics.
2. **Representation Topology**: Centered Kernel Alignment (CKA), latent clustering, and feature separability.
3. **Quantitative XAI**: Faithfulness (Deletion/Insertion AUC), explanation stability under perturbation, and localization overlap with clinical ground-truth regions of interest (ROI).

---

## 2. Model Architectures & Theoretical Foundations

### 2.1 Standard CNN Baseline
- **Architecture**: Stack of convolutional blocks with Batch Normalization and ReLU nonlinearities, followed by Global Average Pooling (GAP) and a linear classification head.
- **Inductive Bias**: Strong local connectivity and translation equivariance.
- **Mathematical Form**:
  $$z = f(W * x + b)$$

### 2.2 Volterra Neural Network (VNN)
- **Architecture**: Cascaded layers using discrete Volterra series approximations.
- **Volterra Series Formulation**:
  $$y[m, n] = h_0 + \sum_{i,j} h_1[i, j] x[m-i, n-j] + \sum_{i_1, j_1} \sum_{i_2, j_2} h_2[i_1, j_1, i_2, j_2] x[m-i_1, n-j_1] x[m-i_2, n-j_2] + \dots$$
- **Low-Rank (Rank-$R$) Factorization**:
  To prevent the combinatorial parameter explosion of a full second-order kernel $h_2$, the quadratic term is factorized into $R$ rank-1 separable branch convolutions:
  $$y = (W_{\text{lin}} * x) + \sum_{q=1}^{R} \Big( (A_q * x) \odot (B_q * x) \Big)$$
  where $\odot$ denotes the Hadamard (elementwise) product.
- **Key Property**: Nonlinear representation capability is achieved algebraically without piecewise linear activation functions (activation-free polynomial representation).

### 2.3 Vision Transformer (ViT)
- **Architecture**: Patch embedding layer ($P \times P$ non-overlapping patches), learnable class token `[CLS]`, 1D learnable positional embeddings, and alternating Multi-Head Self-Attention (MHSA) and MLP blocks with LayerNorm.
- **Attention Mechanism**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Inductive Bias**: Minimal spatial inductive bias, learning global relationships across all patch pairs directly from data.

---

## 3. Medical Datasets & Preprocessing Pipeline

### 3.1 Target Benchmarks
- **DermaMNIST** / **HAM10000**: Dermatoscopic lesion classification (7 classes: melanocytic nevi, melanoma, benign keratosis, basal cell carcinoma, actinic keratoses, vascular lesions, dermatofibroma).
- **PathMNIST**: Colon pathology histology tiles (9 tissue types).
- **Chest X-ray / PneumoniaMNIST**: Binary and multi-label thoracic pathology.

> [!IMPORTANT]
> **Data Integrity & Leakage Prevention:**  
> When handling medical image datasets (such as HAM10000 / DermaMNIST), ensure that all lesion images from the same **patient ID** are grouped strictly into the same partition (train, validation, or test) to prevent optimistic bias.

### 3.2 Preprocessing & Data Augmentation

```text
Input Medical Image
       │
       ▼
Resize / Center Crop (32×32 or 224×224)
       │
       ▼
Intensity Scaling & Per-Channel Normalization: x' = (x - μ) / σ
       │
       ├── Training Set ──► Medically Valid Augmentations (Random Flip, Rotation ±15°, Affine)
       │
       └── Validation / Test Set ──► Deterministic Transform (No Random Augmentations)
```

---

## 4. Hyperparameter Optimization & Search Space

A systematic grid/Bayesian search is applied across all architectures to ensure unbiased empirical comparison:

| Parameter Category | Hyperparameter | Search Candidates |
| :--- | :--- | :--- |
| **Optimization** | Optimizer | AdamW, Adam, SGD with Momentum (0.9) |
| | Initial Learning Rate ($\eta$) | $1 \times 10^{-4}, 3 \times 10^{-4}, 1 \times 10^{-3}, 3 \times 10^{-3}$ |
| | Weight Decay | $1 \times 10^{-5}, 1 \times 10^{-4}, 1 \times 10^{-2}$ |
| | LR Scheduler | Cosine Annealing, StepLR, ReduceLROnPlateau |
| **Architecture (CNN)** | Base Channels | 32, 64, 128 |
| | Kernel Size | $3 \times 3, 5 \times 5$ |
| | Dropout Rate | 0.0, 0.1, 0.2 |
| **Architecture (VNN)** | Decomposition Rank ($R$) | 1, 2, 4 |
| | Base Channels | 32, 64, 128 |
| **Architecture (ViT)** | Patch Size ($P$) | $4 \times 4, 8 \times 8$ |
| | Embedding Dim ($D$) | 64, 128, 256 |
| | Transformer Depth / Heads | Depth $\in \{3, 4, 6\}$, Heads $\in \{2, 4, 8\}$ |

---

## 5. Classification & Clinical Performance Metrics

In medical diagnostic tasks, class imbalance is common. Standard overall accuracy must be supplemented with threshold-independent and sensitivity-focused metrics:

- **Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
- **Macro Sensitivity / Recall (Minimizing False Negatives)**:
  $$\text{Recall}_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C}\frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
- **Macro Precision**:
  $$\text{Precision}_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C}\frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$
- **Macro F1-Score**: Harmonic mean of macro precision and recall.
- **Macro One-vs-Rest ROC-AUC & PR-AUC**: Area under the receiver operating characteristic and precision-recall curves.
- **Confusion Matrix**: Full breakdown of misclassification trends between critical malignant and benign categories.

---

## 6. Explainable AI (XAI) Methods

```text
                        Input Image (x)
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
    CNN Model               VNN Model               ViT Model
       │                       │                       │
       ▼                       ▼                       ▼
   Grad-CAM              Volterra Pairwise         Attention
 (Gradient Saliency)    Interaction Heatmap         Rollout
```

### 6.1 CNN: Grad-CAM
Computes class-specific activation weights via pooled gradients flowing into the final convolutional feature maps:
$$\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial Y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k}\alpha_k^c A^k\right)$$

### 6.2 ViT: Attention Rollout
Aggregates self-attention matrices across all $L$ transformer layers to trace information flow from the `[CLS]` token to input spatial patches:
$$A_{\text{aug}}^{(l)} = 0.5 A^{(l)} + 0.5 I, \quad R^{(l)} = A_{\text{aug}}^{(l)} \cdot R^{(l-1)}$$
The resulting rollout vector $R^{(L)}_{0, 1:N}$ maps the influence of each patch onto the classification token.

### 6.3 VNN: Volterra Pairwise Interaction Map
Exploits the intrinsic quadratic structure of `VolterraConv2d` to visualize the spatial distribution of second-order multiplicative feature interactions:
$$M_{\text{Volterra}}[m, n] = \text{ReLU}\left(\sum_{q=1}^{R} \left(A_q * x\right)[m, n] \cdot \left(B_q * x\right)[m, n]\right)$$
Unlike first-order saliency maps, this reflects where non-linear spatial coupling occurs within the input field.

---

## 7. Quantitative XAI Evaluation

Qualitative heatmaps alone are insufficient to confirm explanation validity. We employ rigorous quantitative metrics:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Quantitative XAI Evaluation Suite                     │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│     Deletion Metric     │     Insertion Metric    │  Explanation Stability  │
│  - Remove top saliency  │  - Add top saliency to  │  - Measure map change   │
│    pixels iteratively   │    blurred baseline     │    under small input    │
│  - Lower AUC = better   │  - Higher AUC = better  │    perturbations (SSIM) │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

### 7.1 Deletion Metric (Faithfulness)
1. Sort image pixels in descending order of importance according to explanation map $M$.
2. Iteratively replace the top $k\%$ ($k \in [0, 100]$) pixels with neutral gray/blur values.
3. Compute the model's prediction confidence $p(y|x_k)$ at each step.
4. Calculate **Deletion AUC** (Area Under Curve). A faithful explanation causes a steep drop in confidence (**lower AUC is better**).

### 7.2 Insertion Metric (Early Salience)
1. Initialize a blurred or blank baseline image.
2. Iteratively restore pixels in order of descending importance according to map $M$.
3. Compute model confidence $p(y|x_k)$ at each step.
4. Calculate **Insertion AUC** (**higher AUC is better**).

### 7.3 Explanation Stability & Robustness
Evaluate explanation invariance under imperceptible, medically non-altering input perturbations $x' = x + \delta$ (Gaussian noise $\sigma = 0.01$, mild brightness shift $\pm 5\%$, sub-pixel translation):
- **Structural Similarity Index Measure (SSIM)**: $\text{SSIM}(M(x), M(x'))$
- **Pearson Correlation ($r$)**: Correlation between flattened saliency vectors.
- **Intersection over Union (IoU)** of top $20\%$ salient region masks:
  $$\text{IoU}_{\text{stability}} = \frac{|\mathcal{S}_{0.2}(x) \cap \mathcal{S}_{0.2}(x')|}{|\mathcal{S}_{0.2}(x) \cup \mathcal{S}_{0.2}(x')|}$$

### 7.4 Ground-Truth Localization & Pointing Game
When lesion segmentation masks or bounding boxes are available:
- **Pointing Game Accuracy**: Fraction of test instances where $\arg\max_{(i,j)} M[i,j]$ falls within the ground-truth lesion mask.
- **Salience IoU / Dice**: Overlap between the binarized top-$k\%$ saliency mask and the true clinical segmentation mask.

---

## 8. Deep Representation & Latent Space Analysis

### 8.1 Intermediate Feature Activations
Extract and visualize feature channel statistics across early, middle, and final representation layers to observe pattern selectivity (edges, textures, lesion borders, tissue morphology).

### 8.2 Feature-Space Geometry (t-SNE / UMAP / PCA)
Project penultimate-layer embedding vectors $z \in \mathbb{R}^D$ into 2D manifolds to assess class separability and cluster compactness.

### 8.3 Quantitative Separability (Silhouette Score)
Compute cluster quality in the native latent embedding space:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad S = \frac{1}{N}\sum_{i=1}^{N} s(i)$$
where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance.

### 8.4 Cross-Model Representation Similarity (CKA)
Centered Kernel Alignment (CKA) measures the similarity of representations learned by different architectures across layers:
$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K)\text{HSIC}(L, L)}}$$
where $K = X X^T$ and $L = Y Y^T$ are Gram matrices of layer representations from two models.

### 8.5 Analysis of Volterra Higher-Order Interactions
Measure the relative energy and feature correlation between the linear branch ($W_{\text{lin}} * x$) and the quadratic Volterra interaction branch ($\sum (A_q * x) \odot (B_q * x)$) to quantify how non-linear interactions contribute to classification boundaries.

---

## 9. Computational Efficiency & Complexity Analysis

Evaluate each model across key hardware and deployment metrics:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Computational Benchmark Grid                        │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│  Model Parameters   │  FLOPs / MAC Count  │  Inference Latency (ms/sample)  │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│  Training Time / Ep │ Peak Memory (VRAM)  │  Accuracy per 100k Parameters   │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
```

---

## 10. Unified XAI & Benchmark Evaluation Matrix

Results should be compiled into the standardized benchmark schema below:

### 10.1 Classification Performance
| Model | Params | Macro F1 ↑ | Top-1 Acc ↑ | Macro Recall ↑ | ROC-AUC ↑ | PR-AUC ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SimpleCNN** | ~95K | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **SimpleVNN** | ~281K | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **SimpleViT** | ~546K | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

### 10.2 Quantitative Explainability & Representation
| Model | XAI Method | Deletion AUC ↓ | Insertion AUC ↑ | Stability (SSIM) ↑ | Pointing Game Acc ↑ | Silhouette Score ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN** | Grad-CAM | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **VNN** | Pairwise Map | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **ViT** | Attn Rollout | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

### 10.3 Cross-Architecture CKA Representation Similarity
| Layer Comparison | CNN vs. VNN | CNN vs. ViT | VNN vs. ViT |
| :--- | :--- | :--- | :--- |
| **Early Features** | *TBD* | *TBD* | *TBD* |
| **Mid-level Features** | *TBD* | *TBD* | *TBD* |
| **Penultimate Embeddings** | *TBD* | *TBD* | *TBD* |

---

## 11. Repository Structure & Execution Guide

### 11.1 File Overview
```text
.
├── models.py           # Model definitions (SimpleCNN, SimpleVNN, SimpleViT, VolterraConv2d)
├── train.py            # Training pipeline with checkpointing & results logging
├── viz.py              # XAI visualization engines (Grad-CAM, Attention Rollout, Volterra Map)
├── plot_comparison.py  # Multi-model qualitative comparison figure generator
├── results.csv         # Experiment logging (parameters, accuracy, runtime)
└── README.md           # Quickstart and CIFAR-10 baseline overview
```

### 11.2 Reproducibility & Commands

```bash
# 1. Install dependencies
pip install torch torchvision matplotlib numpy scipy

# 2. Train baseline models (saves checkpoints cnn.pt, vnn.pt, vit.pt)
python train.py --model cnn --epochs 20
python train.py --model vnn --epochs 20
python train.py --model vit --epochs 20

# 3. Generate qualitative 4-panel XAI comparison figure
python plot_comparison.py --cnn_ckpt cnn.pt --vit_ckpt vit.pt --vnn_ckpt vnn.pt --image_index 0
```
