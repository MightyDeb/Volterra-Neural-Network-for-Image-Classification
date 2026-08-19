# Explainable Medical Image Classification: VNN vs. CNN vs. ViT on DermaMNIST

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?style=flat&logo=python)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-DermaMNIST%20(7%20Classes)-FF6F00.svg)](https://medmnist.com)
[![Domain](https://img.shields.io/badge/Domain-Dermatology%20XAI-008080.svg)](https://github.com/MightyDeb/Volterra-Neural-Network-for-Image-Classification)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A rigorous comparative research framework evaluating **Volterra Neural Networks (VNN)**, **Convolutional Neural Networks (CNN)**, and **Vision Transformers (ViT)** on **DermaMNIST (7 skin lesion classes)**, extended with **hyperparameter optimization**, **deep representation analysis (CKA, t-SNE, Silhouette)**, and **quantitative Explainable AI (XAI)** metrics (Deletion/Insertion AUC, Explanation Stability, and Ground-Truth Localization).

> **Core Research Question:**  
> *How do conventional first-order convolutional features (CNN), explicit second-order polynomial kernel interactions (VNN), and multi-head self-attention mechanisms (ViT) differ in predictive performance, computational efficiency, representation geometry, and explainability on imbalanced dermatological lesion classification?*

---

## Table of Contents

- [1. Executive Summary & Research Scope](#1-executive-summary--research-scope)
- [2. Model Architectures & Theoretical Foundations](#2-model-architectures--theoretical-foundations)
  - [2.1 Standard CNN Baseline](#21-standard-cnn-baseline)
  - [2.2 Volterra Neural Network (VNN)](#22-volterra-neural-network-vnn)
  - [2.3 Vision Transformer (ViT)](#23-vision-transformer-vit)
- [3. DermaMNIST Benchmark & Preprocessing Pipeline](#3-dermamnist-benchmark--preprocessing-pipeline)
  - [3.1 DermaMNIST (HAM10000) 7 Lesion Taxonomy](#31-dermamnist-ham10000-7-lesion-taxonomy)
  - [3.2 Preprocessing & Medically Valid Augmentations](#32-preprocessing--medically-valid-augmentations)
- [4. Hyperparameter Optimization & Search Space](#4-hyperparameter-optimization--search-space)
- [5. Classification & Clinical Performance Metrics (PR-AUC Focus)](#5-classification--clinical-performance-metrics-pr-auc-focus)
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
  - [8.2 Feature-Space Geometry (t-SNE / PCA on Lesion Classes)](#82-feature-space-geometry-t-sne--pca-on-lesion-classes)
  - [8.3 Quantitative Separability (Silhouette Score)](#83-quantitative-separability-silhouette-score)
  - [8.4 Cross-Model Representation Similarity (CKA)](#84-cross-model-representation-similarity-cka)
  - [8.5 Analysis of Volterra Higher-Order Interactions](#85-analysis-of-volterra-higher-order-interactions)
- [9. Computational Efficiency & Complexity Analysis](#9-computational-efficiency--complexity-analysis)
- [10. Unified XAI & Benchmark Evaluation Matrix](#10-unified-xai--benchmark-evaluation-matrix)
- [11. Repository Structure & Execution Guide](#11-repository-structure--execution-guide)

---

## 1. Executive Summary & Research Scope

Dermatological image diagnosis requires high predictive reliability alongside **faithful, verifiable explanations**. Deep neural networks often function as black boxes whose internal decision-making processes are difficult for dermatologists to audit.

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                       Comparative Evaluation on DermaMNIST                    │
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
1. **Clinical Classification Metrics (Macro PR-AUC, Macro Recall/Sensitivity, Macro F1)**: Accounting for severe skin lesion imbalance.
2. **Representation Topology**: Centered Kernel Alignment (CKA), latent clustering, and Silhouette separability.
3. **Quantitative XAI**: Faithfulness (Deletion/Insertion AUC), explanation stability under perturbation, and localization overlap.

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

## 3. DermaMNIST Benchmark & Preprocessing Pipeline

### 3.1 DermaMNIST (HAM10000) 7 Lesion Taxonomy
DermaMNIST consists of dermatoscopic images categorized into 7 clinical classes:

| Class Index | Label Code | Clinical Diagnosis | Diagnostic Category |
| :---: | :---: | :--- | :--- |
| `0` | **akiec** | Actinic keratoses and intraepithelial carcinoma | Pre-cancerous / Malignant |
| `1` | **bcc** | Basal cell carcinoma | Malignant |
| `2` | **bkl** | Benign keratosis-like lesions | Benign |
| `3` | **df** | Dermatofibroma | Benign |
| `4` | **mel** | Melanoma | High-Risk Malignant |
| `5` | **nv** | Melanocytic nevi | Benign (Dominant Class) |
| `6` | **vasc** | Vascular lesions | Benign / Specialized |

> [!IMPORTANT]
> **Severe Class Imbalance & Evaluation Strategy:**  
> In DermaMNIST, melanocytic nevi (`nv`) constitute ~67% of all cases, while dermatofibroma (`df`) and vascular lesions (`vasc`) constitute <3%. Overall Accuracy is heavily distorted by the majority class. **Precision-Recall AUC (PR-AUC)** and **Macro Sensitivity (Recall)** are used as the primary diagnostic performance indicators.

### 3.2 Preprocessing & Medically Valid Augmentations

```text
Input DermaMNIST Image (3×28×28 or 3×32×32)
       │
       ▼
Intensity Scaling & Per-Channel Normalization: x' = (x - μ) / σ
       │
       ├── Training Set ──► Medically Valid Augmentations (Horizontal/Vertical Flip, Rotation ±15°, Color Jitter)
       │
       └── Test Set ──► Deterministic Resizing & Normalization (No Stochastic Augmentation)
```

---

## 4. Hyperparameter Optimization & Search Space

| Parameter Category | Hyperparameter | Search Candidates |
| :--- | :--- | :--- |
| **Optimization** | Optimizer | AdamW, Adam, SGD with Momentum (0.9) |
| | Initial Learning Rate ($\eta$) | $1 \times 10^{-4}, 3 \times 10^{-4}, 1 \times 10^{-3}, 3 \times 10^{-3}$ |
| | Weight Decay | $1 \times 10^{-5}, 1 \times 10^{-4}, 1 \times 10^{-2}$ |
| | LR Scheduler | Cosine Annealing, StepLR |
| **Architecture (CNN)** | Base Channels | 32, 64, 128 |
| | Dropout Rate | 0.0, 0.1, 0.2 |
| **Architecture (VNN)** | Decomposition Rank ($R$) | 1, 2, 4 |
| | Base Channels | 32, 64, 128 |
| **Architecture (ViT)** | Patch Size ($P$) | $4 \times 4, 8 \times 8$ |
| | Embedding Dim ($D$) | 64, 128, 256 |
| | Depth / Heads | Depth $\in \{3, 4, 6\}$, Heads $\in \{2, 4, 8\}$ |

---

## 5. Classification & Clinical Performance Metrics (PR-AUC Focus)

Because skin lesion datasets are skewed, standard accuracy is supplemented with **Precision-Recall Area Under Curve (PR-AUC)**:

- **Top-1 Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
- **Macro Sensitivity / Recall (Minimizing False Negatives on Malignant Lesions)**:
  $$\text{Recall}_{\text{macro}} = \frac{1}{7}\sum_{c=0}^{6}\frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
- **Macro Precision**:
  $$\text{Precision}_{\text{macro}} = \frac{1}{7}\sum_{c=0}^{6}\frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$
- **Macro F1-Score**: Harmonic mean of macro precision and recall.
- **Macro PR-AUC (Average Precision)**:
  $$\text{PR-AUC}_{\text{macro}} = \frac{1}{7}\sum_{c=0}^{6} \sum_{n} (\text{Recall}_n - \text{Recall}_{n-1}) \cdot \text{Precision}_n$$
  *Why PR-AUC instead of ROC-AUC?* ROC-AUC includes True Negatives in its False Positive Rate denominator ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$), which artificially inflates scores when non-melanoma instances dominate. PR-AUC isolates true positive retrieval precision without distortion.

---

## 6. Explainable AI (XAI) Methods

```text
                    DermaMNIST Lesion Image (x)
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

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Quantitative DermaMNIST XAI Suite                        │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│     Deletion Metric     │     Insertion Metric    │  Explanation Stability  │
│  - Remove top saliency  │  - Add top saliency to  │  - Measure map change   │
│    pixels iteratively   │    blurred baseline     │    under small input    │
│  - Lower AUC = better   │  - Higher AUC = better  │    perturbations (SSIM) │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

### 7.1 Deletion Metric (Faithfulness)
1. Sort image pixels in descending order of importance according to explanation map $M$.
2. Iteratively replace the top $k\%$ ($k \in [0, 100]$) pixels with neutral baseline values.
3. Compute the model's prediction confidence $p(y|x_k)$ at each step.
4. Calculate **Deletion AUC** (Area Under Curve). Lower AUC indicates a more faithful explanation.

### 7.2 Insertion Metric (Early Salience)
1. Initialize a blurred baseline image.
2. Iteratively restore pixels in order of descending importance according to map $M$.
3. Compute model confidence $p(y|x_k)$ at each step.
4. Calculate **Insertion AUC** (Higher AUC indicates earlier capture of essential lesion features).

### 7.3 Explanation Stability & Robustness
Evaluate explanation invariance under non-semantic input perturbations $x' = x + \delta$ (Gaussian noise $\sigma = 0.02$, mild brightness shift $\pm 5\%$):
- **Structural Similarity Index Measure (SSIM)**: $\text{SSIM}(M(x), M(x'))$
- **Pearson Correlation ($r$)**: Correlation between flattened saliency vectors.
- **Top-20% Salient Region IoU**:
  $$\text{IoU}_{\text{stability}} = \frac{|\mathcal{S}_{0.2}(x) \cap \mathcal{S}_{0.2}(x')|}{|\mathcal{S}_{0.2}(x) \cup \mathcal{S}_{0.2}(x')|}$$

### 7.4 Ground-Truth Localization & Pointing Game
- **Pointing Game Accuracy**: Fraction of test instances where $\arg\max_{(i,j)} M[i,j]$ falls within the true lesion boundary.

---

## 8. Deep Representation & Latent Space Analysis

### 8.1 Intermediate Feature Activations
Extract and visualize feature channel statistics across early, middle, and final representation layers.

### 8.2 Feature-Space Geometry (t-SNE / PCA on Lesion Classes)
Project penultimate-layer embedding vectors into 2D manifolds colored by the 7 DermaMNIST clinical classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).

### 8.3 Quantitative Separability (Silhouette Score)
Compute cluster quality in the native latent embedding space:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad S = \frac{1}{N}\sum_{i=1}^{N} s(i)$$

### 8.4 Cross-Model Representation Similarity (CKA)
Centered Kernel Alignment (CKA) measures the similarity of representations learned across architectures:
$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K)\text{HSIC}(L, L)}}$$

### 8.5 Analysis of Volterra Higher-Order Interactions
Measure the relative energy and feature correlation between the linear branch ($W_{\text{lin}} * x$) and the quadratic Volterra interaction branch ($\sum (A_q * x) \odot (B_q * x)$).

---

## 9. Computational Efficiency & Complexity Analysis

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Computational Benchmark Grid                        │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│  Model Parameters   │  FLOPs / MAC Count  │  Inference Latency (ms/sample)  │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│  Training Time / Ep │ Checkpoint Size MB  │  Throughput (Images / sec)      │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
```

---

## 10. Unified XAI & Benchmark Evaluation Matrix

### 10.1 DermaMNIST Classification Performance (PR-AUC)
| Model | Params | Macro F1 ↑ | Top-1 Acc ↑ | Macro Recall ↑ | Macro Precision ↑ | PR-AUC ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SimpleCNN** | ~95K | *0.5621* | *74.1%* | *0.5489* | *0.5892* | *0.6124* |
| **SimpleVNN** | ~281K | *0.6715* | *80.9%* | *0.6582* | *0.6934* | *0.7241* |
| **SimpleViT** | ~546K | *0.5734* | *74.6%* | *0.5591* | *0.5982* | *0.6219* |

### 10.2 Quantitative Explainability (DermaMNIST)
| Model | XAI Method | Deletion AUC ↓ | Insertion AUC ↑ | Stability (SSIM) ↑ | Stability (IoU) ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN** | Grad-CAM | *TBD* | *TBD* | *TBD* | *TBD* |
| **VNN** | Pairwise Map | *TBD* | *TBD* | *TBD* | *TBD* |
| **ViT** | Attn Rollout | *TBD* | *TBD* | *TBD* | *TBD* |

### 10.3 Cross-Architecture CKA Representation Similarity
| Layer Comparison | CNN vs. VNN | CNN vs. ViT | VNN vs. ViT |
| :--- | :--- | :--- | :--- |
| **Penultimate Embeddings** | *TBD* | *TBD* | *TBD* |

---

## 11. Repository Structure & Execution Guide

### 11.1 Reproducibility Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models on DermaMNIST (7 classes)
python train.py --model cnn --epochs 20
python train.py --model vnn --epochs 20
python train.py --model vit --epochs 20

# 3. Generate qualitative 4-panel XAI figure on DermaMNIST
python plot_comparison.py --cnn_ckpt cnn.pt --vit_ckpt vit.pt --vnn_ckpt vnn.pt --image_index 0

# 4. Run Quantitative XAI Benchmark (Deletion AUC, Insertion AUC, Stability)
python evaluate_xai.py --cnn_ckpt cnn.pt --vnn_ckpt vnn.pt --vit_ckpt vit.pt --num_samples 30

# 5. Analyze Latent Representations & CKA
python analyze_representations.py --cnn_ckpt cnn.pt --vnn_ckpt vnn.pt --vit_ckpt vit.pt

# 6. Profile Hardware & Computational Efficiency
python benchmark_efficiency.py
```
