# Volterra Neural Network for Image Classification & Explainable AI (XAI)

[![GitHub](https://img.shields.io/badge/GitHub-MightyDeb-blue.svg)](https://github.com/MightyDeb/Volterra-Neural-Network-for-Image-Classification)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A rigorous comparative evaluation framework contrasting **Volterra Neural Networks (VNN)**, **Convolutional Neural Networks (CNN)**, and **Vision Transformers (ViT)** across medical benchmarks (DermaMNIST, PathMNIST) and natural images (CIFAR-10), extended with **quantitative Explainable AI (XAI)** metrics, **deep representation analysis (CKA, t-SNE, Silhouette)**, and **computational efficiency profiling**.

---

## 📁 Repository Structure

| File | Description |
| :--- | :--- |
| [`models.py`](file:///models.py) | Full model definitions (`SimpleCNN`, `SimpleVNN`, `SimpleViT`, `VolterraConv2d`) with feature extractors & intermediate activation hooks |
| [`train.py`](file:///train.py) | Training & validation pipeline with medical metrics (Macro Recall/Sensitivity, Precision, F1, ROC-AUC, PR-AUC) and checkpointing |
| [`viz.py`](file:///viz.py) | XAI visualization engine for CNN (Grad-CAM), ViT (Attention Rollout), and VNN (Volterra 2nd-order Pairwise Map) |
| [`plot_comparison.py`](file:///plot_comparison.py) | Generates side-by-side 4-panel interpretability comparison figures (`interpretability_comparison.png`) |
| [`evaluate_xai.py`](file:///evaluate_xai.py) | Quantitative XAI benchmark suite: Deletion AUC, Insertion AUC, and Perturbation Stability (SSIM, Pearson, IoU) |
| [`analyze_representations.py`](file:///analyze_representations.py) | Latent space analysis: CKA similarity matrix, t-SNE clustering, Silhouette scores, and Volterra branch energy ratios |
| [`benchmark_efficiency.py`](file:///benchmark_efficiency.py) | Computational complexity profiler: Parameters, FLOPs/MACs, inference latency (ms), throughput (FPS), and memory |
| [`results.csv`](file:///results.csv) | Experiment log tracking parameters, training time, accuracy, and clinical metrics |
| [`README_XAI_Medical_VNN_CNN_ViT.md`](file:///README_XAI_Medical_VNN_CNN_ViT.md) | Full publication-grade research specification & mathematical formulations |

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Model Training
Train CNN, VNN, and ViT models on DermaMNIST or CIFAR-10:
```bash
# Medical Dataset (DermaMNIST - 7 classes)
python train.py --dataset dermamnist --model cnn --epochs 20
python train.py --model vnn --dataset dermamnist --epochs 20
python train.py --model vit --dataset dermamnist --epochs 20

# Natural Dataset (CIFAR-10 - 10 classes)
python train.py --dataset cifar10 --model cnn --epochs 20
python train.py --model vnn --dataset cifar10 --epochs 20
python train.py --model vit --dataset cifar10 --epochs 20
```

### 3. Qualitative XAI Visualization
Generate a side-by-side 4-panel comparison figure (`interpretability_comparison.png`):
```bash
python plot_comparison.py --cnn_ckpt cnn.pt --vit_ckpt vit.pt --vnn_ckpt vnn.pt --dataset dermamnist --image_index 0
```

### 4. Quantitative XAI Benchmarking
Evaluate Deletion AUC, Insertion AUC, and Explanation Stability (SSIM, IoU):
```bash
python evaluate_xai.py --cnn_ckpt cnn.pt --vnn_ckpt vnn.pt --vit_ckpt vit.pt --dataset dermamnist --num_samples 30
```

### 5. Deep Representation & CKA Analysis
Compute cross-model CKA similarity, Silhouette scores, and t-SNE latent embeddings:
```bash
python analyze_representations.py --cnn_ckpt cnn.pt --vnn_ckpt vnn.pt --vit_ckpt vit.pt --dataset dermamnist
```

### 6. Computational Efficiency Profiling
Measure parameters, FLOPs/MACs, latency, and throughput:
```bash
python benchmark_efficiency.py
```

---

## 🔬 Model Comparison Overview

- **CNN**: Standard baseline with $3 \times 3$ convolutional filters, BatchNorm, and ReLU activation. Explainability via **Grad-CAM**.
- **VNN**: Second-order polynomial filter with Rank-$R$ branch factorization:
  $$y = (W_{\text{lin}} * x) + \sum_{q=1}^R \Big((A_q * x) \odot (B_q * x)\Big)$$
  Activation-free representation. Explainability via **Pairwise Volterra Interaction Maps**.
- **ViT**: Vision Transformer with patch projection and multi-head self-attention. Explainability via **Attention Rollout**.
