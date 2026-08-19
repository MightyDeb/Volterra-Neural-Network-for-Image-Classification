# Interpretability Comparison of Volterra Neural Networks, CNNs, and Vision Transformers on Skin Lesion Classification

**Author**: Debapriya Maity
**Dataset**: DermaMNIST (7-class skin lesion classification, MedMNIST collection)
**Date**: August 2026

---

## Abstract

This report compares three architecturally distinct image classifiers — a
standard Convolutional Neural Network (CNN), a Volterra Neural Network
(VNN), and a Vision Transformer (ViT) — on the DermaMNIST skin lesion
dataset, with a focus on **explainability** rather than raw accuracy. Each
model's explanation method (Grad-CAM for CNN, Attention Rollout for ViT,
and a novel Pairwise Volterra Interaction Map for VNN) is evaluated both
qualitatively (visual inspection) and quantitatively (Deletion/Insertion
AUC, explanation stability under perturbation). Results are reported across
two sampling conditions — an initial class-imbalanced test set and a
corrected class-balanced test set — to check whether findings hold up under
a more rigorous evaluation. The central finding is that **ViT's attention
maps localize the most decision-relevant pixels most precisely, while both
ViT and VNN produce explanations that are substantially more stable under
small input perturbations than CNN's Grad-CAM**, and this pattern holds
across both sampling conditions.

---

## 1. Introduction & Motivation

Deep learning models used in medical image classification are often
"black boxes" — highly accurate, but offering no insight into *why* a
particular diagnosis was predicted. This is a genuine problem in a clinical
context: a clinician cannot trust, or safely override, a prediction they
cannot inspect.

This project compares three ways of building that inspection window:

- **Grad-CAM** (for CNNs) — a widely-used post-hoc method that highlights
  which pixels most influenced the prediction, using gradients flowing
  back through the final convolutional layer.
- **Attention Rollout** (for ViTs) — traces which image patches the
  transformer's classification token attended to most, aggregated across
  all layers.
- **Pairwise Volterra Interaction Map** (for VNNs) — the novel
  contribution of this project. VNNs replace a standard convolution with a
  low-rank second-order Volterra filter, which explicitly models
  *pairwise* pixel interactions (`a(x) · b(x)`) rather than only
  single-pixel responses. Visualizing this term shows *where the model's
  learned pixel-pair correlations are strongest* — a genuinely different
  kind of explanation from single-pixel saliency or patch attention.

The goal is not to declare one architecture universally "best," but to
characterize the trade-offs each explanation method makes.

---

## 2. Methodology

### 2.1 Dataset
**DermaMNIST**, a 7-class skin lesion classification benchmark from the
MedMNIST collection: `akiec` (actinic keratosis), `bcc` (basal cell
carcinoma), `bkl` (benign keratosis), `df` (dermatofibroma), `mel`
(melanoma), `nv` (melanocytic nevus), `vasc` (vascular lesion).

Two evaluation sets were used:
- **Run 1 (class-imbalanced)**: 100 test samples reflecting the dataset's
  natural class frequencies (`nv: 68, mel: 11, bkl: 11, bcc: 5, akiec: 3,
  df: 1, vasc: 1`).
- **Run 2 (class-balanced)**: 100 test samples stratified to roughly
  14-15 per class, correcting the skew above so no single class dominates
  the evaluation.

Reporting both runs lets us check whether findings are genuine patterns or
artifacts of the dominant `nv` class in Run 1.

### 2.2 Models
All three models were trained on identical data splits for a fair
comparison:
- **CNN**: standard 3-layer convolutional network (Conv+BatchNorm+ReLU
  blocks).
- **VNN**: same overall depth, but each convolution is replaced with a
  low-rank second-order Volterra layer: `y = (linear conv) + Σ (a_q(x) ·
  b_q(x))`, with no activation function needed since the nonlinearity is
  built into the polynomial term itself.
- **ViT**: minimal Vision Transformer with patch embedding and multi-head
  self-attention blocks.

### 2.3 Quantitative XAI Metrics
- **Deletion AUC** (lower is better): removes the pixels an explanation
  marks "most important" and measures how quickly prediction confidence
  collapses. A fast collapse means the explanation correctly identified
  decision-critical pixels.
- **Insertion AUC** (higher is better): the inverse test — starts blank,
  adds back "important" pixels first, measures how quickly confidence
  builds up.
- **Stability (SSIM, IoU, Pearson — higher is better)**: measures whether
  the explanation stays consistent when the input image is very slightly
  perturbed. A trustworthy explanation should not change drastically from
  imperceptible noise.

### 2.4 Representation Analysis Metrics
- **t-SNE**: compresses each model's 128-dimensional internal feature
  vector down to 2D for visualization, to see whether same-class images
  cluster together.
- **Silhouette Score** (higher/more positive is better) and
  **Davies-Bouldin Index** (lower is better): numeric measures of how
  well-separated the learned clusters are.
- **CKA (Centered Kernel Alignment)**: measures how similar two models'
  internal representations are to each other (0 = unrelated, 1 =
  identical).
- **Volterra 2nd-Order Energy Ratio**: VNN-specific — measures what
  fraction of a Volterra layer's output comes from the nonlinear pairwise
  term vs. the plain linear term, tracked across early/middle/deep layers.

---

## 3. Results

### 3.1 Quantitative XAI Results

**Run 1 — Class-Imbalanced (100 samples, `nv`-dominated)**

| Model | XAI Method | Deletion AUC ↓ | Insertion AUC ↑ | SSIM ↑ | IoU ↑ | Pearson ↑ |
|---|---|---|---|---|---|---|
| CNN | Grad-CAM | 0.6515 | 0.7813 | 0.7323 | 0.7777 | 0.9067 |
| VNN | Pairwise Volterra Map | 0.5196 | 0.7391 | 0.9965 | 0.9008 | 0.9968 |
| ViT | Attention Rollout | **0.4026** | 0.7885 | 0.9959 | **0.9465** | 0.9977 |

**Run 2 — Class-Balanced (100 samples, ~14-15 per class)**

| Model | XAI Method | Deletion AUC ↓ | Insertion AUC ↑ | SSIM ↑ | IoU ↑ | Pearson ↑ |
|---|---|---|---|---|---|---|
| CNN | Grad-CAM | 0.3119 | 0.6144 | 0.8557 | 0.7907 | 0.9216 |
| VNN | Pairwise Volterra Map | 0.3250 | 0.5725 | 0.9947 | 0.8880 | 0.9949 |
| ViT | Attention Rollout | **0.2129** | **0.7129** | 0.9939 | **0.9321** | 0.9966 |

**Key observation**: ViT has the best (lowest) Deletion AUC and best IoU
stability in *both* runs. CNN's Grad-CAM has consistently and substantially
lower SSIM/Pearson stability than VNN or ViT in both runs (~0.73-0.86 vs.
~0.99). This consistency across two independently sampled test sets is
strong evidence these are genuine architectural properties, not artifacts
of one lucky/unlucky sample draw.

### 3.2 Representation Analysis Results

**Run 1 (Imbalanced)**

| Model | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|
| CNN | -0.0487 | 1.8812 |
| VNN | -0.0885 | 1.8681 |
| ViT | **0.0747** | **1.5256** |

**Run 2 (Balanced)**

| Model | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|
| CNN | 0.0159 | 3.0339 |
| VNN | 0.0065 | 2.8702 |
| ViT | **0.0484** | **2.6275** |

ViT produces the best-separated latent representations in both runs, though
all three models' Silhouette scores are close to zero — indicating none of
the three achieve strongly separated clusters on this small, low-resolution
dataset. The Davies-Bouldin increase from Run 1 to Run 2 across all models
reflects a harder, more honest evaluation once the dominant `nv` class no
longer inflates apparent cluster tightness.

**CKA Cross-Architecture Similarity**

| | Run 1 (Imbalanced) | Run 2 (Balanced) |
|---|---|---|
| CNN ↔ VNN | 0.7902 | 0.7847 |
| CNN ↔ ViT | 0.7276 | 0.5287 |
| VNN ↔ ViT | 0.6417 | 0.5915 |

CNN and VNN remain consistently similar to each other (~0.78-0.79) across
both runs — expected, since VNN is fundamentally a convolutional
architecture with an added nonlinear term. ViT's representations are
consistently the most distinct from both, and this gap widens under
balanced sampling.

**Volterra 2nd-Order Energy Ratio by Layer**

| Layer | Run 1 (Imbalanced) | Run 2 (Balanced) |
|---|---|---|
| Early (Stage 1) | 32.21% | 27.94% |
| Middle (Stage 2) | 64.88% | 59.72% |
| Deep (Stage 3) | 80.05% | 67.55% |

The trend — deeper layers rely increasingly on the nonlinear pairwise term
rather than the plain linear filter — holds in both runs, though the
balanced run gives somewhat more conservative (lower) magnitudes. This
suggests VNNs learn to lean more heavily on higher-order feature
interactions as representations become more abstract.

### 3.3 t-SNE Visualization (Balanced Run)

![t-SNE latent space comparison](assets/representation_tsne.png)

All three models show substantial intermixing of classes in their latent
space, consistent with the near-zero Silhouette scores above. This is not
surprising given DermaMNIST's low resolution (28×28) and genuine visual
similarity between several lesion categories (see qualitative examples
below).

### 3.4 Qualitative Interpretability Examples (Balanced Run)

Representative examples, one per class, all correctly classified by all
three models with reasonably high confidence:

**akiec** (actinic keratosis)
![akiec example](assets/example_akiec.png)

**bcc** (basal cell carcinoma)
![bcc example](assets/example_bcc.png)

**bkl** (benign keratosis)
![bkl example](assets/example_bkl.png)

**df** (dermatofibroma)
![df example](assets/example_df.png)

**nv** (melanocytic nevus)
![nv example](assets/example_nv.png)

**vasc** (vascular lesion)
![vasc example](assets/example_vasc.png)

Across these successful cases, a consistent visual pattern emerges:
Grad-CAM tends to produce one large, smooth, single "hot blob" roughly
centered on the lesion. Attention Rollout tends to be sparser, sometimes
picking out a small sub-region within the lesion. The Volterra map is
visually the most different of the three — it frequently highlights the
lesion's **boundary and surrounding texture**, consistent with it capturing
pairwise pixel-interaction (edge/texture) information rather than a single
smooth region of interest.

**A notable failure/success divergence case**:

![mel failure case](assets/example_mel_failure_case.png)

On this melanoma (`mel`) sample, both CNN (predicting `vasc` at 100%
confidence) and ViT (predicting `vasc` at 98.5% confidence) misclassify the
lesion — while VNN correctly predicts `mel` at 98.9% confidence. This is a
single anecdote and should not be over-generalized, but it is a concrete,
illustrative case worth discussing: melanoma and vascular lesions can share
visual features (both often present as dark, roughly circular regions),
and this example suggests the Volterra pairwise term may be picking up on
a distinguishing texture cue the other two architectures missed on this
particular image.

---

## 4. Discussion & Inferences

1. **No single architecture "wins" outright** — each offers a different
   trade-off. ViT localizes the most precisely (best Deletion AUC, best
   IoU). VNN and ViT are both dramatically more stable under input
   perturbation than CNN's Grad-CAM. This is a more nuanced and more
   credible finding than a one-sided "VNN beats everything" claim would
   have been.

2. **The Volterra pairwise map is a genuinely different kind of
   explanation**, not just a lower-resolution Grad-CAM. Its tendency to
   highlight edges/boundaries rather than a single smooth blob is
   consistent with its underlying mathematics: it visualizes pairwise
   pixel-interaction strength, which naturally concentrates where
   neighboring pixels vary sharply (i.e., edges and textures).

3. **Robustness of findings across sampling conditions** is the strongest
   part of this report's evidence. Because the Deletion AUC ranking
   (ViT < CNN ≈ VNN) and the stability ranking (VNN ≈ ViT >> CNN) both hold
   under two independently drawn test sets — one badly imbalanced, one
   properly balanced — these are unlikely to be artifacts of a particular
   sample draw.

4. **Deeper Volterra layers lean more on nonlinear interactions.** The
   consistent early→middle→deep increase in quadratic energy ratio
   (~28%→60%→68% in the balanced run) suggests the network learns to
   depend more on higher-order feature interactions as representations
   become more abstract — an interpretable, testable claim distinguishing
   VNNs from standard CNNs.

---

## 5. Limitations

- **Dataset scale and resolution**: DermaMNIST images are only 28×28
  pixels, and all models are intentionally small/lightweight. Findings may
  not directly transfer to larger, higher-resolution medical imaging
  tasks.
- **No hyperparameter tuning** was performed on any of the three models;
  default training settings (AdamW, fixed learning rate, cosine schedule)
  were used throughout. Relative rankings could shift somewhat under
  tuned, longer training.
- **Class-imbalanced Run 1 results should be read with caution** on their
  own — they are included primarily as a robustness check against Run 2,
  not as an independent finding.
- **The `mel` failure/success anecdote (Section 3.4) is a single example**
  and should not be treated as statistical evidence that VNN is generally
  better at distinguishing melanoma from vascular lesions — a systematic,
  per-class breakdown across many samples would be needed to support that
  claim.
- **No formal significance testing** (e.g., confidence intervals, paired
  statistical tests) was performed on the metric differences between
  models; reported differences are descriptive, not statistically
  validated.

---

## 6. Conclusion & Future Work

This project set out to determine whether a Volterra Neural Network's
explicit, pairwise-interaction-based interpretability offers something
genuinely different from established CNN and ViT explanation methods on a
medical image classification task. The evidence — consistent across two
independently sampled evaluation sets — supports a nuanced "yes": VNN
explanations are substantially more stable under perturbation than
Grad-CAM, and visually distinct from both Grad-CAM and attention rollout in
a way that is mathematically explicable (pairwise interactions concentrate
at edges/boundaries). ViT remains the strongest performer on localization
precision (Deletion AUC).

**Natural next steps**, if continuing this work:
- Scale to a larger, higher-resolution dataset (e.g., ISIC 2019/2020) to
  test whether findings hold beyond DermaMNIST's 28×28 constraint.
- Add a systematic per-class breakdown of the quantitative XAI metrics,
  rather than only aggregate numbers, to see whether VNN's advantage (if
  any) concentrates in specific, visually-similar class pairs like
  mel/vasc.
- Run multiple training seeds per model and report confidence intervals,
  to support the current single-run comparisons with proper statistical
  testing.