# VNN vs CNN vs ViT --- Explainable Medical Image Classification

A comparative study of a Volterra Neural Network (VNN), standard CNN,
and Vision Transformer (ViT) for medical image classification, extended
with **hyperparameter optimization and quantitative Explainable AI
(XAI)**.

The project focuses not only on which model predicts better, but also on
**why the models make their predictions and how trustworthy their
explanations are**.

## 1. Project Goals

Compare CNN, VNN, and ViT on:

1.  Classification performance
2.  Model complexity and computational efficiency
3.  Hyperparameter sensitivity and optimized performance
4.  Explainability methods
5.  Quantitative quality of explanations
6.  Robustness/stability of explanations

### Core Research Question

> How do conventional CNN features, VNN second-order interactions, and
> ViT attention differ in predictive performance and explainability for
> medical image classification?

## 2. Models

### CNN

-   Standard convolutional neural network
-   XAI: **Grad-CAM**

### VNN

-   Volterra Neural Network
-   XAI: **Volterra pairwise interaction map**
-   Focus on the quadratic/second-order interaction term

### ViT

-   Small Vision Transformer
-   XAI: **Attention rollout**

The current implementation already contains all three model definitions
and the three visualization methods.

## 3. Dataset

Start with a small medical-image benchmark such as:

-   **DermaMNIST** --- skin-lesion classification

Prefer a dataset with meaningful localization annotations if
quantitative localization evaluation is planned.

### Important

For final results: - use the complete available dataset where possible - preserve
train/test separation

Avoid data leakage, especially when multiple images can originate from
the same patient.

## 4. Preprocessing

Typical pipeline:

``` text
Medical Image
     ↓
Resize / crop if required
     ↓
Convert to Tensor
     ↓
Normalize
     ↓
Data augmentation (training only)
     ↓
Model
```

For training, consider: - random crop - horizontal flip where medically
valid - small rotations where medically valid - mild color/intensity
augmentation where appropriate

Do not apply random augmentation to the test set.


## 5. Hyperparameter Optimization

Add a systematic hyperparameter-search stage after the baseline models
work.

### Hyperparameters to consider

**Optimization** - Learning rate - Weight decay - Optimizer: Adam,
AdamW, SGD

**Architecture** - Number of channels - Hidden dimension - Number of
layers - Kernel size - Dropout - VNN rank - ViT embedding dimension -
Number of attention heads - Number of transformer layers - Patch size

**Training** - Batch size - Number of epochs - Learning-rate scheduler

Record: - best hyperparameters - validation score - number of trials -
trial history - final test performance

## 6. Classification Metrics

Do not rely only on accuracy, especially for medical data.

### Multiclass classification

-   Accuracy
-   Macro Precision
-   Macro Recall
-   Macro F1
-   ROC-AUC , PR-AUC
-   Confusion matrix

For medical applications, explicitly discuss **false negatives** and
sensitivity.

## 7. Explainable AI

Generate explanations for the same test examples across all three
models.

### CNN

``` text
CNN → Grad-CAM
```

### ViT

``` text
ViT → Attention Rollout
```

### VNN

``` text
VNN → Pairwise Volterra Interaction Map
```

For qualitative analysis, show:

``` text
Original image
      ↓
CNN explanation
      ↓
VNN explanation
      ↓
ViT explanation
```

Use representative correctly classified and misclassified examples.

Do not evaluate explanations only on cherry-picked images.

## 8. Quantitative XAI Evaluation

This is the major extension of the current project.

### 8.1 Deletion Metric

Progressively remove the pixels/regions considered most important by the
explanation.

``` text
Original image
      ↓
Remove top 10% important pixels
      ↓
Remove top 20%
      ↓
...
      ↓
Measure model confidence
```

A faithful explanation should cause model confidence to decrease when
highly important regions are removed.

Calculate **Deletion AUC**.

### 8.2 Insertion Metric

Start from a blurred/empty baseline and progressively insert the most
important regions.

Measure prediction confidence and calculate **Insertion AUC**.

Higher insertion performance indicates that important regions are being
identified early.

### 8.3 Explanation Stability

Create small perturbations of the same image: - small crop - mild
brightness change - small Gaussian noise - other medically valid
perturbations

Generate explanations again and compare using: - IoU - Pearson
correlation - SSIM

A small irrelevant input change should not completely change the
explanation.

### 8.4 Localization

Only use this when the dataset provides suitable ground-truth
localization annotations such as masks or bounding boxes.

Report: - IoU - Dice coefficient - localization accuracy if appropriate

Compare CNN, VNN and ViT explanations using the same ground truth.

## 9. XAI Evaluation Table

Create a final table:

  -------------------------------------------------------------------------------
  Model    XAI Method      Deletion   Insertion Stability ↑      IoU ↑     Dice ↑
                              AUC ↓       AUC ↑                        
  -------- ------------- ---------- ----------- ----------- ---------- ----------
  CNN      Grad-CAM                                                    

  VNN      Interaction                                                 
           Map                                                         

  ViT      Attention                                                   
           Rollout                                                     
  -------------------------------------------------------------------------------

Do not include localization metrics if the dataset does not provide
suitable ground truth.

## 10. Deep Feature / Representation Analysis

In addition to explaining where the model looks, analyze what the model has learned internally.

The goal is to investigate whether CNN, VNN, and ViT learn different internal representations of medical images, and whether those differences relate to predictive performance and explainability.

## 11. Model Efficiency

Since the models are not necessarily parameter-matched, report their
actual complexity.

Measure: - Trainable parameters - Model size - Training time - Inference
time per image - Peak GPU memory if available - FLOPs/MACs if practical

### 11.1 Intermediate Feature Maps

Extract activations from different depths of each model.

For CNN:

Input image
    ↓
Early layers      → low-level patterns
    ↓
Middle layers     → textures / shapes / structures
    ↓
Deep layers       → higher-level representations
    ↓
Classifier

Visualize representative activation maps from:

early layers
middle layers
final feature layer

Do not automatically assign a semantic meaning such as "lesion detector" to a feature map.

Treat these interpretations as hypotheses and support them with quantitative analysis.

### 11.2 Feature-Space Visualization

Extract the feature vector immediately before the final classifier.

Image
  ├── CNN → feature vector
  ├── VNN → feature vector
  └── ViT → feature vector

Reduce the high-dimensional representations using:

PCA
t-SNE
UMAP

Plot the representations with points colored by class.

Questions to investigate
Do different medical classes form distinct clusters?
Are some classes strongly overlapping?
Which model produces the clearest class separation?
Where do misclassified samples appear in feature space?

### 11.3 Quantitative Feature Separability

Do not rely only on PCA/t-SNE/UMAP plots.

Calculate a quantitative cluster-separation metric such as:

Silhouette score

Compare:

Model	Feature Dimension	Silhouette Score ↑
CNN		
VNN		
ViT		

This provides quantitative evidence about how well the learned representations separate the classes.

### 11.4 Correct vs Incorrect Feature Representations

Analyze feature representations of:

correctly classified samples
incorrectly classified samples
high-confidence predictions
low-confidence predictions

Investigate whether misclassified samples:

lie close to another class cluster
occur in overlapping regions
have unusual feature representations

This can help explain why particular medical images are difficult to classify.

### 11.5 Representation Similarity Between Models

Compare the internal representations learned by CNN, VNN, and ViT.

Possible methods:

Cosine similarity
Feature correlation
CKA (Centered Kernel Alignment)

A representation-similarity matrix can be reported as:

	CNN	VNN	ViT
CNN	1.00		
VNN		1.00	
ViT			1.00

CKA is particularly useful for comparing representations across architectures and layers.

Questions to investigate
Do CNN and VNN learn similar representations?
Does the VNN learn representations that differ from CNN despite using convolution-like operations?
How different are ViT representations from CNN/VNN?
At which depth do representations become more architecture-specific?

### 11.6 VNN Second-Order Feature Interactions

The VNN provides an additional analysis opportunity because its representation contains second-order/Volterra interactions.

Conceptually:

Input
  ↓
VNN features
  ↓
First-order terms
  +
Second-order interactions
  ↓
Prediction

Analyze the strength and spatial distribution of the pairwise interaction terms.

Key Question

Do the VNN's second-order interactions capture useful relationships between visual patterns that are different from the representations learned by CNN and ViT?

Do not assume that a particular interaction corresponds to a specific medical feature unless supported by the experiment.

