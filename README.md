# VNN vs CNN vs ViT — CIFAR-10 Comparison

A minimal, honest comparison of a Volterra Neural Network against a standard
CNN and a small ViT, matched roughly (not exactly) on scale.

## Files
- `models.py` — all three model definitions + a sanity-check script
- `train.py` — trains one model on CIFAR-10, logs results to `results.csv`
- `results.csv` — created automatically after your first run

## Quickstart (Google Colab, GPU runtime)

1. Upload `models.py` and `train.py` to Colab (or `git clone` if you push
   this to a repo).
2. Set runtime to GPU: Runtime -> Change runtime type -> T4 GPU
3. Run:

```bash
!pip install torch torchvision --quiet
!python train.py --model cnn --epochs 20
!python train.py --model vnn --epochs 20
!python train.py --model vit --epochs 20
```

Each run appends a row to `results.csv` with model name, parameter count,
final test accuracy, and training time. After all three runs, open
`results.csv` and plot accuracy vs. params — that's your headline result.

CIFAR-10 downloads automatically via torchvision on first run (~170MB).

## What's already verified
- All three models forward-pass and backward-pass correctly (checked with
  synthetic data on CPU in this environment — no GPU/internet was available
  here to run real CIFAR-10 training, so **you must run the actual training
  yourself on Colab/Kaggle**).
- Current parameter counts: CNN ~95K, VNN ~281K (rank=1), ViT ~546K.
  These are NOT exactly matched — that's normal and fine. Report the actual
  numbers and discuss accuracy-per-parameter rather than pretending they're
  equal. If you want VNN closer to the CNN's ~95K, reduce channel widths in
  `SimpleVNN.__init__` (currently 32→64→128) rather than fighting with rank.

## Interpretability (viz.py + plot_comparison.py)

This is the genuinely novel part of the project, so it's included:

- **CNN**: Grad-CAM — highlights which pixels drove the prediction.
- **ViT**: Attention rollout — shows which patches the [CLS] token attended
  to, aggregated across all transformer layers.
- **VNN**: Pairwise Volterra interaction map — visualizes where the
  *quadratic* term (the `a_q(x) * b_q(x)` product in `VolterraConv2d`) is
  strongest. This is the part a CNN or ViT explanation cannot show directly:
  single-pixel saliency and patch attention are first-order explanations,
  while this map reflects a genuine pairwise interaction, which is the
  actual selling point of doing interpretability on a VNN specifically.

### Usage
After training and saving checkpoints (train.py now saves `{model}.pt`
automatically at the end of each run):

```bash
python plot_comparison.py --cnn_ckpt cnn.pt --vit_ckpt vit.pt --vnn_ckpt vnn.pt
```

This saves `interpretability_comparison.png` — a 4-panel figure: original
image, Grad-CAM, attention rollout, Volterra pairwise map. Run it on a
handful of test images (change `--image_index`) and pick 2-3 representative
examples for your write-up — don't try to systematically evaluate this
quantitatively (deletion/insertion metrics, IoU against ground truth) this
month, that's real additional work for later.

**Verified**: all three visualization functions (`gradcam_cnn`,
`attention_rollout_vit`, `volterra_pairwise_map`) were run end-to-end here
with real PyTorch on synthetic data and confirmed to produce correctly
shaped outputs without errors. `plot_comparison.py`'s figure assembly was
also verified. What was NOT verified here (no GPU/internet in this sandbox):
whether the visualizations look *meaningful* once the models are actually
trained on real CIFAR-10 — that only becomes visible once you've trained
real weights on Colab. Untrained/random-weight Grad-CAM and attention maps
are meaningless noise by definition; don't be alarmed if the sanity-check
images look like nothing.

## What this project deliberately does NOT include (yet)
- **Quantitative interpretability metrics** (deletion/insertion curves,
  localization IoU against bounding boxes, human evaluation) — the
  qualitative visualizations above are the scoped deliverable for this
  month; quantitative evaluation is real additional work, save it for later
  if you continue this after placements.
- CIFAR-100 / ImageNet — start with CIFAR-10 for speed; scale up later only
  if you have time and compute.
- Hyperparameter tuning — the defaults (AdamW, lr=1e-3, cosine schedule,
  20 epochs) are reasonable starting points, not tuned. Don't burn your
  week tuning hyperparameters; get one clean run per model first.

## Suggested minimal timeline
- Day 1: Get training running on Colab, confirm CIFAR-10 downloads, run cnn
  for 5 epochs just to confirm the pipeline works end-to-end.
- Day 2-4: Run all three models for the full 20 epochs (can run overnight /
  between placement prep sessions — each run is unattended once started;
  checkpoints save automatically).
- Day 5: Run `plot_comparison.py` on 2-3 test images, pick the clearest
  example, write 3-4 sentences on what you observed (e.g. does the Volterra
  map highlight edges/textures differently from Grad-CAM's blob-like focus?).

That's a complete, honest, small project with a genuinely distinctive piece
(the pairwise interaction map). Don't scope-creep further this month —
quantitative evaluation and multi-dataset results are legitimate next steps
for after placements, not now.
