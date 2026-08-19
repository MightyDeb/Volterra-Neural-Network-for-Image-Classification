"""
Interpretability visualizations for the three models.

  - CNN : Grad-CAM (which pixels drove the prediction)
  - ViT : Attention rollout (which patches the [CLS] token attended to)
  - VNN : Volterra pairwise-interaction map (which PIXEL PAIRS the 2nd-order
          term weighted most heavily) -- this is the genuinely novel part,
          since CNN/ViT explanations are single-pixel/single-patch importance,
          but the Volterra term is explicitly modeling pairwise interactions.

Usage:
    from viz import gradcam_cnn, attention_rollout_vit, volterra_pairwise_map

Each function takes a trained model + a single image tensor (1, 3, 32, 32)
and returns a numpy array you can plot with matplotlib / imshow.
"""

import torch
import torch.nn.functional as F
import numpy as np

from models import SimpleCNN, SimpleViT, SimpleVNN, VolterraConv2d


# ---------------------------------------------------------------------------
# CNN: Grad-CAM
# ---------------------------------------------------------------------------
def gradcam_cnn(model: SimpleCNN, img: torch.Tensor, target_class: int = None):
    """
    Standard Grad-CAM using the last conv layer's activations + gradients.
    img: (1, 3, 32, 32)
    Returns: (H, W) numpy heatmap, resized to input resolution.
    """
    model.eval()
    activations = {}
    gradients = {}

    last_conv = model.features[6]  # the third Conv2d layer (index into Sequential)

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_full_backward_hook(bwd_hook)

    out = model(img)
    if target_class is None:
        target_class = out.argmax(dim=1).item()

    model.zero_grad()
    out[0, target_class].backward()

    acts = activations["value"][0]      # (C, h, w)
    grads = gradients["value"][0]       # (C, h, w)
    weights = grads.mean(dim=(1, 2))    # (C,) global-average-pooled gradients

    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # (h, w)
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=img.shape[-2:], mode="bilinear", align_corners=False)

    h1.remove()
    h2.remove()
    return cam[0, 0].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# ViT: Attention rollout
# ---------------------------------------------------------------------------
def attention_rollout_vit(model: SimpleViT, img: torch.Tensor):
    """
    Captures attention weights from every transformer block and "rolls" them
    together (Abnar & Zuidema 2020 style, simplified) to get a single
    CLS-token-to-patch importance map.
    Returns: (H_patches, W_patches) numpy heatmap.
    """
    model.eval()
    attn_maps = []

    hooks = []
    for blk in model.blocks:
        def make_hook(store):
            def hook(module, inp, out):
                # nn.MultiheadAttention with need_weights=False returns None weights,
                # so we re-run with need_weights=True inside the hook's owning call.
                pass
            return hook
        hooks.append(blk)

    # Simpler approach: monkey-patch forward to capture attention weights directly.
    original_forwards = []
    for blk in model.blocks:
        def make_forward(blk):
            def forward(x):
                h = blk.norm1(x)
                attn_out, attn_w = blk.attn(h, h, h, need_weights=True, average_attn_weights=True)
                attn_maps.append(attn_w.detach())  # (B, tokens, tokens)
                x = x + attn_out
                x = x + blk.mlp(blk.norm2(x))
                return x
            return forward
        original_forwards.append(blk.forward)
        blk.forward = make_forward(blk)

    with torch.no_grad():
        model(img)

    # restore
    for blk, fwd in zip(model.blocks, original_forwards):
        blk.forward = fwd

    # Rollout: multiply attention matrices across layers (with residual/identity add)
    n_tokens = attn_maps[0].shape[-1]
    result = torch.eye(n_tokens)
    for attn in attn_maps:
        a = attn[0]  # (tokens, tokens)
        a = a + torch.eye(n_tokens)  # account for residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    cls_attention = result[0, 1:]  # attention from CLS token to all patches (skip CLS-CLS)
    n_patches = cls_attention.shape[0]
    side = int(n_patches ** 0.5)
    heatmap = cls_attention.reshape(side, side).cpu().numpy()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


# ---------------------------------------------------------------------------
# VNN: Pairwise Volterra interaction map
# ---------------------------------------------------------------------------
def volterra_pairwise_map(layer: VolterraConv2d, img: torch.Tensor, out_channel: int = 0):
    """
    For a single VolterraConv2d layer, visualizes the *quadratic* term's
    contribution: sum_q (a_q * x) elementwise* (b_q * x), restricted to one
    output channel. This is what a CNN/ViT saliency map CANNOT show directly:
    it highlights *where the pairwise multiplicative interaction is strongest*,
    not just where a single pixel mattered.

    Returns: (H, W) numpy heatmap at the layer's output resolution.
    """
    with torch.no_grad():
        quad_total = torch.zeros(1, layer.linear.out_channels, *img.shape[-2:])
        for a, b in zip(layer.branch_a, layer.branch_b):
            quad_total = quad_total + a(img) * b(img)
        heatmap = quad_total[0, out_channel]
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.cpu().numpy()


if __name__ == "__main__":
    # Sanity check with random weights + random image (no trained checkpoint needed
    # to verify the code runs end-to-end).
    img = torch.randn(1, 3, 32, 32)

    cnn = SimpleCNN()
    cam = gradcam_cnn(cnn, img)
    print("Grad-CAM output shape:", cam.shape)

    vit = SimpleViT()
    rollout = attention_rollout_vit(vit, img)
    print("Attention rollout shape:", rollout.shape)

    vnn = SimpleVNN()
    first_volterra_layer = vnn.features[0]
    vmap = volterra_pairwise_map(first_volterra_layer, img)
    print("Volterra pairwise map shape:", vmap.shape)
