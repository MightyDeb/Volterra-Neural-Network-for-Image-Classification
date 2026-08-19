"""
Explainable AI (XAI) Visualization Engine:
  - CNN : Grad-CAM (Target-class gradient backpropagation through final conv layer)
  - ViT : Attention Rollout (Multi-layer self-attention weight recursion from [CLS] token)
  - VNN : Volterra Pairwise Interaction Map (Explicit 2nd-order quadratic polynomial interaction energy)

Provides unified interfaces for generating normalized saliency maps across all three architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import SimpleCNN, SimpleViT, SimpleVNN, VolterraConv2d


# ---------------------------------------------------------------------------
# 1. CNN: Grad-CAM (Gradient-Weighted Class Activation Mapping)
# ---------------------------------------------------------------------------
def gradcam_cnn(model: SimpleCNN, img: torch.Tensor, target_class: int = None) -> np.ndarray:
    """
    Computes standard Grad-CAM using activations and gradients from the final conv layer.
    
    Args:
        model: Trained SimpleCNN model
        img: Input image tensor (1, C, H, W) or (C, H, W)
        target_class: Integer class index to explain (defaults to argmax predicted class)

    Returns:
        (H, W) float numpy array normalized in [0, 1]
    """
    model.eval()
    if img.ndim == 3:
        img = img.unsqueeze(0)

    device = next(model.parameters()).device
    img = img.to(device)

    # Locate the final conv layer
    target_conv = None
    if hasattr(model, "block3") and len(model.block3) > 0:
        target_conv = model.block3[0]
    elif hasattr(model, "features"):
        for module in reversed(list(model.features.modules())):
            if isinstance(module, nn.Conv2d):
                target_conv = module
                break

    if target_conv is None:
        raise ValueError("Could not find a Conv2d layer in SimpleCNN for Grad-CAM.")

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_conv.register_forward_hook(fwd_hook)
    h2 = target_conv.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    out = model(img)
    if target_class is None:
        target_class = out.argmax(dim=1).item()

    score = out[0, target_class]
    score.backward()

    acts = activations["value"][0]       # (C, h, w)
    grads = gradients["value"][0]        # (C, h, w)
    weights = grads.mean(dim=(1, 2))     # (C,) GAP pooled gradients

    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # (h, w)
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(
        cam[None, None], size=img.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]

    h1.remove()
    h2.remove()
    return cam.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 2. ViT: Attention Rollout
# ---------------------------------------------------------------------------
def attention_rollout_vit(model: SimpleViT, img: torch.Tensor,
                          discard_ratio: float = 0.0,
                          head_fusion: str = "mean") -> np.ndarray:
    """
    Computes Attention Rollout (Abnar & Zuidema 2020) by recursively chaining
    self-attention matrices across all transformer blocks.

    Args:
        model: Trained SimpleViT model
        img: Input image tensor (1, C, H, W) or (C, H, W)
        discard_ratio: Ratio of lowest attention values to prune (default 0.0)
        head_fusion: How to fuse multi-head attention ('mean', 'max', 'min')

    Returns:
        (H, W) float numpy array interpolated to input resolution in [0, 1]
    """
    model.eval()
    if img.ndim == 3:
        img = img.unsqueeze(0)

    device = next(model.parameters()).device
    img = img.to(device)

    attn_maps = []
    original_forwards = []

    for blk in model.blocks:
        def make_forward(block_module):
            def forward(x):
                h = block_module.norm1(x)
                # MultiheadAttention call with need_weights=True to get raw attention matrix
                attn_out, attn_w = block_module.attn(
                    h, h, h, need_weights=True, average_attn_weights=(head_fusion == "mean")
                )
                attn_maps.append(attn_w.detach())
                x = x + attn_out
                x = x + block_module.mlp(block_module.norm2(x))
                return x
            return forward

        original_forwards.append(blk.forward)
        blk.forward = make_forward(blk)

    with torch.no_grad():
        _ = model(img)

    # Restore original forward methods
    for blk, fwd in zip(model.blocks, original_forwards):
        blk.forward = fwd

    if len(attn_maps) == 0:
        raise RuntimeError("Failed to capture attention weights from ViT transformer blocks.")

    # Rollout computation
    n_tokens = attn_maps[0].shape[-1]
    result = torch.eye(n_tokens, device=device)

    for attn in attn_maps:
        if attn.ndim == 4:  # (B, heads, N, N)
            if head_fusion == "mean":
                a = attn.mean(dim=1)[0]
            elif head_fusion == "max":
                a = attn.max(dim=1)[0]
            else:
                a = attn.min(dim=1)[0]
        else:
            a = attn[0]

        # Account for residual connection (identity add)
        a = a + torch.eye(n_tokens, device=device)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    # Saliency of patches to the [CLS] classification token (index 0)
    cls_attention = result[0, 1:]  # Exclude CLS self-attention
    n_patches = cls_attention.shape[0]
    side = int(round(n_patches ** 0.5))

    heatmap = cls_attention.reshape(1, 1, side, side)
    heatmap = F.interpolate(heatmap, size=img.shape[-2:], mode="bilinear", align_corners=False)
    heatmap = heatmap[0, 0]
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap.cpu().numpy()


# ---------------------------------------------------------------------------
# 3. VNN: Volterra Pairwise Interaction Map
# ---------------------------------------------------------------------------
def volterra_pairwise_map(vnn_model_or_layer, img: torch.Tensor,
                          out_channel: int = None) -> np.ndarray:
    """
    Computes the 2nd-order Volterra quadratic interaction energy map:
        M = ReLU( sum_{q=1..R} (A_q * x) * (B_q * x) )

    Args:
        vnn_model_or_layer: SimpleVNN model or a single VolterraConv2d layer
        img: Input image tensor (1, C, H, W) or (C, H, W)
        out_channel: Specific output channel index (if None, averages across all channels)

    Returns:
        (H, W) float numpy array interpolated to input resolution in [0, 1]
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)

    # Determine target Volterra layer
    if isinstance(vnn_model_or_layer, SimpleVNN):
        layer = vnn_model_or_layer.layer1 if hasattr(vnn_model_or_layer, "layer1") else vnn_model_or_layer.features[0]
    elif isinstance(vnn_model_or_layer, VolterraConv2d):
        layer = vnn_model_or_layer
    else:
        raise TypeError("Expected SimpleVNN instance or VolterraConv2d layer.")

    device = next(layer.parameters()).device
    img = img.to(device)

    with torch.no_grad():
        quad_total = torch.zeros(1, layer.linear.out_channels, *img.shape[-2:], device=device)
        for a, b in zip(layer.branch_a, layer.branch_b):
            quad_total = quad_total + a(img) * b(img)

        if out_channel is not None and 0 <= out_channel < quad_total.shape[1]:
            heatmap = quad_total[0, out_channel]
        else:
            # Average or L2-norm magnitude across all feature channels
            heatmap = torch.norm(quad_total[0], dim=0, p=2)

        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        if heatmap.shape != img.shape[-2:]:
            heatmap = F.interpolate(
                heatmap[None, None], size=img.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0]

    return heatmap.cpu().numpy()


# ---------------------------------------------------------------------------
# 4. Unified Explainability Interface
# ---------------------------------------------------------------------------
def generate_saliency_map(model: nn.Module, model_type: str, img: torch.Tensor,
                          target_class: int = None) -> np.ndarray:
    """
    Unified dispatcher to obtain an (H, W) explanation map in [0, 1] for any model.
    """
    m_type = model_type.lower().strip()
    if m_type == "cnn":
        return gradcam_cnn(model, img, target_class=target_class)
    elif m_type == "vit":
        return attention_rollout_vit(model, img)
    elif m_type == "vnn":
        return volterra_pairwise_map(model, img)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose 'cnn', 'vnn', or 'vit'.")


if __name__ == "__main__":
    from models import build_model
    x = torch.randn(1, 3, 32, 32)
    cnn = build_model("cnn", num_classes=7)
    vnn = build_model("vnn", num_classes=7)
    vit = build_model("vit", num_classes=7)

    cam = generate_saliency_map(cnn, "cnn", x)
    vmap = generate_saliency_map(vnn, "vnn", x)
    rollout = generate_saliency_map(vit, "vit", x)

    print("Sanity Check: Saliency Map Shapes & Ranges")
    print(f"CNN Grad-CAM: shape={cam.shape}, min={cam.min():.3f}, max={cam.max():.3f}")
    print(f"VNN Volterra: shape={vmap.shape}, min={vmap.min():.3f}, max={vmap.max():.3f}")
    print(f"ViT Rollout : shape={rollout.shape}, min={rollout.min():.3f}, max={rollout.max():.3f}")
