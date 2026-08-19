"""
Model Architectures for Medical Image Classification & XAI Comparison:
  1. SimpleCNN : Standard Conv2d + BatchNorm2d + ReLU baseline with feature extraction
  2. SimpleVNN : Cascaded 2nd-order Volterra filter layers with low-rank branch factorization
  3. SimpleViT : Minimal Vision Transformer with patch embeddings & multi-head self-attention

All models support configurable input channels, image dimensions, class counts,
penultimate feature extraction, and intermediate layer representations.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Volterra Neural Network (VNN) Components
# ---------------------------------------------------------------------------
class VolterraConv2d(nn.Module):
    """
    Second-order discrete Volterra filter with low-rank (Rank-R) factorization:

        y = (W_lin * x) + sum_{q=1..R} (A_q * x) elementwise* (B_q * x)

    This factorized formulation reduces the combinatorial complexity of full
    pairwise cross-terms O(K^4) to O(2 * R * K^2), enabling deep activation-free
    polynomial feature representation.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, rank: int = 1,
                 stride: int = 1, padding: int = 1, bias: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.rank = rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 1st-order linear convolution branch
        self.linear = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)

        # 2nd-order factorized quadratic branches
        self.branch_a = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
            for _ in range(rank)
        ])
        self.branch_b = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
            for _ in range(rank)
        ])

        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        for a, b in zip(self.branch_a, self.branch_b):
            out = out + a(x) * b(x)
        return self.bn(out)

    def compute_branch_energies(self, x: torch.Tensor):
        """
        Computes the separate Frobenius energy norm of the linear vs.
        quadratic interaction components for representation analysis.
        """
        with torch.no_grad():
            lin_out = self.linear(x)
            quad_out = torch.zeros_like(lin_out)
            for a, b in zip(self.branch_a, self.branch_b):
                quad_out = quad_out + a(x) * b(x)

            e_lin = torch.norm(lin_out, p="fro").item()
            e_quad = torch.norm(quad_out, p="fro").item()
            total = e_lin + e_quad + 1e-8
            ratio = e_quad / total
            return {"linear_energy": e_lin, "quadratic_energy": e_quad, "quadratic_ratio": ratio}


class SimpleVNN(nn.Module):
    """
    Volterra Neural Network featuring 3 cascaded VolterraConv2d stages,
    global average pooling, and a linear classifier head.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 base_channels: int = 32, rank: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.rank = rank

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.layer1 = VolterraConv2d(in_channels, c1, kernel_size=3, rank=rank, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = VolterraConv2d(c1, c2, kernel_size=3, rank=rank, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.layer3 = VolterraConv2d(c2, c3, kernel_size=3, rank=rank, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)

        # Legacy features container for backward compatibility with viz.py
        self.features = nn.Sequential(
            self.layer1,
            self.pool1,
            self.layer2,
            self.pool2,
            self.layer3,
            self.pool3,
        )
        self.classifier = nn.Linear(c3, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate feature embedding (B, c3) before the linear head."""
        x = self.features(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)

    def get_last_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the spatial feature map tensor [B, C, H, W] from layer3 before pooling."""
        out1 = self.layer1(x)
        p1 = self.pool1(out1)
        out2 = self.layer2(p1)
        p2 = self.pool2(out2)
        out3 = self.layer3(p2)
        return out3

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, feature_maps) where feature_maps has shape [B, C, H, W].
        """
        feat_map = self.get_last_feature_map(x)
        pooled = self.pool3(feat_map).flatten(1)
        logits = self.classifier(pooled)
        return logits, feat_map

    def get_intermediate_activations(self, x: torch.Tensor):
        """Returns activations at early (layer1), mid (layer2), and deep (layer3) stages."""
        out1 = self.layer1(x)
        p1 = self.pool1(out1)
        out2 = self.layer2(p1)
        p2 = self.pool2(out2)
        out3 = self.layer3(p2)
        return {"early": out1, "mid": out2, "deep": out3}


# ---------------------------------------------------------------------------
# 2. Convolutional Neural Network (CNN) Baseline
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    Standard Convolutional Baseline with 3 Conv-BN-ReLU stages,
    global average pooling, and linear classification head.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 base_channels: int = 32, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )

        # Sequential container for backward compatibility
        self.features = nn.Sequential(
            self.block1[0], self.block1[1], self.block1[2], self.block1[3],
            self.block2[0], self.block2[1], self.block2[2], self.block2[3],
            self.block3[0], self.block3[1], self.block3[2], self.block3[3],
        )
        self.classifier = nn.Linear(c3, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate feature embedding (B, c3) before linear classifier."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)

    def get_intermediate_activations(self, x: torch.Tensor):
        """Returns activations at early (block1), mid (block2), and deep (block3 conv) stages."""
        out1 = self.block1[0](x)
        a1 = self.block1(x)
        out2 = self.block2[0](a1)
        a2 = self.block2(a1)
        out3 = self.block3[0](a2)
        return {"early": out1, "mid": out2, "deep": out3}


# ---------------------------------------------------------------------------
# 3. Vision Transformer (ViT)
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Splits an image into non-overlapping patches and projects to embedding dim."""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_ch: int = 3, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm and Multi-Head Self-Attention."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    """
    Compact Vision Transformer for small medical/natural images.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 img_size: int = 32, patch_size: int = 4,
                 embed_dim: int = 128, depth: int = 4, heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate [CLS] token embedding (B, embed_dim)."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_feat = self.extract_features(x)
        return self.head(cls_feat)

    def get_intermediate_activations(self, x: torch.Tensor):
        """Returns token embeddings across transformer depth (early, mid, deep)."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.pos_drop(x + self.pos_embed)

        activations = []
        for blk in self.blocks:
            x = blk(x)
            activations.append(x)

        early_idx = 0
        mid_idx = len(self.blocks) // 2
        deep_idx = len(self.blocks) - 1
        return {
            "early": activations[early_idx],
            "mid": activations[mid_idx],
            "deep": activations[deep_idx],
        }


# ---------------------------------------------------------------------------
# Helper Utility Functions
# ---------------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    """Returns the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, num_classes: int = 10, in_channels: int = 3,
                img_size: int = 32, rank: int = 1, base_channels: int = 32,
                embed_dim: int = 128, depth: int = 4, heads: int = 4,
                dropout: float = 0.0, **kwargs) -> nn.Module:
    """
    Factory constructor for CNN, VNN, and ViT architectures with medical
    dataset customization support.
    """
    name = name.lower().strip()
    if name == "cnn":
        return SimpleCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
        )
    elif name == "vnn":
        return SimpleVNN(
            num_classes=num_classes,
            in_channels=in_channels,
            base_channels=base_channels,
            rank=rank,
        )
    elif name == "vit":
        return SimpleViT(
            num_classes=num_classes,
            in_channels=in_channels,
            img_size=img_size,
            patch_size=4,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model architecture: '{name}'. Choose from 'cnn', 'vnn', 'vit'.")


def load_model_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device = None) -> bool:
    """
    Robust checkpoint loader that safely loads model weights, handling potential
    class count differences (e.g., 10-class pretraining vs 7-class DermaMNIST) via strict=False.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    try:
        state = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(state)
            return True
        except Exception:
            model_dict = model.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
            if filtered:
                model.load_state_dict(filtered, strict=False)
                return True
            return False
    except Exception:
        return False


if __name__ == "__main__":
    # Sanity-check forward passes and feature extraction for 32x32 inputs
    x = torch.randn(2, 3, 32, 32)
    print("=" * 60)
    print("Architecture Sanity Check & Representation Dimensions")
    print("=" * 60)
    for model_name in ["cnn", "vnn", "vit"]:
        m = build_model(model_name, num_classes=7, in_channels=3, img_size=32)
        logits = m(x)
        feat = m.extract_features(x)
        acts = m.get_intermediate_activations(x)
        print(f"[{model_name.upper():<4}] Params: {count_params(m):,d} | Logits: {tuple(logits.shape)} | Feature: {tuple(feat.shape)}")
