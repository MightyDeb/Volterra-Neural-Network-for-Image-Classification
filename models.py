"""
Three small, roughly parameter-matched models for CIFAR-10 comparison:
  - SimpleCNN : standard conv + BN + ReLU baseline
  - SimpleVNN : 2nd-order Volterra layers (low-rank / cascaded factorization)
  - SimpleViT : minimal vision transformer

All models take (B, 3, 32, 32) input and output (B, num_classes) logits.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# VNN: Volterra layer
# ---------------------------------------------------------------------------
class VolterraConv2d(nn.Module):
    """
    2nd-order Volterra filter, low-rank ("Q-rank") factorized to avoid the
    combinatorial blowup of a full pairwise kernel:

        y = h1 * x  +  sum_{q=1..rank} (a_q * x) elementwise* (b_q * x)

    This is the standard cascaded/low-rank approximation used in the VNN
    literature (Roheda & Krim 2019; kVNN 2026) to make 2nd-order Volterra
    filtering tractable inside a deep network. No activation function is
    used after this layer -- the polynomial nonlinearity is built in
    ("activation-free learning").
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, rank=4, stride=1, padding=1):
        super().__init__()
        self.linear = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.branch_a = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding) for _ in range(rank)]
        )
        self.branch_b = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding) for _ in range(rank)]
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = self.linear(x)
        for a, b in zip(self.branch_a, self.branch_b):
            out = out + a(x) * b(x)
        return self.bn(out)


class SimpleVNN(nn.Module):
    def __init__(self, num_classes=10, rank=1):
        super().__init__()
        self.features = nn.Sequential(
            VolterraConv2d(3, 32, rank=rank),      # 32x32 -> 32x32
            nn.MaxPool2d(2),                        # -> 16x16
            VolterraConv2d(32, 64, rank=rank),
            nn.MaxPool2d(2),                        # -> 8x8
            VolterraConv2d(64, 128, rank=rank),
            nn.AdaptiveAvgPool2d(1),                # -> 1x1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# CNN baseline
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Minimal ViT
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    def __init__(self, num_classes=10, img_size=32, patch_size=4,
                 embed_dim=128, depth=4, heads=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name, num_classes=10):
    name = name.lower()
    if name == "cnn":
        return SimpleCNN(num_classes)
    elif name == "vnn":
        return SimpleVNN(num_classes)
    elif name == "vit":
        return SimpleViT(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Choose from cnn, vnn, vit.")


if __name__ == "__main__":
    # Quick sanity check: forward pass + parameter count for each model.
    x = torch.randn(4, 3, 32, 32)
    for name in ["cnn", "vnn", "vit"]:
        model = build_model(name)
        out = model(x)
        print(f"{name:>4}: output shape {tuple(out.shape)}, params = {count_params(model):,}")
