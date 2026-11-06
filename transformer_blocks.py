import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim=256, heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        nx = self.norm1(x)
        ax, _ = self.attn(nx, nx, nx)
        x = x + ax
        x = x + self.mlp(self.norm2(x))
        return x
