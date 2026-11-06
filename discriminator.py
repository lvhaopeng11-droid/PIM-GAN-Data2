import torch
import torch.nn as nn
from .transformer_blocks import TransformerEncoderBlock

class PIMGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, 256) * 0.02)
        self.trans1 = TransformerEncoderBlock(dim=256, heads=8)
        self.trans2 = TransformerEncoderBlock(dim=256, heads=8)
        self.head = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x_pair):
        s = self.stem(x_pair)
        d = self.down(s)
        B, C, H, W = d.shape
        tokens = d.view(B, C, H * W).permute(0, 2, 1) + self.pos_embed
        tokens = self.trans2(self.trans1(tokens))
        sp = tokens.permute(0, 2, 1).view(B, C, H, W)
        logits = self.head(sp)
        return logits
