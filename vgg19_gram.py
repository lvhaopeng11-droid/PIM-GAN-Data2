import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGG19GramTextureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.s1 = nn.Sequential(*[vgg[i] for i in range(4)])
        self.s2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])
        self.s3 = nn.Sequential(*[vgg[i] for i in range(9, 18)])
        for p in self.parameters():
            p.requires_grad = False

    def gram(self, x):
        B, C, H, W = x.size()
        f = x.view(B, C, H * W)
        return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)

    def forward(self, gen, target):
        g = ((gen + 1.0) / 2.0).repeat(1, 3, 1, 1)
        t = ((target + 1.0) / 2.0).repeat(1, 3, 1, 1)
        loss = 0.0
        for s in [self.s1, self.s2, self.s3]:
            g = s(g)
            with torch.no_grad():
                t = s(t)
            loss += nn.functional.mse_loss(self.gram(g), self.gram(t))
        return loss
