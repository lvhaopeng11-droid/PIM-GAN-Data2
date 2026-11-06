import torch
import torch.nn as nn

class ConvStage(nn.Module):
    def __init__(self, in_ch, out_ch, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvStage(1, 64, use_norm=False)
        self.c2 = ConvStage(64, 128)
        self.c3 = ConvStage(128, 256)
        self.c4 = ConvStage(256, 512)
        self.c5 = ConvStage(512, 512)
        self.c6 = ConvStage(512, 512, use_norm=False)

    def forward(self, x):
        e1 = self.c1(x)
        e2 = self.c2(e1)
        e3 = self.c3(e2)
        e4 = self.c4(e3)
        e5 = self.c5(e4)
        e6 = self.c6(e5)
        return e6, [e1, e2, e3, e4, e5]
