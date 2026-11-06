import torch
import torch.nn as nn

class DeconvStage(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DeconvStage(512, 512, use_dropout=True)
        self.d2 = DeconvStage(1024, 512)
        self.d3 = DeconvStage(1024, 256)
        self.d4 = DeconvStage(512, 128)
        self.d5 = DeconvStage(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, f_bottleneck, skips):
        e1, e2, e3, e4, e5 = skips
        d1 = self.d1(f_bottleneck)
        d2 = self.d2(torch.cat([d1, e5], dim=1))
        d3 = self.d3(torch.cat([d2, e4], dim=1))
        d4 = self.d4(torch.cat([d3, e3], dim=1))
        d5 = self.d5(torch.cat([d4, e2], dim=1))
        out = self.final(torch.cat([d5, e1], dim=1))
        return out
