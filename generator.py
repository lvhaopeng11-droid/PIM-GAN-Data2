import torch
import torch.nn as nn
from .encoder import UNetEncoder
from .decoder import UNetDecoder
from .piam import PIAM
from .process_mlp import ProcessMLP

class PIMGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        self.process_mlp = ProcessMLP(d_model=512)
        self.piam = PIAM(in_channels=512, d_model=512)
        self.decoder = UNetDecoder()

    def forward(self, x_init, T_celsius, t_min):
        e6, skips = self.encoder(x_init)
        query, kinetic_params, Q_eff = self.process_mlp(T_celsius, t_min)
        f_cond, attn_map = self.piam(e6, query)
        out = self.decoder(f_cond, skips)
        return out, kinetic_params, attn_map
