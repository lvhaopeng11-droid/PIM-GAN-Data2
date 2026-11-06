import torch
import torch.nn as nn
import math

class PIAM(nn.Module):
    def __init__(self, in_channels=512, d_model=512):
        super().__init__()
        self.d_model = d_model
        self.to_k = nn.Linear(in_channels, d_model, bias=False)
        self.to_v = nn.Linear(in_channels, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)
        self.out_proj = nn.Linear(d_model, in_channels)

    def forward(self, F_latent, query):
        B, C, H, W = F_latent.shape
        tokens = F_latent.view(B, C, H * W).permute(0, 2, 1)
        K = self.to_k(tokens)
        V = self.to_v(tokens)
        Q = query.unsqueeze(1)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, V)
        
        modulated = tokens * torch.sigmoid(self.out_proj(context))
        F_out = modulated.permute(0, 2, 1).view(B, C, H, W)
        return F_out, attn.view(B, H, W)
