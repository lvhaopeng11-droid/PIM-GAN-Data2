import torch
import torch.nn.functional as F
import math

def get_log_kernel(size=9, sigma=2.0, device="cpu"):
    coord = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    y, x = torch.meshgrid(coord, coord, indexing='ij')
    r2 = x**2 + y**2
    k = -(1.0 - r2 / (2.0 * sigma**2)) * torch.exp(-r2 / (2.0 * sigma**2)) / (math.pi * sigma**4)
    return k.view(1, 1, size, size)

def estimate_particle_number_continuous(P_delta, R_scale=1.0):
    k = get_log_kernel(device=P_delta.device)
    c = F.conv2d(P_delta, k, padding=4)
    N = R_scale * torch.sum(F.relu(c), dim=[-2, -1]) + 1e-5
    return N
