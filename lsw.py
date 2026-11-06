import torch
from .log_particle_count import estimate_particle_number_continuous

def lsw_coarsening_residual(P_delta, A_delta, t_hold, k_lsw, zeta=1.0):
    B, C, H, W = P_delta.shape
    N = estimate_particle_number_continuous(P_delta)
    area = A_delta * (H * W)
    r_cubed = zeta * torch.pow(area / N, 1.5)
    t_safe = torch.clamp(t_hold, min=1.0)
    return (r_cubed - k_lsw * t_safe)**2
