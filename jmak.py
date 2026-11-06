import torch

def jmak_dissolution_residual(A_delta, t_hold, n, log_k):
    t_safe = torch.clamp(t_hold, min=1.0)
    clamped_A = torch.clamp(A_delta, 1e-4, 1.0 - 1e-4)
    pred = torch.log(-torch.log(1.0 - clamped_A))
    target = n * torch.log(t_safe) + log_k
    return (pred - target)**2
