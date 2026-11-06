import torch

def compute_soft_area_fraction(P_delta, tau=0.5, k=20.0):
    occ = torch.sigmoid(k * (P_delta - tau))
    return occ.mean(dim=[-2, -1])
