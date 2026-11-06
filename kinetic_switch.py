import torch

def get_kinetic_smoothing_weight(A_delta, A_eq, gamma=20.0, epsilon=0.01):
    diff = torch.abs(A_delta - A_eq)
    return torch.sigmoid(gamma * (epsilon - diff))
