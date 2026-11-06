import torch

def get_continuous_phase_score(gen_tanh, T_ref=0.5):
    img01 = (gen_tanh + 1.0) / 2.0
    return torch.sigmoid(20.0 * (img01 - T_ref))
