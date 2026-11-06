import torch

def encode_process_parameters(temp1, time1, temp2=0.0, time2=0.0):
    vec = torch.tensor([float(temp1), float(time1), float(temp2), float(time2)], dtype=torch.float32)
    return vec
