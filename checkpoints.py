import torch
import os

def save_checkpoint(model_g, model_d, epoch, path="checkpoints/latest.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "g_state": model_g.state_dict(),
        "d_state": model_d.state_dict()
    }, path)

def load_checkpoint(path, model_g, model_d=None):
    data = torch.load(path, map_location="cpu")
    model_g.load_state_dict(data["g_state"])
    if model_d and "d_state" in data:
        model_d.load_state_dict(data["d_state"])
    return data.get("epoch", 0)
