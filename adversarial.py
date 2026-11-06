import torch
import torch.nn as nn

class AdversarialBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, target_is_real=True):
        targets = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
        return self.loss(logits, targets)
