import torch
from torch import nn
import torch.nn.functional as F

class AASIST2_Loss(nn.Module):
    """
    Example of a loss function to use.
    """
    def __init__(self):
        super().__init__()
        self.sr = 16000
        self.A = 3/50
        self.B = 7/50
        self.s = 15.0


    def forward(self, vectors: torch.Tensor, W: torch.Tensor, labels: torch.Tensor, duration, **batch):
        normW = F.normalize(W)
        normv = F.normalize(vectors)
        logits = normv @ normW
        one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
        m = (duration / self.sr) * self.A + self.B
        logits = self.s * (logits - m * one_hot)
        return {"loss": F.cross_entropy(logits, labels)}