import torch
from torch import nn

class AASIST_Loss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        weight = torch.FloatTensor([0.1, 0.9])
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        return {"loss": self.loss(logits, labels)}