import torch
from torch import nn

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int):
        x_stable = x - x.amax(dim=dim, keepdim=True)
        exp_x = torch.exp(x_stable)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)
