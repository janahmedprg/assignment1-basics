import math
import torch
from torch import nn
from src.softmax import Softmax

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dk = Q.shape[-1]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dk)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = self.softmax(scores, dim=-1)

        return attn @ V