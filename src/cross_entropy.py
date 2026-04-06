import torch
from torch import nn


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x_stable = x - torch.amax(x, dim=-1, keepdim=True)
        logsumexp = torch.log(torch.exp(x_stable).sum(dim=-1))

        target_logits = x_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        loss = -target_logits + logsumexp

        return loss.mean()