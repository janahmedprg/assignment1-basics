import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.W = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.W

        return result.to(in_dtype)