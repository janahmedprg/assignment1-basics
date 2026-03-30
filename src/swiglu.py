import torch
from torch import nn
import torch.nn.init as init
import math


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        if d_ff is None:
            d_ff = 64 * math.ceil((8 * d_model) / (3 * 64))
        self.d_ff = d_ff

        self.W1 = nn.Parameter(
            torch.empty((d_ff, d_model), **factory_kwargs)
        )
        self.W2 = nn.Parameter(
            torch.empty((d_model, d_ff), **factory_kwargs)
        )
        self.W3 = nn.Parameter(
            torch.empty((d_ff, d_model), **factory_kwargs)
        )

        self.reset_parameters()

    def reset_parameters(self):
        std = (2.0 / (self.d_ff + self.d_model)) ** 0.5
        init.trunc_normal_(self.W1, mean=0.0, std=std, a=-3*std, b=3*std)
        init.trunc_normal_(self.W2, mean=0.0, std=std, a=-3*std, b=3*std)
        init.trunc_normal_(self.W3, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x @ self.W1.T
        x3 = x @ self.W3.T
        return ((x1 * torch.sigmoid(x1)) * x3) @ self.W2.T