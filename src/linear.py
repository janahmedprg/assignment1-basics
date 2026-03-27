import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}

        self.W = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        self.reset_parameters()

    def reset_parameters(self):
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T