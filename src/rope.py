import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (pair_indices / d_k))

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # theta_ik matrix (without dobles)
        angles = torch.outer(positions, inv_freq)

        # Doubling cos(theta_ik) and sin(theta_ik)
        cos_cached = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
        sin_cached = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # Coefficients for sin
        rotated = torch.empty_like(x)
        rotated[..., ::2] = -x2
        rotated[..., 1::2] = x1

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        return x * cos + rotated * sin