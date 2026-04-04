import torch
from torch import nn
from src.scaled_dot_product_attention import ScaledDotProductAttention
from src.linear import Linear


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.dv = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.o_proj = Linear(d_model, d_model, device, dtype)

        self.rope = rope
        self.attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qx = self.q_proj(x)
        kx = self.k_proj(x)
        vx = self.v_proj(x)

        qx = qx.view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        kx = kx.view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        vx = vx.view(batch_size, seq_len, self.num_heads, self.dv).transpose(1, 2)

        if self.rope is not None:
            qx = self.rope(qx, token_positions)
            kx = self.rope(kx, token_positions)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        out = self.attention(qx, kx, vx, causal_mask)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.o_proj(out)

        return out