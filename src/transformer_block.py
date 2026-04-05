import torch
from torch import nn
from src.rope import RotaryPositionalEmbedding
from src.multihead_self_attention import MultiHeadSelfAttention
from src.rmsnorm import RMSNorm
from src.swiglu import SwiGLU

class TransformerBlock(nn.Module):

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope: RotaryPositionalEmbedding = None,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, device=x.device)
        y1 = x + self.attn(self.ln1(x), token_positions)

        return y1 + self.ffn(self.ln2(y1))