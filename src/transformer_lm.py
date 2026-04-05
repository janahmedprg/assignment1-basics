import torch
from torch import nn
from src.transformer_block import TransformerBlock
from src.rope import RotaryPositionalEmbedding
from src.embedding import Embedding
from src.rmsnorm import RMSNorm
from src.linear import Linear
from src.softmax import Softmax

class TransformerLM(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope: RotaryPositionalEmbedding = None,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope=rope,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(self, x):
        y = self.token_embeddings(x)
        for layer in self.layers:
            y = layer(y)
        
        y = self.ln_final(y)

        return self.lm_head(y)