import torch
import torch.nn as nn
import torch.nn.init as init

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )

        self.reset_parameters()

    def reset_parameters(self):
        std = (2.0 / (self.num_embeddings + self.embedding_dim)) ** 0.5
        init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]