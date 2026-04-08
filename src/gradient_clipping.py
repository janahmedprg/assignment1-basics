import torch

def gradient_clipping(params: torch.Tensor, max_l2_norm: float):
    eps = 10e-6

    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += p.grad.data.pow(2).sum()
    
    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)

        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale)