from collections.abc import Callable
from typing import Optional, Tuple, Union
import math
import torch
from torch.optim.optimizer import ParamsT
from torch import Tensor

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                grad = p.grad # Get the gradient of loss with respect to p.
                m1 = state.get("m1", torch.zeros_like(p))
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                m2 = state.get("m2", torch.zeros_like(p))
                m2.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                denom =  m2.sqrt().add_(eps)
                p.addcdiv_(m1, denom, value=-lr_t)  # Update weight tensor in-place.
                p.mul_(1 - lr * weight_decay)
                state["t"] = t + 1 # Increment iteration number.
                state["m1"] = m1
                state["m2"] = m2
        return loss