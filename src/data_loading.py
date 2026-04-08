import numpy.typing as npt
import torch
import random

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []

    max_start = len(dataset) - context_length - 1

    for _ in range(batch_size):
        idx = random.randint(0, max_start)

        inputs.append(dataset[idx : idx + context_length])
        targets.append(dataset[idx + 1 : idx + context_length + 1])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets