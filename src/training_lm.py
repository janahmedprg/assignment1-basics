from src.data_loading import get_batch
from src.transformer_lm import TransformerLM
from src.adamw import AdamW
from src.checkpointing import load_checkpoint, save_checkpoint
from src.rope import RotaryPositionalEmbedding
from src.cross_entropy import CrossEntropy
from src.gradient_clipping import gradient_clipping
from src.learning_rate_schedule import learning_rate_schedule
import random

import torch
import numpy as np
import os


@torch.no_grad()
def evaluate_loss(model, train_data, val_data, batch_size, context_length, eval_iters, device):
    criterion = CrossEntropy()
    model.eval()
    losses = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, context_length, device)
            logits = model(x)
            loss = criterion(logits, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def main():
    train_data_path = input("Train data path: ")
    val_data_path = input("Validation data path: ")

    batch_size = int(input("Batch size (default 32): ") or 32)
    context_length = int(input("Context length (default 128): ") or 128)
    max_iters = int(input("Max iterations (default 10000): ") or 10000)
    eval_interval = int(input("Eval interval (default 500): ") or 500)
    eval_iters = int(input("Eval iters (default 50): ") or 50)
    log_interval = int(input("Log interval (default 50): ") or 50)
    save_interval = int(input("Save interval (default 1000): ") or 1000)
    checkpoint_path = input("Checkpoint path (default checkpoint.pt): ") or "checkpoint.pt"
    resume = input("Resume? (y/n): ").strip().lower() == "y"
    seed = int(input("Seed (default 42): ") or 42)
    device = input("Device (default cuda): ") or "cuda"

    learning_rate = float(input("Max learning rate (default 3e-4): ") or 3e-4)
    min_learning_rate = float(input("Min learning rate (default 3e-5): ") or 3e-5)
    warmup_iters = int(input("Warmup iterations (default 500): ") or 500)
    cosine_end_iter = int(input("Cosine decay end iteration (default max_iters): ") or max_iters)

    weight_decay = float(input("Weight decay (default 0.01): ") or 0.01)
    beta1 = float(input("Beta1 (default 0.9): ") or 0.9)
    beta2 = float(input("Beta2 (default 0.999): ") or 0.999)
    eps = float(input("Eps (default 1e-8): ") or 1e-8)
    grad_clip = float(input("Grad clip (default 1.0): ") or 1.0)

    vocab_size = int(input("Vocab size: "))
    d_model = int(input("d_model (default 256): ") or 256)
    num_layers = int(input("Num layers (default 4): ") or 4)
    num_heads = int(input("Num heads (default 8): ") or 8)
    d_ff = int(input("d_ff (default 1024): ") or 1024)
    theta = float(input("RoPE theta (default 10000.0): ") or 10000.0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = device if torch.cuda.is_available() else "cpu"

    train_data = np.memmap(train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode="r")

    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_model // num_heads,
        max_seq_len=context_length,
        device=device,
    )

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope=rope,
        device=device,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    start_iter = 0
    if resume and os.path.exists(checkpoint_path):
        start_iter = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    model.train()
    criterion = CrossEntropy()

    for iteration in range(start_iter, max_iters):
        lr = learning_rate_schedule(
            t=iteration,
            alpha_max=learning_rate,
            alpha_min=min_learning_rate,
            Tw=warmup_iters,
            Tc=cosine_end_iter,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, batch_size, context_length, device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        gradient_clipping(model.parameters(), grad_clip)
        optimizer.step()

        if iteration % log_interval == 0:
            print(f"iter {iteration}: step loss {loss.item():.4f}")

        if iteration % eval_interval == 0:
            losses = evaluate_loss(
                model,
                train_data,
                val_data,
                batch_size,
                context_length,
                eval_iters,
                device,
            )
            print(
                f"iter {iteration}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        if iteration % save_interval == 0 and iteration > start_iter:
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    save_checkpoint(model, optimizer, max_iters, checkpoint_path)
    print(f"Final checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()