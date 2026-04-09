from src.data_loading import get_batch
from src.transformer_lm import TransformerLM
from src.adamw import AdamW
from src.checkpointing import load_checkpoint, save_checkpoint
from src.rope import RotaryPositionalEmbedding
from src.cross_entropy import CrossEntropy
from src.gradient_clipping import gradient_clipping
from src.learning_rate_schedule import learning_rate_schedule
import random
import time
import json
import csv
from pathlib import Path

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


def ensure_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def write_run_config(exp_dir: Path, config: dict) -> None:
    config_path = exp_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_run_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def initialize_metrics_csv(csv_path: Path) -> None:
    if csv_path.exists():
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event",
                "iteration",
                "wallclock_seconds",
                "learning_rate",
                "train_step_loss",
                "train_eval_loss",
                "val_eval_loss",
            ]
        )


def append_metrics_csv(
    csv_path: Path,
    event: str,
    iteration: int,
    wallclock_seconds: float,
    learning_rate: float,
    train_step_loss: float | None = None,
    train_eval_loss: float | None = None,
    val_eval_loss: float | None = None,
) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                event,
                iteration,
                wallclock_seconds,
                learning_rate,
                train_step_loss,
                train_eval_loss,
                val_eval_loss,
            ]
        )


def append_metrics_jsonl(
    jsonl_path: Path,
    event: str,
    iteration: int,
    wallclock_seconds: float,
    learning_rate: float,
    train_step_loss: float | None = None,
    train_eval_loss: float | None = None,
    val_eval_loss: float | None = None,
) -> None:
    record = {
        "event": event,
        "iteration": iteration,
        "wallclock_seconds": wallclock_seconds,
        "learning_rate": learning_rate,
        "train_step_loss": train_step_loss,
        "train_eval_loss": train_eval_loss,
        "val_eval_loss": val_eval_loss,
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_experiment_log(log_path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def main():
    use_config = input("Load from config? (y/n): ").strip().lower() == "y"

    if use_config:
        config_path = input("Path to config.json: ")
        cfg = load_run_config(config_path)

        train_data_path = cfg["train_data_path"]
        val_data_path = cfg["val_data_path"]
        batch_size = cfg["batch_size"]
        context_length = cfg["context_length"]
        max_iters = cfg["max_iters"]
        eval_interval = cfg["eval_interval"]
        eval_iters = cfg["eval_iters"]
        log_interval = cfg["log_interval"]
        save_interval = cfg["save_interval"]
        checkpoint_path = cfg["checkpoint_path"]
        resume = cfg["resume"]
        seed = cfg["seed"]
        device = cfg["device"]
        learning_rate = cfg["learning_rate"]
        min_learning_rate = cfg["min_learning_rate"]
        warmup_iters = cfg["warmup_iters"]
        cosine_end_iter = cfg["cosine_end_iter"]
        weight_decay = cfg["weight_decay"]
        beta1 = cfg["beta1"]
        beta2 = cfg["beta2"]
        eps = cfg["eps"]
        grad_clip = cfg["grad_clip"]
        vocab_size = cfg["vocab_size"]
        d_model = cfg["d_model"]
        num_layers = cfg["num_layers"]
        num_heads = cfg["num_heads"]
        d_ff = cfg["d_ff"]
        theta = cfg["theta"]
        experiment_root = cfg["experiment_root"]
        experiment_name = cfg["experiment_name"]
    else:
        train_data_path = input("Train data path: ")
        val_data_path = input("Validation data path: ")

        batch_size = int(input("Batch size (default 32): ") or 32)
        context_length = int(input("Context length (default 256): ") or 256)
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

        experiment_root = input("Experiment root dir (default experiments): ") or "experiments"
        experiment_name = input("Experiment name (default run): ") or "run"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = device if torch.cuda.is_available() else "cpu"

    exp_dir = ensure_experiment_dir(experiment_root, experiment_name)
    metrics_csv_path = exp_dir / "metrics.csv"
    metrics_jsonl_path = exp_dir / "metrics.jsonl"
    experiment_log_path = exp_dir / "experiment_log.txt"
    initialize_metrics_csv(metrics_csv_path)

    run_config = {
        "train_data_path": train_data_path,
        "val_data_path": val_data_path,
        "batch_size": batch_size,
        "context_length": context_length,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "eval_iters": eval_iters,
        "log_interval": log_interval,
        "save_interval": save_interval,
        "checkpoint_path": checkpoint_path,
        "resume": resume,
        "seed": seed,
        "device": device,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "cosine_end_iter": cosine_end_iter,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "grad_clip": grad_clip,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "theta": theta,
        "experiment_root": experiment_root,
        "experiment_name": experiment_name,
    }
    write_run_config(exp_dir, run_config)

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

    start_iter = 1
    if resume and os.path.exists(checkpoint_path):
        start_iter = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    model.train()
    criterion = CrossEntropy()
    train_start_time = time.perf_counter()

    for iteration in range(start_iter, max_iters + 1):
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

        wallclock_seconds = time.perf_counter() - train_start_time

        if iteration % log_interval == 0:
            print(f"iter {iteration}: step loss {loss.item():.4f}")
            append_metrics_csv(
                metrics_csv_path,
                "train_step",
                iteration,
                wallclock_seconds,
                lr,
                loss.item(),
            )

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
            append_metrics_csv(
                metrics_csv_path,
                "eval",
                iteration,
                wallclock_seconds,
                lr,
                None,
                losses["train"],
                losses["val"],
            )

        if iteration % save_interval == 0 and iteration > start_iter:
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    save_checkpoint(model, optimizer, max_iters, checkpoint_path)
    print(f"Final checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()