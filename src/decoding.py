import torch
import json

from src.transformer_lm import TransformerLM
from src.checkpointing import load_checkpoint
from src.rope import RotaryPositionalEmbedding
from src.adamw import AdamW
from src.tokenizer import Tokenizer
from src.softmax import Softmax
import warnings
warnings.filterwarnings("ignore", message=".*hipBLASLt.*")


def load_run_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    keep_mask = cumulative_probs <= top_p
    keep_mask[..., 0] = True

    filtered_probs = sorted_probs * keep_mask
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples=1)
    sampled_token = sorted_indices.gather(dim=-1, index=sampled_sorted_idx)

    return sampled_token.squeeze(-1)


@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    context_length: int,
    device: str,
) -> list[int]:
    model.eval()

    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    softmax = Softmax()

    for _ in range(max_new_tokens):
        x = tokens[:, -context_length:]
        logits = model(x)
        next_token_logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_token_logits, dim=-1)
        else:
            scaled_logits = next_token_logits / temperature
            probs = softmax(scaled_logits, dim=-1)
            next_token = sample_top_p(probs, top_p)

        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == eos_token_id:
            break

    return tokens.squeeze(0).tolist()


def main():
    use_config = input("Load from config? (y/n): ").strip().lower() == "y"

    if use_config:
        config_path = input("Path to config.json: ")
        cfg = load_run_config(config_path)

        vocab_size = cfg["vocab_size"]
        context_length = cfg["context_length"]
        num_layers = cfg["num_layers"]
        d_model = cfg["d_model"]
        num_heads = cfg["num_heads"]
        d_ff = cfg["d_ff"]
        theta = cfg["theta"]
        device = cfg["device"]
        checkpoint_path = cfg["checkpoint_path"]

        vocab_path = input("Vocab path: ")
        merges_path = input("Merges path: ")

        max_new_tokens = int(input("Max new tokens (default 100): ") or 100)
        temperature = float(input("Temperature (default 1.0): ") or 1.0)
        top_p = float(input("Top-p (default 1.0): ") or 1.0)

    else:
        checkpoint_path = input("Checkpoint path (default checkpoint.pt): ") or "checkpoint.pt"
        vocab_path = input("Vocab path: ")
        merges_path = input("Merges path: ")

        vocab_size = int(input("Vocab size: "))
        context_length = int(input("Context length (default 256): ") or 256)
        num_layers = int(input("Num layers (default 4): ") or 4)
        d_model = int(input("d_model (default 256): ") or 256)
        num_heads = int(input("Num heads (default 8): ") or 8)
        d_ff = int(input("d_ff (default 1024): ") or 1024)
        theta = float(input("RoPE theta (default 10000.0): ") or 10000.0)

        device = input("Device (default cuda): ") or "cuda"

        max_new_tokens = int(input("Max new tokens (default 100): ") or 100)
        temperature = float(input("Temperature (default 1.0): ") or 1.0)
        top_p = float(input("Top-p (default 1.0): ") or 1.0)

    device = device if torch.cuda.is_available() else "cpu"

    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

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
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )

    loaded_iteration = load_checkpoint(checkpoint_path, model, optimizer)
    print(f"Loaded checkpoint from iteration {loaded_iteration}")

    print("\nType 'exit' to quit.\n")

    while True:
        prompt = input("Prompt: ")
        if prompt.strip().lower() in ["exit", "quit"]:
            break

        prompt_tokens = tokenizer.encode(prompt)

        generated_tokens = decode(
            model=model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            context_length=context_length,
            device=device,
        )

        print(tokenizer.decode(generated_tokens))
        print("-" * 50)


if __name__ == "__main__":
    main()