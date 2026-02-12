from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from multiprocessing import Pool
import os

file_path = "/home/janahmed/Desktop/code/assignment1-basics/data/test_data.txt" # TinyStoriesV2-GPT4-valid.txt"

special_tokens = ["<|endoftext|>"]

with open(file_path, "r") as f:
    text = f.read()
    print(text)
    print("----------------------------------------------------------------------------------")

def pre_tokenize(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    special_tok_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    splitted_chunk = re.split(f"({special_tok_pattern})", chunk)

    token_freq_dict = {}

    for chunk_split in splitted_chunk:
        if chunk_split in special_tokens:
            continue

        for pre_token in re.finditer(PAT, chunk_split):
            token_byte_tuple = tuple(pre_token.group().encode("utf-8"))
            token_freq_dict[token_byte_tuple] = token_freq_dict.get(token_byte_tuple, 0) + 1

    return token_freq_dict



def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i : bytes([i]) for i in range(256)}
    
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        f.seek(boundaries[0])

        async_results = []
        with Pool() as pool:
            for start, end in zip(boundaries[:-1], boundaries[1:]): 
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                async_results.append(pool.apply_async(pre_tokenize, args=(chunk, special_tokens)))
            
            results = [r.get() for r in async_results]
        
    merged_dicts = {}
    for r in results:
        combined_tokens = set(merged_dicts.keys()) | set(r.keys())
        merged_dicts = {k: merged_dicts.get(k, 0) + r.get(k, 0) for k in combined_tokens}

    byte_pairs = {}
    for key in merged_dicts.keys():
        for i in range(len(key)-1):
            byte_pair = (key[i], key[i+1])
            get_key = byte_pairs.get(byte_pair, (0, []))
            byte_pairs[byte_pair] = (get_key[0] + merged_dicts[key], get_key[1] + [key])

    print(byte_pairs)

train_bpe("/home/janahmed/Desktop/code/assignment1-basics/data/test_data.txt", 1000, ["<|endoftext|>"])