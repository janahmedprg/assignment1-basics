import os
import numpy as np
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
from src.tokenizer import Tokenizer
import codecs


n_proc = int(input("Enter number of processes: "))
NUM_PROCESSES = n_proc

n_chunks = int(input("Enter number of chunks: "))
N_CHUNKS = n_chunks

file_name = input("Enter file name: ")
INPUT_PATH = "data/" + file_name
tokenizer_type = input("Enter tokenizer (type \"tiny\" or \"owt\"): ")
VOCAB_PATH = "tokenizer_params/" + tokenizer_type + "_vocab.txt"
MERGES_PATH = "tokenizer_params/" + tokenizer_type + "_merges.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]

output_file_name = input("Enter output file name: ")
OUTPUT_PATH = "tokenized_data/" + output_file_name
TMP_DIR = "tokenized_data/tmp_tokenization"

TOKENIZER = None


def init_worker(vocab_path: str, merges_path: str, special_tokens: list[str]) -> None:
    global TOKENIZER
    TOKENIZER = Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def bounded_text_reader(file_obj, num_bytes: int, chunk_size: int = 1 << 20):
    decoder = codecs.getincrementaldecoder("utf-8")()
    remaining = num_bytes

    while remaining > 0:
        to_read = min(chunk_size, remaining)
        data = file_obj.read(to_read)
        if not data:
            break
        remaining -= len(data)

        text = decoder.decode(data, final=False)
        if text:
            yield text

    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail

def encode_job(job_args: tuple[int, str, int, int, str]) -> tuple[int, str]:
    shard_idx, corpus_path, start, end, tmp_dir = job_args
    shard_path = os.path.join(tmp_dir, f"shard_{shard_idx:05d}.bin")

    with open(corpus_path, "rb") as file, open(shard_path, "wb") as out:
        file.seek(start)

        token_buffer: list[int] = []
        flush_size = 100_000

        text_iter = bounded_text_reader(file, end - start)

        for token_id in TOKENIZER.encode_iterable(text_iter):
            token_buffer.append(token_id)

            if len(token_buffer) >= flush_size:
                np.array(token_buffer, dtype=np.uint16).tofile(out)
                token_buffer.clear()

        if token_buffer:
            np.array(token_buffer, dtype=np.uint16).tofile(out)

    return shard_idx, shard_path


def concatenate_shards(shard_paths: list[str], output_path: str) -> None:
    with open(output_path, "wb") as out:
        for shard_path in shard_paths:
            with open(shard_path, "rb") as shard_file:
                while True:
                    chunk = shard_file.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS)
    max_token_id = max(tokenizer.vocab.keys())
    if max_token_id >= 65536:
        raise ValueError(
            f"Max token id is {max_token_id}, which does not fit in uint16."
        )

    jobs: list[tuple[int, str, int, int, str]] = []

    with open(INPUT_PATH, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            N_CHUNKS,
            b"<|endoftext|>",
        )

    for shard_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if start == end:
            continue
        jobs.append((shard_idx, INPUT_PATH, start, end, TMP_DIR))

    with Pool(
        processes=NUM_PROCESSES,
        initializer=init_worker,
        initargs=(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS),
    ) as pool:
        results = pool.map(encode_job, jobs)

    results.sort(key=lambda x: x[0])
    shard_paths = [path for _, path in results]

    concatenate_shards(shard_paths, OUTPUT_PATH)

    for shard_path in shard_paths:
        os.remove(shard_path)
    os.rmdir(TMP_DIR)


if __name__ == "__main__":
    main()