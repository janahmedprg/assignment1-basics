import gc
import os
import regex as re

from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"

PAT_RE = re.compile(PAT)
EOT_RE = re.compile(re.escape(EOT))

def cnt_freq(chunk: BinaryIO):
    word_freq = {}

    prev_end = 0
    for eot_indices in EOT_RE.finditer(chunk):
        eot_start, eot_end = eot_indices.span()

        for word_indices in PAT_RE.finditer(chunk, prev_end, eot_start):
            word = word_indices.group()
            word_freq[word] = word_freq.get(word, 0) + 1

        prev_end = eot_end

    return word_freq

def cnt_freq_job(corpus_path, start, end):
    with open(corpus_path, 'rb') as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
        return cnt_freq(chunk)

def train_bpe(corpus_path: str, vocab_size: int, special_tokens: list[str]):
    assert vocab_size >= 256, "Vocab size must be at least 256"

    N_CHUNKS = 32
    NUM_PROCESSES = 8

    chunk_ranges = []
    with open(corpus_path, 'rb') as f:
        chunks = find_chunk_boundaries(f, N_CHUNKS, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(chunks[:-1], chunks[1:]):
            chunk_ranges.append((corpus_path, start, end))

    token_freq = {}
    with Pool(NUM_PROCESSES) as p:
        word_freq_shards = p.starmap(cnt_freq_job, chunk_ranges)
        for chunk_freq_map in word_freq_shards:
            for token, freq in chunk_freq_map.items():
                token_freq[token] = token_freq.get(token, 0) + freq


script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, "..", "data/owt_train.txt")

train_bpe(file_path, 10000, ["<|endoftext|>"])

# print(max(bpe[0].values(), key=len))

# with open('bpe_output.txt', 'w') as f:
#     f.write('Vocabulary = ' + str(bpe[0]) + '\n')
#     f.write('Merges = ' + str(bpe[1]) + '\n')