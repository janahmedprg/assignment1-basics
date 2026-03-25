from typing import Iterable, Iterator
import ast
import regex as re
import os

class Tokenizer:

    def __init__(
        self,
        vocab : dict[int, bytes],
        merges : list[tuple[bytes, bytes]],
        special_tokens : list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.encode_map = {val : key for key, val in self.vocab.items()}

        self.pat_re = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.merge_rank = {b_pair: i for i, b_pair in enumerate(self.merges)}

        if self.special_tokens:
            special_tok_pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
            self.special_token_re = re.compile(f"({special_tok_pattern})")
        else:
            self.special_token_re = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r') as file:
            vocab = ast.literal_eval(file.read())

        with open(merges_filepath, 'r') as file:
            merges = ast.literal_eval(file.read())

        return cls(vocab, merges, special_tokens)


    def _merge_pretoken(self, pretoken : str):
        pretoken_b = [bytes([b]) for b in pretoken.encode("utf-8")]
        while (len(pretoken_b) > 1):
            best_idx = -1
            best_rank = float("inf")

            for i in range(len(pretoken_b) - 1):
                rank = self.merge_rank.get((pretoken_b[i], pretoken_b[i + 1]), float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break 

            pretoken_b = (
                pretoken_b[:best_idx] 
                + [pretoken_b[best_idx] + pretoken_b[best_idx + 1]]
                + pretoken_b[best_idx + 2:]
            )

        return pretoken_b


    def encode(self, text: str) -> list[int]:
        encoded_bytes = []
        if self.special_tokens:
            special_token_split = self.special_token_re.split(text)
        else:
            special_token_split = [text]
        
        for text_split in special_token_split:
            if self.special_tokens and text_split in self.special_tokens:
                encoded_bytes.append(self.encode_map[text_split.encode("utf-8")])
                continue
            
            for pretoken_match in self.pat_re.finditer(text_split):
                merged_pretoken = self._merge_pretoken(pretoken_match.group())
                encoded_bytes.extend(self.encode_map[b] for b in merged_pretoken)
        
        return encoded_bytes


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        bytes_str = b"".join(
            self.vocab.get(bid, b"\xef\xbf\xbd")
            for bid in ids
        )
        return bytes_str.decode("utf-8", errors="replace")


# script_dir = os.path.dirname(os.path.abspath(__file__))

# vocab_path = os.path.join(script_dir, "..", "vocab.txt")
# merges_path = os.path.join(script_dir, "..", "merges.txt")

# tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])

# print(tokenizer.encode("Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"))
# print(tokenizer.decode([1202, 45, 763, 33, 0, 0, 488, 350, 64, 0]))