from typing import Iterable, Iterator
import ast
import regex as re

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
        buffer = ""

        for chunk in iterable:
            if not chunk:
                continue

            buffer += chunk
            partial_special_len = self._partial_special_suffix_len(buffer)
            processable = buffer[: len(buffer) - partial_special_len]

            if self.special_tokens:
                last_end = 0
                for special_match in self.special_token_re.finditer(processable):
                    segment = processable[last_end : special_match.start()]
                    for m in self.pat_re.finditer(segment):
                        yield from self._encode_text_segment(m.group())
                    yield self.encode_map[special_match.group().encode("utf-8")]
                    last_end = special_match.end()
                trailing_text = processable[last_end:]
            else:
                trailing_text = processable

            matches = list(self.pat_re.finditer(trailing_text))
            for pretoken_match in matches[:-1]:
                yield from self._encode_text_segment(pretoken_match.group())

            if matches:
                buffer = trailing_text[matches[-1].start():] + buffer[len(processable):]
            else:
                buffer = trailing_text + buffer[len(processable):]

        if buffer:
            yield from self.encode(buffer)

    def _encode_text_segment(self, text: str) -> Iterator[int]:
        merged_pretoken = self._merge_pretoken(text)
        yield from (self.encode_map[b] for b in merged_pretoken)

    def _partial_special_suffix_len(self, text: str) -> int:
        if not self.special_tokens:
            return 0

        max_suffix_len = 0
        # Note the special_tokens are sorted by length descending order
        max_check_len = min(len(text), len(self.special_tokens[0]) - 1)
        for suffix_len in range(1, max_check_len + 1):
            suffix = text[-suffix_len:]
            if any(tok.startswith(suffix) for tok in self.special_tokens):
                max_suffix_len = suffix_len

        return max_suffix_len

    def decode(self, ids: list[int]) -> str:
        bytes_str = b"".join(
            self.vocab.get(bid, b"\xef\xbf\xbd")
            for bid in ids
        )
        return bytes_str.decode("utf-8", errors="replace")
