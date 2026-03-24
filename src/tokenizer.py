from typing import Iterable, Iterator


class Tokenizer:

    def __init__(
        self,
        vocab : dict[int, bytes],
        merges : list[tuple[bytes, bytes]],
        special_tokens : list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges

        idx = len(vocab)
        for spt in special_tokens:
            vocab[idx] = spt
            idx += 1

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass
    