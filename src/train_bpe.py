from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from multiprocessing import Pool
import os

class BytePairMap:

    def __init__(self, pre_token_dict: dict[bytes, int]):
        self.byte_pairs_freq: dict[tuple[bytes, bytes], int] = {}
        self.byte_pairs_indices: dict[tuple[bytes, bytes], set[str]] = {}

        for key in pre_token_dict.keys():
            for i in range(len(key) - 1):
                byte_pair = (bytes([key[i]]), bytes([key[i+1]]))
                self.byte_pairs_freq[byte_pair] = self.byte_pairs_freq.get(byte_pair, 0) + pre_token_dict[key]
                self.byte_pairs_indices.setdefault(byte_pair, set()).add(key)

    def getMaxFreq(self) -> tuple[bytes, bytes]:
        maxKey = (b"", b"")
        maxVal = 0 
        for key, value in self.byte_pairs_freq.items():
            if maxVal < value:
                maxKey = key
                maxVal = value
            elif maxVal == value and maxKey < key:
                maxKey = key
        return maxKey

    
    def getIndices(self, byte_pair: tuple[bytes, bytes]) -> list[tuple[int, ...]]:
        return self.byte_pairs_indices[byte_pair]
    
    def subtractFreq(self, byte_pair: tuple[bytes, bytes], ammount: int):
        self.byte_pairs_freq[byte_pair] -= ammount
        if self.byte_pairs_freq[byte_pair] == 0:
            del self.byte_pairs_freq[byte_pair]
            del self.byte_pairs_indices[byte_pair]
    
    def addBytePair(self, byte_pair: tuple[bytes, bytes], freq: int, index: str):
        self.byte_pairs_freq[byte_pair] = freq + self.byte_pairs_freq.get(byte_pair, 0)
        self.byte_pairs_indices.setdefault(byte_pair, set()).add(index)
    
    def deleteBytePair(self, byte_pair: tuple[bytes, bytes]):
        del self.byte_pairs_freq[byte_pair]
        del self.byte_pairs_indices[byte_pair]
    
    def isEmpty(self) -> bool:
        if len(self.byte_pairs_freq) == 0:
            return True
        return False

def pre_tokenize(chunk_bytes: bytes, special_tokens: list[str]) -> dict[bytes, int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tok_pattern = "|".join(re.escape(tok) for tok in special_tokens)

    chunk_str = chunk_bytes.decode("utf-8", errors="ignore")

    token_freq_dict = {}
    for chunk_split in re.split(f"({special_tok_pattern})", chunk_str):
        if chunk_split in special_tokens:
            continue
        for pre_token in re.finditer(PAT, chunk_split):
            token_byte = pre_token.group().encode("utf-8")
            token_freq_dict[token_byte] = token_freq_dict.get(token_byte, 0) + 1

    return token_freq_dict

def pre_tokenize_job(corpus_path, start, end, special_tokens):
    with open(corpus_path, 'rb') as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)
        return pre_tokenize(chunk_bytes, special_tokens)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    NUM_PROCESSES = 16
    N_CHUNKS = 48
    vocab = {}

    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode("utf-8")

    vocab_index = len(special_tokens)

    for i in range(256):
        vocab[vocab_index + i] = bytes([i])
    
    vocab_index += 256

    # Start pre-tokenization
    chunk_ranges = []
    with open(input_path, 'rb') as f:
        chunks = find_chunk_boundaries(f, N_CHUNKS, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(chunks[:-1], chunks[1:]):
            chunk_ranges.append((input_path, start, end, special_tokens))

    pre_token_dict = {}
    with Pool(NUM_PROCESSES) as p:
        word_freq_shards = p.starmap(pre_tokenize_job, chunk_ranges)
        for chunk_freq_map in word_freq_shards:
            for token, freq in chunk_freq_map.items():
                pre_token_dict[token] = pre_token_dict.get(token, 0) + freq

    # Make byte pairs
    byte_pairs = BytePairMap(pre_token_dict)
    

    # Start merging step
    merge_list: list[tuple[bytes, bytes]] = []
    merged_pre_tokens: dict[tuple[int, ...], tuple[bytes, ...]] = {}
    for _ in range(vocab_size - 256 - len(special_tokens)):
        if byte_pairs.isEmpty():
            break
        max_byte_pairs = byte_pairs.getMaxFreq()
        merge_list.append((max_byte_pairs[0], max_byte_pairs[1])) # Merge the pairs
        vocab[vocab_index] = max_byte_pairs[0] + max_byte_pairs[1]
        vocab_index += 1
        
        most_freq_byte_pair_indices = byte_pairs.getIndices((max_byte_pairs[0], max_byte_pairs[1]))

        # Iterate over indices
        for pre_token in most_freq_byte_pair_indices:
            # Check if index has any merges if not return unmerged pre-token
            merged_pre_token = merged_pre_tokens.get(pre_token, tuple(bytes([b]) for b in pre_token))
            # Iterate over (merged) bytes of the pre-token index to find the (merged) bytes pair
            new_merged_pre_token = ()
            i = 0
            while i < len(merged_pre_token):
                if i < len(merged_pre_token) - 1 and merged_pre_token[i] == max_byte_pairs[0] and merged_pre_token[i+1] == max_byte_pairs[1]:
                    # Make appropriate changes to the byte to the left of the pair
                    if i != 0:
                        # Subtract byte pair freq by the amount of freq of pre-token
                        byte_pairs.subtractFreq((new_merged_pre_token[-1], merged_pre_token[i]), pre_token_dict[pre_token])
                        # Add merged byte pair with left byte to byte_pairs
                        byte_pairs.addBytePair((new_merged_pre_token[-1], merged_pre_token[i] + merged_pre_token[i + 1]), pre_token_dict[pre_token], pre_token)
                                                
                    # Make appropriate changes to the byte to the right of the pair
                    if i != len(merged_pre_token) - 2:
                        # Subtract byte pair freq by the amount of freq of pre-token
                        byte_pairs.subtractFreq((merged_pre_token[i+1], merged_pre_token[i+2]), pre_token_dict[pre_token])
                        # Add merged byte pair with right byte to byte_pairs
                        byte_pairs.addBytePair((merged_pre_token[i] + merged_pre_token[i+1], merged_pre_token[i + 2]), pre_token_dict[pre_token], pre_token)
                        
                    new_merged_pre_token += (merged_pre_token[i] + merged_pre_token[i+1],)
                    i += 1
                else:
                    new_merged_pre_token += (merged_pre_token[i],)
                
                i += 1

            merged_pre_tokens[pre_token] = new_merged_pre_token

        byte_pairs.deleteBytePair((max_byte_pairs[0], max_byte_pairs[1]))

    return (vocab, merge_list)


script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, "..", "data/owt_train.txt")

bpe = train_bpe(file_path, 10000, ["<|endoftext|>"])

print(max(bpe[0].values(), key=len))

with open('bpe_output.txt', 'w') as f:
    f.write('Vocabulary = ' + str(bpe[0]) + '\n')
    f.write('Merges = ' + str(bpe[1]) + '\n')