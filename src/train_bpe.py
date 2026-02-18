from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from multiprocessing import Pool
import os

class BytePairMap:

    def __init__(self, pre_token_dict: dict[tuple[int, ...], int]):
        self.byte_pairs_freq: dict[tuple[bytes, bytes], int] = {}
        self.byte_pairs_indices: dict[tuple[bytes, bytes], list[tuple[int, ...]]] = {}

        for key in pre_token_dict.keys():
            for i in range(len(key)-1):
                byte_pair = (bytes([key[i]]), bytes([key[i+1]]))
                get_freq_val = self.byte_pairs_freq.get(byte_pair, 0)
                get_indices_val = self.byte_pairs_indices.get(byte_pair, [])
                self.byte_pairs_freq[byte_pair] = get_freq_val + pre_token_dict[key]
                self.byte_pairs_indices[byte_pair] = get_indices_val + [key]

    def sort(self) -> list[tuple[tuple[bytes, bytes], int]]:
        return sorted(self.byte_pairs_freq.items(), key=lambda item: (item[1], item[0]), reverse=True)
    
    def getIndices(self, byte_pair: tuple[bytes, bytes]) -> list[tuple[int, ...]]:
        return self.byte_pairs_indices[byte_pair]
    
    def subtractFreq(self, byte_pair: tuple[bytes, bytes], ammount: int):
        self.byte_pairs_freq[byte_pair] -= ammount

        if self.byte_pairs_freq[byte_pair] == 0:
            del self.byte_pairs_freq[byte_pair]
            del self.byte_pairs_indices[byte_pair]
    
    def addBytePair(self, byte_pair: tuple[bytes, bytes], freq: int, index: tuple[int, ...]):
        self.byte_pairs_freq[byte_pair] = freq + self.byte_pairs_freq.get(byte_pair, 0)
        self.byte_pairs_indices[byte_pair] = [index] + self.byte_pairs_indices.get(byte_pair, [])
    
    def deleteBytePair(self, byte_pair: tuple[bytes, bytes]):
        del self.byte_pairs_freq[byte_pair]
        del self.byte_pairs_indices[byte_pair]
    
    def isEmpty(self) -> bool:
        if len(self.byte_pairs_freq) == 0:
            return True
        return False



def pre_tokenize(chunk: str, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
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
    vocab = {}

    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode("utf-8")

    vocab_index = len(special_tokens)

    for i in range(256):
        vocab[vocab_index + i] = bytes([i])
    
    vocab_index += 256
    
    # Read file using parallel processing
    with open(input_path, "rb") as f:
        num_processes = 32
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        f.seek(boundaries[0])

        async_results = []
        with Pool() as pool:
            for start, end in zip(boundaries[:-1], boundaries[1:]): 
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                async_results.append(pool.apply_async(pre_tokenize, args=(chunk, special_tokens)))
            
            results = [r.get() for r in async_results]
        
    # Start pre-tokenization
    pre_token_dict: dict[tuple[int, ...], int] = {}
    for r in results:
        combined_tokens = set(pre_token_dict.keys()) | set(r.keys())
        pre_token_dict = {k: pre_token_dict.get(k, 0) + r.get(k, 0) for k in combined_tokens}

    # Make byte pairs
    byte_pairs = BytePairMap(pre_token_dict)
    

    # Start merging step
    merge_list: list[tuple[bytes, bytes]] = []
    merged_pre_tokens: dict[tuple[int, ...], tuple[bytes, ...]] = {}
    for _ in range(vocab_size - 256 - len(special_tokens)):
        if byte_pairs.isEmpty():
            break
        sorted_byte_pairs = byte_pairs.sort()
        merge_list.append((sorted_byte_pairs[0][0][0], sorted_byte_pairs[0][0][1])) # Merge the pairs
        vocab[vocab_index] = sorted_byte_pairs[0][0][0] + sorted_byte_pairs[0][0][1]
        vocab_index += 1
        
        most_freq_byte_pair_indices = byte_pairs.getIndices((sorted_byte_pairs[0][0][0], sorted_byte_pairs[0][0][1]))

        # Iterate over indices
        for pre_token in most_freq_byte_pair_indices:
            # Check if index has any merges if not return unmerged pre-token
            merged_pre_token = merged_pre_tokens.get(pre_token, tuple(bytes([b]) for b in pre_token))
            # Iterate over (merged) bytes of the pre-token index to find the (merged) bytes pair
            new_merged_pre_token = ()
            i = 0
            while i < len(merged_pre_token):
                if i < len(merged_pre_token) - 1 and merged_pre_token[i] == sorted_byte_pairs[0][0][0] and merged_pre_token[i+1] == sorted_byte_pairs[0][0][1]:
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

        byte_pairs.deleteBytePair((sorted_byte_pairs[0][0][0], sorted_byte_pairs[0][0][1]))

    return (vocab, merge_list)

bpe = train_bpe("/assignment1-basics/data/owt_train.txt", 32000, ["<|endoftext|>"])

print(max(bpe[0].values(), key=len))

with open('bpe_output.txt', 'w') as f:
    f.write('Vocabulary = ' + str(bpe[0]) + '\n')
    f.write('Merges = ' + str(bpe[1]) + '\n')