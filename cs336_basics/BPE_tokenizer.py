from collections.abc import Iterable, Iterator
import json
import os
import regex as re
from tests.common import gpt2_bytes_to_unicode
from tests import adapters

class MyBPETokenizer:
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        self.encoder = {}
        self.decoder = {}
        self.bpe_ranks = {}
        self.cache = {}
        self.special_tokens = special_tokens if special_tokens else []

        if vocab and merges:
            self.vocab = vocab
            self.merges = merges
            self.decoder = vocab
            self.encoder = {v: k for k, v in vocab.items()}
            self.bpe_ranks = dict(zip(merges, range(len(merges))))

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(bytes([b]) for b in token)
        pairs = [word[i:i+2] for i in range(len(word)-1)]

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word)-1 and word[i] == first and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = [word[i:i+2] for i in range(len(word)-1)]

        self.cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        bpe_tokens = []
        
        # Handle special tokens splitting
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_tokens_pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
            segments = re.split(f"({special_tokens_pattern})", text)
        else:
            segments = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for segment in segments:
            if not segment:
                continue
            
            # If segment is a special token
            if self.special_tokens and segment in self.special_tokens:
                # Special tokens are stored as bytes in encoder
                token_bytes = segment.encode('utf-8')
                if token_bytes in self.encoder:
                    bpe_tokens.append(self.encoder[token_bytes])
                continue

            # Regular text processing
            for match in re.finditer(PAT, segment, re.UNICODE):
                token = match.group(0)
                token_bytes = token.encode('utf-8')
                
                # Apply BPE
                word_tokens = self.bpe(token_bytes)
                
                # If result is a tuple of parts, map each part
                if isinstance(word_tokens, tuple):
                    for part in word_tokens:
                        if part in self.encoder:
                            bpe_tokens.append(self.encoder[part])
                # If result is single bytes object (rare case if logic changes, but safe to handle)
                elif word_tokens in self.encoder:
                    bpe_tokens.append(self.encoder[word_tokens])
        
        return bpe_tokens

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text)

    def decode(self, tokens: list[int]) -> str:
        text_bytes = b"".join(self.decoder[token] for token in tokens)
        return text_bytes.decode('utf-8', errors='replace')

    def train_from_corpus(self, corpus_file, vocab_size, special_tokens, auto_save=False, save_path_prefix=None):
        vocab, merges = adapters.run_train_bpe(
            input_path=corpus_file,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            timer_enable=True,
        )
        # Update internal state
        self.decoder = vocab
        self.encoder = {v: k for k, v in vocab.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.special_tokens = special_tokens
        
        # For saving, we need self.vocab and self.merges to match the save method expectation
        # The save method expects self.vocab to be {id: bytes} and self.merges to be list of (bytes, bytes)
        self.vocab = vocab 
        self.merges = merges

        if auto_save:
            corpus_name = os.path.splitext(os.path.basename(corpus_file))[0]
            vocab_path = f"{corpus_name}_bpe_vocab.json"
            merges_path = f"{corpus_name}_bpe_merges.txt"
            if save_path_prefix:
                os.makedirs(save_path_prefix, exist_ok=True)
                vocab_path = os.path.join(save_path_prefix, os.path.basename(vocab_path))
                merges_path = os.path.join(save_path_prefix, os.path.basename(merges_path))
            self.save(vocab_path, merges_path)

    def from_files(self, vocab_path, merges_path, special_tokens=None):
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        def convert_token_str_to_bytes(token_str):
            return bytes([byte_decoder[c] for c in token_str])

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = {} # id -> bytes
        self.encoder = {} # bytes -> id
        for token_str, token_id in vocab_data.items():
            token_bytes = convert_token_str_to_bytes(token_str)
            self.vocab[int(token_id)] = token_bytes
            self.encoder[token_bytes] = int(token_id)
        
        self.decoder = self.vocab

        self.merges = []
        self.bpe_ranks = {}
        with open(merges_path, 'r', encoding='utf-8') as f:
            rank = 0
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                parts = line.split(' ')
                if len(parts) == 2:
                    pair = (
                        convert_token_str_to_bytes(parts[0]),
                        convert_token_str_to_bytes(parts[1])
                    )
                    self.merges.append(pair)
                    self.bpe_ranks[pair] = rank
                    rank += 1
        
        self.special_tokens = special_tokens if special_tokens else []
    
    def save(self, vocab_path, merges_path):
        # Get the byte to unicode mapping
        byte_encoder = gpt2_bytes_to_unicode()
        
        def convert_token_bytes_to_str(token_bytes):
            return "".join(byte_encoder[b] for b in token_bytes)

        vocab_to_save = {}
        for token_id, token_bytes in self.vocab.items():
            token_str = convert_token_bytes_to_str(token_bytes)
            vocab_to_save[token_str] = token_id

        with open(vocab_path, 'w', encoding='utf-8') as f: 
            json.dump(vocab_to_save, f, ensure_ascii=False, indent=4)

        with open(merges_path, 'w', encoding='utf-8') as f: 
            for token1_bytes, token2_bytes in self.merges:
                token1_str = convert_token_bytes_to_str(token1_bytes)
                token2_str = convert_token_bytes_to_str(token2_bytes)
                f.write(f"{token1_str} {token2_str}\n")

    
if __name__ == '__main__':
    # vocab = {
    #     0: b' ', 1: b'a', 2:
    #     b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'
    # }
    # merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'),
    #           (b' a', b't')]
    # tokenizer = MyBpeTokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    # decode_ex = tokenizer.encode("the cat ate")
    # print(decode_ex)
    # print(tokenizer.decode(decode_ex))
    vocab_path = 'tests/fixtures/gpt2_vocab.json'
    merges_path = 'tests/fixtures/gpt2_merges.txt'
    tokenizer = MyBPETokenizer()
    tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    test_string = " dedicated"
    encoded_ids = tokenizer.encode(test_string)
    print(encoded_ids)
    decoded_string = tokenizer.decode(encoded_ids)
    print(decoded_string)