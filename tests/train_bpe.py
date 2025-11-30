import os
import json

import adapters
from common import gpt2_bytes_to_unicode


class MyBPETokenizer:
    def __init__(self, vocal=None, merges=None, special_tokens=None):
        if vocal:
            if os.path.isfile(vocal):
                with open(vocal, 'r', encoding='utf-8') as f:
                    self.vocab = [line.strip() for line in f]
            else:
                raise ValueError(f"Vocab file {vocal} does not exist.")
        if merges:
            if os.path.isfile(merges):
                with open(merges, 'r', encoding='utf-8') as f:
                    self.merges = [line.strip() for line in f]
            else:
                raise ValueError(f"Merges file {merges} does not exist.")
        self.special_tokens = special_tokens if special_tokens else []

    def encode(self, text):
        pass

    def train_from_corpus(self, corpus_file, vocab_size, special_tokens, auto_save=False, save_path_prefix=None):
        vocab, merges = adapters.run_train_bpe(
            input_path=corpus_file,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            timer_enable=True,
        )
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        if auto_save:
            corpus_name = os.path.splitext(os.path.basename(corpus_file))[0]
            vocab_path = f"{corpus_name}_bpe_vocab.json"
            merges_path = f"{corpus_name}_bpe_merges.txt"
            if save_path_prefix:
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
        
        self.vocab = {}
        for token_str, token_id in vocab_data.items():
            self.vocab[token_id] = convert_token_str_to_bytes(token_str)

        self.merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                parts = line.split(' ')
                if len(parts) == 2:
                    self.merges.append((
                        convert_token_str_to_bytes(parts[0]),
                        convert_token_str_to_bytes(parts[1])
                    ))
        
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

    input_file_path = '/data/zzx/test/data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = ['<|endoftext>|']

    BPE_tokenizer = MyBPETokenizer()
    BPE_tokenizer.train_from_corpus(
        corpus_file=input_file_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        auto_save=True,
        save_path_prefix='/data/zzx/test/assignment1-basics/cs336_basics',
    )