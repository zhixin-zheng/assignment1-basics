# trains a BPE tokenizer on a given text corpus and saves the vocabulary and merges.

import os
import argparse
from cs336_basics.BPE_tokenizer import MyBPETokenizer        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument('--input_file', type=str, default="/data/zzx/test/data/TinyStoriesV2-GPT4-train.txt", required=True, help='Path to the input text corpus.')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Desired vocabulary size.')
    parser.add_argument('--special_tokens', type=str, nargs='*', default=["<|endoftext|>"], help='List of special tokens to include in the vocabulary.')
    parser.add_argument('--auto_save', action='store_true', help='Automatically save the trained vocab and merges to files.')
    parser.add_argument('--save_path_prefix', default="/data/zzx/test/assignment1-basics/data", type=str, help='Prefix path to save vocab and merges files.')
    args = parser.parse_args()

    input_file_path = args.input_file
    vocab_size = args.vocab_size
    special_tokens = args.special_tokens
    auto_save = args.auto_save
    save_path_prefix = args.save_path_prefix

    BPE_tokenizer = MyBPETokenizer()
    BPE_tokenizer.train_from_corpus(
        corpus_file=input_file_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        auto_save=auto_save,
        save_path_prefix=save_path_prefix,
    )
