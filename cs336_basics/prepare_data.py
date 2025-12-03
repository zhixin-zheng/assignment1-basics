import os
import json
import numpy as np
import re
import argparse
from typing import List, Dict
from cs336_basics import MyBPETokenizer
from tqdm import tqdm

def load_bpe_tokenizer(vocab_path: str, merge_path: str, special_tokens: List[str] = None) -> MyBPETokenizer:
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    
    tokenizer = MyBPETokenizer()
    tokenizer.from_files(vocab_path, merge_path, special_tokens=special_tokens)
    return tokenizer

def process_txt_file(input_text_path: str, tokenizer: MyBPETokenizer) -> List[int]:
    token_ids = []
    with open(input_text_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Processing txt lines"):
            if line:
                line_token_ids = tokenizer.encode(line)
                token_ids.extend(line_token_ids)
    return token_ids

def main():
    parser = argparse.ArgumentParser(description="Prepare data using BPE tokenizer.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the BPE vocabulary JSON file.")
    parser.add_argument("--merge_path", type=str, required=True, help="Path to the BPE merges file.")
    parser.add_argument("--input_text_path", type=str, 
                        default="data/input.txt", required=True, help="Path to the input text file.")
    parser.add_argument("--output_data_dir", type=str, 
                        default="data/", help="Path to save the processed token IDs (as .bin file).")
    args = parser.parse_args()

    tokenizer = load_bpe_tokenizer(args.vocab_path, args.merge_path)

    token_ids = process_txt_file(args.input_text_path, tokenizer)

    # print("Decoded sample:", tokenizer.decode(token_ids[:100]))  # Decode first 100 tokens for verification

    data = np.array(token_ids, dtype=np.uint16)
    output_data_path = os.path.join(args.output_data_dir, args.input_text_path.split('/')[-1].replace('.txt', '_bpe_token_ids.bin'))
    data.tofile(output_data_path)
    stats = {
        'vocab_size': len(tokenizer.vocab),
        'total_tokens': len(data),
        'vocab_path': args.vocab_path,
        'merges_path': args.merge_path,
        'output_file': output_data_path
    }
    stats_path = output_data_path.replace('.bin', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Processed data saved to {output_data_path}")
    print(f"Total tokens: {len(token_ids)}")

if __name__ == "__main__":
    main()