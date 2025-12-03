python  cs336_basics/generate_text.py --model_checkpoint logs/train_exp_6/checkpoints/last.pth \
    --vocab_path data/TinyStoriesV2-GPT4-train_bpe_vocab.json \
    --merges_path data/TinyStoriesV2-GPT4-train_bpe_merges.txt \
    --model_config logs/train_exp_6/config.json \
    --prompt "Once upon a time" \
    --max_length 100 \
    --temperature 1.0 \
    --top_p 1.0 \