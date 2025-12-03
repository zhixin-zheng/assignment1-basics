import torch
import json
import argparse
from tqdm import tqdm
from cs336_basics import LM, MyBPETokenizer, load_checkpoint

class TextGenerator:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 1.0, top_p: float = 1.0) -> str:
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        generated_ids = input_tensor

        for _ in tqdm(range(max_length), desc="Generating text"):
            outputs = self.model(generated_ids)
            logits = outputs[:, -1, :] / temperature

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

            probabilities = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)

            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

        generated_ids = generated_ids.squeeze().tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text

def load_tokenizer(vocab_path: str, merges_path: str, special_tokens=None) -> MyBPETokenizer:
    tokenizer = MyBPETokenizer()
    tokenizer.from_files(vocab_path=vocab_path, merges_path=merges_path, special_tokens=special_tokens)
    return tokenizer

def load_model_checkpoint(model_class, config_path, checkpoint_path: str, device='cpu'):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = model_class(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        num_layers=config['num_layers'],
        theta=config.get('rope_theta', 10000.0)
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Text Generation using a trained Transformer model.")

    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the BPE vocabulary file.')
    parser.add_argument('--merges_path', type=str, required=True, help='Path to the BPE merges file.')
    parser.add_argument('--special_tokens', type=str, nargs='*', default=["<|endoftext|>"], help='List of special tokens used in the tokenizer.')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration JSON file.')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt text for generation.')
    parser.add_argument('--output_file', type=str, default='data/generated_text.txt', help='File to save the generated text.')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of generated text.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p (nucleus) sampling parameter.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cpu or cuda).')

    args = parser.parse_args()

    tokenizer = load_tokenizer(args.vocab_path, args.merges_path, args.special_tokens)

    model = load_model_checkpoint(LM, args.model_config, args.model_checkpoint, device=args.device)

    text_generator = TextGenerator(model, tokenizer, device=args.device)
    generated_text = text_generator.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print("Generated Text:\n-------------------")
    print(generated_text)

    with open(args.output_file, 'w') as f:
        f.write(generated_text)

    print(f"--------------------\nGenerated text saved to {args.output_file}")