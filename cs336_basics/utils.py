import torch
import regex as re
import math
import numpy as np
from jaxtyping import Bool, Float, Int
from typing import Iterable, Optional
from torch import Tensor

def process_chunk_freqs(chunk_data, special_tokens, PAT):
    start, end, file_path = chunk_data

    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')
    
    special_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)
    segments = re.split(f"({special_tokens_pattern})", chunk)

    token_freqs = {}

    special_token_bytes = {token.encode('utf-8') for token in special_tokens}

    for segment in segments:
        if not segment or segment in special_tokens:
            continue
        
        for match in re.finditer(PAT, segment):
            token = match.group(0)
            token_bytes = token.encode('utf-8')
            if token_bytes in special_token_bytes:
                continue
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            token_freqs[token_tuple] = token_freqs.get(token_tuple, 0) + 1

    return token_freqs

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_values = torch.max(x, dim = dim, keepdim = True).values
    exp_x = torch.exp(x - max_values)
    return exp_x / torch.sum(exp_x, dim = dim, keepdim = True)

def cross_entropy_loss(logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    max_logits = logits.max(dim=-1, keepdim=True).values
    logits_stabilized = logits - max_logits # [batch_size, vocab_size]

    log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(logits_stabilized), dim=-1, keepdim=True))

    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1))

    loss = log_sum_exp - target_logits
    
    return loss.mean()

def get_lr_cosine_schedule(it, min_lr, max_lr, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if warmup_iters <= it and it <= cosine_cycle_iters:
        return min_lr + (max_lr - min_lr) * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) / 2.0
    return min_lr

def get_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if getattr(p, "grad", None) is not None]
    if not grads:
        return

    device = grads[0].device
    total_sq = torch.zeros((), device=device)
    for g in grads:
        total_sq = total_sq + g.detach().float().pow(2).sum()
    total_norm = torch.sqrt(total_sq)

    eps = 1e-6
    clip_coef = (max_l2_norm / (total_norm + eps)).item()
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)
            
    return total_norm

def get_batch(x: np.array, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = x.shape[0]
    starts = np.random.randint(0, num_tokens - context_length, size=batch_size)
    inputs = np.stack([x[start : start + context_length] for start in starts])
    targets = np.stack([x[start + 1 : start + context_length + 1] for start in starts])
    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)
    
def save_checkpoint(model, optimizer, iteration, out):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration