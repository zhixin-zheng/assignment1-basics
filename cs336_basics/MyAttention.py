import torch
import torch.nn as nn
from einops import einsum, rearrange
from cs336_basics.MyRoPE import MyRoPE
from cs336_basics.utils import softmax
# from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def scaled_dot_product_attention(q, k, v, mask=None):
    dim = k.shape[-1]
    scale = dim ** 0.5

    attention_scores = einsum(q, k, '... i d, ... j d -> ... i j') / scale

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = softmax(attention_scores, dim=-1)

    return attention_weights @ v

class Multihead_Self_Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dtype=None, device=None):
        super(Multihead_Self_Attention, self).__init__()
        assert d_model % num_heads == 0, 'd_model should be divided by number of heads.'

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_head = int(d_model / num_heads)

        self.Wq = nn.Parameter(torch.empty(d_model, d_model, dtype=dtype, device=device))
        self.Wk = nn.Parameter(torch.empty(d_model, d_model, dtype=dtype, device=device))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model, dtype=dtype, device=device))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model, dtype=dtype, device=device))

        # nn.init.xavier_uniform_(self.Wq)
        # nn.init.xavier_uniform_(self.Wk)
        # nn.init.xavier_uniform_(self.Wv)
        # nn.init.xavier_uniform_(self.Wo)

    def forward(self, x: torch.Tensor, token_positions=None, theta: float=10000.0) -> torch.Tensor:
        B, N, _ = x.shape

        q = einsum(x, self.Wq, '... i, j i -> ... j')
        k = einsum(x, self.Wk, '... i, j i -> ... j')
        v = einsum(x, self.Wv, '... i, j i -> ... j')
        
        q = rearrange(q, 'batch seq_len (num_heads d_head) -> batch num_heads seq_len d_head', num_heads=self.num_heads, d_head=self.d_head)
        k = rearrange(k, 'batch seq_len (num_heads d_head) -> batch num_heads seq_len d_head', num_heads=self.num_heads, d_head=self.d_head)
        v = rearrange(v, 'batch seq_len (num_heads d_head) -> batch num_heads seq_len d_head', num_heads=self.num_heads, d_head=self.d_head)

        casual_mask = torch.tril(torch.ones(N, N, device=x.device)).bool()
        attention_mask = casual_mask[None, None,...].expand(B, self.num_heads, N, N)

        if token_positions is None:
            token_positions = torch.arange(0, N, device=x.device)[None,:]
        
        if token_positions.dim() == 1:
            token_positions = token_positions[None,:]
        
        rope = MyRoPE(theta, self.d_head, max(N, self.max_seq_len))
        q = rope.forward(q, token_positions)
        k = rope.forward(k, token_positions)

        attention_output = scaled_dot_product_attention(q, k, v, attention_mask)

        attention_output = rearrange(attention_output, 'batch num_heads seq_len d_head -> batch seq_len (num_heads d_head)')

        o = einsum(attention_output, self.Wo, '... i, j i -> ... j')

        return o
    
    def forward_without_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        q = einsum(x, self.Wq, '... i, j i -> ... j')
        k = einsum(x, self.Wk, '... i, j i -> ... j')
        v = einsum(x, self.Wv, '... i, j i -> ... j')
        
        q = rearrange(q, 'batch seq_len (num_heads d_head) -> batch num_heads seq_len d_head', num_heads=self.num_heads, d_head=self.d_head)
        k = rearrange(k, 'batch seq_len (num_heads d_head) -> batch num_heads seq_len d_head', num_heads=self.num_heads, d_head=self.d_head)
        v = rearrange(v, 'batch seq_len (num_heads d_head) -> batch num_heads seq_len d_head', num_heads=self.num_heads, d_head=self.d_head)

        casual_mask = torch.tril(torch.ones(N, N, device=x.device)).bool()
        attention_mask = casual_mask[None, None,...].expand(B, self.num_heads, N, N)

        attention_output = scaled_dot_product_attention(q, k, v, attention_mask)

        attention_output = rearrange(attention_output, 'batch num_heads seq_len d_head -> batch seq_len (num_heads d_head)')

        o = einsum(attention_output, self.Wo, '... i, j i -> ... j')

        return o

