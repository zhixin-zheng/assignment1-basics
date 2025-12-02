import torch
import torch.nn as nn
from cs336_basics.MyAttention import Multihead_Self_Attention
from cs336_basics.MySwiGLU import MySwiGLU
from cs336_basics.MyRMSNorm import MyRMSNorm


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len):
        super(Transformer, self).__init__()
        
        self.ln1 = MyRMSNorm(d_model)
        self.attn = Multihead_Self_Attention(d_model, num_heads, max_seq_len)
        self.ln2 = MyRMSNorm(d_model)
        self.ffn = MySwiGLU(d_model, d_ff)

    def load_weights(self, weights):
        self.attn.Wq.data      = weights['attn.q_proj.weight']
        self.attn.Wk.data      = weights['attn.k_proj.weight']
        self.attn.Wv.data      = weights['attn.v_proj.weight']
        self.attn.Wo.data      = weights['attn.output_proj.weight']
        self.ln1.gain.data     = weights['ln1.weight']
        self.ffn.weights1.data = weights['ffn.w1.weight']
        self.ffn.weights2.data = weights['ffn.w2.weight']
        self.ffn.weights3.data = weights['ffn.w3.weight']
        self.ln2.gain.data     = weights['ln2.weight']

    def forward(self, x: torch.Tensor, token_positions = None, theta: float = 10000.0) -> torch.Tensor:

        x = x + self.attn(self.ln1(x), token_positions, theta)

        x = x + self.ffn(self.ln2(x))
        return x
