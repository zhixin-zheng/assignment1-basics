import torch
import torch.nn as nn

from cs336_basics.MyTransformer import Transformer
from cs336_basics.MyEmbedding import MyEmbedding
from cs336_basics.MyRMSNorm import MyRMSNorm
from cs336_basics.MyLinear import Linear
from cs336_basics.MyAttention import softmax

class LM(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, theta=10000.0):
        super(LM, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.theta = theta
        self.token_embeddings = MyEmbedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [Transformer(d_model, num_heads, d_ff, context_length) for i in range(num_layers)]
        )
        self.ln_final = MyRMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def load_weights(self, weights):
        self.token_embeddings.weights.data = weights['token_embeddings.weight']
        self.lm_head.weights.data = weights['lm_head.weight']
        self.ln_final.gain.data = weights['ln_final.weight']
        for idx, transformer_block in enumerate(self.transformer_blocks):
            current_weights = {
                'attn.q_proj.weight': weights[f'layers.{idx}.attn.q_proj.weight'],
                'attn.k_proj.weight': weights[f'layers.{idx}.attn.k_proj.weight'],
                'attn.v_proj.weight': weights[f'layers.{idx}.attn.v_proj.weight'],
                'attn.output_proj.weight': weights[f'layers.{idx}.attn.output_proj.weight'],
                'ln1.weight': weights[f'layers.{idx}.ln1.weight'],
                'ln2.weight': weights[f'layers.{idx}.ln2.weight'],
                'ffn.w1.weight': weights[f'layers.{idx}.ffn.w1.weight'],
                'ffn.w2.weight': weights[f'layers.{idx}.ffn.w2.weight'],
                'ffn.w3.weight': weights[f'layers.{idx}.ffn.w3.weight']
            }
            transformer_block.load_weights(current_weights)

    def forward(self, in_indices: torch.Tensor, test_flops: bool = False) -> torch.Tensor:
        if test_flops:
            B, N = in_indices.shape
            d, d_ff = self.d_model, self.d_ff
            FLOPs = 0
            FLOPs += self.num_layers * (
                8 * B * N * d * d + 
                4 * B * N * N * d +
                3 * (2 * B * N * d * d_ff)
            ) + 2 * B * N * d * self.vocab_size
            print(f"Estimated FLOPs for a forward pass: {FLOPs / 1e12} TFLOPs")
            
        features = self.token_embeddings(in_indices)
        for transformer_block in self.transformer_blocks:
            features = transformer_block(features, theta=self.theta)
        features = self.lm_head(self.ln_final(features))

        return features