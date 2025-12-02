import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        input_std = math.sqrt(2 / (in_features + out_features))
        self.weights = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        torch.nn.init.trunc_normal_(self.weights, 0, input_std, a=-3*input_std, b=3*input_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, '... in_dim, out_dim in_dim -> ... out_dim')