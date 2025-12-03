import torch
import torch.nn as nn
from einops import einsum
    
class MySwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dtype=None, device=None):
        super(MySwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff else round(self.d_model * 8 / 3)
        self.weights1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, dtype=dtype, device=device))
        self.weights2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, dtype=dtype, device=device))
        self.weights3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, dtype=dtype, device=device))

        nn.init.xavier_uniform_(self.weights1)
        nn.init.xavier_uniform_(self.weights2)
        nn.init.xavier_uniform_(self.weights3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = einsum(x, self.weights1, '... d_model, d_ff d_model -> ... d_ff')
        silu = gate * torch.sigmoid(gate)
        value = einsum(x, self.weights3, '... d_model, d_ff d_model -> ... d_ff')
        x = silu * value
        return einsum(x, self.weights2, '... d_ff, d_model d_ff -> ... d_model')


