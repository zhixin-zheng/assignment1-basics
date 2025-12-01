import torch
import torch.nn as nn

class MyRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(MyRMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(self.d_model, dtype=dtype, device=device))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)+ self.eps)
        x = (x / rms) * self.gain

        return x.to(in_dtype)