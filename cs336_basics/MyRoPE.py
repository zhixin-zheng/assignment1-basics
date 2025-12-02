import torch
import torch.nn as nn
from einops import einsum

class MyRoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(MyRoPE, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.inv_freq = 1.0 / self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k) # \frac{1}{\theta^{(2k - 2)/d}}
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = einsum(t, self.inv_freq, 'i,j -> i j') # (max_seq_len, dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        current_cos, current_sin = self.cos_cached[token_positions], self.sin_cached[token_positions]

        if x.dim() == 4:
            current_cos = current_cos.unsqueeze(1)
            current_sin = current_sin.unsqueeze(1)

        x_odd  = x[...,1::2]
        x_even = x[..., ::2]

        rot_even = x_even * current_cos - x_odd * current_sin
        rot_odd  = x_even * current_sin + x_odd * current_cos

        rotated_x = torch.empty_like(x)
        rotated_x[..., ::2] = rot_even
        rotated_x[...,1::2] = rot_odd
        return rotated_x

class MyRoPE_Complex(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j -> i j', t, inv_freq) # (Seq, Dim/2)
        
        # [优化点 1]: 直接构建复数形式的旋转因子 (Polar Form)
        # e^{i\theta} = cos(theta) + i*sin(theta)
        # shape: (Seq, Dim/2) -> complex64
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        
        # 注册 Buffer (直接存复数)
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq, Head, Dim)
        
        # [优化点 2]: 将输入 x 变形为复数视图
        # (..., Dim) -> (..., Dim/2, 2) -> (..., Dim/2) complex
        # 这步是 Zero-Copy (零拷贝) 的，非常快
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        
        # 取出对应的旋转因子
        # shape: (Batch, Seq, Dim/2)
        current_freqs = self.freqs_complex[token_positions]
        
        # 处理 Head 维度的广播 (Batch, Seq, 1, Dim/2)
        if x.dim() == 4:
            current_freqs = current_freqs.unsqueeze(2)
            
        # [优化点 3]: 核心计算 -> 一次复数乘法
        # 这一行替代了之前所有的 slice, mul, sub, add, empty_like, assign
        x_rotated_complex = x_complex * current_freqs
        
        # [优化点 4]: 变回实数并展平
        # (..., Dim/2) complex -> (..., Dim/2, 2) real -> (..., Dim)
        x_out = torch.view_as_real(x_rotated_complex).flatten(-2)
        
        return x_out
