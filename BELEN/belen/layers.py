"""  
|---|---|
| `layers.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Break it. |
"""
-----

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def autocast_for(device) -> torch.autocast:
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


class RMSNorm(nn.Module):

    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return (self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype))


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    h = x.size(-1) // 2
    x1, x2 = (x[..., :h], x[..., h:])
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


class GQAttention(nn.Module):

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, rope_base: float, qk_gain_init: float = 1.5):
        super().__init__()
        assert dim % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        kv_dim = n_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.q_gain = nn.Parameter(torch.full((n_heads,), qk_gain_init, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, D))


class MLP(nn.Module):

    def __init__(self, dim: int, mult: int):
        super().__init__()
        self.fc = CastedLinear(dim, mult * dim, bias=False)
        self.proj = CastedLinear(mult * dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())
