"""  
|---|---|
| `soma.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Love it. |
"""
-----

from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers import GQAttention, MLP, RMSNorm


class SomaBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float = 1.5,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = GQAttention(dim, n_heads, n_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(x.dtype) * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype) * self.mlp(self.mlp_norm(x))
        return x
