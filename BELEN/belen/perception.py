"""  
|---|---|
| `perception.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Let it. |
"""
-----


from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import CastedLinear, RMSNorm


class PerceptionAgent(nn.Module):

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, dim, bias=False)
        self.c_v = CastedLinear(dim, dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.query_norm = RMSNorm()
        self.key_norm = RMSNorm()
        self.percept_gate = nn.Parameter(torch.full((dim,), -1.5, dtype=torch.float32))
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, dendritic_out: Tensor, resonance_slots: Tensor) -> Tensor:
        B, T, D = dendritic_out.shape
        K = resonance_slots.size(0)
        q = self.c_q(self.query_norm(dendritic_out))
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        slots_norm = self.key_norm(resonance_slots.unsqueeze(0)).squeeze(0)
        k_proj = self.c_k(slots_norm)
        v_proj = self.c_v(slots_norm)
        k = k_proj.reshape(K, self.n_heads, self.head_dim).transpose(0, 1)
        v = v_proj.reshape(K, self.n_heads, self.head_dim).transpose(0, 1)
        k = F.rms_norm(k, (k.size(-1),))
        k = k.unsqueeze(0).expand(B, -1, -1, -1)
        v = v.unsqueeze(0).expand(B, -1, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        out = self.proj(out)
        gate = torch.sigmoid(self.percept_gate.to(dendritic_out.dtype))
        return out * gate.unsqueeze(0).unsqueeze(0)
