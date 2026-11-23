"""  
|---|---|
| `dendrite.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Escalate it. |
-----
"""


from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import CastedLinear, Rotary, apply_rope


class PlasticDendriticField(nn.Module):

    def __init__(self, dim: int, n_proto: int):
        super().__init__()
        self.dim = dim
        self.n_proto = n_proto
        self.prototypes = nn.Parameter(torch.randn(n_proto, dim) * 0.02)
        self.log_beta = nn.Parameter(torch.tensor(math.log(math.sqrt(dim)), dtype=torch.float32))

    def forward(self, h: Tensor) -> Tensor:
        B, T, D = h.shape
        dtype = h.dtype
        h_norm = F.normalize(h.float(), dim=-1)
        P = F.normalize(self.prototypes.float(), dim=-1)
        beta = self.log_beta.float().exp()
        sim = h_norm @ P.T
        weights = torch.softmax(beta * sim, dim=-1)
        recall = weights @ self.prototypes.float()
        alpha = sim.max(dim=-1, keepdim=True).values.clamp(0.0, 1.0)
        return ((1.0 - alpha) * h.float() + alpha * recall).to(dtype)


class DendriticEncoder(nn.Module):

    def __init__(
        self,
        dim: int,
        field_heads: list[int],
        local_span: int,
        mid_span: int,
        n_kv_heads: int,
        rope_base: float,
        n_proto: int = 16,
        layout: str = "overlap",
        geometry: nn.Module | None = None,
    ):
        super().__init__()
        assert layout in ("concat", "overlap")
        assert len(field_heads) == 3 and all(
            (h % n_kv_heads == 0 for h in field_heads)
        ), f"cada campo dendrítico debe ser múltiplo de las cabezas KV: {field_heads} vs {n_kv_heads}"
        assert (
            dim % sum(field_heads) == 0 and dim // sum(field_heads) % 2 == 0
        ), f"MODEL_DIM={dim} debe dividirse entre {sum(field_heads)} cabezas dendríticas con head_dim par (RoPE)"
        self.layout = layout
        self.dim = dim
        self.field_heads = field_heads
        self.n_heads = sum(field_heads)
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // self.n_heads
        self.local_span = local_span
        self.mid_span = mid_span
        kv_dim = n_kv_heads * self.head_dim
        self.q_local = CastedLinear(dim, field_heads[0] * self.head_dim, bias=False)
        self.q_mid = CastedLinear(dim, field_heads[1] * self.head_dim, bias=False)
        self.q_global = CastedLinear(dim, field_heads[2] * self.head_dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.plastic_local = PlasticDendriticField(dim, n_proto)
        self.plastic_mid = PlasticDendriticField(dim, n_proto)
        self.plastic_global = PlasticDendriticField(dim, n_proto)
        self.field_mix = nn.Parameter(torch.ones(3, dtype=torch.float32) / 3.0)
        self.geometry = geometry
        self._mask_cache: dict = {}

    def _make_span_mask(self, T: int, span: int, device, dtype) -> Tensor:
        key = (T, span, str(device), str(dtype))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        causal = torch.ones(T, T, device=device, dtype=torch.bool).tril()
        if span < T:
            idx = torch.arange(T, device=device)
            dist_mask = idx.unsqueeze(0) - idx.unsqueeze(1) >= -span
            causal = causal & dist_mask
        mask = torch.zeros(T, T, device=device, dtype=dtype)
        mask.masked_fill_(~causal, float("-inf"))
        mask = mask.unsqueeze(0).unsqueeze(0)
        self._mask_cache[key] = mask
        return mask

    def forward(self, x: Tensor) -> Tensor:
        if self.geometry is not None:
            return self._forward_geometric(x)
        B, T, D = x.shape
        k = self.c_k(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, k.dtype)
        k = apply_rope(k, cos, sin)
        q_projs = [self.q_local, self.q_mid, self.q_global]
        plastics = [self.plastic_local, self.plastic_mid, self.plastic_global]
        spans = [self.local_span, self.mid_span, T]
        field_outputs = []
        offset = 0
        for q_proj, plastic, span, nh in zip(q_projs, plastics, spans, self.field_heads):
            q = q_proj(x).reshape(B, T, nh, self.head_dim).transpose(1, 2)
            q = F.rms_norm(q, (q.size(-1),))
            q = apply_rope(q, cos, sin)
            k_exp = k
            v_exp = v
            if span < T:
                attn_mask = self._make_span_mask(T, span, x.device, q.dtype)
                y = F.scaled_dot_product_attention(
                    q, k_exp, v_exp, attn_mask=attn_mask, is_causal=False, enable_gqa=nh != self.n_kv_heads
                )
            else:
                y = F.scaled_dot_product_attention(
                    q, k_exp, v_exp, attn_mask=None, is_causal=True, enable_gqa=nh != self.n_kv_heads
                )
            field_h = y.transpose(1, 2).contiguous().reshape(B, T, nh * self.head_dim)
            width = field_h.size(-1)
            if width < D:
                left = offset if self.layout == "concat" else 0
                field_h_full = F.pad(field_h, (left, D - width - left))
            else:
                field_h_full = field_h
            offset += width
            field_h_plastic = plastic(field_h_full)
            field_outputs.append(field_h_plastic)
        mix = F.softmax(self.field_mix.to(x.dtype), dim=0)
        mixed = sum((mix[i] * field_outputs[i] for i in range(3)))
        return self.proj(mixed)

    def _place(self, field_h: Tensor, offset: int, D: int) -> Tensor:
        width = field_h.size(-1)
        if width >= D:
            return field_h
        left = offset if self.layout == "concat" else 0
        return F.pad(field_h, (left, D - width - left))

    def _forward_geometric(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        geo = self.geometry
        k_base = self.c_k(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_base = self.c_v(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k_base = F.rms_norm(k_base, (k_base.size(-1),))
        cos, sin = self.rotary(T, x.device, k_base.dtype)
        gamma_prior = geo.gamma_prior(x)
        q_projs = [self.q_local, self.q_mid, self.q_global]
        plastics = [self.plastic_local, self.plastic_mid, self.plastic_global]
        spans = [self.local_span, self.mid_span, T]
        field_outputs = []
        offset = 0
        for idx, (q_proj, plastic, span, nh) in enumerate(zip(q_projs, plastics, spans, self.field_heads)):
            q = q_proj(x).reshape(B, T, nh, self.head_dim).transpose(1, 2)
            q = F.rms_norm(q, (q.size(-1),))
            g = gamma_prior[:, :, idx]
            q = geo.metric_transform(q, idx, g, values=False)
            k = geo.metric_transform(k_base, idx, g, values=False)
            v = geo.metric_transform(v_base, idx, g, values=True)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            if span < T:
                attn_mask = self._make_span_mask(T, span, x.device, q.dtype)
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=nh != self.n_kv_heads
                )
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=True, enable_gqa=nh != self.n_kv_heads
                )
            field_h = y.transpose(1, 2).contiguous().reshape(B, T, nh * self.head_dim)
            field_outputs.append(plastic(self._place(field_h, offset, D)))
            offset += field_h.size(-1)
        projections, residuals = [], []
        for i, h_field in enumerate(field_outputs):
            p, r = geo.project(h_field, i)
            projections.append(p)
            residuals.append(r)
        gamma, disagreement = geo.posterior_gamma(gamma_prior, field_outputs, projections, residuals)
        transported, taus = [], []
        for i, (h_field, projected) in enumerate(zip(field_outputs, projections)):
            out, tau = geo.geodesic_transport(h_field, projected, i, gamma[:, :, i])
            transported.append(out)
            taus.append(tau)
        geo._last_gamma = gamma.detach()
        geo._last_tau = torch.stack(taus, dim=-1).detach()
        geo._last_disagreement = disagreement.detach()
        geo._last_projection_residual = torch.stack(residuals, dim=-1).detach()
        geo.update_centers(field_outputs)
        mix = F.softmax(self.field_mix.to(x.dtype), dim=0)
        mixed = sum((mix[i] * transported[i] for i in range(3)))
        return self.proj(mixed)

    def geometry_aux_loss(self) -> Tensor:
        if self.geometry is None:
            return self.field_mix.new_zeros(())
        return self.geometry.regularization_loss()

    @torch.no_grad()
    def geometry_snapshot(self) -> dict:
        return self.geometry.snapshot() if self.geometry is not None else {"enabled": False}

    @torch.no_grad()
    def calibrate_geometry(self, neutral: Tensor, left: Tensor, right: Tensor) -> dict:
        if self.geometry is None:
            raise RuntimeError("GEOMETRIC_DENDRITE=0: there is no geometry to calibrate")
        return self.geometry.calibrate(neutral, left, right)
