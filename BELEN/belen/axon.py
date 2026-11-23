"""  
|---|---|
| `axon.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Understand it. |
"""
-----


from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quarks import QuarkBundle, quark_views


class AxonViews(nn.Module):

    def __init__(self, dim: int, n_views: int, lam: float, momentum: float = 0.9):
        super().__init__()
        self.dim, self.n_views, self.lam, self.momentum = (dim, n_views, lam, momentum)
        self.conductance_base = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.rho_k = nn.Parameter(torch.tensor(-1.3862944))
        self.register_buffer("S_mean", torch.zeros(()))
        self.register_buffer("S_var", torch.ones(()))
        self.register_buffer("S_count", torch.zeros(()))
        self._last: tuple | None = None

    def k_value(self) -> float:
        return float(0.5 + 7.5 * torch.sigmoid(self.rho_k.detach()))

    @property
    def last_S(self) -> float:
        return float(self._last[0]) if self._last else 0.0

    @property
    def last_rho(self) -> float:
        return float(self._last[1]) if self._last else 0.0

    @property
    def last_mask_mean(self) -> float:
        return float(self._last[2]) if self._last else 1.0

    def forward(self, h: Tensor, views: list[Tensor]) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            phis = [F.layer_norm(v.float(), (self.dim,)) for v in views[-self.n_views :]]
            S, dim_div = quark_views(phis)
            first = (self.S_count == 0).to(S.dtype)
            if self.training:
                bm, bv = (S.mean(), S.var().clamp_min(1e-08))
                ref_m = first * bm + (1 - first) * self.S_mean
                ref_v = first * bv + (1 - first) * self.S_var
            else:
                ref_m, ref_v = (self.S_mean, self.S_var)
            z = (S - ref_m) / (ref_v + 1e-08).sqrt()
            if self.training:
                mom = self.momentum * (1 - first)
                self.S_mean.copy_(mom * self.S_mean + (1 - mom) * bm)
                self.S_var.copy_(mom * self.S_var + (1 - mom) * bv)
                self.S_count.add_(1.0)
            ddn = dim_div / (dim_div.amax(dim=-1, keepdim=True) + 1e-09)
        k = 0.5 + 7.5 * torch.sigmoid(self.rho_k)
        rho = torch.sigmoid(k * z).to(h.dtype)
        c_base = torch.sigmoid(self.conductance_base.to(h.dtype))
        c_stressed = (c_base * (1.0 - self.lam * ddn.to(h.dtype))).clamp(min=0.05)
        mask = c_base + rho.unsqueeze(-1) * (c_stressed - c_base)
        self._last = (S.mean(), rho.detach().mean(), mask.detach().mean())
        return (h * mask, (2.0 * rho - 1.0).clamp(min=0.0))


class AxonProjector(nn.Module):

    def __init__(
        self, dim: int, n_views: int, t_min: float, t_max: float, lam: float, stress_threshold: float
    ):
        super().__init__()
        self.dim = dim
        self.n_views = n_views
        self.t_min = t_min
        self.t_max = t_max
        self.lam = lam
        self.stress_threshold = stress_threshold
        self.conductance_base = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
        self.rho_k = nn.Parameter(torch.tensor(2.0))
        self.quarks = QuarkBundle()
        self.last_S = 0.0
        self.last_rho = 0.0
        self.last_mask_mean = 1.0

    def reset_quarks(self):
        self.quarks.reset()

    def forward(self, h: Tensor) -> Tensor:
        h_flat = h.reshape(-1, self.dim)
        h_mean = h_flat.mean(dim=0, keepdim=True)
        with torch.no_grad():
            temps = np.linspace(self.t_min, self.t_max, self.n_views)
            phi_list = []
            for t in temps:
                noise = torch.randn_like(h_mean) * t * 0.25
                r = h_mean + noise
                phi_list.append(F.layer_norm(r, r.shape[-1:]))
            S, dim_div = self.quarks.observe(phi_list)
        self.last_S = S
        k = self.rho_k.clamp(min=0.5, max=8.0)
        S_t = torch.tensor(S, dtype=h.dtype, device=h.device)
        rho = torch.sigmoid(k * (S_t - self.stress_threshold))
        self.last_rho = float(rho.item())
        cb = self.conductance_base.to(h.dtype)
        c_base = torch.sigmoid(cb)
        dim_div_t = dim_div.to(device=h.device, dtype=h.dtype)
        dim_div_norm = dim_div_t / (dim_div_t.max() + 1e-09)
        c_stressed = c_base * (1.0 - self.lam * dim_div_norm)
        c_stressed = c_stressed.clamp(min=0.05)
        mask = (1.0 - rho) * c_base + rho * c_stressed
        self.last_mask_mean = float(mask.mean().item())
        return h * mask.unsqueeze(0).unsqueeze(0)
