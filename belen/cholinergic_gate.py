"""  
|---|---|
| `cholinergic_gate.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Invent it. |
-----
"""


from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def cross_layer_incoherence(trace: Tensor) -> Tensor:
    if trace.ndim != 4:
        raise ValueError(f"trace debe ser (B, T, L, D); recibido {tuple(trace.shape)}")
    B, T, L, _ = trace.shape
    if L < 3:
        return trace.new_full((B, T), 0.5, dtype=torch.float32)
    deltas = trace.float()[:, :, 1:, :] - trace.float()[:, :, :-1, :]
    norms = deltas.norm(dim=-1)
    directions = F.normalize(deltas, dim=-1, eps=1e-06)
    cos_seq = (directions[:, :, 1:, :] * directions[:, :, :-1, :]).sum(dim=-1)
    valid = (norms[:, :, 1:] > 1e-06) & (norms[:, :, :-1] > 1e-06)
    weights = valid.to(cos_seq.dtype)
    denom = weights.sum(dim=-1)
    coh_weighted = (cos_seq * weights).sum(dim=-1) / denom.clamp_min(1.0)
    coh = torch.where(denom > 0, coh_weighted, torch.zeros_like(coh_weighted))
    return ((1.0 - coh) * 0.5).clamp(0.0, 1.0)


class CholinergicGate(nn.Module):
    K_MIN, K_MAX = (0.5, 8.0)
    THETA_MIN, THETA_MAX = (-3.0, 3.0)

    def __init__(
        self,
        dim: int,
        lam: float = 0.12,
        theta_init: float = 0.0,
        rho_k_init: float = 2.0,
        ach_enabled: bool = True,
        base_scale: float = 0.1,
        freeze_conductance: bool | None = None,
        ema_momentum: float = 0.99,
    ):
        super().__init__()
        if not 0.0 <= lam < 1.0:
            raise ValueError(f"lam debe estar en [0,1), recibido {lam}")
        if not 0.0 <= base_scale < 1.0:
            raise ValueError(f"base_scale debe estar en [0,1), recibido {base_scale}")
        if not self.K_MIN <= rho_k_init <= self.K_MAX:
            raise ValueError(f"rho_k_init fuera de [{self.K_MIN},{self.K_MAX}]")
        if not self.THETA_MIN < theta_init < self.THETA_MAX:
            raise ValueError(f"theta_init fuera de ({self.THETA_MIN},{self.THETA_MAX})")
        self.dim = dim
        self.lam = float(lam)
        self.base_scale = float(base_scale)
        self.ach_enabled = ach_enabled
        self.ema_momentum = float(ema_momentum)
        if freeze_conductance is None:
            freeze_conductance = os.environ.get("ACH_FREEZE_BASE", "0") == "1"
        self.freeze_conductance = bool(freeze_conductance)
        self.conductance_base = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.conductance_base.requires_grad_(not self.freeze_conductance)
        self.w_supp = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.theta = nn.Parameter(
            torch.tensor(self._inv_affine_sigmoid(theta_init, self.THETA_MIN, self.THETA_MAX))
        )
        self.rho_k = nn.Parameter(torch.tensor(self._inv_affine_sigmoid(rho_k_init, self.K_MIN, self.K_MAX)))
        self.register_buffer("u_mu", torch.tensor(0.0))
        self.register_buffer("u_var", torch.tensor(0.0))
        self.register_buffer("u_seen", torch.tensor(0.0))
        for name in (
            "last_u_mean",
            "last_u_std",
            "last_z_mean",
            "last_rho",
            "last_mask_mean",
            "last_mask_min",
            "last_mask_max",
        ):
            self.register_buffer(name, torch.tensor(0.0), persistent=False)
        self.last_mask_mean.fill_(1.0)
        self.last_mask_min.fill_(1.0)
        self.last_mask_max.fill_(1.0)
        self.last_u_mean.fill_(-1.0)

    @staticmethod
    def _inv_affine_sigmoid(value: float, lo: float, hi: float) -> float:
        p = (float(value) - lo) / (hi - lo)
        p = min(max(p, 0.0001), 1.0 - 0.0001)
        return float(torch.logit(torch.tensor(p)).item())

    @staticmethod
    def _affine_sigmoid(raw: Tensor, lo: float, hi: float) -> Tensor:
        return lo + (hi - lo) * torch.sigmoid(raw)

    @property
    def theta_eff(self) -> Tensor:
        return self._affine_sigmoid(self.theta, self.THETA_MIN, self.THETA_MAX)

    @property
    def rho_k_eff(self) -> Tensor:
        return self._affine_sigmoid(self.rho_k, self.K_MIN, self.K_MAX)

    def _base_conductance(self, dtype: torch.dtype) -> Tensor:
        return 1.0 + self.base_scale * torch.tanh(self.conductance_base.to(dtype))

    @torch.no_grad()
    def _standardize(self, u: Tensor) -> Tensor:
        if self.training:
            m = self.ema_momentum
            self.u_seen.mul_(m).add_(1.0 - m)
            self.u_mu.mul_(m).add_((1.0 - m) * u.mean())
            self.u_var.mul_(m).add_((1.0 - m) * u.var(unbiased=False))
        if float(self.u_seen) < 0.001:
            return (u - u.mean()) / u.var(unbiased=False).clamp_min(1e-08).sqrt()
        bias = self.u_seen
        return (u - self.u_mu / bias) / (self.u_var / bias).clamp_min(1e-08).sqrt()

    def forward(self, h: Tensor, trace: Tensor | None = None, return_signal: bool = False):
        c_base = self._base_conductance(h.dtype)
        if not self.ach_enabled or trace is None:
            mask = c_base.view(1, 1, -1)
            with torch.no_grad():
                self.last_u_mean.fill_(-1.0)
                self.last_u_std.zero_()
                self.last_z_mean.zero_()
                self.last_rho.zero_()
                self.last_mask_mean.copy_(c_base.float().mean())
                self.last_mask_min.copy_(c_base.float().min())
                self.last_mask_max.copy_(c_base.float().max())
            out = h * mask
            return (out, None) if return_signal else out
        with torch.no_grad():
            u = cross_layer_incoherence(trace)
            z = self._standardize(u)
        k = self.rho_k_eff.to(h.dtype)
        th = self.theta_eff.to(h.dtype)
        rho = torch.sigmoid(k * (z.to(h.dtype) - th)).unsqueeze(-1)
        supp = torch.sigmoid(self.w_supp.to(h.dtype))
        mask = c_base.view(1, 1, -1) * (1.0 - self.lam * rho * supp.view(1, 1, -1))
        mask = mask.clamp(min=0.05, max=1.5)
        with torch.no_grad():
            self.last_u_mean.copy_(u.mean())
            self.last_u_std.copy_(u.std(unbiased=False))
            self.last_z_mean.copy_(z.mean())
            self.last_rho.copy_(rho.float().mean())
            self.last_mask_mean.copy_(mask.float().mean())
            self.last_mask_min.copy_(mask.float().min())
            self.last_mask_max.copy_(mask.float().max())
        out = h * mask
        if return_signal:
            return (out, (2.0 * rho.squeeze(-1) - 1.0).clamp(min=0.0))
        return out

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        c_base = self._base_conductance(torch.float32)
        supp = torch.sigmoid(self.w_supp.float())
        rho = float(self.last_rho.item())
        ach_authority = self.lam * rho * float(supp.mean().item())
        base_authority = abs(float(c_base.mean().item()) - 1.0)
        return {
            "u_mean": float(self.last_u_mean.item()),
            "u_std": float(self.last_u_std.item()),
            "u_mu_ema": float((self.u_mu / self.u_seen.clamp_min(1e-06)).item()),
            "z_mean": float(self.last_z_mean.item()),
            "rho_mean": rho,
            "mask_mean": float(self.last_mask_mean.item()),
            "mask_min": float(self.last_mask_min.item()),
            "mask_max": float(self.last_mask_max.item()),
            "theta": float(self.theta_eff.item()),
            "rho_k": float(self.rho_k_eff.item()),
            "theta_latent": float(self.theta.item()),
            "rho_k_latent": float(self.rho_k.item()),
            "base_mean": float(c_base.mean().item()),
            "supp_mean": float(supp.mean().item()),
            "lambda": self.lam,
            "ach_authority": ach_authority,
            "base_authority": base_authority,
            "authority_ratio": base_authority / max(ach_authority, 1e-09),
            "base_frozen": float(self.freeze_conductance),
        }
