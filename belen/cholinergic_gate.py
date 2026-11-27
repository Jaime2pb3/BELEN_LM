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
        auto: bool | None = None,
        auto_eps: float = 0.02,
        auto_patience: int = 200,
        auto_max: int = 4,
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
        self.register_buffer("last_rho_std", torch.tensor(0.0), persistent=False)
        self.static_rho: float | None = None
        # Compuerta automática (ACH_AUTO=1). Apagada por defecto: sin ella el state_dict y el
        # forward son idénticos al modelo auditado.
        if auto is None:
            auto = os.environ.get("ACH_AUTO", "0") == "1"
        self.auto = bool(auto)
        if self.auto:
            self.auto_eps = float(auto_eps)
            self.auto_patience = int(auto_patience)
            self.auto_max = int(auto_max)
            # Almacén: escala por canal sin límites (en log). Guarda lo que la compuerta ya aprendió
            # cuando sus parámetros llegan al borde de su caja, y sigue en el camino del gradiente.
            self.store_log = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.register_buffer("rho_bar", torch.tensor(0.0))
            self.register_buffer("rho_ref", torch.tensor(0.0))
            self.register_buffer("n_consolidations", torch.tensor(0))
            self.register_buffer("sat_count", torch.tensor(0))
            self.register_buffer("last_jump", torch.tensor(0.0))
            self.register_buffer("forwards_seen", torch.tensor(0))
            self.register_buffer("consolidated_at", torch.full((self.auto_max,), -1, dtype=torch.long))

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

    # ── compuerta automática ─────────────────────────────────────────────────
    @torch.no_grad()
    def saturation(self) -> dict[str, float]:
        """Distancia relativa de cada parámetro acotado al borde que lo apaga (0 = en el borde)."""
        k = float(self.rho_k_eff)
        th = float(self.theta_eff)
        c = float(self._base_conductance(torch.float32).mean())
        lo = 1.0 - self.base_scale
        return {
            "k": (k - self.K_MIN) / (self.K_MAX - self.K_MIN),
            "theta": (self.THETA_MAX - th) / (self.THETA_MAX - self.THETA_MIN),
            "base": (c - lo) / (2.0 * self.base_scale) if self.base_scale > 0 else 1.0,
        }

    def _rho(self, z: Tensor, dtype) -> Tensor:
        k = self.rho_k_eff.to(dtype)
        th = self.theta_eff.to(dtype)
        return torch.sigmoid(k * (z.to(dtype) - th))

    @staticmethod
    @torch.no_grad()
    def _release_latent(p: Tensor, frac: float = 0.95) -> None:
        lim = float(torch.logit(torch.tensor(frac)))
        p.clamp_(-lim, lim)

    @torch.no_grad()
    def _consolidate(self, z: Tensor) -> None:
        """Pasa al almacén la parte constante de la máscara en su punto de operación y libera la
        dinámica. Antes: m = e^store c (1 - λ(ρ - ρ_ref)s). Después: e^store' (1 - λ(ρ' - ρ_ref')s),
        con e^store' = e^store c (1 - λ(ρ̄ - ρ_ref)s), c := 1, y k, θ sacados del borde."""
        dt = torch.float32
        supp = torch.sigmoid(self.w_supp.float())
        c = self._base_conductance(dt)
        rho_old = self._rho(z, dt)
        before = torch.exp(self.store_log) * c * (1.0 - self.lam * (rho_old.unsqueeze(-1) - self.rho_ref) * supp)
        fold = c * (1.0 - self.lam * (rho_old.mean() - self.rho_ref) * supp)
        self.store_log.add_(torch.log(fold.clamp_min(1e-06)))
        self.conductance_base.zero_()
        self._release_latent(self.rho_k)
        self._release_latent(self.theta)
        rho_new = self._rho(z, dt)
        self.rho_ref.copy_(rho_new.mean())
        after = torch.exp(self.store_log) * (1.0 - self.lam * (rho_new.unsqueeze(-1) - self.rho_ref) * supp)
        self.last_jump.copy_((after - before).abs().max())
        self.consolidated_at[int(self.n_consolidations)] = int(self.forwards_seen)
        self.n_consolidations.add_(1)
        self.sat_count.zero_()

    @torch.no_grad()
    def _auto_step(self, z: Tensor) -> None:
        self.forwards_seen.add_(1)
        if int(self.n_consolidations) >= self.auto_max:
            return
        if min(self.saturation().values()) < self.auto_eps:
            self.sat_count.add_(1)
        else:
            self.sat_count.zero_()
        if int(self.sat_count) >= self.auto_patience:
            self._consolidate(z)

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
            if self.auto and self.training:
                self._auto_step(z)
        if self.auto:
            # Fuera de no_grad: la consolidación pudo cambiar conductance_base y la máscara
            # tiene que seguir llevando su gradiente.
            c_base = self._base_conductance(h.dtype)
        rho = self._rho(z, h.dtype).unsqueeze(-1)
        if self.static_rho is not None:
            rho = torch.full_like(rho, self.static_rho)
        supp = torch.sigmoid(self.w_supp.to(h.dtype))
        if self.auto:
            rho_eff = rho - self.rho_ref.to(h.dtype)
        else:
            rho_eff = rho
        mask = c_base.view(1, 1, -1) * (1.0 - self.lam * rho_eff * supp.view(1, 1, -1))
        mask = mask.clamp(min=0.05, max=1.5)
        if self.auto:
            mask = mask * torch.exp(self.store_log.to(h.dtype)).view(1, 1, -1)
        with torch.no_grad():
            self.last_u_mean.copy_(u.mean())
            self.last_u_std.copy_(u.std(unbiased=False))
            self.last_z_mean.copy_(z.mean())
            self.last_rho.copy_(rho.float().mean())
            self.last_rho_std.copy_(rho.float().std(unbiased=False))
            if self.auto and self.training:
                m = self.ema_momentum
                self.rho_bar.mul_(m).add_((1.0 - m) * rho.float().mean())
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
            "rho_std": float(self.last_rho_std.item()),
            **(
                {
                    "auto": 1.0,
                    "n_consolidations": float(self.n_consolidations.item()),
                    "store_mean": float(torch.exp(self.store_log.float()).mean().item()),
                    "store_min": float(torch.exp(self.store_log.float()).min().item()),
                    "rho_ref": float(self.rho_ref.item()),
                    "last_jump": float(self.last_jump.item()),
                    **{f"sat_{k}": v for k, v in self.saturation().items()},
                }
                if self.auto
                else {}
            ),
        }
