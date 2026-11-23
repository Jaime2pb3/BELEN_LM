"""  
|---|---|
| `pfv_monitor.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Watch it. |
-----
"""


from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class IsomericPolarizationMonitor(nn.Module):

    def __init__(
        self,
        n_realizations: int,
        window: int = 50,
        agg: str = "mean",
        epsilon_quantile: float = 0.1,
        activation_ema: float = 0.95,
    ):
        super().__init__()
        assert agg in ("mean", "trimmed_mean")
        assert 0.0 < epsilon_quantile < 1.0
        self.n_realizations = n_realizations
        self.window = window
        self.agg = agg
        self.epsilon_quantile = epsilon_quantile
        self.activation_ema = activation_ema
        self.register_buffer("P_history", torch.zeros(window))
        self.register_buffer("P_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("P_fill", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("dP_history", torch.zeros(window))
        self.register_buffer("activation_count", torch.zeros(n_realizations))

    @torch.no_grad()
    def compute(self, realizations: Tensor) -> dict:
        K = realizations.size(0)
        if realizations.ndim == 3:
            r = realizations.float().mean(dim=1)
        else:
            r = realizations.float()
        r_norm = F.normalize(r, dim=-1)
        sim = r_norm @ r_norm.T
        d = 1.0 - sim
        mask_upper = torch.triu(torch.ones(K, K, device=r.device, dtype=torch.bool), diagonal=1)
        d_pairs = d[mask_upper]
        if self.agg == "trimmed_mean":
            sorted_d, _ = d_pairs.sort()
            n_trim = max(1, int(len(sorted_d) * 0.1))
            P_t = sorted_d[n_trim:-n_trim].mean() if len(sorted_d) > 2 * n_trim else sorted_d.mean()
        else:
            P_t = d_pairs.mean()
        epsilon = torch.quantile(d_pairs, self.epsilon_quantile)
        iso_mask = (d <= epsilon) & mask_upper
        n_isomers = iso_mask.sum().float()
        ptr_mask = F.one_hot(self.P_ptr, num_classes=self.window).squeeze(0).float()
        prev_ptr = (self.P_ptr - 1) % self.window
        prev_mask = F.one_hot(prev_ptr, num_classes=self.window).squeeze(0).float()
        prev_P = (self.P_history * prev_mask).sum()
        prev_dP = (self.dP_history * prev_mask).sum()
        delta_P = P_t - prev_P
        delta2_P = delta_P - prev_dP
        new_P_hist = self.P_history * (1.0 - ptr_mask) + P_t * ptr_mask
        new_dP_hist = self.dP_history * (1.0 - ptr_mask) + delta_P * ptr_mask
        self.P_history.copy_(new_P_hist)
        self.dP_history.copy_(new_dP_hist)
        self.P_ptr.add_(1).remainder_(self.window)
        self.P_fill.add_(1.0)
        effective_n = torch.clamp(self.P_fill, min=1.0, max=float(self.window))
        P_regime = self.P_history.sum() / effective_n
        if self.activation_count.numel() > 0 and self.activation_count.sum() > 0:
            median = torch.median(self.activation_count)
            dead_mask = self.activation_count < 0.1 * median
        else:
            dead_mask = torch.zeros_like(self.activation_count, dtype=torch.bool)
        dead_count = dead_mask.sum().float()
        return {
            "P_t": P_t,
            "P_regime": P_regime,
            "delta_P": delta_P,
            "delta2_P": delta2_P,
            "epsilon": epsilon,
            "n_isomers": n_isomers,
            "iso_mask": iso_mask,
            "dead_mask": dead_mask,
            "dead_count": dead_count,
        }

    @torch.no_grad()
    def update_activations(self, scores: Tensor) -> None:
        assert (
            scores.shape == self.activation_count.shape
        ), f"expected {self.activation_count.shape}, got {scores.shape}"
        self.activation_count.mul_(self.activation_ema).add_(scores.float(), alpha=1.0 - self.activation_ema)

    def compute_ttt_loss(
        self, realizations_frozen: Tensor, realization_new: Tensor, beta: float = 0.1
    ) -> Tensor:
        K = realizations_frozen.size(0)
        Rf = realizations_frozen.detach().float()
        pn = realization_new.float()
        R_aug = torch.cat([Rf, pn.unsqueeze(0)], dim=0)
        R_norm = F.normalize(R_aug, dim=-1)
        sim = R_norm @ R_norm.T
        d = 1.0 - sim
        mask_upper = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
        d_pairs = d[mask_upper]
        if self.agg == "trimmed_mean":
            sorted_d, _ = d_pairs.sort()
            n_trim = max(1, int(len(sorted_d) * 0.1))
            if len(sorted_d) > 2 * n_trim:
                P_t_aug = sorted_d[n_trim:-n_trim].mean()
            else:
                P_t_aug = sorted_d.mean()
        else:
            P_t_aug = d_pairs.mean()
        effective_n = torch.clamp(self.P_fill, min=1.0, max=float(self.window))
        P_regime = (self.P_history.sum() / effective_n).detach()
        loss_regime = (P_t_aug - P_regime).abs()
        prev_ptr = (self.P_ptr - 1) % self.window
        prev2_ptr = (self.P_ptr - 2) % self.window
        m1 = F.one_hot(prev_ptr, num_classes=self.window).squeeze(0).float()
        m2 = F.one_hot(prev2_ptr, num_classes=self.window).squeeze(0).float()
        P_prev = (self.P_history * m1).sum().detach()
        P_prev2 = (self.P_history * m2).sum().detach()
        fill_frac = torch.clamp(self.P_fill / 2.0, max=1.0).detach()
        delta2_P = P_t_aug - 2.0 * P_prev + P_prev2
        loss_accel = fill_frac * delta2_P.abs()
        return loss_regime + beta * loss_accel
