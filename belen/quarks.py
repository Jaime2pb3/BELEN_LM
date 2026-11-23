"""  
|---|---|
| `quarks.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Show it. |
-----
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


class DownQuark:

    def compute(self, phi_list: list[Tensor]) -> tuple[float, Tensor]:
        m = len(phi_list)
        if m < 2:
            return (0.0, torch.zeros(phi_list[0].size(-1)))
        diffs = []
        scalar_dists = []
        for i in range(m):
            for j in range(i + 1, m):
                diff = (phi_list[i] - phi_list[j]).abs()
                diffs.append(diff)
                cos = F.cosine_similarity(phi_list[i], phi_list[j], dim=-1).mean()
                scalar_dists.append(float(1.0 - cos.item()))
        dim_div = torch.stack(diffs).mean(0).squeeze(0)
        return (float(np.mean(scalar_dists)), dim_div)


class StrangeQuark:

    def compute(self, phi_list: list[Tensor]) -> float:
        m = len(phi_list)
        if m < 2:
            return 0.0
        stds = []
        for i in range(m):
            for j in range(i + 1, m):
                a, b = (phi_list[i], phi_list[j])
                d1 = float(1.0 - F.cosine_similarity(a, b, dim=-1).mean().item())
                d2 = float((a - b).norm(dim=-1).mean().item() / (a.norm(dim=-1).mean().item() + 1e-09))
                va = float(a.var(dim=-1).mean().item())
                vb = float(b.var(dim=-1).mean().item())
                d3 = abs(va - vb) / (max(va, vb) + 1e-09)
                stds.append(float(np.std([d1, d2, d3])))
        return float(np.mean(stds)) if stds else 0.0


class UpQuark:

    def compute(self, phi_list: list[Tensor]) -> float:
        m = len(phi_list)
        if m < 4:
            return 0.0
        dists = []
        for i in range(m):
            for j in range(i + 1, m):
                cos = F.cosine_similarity(phi_list[i], phi_list[j], dim=-1).mean()
                dists.append(float(1.0 - cos.item()))
        if len(dists) < 3:
            return 0.0
        arr = np.array(dists)
        n = len(arr)
        mu = arr.mean()
        if arr.std() < 1e-09:
            return 0.0
        skew = float(np.mean((arr - mu) ** 3) / (arr.std() ** 3 + 1e-09))
        kurt = float(np.mean((arr - mu) ** 4) / (arr.std() ** 4 + 1e-09))
        bc = (skew**2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3) + 1e-09))
        return float(bc)


class CharmQuark:

    def compute(self, phi_list: list[Tensor]) -> float:
        m = len(phi_list)
        if m < 2:
            return 0.0
        sims = []
        for i in range(m):
            for j in range(i + 1, m):
                cos = float(F.cosine_similarity(phi_list[i], phi_list[j], dim=-1).mean().item())
                sims.append(cos)
        if not sims:
            return 0.0
        arr = np.array(sims)
        hi = arr[arr > arr.mean()]
        lo = arr[arr <= arr.mean()]
        if len(hi) == 0 or len(lo) == 0:
            return 0.0
        return float(abs(hi.mean() - lo.mean()))


class TopQuark:

    def compute(self, phi_list: list[Tensor]) -> float:
        m = len(phi_list)
        if m < 2:
            return 0.0
        try:
            stacked = torch.stack([p.squeeze(0).float() for p in phi_list])
            xc = stacked - stacked.mean(dim=0, keepdim=True)
            eigs = torch.linalg.eigvalsh(xc @ xc.T / (stacked.size(0) - 1))
            eigs = eigs.abs()
            total = eigs.sum().item()
            if total < 1e-09:
                return 0.0
            return float(eigs.max().item() / total)
        except Exception:
            return 0.0


class BottomQuark:

    def compute(self, phi_list: list[Tensor]) -> float:
        m = len(phi_list)
        if m < 2:
            return 0.0
        sims = []
        for i in range(m):
            for j in range(i + 1, m):
                cos = float(F.cosine_similarity(phi_list[i], phi_list[j], dim=-1).mean().item())
                sims.append(abs(cos))
        return float(np.mean(sims)) if sims else 0.0


class QuarkBundle:

    def __init__(self):
        self.down = DownQuark()
        self.strange = StrangeQuark()
        self.up = UpQuark()
        self.charm = CharmQuark()
        self.top = TopQuark()
        self.bottom = BottomQuark()
        self._history: list[float] = []

    def reset(self):
        self._history.clear()

    def observe(self, phi_list: list[Tensor]) -> tuple[float, Tensor]:
        s_down, dim_div = self.down.compute(phi_list)
        s_strange = self.strange.compute(phi_list)
        s_up = self.up.compute(phi_list)
        s_charm = self.charm.compute(phi_list)
        s_top = self.top.compute(phi_list)
        s_bottom = self.bottom.compute(phi_list)
        S = float(np.mean([s_down, s_strange, s_up, s_charm, s_top, s_bottom]))
        self._history.append(S)
        return (S, dim_div)


@torch.no_grad()
def quark_views(phis: list[Tensor]) -> tuple[Tensor, Tensor]:
    m = len(phis)
    P = torch.stack([p.float() for p in phis], dim=-2)
    lead, D = (P.shape[:-2], P.shape[-1])
    if m < 2:
        return (P.new_zeros(lead), P.new_zeros(*lead, D))
    nrm = P.norm(dim=-1)
    var = P.var(dim=-1)
    cos, d2, d3 = ([], [], [])
    dim_div = P.new_zeros(*lead, D)
    for i in range(m):
        for j in range(i + 1, m):
            diff = P[..., i, :] - P[..., j, :]
            dim_div += diff.abs()
            cos.append(
                (P[..., i, :] * P[..., j, :]).sum(-1)
                / (nrm[..., i].clamp_min(1e-08) * nrm[..., j].clamp_min(1e-08))
            )
            d2.append(diff.norm(dim=-1) / (nrm[..., i] + 1e-09))
            va, vb = (var[..., i], var[..., j])
            d3.append((va - vb).abs() / (torch.maximum(va, vb) + 1e-09))
    n = len(cos)
    dim_div /= n
    cos, d2, d3 = (torch.stack(cos, -1), torch.stack(d2, -1), torch.stack(d3, -1))
    dist = 1.0 - cos
    s_down = dist.mean(-1)
    s_strange = torch.stack([dist, d2, d3], -1).std(dim=-1, correction=0).mean(-1)
    s_bottom = cos.abs().mean(-1)
    if m >= 4 and n >= 3:
        mu = dist.mean(-1, keepdim=True)
        sd = dist.std(dim=-1, correction=0, keepdim=True)
        skew = ((dist - mu) ** 3).mean(-1, keepdim=True) / (sd**3 + 1e-09)
        kurt = ((dist - mu) ** 4).mean(-1, keepdim=True) / (sd**4 + 1e-09)
        bc = (skew**2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3) + 1e-09))
        s_up = torch.where(sd < 1e-09, torch.zeros_like(bc), bc).squeeze(-1)
    else:
        s_up = torch.zeros_like(s_down)
    mu = cos.mean(-1, keepdim=True)
    hi = (cos > mu).to(cos.dtype)
    lo = 1.0 - hi
    n_hi, n_lo = (hi.sum(-1), lo.sum(-1))
    s_charm = ((cos * hi).sum(-1) / n_hi.clamp_min(1) - (cos * lo).sum(-1) / n_lo.clamp_min(1)).abs()
    s_charm = torch.where((n_hi == 0) | (n_lo == 0), torch.zeros_like(s_charm), s_charm)
    xc = P - P.mean(dim=-2, keepdim=True)
    with torch.autocast(device_type=P.device.type, enabled=False):
        eig = torch.linalg.eigvalsh(xc @ xc.transpose(-1, -2) / (m - 1)).abs()
    tot = eig.sum(-1)
    s_top = torch.where(tot < 1e-09, torch.zeros_like(tot), eig.amax(-1) / tot.clamp_min(1e-09))
    S = (s_down + s_strange + s_up + s_charm + s_top + s_bottom) / 6.0
    return (S, dim_div)
