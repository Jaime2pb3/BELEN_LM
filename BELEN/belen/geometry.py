"""  
|---|---|
| `geometry.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Draw it. |
"""
-----

from __future__ import annotations

import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import CastedLinear


def _inverse_softplus(x: float) -> float:
    x = max(float(x), 1e-08)
    return math.log(math.expm1(x))


def _logit(x: float) -> float:
    x = min(max(float(x), 1e-06), 1.0 - 1e-06)
    return math.log(x / (1.0 - x))


class GeometricFrameBank(nn.Module):
    n_fields = 3
    n_axes = 2

    def __init__(
        self,
        dim: int,
        head_dim: int,
        *,
        enabled: bool,
        center_decay: float,
        axis_init_cos: float,
        gamma_init: float,
        metric_max: float,
        transport_max: float,
        tau_max: float,
        quantize: bool,
        quant_step_init: float,
        axis_max_abs_cos: float,
        cross_field_target: float,
        eps: float,
    ):
        super().__init__()
        if not 0.0 <= center_decay < 1.0:
            raise ValueError("geo center_decay must be in [0,1)")
        if not -0.95 < axis_init_cos < 0.95:
            raise ValueError("geo axis_init_cos must be in (-0.95,0.95)")
        if dim < 2 or head_dim < 2:
            raise ValueError("geometric frames require dim/head_dim >= 2")
        self.dim = int(dim)
        self.head_dim = int(head_dim)
        self.enabled = bool(enabled)
        self.center_decay = float(center_decay)
        self.metric_max = float(metric_max)
        self.transport_max = float(transport_max)
        self.tau_max = float(tau_max)
        self.quantize = bool(quantize)
        self.axis_max_abs_cos = float(axis_max_abs_cos)
        self.cross_field_target = float(cross_field_target)
        self.eps = float(eps)
        self.axes = nn.Parameter(self._init_frames(dim, axis_init_cos))
        self.head_axes = nn.Parameter(self._init_frames(head_dim, axis_init_cos))
        self.gamma_projector = CastedLinear(dim, self.n_fields, bias=True)
        nn.init.zeros_(self.gamma_projector.weight)
        nn.init.constant_(self.gamma_projector.bias, _logit(gamma_init))
        self.metric_raw = nn.Parameter(torch.full((self.n_fields,), _logit(0.25)))
        self.transport_raw = nn.Parameter(torch.full((self.n_fields,), _logit(0.25)))
        self.eta_raw = nn.Parameter(torch.full((self.n_fields,), _logit(0.25)))
        self.disagreement_gain_raw = nn.Parameter(torch.full((self.n_fields,), _inverse_softplus(2.0)))
        self.residual_gain_raw = nn.Parameter(torch.full((self.n_fields,), _inverse_softplus(2.0)))
        self.log_coord_scale = nn.Parameter(torch.zeros(self.n_fields, self.n_axes))
        self.register_buffer("quant_step", torch.full((self.n_fields, self.n_axes), float(quant_step_init)))
        self.register_buffer("center", torch.zeros(self.n_fields, dim))
        self.register_buffer("center_initialized", torch.zeros(self.n_fields, dtype=torch.bool))
        self._last_gamma = None
        self._last_tau = None
        self._last_disagreement = None
        self._last_projection_residual = None
        if not self.enabled:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _init_frames(dim: int, target_cos: float) -> Tensor:
        frames = []
        sin = math.sqrt(max(1.0 - target_cos * target_cos, 1e-08))
        for _ in range(GeometricFrameBank.n_fields):
            q, _ = torch.linalg.qr(torch.randn(dim, 2), mode="reduced")
            left = q[:, 0]
            right = target_cos * q[:, 0] + sin * q[:, 1]
            frames.append(torch.stack([left, right], dim=0))
        return torch.stack(frames, dim=0)

    def _normalized_axes(self, head_space: bool = False) -> Tensor:
        raw = self.head_axes if head_space else self.axes
        return F.normalize(raw.float(), dim=-1, eps=self.eps)

    def _gram(self, field_idx: int, head_space: bool = False) -> tuple[Tensor, Tensor]:
        axes = self._normalized_axes(head_space)[field_idx]
        gram = axes @ axes.T
        eye = torch.eye(self.n_axes, device=gram.device, dtype=gram.dtype)
        with torch.autocast(device_type=gram.device.type, enabled=False):
            gram_fp32 = gram.float()
            eye_fp32 = torch.eye(gram_fp32.size(-1), device=gram_fp32.device, dtype=torch.float32)
            regularization = max(float(self.eps), 0.0001)
            gram_inv = torch.linalg.pinv(
                gram_fp32 + regularization * eye_fp32, hermitian=True, rtol=1e-05
            ).to(dtype=gram.dtype)
        return (axes, gram_inv)

    def _dual_projection(self, z: Tensor, field_idx: int, head_space: bool = False) -> tuple[Tensor, Tensor]:
        axes, gram_inv = self._gram(field_idx, head_space=head_space)
        cov = torch.einsum("...d,ad->...a", z.float(), axes)
        coords = cov @ gram_inv
        plane = torch.einsum("...a,ad->...d", coords, axes)
        return (plane, coords)

    @staticmethod
    def _preserve_rms(reference: Tensor, transformed: Tensor, eps: float) -> Tensor:
        ref_rms = reference.float().square().mean(dim=-1, keepdim=True).sqrt()
        new_rms = transformed.float().square().mean(dim=-1, keepdim=True).sqrt()
        return transformed * (ref_rms / new_rms.clamp_min(eps)).to(transformed.dtype)

    def gamma_prior(self, x: Tensor) -> Tensor:
        logits = self.gamma_projector(F.rms_norm(x, (x.size(-1),)))
        return torch.sigmoid(logits.float())

    def metric_transform(self, z: Tensor, field_idx: int, gamma: Tensor, *, values: bool = False) -> Tensor:
        if not self.enabled:
            return z
        plane, _ = self._dual_projection(z, field_idx, head_space=True)
        if values:
            strength = self.transport_max * torch.sigmoid(self.transport_raw[field_idx])
        else:
            strength = self.metric_max * torch.sigmoid(self.metric_raw[field_idx])
        g = gamma[:, None, :, None].to(z.dtype)
        out = z + strength.to(z.dtype) * g * plane.to(z.dtype)
        if values:
            return self._preserve_rms(z, out, self.eps)
        return F.rms_norm(out, (out.size(-1),), eps=self.eps)

    def effective_center(self, h: Tensor, field_idx: int) -> Tensor:
        stored = self.center[field_idx].float()
        batch_center = h.detach().float().mean(dim=(0, 1))
        init = self.center_initialized[field_idx]
        return torch.where(init, stored, batch_center)

    def project(self, h: Tensor, field_idx: int) -> tuple[Tensor, Tensor]:
        dtype = h.dtype
        center = self.effective_center(h, field_idx)
        z = h.float() - center
        plane, coords = self._dual_projection(z, field_idx, head_space=False)
        z_perp = z - plane
        if self.quantize:
            scale = self.log_coord_scale[field_idx].float().exp().clamp(0.1, 10.0)
            delta = self.quant_step[field_idx].float().clamp_min(0.0001)
            scaled = coords * scale
            discrete = torch.round(scaled / delta) * delta
            scaled_q = scaled + (discrete - scaled).detach()
            coords_q = scaled_q / scale
        else:
            coords_q = coords
        axes = self._normalized_axes(False)[field_idx]
        plane_q = torch.einsum("...a,ad->...d", coords_q, axes)
        projected = center + z_perp + plane_q
        residual = (projected - h.float()).norm(dim=-1) / z.norm(dim=-1).clamp_min(self.eps)
        return (projected.to(dtype), residual.clamp(0.0, 4.0))

    def geodesic_transport(
        self, h: Tensor, projected: Tensor, field_idx: int, gamma: Tensor
    ) -> tuple[Tensor, Tensor]:
        center = self.effective_center(h, field_idx)
        z = h.float() - center
        p = projected.float() - center
        r0 = z.norm(dim=-1, keepdim=True)
        r1 = p.norm(dim=-1, keepdim=True)
        u = z / r0.clamp_min(self.eps)
        q = p / r1.clamp_min(self.eps)
        dot = (u * q).sum(dim=-1, keepdim=True).clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        tangent = (q - dot * u) / sin_theta.clamp_min(self.eps)
        eta = torch.sigmoid(self.eta_raw[field_idx])
        tau = (self.tau_max * eta * gamma.float()).clamp(0.0, 1.0).unsqueeze(-1)
        direction = torch.cos(tau * theta) * u + torch.sin(tau * theta) * tangent
        lerp_direction = F.normalize((1.0 - tau) * u + tau * q, dim=-1, eps=self.eps)
        direction = torch.where((theta < 0.0001).expand_as(direction), lerp_direction, direction)
        log_r = (1.0 - tau) * torch.log(r0.clamp_min(self.eps)) + tau * torch.log(r1.clamp_min(self.eps))
        out = center + torch.exp(log_r) * direction
        degenerate = ((r0 < self.eps) | (r1 < self.eps)).expand_as(out)
        linear = h.float() + tau * (projected.float() - h.float())
        out = torch.where(degenerate, linear, out)
        out = torch.where((tau <= self.eps).expand_as(out), h.float(), out)
        out = torch.where((tau >= 1.0 - self.eps).expand_as(out), projected.float(), out)
        return (out.to(h.dtype), tau.squeeze(-1))

    def posterior_gamma(
        self, prior: Tensor, field_outputs: list[Tensor], projections: list[Tensor], residuals: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        stack = torch.stack([x.float() for x in field_outputs], dim=2)
        total = stack.sum(dim=2, keepdim=True)
        other = (total - stack) / float(self.n_fields - 1)
        disagreement = 0.5 * (1.0 - F.cosine_similarity(stack, other, dim=-1))
        disagreement = disagreement.clamp(0.0, 1.0)
        residual = torch.stack(residuals, dim=-1).tanh()
        axes = self._normalized_axes(False)
        axis_cos = (axes[:, 0] * axes[:, 1]).sum(dim=-1)
        confidence = torch.sqrt((1.0 - axis_cos.square()).clamp_min(self.eps))
        prior_logit = torch.logit(prior.clamp(1e-05, 1.0 - 1e-05))
        gd = F.softplus(self.disagreement_gain_raw).view(1, 1, -1)
        gr = F.softplus(self.residual_gain_raw).view(1, 1, -1)
        logits = prior_logit + gd * (disagreement - 0.15) + gr * (residual - 0.1)
        gamma = torch.sigmoid(logits) * confidence.view(1, 1, -1)
        return (gamma.clamp(0.0, 1.0), disagreement)

    @torch.no_grad()
    def update_centers(self, field_outputs: list[Tensor]) -> None:
        if not self.training:
            return
        currents = torch.stack([h.detach().float().mean(dim=(0, 1)) for h in field_outputs], dim=0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(currents, op=dist.ReduceOp.SUM)
            currents.div_(dist.get_world_size())
        for i, current in enumerate(currents):
            initialized = self.center_initialized[i].to(current.dtype)
            ema = self.center[i].float() * self.center_decay + current * (1.0 - self.center_decay)
            updated = initialized * ema + (1.0 - initialized) * current
            self.center[i].copy_(updated)
            self.center_initialized[i].fill_(True)

    def regularization_loss(self) -> Tensor:
        axes = self._normalized_axes(False)
        head_axes = self._normalized_axes(True)
        cos_full = (axes[:, 0] * axes[:, 1]).sum(dim=-1)
        cos_head = (head_axes[:, 0] * head_axes[:, 1]).sum(dim=-1)
        max_cos = self.axis_max_abs_cos
        independence = F.relu(cos_full.abs() - max_cos).square().mean()
        independence = independence + 0.5 * F.relu(cos_head.abs() - max_cos).square().mean()
        qs = []
        for i in range(self.n_fields):
            q, _ = torch.linalg.qr(axes[i].T, mode="reduced")
            qs.append(q)
        overlaps = []
        for i, j in ((0, 1), (1, 2), (0, 2)):
            c = qs[i].T @ qs[j]
            overlaps.append(c.square().sum() / self.n_axes)
        overlap = torch.stack(overlaps)
        coherence = (overlap - self.cross_field_target).square().mean()
        return independence + 0.25 * coherence

    @torch.no_grad()
    def calibrate(self, neutral: Tensor, left: Tensor, right: Tensor) -> dict:

        def canonical(name: str, x: Tensor) -> Tensor:
            if x.ndim == 2:
                x = x[:, None, :].expand(-1, self.n_fields, -1)
            if x.ndim != 3 or x.size(1) != self.n_fields or x.size(2) != self.dim:
                raise ValueError(f"{name} must be (N,3,{self.dim}) or (N,{self.dim}); got {tuple(x.shape)}")
            if x.size(0) < 2 or not torch.isfinite(x).all():
                raise ValueError(f"{name} requires >=2 finite samples")
            return x.to(device=self.axes.device, dtype=torch.float32)

        n = canonical("neutral", neutral)
        l = canonical("left", left)
        r = canonical("right", right)
        mu_n = n.median(dim=0).values
        raw_l = l.median(dim=0).values - mu_n
        raw_r = r.median(dim=0).values - mu_n
        if (raw_l.norm(dim=-1) < self.eps).any() or (raw_r.norm(dim=-1) < self.eps).any():
            raise RuntimeError("degenerate geometric calibration: near-zero axis")
        axis_l = F.normalize(raw_l, dim=-1, eps=self.eps)
        axis_r = F.normalize(raw_r, dim=-1, eps=self.eps)
        cos_lr = (axis_l * axis_r).sum(dim=-1)
        if (cos_lr.abs() > 0.985).any():
            raise RuntimeError("degenerate geometric calibration: nearly collinear L/R axes")
        self.center.copy_(mu_n)
        self.center_initialized.fill_(True)
        self.axes[:, 0].copy_(axis_l)
        self.axes[:, 1].copy_(axis_r)
        for i in range(self.n_fields):
            axes, gram_inv = self._gram(i, head_space=False)
            cov_l = torch.einsum("nd,ad->na", l[:, i] - mu_n[i], axes)
            cov_r = torch.einsum("nd,ad->na", r[:, i] - mu_n[i], axes)
            coords_l = cov_l @ gram_inv
            coords_r = cov_r @ gram_inv
            typical = torch.stack([coords_l[:, 0].abs().median(), coords_r[:, 1].abs().median()]).clamp_min(
                0.001
            )
            self.log_coord_scale[i].copy_(-typical.log())
        return self.snapshot()

    @torch.no_grad()
    def snapshot(self) -> dict:
        axes = self._normalized_axes(False)
        head_axes = self._normalized_axes(True)
        axis_cos = (axes[:, 0] * axes[:, 1]).sum(dim=-1)
        head_cos = (head_axes[:, 0] * head_axes[:, 1]).sum(dim=-1)
        out = {
            "enabled": self.enabled,
            "axis_cos_LR": [float(x) for x in axis_cos.cpu()],
            "head_axis_cos_LR": [float(x) for x in head_cos.cpu()],
            "gram_det": [float(x) for x in (1.0 - axis_cos.square()).cpu()],
            "center_initialized": [bool(x) for x in self.center_initialized.cpu()],
            "metric_strength": [float(x) for x in (self.metric_max * torch.sigmoid(self.metric_raw)).cpu()],
            "transport_strength": [
                float(x) for x in (self.transport_max * torch.sigmoid(self.transport_raw)).cpu()
            ],
            "eta": [float(x) for x in torch.sigmoid(self.eta_raw).cpu()],
            "quant_step": [[float(v) for v in row] for row in self.quant_step.cpu()],
        }
        if self._last_gamma is not None:
            out["gamma_mean"] = [float(x) for x in self._last_gamma.float().mean(dim=(0, 1)).cpu()]
        if self._last_tau is not None:
            out["tau_mean"] = [float(x) for x in self._last_tau.float().mean(dim=(0, 1)).cpu()]
        return out
