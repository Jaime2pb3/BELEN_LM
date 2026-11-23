"""  
|---|---|
| `model.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | DO IT. |
-----
"""


from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .axon import AxonProjector, AxonViews
from .cholinergic_gate import CholinergicGate
from .config import Hyperparameters
from .dendrite import DendriticEncoder
from .geometry import GeometricFrameBank
from .layers import RMSNorm
from .molecule_gate import MoleculeGate
from .perception import PerceptionAgent
from .pfv_monitor import IsomericPolarizationMonitor
from .resonance import ResonanceBuffer
from .soma import SomaBlock


class Belen(nn.Module):

    def __init__(self, args: Hyperparameters):
        super().__init__()
        d = args.model_dim
        geometry = None
        if int(getattr(args, "geometric_dendrite", 0)):
            geometry = GeometricFrameBank(
                d,
                d // sum(args.dendrite_field_heads),
                enabled=True,
                center_decay=args.geo_center_decay,
                axis_init_cos=args.geo_axis_init_cos,
                gamma_init=args.geo_gamma_init,
                metric_max=args.geo_metric_max,
                transport_max=args.geo_transport_max,
                tau_max=args.geo_tau_max,
                quantize=bool(args.geo_quantize),
                quant_step_init=args.geo_quant_step_init,
                axis_max_abs_cos=args.geo_axis_max_abs_cos,
                cross_field_target=args.geo_cross_field_target,
                eps=args.geo_eps,
            )
        self.logit_softcap = args.logit_softcap
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, d)
        self.input_norm = RMSNorm()
        self.dendrite = DendriticEncoder(
            dim=d,
            field_heads=list(args.dendrite_field_heads),
            local_span=args.dendrite_local_span,
            mid_span=args.dendrite_mid_span,
            n_kv_heads=args.dendrite_kv_heads,
            rope_base=args.rope_base,
            n_proto=args.dendrite_n_proto,
            layout=args.dendrite_layout,
            geometry=geometry,
        )
        self.dendrite_norm = RMSNorm()
        self.resonance = ResonanceBuffer(
            dim=d,
            n_slots=args.resonance_slots,
            ema_decay=args.resonance_ema_decay,
            mode=args.resonance_mode,
            mem_span=args.resonance_mem_span,
            write=args.resonance_write,
        )
        self.perception = PerceptionAgent(dim=d, n_heads=args.perception_heads)
        self.percept_norm = RMSNorm()
        self.use_percept_norm = bool(args.percept_norm)
        self.n_enc = args.num_layers // 2
        self.n_dec = args.num_layers - self.n_enc
        self.blocks = nn.ModuleList(
            [
                SomaBlock(
                    d, args.num_heads, args.num_kv_heads, args.mlp_mult, args.rope_base, qk_gain_init=1.5
                )
                for _ in range(args.num_layers)
            ]
        )
        n_skip = min(self.n_enc, self.n_dec)
        self.skip_weights = nn.Parameter(
            torch.full((n_skip, d), float(getattr(args, "skip_init", 1.0)), dtype=torch.float32)
        )
        self.final_norm = RMSNorm()
        assert args.axon_mode in ("views", "noise", "ach")
        self.axon_mode = args.axon_mode
        self.ach_first_block = min(max(int(getattr(args, "ach_first_block", 2)), 0), args.num_layers - 3)
        if self.axon_mode == "ach":
            self.axon = CholinergicGate(
                dim=d,
                lam=args.ach_lambda,
                theta_init=args.ach_theta,
                rho_k_init=args.ach_rho_k,
                base_scale=args.ach_base_scale,
                ema_momentum=args.ach_ema_momentum,
            )
        elif self.axon_mode == "views":
            self.axon = AxonViews(
                dim=d, n_views=args.axon_n_views, lam=args.axon_lambda, momentum=args.axon_regime_momentum
            )
        else:
            self.axon = AxonProjector(
                dim=d,
                n_views=args.axon_n_views,
                t_min=args.axon_t_min,
                t_max=args.axon_t_max,
                lam=args.axon_lambda,
                stress_threshold=args.axon_stress_threshold,
            )
        self.molecule_gate = MoleculeGate(d, args.vocab_size)
        self.pfv_enabled = bool(int(os.environ.get("PFV_MONITOR", "1")))
        if self.pfv_enabled:
            K_resonance = (
                args.resonance_slots * args.resonance_mem_span
                if args.resonance_mode == "seq"
                else args.resonance_slots
            )
            self.monitor_resonance = IsomericPolarizationMonitor(
                n_realizations=K_resonance,
                window=int(os.environ.get("PFV_WINDOW", 50)),
                agg=os.environ.get("PFV_AGG", "mean"),
                epsilon_quantile=float(os.environ.get("PFV_EPS_Q", 0.1)),
            )
            n_proto = args.dendrite_n_proto
            self.monitor_plastic_local = IsomericPolarizationMonitor(n_realizations=n_proto)
            self.monitor_plastic_mid = IsomericPolarizationMonitor(n_realizations=n_proto)
            self.monitor_plastic_global = IsomericPolarizationMonitor(n_realizations=n_proto)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)

    @torch.no_grad()
    def pfv_snapshot(self) -> dict:
        if not self.pfv_enabled:
            return {}
        out = {}
        slots = self.resonance.get_slots()
        out["resonance"] = self.monitor_resonance.compute(slots)
        out["plastic_local"] = self.monitor_plastic_local.compute(self.dendrite.plastic_local.prototypes)
        out["plastic_mid"] = self.monitor_plastic_mid.compute(self.dendrite.plastic_mid.prototypes)
        out["plastic_global"] = self.monitor_plastic_global.compute(self.dendrite.plastic_global.prototypes)
        return out

    def _encode_decode(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        h, x, _ = self._encode(input_ids)
        return (h, x)

    def _encode(self, input_ids: Tensor) -> tuple[Tensor, Tensor, list]:
        x = self.tok_emb(input_ids)
        x = self.input_norm(x)
        x0 = x
        dendritic_out = self.dendrite(self.dendrite_norm(x))
        resonance_slots = self.resonance.get_slots()
        perceptual_flow = self.perception(dendritic_out, resonance_slots)
        if self.use_percept_norm:
            x = x + dendritic_out + self.percept_norm(perceptual_flow)
        else:
            x = x + dendritic_out + perceptual_flow
        skips = []
        trace = []
        for i in range(self.n_enc):
            x = self.blocks[i](x, x0)
            skips.append(x)
            trace.append(x)
        for i in range(self.n_dec):
            if skips:
                sw = self.skip_weights[i].to(dtype=x.dtype)
                x = x + sw[None, None, :] * skips.pop()
            x = self.blocks[self.n_enc + i](x, x0)
            trace.append(x)
        if self.axon_mode == "ach":
            views = torch.stack([t.detach() for t in trace[self.ach_first_block :]], dim=2)
        else:
            views = [t.detach() for t in trace[-max(2, self.args.axon_n_views) :]]
        h = self.final_norm(x)
        self.resonance.update(h)
        return (h, x, views)

    def _logits(self, input_ids: Tensor, use_axon: bool = True) -> Tensor:
        h, _, views = self._encode(input_ids)
        w = None
        if use_axon:
            if self.axon_mode == "ach":
                h, w = self.axon(h, views, return_signal=True)
            elif self.axon_mode == "views":
                h, w = self.axon(h, views)
            else:
                h = self.axon(h)
        h_flat = h.reshape(-1, h.size(-1))
        logits = F.linear(h_flat, self.tok_emb.weight.to(h_flat.dtype))
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if use_axon:
            logits = self.molecule_gate(
                logits,
                h_flat,
                0.0 if w is not None or self.axon_mode == "ach" else self.axon.last_S,
                self.args.axon_stress_threshold,
                w=w,
            )
        return logits

    def forward(self, input_ids: Tensor, target_ids: Tensor, use_axon: bool = True) -> Tensor:
        logits = self._logits(input_ids, use_axon)
        targets = target_ids.reshape(-1)
        loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.dendrite.geometry is not None and self.args.geo_aux_scale > 0.0:
            loss = loss + self.args.geo_aux_scale * self.dendrite.geometry_aux_loss()
        return loss

    def get_logits(self, input_ids: Tensor) -> Tensor:
        return self._logits(input_ids, use_axon=True)

    def forward_with_percept(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        x = self.tok_emb(input_ids)
        x = self.input_norm(x)
        x0 = x
        dendritic_out = self.dendrite(self.dendrite_norm(x))
        resonance_slots = self.resonance.get_slots()
        perceptual_flow = self.perception(dendritic_out, resonance_slots)
        return (dendritic_out, perceptual_flow)
