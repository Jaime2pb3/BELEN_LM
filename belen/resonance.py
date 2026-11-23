"""  
|---|---|
| `resonance.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Feel it. |
-----
"""


from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResonanceBuffer(nn.Module):

    def __init__(
        self,
        dim: int,
        n_slots: int = 64,
        ema_decay: float = 0.99,
        mode: str = "mean",
        mem_span: int = 8,
        write: str = "mean",
    ):
        super().__init__()
        assert mode in ("mean", "seq", "topk")
        assert write in ("mean", "sample")
        self.write = write
        self.frozen = False
        self.dim = dim
        self.n_slots = n_slots
        self.ema_decay = ema_decay
        self.mode = mode
        self.mem_span = mem_span if mode == "seq" else 1
        if mode == "seq":
            slots_shape = (n_slots, self.mem_span, dim)
        else:
            slots_shape = (n_slots, dim)
        self.register_buffer("slots", torch.zeros(*slots_shape))
        self.register_buffer("slot_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("fill_count", torch.zeros(1, dtype=torch.float32))

    @torch.no_grad()
    def update(self, h: Tensor) -> None:
        if self.frozen:
            return
        if self.write == "sample":
            h = h.detach().index_select(0, self.slot_ptr % h.size(0))
        if self.mode == "mean":
            sample = h.detach().float().mean(dim=(0, 1))
            target = sample
        elif self.mode == "topk":
            h_flat = h.detach().float().reshape(-1, h.size(-1))
            norms = h_flat.norm(dim=-1)
            idx = norms.argmax()
            target = h_flat[idx]
        else:
            h_mean_batch = h.detach().float().mean(dim=0)
            T = h_mean_batch.size(0)
            L = self.mem_span
            idx = torch.linspace(0, T - 1, L, device=h.device).long()
            target = h_mean_batch.index_select(0, idx)
        ptr_mask_flat = (
            torch.nn.functional.one_hot(self.slot_ptr, num_classes=self.n_slots).squeeze(0).float()
        )
        if self.mode == "seq":
            ptr_mask = ptr_mask_flat.view(self.n_slots, 1, 1)
            target_bcast = target.unsqueeze(0)
        else:
            ptr_mask = ptr_mask_flat.view(self.n_slots, 1)
            target_bcast = target.unsqueeze(0)
        warmup_weight = torch.clamp(self.fill_count / float(self.n_slots), max=1.0)
        effective_decay = self.ema_decay * warmup_weight
        old = self.slots.float()
        new_slot_val = effective_decay * old + (1.0 - effective_decay) * target_bcast
        updated = ptr_mask * new_slot_val + (1.0 - ptr_mask) * old
        self.slots.copy_(updated.to(self.slots.dtype))
        self.slot_ptr.add_(1).remainder_(self.n_slots)
        self.fill_count.add_(1.0)

    def get_slots(self) -> Tensor:
        if self.mode == "seq":
            return self.slots.reshape(-1, self.dim)
        return self.slots

    def resonance_score(self, h: Tensor) -> float:
        h_mean = h.detach().float().mean(dim=(0, 1))
        h_norm = F.normalize(h_mean.unsqueeze(0), dim=-1)
        slots_flat = self.get_slots().float()
        s_norm = F.normalize(slots_flat, dim=-1)
        sims = (h_norm @ s_norm.T).squeeze(0)
        return float(sims.max().item())
