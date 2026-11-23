"""  
|---|---|
| `molecule_gate.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Open it. |
-----
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers import CastedLinear


class MoleculeGate(nn.Module):

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        rank = min(32, dim // 8)
        self.gate_down = CastedLinear(dim, rank, bias=False)
        self.gate_up = CastedLinear(rank, vocab_size, bias=False)
        nn.init.normal_(self.gate_down.weight, std=0.01)
        nn.init.zeros_(self.gate_up.weight)

    def forward(
        self, logits: Tensor, h: Tensor, S: float, stress_threshold: float, w: Tensor | None = None
    ) -> Tensor:
        if w is not None:
            correction = self.gate_up(torch.tanh(self.gate_down(h)))
            return logits + w.reshape(-1, 1).to(correction.dtype) * correction
        if S <= stress_threshold:
            return logits
        correction = self.gate_up(torch.tanh(self.gate_down(h)))
        stress_weight = min((S - stress_threshold) / (1.0 - stress_threshold + 1e-09), 1.0)
        return logits + stress_weight * correction
