"""  
|---|---|
| `ttt_iso.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Improve it. |
"""
-----

from __future__ import annotations

from torch import Tensor, nn

from .layers import autocast_for
from .model import Belen
from .optim import Lion


class TTTIsoTrainer:

    def __init__(
        self,
        base_model: "Belen",
        lr: float = 0.0005,
        inner_steps: int = 3,
        beta: float = 0.1,
        reset_each_chunk: bool = False,
    ):
        self.base_model = base_model
        self.inner_steps = inner_steps
        self.beta = beta
        self.reset_each_chunk = reset_each_chunk
        self.percept_params = list(base_model.perception.parameters())
        self.opt = Lion(self.percept_params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
        self._init_state = None
        if reset_each_chunk:
            self._init_state = {k: v.detach().clone() for k, v in base_model.perception.state_dict().items()}
        self.n_chunks_adapted = 0
        self.last_loss = float("nan")

    def snapshot(self) -> dict:
        return {k: v.detach().clone() for k, v in self.base_model.perception.state_dict().items()}

    def restore(self, snap: dict) -> None:
        self.base_model.perception.load_state_dict(snap)
        self.opt.state.clear()

    def _freeze_all_except_perception(self) -> dict:
        prev = {}
        for name, p in self.base_model.named_parameters():
            prev[name] = p.requires_grad
            p.requires_grad_(name.startswith("perception."))
        return prev

    def _restore_grad_state(self, prev: dict) -> None:
        for name, p in self.base_model.named_parameters():
            if name in prev:
                p.requires_grad_(prev[name])

    def adapt(self, input_ids: Tensor) -> dict:
        if self.reset_each_chunk and self._init_state is not None:
            self.base_model.perception.load_state_dict(self._init_state)
        prev_grad_state = self._freeze_all_except_perception()
        self.base_model.perception.train()
        diag = {"losses": [], "grad_norms": []}
        try:
            for step in range(self.inner_steps):
                self.opt.zero_grad(set_to_none=True)
                with autocast_for(input_ids.device):
                    _, perceptual_flow = self.base_model.forward_with_percept(input_ids)
                p_new = perceptual_flow.float().mean(dim=(0, 1))
                R_frozen = self.base_model.resonance.get_slots().detach().float()
                loss = self.base_model.monitor_resonance.compute_ttt_loss(
                    realizations_frozen=R_frozen, realization_new=p_new, beta=self.beta
                )
                diag["losses"].append(float(loss.item()))
                loss.backward()
                total_grad = 0.0
                for p in self.percept_params:
                    if p.grad is not None:
                        total_grad += float(p.grad.norm().item() ** 2)
                diag["grad_norms"].append(float(total_grad**0.5))
                self.opt.step()
            self.last_loss = diag["losses"][-1] if diag["losses"] else float("nan")
            self.n_chunks_adapted += 1
        finally:
            self.base_model.perception.eval()
            self._restore_grad_state(prev_grad_state)
        return diag
