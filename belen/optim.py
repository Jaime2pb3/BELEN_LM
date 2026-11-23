"""  
|---|---|
| `optim.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Build it. |
-----
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .model import Belen


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-07) -> Tensor:
    a, b, c = (3.4445, -4.775, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):

    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov)
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum((int(p.numel()) for p in params))
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


class Lion(torch.optim.Optimizer):

    def __init__(
        self, params, lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0
    ):
        if lr <= 0:
            raise ValueError(f"Lion lr must be positive, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Lion betas must be in [0,1), got {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                m = state["exp_avg"]
                update = m.mul(beta1).add_(g.float(), alpha=1.0 - beta1).sign_()
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update.to(dtype=p.dtype), alpha=-lr)
                m.mul_(beta2).add_(g.float(), alpha=1.0 - beta2)
        return loss


def build_param_groups(base_model: "Belen") -> dict:
    emb = [base_model.tok_emb.weight]
    ctrl = ("resid_mix", "scale", "q_gain")
    matrix = (
        [
            p
            for n, p in base_model.dendrite.named_parameters()
            if p.ndim == 2 and not n.startswith("geometry.")
        ]
        + [p for n, p in base_model.perception.named_parameters() if p.ndim == 2]
        + [
            p
            for n, p in base_model.blocks.named_parameters()
            if p.ndim == 2 and (not any((c in n for c in ctrl)))
        ]
        + [p for n, p in base_model.molecule_gate.named_parameters() if p.ndim == 2]
    )
    nowd = [p for n, p in base_model.blocks.named_parameters() if "resid_mix" in n] + [
        base_model.skip_weights
    ]
    geometry = [p for n, p in base_model.dendrite.named_parameters() if n.startswith("geometry.")]
    taken = {id(p) for p in emb + matrix + nowd + geometry}
    scalar = [p for p in base_model.parameters() if id(p) not in taken]
    groups = {"emb": emb, "matrix": matrix, "scalar": scalar, "scalar_nowd": nowd}
    if geometry:
        groups["geometry"] = geometry
    seen: dict = {}
    for g, ps in groups.items():
        for p in ps:
            seen.setdefault(id(p), []).append(g)
    names = {id(p): n for n, p in base_model.named_parameters()}
    dup = [names[i] for i, v in seen.items() if len(v) > 1]
    miss = [n for n, p in base_model.named_parameters() if id(p) not in seen]
    if dup or miss:
        raise RuntimeError(f"grupos de optimizador mal formados: repetidos {dup}, sin grupo {miss}")
    return groups


def lr_schedule(
    step: int, elapsed_ms: float, iterations: int, warmdown_iters: int, max_wallclock_seconds: float
) -> float:
    if warmdown_iters <= 0:
        return 1.0
    if max_wallclock_seconds > 0:
        max_ms = 1000.0 * max_wallclock_seconds
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = warmdown_iters * step_ms
        remaining = max(max_ms - elapsed_ms, 0.0)
        wall = remaining / max(warmdown_ms, 1e-09) if remaining <= warmdown_ms else 1.0
    else:
        wall = 1.0
    start = max(iterations - warmdown_iters, 0)
    by_step = max((iterations - step) / max(warmdown_iters, 1), 0.0) if step >= start else 1.0
    return min(wall, by_step)


def resolve_warmdown(args) -> int:
    return int(0.3 * args.iterations) if args.warmdown_iters < 0 else args.warmdown_iters
