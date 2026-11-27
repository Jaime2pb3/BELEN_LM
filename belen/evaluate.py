"""  
|---|---|
| `evaluate.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Face it. |
-----
"""

from __future__ import annotations

import math
import torch
import torch.distributed as dist

from .layers import autocast_for


def eval_val(
    args,
    model,
    rank,
    world_size,
    device,
    grad_accum,
    val_tokens,
    base_bytes_lut,
    leading_space_lut,
    boundary_lut,
    ttt_trainer=None,
    log_fn=None,
    full: bool = True,
    base_model=None,
):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum)
    local_batch_seqs = max(1, local_batch_tokens // args.train_seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    if not full and args.val_max_tokens > 0:
        total_seqs = min(total_seqs, max(local_batch_seqs, args.val_max_tokens // args.train_seq_len))
    if base_model is None:
        base_model = model.module if hasattr(model, "module") else model
    buf_snap = {k: b.detach().clone() for k, b in base_model.named_buffers()} if args.eval_isolated else None
    ttt_snap = (
        ttt_trainer.snapshot()
        if ttt_trainer is not None and args.ttt_enabled and args.ttt_restore_after_eval
        else None
    )
    seq_start = total_seqs * rank // world_size
    seq_end = total_seqs * (rank + 1) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
    ttt_active = ttt_trainer is not None and bool(args.ttt_enabled)
    warmup_chunks_left = int(args.ttt_warmup_chunks) if ttt_active else 0
    chunk_idx = 0
    ttt_diag_all = {"losses": [], "grad_norms": []}
    model.eval()
    for bs in range(seq_start, seq_end, local_batch_seqs):
        be = min(bs + local_batch_seqs, seq_end)
        raw = val_tokens[bs * args.train_seq_len : be * args.train_seq_len + 1]
        raw = raw.to(device=device, dtype=torch.int64, non_blocking=True)
        x = raw[:-1].reshape(-1, args.train_seq_len)
        y = raw[1:].reshape(-1, args.train_seq_len)
        with torch.no_grad():
            with autocast_for(device):
                batch_loss = model(x, y, use_axon=True).detach()
        n = float(y.numel())
        loss_sum += batch_loss.to(torch.float64) * n
        token_cnt += n
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        tb += (leading_space_lut[tgt_ids] & ~boundary_lut[prev_ids]).to(torch.int16)
        byte_cnt += tb.to(torch.float64).sum()
        if ttt_active:
            base = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                _ = ttt_trainer.base_model.monitor_resonance.compute(
                    ttt_trainer.base_model.resonance.get_slots()
                )
            if warmup_chunks_left <= 0:
                diag = ttt_trainer.adapt(x)
                ttt_diag_all["losses"].extend(diag["losses"])
                ttt_diag_all["grad_norms"].extend(diag["grad_norms"])
            else:
                warmup_chunks_left -= 1
        chunk_idx += 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_cnt, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_cnt, op=dist.ReduceOp.SUM)
    val_loss = float(loss_sum / token_cnt)
    bpt = val_loss / math.log(2.0)
    tpb = float(token_cnt / byte_cnt)
    if ttt_active and ttt_diag_all["losses"] and (log_fn is not None):
        avg_loss = sum(ttt_diag_all["losses"]) / len(ttt_diag_all["losses"])
        avg_gnorm = sum(ttt_diag_all["grad_norms"]) / len(ttt_diag_all["grad_norms"])
        log_fn(
            f"  [TTT-Iso]  chunks_adapted={ttt_trainer.n_chunks_adapted}  avg_L_iso={avg_loss:.4f}  avg_|∇|={avg_gnorm:.4f}"
        )
    if ttt_snap is not None:
        ttt_trainer.restore(ttt_snap)
    if buf_snap is not None:
        with torch.no_grad():
            for k, b in base_model.named_buffers():
                b.copy_(buf_snap[k])
    model.train()
    return (val_loss, float(bpt * tpb))


@torch.no_grad()
def ablation_report(args, base_model, val_tokens, device) -> dict:
    was_training = base_model.training
    base_model.eval()
    T = args.train_seq_len
    B = max(1, min(8, args.val_batch_size // T))
    n_seq = min((val_tokens.numel() - 1) // T, max(B, args.ablate_tokens // T))
    snap = {k: b.detach().clone() for k, b in base_model.named_buffers()}
    res = base_model.resonance
    hooks: list = []

    def restore():
        for k, b in base_model.named_buffers():
            b.copy_(snap[k])
        res.frozen = False
        while hooks:
            hooks.pop().remove()

    def score(use_axon=True) -> float:
        tot, cnt = (0.0, 0)
        for s in range(0, n_seq, B):
            e = min(s + B, n_seq)
            raw = val_tokens[s * T : e * T + 1].to(device=device, dtype=torch.int64)
            with autocast_for(device):
                loss = base_model(raw[:-1].reshape(-1, T), raw[1:].reshape(-1, T), use_axon=use_axon)
            tot += float(loss) * (e - s)
            cnt += e - s
        return tot / max(cnt, 1)

    out = {}
    restore()
    out["completo (memoria viva)"] = score()
    restore()
    res.frozen = True
    out["memoria congelada"] = score()
    restore()
    res.slots.copy_(snap["resonance.slots"].mean(0, keepdim=True).expand_as(res.slots))
    res.frozen = True
    out["memoria = un solo slot promedio"] = score()
    restore()
    res.slots.zero_()
    res.frozen = True
    out["memoria en cero"] = score()
    restore()
    out["sin axón ni compuerta"] = score(use_axon=False)
    restore()
    gate = getattr(base_model, "axon", None)
    if base_model.axon_mode == "ach" and hasattr(gate, "static_rho"):
        # ¿La compuerta aporta por su dinámica o por su escala promedio? ρ fijo en su media
        # deja la escala aprendida y quita solo la respuesta a la incoherencia.
        rhos = []
        h = gate.register_forward_hook(lambda m, i, o: rhos.append(float(m.last_rho)))
        score()
        h.remove()
        restore()
        gate.static_rho = sum(rhos) / max(len(rhos), 1)
        out["compuerta sin dinámica (ρ fijo en su media)"] = score()
        gate.static_rho = None
        restore()
    hooks.append(base_model.dendrite.register_forward_hook(lambda m, i, o: o * 0.0))
    out["sin dendrita"] = score()
    restore()
    base_model.train(was_training)
    return out
