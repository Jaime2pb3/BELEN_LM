"""  
|---|---|
| `train.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Learn it. |
"""
-----

from __future__ import annotations

import math
import os
import random
import time
import zlib
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .checkpoint import save_checkpoint
from .compress import distill_resonance, quantize_and_compress
from .config import Hyperparameters, LEGACY
from .data import DistributedTokenLoader, build_sentencepiece_luts, load_data_shard, load_validation_tokens
from .evaluate import ablation_report, eval_val
from .layers import autocast_for
from .model import Belen
from .optim import Lion, Muon, build_param_groups, lr_schedule, resolve_warmdown
from .ttt_iso import TTTIsoTrainer

RUN_SUMMARY: dict = {}


def axon_status(base_model) -> str:
    axon = base_model.axon
    if base_model.axon_mode == "ach":
        d = axon.diagnostics()
        return (
            f"  ach_u:{d['u_mean']:.3f}  z:{d['z_mean']:+.3f}  theta:{d['theta']:+.2f}  k:{d['rho_k']:.2f}"
            f"  rho:{d['rho_mean']:.3f}  mask_μ:{d['mask_mean']:.3f}  authority:{d['authority_ratio']:.2f}"
        )
    k = axon.k_value() if hasattr(axon, "k_value") else axon.rho_k.item()
    return f"  axon_S:{axon.last_S:.3f}  mask_μ:{axon.last_mask_mean:.3f}  ρ:{axon.last_rho:.3f}  k:{k:.2f}"


PACKAGE_FILES = sorted(str(p) for p in Path(__file__).resolve().parent.glob("*.py"))


def main() -> None:
    code = "\n".join(Path(p).read_text(encoding="utf-8") for p in PACKAGE_FILES)
    args = Hyperparameters()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum = args.grad_accum if args.grad_accum > 0 else max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum
    per_micro = args.train_batch_tokens // (world_size * grad_accum)
    if (
        args.train_batch_tokens % (world_size * grad_accum)
        or per_micro == 0
        or per_micro % args.train_seq_len
    ):
        raise ValueError(
            f"TRAIN_BATCH_TOKENS={args.train_batch_tokens} debe repartirse en world_size·GRAD_ACCUM={world_size * grad_accum} micro-batches con un número entero de secuencias de TRAIN_SEQ_LEN={args.train_seq_len} (salen {per_micro} tokens por micro-batch)"
        )
    want = args.device_pref
    if want == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required (para una corrida reducida en CPU: DEVICE=cpu)")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    on_cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if on_cuda else lambda: None
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if args.sdp_backend == "math":
            from torch.backends.cuda import (
                enable_flash_sdp,
                enable_math_sdp,
                enable_mem_efficient_sdp,
                enable_cudnn_sdp,
            )

            enable_flash_sdp(False)
            enable_cudnn_sdp(False)
            enable_mem_efficient_sdp(False)
            enable_math_sdp(True)
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt" if master else None

    def log0(msg: str, console: bool = True):
        if not master:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 80, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if on_cuda:
        torch.cuda.manual_seed_all(args.seed)
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb_lut, ls_lut, bd_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    base_model = Belen(args).to(device).float()
    assert all((p.dtype == torch.float32 for p in base_model.parameters()))
    use_compile = os.environ.get("USE_COMPILE", "0") == "1"
    if use_compile:
        compiled_model = torch.compile(base_model, mode="default", fullgraph=False, dynamic=False)
        log0(f"torch.compile: enabled (mode=default)")
    else:
        compiled_model = base_model
        log0(f"torch.compile: disabled (eager mode)")
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True
        )
    else:
        model = compiled_model
    n_params = sum((p.numel() for p in base_model.parameters()))
    log0(f"BELEN  (LEGACY={int(LEGACY)})")
    log0(f"params: {n_params:,}  |  layers: {args.num_layers}  |  dim: {args.model_dim}")
    log0(f"heads: {args.num_heads}Q / {args.num_kv_heads}KV  |  mlp_mult: {args.mlp_mult}")
    log0(f"vocab: {args.vocab_size}  |  weight_tying: True  |  RoPE: True")
    log0(
        f"device: {device}  |  params float32, cómputo {('bfloat16 (autocast)' if on_cuda else 'float32')}  |  sdp: {args.sdp_backend}"
    )
    log0(
        f"interruptores: dendrite_layout={args.dendrite_layout}  percept_norm={args.percept_norm}  axon_mode={args.axon_mode}  resonance_write={args.resonance_write}  eval_isolated={args.eval_isolated}  ttt_restore={args.ttt_restore_after_eval}"
    )
    log0(
        f"batch: {args.train_batch_tokens} tokens/paso = {grad_accum} micro-batches × {per_micro // args.train_seq_len} secuencias × {args.train_seq_len} tokens"
    )
    if args.dendrite_mid_span >= args.train_seq_len:
        log0(
            f"AVISO: DENDRITE_MID_SPAN={args.dendrite_mid_span} >= TRAIN_SEQ_LEN={args.train_seq_len}: el campo mid ve lo mismo que el global"
        )
    log0(
        f"DendriticEncoder: field_heads={args.dendrite_field_heads}  kv_heads={args.dendrite_kv_heads}  local={args.dendrite_local_span}  mid={args.dendrite_mid_span}  n_proto={args.dendrite_n_proto}"
    )
    log0(
        f"ResonanceBuffer: {args.resonance_slots} slots × {args.model_dim}d  EMA_decay={args.resonance_ema_decay}  mode={args.resonance_mode}{('  span=' + str(args.resonance_mem_span) if args.resonance_mode == 'seq' else '')}  [no trainable params]"
    )
    log0(f"PerceptionAgent: {args.perception_heads} heads  cross-attn(Dendritic × Resonance)")
    log0(f"SomaCore: {args.num_layers} blocks — pure medium, no nucleus")
    log0(
        f"phase3={args.axon_mode}  geometric_dendrite={args.geometric_dendrite}  n_views={args.axon_n_views}  λ={args.axon_lambda}"
    )
    log0(f"seed: {args.seed}  |  world_size: {world_size}  |  grad_accum: {grad_accum}")
    groups = build_param_groups(base_model)
    log0("optimizadores: " + "  ".join((f"{g}={sum((p.numel() for p in ps)):,}" for g, ps in groups.items())))
    opt_emb = Lion(
        [{"params": groups["emb"], "lr": args.lion_lr}],
        betas=(args.lion_beta1, args.lion_beta2),
        weight_decay=args.lion_wd,
    )
    opt_muon = Muon(
        groups["matrix"],
        lr=args.matrix_lr,
        momentum=args.muon_momentum_warmup_start,
        backend_steps=args.muon_backend_steps,
    )
    opt_scalar = torch.optim.AdamW(
        [
            {"params": groups["scalar"], "lr": args.scalar_lr, "weight_decay": args.scalar_wd},
            {"params": groups["scalar_nowd"], "lr": args.scalar_lr, "weight_decay": 0.0},
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=on_cuda,
    )
    optimizers = [opt_emb, opt_muon, opt_scalar]
    if "geometry" in groups:
        optimizers.append(
            torch.optim.AdamW(
                [{"params": groups["geometry"], "lr": args.geo_lr, "weight_decay": args.geo_wd}],
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
            )
        )
    for opt in optimizers:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]

    def zero_grad_all():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    max_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    warmdown_iters = resolve_warmdown(args)
    log0(
        f"LR: warmdown lineal en los últimos {warmdown_iters} de {args.iterations} pasos"
        + (
            f"  |  límite de reloj {args.max_wallclock_seconds:.0f}s"
            if max_ms
            else "  |  sin límite de reloj"
        )
    )

    def lr_mul(step: int, elapsed_ms: float) -> float:
        return lr_schedule(step, elapsed_ms, args.iterations, warmdown_iters, args.max_wallclock_seconds)

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    if args.ttt_enabled and base_model.pfv_enabled:
        ttt_trainer = TTTIsoTrainer(
            base_model=base_model,
            lr=args.ttt_lr,
            inner_steps=args.ttt_inner_steps,
            beta=args.ttt_beta,
            reset_each_chunk=bool(args.ttt_reset_each_chunk),
        )
        log0(
            f"TTT-Iso: ENABLED  lr={args.ttt_lr}  K={args.ttt_inner_steps}  β={args.ttt_beta}  reset={args.ttt_reset_each_chunk}  warmup_chunks={args.ttt_warmup_chunks}"
        )
    else:
        ttt_trainer = None
        if args.ttt_enabled and (not base_model.pfv_enabled):
            log0(f"TTT-Iso: DISABLED (requiere PFV_MONITOR=1 para monitor_resonance)")
        else:
            log0(f"TTT-Iso: DISABLED (TTT_ENABLED=0)")
    if args.warmup_steps > 0:
        import copy

        init_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for _ in range(grad_accum):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
                with autocast_for(device):
                    loss = model(x, y, use_axon=True)
                (loss * grad_scale).backward()
            for o in optimizers:
                o.step()
            zero_grad_all()
            if (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup {ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_state, strict=True)
        for o, s in zip(optimizers, init_opt):
            o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    training_ms = 0.0
    step = 0
    tokens_seen = 0
    last_val = {}

    def resume_state() -> dict:
        return {
            "step": step,
            "training_ms": training_ms,
            "tokens_seen": tokens_seen,
            "optimizers": [o.state_dict() for o in optimizers],
            "loader": {"file_idx": train_loader.stream.file_idx, "pos": train_loader.stream.pos},
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if on_cuda else [],
        }

    def ckpt_extra() -> dict:
        return {
            "script": "belen",
            "run_id": args.run_id,
            "steps_done": step,
            "tokens_seen": tokens_seen,
            "last_val": last_val,
            "legacy": int(LEGACY),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    if args.resume_from:
        payload = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        if "resume" not in payload:
            raise ValueError(f"{args.resume_from} no trae estado de reanudación (se guardó sin CKPT_EVERY)")
        base_model.load_state_dict(payload["model_state_dict"], strict=True)
        rs = payload["resume"]
        for o, s in zip(optimizers, rs["optimizers"]):
            o.load_state_dict(s)
        for o in optimizers:
            for g in o.param_groups:
                g.setdefault("base_lr", g["lr"])
        step, training_ms, tokens_seen = (int(rs["step"]), float(rs["training_ms"]), int(rs["tokens_seen"]))
        st = train_loader.stream
        st.file_idx = int(rs["loader"]["file_idx"])
        st.tokens = load_data_shard(st.files[st.file_idx])
        st.pos = int(rs["loader"]["pos"])
        torch.set_rng_state(rs["torch_rng"])
        if on_cuda and rs["cuda_rng"]:
            torch.cuda.set_rng_state_all(rs["cuda_rng"])
        if 0 < step < args.soma_freeze_steps:
            for p in base_model.blocks.parameters():
                p.requires_grad_(False)
        log0(f"RESUME: continúo desde el paso {step} ({tokens_seen / 1000000.0:.1f}M tokens vistos)")
    sync()
    t0 = time.perf_counter()
    while True:
        last_step = step == args.iterations
        should_val = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_val:
            sync()
            training_ms += 1000.0 * (time.perf_counter() - t0)
            val_full = bool(last_step and args.val_final_full) or args.val_max_tokens <= 0
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum,
                val_tokens,
                bb_lut,
                ls_lut,
                bd_lut,
                ttt_trainer=ttt_trainer,
                log_fn=log0,
                full=val_full,
                base_model=base_model,
            )
            last_val = {
                "step": step,
                "val_loss": val_loss,
                "val_bpb": val_bpb,
                "scope": "todo val" if val_full else f"primeros {args.val_max_tokens} tokens de val",
            }
            resonance_filled = base_model.resonance.fill_count.item() > 0
            resonance_norm = base_model.resonance.slots.float().norm(dim=-1).mean().item()
            percept_gate_act = torch.sigmoid(base_model.perception.percept_gate).mean().item()
            axon_info = axon_status(base_model)
            resonance_info = f"  res_filled:{resonance_filled}  res_norm:{resonance_norm:.3f}  percept_gate:{percept_gate_act:.3f}"
            pfv_info = ""
            if base_model.pfv_enabled:
                pfv = base_model.pfv_snapshot()
                if pfv:
                    r = pfv["resonance"]
                    pl = pfv["plastic_local"]
                    pm = pfv["plastic_mid"]
                    pg = pfv["plastic_global"]
                    pfv_info = f"  PfV[res:P={r['P_t'].item():.3f},ΔP={r['delta_P'].item():+.3f},iso={int(r['n_isomers'].item())},ε={r['epsilon'].item():.3f}] [den:L={pl['P_t'].item():.2f}/M={pm['P_t'].item():.2f}/G={pg['P_t'].item():.2f}]"
            elapsed_s = training_ms / 1000.0
            elapsed_min = int(elapsed_s // 60)
            elapsed_sec = elapsed_s % 60
            log0(
                f"step:{step}/{args.iterations}  val_loss:{val_loss:.4f}  val_bpb:{val_bpb:.4f}  [{('val completo' if val_full else 'val parcial')}]  time:{elapsed_min}m{elapsed_sec:.1f}s  train_ms:{training_ms:.0f}{axon_info}{resonance_info}{pfv_info}"
            )
            sync()
            t0 = time.perf_counter()
        if last_step:
            break
        sync()
        elapsed_ms = training_ms + 1000.0 * (time.perf_counter() - t0)
        if max_ms is not None and elapsed_ms >= max_ms:
            log0(f"Wallclock limit reached at step {step}")
            break
        mul = lr_mul(step, elapsed_ms)
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * mul
        model.train()
        zero_grad_all()
        if step == 0 and args.soma_freeze_steps > 0:
            for p in base_model.blocks.parameters():
                p.requires_grad_(False)
            log0(f"SomaBlocks frozen for first {args.soma_freeze_steps} steps")
        if step == args.soma_freeze_steps and args.soma_freeze_steps > 0:
            for p in base_model.blocks.parameters():
                p.requires_grad_(True)
            log0(f"SomaBlocks unfrozen at step {step}")
        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_momentum_now = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups:
            g["momentum"] = muon_momentum_now
        for micro in range(grad_accum):
            if distributed:
                model.require_backward_grad_sync = micro == grad_accum - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
            with autocast_for(device):
                loss = model(x, y, use_axon=True)
            (loss * grad_scale).backward()
        for o in optimizers:
            o.step()
        zero_grad_all()
        tokens_seen += args.train_batch_tokens
        if master and args.train_log_every > 0 and (step % args.train_log_every == 0):
            sync()
            elapsed = training_ms + 1000.0 * (time.perf_counter() - t0)
            e_s = elapsed / 1000.0
            e_min = int(e_s // 60)
            e_sec = e_s % 60
            log0(
                f"step:{step}  loss:{loss.item():.4f}  lr_mul:{mul:.4f}  time:{e_min}m{e_sec:.1f}s  tok/s:{tokens_seen / max(e_s, 1e-09):.0f}"
            )
        step += 1
        if master and args.ckpt_every > 0 and (step % args.ckpt_every == 0) and (step < args.iterations):
            sync()
            training_ms += 1000.0 * (time.perf_counter() - t0)
            path = save_checkpoint(
                os.path.join(args.ckpt_dir, f"{args.run_id}_resume.pt"),
                base_model,
                ckpt_extra(),
                resume_state(),
            )
            log0(f"checkpoint: {path} (paso {step})")
            sync()
            t0 = time.perf_counter()
    sync()
    training_ms += 1000.0 * (time.perf_counter() - t0)
    total_s = training_ms / 1000.0
    total_min = int(total_s // 60)
    total_sec = total_s % 60
    log0(f"\n── Training complete ──")
    log0(f"  Total time : {total_min}m {total_sec:.1f}s  ({total_s:.1f}s)")
    log0(f"  Steps done : {step}")
    log0(f"  ms/step    : {training_ms / max(step, 1):.1f}ms")
    log0(
        f"  tokens     : {tokens_seen / 1000000.0:.1f}M  ({tokens_seen / max(total_s, 1e-09):.0f} tok/s de entrenamiento)"
    )
    RUN_SUMMARY.update(ckpt_extra())
    if master:
        raw_path = save_checkpoint(
            os.path.join(args.ckpt_dir, f"{args.run_id}_raw.pt"), base_model, ckpt_extra()
        )
        log0(f"\n── Checkpoint ──")
        log0(f"  pesos crudos : {raw_path}  ({os.path.getsize(raw_path) / 1000000.0:.0f} MB)")
    if master:
        size_info = quantize_and_compress(base_model, PACKAGE_FILES)
        log0(f"\n── Artifact Size ──")
        log0(f"  model compressed : {size_info['model_compressed_kb']:.1f} KB")
        log0(f"  code compressed  : {size_info['code_compressed_kb']:.1f} KB")
        log0(f"  TOTAL            : {size_info['total_kb']:.1f} KB")
        log0(f"  fits 16MB        : {size_info['fits_16mb']}")
        log0(f"  fits 14MB        : {size_info['fits_14mb']}")
        log0(f"  SVD tensors      : {size_info['n_svd_tensors']}")
        log0(f"  int8 tensors     : {size_info['n_int8_tensors']}")
        log0(f"\n── Axon + Molecule diagnostics ──")
        log0(f"  axon             :{axon_status(base_model)}")
        log0(f"  field_mix        : {F.softmax(base_model.dendrite.field_mix, dim=0).tolist()}")
        log0(f"\n── Resonance + Perception diagnostics ──")
        log0(f"  resonance filled : {base_model.resonance.fill_count.item() > 0}")
        log0(f"  resonance ptr    : {base_model.resonance.slot_ptr.item()}")
        slot_norms = base_model.resonance.slots.float().norm(dim=-1)
        log0(f"  slot_norm mean   : {slot_norms.mean().item():.4f}")
        log0(f"  slot_norm std    : {slot_norms.std().item():.4f}")
        percept_gate = torch.sigmoid(base_model.perception.percept_gate)
        log0(f"  percept_gate μ   : {percept_gate.mean().item():.4f}")
        log0(f"  percept_gate max : {percept_gate.max().item():.4f}")
        log0(f"\n── SomaCore diagnostics ──")
        attn_scales = [
            float(p.data.abs().mean().item())
            for n, p in base_model.blocks.named_parameters()
            if "attn_scale" in n
        ]
        mlp_scales = [
            float(p.data.abs().mean().item())
            for n, p in base_model.blocks.named_parameters()
            if "mlp_scale" in n
        ]
        log0(f"  attn_scale |μ|   : {sum(attn_scales) / max(len(attn_scales), 1):.4f}")
        log0(f"  mlp_scale  |μ|   : {sum(mlp_scales) / max(len(mlp_scales), 1):.4f}")
        if args.ablate_at_end:
            ab = ablation_report(args, base_model, val_tokens, device)
            full = ab["completo (memoria viva)"]
            log0(f"\n── Ablación de componentes (primeros {args.ablate_tokens} tokens de val, sin TTT) ──")
            for k, v in ab.items():
                log0(
                    f"  {k:34s} loss {v:.4f}   Δ {v - full:+.4f}"
                    + (
                        ""
                        if k.startswith("completo")
                        else (
                            "   (ayuda)"
                            if v - full > 0.005
                            else "   (ESTORBA)" if v - full < -0.005 else "   (no se nota)"
                        )
                    )
                )
            log0(
                "  Δ > 0: quitar el componente empeora, o sea que aporta. Es una sola semilla de entrenamiento."
            )
            RUN_SUMMARY["ablation"] = ab
            if base_model.pfv_enabled and (not distributed):
                T_ = args.train_seq_len
                sub = val_tokens[: max(1, args.ablate_tokens // T_) * T_ + 1]
                keep = (args.ttt_enabled, args.eval_isolated, args.ttt_restore_after_eval)
                args.ttt_enabled, args.eval_isolated, args.ttt_restore_after_eval = (1, 1, 1)
                l_off = eval_val(
                    args,
                    model,
                    rank,
                    world_size,
                    device,
                    grad_accum,
                    sub,
                    bb_lut,
                    ls_lut,
                    bd_lut,
                    ttt_trainer=None,
                    full=True,
                    base_model=base_model,
                )[0]
                tt = TTTIsoTrainer(
                    base_model, lr=args.ttt_lr, inner_steps=args.ttt_inner_steps, beta=args.ttt_beta
                )
                l_on = eval_val(
                    args,
                    model,
                    rank,
                    world_size,
                    device,
                    grad_accum,
                    sub,
                    bb_lut,
                    ls_lut,
                    bd_lut,
                    ttt_trainer=tt,
                    full=True,
                    base_model=base_model,
                )[0]
                args.ttt_enabled, args.eval_isolated, args.ttt_restore_after_eval = keep
                log0(
                    f"  TTT-Iso en los mismos pesos        apagado {l_off:.4f} · encendido {l_on:.4f}   Δ {l_on - l_off:+.4f}"
                    + (
                        "   (el TTT ayuda)"
                        if l_on - l_off < -0.005
                        else "   (el TTT ESTORBA)" if l_on - l_off > 0.005 else "   (no se nota)"
                    )
                )
                RUN_SUMMARY["ttt_same_weights"] = {"off": l_off, "on": l_on}
        resonance_py = distill_resonance(base_model)
        resonance_path = f"logs/resonance_distilled_{args.run_id}.py"
        with open(resonance_path, "w") as f:
            f.write(resonance_py)
        resonance_compressed_kb = len(__import__("zlib").compress(resonance_py.encode(), level=9)) / 1000
        log0(f"  resonance_distilled : {resonance_path}")
        log0(f"  resonance_compressed: {resonance_compressed_kb:.1f} KB")
    if distributed:
        dist.destroy_process_group()


def params_report() -> int:
    args = Hyperparameters()
    model = Belen(args)
    by: dict = {}
    for n, p in model.named_parameters():
        by[n.split(".")[0]] = by.get(n.split(".")[0], 0) + p.numel()
    total = sum(by.values())
    print(
        f"BELEN · {args.num_layers} capas × {args.model_dim} · {args.num_heads}Q/{args.num_kv_heads}KV · MLP ×{args.mlp_mult} · vocab {args.vocab_size}"
    )
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:>13,}  {100 * v / total:5.1f}%")
    print(f"  {'TOTAL':14s} {total:>13,}")
    g = build_param_groups(model)
    print("  optimizadores: " + "  ".join((f"{k}={sum((p.numel() for p in v)):,}" for k, v in g.items())))
    return 0
