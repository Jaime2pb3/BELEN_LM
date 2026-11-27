"""  
|---|---|
| `config.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Adapt it. |
-----
"""

from __future__ import annotations

import os
import uuid

LEGACY = os.environ.get("LEGACY", "0") == "1"


def switch(name: str, new: str, old: str) -> str:
    return os.environ.get(name, old if LEGACY else new)


class Hyperparameters:
    device_pref = os.environ.get("DEVICE", "auto")
    grad_accum = int(os.environ.get("GRAD_ACCUM", 0))
    sdp_backend = os.environ.get("SDP_BACKEND", "auto")
    ckpt_every = int(os.environ.get("CKPT_EVERY", 0))
    ckpt_dir = os.environ.get("CKPT_DIR", "logs")
    resume_from = os.environ.get("RESUME", "")
    _here = os.path.dirname(os.path.abspath(__file__))
    _base = _here
    for _ in range(3):
        if os.path.isdir(os.path.join(_base, "data")):
            break
        _base = os.path.dirname(_base)
    data_path = os.environ.get("DATA_PATH", os.path.join(_base, "data", "datasets", "fineweb10B_sp1024"))
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", os.path.join(_base, "data", "tokenizers", "fineweb_1024_bpe.model")
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", os.environ.get("TRAIN_BATCH_TOKENS", 65536)))
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 2097152))
    val_final_full = int(os.environ.get("VAL_FINAL_FULL", 1))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 250))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    iterations = int(os.environ.get("ITERATIONS", 1000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", -1))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 10))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 65536))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0.0))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 14))
    model_dim = int(os.environ.get("MODEL_DIM", 768))
    num_heads = int(os.environ.get("NUM_HEADS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 12))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    dendrite_field_heads = [int(v) for v in os.environ.get("DENDRITE_FIELD_HEADS", "8,4,4").split(",")]
    dendrite_local_span = int(os.environ.get("DENDRITE_LOCAL_SPAN", 64))
    dendrite_mid_span = int(os.environ.get("DENDRITE_MID_SPAN", 256))
    dendrite_kv_heads = int(os.environ.get("DENDRITE_KV_HEADS", 4))
    dendrite_n_proto = int(os.environ.get("DENDRITE_N_PROTO", 16))
    dendrite_layout = switch("DENDRITE_LAYOUT", "concat", "overlap")
    axon_n_views = int(os.environ.get("AXON_N_VIEWS", 4))
    axon_t_min = float(os.environ.get("AXON_T_MIN", 0.2))
    axon_t_max = float(os.environ.get("AXON_T_MAX", 0.6))
    axon_lambda = float(os.environ.get("AXON_LAMBDA", 0.12))
    axon_stress_threshold = float(os.environ.get("AXON_STRESS_THRESHOLD", 0.3))
    axon_mode = switch("AXON_MODE", "views", "noise")
    axon_regime_momentum = float(os.environ.get("AXON_REGIME_MOMENTUM", 0.9))
    ach_theta = float(os.environ.get("ACH_THETA", 0.0))
    ach_rho_k = float(os.environ.get("ACH_RHO_K", 2.0))
    ach_lambda = float(os.environ.get("ACH_LAMBDA", 0.12))
    ach_base_scale = float(os.environ.get("ACH_BASE_SCALE", 0.10))
    ach_first_block = int(os.environ.get("ACH_FIRST_BLOCK", 2))
    ach_ema_momentum = float(os.environ.get("ACH_EMA_MOMENTUM", 0.99))
    ach_auto = int(os.environ.get("ACH_AUTO", 0))
    ach_auto_eps = float(os.environ.get("ACH_AUTO_EPS", 0.02))
    ach_auto_patience = int(os.environ.get("ACH_AUTO_PATIENCE", 200))
    ach_auto_max = int(os.environ.get("ACH_AUTO_MAX", 4))
    geometric_dendrite = int(os.environ.get("GEOMETRIC_DENDRITE", 0))
    geo_center_decay = float(os.environ.get("GEO_CENTER_DECAY", 0.99))
    geo_axis_init_cos = float(os.environ.get("GEO_AXIS_INIT_COS", 0.35))
    geo_gamma_init = float(os.environ.get("GEO_GAMMA_INIT", 0.08))
    geo_metric_max = float(os.environ.get("GEO_METRIC_MAX", 0.30))
    geo_transport_max = float(os.environ.get("GEO_TRANSPORT_MAX", 0.12))
    geo_tau_max = float(os.environ.get("GEO_TAU_MAX", 0.35))
    geo_quantize = int(os.environ.get("GEO_QUANTIZE", 1))
    geo_quant_step_init = float(os.environ.get("GEO_QUANT_STEP_INIT", 0.25))
    geo_axis_max_abs_cos = float(os.environ.get("GEO_AXIS_MAX_ABS_COS", 0.92))
    geo_cross_field_target = float(os.environ.get("GEO_CROSS_FIELD_TARGET", 0.10))
    geo_aux_scale = float(os.environ.get("GEO_AUX_SCALE", 0.01))
    geo_eps = float(os.environ.get("GEO_EPS", 1e-5))
    geo_lr = float(os.environ.get("GEO_LR", 1e-3))
    geo_wd = float(os.environ.get("GEO_WD", 1e-4))
    resonance_slots = int(os.environ.get("RESONANCE_SLOTS", 64))
    resonance_ema_decay = float(switch("RESONANCE_EMA_DECAY", "0.0", "0.95"))
    resonance_write = switch("RESONANCE_WRITE", "sample", "mean")
    resonance_mode = os.environ.get("RESONANCE_MODE", "seq")
    resonance_mem_span = int(os.environ.get("RESONANCE_MEM_SPAN", 8))
    perception_heads = int(os.environ.get("PERCEPTION_HEADS", 12))
    percept_norm = int(switch("PERCEPT_NORM", "0", "1"))
    ttt_enabled = int(os.environ.get("TTT_ENABLED", 1))
    ttt_inner_steps = int(os.environ.get("TTT_INNER_STEPS", 3))
    ttt_lr = float(os.environ.get("TTT_LR", 0.0005))
    ttt_beta = float(os.environ.get("TTT_BETA", 0.1))
    ttt_reset_each_chunk = int(os.environ.get("TTT_RESET_EACH_CHUNK", 0))
    ttt_warmup_chunks = int(os.environ.get("TTT_WARMUP_CHUNKS", 2))
    ablate_at_end = int(os.environ.get("ABLATE_AT_END", 1))
    ablate_tokens = int(os.environ.get("ABLATE_TOKENS", 524288))
    ttt_restore_after_eval = int(switch("TTT_RESTORE_AFTER_EVAL", "1", "0"))
    eval_isolated = int(switch("EVAL_ISOLATED", "1", "0"))
    soma_freeze_steps = int(os.environ.get("SOMA_FREEZE_STEPS", 100))
    skip_init = float(os.environ.get("SKIP_INIT", 1.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    scalar_wd = float(os.environ.get("SCALAR_WD", 0.01))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 10))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 2500))
    lion_lr = float(os.environ.get("LION_LR", 0.001))
    lion_beta1 = float(os.environ.get("LION_BETA1", 0.9))
    lion_beta2 = float(os.environ.get("LION_BETA2", 0.99))
    lion_wd = float(os.environ.get("LION_WD", 0.0))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-08))


def hparams_dict(obj=Hyperparameters) -> dict:
    simple, out = ((str, int, float, bool, type(None)), {})
    for k in dir(obj):
        if k.startswith("_"):
            continue
        v = getattr(obj, k)
        if isinstance(v, simple):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all((isinstance(x, simple) for x in v)):
            out[k] = list(v)
    return out
