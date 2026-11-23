"""  
|---|---|
| `compress.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Xploit it. |
"""
-----


from __future__ import annotations

import io
import zlib
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn


def distill_resonance(model: nn.Module) -> str:
    slots = model.resonance.get_slots().detach().float().cpu().numpy()
    K, D = slots.shape
    U, S, Vt = np.linalg.svd(slots, full_matrices=False)
    top_k = min(16, K)
    U_k = U[:, :top_k]
    S_k = S[:top_k]
    Vt_k = Vt[:top_k, :]
    var_explained = float((S_k**2).sum() / (S**2).sum())
    filled = bool(model.resonance.fill_count.item() > 0)
    ptr = int(model.resonance.slot_ptr.item())
    lines = [
        "# ResonanceBuffer destilado — snapshot del flujo final",
        f"# Slots: {K}×{D}  |  EMA decay: {model.resonance.ema_decay}",
        f"# Buffer filled: {filled}  |  Last ptr: {ptr}",
        f"# Varianza explicada con top_{top_k}: {var_explained:.4f}",
        "import numpy as np",
        "",
        f"RESONANCE_N_SLOTS  = {K}",
        f"RESONANCE_DIM      = {D}",
        f"RESONANCE_TOP_K    = {top_k}",
        "",
        "# Bases principales del flujo (top_k × D)",
        f"RESONANCE_BASES  = np.array({Vt_k.tolist()}, dtype=np.float32)",
        f"RESONANCE_SCALES = np.array({S_k.tolist()}, dtype=np.float32)",
        f"RESONANCE_COEFS  = np.array({U_k.tolist()}, dtype=np.float32)",
        "",
        "def reconstruct_slots():",
        '    """Reconstruye los slots de resonancia desde constantes destiladas."""',
        "    return (RESONANCE_COEFS * RESONANCE_SCALES[None, :]) @ RESONANCE_BASES",
    ]
    return "\n".join(lines)


def svd_compress_tensor(t32: torch.Tensor, variance_target: float = 0.9, rank_cap: int = 96) -> dict:
    m, n = t32.shape
    try:
        U, S, Vt = torch.linalg.svd(t32, full_matrices=False)
    except Exception:
        return None
    var_total = (S**2).sum()
    var_cum = (S**2).cumsum(0)
    rank_needed = int((var_cum / var_total >= variance_target).nonzero()[0].item()) + 1
    rank = min(rank_needed, rank_cap, min(m, n))
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vt_k = Vt[:rank, :]
    var_explained = float((S_k**2).sum() / var_total)

    def _to_int8(x):
        clip = float(torch.quantile(x.abs().flatten(), 0.9999).item())
        scale = torch.tensor(clip / 127.0 if clip > 0 else 1.0, dtype=torch.float32)
        if x.ndim == 2:
            clip_r = torch.quantile(x.abs(), 0.9999, dim=1)
            clipped = torch.clamp(x, -clip_r[:, None], clip_r[:, None])
            scale_r = (clip_r / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / scale_r[:, None]), -127, 127).to(torch.int8)
            return (q, scale_r.to(torch.float16))
        else:
            q = torch.clamp(torch.round(x.clamp(-clip, clip) / scale), -127, 127).to(torch.int8)
            return (q, scale.to(torch.float16))

    U_q, U_s = _to_int8(U_k)
    Vt_q, Vt_s = _to_int8(Vt_k)
    return {
        "type": "svd",
        "shape": (m, n),
        "rank": rank,
        "var_explained": var_explained,
        "U_q": U_q,
        "U_s": U_s,
        "S_k": S_k.to(torch.float16),
        "Vt_q": Vt_q,
        "Vt_s": Vt_s,
    }


def quantize_and_compress(
    model: nn.Module,
    code_path,
    svd_threshold: int = 200000,
    variance_target: float = 0.9,
    rank_cap: int = 96,
) -> dict:
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    quant = {}
    scales = {}
    svd_data = {}
    passthrough = {}
    stats = {"svd": 0, "int8": 0, "fp16": 0, "total_orig_kb": 0.0}
    for name, t in sd.items():
        orig_kb = t.numel() * t.element_size() / 1000
        stats["total_orig_kb"] += orig_kb
        if not t.is_floating_point() or t.numel() <= 65536:
            passthrough[name] = t.to(torch.float16) if t.is_floating_point() else t
            stats["fp16"] += 1
            continue
        t32 = t.float()
        if t32.ndim == 2 and t32.numel() > svd_threshold:
            result = svd_compress_tensor(t32, variance_target, rank_cap)
            if result is not None:
                svd_data[name] = result
                stats["svd"] += 1
                continue
        if t32.ndim == 2:
            clip = torch.quantile(t32.abs(), 0.9999, dim=1)
            clipped = torch.clamp(t32, -clip[:, None], clip[:, None])
            scale = (clip / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
            quant[name] = q
            scales[name] = scale.to(torch.float16)
        else:
            clip = float(torch.quantile(t32.abs().flatten(), 0.9999).item())
            scale = torch.tensor(clip / 127.0 if clip > 0 else 1.0)
            q = torch.clamp(torch.round(t32.clamp(-clip, clip) / scale), -127, 127).to(torch.int8)
            quant[name] = q
            scales[name] = scale
        stats["int8"] += 1
    obj = {
        "quantized": quant,
        "scales": scales,
        "svd": svd_data,
        "passthrough": passthrough,
        "compression": {
            "svd_threshold": svd_threshold,
            "variance_target": variance_target,
            "rank_cap": rank_cap,
            "n_svd": stats["svd"],
            "n_int8": stats["int8"],
            "n_fp16": stats["fp16"],
        },
    }
    buf = io.BytesIO()
    torch.save(obj, buf)
    model_bytes = zlib.compress(buf.getvalue(), level=9)
    paths = [code_path] if isinstance(code_path, (str, Path)) else list(code_path)
    code_bytes = b"".join(Path(p).read_bytes() for p in paths)
    code_compressed = zlib.compress(code_bytes, level=9)
    total_bytes = len(model_bytes) + len(code_compressed)
    return {
        "model_compressed_kb": len(model_bytes) / 1000,
        "code_compressed_kb": len(code_compressed) / 1000,
        "total_kb": total_bytes / 1000,
        "fits_16mb": total_bytes < 16000000,
        "fits_14mb": total_bytes < 14000000,
        "n_svd_tensors": stats["svd"],
        "n_int8_tensors": stats["int8"],
        "model_bytes": model_bytes,
    }
