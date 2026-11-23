"""  
|---|---|
| `checkpoint.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Enjoy it. |
"""
-----


from __future__ import annotations

import os
import torch
from torch import Tensor, nn

from .config import hparams_dict

CKPT_FORMAT = "belen-raw-state-dict-v1"


def save_checkpoint(
    path: str, model: nn.Module, extra: dict | None = None, resume: dict | None = None
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "format": CKPT_FORMAT,
        "hparams": hparams_dict(),
        "extra": extra or {},
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if resume is not None:
        payload["resume"] = resume
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path
