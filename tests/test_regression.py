"""  
|---|---|
| `test_regression.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Check it. |
-----
"""


import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SMALL = {
    "VOCAB_SIZE": "64", "NUM_LAYERS": "4", "MODEL_DIM": "64", "NUM_HEADS": "4", "NUM_KV_HEADS": "4",
    "DENDRITE_FIELD_HEADS": "2,1,1", "DENDRITE_KV_HEADS": "1", "DENDRITE_LOCAL_SPAN": "4",
    "DENDRITE_MID_SPAN": "8", "PERCEPTION_HEADS": "4", "RESONANCE_SLOTS": "8", "RESONANCE_MEM_SPAN": "2",
    "TRAIN_SEQ_LEN": "16", "AXON_N_VIEWS": "4", "DEVICE": "cpu", "RUN_ID": "regress", "SEED": "7",
    "GEOMETRIC_DENDRITE": "0",
}

REFERENCE = {
    "0": {"n_params": 232727, "keys": "dbd125ec6d430509", "losses": [4.1541213989, 4.1001000404, 4.0510425568]},
    "1": {"n_params": 232727, "keys": "791ddca4cb11ae33", "losses": [4.1537551880, 4.1187295914, 4.0820918083]},
}


def measure():
    import torch
    sys.path.insert(0, str(ROOT))
    from belen.config import Hyperparameters
    from belen.model import Belen
    from belen.optim import Lion, Muon, build_param_groups

    torch.manual_seed(7)
    model = Belen(Hyperparameters()).float()
    g = torch.Generator().manual_seed(11)
    x = torch.randint(0, 64, (2, 16), generator=g)
    y = torch.randint(0, 64, (2, 16), generator=g)
    keys = hashlib.sha256(json.dumps([(k, list(v.shape)) for k, v in model.state_dict().items()]).encode()).hexdigest()[:16]
    groups = build_param_groups(model)
    opts = [
        Lion(groups["emb"], lr=1e-3),
        Muon(groups["matrix"], lr=0.02, momentum=0.9, backend_steps=5),
        torch.optim.AdamW(groups["scalar"] + groups["scalar_nowd"], lr=1e-3),
    ]
    losses = []
    model.train()
    for step in range(3):
        for o in opts:
            o.zero_grad(set_to_none=True)
        torch.manual_seed(100 + step)
        loss = model(x, y)
        loss.backward()
        losses.append(float(loss.detach()))
        for o in opts:
            o.step()
    return {"n_params": sum(p.numel() for p in model.parameters()), "keys": keys, "losses": losses}


def main():
    if os.environ.get("_BELEN_REGRESSION_CHILD"):
        print(json.dumps(measure()))
        return 0
    worst = 0.0
    for legacy, ref in REFERENCE.items():
        env = {**os.environ, **SMALL, "LEGACY": legacy, "_BELEN_REGRESSION_CHILD": "1"}
        env.pop("AXON_MODE", None)
        out = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=True)
        got = json.loads(out.stdout.strip().splitlines()[-1])
        assert got["n_params"] == ref["n_params"], (legacy, got["n_params"])
        assert got["keys"] == ref["keys"], (legacy, "state_dict layout changed")
        diff = max(abs(a - b) for a, b in zip(got["losses"], ref["losses"]))
        worst = max(worst, diff)
        assert diff < 1e-5, (legacy, got["losses"], ref["losses"])
        print(f"  [ok] LEGACY={legacy}: parameters, layout and 3 training losses match the audited reference (max diff {diff:.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
