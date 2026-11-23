"""  
|---|---|
| `calibrate_geometry.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Tune it. |
"""
-----


import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Empirical L/R calibration of the geometric dendrite")
    ap.add_argument("samples", type=Path, help=".pt dict with neutral, left and right tensors, (N, 3, D) or (N, D)")
    ap.add_argument("--ckpt", type=Path, help="raw BELEN checkpoint to calibrate (optional)")
    ap.add_argument("--output", type=Path, default=Path("geometry_calibrated.pt"))
    ns = ap.parse_args()
    os.environ["GEOMETRIC_DENDRITE"] = "1"
    sys.path.insert(0, str(ROOT))
    import torch
    from belen.config import Hyperparameters
    from belen.model import Belen

    args = Hyperparameters()
    if ns.ckpt:
        payload = torch.load(ns.ckpt, map_location="cpu", weights_only=True)
        for k, v in payload["hparams"].items():
            setattr(args, k, v)
        if not int(getattr(args, "geometric_dendrite", 0)):
            raise SystemExit("the checkpoint was trained with GEOMETRIC_DENDRITE=0")
        model = Belen(args).cpu()
        model.load_state_dict(payload["model_state_dict"], strict=True)
    else:
        model = Belen(args).cpu()
    data = torch.load(ns.samples, map_location="cpu")
    snapshot = model.dendrite.calibrate_geometry(data["neutral"], data["left"], data["right"])
    torch.save({"geometry_state_dict": model.dendrite.geometry.state_dict(), "geometry_snapshot": snapshot}, ns.output)
    print(snapshot)
    print(f"saved: {ns.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
