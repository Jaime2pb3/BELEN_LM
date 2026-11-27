"""  
|---|---|
| `ab_run.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Run it. |
-----
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    # ACH_AUTO va explícito en cada brazo: el entorno se hereda y un brazo no debe contaminar a otro.
    "base": {"AXON_MODE": "views", "GEOMETRIC_DENDRITE": "0", "ACH_AUTO": "0"},
    "geometry": {"AXON_MODE": "views", "GEOMETRIC_DENDRITE": "1", "ACH_AUTO": "0"},
    "ach": {"AXON_MODE": "ach", "GEOMETRIC_DENDRITE": "0", "ACH_AUTO": "0"},
    "ach_geometry": {"AXON_MODE": "ach", "GEOMETRIC_DENDRITE": "1", "ACH_AUTO": "0"},
    "ach_auto": {"AXON_MODE": "ach", "GEOMETRIC_DENDRITE": "0", "ACH_AUTO": "1"},
    "legacy": {"LEGACY": "1"},
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired runs: same seed, data and schedule, one factor at a time")
    ap.add_argument("--arms", nargs="+", default=["base", "geometry", "ach", "ach_geometry"], choices=sorted(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--prefix", default="ab")
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args()
    for seed in ns.seeds:
        for arm in ns.arms:
            run_id = f"{ns.prefix}_{arm}_s{seed}"
            env = {**os.environ, "TTT_ENABLED": os.environ.get("TTT_ENABLED", "0"), "SEED": str(seed), "RUN_ID": run_id}
            env.update(ARMS[arm])
            cmd = [sys.executable, str(ROOT / "train_belen.py")]
            print(f"=== {run_id}: " + " ".join(f"{k}={v}" for k, v in ARMS[arm].items()), flush=True)
            if ns.dry_run:
                continue
            rc = subprocess.call(cmd, env=env)
            if rc:
                print(f"{run_id} failed with exit code {rc}")
                return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
