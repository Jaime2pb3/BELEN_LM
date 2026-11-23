"""  
|---|---|
| `train_belen.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Dreams too much. |
-----
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
for _name in [k for k in sys.modules if k == "belen" or k.startswith("belen.")]:
    del sys.modules[_name]

from belen import train as train_module
from belen.checkpoint import CKPT_FORMAT
from belen.compress import quantize_and_compress
from belen.config import Hyperparameters
from belen.model import Belen
from belen.selftest import run_selftest
from belen.train import RUN_SUMMARY, main, params_report

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())
    if "--params" in sys.argv:
        sys.exit(params_report())
    main()
