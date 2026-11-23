"""  
|---|---|
| `__main__.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Dream it. |
"""
-----

import sys

from .train import main
from .train import params_report
from .selftest import run_selftest

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())
    if "--params" in sys.argv:
        sys.exit(params_report())
    main()
