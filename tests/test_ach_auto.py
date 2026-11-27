"""
|---|---|
| `test_ach_auto.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Store it. |
-----
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ach_and_geometry import SMALL  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build(auto: bool, **extra):
    os.environ.update(SMALL)
    os.environ.update({"AXON_MODE": "ach", "GEOMETRIC_DENDRITE": "0", "ACH_AUTO": "1" if auto else "0"})
    os.environ.update({k: str(v) for k, v in extra.items()})
    for m in [k for k in sys.modules if k == "belen" or k.startswith("belen.")]:
        del sys.modules[m]
    import torch
    from belen.config import Hyperparameters
    from belen.model import Belen
    from belen.optim import build_param_groups

    torch.manual_seed(3)
    return torch, Belen(Hyperparameters()).float(), build_param_groups


def batch(torch):
    g = torch.Generator().manual_seed(5)
    return torch.randint(0, 64, (2, 16), generator=g), torch.randint(0, 64, (2, 16), generator=g)


def main() -> int:
    # 1. Apagada por defecto: mismo state_dict que el modelo auditado.
    torch, off, _ = build(False)
    keys_off = list(off.state_dict().keys())
    torch, on, groups_fn = build(True, ACH_AUTO_PATIENCE=1)
    extra = [k for k in on.state_dict() if k not in keys_off]
    assert all(k.startswith("axon.") for k in extra), extra
    assert "axon.store_log" in extra
    print(f"  [ok] ACH_AUTO=0 no cambia el layout · ACH_AUTO=1 agrega {len(extra)} entradas, todas en axon.")

    # 1b. Con ACH_AUTO=1 todos los parámetros de la compuerta reciben gradiente.
    on.train()
    x, y = batch(torch)
    on(x, y).backward()
    for n in ("theta", "rho_k", "w_supp", "conductance_base", "store_log"):
        p = getattr(on.axon, n)
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, f"ach_auto {n} sin gradiente"
    on.zero_grad(set_to_none=True)
    torch, on, groups_fn = build(True, ACH_AUTO_PATIENCE=1)
    print("  [ok] ACH_AUTO=1: theta, rho_k, w_supp, conductance_base y store_log reciben gradiente")

    # 2. Antes de consolidar, la compuerta automática es idéntica a la original.
    x, y = batch(torch)
    sd = {k: v for k, v in on.state_dict().items() if k in keys_off}
    off.load_state_dict(sd, strict=True)
    on.eval(), off.eval()
    on.resonance.frozen = off.resonance.frozen = True
    with torch.no_grad():
        d0 = float((on(x, y) - off(x, y)).abs())
    assert d0 == 0.0, d0
    print(f"  [ok] sin consolidar: misma loss que la compuerta original (Δ {d0:.1e})")

    # 3. Forzar el borde: θ y k pegados al lado que apaga la compuerta, como en la corrida real.
    g = on.axon
    with torch.no_grad():
        g.theta.fill_(12.0)
        g.rho_k.fill_(-12.0)
        g.conductance_base.fill_(-4.0)
    sat0 = g.saturation()
    assert max(sat0.values()) < 0.02, sat0
    on.train()
    on.resonance.frozen = True
    loss = on(x, y)
    d = g.diagnostics()
    assert int(d["n_consolidations"]) == 1, d
    sat1 = g.saturation()
    assert sat1["k"] > 0.04 and sat1["theta"] > 0.04 and abs(d["base_mean"] - 1.0) < 1e-6, sat1
    assert d["last_jump"] < 0.05, d["last_jump"]
    assert 0.85 < d["store_mean"] < 0.95, d["store_mean"]
    print(
        f"  [ok] consolidación: borde k/θ/c {sat0['k']:.3f}/{sat0['theta']:.3f}/{sat0['base']:.3f}"
        f" -> {sat1['k']:.2f}/{sat1['theta']:.2f}/1.00 · almacén μ {d['store_mean']:.3f}"
        f" · salto de máscara {d['last_jump']:.1e}"
    )

    # 4. El almacén recibe gradiente y va sin weight decay.
    loss.backward()
    assert g.store_log.grad is not None and float(g.store_log.grad.abs().sum()) > 0.0
    groups = groups_fn(on)
    assert any(p is g.store_log for p in groups["scalar_nowd"])
    assert not any(p is g.store_log for p in groups["scalar"])
    print("  [ok] almacén: recibe gradiente, en scalar_nowd")

    # 5. Tope de consolidaciones y ablación sin dinámica.
    with torch.no_grad():
        for _ in range(12):
            g.theta.fill_(12.0)
            g.rho_k.fill_(-12.0)
            on(x, y)
    assert int(g.n_consolidations) == g.auto_max, int(g.n_consolidations)
    on.eval()
    g.static_rho = 0.2
    with torch.no_grad():
        a, b = on(x, y), on(x, y)
    g.static_rho = None
    assert float((a - b).abs()) == 0.0 and torch.isfinite(a)
    print(f"  [ok] tope de {g.auto_max} consolidaciones · ρ fijo determinista en eval")

    # 6. Ida y vuelta del checkpoint con el estado del almacén.
    torch2, again, _ = build(True)
    again.load_state_dict(on.state_dict(), strict=True)
    assert int(again.axon.n_consolidations) == g.auto_max
    assert torch.equal(again.axon.store_log, g.store_log)
    print("  [ok] checkpoint: almacén y contador de consolidaciones sobreviven")
    print("Honestidad: modelo diminuto. Valida la MECÁNICA de la consolidación, no si mejora a BELEN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
