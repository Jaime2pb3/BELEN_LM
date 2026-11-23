"""  
|---|---|
| `test_ach_and_geometry.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Secure it. |
"""
-----

import os
import sys
from pathlib import Path

SMALL = {
    "VOCAB_SIZE": "64",
    "NUM_LAYERS": "5",
    "MODEL_DIM": "64",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "4",
    "DENDRITE_FIELD_HEADS": "2,1,1",
    "DENDRITE_KV_HEADS": "1",
    "DENDRITE_LOCAL_SPAN": "4",
    "DENDRITE_MID_SPAN": "8",
    "PERCEPTION_HEADS": "4",
    "RESONANCE_SLOTS": "8",
    "RESONANCE_MEM_SPAN": "2",
    "TRAIN_SEQ_LEN": "16",
    "AXON_N_VIEWS": "4",
    "DEVICE": "cpu",
    "RUN_ID": "test",
    "SEED": "3",
}


def build(axon_mode, geometric):
    os.environ.update(SMALL)
    os.environ["AXON_MODE"] = axon_mode
    os.environ["GEOMETRIC_DENDRITE"] = "1" if geometric else "0"
    for m in [k for k in sys.modules if k == "belen" or k.startswith("belen.")]:
        del sys.modules[m]
    import torch
    from belen.config import Hyperparameters
    from belen.model import Belen
    from belen.optim import build_param_groups

    torch.manual_seed(3)
    model = Belen(Hyperparameters()).float()
    return torch, model, build_param_groups


def check(axon_mode, geometric):
    torch, model, groups_fn = build(axon_mode, geometric)
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, 64, (2, 16), generator=g)
    y = torch.randint(0, 64, (2, 16), generator=g)
    model.train()
    loss = model(x, y)
    loss.backward()
    assert torch.isfinite(loss), loss
    groups = groups_fn(model)
    names = {id(p): n for n, p in model.named_parameters()}
    if geometric:
        assert "geometry" in groups and groups["geometry"], "geometry params have no optimizer group"
        assert not any(names[id(p)].startswith("dendrite.geometry.") for p in groups["matrix"])
        for p in groups["geometry"]:
            assert p.grad is not None and torch.isfinite(p.grad).all(), names[id(p)]
        dead = [names[id(p)] for p in groups["geometry"] if float(p.grad.abs().sum()) == 0.0]
        assert len(dead) < len(groups["geometry"]), "no geometry parameter received gradient"
    else:
        assert "geometry" not in groups
    if axon_mode == "ach":
        for n in ("theta", "rho_k", "w_supp", "conductance_base"):
            p = getattr(model.axon, n)
            assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, f"ach {n} has no gradient"
        d = model.axon.diagnostics()
        assert all(isinstance(v, float) for v in d.values())
        assert -3.0 < d["theta"] < 3.0 and 0.5 < d["rho_k"] < 8.0
    for _ in range(3):
        model(x, y)
    model.eval()
    model.resonance.frozen = True
    with torch.no_grad():
        a = model.get_logits(x)
        b = model.get_logits(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 64
        c = model.get_logits(x2)
    a = a.reshape(2, 16, -1)
    b = b.reshape(2, 16, -1)
    c = c.reshape(2, 16, -1)
    det = float((a - b).abs().max())
    causal = float((a[:, :-1] - c[:, :-1]).abs().max())
    if axon_mode != "noise":
        assert det == 0.0, f"eval not deterministic: {det}"
        assert causal == 0.0, f"eval not causal: {causal}"
    sd = model.state_dict()
    torch2, model2, _ = build(axon_mode, geometric)
    model2.load_state_dict(sd, strict=True)
    return float(loss), det, causal


def geometry_invariants():
    import torch
    from belen.geometry import GeometricFrameBank

    frame = GeometricFrameBank(
        dim=16,
        head_dim=4,
        enabled=True,
        center_decay=0.9,
        axis_init_cos=0.35,
        gamma_init=0.08,
        metric_max=0.3,
        transport_max=0.12,
        tau_max=1.0,
        quantize=True,
        quant_step_init=0.25,
        axis_max_abs_cos=0.92,
        cross_field_target=0.10,
        eps=1e-5,
    ).float()
    assert all(v > 0.05 for v in frame.snapshot()["gram_det"])
    frame.center.zero_()
    frame.center_initialized.fill_(True)
    h = torch.randn(3, 5, 16)
    projected, residual = frame.project(h, 0)
    plane_h, _ = frame._dual_projection(h, 0, head_space=False)
    plane_p, _ = frame._dual_projection(projected, 0, head_space=False)
    assert torch.allclose(h - plane_h, projected - plane_p, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(residual).all()
    frame.eta_raw.data.fill_(20.0)
    out0, _ = frame.geodesic_transport(h, projected, 0, torch.zeros(3, 5))
    out1, _ = frame.geodesic_transport(h, projected, 0, torch.ones(3, 5))
    assert torch.equal(out0, h)
    assert torch.allclose(out1, projected, atol=2e-5, rtol=2e-5)
    neutral = torch.randn(12, 3, 16) * 0.05
    left = neutral + 0.4 * torch.nn.functional.normalize(torch.randn(1, 3, 16), dim=-1)
    right = neutral + 0.4 * torch.nn.functional.normalize(torch.randn(1, 3, 16), dim=-1)
    cal = frame.calibrate(neutral, left, right)
    assert all(cal["center_initialized"]) and all(v > 1e-3 for v in cal["gram_det"])


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    for mode in ("views", "noise", "ach"):
        for geo in (False, True):
            loss, det, causal = check(mode, geo)
            print(
                f"  [ok] phase3={mode:5s} geometry={int(geo)}  loss {loss:.4f}  eval det {det:.1e}  causal {causal:.1e}"
            )
    geometry_invariants()
    print("  [ok] geometry invariants: gram, orthogonal complement, geodesic endpoints, calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
