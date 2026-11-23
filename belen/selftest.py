"""  
|---|---|
| `selftest.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Test it. |
-----
"""


from __future__ import annotations

import math
import os
import torch
import torch.nn.functional as F

from .axon import AxonProjector
from .checkpoint import CKPT_FORMAT, save_checkpoint
from .compress import distill_resonance
from .config import Hyperparameters
from .evaluate import eval_val
from .model import Belen
from .optim import Lion, Muon, build_param_groups, lr_schedule
from .quarks import QuarkBundle, TopQuark, quark_views
from .ttt_iso import TTTIsoTrainer


def run_selftest() -> int:
    import tempfile

    tests: list = []
    torch.manual_seed(0)

    def tiny(**over):
        a = Hyperparameters()
        base = dict(
            model_dim=64,
            num_layers=4,
            num_heads=4,
            num_kv_heads=4,
            mlp_mult=2,
            vocab_size=96,
            dendrite_field_heads=[4, 2, 2],
            dendrite_kv_heads=2,
            dendrite_local_span=4,
            dendrite_mid_span=8,
            resonance_slots=6,
            resonance_mem_span=2,
            perception_heads=4,
            train_seq_len=16,
            axon_n_views=4,
            dendrite_n_proto=8,
            dendrite_layout="concat",
            percept_norm=0,
            axon_mode="views",
            resonance_write="sample",
            resonance_ema_decay=0.0,
            resonance_mode="seq",
            eval_isolated=1,
            ttt_restore_after_eval=1,
            ttt_enabled=1,
            ttt_warmup_chunks=0,
            val_batch_size=64,
            val_max_tokens=0,
        )
        base.update(over)
        for k, v in base.items():
            setattr(a, k, v)
        return (a, Belen(a).float())

    qb, worst = (QuarkBundle(), 0.0)
    for trial in range(40):
        D = 48
        base_v = torch.randn(1, D)
        phis = [F.layer_norm(base_v + torch.randn(1, D) * s, (D,)) for s in (0.05, 0.3, 0.8, 1.5)]
        if trial % 4 == 3:
            phis[1] = phis[0].clone()
        S_ref, dd_ref = qb.observe(phis)
        S_new, dd_new = quark_views(phis)
        worst = max(worst, abs(S_ref - float(S_new)), float((dd_ref - dd_new.squeeze(0)).abs().max()))
    batch = [torch.randn(3, 5, 48) for _ in range(4)]
    Sb, db = quark_views([F.layer_norm(v, (48,)) for v in batch])
    one = quark_views([F.layer_norm(v[1:2, 2], (48,)) for v in batch])[0]
    tests.append(
        (
            f"quarks por token == QuarkBundle original (Δ máx {worst:.1e}), y en lote da lo mismo que suelto",
            worst < 0.0002
            and Sb.shape == (3, 5)
            and (db.shape == (3, 5, 48))
            and (abs(float(Sb[1, 2]) - float(one)) < 1e-05),
        )
    )
    phis = [torch.randn(1, 40) for _ in range(4)]
    st = torch.stack([p.squeeze(0) for p in phis])
    e = torch.linalg.eigvalsh(torch.cov(st.T)).abs()
    tests.append(
        (
            "TopQuark: Gram m×m == covarianza D×D",
            abs(TopQuark().compute(phis) - float(e.max() / e.sum())) < 1e-05,
        )
    )

    def field_dims(layout):
        a, m = tiny(dendrite_layout=layout)
        caps, den = ([], m.dendrite)
        for pl in (den.plastic_local, den.plastic_mid, den.plastic_global):
            pl.register_forward_pre_hook(
                lambda _m, inp: caps.append((inp[0].abs().sum((0, 1)) > 0).nonzero().flatten())
            )
        with torch.no_grad():
            den(torch.randn(2, 16, 64))
        return [(int(c.min()), int(c.max())) for c in caps]

    fc, fo = (field_dims("concat"), field_dims("overlap"))
    tests.append(
        (
            f"dendrita: concat {fc} cubre 0..63 sin encimarse · overlap {fo} (original)",
            fc == [(0, 31), (32, 47), (48, 63)] and fo == [(0, 31), (0, 15), (0, 15)],
        )
    )
    a, m = tiny()
    m.eval()
    x = torch.randint(0, 96, (2, 16))
    with torch.no_grad():
        for _ in range(3):
            m(torch.randint(0, 96, (2, 16)), torch.randint(0, 96, (2, 16)))
        m.train()
        m(x, x)
        m.eval()
        m.resonance.frozen = True
        la = m.get_logits(x).reshape(2, 16, -1)
        lb = m.get_logits(x).reshape(2, 16, -1)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 5) % 96
        lc = m.get_logits(x2).reshape(2, 16, -1)
        solo = m.get_logits(x[:1]).reshape(1, 16, -1)
    leak = float((la - lc)[:, :-1].abs().max())
    tests.append(
        (
            f"axón views: mismo input dos veces Δ {float((la - lb).abs().max()):.1e} · cambiar el último token mueve las posiciones previas {leak:.1e} · una secuencia sola == la misma dentro del batch Δ {float((la[:1] - solo).abs().max()):.1e}",
            float((la - lb).abs().max()) == 0.0
            and leak == 0.0
            and (float((la[:1] - solo).abs().max()) < 1e-05),
        )
    )
    a_n, m_n = tiny(axon_mode="noise")
    m_n.eval()
    m_n.resonance.frozen = True
    with torch.no_grad():
        na = m_n.get_logits(x)
        nb = m_n.get_logits(x)
    tests.append(
        (
            f"axón noise (original): mismo input dos veces en eval Δ {float((na - nb).abs().max()):.1e} (no determinista)",
            float((na - nb).abs().max()) > 0.0,
        )
    )

    def gate_effect(pn):
        a, m = tiny(percept_norm=pn)
        with torch.no_grad():
            for _ in range(8):
                m._encode(torch.randint(0, 96, (2, 16)))
            m.resonance.frozen = True
            e = m.input_norm(m.tok_emb(x))
            d = m.dendrite(m.dendrite_norm(e))
            out = []
            for g in (-1.5, -4.0):
                m.perception.percept_gate.fill_(g)
                pf = m.perception(d, m.resonance.get_slots())
                pf = m.percept_norm(pf) if pn else pf
                out.append(float(pf.pow(2).mean().sqrt()))
        return out[1] / max(out[0], 1e-12)

    r0, r1 = (gate_effect(0), gate_effect(1))
    tests.append(
        (
            f"percepción: cerrar el gate 10× reduce lo que entra al Soma a {r0:.3f}× sin norm, y a {r1:.3f}× con norm (original)",
            r0 < 0.15 and r1 > 0.8,
        )
    )
    new = [lr_schedule(s, s * 950.0, 6500, 1950, 0.0) for s in range(6501)]
    old_branch = [lr_schedule(s, s * 950.0, 6500, 0, 99999.0) for s in range(6501)]
    wall = [lr_schedule(s, s * 1000.0, 10**9, 100, 600.0) for s in range(601)]
    tests.append(
        (
            f"LR: constante hasta el paso 4550, lineal a 0 en el 6500 (final {new[-1]:.3f}, mitad del warmdown {new[5525]:.3f}) · modo reloj explícito baja a {wall[-1]:.2f}",
            new[4550] == 1.0
            and new[4551] < 1.0
            and (new[-1] == 0.0)
            and (abs(new[5525] - 0.5) < 1e-09)
            and all((v == 1.0 for v in old_branch))
            and (wall[0] == 1.0)
            and (wall[-1] == 0.0),
        )
    )
    a, m = tiny()
    g = build_param_groups(m)
    ids = [id(p) for ps in g.values() for p in ps]
    rm = {id(p) for n, p in m.named_parameters() if "resid_mix" in n}
    tests.append(
        (
            f"optimizadores: {len(ids)} tensores, cada uno en un solo grupo, resid_mix solo en scalar_nowd",
            len(ids) == len(set(ids)) == len(list(m.parameters()))
            and (not rm & {id(p) for p in g["matrix"]})
            and (rm <= {id(p) for p in g["scalar_nowd"]})
            and all((p.dtype == torch.float32 for p in m.parameters())),
        )
    )

    def slot_cos(write, decay):
        a, m = tiny(resonance_write=write, resonance_ema_decay=decay, resonance_slots=6)
        gen = torch.Generator().manual_seed(3)
        common = torch.randn(64, generator=gen)
        with torch.no_grad():
            for i in range(60):
                h = (
                    common
                    + torch.randn(4, 1, 64, generator=gen) * 2.0
                    + torch.randn(4, 16, 64, generator=gen) * 0.3
                )
                m.resonance.update(h)
        s = F.normalize(m.resonance.slots[:, 0], dim=-1)
        c = s @ s.T
        return (float(c[~torch.eye(6, dtype=torch.bool)].mean()), m)

    c_new, m_new = slot_cos("sample", 0.0)
    c_old, _ = slot_cos("mean", 0.95)
    before = m_new.resonance.slots.clone()
    m_new.resonance.frozen = True
    m_new.resonance.update(torch.randn(4, 16, 64))
    tests.append(
        (
            f"memoria: coseno medio entre slots {c_new:+.2f} con sample, {c_old:+.2f} con promedio de batch + EMA 0.95 (original)",
            c_old > 0.6 and c_new < c_old - 0.4 and torch.equal(before, m_new.resonance.slots),
        )
    )

    def eval_trace(restore, isolated):
        a, m = tiny(ttt_restore_after_eval=restore, eval_isolated=isolated)
        with torch.no_grad():
            for _ in range(8):
                m._encode(torch.randint(0, 96, (2, 16)))
        val = torch.randint(0, 96, (16 * 12 + 1,))
        lut_b = torch.ones(96, dtype=torch.int16)
        lut_f = torch.zeros(96, dtype=torch.bool)
        ttt = TTTIsoTrainer(m, lr=0.0005, inner_steps=2, beta=0.1)
        w0 = {k: v.clone() for k, v in m.perception.state_dict().items()}
        b0 = {k: v.clone() for k, v in m.named_buffers()}
        eval_val(a, m, 0, 1, torch.device("cpu"), 1, val, lut_b, lut_f, lut_f, ttt_trainer=ttt, base_model=m)
        dw = max((float((m.perception.state_dict()[k] - w0[k]).abs().max()) for k in w0))
        db = max((float((b.float() - b0[k].float()).abs().max()) for k, b in m.named_buffers()))
        return (dw, db, m.training)

    dw1, db1, tr1 = eval_trace(1, 1)
    dw0, db0, _ = eval_trace(0, 0)
    tests.append(
        (
            f"evaluación aislada: Δ pesos de percepción {dw1:.1e}, Δ buffers {db1:.1e} · con el comportamiento original Δ pesos {dw0:.1e}, Δ buffers {db0:.1e}",
            dw1 == 0.0 and db1 == 0.0 and (dw0 > 0) and (db0 > 0) and tr1,
        )
    )
    a, m = tiny()
    m.train()
    g = build_param_groups(m)
    opts = [
        Lion([{"params": g["emb"], "lr": 0.001}]),
        Muon(g["matrix"], lr=0.04, momentum=0.9, backend_steps=5),
        torch.optim.AdamW(
            [
                {"params": g["scalar"], "weight_decay": 0.01},
                {"params": g["scalar_nowd"], "weight_decay": 0.0},
            ],
            lr=0.04,
        ),
    ]
    xb = torch.randint(0, 96, (4, 16))
    yb = torch.roll(xb, -1, 1)
    losses, nograd, bad = ([], [], [])
    for it in range(30):
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss = m(xb, yb)
        loss.backward()
        if it == 1:
            nograd = [
                n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0
            ]
            bad = [
                n
                for n, p in m.named_parameters()
                if p.grad is not None and (not torch.isfinite(p.grad).all())
            ]
        for o in opts:
            o.step()
        losses.append(float(loss.detach()))
    tests.append(
        (
            f"entrenamiento: loss {losses[0]:.3f} -> {losses[-1]:.3f} en 30 pasos sobre un batch fijo · sin gradiente: {nograd or 'ninguno'}",
            losses[-1] < losses[0] - 0.5
            and (not nograd)
            and (not bad)
            and all((math.isfinite(v) for v in losses)),
        )
    )
    a, m = tiny()
    grads = []
    for raw in (-1.386, -6.0, 6.0):
        m.axon.rho_k.data.fill_(raw)
        m.zero_grad()
        m(xb, yb).backward()
        grads.append(float(m.axon.rho_k.grad.abs()))
    tests.append(
        (
            f"rho_k: gradiente en k=2.0, k≈0.52 y k≈7.98 -> {['%.1e' % v for v in grads]}",
            all((v > 0 for v in grads)),
        )
    )
    a, m = tiny()
    d = tempfile.mkdtemp()
    p = save_checkpoint(os.path.join(d, "x_raw.pt"), m, {"steps_done": 3})
    pay = torch.load(p, map_location="cpu", weights_only=True)
    a2, m2 = tiny()
    m2.load_state_dict(pay["model_state_dict"], strict=True)
    same = all((torch.equal(u, v) for u, v in zip(m.state_dict().values(), m2.state_dict().values())))
    try:
        distill_resonance(m)
        dist_ok = True
    except Exception:
        dist_ok = False
    tests.append(
        (
            f"checkpoint: formato {pay['format']}, state_dict idéntico, steps_done guardado · distill_resonance en modo seq",
            same
            and pay["format"] == CKPT_FORMAT
            and (pay["extra"]["steps_done"] == 3)
            and dist_ok
            and ("dendrite_layout" in pay["hparams"]),
        )
    )
    a, m = tiny(
        dendrite_layout="overlap",
        percept_norm=1,
        axon_mode="noise",
        resonance_write="mean",
        resonance_ema_decay=0.95,
    )
    with torch.no_grad():
        lo = float(m(xb, yb))
    tests.append(
        (
            f"interruptores en valor original: el modelo corre (loss {lo:.3f}) y el axón es AxonProjector",
            math.isfinite(lo) and isinstance(m.axon, AxonProjector),
        )
    )
    print("belen selftest")
    for name, ok in tests:
        print(f"  [{('ok' if ok else 'FAIL')}] {name}")
    print(
        "Honestidad: modelo diminuto con datos aleatorios. Valida la MECÁNICA de cada corrección, no el desempeño."
    )
    return 0 if all((ok for _, ok in tests)) else 1
