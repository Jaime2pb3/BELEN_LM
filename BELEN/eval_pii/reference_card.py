"""  
|---|---|
| `reference_card.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Compare it. |
"""
-----

from __future__ import annotations
import argparse
import json
import os
import sys
import zlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leak_forge as F
import context_grammar as G

VERSION = "0.2-belenc-pii"
Detector = Callable[[List[str]], np.ndarray]


class HashStates:

    def __init__(self, dim: int = 160, buckets: int = 1 << 15, seed: int = 0, nmax: int = 4):
        self.table = np.random.default_rng(seed).standard_normal((buckets, dim)).astype(np.float32) / np.sqrt(
            nmax
        )
        self.buckets, self.nmax, self.dim = (buckets, nmax, dim)
        self._cache: Dict[str, np.ndarray] = {}

    def idx(self, t: str) -> np.ndarray:
        if t not in self._cache:
            if len(self._cache) > 20000:
                self._cache.clear()
            t2 = t + "\x03" * self.nmax
            self._cache[t] = np.array(
                [
                    [
                        zlib.crc32(t2[i : i + n].encode("utf-8")) % self.buckets
                        for n in range(1, self.nmax + 1)
                    ]
                    for i in range(len(t))
                ],
                np.int64,
            ).reshape(-1, self.nmax)
        return self._cache[t]

    def __call__(self, t: str) -> np.ndarray:
        return self.table[self.idx(t)].sum(1)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class FactorProbe:

    def __init__(self, dim: int, factorized: bool = True, seed: int = 0):
        r = np.random.default_rng(seed)
        self.we = (r.standard_normal(dim) * 0.01).astype(np.float64)
        self.wr = (r.standard_normal(dim) * 0.01).astype(np.float64)
        self.be = self.br = 0.0
        self.factorized = factorized

    def forward(self, H: np.ndarray) -> Dict:
        H = np.asarray(H, np.float64)
        e = H @ self.we + self.be
        m = e.max()
        w = np.exp(e - m)
        a = w / w.sum()
        E = m + np.log(w.mean())
        T = len(H)
        c = ((1.0 - a)[:, None] * H).sum(0) / max(T - 1, 1)
        R = float(c @ self.wr + self.br)
        return {"E": float(E), "R": R, "a": a, "c": c, "g_e": a @ H}

    def prob(self, H) -> float:
        f = self.forward(H)
        return float(_sig(f["E"]) * (_sig(f["R"]) if self.factorized else 1.0))

    def loss_and_grads(self, H, y_ev: float, y_role: Optional[float], y: float):
        f = self.forward(H)
        pe, pr = (_sig(f["E"]), _sig(f["R"]))
        eps = 1e-12
        if not self.factorized:
            L = -(y * np.log(pe + eps) + (1 - y) * np.log(1 - pe + eps))
            return (L, (pe - y) * f["g_e"], pe - y, np.zeros_like(self.wr), 0.0)
        L = -(y_ev * np.log(pe + eps) + (1 - y_ev) * np.log(1 - pe + eps))
        gwe, gbe = ((pe - y_ev) * f["g_e"], pe - y_ev)
        gwr, gbr = (np.zeros_like(self.wr), 0.0)
        if y_role is not None:
            L += -(y_role * np.log(pr + eps) + (1 - y_role) * np.log(1 - pr + eps))
            gwr, gbr = ((pr - y_role) * f["c"], pr - y_role)
        return (L, gwe, gbe, gwr, gbr)

    def fit(
        self,
        states: Sequence[np.ndarray],
        y_ev,
        y_role,
        y,
        epochs: int = 6,
        lr: float = 0.02,
        seed: int = 0,
        l2: float = 0.0001,
    ):
        r = np.random.default_rng(seed)
        P = [self.we, self.wr]
        m = [np.zeros_like(p) for p in P] + [0.0, 0.0]
        v = [np.zeros_like(p) for p in P] + [0.0, 0.0]
        b1, b2, step = (0.9, 0.999, 0)
        for _ in range(epochs):
            for i in r.permutation(len(states)):
                _, gwe, gbe, gwr, gbr = self.loss_and_grads(states[i], y_ev[i], y_role[i], y[i])
                gs = [gwe + l2 * self.we, gwr + l2 * self.wr, gbe, gbr]
                step += 1
                upd = []
                for j, g in enumerate(gs):
                    m[j] = b1 * m[j] + (1 - b1) * g
                    v[j] = b2 * v[j] + (1 - b2) * g * g
                    upd.append(lr * (m[j] / (1 - b1**step)) / (np.sqrt(v[j] / (1 - b2**step)) + 1e-08))
                self.we -= upd[0]
                self.wr -= upd[1]
                self.be -= upd[2]
                self.br -= upd[3]
        return self


def training_set(n_leak: int, n_ctx: int, seed: int, lang: str = "es", evasions: Sequence[str] = F.EVASIONS):
    texts, ev, role, y = ([], [], [], [])
    for s in F.generate(n_leak, seed=seed, blank_frac=0.5, lang=lang, evasions=evasions):
        texts.append(s.text)
        ev.append(float(s.label))
        role.append(None if s.kind == "decoy" else 1.0)
        y.append(float(s.label))
    for g in G.train_groups(n_ctx, seed, lang):
        e = float(g["dose"] > 0)
        for v in g["variants"]:
            texts.append(v["text"])
            ev.append(e)
            role.append(float(v["label"]))
            y.append(e * v["label"])
    return (texts, ev, role, y)


def make_probe_detector(
    states_fn, factorized: bool, n_leak: int, n_ctx: int, seed: int, epochs: int = 6, lang: str = "es"
) -> Detector:
    texts, ev, role, y = training_set(n_leak, n_ctx, seed, lang)
    probe = FactorProbe(states_fn(texts[0]).shape[1], factorized, seed).fit(
        [states_fn(t) for t in texts], ev, role, y, epochs=epochs, seed=seed
    )

    def det(ts: List[str]) -> np.ndarray:
        out = []
        for t in ts:
            H = states_fn(t)
            out.append(probe.prob(H) if len(H) else 0.0)
        return np.array(out)

    det.probe = probe
    return det


def summarize_leak(res: Dict) -> Dict:
    cells = res["cells"]
    lods = [c["lod95"] for c in cells]
    return {
        "fpr_blank_mean": float(np.mean([c["fpr_blank"] for c in cells if c.get("fpr_blank") is not None])),
        "fpr_decoy": res.get("fpr_decoy"),
        "cells": len(cells),
        "lod_le_0.2": sum((l is not None and l <= 0.2 for l in lods)),
        "lod_le_0.5": sum((l is not None and l <= 0.5 for l in lods)),
        "lod_none": sum((l is None for l in lods)),
        "rate_d0.1": (
            float(np.mean([c["rates"]["0.1"] for c in cells])) if "0.1" in cells[0]["rates"] else None
        ),
        "rate_d1": float(np.mean([c["rates"]["1.0"] for c in cells])),
    }


def context_block(det: Detector, n: int, seed: int, lang: str, text_fn=None) -> Dict:
    out = {}
    for c in G.CONDITIONS:
        r = G.context_eval(det, n, seed, c, lang, text_fn=text_fn)
        r.pop("per_group")
        out[c] = r
    return out


def run_ficha(
    instruments: Dict[str, Detector],
    reps: int,
    seed: int,
    images: bool = False,
    lang: str = "es",
    log: Callable = print,
) -> Dict:
    out = {
        "version": VERSION,
        "reps": reps,
        "seed": seed,
        "lang": lang,
        "text": {},
        "image": {},
        "grammar_audit": {
            k: v for k, v in G.audit(lang, n=max(60, reps * 3), seed=seed).items() if k != "conditions"
        },
    }
    for name, det in instruments.items():
        leak = F.evaluate(det, reps, seed, lang=lang)
        out["text"][name] = {
            "leak": summarize_leak(leak),
            "leak_cells": leak["cells"],
            "context": context_block(det, reps * 3, seed, lang),
        }
        log(f"[belenc-pii] texto · {name}: listo")
    if images:
        import image_forge as I

        font = I._font(0)
        to_img = lambda v: I.render(v["text"], v["span"], "light", font)[0]
        for name, det in instruments.items():
            idet = I.ocr_then(det)
            leak = I.evaluate(
                idet, max(4, reps // 6), seed, types=("PHONE", "IP"), doses=(0.0, 0.2, 0.5, 1.0), lang=lang
            )
            zi = [c for c in leak["cells"] if c["method"] in I.ZERO_INFO]
            ri = [c for c in leak["cells"] if c["method"] not in I.ZERO_INFO]
            out["image"][f"ocr:{name}"] = {
                "fpr_blank_mean": float(np.mean([c["fpr_blank"] for c in zi])),
                "residual_d0_mean": float(np.mean([c["residual_at_d0"] for c in ri])),
                "rate_d1": float(np.mean([c["rates"]["1.0"] for c in leak["cells"]])),
                "leak_cells": leak["cells"],
                "context": context_block(idet, max(12, reps // 2), seed, lang, text_fn=to_img),
            }
            log(f"[belenc-pii] imagen · ocr:{name}: listo")
    return out


COND_HEAD = {
    "cong_seen": "visto",
    "cong_lex": "dueños nuevos",
    "cong_frame": "marcos nuevos",
    "contrast": "contraste",
    "incong": "INCONGRUENTE",
}


def report(res: Dict) -> str:
    f2 = lambda x: "—" if x is None else f"{x:.2f}"
    cell = lambda c: f"**{c['pair_acc']:.2f}** ({c['pair_acc_ci'][0]:.2f}–{c['pair_acc_ci'][1]:.2f})"
    cert = res["grammar_audit"].get("certifies", {})
    L = [
        f"# BELENC-PII v{VERSION} — ficha de validación",
        "",
        f"Idioma: {res['lang']} · réplicas por celda: {res['reps']} · semilla {res['seed']} · matriz sintética",
        "",
        "**Un negativo solo informa por encima del LOD del instrumento. Un positivo solo si supera su tasa en blancos y señuelos. En contexto, la exactitud de par de un detector de pura forma es 0.50 por construcción.**",
        "",
        f"Auditoría de la gramática: {('aprobada' if res['grammar_audit']['ok'] else 'CON FALLAS')} · condiciones que certifican comprensión (las palabras clave fallan en ellas): "
        + ", ".join((f"{COND_HEAD[k]} {('sí' if v else 'NO')}" for k, v in cert.items())),
        "",
        "## Texto · eje de dosis (6 tipos × 5 evasiones)",
        "",
        "| instrumento | FPR blanco | FPR señuelo | detecta d=0.1 | detecta d=1 | celdas LOD≤0.2 | LOD≤0.5 | sin LOD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, r in res["text"].items():
        s = r["leak"]
        L.append(
            f"| {n} | {f2(s['fpr_blank_mean'])} | {f2(s['fpr_decoy'])} | {f2(s['rate_d0.1'])} | {f2(s['rate_d1'])} | {s['lod_le_0.2']} | {s['lod_le_0.5']} | {s['lod_none']} |"
        )
    L += [
        "",
        "## Texto · eje de contexto (exactitud de par, IC 95% por bootstrap sobre grupos)",
        "",
        "| instrumento | " + " | ".join((COND_HEAD[c] for c in G.CONDITIONS)) + " |",
        "|---|" + "---|" * len(G.CONDITIONS),
    ]
    for n, r in res["text"].items():
        L.append(f"| {n} | " + " | ".join((cell(r["context"][c]) for c in G.CONDITIONS)) + " |")
    L += [
        "",
        "Lectura: *visto* es memoria de plantilla. *dueños nuevos* exige conocer palabras que no se vieron al entrenar. *marcos nuevos* no distingue comprensión (las palabras clave lo pasan). *contraste* lo decide la negación. *INCONGRUENTE* solo lo pasa quien compone: la única palabra con carga apunta al papel contrario.",
    ]
    if res["image"]:
        L += [
            "",
            "## Imagen (capturas renderizadas, detrás de Tesseract)",
            "",
            "| instrumento | FPR blanco (barra/borrado) | residual d=0 (mosaico/desenfoque) | detecta d=1 | "
            + " | ".join((COND_HEAD[c] for c in G.CONDITIONS))
            + " |",
            "|---|---|---|---|" + "---|" * len(G.CONDITIONS),
        ]
        for n, r in res["image"].items():
            L.append(
                f"| {n} | {r['fpr_blank_mean']:.2f} | {r['residual_d0_mean']:.2f} | {r['rate_d1']:.2f} | "
                + " | ".join((f"{r['context'][c]['pair_acc']:.2f}" for c in G.CONDITIONS))
                + " |"
            )
        L += [
            "",
            "Residual d=0: con mosaico o desenfoque el dato está tapado pero no borrado. Un valor mayor que el FPR de barra significa que el detector todavía lo lee a través de la redacción.",
        ]
    return "\n".join(L)


def selftest() -> int:
    tests = []
    r = np.random.default_rng(2)
    for lang in ("en", "es"):
        texts, ev, role, y = training_set(200, 60, 5, lang)
        ok = all(
            (
                yy == (e * (ro if ro is not None else 0.0) if ro is not None else e) or ro is None
                for e, ro, yy in zip(ev, role, y)
            )
        )
        ok &= (
            len(texts) == len(ev) == len(role) == len(y)
            and any((ro == 0.0 for ro in role))
            and any((e == 0.0 for e in ev))
        )
        tests.append((f"[{lang}] entrenamiento: {len(texts)} textos, etiqueta final = evidencia × papel", ok))
    worst = 0.0
    for fac in (True, False):
        p = FactorProbe(12, fac, seed=1)
        p.we = r.standard_normal(12) * 0.3
        p.wr = r.standard_normal(12) * 0.3
        H = r.standard_normal((9, 12))
        _, gwe, gbe, gwr, gbr = p.loss_and_grads(H, 1.0, 0.0, 0.0)
        for vec, g in ((p.we, gwe), (p.wr, gwr)):
            for j in (0, 5, 11):
                old = vec[j]
                R0 = p.forward(H)["R"]

                def Lnum():
                    f = p.forward(H)
                    pe = _sig(f["E"])
                    pr = _sig(f["R"] if vec is p.wr else R0)
                    if not fac:
                        return -np.log(1 - pe + 1e-12)
                    return -np.log(pe + 1e-12) - np.log(1 - pr + 1e-12)

                vec[j] = old + 1e-05
                lp = Lnum()
                vec[j] = old - 1e-05
                lm = Lnum()
                vec[j] = old
                worst = max(worst, abs((lp - lm) / 2e-05 - g[j]))
    tests.append((f"gradientes analíticos == diferencias finitas (error máx {worst:.1e})", worst < 1e-06))
    D, T = (24, 14)
    u_ev, u_ro = (np.eye(D)[0] * 4, np.eye(D)[1] * 4)

    def planted(ev, ro, rr):
        H = rr.standard_normal((T, D)) * 0.5
        H[rr.integers(0, T)] += u_ev * (1 if ev else -1)
        H += u_ro * (1 if ro else -1) / 4
        return H

    rr = np.random.default_rng(8)
    lab = [(int(rr.integers(0, 2)), int(rr.integers(0, 2))) for _ in range(600)]
    Hs = [planted(e, o, rr) for e, o in lab]
    pf = FactorProbe(D, True).fit(
        Hs[:400],
        [e for e, _ in lab[:400]],
        [float(o) for _, o in lab[:400]],
        [e * o for e, o in lab[:400]],
        epochs=4,
    )
    pred = np.array([pf.prob(H) >= 0.5 for H in Hs[400:]])
    acc = float((pred == np.array([e * o for e, o in lab[400:]], bool)).mean())
    tests.append(
        (
            f"estados plantados: la sonda factorizada recupera evidencia × papel (exactitud {acc:.3f})",
            acc > 0.95,
        )
    )
    accs = {
        (lg, c): G.context_eval(F.regex_detector, 45, 3, c, lg)["pair_acc"]
        for lg in ("en", "es")
        for c in G.CONDITIONS
    }
    tests.append(
        (
            f"regex en pares mínimos: {sorted(set((round(v, 2) for v in accs.values())))} en las 5 condiciones × 2 idiomas",
            all((abs(v - 0.5) < 1e-09 for v in accs.values())),
        )
    )
    hs = HashStates(dim=32, buckets=1 << 10)
    A, B_ = (hs("ID del cliente: ABCD123456-H42"), hs("ID del cliente: ABCD123456-H42"))
    tests.append(
        (
            "HashStates: [T, D], determinista, finito, y tolera texto vacío",
            A.shape == (30, 32)
            and np.array_equal(A, B_)
            and np.isfinite(A).all()
            and (hs("").shape == (0, 32)),
        )
    )
    hs = HashStates(dim=64, buckets=1 << 13)
    det = make_probe_detector(hs, True, n_leak=400, n_ctx=90, seed=4, epochs=3, lang="en")
    res = run_ficha(
        {"regex": F.regex_detector, "factor": det}, reps=6, seed=4, lang="en", log=lambda *_: None
    )
    md = report(res)
    ctx = res["text"]["factor"]["context"]
    tests.append(
        (
            f"pipe: entrenar sonda -> ficha -> reporte (factor en visto: {ctx['cong_seen']['pair_acc']:.2f})",
            "eje de contexto" in md
            and all((0.0 <= ctx[c]["pair_acc"] <= 1.0 for c in G.CONDITIONS))
            and res["grammar_audit"]["ok"],
        )
    )
    print(f"BELENC-PII v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{('ok' if ok else 'FAIL')}] {name}")
    print(
        "Honestidad: [3] es sintético plantado, valida la MECÁNICA de la sonda. El resultado real lo da `ficha`."
    )
    return 0 if all((ok for _, ok in tests)) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="BELENC-PII: ficha de validación de detectores de fuga de PII")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("ficha")
    p.add_argument("--lang", default="en", choices=G.LANGS)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--train-leak", type=int, default=3000)
    p.add_argument("--train-ctx", type=int, default=450)
    p.add_argument(
        "--images", action="store_true", help="agrega el brazo de capturas (necesita PIL + Tesseract)"
    )
    p.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="archivo.py:funcion que devuelve {nombre: detector}. Repetible. Ver detector_adapters.py",
    )
    p.add_argument("--only-plugins", action="store_true", help="omite los instrumentos de referencia")
    p.add_argument("--out-dir", default="ficha_out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "ficha":
        import detector_adapters as A

        tseed = a.seed + 1000
        inst = {}
        if not a.only_plugins:
            texts, ev, role, y = training_set(a.train_leak, a.train_ctx, tseed, a.lang)
            nb = F.NgramLogReg().fit(texts, [int(v) for v in y])
            hs = HashStates()
            inst = {
                "regex": F.regex_detector,
                "ngram": nb,
                "monolítica(hash)": make_probe_detector(
                    hs, False, a.train_leak, a.train_ctx, tseed, lang=a.lang
                ),
                "factor(hash)": make_probe_detector(hs, True, a.train_leak, a.train_ctx, tseed, lang=a.lang),
                "regex+papel(ngram)": A.with_role(
                    A.spans_regex(), A.train_role_ngram(tseed, a.train_leak, a.train_ctx, a.lang)
                ),
            }
        for spec in a.plugin:
            inst.update(A.load_plugin(spec))
        for name in list(inst):
            c = A.conformance(inst[name])
            if not c["ok"]:
                print(
                    f"[belenc-pii] {name}: NO cumple el contrato y se excluye -> "
                    + "; ".join((f"{n}: {d}" for n, ok, d in c["checks"] if not ok))
                )
                del inst[name]
        if not inst:
            raise SystemExit("[belenc-pii] ningún instrumento cumple el contrato")
        res = run_ficha(inst, a.reps, a.seed, images=a.images, lang=a.lang)
        os.makedirs(a.out_dir, exist_ok=True)
        md = report(res)
        with open(os.path.join(a.out_dir, f"BELENC_PII_{a.lang}.md"), "w", encoding="utf-8") as f:
            f.write(md + "\n")
        with open(os.path.join(a.out_dir, f"belenc_pii_{a.lang}.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
        print(md)
        print(f"\n[belenc-pii] escribí {a.out_dir}/")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
