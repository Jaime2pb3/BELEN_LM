"""  
|---|---|
| `text_fingerprint.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Read it. |
-----
"""


from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leak_forge as F
import context_grammar as G
import model_probe as DP

VERSION = "huella-0.2"
KS = (4, 8, 16, 32, 0)
KNAME = {4: "4", 8: "8", 16: "16", 32: "32", 0: "todo"}
CFG = {
    "n_train": 480,
    "n_val": 160,
    "n_test": 240,
    "seeds": (31, 32, 33),
    "controls": [7, 8, 9],
    "real_len": 48,
    "pca": 64,
}
SMOKE = {
    "n_train": 40,
    "n_val": 20,
    "n_test": 30,
    "seeds": (31, 32, 33),
    "controls": [7, 8],
    "real_len": 48,
    "pca": 8,
}
DELTAS = (0.5, 1.0, 2.0)


def shuffle_words(text: str, r) -> str:
    lines = [ln.split(" ") for ln in text.split("\n")]
    w = [x for ln in lines for x in ln]
    if len(set(w)) < 2:
        return text
    out = w
    for _ in range(20):
        out = [w[i] for i in r.permutation(len(w))]
        if out != w:
            break
    else:
        out = w[::-1]
    res, pos = ([], 0)
    for ln in lines:
        res.append(" ".join(out[pos : pos + len(ln)]))
        pos += len(ln)
    return "\n".join(res)


def _mixed(n: int, seed: int, lang: str) -> List[str]:
    a = [s.text for s in F.generate(n // 2 + 1, seed=seed, lang=lang, evasions=("none",))]
    b = [v["text"] for g in G.groups(n // 6 + 1, seed, "cong_seen", lang) for v in g["variants"]]
    out = [x for pair in zip(a, b) for x in pair]
    return out[:n]


def task_idioma(n: int, seed: int) -> Tuple[List[str], List[int]]:
    en, es = (_mixed(n // 2, seed, "en"), _mixed(n // 2, seed + 500, "es"))
    return (en + es, [0] * len(en) + [1] * len(es))


def task_portadora(n: int, seed: int) -> Tuple[List[str], List[int]]:
    r = np.random.default_rng([seed, 7])
    X, y = ([], [])
    for i in range(n):
        c = i % len(F.CARRIERS)
        X.append(
            F.make_sample(
                r,
                str(r.choice(F.PII_TYPES)),
                float(r.choice(F.DOSES)),
                str(r.choice(F.OPERATORS)),
                "none",
                F.CARRIERS[c],
                lang="en",
            ).text
        )
        y.append(c)
    return (X, y)


def window(text: str, r, lo: int = 6, hi: int = 14) -> str:
    w = [x for x in text.replace("\n", " ").split(" ") if x]
    if len(w) <= lo:
        return " ".join(w)
    n = int(r.integers(lo, min(hi, len(w)) + 1))
    i = int(r.integers(0, len(w) - n + 1))
    return " ".join(w[i : i + n])


def task_orden(n: int, seed: int) -> Tuple[List[str], List[int]]:
    r = np.random.default_rng([seed, 8])
    base = [window(t, r) for t in _mixed(n // 2, seed, "en")]
    return (base + [shuffle_words(t, r) for t in base], [0] * len(base) + [1] * len(base))


def real_fragments(
    shard: str, decode: Callable[[List[int]], str], n: int, seed: int, length: int, bos: int
) -> List[str]:
    tok = np.memmap(shard, dtype="<u2", mode="r", offset=256 * 4)
    r = np.random.default_rng([seed, 9])
    out = []
    while len(out) < n:
        i = int(r.integers(0, len(tok) - length - 1))
        w = np.asarray(tok[i : i + length])
        if (w == bos).any():
            continue
        t = decode(w.tolist()).strip()
        if len(t.split()) >= 6:
            out.append(t)
    return out


def task_orden_real(
    n: int, seed: int, shard: str, decode, length: int, bos: int
) -> Tuple[List[str], List[int]]:
    r = np.random.default_rng([seed, 10])
    base = real_fragments(shard, decode, n // 2, seed, length, bos)
    return (base + [shuffle_words(t, r) for t in base], [0] * len(base) + [1] * len(base))


def prefix_means(H: np.ndarray, ks: Sequence[int] = KS) -> np.ndarray:
    D = H.shape[1]
    if len(H) == 0:
        return np.zeros((len(ks), D), np.float32)
    c = np.cumsum(H.astype(np.float64), 0)
    return np.stack(
        [c[(min(k, len(H)) if k else len(H)) - 1] / (min(k, len(H)) if k else len(H)) for k in ks]
    ).astype(np.float32)


class Softmax:

    def __init__(self, n_classes: int, l2: float = 0.001, iters: int = 300, lr: float = 0.5):
        self.C, self.l2, self.iters, self.lr = (n_classes, l2, iters, lr)

    def fit(self, X: np.ndarray, y: Sequence[int]):
        X = np.asarray(X, np.float64)
        self.mu, self.sd = (X.mean(0), X.std(0) + 1e-06)
        Z = (X - self.mu) / self.sd
        Y = np.eye(self.C)[np.asarray(y)]
        self.W, self.b = (np.zeros((Z.shape[1], self.C)), np.zeros(self.C))
        for _ in range(self.iters):
            L = Z @ self.W + self.b
            P = np.exp(L - L.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            G_ = (P - Y) / len(Z)
            self.W -= self.lr * (Z.T @ G_ + self.l2 * self.W)
            self.b -= self.lr * G_.sum(0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, np.float64) - self.mu) / self.sd @ self.W + self.b).argmax(1)


class Probe:

    def __init__(self, n_classes: int, pca: int = 0):
        self.C, self.pca = (n_classes, pca)

    def _z(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, np.float64) - self.mu) / self.sd
        return Z @ self.V.T if self.V is not None else Z

    def fit(self, X: np.ndarray, y: Sequence[int]):
        X = np.asarray(X, np.float64)
        self.mu, self.sd = (X.mean(0), X.std(0) + 1e-06)
        Z = (X - self.mu) / self.sd
        k = min(self.pca, Z.shape[0] - 1, Z.shape[1])
        self.V = np.linalg.svd(Z, full_matrices=False)[2][:k] if k > 0 else None
        self.clf = Softmax(self.C).fit(self._z(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self._z(X))


def reducers(cfg_pca: int) -> List[Tuple[str, int]]:
    return [("sin", 0)] + ([(f"pca{cfg_pca}", cfg_pca)] if cfg_pca > 0 else [])


def sensitivity(
    Xtr: np.ndarray, Xte: np.ndarray, pca: int, deltas: Sequence[float] = DELTAS, seed: int = 0
) -> Dict[str, float]:
    r = np.random.default_rng([seed, 77])
    u = r.standard_normal(Xtr.shape[1])
    u /= np.linalg.norm(u)
    sd = float((np.asarray(Xtr, np.float64) @ u).std()) + 1e-12
    ytr, yte = (r.integers(0, 2, len(Xtr)), r.integers(0, 2, len(Xte)))
    out = {}
    for d in (0.0,) + tuple(deltas):
        a = np.asarray(Xtr, np.float64) + np.outer((2 * ytr - 1) * d * sd, u)
        b = np.asarray(Xte, np.float64) + np.outer((2 * yte - 1) * d * sd, u)
        out[f"{d:g}"] = float((Probe(2, pca).fit(a, ytr).predict(b) == yte).mean())
    return out


def features(ex, texts: Sequence[str]) -> Dict[str, np.ndarray]:
    out = {s: [] for s in ex.streams}
    for t in texts:
        st = ex.run(t)
        for s in ex.streams:
            out[s].append(prefix_means(st[s]))
    return {s: np.stack(v) for s, v in out.items()}


def probe_model(ex, task: Dict, pca: int = 0, sens: bool = False) -> Dict:
    Ftr, Fva, Fte = (features(ex, task[k][0]) for k in ("train", "val", "test"))
    ytr, yva, yte = (np.asarray(task[k][1]) for k in ("train", "val", "test"))
    perm = np.random.default_rng(991).permutation(len(ytr))
    res = {}
    for ki, k in enumerate(KS):
        fits = {
            (s, rn): Probe(task["C"], rk).fit(Ftr[s][:, ki], ytr)
            for s in ex.streams
            for rn, rk in reducers(pca)
        }
        va = {c: float((p.predict(Fva[c[0]][:, ki]) == yva).mean()) for c, p in fits.items()}
        best = max(fits, key=lambda c: va[c])
        ebest = max((c for c in fits if c[0] == "emb"), key=lambda c: va[c])
        hit = (fits[best].predict(Fte[best[0]][:, ki]) == yte).astype(float)
        emb = (fits[ebest].predict(Fte["emb"][:, ki]) == yte).astype(float)
        sh = Probe(task["C"], dict(reducers(pca))[best[1]]).fit(Ftr[best[0]][:, ki], ytr[perm])
        by = {
            s: max((float((fits[s, rn].predict(Fte[s][:, ki]) == yte).mean()) for rn, _ in reducers(pca)))
            for s in ex.streams
        }
        res[KNAME[k]] = {
            "stream": best[0],
            "reducer": best[1],
            "acc": float(hit.mean()),
            "hits": hit,
            "emb": float(emb.mean()),
            "emb_hits": emb,
            "emb_reducer": ebest[1],
            "shuf_fit": float((sh.predict(Ftr[best[0]][:, ki]) == ytr[perm]).mean()),
            "shuf_test": float((sh.predict(Fte[best[0]][:, ki]) == yte).mean()),
            "by_stream": by,
        }
        if sens and k == 0:
            res[KNAME[k]]["sensitivity"] = sensitivity(
                Ftr[best[0]][:, ki], Fte[best[0]][:, ki], dict(reducers(pca))[best[1]]
            )
    return res


def paired_ci(
    a: np.ndarray, b: Optional[np.ndarray] = None, n_boot: int = 2000, seed: int = 0
) -> Tuple[float, float]:
    d = a - (b if b is not None else 0.0)
    i = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    m = d[i].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def build_tasks(cfg: Dict, shard: Optional[str], decode, bos: int) -> Dict[str, Dict]:
    s_tr, s_va, s_te = cfg["seeds"]
    sizes = {"train": (cfg["n_train"], s_tr), "val": (cfg["n_val"], s_va), "test": (cfg["n_test"], s_te)}
    mk = lambda fn, C, chance: {**{k: fn(n, sd) for k, (n, sd) in sizes.items()}, "C": C, "chance": chance}
    tasks = {
        "idioma": mk(task_idioma, 2, 0.5),
        "portadora": mk(task_portadora, len(F.CARRIERS), 1 / len(F.CARRIERS)),
        "orden": mk(task_orden, 2, 0.5),
    }
    if shard:
        tasks["orden_real"] = mk(
            lambda n, sd: task_orden_real(n, sd, shard, decode, cfg["real_len"], bos), 2, 0.5
        )
    for name, t in tasks.items():
        dup = set(t["train"][0]) & set(t["test"][0])
        t["train_test_dups"] = len(dup)
    return tasks


def run(
    script: str,
    ckpt: str,
    tokenizer: str,
    out_dir: str,
    cfg: Dict,
    shard: Optional[str] = None,
    device: Optional[str] = None,
    smoke: bool = False,
    log: Callable = print,
) -> Dict:
    torch = DP._torch()
    import sentencepiece as spm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=tokenizer)
    enc, dec = (lambda t: sp.encode(t), lambda ids: sp.decode(ids))
    mod = DP.load_script(script)
    model, payload = DP.load_raw(mod, ckpt, device)
    hp = payload["hparams"]
    exs = {"entrenado": DP.Extractor(model, enc, "frozen", sp.bos_id())}
    for rs in cfg["controls"]:
        m, _ = DP.build_model(mod, hp, init_seed=rs, device=device)
        exs[f"aleatorio#{rs}"] = DP.Extractor(m, enc, "empty", sp.bos_id())
    tasks = build_tasks(cfg, shard, dec, sp.bos_id())
    out = {
        "version": VERSION,
        "smoke": smoke,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
        "real_shard": os.path.basename(shard) if shard else None,
        "tasks": {},
        "model": {
            "n_params": int(sum((p.numel() for p in model.parameters()))),
            "trained": DP.trained_line(payload.get("extra") or {}, hp),
        },
        "provenance": {
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": device,
            "fingerprint": {
                **DP.fingerprint(script, ckpt, tokenizer),
                "text_fingerprint.py": DP.sha256(os.path.abspath(__file__)),
            },
        },
    }
    for name, task in tasks.items():
        per = {
            tag: probe_model(ex, task, cfg.get("pca", 0), sens=tag == "entrenado") for tag, ex in exs.items()
        }
        rows = {}
        for k in KNAME.values():
            tr = per["entrenado"][k]
            ctl = max((per[t][k] for t in per if t != "entrenado"), key=lambda x: x["acc"])
            ctl_accs = [per[t][k]["acc"] for t in per if t != "entrenado"]
            rows[k] = {
                "acc": tr["acc"],
                "ci": paired_ci(tr["hits"]),
                "stream": tr["stream"],
                "reducer": tr["reducer"],
                "emb": tr["emb"],
                "over_emb_ci": paired_ci(tr["hits"], tr["emb_hits"]),
                "control_range": (min(ctl_accs), max(ctl_accs)),
                "over_control": tr["acc"] - ctl["acc"],
                "over_control_ci": paired_ci(tr["hits"], ctl["hits"]),
                "shuf_fit": tr["shuf_fit"],
                "shuf_test": tr["shuf_test"],
                "by_stream": tr["by_stream"],
            }
        out["tasks"][name] = {
            "chance": task["chance"],
            "n_test": len(task["test"][0]),
            "n_train": len(task["train"][0]),
            "dups": task["train_test_dups"],
            "rows": rows,
            "sensitivity": per["entrenado"][KNAME[0]].get("sensitivity"),
        }
        log(f"[huella] {name}: listo")
    for ex in exs.values():
        ex.close()
    os.makedirs(out_dir, exist_ok=True)
    stem = "HUELLA_SMOKE" if smoke else "HUELLA"
    out = json.loads(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    json.dump(out, open(os.path.join(out_dir, stem + ".json"), "w"), indent=1)
    open(os.path.join(out_dir, stem + ".md"), "w", encoding="utf-8").write(report(out) + "\n")
    return out


NOTE = {
    "idioma": "Trivial por identidad de token. Si emb y el control ya están arriba, esto no dice nada del modelo.",
    "portadora": "Trivial por palabras delatoras de plantilla. Misma lectura que idioma.",
    "orden": "Ventanas tomadas de en medio de textos sintéticos, contra las mismas palabras barajadas. Sigue siendo texto de plantilla: si hay fragmento real, la que vale es orden_real.",
    "orden_real": "Igual que orden, sobre fragmentos REALES de FineWeb. El modelo se entrenó con este tipo de texto.",
}


def report(res: Dict) -> str:
    ci = lambda c: f"({c[0]:+.2f}, {c[1]:+.2f})"
    L = [
        f"# Huella textual sobre BELEN — {VERSION}",
        "",
        "> **EXPLORATORIO.** Sin prerregistro ni veredicto. Números con sus controles, nada más.",
    ]
    if res["smoke"]:
        L += ["> **CORRIDA DE HUMO: tamaños mínimos. Valida el cableado. NO es un resultado.**"]
    M = res.get("model") or {}
    L += [
        "",
        f"Sonda lineal sobre el promedio de los estados de los primeros k tokens. Flujo y reductor ({' o '.join((n for n, _ in reducers(res['config'].get('pca', 0))))}) elegidos en validación. Controles aleatorios: {res['config']['controls']}. Fragmento real: {res['real_shard'] or 'no se dio'}.",
        f"Modelo: {M.get('n_params', 0):,} parámetros · {M.get('trained', 'pasos entrenados no registrados')}.",
        "",
    ]
    for name, T in res["tasks"].items():
        L += [
            f"## {name}  (azar {T['chance']:.2f}, {T.get('n_train', '?')} textos de entrenamiento, {T['n_test']} de prueba, {T['dups']} repetidos entre ambos)",
            "",
            NOTE[name],
            "",
            "| tokens leídos | entrenado (IC 95%) | flujo · reductor | emb del entrenado | controles (rango) | ventaja sobre emb (IC) | ventaja sobre el control más fuerte (IC) | barajada: ajuste / prueba |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for k, r in T["rows"].items():
            L.append(
                f"| {k} | **{r['acc']:.2f}** ({r['ci'][0]:.2f}, {r['ci'][1]:.2f}) | `{r['stream']}` · {r.get('reducer', 'sin')} | {r['emb']:.2f} | {r['control_range'][0]:.2f} a {r['control_range'][1]:.2f} | {r['acc'] - r['emb']:+.2f} {ci(r['over_emb_ci'])} | {r['over_control']:+.2f} {ci(r['over_control_ci'])} | {r.get('shuf_fit', float('nan')):.2f} / {r.get('shuf_test', float('nan')):.2f} |"
            )
        if T.get("sensitivity"):
            sv = T["sensitivity"]
            L += [
                "",
                "Sensibilidad (señal plantada en los estados del flujo elegido, texto completo): "
                + " · ".join((f"δ = {d} sd → {a:.2f}" for d, a in sv.items()))
                + ". Una señal más chica que la que aquí llega a ~0.75 pasaría inadvertida.",
            ]
        L.append("")
    L += [
        "## Cómo leer esto",
        "",
        "- Una huella dice algo del modelo solo si el entrenado supera a **emb** y al **control más fuerte**. Si los tres están igual de altos, la respuesta estaba en los tokens.",
        "- Con 3 controles, una ventaja es provisional frente al ruido de inicialización (p >= 0.25).",
        "- El IC solo cubre el muestreo de textos, no la semilla de entrenamiento.",
        "- **barajada: ajuste / prueba**. El ajuste cerca de 1.00 dice que la sonda memoriza sus datos de entrenamiento: por eso solo cuenta la exactitud en PRUEBA. La de prueba debe quedar en el azar. Si no, hay una fuga en el pipeline.",
        "- **Sensibilidad**. Una huella en el azar NO prueba que el modelo no sepa: prueba que no hay una señal lineal del tamaño que este instrumento detecta. La línea de sensibilidad dice cuál es ese tamaño.",
    ]
    return "\n".join(L)


def selftest() -> int:
    tests = []
    r = np.random.default_rng(0)
    ok = True
    for t in _mixed(60, 3, "en") + _mixed(60, 4, "es"):
        s = shuffle_words(t, r)
        ok &= sorted(t.replace("\n", " ").split(" ")) == sorted(s.replace("\n", " ").split(" ")) and s != t
        ok &= sorted(t) == sorted(s) and [len(x.split(" ")) for x in t.split("\n")] == [
            len(x.split(" ")) for x in s.split("\n")
        ]
    tests.append(
        (
            "orden: barajar conserva exactamente las mismas palabras y caracteres, y siempre cambia el texto",
            ok,
        )
    )
    H = np.arange(40, dtype=np.float32).reshape(10, 4)
    P = prefix_means(H)
    tests.append(
        (
            "prefijos: [K, D], k=4 promedia 4 tokens, k mayor que el largo usa todos, texto vacío no truena",
            P.shape == (5, 4)
            and np.allclose(P[0], H[:4].mean(0))
            and np.allclose(P[2], H.mean(0))
            and np.allclose(P[4], H.mean(0))
            and (prefix_means(np.zeros((0, 4))).shape == (5, 4)),
        )
    )
    X = r.standard_normal((300, 8))
    y = (X[:, 0] + 0.2 * r.standard_normal(300) > 0).astype(int)
    a = float((Softmax(2).fit(X[:200], y[:200]).predict(X[200:]) == y[200:]).mean())
    yr = r.integers(0, 2, 300)
    b = float((Softmax(2).fit(X[:200], yr[:200]).predict(X[200:]) == yr[200:]).mean())
    y3 = np.argmax(X[:, :3], 1)
    c = float((Softmax(3).fit(X[:200], y3[:200]).predict(X[200:]) == y3[200:]).mean())
    tests.append(
        (
            f"sonda: separable {a:.2f}, etiquetas al azar {b:.2f}, tres clases {c:.2f}",
            a > 0.9 and abs(b - 0.5) < 0.15 and (c > 0.8),
        )
    )
    rs = np.random.default_rng(4)
    sdv = 1.0 / np.sqrt(1 + np.arange(512))
    Q = np.linalg.qr(rs.standard_normal((512, 512)))[0]

    def planted(n, j, delta):
        Xp = rs.standard_normal((n, 512)) * sdv
        yp = rs.integers(0, 2, n)
        Xp[:, j] += (2 * yp - 1) * delta * sdv[j]
        return (Xp @ Q, yp)

    (Xa, ya), (Xb, yb) = (planted(480, 400, 1.5), planted(400, 400, 1.5))
    lo_none = float((Probe(2, 0).fit(Xa, ya).predict(Xb) == yb).mean())
    lo_pca = float((Probe(2, 64).fit(Xa, ya).predict(Xb) == yb).mean())
    (Xa2, ya2), (Xb2, yb2) = (planted(480, 2, 1.0), planted(400, 2, 1.0))
    hi_pca = float((Probe(2, 64).fit(Xa2, ya2).predict(Xb2) == yb2).mean())
    pm = rs.permutation(480)
    shp = Probe(2, 0).fit(Xa, ya[pm])
    sh_fit, sh_te = (float((shp.predict(Xa) == ya[pm]).mean()), float((shp.predict(Xb) == yb).mean()))
    X0tr, X0te = (rs.standard_normal((480, 512)) * sdv @ Q, rs.standard_normal((400, 512)) * sdv @ Q)
    sv = sensitivity(X0tr, X0te, 0)
    tests.append(
        (
            f"sonda a D=512: señal en varianza BAJA sin reducir {lo_none:.2f}, con PCA-64 {lo_pca:.2f} (PCA la tira) · en varianza alta PCA-64 {hi_pca:.2f} · barajada ajusta {sh_fit:.2f} y en prueba da {sh_te:.2f}",
            lo_none > 0.62
            and lo_pca < lo_none - 0.08
            and (hi_pca > 0.72)
            and (sh_fit > 0.9)
            and (abs(sh_te - 0.5) < 0.1),
        )
    )
    tests.append(
        (
            f"sensibilidad con señal plantada: δ=0 → {sv['0']:.2f} (azar), δ=1 → {sv['1']:.2f}, δ=2 → {sv['2']:.2f} (crece con δ)",
            abs(sv["0"] - 0.5) < 0.1 and sv["2"] > sv["1"] > sv["0"] and (sv["2"] > 0.75),
        )
    )

    def first_word_acc(make):
        (Xa, ya), (Xb, yb) = (make(600, 5), make(400, 6))
        votes = {}
        for t, c in zip(Xa, ya):
            votes.setdefault(t.split(" ")[0], []).append(c)
        pred = [float(np.mean(votes[w]) > 0.5) if (w := t.split(" ")[0]) in votes else 0.5 for t in Xb]
        return float(np.mean([1.0 if p == c else 0.5 if p == 0.5 else 0.0 for p, c in zip(pred, yb)]))

    def no_window(n, seed):
        rr = np.random.default_rng([seed, 8])
        base = _mixed(n // 2, seed, "en")
        return (base + [shuffle_words(t, rr) for t in base], [0] * len(base) + [1] * len(base))

    fw_old, fw_new = (first_word_acc(no_window), first_word_acc(task_orden))
    tests.append(
        (
            f"orden: quien solo mira la primera palabra acierta {fw_old:.2f} sin ventana y {fw_new:.2f} con ventana",
            fw_old > 0.75 and fw_new < 0.62,
        )
    )
    T = build_tasks(SMOKE, None, None, 1)
    bal = all(
        (len(set(np.bincount(t[k][1], minlength=t["C"]))) == 1 for t in T.values() for k in ("train", "test"))
    )
    tests.append(
        (
            f"tareas: clases balanceadas · repetidos entrenamiento/prueba: { {n: t['train_test_dups'] for n, t in T.items()}}",
            bal and all((t["train_test_dups"] == 0 for t in T.values())),
        )
    )
    torch = DP._torch()
    torch.manual_seed(0)
    ex = DP.Extractor(
        DP._fake_model(D=24, V=140)(), lambda t: [ord(ch) % 130 + 2 for ch in t], "empty", bos_id=1
    )
    task = T["orden"]
    Fe = features(ex, task["test"][0][:8])
    n = len(task["test"][0]) // 2
    nat, shf = (
        features(ex, task["test"][0][:6])["emb"][:, -1],
        features(ex, task["test"][0][n : n + 6])["emb"][:, -1],
    )
    tests.append(
        (
            f"orden a prueba de bolsa: emb promedio del texto natural == el del barajado (Δ máx {np.abs(nat - shf).max():.1e})",
            Fe["emb"].shape == (8, len(KS), 24) and np.abs(nat - shf).max() < 1e-05,
        )
    )
    res = probe_model(ex, task, pca=8, sens=True)
    tests.append(
        (
            f"orden en el juguete: emb con todo el texto {res['todo']['emb']:.2f} (azar por construcción), mejor flujo {res['todo']['acc']:.2f} en `{res['todo']['stream']}` · {res['todo']['reducer']}, barajada en prueba {res['todo']['shuf_test']:.2f}, sensibilidad {res['todo']['sensitivity']}",
            abs(res["todo"]["emb"] - 0.5) <= 0.2
            and "shuf_fit" in res["4"]
            and (set(res["todo"]["sensitivity"]) == {"0", "0.5", "1", "2"}),
        )
    )
    ex.close()
    lo, hi = paired_ci(np.r_[np.ones(60), np.zeros(40)], np.r_[np.ones(50), np.zeros(50)])
    tests.append(
        (f"IC pareado: 0.60 contra 0.50 con los mismos textos -> ({lo:+.2f}, {hi:+.2f})", lo > 0 and hi < 0.2)
    )
    print(f"text_fingerprint v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{('ok' if ok else 'FAIL')}] {name}")
    return 0 if all((ok for _, ok in tests)) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Sondas de huella textual (exploratorio)")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run")
    p.add_argument("--script", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--val-shard", help="fragmento de validación de FineWeb, para la huella de orden sobre texto real"
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        res = run(
            a.script,
            a.ckpt,
            a.tokenizer,
            a.out_dir,
            SMOKE if a.smoke else CFG,
            a.val_shard,
            a.device,
            a.smoke,
        )
        print(
            open(
                os.path.join(a.out_dir, ("HUELLA_SMOKE" if a.smoke else "HUELLA") + ".md"), encoding="utf-8"
            ).read()
        )
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
