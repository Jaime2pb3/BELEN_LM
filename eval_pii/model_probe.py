"""  
|---|---|
| `model_probe.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Care it. |
-----
"""


from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leak_forge as F
import context_grammar as G
import reference_card as B

VERSION = "belen-pii-0.2"
CKPT_FORMAT = "belen-raw-state-dict-v1"
MODULES = ("leak_forge.py", "context_grammar.py", "reference_card.py", "model_probe.py")
SMALL_ENV = {
    "MODEL_DIM": "64",
    "NUM_LAYERS": "4",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "2",
    "MLP_MULT": "2",
    "RESONANCE_SLOTS": "4",
    "RESONANCE_MEM_SPAN": "2",
    "PERCEPTION_HEADS": "4",
    "DENDRITE_LOCAL_SPAN": "8",
    "DENDRITE_MID_SPAN": "16",
    "VOCAB_SIZE": "1024",
}
DEFAULT_CONFIG = {
    "langs": ["en"],
    "seed_train": 20261111,
    "seed_val": 20261112,
    "seed_test": 20261113,
    "n_leak": 3000,
    "n_ctx": 500,
    "n_dose": 300,
    "n_groups_val": 90,
    "n_groups_test": 240,
    "lod_reps": 30,
    "probe_seeds": [0, 1, 2],
    "epochs": 5,
    "random_init_seeds": [7, 8, 9],
    "memory": "frozen",
}
SMOKE_CONFIG = dict(
    DEFAULT_CONFIG,
    n_leak=240,
    n_ctx=45,
    n_dose=36,
    n_groups_val=18,
    n_groups_test=24,
    lod_reps=4,
    probe_seeds=[0, 1],
    epochs=2,
    random_init_seeds=[7, 8],
)


def _torch():
    import torch

    return torch


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_script(path: str, env: Optional[Dict[str, str]] = None, name: str = "belen_script"):
    if env:
        os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def hparams_of(obj) -> Dict:
    simple = (str, int, float, bool, type(None))
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        v = getattr(obj, k)
        if isinstance(v, simple):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all((isinstance(x, simple) for x in v)):
            out[k] = list(v)
    return out


def build_model(mod, hparams: Optional[Dict] = None, init_seed: Optional[int] = None, device: str = "cpu"):
    torch = _torch()
    args = mod.Hyperparameters()
    for k, v in (hparams or {}).items():
        setattr(args, k, v)
    if init_seed is not None:
        torch.manual_seed(init_seed)
    return (mod.Belen(args).float().eval().to(device), args)


def save_raw(model, hparams: Dict, path: str, extra: Optional[Dict] = None) -> str:
    torch = _torch()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "format": CKPT_FORMAT,
            "hparams": hparams,
            "extra": extra or {},
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        path,
    )
    return path


def load_raw(mod, path: str, device: str = "cpu"):
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fmt = str(payload.get("format", ""))
    if not (fmt == CKPT_FORMAT or fmt.startswith("ach-")):
        raise ValueError(f"formato de checkpoint inesperado: {fmt!r}")
    model, args = build_model(mod, payload["hparams"], device="cpu")
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"{e}\n[load_raw] Pista: los monitores PfV existen o no según PFV_MONITOR al construir. Usa el mismo valor que en el entrenamiento."
        ) from e
    return (model.to(device), payload)


def train_and_save(script: str, out_ckpt: str) -> str:
    mod = load_script(script, name="belen_train")
    if not hasattr(mod, "quantize_and_compress") or not hasattr(mod, "main"):
        raise SystemExit("[train] el script no tiene main() y quantize_and_compress(): no sé dónde enganchar")
    target = getattr(mod, "train_module", mod)
    orig, saved = (target.quantize_and_compress, {})

    def wrapped(model, *a, **k):
        extra = {
            "script": os.path.basename(script),
            "script_sha256": sha256(script),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        extra.update(
            {
                k: v
                for k, v in dict(getattr(mod, "RUN_SUMMARY", {}) or {}).items()
                if k in ("steps_done", "tokens_seen", "last_val", "legacy", "run_id")
            }
        )
        saved["path"] = save_raw(model, hparams_of(mod.Hyperparameters), out_ckpt, extra)
        print(f"[train] pesos crudos -> {out_ckpt}", flush=True)
        return orig(model, *a, **k)

    target.quantize_and_compress = wrapped
    mod.main()
    if "path" not in saved:
        raise SystemExit("[train] main() terminó sin llamar a quantize_and_compress: no se guardó nada")
    return saved["path"]


class Extractor:

    def __init__(
        self,
        model,
        encode: Callable[[str], List[int]],
        memory: str = "frozen",
        bos_id: Optional[int] = 1,
        max_tokens: int = 1024,
        train_seq_len: Optional[int] = None,
    ):
        torch = _torch()
        if memory not in ("frozen", "empty"):
            raise ValueError(
                "memory debe ser 'frozen' (la memoria que dejó el entrenamiento) o 'empty' (ablación)"
            )
        self.model, self.encode, self.bos, self.max_tokens = (model.eval(), encode, bos_id, max_tokens)
        self.train_seq_len, self.memory = (train_seq_len, memory)
        self.device = next(model.parameters()).device
        if memory == "empty":
            with torch.no_grad():
                for _, b in model.resonance.named_buffers():
                    b.zero_()
        self.snapshot = {n: b.detach().clone() for n, b in model.named_buffers()}
        self.mutated: set = set()
        self.n_forward = self.n_over = self.max_T = 0
        self._cap: Dict[str, object] = {}
        self._hooks = []
        first = lambda o: o[0] if isinstance(o, (tuple, list)) else o
        named = [("emb", model.input_norm), ("dendritic", model.dendrite), ("percept", model.perception)]
        named += [(f"soma{i}", blk) for i, blk in enumerate(model.blocks)]
        for name, module in named:
            self._hooks.append(
                module.register_forward_hook(
                    lambda _m, _i, out, name=name: self._cap.__setitem__(name, first(out).detach())
                )
            )
        self.streams = [n for n, _ in named] + ["final"]

    def close(self):
        for h in self._hooks:
            h.remove()

    def restore(self):
        torch = _torch()
        with torch.no_grad():
            for n, b in self.model.named_buffers():
                if not torch.equal(b, self.snapshot[n]):
                    b.copy_(self.snapshot[n])
                    self.mutated.add(n)

    def ids(self, text: str) -> List[int]:
        ids = list(self.encode(text))[: self.max_tokens]
        return ([self.bos] if self.bos is not None else []) + ids

    def run(
        self, text: str, streams: Optional[Sequence[str]] = None, restore: bool = True
    ) -> Dict[str, np.ndarray]:
        torch = _torch()
        ids = self.ids(text)
        off = 1 if self.bos is not None else 0
        want = list(streams or self.streams)
        if len(ids) <= off:
            D = self.model.tok_emb.weight.shape[1]
            return {s: np.zeros((0, D), np.float32) for s in want}
        self._cap.clear()
        with torch.no_grad():
            out = self.model._encode_decode(torch.tensor([ids], dtype=torch.long, device=self.device))
        self._cap["final"] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        if restore:
            self.restore()
        cache = getattr(self.model.dendrite, "_mask_cache", None)
        if cache is not None and len(cache) > 96:
            cache.clear()
        T = len(ids) - off
        self.n_forward += 1
        self.max_T = max(self.max_T, T)
        self.n_over += int(self.train_seq_len is not None and len(ids) > self.train_seq_len)
        return {s: self._cap[s][0, off:].float().cpu().numpy() for s in want}

    def states_fn(self, stream: str):
        return lambda text: self.run(text, [stream])[stream]

    def check_independence(self, probe_text: str, others: Sequence[str]) -> Dict:
        a = self.run(probe_text)
        for t in others:
            self.run(t)
        b = self.run(probe_text)
        frozen = max((float(np.abs(a[s] - b[s]).max()) for s in self.streams))
        for t in others:
            self.run(t, restore=False)
        c = self.run(probe_text, restore=False)
        self.restore()
        leak = {s: float(np.abs(a[s] - c[s]).max()) for s in self.streams}
        d = self.run(probe_text)
        back = max((float(np.abs(a[s] - d[s]).max()) for s in self.streams))
        return {
            "ok": frozen <= 1e-05 and back <= 1e-05,
            "max_diff_frozen": frozen,
            "max_diff_after_restore": back,
            "contamination_avoided": leak,
            "mutated_buffers": sorted(self.mutated),
        }

    def check_causality(self, text: str) -> Dict[str, float]:
        full = self.run(text)
        ids = self.encode(text)
        k = max(1, len(ids) // 2)
        torch = _torch()
        self._cap.clear()
        pre = ([self.bos] if self.bos is not None else []) + list(ids[:k])
        with torch.no_grad():
            out = self.model._encode_decode(torch.tensor([pre], dtype=torch.long, device=self.device))
        self._cap["final"] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        off = 1 if self.bos is not None else 0
        part = {s: self._cap[s][0, off:].float().cpu().numpy() for s in self.streams}
        self.restore()
        # Δ relativo a la escala del flujo: el residual llega a ~1e4 en capas profundas y un umbral
        # absoluto confunde ruido de kernel bf16 con dependencia del futuro.
        return {
            s: float(np.abs(full[s][:k] - part[s]).max() / (np.abs(full[s][:k]).max() + 1e-06))
            for s in self.streams
        }

    def stream_norms(self, text: str) -> Dict[str, float]:
        r = self.run(text)
        return {s: float(np.median(np.linalg.norm(r[s], axis=-1))) for s in self.streams if len(r[s])}


def tokenizer_equivalences(
    encode: Callable[[str], List[int]], lang: str = "en", n: int = 120, seed: int = 5
) -> Dict:
    evs = list(F.EVASIONS)
    same = {}
    for i, a in enumerate(evs):
        for b in evs[i + 1 :]:
            eq = 0
            for k in range(n):
                t = F.PII_TYPES[k % len(F.PII_TYPES)]
                d = float(F.DOSES[1 + k % (len(F.DOSES) - 1)])
                sa = F.make_sample(
                    np.random.default_rng([seed, k]), t, d, "scatter", a, "form", lang=lang
                ).text
                sb = F.make_sample(
                    np.random.default_rng([seed, k]), t, d, "scatter", b, "form", lang=lang
                ).text
                eq += int(list(encode(sa)) == list(encode(sb)))
            same[f"{a}={b}"] = eq / n
    canon, alias = ([], {})
    for e in evs:
        hit = next((c for c in canon if same.get(f"{c}={e}", 0.0) >= 0.99), None)
        if hit is None:
            canon.append(e)
        else:
            alias[e] = hit
    return {"canonical": canon, "aliases": alias, "identical_fraction": same}


def token_stats(encode, texts: Sequence[str], unk_id: Optional[int] = None) -> Dict:
    L = [len(encode(t)) for t in texts]
    unk = (
        float(np.mean([np.mean([i == unk_id for i in encode(t)] or [0]) for t in texts]))
        if unk_id is not None
        else None
    )
    return {
        "median": int(np.median(L)),
        "p95": int(np.percentile(L, 95)),
        "max": int(max(L)),
        "unk_rate": unk,
        "chars_per_token": float(sum(map(len, texts)) / max(sum(L), 1)),
    }


class StdProbe:

    def __init__(self, probe: B.FactorProbe, mu: np.ndarray, sd: np.ndarray):
        self.probe, self.mu, self.sd = (probe, mu, sd)
        self.fit: Dict[str, float] = {}

    def z(self, H: np.ndarray) -> np.ndarray:
        return (np.asarray(H, np.float32) - self.mu) / self.sd

    def prob(self, H: np.ndarray) -> float:
        return self.probe.prob(self.z(H)) if len(H) else 0.0


def _fit_quality(probe: B.FactorProbe, Z: List[np.ndarray], ev: Sequence, role: Sequence) -> Dict[str, float]:
    f = [probe.forward(h) for h in Z]

    def ba(pred, lab):
        pred, lab = (np.asarray(pred, bool), np.asarray(lab, bool))
        if lab.all() or not lab.any():
            return 0.5
        return float(0.5 * (pred[lab].mean() + (~pred[~lab]).mean()))

    keep = [i for i, r in enumerate(role) if r is not None]
    return {
        "ev_ba": ba([x["E"] >= 0 for x in f], [e >= 0.5 for e in ev]),
        "role_ba": ba([f[i]["R"] >= 0 for i in keep], [role[i] >= 0.5 for i in keep]),
    }


def fit_stream(Hs: List[np.ndarray], ev, role, y, seeds: Sequence[int], epochs: int) -> Dict[str, StdProbe]:
    cat = np.concatenate([np.asarray(h, np.float32) for h in Hs[:400] if len(h)], 0)
    mu, sd = (cat.mean(0), cat.std(0) + 1e-06)
    keep = [i for i, h in enumerate(Hs) if len(h)]
    Zk = [(np.asarray(Hs[i], np.float32) - mu) / sd for i in keep]
    lab = lambda v: [v[i] for i in keep]
    out = {}
    for s in seeds:
        p = StdProbe(
            B.FactorProbe(Zk[0].shape[1], True, seed=s).fit(
                Zk, lab(ev), lab(role), lab(y), epochs=epochs, seed=s
            ),
            mu,
            sd,
        )
        p.fit = _fit_quality(p.probe, Zk, lab(ev), lab(role))
        out[f"s{s}"] = p
    perm = np.random.default_rng(991).permutation(len(keep))
    sh = lambda v: [lab(v)[j] for j in perm]
    p = StdProbe(
        B.FactorProbe(Zk[0].shape[1], True, seed=seeds[0]).fit(
            Zk, sh(ev), sh(role), sh(y), epochs=epochs, seed=seeds[0]
        ),
        mu,
        sd,
    )
    p.fit = _fit_quality(p.probe, Zk, sh(ev), sh(role))
    out["shuf"] = p
    return out


def fit_all(
    ex: Extractor,
    texts: List[str],
    ev,
    role,
    y,
    seeds: Sequence[int],
    epochs: int,
    budget_gb: float = 3.0,
    log: Callable = print,
) -> Dict[str, StdProbe]:
    D = ex.model.tok_emb.weight.shape[1]
    toks = sum((len(ex.encode(t)) for t in texts[:200])) / max(min(len(texts), 200), 1) * len(texts)
    per_stream_gb = toks * D * 4 / 1000000000.0
    k = max(1, int(budget_gb / max(per_stream_gb, 1e-09)))
    probes: Dict[str, StdProbe] = {}
    for g0 in range(0, len(ex.streams), k):
        group = ex.streams[g0 : g0 + k]
        store = {s: [] for s in group}
        for t in texts:
            r = ex.run(t, group)
            for s in group:
                h = np.asarray(r[s], np.float32)
                if not np.isfinite(h).all():
                    raise FloatingPointError(f"estado no finito en el flujo {s}")
                store[s].append(h)
        for s in group:
            for kind, p in fit_stream(store[s], ev, role, y, seeds, epochs).items():
                probes[f"{s}#{kind}"] = p
            store[s] = None
            log(f"[sondas] {s}: {len(seeds)} semillas + control barajado")
    return probes


class Bank:

    def __init__(self, ex: Extractor, probes: Dict[str, StdProbe]):
        self.ex, self.probes, self.names = (ex, probes, list(probes))
        self.streams = sorted({n.split("#")[0] for n in self.names}, key=ex.streams.index)
        self.memo: Dict[str, np.ndarray] = {}

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        out = np.empty((len(texts), len(self.names)))
        for i, t in enumerate(texts):
            if t not in self.memo:
                st = self.ex.run(t, self.streams)
                self.memo[t] = np.array([self.probes[n].prob(st[n.split("#")[0]]) for n in self.names])
            out[i] = self.memo[t]
        return out

    def detector(self, name: str):
        j = self.names.index(name)
        return lambda texts: self.scores(list(texts))[:, j]


class Columns:

    def __init__(self, dets: Dict[str, Callable]):
        self.dets, self.names = (dets, list(dets))

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([np.asarray(self.dets[n](list(texts)), float) for n in self.names], 1)


def dose_set(n: int, seed: int, lang: str, evasions: Sequence[str], doses=(0.0, 0.2, 1.0)) -> Dict:
    r = np.random.default_rng([seed, 31])
    out = {}
    for d in doses:
        out[str(d)] = [
            F.make_sample(
                r,
                F.PII_TYPES[i % len(F.PII_TYPES)],
                d,
                str(r.choice(F.OPERATORS)),
                evasions[i // len(F.PII_TYPES) % len(evasions)],
                str(r.choice(F.CARRIERS)),
                lang=lang,
            ).text
            for i in range(n)
        ]
    out["decoy"] = [F.make_decoy_only(r, str(r.choice(F.CARRIERS)), lang).text for _ in range(n)]
    return out


def measure(src, cfg: Dict, seed: int, lang: str, evasions: Sequence[str], n_groups: int) -> Dict:
    ds = dose_set(cfg["n_dose"], seed, lang, evasions)
    raw = {k: src.scores(v) for k, v in ds.items()}
    res = {
        "names": src.names,
        "dose": {k: (v >= 0.5).astype(float) for k, v in raw.items()},
        "ctx": {},
        "score_sd": np.concatenate(list(raw.values()), 0).std(0),
    }
    for cond in G.CONDITIONS:
        gs = G.groups(n_groups, seed, cond, lang)
        flat = [v["text"] for g in gs for v in g["variants"]]
        S = src.scores(flat)
        per, pos = (np.empty((len(gs), S.shape[1])), 0)
        for gi, g in enumerate(gs):
            block = S[pos : pos + len(g["variants"])]
            pos += len(g["variants"])
            w = [(block[i] > block[j]) + 0.5 * (block[i] == block[j]) for i, j in g["pairs"]]
            per[gi] = np.mean(w, 0)
        res["ctx"][cond] = per
    return res


def stream_view(m: Dict, stream: str, seeds: Sequence[int], kind: str = "probe") -> Dict:
    cols = (
        [m["names"].index(f"{stream}#shuf")]
        if kind == "shuf"
        else [m["names"].index(f"{stream}#s{s}") for s in seeds]
    )
    dose = {k: v[:, cols] for k, v in m["dose"].items()}
    j02 = [float(dose["0.2"][:, c].mean() - dose["0.0"][:, c].mean()) for c in range(len(cols))]
    return {
        "h": {k: v.mean(1) for k, v in dose.items()},
        "ctx": {c: v[:, cols].mean(1) for c, v in m["ctx"].items()},
        "J02_range": (min(j02), max(j02)),
        "ctx_range": {
            c: (float(v[:, cols].mean(0).min()), float(v[:, cols].mean(0).max())) for c, v in m["ctx"].items()
        },
    }


def col_view(m: Dict, name: str) -> Dict:
    j = m["names"].index(name)
    return {"h": {k: v[:, j] for k, v in m["dose"].items()}, "ctx": {c: v[:, j] for c, v in m["ctx"].items()}}


def J(view: Dict, d: str = "0.2") -> float:
    return float(view["h"][d].mean() - view["h"]["0.0"].mean())


def boot_J(
    view: Dict, other: Optional[Dict] = None, d: str = "0.2", n_boot: int = 2000, seed: int = 0
) -> Tuple[float, float]:
    r = np.random.default_rng(seed)
    n0, n1 = (len(view["h"]["0.0"]), len(view["h"][d]))
    i0, i1 = (r.integers(0, n0, (n_boot, n0)), r.integers(0, n1, (n_boot, n1)))
    f = lambda v: v["h"][d][i1].mean(1) - v["h"]["0.0"][i0].mean(1)
    x = f(view) - (f(other) if other is not None else 0.0)
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))


def boot_ctx(a: np.ndarray, b: Optional[np.ndarray] = None, seed: int = 0) -> Tuple[float, float]:
    return G.boot_ci(a - (b if b is not None else 0.0), seed=seed)


def summarize(view: Dict) -> Dict:
    return {
        "fpr_blank": float(view["h"]["0.0"].mean()),
        "fpr_decoy": float(view["h"]["decoy"].mean()),
        "det_0.2": float(view["h"]["0.2"].mean()),
        "det_1.0": float(view["h"]["1.0"].mean()),
        "J02": J(view),
        "J02_ci": boot_J(view),
        "ctx": {c: {"acc": float(v.mean()), "ci": boot_ctx(v)} for c, v in view["ctx"].items()},
    }


HYP = ("evidence", "cong_lex", "contrast", "incong")


def dead_streams(val: Dict, streams: Sequence[str], seeds: Sequence[int]) -> List[str]:
    sd = val.get("score_sd")
    if sd is None:
        return []
    return [s for s in streams if all((sd[val["names"].index(f"{s}#s{k}")] < 1e-09 for k in seeds))]


def select(val: Dict, streams: Sequence[str], seeds: Sequence[int]) -> Dict:
    dead = dead_streams(val, streams, seeds)
    live = [s for s in streams if s not in dead] or list(streams)
    views = {s: stream_view(val, s, seeds) for s in live}
    out = {"evidence": max(live, key=lambda s: J(views[s]))}
    for c in ("cong_lex", "contrast", "incong"):
        out[c] = max(live, key=lambda s: float(views[s]["ctx"][c].mean()))
    out["dead"] = dead
    return out


def hyp_views(m: Dict, sel: Dict, seeds: Sequence[int]) -> Dict[str, Dict]:
    return {h: stream_view(m, sel[h], seeds) for h in HYP}


BAR = {"cong_lex": 0.65, "contrast": 0.6, "incong": 0.6}
MARGIN = 0.05
CRITERIA = 'Criterios congelados. Se evalúan en PRUEBA, sobre el flujo elegido en VALIDACIÓN.\n"Control" = la misma arquitectura con pesos aleatorios. Hay varias inicializaciones y SIEMPRE se compara\ncontra la más fuerte en esa medida. Un veredicto positivo exige DOS cosas a la vez:\n  (a) ventaja pareada sobre el control más fuerte: el IC95 de la diferencia queda por encima de 0.05\n  (b) nivel absoluto: el IC95 inferior supera el listón de la condición\nLos listones son 0.5 más la tolerancia con la que la auditoría certifica la condición (instrumentos sin\ncomprensión llegan hasta ahí por sesgos residuales): dueños nuevos 0.65, contraste 0.60, incongruente 0.60.\n\n  H1 EVIDENCIA   J(0.2) entrenado − control. "El entrenamiento aporta" si (a). "No aporta" si el IC95 de la\n                 diferencia queda por debajo de 0.05. Si lo cruza: indeterminado. (Sin listón absoluto.)\n  H2 LÉXICO      exactitud de par en dueños nuevos. "El entrenamiento aporta" si (a) y (b).\n                 "No aporta" si el IC de la diferencia queda por debajo de 0.05. Si no: indeterminado.\n  H3 NEGACIÓN    exactitud de par en contraste. "Lee la negación" si (a) y (b). Si no: "no la lee".\n  H4 COMPOSICIÓN exactitud de par en incongruente. "COMPONE" si (a) y (b).\n                 Si el IC95 superior queda por debajo de 0.40, el flujo sigue palabras clave. Eso DESCRIBE al\n                 flujo, y un modelo sin entrenar también puede hacerlo (una bolsa de tokens lo hace). Solo se le\n                 atribuye al entrenamiento si además queda por debajo del control MÁS DÉBIL: IC95 de la\n                 diferencia pareada por debajo de -0.05. Entonces: "SIGUE PALABRAS CLAVE, más que los controles".\n                 Si no: "sigue palabras clave, igual que un modelo sin entrenar". Sobre 0.40: indeterminado.\n  H5 CAPACIDAD   la sonda entrenada con etiquetas BARAJADAS, medida sobre sus propios datos de entrenamiento\n                 (exactitud balanceada de cada factor). "Sonda selectiva" si no pasa de 0.60 en ninguno.\n                 Si pasa, puede ajustar ruido: memoriza, y H1 a H4 se marcan "con cautela".\n  Una condición que la auditoría de la gramática no certifica en esta corrida no da veredicto.\n  J(d) = tasa de detección a dosis d − tasa en blancos (d = 0), con umbral fijo 0.5.\n  Todos los IC son bootstrap al 95%: por textos en dosis, por grupos en contexto. Las diferencias son pareadas.\n  El IC NO incluye la variación entre semillas de ENTRENAMIENTO: un solo checkpoint es una sonda, no una prueba.\n\nROTACIÓN NULA Y RESOLUCIÓN\n  Cada control aleatorio se evalúa con estos mismos criterios como si fuera el entrenado, contra los demás.\n  Si algún control obtiene un positivo en una hipótesis, el positivo del entrenado en ESA hipótesis pasa a\n  "NO DISTINGUIBLE DEL RUIDO DE INICIALIZACIÓN". Lo mismo vale para el negativo atribuido de H4.\n  Con K controles, lo mínimo que se puede afirmar frente al ruido de inicialización es p = 1/(K+1).\n  K = 3 da p >= 0.25: toda atribución al entrenamiento (positiva o negativa) es PROVISIONAL. Para afirmarla\n  con p <= 0.05 hacen falta K >= 19.'
PREDICTIONS = 'Predicciones de Claude, escritas el 10-nov-2026 ANTES de cualquier corrida en GPU:\n  P1 La evidencia casi no necesita entrenamiento: el modelo aleatorio dará J(0.2) >= 0.6 y H1 saldrá\n     "no aporta" o indeterminado. Ver un dígito junto a un asterisco es identidad de token, y una\n     proyección aleatoria la conserva.\n  P2 H2 saldrá "el entrenamiento aporta": saber que juror es una persona exige haber leído inglés.\n  P3 H4 NO saldrá "COMPONE". Un LM de ~22M parámetros, con sonda lineal y sin un solo ejemplo de la\n     construcción, no va a separar núcleo de argumento. Si sale "COMPONE", es lo más interesante de\n     la ficha y hay que replicarlo con otra semilla de entrenamiento antes de creerlo.\n  P4 H3 saldrá "no la lee".\n  P5 El flujo percept quedará cerca del piso (emb), por el gate de percepción colapsado que se vio\n     en el run de 6500 steps, si el checkpoint es de esa familia.\n  P6 En español todo lo léxico saldrá peor que en inglés.\nPredicciones de Jaime (escríbelas aquí antes de correr `ficha`):\n  J1\n  J2'
NEG_ATTR = "SIGUE PALABRAS CLAVE, más que los controles"
NEG_DESC = "sigue palabras clave, igual que un modelo sin entrenar"


def tri(ci: Tuple[float, float], thr: float, yes: str, no: str) -> str:
    return yes if ci[0] > thr else no if ci[1] < thr else "indeterminado"


def verdict(tr: Dict[str, Dict], rnds: Sequence[Dict[str, Dict]], capacity: Dict, certifies: Dict) -> Dict:
    selective = max(capacity["role_ba"], capacity["ev_ba"]) <= 0.6
    cert = lambda c: bool(certifies.get(c, False))
    ev_ctl = max((r["evidence"] for r in rnds), key=J)
    d1 = boot_J(tr["evidence"], ev_ctl)
    js = [J(r["evidence"]) for r in rnds]
    out = {
        "H1": {
            "value": J(tr["evidence"]),
            "delta": J(tr["evidence"]) - max(js),
            "ci": d1,
            "control_range": (min(js), max(js)),
            "verdict": tri(d1, MARGIN, "el entrenamiento aporta", "no aporta"),
        }
    }

    def ctx(cond: str) -> Dict:
        mine = tr[cond]["ctx"][cond]
        accs = [float(r[cond]["ctx"][cond].mean()) for r in rnds]
        ctl = rnds[int(np.argmax(accs))][cond]["ctx"][cond]
        d, ab = (boot_ctx(mine, ctl), boot_ctx(mine))
        return {
            "acc": float(mine.mean()),
            "ci": ab,
            "delta": float(mine.mean() - max(accs)),
            "delta_ci": d,
            "control_range": (min(accs), max(accs)),
            "bar": BAR[cond],
            "passes": bool(d[0] > MARGIN and ab[0] > BAR[cond]),
            "below_margin": bool(d[1] < MARGIN),
        }

    h2, h3, h4 = (ctx("cong_lex"), ctx("contrast"), ctx("incong"))
    h2["verdict"] = (
        "NO CERTIFICADA"
        if not cert("cong_lex")
        else (
            "el entrenamiento aporta"
            if h2["passes"]
            else "no aporta" if h2["below_margin"] else "indeterminado"
        )
    )
    h3["verdict"] = (
        "NO CERTIFICADA" if not cert("contrast") else "lee la negación" if h3["passes"] else "no la lee"
    )
    inc_accs = [float(r["incong"]["ctx"]["incong"].mean()) for r in rnds]
    weakest = rnds[int(np.argmin(inc_accs))]["incong"]["ctx"]["incong"]
    h4["delta_weakest_ci"] = boot_ctx(tr["incong"]["ctx"]["incong"], weakest)
    h4["below_chance"] = bool(h4["ci"][1] < 0.4)
    h4["attributed_negative"] = bool(
        cert("incong") and h4["below_chance"] and (h4["delta_weakest_ci"][1] < -MARGIN)
    )
    h4["verdict"] = (
        "NO CERTIFICADA"
        if not cert("incong")
        else (
            "COMPONE"
            if h4["passes"]
            else (
                NEG_ATTR if h4["attributed_negative"] else NEG_DESC if h4["below_chance"] else "indeterminado"
            )
        )
    )
    out.update(
        {
            "H2": h2,
            "H3": h3,
            "H4": h4,
            "H5": {
                "shuffled_role_ba": capacity["role_ba"],
                "shuffled_ev_ba": capacity["ev_ba"],
                "real_role_ba": capacity.get("real_role_ba"),
                "real_ev_ba": capacity.get("real_ev_ba"),
                "verdict": "sonda selectiva" if selective else "LA SONDA MEMORIZA",
            },
        }
    )
    out["caution"] = not selective
    out["positives"] = [
        h
        for h in ("H1", "H2", "H3", "H4")
        if out[h]["verdict"] in ("el entrenamiento aporta", "lee la negación", "COMPONE")
    ]
    out["negatives"] = ["H4"] if out["H4"]["verdict"] == NEG_ATTR else []
    return out


def null_rotation(ctl_views: Sequence[Dict[str, Dict]], ctl_caps: Sequence[Dict], certifies: Dict) -> Dict:
    K = len(ctl_views)
    if K < 2:
        return {
            "K": K,
            "rotations": [],
            "false_positives": [],
            "neg_rotations": [],
            "false_negatives": [],
            "p_floor": 1.0 / (K + 1),
        }
    rot, neg = ([], [])
    for i in range(K):
        others = [v for j, v in enumerate(ctl_views) if j != i]
        vi = verdict(ctl_views[i], others, ctl_caps[i], certifies)
        rot.append(vi["positives"])
        neg.append(vi["negatives"])
    return {
        "K": K,
        "rotations": rot,
        "false_positives": sorted({h for r in rot for h in r}),
        "neg_rotations": neg,
        "false_negatives": sorted({h for r in neg for h in r}),
        "p_floor": 1.0 / (K + 1),
    }


def apply_null(v: Dict, null: Dict) -> Dict:
    v["null"] = null
    neg_deg = [h for h in v.get("negatives", []) if h in null.get("false_negatives", [])]
    v["degraded"] = [h for h in v["positives"] if h in null["false_positives"]] + neg_deg
    for h in v["degraded"]:
        v[h]["verdict"] = "NO DISTINGUIBLE DEL RUIDO DE INICIALIZACIÓN"
    v["positives"] = [h for h in v["positives"] if h not in v["degraded"]]
    v["negatives"] = [h for h in v.get("negatives", []) if h not in neg_deg]
    v["attributed"] = v["positives"] + [h + " (negativo)" for h in v["negatives"]]
    v["provisional"] = bool(v["attributed"]) and null["p_floor"] > 0.05
    return v


def fingerprint(script: str, ckpt: Optional[str], tokenizer: str) -> Dict[str, str]:
    fp = {m: sha256(os.path.join(HERE, m)) for m in MODULES}
    fp["script:" + os.path.basename(script)] = sha256(script)
    pkg = os.path.join(os.path.dirname(os.path.abspath(script)), "belen")
    if os.path.isdir(pkg):
        for name in sorted(os.listdir(pkg)):
            if name.endswith(".py"):
                fp["belen/" + name] = sha256(os.path.join(pkg, name))
    fp["tokenizer:" + os.path.basename(tokenizer)] = sha256(tokenizer)
    if ckpt:
        fp["ckpt:" + os.path.basename(ckpt)] = sha256(ckpt)
    return fp


def write_prereg(out_dir: str, script: str, ckpt: str, tokenizer: str, cfg: Dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    md, js = (os.path.join(out_dir, "PREREG.md"), os.path.join(out_dir, "prereg.json"))
    if os.path.exists(js):
        raise SystemExit(
            f"[prereg] ya existe {js}. El prerregistro se escribe UNA vez. Usa otro --out-dir para otra corrida."
        )
    fp = fingerprint(script, ckpt, tokenizer)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(js, "w") as f:
        json.dump({"version": VERSION, "written_at": stamp, "fingerprint": fp, "config": cfg}, f, indent=1)
    with open(md, "w", encoding="utf-8") as f:
        f.write(
            f"# Prerregistro — BELENC-PII sobre BELEN\n\nEscrito: {stamp} · {VERSION}\n\n## Configuración congelada\n\n```json\n{json.dumps(cfg, indent=1)}\n```\n\n## Criterios\n\n```\n{CRITERIA}\n```\n\n## Predicciones\n\n```\n{PREDICTIONS}\n```\n\n## Huellas (sha256)\n\n"
            + "\n".join((f"- `{k}` {v}" for k, v in fp.items()))
            + "\n"
        )
    return md


def check_prereg(out_dir: str, script: str, ckpt: str, tokenizer: str) -> Tuple[Dict, List[str]]:
    js = os.path.join(out_dir, "prereg.json")
    if not os.path.exists(js):
        raise SystemExit(f"[ficha] no hay prerregistro en {out_dir}. Corre primero `prereg`.")
    pre = json.load(open(js))
    now = fingerprint(script, ckpt, tokenizer)
    drift = sorted((k for k in set(pre["fingerprint"]) | set(now) if pre["fingerprint"].get(k) != now.get(k)))
    return (pre, drift)


def run_arm(
    tag: str, ex: Extractor, cfg: Dict, lang: str, evasions: Sequence[str], log: Callable = print
) -> Dict:
    texts, ev, role, y = B.training_set(cfg["n_leak"], cfg["n_ctx"], cfg["seed_train"], lang, evasions)
    probes = fit_all(
        ex, texts, ev, role, y, cfg["probe_seeds"], cfg["epochs"], log=lambda m: log(f"[{tag}/{lang}] {m}")
    )
    bank = Bank(ex, probes)
    val = measure(bank, cfg, cfg["seed_val"], lang, evasions, cfg["n_groups_val"])
    sel = select(val, ex.streams, cfg["probe_seeds"])
    bank.memo.clear()
    test = measure(bank, cfg, cfg["seed_test"], lang, evasions, cfg["n_groups_test"])
    log(
        f"[{tag}/{lang}] elegido en validación: "
        + " ".join((f"{h}={sel[h]}" for h in HYP))
        + (f" · muertos: {sel['dead']}" if sel["dead"] else "")
    )
    cap = {
        s: {
            "role_ba": probes[f"{s}#shuf"].fit["role_ba"],
            "ev_ba": probes[f"{s}#shuf"].fit["ev_ba"],
            "real_role_ba": float(np.mean([probes[f"{s}#s{k}"].fit["role_ba"] for k in cfg["probe_seeds"]])),
            "real_ev_ba": float(np.mean([probes[f"{s}#s{k}"].fit["ev_ba"] for k in cfg["probe_seeds"]])),
        }
        for s in ex.streams
    }
    return {
        "train_texts": texts,
        "train_y": y,
        "bank": bank,
        "val": val,
        "test": test,
        "select": sel,
        "capacity": cap,
    }


def leak_audit(train_texts: Sequence[str], train_y: Sequence[float], cfg: Dict, lang: str, evasions) -> Dict:
    tr = {t: yy for t, yy in zip(train_texts, train_y)}
    ds = dose_set(cfg["n_dose"], cfg["seed_test"], lang, evasions)
    test = [(t, k != "0.0" and k != "decoy") for k, v in ds.items() for t in v]
    test += [
        (v["text"], bool(v["label"]))
        for c in G.CONDITIONS
        for g in G.groups(cfg["n_groups_test"], cfg["seed_test"], c, lang)
        for v in g["variants"]
    ]
    dup = [(t, pos) for t, pos in test if t in tr]
    return {
        "n_test": len(test),
        "duplicates": len(dup),
        "positive_duplicates": sum((1 for _, pos in dup if pos)),
    }


def run_ficha(
    script: str,
    ckpt: str,
    tokenizer: str,
    out_dir: str,
    cfg: Dict,
    device: Optional[str] = None,
    smoke: bool = False,
    drift: Sequence[str] = (),
    log: Callable = print,
) -> Dict:
    torch = _torch()
    import sentencepiece as spm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=tokenizer)
    encode = lambda t: sp.encode(t)
    mod = load_script(script)
    model, payload = load_raw(mod, ckpt, device)
    hp = payload["hparams"]
    exs = {
        "entrenado": Extractor(
            model, encode, cfg["memory"], sp.bos_id(), train_seq_len=hp.get("train_seq_len")
        )
    }
    for rs in cfg["random_init_seeds"]:
        rnd_model, _ = build_model(mod, hp, init_seed=rs, device=device)
        exs[f"aleatorio#{rs}"] = Extractor(
            rnd_model, encode, "empty", sp.bos_id(), train_seq_len=hp.get("train_seq_len")
        )
    ctl_tags = [t for t in exs if t != "entrenado"]
    out = {
        "version": VERSION,
        "smoke": smoke,
        "config": cfg,
        "drift": list(drift),
        "arms": {},
        "controls": ctl_tags,
        "provenance": {
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "fingerprint": fingerprint(script, ckpt, tokenizer),
            "ckpt_extra": payload.get("extra", {}),
            "n_params": int(sum((p.numel() for p in model.parameters()))),
            "hparams": {
                k: hp.get(k)
                for k in (
                    "num_layers",
                    "model_dim",
                    "vocab_size",
                    "train_seq_len",
                    "resonance_slots",
                    "resonance_mode",
                    "iterations",
                )
            },
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    dump = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
    part = os.path.join(out_dir, ("SMOKE" if smoke else "MODEL_PROBE") + ".partial.json")
    key = {
        "config": json.loads(json.dumps(cfg)),
        "fingerprint": out["provenance"]["fingerprint"],
        "device": device,
    }
    done = {}
    if os.path.exists(part):
        prev = json.load(open(part))
        if prev.get("key") == key:
            done = prev["arms"]
        else:
            log("[ficha] hay un avance parcial de OTRA configuración o de otro código: se ignora")
    for lang in cfg["langs"]:
        if lang in done:
            out["arms"][lang] = done[lang]
            log(f"[ficha] idioma {lang}: retomado del avance guardado (mismas huellas y configuración)")
            continue
        for ex in exs.values():
            ex.n_forward = ex.n_over = 0
        eq = tokenizer_equivalences(encode, lang)
        evasions = eq["canonical"]
        ind = {}
        for tag, ex in exs.items():
            probe_t = F.make_sample(
                np.random.default_rng(1), "ID", 0.5, "suffix", "none", "form", lang=lang
            ).text
            others = [s.text for s in F.generate(12, seed=2, lang=lang)]
            ind[tag] = {
                "independence": ex.check_independence(probe_t, others),
                "causality": ex.check_causality(probe_t),
                "norms": ex.stream_norms(probe_t),
            }
            if not ind[tag]["independence"]["ok"]:
                raise SystemExit(
                    f"[ficha] {tag}: el estado de un texto depende de lo procesado antes (Δ {ind[tag]['independence']['max_diff_frozen']:.2e}). La memoria no quedó congelada. ABORTO."
                )
        audit = G.audit(lang, n=max(150, cfg["n_groups_test"]), seed=cfg["seed_test"])
        arms = {tag: run_arm(tag, ex, cfg, lang, evasions, log) for tag, ex in exs.items()}
        la = leak_audit(arms["entrenado"]["train_texts"], arms["entrenado"]["train_y"], cfg, lang, evasions)
        if la["positive_duplicates"]:
            raise SystemExit(
                f"[ficha] {la['positive_duplicates']} textos positivos de prueba están en el entrenamiento. ABORTO."
            )
        seeds = cfg["probe_seeds"]
        tr = arms["entrenado"]
        nb = F.NgramLogReg().fit(tr["train_texts"], [int(v) for v in tr["train_y"]])
        base = measure(
            Columns({"regex": F.regex_detector, "ngram": nb}),
            cfg,
            cfg["seed_test"],
            lang,
            evasions,
            cfg["n_groups_test"],
        )
        prof = {
            tag: {
                s: summarize(stream_view(a["test"], s, seeds))
                | {
                    "J02_range": stream_view(a["test"], s, seeds)["J02_range"],
                    "ctx_range": stream_view(a["test"], s, seeds)["ctx_range"],
                }
                for s in exs[tag].streams
            }
            for tag, a in arms.items()
        }

        def cap_of(a):
            c = a["capacity"]
            return {
                "role_ba": c[a["select"]["cong_lex"]]["role_ba"],
                "ev_ba": c[a["select"]["evidence"]]["ev_ba"],
                "real_role_ba": c[a["select"]["cong_lex"]]["real_role_ba"],
                "real_ev_ba": c[a["select"]["evidence"]]["real_ev_ba"],
            }

        ctl_views = [hyp_views(arms[t]["test"], arms[t]["select"], seeds) for t in ctl_tags]
        v = verdict(hyp_views(tr["test"], tr["select"], seeds), ctl_views, cap_of(tr), audit["certifies"])
        v = apply_null(v, null_rotation(ctl_views, [cap_of(arms[t]) for t in ctl_tags], audit["certifies"]))
        lod = {}
        for tag in ("entrenado", ctl_tags[0]):
            a = arms[tag]
            det = a["bank"].detector(f"{a['select']['evidence']}#s{seeds[0]}")
            lod[tag] = B.summarize_leak(
                F.evaluate(det, cfg["lod_reps"], cfg["seed_test"], evasions=evasions, lang=lang)
            )
        for name, det in (("regex", F.regex_detector), ("ngram", nb)):
            lod[name] = B.summarize_leak(
                F.evaluate(det, cfg["lod_reps"], cfg["seed_test"], evasions=evasions, lang=lang)
            )
        out["arms"][lang] = {
            "tokenizer": eq,
            "token_stats": token_stats(encode, tr["train_texts"][:600], sp.unk_id()),
            "checks": ind,
            "grammar_audit": {k: v_ for k, v_ in audit.items() if k != "checks"}
            | {"failed": [n for n, ok, _ in audit["checks"] if not ok]},
            "leak_audit": la,
            "select": {tag: a["select"] for tag, a in arms.items()},
            "profile": prof,
            "capacity": {tag: a["capacity"] for tag, a in arms.items()},
            "baselines": {n: summarize(col_view(base, n)) for n in base["names"]},
            "lod": lod,
            "verdict": v,
            "over_train_len": {tag: [ex.n_over, ex.n_forward] for tag, ex in exs.items()},
        }
        with open(part + ".tmp", "w") as f:
            json.dump({"key": key, "arms": out["arms"]}, f, default=dump)
        os.replace(part + ".tmp", part)
        log(f"[ficha] idioma {lang}: guardado en {os.path.basename(part)}")
    for ex in exs.values():
        ex.close()
    out = json.loads(json.dumps(out, default=dump))
    stem = "SMOKE" if smoke else "MODEL_PROBE"
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump(out, f, indent=1)
    md = report(out)
    with open(os.path.join(out_dir, f"{stem}.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    if os.path.exists(part):
        os.remove(part)
    return out


def trained_line(extra: Dict, hp: Dict) -> str:
    cfg = f"ITERATIONS configurado {hp.get('iterations')}"
    if "steps_done" in extra:
        tok = extra.get("tokens_seen")
        return (
            f"**{extra['steps_done']} pasos entrenados**"
            + (f" ({tok / 1000000.0:.0f}M tokens)" if tok else "")
            + f" · {cfg}"
        )
    if "ALEATORIOS" in str(extra.get("note", "")).upper():
        return f"**PESOS ALEATORIOS, 0 pasos entrenados** · {cfg}, sin efecto"
    return f"pasos entrenados NO registrados en el checkpoint · {cfg}, sin verificar"


def report(res: Dict) -> str:
    P, f2 = (res["provenance"], lambda x: "—" if x is None else f"{x:.2f}")
    ci = lambda c: f"({c[0]:+.2f}, {c[1]:+.2f})"
    L = [f"# BELENC-PII sobre BELEN — {VERSION}", ""]
    if res["smoke"]:
        L += [
            "> **CORRIDA DE HUMO. Tamaños mínimos y, si el modelo es chico o aleatorio, sin significado. Valida el cableado. NO es un resultado.**",
            "",
        ]
    if res["drift"]:
        L += [f"> **DERIVA DECLARADA respecto al prerregistro:** {', '.join(res['drift'])}", ""]
    hp = P["hparams"]
    L += [
        "## Procedencia",
        "",
        f"- Escrito {P['written_at']} · {P['device']}"
        + (f" ({P['gpu']})" if P["gpu"] else "")
        + f" · torch {P['torch']} · numpy {P['numpy']} · python {P['python']}",
        f"- Modelo: {P['n_params']:,} parámetros · {hp['num_layers']} capas · dim {hp['model_dim']} · vocab {hp['vocab_size']} · seq de entrenamiento {hp['train_seq_len']} · {trained_line(P.get('ckpt_extra') or {}, hp)}",
        f"- Memoria: entrenado = `{res['config']['memory']}` (la que dejó el entrenamiento, restaurada tras cada forward) · controles aleatorios ({len(res['controls'])} inicializaciones) = vacía",
        f"- Semillas: entrenamiento {res['config']['seed_train']}, validación {res['config']['seed_val']}, prueba {res['config']['seed_test']}, sonda {res['config']['probe_seeds']}, controles {res['config']['random_init_seeds']}",
        "- Huellas sha256: " + " · ".join((f"`{k}` {v[:12]}" for k, v in P["fingerprint"].items())),
        "",
    ]
    for lang, A in res["arms"].items():
        L += [
            f"## Idioma: {lang}"
            + (
                "  (el del entrenamiento)"
                if lang == "en"
                else "  (FUERA DE DISTRIBUCIÓN para un modelo entrenado en inglés)"
            ),
            "",
        ]
        ind = A["checks"]["entrenado"]["independence"]
        worst = max(ind["contamination_avoided"].items(), key=lambda kv: kv[1])
        caus = A["checks"]["entrenado"]["causality"]
        tk = A["token_stats"]
        L += [
            "**Verificaciones previas**",
            "",
            f"- Independencia de orden (memoria congelada): Δ máx {ind['max_diff_frozen']:.1e}. Sin congelar, el mismo texto habría cambiado hasta {worst[1]:.3g} en `{worst[0]}` tras 12 textos. Buffers que mutan en eval: {', '.join(('`' + b + '`' for b in ind['mutated_buffers'])) or 'ninguno'}",
            f"- Causalidad (el estado en t no depende del futuro): Δ relativo máx {max(caus.values()):.1e}"
            + (
                ""
                if max(caus.values()) <= 0.0001
                else f" · NO causal en: {[s for s, v in caus.items() if v > 0.0001]}"
            ),
            "- Norma del residual por flujo (mediana por token): "
            + " · ".join(f"`{k}` {v:.3g}" for k, v in A["checks"]["entrenado"].get("norms", {}).items()),
            f"- Tokenizer: {tk['chars_per_token']:.2f} caracteres por token · largo mediano {tk['median']}, p95 {tk['p95']}, máx {tk['max']} · tasa de desconocidos {f2(tk['unk_rate'])} · textos más largos que el entrenamiento: {A['over_train_len']['entrenado'][0]} de {A['over_train_len']['entrenado'][1]}",
            "- Evasiones que el tokenizer colapsa: "
            + (", ".join((f"`{a}` ≡ `{c}`" for a, c in A["tokenizer"]["aliases"].items())) or "ninguna")
            + f" · se evalúan: {A['tokenizer']['canonical']}",
            f"- Gramática: auditoría {('aprobada' if A['grammar_audit']['ok'] else 'CON FALLAS ' + str(A['grammar_audit']['failed']))} · certifican: {A['grammar_audit']['certifies']}",
            f"- Fuga entrenamiento/prueba: {A['leak_audit']['duplicates']} textos repetidos de {A['leak_audit']['n_test']} ({A['leak_audit']['positive_duplicates']} positivos)",
            "",
        ]
        V = A["verdict"]
        N = V["null"]
        flags = []
        if V["caution"]:
            flags.append("la sonda puede ajustar ruido (H5)")
        if V["degraded"]:
            flags.append(f"{V['degraded']} degradadas: un control aleatorio también las obtiene")
        if V["provisional"]:
            flags.append(
                f"atribuciones PROVISIONALES {V.get('attributed', V['positives'])}: con K = {N['K']} controles, p >= {N['p_floor']:.2f} frente al ruido de inicialización"
            )
        L += [
            "**Veredicto contra los criterios prerregistrados**"
            + ("  ⚠ " + " · ".join(flags) if flags else ""),
            "",
            "| | modelo evaluado (IC 95%) | controles (rango) | ventaja sobre el más fuerte (IC 95%) | listón | veredicto |",
            "|---|---|---|---|---|---|",
            f"| H1 evidencia, J(0.2) | {V['H1']['value']:+.2f} | {V['H1']['control_range'][0]:+.2f} a {V['H1']['control_range'][1]:+.2f} | {V['H1']['delta']:+.2f} {ci(V['H1']['ci'])} | — | **{V['H1']['verdict']}** |",
        ]
        for h, name in (
            ("H2", "dueños nuevos"),
            ("H3", "contraste (negación)"),
            ("H4", "incongruente (composición)"),
        ):
            x = V[h]
            L.append(
                f"| {h} {name} | {x['acc']:.2f} ({x['ci'][0]:.2f}, {x['ci'][1]:.2f}) | {x['control_range'][0]:.2f} a {x['control_range'][1]:.2f} | {x['delta']:+.2f} {ci(x['delta_ci'])} | {x['bar']:.2f} | **{x['verdict']}** |"
            )
        L += [
            f"| H5 capacidad: ajuste a etiquetas barajadas (papel / evidencia) | {V['H5']['shuffled_role_ba']:.2f} / {V['H5']['shuffled_ev_ba']:.2f} | con etiquetas reales {V['H5']['real_role_ba']:.2f} / {V['H5']['real_ev_ba']:.2f} | — | tope 0.60 | **{V['H5']['verdict']}** |",
            "",
            (
                f"H4 contra el control más débil: {ci(V['H4']['delta_weakest_ci'])}"
                if "delta_weakest_ci" in V["H4"]
                else ""
            ),
            "",
            f"Rotación nula (cada control en el asiento del evaluado): "
            + (
                "ningún control obtiene un positivo"
                if not N["false_positives"]
                else f"positivos de modelos aleatorios en {N['false_positives']} · por control: {N['rotations']}"
            )
            + (
                ""
                if not N.get("false_negatives")
                else f" · negativos atribuidos de modelos aleatorios en {N['false_negatives']}"
            )
            + f" · resolución con K = {N['K']}: p >= {N['p_floor']:.2f}",
            "",
            "Flujo elegido en validación, por hipótesis:",
            "",
        ]
        L += [
            f"- {t}: "
            + ", ".join((f"{h} `{A['select'][t][h]}`" for h in HYP))
            + (
                f" · flujos muertos (no elegibles): {A['select'][t]['dead']}"
                if A["select"][t]["dead"]
                else ""
            )
            for t in A["select"]
        ]
        L += [""]
        head = (
            "| flujo | FPR blanco | FPR señuelo | J(0.2) | "
            + " | ".join((B.COND_HEAD[c] for c in G.CONDITIONS))
            + " |"
        )
        row = (
            lambda n, s: f"| {n} | {s['fpr_blank']:.2f} | {s['fpr_decoy']:.2f} | {s['J02']:+.2f} | "
            + " | ".join((f"{s['ctx'][c]['acc']:.2f}" for c in G.CONDITIONS))
            + " |"
        )
        for tag in A["profile"]:
            L += [
                f"**Perfil por flujo · {tag}** (prueba, media de {len(res['config']['probe_seeds'])} semillas de sonda)",
                "",
                head,
                "|---|" + "---|" * (3 + len(G.CONDITIONS)),
            ]
            chosen = {A["select"][tag][h] for h in HYP}
            L += [
                row("**" + s + "**" if s in chosen else s + " †" if s in A["select"][tag]["dead"] else s, v)
                for s, v in A["profile"][tag].items()
            ]
            L += [""]
        L += [
            "**Bases externas** (mismos datos de entrenamiento, mismos textos de prueba)",
            "",
            head,
            "|---|" + "---|" * (3 + len(G.CONDITIONS)),
        ]
        L += [row(n, v) for n, v in A["baselines"].items()]
        L += [
            "",
            "**Ficha de dosis completa** (celdas = 6 tipos × evasiones canónicas, flujo de evidencia elegido)",
            "",
            "| instrumento | FPR blanco | FPR señuelo | detecta d=0.1 | detecta d=1 | celdas LOD≤0.2 | LOD≤0.5 | sin LOD | de |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for n, s in A["lod"].items():
            L.append(
                f"| {n} | {f2(s['fpr_blank_mean'])} | {f2(s['fpr_decoy'])} | {f2(s['rate_d0.1'])} | {f2(s['rate_d1'])} | {s['lod_le_0.2']} | {s['lod_le_0.5']} | {s['lod_none']} | {s['cells']} |"
            )
        L.append("")
    L += [
        "## Cómo leer esto",
        "",
        "- Lo que cuenta de un flujo es **entrenado − control más fuerte** y **flujo − emb**. Un número alto que un modelo aleatorio también alcanza es mérito de la sonda o del estándar, no del modelo.",
        "- El rango entre controles es el ruido de inicialización. Una ventaja que cabe dentro de ese rango no es ventaja.",
        "- † flujo muerto: sus estados no cambian con el texto (todo empata en 0.50). No se elige nunca.",
        "- `emb` con esta sonda es una bolsa de tokens. Si un flujo profundo no lo supera, no aporta contexto.",
        "- *marcos nuevos* no distingue comprensión: las palabras clave lo pasan. Está para ver robustez sintáctica.",
        "- Una ficha vale para este checkpoint, esta matriz sintética y estas semillas. No es una afirmación sobre datos reales de producción, ni sobre otros checkpoints.",
    ]
    return "\n".join(L)


def _fake_model(D: int = 16, V: int = 64, L: int = 2):
    torch = _torch()
    nn = torch.nn

    class Res(nn.Module):

        def __init__(self):
            super().__init__()
            self.register_buffer("slots", torch.zeros(3, D))
            self.register_buffer("slot_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("fill_count", torch.zeros(1))

        def get_slots(self):
            return self.slots

        @torch.no_grad()
        def update(self, h):
            self.slots[int(self.slot_ptr) % 3] = h.mean((0, 1))
            self.slot_ptr.add_(1)
            self.fill_count.add_(1.0)

    class Per(nn.Module):

        def forward(self, d, slots):
            return d * 0 + slots.mean(0) * (1 + d.mean(-1, keepdim=True))

    class Blk(nn.Module):

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(D, D)

        def forward(self, x, x0):
            return (
                torch.tanh(self.lin(torch.cumsum(x, 1) / (1 + torch.arange(x.shape[1])[None, :, None])))
                + 0.1 * x0
            )

    class Fake(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self.tok_emb = nn.Embedding(V, D)
            self.input_norm = nn.LayerNorm(D)
            self.dendrite = nn.Linear(D, D)
            self.perception, self.resonance = (Per(), Res())
            self.blocks = nn.ModuleList([Blk() for _ in range(L)])
            self.final_norm = nn.LayerNorm(D)

        def _encode_decode(self, ids):
            x = self.input_norm(self.tok_emb(ids))
            x0 = x
            d = self.dendrite(x)
            x = x + d + self.perception(d, self.resonance.get_slots())
            for b in self.blocks:
                x = b(x, x0)
            h = self.final_norm(x)
            self.resonance.update(h)
            return (h, x)

    return Fake


def selftest() -> int:
    torch = _torch()
    import tempfile

    tests = []
    torch.manual_seed(0)
    Fake = _fake_model()
    enc = lambda t: [ord(c) % 60 + 2 for c in t]
    m = Fake()
    m.resonance.slots.normal_()
    ex = Extractor(m, enc, "frozen", bos_id=1)
    st = ex.run("ID 12-34")
    tests.append(
        (
            f"flujos {ex.streams}: [T, D] con T = tokens del texto (sin BOS), y texto vacío -> T = 0",
            all((v.shape == (8, 16) for v in st.values())) and ex.run("")["final"].shape == (0, 16),
        )
    )
    ind = ex.check_independence("ID 12-34", ["otro texto", "y otro más", "tercero"])
    tests.append(
        (
            f"memoria congelada: Δ {ind['max_diff_frozen']:.0e} con congelado, {max(ind['contamination_avoided'].values()):.2f} sin congelar, mutan {ind['mutated_buffers']}",
            ind["ok"]
            and max(ind["contamination_avoided"].values()) > 0.001
            and ("resonance.slots" in ind["mutated_buffers"]),
        )
    )
    m2 = Fake()
    m2.resonance.slots.normal_()
    ex2 = Extractor(m2, enc, "empty", bos_id=1)
    tests.append(
        (
            "memoria vacía: resonancia en cero y percept nulo",
            float(m2.resonance.slots.abs().max()) == 0.0
            and float(np.abs(ex2.run("abc")["percept"]).max()) == 0.0,
        )
    )
    cz = ex.check_causality("un texto de prueba largo")
    tests.append(
        (
            f"diagnóstico de causalidad: Δ máx {max(cz.values()):.0e} en un modelo causal",
            max(cz.values()) < 1e-05,
        )
    )

    class Mod:

        class Hyperparameters:
            alpha, beta, lst, _hidden = (3, "x", [1, 2], object())

        Belen = Fake

    d = tempfile.mkdtemp()
    p = save_raw(m, hparams_of(Mod.Hyperparameters), os.path.join(d, "a.pt"), {"k": 1})
    m3, payload = load_raw(Mod, p)
    same = all((torch.equal(a, b) for a, b in zip(m.state_dict().values(), m3.state_dict().values())))
    torch.save({"format": "otro", "hparams": {}, "model_state_dict": {}}, os.path.join(d, "b.pt"))
    try:
        load_raw(Mod, os.path.join(d, "b.pt"))
        rejected = False
    except ValueError:
        rejected = True
    tests.append(
        (
            "checkpoint: state_dict idéntico tras guardar y cargar, hparams simples, formato ajeno rechazado",
            same and payload["hparams"] == {"alpha": 3, "beta": "x", "lst": [1, 2]} and rejected,
        )
    )
    script = os.path.join(d, "fake_train.py")
    with open(script, "w") as f:
        f.write(
            "import torch\nfrom torch import nn\nORDER = []\nclass Hyperparameters:\n    model_dim = 4\n    vocab_size = 8\nclass Belen(nn.Module):\n    def __init__(self, args):\n        super().__init__(); self.w = nn.Linear(args.model_dim, 2)\ndef quantize_and_compress(model, code_path):\n    ORDER.append('quant'); model.w.weight.data.zero_(); return {}\ndef main():\n    torch.manual_seed(1); m = Belen(Hyperparameters()); ORDER.append('trained')\n    quantize_and_compress(m, __file__)\n"
        )
    out = train_and_save(script, os.path.join(d, "raw.pt"))
    w = torch.load(out, weights_only=True)["model_state_dict"]["w.weight"]
    tests.append(
        (
            "train: pesos guardados antes de que la cuantización los toque (no son cero)",
            float(w.abs().sum()) > 0,
        )
    )
    texts, ev, role, y = B.training_set(90, 18, 3, "en", ["none", "spaced"])
    probes = fit_all(ex, texts, ev, role, y, [0, 1], 2, log=lambda *_: None)
    bank = Bank(ex, probes)
    s1 = bank.scores(texts[:10])
    n_fw = ex.n_forward
    s2 = bank.scores(texts[:10])
    memo_free = ex.n_forward == n_fw
    direct = probes["final#s0"].prob(ex.run(texts[3], ["final"])["final"])
    tests.append(
        (
            f"banco: {len(probes)} sondas ({len(ex.streams)} flujos × (2 semillas + barajada)), memo sin forwards extra, score == cálculo directo",
            len(probes) == len(ex.streams) * 3
            and np.array_equal(s1, s2)
            and memo_free
            and (abs(s1[3, bank.names.index("final#s0")] - direct) < 1e-09),
        )
    )
    cfg = dict(SMOKE_CONFIG, n_dose=12, n_groups_val=6, n_groups_test=6)
    val = measure(bank, cfg, 11, "en", ["none"], 6)
    sel = select(val, ex.streams, [0, 1])
    fake_val = {
        "names": val["names"],
        "dose": {k: np.zeros_like(v) for k, v in val["dose"].items()},
        "ctx": {c: np.zeros_like(v) for c, v in val["ctx"].items()},
    }
    j = fake_val["names"].index("soma1#s0")
    fake_val["dose"]["0.2"][:, [j, j + 1]] = 1.0
    k = fake_val["names"].index("dendritic#s0")
    fake_val["ctx"]["incong"][:, [k, k + 1]] = 1.0
    sel2 = select(fake_val, ex.streams, [0, 1])
    dead_val = dict(val, score_sd=np.where([n.startswith("percept#") for n in val["names"]], 0.0, 1.0))
    dead_val["ctx"] = {
        c: np.where([n.startswith("percept#") for n in val["names"]], 0.5, 0.2) * np.ones_like(v)
        for c, v in val["ctx"].items()
    }
    sel3 = select(dead_val, ex.streams, [0, 1])
    tests.append(
        (
            f"elección por hipótesis en validación: plantado -> evidence {sel2['evidence']}, incong {sel2['incong']} · un flujo muerto en 0.50 NO le gana a los vivos en 0.20 -> muertos {sel3['dead']}, incong {sel3['incong']}",
            sel2["evidence"] == "soma1"
            and sel2["incong"] == "dendritic"
            and (set(sel) == set(HYP) | {"dead"})
            and (sel3["dead"] == ["percept"])
            and ("percept" not in [sel3[h] for h in HYP]),
        )
    )

    def view(j0, j02, lex, con, inc, seen=1.0, n=400):
        r = np.random.default_rng(0)
        b = lambda p: (r.random(n) < p).astype(float)
        return {
            "h": {"0.0": b(j0), "0.2": b(j02), "1.0": b(0.99), "decoy": b(0.0)},
            "ctx": {
                "cong_seen": b(seen),
                "cong_lex": b(lex),
                "cong_frame": b(1.0),
                "contrast": b(con),
                "incong": b(inc),
            },
        }

    cert = {"cong_lex": True, "contrast": True, "incong": True}
    pk = lambda v: {h: v for h in HYP}
    cap_ok, cap_bad = ({"role_ba": 0.52, "ev_ba": 0.55}, {"role_ba": 0.93, "ev_ba": 0.5})
    va = verdict(
        pk(view(0.02, 0.9, 0.9, 0.5, 0.1)),
        [pk(view(0.02, 0.88, 0.5, 0.5, 0.1)), pk(view(0.02, 0.8, 0.45, 0.5, 0.2))],
        cap_ok,
        cert,
    )
    vb = verdict(
        pk(view(0.02, 0.9, 0.9, 0.8, 0.8)),
        [pk(view(0.4, 0.5, 0.5, 0.5, 0.1)), pk(view(0.3, 0.5, 0.55, 0.5, 0.2))],
        cap_bad,
        dict(cert, incong=False),
    )
    vn0 = verdict(
        pk(view(0.02, 0.3, 0.52, 0.55, 0.54)),
        [pk(view(0.02, 0.3, 0.5, 0.52, 0.5)), pk(view(0.02, 0.3, 0.55, 0.5, 0.53))],
        cap_ok,
        cert,
    )
    vm = verdict(
        pk(view(0.02, 0.9, 0.8, 0.5, 0.1)),
        [pk(view(0.02, 0.9, 0.5, 0.5, 0.1)), pk(view(0.02, 0.9, 0.78, 0.5, 0.1))],
        cap_ok,
        cert,
    )
    ctl = [
        pk(view(0.02, 0.3, 0.5, 0.5, 0.67)),
        pk(view(0.02, 0.3, 0.5, 0.5, 0.5)),
        pk(view(0.02, 0.3, 0.5, 0.5, 0.54)),
    ]
    nr = null_rotation(ctl, [cap_ok] * 3, cert)
    vdeg = apply_null(verdict(pk(view(0.02, 0.3, 0.5, 0.5, 0.8)), ctl, cap_ok, cert), nr)
    clean = null_rotation(
        [
            pk(view(0.02, 0.3, 0.5, 0.5, 0.5)),
            pk(view(0.02, 0.3, 0.5, 0.5, 0.52)),
            pk(view(0.02, 0.3, 0.5, 0.5, 0.49)),
        ],
        [cap_ok] * 3,
        cert,
    )
    vprov = apply_null(verdict(pk(view(0.02, 0.3, 0.5, 0.5, 0.8)), ctl[1:], cap_ok, cert), clean)
    tests.append(
        (
            f"veredicto A: {[va[h]['verdict'] for h in ('H1', 'H2', 'H3', 'H4', 'H5')]}",
            [va[h]["verdict"] for h in ("H1", "H2", "H3", "H4", "H5")]
            == ["no aporta", "el entrenamiento aporta", "no la lee", NEG_DESC, "sonda selectiva"]
            and (not va["caution"])
            and (va["negatives"] == []),
        )
    )
    cn = [
        pk(view(0.02, 0.3, 0.5, 0.5, 0.5)),
        pk(view(0.02, 0.3, 0.5, 0.5, 0.45)),
        pk(view(0.02, 0.3, 0.5, 0.5, 0.52)),
    ]
    vneg = apply_null(
        verdict(pk(view(0.02, 0.3, 0.5, 0.5, 0.1)), cn, cap_ok, cert), null_rotation(cn, [cap_ok] * 3, cert)
    )
    cbad = [
        pk(view(0.02, 0.3, 0.5, 0.5, 0.1)),
        pk(view(0.02, 0.3, 0.5, 0.5, 0.5)),
        pk(view(0.02, 0.3, 0.5, 0.5, 0.52)),
    ]
    nrn = null_rotation(cbad, [cap_ok] * 3, cert)
    vnd = apply_null(verdict(pk(view(0.02, 0.3, 0.5, 0.5, 0.02)), cbad[1:], cap_ok, cert), nrn)
    tests.append(
        (
            f"H4 negativo atribuido: 0.10 contra controles en 0.45 a 0.52 -> '{vneg['H4']['verdict']}', provisional {vneg['provisional']} ({vneg['attributed']}) · si un control aleatorio también cae ({nrn['false_negatives']}) -> '{vnd['H4']['verdict']}'",
            vneg["H4"]["verdict"] == NEG_ATTR
            and vneg["negatives"] == ["H4"]
            and vneg["provisional"]
            and (vneg["positives"] == [])
            and (nrn["false_negatives"] == ["H4"])
            and (vnd["H4"]["verdict"] == "NO DISTINGUIBLE DEL RUIDO DE INICIALIZACIÓN")
            and (vnd["negatives"] == [])
            and (vnd["attributed"] == []),
        )
    )
    tests.append(
        (
            f"procedencia: '{trained_line({'note': 'pesos ALEATORIOS, solo cableado'}, {'iterations': 1000})}' · '{trained_line({'steps_done': 6500, 'tokens_seen': 425984000}, {'iterations': 6500})}'",
            "0 pasos" in trained_line({"note": "pesos ALEATORIOS, solo cableado"}, {"iterations": 1000})
            and "6500 pasos entrenados"
            in trained_line({"steps_done": 6500, "tokens_seen": 425984000}, {"iterations": 6500})
            and ("NO registrados" in trained_line({}, {"iterations": 1000})),
        )
    )
    tests.append(
        (
            f"veredicto B: {[vb[h]['verdict'] for h in ('H1', 'H2', 'H3', 'H4', 'H5')]}",
            [vb[h]["verdict"] for h in ("H1", "H2", "H3", "H4", "H5")]
            == [
                "el entrenamiento aporta",
                "el entrenamiento aporta",
                "lee la negación",
                "NO CERTIFICADA",
                "LA SONDA MEMORIZA",
            ]
            and vb["caution"],
        )
    )
    tests.append(
        (
            f"veredicto: 0.55 en contraste y 0.54 en incongruente ya NO son positivos -> H3 '{vn0['H3']['verdict']}', H4 '{vn0['H4']['verdict']}', positivos {vn0['positives']}",
            vn0["positives"] == []
            and vn0["H3"]["verdict"] == "no la lee"
            and (vn0["H4"]["verdict"] == "indeterminado"),
        )
    )
    tests.append(
        (
            f"veredicto: ganarle a un control débil no basta si otro control llega igual -> H2 '{vm['H2']['verdict']}' (controles {vm['H2']['control_range'][0]:.2f} a {vm['H2']['control_range'][1]:.2f})",
            vm["H2"]["verdict"] != "el entrenamiento aporta",
        )
    )
    tests.append(
        (
            f"rotación nula: un control aleatorio da {nr['false_positives']} -> el H4 del evaluado pasa a '{vdeg['H4']['verdict']}'",
            nr["false_positives"] == ["H4"] and vdeg["degraded"] == ["H4"] and (vdeg["positives"] == []),
        )
    )
    tests.append(
        (
            f"resolución: controles limpios y K = 3 -> H4 '{vprov['H4']['verdict']}' pero provisional (p >= {clean['p_floor']:.2f})",
            vprov["positives"] == ["H4"]
            and vprov["provisional"]
            and (clean["false_positives"] == [])
            and (abs(clean["p_floor"] - 0.25) < 1e-09),
        )
    )
    tok = os.path.join(d, "tok.model")
    open(tok, "w").write("x")
    pre_dir = os.path.join(d, "pre")
    write_prereg(pre_dir, script, out, tok, SMOKE_CONFIG)
    _, drift0 = check_prereg(pre_dir, script, out, tok)
    open(tok, "w").write("y")
    _, drift1 = check_prereg(pre_dir, script, out, tok)
    try:
        write_prereg(pre_dir, script, out, tok, SMOKE_CONFIG)
        once = False
    except SystemExit:
        once = True
    tests.append(
        (
            f"prerregistro: sin deriva -> {drift0}, tras cambiar el tokenizer -> {drift1}, segunda escritura rechazada",
            drift0 == [] and drift1 == ["tokenizer:tok.model"] and once,
        )
    )
    norm = lambda t: [ord(c) for c in t.replace(F.ZW, " ")]
    eq = tokenizer_equivalences(norm, "en", n=40)
    raw = tokenizer_equivalences(lambda t: [ord(c) for c in t], "en", n=40)
    tests.append(
        (
            f"equivalencias de tokenizer: normalizador -> alias {eq['aliases']}, sin normalizar -> {raw['aliases']}",
            eq["aliases"] == {"zwsp": "spaced"} and raw["aliases"] == {} and ("zwsp" not in eq["canonical"]),
        )
    )
    ds = dose_set(cfg["n_dose"], cfg["seed_test"], "en", ["none"])
    la0 = leak_audit(texts, y, cfg, "en", ["none"])
    la1 = leak_audit(texts + [ds["1.0"][0]], y + [1.0], cfg, "en", ["none"])
    tests.append(
        (
            f"auditoría de fuga: limpio -> {la0['positive_duplicates']} positivos, plantado -> {la1['positive_duplicates']}",
            la0["positive_duplicates"] == 0 and la1["positive_duplicates"] == 1,
        )
    )
    ex.close()
    ex2.close()
    import zlib
    from types import SimpleNamespace

    class Planted:
        streams = ["emb", "final"]

        def __init__(self, plant: bool):
            self.plant, self.orc, self.hs = (plant, G.oracle("en"), B.HashStates(dim=24, buckets=1 << 12))
            self.encode = lambda t: list(t)
            self.model = SimpleNamespace(tok_emb=SimpleNamespace(weight=np.zeros((1, 24))))

        def run(self, text, streams=None):
            n = len(text)
            noise = (
                np.random.default_rng(zlib.crc32(text.encode())).standard_normal((n, 23)).astype(np.float32)
            )
            role = np.zeros((n, 1), np.float32)
            if self.plant and text.count("\n") == 2:
                o = float(self.orc([text])[0])
                role += 1.0 if o == 1.0 else -1.0 if o == 0.0 else 0.0
            return {
                "emb": self.hs(text) if n else np.zeros((0, 24), np.float32),
                "final": np.concatenate([noise, role], 1),
            }

    pc = dict(SMOKE_CONFIG, n_leak=150, n_ctx=60, n_dose=24, n_groups_val=24, n_groups_test=60, epochs=3)
    arms = {
        k: run_arm(k, Planted(k == "plantado"), pc, "en", ["none"], log=lambda *_: None)
        for k in ("plantado", "vacío")
    }
    cert = {"cong_lex": True, "contrast": True, "incong": True}

    def vd(a, b):
        sv = lambda arm: {h: stream_view(arm["test"], "final", pc["probe_seeds"]) for h in HYP}
        return verdict(sv(a), [sv(b)], a["capacity"]["final"], cert)

    vp, vn = (vd(arms["plantado"], arms["vacío"]), vd(arms["vacío"], arms["vacío"]))
    tests.append(
        (
            f"control POSITIVO (papel plantado): H2 '{vp['H2']['verdict']}', H3 '{vp['H3']['verdict']}', H4 '{vp['H4']['verdict']}' ({vp['H4']['acc']:.2f}), H5 '{vp['H5']['verdict']}' (barajada ajusta {vp['H5']['shuffled_role_ba']:.2f}, real {vp['H5']['real_role_ba']:.2f})",
            vp["H2"]["verdict"] == "el entrenamiento aporta"
            and vp["H3"]["verdict"] == "lee la negación"
            and (vp["H4"]["verdict"] == "COMPONE")
            and (not vp["caution"]),
        )
    )
    tests.append(
        (
            f"control NEGATIVO (sin plantar): H2 '{vn['H2']['verdict']}', H3 '{vn['H3']['verdict']}', H4 '{vn['H4']['verdict']}' ({vn['H4']['acc']:.2f})",
            vn["H2"]["verdict"] != "el entrenamiento aporta"
            and vn["H3"]["verdict"] == "no la lee"
            and (vn["H4"]["verdict"] != "COMPONE"),
        )
    )
    print(f"model_probe v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{('ok' if ok else 'FAIL')}] {name}")
    print(
        "Honestidad: modelo de juguete con la misma interfaz. Valida la MECÁNICA. El código real lo prueba `wiring`."
    )
    return 0 if all((ok for _, ok in tests)) else 1


def wiring(a) -> int:
    torch = _torch()
    import sentencepiece as spm
    import tempfile

    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    mod = load_script(a.script, SMALL_ENV if a.small else None)
    need = ["Hyperparameters", "Belen"]
    miss = [n for n in need if not hasattr(mod, n)]
    if miss:
        raise SystemExit(f"[wiring] al script le falta: {miss}")
    d = tempfile.mkdtemp()
    if a.ckpt:
        ckpt = a.ckpt
    else:
        model, args = build_model(mod, init_seed=11)
        with torch.no_grad():
            model._encode_decode(torch.randint(4, 1000, (2, 24)))
        ckpt = save_raw(
            model,
            hparams_of(args),
            os.path.join(d, "wiring_raw.pt"),
            {"note": "pesos ALEATORIOS, solo cableado"},
        )
    model, payload = load_raw(mod, ckpt)
    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{('ok' if good else 'FAIL')}] {name}" + (f": {detail}" if detail else ""), flush=True)

    attrs = ["tok_emb", "input_norm", "dendrite", "perception", "resonance", "blocks", "_encode_decode"]
    line(
        "interfaz de BELEN",
        all((hasattr(model, x) for x in attrs)),
        f"faltan {[x for x in attrs if not hasattr(model, x)]}",
    )
    if not ok:
        return 1
    ex = Extractor(
        model,
        lambda t: sp.encode(t),
        "frozen",
        sp.bos_id(),
        train_seq_len=payload["hparams"].get("train_seq_len"),
    )
    t0 = "Customer ID: ABCD123456-H42 | Reply"
    st = ex.run(t0)
    line(
        "flujos y formas",
        all((v.ndim == 2 and v.shape[0] == len(sp.encode(t0)) and np.isfinite(v).all() for v in st.values())),
        f"{len(ex.streams)} flujos, [T={len(sp.encode(t0))}, D={st['final'].shape[1]}], finitos",
    )
    ind = ex.check_independence(t0, [s.text for s in F.generate(8, seed=3, lang="en")])
    line(
        "memoria congelada (independencia de orden)",
        ind["ok"],
        f"Δ {ind['max_diff_frozen']:.1e} congelada · hasta {max(ind['contamination_avoided'].values()):.3g} sin congelar · mutan {ind['mutated_buffers']}",
    )
    cz = ex.check_causality(t0)
    line(
        "diagnóstico de causalidad",
        True,
        f"Δ máx {max(cz.values()):.1e}"
        + ("" if max(cz.values()) <= 0.0001 else f" · no causal: {[s for s, v in cz.items() if v > 0.0001]}"),
    )
    if not a.ckpt:
        m2, _ = load_raw(mod, ckpt)
        line(
            "checkpoint: ida y vuelta estricta",
            all((torch.equal(x, y) for x, y in zip(model.state_dict().values(), m2.state_dict().values()))),
        )
    eq = tokenizer_equivalences(lambda t: sp.encode(t), "en", n=60)
    line("auditoría del tokenizer real", True, f"alias {eq['aliases']} · canónicas {eq['canonical']}")
    ex.close()
    out_dir = os.path.join(d, "humo")
    res = run_ficha(
        a.script,
        ckpt,
        a.tokenizer,
        out_dir,
        dict(SMOKE_CONFIG, langs=a.langs),
        device=None,
        smoke=True,
        log=lambda m: print("   ", m, flush=True),
    )
    md = open(os.path.join(out_dir, "SMOKE.md"), encoding="utf-8").read()
    line(
        "ficha de humo de extremo a extremo",
        "Veredicto" in md and all((l in res["arms"] for l in a.langs)),
        f"{len(md.splitlines())} líneas -> {out_dir}/SMOKE.md",
    )
    if not a.ckpt:
        pos = {
            l: res["arms"][l]["verdict"].get("attributed", res["arms"][l]["verdict"]["positives"])
            for l in a.langs
        }
        rotp = {l: res["arms"][l]["verdict"]["null"]["false_positives"] for l in a.langs}
        if any(pos.values()):
            print(
                f"  [aviso] calibración nula de humo: al modelo aleatorio se le atribuyó {pos} (rotación: {rotp}). Con K = {len(SMOKE_CONFIG['random_init_seeds'])} controles y {SMOKE_CONFIG['n_groups_test']} grupos eso cabe en la resolución declarada. No es un resultado ni una falla de cableado.",
                flush=True,
            )
        else:
            line(
                "calibración nula de humo: al modelo aleatorio no se le atribuye nada",
                True,
                f"rotación: {rotp}",
            )
    if a.keep:
        import shutil

        shutil.copy(os.path.join(out_dir, "SMOKE.md"), a.keep)
    print(
        "[wiring] "
        + ("TODO OK" if ok else "HAY FALLAS")
        + (" · pesos aleatorios: valida cableado, no desempeño" if not a.ckpt else "")
    )
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="BELENC-PII dentro de BELEN")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    w = sub.add_parser("wiring")
    w.add_argument("--script", required=True)
    w.add_argument("--tokenizer", required=True)
    w.add_argument("--ckpt")
    w.add_argument("--small", action="store_true")
    w.add_argument("--langs", nargs="+", default=["en"], choices=G.LANGS)
    w.add_argument("--keep")
    t = sub.add_parser("train")
    t.add_argument("--script", required=True)
    t.add_argument("--out-ckpt", required=True)
    for name in ("prereg", "ficha"):
        p = sub.add_parser(name)
        p.add_argument("--script", required=True)
        p.add_argument("--ckpt", required=True)
        p.add_argument("--tokenizer", required=True)
        p.add_argument("--out-dir", required=True)
        if name == "prereg":
            p.add_argument("--langs", nargs="+", default=["en"], choices=G.LANGS)
            p.add_argument(
                "--controls",
                type=int,
                default=3,
                help="inicializaciones aleatorias del control. 3 = positivos provisionales. 19 = p <= 0.05",
            )
        else:
            p.add_argument("--allow-drift", action="store_true")
            p.add_argument("--device")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "wiring":
        if not a.ckpt and (not a.small):
            raise SystemExit("[wiring] da --ckpt, o --small para el modelo chico aleatorio")
        return wiring(a)
    if a.cmd == "train":
        train_and_save(a.script, a.out_ckpt)
        return 0
    if a.cmd == "prereg":
        if a.controls < 2:
            raise SystemExit("[prereg] hacen falta al menos 2 controles para la rotación nula")
        cfg = dict(DEFAULT_CONFIG, langs=a.langs, random_init_seeds=list(range(7, 7 + a.controls)))
        print(f"[prereg] escrito {write_prereg(a.out_dir, a.script, a.ckpt, a.tokenizer, cfg)}")
        print(
            f"[prereg] {a.controls} controles: frente al ruido de inicialización, p >= {1 / (a.controls + 1):.2f}"
        )
        print("[prereg] Ábrelo, agrega tus predicciones en J1/J2 y después corre `ficha`.")
        return 0
    if a.cmd == "ficha":
        pre, drift = check_prereg(a.out_dir, a.script, a.ckpt, a.tokenizer)
        if drift and (not a.allow_drift):
            raise SystemExit(
                f"[ficha] el código o los archivos cambiaron desde el prerregistro: {drift}\n        Si es intencional, corre con --allow-drift y quedará declarado en la ficha."
            )
        res = run_ficha(a.script, a.ckpt, a.tokenizer, a.out_dir, pre["config"], a.device, drift=drift)
        print(open(os.path.join(a.out_dir, "MODEL_PROBE.md"), encoding="utf-8").read())
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
