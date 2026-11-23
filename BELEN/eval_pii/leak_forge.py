"""  
|---|---|
| `leak_forge.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Guide it. |
"""
-----


from __future__ import annotations
import argparse
import json
import re
import sys
import zlib
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

VERSION = "forge-0.2"
PII_TYPES = ("PERSON", "EMAIL", "PHONE", "ID", "IP", "ADDRESS")
OPERATORS = ("suffix", "prefix", "edges", "scatter")
EVASIONS = ("none", "spaced", "zwsp", "homoglyph", "split")
CARRIERS = ("chat", "email", "form", "log", "ocr")
DOSES = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
MASK_CHARS = ("*", "X", "•", "#")
ZW = "\u200b"
HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "A": "А", "E": "Е", "O": "О"}
FIRST = [
    "Mariela",
    "Tobias",
    "Renata",
    "Ilan",
    "Odalys",
    "Bruno",
    "Xiomara",
    "Elian",
    "Petra",
    "Damaso",
    "Yolotl",
    "Severin",
    "Itzel",
    "Kasimir",
    "Noemi",
    "Fausto",
    "Lizbeth",
    "Orrin",
]
LAST = [
    "Quintanar",
    "Vellmer",
    "Aranzubia",
    "Ostrikov",
    "Peñaloza",
    "Haldane",
    "Zubiaur",
    "Tremblay",
    "Ocampo",
    "Lindqvist",
    "Barragán",
    "Molnar",
    "Cervantes",
    "Ashdown",
]
DOMAINS = ["mailbox.test", "correo.example", "corp.invalid", "tq-labs.example", "nube.test"]
STREETS = ["Av. Lomas Altas", "Calle Encino", "Rue Valmont", "Birch Hollow Rd", "Priv. Tepeyac"]
CHANNELS = ["#general", "#soporte", "#ops", "#ventas"]
UI_CHROME = [
    "Enviar mensaje a {ch}",
    "13 en línea",
    "ayer a las 21:14",
    "(editado)",
    "Ver todos los mensajes fijados.",
    "Responder",
    "Reaccionar",
    "Copiar enlace",
    "Mostrar más",
    "Archivo adjunto · 2.1 MB",
    "Visto 23:58",
]
UI_CHROME_EN = [
    "Send a message to {ch}",
    "13 online",
    "yesterday at 21:14",
    "(edited)",
    "View all pinned messages.",
    "Reply",
    "React",
    "Copy link",
    "Show more",
    "Attachment · 2.1 MB",
    "Seen 23:58",
]
CHROME_BY_LANG = {"es": UI_CHROME, "en": UI_CHROME_EN}
LABELS = {
    "es": {
        "PERSON": "Nombre",
        "EMAIL": "Correo",
        "PHONE": "Tel",
        "ID": "ID cliente",
        "IP": "IP origen",
        "ADDRESS": "Dirección",
        "DECOY": "Ref",
    },
    "en": {
        "PERSON": "Name",
        "EMAIL": "Email",
        "PHONE": "Phone",
        "ID": "Customer ID",
        "IP": "Source IP",
        "ADDRESS": "Address",
        "DECOY": "Ref",
    },
}
WORDS = {
    "es": {
        "chat_pre": "oye, el {l} es ",
        "chat_post": " ya lo pasé",
        "from": "De",
        "subject": "Asunto: Seguimiento ticket",
        "hello": "Hola, confirmamos",
        "save": "[ Guardar ]  [ Cancelar ]",
        "order": "Pedido",
        "apt": "int.",
    },
    "en": {
        "chat_pre": "hey, the {l} is ",
        "chat_post": " already sent it",
        "from": "From",
        "subject": "Subject: Ticket follow-up",
        "hello": "Hi, we are confirming",
        "save": "[ Save ]  [ Cancel ]",
        "order": "Order",
        "apt": "apt.",
    },
}
LANGS = ("es", "en")


def _person(r) -> str:
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def _email(r) -> str:
    sep = r.choice([".", "_", ""])
    f, l = (r.choice(FIRST).lower(), r.choice(LAST).lower())
    f = f.encode("ascii", "ignore").decode() or "user"
    l = l.encode("ascii", "ignore").decode() or "mail"
    num = str(r.integers(1, 99)) if r.random() < 0.4 else ""
    return f"{f}{sep}{l}{num}@{r.choice(DOMAINS)}"


def _phone(r) -> str:
    d = "".join((str(x) for x in r.integers(0, 10, size=10)))
    fmt = r.integers(0, 3)
    if fmt == 0:
        return f"+52 {d[:2]} {d[2:6]} {d[6:]}"
    if fmt == 1:
        return f"({d[:3]}) {d[3:6]}-{d[6:]}"
    return f"{d[:3]}-{d[3:6]}-{d[6:]}"


def _idnum(r) -> str:
    letters = "".join(r.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ"), size=4))
    digits = "".join((str(x) for x in r.integers(0, 10, size=6)))
    return f"{letters}{digits}-{r.choice(list('HM'))}{r.integers(10, 99)}"


def _ip(r) -> str:
    return ".".join((str(x) for x in [r.integers(11, 223), *r.integers(0, 255, size=3)]))


def _address(r, lang: str = "es") -> str:
    return f"{r.choice(STREETS)} {r.integers(12, 9999)}, {WORDS[lang]['apt']} {r.integers(1, 40)}"


GEN = {"PERSON": _person, "EMAIL": _email, "PHONE": _phone, "ID": _idnum, "IP": _ip, "ADDRESS": _address}


def _decoy(r, lang: str = "es") -> str:
    k = r.integers(0, 6)
    if k == 0:
        return f"{WORDS[lang]['order']} #{r.integers(100000, 999999)}-{r.integers(10, 99)}"
    if k == 1:
        return f"commit {''.join(r.choice(list('0123456789abcdef'), size=12))}"
    if k == 2:
        return f"v{r.integers(1, 9)}.{r.integers(0, 30)}.{r.integers(0, 200)}"
    if k == 3:
        return f"{r.integers(0, 24):02d}:{r.integers(0, 60):02d}:{r.integers(0, 60):02d}"
    if k == 4:
        return f"noreply@{r.choice(DOMAINS)}"
    return f"build-{r.integers(1000, 9999)}-{r.integers(1000, 9999)}"


@dataclass
class Span:
    start: int
    end: int
    pii_type: str
    original: str
    rendered: str
    dose_requested: float
    revealed: int
    total: int
    operator: str
    evasion: str

    @property
    def dose(self) -> float:
        return self.revealed / max(self.total, 1)


def ident_positions(v: str) -> List[int]:
    return [i for i, ch in enumerate(v) if ch.isalnum()]


def reveal_set(v: str, dose: float, op: str, pii_type: str, r) -> List[int]:
    idx = ident_positions(v)
    n = len(idx)
    k = int(round(dose * n))
    if k <= 0:
        return []
    if k >= n:
        return idx
    if pii_type == "EMAIL" and op != "scatter":
        at = v.index("@")
        dom = [i for i in idx if i > at]
        loc = [i for i in idx if i < at]
        order = dom + (
            loc
            if op == "prefix"
            else loc[::-1] if op == "suffix" else [x for pair in zip(loc, loc[::-1]) for x in pair]
        )
        seen, out = (set(), [])
        for i in order:
            if i not in seen:
                seen.add(i)
                out.append(i)
            if len(out) == k:
                break
        return sorted(out)
    if op == "suffix":
        return idx[-k:]
    if op == "prefix":
        return idx[:k]
    if op == "edges":
        a = (k + 1) // 2
        return sorted(set(idx[:a] + idx[n - (k - a) :]))
    return sorted(r.choice(idx, size=k, replace=False).tolist())


def mask(v: str, keep: Sequence[int], mc: str) -> str:
    ks = set(keep)
    return "".join((ch if not ch.isalnum() or i in ks else mc for i, ch in enumerate(v)))


def evade(s: str, ev: str, r, lang: str = "es") -> str:
    if ev == "none":
        return s
    if ev == "spaced":
        return " ".join(s)
    if ev == "zwsp":
        return ZW.join(s)
    if ev == "homoglyph":
        return "".join((HOMO.get(ch, ch) if r.random() < 0.6 else ch for ch in s))
    if ev == "split":
        cut = max(1, len(s) // 2)
        return s[:cut] + f" ⏎ {r.choice(CHROME_BY_LANG[lang][1:4])} ⏎ " + s[cut:]
    raise ValueError(ev)


def carrier_pieces(kind: str, slot: str, r, lang: str = "es") -> List[str]:
    ch = r.choice(CHANNELS)
    UI, W = (CHROME_BY_LANG[lang], WORDS[lang])
    chrome = [c.format(ch=ch) for c in r.choice(UI, size=2, replace=False)]
    label = LABELS[lang][slot]
    if kind == "chat":
        user = r.choice(["timthepost", "kyuboid", "sudocorn", "frencdec", "agente_7"])
        return [
            f"{ch} · {chrome[0]}\n{user} {chrome[1]}\n" + W["chat_pre"].format(l=label.lower()),
            W["chat_post"],
        ]
    if kind == "email":
        return [
            f"{W['from']}: soporte@{r.choice(DOMAINS)}\n{W['subject']} {r.integers(1000, 9999)}\n{W['hello']} {label.lower()}: ",
            f"\n--\n{chrome[0]}",
        ]
    if kind == "form":
        return [f"[{chrome[0]}]\n{label}: ", f"\n{W['save']}  {chrome[1]}"]
    if kind == "log":
        lvl = r.choice(["INFO", "WARN", "DEBUG"])
        return [
            f'2026-11-{r.integers(1, 28):02d}T{r.integers(0, 24):02d}:{r.integers(0, 60):02d}:00Z {lvl} svc=auth msg="lookup {label.lower()}=',
            f'" latency_ms={r.integers(3, 900)}',
        ]
    if kind == "ocr":
        return [f"{chrome[0]} | {chrome[1]} | {label} ", f" | {r.choice(UI)}"]
    raise ValueError(kind)


OCR_SUB = {"O": "0", "l": "1", "I": "l", "S": "5", "B": "8"}


def ocr_noise(text: str, protect: List[Tuple[int, int]], r, p: float = 0.06) -> str:
    inside = np.zeros(len(text), bool)
    for a, b in protect:
        inside[a:b] = True
    return "".join(
        (
            OCR_SUB[ch] if not inside[i] and ch in OCR_SUB and (r.random() < p) else ch
            for i, ch in enumerate(text)
        )
    )


@dataclass
class Sample:
    text: str
    spans: List[Span]
    kind: str
    carrier: str
    decoys: List[Tuple[int, int, str]] = field(default_factory=list)

    @property
    def dose(self) -> float:
        return max((s.dose for s in self.spans), default=0.0)

    @property
    def label(self) -> int:
        return int(any((s.revealed > 0 for s in self.spans)))

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "kind": self.kind,
            "carrier": self.carrier,
            "label": self.label,
            "dose": round(self.dose, 4),
            "spans": [asdict(s) for s in self.spans],
            "decoys": self.decoys,
            "forge_version": VERSION,
        }


def make_sample(
    r, pii_type: str, dose: float, op: str, ev: str, carrier: str, add_decoy: bool = False, lang: str = "es"
) -> Sample:
    v = _address(r, lang) if pii_type == "ADDRESS" else GEN[pii_type](r)
    mc = r.choice(MASK_CHARS)
    keep = reveal_set(v, dose, op, pii_type, r)
    if not keep and r.random() < 0.25:
        shown = f"[{pii_type}]" if r.random() < 0.5 else "[REDACTED]"
    else:
        shown = mask(v, keep, mc)
    shown = evade(shown, ev, r, lang)
    pre, post = carrier_pieces(carrier, pii_type, r, lang)
    decoys = []
    if add_decoy:
        d = _decoy(r, lang)
        post = post + f" · {d}"
    text = pre + shown + post
    s0 = len(pre)
    span = Span(s0, s0 + len(shown), pii_type, v, shown, dose, len(keep), len(ident_positions(v)), op, ev)
    if add_decoy:
        di = text.rfind(d)
        decoys.append((di, di + len(d), d))
    if carrier == "ocr":
        text = ocr_noise(text, [(span.start, span.end)] + [(a, b) for a, b, _ in decoys], r)
    return Sample(text, [span], "analyte", carrier, decoys)


def make_decoy_only(r, carrier: str, lang: str = "es") -> Sample:
    d = _decoy(r, lang)
    pre, post = carrier_pieces(carrier, "DECOY", r, lang)
    text = pre + d + post
    a = len(pre)
    if carrier == "ocr":
        text = ocr_noise(text, [(a, a + len(d))], r)
    return Sample(text, [], "decoy", carrier, [(a, a + len(d), d)])


def generate(
    n: int,
    seed: int,
    doses: Sequence[float] = DOSES,
    decoy_frac: float = 0.15,
    blank_frac: Optional[float] = None,
    lang: str = "es",
    evasions: Sequence[str] = EVASIONS,
) -> List[Sample]:
    r = np.random.default_rng(seed)
    pos = [d for d in doses if d > 0]
    out = []
    for _ in range(n):
        car = str(r.choice(CARRIERS))
        if r.random() < decoy_frac:
            out.append(make_decoy_only(r, car, lang))
            continue
        d = (
            (0.0 if r.random() < blank_frac else float(r.choice(pos)))
            if blank_frac is not None
            else float(r.choice(doses))
        )
        out.append(
            make_sample(
                r,
                str(r.choice(PII_TYPES)),
                d,
                str(r.choice(OPERATORS)),
                str(r.choice(list(evasions))),
                car,
                add_decoy=r.random() < 0.3,
                lang=lang,
            )
        )
    return out


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def lod_from_rates(doses: Sequence[float], rates: Sequence[float], target: float = 0.95) -> Optional[float]:
    ds = [(d, x) for d, x in zip(doses, rates) if d > 0]
    for i, (d, _) in enumerate(ds):
        if all((x >= target for _, x in ds[i:])):
            return d
    return None


Detector = Callable[[List[str]], np.ndarray]


def evaluate(
    detector: Detector,
    reps: int,
    seed: int,
    doses: Sequence[float] = DOSES,
    types: Sequence[str] = PII_TYPES,
    evasions: Sequence[str] = EVASIONS,
    thr: float = 0.5,
    lang: str = "es",
) -> Dict:
    cells = []
    for t in types:
        for ev in evasions:
            rates, ks = ([], [])
            for di, d in enumerate(doses):
                r = np.random.default_rng([seed, PII_TYPES.index(t), EVASIONS.index(ev), di])
                texts = [
                    make_sample(
                        r, t, d, str(r.choice(OPERATORS)), ev, str(r.choice(CARRIERS)), lang=lang
                    ).text
                    for _ in range(reps)
                ]
                hit = np.asarray(detector(texts), float) >= thr
                ks.append(int(hit.sum()))
                rates.append(float(hit.mean()))
            fpr_k = ks[list(doses).index(0.0)] if 0.0 in doses else 0
            cells.append(
                {
                    "pii_type": t,
                    "evasion": ev,
                    "rates": dict(zip(map(str, doses), rates)),
                    "fpr_blank": rates[list(doses).index(0.0)] if 0.0 in doses else None,
                    "fpr_blank_ci": wilson(fpr_k, reps),
                    "lod95": lod_from_rates(doses, rates),
                }
            )
    r = np.random.default_rng([seed, 999])
    dtexts = [make_decoy_only(r, str(r.choice(CARRIERS)), lang).text for _ in range(reps * 4)]
    dk = int((np.asarray(detector(dtexts), float) >= thr).sum())
    return {
        "doses": list(doses),
        "reps": reps,
        "cells": cells,
        "lang": lang,
        "fpr_decoy": dk / len(dtexts),
        "fpr_decoy_ci": wilson(dk, len(dtexts)),
    }


def report(res: Dict, name: str) -> str:
    doses = res["doses"]
    fmt = lambda x: "—" if x is None else f"{x:g}"
    L = [
        f"## {name}",
        "",
        f"FPR señuelos: {res['fpr_decoy']:.2f} (IC95 {res['fpr_decoy_ci'][0]:.2f}–{res['fpr_decoy_ci'][1]:.2f}) · réplicas por dosis: {res['reps']}",
        "",
        "| tipo | evasión | FPR blanco | " + " | ".join((f"d={d:g}" for d in doses if d > 0)) + " | LOD95 |",
        "|" + "---|" * (4 + len([d for d in doses if d > 0])),
    ]
    for c in res["cells"]:
        L.append(
            f"| {c['pii_type']} | {c['evasion']} | {c['fpr_blank']:.2f} | "
            + " | ".join((f"{c['rates'][str(d)]:.2f}" for d in doses if d > 0))
            + f" | **{fmt(c['lod95'])}** |"
        )
    return "\n".join(L)


RX = [
    re.compile(p)
    for p in (
        "[\\w.+-]+@[\\w-]+\\.[\\w.]+",
        "(\\+\\d{1,3}[\\s-]?)?\\(?\\d{2,3}\\)?[\\s-]?\\d{3,4}[\\s-]?\\d{4}",
        "\\b[A-Z]{4}\\d{6}-[HM]\\d{2}\\b",
        "\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b",
        "\\b[A-Z][a-záéíóúñ]+ [A-Z][a-záéíóúñ]+\\b",
    )
]


def regex_detector(texts: List[str]) -> np.ndarray:
    return np.array([float(any((rx.search(t) for rx in RX))) for t in texts])


class NgramLogReg:

    def __init__(self, dim: int = 1 << 18, nmin: int = 1, nmax: int = 4, seed: int = 0):
        self.dim, self.nmin, self.nmax = (dim, nmin, nmax)
        self.w = np.zeros(dim, np.float32)
        self.b = 0.0
        self.seed = seed

    def feats(self, t: str) -> np.ndarray:
        t = f"\x02{t}\x03"
        h = [
            zlib.crc32(t[i : i + n].encode("utf-8")) % self.dim
            for n in range(self.nmin, self.nmax + 1)
            for i in range(len(t) - n + 1)
        ]
        return np.unique(np.asarray(h, np.int64))

    def fit(self, texts: List[str], y: Sequence[int], epochs: int = 4, lr: float = 0.2, l2: float = 1e-06):
        X = [self.feats(t) for t in texts]
        y = np.asarray(y, np.float32)
        r = np.random.default_rng(self.seed)
        for _ in range(epochs):
            for i in r.permutation(len(X)):
                f = X[i]
                z = float(self.w[f].sum()) / np.sqrt(len(f)) + self.b
                g = 1.0 / (1.0 + np.exp(-z)) - y[i]
                self.w[f] -= lr * (g / np.sqrt(len(f)) + l2 * self.w[f])
                self.b -= lr * g
        return self

    def __call__(self, texts: List[str]) -> np.ndarray:
        out = np.empty(len(texts))
        for i, t in enumerate(texts):
            f = self.feats(t)
            out[i] = 1.0 / (1.0 + np.exp(-(float(self.w[f].sum()) / np.sqrt(len(f)) + self.b)))
        return out


def auc(y, s) -> float:
    y, s = (np.asarray(y), np.asarray(s, float))
    p, n = (s[y == 1], s[y == 0])
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


def selftest() -> int:
    tests = []
    r = np.random.default_rng(3)
    ok = True
    for t in PII_TYPES:
        for op in OPERATORS:
            for d in DOSES:
                v = GEN[t](r)
                k = len(reveal_set(v, d, op, t, r))
                ok &= k == int(round(d * len(ident_positions(v))))
    tests.append(("dosis exacta: |visibles| = round(d·n) en 6 tipos × 4 operadores × 7 dosis", ok))
    v = "mariela.quintanar7@mailbox.test"
    keep = reveal_set(v, 0.5, "suffix", "EMAIL", r)
    m = mask(v, keep, "*")
    vis = [i for i, ch in enumerate(m) if ch.isalnum()]
    tests.append(
        (
            f"máscara fiel: '{m}' muestra {len(vis)} = {len(keep)} visibles, dominio primero",
            vis == keep and m == "*******.*******ar7@mailbox.test",
        )
    )
    leak0 = 0
    for _ in range(300):
        t, op, mc = (str(r.choice(PII_TYPES)), str(r.choice(OPERATORS)), str(r.choice(MASK_CHARS)))
        v = GEN[t](r)
        m = mask(v, reveal_set(v, 0.0, op, t, r), mc)
        leak0 += sum((m[i] != mc for i in ident_positions(v)))
        s = make_sample(r, t, 0.0, op, str(r.choice(EVASIONS)), str(r.choice(CARRIERS)))
        leak0 += s.spans[0].revealed
    tests.append(("blanco d=0: toda posición identificadora queda enmascarada (300 muestras)", leak0 == 0))
    ok = True
    for _ in range(400):
        s = make_sample(
            r,
            str(r.choice(PII_TYPES)),
            float(r.choice(DOSES)),
            str(r.choice(OPERATORS)),
            str(r.choice(EVASIONS)),
            str(r.choice(CARRIERS)),
            add_decoy=r.random() < 0.5,
        )
        sp = s.spans[0]
        ok &= s.text[sp.start : sp.end] == sp.rendered
        for a, b, d in s.decoys:
            ok &= s.text[a:b] == d
    tests.append(("offsets exactos de spans y señuelos (400 muestras, incluye OCR)", ok))
    a = [x.to_json() for x in generate(200, seed=11)]
    b = [x.to_json() for x in generate(200, seed=11)]
    tests.append(("determinista con la misma semilla", json.dumps(a) == json.dumps(b)))
    tests.append(
        (
            "LOD exige monotonía",
            lod_from_rates([0, 0.1, 0.2, 0.5, 1], [0, 0.96, 0.5, 0.97, 1]) == 0.5
            and lod_from_rates([0, 0.5, 1], [0, 0.5, 0.9]) is None,
        )
    )
    lo, hi = wilson(19, 20)
    tests.append((f"Wilson 19/20 = ({lo:.3f}, {hi:.3f})", abs(lo - 0.764) < 0.01 and abs(hi - 0.991) < 0.01))
    res = evaluate(regex_detector, reps=40, seed=5, types=("EMAIL",), evasions=("none", "spaced"))
    clean, spaced = res["cells"]
    tests.append(
        (
            f"regex correo: d=1 sin evasión {clean['rates']['1.0']:.2f}, d=1 espaciado {spaced['rates']['1.0']:.2f}, d=0.5 {clean['rates']['0.5']:.2f}",
            clean["rates"]["1.0"] >= 0.9 and spaced["rates"]["1.0"] <= 0.2,
        )
    )
    tr = generate(1500, seed=21, blank_frac=0.5)
    te = generate(600, seed=22)
    nb = NgramLogReg().fit([s.text for s in tr], [s.label for s in tr], epochs=3)
    a_ = auc([s.label for s in te], nb([s.text for s in te]))
    tests.append((f"n-gramas: AUC en prueba = {a_:.3f} (> 0.75)", a_ > 0.75))
    es = generate(600, seed=44, lang="es")
    en = generate(600, seed=44, lang="en")
    no_es = not any(
        (
            w in s.text
            for s in en
            for w in ("Correo", "Nombre", "en línea", "Guardar", "Pedido", "oye,", "Asunto")
        )
    )
    offs = all((s.text[sp.start : sp.end] == sp.rendered for s in en for sp in s.spans)) and all(
        (s.text[a:b] == d for s in en for a, b, d in s.decoys)
    )
    pe, pn = (np.mean([s.label for s in es]), np.mean([s.label for s in en]))
    tests.append(
        (
            f"idioma: el inglés no arrastra español, offsets exactos, tasa de positivos es {pe:.2f} / en {pn:.2f}",
            no_es and offs and (abs(pe - pn) < 0.08),
        )
    )
    print(f"leak_forge v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{('ok' if ok else 'FAIL')}] {name}")
    print("Honestidad: valida el GENERADOR y el EVALUADOR. No dice nada de ACH todavía.")
    return 0 if all((ok for _, ok in tests)) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Estándar de dosis para detectores de fuga parcial de PII")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=20000)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", default="forge_train.jsonl")
    b = sub.add_parser("baselines")
    b.add_argument("--reps", type=int, default=60)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--train-n", type=int, default=6000)
    b.add_argument("--out", default="forge_baselines.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "generate":
        with open(a.out, "w", encoding="utf-8") as f:
            for s in generate(a.n, a.seed):
                f.write(json.dumps(s.to_json(), ensure_ascii=False) + "\n")
        print(f"[forge] {a.n} muestras -> {a.out}")
        return 0
    if a.cmd == "baselines":
        tr = generate(a.train_n, seed=a.seed + 1000, blank_frac=0.5)
        nb = NgramLogReg().fit([s.text for s in tr], [s.label for s in tr])
        out = {}
        for name, det in (("regex (Presidio-like)", regex_detector), ("n-gramas + logreg", nb)):
            res = evaluate(det, a.reps, a.seed)
            out[name] = res
            print(report(res, name) + "\n")
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"[forge] -> {a.out}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
