"""  
|---|---|
| `context_grammar.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Propose it. |
"""
-----

from __future__ import annotations
import argparse
import itertools
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leak_forge as F

VERSION = "grammar-0.2"
TYPES = ("IP", "PHONE", "ID")
CLASSES = ("PER", "SYS", "EXM")
LABEL = {"PER": 1, "SYS": 0, "EXM": 0}
CONDITIONS = ("cong_seen", "cong_lex", "cong_frame", "contrast", "incong")
LANGS = ("en", "es")
PH = "\x00X\x00"
LEX: Dict[str, dict] = {
    "en": {
        "noun": {
            "IP": ["IP address", "network address"],
            "PHONE": ["phone number", "contact number"],
            "ID": ["ID code", "record number"],
        },
        "PER": {
            "seen": [
                "the customer",
                "the patient",
                "the applicant",
                "the tenant",
                "the night nurse",
                "the bus driver",
                "the school teacher",
                "the store clerk",
            ],
            "held": ["the juror", "the refugee", "the hotel guest", "the truck mechanic"],
        },
        "SYS": {
            "IP": {
                "seen": [
                    "the firewall",
                    "the router",
                    "the mainframe",
                    "the mail server",
                    "the load balancer",
                    "the backup node",
                ],
                "held": ["the hypervisor", "the modem", "the cache cluster", "the staging proxy"],
            },
            "PHONE": {
                "seen": [
                    "the switchboard",
                    "the hotline",
                    "the helpdesk",
                    "the front desk",
                    "the billing department",
                    "the sales office",
                ],
                "held": ["the pharmacy", "the laundromat", "the reception kiosk", "the claims counter"],
            },
            "ID": {
                "seen": [
                    "the forklift",
                    "the pallet",
                    "the invoice",
                    "the shipping crate",
                    "the spare part",
                    "the storage rack",
                ],
                "held": ["the turbine", "the cartridge", "the circuit board", "the cargo container"],
            },
        },
        "EXM": {
            "seen": [
                "the placeholder",
                "the mockup",
                "the template",
                "the dummy account",
                "the demo profile",
                "the sample row",
            ],
            "held": ["the specimen", "the stub", "the fictional persona", "the test fixture"],
        },
        "v_sys": ["handles", "inspected", "registered", "manages"],
        "v_exm": ["created", "drafted", "submitted", "reviewed"],
        "v_per": ["serves", "greets", "assists", "notifies"],
        "generic": ["the unit", "the station", "the terminal", "the machine"],
        "fict": [("an", "imaginary"), ("an", "invented"), ("a", "pretend"), ("a", "hypothetical")],
        "frames": {
            "seen": [
                ("pre", "{N} of {H}: {X}"),
                ("pre", "{Hc}'s {n} is {X}"),
                ("pre", "For {H}, {an} on file is {X}"),
                ("post", "{X} is {an} of {H}"),
                ("post", "{X} ({n} of {H})"),
            ],
            "held": [
                ("pre", "On record as {an} belonging to {H}: {X}"),
                ("post", "{X} was logged as {H}'s {n}"),
            ],
        },
        "contrast": [
            (True, "{N} of {S1}, not of {S2}: {X}"),
            (False, "{N} not of {S1}, but of {S2}: {X}"),
            (True, "{X} is {an} of {S1}, not of {S2}"),
            (False, "{X} is {an} not of {S1}, but of {S2}"),
        ],
        "incong": {
            "whoever": [("pre", "{N} of whoever {v} {O}: {X}"), ("post", "{X} is {an} of whoever {v} {O}")],
            "generic": [
                ("pre", "{N} of {G} that {v} every {p}: {X}"),
                ("post", "{X} is {an} of {G} that {v} every {p}"),
            ],
            "fict": [("pre", "{N} of {art} {f} {p}: {X}"), ("post", "{X} is {an} of {art} {f} {p}")],
        },
        "chrome": ["{P}", "Tab · {P} — Browser", "Notification: {P}", "Window | {P} | 3 unread"],
    },
    "es": {
        "noun": {
            "IP": ["la dirección IP", "la dirección de red"],
            "PHONE": ["el número de teléfono", "el número de contacto"],
            "ID": ["el código de identificación", "el número de expediente"],
        },
        "PER": {
            "seen": [
                "el cliente",
                "la paciente",
                "el solicitante",
                "la inquilina",
                "la enfermera de noche",
                "el chofer de autobús",
                "la maestra de escuela",
                "el empleado de mostrador",
            ],
            "held": ["el heredero", "la refugiada", "el huésped del hotel", "la mecánica de camiones"],
        },
        "SYS": {
            "IP": {
                "seen": [
                    "el cortafuegos",
                    "el enrutador",
                    "la intranet",
                    "el servidor de correo",
                    "la pasarela de cobro",
                    "la subred de pruebas",
                ],
                "held": ["el hipervisor", "la antena", "el clúster de caché", "la puerta VPN"],
            },
            "PHONE": {
                "seen": [
                    "el conmutador",
                    "la recepción",
                    "el consultorio",
                    "la línea de soporte",
                    "el área de facturación",
                    "la oficina de ventas",
                ],
                "held": ["el taller", "la lavandería", "el módulo de informes", "la mesa de reclamos"],
            },
            "ID": {
                "seen": [
                    "el montacargas",
                    "la tarima",
                    "la factura",
                    "la caja de envío",
                    "el lote de refacciones",
                    "el anaquel de bodega",
                ],
                "held": ["el cartucho", "la turbina", "el contenedor de carga", "la placa de circuito"],
            },
        },
        "EXM": {
            "seen": [
                "la plantilla",
                "la maqueta",
                "el simulacro",
                "la cuenta de prueba",
                "el perfil de demostración",
                "el registro de relleno",
            ],
            "held": ["el espécimen", "la réplica", "el personaje ficticio", "la ficha de ensayo"],
        },
        "v_sys": ["maneja", "inspeccionó", "etiquetó", "administra"],
        "v_exm": ["creó", "redactó", "envió", "revisó"],
        "v_per": ["atiende", "recibe", "saluda", "avisa"],
        "generic": ["la unidad", "la estación", "la terminal", "la máquina"],
        "fict": [
            ("imaginario", "imaginaria"),
            ("inventado", "inventada"),
            ("hipotético", "hipotética"),
            ("de mentira", "de mentira"),
        ],
        "frames": {
            "seen": [
                ("pre", "{N} de {H}: {X}"),
                ("pre", "{An} de {H} es {X}"),
                ("pre", "Para {H}, {an} en el expediente es {X}"),
                ("post", "{X} es {an} de {H}"),
                ("post", "{X} ({n} de {H})"),
            ],
            "held": [
                ("pre", "Consta como {an} que pertenece a {H}: {X}"),
                ("post", "{X} quedó anotado como {an} de {H}"),
            ],
        },
        "contrast": [
            (True, "{N} de {S1}, no de {S2}: {X}"),
            (False, "{N} no de {S1}, sino de {S2}: {X}"),
            (True, "{X} es {an} de {S1}, no de {S2}"),
            (False, "{X} es {an} no de {S1}, sino de {S2}"),
        ],
        "incong": {
            "whoever": [("pre", "{N} de quien {v} {O}: {X}"), ("post", "{X} es {an} de quien {v} {O}")],
            "generic": [
                ("pre", "{N} de {G} que {v} a cada {p}: {X}"),
                ("post", "{X} es {an} de {G} que {v} a cada {p}"),
            ],
            "fict": [("pre", "{N} de {art} {p} {f}: {X}"), ("post", "{X} es {an} de {art} {p} {f}")],
        },
        "chrome": ["{P}", "Pestaña · {P} — Navegador", "Notificación: {P}", "Ventana | {P} | 3 sin leer"],
    },
}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _bare(holder: str) -> str:
    return holder.split(" ", 1)[1]


def _contract(lang: str, s: str) -> str:
    return s.replace(" de el ", " del ").replace(" a el ", " al ") if lang == "es" else s


def _noun_fields(lang: str, noun: str) -> dict:
    if lang == "en":
        return {"N": _cap(noun), "n": noun, "an": "the " + noun, "An": "The " + noun}
    return {"N": _cap(_bare(noun)), "n": _bare(noun), "an": noun, "An": _cap(noun)}


def holders(lang: str, cls: str, t: str, split: str) -> List[str]:
    L = LEX[lang]
    return L["SYS"][t][split] if cls == "SYS" else L[cls][split]


def _congruent(lang: str, tmpl: str, noun: str, H: str) -> str:
    return _contract(lang, tmpl.format(H=H, Hc=_cap(H), X=PH, **_noun_fields(lang, noun)))


def _contrast(lang: str, tmpl: str, noun: str, first: bool, A: str, B: str) -> str:
    return _contract(
        lang, tmpl.format(S1=A if first else B, S2=B if first else A, X=PH, **_noun_fields(lang, noun))
    )


def _incong(lang: str, kind: str, tmpl: str, noun: str, r, t: str) -> Tuple[str, str, str]:
    L, nf = (LEX[lang], _noun_fields(lang, noun))
    per = str(r.choice(L["PER"]["seen"]))
    if kind in ("whoever_sys", "whoever_exm"):
        O = str(r.choice(holders(lang, "SYS", t, "seen") if kind == "whoever_sys" else L["EXM"]["seen"]))
        v = str(r.choice(L["v_sys"] if kind == "whoever_sys" else L["v_exm"]))
        return (
            _contract(lang, tmpl.format(v=v, O=O, X=PH, **nf)),
            "PER",
            "SYS" if kind == "whoever_sys" else "EXM",
        )
    if kind == "generic":
        return (
            _contract(
                lang,
                tmpl.format(
                    G=str(r.choice(L["generic"])), v=str(r.choice(L["v_per"])), p=_bare(per), X=PH, **nf
                ),
            ),
            "SYS",
            "PER",
        )
    f = L["fict"][int(r.integers(len(L["fict"])))]
    if lang == "en":
        art, fw = f
    else:
        fem = per.startswith("la ")
        art, fw = ("una" if fem else "un", f[1] if fem else f[0])
    return (_contract(lang, tmpl.format(art=art, f=fw, p=_bare(per), X=PH, **nf)), "EXM", "PER")


def _place(
    lang: str, r, phrase: str, x: str, chrome_i: int, pad: Tuple[str, str]
) -> Tuple[str, Tuple[int, int]]:
    body = LEX[lang]["chrome"][chrome_i].format(P=phrase)
    text = pad[0] + "\n" + body + "\n" + pad[1]
    assert text.count(PH) == 1, "el hueco del dato debe aparecer exactamente una vez"
    a = text.index(PH)
    return (text.replace(PH, x), (a, a + len(x)))


def make_group(
    r, t: str, cond: str, lang: str = "en", dose: float = 1.0, frame_i: Optional[int] = None
) -> Dict:
    L = LEX[lang]
    v = F.GEN[t](r)
    keep = F.reveal_set(v, dose, "suffix", t, r)
    x = F.mask(v, keep, "*")
    noun = str(r.choice(L["noun"][t]))
    chrome_i = int(r.integers(len(L["chrome"])))
    UI = F.CHROME_BY_LANG[lang]
    pad = tuple((str(c).format(ch="#general") for c in r.choice(UI, size=2, replace=False)))
    variants, pairs = ([], [])

    def add(phrase, cls, cue, pos, frame):
        text, span = _place(lang, r, phrase, x, chrome_i, pad)
        variants.append(
            {
                "text": text,
                "span": span,
                "label": LABEL[cls],
                "cls": cls,
                "cue": cue,
                "pos": pos,
                "frame": frame,
                "chrome": chrome_i,
            }
        )
        return len(variants) - 1

    if cond in ("cong_seen", "cong_lex", "cong_frame"):
        fsplit = "held" if cond == "cong_frame" else "seen"
        lsplit = "held" if cond == "cong_lex" else "seen"
        fi = int(r.integers(len(L["frames"][fsplit])))
        pos, tmpl = L["frames"][fsplit][fi]
        idx = {
            c: add(
                _congruent(lang, tmpl, noun, str(r.choice(holders(lang, c, t, lsplit)))),
                c,
                c,
                pos,
                f"{fsplit}{fi}",
            )
            for c in CLASSES
        }
        pairs = [(idx["PER"], idx["SYS"]), (idx["PER"], idx["EXM"])]
    elif cond == "contrast":
        fi = int(r.integers(len(L["contrast"]))) if frame_i is None else frame_i % len(L["contrast"])
        first, tmpl = L["contrast"][fi]
        per = str(r.choice(L["PER"]["seen"]))
        for neg in ("SYS", "EXM"):
            other = str(r.choice(holders(lang, neg, t, "seen")))
            i = add(
                _contrast(lang, tmpl, noun, first, per, other),
                "PER",
                "both",
                "first" if first else "second",
                f"c{fi}",
            )
            j = add(
                _contrast(lang, tmpl, noun, first, other, per),
                neg,
                "both",
                "first" if first else "second",
                f"c{fi}",
            )
            pairs.append((i, j))
    elif cond == "incong":
        pi = int(r.integers(2))
        ids = {}
        for kind, key in (
            ("whoever_sys", "whoever"),
            ("whoever_exm", "whoever"),
            ("generic", "generic"),
            ("fict", "fict"),
        ):
            pos, tmpl = L["incong"][key][pi]
            phrase, cls, cue = _incong(lang, kind, tmpl, noun, r, t)
            ids[kind] = add(phrase, cls, cue, pos, kind)
        pairs = [(ids[p], ids[n]) for p in ("whoever_sys", "whoever_exm") for n in ("generic", "fict")]
    else:
        raise ValueError(cond)
    return {
        "pii_type": t,
        "cond": cond,
        "lang": lang,
        "original": v,
        "rendered": x,
        "dose": len(keep) / len(F.ident_positions(v)),
        "variants": variants,
        "pairs": pairs,
    }


def groups(
    n: int, seed: int, cond: str, lang: str = "en", dose_choices: Sequence[float] = (1.0,)
) -> List[Dict]:
    r = np.random.default_rng([seed, CONDITIONS.index(cond), LANGS.index(lang)])
    return [
        make_group(
            r,
            TYPES[i % len(TYPES)],
            cond,
            lang,
            float(r.choice(list(dose_choices))),
            frame_i=i // len(TYPES) if cond == "contrast" else None,
        )
        for i in range(n)
    ]


def train_groups(n: int, seed: int, lang: str = "en") -> List[Dict]:
    return groups(n, seed, "cong_seen", lang, dose_choices=(0.0, 1.0, 1.0))


Detector = Callable[[List[str]], np.ndarray]


def boot_ci(per_group: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    g = np.asarray(per_group, float)
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    m = g[idx].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def context_eval(
    detector: Detector,
    n: int,
    seed: int,
    cond: str,
    lang: str = "en",
    thr: float = 0.5,
    text_fn: Optional[Callable[[Dict], str]] = None,
) -> Dict:
    gs = groups(n, seed, cond, lang)
    flat = [(gi, vi, v) for gi, g in enumerate(gs) for vi, v in enumerate(g["variants"])]
    scores = np.asarray(detector([text_fn(v) if text_fn else v["text"] for _, _, v in flat]), float)
    S = {(gi, vi): s for (gi, vi, _), s in zip(flat, scores)}
    per_group, by = ([], {})
    for gi, g in enumerate(gs):
        w = []
        for i, j in g["pairs"]:
            a, b = (S[gi, i], S[gi, j])
            val = 1.0 if a > b else 0.5 if a == b else 0.0
            w.append(val)
            key = g["variants"][i]["pos"] if cond != "incong" else g["variants"][j]["frame"]
            by.setdefault(key, []).append(val)
        per_group.append(float(np.mean(w)))
    per_group = np.array(per_group)
    fire = {}
    for c in CLASSES:
        s = [sc for (_, _, v), sc in zip(flat, scores) if v["cls"] == c]
        fire[c] = float(np.mean(np.asarray(s) >= thr)) if s else None
    return {
        "cond": cond,
        "lang": lang,
        "n_groups": n,
        "pair_acc": float(per_group.mean()),
        "pair_acc_ci": boot_ci(per_group, seed=seed),
        "per_group": per_group.tolist(),
        "fire_rate": fire,
        "by": {k: float(np.mean(v)) for k, v in sorted(by.items())},
    }


def inventory(lang: str = "en") -> Dict:
    L, out = (LEX[lang], {})
    for t in TYPES:
        nn = len(L["noun"][t])
        seen = sum((len(holders(lang, c, t, "seen")) for c in CLASSES))
        held = sum((len(holders(lang, c, t, "held")) for c in CLASSES))
        nper, nsys, nexm = (
            len(L["PER"]["seen"]),
            len(holders(lang, "SYS", t, "seen")),
            len(L["EXM"]["seen"]),
        )
        out[t] = {
            "cong_seen": nn * len(L["frames"]["seen"]) * seen,
            "cong_lex": nn * len(L["frames"]["seen"]) * held,
            "cong_frame": nn * len(L["frames"]["held"]) * seen,
            "contrast": nn * len(L["contrast"]) * 2 * nper * (nsys + nexm),
            "incong": nn
            * 2
            * (
                len(L["v_sys"]) * nsys
                + len(L["v_exm"]) * nexm
                + len(L["generic"]) * len(L["v_per"]) * nper
                + len(L["fict"]) * nper
            ),
        }
    out["total"] = {c: sum((out[t][c] for t in TYPES)) for c in CONDITIONS}
    return out


def oracle(lang: str = "en") -> Detector:
    L = LEX[lang]
    cls_of = {}
    for c in CLASSES:
        for t in TYPES:
            for sp in ("seen", "held"):
                for h in holders(lang, c, t, sp):
                    cls_of[_bare(h)] = c
    bare_sorted = sorted(cls_of, key=len, reverse=True)
    fict_words = {w for f in L["fict"] for w in (f if lang == "es" else f[1:])}
    who = "whoever" if lang == "en" else " de quien "
    gen = [_bare(g) for g in L["generic"]]
    neg = re.compile("\\b[Nn]ot\\b" if lang == "en" else "\\b[Nn]o\\b")

    def found(body: str) -> List[Tuple[int, str]]:
        hits, taken = ([], np.zeros(len(body), bool))
        for b_ in bare_sorted:
            for m in re.finditer(f"\\b{re.escape(b_)}\\b", body):
                if not taken[m.start() : m.end()].any():
                    taken[m.start() : m.end()] = True
                    hits.append((m.start(), b_))
        return sorted(hits)

    def classify(text: str) -> float:
        body = text.split("\n")[1]
        if who in body:
            return 1.0
        if any((re.search(f"\\b{re.escape(w)}\\b", body) for w in fict_words)):
            return 0.0
        if any((re.search(f"\\b{re.escape(g)}\\b", body) for g in gen)):
            return 0.0
        hs, m = (found(body), neg.search(body))
        if len(hs) >= 2 and m:
            after = [h for h in hs if h[0] > m.start()]
            negated = after[0][1]
            truth = [h[1] for h in hs if h[1] != negated][0]
            return float(LABEL[cls_of[truth]])
        return float(LABEL[cls_of[hs[0][1]]]) if hs else 0.5

    return lambda texts: np.array([classify(t) for t in texts])


def keyword_baseline(seed: int, lang: str = "en", n_train: int = 400) -> Detector:
    X, y = ([], [])
    for g in train_groups(n_train, seed, lang):
        for v in g["variants"]:
            X.append(v["text"])
            y.append(v["label"])
    clf = F.NgramLogReg(seed=seed).fit(X, y, epochs=5)
    return lambda texts: clf(list(texts))


class WordLogReg(F.NgramLogReg):

    def feats(self, t: str) -> np.ndarray:
        import zlib

        w = re.findall("[^\\W\\d_]+", t.lower())
        g = w + [a + " " + b for a, b in zip(w, w[1:])]
        return np.unique(np.asarray([zlib.crc32(x.encode("utf-8")) % self.dim for x in g] or [0], np.int64))


def word_baseline(seed: int, lang: str = "en", n_train: int = 400) -> Detector:
    X, y = ([], [])
    for g in train_groups(n_train, seed, lang):
        for v in g["variants"]:
            X.append(v["text"])
            y.append(v["label"])
    clf = WordLogReg(seed=seed).fit(X, y, epochs=5)
    return lambda texts: clf(list(texts))


def audit(lang: str = "en", n: int = 150, seed: int = 20261110) -> Dict:
    L, checks = (LEX[lang], [])

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    seen_all = {h for c in CLASSES for t in TYPES for h in holders(lang, c, t, "seen")}
    held_all = {h for c in CLASSES for t in TYPES for h in holders(lang, c, t, "held")}
    add(
        "léxico reservado disjunto del visto",
        not seen_all & held_all,
        f"{len(seen_all)} vistos, {len(held_all)} reservados",
    )
    words = lambda hs: {w.lower() for h in hs for w in re.findall("\\w+", _bare(h))}
    shared = (words(held_all) & words(seen_all)) - {"de", "del", "la", "el"}
    add(
        "ningún sustantivo reservado comparte palabras con los vistos",
        not shared,
        f"compartidas: {sorted(shared)}",
    )
    train_vocab = {
        w.lower()
        for g in train_groups(300, seed + 1, lang)
        for v in g["variants"]
        for w in re.findall("\\w+", v["text"])
    }
    scaffold = set(L["v_sys"]) | set(L["v_exm"]) | set(L["v_per"]) | {_bare(g) for g in L["generic"]}
    scaffold |= {w for f in L["fict"] for w in (f if lang == "es" else f[1:])} | (
        {"whoever", "every", "that"} if lang == "en" else {"quien", "cada"}
    )
    leak = {w for s in scaffold for w in re.findall("\\w+", s.lower())} & train_vocab
    leak -= {"de"}
    add("el andamiaje incongruente no aparece al entrenar", not leak, f"aparecen: {sorted(leak)}")
    orc = oracle(lang)
    fam = {"ngramas": keyword_baseline(seed + 2, lang), "palabras": word_baseline(seed + 2, lang)}
    res = {}
    for cond in CONDITIONS:
        o = context_eval(orc, n, seed, cond, lang)["pair_acc"]
        res[cond] = {"oracle": o}
        for name, det in fam.items():
            k = context_eval(det, n, seed, cond, lang)
            res[cond][name] = k["pair_acc"]
            res[cond][name + "_by"] = k["by"]
        add(f"oráculo de reglas en {cond}", o == 1.0, f"{o:.3f}")
    kws = lambda c: {k: round(res[c][k], 2) for k in fam}
    add(
        "palabras clave APRENDEN cong_seen",
        min((res["cong_seen"][k] for k in fam)) >= 0.9,
        str(kws("cong_seen")),
    )
    add(
        "palabras clave en el AZAR con dueños nunca vistos, cong_lex (±0.15)",
        max((abs(res["cong_lex"][k] - 0.5) for k in fam)) <= 0.15,
        str(kws("cong_lex")),
    )
    multi = {
        (c, t, sp): float(np.mean([" " in _bare(h) for h in holders(lang, c, t, sp)]))
        for c in CLASSES
        for t in TYPES
        for sp in ("seen", "held")
    }
    add(
        "longitud balanceada: mitad de los dueños de una palabra en cada lista",
        all((abs(v - 0.5) < 1e-09 for v in multi.values())),
        f"fracción de varias palabras: {sorted(set((round(v, 2) for v in multi.values())))}",
    )
    short = lambda texts: np.array([-float(len(t)) for t in texts])
    for cond in ("cong_seen", "cong_lex"):
        h = context_eval(short, n, seed, cond, lang)["pair_acc"]
        res[cond]["longitud"] = h
        add(
            f"la heurística 'texto corto = persona' queda en el azar en {cond} (±0.15)",
            abs(h - 0.5) <= 0.15,
            f"{h:.2f}",
        )
    add(
        "palabras clave en el AZAR en contrast (las dos familias, ±0.10)",
        max((abs(res["contrast"][k] - 0.5) for k in fam)) <= 0.1,
        str(kws("contrast")),
    )
    add(
        "palabras clave POR DEBAJO del azar en incong",
        max((res["incong"][k] for k in fam)) <= 0.25,
        str(kws("incong")),
    )
    gs = groups(400, seed, "contrast", lang)
    first = np.mean([g["variants"][i]["pos"] == "first" for g in gs for i, _ in g["pairs"]])
    add(
        "contrast: el verdadero va primero la mitad de las veces (exacto por rotación)",
        abs(first - 0.5) <= 0.01,
        f"{first:.3f}",
    )
    tg = train_groups(600, seed + 3, lang)
    by = {}
    for g in tg:
        for v in g["variants"]:
            by.setdefault(v["chrome"], []).append(v["label"])
    rates = [float(np.mean(v)) for v in by.values()]
    add(
        "el cromo no predice la etiqueta",
        max(rates) - min(rates) < 1e-09,
        f"tasa de positivos por cromo: {[round(x, 3) for x in rates]}",
    )
    if lang == "es":
        fem = {
            (c, t): float(np.mean([h.startswith("la ") for h in holders(lang, c, t, "seen")]))
            for c in CLASSES
            for t in TYPES
        }
        add(
            "es: género balanceado en cada lista vista (del / de la no delata la clase)",
            all((abs(v - 0.5) < 1e-09 for v in fem.values())),
            f"fracción femenina: {sorted(set((round(v, 2) for v in fem.values())))}",
        )
    return {
        "lang": lang,
        "ok": all((ok for _, ok, _ in checks)),
        "checks": checks,
        "conditions": res,
        "certifies": {
            "cong_lex": max((abs(res["cong_lex"][k] - 0.5) for k in fam)) <= 0.15,
            "contrast": max((abs(res["contrast"][k] - 0.5) for k in fam)) <= 0.1,
            "incong": max((res["incong"][k] for k in fam)) <= 0.25,
        },
    }


def selftest() -> int:
    tests = []
    for lang in LANGS:
        r = np.random.default_rng(5)
        ok = True
        for cond in CONDITIONS:
            for i in range(60):
                g = make_group(r, TYPES[i % 3], cond, lang)
                for v in g["variants"]:
                    a, b = v["span"]
                    ok &= v["text"][a:b] == g["rendered"] and PH not in v["text"] and ("{" not in v["text"])
                ok &= all(
                    (g["variants"][i]["label"] == 1 and g["variants"][j]["label"] == 0 for i, j in g["pairs"])
                )
                ok &= len({v["chrome"] for v in g["variants"]}) == 1
                ok &= len({v["text"].split("\n")[0] + v["text"].split("\n")[2] for v in g["variants"]}) == 1
        tests.append(
            (
                f"[{lang}] grupos bien formados: dato idéntico, offsets exactos, pares (1, 0), mismo cromo y relleno",
                ok,
            )
        )
        if lang == "es":
            alltx = [v for c in CONDITIONS for g in groups(120, 1, c, "es") for v in g["variants"]]
            contr = [v["text"] for v in alltx if re.search("\\bde el\\b|\\ba el\\b", v["text"])]
            masc, fem = ({f[0] for f in LEX["es"]["fict"]}, {f[1] for f in LEX["es"]["fict"]})
            gen_bad = []
            for v in alltx:
                if v["frame"] == "fict":
                    body = v["text"].split("\n")[1]
                    ok_g = re.search(
                        "\\bun [\\w ]+? (" + "|".join(masc) + ")(:|$| \\||\\s—)", body
                    ) or re.search("\\buna [\\w ]+? (" + "|".join(fem) + ")(:|$| \\||\\s—)", body)
                    if not ok_g:
                        gen_bad.append(body)
            tests.append(
                (
                    f"[es] contracciones del/al ({len(contr)} errores) y concordancia de género ({len(gen_bad)} errores)",
                    not contr and (not gen_bad),
                )
            )
        a = audit(lang, n=120)
        for name, ok_, detail in a["checks"]:
            tests.append((f"[{lang}] auditoría · {name}: {detail}", ok_))
        inv = inventory(lang)["total"]
        tests.append(
            (
                f"[{lang}] inventario de frases distintas: {inv}",
                inv["incong"] >= 500 and inv["cong_lex"] >= 300,
            )
        )
    g1 = [v["text"] for g in groups(40, 9, "incong", "en") for v in g["variants"]]
    g2 = [v["text"] for g in groups(40, 9, "incong", "en") for v in g["variants"]]
    tests.append(("determinista con la misma semilla", g1 == g2))
    lo, hi = boot_ci(np.r_[np.ones(50), np.zeros(50)], seed=1)
    tests.append(
        (
            f"bootstrap por grupos: 50/100 -> IC ({lo:.2f}, {hi:.2f}) contiene 0.5 y no es degenerado",
            lo < 0.5 < hi and hi - lo > 0.1,
        )
    )
    print(f"context_grammar v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{('ok' if ok else 'FAIL')}] {name}")
    return 0 if all((ok for _, ok in tests)) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Gramática factorial del eje de contexto")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("inventory", "dump", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--lang", default="en", choices=LANGS)
        if name == "dump":
            p.add_argument("--cond", default="incong", choices=CONDITIONS)
            p.add_argument("--n", type=int, default=12)
            p.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "inventory":
        inv = inventory(a.lang)
        for k, v in inv.items():
            print(f"{k:6s} {v}")
    elif a.cmd == "dump":
        for g in groups(a.n, a.seed, a.cond, a.lang):
            for v in g["variants"]:
                print(
                    f"  [{v['label']}] {v['cls']} (pista {v['cue']}, {v['pos']}):  {v['text'].splitlines()[1]}"
                )
            print()
    elif a.cmd == "audit":
        r = audit(a.lang)
        for name, ok, detail in r["checks"]:
            print(f"  [{('ok' if ok else 'FAIL')}] {name}: {detail}")
        return 0 if r["ok"] else 1
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
