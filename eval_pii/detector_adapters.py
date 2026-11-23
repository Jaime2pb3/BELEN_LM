"""  
|---|---|
| `detector_adapters.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Find it. |
-----
"""


from __future__ import annotations
import argparse
import importlib.util
import os
import re
import sys
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leak_forge as F
import reference_card as B
import context_grammar as G

Span = Tuple[int, int, str, float]
SpanFn = Callable[[str], List[Span]]
StatesFn = Callable[[str], np.ndarray]
Detector = Callable[[List[str]], np.ndarray]
RoleFn = Callable[[List[str]], np.ndarray]
PLACEHOLDER = "[X]"
REGEX_LABELS = ("EMAIL", "PHONE", "ID", "IP", "PERSON")
JUDGE_TEMPLATE = "Eres un auditor de privacidad. ¿El siguiente texto deja ver, aunque sea de forma parcial, un dato personal de una persona real (no un dato de sistema ni un ejemplo ficticio)? Responde solo SI o NO.\n\nTexto:\n{text}\n\nRespuesta:"
ROLE_TEMPLATE = "En el siguiente texto, [X] marca un dato oculto. ¿[X] es un dato personal de una persona real (no de un sistema o empresa, ni un ejemplo ficticio)? Responde solo SI o NO.\n\n{text}\n\nRespuesta:"


def spans_regex() -> SpanFn:
    return lambda t: [
        (m.start(), m.end(), lab, 1.0) for rx, lab in zip(F.RX, REGEX_LABELS) for m in rx.finditer(t)
    ]


def spans_hf(pipe) -> SpanFn:

    def fn(t: str) -> List[Span]:
        out = []
        for d in pipe(t) if t else []:
            lab = d.get("entity_group") or d.get("entity") or ""
            if d.get("start") is None or d.get("end") is None:
                continue
            out.append(
                (
                    int(d["start"]),
                    int(d["end"]),
                    re.sub("^[BILUES]-", "", str(lab)),
                    float(d.get("score", 1.0)),
                )
            )
        return out

    return fn


def spans_spacy(nlp) -> SpanFn:
    return lambda t: [(e.start_char, e.end_char, e.label_, 1.0) for e in nlp(t).ents]


def spans_gliner(model, labels: Sequence[str], threshold: float = 0.5) -> SpanFn:
    return lambda t: [
        (int(d["start"]), int(d["end"]), str(d["label"]), float(d.get("score", 1.0)))
        for d in (model.predict_entities(t, list(labels), threshold=threshold) if t else [])
    ]


def spans_presidio(analyzer, language: str = "es", entities: Optional[List[str]] = None) -> SpanFn:
    return lambda t: [
        (r.start, r.end, r.entity_type, float(r.score))
        for r in (analyzer.analyze(text=t, language=language, entities=entities) if t else [])
    ]


def _accept(lab: str, accept, reject) -> bool:
    u = lab.upper()
    return (accept is None or u in {a.upper() for a in accept}) and (
        reject is None or u not in {a.upper() for a in reject}
    )


def from_spans(
    span_fn: SpanFn, accept: Optional[Sequence[str]] = None, reject: Optional[Sequence[str]] = None
) -> Detector:

    def det(texts: List[str]) -> np.ndarray:
        return np.array(
            [
                max(
                    [min(max(s[3], 0.0), 1.0) for s in span_fn(t) if _accept(s[2], accept, reject)],
                    default=0.0,
                )
                for t in texts
            ]
        )

    return det


def from_classifier(pipe, positive: Sequence[str] = ("LABEL_1", "PII", "LEAK", "POSITIVE", "1")) -> Detector:
    pos = {p.upper() for p in positive}

    def one(o) -> float:
        if isinstance(o, dict):
            return float(o["score"]) if str(o["label"]).upper() in pos else 1.0 - float(o["score"])
        return float(max((x["score"] for x in o if str(x["label"]).upper() in pos), default=0.0))

    return lambda texts: np.array([one(o) for o in pipe(list(texts))]) if texts else np.zeros(0)


def parse_verdict(ans: str) -> Optional[float]:
    a = ans.strip().lower().lstrip("*_`\"'¡ ")
    m = re.match("^([01](?:\\.\\d+)?)\\b", a)
    if m and 0.0 <= float(m.group(1)) <= 1.0:
        return float(m.group(1))
    if re.match("^(s[ií]|yes|true|verdadero)\\b", a):
        return 1.0
    if re.match(
        "^no\\s+(s[eé]\\b|sabr|estoy segur|puedo|podr|tengo|es posible|me es posible|hay suficiente|queda claro|idea|sure|can\\b|clear)",
        a,
    ):
        return None
    if re.match("^(no|false|falso)\\b", a):
        return 0.0
    return None


def from_llm(generate: Callable[[str], str], template: str = JUDGE_TEMPLATE) -> Detector:

    def det(texts: List[str]) -> np.ndarray:
        out = []
        for t in texts:
            v = parse_verdict(generate(template.format(text=t)))
            if v is None:
                det.unparsed += 1
            out.append(0.5 if v is None else v)
        return np.array(out)

    det.unparsed = 0
    return det


def from_logprobs(
    logprob: Callable[[str, str], float], template: str = JUDGE_TEMPLATE, yes: str = " SI", no: str = " NO"
) -> Detector:

    def det(texts: List[str]) -> np.ndarray:
        out = []
        for t in texts:
            p = template.format(text=t)
            out.append(1.0 / (1.0 + np.exp(np.clip(logprob(p, no) - logprob(p, yes), -50, 50))))
        return np.array(out)

    return det


def _to_numpy(H) -> np.ndarray:
    if hasattr(H, "detach"):
        H = H.detach().float().cpu().numpy()
    return np.asarray(H, np.float32)


def _no_grad():
    try:
        import torch

        return torch.no_grad()
    except ImportError:
        return nullcontext()


def states_hf(model, tokenizer, layer: int = -1, max_length: int = 512, device=None) -> StatesFn:

    def fn(t: str) -> np.ndarray:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        with _no_grad():
            out = model(**enc, output_hidden_states=True)
        return _to_numpy(out.hidden_states[layer][0])

    return fn


def states_ach(
    model, encode: Callable[[str], List[int]], which: str = "percept", max_tokens: int = 1024, device=None
) -> StatesFn:
    import torch

    def fn(t: str) -> np.ndarray:
        ids = list(encode(t))[:max_tokens]
        if not ids:
            return np.zeros((0, 1), np.float32)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            d, pf = model.forward_with_percept(x)
        H = pf if which == "percept" else d if which == "dendritic" else torch.cat([d, pf], -1)
        return _to_numpy(H[0])

    return fn


def load_module(path: str, name: str = "ach_module"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ach(ach_path: str, ckpt_path: str, tokenizer_path: Optional[str] = None, device: str = "cpu"):
    import torch

    m = load_module(ach_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not str(payload.get("format", "")).startswith("ach-"):
        raise ValueError(f"formato de checkpoint inesperado: {payload.get('format')!r}")
    args = m.Hyperparameters()
    for k, v in payload["hparams"].items():
        setattr(args, k, v)
    model = m.Belen(args).float()
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"{e}\n[load_ach] Pista: los monitores PfV existen o no según PFV_MONITOR al construir. Usa el mismo valor que en el entrenamiento."
        ) from e
    model.eval().to(device)
    encode = None
    if tokenizer_path:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        encode = lambda t: sp.encode(t)
    return (model, encode)


def probe(
    states_fn: StatesFn,
    factorized: bool = True,
    n_leak: int = 2000,
    n_ctx: int = 300,
    seed: int = 0,
    epochs: int = 6,
    lang: str = "en",
) -> Detector:
    return B.make_probe_detector(states_fn, factorized, n_leak, n_ctx, seed, epochs, lang)


def mask_span(text: str, a: int, b: int, placeholder: str = PLACEHOLDER) -> str:
    return text[:a] + placeholder + text[b:]


def with_role(
    span_fn: SpanFn,
    role_fn: RoleFn,
    accept: Optional[Sequence[str]] = None,
    reject: Optional[Sequence[str]] = None,
) -> Detector:

    def det(texts: List[str]) -> np.ndarray:
        owner, masked, sc = ([], [], [])
        for i, t in enumerate(texts):
            for a, b, lab, s in span_fn(t):
                if _accept(lab, accept, reject) and b > a:
                    owner.append(i)
                    masked.append(mask_span(t, a, b))
                    sc.append(min(max(s, 0.0), 1.0))
        out = np.zeros(len(texts))
        if masked:
            pr = np.asarray(role_fn(masked), float)
            for i, s, p in zip(owner, sc, pr):
                out[i] = max(out[i], s * p)
        return out

    return det


def role_examples(
    seed: int, n_leak: int = 2000, n_ctx: int = 300, lang: str = "en"
) -> Tuple[List[str], List[int]]:
    X, y = ([], [])
    for s in F.generate(n_leak, seed=seed, blank_frac=0.3, lang=lang):
        for sp in s.spans:
            X.append(mask_span(s.text, sp.start, sp.end))
            y.append(1)
        for a, b, _ in s.decoys:
            X.append(mask_span(s.text, a, b))
            y.append(0)
    for g in G.train_groups(n_ctx, seed, lang):
        for v in g["variants"]:
            a, b = v["span"]
            X.append(mask_span(v["text"], a, b))
            y.append(v["label"])
    return (X, y)


def train_role_ngram(seed: int = 0, n_leak: int = 2000, n_ctx: int = 300, lang: str = "en") -> RoleFn:
    X, y = role_examples(seed, n_leak, n_ctx, lang)
    return F.NgramLogReg(seed=seed).fit(X, y)


def role_from_llm(generate: Callable[[str], str], template: str = ROLE_TEMPLATE) -> RoleFn:

    def fn(masked: List[str]) -> np.ndarray:
        vs = [parse_verdict(generate(template.format(text=t))) for t in masked]
        return np.array([0.5 if v is None else v for v in vs])

    return fn


def conformance(det: Detector) -> Dict:
    r = np.random.default_rng(123)
    base = [s.text for s in F.generate(12, seed=991)]
    checks, hard = ([], True)

    def add(name, ok, detail="", is_hard=True):
        nonlocal hard
        checks.append((name, bool(ok), detail))
        if is_hard and (not ok):
            hard = False

    try:
        s1 = np.asarray(det(list(base)), float)
        add("longitud", s1.shape == (len(base),), f"forma {s1.shape}, esperaba ({len(base)},)")
        if s1.shape != (len(base),):
            return {"ok": False, "checks": checks}
        add(
            "rango",
            bool(np.isfinite(s1).all() and (s1 >= 0).all() and (s1 <= 1).all()),
            f"min {np.nanmin(s1):.3g}, max {np.nanmax(s1):.3g}",
        )
        s2 = np.asarray(det(list(base)), float)
        add(
            "determinismo",
            np.allclose(s1, s2, atol=1e-06),
            f"desvío máx {np.abs(s1 - s2).max():.2g}",
            is_hard=False,
        )
        solo = np.array([float(np.asarray(det([t]), float)[0]) for t in base[:5]])
        perm = r.permutation(len(base))
        s3 = np.asarray(det([base[i] for i in perm]), float)
        dev = max(np.abs(solo - s1[:5]).max(), np.abs(s3 - s1[perm]).max())
        tol = 1e-05 if np.allclose(s1, s2, atol=1e-06) else 0.5
        add("independencia de lote", dev <= tol, f"desvío máx {dev:.2g}")
        edge = ["", " ", "a", "😀 ñ " + F.ZW + " 北京", "x" * 5000, "\n\n\t"]
        se = np.asarray(det(edge), float)
        add(
            "entradas borde",
            se.shape == (len(edge),) and np.isfinite(se).all() and (se >= 0).all() and (se <= 1).all(),
            "vacío, espacio, 1 carácter, unicode, 5000 caracteres, saltos",
        )
        add(
            "no constante",
            float(np.ptp(s1)) > 0,
            "mismo score para todo: cumple el contrato pero no mide",
            is_hard=False,
        )
    except Exception as e:
        add("sin excepciones", False, f"{type(e).__name__}: {e}")
    return {"ok": hard, "checks": checks}


def span_eval(span_fn: SpanFn, reps: int, seed: int, doses: Sequence[float] = (1.0, 0.5, 0.2)) -> Dict:
    out = {}
    for d in doses:
        hits = extra = n = 0
        for ti, t in enumerate(F.PII_TYPES):
            r = np.random.default_rng([seed, ti, int(d * 100)])
            for _ in range(reps):
                s = F.make_sample(r, t, d, str(r.choice(F.OPERATORS)), "none", str(r.choice(F.CARRIERS)))
                sp = s.spans[0]
                cov = [
                    max(0, min(b, sp.end) - max(a, sp.start)) / (sp.end - sp.start)
                    for a, b, _, _ in span_fn(s.text)
                ]
                hits += int(any((c >= 0.5 for c in cov)))
                extra += sum((c == 0 for c in cov))
                n += 1
        out[str(d)] = {"span_recall": hits / n, "ci": F.wilson(hits, n), "spans_fuera_por_texto": extra / n}
    return out


def load_plugin(spec: str) -> Dict[str, Detector]:
    path, _, func = spec.partition(":")
    inst = getattr(
        load_module(path, "pii_plugin_" + re.sub("\\W", "_", os.path.basename(path))), func or "build"
    )()
    if not isinstance(inst, dict) or not all((callable(v) for v in inst.values())):
        raise TypeError(f"{spec}: debe devolver {{nombre: detector}}")
    return inst


class _FakeTensor:

    def __init__(self, a):
        self.a = np.asarray(a, np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __getitem__(self, i):
        return _FakeTensor(self.a[i])


def selftest() -> int:
    tests = []
    ref = spans_regex()
    texts = [s.text for s in F.generate(150, seed=31)]
    want = F.regex_detector(texts)

    class HFAgg:

        def __call__(self, t):
            return [
                {"entity_group": l, "score": s, "start": a, "end": b, "word": t[a:b]} for a, b, l, s in ref(t)
            ]

    class HFTok:

        def __call__(self, t):
            return [
                {"entity": "B-" + l, "score": s, "start": a, "end": b, "index": 1} for a, b, l, s in ref(t)
            ]

    class Ent:

        def __init__(self, a, b, l):
            self.start_char, self.end_char, self.label_ = (a, b, l)

    class Spacy:

        def __call__(self, t):
            return type("Doc", (), {"ents": [Ent(a, b, l) for a, b, l, _ in ref(t)]})()

    class Gliner:

        def predict_entities(self, t, labels, threshold=0.5):
            return [{"start": a, "end": b, "label": l, "score": s, "text": t[a:b]} for a, b, l, s in ref(t)]

    class PRes:

        def __init__(self, a, b, l, s):
            self.start, self.end, self.entity_type, self.score = (a, b, l, s)

    class Presidio:

        def analyze(self, text, language, entities=None):
            return [PRes(a, b, l, s) for a, b, l, s in ref(text)]

    fns = {
        "HF agregado": spans_hf(HFAgg()),
        "HF por token": spans_hf(HFTok()),
        "spaCy": spans_spacy(Spacy()),
        "GLiNER": spans_gliner(Gliner(), ["person"]),
        "Presidio": spans_presidio(Presidio()),
    }
    ok = all((np.array_equal(from_spans(f)(texts), want) for f in fns.values()))
    ok &= spans_hf(HFTok())("Tel 333-444-5555")[0][2] == "PHONE"
    tests.append((f"NER: {', '.join(fns)} -> mismos scores que la referencia (150 textos)", ok))
    d_no_person = from_spans(ref, reject=["PERSON"])(["Nombre: Mariela Quintanar"])
    tests.append(
        (
            "NER: accept/reject filtra etiquetas",
            d_no_person[0] == 0.0 and from_spans(ref)(["Nombre: Mariela Quintanar"])[0] == 1.0,
        )
    )
    top1 = lambda ts: [{"label": "LABEL_1" if w else "LABEL_0", "score": 0.9} for w in F.regex_detector(ts)]
    full = lambda ts: [
        [{"label": "LABEL_0", "score": 0.1 if w else 0.9}, {"label": "LABEL_1", "score": 0.9 if w else 0.1}]
        for w in F.regex_detector(ts)
    ]
    a_, b_ = (from_classifier(top1)(texts), from_classifier(full)(texts))
    tests.append(
        (
            "clasificador: top-1 y lista completa coinciden",
            np.allclose(a_, b_) and np.array_equal(a_ >= 0.5, want >= 0.5),
        )
    )
    table = {
        "SI": 1.0,
        "Sí, contiene un correo.": 1.0,
        "**No**": 0.0,
        "no.": 0.0,
        "yes": 1.0,
        "0.8": 0.8,
        "1": 1.0,
        "Depende del contexto": None,
        "": None,
        "no sabría decir": None,
        "No estoy seguro.": None,
        "No contiene datos personales": 0.0,
        "not sure": None,
    }
    ok = all((parse_verdict(k) == v for k, v in table.items()))
    gen = lambda p: (
        "SI" if F.regex_detector([p.split("Texto:\n", 1)[1].rsplit("\n\nRespuesta:", 1)[0]])[0] else "No"
    )
    ok &= np.array_equal(from_llm(gen)(texts), want)
    bad = from_llm(lambda p: "no sabría decir")
    ok &= np.allclose(bad(texts[:7]), 0.5) and bad.unparsed == 7
    tests.append(("LM generativo: lee SI/NO/número, reproduce la referencia y cuenta 7/7 ilegibles", ok))
    lp = lambda p, c: (
        0.0
        if (c.strip() == "SI")
        == bool(F.regex_detector([p.split("Texto:\n", 1)[1].rsplit("\n\nRespuesta:", 1)[0]])[0])
        else -6.0
    )
    tests.append(
        (
            "LM por logprobs: softmax SI/NO reproduce la referencia",
            np.array_equal(from_logprobs(lp)(texts) >= 0.5, want >= 0.5),
        )
    )
    hs = B.HashStates(dim=48, buckets=1 << 12)
    tok = lambda t, **kw: {"input_ids": t}

    class HFModel:

        def __call__(self, input_ids, output_hidden_states=False):
            H = hs(input_ids) if input_ids else np.zeros((0, 48), np.float32)
            return type("Out", (), {"hidden_states": (_FakeTensor(H[None] * 0), _FakeTensor(H[None]))})()

    sf = states_hf(HFModel(), tok, layer=-1)
    det = probe(sf, n_leak=250, n_ctx=45, seed=3, epochs=2)
    c = conformance(det)
    tests.append(
        (
            f"estados HF (doble) -> sonda: forma {sf('IP 10.0.0.1').shape}, contrato {('cumplido' if c['ok'] else [n for n, ok, _ in c['checks'] if not ok])}",
            sf("IP 10.0.0.1").shape == (11, 48) and c["ok"],
        )
    )
    seen = []
    spy = lambda ms: (seen.extend(ms), np.ones(len(ms)))[1]
    w = with_role(ref, spy)(["IP of the customer: 187.20.3.44", "nothing to see here"])
    tests.append(
        (
            "envoltura de papel: el juez recibe [X] y nunca la cadena; sin span -> 0",
            w[0] == 1.0
            and w[1] == 0.0
            and all((PLACEHOLDER in m for m in seen))
            and (not any(("187.20.3.44" in m for m in seen))),
        )
    )
    good = conformance(F.regex_detector)["ok"]
    broken = {
        "longitud": lambda ts: np.zeros(len(ts) + 1),
        "rango": lambda ts: np.full(len(ts), 1.7),
        "lote": lambda ts: np.full(len(ts), min(1.0, len(ts) / 12)),
        "excepción": lambda ts: np.array([1 / len(t) for t in ts]),
    }
    rej = {k: not conformance(v)["ok"] for k, v in broken.items()}
    tests.append(
        (
            f"compuerta: referencia pasa, rechaza {[k for k, v in rej.items() if v]}",
            good and all(rej.values()),
        )
    )
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import leak_forge as F\ndef build():\n    return {'mi-regex': F.regex_detector}\n")
    pl = load_plugin(f.name + ":build")
    os.unlink(f.name)
    tests.append(
        (
            "plugin: archivo.py:build -> {nombre: detector}",
            list(pl) == ["mi-regex"] and callable(pl["mi-regex"]),
        )
    )
    comp = with_role(ref, train_role_ngram(seed=5, n_leak=600, n_ctx=150, lang="en"))
    seen_acc = G.context_eval(comp, 60, 9, "cong_seen", "en")["pair_acc"]
    trap_acc = G.context_eval(comp, 60, 9, "incong", "en")["pair_acc"]
    tests.append(
        (
            f"regex+papel(n-gramas): par en visto {seen_acc:.2f} (regex solo: 0.50), en INCONGRUENTE {trap_acc:.2f}",
            seen_acc > 0.9 and trap_acc < 0.3,
        )
    )
    se = span_eval(ref, 10, 3)
    tests.append(
        (
            f"localización: recall de span regex d=1 {se['1.0']['span_recall']:.2f}, d=0.2 {se['0.2']['span_recall']:.2f}",
            se["1.0"]["span_recall"] > se["0.2"]["span_recall"],
        )
    )
    print("detector_adapters selftest")
    for name, ok in tests:
        print(f"  [{('ok' if ok else 'FAIL')}] {name}")
    print(
        "Honestidad: [1,3,4,5,6] usan DOBLES con el formato de salida de cada librería, no la librería real."
    )
    return 0 if all((ok for _, ok in tests)) else 1


def ach_wiring(a) -> int:
    import torch

    torch.manual_seed(0)
    if a.ckpt:
        model, encode = load_ach(a.ach, a.ckpt, a.tokenizer)
        if encode is None:
            raise SystemExit("[ach-wiring] con --ckpt hace falta --tokenizer")
        tag = f"checkpoint {os.path.basename(a.ckpt)}"
    else:
        m = load_module(a.ach)
        args = load_module(a.small_args, "ach_test_args").small_args(m, True)
        model = m.Belen(args).float().eval()
        V = args.vocab_size
        encode = lambda t: [b % V for b in t.encode("utf-8")]
        tag = f"modelo chico aleatorio (dim {args.model_dim}, {args.num_layers} capas, vocab {V})"
        import tempfile

        d = tempfile.mkdtemp()
        path, _ = m.save_raw_checkpoint(model, args, 0, d)
        model2, _ = load_ach(a.ach, str(path))
        same = all(
            (torch.equal(p, q) for p, q in zip(model.state_dict().values(), model2.state_dict().values()))
        )
        print(
            f"  [{('ok' if same else 'FAIL')}] load_ach: ida y vuelta de save_raw_checkpoint, state_dict idéntico"
        )
        if not same:
            return 1
    print(f"[ach-wiring] {tag}")
    ok_all = True
    for which in ("dendritic", "percept", "both"):
        sf = states_ach(model, encode, which=which)
        H = sf("ID del cliente: ABCD123456-H42")
        fin = bool(np.isfinite(H).all())
        det = probe(sf, n_leak=a.n_leak, n_ctx=a.n_ctx, seed=1, epochs=2) if fin else None
        c = (
            conformance(det)
            if det
            else {"ok": False, "checks": [("estados finitos", False, "NaN/Inf en los estados")]}
        )
        ok_all &= c["ok"]
        print(
            f"  [{('ok' if c['ok'] else 'FAIL')}] estados '{which}': forma {H.shape}, finitos {fin}, contrato "
            + ("cumplido" if c["ok"] else str([(n, d) for n, ok, d in c["checks"] if not ok]))
        )
        for n, ok, d in c["checks"]:
            if not ok and c["ok"]:
                print(f"         aviso: {n}: {d}")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Adaptadores BELENC-PII")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    w = sub.add_parser("ach-wiring")
    w.add_argument("--ach", required=True, help="ruta al modulo del modelo")
    w.add_argument("--small-args", help="ruta al modulo con small_args (modelo chico aleatorio)")
    w.add_argument("--ckpt", help="checkpoint crudo de save_raw_checkpoint")
    w.add_argument("--tokenizer", help="modelo de sentencepiece")
    w.add_argument("--n-leak", type=int, default=200)
    w.add_argument("--n-ctx", type=int, default=36)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "ach-wiring":
        if not a.ckpt and (not a.small_args):
            raise SystemExit("[ach-wiring] da --ckpt y --tokenizer, o --small-args para el modelo chico")
        return ach_wiring(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
