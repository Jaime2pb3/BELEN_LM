"""  
|---|---|
| `image_forge.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | See it. |
-----
"""


from __future__ import annotations
import argparse
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leak_forge as F

VERSION = "image-forge-0.1"
METHODS = ("bar", "erase", "pixelate", "blur")
ZERO_INFO = ("bar", "erase")
STRENGTH = {"bar": (0,), "erase": (0,), "pixelate": (3, 6, 10), "blur": (1.5, 3.0, 5.0)}
IMG_DOSES = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
THEMES = {
    "dark": ((49, 51, 56), (219, 222, 225), (0, 0, 0)),
    "light": ((255, 255, 255), (32, 33, 36), (0, 0, 0)),
}
W, MARGIN, FONT_SIZE, LINE_H = (760, 14, 17, 26)


@dataclass
class ImgMeta:
    pii_type: str
    original: str
    dose: float
    revealed: int
    total: int
    method: str
    strength: float
    operator: str
    carrier: str
    theme: str
    jpeg_q: Optional[int]
    span_boxes: List[Tuple[int, int, int, int]]
    masked_boxes: List[Tuple[int, int, int, int]]

    @property
    def zero_info_blank(self) -> bool:
        return self.revealed == 0 and self.method in ZERO_INFO


def _font(path_i: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATHS[path_i % len(FONT_PATHS)], FONT_SIZE)


def render(
    text: str, span: Tuple[int, int], theme: str, font
) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    lines = text.replace(F.ZW, "").split("\n")
    H = MARGIN * 2 + LINE_H * len(lines)
    bg, fg, _ = THEMES[theme]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    Wd = max(W, int(np.ceil(max((probe.textlength(L, font=font) for L in lines)))) + 2 * MARGIN)
    img = Image.new("RGB", (Wd, H), bg)
    dr = ImageDraw.Draw(img)
    boxes, pos = ([], 0)
    asc, desc = font.getmetrics()
    for li, L in enumerate(lines):
        y = MARGIN + li * LINE_H
        dr.text((MARGIN, y), L, font=font, fill=fg)
        for c in range(len(L)):
            g = pos + c
            if span[0] <= g < span[1]:
                x0 = MARGIN + dr.textlength(L[:c], font=font)
                x1 = MARGIN + dr.textlength(L[: c + 1], font=font)
                boxes.append((int(np.floor(x0)), y - 1, int(np.ceil(x1)), y + asc + desc + 1))
        pos += len(L) + 1
    return (img, boxes)


def _runs(masked: List[int]) -> List[List[int]]:
    out, cur = ([], [])
    for i in masked:
        if cur and i != cur[-1] + 1:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def treat(img: Image.Image, box: Tuple[int, int, int, int], method: str, s: float, theme: str):
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    bg, _, bar = THEMES[theme]
    if method == "bar":
        ImageDraw.Draw(img).rectangle([x0, y0, x1 - 1, y1 - 1], fill=bar)
    elif method == "erase":
        ImageDraw.Draw(img).rectangle([x0, y0, x1 - 1, y1 - 1], fill=bg)
    elif method == "pixelate":
        crop = img.crop(box)
        b = max(1, int(s))
        small = crop.resize((max(1, (x1 - x0) // b), max(1, (y1 - y0) // b)), Image.BILINEAR)
        img.paste(small.resize(crop.size, Image.NEAREST), box)
    elif method == "blur":
        img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(s)), box)
    else:
        raise ValueError(method)


def make_image(
    r,
    pii_type: str,
    dose: float,
    method: str,
    strength: float,
    op: Optional[str] = None,
    carrier: Optional[str] = None,
    jpeg_q: Optional[int] = None,
    lang: str = "es",
) -> Tuple[Image.Image, ImgMeta]:
    op = op or str(r.choice(F.OPERATORS))
    carrier = carrier or str(r.choice(F.CARRIERS))
    theme = str(r.choice(list(THEMES)))
    font = _font(int(r.integers(0, 2)))
    s = F.make_sample(r, pii_type, 1.0, "suffix", "none", carrier, lang=lang)
    sp = s.spans[0]
    img, boxes = render(s.text, (sp.start, sp.end), theme, font)
    v = sp.original
    keep = set(F.reveal_set(v, dose, op, pii_type, r))
    ident = F.ident_positions(v)
    masked = [i for i in ident if i not in keep]
    mboxes = []
    for run in _runs(masked):
        bx = (
            min((boxes[i][0] for i in run)),
            min((boxes[i][1] for i in run)),
            max((boxes[i][2] for i in run)),
            max((boxes[i][3] for i in run)),
        )
        treat(img, bx, method, strength, theme)
        mboxes.append(bx)
    if jpeg_q:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(jpeg_q))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    meta = ImgMeta(
        pii_type,
        v,
        dose,
        len(keep),
        len(ident),
        method,
        float(strength),
        op,
        carrier,
        theme,
        jpeg_q,
        boxes,
        mboxes,
    )
    return (img, meta)


ImgDetector = Callable[[List[Image.Image]], np.ndarray]


def evaluate(
    detector: ImgDetector,
    reps: int,
    seed: int,
    types: Sequence[str] = F.PII_TYPES,
    doses: Sequence[float] = IMG_DOSES,
    thr: float = 0.5,
    lang: str = "es",
) -> Dict:
    cells = []
    for t in types:
        for m in METHODS:
            for s in STRENGTH[m]:
                rates, ks = ([], [])
                for di, d in enumerate(doses):
                    r = np.random.default_rng([seed, F.PII_TYPES.index(t), METHODS.index(m), int(s * 10), di])
                    imgs = [make_image(r, t, d, m, s, lang=lang)[0] for _ in range(reps)]
                    hit = np.asarray(detector(imgs), float) >= thr
                    ks.append(int(hit.sum()))
                    rates.append(float(hit.mean()))
                c = {
                    "pii_type": t,
                    "method": m,
                    "strength": s,
                    "rates": dict(zip(map(str, doses), rates)),
                    "lod95": F.lod_from_rates(doses, rates),
                }
                i0 = list(doses).index(0.0)
                if m in ZERO_INFO:
                    c["fpr_blank"], c["fpr_blank_ci"] = (rates[i0], F.wilson(ks[i0], reps))
                else:
                    c["residual_at_d0"] = rates[i0]
                cells.append(c)
    return {"doses": list(doses), "reps": reps, "cells": cells, "version": VERSION}


def report(res: Dict, name: str) -> str:
    doses = res["doses"]
    L = [
        f"## {name}",
        "",
        f"réplicas por dosis: {res['reps']} · d=0 con bar/erase = FPR · con pixelate/blur = residual (no es blanco)",
        "",
        "| tipo | método | fuerza | d=0 | " + " | ".join((f"d={d:g}" for d in doses if d > 0)) + " | LOD95 |",
        "|" + "---|" * (5 + len([d for d in doses if d > 0])),
    ]
    for c in res["cells"]:
        L.append(
            f"| {c['pii_type']} | {c['method']} | {c['strength']:g} | {c['rates']['0.0']:.2f} | "
            + " | ".join((f"{c['rates'][str(d)]:.2f}" for d in doses if d > 0))
            + f" | **{('—' if c['lod95'] is None else c['lod95'])}** |"
        )
    return "\n".join(L)


_OCR_CACHE: Dict[int, str] = {}


def ocr_texts(imgs: List[Image.Image]) -> List[str]:
    import pytesseract

    out = []
    for im in imgs:
        k = hash((im.size, im.tobytes()))
        if k not in _OCR_CACHE:
            _OCR_CACHE[k] = pytesseract.image_to_string(im, lang="eng", config="--psm 6")
        out.append(_OCR_CACHE[k])
    return out


def ocr_then(text_detector: Callable[[List[str]], np.ndarray]) -> ImgDetector:
    return lambda imgs: np.asarray(text_detector(ocr_texts(imgs)), float)


def selftest() -> int:
    tests = []
    r = np.random.default_rng(4)
    ok = True
    for _ in range(40):
        img, m = make_image(r, str(r.choice(F.PII_TYPES)), 1.0, "bar", 0)
        ok &= len(m.span_boxes) == len(m.original)
        ok &= all((0 <= b[0] <= b[2] <= img.width and 0 <= b[1] < b[3] <= img.height for b in m.span_boxes))
    tests.append(("cajas de glifo: una por carácter, dentro de la imagen (40 capturas)", ok))
    ok2 = True
    for t in F.PII_TYPES:
        for d in IMG_DOSES:
            _, m = make_image(r, t, d, "bar", 0)
            ok2 &= m.revealed == int(round(d * m.total))
    tests.append(("dosis exacta: revelados = round(d·n) en 6 tipos × 7 dosis", ok2))
    ok = True
    for _ in range(30):
        img, m = make_image(r, str(r.choice(F.PII_TYPES)), 0.0, "bar", 0)
        a = np.asarray(img)
        for x0, y0, x1, y1 in m.masked_boxes:
            ok &= a[y0:y1, x0:x1].reshape(-1, 3).std(0).max() == 0
    tests.append(("blanco bar d=0: toda región tratada es uniforme (30 capturas)", ok))
    img, m = make_image(np.random.default_rng(1), "EMAIL", 0.0, "pixelate", 3)
    a = np.asarray(img)
    sd = max((a[y0:y1, x0:x1].std() for x0, y0, x1, y1 in m.masked_boxes))
    tests.append((f"pixelate d=0 conserva señal: sd región {sd:.1f} > 0 (no es blanco)", sd > 0))

    def png(seed):
        im, _ = make_image(np.random.default_rng(seed), "PHONE", 0.35, "blur", 3.0, jpeg_q=70)
        b = io.BytesIO()
        im.save(b, "PNG")
        return b.getvalue()

    tests.append(("determinista: mismos bytes PNG con la misma semilla", png(5) == png(5)))
    try:
        det = ocr_then(F.regex_detector)
        rr = np.random.default_rng(12)
        clean = det([make_image(rr, "EMAIL", 1.0, "bar", 0, carrier="form")[0] for _ in range(10)]).mean()
        blank = det([make_image(rr, "IP", 0.0, "bar", 0, carrier="form")[0] for _ in range(10)]).mean()
        tests.append(
            (
                f"OCR+regex: IP tapada d=0 {blank:.2f}, correo limpio d=1 {clean:.2f}",
                clean >= 0.7 and blank <= 0.3,
            )
        )
    except Exception as e:
        print(f"  [skip] OCR no disponible: {e}")
    print(f"image_forge v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{('ok' if ok else 'FAIL')}] {name}")
    print("Honestidad: valida el GENERADOR. No dice nada de ningún modelo de visión todavía.")
    return 0 if all((ok for _, ok in tests)) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Estándar de dosis para PII en capturas")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("sample")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outdir", default="forge_img_peek")
    b = sub.add_parser("baselines")
    b.add_argument("--reps", type=int, default=20)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--types", nargs="+", default=list(F.PII_TYPES))
    b.add_argument("--out", default="image_baselines.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "sample":
        os.makedirs(a.outdir, exist_ok=True)
        r = np.random.default_rng(a.seed)
        with open(os.path.join(a.outdir, "meta.jsonl"), "w") as f:
            for i in range(a.n):
                m_ = str(r.choice(METHODS))
                img, meta = make_image(
                    r,
                    str(r.choice(F.PII_TYPES)),
                    float(r.choice(IMG_DOSES)),
                    m_,
                    float(r.choice(STRENGTH[m_])),
                    jpeg_q=int(r.choice([0, 60, 85])) or None,
                )
                img.save(os.path.join(a.outdir, f"{i:04d}.png"))
                f.write(json.dumps({"file": f"{i:04d}.png", **asdict(meta)}, ensure_ascii=False) + "\n")
        print(f"[image-forge] {a.n} capturas -> {a.outdir}")
        return 0
    if a.cmd == "baselines":
        tr = F.generate(6000, seed=a.seed + 1000, blank_frac=0.5)
        nb = F.NgramLogReg().fit([s.text for s in tr], [s.label for s in tr])
        out = {}
        for name, det in (("OCR + regex", ocr_then(F.regex_detector)), ("OCR + n-gramas", ocr_then(nb))):
            res = evaluate(det, a.reps, a.seed, types=a.types)
            out[name] = res
            print(report(res, name) + "\n", flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"[image-forge] -> {a.out}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
