"""  
|---|---|
| `ladder.py` | Luis Jaime Ledesma. |
| `TwoQuarksResearch` | Drive it. |
"""
-----


from __future__ import annotations
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from typing import Dict, List, Optional
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("env", "selftest", "data", "wiring", "preflight", "train", "prereg", "ficha", "huella", "pack")
REQUIRES = {
    "env": [],
    "selftest": ["env"],
    "data": ["selftest"],
    "wiring": ["data"],
    "preflight": ["wiring"],
    "train": ["preflight"],
    "prereg": ["train"],
    "ficha": ["prereg"],
    "huella": ["train"],
    "pack": ["ficha"],
}
MODULES = (
    "leak_forge",
    "context_grammar",
    "reference_card",
    "detector_adapters",
    "model_probe",
    "text_fingerprint",
)
HF = "https://huggingface.co/datasets/willdepueoai/parameter-golf/resolve/main/datasets"
TOK_REMOTE = f"{HF}/tokenizers/fineweb_1024_bpe.model"
SHARD_REMOTE = f"{HF}/datasets/fineweb10B_sp1024/{{name}}"
MAGIC, HEADER_INTS = (20240520, 256)
TRAIN_DEFAULTS = {"TTT_ENABLED": "0", "VAL_LOSS_EVERY": "0"}
PREFLIGHT_ENV = {
    "ITERATIONS": "2",
    "WARMUP_STEPS": "0",
    "VAL_LOSS_EVERY": "0",
    "TRAIN_BATCH_TOKENS": "8192",
    "TRAIN_SEQ_LEN": "256",
    "TTT_ENABLED": "0",
    "RUN_ID": "preflight",
}
TRAIN_DEFAULTS_100M = {"TTT_ENABLED": "0", "VAL_LOSS_EVERY": "500", "CKPT_EVERY": "250", "CKPT_DIR": "ckpt"}
PREFLIGHT_ENV_100M = {
    "ITERATIONS": "2",
    "WARMUP_STEPS": "0",
    "VAL_LOSS_EVERY": "0",
    "TTT_ENABLED": "0",
    "RUN_ID": "preflight",
    "CKPT_EVERY": "0",
    "SOMA_FREEZE_STEPS": "0",
    "ABLATE_TOKENS": "65536",
}


class Ladder:

    def __init__(self, work: str):
        self.work = os.path.abspath(work)
        os.makedirs(self.work, exist_ok=True)
        self.path = os.path.join(self.work, "ladder.json")
        self.state: Dict[str, dict] = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def green(self, stage: str) -> bool:
        return self.state.get(stage, {}).get("ok") is True

    def mark(self, stage: str, ok: bool, detail: str = "", seconds: float = 0.0):
        self.state[stage] = {
            "ok": bool(ok),
            "detail": detail,
            "seconds": round(seconds, 1),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=1)

    def invalidate_after(self, stage: str):
        for s in STAGES[STAGES.index(stage) + 1 :]:
            if s in self.state:
                self.state[s]["ok"] = None
                self.state[s]["detail"] = f"pendiente: se rehízo '{stage}'"

    def show(self):
        for s in STAGES:
            st = self.state.get(s)
            mark = (
                "  ·  " if st is None else " ok  " if st["ok"] else "FALLA" if st["ok"] is False else " ¿?  "
            )
            print(f"  [{mark}] {s:10s} {('' if st is None else st.get('detail', ''))}")


def sh(cmd: List[str], log_path: str, env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None) -> int:
    full = dict(os.environ, PYTHONUNBUFFERED="1", **env or {})
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=full, cwd=cwd, bufsize=1
        )
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return p.wait()


def download(url: str, dst: str, expect_min: int = 1) -> int:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:
            have = 0
        total = have + int(r.headers.get("Content-Length") or 0)
        with open(part, "ab" if have else "wb") as f:
            done, t0 = (have, time.time())
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - t0 > 5:
                    print(
                        f"    {os.path.basename(dst)}: {done / 1000000.0:.0f} / {total / 1000000.0:.0f} MB",
                        flush=True,
                    )
                    t0 = time.time()
    size = os.path.getsize(part)
    if size < expect_min or (total and size != total):
        raise IOError(f"descarga incompleta de {os.path.basename(dst)}: {size} de {total} bytes")
    os.replace(part, dst)
    return size


def shard_info(path: str) -> Dict:
    h = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if h.size != HEADER_INTS or int(h[0]) != MAGIC or int(h[1]) != 1:
        raise ValueError(f"cabecera inválida: {path}")
    n = int(h[2])
    expect = HEADER_INTS * 4 + n * 2
    if os.path.getsize(path) != expect:
        raise ValueError(f"{path}: declara {n} tokens ({expect} bytes) y mide {os.path.getsize(path)}")
    return {"tokens": n}


def truncate_shard(src: str, dst: str, n_tokens: int) -> int:
    h = np.fromfile(src, dtype="<i4", count=HEADER_INTS).copy()
    n = min(int(h[2]), n_tokens)
    toks = np.fromfile(src, dtype="<u2", count=n, offset=HEADER_INTS * 4)
    h[2] = n
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(h.astype("<i4").tobytes())
        f.write(toks.astype("<u2").tobytes())
    return n


class Run:

    def __init__(self, a):
        self.a, self.L = (a, Ladder(a.work))
        self.work = self.L.work
        self.script = os.path.abspath(a.script)
        self.data = os.path.join(self.work, "data")
        self.shards = os.path.join(self.data, "datasets", "fineweb10B_sp1024")
        self.tok = (
            os.path.abspath(a.tokenizer)
            if a.tokenizer
            else os.path.join(self.data, "tokenizers", "fineweb_1024_bpe.model")
        )
        self.user_ckpt = os.path.abspath(a.ckpt) if a.ckpt else None
        self.ckpt = self.user_ckpt or os.path.join(self.work, "run_raw.pt")
        self.ficha_dir = os.path.join(self.work, "ficha")
        self.py = sys.executable

    def log(self, stage: str) -> str:
        return os.path.join(self.work, f"{stage}.log")

    def dp(self, *args: str) -> List[str]:
        return [self.py, os.path.join(HERE, "model_probe.py"), *args]

    def env(self):
        import platform

        info = [f"python {platform.python_version()}", f"numpy {np.__version__}"]
        missing = []
        try:
            import torch

            info.append(f"torch {torch.__version__}")
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                info.append(f"GPU {p.name} {p.total_memory / 2 ** 30:.0f} GB")
            else:
                info.append(
                    "SIN GPU (alcanza hasta 'wiring'. preflight, train y una ficha real necesitan GPU)"
                )
        except ImportError:
            missing.append("torch")
        try:
            import sentencepiece
        except ImportError:
            missing.append("sentencepiece")
        free = shutil.disk_usage(self.work).free / 2**30
        info.append(f"disco libre {free:.0f} GB")
        if not os.path.exists(self.script):
            return (False, f"no encuentro el script de entrenamiento: {self.script}")
        if missing:
            return (False, f"faltan dependencias: pip install {' '.join(missing)}")
        if free < 2:
            return (False, f"disco insuficiente ({free:.1f} GB libres)")
        return (True, " · ".join(info))

    def selftest(self):
        files = [os.path.join(HERE, m + ".py") for m in MODULES] + [self.script, os.path.abspath(__file__)]
        for f in files:
            try:
                ast.parse(open(f, encoding="utf-8").read())
            except SyntaxError as e:
                return (False, f"sintaxis en {os.path.basename(f)}: {e}")
        n_ok = 0
        for m in MODULES:
            if sh([self.py, os.path.join(HERE, m + ".py"), "--selftest"], self.log("selftest")):
                return (False, f"{m}.py --selftest falló (ver selftest.log)")
            n_ok += 1
        if _self_checks():
            return (False, "ladder: verificación interna de fragmentos falló")
        return (True, f"sintaxis de {len(files)} archivos y selftest de {n_ok} módulos")

    def data_stage(self):
        got = []
        if not os.path.exists(self.tok):
            download(TOK_REMOTE, self.tok, 1000)
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=self.tok)
        got.append(f"tokenizer vocab {sp.get_piece_size()}")
        if self.user_ckpt:
            return (True, got[0] + " · sin fragmentos (se usa --ckpt)")
        names = ["fineweb_val_000000.bin"] + [
            f"fineweb_train_{i:06d}.bin" for i in range(self.a.train_shards)
        ]
        total = 0
        for n in names:
            dst = os.path.join(self.shards, n)
            if not os.path.exists(dst):
                print(f"  descargando {n}", flush=True)
                download(SHARD_REMOTE.format(name=n), dst, HEADER_INTS * 4)
            total += shard_info(dst)["tokens"]
        need = self.a.iterations * int(
            self.train_env().get("TRAIN_BATCH_TOKENS", 65536 if self.is_100m else 16384)
        )
        have = sum((shard_info(os.path.join(self.shards, n))["tokens"] for n in names[1:]))
        note = (
            ""
            if have >= need
            else f" · AVISO: {self.a.iterations} iteraciones consumen ~{need / 1000000.0:.0f}M tokens y hay {have / 1000000.0:.0f}M: los fragmentos se repetirán"
        )
        return (
            True,
            f"{got[0]} · {len(names)} fragmentos, {total / 1000000.0:.0f}M tokens, cabeceras verificadas{note}",
        )

    def wiring(self):
        rc = sh(
            self.dp("wiring", "--script", self.script, "--tokenizer", self.tok, "--small"), self.log("wiring")
        )
        if rc:
            return (False, "cableado en chico falló (ver wiring.log)")
        if self.user_ckpt:
            rc = sh(
                self.dp(
                    "wiring",
                    "--script",
                    self.script,
                    "--tokenizer",
                    self.tok,
                    "--ckpt",
                    self.user_ckpt,
                    "--keep",
                    os.path.join(self.work, "SMOKE_ckpt.md"),
                ),
                self.log("wiring"),
            )
            if rc:
                return (False, "cableado con tu checkpoint falló (ver wiring.log)")
            return (True, "script real en chico + tu checkpoint a tamaño real, con ficha de humo")
        return (True, "script real en chico, con ficha de humo")

    @property
    def is_100m(self) -> bool:
        name = os.path.basename(self.script).lower()
        return "100m" in name or name == "train_belen.py"

    def train_env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = dict(TRAIN_DEFAULTS_100M if self.is_100m else TRAIN_DEFAULTS)
        env.update(
            {
                "DATA_PATH": self.shards + os.sep,
                "TOKENIZER_PATH": self.tok,
                "ITERATIONS": str(self.a.iterations),
                "RUN_ID": "belenc_run",
            }
        )
        for kv in self.a.env or []:
            k, _, v = kv.partition("=")
            env[k] = v
        env.update(extra or {})
        return env

    def preflight(self):
        import torch

        if not torch.cuda.is_available():
            return (False, "preflight necesita GPU (el script exige CUDA)")
        tiny = os.path.join(self.work, "preflight_data")
        shutil.rmtree(tiny, ignore_errors=True)
        truncate_shard(
            os.path.join(self.shards, "fineweb_train_000000.bin"),
            os.path.join(tiny, "fineweb_train_000000.bin"),
            2000000,
        )
        truncate_shard(
            os.path.join(self.shards, "fineweb_val_000000.bin"),
            os.path.join(tiny, "fineweb_val_000000.bin"),
            300000,
        )
        ck = os.path.join(self.work, "preflight_raw.pt")
        if os.path.exists(ck):
            os.remove(ck)
        env = self.train_env(
            dict(PREFLIGHT_ENV_100M if self.is_100m else PREFLIGHT_ENV, DATA_PATH=tiny + os.sep)
        )
        rc = sh(
            self.dp("train", "--script", self.script, "--out-ckpt", ck),
            self.log("preflight"),
            env,
            cwd=self.work,
        )
        if rc or not os.path.exists(ck):
            return (
                False,
                "el script real no terminó 2 iteraciones o no guardó pesos (ver preflight.log). NO sigas a train.",
            )
        rc = sh(
            self.dp(
                "wiring",
                "--script",
                self.script,
                "--tokenizer",
                self.tok,
                "--ckpt",
                ck,
                "--keep",
                os.path.join(self.work, "SMOKE_preflight.md"),
            ),
            self.log("preflight"),
        )
        if rc:
            return (False, "los pesos del preflight no pasan el cableado a tamaño real (ver preflight.log)")
        return (
            True,
            f"2 iteraciones reales, pesos guardados ({os.path.getsize(ck) / 1000000.0:.0f} MB), ficha de humo con el modelo real",
        )

    def train(self):
        if self.user_ckpt:
            return (True, f"omitido: se usa {os.path.basename(self.user_ckpt)}")
        env = self.train_env()
        resume = os.path.join(self.work, env.get("CKPT_DIR", "ckpt"), f"{env['RUN_ID']}_resume.pt")
        if self.is_100m and os.path.exists(resume) and ("RESUME" not in env):
            env["RESUME"] = resume
            print(
                f"  hay un checkpoint reanudable: continúo desde {os.path.relpath(resume, self.work)}",
                flush=True,
            )
        print("  entorno de entrenamiento: " + " ".join((f"{k}={v}" for k, v in env.items())), flush=True)
        if os.path.exists(self.ckpt):
            os.replace(self.ckpt, self.ckpt + f".previo-{int(time.time())}")
        rc = sh(
            self.dp("train", "--script", self.script, "--out-ckpt", self.ckpt),
            self.log("train"),
            env,
            cwd=self.work,
        )
        if rc or not os.path.exists(self.ckpt):
            return (False, "el entrenamiento no terminó o no guardó pesos (ver train.log)")
        return (
            True,
            f"pesos crudos en {os.path.relpath(self.ckpt, self.work)} ({os.path.getsize(self.ckpt) / 1000000.0:.0f} MB) · ITERATIONS={env['ITERATIONS']}",
        )

    def prereg(self):
        if os.path.exists(os.path.join(self.ficha_dir, "prereg.json")):
            return (True, "ya existe (el prerregistro se escribe una sola vez)")
        rc = sh(
            self.dp(
                "prereg",
                "--script",
                self.script,
                "--ckpt",
                self.ckpt,
                "--tokenizer",
                self.tok,
                "--out-dir",
                self.ficha_dir,
                "--controls",
                str(self.a.controls),
                "--langs",
                *self.a.langs,
            ),
            self.log("prereg"),
        )
        return (not rc, f"escrito {os.path.relpath(os.path.join(self.ficha_dir, 'PREREG.md'), self.work)}")

    def ficha(self):
        cmd = self.dp(
            "ficha",
            "--script",
            self.script,
            "--ckpt",
            self.ckpt,
            "--tokenizer",
            self.tok,
            "--out-dir",
            self.ficha_dir,
        )
        if self.a.allow_drift:
            cmd.append("--allow-drift")
        rc = sh(cmd, self.log("ficha"))
        return (not rc, "MODEL_PROBE.md escrito" if not rc else "la ficha no terminó (ver ficha.log)")

    def huella(self):
        val = os.path.join(self.shards, "fineweb_val_000000.bin")
        if not os.path.exists(val):
            print(
                "  descargando fineweb_val_000000.bin (124 MB) para la huella de orden sobre texto real",
                flush=True,
            )
            download(SHARD_REMOTE.format(name="fineweb_val_000000.bin"), val, HEADER_INTS * 4)
        shard_info(val)
        out = os.path.join(self.work, "huella")
        rc = sh(
            [
                self.py,
                os.path.join(HERE, "text_fingerprint.py"),
                "run",
                "--script",
                self.script,
                "--ckpt",
                self.ckpt,
                "--tokenizer",
                self.tok,
                "--out-dir",
                out,
                "--val-shard",
                val,
            ],
            self.log("huella"),
        )
        return (
            not rc,
            "HUELLA.md escrito (exploratorio)" if not rc else "la huella no terminó (ver huella.log)",
        )

    def pack(self):
        out = os.path.join(self.work, "belenc_pii_resultados.zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for d in (self.ficha_dir, os.path.join(self.work, "huella")):
                for root, _, files in os.walk(d):
                    for f in files:
                        p = os.path.join(root, f)
                        z.write(p, os.path.relpath(p, self.work))
            for f in os.listdir(self.work):
                if f.endswith((".log", ".md")) or f == "ladder.json":
                    z.write(os.path.join(self.work, f), f)
            if self.a.with_ckpt and os.path.exists(self.ckpt):
                z.write(self.ckpt, os.path.basename(self.ckpt))
        return (True, f"{os.path.relpath(out, self.work)} ({os.path.getsize(out) / 1000000.0:.1f} MB)")

    def run(self, stage: str) -> bool:
        need = [s for s in REQUIRES[stage] if not self.L.green(s)]
        if self.user_ckpt:
            need = [s for s in need if s not in ("preflight", "train")]
            if stage in ("prereg", "huella") and (not self.L.green("wiring")):
                need.append("wiring")
        if need:
            print(
                f"[escalera] '{stage}' no corre: falta en verde -> {need}. Corre primero: python ladder.py {need[0]}"
            )
            return False
        if stage in ("preflight", "train") and self.user_ckpt:
            self.L.mark(stage, True, f"omitido: se usa {os.path.basename(self.user_ckpt)}")
            print(f"[escalera] {stage}: omitido (se usa --ckpt)")
            return True
        print(f"\n[escalera] ── {STAGES.index(stage)} · {stage} " + "─" * 40, flush=True)
        t0 = time.time()
        fn = {
            "env": self.env,
            "selftest": self.selftest,
            "data": self.data_stage,
            "wiring": self.wiring,
            "preflight": self.preflight,
            "train": self.train,
            "prereg": self.prereg,
            "ficha": self.ficha,
            "huella": self.huella,
            "pack": self.pack,
        }[stage]
        try:
            ok, detail = fn()
        except Exception as e:
            import traceback

            traceback.print_exc()
            ok, detail = (False, f"{type(e).__name__}: {e}")
        if stage not in ("pack", "env", "prereg", "huella"):
            self.L.invalidate_after(stage)
        self.L.mark(stage, ok, detail, time.time() - t0)
        print(
            f"[escalera] {stage}: {('VERDE' if ok else 'ROJO')} · {detail} · {time.time() - t0:.0f}s",
            flush=True,
        )
        if not ok:
            print(
                "[escalera] ALTO. No sigas a la siguiente etapa: corrige esto y vuelve a correr esta misma.",
                flush=True,
            )
        return ok


def _self_checks() -> int:
    import tempfile

    d = tempfile.mkdtemp()
    h = np.zeros(HEADER_INTS, "<i4")
    h[0], h[1], h[2] = (MAGIC, 1, 5000)
    toks = (np.arange(5000) % 1024).astype("<u2")
    src = os.path.join(d, "a.bin")
    with open(src, "wb") as f:
        f.write(h.tobytes())
        f.write(toks.tobytes())
    ok = shard_info(src)["tokens"] == 5000
    dst = os.path.join(d, "sub", "b.bin")
    ok &= truncate_shard(src, dst, 1200) == 1200 and shard_info(dst)["tokens"] == 1200
    ok &= np.array_equal(np.fromfile(dst, dtype="<u2", count=1200, offset=HEADER_INTS * 4), toks[:1200])
    with open(src, "ab") as f:
        f.write(b"\x00\x00")
    try:
        shard_info(src)
        ok = False
    except ValueError:
        pass
    shutil.rmtree(d, ignore_errors=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Escalera BELENC-PII sobre BELEN")
    ap.add_argument("stage", choices=STAGES + ("all", "status"))
    ap.add_argument(
        "--work",
        default="belenc_work",
        help="carpeta de trabajo (ponla en Drive para sobrevivir desconexiones)",
    )
    ap.add_argument(
        "--script",
        default=os.path.join(os.path.dirname(HERE), "train_belen.py"),
        help="por defecto train_belen.py en la raíz del repo",
    )
    ap.add_argument("--ckpt", help="pesos crudos ya entrenados: se salta preflight y train")
    ap.add_argument("--tokenizer", help="por defecto se descarga el de Parameter Golf")
    ap.add_argument(
        "--train-shards",
        type=int,
        default=5,
        help="fragmentos de 100M tokens (hay 195). 100M: 6500 iteraciones × 65536 ≈ 426M tokens = 5 fragmentos",
    )
    ap.add_argument("--iterations", type=int, default=6500)
    ap.add_argument("--env", action="append", help="variable para el entrenamiento, CLAVE=VALOR. Repetible")
    ap.add_argument("--langs", nargs="+", default=["en"], choices=("en", "es"))
    ap.add_argument(
        "--controls",
        type=int,
        default=3,
        help="controles aleatorios. 3: un positivo es provisional (p >= 0.25). 19: p <= 0.05. Cada uno cuesta lo que el entrenado",
    )
    ap.add_argument("--allow-drift", action="store_true")
    ap.add_argument("--with-ckpt", action="store_true", help="incluye los pesos en el zip de resultados")
    a = ap.parse_args()
    r = Run(a)
    if a.stage == "status":
        r.L.show()
        return 0
    if a.stage == "all":
        for s in STAGES[: STAGES.index("prereg") + 1]:
            if not r.run(s):
                return 1
        print(
            f"\n[escalera] Prerregistro escrito. ANTES de medir:\n  1. Abre {os.path.join(r.ficha_dir, 'PREREG.md')} y escribe tus predicciones en J1 y J2.\n  2. Corre:  python ladder.py ficha\n  3. Corre:  python ladder.py pack"
        )
        return 0
    return 0 if r.run(a.stage) else 1


if __name__ == "__main__":
    raise SystemExit(main())
