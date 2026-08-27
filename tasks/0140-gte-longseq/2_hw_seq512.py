"""gte on the seq-512 design set: exe (s512 container + seq512 artifacts)
vs the numpy oracle, on long inputs. Mirrors 0136's e2e check at seq 64.
Run in .venv-ref from the repo root."""
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "reference"))
import encoder_gte
from safetensors_io import load

tok = AutoTokenizer.from_pretrained(str(REPO / "models" / "gte-multilingual-base"))
raw, _ = load(REPO / "models" / "gte-multilingual-base" / "model.safetensors")
w = {}
for k, v in raw.items():
    kk = k[4:] if k.startswith("new.") else k
    w[kk] = v.astype(np.float32) if v.dtype == np.float16 else v
enc = encoder_gte.GteEncoder(w)

para = ("Retrieval systems live or die on the embedding model, and a "
        "long-context encoder must hold its geometry out to the last "
        "position it claims to support, not only near the origin. ")
no_para = ("Dokumentgjenfinning på tvers av språk krever at modellen "
           "holder geometrien også for norsk tekst, hele veien ut til "
           "siste posisjon. ")

def fit_text(unit, limit):
    """repeat `unit` while the tokenized length stays under `limit` --
    the runtime REFUSES oversize inputs (0110), so the harness must fit."""
    text = unit
    while True:
        cand = text + unit
        if len(tok(cand)["input_ids"]) > limit:
            return text
        text = cand

def make_texts(seq):
    lim = seq - 2
    return [
        fit_text(para, lim),
        fit_text(para, max(8, lim // 2)),
        "A single short sentence rides along as the short-input control.",
        fit_text(no_para, lim),
    ]

exe = REPO / "runtime" / "build" / "npuembed.exe"
for artifacts, seq in (("artifacts_nomic_bfp16_seq512", 512),
                       ("artifacts_nomic_bfp16_seq256", 256)):
    TEXTS = make_texts(seq)
    with tempfile.TemporaryDirectory(prefix="gte_ls_") as td:
        d = Path(td)
        (d / "in.txt").write_text("\n".join(t.replace("\n", " ") for t in TEXTS) + "\n",
                                  encoding="utf-8")
        p = subprocess.run([str(exe), str(REPO), "--model",
                            str(REPO / "models" / "gte-multilingual-base.s512.npue"),
                            "--artifacts", artifacts,
                            "--embed", str(d / "in.txt"), str(d / "out.f32"),
                            "--threads", "24"],
                           capture_output=True, text=True, encoding="utf-8")
        if p.returncode != 0:
            print(f"[{artifacts}] FAILED:\n{p.stdout[-1500:]}\n{p.stderr[-800:]}")
            continue
        status = [l.strip() for l in p.stdout.splitlines()
                  if "datapath" in l or "shape " in l]
        a = array("f")
        a.frombytes((d / "out.f32").read_bytes())
        got = np.asarray(a, dtype=np.float64).reshape(len(TEXTS), 768)
    print(f"\n[{artifacts}] " + " | ".join(status))
    for i, text in enumerate(TEXTS):
        e = tok(text, truncation=True, max_length=seq)
        ids = np.array(e["input_ids"])
        ref = enc.encode(ids)["cls_normalized"]
        g = got[i] / np.linalg.norm(got[i])
        print(f"  text {i} ({len(ids):3d} tok): 1-cos {1.0 - float(g @ ref):.3e}")
