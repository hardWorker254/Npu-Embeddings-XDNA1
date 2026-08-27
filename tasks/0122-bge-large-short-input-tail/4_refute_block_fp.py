"""Hypothesis 2: block-floating-point outliers.

The adopted datapath is `--emulate-bf16-mmul-with-bfp16`, and bfp16 shares ONE
exponent across a block of 8 elements (`to_v64bfp16ebs8`). So the precision every
element in a block gets is set by the LARGEST element in that block: a single
outlier channel costs the other seven mantissa bits. That is the classic block-FP
failure mode, and it is the same phenomenon SmoothQuant exists to fix for int8
(tasks/0078).

Cancellation was refuted first (why.py): ||CLS|| is 16.5-19.1 for every word and
`test` has the LARGEST norm, corr(log||CLS||, log 1-cos) = +0.111.

PREDICTION if this hypothesis holds: the per-word error tracks how outlier-heavy
that word's activations are, measured the way the hardware sees them -- within
aligned groups of 8 along the reduction axis.

Run in .venv-ref.
"""
import subprocess
import tempfile
from array import array
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path.cwd()
DIM, SEQ, BLK = 1024, 64, 8
WORDS = ["test", "rain", "mechanic", "hello", "banana", "check", "example",
         "sample", "river", "chair", "telescope", "antibiotics",
         "the mechanic replaced the front brake pads and topped up the oil"]

exe = REPO / "runtime" / "build" / "npuembeddings.exe"
with tempfile.TemporaryDirectory(prefix="why2_") as td:
    d = Path(td)
    (d / "in.txt").write_text("\n".join(WORDS) + "\n", encoding="utf-8")
    p = subprocess.run([str(exe), "embed", "bge-large-en-v1.5",
                        str(d / "in.txt"), str(d / "out.f32"),
                        "--root", str(REPO), "--threads", "8"],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stdout + p.stderr
    a = array("f")
    a.frombytes((d / "out.f32").read_bytes())
    ours = np.asarray(a, dtype=np.float64).reshape(len(WORDS), DIM)

MODEL = REPO / "models" / "bge-large-en-v1.5"
tok = AutoTokenizer.from_pretrained(str(MODEL))
mdl = AutoModel.from_pretrained(str(MODEL)).eval()
enc = tok(WORDS, padding=True, truncation=True, max_length=SEQ,
          return_tensors="pt")
with torch.no_grad():
    out = mdl(**enc, output_hidden_states=True)
hs = [h.double().numpy() for h in out.hidden_states]     # 25 x [B, T, 1024]
cls = out.last_hidden_state[:, 0].double().numpy()
ref = cls / np.linalg.norm(cls, axis=1, keepdims=True)
got = ours / np.linalg.norm(ours, axis=1, keepdims=True)
err = 1.0 - (got * ref).sum(axis=1)
mask = enc["attention_mask"].numpy().astype(bool)


def block_crest(x):
    """Mean over aligned blocks of 8 of (max|x| / rms|x|).

    1.0 means every element in the block is the same size (bfp16 loses nothing);
    sqrt(8) = 2.83 means one element carries the whole block (the other seven
    are pushed toward the exponent floor).
    """
    b = np.abs(x).reshape(-1, BLK)
    rms = np.sqrt((b ** 2).mean(axis=1)) + 1e-12
    return float((b.max(axis=1) / rms).mean())


print("%-16s %10s %12s %12s %12s"
      % ("text", "1-cos", "crest(all)", "crest(CLS)", "max|h|"))
print("-" * 66)
rows = []
for i, w in enumerate(WORDS):
    real = [h[i][mask[i]] for h in hs]                 # real tokens only
    c_all = float(np.mean([block_crest(r) for r in real]))
    c_cls = float(np.mean([block_crest(h[i, 0]) for h in hs]))
    mx = float(max(np.abs(r).max() for r in real))
    rows.append((w, err[i], c_all, c_cls, mx))

for w, e, c_all, c_cls, mx in sorted(rows, key=lambda r: -r[1]):
    print("%-16s %10.3e %12.4f %12.4f %12.1f" % (w[:15], e, c_all, c_cls, mx))

e = np.array([r[1] for r in rows])
for name, k in (("crest(all)", 2), ("crest(CLS)", 3), ("max|h|", 4)):
    v = np.array([r[k] for r in rows])
    print("  corr(log %-11s, log 1-cos) = %+.3f"
          % (name, np.corrcoef(np.log(v), np.log(e))[0, 1]))
