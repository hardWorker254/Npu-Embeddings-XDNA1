"""Why is `test` 5x worse than other single words on bge-large?

THE HYPOTHESIS: cancellation. bf16/bfp16 rounding lands on the pooled vector as
a roughly constant ABSOLUTE perturbation, while `1-cos` is RELATIVE. If a text's
pre-normalisation CLS vector is unusually short, the same absolute noise buys a
much larger angle -- so the error would track 1/||CLS||^2 and nothing else.

The first attempt at this probe was VOID: `SentenceTransformer.encode(
normalize_embeddings=False)` still returned unit vectors, because bge-large's ST
config carries a Normalize MODULE in the pipeline and the flag only governs the
final step. This uses AutoModel directly and takes last_hidden_state[:, 0], which
is the CLS vector before any pooling module touches it.

Run in .venv-ref.
"""
import math
import subprocess
import tempfile
from array import array
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path.cwd()
MODEL = REPO / "models" / "bge-large-en-v1.5"
DIM, SEQ = 1024, 64

WORDS = ["test", "rain", "mechanic", "hello", "banana", "check", "example",
         "sample", "river", "chair", "telescope", "antibiotics",
         "the mechanic replaced the front brake pads and topped up the oil"]

# --- our runtime ----------------------------------------------------------
exe = REPO / "runtime" / "build" / "npuembeddings.exe"
with tempfile.TemporaryDirectory(prefix="why_") as td:
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

# --- fp32 reference, CLS BEFORE normalisation ------------------------------
tok = AutoTokenizer.from_pretrained(str(MODEL))
mdl = AutoModel.from_pretrained(str(MODEL)).eval()
enc = tok(WORDS, padding=True, truncation=True, max_length=SEQ,
          return_tensors="pt")
with torch.no_grad():
    cls = mdl(**enc).last_hidden_state[:, 0].double().numpy()   # raw CLS

norms = np.linalg.norm(cls, axis=1)
ref = cls / norms[:, None]
got = ours / np.linalg.norm(ours, axis=1)[:, None]
one_m_cos = 1.0 - (got * ref).sum(axis=1)
ntok = enc["attention_mask"].sum(axis=1).numpy()

print("%-16s %5s %10s %12s %14s"
      % ("text", "tok", "||CLS||", "1-cos", "1-cos*||CLS||^2"))
print("-" * 62)
order = np.argsort(-one_m_cos)
for i in order:
    print("%-16s %5d %10.3f %12.3e %14.3e"
          % (WORDS[i][:15], ntok[i], norms[i], one_m_cos[i],
             one_m_cos[i] * norms[i] ** 2))

print("\nIf cancellation is the mechanism the LAST column is flat.")
sc = one_m_cos * norms ** 2
print("  1-cos            spread %.2fx" % (one_m_cos.max() / one_m_cos.min()))
print("  1-cos*||CLS||^2  spread %.2fx" % (sc.max() / sc.min()))
r = np.corrcoef(np.log(norms), np.log(one_m_cos))[0, 1]
print("  corr(log ||CLS||, log 1-cos) = %+.3f  (cancellation predicts ~-1)" % r)
