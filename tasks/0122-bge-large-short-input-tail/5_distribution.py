"""Hypothesis 3: `test` is not special, it is the tail.

Two mechanistic hypotheses were tested and REFUTED:
  1. cancellation      -- ||CLS|| is 16.5-19.1 for every word and `test` has the
                          LARGEST; corr(log||CLS||, log 1-cos) = +0.111.
  2. block-FP outliers -- the crest factor within aligned blocks of 8 is
                          1.87-1.91 for every word and max|h| is 17.6-18.0;
                          corr = -0.00 / -0.20 / -0.28, i.e. nothing.

Neither input statistic predicts the error. That is itself the finding: if the
bf16/bfp16 perturbation is quasi-random in direction, `1-cos` is the squared
angle it happens to make with the CLS direction, and it should be DISTRIBUTED
rather than caused. Then `test` is not broken -- it is a high draw.

The question becomes: what IS the distribution, and where does the project's own
2e-03 tolerance sit in it? Run bge-large (24 layers, the deepest) against
bge-base (12) as the control.

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
SEQ = 64
CASES = [("bge-large-en-v1.5", 1024), ("bge-base-en-v1.5", 768)]

# A varied corpus: single words (the regime where the spread showed), short
# phrases, and full sentences. Words are drawn from the model's OWN vocabulary,
# deterministically, so this is reproducible and not hand-picked.
tok0 = AutoTokenizer.from_pretrained(str(REPO / "models" / "bge-base-en-v1.5"))
vocab = [w for w in tok0.get_vocab()
         if w.isalpha() and len(w) >= 4 and w.islower()]
vocab.sort()
words = vocab[::max(1, len(vocab) // 180)][:180]

import json
corpus = json.loads((REPO / "tools" / "semantic_corpus.json")
                    .read_text(encoding="utf-8"))
sents = [s["text"] for s in corpus["sentences"]]
TEXTS = words + sents + [" ".join(words[i:i + 5]) for i in range(0, 40, 5)]
print("corpus: %d texts (%d single words, %d sentences, %d phrases)"
      % (len(TEXTS), len(words), len(sents), 8))

exe = REPO / "runtime" / "build" / "npuembeddings.exe"
for model, dim in CASES:
    with tempfile.TemporaryDirectory(prefix="dist_") as td:
        d = Path(td)
        (d / "in.txt").write_text("\n".join(TEXTS) + "\n", encoding="utf-8")
        p = subprocess.run([str(exe), "embed", model, str(d / "in.txt"),
                            str(d / "out.f32"), "--root", str(REPO),
                            "--threads", "24"],
                           capture_output=True, text=True, encoding="utf-8")
        assert p.returncode == 0, p.stdout + p.stderr
        a = array("f")
        a.frombytes((d / "out.f32").read_bytes())
        ours = np.asarray(a, dtype=np.float64).reshape(len(TEXTS), dim)

    m = REPO / "models" / model
    tk = AutoTokenizer.from_pretrained(str(m))
    md = AutoModel.from_pretrained(str(m)).eval()
    ref = []
    for i in range(0, len(TEXTS), 64):
        enc = tk(TEXTS[i:i + 64], padding=True, truncation=True,
                 max_length=SEQ, return_tensors="pt")
        with torch.no_grad():
            c = md(**enc).last_hidden_state[:, 0].double().numpy()
        ref.append(c / np.linalg.norm(c, axis=1, keepdims=True))
    ref = np.concatenate(ref)
    got = ours / np.linalg.norm(ours, axis=1, keepdims=True)
    e = 1.0 - (got * ref).sum(axis=1)

    nw, ns = len(words), len(sents)
    groups = [("single words", e[:nw]), ("sentences", e[nw:nw+ns]),
              ("5-word phrases", e[nw+ns:])]
    q = np.percentile(e, [50, 90, 99])
    print("\n=== %s  (%d texts)" % (model, len(TEXTS)))
    print("    mean %.3e   median %.3e   p90 %.3e   p99 %.3e   max %.3e"
          % (e.mean(), q[0], q[1], q[2], e.max()))
    print("    over the 2e-03 tolerance: %d / %d  (%.1f%%)"
          % ((e > 2e-3).sum(), len(e), 100 * (e > 2e-3).mean()))
    print("    max/median = %.1fx" % (e.max() / q[0]))
    for gname, ge in groups:
        print("      %-15s n=%3d  median %.2e  max %.2e  over 2e-03: %d"
              % (gname, len(ge), np.median(ge), ge.max(), (ge > 2e-3).sum()))
    worst = np.argsort(-e)[:5]
    print("    worst: " + ", ".join("%r %.2e" % (TEXTS[i][:18], e[i])
                                    for i in worst))
