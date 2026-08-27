"""T51 step 1: the same 224 texts through int8 bge-large at tile_n = 64.

0122 measured a 100x max/median tail on bfp16 bge-large (tile_n 32), entirely
in single-word inputs, with bge-base (identical bfp16 datapath, tile_n 48,
12 layers) showing a 3.6x spread and zero violations. Three candidate
discriminators were left standing: depth (24 vs 12), width (1024 vs 768), and
tile_n (32 -- bge-large is the only model not at 48).

int8 bge-large at tile_n 64 (shipped, tasks/0081) changes the tile geometry
AND the datapath while keeping depth and width -- so:
  * tail COMPRESSED here  -> the mechanism is tied to the bfp16 datapath or
    its tile_n-32 layout, not to depth/width alone;
  * tail PERSISTS (~100x) -> depth/width (or something both datapaths share)
    carries it, and tile_n/datapath are exonerated.
Absolute error is expected higher (int8 bge-large 1-cos FAILS at 2.968e-03,
0085); the question is the SHAPE of the distribution, not its level.

bfp16 bge-large is re-run in the same session as the control, so the
comparison never crosses an environment boundary.

Run in .venv-ref, from the repo root.
"""
import json
import subprocess
import tempfile
from array import array
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path.cwd()
SEQ = 64

# The corpus is 0122's, byte for byte: deterministic draw from bge-base's
# vocab + the semantic corpus sentences + 8 five-word phrases.
tok0 = AutoTokenizer.from_pretrained(str(REPO / "models" / "bge-base-en-v1.5"))
vocab = [w for w in tok0.get_vocab()
         if w.isalpha() and len(w) >= 4 and w.islower()]
vocab.sort()
words = vocab[::max(1, len(vocab) // 180)][:180]
corpus = json.loads((REPO / "tools" / "semantic_corpus.json")
                    .read_text(encoding="utf-8"))
sents = [s["text"] for s in corpus["sentences"]]
TEXTS = words + sents + [" ".join(words[i:i + 5]) for i in range(0, 40, 5)]
print("corpus: %d texts (%d single words, %d sentences, %d phrases)"
      % (len(TEXTS), len(words), len(sents), 8))

# (name shown, container --model, artifacts set, hidden)
CASES = [
    ("bge-large int8 tile_n=64", "bge-large-en-v1.5.int8n64",
     "artifacts_int8c_large_n64", 1024),
    ("bge-large bfp16 tile_n=32 (control)", "bge-large-en-v1.5",
     "artifacts_large_bfp16", 1024),
]

exe = REPO / "runtime" / "build" / "npuembed.exe"

# One fp32 reference, shared by both cases (same checkpoint).
m = REPO / "models" / "bge-large-en-v1.5"
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

for label, model, artifacts, dim in CASES:
    with tempfile.TemporaryDirectory(prefix="t51_") as td:
        d = Path(td)
        (d / "in.txt").write_text("\n".join(TEXTS) + "\n", encoding="utf-8")
        p = subprocess.run([str(exe), str(REPO), "--model", model,
                            "--artifacts", artifacts,
                            "--embed", str(d / "in.txt"), str(d / "out.f32"),
                            "--threads", "24"],
                           capture_output=True, text=True, encoding="utf-8")
        assert p.returncode == 0, p.stdout + p.stderr
        status = [l for l in p.stdout.splitlines() if "datapath" in l]
        a = array("f")
        a.frombytes((d / "out.f32").read_bytes())
        ours = np.asarray(a, dtype=np.float64).reshape(len(TEXTS), dim)

    got = ours / np.linalg.norm(ours, axis=1, keepdims=True)
    e = 1.0 - (got * ref).sum(axis=1)

    nw, ns = len(words), len(sents)
    groups = [("single words", e[:nw]), ("sentences", e[nw:nw + ns]),
              ("5-word phrases", e[nw + ns:])]
    q = np.percentile(e, [50, 90, 99])
    print("\n=== %s" % label)
    for l in status:
        print("   " + l.strip())
    print("    mean %.3e   median %.3e   p90 %.3e   p99 %.3e   max %.3e"
          % (e.mean(), q[0], q[1], q[2], e.max()))
    print("    over the 2e-03 tolerance: %d / %d  (%.1f%%)"
          % ((e > 2e-3).sum(), len(e), 100 * (e > 2e-3).mean()))
    print("    max/median = %.1fx    p99/median = %.1fx"
          % (e.max() / q[0], q[2] / q[0]))
    for gname, ge in groups:
        print("      %-15s n=%3d  median %.2e  max %.2e  over 2e-03: %d"
              % (gname, len(ge), np.median(ge), ge.max(), (ge > 2e-3).sum()))
    worst = np.argsort(-e)[:5]
    print("    worst: " + ", ".join("%r %.2e" % (TEXTS[i][:18], e[i])
                                    for i in worst))

# -- appended after the first run: quantify "same tail" with a per-text
#    rank correlation between the two datapaths' errors. Saves the error
#    vectors so the correlation is computable without re-encoding.
