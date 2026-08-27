"""Correlate per-text error between int8n64 and bfp16 bge-large.

If the two datapaths' per-text errors rank-correlate strongly, the tail is a
property of (model, input) that any low-precision perturbation exposes -- not
of one number format's failure mode. Saves e-vectors to .npy for reuse.
Run in .venv-ref from the repo root.
"""
import json, subprocess, tempfile
from array import array
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path.cwd(); SEQ = 64
tok0 = AutoTokenizer.from_pretrained(str(REPO / "models" / "bge-base-en-v1.5"))
vocab = sorted(w for w in tok0.get_vocab()
               if w.isalpha() and len(w) >= 4 and w.islower())
words = vocab[::max(1, len(vocab) // 180)][:180]
corpus = json.loads((REPO / "tools" / "semantic_corpus.json").read_text(encoding="utf-8"))
sents = [s["text"] for s in corpus["sentences"]]
TEXTS = words + sents + [" ".join(words[i:i + 5]) for i in range(0, 40, 5)]

m = REPO / "models" / "bge-large-en-v1.5"
tk = AutoTokenizer.from_pretrained(str(m)); md = AutoModel.from_pretrained(str(m)).eval()
ref = []
for i in range(0, len(TEXTS), 64):
    enc = tk(TEXTS[i:i + 64], padding=True, truncation=True, max_length=SEQ,
             return_tensors="pt")
    with torch.no_grad():
        c = md(**enc).last_hidden_state[:, 0].double().numpy()
    ref.append(c / np.linalg.norm(c, axis=1, keepdims=True))
ref = np.concatenate(ref)

exe = REPO / "runtime" / "build" / "npuembed.exe"
out = {}
for key, model, artifacts in [("int8n64", "bge-large-en-v1.5.int8n64", "artifacts_int8c_large_n64"),
                              ("bfp16", "bge-large-en-v1.5", "artifacts_large_bfp16")]:
    with tempfile.TemporaryDirectory(prefix="t51c_") as td:
        d = Path(td)
        (d / "in.txt").write_text("\n".join(TEXTS) + "\n", encoding="utf-8")
        p = subprocess.run([str(exe), str(REPO), "--model", model, "--artifacts",
                            artifacts, "--embed", str(d / "in.txt"),
                            str(d / "out.f32"), "--threads", "24"],
                           capture_output=True, text=True, encoding="utf-8")
        assert p.returncode == 0, p.stdout + p.stderr
        a = array("f"); a.frombytes((d / "out.f32").read_bytes())
        v = np.asarray(a, dtype=np.float64).reshape(len(TEXTS), 1024)
    g = v / np.linalg.norm(v, axis=1, keepdims=True)
    out[key] = 1.0 - (g * ref).sum(axis=1)
    np.save(REPO / "tasks" / "0129-t51-int8-tail" / f"e_{key}.npy", out[key])

def rank(a):
    r = np.empty_like(a); r[np.argsort(a)] = np.arange(len(a)); return r
ra, rb = rank(out["int8n64"]), rank(out["bfp16"])
rho = np.corrcoef(ra, rb)[0, 1]
pearson_log = np.corrcoef(np.log(out["int8n64"]), np.log(out["bfp16"]))[0, 1]
print("spearman rho (all 224):        %.3f" % rho)
print("pearson on log errors:         %.3f" % pearson_log)
w = slice(0, 180)
rho_w = np.corrcoef(rank(out["int8n64"][w]), rank(out["bfp16"][w]))[0, 1]
print("spearman rho (180 words only): %.3f" % rho_w)
top = set(np.argsort(-out["int8n64"])[:10]) & set(np.argsort(-out["bfp16"])[:10])
print("overlap of top-10 worst texts: %d / 10  -> %s"
      % (len(top), sorted(TEXTS[i] for i in top)))
