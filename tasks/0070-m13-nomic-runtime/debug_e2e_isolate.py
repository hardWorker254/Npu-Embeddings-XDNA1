"""Isolate the verify_embed_e2e.py FAIL on nomic: is it the C++ runtime, or
the sentence-transformers reference side of that harness?

Runs our OWN verified numpy oracle (reference/encoder_nomic.py, rel_fro
3.573e-13 vs real HF per tasks/0069) on the exact same 13-sentence DEFAULT
corpus from tools/verify_embed_e2e.py, WITH the same "search_document: "
prefix, and compares it against:
  (a) the C++ runtime's --embed output for these texts
  (b) sentence-transformers(trust_remote_code=True).encode() for these texts
"""
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "reference"))
sys.path.insert(0, str(REPO / "tools"))

from encoder_nomic import NomicEmbeddingReference, PROMPTS  # noqa: E402
from safetensors_io import load  # noqa: E402

DEFAULT = [
    "A man is playing a guitar on stage.",
    "Someone plays a guitar at a concert.",
    "A woman is slicing an onion in the kitchen.",
    "The stock market closed lower on Tuesday.",
    "Interest rates were raised by the central bank.",
    "Blåbærsyltetøy smaker godt på en skive brød.",
    "机器学习模型可以生成句子向量。",
    "Tokenization turns text into subword pieces.",
    "pneumonoultramicroscopicsilicovolcanoconiosis is a long word",
    "",
    "   ",
    "Short.",
    "The quick brown fox jumps over the lazy dog while the cat sleeps quietly "
    "in the warm afternoon sun near the window of an old wooden house.",
]

prefix = PROMPTS["search_document"]
prefixed = [prefix + t for t in DEFAULT]

model_dir = REPO / "models" / "nomic-embed-text-v1.5"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(model_dir))
enc = tok(prefixed, padding="max_length", max_length=64, truncation=True,
          return_tensors="np")
for i, t in enumerate(prefixed):
    n = int(enc["attention_mask"][i].sum())
    print(f"[{i:2d}] n_tok={n:3d}  truncated={n>=64}  {t[:50]!r}")

w, _ = load(model_dir / "model.safetensors")
ref = NomicEmbeddingReference(w, num_layers=12, hidden=768, num_heads=12,
                              head_dim=64, intermediate=3072, eps=1e-12,
                              rope_theta=1000.0)
raw, l2n = ref.encode(enc["input_ids"], enc["attention_mask"])
print("\noracle pooled (L2-normalized) shape", l2n.shape)

out_dir = REPO / "tasks" / "0070-m13-nomic-runtime" / "embed_check"
out_dir.mkdir(exist_ok=True)
(out_dir / "e2e13_in.txt").write_text("\n".join(prefixed) + "\n", encoding="utf-8")

import subprocess
exe = REPO / "runtime" / "build" / "npuembed.exe"
r = subprocess.run(
    [str(exe), "..", "--model", "nomic-embed-text-v1.5",
     "--artifacts", "artifacts_nomic", "--threads", "16",
     "--embed", str(out_dir / "e2e13_in.txt"), str(out_dir / "e2e13_out.f32")],
    cwd=str(REPO / "runtime"), capture_output=True, text=True, encoding="utf-8")
print(r.stdout[-800:])
if r.returncode != 0:
    print("FAIL", r.stderr)
    sys.exit(1)
got = np.frombuffer((out_dir / "e2e13_out.f32").read_bytes(),
                    dtype=np.float32).reshape(len(DEFAULT), 768)

cos_oracle = (got.astype(np.float64) * l2n.astype(np.float64)).sum(axis=1)
print("\nC++ runtime vs OUR OWN oracle (same prefix, same tokenizer):")
for i in range(len(DEFAULT)):
    print(f"  [{i:2d}] 1-cos {1.0 - cos_oracle[i]:.3e}   {DEFAULT[i][:40]!r}")
print(f"  worst 1-cos: {(1.0 - cos_oracle).max():.3e}")
