# tasks/0136: gte NPU output vs the numpy oracle (reference/encoder_gte.py).
# Runs in .venv-ref (needs transformers for AutoTokenizer only -- the forward
# pass is the numpy oracle, which already carries 0134's repairs by
# construction).
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\vegar\Documents\GitHub\NpuEmbeddings")
SCRATCH = Path(r"C:\Users\vegar\AppData\Local\Temp\claude"
               r"\C--Users-vegar-Documents-GitHub-NpuEmbeddings"
               r"\26879072-e944-41ac-96c5-a744da5f297a\scratchpad")
sys.path.insert(0, str(ROOT / "reference"))

from safetensors_io import load          # noqa: E402
import encoder_gte                        # noqa: E402

texts = io.open(SCRATCH / "gte_in.txt", encoding="utf-8").read().splitlines()
npu = np.fromfile(SCRATCH / "gte_out.f32", dtype=np.float32).reshape(-1, 768)
assert len(texts) == npu.shape[0], (len(texts), npu.shape)

from transformers import AutoTokenizer    # noqa: E402
tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gte-multilingual-base"))

raw, _ = load(str(ROOT / "models" / "gte-multilingual-base" / "model.safetensors"))
w = {}
for k, v in raw.items():
    kk = k[4:] if k.startswith("new.") else k
    w[kk] = v.astype(np.float32) if v.dtype == np.float16 else v

enc = encoder_gte.GteEncoder(w)

print(f"{'idx':>3} {'tok':>4} {'1-cos':>12}   text-id")
worst = 0.0
for i, t in enumerate(texts):
    ids = tok(t, truncation=True, max_length=64)["input_ids"]
    out = enc.encode(np.asarray(ids, dtype=np.int64))
    ref = out["cls_normalized"].astype(np.float64)
    got = npu[i].astype(np.float64)
    got = got / np.linalg.norm(got)
    one_cos = 1.0 - float(ref @ got)
    worst = max(worst, one_cos)
    print(f"{i:>3} {len(ids):>4} {one_cos:>12.3e}   text_{i}")
print(f"worst 1-cos: {worst:.3e}")
