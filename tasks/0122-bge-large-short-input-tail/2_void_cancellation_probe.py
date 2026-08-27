"""Is the outlier a CANCELLATION effect?

bf16/bfp16 rounding noise is roughly a fixed fraction of the intermediate
magnitudes, so it lands on the pooled vector as a near-constant ABSOLUTE
perturbation. `1-cos` is a RELATIVE measure. If a text's pre-normalisation CLS
vector is unusually short, the same absolute noise buys a much larger angle --
and the error would track 1/||v|| rather than length or token count.

Run in .venv-ref (needs sentence-transformers).
"""
import math
from pathlib import Path

from sentence_transformers import SentenceTransformer

REPO = Path.cwd()
TEXTS = [
    "test",
    "rain",
    "mechanic",
    "brake pads",
    "heavy rain tonight",
    "the mechanic replaced the brake pads",
    "the mechanic replaced the front brake pads and topped up the oil",
    "the mechanic replaced the front brake pads and topped up the oil "
    "before the annual inspection",
]
# measured by tools/verify_embed_e2e.py, bge-large, artifacts_large_bfp16
ERR = {
    "test": 2.234e-03, "rain": 4.204e-04, "mechanic": 4.442e-04,
    "brake pads": 1.931e-04, "heavy rain tonight": None,
    "the mechanic replaced the brake pads": 1.634e-04,
    "the mechanic replaced the front brake pads and topped up the oil": 1.618e-04,
    "the mechanic replaced the front brake pads and topped up the oil "
    "before the annual inspection": 1.773e-04,
}

st = SentenceTransformer(str(REPO / "models" / "bge-large-en-v1.5"),
                         device="cpu")
st.max_seq_length = 64
raw = st.encode(TEXTS, convert_to_numpy=True, normalize_embeddings=False)

print("%-62s %8s %10s %12s" % ("text", "||v||", "1-cos", "1-cos*||v||^2"))
print("-" * 96)
rows = []
for t, v in zip(TEXTS, raw):
    n = float((v.astype("float64") ** 2).sum() ** 0.5)
    e = ERR.get(t)
    rows.append((t, n, e))
    print("%-62s %8.3f %10s %12s"
          % (t[:60], n,
             "%.3e" % e if e else "-",
             "%.3e" % (e * n * n) if e else "-"))

known = [(n, e) for _, n, e in rows if e]
print()
print("If the mechanism is cancellation, 1-cos should scale as 1/||v||^2 and")
print("the last column should be roughly FLAT. Spread of that column:")
vals = [e * n * n for n, e in known]
print("  min %.3e  max %.3e  ratio %.2fx"
      % (min(vals), max(vals), max(vals) / min(vals)))
print("Spread of 1-cos itself, for comparison:")
es = [e for _, e in known]
print("  min %.3e  max %.3e  ratio %.2fx"
      % (min(es), max(es), max(es) / min(es)))
