# NpuEmbeddings -- M3 GATE: prove reference/encoder.py IS the oracle.
#
# Runs the pure-numpy reference against the HuggingFace goldens, tensor by
# tensor. Passing this is what makes encoder.py usable as the thing M5 kernels
# are validated against.
#
# Deliberately runs with numpy ONLY, so it can execute in the iron env where
# the NPU work happens -- no torch, no transformers, no safetensors package.
# That is the whole point of the file-based env boundary.
#
# What is compared, and what is not:
#   emb.ln, L{i}.ln2, last_hidden_state, pool.mean, out.embedding   <- HF exposes
#   L{i}.ln1, qkv, scores, probs, ctx, ffn_up, gelu, ffn_down       <- HF does NOT
# HF has no hook for the mid-layer or interior tensors. They are covered
# transitively: an error there cannot leave L{i}.ln2 correct.
#
# Env: iron (numpy only) -- also runs in .venv-ref.
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" reference\check_reference.py

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoder import MiniLMReference, read_pooling          # noqa: E402
from safetensors_io import load              # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# fp32 numpy vs fp32 torch on the same weights. Both are IEEE fp32 doing the
# same maths in a different order, so the gap should be accumulation noise:
# ~1e-6 relative Frobenius, growing slightly with depth. Anything at 1e-3 is a
# formula difference (tanh GELU, unbiased variance, wrong eps placement), not
# noise -- which is exactly what this bar is set to catch.
TOL_RELFRO = 2e-5
TOL_COSINE = 1e-6          # 1 - cos for the final embedding


def compare(name, got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return {"name": name, "shape_got": got.shape, "shape_want": want.shape,
                "ok": False, "why": "shape mismatch"}
    denom = np.linalg.norm(want)
    relfro = float(np.linalg.norm(got - want) / denom) if denom else 0.0
    return {
        "name": name,
        "shape": tuple(got.shape),
        "max_abs": float(np.abs(got - want).max()),
        "rel_fro": relfro,
        "ok": relfro <= TOL_RELFRO,
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models" / "all-MiniLM-L6-v2"))
    ap.add_argument("--goldens", default=str(REPO / "reference" / "goldens"
                                             / "minilm_l6_s64_boundary.safetensors"))
    args = ap.parse_args()

    g, meta = load(args.goldens)
    n_layers = int(meta["num_layers"])
    print(f"goldens  : {Path(args.goldens).name}")
    print(f"  model  : {meta['repo_id']}  ({meta['torch']} / {meta['transformers']})")
    print(f"  shape  : batch {meta['batch']}, seq {meta['seq_len']}, {n_layers} layers")

    # A golden compared against a different checkpoint is worse than no golden.
    model_dir = Path(args.model_dir)
    digest = sha256(model_dir / "model.safetensors")
    if digest != meta["source_sha256"]:
        print(f"\nFAIL -- checkpoint sha256 does not match the goldens:\n"
              f"  goldens    {meta['source_sha256']}\n  on disk    {digest}")
        return 1
    print(f"  sha256   : {digest[:16]}... matches")

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    w, _ = load(model_dir / "model.safetensors")
    ref = MiniLMReference(w, num_layers=n_layers,
                          num_heads=cfg["num_attention_heads"],
                          eps=cfg["layer_norm_eps"],
                          pooling=read_pooling(model_dir))

    taps = {}
    out = ref.encode(g["input_ids"], g["attention_mask"], g["token_type_ids"], taps=taps)

    rows = [compare("emb.ln", taps["emb.ln"], g["hf.emb.ln"])]
    for i in range(n_layers):
        rows.append(compare(f"L{i}.ln2", taps[f"L{i}.ln2"], g[f"hf.L{i}.ln2"]))
    rows.append(compare("last_hidden_state", taps["last_hidden_state"],
                        g["hf.last_hidden_state"]))
    rows.append(compare("pool.mean", taps["pool.mean"], g["hf.pool.mean"]))
    rows.append(compare("out.embedding", out, g["hf.out.embedding"]))
    # Second, independent oracle: the real sentence-transformers pipeline.
    rows.append(compare("out.embedding vs sentence-transformers", out, g["st.embedding"]))

    print(f"\n{'tensor':<40} {'shape':>18} {'max_abs':>11} {'rel_fro':>11}   ")
    for r in rows:
        if "why" in r:
            print(f"{r['name']:<40} {'':>18} {'':>11} {'':>11}   FAIL {r['why']}")
            continue
        print(f"{r['name']:<40} {str(r['shape']):>18} "
              f"{r['max_abs']:11.3e} {r['rel_fro']:11.3e}   "
              f"{'ok' if r['ok'] else 'FAIL'}")

    # Cosine is the metric that actually matters for an embedding model: a
    # relative-Frobenius pass with a cosine failure would mean we got the
    # direction wrong while keeping the magnitude, which no downstream user
    # would forgive.
    cos = (out.astype(np.float64) * g["hf.out.embedding"].astype(np.float64)).sum(1)
    cos_st = (out.astype(np.float64) * g["st.embedding"].astype(np.float64)).sum(1)
    print(f"\ncosine vs HF                : min {cos.min():.12f}  "
          f"(1-cos max {1 - cos.min():.3e})")
    print(f"cosine vs sentence-transf.  : min {cos_st.min():.12f}  "
          f"(1-cos max {1 - cos_st.min():.3e})")

    # Taps HF cannot see. Report shapes so M5 knows exactly what it must match.
    interior = [k for k in taps if k not in {r["name"] for r in rows}]
    print(f"\n{len(taps)} taps produced; {len(interior)} have no HF counterpart "
          f"and are validated transitively:")
    for k in sorted(interior, key=lambda s: (len(s), s))[:6]:
        print(f"  {k:<24} {taps[k].shape}")
    print(f"  ... ({len(interior)} total, see goldens --taps)")

    failed = [r for r in rows if not r["ok"]]
    cos_ok = (1 - cos.min()) <= TOL_COSINE and (1 - cos_st.min()) <= TOL_COSINE
    if failed or not cos_ok:
        print(f"\nFAIL -- {len(failed)} tensor(s) over rel_fro {TOL_RELFRO:.0e}"
              f"{'' if cos_ok else ', cosine over tolerance'}")
        return 1
    print(f"\nPASS -- all {len(rows)} comparisons within rel_fro {TOL_RELFRO:.0e}, "
          f"cosine within {TOL_COSINE:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
