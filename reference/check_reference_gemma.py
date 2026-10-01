# NpuEmbeddings -- C1 spike (tasks/0055) GATE: prove encoder_gemma.py is the
# oracle for EmbeddingGemma-300M, same pattern as check_reference.py for M3.
#
# What is compared, and what is not:
#   emb.scaled, L{i}.resid2 (i in 0..22), last_hidden_state, pool.mean,
#   dense2, dense3, out.embedding   <- HF exposes (see make_goldens_gemma.py)
#   L23.resid2 (pre-final-norm), everything inside attention/mlp            <- HF does NOT
# L23.resid2 is covered transitively: an error there cannot leave
# last_hidden_state (= norm(L23.resid2)) correct.
#
# Env: iron (numpy only) -- also runs in .venv-ref.
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" reference\check_reference_gemma.py

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoder_gemma import GemmaEmbeddingReference   # noqa: E402
from safetensors_io import load                     # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Measured (tasks/0055 TASK.md), not guessed a priori: worst per-tensor
# rel_fro across all 30 boundary comparisons was 8.29e-07 (L17.resid2) and
# worst 1-cos was 1.065e-07 (vs the raw HF model) / 2.110e-08 (vs the real
# sentence-transformers pipeline) -- as tight as M3's MiniLM oracle (2.2e-08),
# despite 24 RMSNorm/RoPE/GeGLU layers against MiniLM's 6 LayerNorm/GELU ones.
# Tolerances below have headroom over the measured worst case, not slack for
# an expected-but-unmeasured gap.
TOL_RELFRO = 2e-6
TOL_COSINE = 2e-7


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
    ap.add_argument("--model-dir", default=str(REPO / "models" / "embeddinggemma-300m"))
    ap.add_argument("--goldens", default=str(REPO / "reference" / "goldens_gemma"
                                             / "embeddinggemma-300m_l24_s64_boundary.safetensors"))
    args = ap.parse_args()

    g, meta = load(args.goldens)
    n_layers = int(meta["num_layers"])
    print(f"goldens  : {Path(args.goldens).name}")
    print(f"  model  : {meta['repo_id']}")
    print(f"  shape  : batch {meta['batch']}, seq {meta['seq_len']}, {n_layers} layers")
    print(f"  prompt : {meta['prompt_name']!r} = {meta['prompt_text']!r}")

    model_dir = Path(args.model_dir)
    digest = sha256(model_dir / "model.safetensors")
    if digest != meta["source_sha256"]:
        print(f"\nFAIL -- checkpoint sha256 does not match the goldens:\n"
              f"  goldens    {meta['source_sha256']}\n  on disk    {digest}")
        return 1
    print(f"  sha256   : {digest[:16]}... matches")

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    w, _ = load(model_dir / "model.safetensors")
    d2, _ = load(model_dir / "2_Dense" / "model.safetensors")
    d3, _ = load(model_dir / "3_Dense" / "model.safetensors")
    dense_w = {"2": d2["linear.weight"], "3": d3["linear.weight"]}

    ref = GemmaEmbeddingReference(
        w, dense_w, num_layers=n_layers, hidden=cfg["hidden_size"],
        num_heads=cfg["num_attention_heads"], num_kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], intermediate=cfg["intermediate_size"],
        eps=cfg["rms_norm_eps"], rope_theta_global=cfg["rope_theta"],
        rope_theta_local=cfg["rope_local_base_freq"],
        sliding_window=cfg["sliding_window"],
        sliding_window_pattern=cfg.get("_sliding_window_pattern", 6),
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"])

    taps = {}
    out = ref.encode(g["input_ids"], g["attention_mask"], taps=taps)

    rows = [compare("emb.scaled", taps["emb.scaled"], g["hf.emb.scaled"])]
    for i in range(n_layers - 1):        # L{n_layers-1}.resid2 has no HF tap
        rows.append(compare(f"L{i}.resid2", taps[f"L{i}.resid2"], g[f"hf.L{i}.resid2"]))
    rows.append(compare("last_hidden_state", taps["last_hidden_state"], g["hf.last_hidden_state"]))
    rows.append(compare("pool.mean", taps["pool.mean"], g["hf.pool.mean"]))
    rows.append(compare("dense2", taps["dense2"], g["hf.dense2"]))
    rows.append(compare("dense3", taps["dense3"], g["hf.dense3"]))
    rows.append(compare("out.embedding", out, g["hf.out.embedding"]))
    rows.append(compare("out.embedding vs sentence-transformers", out, g["st.embedding"]))

    print(f"\n{'tensor':<44} {'shape':>18} {'max_abs':>11} {'rel_fro':>11}   ")
    for r in rows:
        if "why" in r:
            print(f"{r['name']:<44} {'':>18} {'':>11} {'':>11}   FAIL {r['why']}")
            continue
        print(f"{r['name']:<44} {str(r['shape']):>18} "
              f"{r['max_abs']:11.3e} {r['rel_fro']:11.3e}   "
              f"{'ok' if r['ok'] else 'FAIL'}")

    cos = (out.astype(np.float64) * g["hf.out.embedding"].astype(np.float64)).sum(1)
    cos_st = (out.astype(np.float64) * g["st.embedding"].astype(np.float64)).sum(1)
    print(f"\ncosine vs HF                : min {cos.min():.12f}  "
          f"(1-cos max {1 - cos.min():.3e})")
    print(f"cosine vs sentence-transf.  : min {cos_st.min():.12f}  "
          f"(1-cos max {1 - cos_st.min():.3e})")

    interior = [k for k in taps if k not in {r["name"] for r in rows}]
    print(f"\n{len(taps)} taps produced; {len(interior)} have no HF counterpart "
          f"and are validated transitively.")

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
