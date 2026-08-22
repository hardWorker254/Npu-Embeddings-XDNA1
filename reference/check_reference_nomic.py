# NpuEmbeddings -- M13 (tasks/0069, reference half) GATE: prove
# encoder_nomic.py is the oracle for nomic-embed-text-v1.5, same pattern as
# check_reference.py (M3) / check_reference_gemma.py (C1).
#
# What is compared, and what is not:
#   emb.ln, L{i}.norm2 (i in 0..num_layers-1), last_hidden_state,
#   pool.mean_raw          <- goldens expose (see make_goldens_nomic.py)
#   emb.sum, L{i}.qkv, L{i}.q/.k/.v, L{i}.q_rope/.k_rope, L{i}.scores,
#   L{i}.probs, L{i}.ctx, L{i}.attn_out, L{i}.norm1, L{i}.fc11/.fc12/.silu/
#   .gated/.fc2             <- goldens do NOT expose (no HF hook for these)
# The un-exposed interior taps are covered TRANSITIVELY: a formula error
# inside a layer cannot leave that layer's L{i}.norm2 correct, since norm2 is
# a deterministic function of everything upstream of it.
#
# Also runs a DISCRIMINATING CONTROL (section at the bottom): the SAME
# goldens, scored against two deliberately WRONG oracle configurations
# (RoPE theta=10000 instead of 1000, and the swapped SiLU-on-fc11 SwiGLU
# candidate) -- proving this file's PASS is not a tautology (a broken metric
# that "passes" everything would never be caught by comparing only the
# correct configuration against itself).
#
# Env: iron (numpy only) -- also runs in .venv-ref.
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" reference\check_reference_nomic.py

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoder_nomic import NomicEmbeddingReference   # noqa: E402
from safetensors_io import load                     # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Set from a real run (see this task's log), not guessed a priori -- the
# BERT oracle (check_reference.py) achieves worst rel_fro ~1e-6 / 1-cos ~1e-8;
# the Gemma oracle (check_reference_gemma.py) 2e-6 / 2e-7 despite 24 RMSNorm/
# RoPE/GeGLU layers. nomic has 12 LayerNorm/RoPE/SwiGLU layers -- structurally
# closer to BERT (LayerNorm, not RMSNorm) but with RoPE added, so headroom is
# set between the two rather than re-guessed from scratch.
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


def build_ref(w, cfg, n_layers, **overrides):
    kwargs = dict(
        num_layers=n_layers, hidden=cfg["hidden_size"],
        num_heads=cfg["num_attention_heads"], head_dim=cfg["head_dim"],
        intermediate=cfg["intermediate_size"], eps=cfg["layer_norm_epsilon"],
        rope_theta=float(cfg.get("rope_parameters", {}).get(
            "rope_theta", cfg.get("rotary_emb_base", 1000.0))))
    kwargs.update(overrides)
    return NomicEmbeddingReference(w, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models" / "nomic-embed-text-v1.5"))
    ap.add_argument("--goldens", default=str(REPO / "reference" / "goldens_nomic"
                                             / "nomic-embed-text-v1.5_l12_s64_boundary.safetensors"))
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

    # ------------------------------------------------------------------
    # 1. The correct oracle vs the goldens.
    # ------------------------------------------------------------------
    ref = build_ref(w, cfg, n_layers)
    taps = {}
    pooled_raw, pooled_norm = ref.encode(g["input_ids"], g["attention_mask"], taps=taps)

    rows = [compare("emb.ln", taps["emb.ln"], g["hf.emb.ln"])]
    for i in range(n_layers):
        rows.append(compare(f"L{i}.norm2", taps[f"L{i}.norm2"], g[f"hf.L{i}.norm2"]))
    rows.append(compare("last_hidden_state", taps["last_hidden_state"], g["hf.last_hidden_state"]))
    rows.append(compare("pool.mean_raw", pooled_raw, g["hf.pool.mean_raw"]))
    rows.append(compare("pool.mean_raw vs sentence-transformers (remote code)",
                        pooled_raw, g["st.pool.mean_raw"]))

    print(f"\n{'tensor':<44} {'shape':>18} {'max_abs':>11} {'rel_fro':>11}   ")
    for r in rows:
        if "why" in r:
            print(f"{r['name']:<44} {'':>18} {'':>11} {'':>11}   FAIL {r['why']}")
            continue
        print(f"{r['name']:<44} {str(r['shape']):>18} "
              f"{r['max_abs']:11.3e} {r['rel_fro']:11.3e}   "
              f"{'ok' if r['ok'] else 'FAIL'}")

    # Cosine on the UNNORMALIZED pooled vector -- this checkpoint is not
    # L2-normalized by sentence-transformers (see encoder_nomic.py header),
    # but cosine similarity is scale-invariant, so it is still the right
    # metric for "did we get the direction right".
    cos = (pooled_raw.astype(np.float64) * g["hf.pool.mean_raw"].astype(np.float64)).sum(1)
    cos = cos / (np.linalg.norm(pooled_raw.astype(np.float64), axis=1)
                 * np.linalg.norm(g["hf.pool.mean_raw"].astype(np.float64), axis=1))
    cos_st = (pooled_raw.astype(np.float64) * g["st.pool.mean_raw"].astype(np.float64)).sum(1)
    cos_st = cos_st / (np.linalg.norm(pooled_raw.astype(np.float64), axis=1)
                       * np.linalg.norm(g["st.pool.mean_raw"].astype(np.float64), axis=1))
    print(f"\ncosine vs native HF port     : min {cos.min():.12f}  "
          f"(1-cos max {1 - cos.min():.3e})")
    print(f"cosine vs sentence-transf.   : min {cos_st.min():.12f}  "
          f"(1-cos max {1 - cos_st.min():.3e})")

    interior = [k for k in taps if k not in {r["name"] for r in rows}]
    print(f"\n{len(taps)} taps produced; {len(interior)} have no HF counterpart "
          f"and are validated transitively.")

    failed = [r for r in rows if not r["ok"]]
    cos_ok = (1 - cos.min()) <= TOL_COSINE and (1 - cos_st.min()) <= TOL_COSINE
    primary_pass = not failed and cos_ok
    if not primary_pass:
        print(f"\nFAIL -- {len(failed)} tensor(s) over rel_fro {TOL_RELFRO:.0e}"
              f"{'' if cos_ok else ', cosine over tolerance'}")
    else:
        print(f"\nPASS -- all {len(rows)} comparisons within rel_fro {TOL_RELFRO:.0e}, "
              f"cosine within {TOL_COSINE:.0e}")

    # ------------------------------------------------------------------
    # 2. DISCRIMINATING CONTROL. Score the SAME goldens against deliberately
    #    wrong oracle configurations. If this oracle's PASS above were a
    #    tautology (e.g. a metric that always reports near-zero error), these
    #    would ALSO pass -- they must not. Mirrors tasks/0068's
    #    probe_nomic_arch.py negative controls, but run through THIS file's
    #    actual comparison code, on the actual goldens, not a one-off script.
    # ------------------------------------------------------------------
    print(f"\n{'='*100}\nDISCRIMINATING CONTROL -- same goldens, deliberately wrong oracle config\n{'='*100}")

    controls = [
        ("RoPE theta=10000 (WRONG -- real value is 1000)",
         build_ref(w, cfg, n_layers, rope_theta=10000.0)),
        ("SwiGLU silu(fc11)*fc12 (WRONG -- real is fc11*silu(fc12))",
         build_ref(w, cfg, n_layers, wrong_swiglu=True)),
    ]

    correct_last_hidden_relfro = next(r["rel_fro"] for r in rows if r["name"] == "last_hidden_state")
    correct_1mcos = 1 - cos.min()

    control_rows = []
    for label, cref in controls:
        c_taps = {}
        c_pooled_raw, _ = cref.encode(g["input_ids"], g["attention_mask"], taps=c_taps)
        c_last = compare("last_hidden_state", c_taps["last_hidden_state"], g["hf.last_hidden_state"])
        c_cos = (c_pooled_raw.astype(np.float64) * g["hf.pool.mean_raw"].astype(np.float64)).sum(1)
        c_cos = c_cos / (np.linalg.norm(c_pooled_raw.astype(np.float64), axis=1)
                         * np.linalg.norm(g["hf.pool.mean_raw"].astype(np.float64), axis=1))
        print(f"\n[{label}]")
        print(f"  last_hidden_state  rel_fro={c_last['rel_fro']:.3e}  "
              f"(correct oracle: {correct_last_hidden_relfro:.3e})")
        print(f"  pooled 1-cos       {1 - c_cos.min():.3e}  "
              f"(correct oracle: {correct_1mcos:.3e})")
        control_rows.append((label, c_last["rel_fro"], 1 - c_cos.min()))

    control_ok = all(c_relfro > TOL_RELFRO * 1000 for _, c_relfro, _ in control_rows)
    print(f"\n{'control':<60} {'last_hidden rel_fro':>20} {'pooled 1-cos':>14}")
    for label, c_relfro, c_1mcos in control_rows:
        print(f"{label:<60} {c_relfro:20.3e} {c_1mcos:14.3e}")
    if control_ok:
        print(f"\nPASS -- both wrong configurations are >>1000x worse than the "
              f"correct oracle's last_hidden_state rel_fro {correct_last_hidden_relfro:.3e}. "
              f"This oracle IS sensitive to what it asserts.")
    else:
        print(f"\nFAIL -- a wrong configuration scored suspiciously close to the "
              f"correct one. The comparison itself may be broken.")

    if not primary_pass or not control_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
