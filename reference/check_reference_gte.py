# NpuEmbeddings -- 0.5.0 (tasks/0136, reference half) GATE: prove
# encoder_gte.py is the oracle for gte-multilingual-base, same pattern as
# check_reference_nomic.py (M13) / check_reference.py (M3).
#
# What is compared, and what is not:
#   emb.ln, L{i}.mlp_ln (i in 0..num_layers-1), last_hidden_state,
#   pool.cls_raw, pool.cls_norm       <- goldens expose (make_goldens_gte.py)
#   qkv, q/k/v splits, q_rope/k_rope, scores, probs, ctx, o_proj, attn_ln,
#   up/gate, gelu(gate)*up, down_proj  <- goldens do NOT expose
# The un-exposed interior is covered TRANSITIVELY: a formula error inside a
# layer cannot leave that layer's mlp_ln output correct.
#
# THIS FILE CARRIES MORE WEIGHT than its nomic sibling: gte has no native
# transformers port, so make_goldens_gte.py's two oracles share the remote
# forward code and its cross-gate only covers the pipeline. The numpy
# encoder here is the one genuinely independent implementation -- which is
# why the DISCRIMINATING CONTROLS section runs THREE deliberately wrong
# configurations (the two wrong RoPE readings 0134 measured, and the
# activation on the wrong GLU half), proving the PASS is not a tautology.
#
# Env: iron (numpy only) -- also runs in .venv-ref.
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" reference\check_reference_gte.py

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import encoder_gte                                  # noqa: E402
from encoder_gte import GteEncoder                  # noqa: E402
from safetensors_io import load                     # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Set from a real run (tasks/0136), not guessed a priori: measured worst
# valid-rows rel_fro on this corpus is 1.4e-6 at layer 12 (0134's per-layer
# probe read 7e-08 -> 1.9e-06 on unpadded singles), 1-cos 1.3e-7 on pooled
# CLS. Same order as the nomic oracle's gate; headroom ~3x.
TOL_RELFRO = 5e-6
TOL_COSINE = 5e-7


def compare(name, got, want, am=None):
    """When `am` (the [B, S] attention mask) is given, the GATED metric is
    computed over VALID rows only, and the padded rows' max|diff| is
    REPORTED beside it, ungated.

    This differs from check_reference_nomic.py's whole-tensor compare, and
    the difference was measured, not assumed (tasks/0136): on this model the
    repaired HF reference and the numpy oracle agree at 1.4e-06 rel_fro on
    every VALID row and diverge up to 1e-03 max-abs on PADDED rows by layer
    11. Padded rows are don't-cares by construction -- CLS pooling reads row
    0 only, the runtime masks them out of pooling, and HF's own downstream
    consumers never read them -- so gating on them would fail the oracle
    over values no pipeline anywhere consumes. They are still printed,
    because a padded-row divergence that suddenly GREW would be worth
    seeing."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return {"name": name, "shape_got": got.shape, "shape_want": want.shape,
                "ok": False, "why": "shape mismatch"}
    shape0 = tuple(got.shape)
    pad_max = None
    if am is not None:
        B = am.shape[0]
        keep_g, keep_w, pad_d = [], [], [0.0]
        for b in range(B):
            n = int(am[b].sum())
            keep_g.append(got[b, :n].ravel())
            keep_w.append(want[b, :n].ravel())
            if n < got.shape[1]:
                pad_d.append(float(np.abs(got[b, n:] - want[b, n:]).max()))
        got = np.concatenate(keep_g)
        want = np.concatenate(keep_w)
        pad_max = max(pad_d)
    denom = np.linalg.norm(want)
    relfro = float(np.linalg.norm(got - want) / denom) if denom else 0.0
    return {
        "name": name,
        "shape": shape0,
        "max_abs": float(np.abs(got - want).max()),
        "rel_fro": relfro,
        "pad_max": pad_max,
        "ok": relfro <= TOL_RELFRO,
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_weights(model_dir):
    raw, _ = load(model_dir / "model.safetensors")
    w = {}
    for k, v in raw.items():
        kk = k[4:] if k.startswith("new.") else k       # strip 'new.'
        w[kk] = v.astype(np.float32) if v.dtype == np.float16 else v
    return w


def run_batch(enc, input_ids, attention_mask, cos, sin, n_layers):
    """Per-sentence forward with per-layer taps. GteEncoder works one
    sequence at a time (its encode() signature); this collects the batch
    the goldens store, tapping each layer boundary the way 0134's probe
    does."""
    B, S = input_ids.shape
    taps = {"emb.ln": np.zeros((B, S, enc.hidden), dtype=np.float32)}
    for i in range(n_layers):
        taps[f"L{i}.mlp_ln"] = np.zeros((B, S, enc.hidden), dtype=np.float32)
    last = np.zeros((B, S, enc.hidden), dtype=np.float32)
    cls_raw = np.zeros((B, enc.hidden), dtype=np.float32)
    cls_norm = np.zeros((B, enc.hidden), dtype=np.float32)
    for b in range(B):
        am = attention_mask[b].astype(np.float32)
        mask = np.where(am > 0, 0.0, -1e9).astype(np.float32)
        h = enc.embed(input_ids[b])
        taps["emb.ln"][b] = h
        for i in range(n_layers):
            h = enc.layer(h, i, cos, sin, mask)
            taps[f"L{i}.mlp_ln"][b] = h
        last[b] = h
        cls_raw[b] = h[0]
        cls_norm[b] = h[0] / np.linalg.norm(h[0])
    return taps, last, cls_raw, cls_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default=str(REPO / "models" / "gte-multilingual-base"))
    ap.add_argument("--goldens",
                    default=str(REPO / "reference" / "goldens_gte" /
                                "gte-multilingual-base_l12_s64_boundary"
                                ".safetensors"))
    args = ap.parse_args()

    g, meta = load(args.goldens)
    n_layers = int(meta["num_layers"])
    seq_len = int(meta["seq_len"])
    print(f"goldens  : {Path(args.goldens).name}")
    print(f"  model  : {meta['repo_id']}")
    print(f"  shape  : batch {meta['batch']}, seq {seq_len}, "
          f"{n_layers} layers, NO prompt (gte has no prompts table)")

    model_dir = Path(args.model_dir)
    digest = sha256(model_dir / "model.safetensors")
    if digest != meta["source_sha256"]:
        print(f"\nFAIL -- checkpoint sha256 does not match the goldens:\n"
              f"  goldens    {meta['source_sha256']}\n  on disk    {digest}")
        return 1
    print(f"  sha256   : {digest[:16]}... matches")

    w = load_weights(model_dir)
    enc = GteEncoder(w)
    cos, sin = encoder_gte.rope_cos_sin(seq_len)

    # ------------------------------------------------------------------
    # 1. The correct oracle vs the goldens.
    # ------------------------------------------------------------------
    taps, last, cls_raw, cls_norm = run_batch(
        enc, g["input_ids"], g["attention_mask"], cos, sin, n_layers)

    am = g["attention_mask"]
    rows = [compare("emb.ln", taps["emb.ln"], g["hf.emb.ln"], am)]
    for i in range(n_layers):
        rows.append(compare(f"L{i}.mlp_ln", taps[f"L{i}.mlp_ln"],
                            g[f"hf.L{i}.mlp_ln"], am))
    rows.append(compare("last_hidden_state", last, g["hf.last_hidden_state"],
                        am))
    rows.append(compare("pool.cls_raw", cls_raw, g["hf.pool.cls_raw"]))
    rows.append(compare("pool.cls_norm", cls_norm, g["hf.pool.cls_norm"]))
    rows.append(compare("pool.cls_norm vs sentence-transformers",
                        cls_norm, g["st.pool.cls_norm"]))

    print(f"\n{'tensor':<44} {'shape':>16} {'max_abs':>11} "
          f"{'rel_fro':>11}          pad rows max|d| (ungated)")
    for r in rows:
        if "why" in r:
            print(f"{r['name']:<44} {'':>16} {'':>11} {'':>11}   "
                  f"FAIL {r['why']}")
            continue
        pad = (f"{r['pad_max']:.3e}" if r.get("pad_max") is not None else "")
        print(f"{r['name']:<44} {str(r['shape']):>16} "
              f"{r['max_abs']:11.3e} {r['rel_fro']:11.3e}   "
              f"{'ok' if r['ok'] else 'FAIL':<6} {pad}")

    cos_sim = (cls_norm.astype(np.float64) *
               g["hf.pool.cls_norm"].astype(np.float64)).sum(1)
    cos_st = (cls_norm.astype(np.float64) *
              g["st.pool.cls_norm"].astype(np.float64)).sum(1)
    print(f"\ncosine vs repaired HF remote : min {cos_sim.min():.12f}  "
          f"(1-cos max {1 - cos_sim.min():.3e})")
    print(f"cosine vs sentence-transf.   : min {cos_st.min():.12f}  "
          f"(1-cos max {1 - cos_st.min():.3e})")

    failed = [r for r in rows if not r["ok"]]
    cos_ok = ((1 - cos_sim.min()) <= TOL_COSINE and
              (1 - cos_st.min()) <= TOL_COSINE)
    primary_pass = not failed and cos_ok
    if not primary_pass:
        print(f"\nFAIL -- {len(failed)} tensor(s) over rel_fro "
              f"{TOL_RELFRO:.0e}"
              f"{'' if cos_ok else ', cosine over tolerance'}")
    else:
        print(f"\nPASS -- all {len(rows)} comparisons within rel_fro "
              f"{TOL_RELFRO:.0e}, cosine within {TOL_COSINE:.0e}")

    # ------------------------------------------------------------------
    # 2. DISCRIMINATING CONTROLS. The SAME goldens, three deliberately wrong
    #    oracle configurations -- the two wrong RoPE readings tasks/0134
    #    measured (both of which "look fine" at 1e-02, the exact regime 0068
    #    warned about) and the activation on the wrong GLU half. If the PASS
    #    above were a tautology these would also pass -- they must not.
    # ------------------------------------------------------------------
    print(f"\n{'='*100}\nDISCRIMINATING CONTROLS -- same goldens, "
          f"deliberately wrong oracle config\n{'='*100}")

    def tables_from_inv_freq(inv):
        t = np.arange(seq_len, dtype=np.float32)
        f = np.einsum("i,j->ij", t, inv.astype(np.float32))
        e = np.concatenate([f, f], axis=-1)
        return np.cos(e), np.sin(e)

    i32 = np.arange(0, 64, 2, dtype=np.float64)
    wrong_rope = [
        ("RoPE theta=20000 plain (WRONG -- NTK cache overwrites it)",
         tables_from_inv_freq(1.0 / (20000.0 ** (i32 / 64.0)))),
        ("RoPE theta=160000 no-correction (WRONG -- misses 8^(-1/32))",
         tables_from_inv_freq(1.0 / (160000.0 ** (i32 / 64.0)))),
    ]

    correct_relfro = next(r["rel_fro"] for r in rows
                          if r["name"] == "last_hidden_state")
    control_rows = []
    for label, (c_cos, c_sin) in wrong_rope:
        _, c_last, _, c_norm = run_batch(
            enc, g["input_ids"], g["attention_mask"], c_cos, c_sin, n_layers)
        c = compare("last_hidden_state", c_last, g["hf.last_hidden_state"],
                    am)
        c1m = 1 - (c_norm.astype(np.float64) *
                   g["hf.pool.cls_norm"].astype(np.float64)).sum(1).min()
        control_rows.append((label, c["rel_fro"], c1m))

    # Activation on the WRONG half of the fused up_gate_proj (act(up)*gate
    # instead of gate-activated) -- 0134's probe measured this structurally
    # wrong at relfro 0.78 on layer 0.
    real_mlp = GteEncoder.mlp

    def wrong_mlp(self, h):
        ww = self.w
        p = self.pfx
        up_gate = self.linear(h, ww[p + "mlp.up_gate_proj.weight"])
        up = up_gate[:, :self.inter]
        gate = up_gate[:, self.inter:]
        act = self.gelu(up) * gate               # WRONG: activation on up
        return self.linear(act, ww[p + "mlp.down_proj.weight"],
                           ww[p + "mlp.down_proj.bias"])

    GteEncoder.mlp = wrong_mlp
    try:
        _, c_last, _, c_norm = run_batch(
            enc, g["input_ids"], g["attention_mask"], cos, sin, n_layers)
    finally:
        GteEncoder.mlp = real_mlp
    c = compare("last_hidden_state", c_last, g["hf.last_hidden_state"], am)
    c1m = 1 - (c_norm.astype(np.float64) *
               g["hf.pool.cls_norm"].astype(np.float64)).sum(1).min()
    control_rows.append(
        ("GELU on the UP half (WRONG -- real is up * gelu(gate))",
         c["rel_fro"], c1m))

    print(f"\n{'control':<60} {'last_hidden rel_fro':>20} "
          f"{'pooled 1-cos':>14}")
    for label, c_relfro, c_1mcos in control_rows:
        print(f"{label:<60} {c_relfro:20.3e} {c_1mcos:14.3e}")

    control_ok = all(c_relfro > TOL_RELFRO * 1000
                     for _, c_relfro, _ in control_rows)
    if control_ok:
        print(f"\nPASS -- all three wrong configurations are >>1000x worse "
              f"than the correct oracle's last_hidden_state rel_fro "
              f"{correct_relfro:.3e}. This oracle IS sensitive to what it "
              f"asserts.")
    else:
        print(f"\nFAIL -- a wrong configuration scored suspiciously close to "
              f"the correct one. The comparison itself may be broken.")

    if not primary_pass or not control_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
