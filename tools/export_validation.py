# NpuEmbeddings -- M7: export validation vectors for the C++ runtime.
# SPDX-License-Identifier: Apache-2.0
#
# The runtime reads weights from the .npue (it has a C++ reader for that) but
# the goldens live in safetensors, and writing a second safetensors parser in
# C++ to check one number would be the wrong trade. So the vectors the runtime
# validates against are dumped here, at build time, as raw little-endian fp32
# with a JSON descriptor.
#
# These are CHECK data, not model data. Nothing in the inference path reads
# them; `main.cpp` uses them only to answer "did the C++ runtime reproduce the
# HuggingFace-derived oracle".
#
# Env: iron env (numpy only, no NPU).
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" tools\export_validation.py

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))
sys.path.insert(0, str(REPO / "tools"))

from npue import Reader, find_goldens                       # noqa: E402
from safetensors_io import load               # noqa: E402



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="name of the container under models/, without .npue")
    ap.add_argument("--seq", type=int, default=64)
    args = ap.parse_args()

    # Per model, so two installed models cannot overwrite each other's
    # fixtures. The runtime's source_sha256 guard would CATCH that, but a
    # collision that cannot happen beats one that is merely detected.
    out = REPO / "runtime" / "artifacts" / "validation" / args.model
    out.mkdir(parents=True, exist_ok=True)

    npue_path = REPO / "models" / f"{args.model}.npue"
    if not npue_path.exists():
        print(f"FAIL -- {npue_path} not found; build it with "
              f"`npuembed --prepare-model models/{args.model}`")
        return 1
    with Reader(npue_path) as r0:
        want_sha = r0.config["source_sha256"]
        n_layers = r0.config["num_layers"]
        arch = r0.config.get("arch", "bert_abs_gelu_postln")

    # arch=2 (nomic-embed-text-v1.5, tasks/0069-0070) keeps its own goldens
    # tree -- a different reference oracle (reference/encoder_nomic.py, RoPE +
    # SwiGLU) wrote them, and find_goldens() matches by source_sha256 content
    # rather than by directory, so pointing it at the right tree is the only
    # branch needed here.
    is_nomic = (arch == "nomic_bert_rope_swiglu")
    goldens_root = REPO / "reference" / ("goldens_nomic" if is_nomic else "goldens")
    make_goldens_hint = "make_goldens_nomic.py" if is_nomic else "make_goldens.py"

    # By checkpoint, not by name -- see tools/npue.py's find_goldens().
    try:
        bpath, tpath = find_goldens(goldens_root, want_sha, args.seq, load)
    except FileNotFoundError as e:
        print(f"FAIL -- {e}")
        return 1
    if not tpath.exists():
        print(f"FAIL -- {tpath.name} not found; the boundary goldens are "
              f"there, so re-run {make_goldens_hint} with --taps")
        return 1

    _, meta = load(bpath)
    taps, _ = load(tpath)

    with Reader(npue_path) as r:
        cfg = r.config
        hidden, head_dim = cfg["hidden"], cfg["head_dim"]
        folded = cfg["fusions"]["qk_scale_folded_into_q"]
        if cfg["source_sha256"] != meta["source_sha256"]:
            print("FAIL -- .npue and goldens are different checkpoints")
            return 1

    a = np.ascontiguousarray(taps["emb.ln"].reshape(-1, hidden), dtype=np.float32)

    # The expected result is the golden put into the space the PACKED pipeline
    # produces: .npue folds 1/sqrt(head_dim) into Q, so the golden's Q third is
    # scaled to match rather than the weights being unfolded. Same reasoning as
    # tasks/0011 -- unfolding would test a different program than the one we run.
    want = np.ascontiguousarray(taps["L0.qkv"].reshape(-1, 3 * hidden),
                                dtype=np.float32).copy()
    if folded:
        want[:, :hidden] *= np.float32(1.0 / math.sqrt(head_dim))

    # -- full-encode vectors --------------------------------------------------
    # emb.sum is the embedding lookup's output BEFORE the first LayerNorm: the
    # gather is a host op in both runtimes, so handing the C++ side its result
    # keeps the two comparable and avoids a 47 MB embedding table read for a
    # check. The mask is the additive form softmax consumes, clamped to a
    # bf16-representable value -- tasks/0021: finfo(float32).min becomes -inf in
    # bf16, which is docs/04-model's NaN landmine arriving through the dtype.
    g, _ = load(bpath)
    emb_sum = np.ascontiguousarray(taps["emb.sum"].reshape(-1, hidden),
                                   dtype=np.float32)
    am = np.ascontiguousarray(g["attention_mask"], dtype=np.float32)
    # Same clamp value the C++ runtime itself uses everywhere (main.cpp's
    # cmask/cmask literals), not the oracle's own np.finfo(float32).min --
    # both are "very negative" enough that softmax zeroes the masked columns
    # either way; matching the runtime's OWN convention here is what makes
    # this a check of the runtime's math, not of a different mask scale.
    add_mask = np.where(am > 0, np.float32(0.0), np.float32(-1.0e30))

    if is_nomic:
        # nomic has no "hf.out.embedding" golden -- sentence-transformers
        # does NOT L2-normalize this checkpoint (no Normalize module in
        # modules.json; tasks/0068 sec 5b), so the boundary file only carries
        # the RAW pooled vector (hf.pool.mean_raw / st.pool.mean_raw). This
        # runtime always L2-normalizes (g_l2_normalize hardcoded true,
        # main.cpp) -- reference/encoder_nomic.py's own encode() already taps
        # BOTH forms explicitly (pool.mean_raw / pool.mean_l2normalized), so
        # the taps file (regenerated with --taps) is read for the normalized
        # one rather than re-deriving it here and risking a second, silently
        # different normalization convention.
        if "pool.mean_l2normalized" not in taps:
            print("FAIL -- goldens_nomic taps have no 'pool.mean_l2normalized' "
                  "-- regenerate with "
                  "`python reference/make_goldens_nomic.py --taps`")
            return 1
        expected_emb = np.ascontiguousarray(taps["pool.mean_l2normalized"],
                                            dtype=np.float32)
    else:
        expected_emb = np.ascontiguousarray(g["hf.out.embedding"], dtype=np.float32)

    (out / "emb_sum.f32").write_bytes(emb_sum.tobytes())
    (out / "add_mask.f32").write_bytes(add_mask.tobytes())
    (out / "attention_mask.f32").write_bytes(am.tobytes())
    (out / "embedding_expected.f32").write_bytes(expected_emb.tobytes())

    (out / "l0_qkv_input.f32").write_bytes(a.tobytes())
    (out / "l0_qkv_expected.f32").write_bytes(want.tobytes())
    (out / "validation.json").write_text(json.dumps({
        "kind": "build artifact -- check data, not model data",
        "source_sha256": meta["source_sha256"],
        "qk_scale_folded_into_q": folded,
        "input": {"file": "l0_qkv_input.f32", "rows": int(a.shape[0]),
                  "cols": int(a.shape[1]), "dtype": "f32le"},
        "expected": {"file": "l0_qkv_expected.f32", "rows": int(want.shape[0]),
                     "cols": int(want.shape[1]), "dtype": "f32le"},
        "tolerance_rel_fro": 5e-3,
        "note": "expected = golden L0.qkv with the Q third scaled by "
                "1/sqrt(head_dim), matching the .npue fold",
    }, indent=2), encoding="utf-8")

    print(f"wrote {out.relative_to(REPO)}   (goldens {bpath.name})")
    print(f"  input    {a.shape} fp32  {a.nbytes / 1024:.1f} KB")
    print(f"  expected {want.shape} fp32  {want.nbytes / 1024:.1f} KB")
    print(f"  emb_sum  {emb_sum.shape} fp32   add_mask {add_mask.shape}")
    print(f"  expected embedding {expected_emb.shape} fp32")
    print(f"  checkpoint {meta['source_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
