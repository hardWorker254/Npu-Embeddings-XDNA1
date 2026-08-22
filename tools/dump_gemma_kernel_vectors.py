# NpuEmbeddings -- tasks/0063: build-time-only test-vector generator for the
# standalone C++ EmbeddingGemma kernels (runtime/src/gemma_kernels.cpp).
#
# Python is fine here per CLAUDE.md rule 5 ("Python for build-time design
# generation and prototyping") -- this script produces a flat binary the C++
# test harness (runtime/src/gemma_kernels_test.cpp) reads with ifstream, no
# Python at runtime. Same role as tools/gen_gemma_tokenizer_table.py.
#
# Source of the "before" tensors: reference/goldens_gemma/
# embeddinggemma-300m_l24_s64_taps.safetensors, the full intermediate dump
# reference/make_goldens_gemma.py --taps already wrote (real checkpoint,
# real sentences, real forward pass) -- see tasks/0055-m10-embeddinggemma-spike
# and tasks/0055's Commands section for how to regenerate it if missing.
# "after" values are the REAL tapped output of the SAME forward pass at the
# point immediately downstream of each op -- not recomputed from the "before"
# tensor by calling the reference function a second time -- so this is
# checking the C++ kernel against genuine intermediate values from a live
# HuggingFace-checkpoint encode, per the task's verification requirement.
#
# Binary format (little-endian, read by gemma_kernels_test.cpp):
#   repeated records until EOF:
#     char[4]  magic      b"GKK1"
#     u32      name_len
#     char[name_len] name (utf-8, not NUL-terminated)
#     u32      kind        0=rmsnorm, 1=rope, 2=geglu
#     u32      rows
#     u32      dim
#     f32      eps         (kind 0 only, else 0)
#     u32      seq_len     (kind 1 only, else 0)
#     f64      theta       (kind 1 only, else 0)
#     payload, all float32, C-contiguous:
#       kind 0: weight[dim]  before[rows*dim]  after[rows*dim]
#       kind 1: before[rows*dim]  after[rows*dim]
#       kind 2: gate[rows*dim]  up[rows*dim]  after[rows*dim]
#
# Usage:
#   & .\.venv-ref\Scripts\python.exe tools\dump_gemma_kernel_vectors.py

import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))

from safetensors_io import load  # noqa: E402

MODEL_DIR = REPO / "models" / "embeddinggemma-300m"
TAPS_PATH = REPO / "reference" / "goldens_gemma" / "embeddinggemma-300m_l24_s64_taps.safetensors"
OUT_PATH = REPO / "runtime" / "build_gemma_kernels" / "gemma_kernel_vectors.bin"


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def write_record(f, name, kind, rows, dim, *, eps=0.0, seq_len=0, theta=0.0,
                 weight=None, blocks=()):
    name_b = name.encode("utf-8")
    f.write(b"GKK1")
    f.write(struct.pack("<I", len(name_b)))
    f.write(name_b)
    f.write(struct.pack("<III", kind, rows, dim))
    f.write(struct.pack("<f", eps))
    f.write(struct.pack("<I", seq_len))
    f.write(struct.pack("<d", theta))
    total = 0
    if weight is not None:
        wb = f32(weight)
        assert wb.size == dim, (name, wb.shape, dim)
        f.write(wb.tobytes())
        total += wb.size
    for b in blocks:
        b = f32(b)
        assert b.size == rows * dim, (name, b.shape, rows, dim)
        f.write(b.tobytes())
        total += b.size
    return total


def main():
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    eps = float(cfg["rms_norm_eps"])
    head_dim = int(cfg["head_dim"])
    hidden = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg["num_key_value_heads"])
    theta_global = float(cfg["rope_theta"])
    theta_local = float(cfg["rope_local_base_freq"])
    swp = int(cfg.get("_sliding_window_pattern", 6))

    w, _ = load(MODEL_DIR / "model.safetensors")
    taps, meta = load(TAPS_PATH)
    B = int(meta["batch"])
    S = int(meta["seq_len"])
    assert taps["L0.q_proj"].shape == (B, S, hidden)

    def is_full(i):
        return (i + 1) % swp == 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    n_elems = 0
    with open(OUT_PATH, "wb") as f:

        # -- 1. RMSNorm, full hidden size (dim=768) -----------------------
        # Four sites per layer, each a REAL (before, after) pair from one
        # forward pass: the tensor immediately upstream of a rms_norm()
        # call, and the tapped tensor immediately downstream of it.
        rows_full = B * S
        for L in (0, 5, 11, 23):
            p = f"layers.{L}."
            sites = [
                ("ln_in", (taps[f"L{L-1}.resid2"] if L > 0 else taps["emb.scaled"]),
                 w[p + "input_layernorm.weight"], taps[f"L{L}.ln_in"]),
                ("ln_post_attn", taps[f"L{L}.attn_out"],
                 w[p + "post_attention_layernorm.weight"], taps[f"L{L}.ln_post_attn"]),
                ("ln_pre_ffn", taps[f"L{L}.resid1"],
                 w[p + "pre_feedforward_layernorm.weight"], taps[f"L{L}.ln_pre_ffn"]),
                ("ln_post_ffn", taps[f"L{L}.down"],
                 w[p + "post_feedforward_layernorm.weight"], taps[f"L{L}.ln_post_ffn"]),
            ]
            for site, before, weight, after in sites:
                name = f"rmsnorm_full_L{L}_{site}"
                n = write_record(f, name, 0, rows_full, hidden, eps=eps,
                                 weight=weight, blocks=(before, after))
                n_records += 1
                n_elems += n

        # -- 2. RMSNorm, per-head (q_norm/k_norm, dim=head_dim) -----------
        # q_normed/k_normed taps are [B,H,S,D]; the "before" (q_proj/k_proj,
        # [B,S,H*D]) is reshaped to [B,S,H,D] then transposed to [B,H,S,D]
        # so both sides share one row order (row = (b,h,s), contiguous D) --
        # a pure axis permutation, not a value change, so this remains a
        # REAL tapped pair, not a value recomputed from scratch.
        for L in (0, 5, 11, 23):
            p = f"layers.{L}.self_attn."
            for label, nh, proj_key, norm_key, normed_key in (
                ("q_norm", n_heads, f"L{L}.q_proj", p + "q_norm.weight", f"L{L}.q_normed"),
                ("k_norm", n_kv_heads, f"L{L}.k_proj", p + "k_norm.weight", f"L{L}.k_normed"),
            ):
                proj = taps[proj_key].reshape(B, S, nh, head_dim).transpose(0, 2, 1, 3)
                after = taps[normed_key]  # already [B,H,S,D] (or [B,KVH,S,D])
                assert proj.shape == after.shape
                rows = B * nh * S
                name = f"rmsnorm_head_L{L}_{label}"
                n = write_record(f, name, 0, rows, head_dim, eps=eps,
                                 weight=w[norm_key], blocks=(proj, after))
                n_records += 1
                n_elems += n

        # -- 3. RoPE -------------------------------------------------------
        # q_normed/k_normed -> q_rope/k_rope, in the taps' native [B,H,S,D]
        # (or [B,KVH,S,D]) layout: row = (b,h,s) flattened, D contiguous --
        # exactly apply_rope_cpu's assumed layout (pos = row % seq_len).
        for L in (0, 5, 6, 11):
            theta = theta_global if is_full(L) else theta_local
            for label, nh, before_key, after_key in (
                ("q", n_heads, f"L{L}.q_normed", f"L{L}.q_rope"),
                ("k", n_kv_heads, f"L{L}.k_normed", f"L{L}.k_rope"),
            ):
                before = taps[before_key]
                after = taps[after_key]
                assert before.shape == (B, nh, S, head_dim)
                rows = B * nh * S
                name = f"rope_L{L}_{label}_full={is_full(L)}"
                n = write_record(f, name, 1, rows, head_dim, seq_len=S,
                                 theta=theta, blocks=(before, after))
                n_records += 1
                n_elems += n

        # -- 4. GeGLU elementwise stage -------------------------------------
        for L in (0, 5, 11, 23):
            gate = taps[f"L{L}.gate"]
            up = taps[f"L{L}.up"]
            after = taps[f"L{L}.geglu"]
            assert gate.shape == up.shape == after.shape
            rows = B * S
            inter = gate.shape[-1]
            name = f"geglu_L{L}"
            n = write_record(f, name, 2, rows, inter, blocks=(gate, up, after))
            n_records += 1
            n_elems += n

    print(f"wrote {OUT_PATH.relative_to(REPO)}  "
          f"({OUT_PATH.stat().st_size/1e6:.2f} MB, {n_records} records, "
          f"{n_elems} float32 values)")
    print(f"config: eps={eps} head_dim={head_dim} hidden={hidden} "
          f"heads={n_heads} kv_heads={n_kv_heads} "
          f"theta_global={theta_global} theta_local={theta_local} swp={swp}")
    print(f"full-attention layers (theta_global) for swp={swp}: "
          f"{[i for i in range(24) if is_full(i)]}")


if __name__ == "__main__":
    main()
