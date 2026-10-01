# NpuEmbeddings -- M6: a full MiniLM encode, with the NPU doing the work.
# SPDX-License-Identifier: Apache-2.0
#
#   tokens -> embeddings -> 6 encoder layers -> mean pool -> L2 normalize
#
# validated end to end against the HuggingFace-derived goldens.
#
# HOW IT IS BUILT
# ---------------
# Not as a second encoder. The M3 reference already routes every matmul through
# a swappable `self.gemm` and (since M6) the activation through `self.gelu_fn`.
# So this file supplies NPU-backed implementations of those two hooks and runs
# the *same* forward pass that check_reference.py proves correct. There is one
# encoder in this project, and it is the oracle.
#
# WHAT RUNS WHERE, AND WHY
# ------------------------
# NPU:
#   * the four projection/FFN GEMMs per layer -- qkv, attn_out, ffn_up, ffn_down.
#     94.7% of encoder FLOPs at seq 128 (docs/04-model). Validated in tasks/0012.
#   * GELU, via our own polynomial kernel (tasks/0015).
#
# Host:
#   * embedding lookup -- a gather, never a multiply. Tiling it would only hurt,
#     and .npue stores it un-tiled fp32 for exactly this reason.
#   * LayerNorm and softmax -- docs/04-model requires both in fp32. tasks/0016
#     established that fp32 IS available on the array, so these are "not written
#     yet", not "not possible".
#   * the attention GEMMs, QK^T and A.V. Their shapes are per-head [S,32]x[32,S],
#     which do not satisfy the whole-array design's tiling constraints, so the
#     NPU GEMM declines them and they fall back. 5.3% of FLOPs at seq 128, and
#     F3 says attention is not where encoder time goes.
#   * bias adds and pooling -- elementwise, and fp32 in .npue by design.
#
# The fallback is automatic and counted: every GEMM the NPU cannot express is
# tallied and reported, so "what still runs on the host" is a measured number in
# the output rather than a claim in a comment.
#
# ON SPEED: this is a CORRECTNESS milestone. Each dispatch costs ~150 us
# (tasks/0010) and this issues ~30 of them per encode with no fusion, so wall
# clock here is meaningless and is not reported. M7 is where that changes.
#
# Env: iron env WITH iron_env.ps1 dot-sourced (needs the NPU).
# Usage:
#   python reference\encode_npu.py
#   python reference\encode_npu.py --cols 8 --emulate-bfp16

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))
sys.path.insert(0, str(REPO / "experiments" / "m5-eltwise"))

import aie.iron as iron                                   # noqa: E402
from aie.iron import str_to_dtype                         # noqa: E402
from aie.iron.device import from_name                     # noqa: E402

from encoder import MiniLMReference, fp32_gemm            # noqa: E402
from gemm_pretiled import pretiled_array                  # noqa: E402
from gelu_kernel import TILE as GELU_TILE, gelu_array     # noqa: E402
from layernorm_kernel import (COLS as LN_COLS,            # noqa: E402
                              ROWS_PER_CALL as LN_ROWS, ln_array)
from softmax_kernel import (COLS as SM_COLS,              # noqa: E402
                            ROWS_PER_CALL as SM_ROWS, sm_array)
from npue import Reader                                   # noqa: E402
from safetensors_io import load                           # noqa: E402


class NpuGemm:
    """(M,K) x (K,N) on the array, falling back to numpy when it does not fit.

    The whole-array design requires M % (m*4) == 0, K % k == 0 and
    N % (n*cols) == 0, plus an L1 budget. Attention's per-head shapes fail the
    first of those, so they fall back. Both counts are reported.
    """

    def __init__(self, cols=4, emulate=False, m=64, k=64, n=48):
        self.cols, self.emulate = cols, emulate
        self.m, self.k, self.n = m, k, n
        self.npu_calls = 0
        self.cpu_calls = 0
        self.shapes_npu = {}
        self.shapes_cpu = {}
        self.npu_seconds = 0.0
        self.cpu_seconds = 0.0

    def fits(self, M, K, N):
        return (M % (self.m * 4) == 0 and K % self.k == 0
                and N % (self.n * self.cols) == 0
                and 2 * (self.m * self.k * 2 + self.k * self.n * 2
                         + self.m * self.n * 4) < 65536)

    def __call__(self, a, b):
        M, K = a.shape
        N = b.shape[1]
        key = f"{M}x{K}x{N}"
        if not self.fits(M, K, N):
            self.cpu_calls += 1
            self.shapes_cpu[key] = self.shapes_cpu.get(key, 0) + 1
            t0 = time.perf_counter()
            out = fp32_gemm(a, b)
            self.cpu_seconds += time.perf_counter() - t0
            return out

        self.npu_calls += 1
        self.shapes_npu[key] = self.shapes_npu.get(key, 0) + 1
        t0 = time.perf_counter()
        dt_in, dt_out = str_to_dtype("bf16"), str_to_dtype("f32")
        A = iron.zeros((M, K), dtype=dt_in, device="npu")
        B = iron.zeros((K, N), dtype=dt_in, device="npu")
        C = iron.zeros(M * N, dtype=dt_out, device="npu")
        a16 = np.ascontiguousarray(a, dtype=np.float32).astype(bfloat16)
        b16 = np.ascontiguousarray(b, dtype=np.float32).astype(bfloat16)
        A[:] = a16                       # __setitem__ syncs to device; see note 0003
        B[:] = b16
        pretiled_array(A, B, C, M=M, K=K, N=N, m=self.m, k=self.k, n=self.n,
                       n_aie_cols=self.cols, dtype_in_str="bf16",
                       dtype_out_str="f32",
                       emulate_bf16_mmul_with_bfp16=self.emulate,
                       pretiled=False, trace_config=None)
        out = C.numpy().reshape(M, N).astype(np.float32)
        self.npu_seconds += time.perf_counter() - t0
        return out


class NpuGelu:
    """Our polynomial GELU on the array (tasks/0015), 1024-element tiles."""

    def __init__(self, cols=1):
        self.cols = cols
        self.calls = 0
        self.cpu_calls = 0
        self.seconds = 0.0

    def __call__(self, x):
        from encoder import gelu as gelu_exact

        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        n = flat.size
        if n % (GELU_TILE * 4 * self.cols) != 0:
            self.cpu_calls += 1
            return gelu_exact(x)
        self.calls += 1
        t0 = time.perf_counter()
        X = iron.zeros(n, dtype=bfloat16, device="npu")
        Y = iron.zeros(n, dtype=bfloat16, device="npu")
        X[:] = flat.astype(bfloat16)
        gelu_array(X, Y, n_elem=n, n_cols=self.cols, use_ours="poly")
        out = Y.numpy().astype(np.float32).reshape(x.shape)
        self.seconds += time.perf_counter() - t0
        return out


class NpuLayerNorm:
    """Our LayerNorm kernel (tasks/0020) behind the reference's ln_fn hook."""

    def __init__(self, cols=1):
        self.cols = cols
        self.calls = 0
        self.cpu_calls = 0
        self.seconds = 0.0

    def __call__(self, x, gamma, beta, eps=1e-12):
        from encoder import layernorm as ln_ref

        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
        rows, cols = flat.shape
        if cols != LN_COLS or rows % (LN_ROWS * 4 * self.cols):
            self.cpu_calls += 1
            return ln_ref(x, gamma, beta, eps)
        self.calls += 1
        t0 = time.perf_counter()
        X = iron.zeros(rows * cols, dtype=bfloat16, device="npu")
        P = iron.zeros(2 * cols, dtype=np.float32, device="npu")
        Y = iron.zeros(rows * cols, dtype=bfloat16, device="npu")
        X[:] = flat.astype(bfloat16).reshape(-1)
        P[:] = np.concatenate([gamma, beta]).astype(np.float32)
        ln_array(X, P, Y, rows=rows, n_cols=self.cols)
        out = Y.numpy().astype(np.float32).reshape(x.shape)
        self.seconds += time.perf_counter() - t0
        return out


class NpuSoftmax:
    """Our row-wise softmax kernel (tasks/0021) behind the softmax_fn hook.

    The mask fill is clamped to a bf16-representable value first. HF uses
    finfo(float32).min = -3.4028e38, which exceeds bf16's largest finite
    magnitude and becomes -inf on this datapath -- docs/04-model's -inf/NaN
    landmine, arriving through the dtype rather than the formula.
    """

    MASK_BF16_SAFE = -1.0e30

    def __init__(self, cols=1):
        self.cols = cols
        self.calls = 0
        self.cpu_calls = 0
        self.seconds = 0.0

    def __call__(self, x):
        from encoder import softmax as sm_ref

        shape = x.shape
        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, shape[-1])
        rows, cols = flat.shape
        if cols != SM_COLS or rows % (SM_ROWS * 4 * self.cols):
            self.cpu_calls += 1
            return sm_ref(x)
        self.calls += 1
        t0 = time.perf_counter()
        flat = np.maximum(flat, np.float32(self.MASK_BF16_SAFE))
        X = iron.zeros(rows * cols, dtype=bfloat16, device="npu")
        Y = iron.zeros(rows * cols, dtype=bfloat16, device="npu")
        X[:] = flat.astype(bfloat16).reshape(-1)
        sm_array(X, Y, rows=rows, n_cols=self.cols)
        out = Y.numpy().astype(np.float32).reshape(shape)
        self.seconds += time.perf_counter() - t0
        return out


def build_from_npue(r, cfg, gemm, gelu_fn, ln_fn=None, softmax_fn=None):
    """An encoder whose every weight comes out of the packed .npue."""
    L, hidden = cfg["num_layers"], cfg["hidden"]
    w = {
        "embeddings.word_embeddings.weight": r.tensor("embeddings.word"),
        "embeddings.position_embeddings.weight": r.tensor("embeddings.position"),
        "embeddings.token_type_embeddings.weight": r.tensor("embeddings.token_type"),
        "embeddings.LayerNorm.weight": r.tensor("embeddings.ln.weight"),
        "embeddings.LayerNorm.bias": r.tensor("embeddings.ln.bias"),
    }
    qkv_w, qkv_b = {}, {}
    for i in range(L):
        qkv_w[i] = np.ascontiguousarray(r.tensor(f"layer.{i}.qkv").T)
        qkv_b[i] = r.tensor(f"layer.{i}.qkv.bias")
        p = f"encoder.layer.{i}."
        for src, dst in (("attn_out", "attention.output.dense"),
                         ("ffn_up", "intermediate.dense"),
                         ("ffn_down", "output.dense")):
            w[p + dst + ".weight"] = np.ascontiguousarray(r.tensor(f"layer.{i}.{src}").T)
            w[p + dst + ".bias"] = r.tensor(f"layer.{i}.{src}.bias")
        w[p + "attention.output.LayerNorm.weight"] = r.tensor(f"layer.{i}.ln1.weight")
        w[p + "attention.output.LayerNorm.bias"] = r.tensor(f"layer.{i}.ln1.bias")
        w[p + "output.LayerNorm.weight"] = r.tensor(f"layer.{i}.ln2.weight")
        w[p + "output.LayerNorm.bias"] = r.tensor(f"layer.{i}.ln2.bias")

    folded = cfg["fusions"]["qk_scale_folded_into_q"]
    return MiniLMReference(
        w, num_layers=L, num_heads=cfg["num_heads"], eps=cfg["layer_norm_eps"],
        gemm=gemm, gelu_fn=gelu_fn, ln_fn=ln_fn, softmax_fn=softmax_fn,
        qkv_w=qkv_w, qkv_b=qkv_b,
        qk_scale=1.0 if folded else math.sqrt(cfg["head_dim"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npue", default=str(REPO / "models" / "all-MiniLM-L6-v2.npue"))
    ap.add_argument("--goldens", default=str(REPO / "reference" / "goldens"))
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--emulate-bfp16", action="store_true")
    ap.add_argument("--cpu-gelu", action="store_true",
                    help="host GELU, to isolate what the GELU kernel costs")
    ap.add_argument("--cpu-eltwise", action="store_true",
                    help="host LayerNorm and softmax, to isolate their cost")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    iron.set_current_device(from_name("npu1", n_cols=None))
    g, meta = load(Path(args.goldens) / "minilm_l6_s64_boundary.safetensors")

    with Reader(args.npue) as r:
        cfg = r.config
        if cfg["source_sha256"] != meta["source_sha256"]:
            print("FAIL -- .npue and goldens are different checkpoints")
            return 1
        gemm = NpuGemm(cols=args.cols, emulate=args.emulate_bfp16)
        gelu_fn = None if args.cpu_gelu else NpuGelu()
        ln_fn = None if args.cpu_eltwise else NpuLayerNorm()
        sm_fn = None if args.cpu_eltwise else NpuSoftmax()
        ref = build_from_npue(r, cfg, gemm, gelu_fn, ln_fn, sm_fn)

        fmt = "bfp16" if args.emulate_bfp16 else "bf16"
        print(f"M6: full MiniLM-L6 encode, GEMMs on the NPU [{fmt}, "
              f"{args.cols} cols], GELU {'on host' if args.cpu_gelu else 'on NPU'}")
        print(f"  checkpoint {meta['source_sha256'][:16]}...  "
              f"weights from {Path(args.npue).name}")
        print(f"  corpus: batch {meta['batch']}, seq {meta['seq_len']}\n")

        taps = {}
        t_all = time.perf_counter()
        emb = ref.encode(g["input_ids"], g["attention_mask"],
                         g["token_type_ids"], taps=taps)
        wall = time.perf_counter() - t_all

    # -- where the work ran ---------------------------------------------------
    print("  dispatched to the NPU:")
    for k, v in sorted(gemm.shapes_npu.items()):
        print(f"    {v:>3} x GEMM {k}")
    if gelu_fn is not None:
        print(f"    {gelu_fn.calls:>3} x GELU (polynomial kernel)")
    if ln_fn is not None:
        print(f"    {ln_fn.calls:>3} x LayerNorm  ({ln_fn.cpu_calls} fell back)")
    if sm_fn is not None:
        print(f"    {sm_fn.calls:>3} x softmax    ({sm_fn.cpu_calls} fell back)")
    print("  fell back to the host:")
    for k, v in sorted(gemm.shapes_cpu.items()):
        print(f"    {v:>3} x GEMM {k}   (attention: per-head shapes do not tile)")
    print("    LayerNorm, softmax, bias adds, embedding gather, pooling")

    # -- where the TIME went --------------------------------------------------
    # Wall clock, legitimate here because it is end-to-end and host cost, never
    # a kernel claim (docs/05-measurement). The point of the breakdown is to say
    # what to fix, not to claim a result.
    n_seq = int(meta["batch"])
    npu_s = gemm.npu_seconds
    for h in (gelu_fn, ln_fn, sm_fn):
        if h is not None:
            npu_s += h.seconds
    host_s = wall - npu_s
    print(f"\n  wall clock for {n_seq} sequences at seq {meta['seq_len']}: "
          f"{wall * 1e3:.1f} ms  ->  {n_seq / wall:.1f} seq/s")
    print(f"    NPU dispatches ({gemm.npu_calls} GEMM"
          f"{'' if gelu_fn is None else f' + {gelu_fn.calls} GELU'})"
          f"{'':<6} {npu_s * 1e3:7.1f} ms  {npu_s / wall * 100:4.1f}%")
    print(f"    host (LayerNorm, softmax, attention, gather, pooling)"
          f"{'':<1} {host_s * 1e3:7.1f} ms  {host_s / wall * 100:4.1f}%")
    n_disp = gemm.npu_calls + (gelu_fn.calls if gelu_fn is not None else 0)
    print(f"    of the NPU time, ~{n_disp * 150 / 1e3:.1f} ms is the "
          f"{n_disp} x 150 us fixed dispatch cost (tasks/0010)")

    # -- correctness ----------------------------------------------------------
    hf = g["hf.out.embedding"].astype(np.float64)
    got = emb.astype(np.float64)
    cos = (got * hf).sum(1)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    sim_shift = float(np.abs(got @ got.T - hf @ hf.T).max())

    print(f"\n  {'tensor':<22} {'rel_fro vs HF golden':>21}")
    for nm in ["emb.ln"] + [f"L{i}.ln2" for i in range(cfg["num_layers"])] + \
              ["last_hidden_state"]:
        print(f"  {nm:<22} {rel(taps[nm].astype(np.float64), g['hf.' + nm].astype(np.float64)):>21.3e}")

    print(f"\n  worst 1 - cos vs HuggingFace   : {1 - cos.min():.3e}")
    print(f"  max sentence-similarity shift  : {sim_shift:.3e}")
    print(f"  embedding rel_fro              : {rel(got, hf):.3e}")

    # M3 measured the bf16 datapath end to end in simulation at 1-cos 1.27e-05.
    # Hardware carries a real GELU approximation on top, so allow 5x that and
    # still be far below anything a downstream user could resolve.
    # With LayerNorm and softmax also on the array the error budget grows: the
    # softmax kernel is limited by aie::exp2 at ~1.7e-02 per call (tasks/0021).
    # 1e-3 is still three orders below anything a consumer of an embedding
    # model can resolve.
    base = {"bf16": 1e-4, "bfp16": 5e-2}[fmt]
    TOL = base if args.cpu_eltwise else max(base, 1e-3)
    ok = (1 - cos.min()) <= TOL
    print(f"\n{'PASS' if ok else 'FAIL'} -- 1-cos {1 - cos.min():.3e} "
          f"vs {fmt} tolerance {TOL:.0e}")

    out = Path(args.out or REPO / "reference" / "goldens" /
               f"encode_npu_{fmt}{'_cpugelu' if args.cpu_gelu else ''}.json")
    out.write_text(json.dumps({
        "kind": "hardware measurement", "milestone": "M6",
        "format": fmt, "cols": args.cols,
        "gelu": "host" if args.cpu_gelu else "npu",
        "source_sha256": meta["source_sha256"],
        "npu_gemms": gemm.shapes_npu, "host_gemms": gemm.shapes_cpu,
        "npu_gemm_calls": gemm.npu_calls, "host_gemm_calls": gemm.cpu_calls,
        "npu_gelu_calls": None if gelu_fn is None else gelu_fn.calls,
        "npu_layernorm_calls": None if ln_fn is None else ln_fn.calls,
        "npu_softmax_calls": None if sm_fn is None else sm_fn.calls,
        "worst_1_minus_cos_vs_hf": float(1 - cos.min()),
        "max_similarity_shift": sim_shift,
        "embedding_rel_fro": rel(got, hf),
        "tolerance": TOL, "pass": bool(ok),
        "wall_seconds": wall, "npu_seconds": npu_s, "host_seconds": host_s,
        "sequences": n_seq, "seq_per_s": n_seq / wall,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
