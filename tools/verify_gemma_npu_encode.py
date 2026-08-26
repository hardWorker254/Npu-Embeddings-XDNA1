# NpuEmbeddings -- gate the arch=1 NPU encode against the host-only control.
# SPDX-License-Identifier: Apache-2.0
#
# tasks/0074. The NPU path (GemmaNpuEncoder) and the host-only path
# (npue::GemmaEncoder) read the SAME checkpoint through two containers packed
# by the same function, tokenize with the same table, and implement the same
# architecture. They differ in exactly two things: the four per-layer GEMMs run
# on the array in bf16 with an fp32 accumulate, and the host eltwise passes are
# AVX2 fp32 rather than scalar double.
#
# So this is a DIFFERENTIAL gate, not an absolute one. The control is already
# tied to the truth -- tasks/0064/0065 measured it at 1-cos 5.496e-13 against
# reference/encoder_gemma.py, which is itself 1.065e-07 against real
# HuggingFace -- and what has to be established here is that the array path did
# not change the answer by more than one bf16 rounding of every GEMM.
#
# WHY THIS CORPUS AND NOT THE GOLDEN FIXTURES. Thread T32: the golden fixtures
# tile ONE batch-4 corpus 32 times, so every "different" row is identical
# content and the gate is structurally blind to any wrong-ROW bug -- an
# in-place threaded swiglu_cpu() race PASSED that gate and failed a real
# 13-sentence encode at 1-cos 0.44. Every sentence here is distinct, and the
# check below asserts they are distinct as vectors before reporting agreement.
#
# Usage:
#   python tools\verify_gemma_npu_encode.py --npu out_npu.f32 --cpu out_cpu.f32
#          [--hidden 768] [--gate 2e-3]

from __future__ import annotations

import argparse
import sys

import numpy as np


def load(path, hidden):
    v = np.fromfile(path, dtype=np.float32)
    if v.size % hidden:
        raise SystemExit(f"{path}: {v.size} floats is not a multiple of "
                         f"hidden={hidden}")
    return v.reshape(-1, hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--hidden", type=int, default=768)
    # The same 2e-03 this project gates every other model's 1-cos on
    # (tasks/0035 established MTEB as the authority for accuracy DECISIONS;
    # this is the cheap pre-check that has to pass before MTEB is worth
    # running at all).
    ap.add_argument("--gate", type=float, default=2e-3)
    args = ap.parse_args()

    a = load(args.npu, args.hidden)
    b = load(args.cpu, args.hidden)
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")
    n = a.shape[0]

    # Both sides are L2-normalized by their own encoder, but normalise again
    # rather than assuming it: an un-normalised side would inflate 1-cos and
    # the failure would look like an accuracy problem instead of a bug.
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    au, bu = unit(a.astype(np.float64)), unit(b.astype(np.float64))
    cos = np.sum(au * bu, axis=1)
    one_minus = 1.0 - cos

    # A ROW-ORDER CHECK, because the metric above cannot see one. If the NPU
    # path mixed up which sequence went where, every row would still be a valid
    # embedding of SOME sentence in the corpus. Compare each NPU row against
    # every control row and require its own to be the nearest.
    sim = au @ bu.T
    best = np.argmax(sim, axis=1)
    misrouted = [(i, int(best[i])) for i in range(n) if best[i] != i]

    # And assert the corpus actually discriminates: if the sentences produced
    # near-identical vectors, "nearest is itself" would be luck. Report the
    # closest OFF-diagonal pair so the margin is visible rather than assumed.
    off = sim.copy()
    np.fill_diagonal(off, -np.inf)
    worst_off = float(np.max(off))

    print(f"  rows            {n}")
    print(f"  1-cos  mean     {one_minus.mean():.3e}")
    print(f"  1-cos  worst    {one_minus.max():.3e}  (row {int(np.argmax(one_minus))})")
    print(f"  gate            {args.gate:.1e}")
    print(f"  nearest-is-self {n - len(misrouted)}/{n}")
    print(f"  closest off-diagonal pair  cos {worst_off:.4f} "
          f"(margin {1.0 - worst_off:.4f} -- a corpus that cannot separate "
          f"its own rows cannot detect a routing bug)")

    ok = True
    if one_minus.max() > args.gate:
        print(f"\n  FAIL: worst 1-cos {one_minus.max():.3e} exceeds the gate")
        ok = False
    if misrouted:
        print(f"\n  FAIL: {len(misrouted)} row(s) match a different sentence "
              f"more closely than their own: {misrouted[:8]}")
        ok = False
    if worst_off > 0.999:
        print("\n  FAIL: the corpus rows are not distinguishable from each "
              "other, so the routing check above proves nothing")
        ok = False
    print("\n  PASS" if ok else "")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
