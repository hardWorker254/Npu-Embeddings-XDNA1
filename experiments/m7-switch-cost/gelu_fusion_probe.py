# NpuEmbeddings -- expert review section 2: GELU fused into ffn_up, variant A.
#
# The claim: GELU is pure elementwise post-processing of ffn_up's output, the
# output tile is fp32 IN L1, and the GEMM core owns the code that releases it.
# So: (1) ride the bias in as an augmented K-block -- A gets a ones column,
# B gets the bias row, so the accumulator holds A@B + bias with no third input
# (a core has 2-in/2-out) -- and (2) run the GELU polynomial on the tile before
# release. The separate GELU dispatch, its design switch, and the entire host
# round trip (convert -> sync -> DMA -> sync -> convert) disappear.
#
# Cost: K 384 -> 448 is +16.7% MACs on ffn_up. This probe measures both sides.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage:
#   python experiments\m7-switch-cost\gelu_fusion_probe.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))
sys.path.insert(0, str(REPO / "tools"))

import aie.iron as iron                              # noqa: E402
from aie.iron.device import from_name                # noqa: E402
from gemm_pretiled import pretiled_array             # noqa: E402
from npue import tile_b, to_bf16_bits                # noqa: E402

M, K_REAL, K_AUG, N = 1024, 384, 448, 1536
TM, TK, TN = 64, 64, 48


def gelu_exact(x):
    # exact erf via math.erf -- no scipy in the iron env, and this is check
    # code, not a hot path
    import math
    v = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + v(x / np.sqrt(2.0)))


def main() -> int:
    iron.set_current_device(from_name("npu1", n_cols=None))
    rng = np.random.default_rng(7)

    a_real = (rng.standard_normal((M, K_REAL)) * 0.5).astype(np.float32)
    b_real = (rng.standard_normal((K_REAL, N)) * 0.05).astype(np.float32)
    bias = (rng.standard_normal(N) * 0.1).astype(np.float32)

    # K-augmentation: one extra k-block. A's column 384 is 1.0, the rest of
    # the block is 0; B's row 384 is the bias, the rest 0.
    a_aug = np.zeros((M, K_AUG), np.float32)
    a_aug[:, :K_REAL] = a_real
    a_aug[:, K_REAL] = 1.0
    b_aug = np.zeros((K_AUG, N), np.float32)
    b_aug[:K_REAL] = b_real
    b_aug[K_REAL] = bias

    # reference in the same precision the datapath sees: bf16 inputs, fp32 acc
    a16 = a_aug.astype(bfloat16).astype(np.float32)
    b16 = b_aug.astype(bfloat16).astype(np.float32)
    want = gelu_exact(a16 @ b16)

    A = iron.zeros((M, K_AUG), dtype=bfloat16, device="npu")
    Bt = tile_b(to_bf16_bits(b_aug), TK, TN, 8, 8)
    B = iron.zeros((K_AUG, N), dtype=bfloat16, device="npu")
    C = iron.zeros(M * N, dtype=np.float32, device="npu")
    A[:] = a_aug.astype(bfloat16)
    B[:] = Bt.view(bfloat16).reshape(K_AUG, N)
    assert np.array_equal(A.numpy(), a_aug.astype(bfloat16)), "A sync"

    pretiled_array(A, B, C, M=M, K=K_AUG, N=N, m=TM, k=TK, n=TN, n_aie_cols=2,
                   dtype_in_str="bf16", dtype_out_str="f32",
                   emulate_bf16_mmul_with_bfp16=False,
                   pretiled=True, trace_config=None, epilogue="gelu")

    got = C.numpy().reshape(M, N).astype(np.float64)
    if not np.isfinite(got).all():
        print(f"FAIL -- {int((~np.isfinite(got)).sum())} non-finite outputs")
        return 1
    rel_fro = np.linalg.norm(got - want) / np.linalg.norm(want)
    worst = np.abs(got - want).max()
    print(f"  fused gelu(A@B + bias) vs exact-erf reference:")
    print(f"    rel_fro   {rel_fro:.3e}")
    print(f"    worst abs {worst:.3e}")
    ok = rel_fro < 1.5e-2
    print("PASS" if ok else "FAIL", "-- tolerance 1.5e-2 rel_fro "
          "(poly GELU alone measured 4.3e-03 on hardware, tasks/0015)")
    if not ok:
        return 1

    # export for --probe-design so the fused dispatch can be timed against
    # plain ffn_up + separate GELU
    import json
    import shutil
    from export_xclbin import find_cache_dir
    src, _ = find_cache_dir(M, K_AUG, N, TK, TN, cols=2)
    dst = REPO / "runtime" / "artifacts_fused" / "ffn_up_gelu"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("final.xclbin", "insts.bin"):
        shutil.copy(src / f, dst / f)
    (dst / "design.json").write_text(json.dumps({
        "name": "ffn_up_gelu", "kernel": "MLIR_AIE", "kind": "gemm",
        "M": M, "K": K_AUG, "N": N,
        "bytes_a": M * K_AUG * 2, "bytes_b": K_AUG * N * 2,
        "bytes_c": M * N * 4,
        "b_layout_hash": "probe-only",
        "insts_bytes": (dst / "insts.bin").stat().st_size,
        "source_cache_dir": src.name,
    }, indent=2), encoding="utf-8")
    print(f"  exported to {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
