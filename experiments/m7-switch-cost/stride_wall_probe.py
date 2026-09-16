# NpuEmbeddings -- which shapes hit the DMA stride wall, exactly?
#
# tasks/0027 hit `'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range` at
# hidden=1536 and recorded it as "a wall without a known door". External review
# proposed the door: the C-drain's row-block stride is m*4*N = 256*N elements
# (gemm_pretiled.py step_tiler, tile_group_repeats=(tb_n_rows, ...)), so the
# wall is per-SHAPE, not per-hidden:
#
#     N          256*N        vs 2^20 = 1,048,576
#     1536       393,216      clear   (attn_out, ffn_down @ h=1536)
#     4096     1,048,576      EXACTLY the limit (ffn_up @ h=1024 -- bge-large!)
#     4608     1,179,648      over    (qkv @ h=1536)
#     6144     1,572,864      over    (ffn_up @ h=1536)
#
# This probe builds each shape in isolation and records pass/fail, testing the
# prediction and -- at N=4096 -- whether the limit is inclusive.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage:
#   python experiments\m7-switch-cost\stride_wall_probe.py

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))

import aie.iron as iron                              # noqa: E402
from aie.iron.device import from_name                # noqa: E402
from gemm_pretiled import pretiled_array             # noqa: E402

M = 1024        # batch 16 -- smallest batch the width sweep used
CASES = [
    # (label, K, N, tile_n, prediction). tile_n=48 unless N % (48*2) != 0 --
    # N=4096 needs n=32; the C row-block stride 256*N is independent of n, so
    # the boundary test stays valid.
    ("attn_out@h1536", 6144 // 4, 1536, 48, "PASS (256*N = 393,216)"),
    ("ffn_down@h1536", 6144,      1536, 48, "PASS (256*N = 393,216)"),
    ("qkv@h1536",      1536,      4608, 48, "FAIL (256*N = 1,179,648)"),
    ("ffn_up@h1536",   1536,      6144, 48, "FAIL (256*N = 1,572,864)"),
    ("ffn_up@h1024",   1024,      4096, 32, "boundary (256*N = 1,048,576 == 2^20)"),
]


def main() -> int:
    iron.set_current_device(from_name("npu1", n_cols=None))
    print(f"stride-wall probe, M={M}, tile 64x64x48, 2 columns\n")
    results = []
    for label, K, N, tile_n, pred in CASES:
        A = iron.zeros((M, K), dtype=bfloat16, device="npu")
        B = iron.zeros((K, N), dtype=bfloat16, device="npu")
        C = iron.zeros(M * N, dtype=np.float32, device="npu")
        try:
            pretiled_array(A, B, C, M=M, K=K, N=N, m=64, k=64, n=tile_n,
                           n_aie_cols=2, dtype_in_str="bf16",
                           dtype_out_str="f32",
                           emulate_bf16_mmul_with_bfp16=False,
                           pretiled=True, trace_config=None)
            got = "BUILT"
        except Exception as e:
            msg = str(e)
            if "Stride" in msg and "exceeds" in msg:
                got = "STRIDE-FAIL"
            else:
                got = f"OTHER-FAIL ({msg.splitlines()[-1][:70]})"
        results.append((label, pred, got))
        print(f"  {label:<16} predicted {pred:<40} -> {got}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
