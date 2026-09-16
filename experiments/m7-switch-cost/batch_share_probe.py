# NpuEmbeddings -- do DIFFERENT BATCH SIZES share one static design?
# SPDX-License-Identifier: Apache-2.0
#
# tasks/0030 proved different SHAPES share an xclbin once the loop bounds are
# runtime parameters. Batch is the same kind of variable: M enters the design
# only through `n_tiles_per_core = (M/m)*(N/n)/cores`, which is RTP[0], and
# through the runtime sequence's tap loops, which live in insts.bin.
#
# If that holds, ONE xclbin serves every shape AND every batch, and an
# embeddings server can right-size each request -- a 4-text request runs a
# 4-sequence encode instead of padding to 128 -- at zero switch cost.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage:
#   python experiments\m7-switch-cost\batch_share_probe.py

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))
CACHE = Path.home() / ".npu" / "cache"

import aie.iron as iron                              # noqa: E402
from aie.iron.device import from_name                # noqa: E402
from gemm_pretiled import pretiled_array             # noqa: E402

SEQ = 64
K, N = 384, 1152          # qkv
COLS = 8
BATCHES = [4, 16, 128]


def core_columns(d):
    import re
    m = d / "input_with_addresses.mlir"
    if not m.exists():
        return None
    tiles = re.findall(r"aie\.tile\((\d+),\s*(\d+)\)",
                       m.read_text(encoding="utf-8", errors="ignore"))
    cols = {int(c) for c, r in tiles if int(r) >= 2}
    return len(cols) if cols else None


def build(M):
    markers = [f"memref<{M * K}xbf16>", f"memref<{M * N}xf32>",
               'sym_name = "rtp_0_0"']
    # Purge first: an ambiguous cache is how five fail-open bugs happened.
    for d in list(CACHE.iterdir()):
        m = d / "aie.mlir"
        if d.is_dir() and m.exists():
            txt = m.read_text(encoding="utf-8", errors="ignore")
            if all(x in txt for x in markers):
                shutil.rmtree(d)
    A = iron.zeros((M, K), dtype=bfloat16, device="npu")
    B = iron.zeros((K, N), dtype=bfloat16, device="npu")
    C = iron.zeros(M * N, dtype=np.float32, device="npu")
    pretiled_array(A, B, C, M=M, K=K, N=N, m=64, k=64, n=48, n_aie_cols=COLS,
                   dtype_in_str="bf16", dtype_out_str="f32",
                   emulate_bf16_mmul_with_bfp16=False, pretiled=True,
                   trace_config=None, rtp=True)
    hits = [d for d in CACHE.iterdir()
            if (d / "aie.mlir").exists() and (d / "final.xclbin").exists()
            and all(x in (d / "aie.mlir").read_text(encoding="utf-8",
                                                    errors="ignore")
                    for x in markers)
            and core_columns(d) == COLS]
    if len(hits) != 1:
        raise SystemExit(f"M={M}: {len(hits)} cache candidates, expected 1")
    return hits[0]


def main() -> int:
    iron.set_current_device(from_name("npu1", n_cols=None))
    dirs = {}
    for b in BATCHES:
        M = b * SEQ
        dirs[b] = build(M)
        print(f"  batch {b:>4} (M={M:>5}) -> {dirs[b].name}  "
              f"insts {(dirs[b] / 'insts.bin').stat().st_size:>6} B")

    base = (dirs[BATCHES[0]] / "final.xclbin").read_bytes()
    ok = True
    print()
    for b in BATCHES[1:]:
        other = (dirs[b] / "final.xclbin").read_bytes()
        if len(base) != len(other):
            print(f"  batch {BATCHES[0]} vs {b}: SIZES DIFFER "
                  f"({len(base)} vs {len(other)})")
            ok = False
            continue
        diffs = sum(1 for x, y in zip(base, other) if x != y)
        verdict = "SHARED (UUID-only)" if diffs <= 80 else "NOT SHARED"
        print(f"  batch {BATCHES[0]} vs {b:>4}: {diffs:>4} differing bytes  "
              f"-- {verdict}")
        ok = ok and diffs <= 80

    print(f"\n{'CONFIRMED' if ok else 'REFUTED'} -- one xclbin "
          f"{'serves' if ok else 'does NOT serve'} every batch size")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
