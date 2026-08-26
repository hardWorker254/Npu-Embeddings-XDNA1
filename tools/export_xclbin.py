# NpuEmbeddings -- M7: export a compiled design for the C++ runtime.
# SPDX-License-Identifier: Apache-2.0
#
# The C++ runtime must not compile anything. IRON has no C++ frontend
# (docs/00-overview, ground rule 3), so the design is built once here -- at BUILD
# time, where Python is allowed -- and the two artifacts XRT actually needs are
# copied out:
#
#   final.xclbin   the configured array
#   insts.bin      the instruction stream the DPU kernel replays
#
# Everything else in the JIT cache (MLIR, CDO blobs, kernel objects) is
# intermediate and the runtime never sees it.
#
# HOW THE CACHE DIRECTORY IS IDENTIFIED
# -------------------------------------
# The JIT hashes the design into ~/.npu/cache/<hash>/ and does not return the
# path. The first version of this script snapshotted mtimes around the build and
# took whatever changed -- which silently copied the SAME xclbin into all four
# design directories, because every design was already cached and nothing
# changed at all.
#
# So the directory is identified by matching what the design provably contains:
# the runtime sequence declares its three buffers as memrefs of M*K, K*N and M*N
# elements. The runtime then asserts those same sizes before dispatching, so a
# mismatch fails loudly rather than producing plausible garbage.
#
# Env: iron env WITH iron_env.ps1 dot-sourced (compiles, so it needs the NPU
#      toolchain; it also runs the design once to force the build).
# Usage:
#   python tools\export_xclbin.py

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))

import aie.iron as iron                       # noqa: E402
from aie.iron import str_to_dtype             # noqa: E402
from aie.iron.device import from_name         # noqa: E402
sys.path.insert(0, str(REPO / "experiments" / "m5-eltwise"))
sys.path.insert(0, str(REPO / "reference"))
sys.path.insert(0, str(REPO / "tools"))

from npue import gemm_b_layout, layout_hash                    # noqa: E402
from toolchain_provenance import write_toolchain_json           # noqa: E402

from gemm_pretiled import pretiled_array      # noqa: E402
from gelu_kernel import (TILE as GELU_TILE,                    # noqa: E402
                         POLY_TILES as GELU_POLY_TILES,
                         PROBE_SYMBOL as GELU_PROBE, gelu_array)
from layernorm_kernel import (COLS as LN_COLS,                 # noqa: E402
                              ROWS_PER_CALL as LN_ROWS, ln_array)
from softmax_kernel import (COLS as SM_COLS,                   # noqa: E402
                            ROWS_PER_CALL as SM_ROWS, sm_array)

CACHE = Path.home() / ".npu" / "cache"

SEQ = 64
HEADS = 12

# The four per-layer GEMMs. M is batch*SEQ: every GEMM in the encoder is over
# all tokens of all sequences at once, so batching is purely a bigger M.
#
# It is the lever that survives tasks/0024. Changing design costs ~25 us +
# 7.2 us per lock and the encoder does it 49 times per encode REGARDLESS of how
# many sequences that encode carries. So the switch bill is fixed per encode and
# batching amortises it directly -- no new kernels, no fusion, just a larger M.
#
# M % (m * 4) == 0 is required by the whole-array design (4 rows of cores), so
# with m=64 any multiple of 256 works, i.e. any batch that is a multiple of 4.
def designs_for(batch, hidden=384):
    """MiniLM's shapes at hidden=384; any hidden for the width sweep.

    The FFN is 4*hidden and QKV is 3*hidden in every BERT-family encoder, so one
    parameter generates the whole set. tasks/0027 uses this to test whether the
    NPU's disadvantage is a property of MiniLM's WIDTH rather than of the
    machine: per layer and token there are 4*hidden GELU elements against
    12*hidden^2 MACs, so the elementwise share of the work falls as 1/hidden --
    independent of implementation.
    """
    M, h = batch * SEQ, hidden
    return {
        "qkv":      dict(M=M, K=h,     N=3 * h),
        "attn_out": dict(M=M, K=h,     N=h),
        "ffn_up":   dict(M=M, K=h,     N=4 * h),
        "ffn_down": dict(M=M, K=4 * h, N=h),
    }


class WidthUnknown(Exception):
    """A candidate's width could not be established. Never a pass."""


def cache_core_columns(d):
    """Design width, read from the PLACED MLIR. RAISES if it cannot be read.

    Not from aie.mlir: that is pre-placement and contains only
    `aie.logical_tile<CoreTile>(?, ?)` with no coordinates at all, so counting
    columns there silently yields 0 for every design. The placed form is
    input_with_addresses.mlir -- the same file the trace flow has to use, for
    the same reason: it is the one that knows where things actually went.

    Tiles appear as `aie.tile(col, row)`; on npu2 row 0 is the shim and row 1
    the mem tile, so rows >= 2 are the cores.

    It raises rather than returning None or 0 because the first version DID
    return 0 -- from reading the wrong file -- and 0 compared unequal to every
    requested width, so the filter silently rejected everything and the error
    surfaced as "no cache dir", pointing at the search instead of at the check.
    A verification that can report "I could not tell" is not a verification.
    This is the second time this class of bug has landed (tasks/0022, 0024), and
    both times it failed OPEN.
    """
    m = d / "input_with_addresses.mlir"
    if not m.exists():
        raise WidthUnknown(f"{d.name}: no input_with_addresses.mlir")
    text = m.read_text(encoding="utf-8", errors="ignore")
    tiles = re.findall(r"aie\.tile\((\d+),\s*(\d+)\)", text)
    if not tiles:
        raise WidthUnknown(f"{d.name}: input_with_addresses.mlir has no "
                           f"aie.tile(col, row) -- is it really placed?")
    cols = {int(c) for c, r in tiles if int(r) >= 2}
    if not cols:
        raise WidthUnknown(f"{d.name}: placed MLIR has no core tiles (row >= 2)")
    return len(cols)


def width_matches(d, want):
    """True only if `d` is provably `want` columns wide.

    A candidate whose width cannot be established is skipped and reported, not
    accepted. The distinction matters: accepting it is exactly how a 2-column
    GELU shipped as a 1-column one.
    """
    if want is None:
        return True
    try:
        return cache_core_columns(d) == want
    except WidthUnknown:
        return False



def purge_ambiguous(match_fn, what):
    """Delete every cache dir matching `match_fn` BEFORE building.

    The fifth fail-open (research/notes/0005): the GEMM matcher pins memrefs,
    stride and column count -- and none of those distinguish a bf16 build from
    an --emulate-bfp16 build of the same shape. Both sat in the cache; mtime
    picked the bfp16 one; the full encode silently regressed to 1-cos 3.470e-03,
    which is exactly 0026's bfp16 number. The same ambiguity exists for any
    flag that changes only the kernel object (gelu stack size, emulation, ...).

    There is no readable discriminator in the artifact -- the kernel hash
    prefix differs but its expected value is not computable here. So the only
    deterministic fix is to remove the ambiguity instead of trying to resolve
    it: purge all matching candidates, build fresh, match against exactly one.
    Costs a recompile whenever candidates existed; correctness over warmth.
    """
    import shutil as _sh
    purged = 0
    for d in CACHE.iterdir():
        if not d.is_dir():
            continue
        try:
            if match_fn(d):
                _sh.rmtree(d)
                purged += 1
        except OSError:
            pass
    if purged:
        print(f"  {what}: purged {purged} ambiguous cache dir(s), building fresh")


def find_cache_by_markers(markers, what, n_cols=None):
    """Find the cache dir whose aie.mlir contains all of `markers`.

    For the elementwise and reduction designs the discriminator is the kernel
    SYMBOL -- gelu_poly_bf16, layernorm_bf16, softmax_bf16 -- which appears in
    the MLIR as the external function it calls. That is unambiguous in a way
    buffer shapes are not, and the GEMM case had to learn that the hard way.

    Markers must pin the BUFFER SIZE too, not just the symbol. Symbol plus
    width still cannot tell a batch-4 GELU from a batch-16 one -- both are
    `gelu_poly_bf16` on 1 column -- and mtime handed back the batch-16 xclbin
    for a batch-4 export. design.json still said batch 4, because this script
    WRITES the sizes from the request rather than reading them from the
    artifact, so the runtime's size assert passed and the answer came out wrong:
    1-cos 8.651e-02 against 3.430e-04. Third instance of the same class of bug
    in this file (tasks/0022, 0024) and the third time it failed open.

    `n_cols` narrows it further, and it is not optional in practice. Symbol
    alone cannot tell a 1-column GELU from a 2-column one, and the tie was
    broken by mtime -- which is WRONG whenever the JIT serves a cache hit,
    because a hit does not restamp the directory. Asking for 1 column after
    building 2 silently produced the 2-column xclbin, sizes and all: exactly
    the failure tasks/0022 hit with GEMM shapes, in a new guise. Match on what
    the artifact contains, never on when it was written.
    """
    hits, rejected = [], []
    for d in CACHE.iterdir():
        if not d.is_dir():
            continue
        if not (d / "final.xclbin").exists() or not (d / "insts.bin").exists():
            continue
        mlir = d / "aie.mlir"
        if not mlir.exists():
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if not all(m in text for m in markers):
            continue
        if not width_matches(d, n_cols):
            try:
                rejected.append(cache_core_columns(d))
            except WidthUnknown as e:
                rejected.append(f"unknown({e})")
            continue
        hits.append(d)
    if not hits:
        raise SystemExit(
            f"no cache dir for {what}; looked for {markers}"
            + (f" at {n_cols} column(s); saw widths {sorted(set(rejected))}"
               if n_cols is not None else ""))
    return max(hits, key=lambda p: p.stat().st_mtime), len(hits)


def export_eltwise(out_root, n_cols=1, batch=4, gelu_tile=1024,
                   gelu_variant="poly", hidden=384,
                   ln_variant="il4", sm_variant="poly_il4"):
    """The three non-GEMM designs the full encoder needs.

    Each is run once so the JIT builds it, then located by its kernel symbol.
    Buffer counts differ -- GELU and softmax take (in, out); LayerNorm takes
    (in, params, out) because a core tile has only 2 input DMA channels and
    gamma+beta had to be packed into one (tasks/0020).

    `n_cols` was hard-coded to 1 through M6 and M7, because these designs were
    written to prove the kernels correct and correctness does not need width.
    That left them on ONE column of eight: tasks/0024 measured GELU at 2184 us
    per call, more than any GEMM in the model despite doing no multiply at all.
    """
    specs = []

    # GELU: one 1024-element tile per call, 4 cores per column.
    n_gelu = batch * SEQ * 4 * hidden

    def _purge_elt(symbol, n_elems, what):
        def _m(d):
            m = d / "aie.mlir"
            if not m.exists():
                return False
            t = m.read_text(encoding="utf-8", errors="ignore")
            return symbol in t and f"memref<{n_elems}xbf16>" in t                 and width_matches(d, n_cols)
        purge_ambiguous(_m, what)
    X = iron.zeros(n_gelu, dtype=bfloat16, device="npu")
    Y = iron.zeros(n_gelu, dtype=bfloat16, device="npu")
    _purge_elt(GELU_PROBE if gelu_variant == "probe2"
               else GELU_POLY_TILES[gelu_tile], n_gelu, "gelu")
    gelu_array(X, Y, n_elem=n_gelu, n_cols=n_cols, use_ours=gelu_variant,
               tile=gelu_tile)
    gelu_sym = (GELU_PROBE if gelu_variant == "probe2"
                else GELU_POLY_TILES[gelu_tile])
    # The symbol differs per tile size, so it also serves as the marker.
    specs.append(("gelu", [gelu_sym,
                           f"memref<{n_gelu}xbf16>"],
                  dict(rows=n_gelu, cols=n_cols, gelu_tile=gelu_tile,
                       buffers=[n_gelu * 2, n_gelu * 2],
                       kind="eltwise", elems=n_gelu)))

    # LayerNorm: [batch*SEQ, 384] with gamma|beta as one 768-float buffer.
    ln_rows = batch * SEQ
    X = iron.zeros(ln_rows * LN_COLS, dtype=bfloat16, device="npu")
    P = iron.zeros(2 * LN_COLS, dtype=np.float32, device="npu")
    Y = iron.zeros(ln_rows * LN_COLS, dtype=bfloat16, device="npu")
    # il4: four rows interleaved (tasks/0031) -- the one-row kernel is
    # latency bound (~7,800 cycles/row against ~1,000 of issued work), and the
    # interleaved variant is bit-identical numerically. Needs the 0x2000 stack.
    ln_sym = "layernorm_il4_bf16" if ln_variant == "il4" else "layernorm_bf16"
    ln_stack = 0x2000 if ln_variant == "il4" else 0xD00
    _purge_elt(ln_sym, ln_rows * LN_COLS, "layernorm")
    ln_array(X, P, Y, rows=ln_rows, n_cols=n_cols, variant=ln_variant,
             stack=ln_stack)
    specs.append(("layernorm", [ln_sym,
                                f"memref<{ln_rows * LN_COLS}xbf16>"],
                  dict(rows=ln_rows, cols=LN_COLS, aie_cols=n_cols,
                       kind="layernorm",
                       buffers=[ln_rows * LN_COLS * 2, 2 * LN_COLS * 4,
                                ln_rows * LN_COLS * 2])))

    # softmax: [batch*12*64, 64] -- batch x heads x queries, keys on the row.
    sm_rows = batch * HEADS * SEQ
    X = iron.zeros(sm_rows * SM_COLS, dtype=bfloat16, device="npu")
    Y = iron.zeros(sm_rows * SM_COLS, dtype=bfloat16, device="npu")
    # variant="poly": exp2_poly, unblocked by the 0x2000 stack (notes/0005).
    # 4.1x more accurate than aie::exp2 on the isolated test (1.744e-02 ->
    # 4.278e-03 vs a bf16 floor of 3.225e-03). An earlier full-encode FAIL
    # blamed on this variant was actually the bfp16 cache poisoning -- the
    # purge above is what makes this switch safe to make.
    sm_sym = {"poly": "softmax_poly_bf16",
              "poly_il4": "softmax_poly_il4_bf16"}[sm_variant]
    # poly_il4 needs 0x4000: at 0x2000 the step-interleaved exp2 frame
    # overruns and the design TIMES OUT on the array (tasks/0031 -- fourth
    # bite of the stack trap, and the first one that hung instead of
    # corrupting).
    sm_stack = 0x4000 if sm_variant == "poly_il4" else 0x2000
    _purge_elt(sm_sym, sm_rows * SM_COLS, "softmax")
    sm_array(X, Y, rows=sm_rows, n_cols=n_cols, variant=sm_variant,
             stack=sm_stack)
    specs.append(("softmax", [sm_sym,
                              f"memref<{sm_rows * SM_COLS}xbf16>"],
                  dict(rows=sm_rows, cols=SM_COLS, aie_cols=n_cols,
                       kind="softmax",
                       buffers=[sm_rows * SM_COLS * 2, sm_rows * SM_COLS * 2])))

    metas = []
    for name, markers, extra in specs:
        src, n_hits = find_cache_by_markers(markers, name, n_cols=n_cols)
        dst = out_root / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("final.xclbin", "insts.bin"):
            shutil.copy(src / f, dst / f)
        meta = dict(name=name, kernel="MLIR_AIE",
                    insts_bytes=(dst / "insts.bin").stat().st_size,
                    source_cache_dir=src.name, **extra)
        (dst / "design.json").write_text(json.dumps(meta, indent=2),
                                         encoding="utf-8")
        write_toolchain_json(dst)   # T39, tasks/0106
        print(f"  {name:<10} {str(extra.get('kind')):>18}  xclbin "
              f"{(dst / 'final.xclbin').stat().st_size / 1024:6.1f} KB  insts "
              f"{meta['insts_bytes'] / 1024:5.1f} KB  "
              f"(matched {n_hits} by symbol)")
        metas.append(meta)
    return metas


def find_cache_dir(M, K, N, k=None, n=None, cols=None):
    """Locate the cache directory for this exact design, deterministically.

    The JIT hashes the design and does not hand back the path. Snapshotting
    mtimes around the build does not work: a cache hit changes nothing, and the
    first version of this script then copied the SAME xclbin into all four
    design directories without noticing.

    Instead match on what the design provably contains. The runtime sequence
    declares its three buffers as memrefs whose element counts are M*K, K*N and
    M*N, and those three together are unique across the shapes we build.
    """
    want = [f"memref<{M * K}xbf16>", f"memref<{K * N}xbf16>",
            f"memref<{M * N}xf32>"]
    # The buffer shapes alone do NOT distinguish a pre-tiled design from a
    # row-major one -- both declare the same three memrefs. What differs is B's
    # innermost DMA stride: pre-tiled reads a contiguous k*n tile, so the
    # size-k dimension strides by n; row-major strides by N. Without this the
    # search happily returns the row-major build and the runtime produces a
    # confidently wrong answer (measured: rel_fro 1.186).
    if k is not None and n is not None:
        want.append(f"<size = {k}, stride = {n}>")
    # Nor do they distinguish a 4-column build from an 8-column one: same M, K,
    # N, same memrefs, same strides, different half of the array. That only
    # stayed correct while every width was built fresh and mtime happened to
    # break the tie the right way, which is luck rather than a method. Count
    # the core columns in the MLIR instead (tasks/0024).
    hits = []
    for d in CACHE.iterdir():
        if not d.is_dir():
            continue
        if not (d / "final.xclbin").exists() or not (d / "insts.bin").exists():
            continue
        mlir = d / "aie.mlir"
        if not mlir.exists():
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if not all(w in text for w in want):
            continue
        if not width_matches(d, cols):
            continue
        hits.append(d)
    if not hits:
        raise SystemExit(f"no cache dir for {M}x{K}x{N} at {cols} column(s); "
                         f"looked for {want}")
    return max(hits, key=lambda p: p.stat().st_mtime), len(hits)


def build_one(name, shape, m, k, n, cols, emulate, out_root):
    dt_in, dt_out = str_to_dtype("bf16"), str_to_dtype("f32")
    M, K, N = shape["M"], shape["K"], shape["N"]

    want = [f"memref<{M * K}xbf16>", f"memref<{K * N}xbf16>",
            f"memref<{M * N}xf32>", f"<size = {k}, stride = {n}>"]

    def _matches(d):
        m = d / "aie.mlir"
        if not m.exists():
            return False
        t = m.read_text(encoding="utf-8", errors="ignore")
        return all(w in t for w in want) and width_matches(d, cols)

    purge_ambiguous(_matches, name)

    A = iron.zeros((M, K), dtype=dt_in, device="npu")
    B = iron.zeros((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_out, device="npu")
    A[:] = np.zeros((M, K), np.float32).astype(bfloat16)
    B[:] = np.zeros((K, N), np.float32).astype(bfloat16)
    pretiled_array(A, B, C, M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=cols,
                   dtype_in_str="bf16", dtype_out_str="f32",
                   emulate_bf16_mmul_with_bfp16=emulate,
                   # PRE-TILED, because that is how .npue stores B. The whole
                   # point of the format is that the runtime hands the mapped
                   # bytes to a DMA descriptor untouched; a row-major design
                   # would force the C++ side to de-tile at load, which is the
                   # work M4 moved offline. tasks/0007 measured pre-tiling as a
                   # throughput wash, so this costs nothing and buys a runtime
                   # that does no transformation at all.
                   pretiled=True, trace_config=None)

    src, n_hits = find_cache_dir(M, K, N, k, n, cols)
    note = f"(matched {n_hits} cache dir{'s' if n_hits != 1 else ''} by shape)"

    dst = out_root / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("final.xclbin", "insts.bin"):
        if not (src / f).exists():
            raise SystemExit(f"{name}: {src} has no {f}")
        shutil.copy(src / f, dst / f)

    b_layout = gemm_b_layout(k, n)
    meta = {
        "name": name, "M": M, "K": K, "N": N,
        "tile": {"m": m, "k": k, "n": n}, "cols": cols,
        "emulate_bfp16": emulate,
        "dtype_in": "bf16", "dtype_out": "f32", "pretiled_b": True,
        # Must match how tools/pack_npue.py laid the weights out, or the
        # runtime feeds correctly-sized bytes in the wrong order.
        "b_layout": b_layout,
        # ...and `b_layout_hash` is what lets the runtime actually CHECK that,
        # instead of assuming it. Same hash tools/npue.py stamps into every
        # tiled tensor, so the design states what it expects and the file states
        # what it is.
        #
        # This closes tasks/0022's rel_fro 1.186: pre-tiled weights fed to a
        # row-major design, right sizes, wrong order, confidently wrong answer.
        # The lesson recorded then was "a buffer-size check catches a wrong
        # size, never a wrong layout" -- and the mechanism to catch a wrong
        # layout already existed in the format and was simply never consulted.
        "b_layout_hash": layout_hash(b_layout),
        # The runtime asserts against these before dispatching, which is what
        # makes "we picked the newest directory" safe.
        "bytes_a": M * K * 2, "bytes_b": K * N * 2, "bytes_c": M * N * 4,
        "insts_bytes": (dst / "insts.bin").stat().st_size,
        "kernel": "MLIR_AIE", "source_cache_dir": src.name,
    }
    (dst / "design.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_toolchain_json(dst)   # T39, tasks/0106
    print(f"  {name:<10} {str([M, K, N]):>18}  xclbin "
          f"{(dst / 'final.xclbin').stat().st_size / 1024:6.1f} KB  insts "
          f"{meta['insts_bytes'] / 1024:5.1f} KB  {note}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runtime" / "artifacts"))
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=48)
    # 2, not 4 and not 8, and this is measured rather than reasoned.
    #
    # Changing design costs ~25 us + ~7.2 us per lock in the design (tasks/0024,
    # research/notes/0004), and the encoder changes design on all 49 of its
    # dispatches. Wider designs compute faster and switch slower, and switching
    # wins. Mean of 3 runs per width:
    #
    #     cols      1      2      4      8
    #     seq/s  52.1   52.8   46.0   34.7      (ranges 0.7, 1.3, 0.7, 0.8)
    #
    # 8 columns is 33% SLOWER end to end than 2, despite every kernel in it
    # being 1.4-1.9x faster in isolation. Note that 1 and 2 are NOT
    # distinguishable -- 0.7 apart with overlapping ranges -- so this is "<= 2",
    # not "2 is the peak". 2 is chosen over 1 only because it leaves headroom as
    # batching shifts the ratio.
    #
    # This default is correct only while switches dominate. Batching moves the
    # switch:compute ratio monotonically, so re-measure after any change to
    # batch size or fusion: the optimum walks back toward wider designs.
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--elt-cols", type=int, default=1,
                    help="AIE columns for gelu/layernorm/softmax. Was 1 "
                         "through M7 -- see tasks/0024.")
    ap.add_argument("--batch", type=int, default=4,
                    help="sequences per encode. M = batch*64; must be a "
                         "multiple of 4 so M %% 256 == 0.")
    ap.add_argument("--gelu-tile", type=int, default=1024,
                    help="elements per GELU DMA transaction. 4096 is ~4x "
                         "fewer transactions and still inside L1 -- tasks/0026.")
    ap.add_argument("--hidden", type=int, default=384,
                    help="model width. 384 is MiniLM; larger values are the "
                         "synthetic sweep of tasks/0027 (GEMMs and GELU only).")
    ap.add_argument("--gelu-variant", default="poly",
                    choices=["poly", "probe2"],
                    help="probe2 is a SPEED PROBE with a degree-2 chain. "
                         "Numerically wrong; never ship it.")
    ap.add_argument("--emulate-bfp16", action="store_true")
    ap.add_argument("--ln-variant", default="il4", choices=["base", "il4"])
    ap.add_argument("--sm-variant", default="poly_il4",
                    choices=["poly", "poly_il4"])
    ap.add_argument("--eltwise-only", action="store_true",
                    help="rebuild only gelu/layernorm/softmax into an EXISTING "
                         "export, keeping its GEMM designs and manifest "
                         "entries. The out dir must already hold a "
                         "manifest.json whose batch matches --batch.")
    args = ap.parse_args()
    if args.batch % 4:
        raise SystemExit(f"--batch {args.batch}: must be a multiple of 4, "
                         f"since the whole-array design needs M % 256 == 0")

    iron.set_current_device(from_name("npu2", n_cols=None))
    out_root = Path(args.out)
    print(f"exporting designs for the C++ runtime -> {out_root}")
    if args.eltwise_only:
        man_path = out_root / "manifest.json"
        if not man_path.exists():
            raise SystemExit(f"--eltwise-only: {man_path} does not exist")
        old = json.loads(man_path.read_text(encoding="utf-8"))
        if old.get("batch") != args.batch:
            raise SystemExit(f"--eltwise-only: existing manifest is batch "
                             f"{old.get('batch')}, requested {args.batch}")
        metas = [m for m in old["designs"] if m.get("kind") == "gemm"
                 or "M" in m]
    else:
        metas = [build_one(nm, sh, args.m, args.k, args.n, args.cols,
                           args.emulate_bfp16, out_root)
                 for nm, sh in designs_for(args.batch, args.hidden).items()]
    metas += export_eltwise(out_root, n_cols=args.elt_cols, batch=args.batch,
                            gelu_tile=args.gelu_tile,
                            gelu_variant=args.gelu_variant,
                            hidden=args.hidden,
                            ln_variant=args.ln_variant,
                            sm_variant=args.sm_variant)

    (out_root / "manifest.json").write_text(json.dumps({
        "kind": "build artifact",
        "note": "compiled by IRON at build time; the C++ runtime only loads",
        "tile": {"m": args.m, "k": args.k, "n": args.n},
        "cols": args.cols, "elt_cols": args.elt_cols, "batch": args.batch,
        "seq": SEQ, "designs": metas,
    }, indent=2), encoding="utf-8")
    # resolve() first: a relative --out is not under REPO as written, and
    # relative_to raised after every artifact had already been produced.
    man = (out_root / "manifest.json").resolve()
    try:
        man = man.relative_to(REPO)
    except ValueError:
        pass
    print(f"\nwrote {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
