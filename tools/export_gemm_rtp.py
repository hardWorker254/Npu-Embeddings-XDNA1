# NpuEmbeddings -- M7: export the FOUR GEMM shapes as ONE xclbin + four
# instruction streams (tasks/0032).
# SPDX-License-Identifier: Apache-2.0
#
# tasks/0030 proved the mechanism (RTP loop bounds, exact results, zero switch
# cost); tasks/0031 measured what it is worth (~2.3 ms per design switch); and
# with LayerNorm, softmax and GELU on the host (tasks/0032), the encode's NPU
# work is 24 GEMM dispatches -- so ONE static design serving all four shapes
# makes every switch disappear.
#
# The export builds each shape with gemm_pretiled(rtp=True), verifies the four
# final.xclbin files are byte-identical modulo UUID metadata (the 0029 check --
# anything beyond ~80 differing bytes is real divergence and the export
# REFUSES), and emits:
#
#   gemm_rtp/final.xclbin              the shared static configuration
#   gemm_rtp/insts.bin                 the largest tier's qkv (slot 0)
#   gemm_rtp/insts_<shape>_b<batch>.bin every (shape, batch) stream
#   gemm_rtp/design.json               per-stream metadata the C++ parser reads
#
# BATCH TIERS (0037). M enters the static design ONLY through the loop bound
# `n_tiles_per_core`, which is a runtime parameter, so a batch-4 stream and a
# batch-128 stream share the same xclbin -- measured, 67-69 differing bytes,
# the UUID footprint (experiments/m7-switch-cost/batch_share_probe.py).
# Exporting several tiers lets a server RIGHT-SIZE each request: four texts run
# a four-sequence encode instead of padding to 128, and switching tiers costs
# nothing because it is the same context.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage:
#   python tools\export_gemm_rtp.py --batch 128 --cols 8 --out runtime\artifacts_b128il

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))
sys.path.insert(0, str(REPO / "tools"))
CACHE = Path.home() / ".npu" / "cache"

import aie.iron as iron                              # noqa: E402
from aie.iron.device import from_name                # noqa: E402
from gemm_pretiled import pretiled_array             # noqa: E402
from npue import gemm_b_layout, layout_hash          # noqa: E402

SEQ = 64
STREAM_ORDER = ["qkv", "attn_out", "ffn_up", "ffn_down"]


def shapes_for(batch, hidden=384, intermediate=None, gated=False):
    # NPUE-M13 (tasks/0069): the FFN width used to be hardcoded `4 * hidden`.
    # That is true of every BERT-family model this project ships, and it is a
    # property of those checkpoints rather than of the architecture -- so it was
    # an assumption wearing a constant's clothes. A GATED FFN (SwiGLU/GeGLU)
    # breaks it twice over: `ffn_up` must emit BOTH halves, N = 2*intermediate,
    # while `ffn_down` still consumes only one, K = intermediate.
    #
    # nomic-embed-text-v1.5: hidden 768, intermediate 3072 (so 4*h happens to
    # hold), gated -> ffn_up N = 6144. Note that 6144 crosses the C-drain guard
    # threshold that tasks/0068 found half-wired; do not build this without that
    # fix in gemm_pretiled.py.
    M, h = batch * SEQ, hidden
    f = 4 * h if intermediate is None else intermediate
    return {
        "qkv":      dict(M=M, K=h, N=3 * h),
        "attn_out": dict(M=M, K=h, N=h),
        "ffn_up":   dict(M=M, K=h, N=2 * f if gated else f),
        "ffn_down": dict(M=M, K=f, N=h),
    }


def core_columns(d):
    import re
    m = d / "input_with_addresses.mlir"
    if not m.exists():
        return None
    tiles = re.findall(r"aie\.tile\((\d+),\s*(\d+)\)",
                       m.read_text(encoding="utf-8", errors="ignore"))
    cols = {int(c) for c, r in tiles if int(r) >= 2}
    return len(cols) if cols else None


def markers_for(shape, m, k, n, c_dtype="f32"):
    """Strings that identify THIS design in the JIT cache.

    The C element type is part of the identity and must be, because
    `--c-bf16` and the fp32 default differ in the cache only by it.

    MATCH THE ORDERED SIGNATURE, NOT THREE LOOSE MEMREF STRINGS. The old form
    listed `memref<M*K xbf16>`, `memref<K*N xbf16>` and `memref<M*N xf32>` and
    asked only whether each appeared SOMEWHERE in the module. With fp32 C the
    `xf32` suffix happened to keep them apart. With bf16 C it does not, and the
    collision is exact rather than theoretical (tasks/0045):

        ffn_up   [8192,  384, 1536]  -> M*K=3145728  K*N=589824  M*N=12582912
        ffn_down [8192, 1536,  384]  -> M*K=12582912 K*N=589824  M*N=3145728

    Same three numbers, all now `bf16`, so the two shapes became
    indistinguishable and `purge()` for ffn_down DELETED ffn_up's build
    mid-export. It surfaced as a FileNotFoundError on a missing final.xclbin,
    which is luck: had ffn_up been built second it would have shipped the wrong
    instruction stream.

    `aie.runtime_sequence(%arg0: ..., %arg1: ..., %arg2: ...)` binds each size
    to an ARGUMENT POSITION, so A, B and C cannot trade places whatever their
    element types are.

    TILE-GEOMETRY MARKER, mlir-aie 1.4.x FORMAT (tasks/0060, T22). Up to and
    including 1.3.4, the sequence-body `aie.dma_bd` op printed each access-
    pattern dimension as a bracket-tuple (`<size = k, stride = n>`), and the
    second marker below matched that literally against B's innermost tiled
    dimension. The 1.4.x MLIR pretty-printer replaced that with a flat
    `sizes = [...] strides = [...]` pair of arrays for `aie.dma_bd` specif-
    ically -- `<size = N, stride = M>` bracket-tuples survive ONLY in
    `aie.objectfifo`'s `dimensionsToStream` attribute, a different op, so the
    old substring silently stopped matching anything in a freshly built
    `aie.mlir` (confirmed by grepping a fresh cache dir: 0 hits for the old
    form, `markers_for` always finding 0 cache candidates).

    Confirmed directly (all four production shapes, m=64/k=64/n=48, cols=2):
    B's (`%arg1`) `aie.dma_bd` always ends its access pattern with the tile
    dims as the LAST TWO entries of `sizes`, immediately followed by the
    `strides` array --
        sizes = [.., .., 64, 48] strides = [.., .., 48, 1]
    -- exactly twice per build (the ping/pong pair), in every one of qkv,
    attn_out, ffn_up and ffn_down, and nowhere else in the file (A's and C's
    `aie.dma_bd` end their `sizes` in different values). So `f"{k}, {n}]
    strides = ["` is the direct translation of the old `<size=k, stride=n>`
    marker into the new textual form: same two numbers, same adjacency
    requirement, just spelled the way 1.4.x's printer spells it.
    """
    M, K, N = shape["M"], shape["K"], shape["N"]
    return [f"aie.runtime_sequence(%arg0: memref<{M * K}xbf16>, "
            f"%arg1: memref<{K * N}xbf16>, "
            f"%arg2: memref<{M * N}x{c_dtype}>)",
            f"{k}, {n}] strides = [",
            'sym_name = "rtp_0_0"']


def find_cache(markers, cols, what):
    hits = []
    for d in CACHE.iterdir():
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists() and (d / "final.xclbin").exists()
                and (d / "insts.bin").exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if not all(x in text for x in markers):
            continue
        if core_columns(d) != cols:
            continue
        hits.append(d)
    if len(hits) != 1:
        raise SystemExit(f"{what}: {len(hits)} cache candidates after purge -- "
                         f"expected exactly 1")
    return hits[0]


def purge(markers, cols, what):
    n = 0
    for d in list(CACHE.iterdir()):
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if all(x in text for x in markers) and core_columns(d) in (cols, None):
            shutil.rmtree(d)
            n += 1
    if n:
        print(f"  {what}: purged {n} cache candidate(s)")


def xclbin_identical_mod_uuid(a: bytes, b: bytes):
    """0029's check: identical size and <= 80 differing bytes (UUID metadata)."""
    if len(a) != len(b):
        return False, f"sizes differ: {len(a)} vs {len(b)}"
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    return diffs <= 80, f"{diffs} differing bytes"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runtime" / "artifacts"))
    ap.add_argument("--batch", type=int, default=128,
                    help="largest tier; also the buffer sizing")
    ap.add_argument("--batches", default=None,
                    help="comma-separated batch tiers, e.g. 4,16,32,128. "
                         "Defaults to just --batch.")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--intermediate", type=int, default=None,
                    help="FFN width. Defaults to 4*hidden, which every "
                         "BERT-family model here happens to satisfy; pass it "
                         "explicitly for anything else.")
    ap.add_argument("--gated-ffn", action="store_true",
                    help="ffn_up emits BOTH halves of a gated FFN "
                         "(N = 2*intermediate), as SwiGLU/GeGLU need, while "
                         "ffn_down still takes K = intermediate. tasks/0069.")
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=48)
    # NPUE-M9 (tasks/0045): narrow C to bf16 on the core, after the fp32 K
    # reduction, so the C DMA and the host readback move half the bytes.
    # tasks/0044 measured that readback at 18.8% of a MiniLM encode.
    ap.add_argument("--c-bf16", action="store_true",
                    help="GEMM emits bf16 C (fp32 accumulate, one round at "
                         "the end). Halves C transport; the runtime reads the "
                         "dtype from design.json.")
    # RESEARCH FLAG (T23). The bfp16-emulated MMAC datapath is 2.9x of array
    # GEMM time (tasks/0049) at 1-cos ~3.5e-03 -- it FAILED the 2e-03 gate in
    # 0026 and MTEB (0035) is the authority on reopening it. This flag exists
    # so that MTEB measurement can be taken; it is NOT a production mode.
    # Cache note: bfp16 and plain-bf16 builds are AMBIGUOUS in aie.mlir (same
    # buffer dtypes, same sym names) -- correctness rests entirely on
    # purge-before-build (tasks/0030, fifth fail-open), which removes every
    # matching candidate before each fresh build. Re-export the plain set
    # after using this, for the same reason.
    ap.add_argument("--emulate-bfp16", action="store_true",
                    help="RESEARCH: build the GEMM on the bfp16-emulated "
                         "MMAC datapath (T23 accuracy measurement)")
    args = ap.parse_args()
    tiers = ([int(x) for x in args.batches.split(",")] if args.batches
             else [args.batch])
    tiers = sorted(set(tiers))
    for b in tiers:
        if b % 4:
            raise SystemExit(f"batch {b}: must be a multiple of 4")
    if max(tiers) != args.batch:
        raise SystemExit(f"--batch {args.batch} must be the largest tier "
                         f"(tiers are {tiers})")

    iron.set_current_device(from_name("npu2", n_cols=None))

    # Build every (shape, tier). The identity check then covers BOTH axes:
    # if any of them diverged, the whole one-context story is false and the
    # export refuses rather than shipping an artifact that lies about it.
    dirs = {}
    for b in tiers:
        shapes_b = shapes_for(b, args.hidden, args.intermediate, args.gated_ffn)
        for name in STREAM_ORDER:
            sh = shapes_b[name]
            mk = markers_for(sh, args.m, args.k, args.n,
                             "bf16" if args.c_bf16 else "f32")
            purge(mk, args.cols, f"{name}@b{b}")
            M, K, N = sh["M"], sh["K"], sh["N"]
            A = iron.zeros((M, K), dtype=bfloat16, device="npu")
            B = iron.zeros((K, N), dtype=bfloat16, device="npu")
            C = iron.zeros(M * N,
                           dtype=bfloat16 if args.c_bf16 else np.float32,
                           device="npu")
            pretiled_array(A, B, C, M=M, K=K, N=N, m=args.m, k=args.k,
                           n=args.n, n_aie_cols=args.cols,
                           dtype_in_str="bf16", dtype_out_str="f32",
                           emulate_bf16_mmul_with_bfp16=args.emulate_bfp16,
                           pretiled=True, trace_config=None, rtp=True,
                           c_bf16=args.c_bf16)
            dirs[(name, b)] = find_cache(mk, args.cols, f"{name}@b{b}")
            print(f"  b{b:<4} {name:<9} {str([M, K, N]):>20} -> "
                  f"{dirs[(name, b)].name}")

    ref_key = ("qkv", max(tiers))
    base = (dirs[ref_key] / "final.xclbin").read_bytes()
    for key, d in dirs.items():
        if key == ref_key:
            continue
        ok, detail = xclbin_identical_mod_uuid(base, (d / "final.xclbin").read_bytes())
        print(f"  identity {ref_key[0]}@b{ref_key[1]} vs "
              f"{key[0]}@b{key[1]:<4} {detail}  {'OK' if ok else 'DIVERGED'}")
        if not ok:
            raise SystemExit(
                "static configurations diverged -- the streams do NOT share "
                "an xclbin, refusing to export a lying artifact")

    out = Path(args.out) / "gemm_rtp"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("insts_*.bin"):
        f.unlink()                        # never leave a stale tier behind
    shutil.copy(dirs[ref_key] / "final.xclbin", out / "final.xclbin")
    shutil.copy(dirs[ref_key] / "insts.bin", out / "insts.bin")

    # Slots 1..N in load order; slot 0 is insts.bin and is never bound to an
    # op, so the mapping stays explicit rather than depending on which stream
    # happened to be copied first.
    slot = 0
    stream_meta = []
    for b in tiers:
        for name in STREAM_ORDER:
            slot += 1
            fn = f"insts_{name}_b{b}.bin"
            shutil.copy(dirs[(name, b)] / "insts.bin", out / fn)
            sh = shapes_for(b, args.hidden, args.intermediate, args.gated_ffn)[name]
            stream_meta.append({"op": name, "batch": b, "slot": slot,
                                "file": fn, "M": sh["M"], "K": sh["K"],
                                "N": sh["N"],
                                "src": dirs[(name, b)].name})

    biggest = shapes_for(max(tiers), args.hidden, args.intermediate, args.gated_ffn)
    c_bytes = 2 if args.c_bf16 else 4
    b_layout = gemm_b_layout(args.k, args.n)
    meta = {
        "name": "gemm_rtp", "kind": "gemm_rtp", "kernel": "MLIR_AIE",
        "M": biggest["qkv"]["M"],        # batch inference in the runtime
        "buffers": [max(sh["M"] * sh["K"] * 2 for sh in biggest.values()),
                    max(sh["K"] * sh["N"] * 2 for sh in biggest.values()),
                    max(sh["M"] * sh["N"] * c_bytes for sh in biggest.values())],
        # The runtime must READ this, never assume it. An artifact that is
        # bf16 and says nothing looks exactly like an fp32 one to a parser
        # that defaults -- the eighth fail-open in CLAUDE.md is a literal that
        # should have been data.
        "c_dtype": "bf16" if args.c_bf16 else "f32",
        "b_layout_hash": layout_hash(b_layout),
        "b_layout": b_layout,
        "cols": args.cols, "batch": max(tiers), "tiers": tiers,
        # The sequence length these designs were compiled for. It is a
        # property of the DESIGN, not of the model: the container's
        # max_seq_len is how many position embeddings were packed (256),
        # which is a different and larger number.
        "seq": SEQ,
        # NPUE-M13 (tasks/0069, thread T31). The geometry this design was built
        # FOR, stated rather than inferred. `design_fits()` used to ask only
        # whether `hidden` appeared as some "K" in this file -- which is true of
        # any design at the same width, whatever its FFN looks like. nomic's K
        # set {768, 3072} is IDENTICAL to bge-base's while its gated ffn_up is
        # N=6144 against bge-base's 3072, so that check passes and the runtime
        # would dispatch a stream built for half the output width, silently.
        # With these three keys the match can be exact.
        "hidden": args.hidden,
        "intermediate": (4 * args.hidden if args.intermediate is None
                         else args.intermediate),
        "gated_ffn": args.gated_ffn,
        "tile": {"m": args.m, "k": args.k, "n": args.n},
        "streams": stream_meta,
    }
    (out / "design.json").write_text(json.dumps(meta, indent=2),
                                     encoding="utf-8")
    print(f"\n  wrote {out} -- ONE xclbin, {len(stream_meta)} streams "
          f"({len(STREAM_ORDER)} shapes x {len(tiers)} batch tiers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
