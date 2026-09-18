#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Usage:
#   python tools/export_gemm_rtp_multiarch.py --arch all --batch 128 --cols 8 \
#       --out runtime/artifacts_b128il
#
#   python tools/export_gemm_rtp_multiarch.py --arch 1 --batch 128 --cols 8 \
#       --out runtime/artifacts_b128il
#
#   python tools/export_gemm_rtp_multiarch.py --arch 2 --batch 128 --cols 8 \
#       --out runtime/artifacts_b128il

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))
sys.path.insert(0, str(REPO / "tools"))


DEFAULT_SEQ = 64
STREAM_ORDER = ["qkv", "attn_out", "ffn_up", "ffn_down"]

# In the original design M tiling assumes 4 AIE rows.
AIE_ROWS = 4

DEFAULT_CACHE_ROOT = Path.home() / ".npu" / "cache"

# If architecture 2 has another device name in your toolchain, override with:
#   NPU_ARCH1_DEVICE=...
#   NPU_ARCH2_DEVICE=...
ARCH_DEVICES = {
    "1": os.environ.get("NPU_ARCH1_DEVICE", "npu1"),
    "2": os.environ.get("NPU_ARCH2_DEVICE", "npu2"),
}

ARCHES = ("1", "2")

TILE_RE = re.compile(r"aie\.tile\((\d+)\s*,\s*(\d+)\)")


@dataclass(frozen=True)
class DataPath:
    a_str: str
    acc_str: str
    a_np: object
    c_np: object
    c_marker: str
    c_bytes_out: int


def shapes_for(
    batch: int,
    hidden: int = 384,
    intermediate: int | None = None,
    gated: bool = False,
    qkv_n: int | None = None,
    seq: int = DEFAULT_SEQ,
) -> dict[str, dict[str, int]]:
    """
    Return M/K/N for the four GEMM streams.

    M is only batch * seq. The GEMM cores never see seq directly.
    """
    M = batch * seq
    h = hidden
    f = 4 * h if intermediate is None else intermediate

    return {
        "qkv": {
            "M": M,
            "K": h,
            "N": 3 * h if qkv_n is None else qkv_n,
        },
        "attn_out": {
            "M": M,
            "K": h,
            "N": h,
        },
        "ffn_up": {
            "M": M,
            "K": h,
            "N": 2 * f if gated else f,
        },
        "ffn_down": {
            "M": M,
            "K": f,
            "N": h,
        },
    }


def datapath_from_args(args: argparse.Namespace) -> DataPath:
    """
    Decide operand/accumulator/transport dtypes.

    bf16 is default.
    int8 is a different MMAC datapath, not just a different container.
    """
    if args.int8 and args.emulate_bfp16:
        raise SystemExit("--int8 and --emulate-bfp16 are both datapath choices; pick one")

    if args.int8:
        a_str = "i8"
        acc_str = "i32"
        a_np = np.int8

        if args.c_bf16:
            c_np = bfloat16
            c_marker = "bf16"
            c_bytes_out = 2
        else:
            c_np = np.int32
            c_marker = "i32"
            c_bytes_out = 4
    else:
        a_str = "bf16"
        acc_str = "f32"
        a_np = bfloat16

        if args.c_bf16:
            c_np = bfloat16
            c_marker = "bf16"
            c_bytes_out = 2
        else:
            c_np = np.float32
            c_marker = "f32"
            c_bytes_out = 4

    return DataPath(
        a_str=a_str,
        acc_str=acc_str,
        a_np=a_np,
        c_np=c_np,
        c_marker=c_marker,
        c_bytes_out=c_bytes_out,
    )


def markers_for(
    shape: dict[str, int],
    m: int,
    k: int,
    n: int,
    c_dtype: str,
    a_dtype: str,
    extra_markers: tuple[str, ...] | list[str] = (),
) -> list[re.Pattern[str]]:
    """
    Return compiled regexes that identify this exact design in the JIT cache.

    The old implementation used exact substring markers. That is brittle when
    MLIR pretty-printing changes whitespace or formatting. Regexes here are
    more tolerant while still strict enough to distinguish shapes.

    Important:
      * A/B operand dtype is part of identity.
      * C transport dtype is part of identity.
      * tile geometry marker is still based on the last two B DMA sizes,
        because that was the most stable structural marker observed.
    """
    M = shape["M"]
    K = shape["K"]
    N = shape["N"]

    raw_patterns = [
        # Runtime sequence signature: binds A/B/C by argument position.
        rf"aie\.runtime_sequence\(\s*"
        rf"%arg0\s*:\s*memref<{M * K}x{a_dtype}>\s*,\s*"
        rf"%arg1\s*:\s*memref<{K * N}x{a_dtype}>\s*,\s*"
        rf"%arg2\s*:\s*memref<{M * N}x{c_dtype}>\s*\)",

        # B tile geometry marker.
        #
        # Old form:
        #   <size = k, stride = n>
        #
        # Newer mlir-aie form:
        #   sizes = [..., k, n] strides = [...]
        #
        # We match the tail: k, n] strides = [
        rf"{k}\s*,\s*{n}\s*\]\s*strides\s*=\s*\[",

        # RTP symbol marker.
        r'sym_name\s*=\s*"rtp_0_0"',
    ]

    raw_patterns.extend(extra_markers)

    try:
        return [re.compile(p) for p in raw_patterns]
    except re.error as exc:
        raise SystemExit(f"invalid marker regex: {exc}") from exc


def all_markers_match(text: str, markers: list[re.Pattern[str]]) -> bool:
    return all(marker.search(text) for marker in markers)


def core_columns(build_dir: Path) -> int | None:
    """
    Count distinct AIE columns used by cores.

    This parses input_with_addresses.mlir and counts distinct column indices
    for tiles with row >= 2. If the file is absent or no such tiles are found,
    returns None.
    """
    mlir = build_dir / "input_with_addresses.mlir"
    if not mlir.exists():
        return None

    try:
        text = mlir.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    cols = set()
    for col_s, row_s in TILE_RE.findall(text):
        try:
            col = int(col_s)
            row = int(row_s)
        except ValueError:
            continue
        if row >= 2:
            cols.add(col)

    return len(cols) if cols else None


def purge(
    markers: list[re.Pattern[str]],
    cols: int,
    what: str,
    cache_dir: Path,
) -> int:
    """
    Remove cache entries matching this design before rebuilding.

    This avoids stale cache entries with identical shape markers but different
    datapath or architecture. It is intentionally conservative: it only purges
    entries whose core column count matches the expected cols. Entries with
    unknown core column count are left alone; they will not be selected by
    find_cache() anyway.
    """
    if not cache_dir.is_dir():
        return 0

    removed = 0

    for d in list(cache_dir.iterdir()):
        if not d.is_dir():
            continue

        mlir = d / "aie.mlir"
        if not mlir.exists():
            continue

        try:
            text = mlir.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if not all_markers_match(text, markers):
            continue

        if core_columns(d) != cols:
            continue

        shutil.rmtree(d, ignore_errors=True)
        removed += 1

    if removed:
        print(f"  {what}: purged {removed} cache candidate(s)")

    return removed


def find_cache(
    markers: list[re.Pattern[str]],
    cols: int,
    what: str,
    cache_dir: Path,
) -> Path:
    """
    Find exactly one cache entry matching the compiled markers and core cols.
    """
    if not cache_dir.is_dir():
        raise SystemExit(
            f"{what}: cache directory does not exist: {cache_dir}\n"
            f"Hint: if you use --per-arch-cache, make sure the toolchain "
            f"actually honors IRON_CACHE_DIR or remove --per-arch-cache."
        )

    hits: list[Path] = []

    for d in cache_dir.iterdir():
        if not d.is_dir():
            continue

        mlir = d / "aie.mlir"
        if not (
            mlir.exists()
            and (d / "final.xclbin").exists()
            and (d / "insts.bin").exists()
        ):
            continue

        try:
            text = mlir.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if not all_markers_match(text, markers):
            continue

        if core_columns(d) != cols:
            continue

        hits.append(d)

    if len(hits) != 1:
        raise SystemExit(
            f"{what}: {len(hits)} cache candidates after purge -- expected exactly 1\n"
            f"Cache dir: {cache_dir}\n"
            f"Hint: if building multiple architectures in the same cache, use "
            f"--per-arch-cache or --require-arch-marker/--arch-marker."
        )

    return hits[0]


def xclbin_identical_mod_uuid(
    a: bytes,
    b: bytes,
    threshold: int,
) -> tuple[bool, str]:
    """
    Original 0029 check: same size and <= threshold differing bytes.

    The difference budget is intended to cover UUID and small metadata.
    If architecture 2 adds more harmless metadata, use --identity-threshold.
    """
    if len(a) != len(b):
        return False, f"sizes differ: {len(a)} vs {len(b)}"

    diffs = sum(1 for x, y in zip(a, b) if x != y)
    return diffs <= threshold, f"{diffs} differing bytes"


def validate_positive_args(args: argparse.Namespace) -> None:
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    if args.cols <= 0:
        raise SystemExit("--cols must be positive")
    if args.hidden <= 0:
        raise SystemExit("--hidden must be positive")
    if args.seq <= 0:
        raise SystemExit("--seq must be positive")
    if args.m <= 0:
        raise SystemExit("-m must be positive")
    if args.k <= 0:
        raise SystemExit("-k must be positive")
    if args.n <= 0:
        raise SystemExit("-n must be positive")
    if args.identity_threshold < 0:
        raise SystemExit("--identity-threshold must be >= 0")

    if args.intermediate is not None and args.intermediate <= 0:
        raise SystemExit("--intermediate must be positive")

    if args.qkv_n is not None and args.qkv_n <= 0:
        raise SystemExit("--qkv-n must be positive")


def parse_tiers(args: argparse.Namespace) -> list[int]:
    if args.batches:
        try:
            tiers = [int(x) for x in args.batches.split(",") if x.strip()]
        except ValueError as exc:
            raise SystemExit(f"--batches must be comma-separated integers: {exc}") from exc
    else:
        tiers = [args.batch]

    tiers = sorted(set(tiers))

    if not tiers:
        raise SystemExit("no batch tiers specified")

    return tiers


def validate_tiers_and_seq(args: argparse.Namespace) -> list[int]:
    validate_positive_args(args)

    tiers = parse_tiers(args)

    for b in tiers:
        if b <= 0:
            raise SystemExit(f"batch tier {b}: must be positive")
        if b % 4:
            raise SystemExit(f"batch tier {b}: must be a multiple of 4")

    if max(tiers) != args.batch:
        raise SystemExit(
            f"--batch {args.batch} must be the largest tier; tiers are {tiers}"
        )

    if args.seq % 8:
        raise SystemExit(
            f"--seq {args.seq}: must be positive and a multiple of 8 "
            f"(the runtime refuses anything else)"
        )

    # M % (m * rows) == 0
    for b in tiers:
        M = b * args.seq
        if M % (args.m * AIE_ROWS):
            raise SystemExit(
                f"--seq {args.seq} x batch tier {b} gives M = {M}, "
                f"which is not a multiple of m*rows = {args.m * AIE_ROWS}. "
                f"Pick a tier or a seq whose product divides it."
            )

    if args.seq != DEFAULT_SEQ:
        print(
            f"  seq        {args.seq} (default {DEFAULT_SEQ}). "
            f"M = batch*seq, so tiers {tiers} give M {[b * args.seq for b in tiers]}."
        )
        print(
            "             Host attention is O(seq^2) and this repo has no "
            "measurement above seq 64. Treat this design's throughput as "
            "unknown until it is traced."
        )

    return tiers


def validate_geometry(args: argparse.Namespace, tiers: list[int]) -> None:
    """
    Fail early if any shape is not tileable with the requested m/k/n/cols.

    The original script relied on asserts inside pretiled_array. That is fine,
    but late: it can die after several expensive builds. This check is cheap
    and gives clearer errors.
    """
    for b in tiers:
        shapes = shapes_for(
            b,
            args.hidden,
            args.intermediate,
            args.gated_ffn,
            args.qkv_n,
            args.seq,
        )

        for name, sh in shapes.items():
            M = sh["M"]
            K = sh["K"]
            N = sh["N"]

            if M % (args.m * AIE_ROWS):
                raise SystemExit(
                    f"batch {b}, stream {name}: M={M} is not divisible by "
                    f"m*rows={args.m * AIE_ROWS}"
                )

            if K % args.k:
                raise SystemExit(
                    f"batch {b}, stream {name}: K={K} is not divisible by k={args.k}"
                )

            if N % (args.n * args.cols):
                raise SystemExit(
                    f"batch {b}, stream {name}: N={N} is not divisible by "
                    f"n*cols={args.n * args.cols}"
                )


def cache_dir_for(args: argparse.Namespace, arch: str) -> Path:
    root = Path(args.cache_root).expanduser()
    if args.per_arch_cache:
        return root / f"arch{arch}"
    return root


def extra_markers_for_arch(
    args: argparse.Namespace,
    arch: str,
    device: str,
) -> list[str]:
    """
    Optional architecture markers.

    By default we do not require an architecture string inside aie.mlir,
    because the exact MLIR spelling is toolchain-dependent. If you know your
    MLIR contains a stable target marker, use --require-arch-marker or
    --arch-marker.
    """
    extra = list(args.arch_marker or [])

    if args.require_arch_marker:
        # Accept either npu1/npu2 or the configured device name.
        extra.append(rf"\bnpu{arch}\b|\b{re.escape(device)}\b")

    return extra


def export_arch(args: argparse.Namespace, arch: str) -> int:
    device = ARCH_DEVICES.get(arch)
    if not device:
        raise SystemExit(f"unknown architecture: {arch}")

    cache_dir = cache_dir_for(args, arch)

    # If per-arch cache is requested, set the environment before importing the
    # AIE/IRON stack. This only helps if the toolchain honors IRON_CACHE_DIR.
    if args.per_arch_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["IRON_CACHE_DIR"] = str(cache_dir)

    # Heavy imports are deferred until after architecture/cache setup.
    import aie.iron as iron
    from aie.iron.device import from_name
    from gemm_pretiled import pretiled_array
    from npue import gemm_b_layout, layout_hash
    from toolchain_provenance import write_toolchain_json

    print(f"[arch {arch}] device={device} cache_dir={cache_dir}")

    iron.set_current_device(from_name(device, n_cols=None))

    tiers = validate_tiers_and_seq(args)
    validate_geometry(args, tiers)

    dp = datapath_from_args(args)
    extra_markers = extra_markers_for_arch(args, arch, device)

    dirs: dict[tuple[str, int], Path] = {}
    shapes_by_batch: dict[int, dict[str, dict[str, int]]] = {}

    for b in tiers:
        shapes_b = shapes_for(
            b,
            args.hidden,
            args.intermediate,
            args.gated_ffn,
            args.qkv_n,
            args.seq,
        )
        shapes_by_batch[b] = shapes_b

        for name in STREAM_ORDER:
            sh = shapes_b[name]
            M = sh["M"]
            K = sh["K"]
            N = sh["N"]

            mk = markers_for(
                sh,
                args.m,
                args.k,
                args.n,
                dp.c_marker,
                dp.a_str,
                extra_markers,
            )

            purge(mk, args.cols, f"{name}@b{b}", cache_dir)

            A = iron.zeros((M, K), dtype=dp.a_np)
            B = iron.zeros((K, N), dtype=dp.a_np)
            C = iron.zeros(M * N, dtype=dp.c_np)

            pretiled_array(
                A,
                B,
                C,
                M=M,
                K=K,
                N=N,
                m=args.m,
                k=args.k,
                n=args.n,
                n_aie_cols=args.cols,
                dtype_in_str=dp.a_str,
                dtype_out_str=dp.acc_str,
                emulate_bf16_mmul_with_bfp16=args.emulate_bfp16,
                pretiled=True,
                trace_config=None,
                rtp=True,
                c_bf16=args.c_bf16,
            )

            dirs[(name, b)] = find_cache(mk, args.cols, f"{name}@b{b}", cache_dir)

            print(
                f"  b{b:<4} {name:<9} {str([M, K, N]):>20} -> "
                f"{dirs[(name, b)].name}"
            )

    # All shapes/tiers must share the same static xclbin modulo UUID metadata.
    ref_key = ("qkv", max(tiers))
    base = (dirs[ref_key] / "final.xclbin").read_bytes()

    for key, d in dirs.items():
        if key == ref_key:
            continue

        ok, detail = xclbin_identical_mod_uuid(
            base,
            (d / "final.xclbin").read_bytes(),
            args.identity_threshold,
        )

        print(
            f"  identity {ref_key[0]}@b{ref_key[1]} vs "
            f"{key[0]}@b{key[1]:<4} {detail}  {'OK' if ok else 'DIVERGED'}"
        )

        if not ok:
            raise SystemExit(
                "static configurations diverged -- the streams do NOT share "
                "an xclbin, refusing to export a lying artifact\n"
                "If this is only extra harmless metadata on a new architecture, "
                "inspect the xclbins and raise --identity-threshold."
            )

    out = Path(args.out) / "gemm_rtp"
    out.mkdir(parents=True, exist_ok=True)

    # Remove stale per-stream instruction files. Do not remove arbitrary files.
    for f in out.glob("insts_*.bin"):
        try:
            f.unlink()
        except OSError:
            pass

    shutil.copy2(dirs[ref_key] / "final.xclbin", out / "final.xclbin")
    shutil.copy2(dirs[ref_key] / "insts.bin", out / "insts.bin")

    slot0_meta = {
        "file": "insts.bin",
        "op": "qkv",
        "batch": max(tiers),
        "src": dirs[ref_key].name,
        "arch": int(arch),
    }

    slot = 0
    stream_meta = []

    for b in tiers:
        for name in STREAM_ORDER:
            slot += 1
            fn = f"insts_{name}_b{b}.bin"
            shutil.copy2(dirs[(name, b)] / "insts.bin", out / fn)

            sh = shapes_by_batch[b][name]

            stream_meta.append(
                {
                    "op": name,
                    "batch": b,
                    "slot": slot,
                    "file": fn,
                    "M": sh["M"],
                    "K": sh["K"],
                    "N": sh["N"],
                    "src": dirs[(name, b)].name,
                    "arch": int(arch),
                }
            )

    biggest_batch = max(tiers)
    biggest = shapes_by_batch[biggest_batch]

    a_bytes = np.dtype(dp.a_np).itemsize
    c_bytes = dp.c_bytes_out

    b_dtype = "I8" if args.int8 else "BF16"
    b_layout = gemm_b_layout(args.k, args.n, dtype=b_dtype)

    meta = {
        "name": "gemm_rtp",
        "kind": "gemm_rtp",
        "kernel": "MLIR_AIE",

        # Architecture metadata. The runtime should read this and refuse a
        # mismatching xclbin instead of assuming a default.
        "arch": int(arch),
        "device": device,

        "M": biggest["qkv"]["M"],

        "buffers": [
            max(sh["M"] * sh["K"] * a_bytes for sh in biggest.values()),
            max(sh["K"] * sh["N"] * a_bytes for sh in biggest.values()),
            max(sh["M"] * sh["N"] * c_bytes for sh in biggest.values()),
        ],

        "c_dtype": dp.c_marker,
        "a_dtype": dp.a_str,

        # Datapath description, explicit rather than inferred.
        "int8": bool(args.int8),
        "c_bf16": bool(args.c_bf16),
        "emulate_bfp16": bool(args.emulate_bfp16),

        "b_layout_hash": layout_hash(b_layout),
        "b_layout": b_layout,

        "cols": args.cols,
        "batch": biggest_batch,
        "tiers": tiers,
        "seq": args.seq,

        "hidden": args.hidden,
        "intermediate": (
            4 * args.hidden if args.intermediate is None else args.intermediate
        ),
        "gated_ffn": args.gated_ffn,
        "qkv_n": biggest["qkv"]["N"],

        "tile": {
            "m": args.m,
            "k": args.k,
            "n": args.n,
        },

        # slot 0 is the default/largest qkv stream copied as insts.bin.
        # It may be unused by some runtimes, but recording it avoids ambiguity.
        "slot0": slot0_meta,
        "slot_count": len(stream_meta) + 1,

        "streams": stream_meta,
    }

    (out / "design.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    try:
        tc = write_toolchain_json(out) or {}
        print(
            f"  toolchain  mlir_aie {tc.get('mlir_aie_version', 'unknown')}, "
            f"peano {tc.get('peano_version', 'unknown')}, "
            f"mlir-aie HEAD {tc.get('mlir_aie_git_head', 'unknown')}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  warning: toolchain provenance failed: {exc}", file=sys.stderr)

    print(
        f"\n  wrote {out} -- ONE xclbin, {len(stream_meta)} streams "
        f"({len(STREAM_ORDER)} shapes x {len(tiers)} batch tiers)"
    )

    return 0


def build_child_argv(
    args: argparse.Namespace,
    arch: str,
    outdir: Path,
) -> list[str]:
    script = Path(__file__).resolve()
    exe = sys.executable or "python"

    cmd = [
        exe,
        str(script),
        "--arch", arch,
        "--out", str(outdir),
        "--batch", str(args.batch),
        "--cols", str(args.cols),
        "--hidden", str(args.hidden),
        "--seq", str(args.seq),
        "-m", str(args.m),
        "-k", str(args.k),
        "-n", str(args.n),
        "--identity-threshold", str(args.identity_threshold),
        "--cache-root", str(args.cache_root),
    ]

    if args.batches:
        cmd += ["--batches", args.batches]

    if args.intermediate is not None:
        cmd += ["--intermediate", str(args.intermediate)]

    if args.qkv_n is not None:
        cmd += ["--qkv-n", str(args.qkv_n)]

    if args.gated_ffn:
        cmd.append("--gated-ffn")

    if args.int8:
        cmd.append("--int8")

    if args.c_bf16:
        cmd.append("--c-bf16")

    if args.emulate_bfp16:
        cmd.append("--emulate-bfp16")

    if args.per_arch_cache:
        cmd.append("--per-arch-cache")

    if args.require_arch_marker:
        cmd.append("--require-arch-marker")

    for marker in args.arch_marker or []:
        cmd += ["--arch-marker", marker]

    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export GEMM RTP artifacts for one or more NPU architectures."
        )
    )

    ap.add_argument(
        "--arch",
        choices=["1", "2", "all"],
        default="all",
        help=(
            "Architecture to export. 'all' exports architecture 1 and 2 in "
            "separate subprocesses by default. Default: all."
        ),
    )

    ap.add_argument("--out", default=str(REPO / "runtime" / "artifacts"))

    ap.add_argument(
        "--batch",
        type=int,
        default=128,
        help="largest tier; also the buffer sizing",
    )

    ap.add_argument(
        "--batches",
        default=None,
        help=(
            "comma-separated batch tiers, e.g. 4,16,32,128. "
            "Defaults to just --batch."
        ),
    )

    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=384)

    ap.add_argument(
        "--intermediate",
        type=int,
        default=None,
        help=(
            "FFN width. Defaults to 4*hidden, which every BERT-family model "
            "here happens to satisfy; pass it explicitly for anything else."
        ),
    )

    ap.add_argument(
        "--qkv-n",
        type=int,
        default=None,
        help=(
            "width of the fused qkv operand. Defaults to 3*hidden, which "
            "holds whenever num_key_value_heads == num_attention_heads; "
            "MQA/GQA models need it stated."
        ),
    )

    ap.add_argument(
        "--gated-ffn",
        action="store_true",
        help=(
            "ffn_up emits BOTH halves of a gated FFN (N = 2*intermediate), "
            "as SwiGLU/GeGLU need, while ffn_down still takes K = intermediate."
        ),
    )

    ap.add_argument(
        "--seq",
        type=int,
        default=DEFAULT_SEQ,
        help=(
            "sequence length this design is built for. Enters only as "
            "M = batch*seq. Must be a multiple of 8."
        ),
    )

    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=48)

    ap.add_argument(
        "--int8",
        action="store_true",
        help=(
            "build the int8 MMAC datapath (i8 operands, int32 accumulator) "
            "instead of bf16. Needs an int8 container."
        ),
    )

    ap.add_argument(
        "--c-bf16",
        action="store_true",
        help=(
            "GEMM emits bf16 C (fp32 accumulate, one round at the end). "
            "Halves C transport; the runtime reads the dtype from design.json."
        ),
    )

    ap.add_argument(
        "--emulate-bfp16",
        action="store_true",
        help=(
            "RESEARCH: build the GEMM on the bfp16-emulated MMAC datapath. "
            "Not a production mode."
        ),
    )

    ap.add_argument(
        "--in-process",
        action="store_true",
        help=(
            "When --arch all, export both architectures in the same process. "
            "Default is to spawn a subprocess per architecture, which is safer "
            "against global device/toolchain state."
        ),
    )

    ap.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help=(
            "Root directory scanned for JIT cache entries. "
            f"Default: {DEFAULT_CACHE_ROOT}"
        ),
    )

    ap.add_argument(
        "--per-arch-cache",
        action="store_true",
        help=(
            "Use and export IRON_CACHE_DIR=<cache-root>/arch<N>. Useful only "
            "if the toolchain actually honors IRON_CACHE_DIR."
        ),
    )

    ap.add_argument(
        "--require-arch-marker",
        action="store_true",
        help=(
            "Require the MLIR text to contain npu<arch> or the configured "
            "device name. Use only if your toolchain emits such a marker."
        ),
    )

    ap.add_argument(
        "--arch-marker",
        action="append",
        default=[],
        help=(
            "Additional regex marker that must appear in aie.mlir. Can be "
            "passed multiple times."
        ),
    )

    ap.add_argument(
        "--identity-threshold",
        type=int,
        default=80,
        help=(
            "Maximum differing bytes allowed between final.xclbin files while "
            "still treating them as identical modulo UUID metadata. Default: 80."
        ),
    )

    args = ap.parse_args()

    # Validate common arguments early, before launching architecture exports.
    tiers = validate_tiers_and_seq(args)
    validate_geometry(args, tiers)

    if args.arch == "all":
        if args.in_process:
            if args.per_arch_cache:
                print(
                    "warning: --in-process with --per-arch-cache may not work "
                    "if the IRON cache path is captured at first import.",
                    file=sys.stderr,
                )

            for arch in ARCHES:
                child_args = copy.deepcopy(args)
                child_args.arch = arch
                child_args.out = str(Path(args.out) / f"arch{arch}")
                export_arch(child_args, arch)
        else:
            for arch in ARCHES:
                outdir = Path(args.out) / f"arch{arch}"
                cmd = build_child_argv(args, arch, outdir)

                print(
                    "[export] "
                    + " ".join(shlex.quote(str(x)) for x in cmd)
                )

                subprocess.run(cmd, check=True)

        return 0

    return export_arch(args, args.arch)


if __name__ == "__main__":
    sys.exit(main())
