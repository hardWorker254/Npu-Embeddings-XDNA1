#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# One invocation writes a complete, self-describing design set per NPU
# generation. The output convention is one name everywhere:
#
#   <out>/artifacts_npu1/gemm_rtp   (arch 1, device npu1)
#   <out>/artifacts_npu2/gemm_rtp   (arch 2, device npu2)
#
# where <out> defaults to runtime/. That is the same path the README and the
# runtime's --artifacts flag name, so the tool and the documents agree.
#
# With --target <model> the model name becomes a subfolder so several models
# can share one artifacts root without overwriting each other:
#
#   <out>/<model>/artifacts_npu1/gemm_rtp
#   <out>/<model>/artifacts_npu2/gemm_rtp
#
# Usage:
#   python tools/export_gemm_rtp.py --target gte-multilingual-base --arch 1
#   python tools/export_gemm_rtp.py --arch all --batch 128 --cols 8
#   python tools/export_gemm_rtp.py --arch 1   --batch 128 --cols 8
#   python tools/export_gemm_rtp.py --arch 2   --batch 128 --cols 8

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
sys.path.insert(0, str(REPO / "tools"))


DEFAULT_SEQ = 64
STREAM_ORDER = ["qkv", "attn_out", "ffn_up", "ffn_down"]

# Fallbacks for a fully manual invocation without --target. These mirror the
# historic argparse defaults; with --target the values come from
# tools/npu_targets.json instead.
FALLBACK_BATCH = 128
FALLBACK_COLS = 8
FALLBACK_COLS_BY_ARCH = {
    "1": 4,
    "2": 8,
}
FALLBACK_HIDDEN = 384
FALLBACK_M = 32
FALLBACK_K = 32
FALLBACK_N = 48
FALLBACK_IDENTITY_THRESHOLD = 80

DEFAULT_TARGETS_FILE = Path(__file__).resolve().parent / "npu_targets.json"

TARGETS_SCHEMA = 1
KNOWN_DEFAULT_KEYS = {
    "seq", "tile_m", "tile_k", "tile_n", "identity_threshold", "c_bf16",
}
KNOWN_DATAPATHS = {"bf16", "bfp16"}

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


def arch_out_dir(base: str | Path, arch: str, target: str | None = None) -> Path:
    """
    The one output convention: one directory per generation, each holding a
    `gemm_rtp` design set. `base` is the artifacts root (default runtime/).

    Without `target` this is the historic `<base>/artifacts_npu<arch>`. With a
    `--target` model the model name is inserted, so
    `<base>/<model>/artifacts_npu<arch>` is the per-model design set.

    The function is idempotent: if `base` already IS the generation root
    (`.../artifacts_npu<arch>`), it is returned unchanged. That keeps the
    subprocess path correct, where the parent hands the resolved set-dir to
    the child as `--out`.
    """
    base = Path(base)
    if base.name == f"artifacts_npu{arch}":
        return base
    if target:
        return base / target / f"artifacts_npu{arch}"
    return base / f"artifacts_npu{arch}"


def set_out_dir(args: argparse.Namespace, arch: str) -> Path:
    """
    Single source of truth for where an architecture's design set is written.

    Both `--arch all` branches (in-process and subprocess) and the single-arch
    path go through here, so the layout cannot drift between them.
    """
    return arch_out_dir(args.out, arch, getattr(args, "target", None))


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

    if args.npu_eltwise:
        # Same generation root, alongside gemm_rtp: the runtime's
        # --npu-eltwise flag resolves these by name. Build them in the same
        # invocation so the two sets never drift apart by a rebuild.
        import export_eltwise  # noqa: E402  (tools/ is on sys.path)
        export_eltwise.export_arch(
            Path(args.out), arch, args.batch, args.hidden, args.seq,
            args.elt_cols, args.gelu_tile, args.ln_variant, args.sm_variant,
            args.cache_root, args.per_arch_cache)

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

    if args.with_eltwise:
        cmd += [
            "--npu-eltwise",
            "--elt-cols", str(args.elt_cols),
            "--gelu-tile", str(args.gelu_tile),
            "--ln-variant", args.ln_variant,
            "--sm-variant", args.sm_variant,
        ]

    if args.require_arch_marker:
        cmd.append("--require-arch-marker")

    for marker in args.arch_marker or []:
        cmd += ["--arch-marker", marker]

    return cmd


def _require_int(
    obj: dict,
    key: str,
    ctx: str,
    minimum: int = 1,
    allow_none: bool = False,
) -> int | None:
    if key not in obj:
        raise SystemExit(f"{ctx}: missing '{key}'")
    value = obj[key]
    if value is None and allow_none:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SystemExit(
            f"{ctx}: '{key}' must be an integer >= {minimum}, got {value!r}"
        )
    return value


def load_targets(path: str | Path) -> dict:
    """
    Read and validate tools/npu_targets.json.

    Validation is structural only: schema version, known keys and value types.
    Parity with the C++ catalog in runtime/src/common/hub.cpp is a separate
    check (tools/verify_targets.py, S7).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"targets file not found: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{p}: invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"{p}: top level must be an object")

    schema = data.get("schema")
    if schema != TARGETS_SCHEMA:
        raise SystemExit(
            f"{p}: unsupported schema {schema!r}; expected {TARGETS_SCHEMA}"
        )

    defaults = data.get("defaults")
    arches = data.get("arches")
    models = data.get("models")
    kinds = data.get("kinds")

    if not isinstance(defaults, dict):
        raise SystemExit(f"{p}: missing or invalid 'defaults' object")
    if not isinstance(arches, dict) or not arches:
        raise SystemExit(f"{p}: missing or empty 'arches' object")
    if not isinstance(models, dict) or not models:
        raise SystemExit(f"{p}: missing or empty 'models' object")
    if kinds is not None and not isinstance(kinds, dict):
        raise SystemExit(f"{p}: 'kinds' must be an object")

    unknown_defaults = set(defaults) - KNOWN_DEFAULT_KEYS
    if unknown_defaults:
        raise SystemExit(
            f"{p}: unknown defaults keys: {sorted(unknown_defaults)}"
        )
    for key in ("seq", "tile_m", "tile_k", "tile_n", "identity_threshold"):
        _require_int(defaults, key, f"{p}: defaults")
    if "c_bf16" in defaults and not isinstance(defaults["c_bf16"], bool):
        raise SystemExit(f"{p}: defaults.c_bf16 must be a boolean")

    for arch, spec in arches.items():
        ctx = f"{p}: arches[{arch!r}]"
        if not isinstance(spec, dict):
            raise SystemExit(f"{ctx}: must be an object")
        if not str(arch).isdigit():
            raise SystemExit(f"{ctx}: architecture key must be numeric")
        _require_int(spec, "cols", ctx)
        _require_int(spec, "batch", ctx)
        if "batches" in spec:
            batches = spec["batches"]
            if (
                not isinstance(batches, list)
                or not batches
                or any(
                    not isinstance(b, int) or isinstance(b, bool) or b <= 0
                    for b in batches
                )
            ):
                raise SystemExit(
                    f"{ctx}: 'batches' must be a non-empty list of "
                    f"positive integers"
                )
        _require_int(spec, "max_batch", ctx, allow_none=True)

    for name, spec in models.items():
        ctx = f"{p}: models[{name!r}]"
        if not isinstance(spec, dict):
            raise SystemExit(f"{ctx}: must be an object")
        _require_int(spec, "hidden", ctx)
        _require_int(spec, "intermediate", ctx)
        if not isinstance(spec.get("gated_ffn"), bool):
            raise SystemExit(f"{ctx}: 'gated_ffn' must be a boolean")
        _require_int(spec, "qkv_n", ctx, allow_none=True)
        datapath = spec.get("datapath")
        if datapath is not None and datapath not in KNOWN_DATAPATHS:
            raise SystemExit(
                f"{ctx}: 'datapath' must be one of {sorted(KNOWN_DATAPATHS)}"
            )
        overrides = spec.get("overrides")
        if overrides is not None:
            if not isinstance(overrides, dict):
                raise SystemExit(f"{ctx}: 'overrides' must be an object")
            unknown = set(overrides) - KNOWN_DEFAULT_KEYS
            if unknown:
                raise SystemExit(
                    f"{ctx}: unknown overrides keys: {sorted(unknown)}"
                )
            for key, value in overrides.items():
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise SystemExit(
                        f"{ctx}: overrides[{key!r}] must be a non-negative "
                        f"integer"
                    )

    return data


def _pick(*candidates: object) -> object:
    """First candidate that is not None (CLI beats overrides beats arch/default)."""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def apply_datapath(
    ns: argparse.Namespace,
    datapath: str | None,
    defaults: dict | None = None,
) -> argparse.Namespace:
    """
    Derive the datapath flags from a target's `datapath` field.

    Mapping: 'bfp16' -> --emulate-bfp16 --c-bf16; 'bf16' -> --c-bf16 only.
    Flags explicitly passed on the command line are never overwritten.
    """
    default_c = None if defaults is None else defaults.get("c_bf16")

    if datapath == "bfp16":
        if ns.emulate_bfp16 is None:
            ns.emulate_bfp16 = True
        if ns.c_bf16 is None:
            ns.c_bf16 = True if default_c is None else bool(default_c)
    elif datapath == "bf16":
        if ns.c_bf16 is None:
            ns.c_bf16 = True if default_c is None else bool(default_c)
    elif datapath is not None:
        raise SystemExit(
            f"unknown datapath {datapath!r}; expected one of "
            f"{sorted(KNOWN_DATAPATHS)}"
        )

    if ns.c_bf16 is None:
        ns.c_bf16 = bool(default_c)

    return ns


@dataclass
class ResolvedArch:
    """One architecture with every compile argument resolved to a value."""

    arch: str
    args: argparse.Namespace


def resolve_args(args: argparse.Namespace) -> list[ResolvedArch]:
    """
    Resolve concrete compile arguments per architecture.

    Priority for every field: explicit CLI value, then model.overrides, then
    arches[arch], then defaults. Without --target the historic fallbacks are
    used, so a fully manual invocation is unchanged.
    """
    arches = list(ARCHES) if args.arch == "all" else [args.arch]

    targets: dict | None = None
    model_spec: dict | None = None
    if args.target:
        targets = load_targets(args.targets_file)
        model_spec = targets["models"].get(args.target)
        if model_spec is None:
            known = ", ".join(sorted(targets["models"]))
            raise SystemExit(
                f"--target {args.target!r} is not defined in "
                f"{args.targets_file}\nKnown targets: {known}"
            )

    resolved: list[ResolvedArch] = []

    for arch in arches:
        if arch not in ARCH_DEVICES:
            raise SystemExit(f"unknown architecture: {arch}")

        ns = copy.deepcopy(args)
        ns.arch = arch

        if targets is None:
            ns.batch = FALLBACK_BATCH if args.batch is None else args.batch
            ns.cols = FALLBACK_COLS_BY_ARCH.get(arch, FALLBACK_COLS) if args.cols is None else args.cols
            ns.hidden = FALLBACK_HIDDEN if args.hidden is None else args.hidden
            ns.seq = DEFAULT_SEQ if args.seq is None else args.seq
            ns.m = FALLBACK_M if args.m is None else args.m
            ns.k = FALLBACK_K if args.k is None else args.k
            ns.n = FALLBACK_N if args.n is None else args.n
            ns.identity_threshold = (
                FALLBACK_IDENTITY_THRESHOLD
                if args.identity_threshold is None
                else args.identity_threshold
            )
            ns.gated_ffn = bool(args.gated_ffn)
            ns.int8 = bool(args.int8)
            ns.c_bf16 = bool(args.c_bf16)
            ns.emulate_bfp16 = bool(args.emulate_bfp16)
            resolved.append(ResolvedArch(arch, ns))
            continue

        assert targets is not None and model_spec is not None
        defaults = targets["defaults"]
        arch_spec = targets["arches"].get(arch)
        if arch_spec is None:
            raise SystemExit(
                f"--target {args.target!r}: architecture {arch} is not defined "
                f"in {args.targets_file}"
            )
        overrides = model_spec.get("overrides") or {}

        ns.batch = _pick(
            args.batch, overrides.get("batch"), arch_spec.get("batch"),
            FALLBACK_BATCH,
        )
        batches = _pick(
            args.batches, overrides.get("batches"), arch_spec.get("batches"),
        )
        if isinstance(batches, (list, tuple)):
            batches = ",".join(str(int(b)) for b in batches)
        ns.batches = batches
        ns.cols = _pick(
            args.cols, overrides.get("cols"), arch_spec.get("cols"),
            FALLBACK_COLS,
        )
        ns.hidden = _pick(
            args.hidden, overrides.get("hidden"), model_spec.get("hidden"),
            defaults.get("hidden"), FALLBACK_HIDDEN,
        )
        ns.intermediate = _pick(
            args.intermediate, overrides.get("intermediate"),
            model_spec.get("intermediate"),
        )
        ns.qkv_n = _pick(
            args.qkv_n, overrides.get("qkv_n"), model_spec.get("qkv_n"),
        )
        ns.gated_ffn = (
            args.gated_ffn
            if args.gated_ffn is not None
            else bool(model_spec.get("gated_ffn"))
        )
        ns.seq = _pick(
            args.seq, overrides.get("seq"), defaults.get("seq"), DEFAULT_SEQ,
        )
        ns.m = _pick(
            args.m, overrides.get("tile_m"), defaults.get("tile_m"), FALLBACK_M,
        )
        ns.k = _pick(
            args.k, overrides.get("tile_k"), defaults.get("tile_k"), FALLBACK_K,
        )
        ns.n = _pick(
            args.n, overrides.get("tile_n"), defaults.get("tile_n"), FALLBACK_N,
        )
        ns.identity_threshold = _pick(
            args.identity_threshold, overrides.get("identity_threshold"),
            defaults.get("identity_threshold"), FALLBACK_IDENTITY_THRESHOLD,
        )

        # int8 is never implied by a datapath; only an explicit flag sets it.
        ns.int8 = bool(args.int8)
        ns.c_bf16 = args.c_bf16
        ns.emulate_bfp16 = args.emulate_bfp16
        apply_datapath(ns, model_spec.get("datapath"), defaults)

        tiers = parse_tiers(ns)
        max_batch = arch_spec.get("max_batch")
        if max_batch is not None:
            offenders = sorted(
                {t for t in tiers if t > max_batch}
                | ({ns.batch} if ns.batch > max_batch else set())
            )
            if offenders:
                raise SystemExit(
                    f"--target {args.target} --arch {arch}: batch tier(s) "
                    f"{offenders} exceed arches[{arch}].max_batch={max_batch} "
                    f"(device {arch_spec.get('device')}).\n"
                    f"Lower --batch or --batches, or export with --arch 2."
                )

        resolved.append(ResolvedArch(arch, ns))

    return resolved


def print_targets(targets: dict) -> None:
    print(f"targets file (schema {targets['schema']})")
    print("arches:")
    for arch, spec in sorted(targets["arches"].items()):
        print(
            f"  {arch}: device={spec.get('device')} cols={spec.get('cols')} "
            f"batch={spec.get('batch')} batches={spec.get('batches')} "
            f"max_batch={spec.get('max_batch')}"
        )
    print("models:")
    for name in sorted(targets["models"]):
        spec = targets["models"][name]
        qkv = spec.get("qkv_n")
        print(
            f"  {name}: hidden={spec.get('hidden')} "
            f"intermediate={spec.get('intermediate')} "
            f"gated_ffn={spec.get('gated_ffn')} "
            f"qkv_n={'3*hidden' if qkv is None else qkv} "
            f"datapath={spec.get('datapath')}"
        )


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

    ap.add_argument(
        "--out",
        default=str(REPO / "runtime"),
        help=(
            "artifacts root. Each generation is written to "
            "<out>/artifacts_npu<N>/gemm_rtp, or with --target to "
            "<out>/<model>/artifacts_npu<N>/gemm_rtp. Default: runtime/"
        ),
    )

    ap.add_argument(
        "--target",
        default=None,
        help=(
            "Named model in the targets file. Selects the model geometry and "
            "the per-arch compile policy. Explicit CLI arguments always win "
            "over the config."
        ),
    )

    ap.add_argument(
        "--targets-file",
        default=str(DEFAULT_TARGETS_FILE),
        help=(
            "JSON file of per-model targets. Default: "
            f"{DEFAULT_TARGETS_FILE}"
        ),
    )

    ap.add_argument(
        "--list-targets",
        action="store_true",
        help="Print the known targets and architectures, then exit.",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve and print the compile command(s) without invoking the "
            "toolchain."
        ),
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=None,
        help=(
            "largest tier; also the buffer sizing. Without --target defaults "
            f"to {FALLBACK_BATCH}."
        ),
    )

    ap.add_argument(
        "--batches",
        default=None,
        help=(
            "comma-separated batch tiers, e.g. 4,16,32,128. "
            "Defaults to just --batch."
        ),
    )

    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)

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
        default=None,
        help=(
            "ffn_up emits BOTH halves of a gated FFN (N = 2*intermediate), "
            "as SwiGLU/GeGLU need, while ffn_down still takes K = intermediate."
        ),
    )

    ap.add_argument(
        "--seq",
        type=int,
        default=None,
        help=(
            "sequence length this design is built for. Enters only as "
            f"M = batch*seq. Must be a multiple of 8. Default: {DEFAULT_SEQ}."
        ),
    )

    ap.add_argument("-m", type=int, default=None)
    ap.add_argument("-k", type=int, default=None)
    ap.add_argument("-n", type=int, default=None)

    ap.add_argument(
        "--int8",
        action="store_true",
        default=None,
        help=(
            "build the int8 MMAC datapath (i8 operands, int32 accumulator) "
            "instead of bf16. Needs an int8 container."
        ),
    )

    ap.add_argument(
        "--c-bf16",
        action="store_true",
        default=None,
        help=(
            "GEMM emits bf16 C (fp32 accumulate, one round at the end). "
            "Halves C transport; the runtime reads the dtype from design.json."
        ),
    )

    ap.add_argument(
        "--emulate-bfp16",
        action="store_true",
        default=None,
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
        "--npu-eltwise",
        action="store_true",
        help=(
            "also build the gelu/layernorm/softmax design directories for the "
            "same generation, alongside gemm_rtp. The runtime only uses them "
            "when --npu-eltwise is given; the host path stays the default."
        ),
    )

    ap.add_argument(
        "--elt-cols",
        type=int,
        default=1,
        help=(
            "AIE columns for the eltwise designs built by --npu-eltwise. "
            "LayerNorm and softmax refuse above 2 (see tools/export_eltwise.py)."
        ),
    )

    ap.add_argument(
        "--gelu-tile",
        type=int,
        default=1024,
        choices=[1024, 4096],
        help="elements per GELU DMA transaction for --npu-eltwise.",
    )

    ap.add_argument(
        "--ln-variant",
        default="il4",
        choices=["base", "il4", "rne", "il4_rne"],
        help="LayerNorm kernel variant for --npu-eltwise.",
    )

    ap.add_argument(
        "--sm-variant",
        default="poly_il4",
        choices=["lib", "poly", "poly_il4", "poly_rne", "poly_il4_rne"],
        help="softmax kernel variant for --npu-eltwise.",
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
        default=None,
        help=(
            "Maximum differing bytes allowed between final.xclbin files while "
            "still treating them as identical modulo UUID metadata. "
            f"Without --target defaults to {FALLBACK_IDENTITY_THRESHOLD}."
        ),
    )

    args = ap.parse_args()

    if args.list_targets:
        print_targets(load_targets(args.targets_file))
        return 0

    if args.arch == "all":
        # Two generations compiled in one run MUST NOT share a JIT cache: with
        # no architecture marker in the MLIR (the default) their shape markers
        # are identical, so find_cache() would see two candidates and refuse.
        # Per-arch caches are the existing mechanism for that separation, so
        # turn them on for the multi-arch case unless the caller already gave
        # an architecture marker. Applied before resolve so each per-arch copy
        # inherits it.
        if (not args.per_arch_cache and not args.require_arch_marker
                and not args.arch_marker):
            print(
                "[export] --arch all: enabling --per-arch-cache so the two "
                "generations do not collide in one JIT cache",
                file=sys.stderr,
            )
            args.per_arch_cache = True

    resolved = resolve_args(args)

    # Validate each architecture against its own resolved values: the npu1
    # batch ceiling must never be compared against npu2 numbers.
    for ra in resolved:
        tiers = validate_tiers_and_seq(ra.args)
        validate_geometry(ra.args, tiers)

    if args.dry_run:
        for ra in resolved:
            outdir = set_out_dir(ra.args, ra.arch)
            cmd = build_child_argv(ra.args, ra.arch, outdir)
            print("[dry-run] " + " ".join(shlex.quote(str(x)) for x in cmd))
        return 0

    if args.arch == "all":
        if args.in_process:
            if args.per_arch_cache:
                print(
                    "warning: --in-process with --per-arch-cache may not work "
                    "if the IRON cache path is captured at first import.",
                    file=sys.stderr,
                )

            for ra in resolved:
                ra.args.out = str(set_out_dir(ra.args, ra.arch))
                export_arch(ra.args, ra.arch)
        else:
            for ra in resolved:
                outdir = set_out_dir(ra.args, ra.arch)
                cmd = build_child_argv(ra.args, ra.arch, outdir)

                print(
                    "[export] "
                    + " ".join(shlex.quote(str(x)) for x in cmd)
                )

                subprocess.run(cmd, check=True)

        return 0

    ra = resolved[0]
    ra.args.out = str(set_out_dir(ra.args, ra.arch))
    return export_arch(ra.args, ra.arch)


if __name__ == "__main__":
    sys.exit(main())
