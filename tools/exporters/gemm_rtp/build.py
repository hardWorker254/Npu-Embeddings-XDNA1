"""Building one architecture's gemm_rtp design set, and re-invoking for --arch all.

export_arch writes <out>/artifacts_npu<arch>/gemm_rtp: one final.xclbin plus one
instruction stream per (stream, batch tier). The xclbin is loaded once at run
time and kept, which is the whole reason a design set is built per geometry
rather than reused.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

import npu_ops
from ..common.cache import find_cache, purge, xclbin_identical_mod_uuid
from ..common.consts import AIE_ROWS, ARCH_DEVICES, MAC_BY_ARCH
from ..common.paths import cache_dir_for
from ..common.validate import validate_tiers_and_seq
from .geometry import datapath_from_args, markers_for, shapes_for_stream_set

# --stream-set -> the directory name the runtime looks the set up by. The
# encoder's set keeps the historical name so every shipping artifact path and
# every documented command is unchanged.
STREAM_SET_DIRS = {"gemm_rtp": "gemm_rtp", "stt": "gemm_rtp_dec"}
from .validate import validate_geometry

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

    # The stream set decides BOTH the operand names and the shapes, and the
    # output directory name, from one value. The directory is named after the
    # set so an STT model's encoder and decoder passes -- which differ in M by
    # 192x -- cannot overwrite each other, and so the runtime can find them by
    # name the way it finds gemm_rtp.
    stream_set = getattr(args, "stream_set", None) or "gemm_rtp"
    if stream_set not in STREAM_SET_DIRS:
        raise SystemExit(f"unknown --stream-set {stream_set!r}")
    out_name = STREAM_SET_DIRS[stream_set]

    dirs: dict[tuple[str, int], Path] = {}
    shapes_by_batch: dict[int, dict[str, dict[str, int]]] = {}

    for b in tiers:
        stream_order, shapes_b = shapes_for_stream_set(
            stream_set,
            b,
            args.hidden,
            args.intermediate,
            args.gated_ffn,
            args.qkv_n,
            args.seq,
        )
        shapes_by_batch[b] = shapes_b

        for name in stream_order:
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
                n_aie_rows=args.rows,
                tb_n_rows=args.tb_rows,
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
    ref_key = (stream_order[0], max(tiers))
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

    out = Path(args.out) / out_name
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
        "op": stream_order[0],
        "batch": max(tiers),
        "src": dirs[ref_key].name,
        "arch": int(arch),
    }

    slot = 0
    stream_meta = []

    for b in tiers:
        for name in stream_order:
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
    # The B panel order is the generation's, not a constant: design.json's
    # b_layout_hash is what the runtime compares against the container's, and
    # that comparison is the ONLY thing standing between a packed file and a
    # silently misread one. Both sides have to be derived from the same device
    # or the hash agrees on a layout the hardware does not use.
    mac_s, mac_t = MAC_BY_ARCH[str(arch)]
    b_layout = gemm_b_layout(args.k, args.n, mac_s, mac_t, dtype=b_dtype)

    meta = {
        "name": out_name,
        "kind": out_name,
        "kernel": "MLIR_AIE",

        # Architecture metadata. The runtime should read this and refuse a
        # mismatching xclbin instead of assuming a default.
        "arch": int(arch),
        "device": device,

        "M": biggest[stream_order[0]]["M"],

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
        "qkv_n": biggest["qkv"]["N"] if "qkv" in biggest else None,

        "tile": {
            "m": args.m,
            "k": args.k,
            "n": args.n,
            # AIE rows per core group. Recorded because M must tile into
            # m*rows, so a design set exported at rows=1 and one exported at
            # rows=4 accept different M and the runtime must be able to tell
            # them apart rather than infer.
            "rows": args.rows,
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
        f"({len(stream_order)} shapes x {len(tiers)} batch tiers, "
        f"m={args.m} rows={args.rows})"
    )
    print(f"  b_layout {device}: mac (s={mac_s}, t={mac_t}), tile "
          f"({args.k}, {args.n}), hash {layout_hash(b_layout)[:16]}...")


    extra = npu_ops.parse_ops(getattr(args, "npu_extra_ops", ""),
                              npu_ops.EXPORTER_FLAG)
    if extra:
        # Same generation root, alongside gemm_rtp: the runtime's --npu-ops
        # resolves these directories by the op's code, which IS the directory
        # name (see tools/npu_ops.py). Built in the same invocation so the two
        # sets never drift apart by a rebuild, and only for the ops that were
        # asked for -- each one is a compile and an xclbin.
        import export_eltwise  # noqa: E402  (tools/ is on sys.path)
        export_eltwise.export_arch(
            Path(args.out), arch, args.batch, args.hidden, args.seq,
            args.elt_cols, args.gelu_tile, args.ln_variant, args.sm_variant,
            args.cache_root, args.per_arch_cache, ops=extra)

    return 0

def build_child_argv(
    args: argparse.Namespace,
    arch: str,
    outdir: Path,
) -> list[str]:
    # The child re-runs the HISTORIC entry point, not this module: --arch all
    # spawns one process per generation, and that process must go through the
    # same argparser the documented command uses.
    script = Path(__file__).resolve().parents[2] / "export_gemm_rtp.py"
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
        # --rows and --stream-set are passed ONLY when they differ from the
        # defaults, so the child of an ordinary embedder build gets byte-for-byte
        # the argv it has always got. The decoder pass needs them (rows=1, the
        # stt stream set); the encoder pass does not, and a diagnostic line that
        # changes for every existing model is a change nobody asked for.
        *([] if args.rows == AIE_ROWS
          else ["--rows", str(args.rows)]),
        *([] if (getattr(args, "stream_set", None) or "gemm_rtp") == "gemm_rtp"
          else ["--stream-set", str(args.stream_set)]),
        *([] if not args.tb_rows else ["--tb-rows", str(args.tb_rows)]),
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

    if getattr(args, "npu_extra_ops", ""):
        # The child re-parses the list with the same parser, so the set that is
        # built cannot differ from the set that was asked for.
        cmd += [
            npu_ops.EXPORTER_FLAG, args.npu_extra_ops,
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
