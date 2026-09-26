"""Argument parsing and the top-level flow of export_eltwise.py.

Usage (the module this replaces was tools/export_eltwise.py, still a working
shim onto this package):
  python tools/export_eltwise.py --arch all --batch 128 --hidden 384
  python tools/export_eltwise.py --arch 1   --batch 128
"""

import argparse
import subprocess
from pathlib import Path

import npu_ops
from ..common.consts import ARCHES, DEFAULT_CACHE_ROOT, DEFAULT_SEQ, REPO
from .build import arch_out_dir, build_child_argv, export_arch
from .consts import GELU_TILE, MAX_LN_SM_COLS

# Re-exported so tools/exporters/gemm_rtp/build.py can build the eltwise
# siblings in the same invocation without importing a private module.
__all__ = ["export_arch", "main"]

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the gelu/layernorm/softmax design directories for "
                    "one or more NPU generations.")
    ap.add_argument(
        "--extra-ops",
        default="gelu,layn,softm",
        help=(
            "which elementwise designs to build: any comma-separated subset of "
            f"[{npu_ops.CODES}] (layn = LayerNorm, softm = softmax). Each one "
            "is a compile and an xclbin, and each is one extra hw_context at "
            f"run time, so build only what the runtime's {npu_ops.RUNTIME_FLAG} "
            "will ask for. Default: all three, which is what this tool has "
            "always built."
        ),
    )
    ap.add_argument("--arch", choices=["1", "2", "all"], default="all")
    ap.add_argument("--out", default=str(REPO / "runtime"),
                    help="artifacts root; each generation is written to "
                         "<out>/artifacts_npu<N>/{gelu,layernorm,softmax}. "
                         "Default: runtime/")
    ap.add_argument("--batch", type=int, default=128,
                    help="largest batch tier; also the eltwise buffer sizing")
    ap.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--elt-cols", type=int, default=1,
                    help="AIE columns for the eltwise designs. LayerNorm and "
                         f"softmax refuse above {MAX_LN_SM_COLS}.")
    ap.add_argument("--gelu-tile", type=int, default=GELU_TILE,
                    choices=[1024, 4096])
    ap.add_argument("--ln-variant", default="il4",
                    choices=["base", "il4", "rne", "il4_rne"])
    ap.add_argument("--sm-variant", default="poly_il4",
                    choices=["lib", "poly", "poly_il4", "poly_rne",
                             "poly_il4_rne"])
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--per-arch-cache", action="store_true")
    args = ap.parse_args()

    if args.arch == "all":
        if not args.per_arch_cache:
            args.per_arch_cache = True
        for arch in ARCHES:
            outdir = arch_out_dir(args.out, arch)
            cmd = build_child_argv(args, arch, outdir)
            print("[export] " + " ".join(repr(x) for x in cmd))
            subprocess.run(cmd, check=True)
        return 0

    args.out = str(arch_out_dir(args.out, args.arch))
    metas = export_arch(
        Path(args.out), args.arch, args.batch, args.hidden, args.seq,
        args.elt_cols, args.gelu_tile, args.ln_variant, args.sm_variant,
        args.cache_root, args.per_arch_cache,
        ops=npu_ops.parse_ops(args.extra_ops, "--extra-ops"))
    print(f"\nwrote {len(metas)} eltwise design(s) under {args.out}")
    return 0
