"""Argument parsing and the top-level flow of export_gemm_rtp.py.

Usage (the module this replaces was tools/export_gemm_rtp.py, still a working
shim onto this package):
  python tools/export_gemm_rtp.py --target gte-multilingual-base --arch 1
  python tools/export_gemm_rtp.py --arch all --batch 128 --cols 8
"""

import argparse
import shlex
import subprocess
import sys

import npu_ops
from ..common.consts import (
    AIE_ROWS,
    DEFAULT_CACHE_ROOT,
    DEFAULT_SEQ,
    DEFAULT_TARGETS_FILE,
    FALLBACK_BATCH,
    FALLBACK_IDENTITY_THRESHOLD,
    REPO,
)
from ..common.paths import set_out_dir
from ..common.targets import load_targets, print_targets
from ..common.validate import validate_tiers_and_seq
from .build import build_child_argv, export_arch
from .geometry import STT_DECODE_SEQ
from .resolve import resolve_args
from .validate import validate_geometry


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
        "--rows",
        type=int,
        default=None,
        help=(
            f"AIE rows per core group. M must tile into m*rows, so this is "
            f"what decides the smallest expressible M (m*rows). Default: "
            f"{AIE_ROWS}, which is what every shipping design set uses. "
            f"Whisper's decoder needs 1: one greedy step decodes one token, "
            f"and rows={AIE_ROWS} would make the smallest design {AIE_ROWS}x "
            f"larger than the work."
        ),
    )
    ap.add_argument(
        "--tb-rows",
        type=int,
        default=None,
        help=(
            "AIE row-blocks the C path ping-pongs over; 0 derives it. The "
            "derived value (2) only tiles M when the row-block COUNT is a "
            "multiple of 4, i.e. M a multiple of 1024 for m=64 rows=4. Every "
            "shipped design sits on 256 or 1024; Whisper's encoder at seq 1536 "
            "is 6 row-blocks and times out. --tb-rows 1 halves the stride and "
            "costs C-path prefetch distance."
        ),
    )
    ap.add_argument(
        "--dec-seq",
        type=int,
        default=None,
        help=(
            "Decoder tokens per design step for a --target of kind stt. This is "
            "the decoder design's M dimension, NOT the encoder's frames: a "
            "greedy step decodes one token and the tier covers up to this many. "
            f"Default: {STT_DECODE_SEQ}."
        ),
    )
    ap.add_argument(
        "--stream-set",
        choices=["gemm_rtp", "stt"],
        default=None,
        help=(
            "Which operand set to build. 'gemm_rtp' is the four-stream encoder "
            "shape every embedder uses; 'stt' is Whisper's seven-stream "
            "decoder, whose cross-attention splits Q from K|V. --target sets "
            "this per model and the manual flags must not contradict it."
        ),
    )

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
        "--npu-extra-ops",
        default="",
        metavar="CODES",
        help=(
            "also build the elementwise design directories for the same "
            f"generation, alongside gemm_rtp: any comma-separated subset of "
            f"[{npu_ops.CODES}] (layn = LayerNorm, softm = softmax, gelu = "
            "GELU). Each is a compile and an xclbin, and each costs one extra "
            "hw_context at run time, so build only what the runtime's "
            f"{npu_ops.RUNTIME_FLAG} will ask for. Default: none, and the "
            "runtime's default -- all three ops on the host -- is the "
            "measured-faster path. This replaces --npu-eltwise, which built all "
            "three or none."
        ),
    )

    ap.add_argument(
        "--elt-cols",
        type=int,
        default=1,
        help=(
            "AIE columns for the eltwise designs built by --npu-extra-ops. "
            "LayerNorm and softmax refuse above 2 (see tools/export_eltwise.py)."
        ),
    )

    ap.add_argument(
        "--gelu-tile",
        type=int,
        default=1024,
        choices=[1024, 4096],
        help="elements per GELU DMA transaction for --npu-extra-ops gelu.",
    )

    ap.add_argument(
        "--gelu-variant",
        default="poly",
        choices=["poly", "erf"],
        help=(
            "which GELU function --npu-extra-ops gelu builds. poly is the "
            "degree-8 fit of the even part (2.49e-3 relative against exact "
            "erf) that the BERT-family designs have always used; erf is "
            "kernels/gelu_erf.cc, the activation a Whisper container declares. "
            "They are different functions, not two accuracies of one -- a "
            "packer refuses a checkpoint whose activation is not `gelu`. A "
            "target with kind stt picks erf on its own."
        ),
    )

    ap.add_argument(
        "--ln-variant",
        default="il4",
        choices=["base", "il4", "rne", "il4_rne", "il4_8", "il4_4"],
        help=(
            "LayerNorm kernel variant for --npu-extra-ops layn. The il4_8 and "
            "il4_4 entries are the same four-row body with fewer rows per call: "
            "L1 is 64 KB and a block of rows x columns is double-buffered on both "
            "sides, so a wide row (Whisper's 1280) fits four rows per call where "
            "MiniLM's 384 fits sixteen. The exporter picks that itself and "
            "overrides this flag."
        ),
    )

    ap.add_argument(
        "--ln-cols",
        type=int,
        default=None,
        help=(
            "row width of the LayerNorm design built by --npu-extra-ops layn. "
            "Default: --hidden. A Whisper container's d_model is the number "
            "here, and the runtime refuses a design whose width is not the "
            "model's."
        ),
    )

    ap.add_argument(
        "--ln-eps",
        type=float,
        default=None,
        help=(
            "epsilon compiled into the LayerNorm design. Default: the target's "
            "layer_norm_eps, else 1e-12. It sits INSIDE a square root, so 1e-5 "
            "(Whisper) and 1e-12 (MiniLM) are not a rounding difference; the "
            "runtime refuses a design whose eps is not the container's."
        ),
    )

    ap.add_argument(
        "--sm-variant",
        default="poly_il4",
        choices=["lib", "poly", "poly_il4", "poly_rne", "poly_il4_rne",
                 "wide", "wide2"],
        help=(
            "softmax kernel variant for --npu-extra-ops softm. `wide` and "
            "`wide2` are kernels/softmax_w.cc at one and two rows per call: a "
            "Whisper attention score row is n_kv (1500, padded) wide, which the "
            "64-column kernel cannot hold in registers or on a worker's stack. "
            "The kernel is chosen from --sm-cols, so this only has to agree "
            "with it."
        ),
    )

    ap.add_argument(
        "--sm-rows",
        type=int,
        default=None,
        help=(
            "row capacity of the softmax design built by --npu-extra-ops softm. "
            "Default: batch*12*seq, which is an embedder's attention. A Whisper "
            "dispatch is one QUERY CHUNK, so a stt target passes batch*seq: every "
            "dispatch fills and drains the whole buffer, and a design 12x wider "
            "than the work syncs 37 MB to move 1.5 MB of scores."
        ),
    )

    ap.add_argument(
        "--sm-cols",
        type=int,
        default=None,
        help=(
            "row width of the softmax design built by --npu-extra-ops softm. "
            "Default: 64. A Whisper target passes its padded n_kv, and the "
            "runtime refuses a design whose width is not the score row it is "
            "about to hand it."
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
        default=None,
        help=(
            "Maximum differing bytes allowed between final.xclbin files while "
            "still treating them as identical modulo UUID metadata. "
            f"Without --target defaults to {FALLBACK_IDENTITY_THRESHOLD}."
        ),
    )

    args = ap.parse_args()

    # --list-targets answers from the target file alone and touches neither the
    # toolchain nor the device, which is the point: it is the cheap way to ask
    # what a --target name is allowed to be. It therefore has to be handled
    # BEFORE anything resolves a target or spawns a child.
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

    # Validated here, not where it is used: a bad code must be refused before a
    # single design is compiled, and the eltwise child would otherwise be the
    # one to notice, after the GEMM set was already built. parse_exportable, not
    # parse_ops, because `conv` is a runtime code with no directory to compile
    # (tools/npu_ops.py) -- and this is also the line --dry-run goes through, so
    # a code that cannot be built is refused in a dry run too rather than
    # printed into a command that would fail later.
    npu_ops.parse_exportable(args.npu_extra_ops, npu_ops.EXPORTER_FLAG)
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

    # Single-architecture path. It runs EVERY resolved pass for that arch, not
    # just the first: an STT model resolves to two (encoder, decoder), and
    # taking resolved[0] built the encoder, reported success, and left the
    # decoder missing -- a design set that does not exist, discovered later as a
    # "no design set found" from the runtime with nothing pointing here.
    for ra in resolved:
        ra.args.out = str(set_out_dir(ra.args, ra.arch))
        rc = export_arch(ra.args, ra.arch)
        if rc:
            return rc
    return 0
