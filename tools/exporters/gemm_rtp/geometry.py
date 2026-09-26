"""The four GEMM streams' shapes and the operand/accumulator/transport dtypes.

shapes_for is where M = batch * seq appears: the GEMM cores never see the
sequence length, only the row count. Everything downstream (design.json, the
runtime's design_fits, the buffer sizing) reads these numbers.
"""

import argparse
import re
from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

from ..common.consts import DEFAULT_SEQ

# The order the four GEMM streams are emitted in, and therefore the order the
# runtime's instruction-stream slots are filled in.
STREAM_ORDER = ["qkv", "attn_out", "ffn_up", "ffn_down"]

# The Whisper DECODER's seven. Not the encoder's four: a decoder layer runs
# self-attention AND cross-attention, and its cross-attention splits into two
# operands because Q's A operand is the decoder state while K|V's is the encoder
# output. A fused Q|K|V would need one A operand, which does not exist here.
#
#   self_qkv        [d, 3d]   decoder state -> q|k|v   (k has no bias)
#   self_attn_out   [d,  d]   self-attention context -> out_proj
#   cross_q         [d,  d]   decoder state -> q
#   cross_kv        [d, 2d]   encoder output -> k|v   (k has no bias)
#   cross_attn_out  [d,  d]   cross-attention context -> out_proj
#   ffn_up          [d, 4d]   decoder state -> fc1
#   ffn_down        [4d, d]   -> fc2
STT_STREAM_ORDER = ["self_qkv", "self_attn_out", "cross_q", "cross_kv",
                    "cross_attn_out", "ffn_up", "ffn_down"]

# The decoder's per-step M is the TOKEN count, not frames -- `seq` here is the
# design's CAPACITY in tokens per step, and the tier is how many utterances (or
# how many tokens of one step) share it. It is 64 because that is m*rows for this
# geometry: one greedy step decodes one token, so the tier-1 design runs 64 rows
# to consume one, and every attempt to make that smaller ran into the design
# library's four-row assumption (see STT_DECODE_ROWS below).
STT_DECODE_SEQ = 64
STT_DECODE_TIERS = [1, 2, 4]

# THE DECODER RUNS FOUR ROWS, LIKE EVERY OTHER DESIGN IN THIS TREE, and that is
# a measured decision rather than the obvious one.
#
# A one-row design (rows=1, m=16) would make the smallest M 16 instead of 64 --
# four times less arithmetic per decode step -- and it builds, but the design
# library is not row-count-agnostic: core_columns() in common/cache.py identifies
# a cached design by counting columns of tiles with row >= 2, so a one-row design
# reports "no columns" and find_cache then rejects the entry it just built. Making
# rows=1 work is a change to the cache-identity scheme AND to the shim placement
# in gemm_pretiled.py, i.e. to every shipping design's ability to be re-exported.
#
# It is also not worth much: at M=16..64 a decode step is DISPATCH-bound, not
# FLOP-bound. A 30-token turbo transcript is 7 streams x 4 layers x 30 steps = 840
# dispatches at ~150 us, so ~126 ms of dispatch against 0.4-2 s of GEMM. Cutting
# the GEMM fourfold moves the total by tens of percent at best, and the whole
# step is under a second either way. rows=1 is a throughput project for later,
# not a correctness prerequisite.
#
# m=16 (not 64) is what makes M=64 rather than 256: M must tile into m*rows, so
# m is the lever on the decoder's M and the row count is not. m must be a
# multiple of 4*r = 16 (a static_assert inside the AIE matmul kernel) and
# m*n = 16*32 = 512 needs the narrow entry point added for it.
#
# k and n stay 64 and 32 -- the SAME tile the encoder and the packed container
# use -- so both design sets read one B layout and the operands need no second
# tiling. m does not appear in that layout, which is why it is free to differ.
STT_DECODE_ROWS = 4
STT_DECODE_TILE_M = 16


def stt_shapes_for(
    batch: int,
    hidden: int = 384,
    intermediate: int | None = None,
    seq: int = STT_DECODE_SEQ,
) -> dict[str, dict[str, int]]:
    """M/K/N for the seven decoder streams. M = batch * tokens_per_step.

    `seq` here means TOKENS, not frames: the design is exported for a capacity of
    `seq` tokens per step and the runtime picks the smallest tier that covers the
    step it is running. It is the same variable name as the encoder's because it
    is the same role in the design -- the M dimension -- and naming it anything
    else would invite the two to be compared.
    """
    M = batch * seq
    h = hidden
    f = 4 * h if intermediate is None else intermediate
    return {
        "self_qkv":       {"M": M, "K": h, "N": 3 * h},
        "self_attn_out":  {"M": M, "K": h, "N": h},
        "cross_q":        {"M": M, "K": h, "N": h},
        "cross_kv":       {"M": M, "K": h, "N": 2 * h},
        "cross_attn_out": {"M": M, "K": h, "N": h},
        "ffn_up":         {"M": M, "K": h, "N": f},
        "ffn_down":       {"M": M, "K": f, "N": h},
    }


def shapes_for_stream_set(
    stream_set: str,
    batch: int,
    hidden: int = 384,
    intermediate: int | None = None,
    gated: bool = False,
    qkv_n: int | None = None,
    seq: int = DEFAULT_SEQ,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """(stream order, shapes) for a named stream set. One dispatch point.

    Every caller that needs a stream list goes through here, because the two
    sets differ in BOTH the names and the shapes, and a caller that picked the
    wrong one would build a design whose K/N do not match the weights the
    runtime hands it -- a wrong answer with no error anywhere.
    """
    if stream_set == "stt":
        return list(STT_STREAM_ORDER), stt_shapes_for(batch, hidden,
                                                      intermediate, seq)
    return list(STREAM_ORDER), shapes_for(batch, hidden, intermediate, gated,
                                          qkv_n, seq)

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
