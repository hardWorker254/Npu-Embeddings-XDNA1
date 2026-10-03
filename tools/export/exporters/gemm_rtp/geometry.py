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


def attn_shapes(M: int, head_dim: int, n_kv: int,
                n_aie_cols: int = 4, tile_n: int = 32) -> dict[str, dict[str, int]]:
    """The two streams that turn attention into GEMMs.

        attn_qk   [M, head_dim] @ [head_dim, n_kv]  -> one head's scores
        attn_av   [M, n_kv]      @ [n_kv, head_dim]  -> that head's context

    `n_kv` is PADDED to the tile (1536 for a 1500-position window at tile_n 32),
    which is the whole reason this is expressible: the B panel has to tile
    evenly, and a score row of 1500 columns does not divide by 32. The padded
    keys are not a mask the runtime has to remember -- the B panel is built from
    the K|V block with zeros past n_kv, the scores that come back are finite, and
    the runtime writes -1e30 into exactly those columns before the softmax. The
    alternative, a design per n_kv, is a design per position.

    `n_kv` is padded to the LEAST COMMON MULTIPLE of tile_k and tile_n, not to
    tile_n alone: attn_qk's N only has to divide by tile_n, but attn_av's K is
    the same number and has to divide by tile_k as well.

    The context width is the other direction of the same wall, and the one that
    bites first: a design's N must be a multiple of tile_n * AIE columns, which
    is 128 on npu1, and head_dim is 64. So attn_av's N is head_dim rounded up to
    that, the extra columns of the B panel are zero, and the runtime reads the
    first head_dim of every C row. Halving the columns to make 64 legal is not
    an option -- every stream in a set shares the set's xclbin.

    head_dim is 64 on every shipped Whisper, and a K of 64 is one k-tile: a thin
    GEMM, which is why this is worth having for the arithmetic and not for the
    dispatch count.
    """
    step_k = _lcm(tile_n, 64)
    kv = -(-n_kv // step_k) * step_k
    ctx = -(-head_dim // (tile_n * n_aie_cols)) * (tile_n * n_aie_cols)
    return {
        "attn_qk": {"M": M, "K": head_dim, "N": kv},
        "attn_av": {"M": M, "K": kv, "N": ctx},
    }


def _lcm(a: int, b: int) -> int:
    from math import gcd
    return a * b // gcd(a, b) if a and b else max(a, b)


# -- the pose stream set -------------------------------------------------------
#
# A convolution on the array is its im2col matrix as a GEMM:
#
#     [out_h * out_w, Cin*kh*kw] @ [Cin*kh*kw, Cout]
#
# so M is the OUTPUT PIXEL COUNT, not a batch and not a sequence length. That is
# the whole difference from every other set in this file, and it is why a pose
# design set cannot borrow a stream set: there is no batch tier to dispatch on,
# and M is a property of the stride.
#
# TWO NUMBERS PER CONVOLUTION, NOT ONE. A YOLOv8-pose trunk strides 2, 4, 8, 16
# and 32, and each stride fixes its own M: 102400, 25600, 6400, 1600, 400. Four
# of the twenty-one padded (K, N) designs are used at TWO of those M values --
# conv1152x128 and conv576x64 and conv192x32 and conv64x32 each appear both as a
# stride-2 and as a stride-4 convolution. A stream has ONE M, so those are
# twenty-FIVE streams, not twenty-one. Padding M up to the largest instead was
# considered and rejected: a stride-4 convolution would then run four times the
# rows it needs, on four of the busiest shapes in the network.
#
# The (K, N) pairs are the checkpoint's own channel counts padded up to tile_k
# and to tile_n*cols, and they are listed here rather than derived from a formula
# because they are not a formula: they are whatever YOLOv8n-pose's widths happen
# to be. tools/pack/packers/pose.py holds the same list, and
# tools/verify/verify_pose_streamset.py checks the two against each other rather
# than trusting that they were written down the same twice.
# ONE M FOR THE WHOLE SET, NOT ONE PER STRIDE. A stride-2 convolution has
# M=102400 rows and a stride-16 one has M=400, and it is tempting to compile a
# stream per (K, N, M) -- twenty-five of them. The runtime does not want that:
# NpuConvBackend reads its row count ONCE, from the design's single top-level
# `M` (runtime/src/pose/session.cpp), and conv() then walks any convolution's
# own rows in chunks of that size (runtime/src/pose/net.cpp). So the design
# carries ONE dispatch chunk and every convolution is cut into pieces of it.
# Twenty-one streams it is, and the M that goes in all of them is
# POSE_DISPATCH_M. Padding M up per stride instead was rejected: the runtime has
# no place to keep twenty-five different row counts.
#
POSE_DISPATCH_M = 1024

POSE_CONV_SHAPES: tuple[tuple[int, int], ...] = (
    (64, 128), (128, 128), (192, 128), (256, 128), (256, 256),
    (320, 128), (384, 128), (384, 256), (512, 128), (512, 256),
    (576, 128), (1152, 128), (1152, 256), (2304, 128),
)


def pose_stream_order() -> list[str]:
    """Stream names, in the order the slots are filled in.

    `conv{K}x{N}` -- deliberately NOT carrying M, because the M is the design's
    single dispatch chunk rather than a property of the convolution. These are
    the names tools/pack/packers/pose.py writes into the container's
    npu_streams, so the two have to be spelled the same.
    """
    return [f"conv{k}x{n}" for k, n in POSE_CONV_SHAPES]


def pose_shapes_for(m: int = POSE_DISPATCH_M) -> dict[str, dict[str, int]]:
    """M/K/N for every pose convolution, keyed by pose_stream_order()'s names."""
    return {
        f"conv{k}x{n}": {"M": m, "K": k, "N": n}
        for k, n in POSE_CONV_SHAPES
    }


def mel_proj_shapes(M: int, n_bins: int, n_mels: int,
                    tile_k: int = 64, tile_n: int = 32,
                    n_aie_cols: int = 4) -> dict[str, dict[str, int]]:
    """The slaney filter bank as a GEMM: power spectrum @ bank.

        mel_proj   [M, n_bins] @ [n_bins, n_mels]   one frame's mel row

    The bank's own shape is (n_bins, n_mels) = (201, 80) or (201, 128), and
    neither tiles: K is 201 against a 64-wide k-tile and N is 80 against the
    128 a four-column design needs. So both are padded UP -- 256 and 128 -- and
    the runtime writes zeros into the A columns past n_bins and reads back only
    the first n_mels of every C row. That is the same padding the attention
    streams need for the same two reasons, and the reason a 201-bin spectrum
    cannot share their designs is that K and N are both wrong by a different
    amount.

    This is the CHEAPEST op in the tree: 48M MAC on a window, a few milliseconds
    on a host that has sixteen of them. It is here because the question was
    which operations CAN run on the array, not which should.
    """
    K = -(-n_bins // tile_k) * tile_k
    N = -(-n_mels // (tile_n * n_aie_cols)) * (tile_n * n_aie_cols)
    return {"mel_proj": {"M": M, "K": K, "N": N}}


def dft_shapes(M: int, n_fft: int = 400, n_bins: int = 201,
               tile_k: int = 64, tile_n: int = 32,
               n_aie_cols: int = 4) -> dict[str, dict[str, int]]:
    """The 400-point transform as one GEMM against a precomputed DFT matrix.

        dft400   [M, n_fft] @ [n_fft, 2 * n_bins]   -> (real, imaginary)

    A direct transform IS a matrix: X[k] = sum_n x[n] exp(-2*pi*i*k*n/400), so
    one GEMM against the matrix of twiddles computes every bin of every frame in
    the chunk. The real and imaginary halves sit side by side in the B panel, and
    the squared magnitude is taken on the host afterwards -- a square is not a
    matrix, and the array has no epilogue to put it in.

    K is 400 padded to 448 (the frame has 400 samples and the k-tile is 64) and N
    is 201 padded to 256, twice: 256 is the smallest multiple of 128 that holds
    201 bins, and the real and imaginary halves each get one. The runtime writes
    zeros into the 48 padded input columns and reads the first 201 bins of each
    half.

    WHAT THIS COSTS, STATED PLAINLY: the host transform is an exact-size
    mixed-radix Cooley-Tukey in fp64, accurate to about 1e-15 relative. This one
    multiplies by bf16 twiddles, so a bin carries roughly 6e-3 relative error and
    a power spectrum 1.2e-2. It computes the same function; it is not the same
    arithmetic, and the mel gate is what says whether that is inside tolerance.
    """
    K = -(-n_fft // tile_k) * tile_k
    half = -(-n_bins // (tile_n * n_aie_cols)) * (tile_n * n_aie_cols)
    return {"dft400": {"M": M, "K": K, "N": 2 * half}}


LOGIT_CHUNK = 6656   # 52 tiles of 32, the widest N this design library takes


def logit_shapes(M: int, hidden: int, vocab: int, n_chunks: int = 8,
                 tile_k: int = 64, tile_n: int = 32,
                 n_aie_cols: int = 4) -> dict[str, dict[str, int]]:
    """The tied token embedding as a GEMM, in chunks of the vocabulary.

        logits_i   [M, hidden] @ [hidden, chunk]   one slice of the vocabulary

    ONE stream cannot hold the projection: 51865 columns is 1621 tiles of 32
    against the 12 the biggest shipping stream uses, and the design library puts
    one tile per column per row-block, so the N is bounded by the silicon, not by
    patience. The vocabulary is therefore cut into chunks, each a stream of its
    own, and a step dispatches one per chunk and takes the argmax over what comes
    back.

    The chunk is padded up to a multiple of tile_n * AIE columns (128 here), and
    the runtime zeroes the padding and reads back only the ids that exist, so a
    vocab_size that is not a multiple of the chunk is not a wrong answer -- the
    last chunk is short and the rest of it is masked.
    """
    out: dict[str, dict[str, int]] = {}
    chunk = LOGIT_CHUNK
    for i in range(n_chunks):
        out[f"logits_{i}"] = {"M": M, "K": hidden, "N": chunk}
    return out


def shapes_for_stream_set(
    stream_set: str,
    batch: int,
    hidden: int = 384,
    intermediate: int | None = None,
    gated: bool = False,
    qkv_n: int | None = None,
    seq: int = DEFAULT_SEQ,
    extra_streams: tuple[str, ...] = (),
    attn: tuple[int, int] | None = None,
    n_aie_cols: int = 4,
    mel: tuple[int, int] | None = None,
    tile_k: int = 64,
    fft: tuple[int, int] | None = None,
    logit: tuple[int, int, int] | None = None,
    drop_streams: tuple[str, ...] = (),
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """(stream order, shapes) for a named stream set. One dispatch point.

    Every caller that needs a stream list goes through here, because the two
    sets differ in BOTH the names and the shapes, and a caller that picked the
    wrong one would build a design whose K/N do not match the weights the
    runtime hands it -- a wrong answer with no error anywhere.

    `extra_streams` names streams the SET gains, and `attn` is the
    (head_dim, n_kv) pair they need. They are appended, never inserted: the
    xclbin is taken from the first stream of the set and every slot number after
    it moves if the order does, so a design set exported without them keeps the
    slot numbers it always had.

    `drop_streams` removes names again, per batch tier. It exists for the
    vocabulary projection: a decode step is one row, so its chunks are only ever
    dispatched at the SMALLEST tier, and building eight streams at the other two
    is eight compiles per tier of an xclbin nothing binds.
    """
    if stream_set == "stt":
        order, shapes = list(STT_STREAM_ORDER), stt_shapes_for(
            batch, hidden, intermediate, seq)
    elif stream_set == "pose":
        # A pose set's M is the dispatch chunk, not b*seq, so `batch` and `seq`
        # are read here ONLY to be ignored -- which is why they are not
        # parameters of pose_shapes_for().
        order, shapes = pose_stream_order(), pose_shapes_for()
    else:
        order, shapes = list(STREAM_ORDER), shapes_for(
            batch, hidden, intermediate, gated, qkv_n, seq)
    if drop_streams:
        order = [n for n in order if n not in drop_streams]
        shapes = {k: v for k, v in shapes.items() if k not in drop_streams}
    if extra_streams:
        if attn is None:
            raise SystemExit(
                f"extra streams {list(extra_streams)} need a head_dim and a "
                f"padded n_kv, and the caller passed neither")
        M = shapes[order[0]]["M"]
        more: dict[str, dict[str, int]] = {}
        if attn is not None:
            head_dim, n_kv = attn
            more.update(attn_shapes(M, head_dim, n_kv, n_aie_cols))
        if mel is not None:
            n_bins, n_mels = mel
            more.update(mel_proj_shapes(M, n_bins, n_mels, tile_k,
                                        32, n_aie_cols))
        if fft is not None:
            n_fft, n_bins = fft
            more.update(dft_shapes(M, n_fft, n_bins, tile_k, 32, n_aie_cols))
        if logit is not None:
            hidden, vocab, n_chunks = logit
            more.update(logit_shapes(M, hidden, vocab, n_chunks, tile_k, 32,
                                     n_aie_cols))
        for name in extra_streams:
            if name in drop_streams:
                continue
            if name not in more:
                raise SystemExit(
                    f"unknown extra stream {name!r}; the ones a design set can "
                    f"gain are {sorted(more) or 'none for this model'}")
            order.append(name)
            shapes[name] = more[name]
    return order, shapes

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
