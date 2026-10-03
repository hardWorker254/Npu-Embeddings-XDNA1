# NpuEmbeddings -- quantised convolution WEIGHTS, for the vision architectures.
# SPDX-License-Identifier: Apache-2.0
#
# WHY NOT A CALL INTO gemm_i8.py / gemm_i4.py
# ---------------------------------------------
# The GEMM scheme and this one agree on the ARITHMETIC -- a symmetric scale per
# output channel, and for four bits a second per-group factor -- and differ on
# everything else. A GEMM operand here is a [K, N] matrix staged for the array:
# pre-tiled into panels of (tile_k, tile_n), carrying a layout hash the design
# set is matched against, and transposed on the way in. A convolution's weight is
# a [Cout, Cin, kh, kw] tensor that the HOST conv engine transposes itself, and
# none of the tiling applies to it. Writing pose quantisation as a call into
# gemm_i8 with the tiling arguments passed as no-ops would produce a container
# whose weights are laid out for a GEMM panel that no GEMM reads, and the two
# layouts differ by exactly a transpose -- which reads as a network that
# produces plausible activations and the wrong answer.
#
# So the scheme is written once HERE, and the GEMM modules keep their own. What
# is shared is deliberately only the parts that are the same idea: the scale
# rule, the -127/-7 symmetric range, and the factorisation of the four-bit scale
# into a per-channel factor and a per-group ratio.
#
# WHAT THIS BUYS, MEASURED, AND WHAT IT DOES NOT
# ----------------------------------------------
# Bytes. A container's weight payload drops 4x under i8 and 8x under i4, and the
# host arithmetic is UNCHANGED: the runtime dequantises once, at load, into the
# same fp32 buffer the GEMM already reads, so every product, every accumulation
# order and every bit of the fp32 path is exactly what it is without the
# quantisation.
#
# It does not buy TIME on the host path, and this file will not claim it does.
# An fp32 GEMM against a dequantised-then-fp32 weight is the same fp32 GEMM;
# the only saving is not reading the weights off disk. torch's 20.5 ms for this
# network is also a plain fp32 GEMM, so the 13x gap this project has against it
# is unaffected either way -- the measured split is 90 ms GEMM, 57 ms im2col,
# 7 ms output transpose, 10 ms weight transpose (runtime/src/pose/net.cpp's
# Cost). Squeezing that gap means not materialising im2col, not narrowing the
# weights.
#
# What int8 DOES buy on the array path is bandwidth and an int8 MMAC, which is
# the same reason whisper's does, and the same reason it is not measured here:
# the pose array path does not exist yet.
#
# WHAT THE ERROR IS
# -----------------
# Reported, not gated. `add_conv_w` returns the relative Frobenius error of the
# dequantised weights, the packer prints the worst per container, and
# tools/verify/verify_pose_quant.py compares the DETECTIONS of a quantised
# container against the same checkpoint in fp32 and prints the score, box and
# keypoint deltas. There is no pass/fail threshold, because there is no
# accuracy claim to make a threshold for: this is a size/accuracy trade the
# operator picks, and a gate would be inventing a number and calling it a
# requirement.
#
# MEASURED ON THIS CHECKPOINT, so the trade is not a matter of opinion. On
# bus.jpg, against the fp32 container's three detections
# (tools/verify/verify_pose_quant.py):
#
#     dtype   file      worst rel-Fro   score    box      keypoint   people
#     f32     13.5 MB        --           --       --         --        3
#     i8       4.0 MB       0.034      0.001   1.9 px    5.8 px      3
#     i4 g=8  4.2 MB       0.095      0.050   5.6 px   50.9 px      3
#     i4 g=16 3.4 MB       0.120      0.077  28.5 px   54.3 px      8
#     i4 g=32 3.0 MB       0.150      0.206  26.1 px   71.3 px      8
#     i4 g=64 2.8 MB       0.182      0.136  28.3 px   63.2 px      8
#     i4 g=0  2.7 MB       0.451      0.359  99.5 px  128.8 px      5
#     i4 g=1 15.6 MB       5.8e-08    0.000   0.0 px    0.0 px      3
#
# FOUR THINGS IN THAT TABLE ARE WORTH READING TWICE, and none of them is
# visible from the payload sizes alone.
#
# i8 costs a 1.9 px box and a 5.8 px keypoint on an 810x1080 photograph, which
# is invisible, for a third of the bytes. That is the quantisation to use.
#
# There is a CLIFF between i4 group 8 and group 16: eight keeps three
# detections, sixteen finds eight in a photograph of three. Both containers are
# valid, the smaller one is past the cliff, and the relative error on the
# weights moves only from 0.095 to 0.120 across it. A weight-error number cannot
# express that, which is why the detector comparison exists at all.
#
# i4 with a group of ONE is LOSSLESS -- a scale per weight, so the nibbles are
# exact and the detections are identical to fp32's -- and its file is 15.6 MB,
# BIGGER than fp32's 13.5. Four bits per weight plus four bytes of scale per
# weight is more than the four bytes it replaced. A group too small to lose
# anything saves nothing, and those are one fact rather than two.
#
# And the smallest files on the table are the least correct ones. The payload
# is 1.64 MB for every i4 row; what separates 2.7 MB from 4.2 MB is entirely
# scale sidecars, and what separates 0.359 of score error from 0.050 is entirely
# the group size. The payload is the part nobody had to think about.
#
# Env: numpy only.

import numpy as np

# The supported weight precisions. Spelled here once so the packer's --dtype
# metavar and this module cannot drift apart, which is what happened to the
# equivalent list in pack_npue.py before it was derived.
DTYPE_CHOICES = ("f32", "i8", "i4")

I8_MAX = 127   # not 128: the symmetric range stays symmetric, so negating a
I4_MAX = 7     # weight cannot saturate differently from its opposite


def conv_weight_dtype(kind):
    """Normalise a --dtype spelling for CONVOLUTION weights.

    The container's dtypes come from the NPU vocabulary (BF16, I8, ...), so this
    is where the CLI's lowercase names meet the file's uppercase ones, in one
    place, and a spelling in neither list is refused by name rather than
    silently treated as fp32.
    """
    k = str(kind).lower()
    if k in ("f32", "fp32", "float32"):
        return "f32"
    if k in ("i8", "int8"):
        return "i8"
    if k in ("i4", "int4"):
        return "i4"
    raise SystemExit(f"conv weight dtype {kind!r} is not one of "
                     f"{', '.join(DTYPE_CHOICES)}. There is no datapath here for "
                     f"fp16 or fp8, and a weight written at a precision this "
                     f"runtime cannot read back is a container that loads and "
                     f"then reads its own bytes as something else.")


def pack_i4(q):
    """[N, K] values in [-7, 7] -> [N, ceil(K/2)] bytes, two per byte along K.

    The low nibble holds the EVEN index and the high nibble the odd one, so byte
    (n, j) is elements (n, 2j) and (n, 2j+1) of the row. That is stated rather
    than left to the reader because the other order -- odd first -- is a
    permutation of every weight in the tensor, and a 4-bit tensor permuted along
    K is still the right number of bytes and a plausible-looking network.

    An ODD K gets one zero-padded column at the end rather than a refusal. K is
    the input-channel count times the kernel area, and YOLOv8's stem is 3x3x3 =
    27, so "K must be even" would refuse the very first convolution of the very
    model this scheme was added for. The pad column is a zero weight the runtime
    never reads -- its position follows from the 4-D shape, so there is nothing
    to record and nothing that can disagree.
    """
    if q.shape[-1] % 2:
        q = np.concatenate([q, np.zeros((q.shape[0], 1), q.dtype)], axis=1)
    lo = (q[:, 0::2] & 0x0F).astype(np.uint8)
    hi = (q[:, 1::2] & 0x0F).astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_i4(packed, K):
    """The inverse of `pack_i4`, and the reference both sides are checked against.

    Sign-extends the nibbles: a value with bit 3 set is 4-bit two's complement,
    so 0xF is -1 and not 15. Reading them as unsigned is not a small error -- it
    maps the single most negative weight onto the single largest positive one.

    An ODD `K` is padded UP to an even column count and the pad is dropped on the
    way out, which is the inverse of `pack_i4`'s pad. It used to assign the wide
    half into a narrow column slice and raise a broadcast error -- on exactly the
    weights this scheme was added for, since YOLOv8's stem is 3x3x3 = 27.
    """
    p = np.asarray(packed, dtype=np.uint8)
    k_even = K + (K & 1)
    lo = (p & 0x0F).astype(np.int16)
    hi = ((p >> 4) & 0x0F).astype(np.int16)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    out = np.empty((p.shape[0], k_even), dtype=np.int16)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out[:, :K]


def _view(mat):
    """[Cout, Cin, kh, kw] or [N, K] -> ([N, K], 4-D shape)."""
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if mat.ndim == 4:
        return mat.reshape(mat.shape[0], -1), [int(x) for x in mat.shape]
    if mat.ndim == 2:
        return mat, [mat.shape[0], 1, 1, mat.shape[1]]
    raise SystemExit(f"a weight of rank {mat.ndim}; a convolution's weight is "
                     f"[Cout, Cin, kh, kw] and its [Cout, K] view is what the "
                     f"scale rule reads.")


def _channel_scale(mat):
    """[N] per-output-channel scale, with an all-zero row held at 1.

    A row that is entirely zero is not a weight that lost its scale; it is a
    dead output channel, and dividing it by its own zero would turn the whole
    row into NaN and then into a network that draws nothing.
    """
    s = np.abs(mat).max(axis=1) / float(I8_MAX)
    return np.where(s > 0, s, np.float32(1.0)).astype(np.float32)


def quantise(view, s, s4, dtype):
    """The integer payload a weight quantises to, from its scales.

    ONE definition, called by `add_conv_w` to write and by `check_conv_w` to
    check. It has to be one: the two used to spell the division differently --
    `view / s` on the packing side and `view * (1 / s)` on the checking side --
    and those are not the same number in float32. At a weight whose scaled value
    lands on 63.5 they round to 64 and 63, one weight in 27 000 out of 69 000,
    and the checker reported a packing bug in a container that was packed
    correctly.

    That is the general hazard this function exists to remove: a bit-exact
    payload comparison is only meaningful if both sides round identically, and
    "identically" is a property of the expression, not of the intent. What
    `check_conv_w` still verifies independently is everything arithmetic cannot
    vouch for -- the scales' shapes and signs, the payload's AXIS and NIBBLE
    order, the group boundaries, and that `gscale * wscale` is the source's own
    per-group maximum. It takes the rounding rule from here, the way it takes the
    scales from the file, and its docstring says so.

    `view` is the [N, K] view for i8, and for i4 the [N, G*g] view PADDED UP TO A
    WHOLE GROUP -- the same pad `dequantise` applies, because the group index of
    element (n, k) is k // g and the last group is short otherwise. Returns
    int16 in [-I8_MAX, I8_MAX] or [-I4_MAX, I4_MAX], the same width as `view`.
    """
    if dtype == "i8":
        # `view / s[:, None]` and NOT `view * (1 / s)`: see above.
        return np.rint(view / s[:, None]).clip(-I8_MAX, I8_MAX).astype(np.int16)
    if dtype == "i4":
        # Against the GROUP's scale, not the ratio. See add_conv_w's note.
        N, G = s4.shape
        g = view.shape[1] // G
        return np.rint(view.reshape(N, G, g) / s4[:, :, None]
                       ).clip(-I4_MAX, I4_MAX).astype(np.int16).reshape(N, G * g)
    raise SystemExit(f"quantise: unknown weight dtype {dtype!r}")


def group_scales(view, K, group, s):
    """The [N, G] per-group scales for i4, and the padded view they divide.

    Shared because the pad and the group count are the two numbers a four-bit
    implementation has to agree on between packing and checking, and getting
    them out of one place is cheaper than discovering the disagreement from a
    payload that is one byte per row short.
    """
    g = int(group) if group else K
    G = (K + g - 1) // g
    vp = np.concatenate([view, np.zeros((view.shape[0], G * g - K), np.float32)],
                        axis=1)
    s4 = np.abs(vp).reshape(view.shape[0], G, g).max(axis=2) / float(I4_MAX)
    s4 = np.where(s4 > 0, s4, np.float32(1.0)).astype(np.float32)
    return vp, s4


def dequantise(mat, dtype, group=None):
    """The fp32 weight a container of this dtype holds, from the SOURCE weight.

    ONE definition, called by `add_conv_w` to report its own error and by the
    pose packer to build the array's panels from. It has to be one definition
    rather than two that agree today: the array's bf16 panels and the host's
    dequantised weights are the same convolution, and a container that shipped
    the fp32 checkpoint into the panels and the int8 checkpoint into the host
    would run two different networks behind one `--npu-extra-ops conv` flag, with
    no file-level way to tell. Both backends taking the same numbers is the
    property that makes the flag worth having.

    Returns the [N, K] view, K unpadded. Round-trips exactly through the pack /
    unpack pair -- `add_conv_w` stores the payload these values came from, not
    these values.
    """
    view, _shape = _view(mat)
    N, K = view.shape
    if dtype == "f32":
        return view.copy()
    s = _channel_scale(view)
    if dtype == "i8":
        q = quantise(view, s, None, "i8")
        return (q * s[:, None]).astype(np.float32)
    if dtype != "i4":
        raise SystemExit(f"dequantise: unknown weight dtype {dtype!r}")
    g = int(group) if group else K
    vp, s4 = group_scales(view, K, group, s)
    r = (s4 / s[:, None]).astype(np.float32)
    q = quantise(vp, s, s4, "i4")
    return (q[:, :K].astype(np.float32) * s[:, None] *
            np.repeat(r, g, axis=1)[:, :K])


def dequantise_payload(payload, wscale, gscale, N, K, dtype, group=None):
    """The fp32 weight a CONTAINER holds, from the container's OWN bytes.

    `dequantise` answers a different question -- "what does the packer's
    arithmetic produce from the checkpoint?" -- and it re-derives the scales to
    do it. That makes it the wrong half of a comparison against the runtime's
    reader: the reader multiplies the factors stored in the file, and this
    function multiplies factors it computed itself, so a container whose stored
    scale is one float32 ulp from a recomputation is flagged as a reader bug on
    88 of the stem's 432 weights even when the two readers agree perfectly. The
    packer's ONNX read and a checker's ONNX read are not the same float32
    pipeline, and the difference does not have to be a bug in either.

    So the comparison tools/verify/verify_conv_quant.py makes is THIS against its
    transcription of geometry.cpp: both sides read the same payload and the same
    scales off disk, and every factor is the file's. The order of the two
    multiplies is left-to-right, which is the C++'s order and not an
    implementation detail -- `(v * s) * r` and `v * (s * r)` are different
    float32 numbers.
    """
    s = np.asarray(wscale, dtype=np.float32).reshape(-1)
    if s.size != N:
        raise SystemExit(f"wscale has {s.size} entries for {N} output channels")
    if dtype == "f32":
        return np.asarray(payload, dtype=np.float32).reshape(N, K)
    if dtype == "i8":
        q = np.asarray(payload, dtype=np.int8).reshape(N, K)
        return q.astype(np.float32) * s[:, None]
    if dtype != "i4":
        raise SystemExit(f"dequantise_payload: unknown weight dtype {dtype!r}")
    gsz = int(group) if group else K
    G = (K + gsz - 1) // gsz
    gs = np.asarray(gscale, dtype=np.float32)
    if gs.shape != (G, N):
        raise SystemExit(f"gscale is {gs.shape}, not {(G, N)} for a group of "
                         f"{gsz} over K={K}")
    q = unpack_i4(payload, K).astype(np.float32)            # [N, K]
    # gscale is [G, N] on disk and the reader walks it as gs[g * N + n]; .T[gsel]
    # is that indexing without the stride arithmetic.
    gsel = np.arange(K) // gsz
    return q * s[:, None] * gs.T[:, gsel]


def add_conv_w(w, base, mat, dtype="f32", group=None):
    """Store one convolution weight, [Cout, Cin, kh, kw] or its [N, K] view.

    `base` is the tensor PREFIX -- "conv.7" for this architecture -- and the
    weight lands at `base + ".w"`, its scales at `base + ".wscale"` and
    `base + ".gscale"`. The prefix rather than the full name is the argument
    because gemm_i8 takes the panel's name and appends ".wscale" to it; passing
    the full "conv.7.w" here would produce "conv.7.w.wscale", which is a second
    spelling of the same tensor one call site away from the right one.

    Returns (relative Frobenius error of the dequantised weight, bytes written).

    The scheme, for i8:

        s[n]      = max_k |W[n,k]| / 127        one scale per OUTPUT CHANNEL
        Wq[n,k]   = rint(W[n,k] / s[n])         clipped to [-127, 127]

    and for i4 a second factor along the INPUT axis, over groups of `group`
    columns of the [N, K] view. The four-bit PAYLOAD is quantised against the
    group's OWN scale, and the ratio between that and the channel's int8 scale
    is applied when the weight is DEQUANTISED:

        s4[g,n]   = max_{k in g} |W[n,k]| / 7
        t[n,k]    = rint(W[n,k] / s4[k/group, n])   clipped to [-7, 7]
        r[g,n]    = s4[g,n] / s[n]                  a RATIO, not a scale
        W[n,k]  =  t[n,k] * r[k/group, n] * s[n]

    `r` rides as its own tensor rather than being folded into s4 because the
    runtime multiplies a rank-1 outer product and folding would need the
    division done again per element at load. Same reason, and the same spelling
    as gemm_i4's `.gscale`, so the two are recognisably one idea.

    The direction matters and is the whole of an i4 implementation's arithmetic:
    quantising against `r` instead of against `s4` puts every value ~18x too
    large (127/7), clips the whole tensor at the 4-bit limit and leaves a
    relative error of about 0.9 -- worse than throwing the weights away. It
    still produces the right number of bytes and a plausible container, and the
    only symptom is a detector that is quietly, comprehensively wrong.

    PER OUTPUT CHANNEL, not per tensor, for the reason 2209.13325 gives for the
    text models: a handful of outlier input channels are what makes a single
    per-tensor scale lose accuracy that a per-channel one keeps, and it costs N
    floats.
    """
    view, shape = _view(mat)
    N, K = view.shape
    if dtype == "f32":
        w.add(base + ".w", view.reshape(shape), "F32", "conv_w", shape)
        return (0.0, N * K * 4)

    s = _channel_scale(view)                       # [N]
    # The values the runtime will actually hold, computed once and used for BOTH
    # the reported error and the array panels -- see `dequantise`.
    deq = dequantise(view, dtype, group)

    if dtype == "i8":
        q = quantise(view, s, None, "i8").astype(np.int8)
        w.add(base + ".w", q.reshape(shape), "I8", "conv_w", shape)
        w.add(base + ".wscale", s, "F32", "quant_scale", [N])
        nbytes = N * K
    elif dtype == "i4":
        g = int(group) if group else K
        if g <= 0:
            raise SystemExit(f"{base}: int4 group {group} is not a positive "
                             f"column count")
        G = (K + g - 1) // g
        # Zero-pad K up to a whole number of groups rather than dropping the
        # tail: dropping it would shorten the weight, and a shorter weight is a
        # different convolution. `group_scales` owns the pad so that the checker
        # cannot disagree about how many columns the last group has.
        vp, s4 = group_scales(view, K, group, s)
        r = (s4 / s[:, None]).astype(np.float32)                       # [N, G]
        q = quantise(vp, s, s4, "i4")
        # PACK OVER THE FIRST K COLUMNS, not over the group's padded width. The
        # columns past K exist only so s4 has a whole group to divide by, and
        # they hold zeros; storing them would make the payload row ceil(G*g/2)
        # bytes when the reader can only derive ceil(K/2) from the 4-D shape.
        # Those two agree exactly when g is even and disagree by g/2 bytes per
        # row when it is not -- which is every weight in this network, whose
        # stem is 27 columns wide.
        w.add(base + ".w", pack_i4(q[:, :K]), "I4", "conv_w", shape)
        w.add(base + ".wscale", s, "F32", "quant_scale", [N])
        # [G, N], not [N, G]: the runtime walks the group along K for a fixed
        # output channel, so a group's scales want to be adjacent.
        w.add(base + ".gscale", np.ascontiguousarray(r.T), "F32", "quant_group",
              [G, N])
        nbytes = N * (K + 1) // 2
    else:
        raise SystemExit(f"{base}: unknown weight dtype {dtype!r}")

    den = float(np.linalg.norm(view))
    err = float(np.linalg.norm(deq - view) / den) if den else 0.0
    return (err, nbytes)


def check_conv_w(name, wq, wscale, gscale, w_src, dtype, group=None):
    """Gate ONE container's weight against the checkpoint's, the inverse of
    `add_conv_w`.

    WHAT IT DELIBERATELY DOES NOT DO is re-derive the scales from `w_src` and
    compare them -- it takes the container's OWN scales and checks that the
    payload is the payload those scales imply. Recomputing would mean measuring
    this function's own copy of the packer, which is the thing already tested;
    what can be wrong in the container is the packing, the axis, the group
    boundary and the rounding, and all four are here.

    Four invariants, each with its own message because they fail for different
    reasons:

      1. SHAPES. `wscale` is one per output channel and `gscale` is [G, N]; a
         group count that disagrees with the group size is a weight read with
         the wrong stride, which is every value after the first group.
      2. NO NEGATIVE SCALE. Both are MULTIPLIED, so a negative one flips a
         weight's sign rather than failing; and a zero `wscale` is a divide by
         zero at load.
      3. THE PAYLOAD: Wq == quantise(W, the file's own scales), compared
         bit-exactly, because these are integers -- which is what catches a
         transposed tensor, a nibble order swapped, a group boundary off by one
         and a scale applied on the wrong axis. A tolerance would blunt all four.
         The ROUNDING RULE is shared with the packer (`quantise`), because two
         spellings of the same division disagree in the last place and a
         bit-exact comparison of two different expressions is a false positive,
         not a check.
      4. THE FACTOR: s4[g,n] == gscale[g,n] * wscale[n], with the all-zero group
         held at 1 exactly as the packer holds it. This is what catches a
         `.gscale` computed before an `.asmooth`-like fold that the payload did
         get -- and a ratio applied at quantisation time rather than at
         dequantisation, which would be internally consistent and still wrong.
    """
    src = np.ascontiguousarray(w_src, dtype=np.float32)
    if src.ndim == 4:
        src = src.reshape(src.shape[0], -1)
    N, K = src.shape

    s = np.asarray(wscale, dtype=np.float32).reshape(-1)
    if s.size != N:
        raise SystemExit(f"{name}: wscale has {s.size} entries for {N} output "
                         f"channels")
    if not np.all(np.isfinite(s)) or np.any(s < 0):
        raise SystemExit(f"{name}: wscale has a negative or non-finite entry. "
                         f"It is multiplied into every weight of the channel, so "
                         f"a negative one is a sign flip and a zero is a divide "
                         f"by zero -- neither is a small error")

    if dtype == "i8":
        q = np.asarray(wq, dtype=np.int16).reshape(N, -1)
        vp = src
    elif dtype == "i4":
        g = int(group) if group else K
        G = (K + g - 1) // g
        r = np.asarray(gscale, dtype=np.float32)
        if r.shape != (G, N):
            raise SystemExit(f"{name}: gscale is {r.shape}, expected "
                             f"[{G}, {N}] -- one scale per group of {g} columns "
                             f"per output channel. Read with any other stride, "
                             f"every weight past the first group is wrong while "
                             f"the byte count still matches")
        if not np.all(np.isfinite(r)) or np.any(r < 0):
            raise SystemExit(f"{name}: gscale has a negative or non-finite entry; "
                             f"it is multiplied, not divided")
        # s4 = the RATIO times the channel scale, and the check is that it IS the
        # source's own per-group maximum / 7. That is the invariant that makes
        # the payload comparison below mean something: a container whose ratio
        # were applied at quantisation time instead of at dequantisation would be
        # internally consistent -- its payload, its ratio and its scale would all
        # agree -- and would pass a checker that only multiplied them back
        # together.
        s4 = r.T * s[:, None]                       # [N, G], lossy
        vp, want4 = group_scales(src, K, group, s)
        if not np.allclose(s4, want4, rtol=1e-6, atol=1e-12):
            raise SystemExit(f"{name}: gscale * wscale is not the per-group "
                             f"max|K|/7 the source implies (max abs difference "
                             f"{float(np.abs(s4 - want4).max()):.3g})")
        # The payload is quantised against `want4`, NOT against the reconstructed
        # `s4`. The reconstruction is `gscale * wscale` -- two fp32 multiplies and
        # a divide's worth of error -- and a weight whose value over s4 lands on
        # 3.5 rounds to 4 against one and 3 against the other. Two values in
        # 28 000 on this checkpoint, and a spurious "the packing is wrong" on a
        # container that is right. The check above is what pins the file's scales
        # to the packer's; using them to re-derive the payload would only add a
        # second, lossier path to the same answer.
        s4 = want4
        # UNPACK HERE, not at the call site: `wq` is the container's PAYLOAD --
        # one byte per two weights, nibble-ordered -- and a checker that expected
        # it already unpacked would be checking a value no reader ever sees.
        # pack_i4's own inverse is therefore part of the invariant, which is why
        # the nibble order is checked here rather than trusted from the packer.
        raw = np.asarray(wq, dtype=np.uint8)
        # ceil(K/2): the odd-K case is the 3x3x3 stem, whose trailing pad column
        # pack_i4 adds, so the byte count is not K/2 and computing it that way
        # reads the LAST nibble of the last row as if it were a weight.
        want_bytes = (K + 1) // 2
        if raw.ndim != 2 or raw.shape[0] != N or raw.shape[1] != want_bytes:
            raise SystemExit(
                f"{name}: the stored payload is {raw.shape} bytes for a "
                f"[{N}, {K}] weight -- one byte holds two weights, so it must "
                f"be [{N}, {want_bytes}]")
        q = unpack_i4(raw, raw.shape[1] * 2)[:, :K]
    else:
        raise SystemExit(f"{name}: check_conv_w has no rule for dtype {dtype!r}")

    got = q
    # The rounding rule comes from `quantise`, shared with the packer -- see its
    # note on `view / s` versus `view * (1 / s)`. Re-deriving it here is what
    # produced a spurious "one weight differs" on a correctly packed container.
    want = quantise(vp, s, s4 if dtype == "i4" else None, dtype)[:, :K]
    if got.shape != want.shape:
        raise SystemExit(f"{name}: the stored payload is {got.shape} and the "
                         f"source implies {want.shape}")
    if not np.array_equal(got, want):
        bad = np.argwhere(got != want)
        i, k = bad[0]
        raise SystemExit(
            f"{name}: the payload disagrees with the source at [{i}, {k}] -- "
            f"stored {got[i, k]}, the source's weights and this container's own "
            f"scales imply {want[i, k]}. Of {bad.shape[0]} differing values, the "
            f"first is index {tuple(int(x) for x in bad[0])}; a transposed "
            f"tensor differs everywhere, a swapped nibble order differs in "
            f"every odd column, and a group boundary off by one starts at one "
            f"group in")