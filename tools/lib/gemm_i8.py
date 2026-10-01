# NpuEmbeddings -- int8 GEMM operands, shared by every packer.
# SPDX-License-Identifier: Apache-2.0
#
# WHY A SEPARATE MODULE: the scheme is one thing and the two architectures that
# use it are two. `pack_npue.py` has carried it since tasks/0078 for the
# BERT-family, and the Whisper packer needs the same arithmetic with a different
# calibration (audio activations, not text) and a different tile geometry
# (64, 32 forced by Whisper's shapes, not (64, 48) chosen for MiniLM's). Two
# copies of a quantisation scheme is one scheme with two futures, so the scheme
# lives here and both packers call it.
#
# Env: numpy only.

import numpy as np

from npue import gemm_b_layout, tile_b

MAC_DEFAULT = (8, 8)


def add_gemm_b_int8(w, name, mat, tile_k, tile_n, fold=None, asmooth=None,
                  mac=MAC_DEFAULT):
    """Stage a [K,N] GEMM operand as INT8, per-output-channel symmetric.

    THE SCHEME, and why this one (tasks/0078).

        s[j]     = max_k |W[k,j]| / 127          one scale per output column
        Wq[k,j]  = round(W[k,j] / s[j])          clipped to [-127, 127]

    and at runtime, with A quantised per ROW (per token) by the same rule,

        Y[i,j] = int32_dot(Aq[i,:], Wq[:,j]) * sa[i] * s[j] + bias[j]

    so dequantisation is a rank-1 outer-product scaling of the int32 result --
    one multiply per output element, folded into the pass that already reads C
    and adds the bias.

    **Per CHANNEL, not per tensor.** 2209.13325 documents the +-50-100 outlier
    dimensions in post-LN BERT that make per-tensor quantisation lose real
    accuracy; a per-column scale costs N floats per operand and removes the
    coupling between an outlier column and every other column. Per-ROW
    activation scales do the same job on the other operand, which is why this
    scheme needs no calibration set at all -- both scales are computed from the
    data in front of it.

    -127 not -128: the symmetric range is kept symmetric so that negating a
    weight cannot saturate differently from negating its opposite. It costs one
    representable value and removes an asymmetry that is invisible until it is
    not.

    THE ACCUMULATOR IS EXACT. int8 x int8 -> int32 has no rounding anywhere in
    the reduction (2608.13756's "integer alibi"), so every bit of error in this
    path is in the two roundings above and in the scale multiply -- NOT in the
    K-reduction, unlike bf16. Overflow is the one thing that would break that,
    and it is asserted rather than assumed.
    """
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if fold is not None:
        mat = mat * fold
    if asmooth is not None:
        # The weight half of the SmoothQuant identity: X @ W == (X/s) @ (sW).
        # Applied BEFORE quantisation so the columns being measured are the
        # ones the array will multiply.
        asmooth = np.ascontiguousarray(asmooth, dtype=np.float32)
        if asmooth.shape != (mat.shape[0],):
            raise SystemExit(f"{name}: asmooth is {asmooth.shape}, expected "
                             f"{(mat.shape[0],)} (one per INPUT channel)")
        mat = mat * asmooth[:, None]
    K, N = mat.shape
    if K * 127 * 127 >= 2 ** 31:
        raise SystemExit(f"{name}: K={K} could overflow the int32 accumulator "
                         f"({K} * 127^2 >= 2^31); refusing to pack a container "
                         f"whose arithmetic is not provably exact")
    scale = np.abs(mat).max(axis=0) / 127.0          # [N]
    # A genuinely all-zero column would divide by zero. Keep its scale at 1 so
    # the column stays exactly zero rather than becoming NaN.
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    q = np.rint(mat / scale[None, :]).clip(-127, 127).astype(np.int8)

    layout = gemm_b_layout(tile_k, tile_n, mac[0], mac[1], dtype="I8")
    flat = tile_b(q, tile_k, tile_n, mac[0], mac[1])
    w.add(name, flat, "I8", "gemm_b", [K, N], layout=layout)
    w.add(name + ".wscale", scale, "F32", "quant_scale", [N])
    # Ships even when it is all ones, so the runtime has ONE code path and
    # cannot be handed a container whose smoothing it silently skips.
    w.add(name + ".asmooth",
          np.ones(K, np.float32) if asmooth is None else asmooth,
          "F32", "quant_smooth", [K])
    # Report what the quantisation actually cost on THESE weights, so a bad
    # tensor is visible at pack time rather than at MTEB time.
    deq = q.astype(np.float32) * scale[None, :]
    den = float(np.linalg.norm(mat))
    return float(np.linalg.norm(deq - mat) / den) if den else 0.0


def check_i8_operand(name, wq, wscale, asmooth, w_src):
    """Gate one int8 B panel against the weights it claims to be. THE INVERSE
    of `add_gemm_b_int8`, in the sense that matters: it takes the container's
    OWN panel and OWN scales and asks whether they are the ones this scheme
    produces from `w_src`.

    WHAT IT DELIBERATELY DOES NOT DO is recompute `asmooth`. That needs a
    calibration corpus and the numpy oracle, which live outside the container;
    a gate that re-derived it would be measuring its own copy of the
    calibration, not the file. So the container's `asmooth` is taken as given
    and everything downstream of it is checked -- which is exactly the span a
    packing bug can live in: the fold, the axis, the tiling, the scale rule,
    the rounding, the saturation.

    Four invariants, each with its own message because they fail for different
    reasons and an operator needs to know which:

      1. SHAPES AND SIGNS. `wscale` is multiplied by the accumulator and
         `asmooth` is RECIPROCATED by the runtime (`1.0f / asmooth[j]`,
         bert_encoder.cpp), so a zero or negative entry in either is an inf or
         a sign flip on hardware, not a small error.
      2. THE SCALE RULE: `s[j] == max_k |W_eff[k,j]| / 127`, with the all-zero
         column held at 1. Catches a per-tensor scale, a scale computed BEFORE
         the asmooth multiply, a scale taken over the wrong axis, a transposed
         one. Compared bit-exactly: the packer's max is a float32 max over an
         already-materialised array, so an exact input gives an exact scale and
         anything else is a real disagreement, not a platform artefact.
      3. THE PAYLOAD: `Wq == rint(W_eff / s)` clipped. Exact, because these are
         integers -- this is what catches a wrong tiling, a transposed operand
         and an asmooth folded on the N axis instead of K, and it is the
         invariant a tolerance would blunt.
      4. NO SATURATION, and zero columns stay zero. With the correct scale the
         largest element of every LIVE column lands on exactly +/-127, so a live
         column that comes up SHORT was rounded at another scale; and a column
         the source says is all zero must come back as zeros, because the host
         slices the fused qkv by offset and relies on the padding having stayed
         zero through quantisation (pack_npue.py:629). Note the direction: a
         payload rounded at a SMALLER scale than the one stored still touches
         127, because the clip put it there -- so this invariant cannot see that
         case, and check 3 exists for it. Neither check replaces the other.

    `w_src` is the weight BEFORE smoothing and BEFORE the 1/sqrt(head_dim)
    fold -- i.e. what the caller read out of the checkpoint -- and the fold
    must already be applied to it, because the packer folds upstream of this
    function (pack_npue.py:1609) and not inside it.

    Returns (problems, rel_err) where rel_err is the packer's own `qerr`
    number, recomputed here from the container: the relative Frobenius error of
    the weight quantisation, measured against the smoothed matrix. It is the
    OPERAND's error, not the model's, and comparing it to an end-to-end gate is
    a category error (STATE.md sec 8.1 item 5).
    """
    problems = []
    wq = np.asarray(wq)
    wscale = np.asarray(wscale, np.float32)
    asmooth = np.asarray(asmooth, np.float32)
    w_src = np.ascontiguousarray(w_src, dtype=np.float32)

    # 1. shapes and signs
    if wq.dtype != np.int8:
        problems.append(f"{name}: panel is {wq.dtype}, expected int8")
    if w_src.ndim != 2:
        problems.append(f"{name}: source is {wq.shape}, expected a [K, N] matrix")
        return problems, 0.0
    K, N = w_src.shape
    if wq.shape != (K, N):
        problems.append(f"{name}: panel is {wq.shape}, source is {(K, N)}")
    if wscale.shape != (N,):
        problems.append(f"{name}.wscale is {wscale.shape}, expected ({N},) -- "
                        f"one scale per OUTPUT channel")
    if asmooth.shape != (K,):
        problems.append(f"{name}.asmooth is {asmooth.shape}, expected ({K},) "
                        f"-- one per INPUT channel")
    if problems:
        return problems, 0.0
    if not np.all(wscale > 0):
        bad = int(np.argmin(wscale))
        problems.append(f"{name}.wscale has {int((wscale <= 0).sum())} "
                        f"non-positive entries (min {wscale[bad]:.3e} at column "
                        f"{bad}); the runtime multiplies by it")
    if not np.all(asmooth > 0):
        bad = int(np.argmin(asmooth))
        problems.append(f"{name}.asmooth has {int((asmooth <= 0).sum())} "
                        f"non-positive entries (min {asmooth[bad]:.3e} at row "
                        f"{bad}); the runtime reciprocates it")
    if problems:
        return problems, 0.0

    # The packer's arithmetic, in its order: ascontiguousarray(f32) ->
    # * asmooth[:, None] -> abs().max(0) / 127 -> where(>0, ., 1) -> rint ->
    # clip. Reproducing that order is what makes (2) and (3) equalities.
    w_eff = w_src * asmooth[:, None]
    want_scale = np.abs(w_eff).max(axis=0) / 127.0
    live = want_scale > 0
    want_scale = np.where(live, want_scale, np.float32(1.0)).astype(np.float32)

    # 2. the scale rule
    if not np.array_equal(wscale, want_scale):
        off = np.nonzero(wscale != want_scale)[0]
        rel = float(np.max(np.abs(wscale[off] / want_scale[off] - 1.0)))
        problems.append(
            f"{name}.wscale disagrees with max|W*asmooth|/127 on {off.size} of "
            f"{N} channels (worst relative {rel:.3e}, first at column "
            f"{off[0]}: file {wscale[off[0]]:.6e}, rule "
            f"{want_scale[off[0]]:.6e}). Per-output-channel symmetric is the "
            f"scheme; a per-tensor or pre-smoothing scale fails here.")

    # 3. the payload
    want_q = np.rint(w_eff / wscale[None, :]).clip(-127, 127).astype(np.int8)
    ndiff = int((wq != want_q).sum())
    if ndiff:
        first = np.argwhere(wq != want_q)[0]
        problems.append(
            f"{name}: {ndiff} of {wq.size} int8 values are not "
            f"rint(W*asmooth/wscale) -- first at [{first[0]},{first[1]}]: file "
            f"{int(wq[first[0], first[1]])}, expected "
            f"{int(want_q[first[0], first[1]])}. A wrong tiling, a transposed "
            f"operand or asmooth folded on the wrong axis all land here.")

    # 4. NO SATURATION, and zero columns that stayed zero.
    #
    # With the correct scale the largest element of every LIVE column lands on
    # exactly +/-127, so a live column that comes up SHORT was rounded at some
    # larger scale than the one stored. And a column the source says is all
    # zero must come back as zeros: the host slices the fused qkv by offset and
    # relies on the padding having stayed zero through quantisation
    # (pack_npue.py:629).
    #
    # NOTE THE DIRECTION, because the other one is a trap. A payload rounded at
    # a SMALLER scale than the stored one still touches 127 -- the clip put it
    # there -- so this invariant cannot see it and check 3 exists for it. Two
    # checks, two directions; neither substitutes for the other.
    sat = np.abs(wq).max(axis=0).astype(np.int32)
    short = np.nonzero(live & (sat != 127))[0]
    if short.size:
        problems.append(
            f"{name}: {short.size} live columns do not reach 127 (first at "
            f"column {short[0]}, max |q| = {int(sat[short[0]])}); with the "
            f"correct scale the largest element of every column lands on "
            f"exactly 127, so this payload was rounded at another scale")
    loud = np.nonzero(~live & (sat != 0))[0]
    if loud.size:
        problems.append(
            f"{name}: {loud.size} all-zero columns came back non-zero (first at "
            f"column {loud[0]}, max |q| = {int(sat[loud[0]])}); the host slices "
            f"the fused qkv by offset and padding must survive quantisation as "
            f"exact zeros")

    deq = wq.astype(np.float32) * wscale[None, :]
    den = float(np.linalg.norm(w_eff))
    rel_err = float(np.linalg.norm(deq - w_eff) / den) if den else 0.0
    return problems, rel_err
