# NpuEmbeddings -- int4 weight-only GEMM operands, shared by every packer.
# SPDX-License-Identifier: Apache-2.0
#
# WHY A SEPARATE MODULE: `gemm_i8.py` describes the scheme the array RUNS, and
# this one the scheme that is WRITTEN for it. They are different questions --
# int4 is a storage format, not a datapath: the array has no 4-bit MAC, so
# every nibble is widened back to int8 before stage() and the design, the
# kernel, the dispatch and the container's own `a_dtype` are all the int8 ones,
# unchanged. Putting the widening inside gemm_i8.py would make that module's
# "one emitter, one scheme" claim false for a reader that has to tell them
# apart; putting it beside gemm_i8.py keeps one file per scheme and lets
# `verify_i4_scheme.py` check this one against itself.
#
# Env: numpy only.

import numpy as np

from npue import gemm_b_layout, pack_i4, tile_b

MAC_DEFAULT = (8, 8)


def add_gemm_b_int4(w, name, mat, tile_k, tile_n, fold=None, asmooth=None,
                    group=32, mac=MAC_DEFAULT):
    """Stage a [K, N] GEMM operand as four-bit weights over the int8 datapath.

    THE SCHEME, and every quantity in it exists because of what the runtime
    already does (`runtime/include/common/host_kernels.hpp`):
        Aq[i,:]  = rint((X[i,:] / asmooth) / sa[i])   sa[i] = max|.| / 127
        Y[i,j]   = int32_dot(Aq[i,:], Wq[:,j]) * sa[i] * s[j] + bias[j]
    so with `q` the panel the array receives and `s` the per-column scale the
    container carries as `.wscale`, the weight is `q * s / asmooth` and
    NOTHING downstream of `q` changes between int8 and int4.

    Given M = W * fold * asmooth (the smoothed matrix, exactly as in
    `add_gemm_b_int8`), and g = the group of input rows containing k:

        s8[j]   = max_k |M[k,j]| / 127      the INT8 scale rule, UNCHANGED
        s4[g,j] = max_{k in g} |M[k,j]| / 7  the int4 rule, per group per column
        t[k,j]  = rint(M[k,j] / s4[g,j])     clipped to +-7 -- four bits
        r[g,j]  = s4[g,j] / s8[j]            widening factor, one float
        q[k,j]  = rint(t[k,j] * r[g,j])      clipped to +-127 -- what stage()
                                             hands the design (npue.fold_i4)

    and q * s8[j] is M[k,j] to within one int4 rounding plus one int8
    re-rounding that is ~18x smaller than it. The container writes `t`
    (pack_i4), `s8` as `.wscale`, `r` as `.gscale`, `asmooth` as `.asmooth`.

    WHY GROUPS, AND WHY ALONG K. Measured on MiniLM's 38 GEMM matrices
    (tasks/0140): per-channel int4 has relative Frobenius error 1.60e-1
    against int8's 9.0e-3, because one scale for a whole output column lets a
    single large input region set the step everywhere else. Groups of 32 rows
    with their own scale, per column, take it to 1.02e-1; groups of 16 to
    8.8e-2. `group=0` means ONE group spanning the whole K, which is exactly
    per-channel int4 and is kept because it is the shape of the thing being
    compared, not because it is the shape to ship.

    WHY THE FACTOR RIDES AS `.gscale` AND NOT IN THE PANEL. The factor varies
    along K, and a value stored in the panel can only carry a per-COLUMN
    scale (that is what `dequantise_c` multiplies by), so the K-varying half
    has nowhere to go but a sidecar that stage() folds in while it still knows
    where each nibble sits. Folding at PACK time instead would mean writing
    int8, i.e. not packing int4 at all.

    WHY `.wscale` KEEPS THE INT8 RULE RATHER THAN THE INT4 ONE. Two reasons,
    one practical and one that a gate depends on: with r in front of it, the
    int8 rule is what makes `q` land inside +-127 by construction
    (|t| * r <= 7 * 127/7 = 127), and `check_i4_operand` can then verify
    `.wscale` against the SAME bit-exact rule `check_i8_operand` uses, so the
    two schemes' containers are checked by one invariant rather than two
    similar ones.

    The int32 accumulator argument is the int8 one and applies unchanged: the
    multiplicands are int8 in both cases, so `K * 127^2 < 2^31` is asserted
    rather than assumed.
    """
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if fold is not None:
        mat = mat * fold
    if asmooth is not None:
        # The weight half of the SmoothQuant identity: X @ W == (X/s) @ (sW).
        # Applied BEFORE quantisation, as in gemm_i8 -- the columns being
        # measured are the ones the array will multiply.
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
    g = int(group or K)
    if g < 1:
        raise SystemExit(f"{name}: int4 group {group} is not a positive row "
                         f"count (0 means per-channel, i.e. one group over all "
                         f"K={K})")

    # s8: the scale dequantise_c multiplies by, with the all-zero column held
    # at 1 so it stays exactly zero instead of becoming NaN (as gemm_i8).
    scale = np.abs(mat).max(axis=0) / 127.0                    # [N]
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)

    # s4: one scale per (group of rows, column). Rows are zero-padded up to a
    # whole number of groups rather than dropped, so t keeps K rows exactly.
    G = -(-K // g)
    padded = np.zeros((G * g, N), dtype=np.float32)
    padded[:K] = mat
    blocks = padded.reshape(G, g, N)
    s4 = np.abs(blocks).max(axis=1) / 7.0                      # [G, N]
    # An all-zero group would divide by zero. Held at 1, as above: its t is 0
    # either way, and the guard keeps the reconstruction below an equality
    # rather than a NaN comparison.
    s4 = np.where(s4 > 0, s4, np.float32(1.0)).astype(np.float32)
    t = np.rint(blocks / s4[:, None, :]).clip(-7, 7).astype(np.int8)
    t = t.reshape(G * g, N)[:K]
    gscale = (s4 / scale[None, :]).astype(np.float32)           # [G, N]

    # What the runtime will actually receive, so the error reported here is
    # the container's rather than the nibbles' alone. Indexed by ROW, not
    # broadcast as [G,1,N] against a [K,N] panel: that broadcast aligns on the
    # right, silently produces [G,K,N], and the norm that follows is then a
    # norm over G copies of the matrix.
    row = np.arange(K, dtype=np.int64) // g
    q = np.clip(np.rint(t.astype(np.float32) * gscale[row, :]),
                -127, 127).astype(np.int8)

    # THE LAYOUT IS THE I8 ONE, and deliberately so: `layout` describes the
    # panel the ARRAY consumes -- what stage() DMAs after the widening -- and
    # its hash must therefore be the int8 hash or every existing int8 design
    # would refuse the container for a difference that happens in the loader.
    # The entry's own `dtype` is what says how the bytes are stored.
    layout = gemm_b_layout(tile_k, tile_n, mac[0], mac[1], dtype="I8")
    flat = tile_b(t, tile_k, tile_n, mac[0], mac[1])
    w.add(name, pack_i4(flat), "I4", "gemm_b", [K, N], layout=layout)
    w.add(name + ".wscale", scale, "F32", "quant_scale", [N])
    w.add(name + ".gscale", gscale, "F32", "quant_group", [G, N])
    # Ships even when it is all ones, exactly as gemm_i8 does, so the runtime
    # has one code path and cannot be handed smoothing it silently skips.
    w.add(name + ".asmooth",
          np.ones(K, np.float32) if asmooth is None else asmooth,
          "F32", "quant_smooth", [K])

    deq = q.astype(np.float32) * scale[None, :]
    den = float(np.linalg.norm(mat))
    return float(np.linalg.norm(deq - mat) / den) if den else 0.0


def check_i4_operand(name, wq, wscale, gscale, asmooth, w_src, group):
    """Gate one int4 B panel against the weights it claims to be. THE INVERSE
    of `add_gemm_b_int4`, in the sense `check_i8_operand` uses: it takes the
    container's OWN panel and OWN scales and asks whether they are the ones
    this scheme produces from `w_src`.

    Like its int8 counterpart it does NOT recompute `asmooth` -- that needs a
    calibration corpus living outside the container -- and takes the file's
    as given, then verifies everything downstream of it.

    Five invariants, each with its own message:

      1. SHAPES AND SIGNS, as in gemm_i8: `wscale` is multiplied and
         `asmooth` is reciprocated on hardware, so a zero or a negative entry
         in either is an inf or a sign flip. `.gscale` is MULTIPLIED, so it
         must be non-negative; it may be exactly zero, which is a dead group.
      2. THE SCALE RULE, `s8[j] == max_k |M[k,j]| / 127`, bit-exact -- the
         same rule `check_i8_operand` checks, because the emitter keeps it.
      3. THE GROUP RULE, `r[g,j] == s4[g,j] / s8[j]` with
         `s4 = max over the group of |M| / 7`, bit-exact. This is the invariant
         that catches a group size that disagrees with config, a group axis
         taken over N instead of K, and a factor computed before the smoothing
         multiply.
      4. THE PAYLOAD: `q == rint(rint(M/s4) * r)` clipped to +-127, exact.
         Integers on the outside, so this is what catches a wrong tiling, a
         transposed operand and an asmooth folded on the wrong axis -- and it
         is the one an accuracy tolerance would blunt.
      5. DEAD GROUPS STAY DEAD: a (group, column) whose source block is all
         zero comes back all zero. The host slices the fused qkv by offset and
         relies on padding surviving quantisation as exact zeros
         (pack_npue.py:629).

    `w_src` is the weight BEFORE smoothing and BEFORE the 1/sqrt(head_dim)
    fold; the fold must already be applied, because the packer folds upstream.

    Returns (problems, rel_err) where rel_err is measured the same way
    `check_i8_operand` measures it -- `q * wscale` against the smoothed matrix
    -- so the two schemes' operand errors are the same number of the same
    kind, and neither is comparable to an end-to-end gate.
    """
    problems = []
    wq = np.asarray(wq)
    wscale = np.asarray(wscale, np.float32)
    gscale = np.asarray(gscale, np.float32)
    asmooth = np.asarray(asmooth, np.float32)
    w_src = np.ascontiguousarray(w_src, dtype=np.float32)

    # 1. shapes and signs
    if wq.dtype != np.int8:
        problems.append(f"{name}: panel is {wq.dtype}, expected int8 -- an I4 "
                        f"container holds nibbles and must be widened first")
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
    if gscale.ndim != 2 or gscale.shape[1] != N:
        problems.append(f"{name}.gscale is {gscale.shape}, expected "
                        f"[groups, {N}]")
        return problems, 0.0
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
    if np.any(gscale < 0):
        problems.append(f"{name}.gscale has {int((gscale < 0).sum())} negative "
                        f"entries (min {float(gscale.min()):.3e}); it "
                        f"multiplies a weight, so a negative one flips a sign")
    if problems:
        return problems, 0.0

    g = int(group or K)
    G = -(-K // g)
    if gscale.shape[0] != G:
        problems.append(
            f"{name}.gscale has {gscale.shape[0]} group rows, but K={K} with "
            f"int4_group={group} needs {G}; the group size is config "
            f"int4_group and the panel and its scales are disagreeing about it")
        return problems, 0.0

    # The emitter's arithmetic, in its order: ascontiguousarray(f32) ->
    # * asmooth[:, None] -> abs().max(0) / 127 -> where(>0, ., 1) -> zero-pad
    # -> reshape -> abs().max(1) / 7 -> where(>0, ., 1) -> rint -> clip.
    # Reproducing that order is what makes (2), (3) and (4) equalities.
    w_eff = w_src * asmooth[:, None]
    want_s8 = np.abs(w_eff).max(axis=0) / 127.0
    live = want_s8 > 0
    want_s8 = np.where(live, want_s8, np.float32(1.0)).astype(np.float32)

    # 2. the int8 scale rule
    if not np.array_equal(wscale, want_s8):
        off = np.nonzero(wscale != want_s8)[0]
        rel = float(np.max(np.abs(wscale[off] / want_s8[off] - 1.0)))
        problems.append(
            f"{name}.wscale disagrees with max|W*asmooth|/127 on {off.size} of "
            f"{N} channels (worst relative {rel:.3e}, first at column "
            f"{off[0]}: file {wscale[off[0]]:.6e}, rule "
            f"{want_s8[off[0]]:.6e}). int4 keeps the INT8 scale rule on "
            f"purpose -- see add_gemm_b_int4.")

    # 3. the group rule
    padded = np.zeros((G * g, N), dtype=np.float32)
    padded[:K] = w_eff
    blocks = padded.reshape(G, g, N)
    want_s4 = np.abs(blocks).max(axis=1) / 7.0
    live_g = want_s4 > 0
    want_s4 = np.where(live_g, want_s4, np.float32(1.0)).astype(np.float32)
    want_r = want_s4 / want_s8[None, :]
    if not np.array_equal(gscale, want_r):
        diff = np.argwhere(gscale != want_r)[0]
        gi, gj = int(diff[0]), int(diff[1])
        problems.append(
            f"{name}.gscale disagrees with (max over the group of "
            f"|W*asmooth|/7) / wscale on "
            f"{int((gscale != want_r).sum())} of {gscale.size} entries (first "
            f"at group {gi}, column {gj}: file {gscale[gi, gj]:.6e}, rule "
            f"{want_r[gi, gj]:.6e}). Either the group size is not "
            f"int4_group={group}, or the factor was taken over the wrong axis.")

    # 4. the payload
    want_t = np.rint(blocks / want_s4[:, None, :]).clip(-7, 7).astype(np.int8)
    want_t = want_t.reshape(G * g, N)[:K]
    row = np.arange(K, dtype=np.int64) // g
    want_q = np.clip(np.rint(want_t.astype(np.float32) * want_r[row, :]),
                     -127, 127).astype(np.int8)
    ndiff = int((wq != want_q).sum())
    if ndiff:
        first = np.argwhere(wq != want_q)[0]
        problems.append(
            f"{name}: {ndiff} of {wq.size} int8 values are not "
            f"rint(rint(W*asmooth/s4) * gscale) -- first at "
            f"[{first[0]},{first[1]}]: file {int(wq[first[0], first[1]])}, "
            f"expected {int(want_q[first[0], first[1]])}. A wrong tiling, a "
            f"transposed operand or asmooth folded on the wrong axis all land "
            f"here.")

    # 5. DEAD GROUPS STAY DEAD -- padding must survive quantisation as exact
    # zeros, because the host slices the fused qkv by offset (pack_npue.py:629).
    # The reverse direction is NOT an invariant and is deliberately absent: a
    # live group whose magnitude is 1/254th of its column's rounds to zero at
    # int4, and reporting that as a fault would be reporting the format.
    padded_q = np.zeros((G * g, N), dtype=np.int8)
    padded_q[:K] = wq
    dead = ~live_g                                    # [G, N]
    zero_q = np.abs(padded_q.reshape(G, g, N)).max(axis=1) == 0
    bad = np.argwhere(dead & ~zero_q)
    if bad.size:
        gi, gj = int(bad[0][0]), int(bad[0][1])
        problems.append(
            f"{name}: {bad.size} all-zero (group, column) blocks came back "
            f"non-zero (first at group {gi}, column {gj}, max |q| = "
            f"{int(np.abs(padded_q.reshape(G, g, N)[gi, :, gj]).max())}); the "
            f"host slices the fused qkv by offset and padding must survive "
            f"quantisation as exact zeros")

    deq = wq.astype(np.float32) * wscale[None, :]
    den = float(np.linalg.norm(w_eff))
    rel_err = float(np.linalg.norm(deq - w_eff) / den) if den else 0.0
    return problems, rel_err
