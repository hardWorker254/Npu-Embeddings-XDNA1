# NpuEmbeddings -- the int4 scheme, checked against itself, and against C++.
# SPDX-License-Identifier: Apache-2.0
#
# WHY THIS EXISTS ALONGSIDE tools/verify/verify_npue.py and verify_i8_scheme.py.
# verify_npue.py answers "is this container the right container for this
# checkpoint" by running a Python encoder against goldens -- it needs
# `reference/`, the oracle and the goldens. verify_i8_scheme.py answers the
# narrower question that needs none of that: does the int8 scheme round-trip
# through the container format, and does its gate CATCH a deliberately broken
# packing. This one asks the same of int4 and adds the one question neither of
# the others can ask.
#
# THAT QUESTION IS THE POINT. int4 is the first format in this tree where the
# runtime does arithmetic the writer also did. An int8 container is written
# once and read as bytes; an int4 container stores `t`, and the int8 panel the
# array multiplies only exists after somebody has folded `.gscale` back in at
# `k // int4_group`. Two implementations of that one order now exist:
#
#     npue.fold_i4()                    (Python: at read time, and at pack time)
#     common/int4_panel.hpp             (C++, at stage time)
#
# Two definitions of one order is exactly what byte equality exists to hold
# apart, because every failure they can have -- a nibble read at the wrong
# index, a group folded at the wrong row, a sign extended from the wrong bit,
# a round that breaks ties the other way -- still produces a RIGHT-SIZED,
# RIGHT-HASHED panel of plausible-looking numbers. No accuracy tolerance sees
# it; the container's own self-checks do not see it either, because each half
# is internally consistent with its own mistake. Section 7 therefore COMPILES
# common/int4_panel.hpp and requires its output to equal Python's byte for
# byte, over several group sizes including 0 (one group over the whole K) and
# a group larger than K.
#
# Sections 1-6 are verify_i8_scheme's question applied to the new scheme: the
# honest round trip, the four group sizes, a zero padding column, an unsmoothed
# panel, ten injected faults each of which the gate must NAME, and the
# refusals that have to be refusals rather than plausible numbers.
#
# Env: numpy, a C++17 compiler, no checkpoint and no NPU. Usage:
#   python tools/verify/verify_i4_scheme.py

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

from gemm_i4 import (add_gemm_b_int4, check_i4_operand)         # noqa: E402
from gemm_i8 import add_gemm_b_int8                             # noqa: E402
from npue import (ASMOOTH_SUFFIX, GSCALE_SUFFIX, WSCALE_SUFFIX,  # noqa: E402
                  MAC_BY_DEVICE, Reader, Writer, dequant_int8,
                  gemm_b_layout, layout_hash, mac_for_device, pack_i4, tile_b,
                  untile_b)

# npu1's sub-tile FOR AN INT8 OPERAND, which is what an int4 container's B
# panel is tiled for -- int4 widens to int8 before staging, so the bytes the
# array multiplies are int8's. This used to be the literal (8, 4), which is
# npu1's *bf16* pair: self-consistent, so every round trip below still passed,
# but no longer the bytes a shipped int4 container carries. Section 8 now
# re-measures the table, so the two cannot drift apart again.
MAC = mac_for_device("npu1", "I8")
TILE_K, TILE_N = 64, 32
SHAPE = (128, 96)          # K, N -- K a multiple of 64, N of 32, neither square
PAD_COL = 64               # stands in for the fused qkv's N-axis zero padding
GROUP = 32                 # the default --int4-group

_failures = []


def report(ok, label, detail=""):
    print(f"   {'ok  ' if ok else 'FAIL'}  {label}"
          + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def expect_clean(problems, label):
    report(not problems, label, "" if not problems else f"-> {problems[0][:110]}")


def expect_caught(problems, needle, label):
    """A fault-injection case PASSES only if the gate NAMED the fault. A gate
    that fails for the wrong reason is not a gate: the operator has to know
    which invariant broke, or they will go looking in the wrong place."""
    hit = any(needle in p for p in problems)
    report(hit, label, "" if hit else f"-> not caught; got {problems[:1]}")


def expect_refused(err, needle, label, problems=None):
    """The same rule for a fault the READER refuses before the gate runs. Two
    implementations of one order means two places a fault can be caught, and
    which one caught it is what this function is asserting -- a reader that
    refuses and a gate that would have agreed with it are not the same claim."""
    hit = err is not None and needle in err
    report(hit, label,
           "" if hit else f"-> not caught; reader said {err!r}, "
                          f"gate saw {problems!r}")


def sample(k=SHAPE[0], n=SHAPE[1], seed=7, zero_col=None):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((k, n), dtype=np.float32) * np.float32(0.05)
    if zero_col is not None:
        w[:, zero_col] = 0.0
    return w


def sample_smooth(k=SHAPE[0], seed=11):
    """An asmooth in the range calibration actually produces: positive, spread
    over more than an order of magnitude. A vector of ones would make every
    wrong-axis fault below invisible -- and `--smooth-alpha 0` really does ship
    ones, which is why section 4 covers it as a case of its own."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(-1.5, 1.5, size=k)).astype(np.float32)


def e_fmt(x):
    return f"{x:.3e}"


# -- the emitter, mirrored ---------------------------------------------------

def scheme(src, asmooth, group=GROUP):
    """`add_gemm_b_int4`'s arithmetic, in its order, with every intermediate
    the injector needs.

    Mirrored rather than called because the injector has to move ONE of these
    quantities without the others following it -- which is precisely the bug
    a packer can have. If this copy and the real emitter drift, the honest
    section above compares against the real one and fails first.
    """
    m = np.ascontiguousarray(src, np.float32) * asmooth[:, None]
    K, N = m.shape

    scale = np.abs(m).max(axis=0) / 127.0
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)

    g = int(group or K)
    G = -(-K // g)
    padded = np.zeros((G * g, N), dtype=np.float32)
    padded[:K] = m
    blocks = padded.reshape(G, g, N)
    s4 = np.abs(blocks).max(axis=1) / 7.0
    s4 = np.where(s4 > 0, s4, np.float32(1.0)).astype(np.float32)
    t = np.rint(blocks / s4[:, None, :]).clip(-7, 7).astype(np.int8)
    t = t.reshape(G * g, N)[:K]
    gscale = (s4 / scale[None, :]).astype(np.float32)
    return t, scale, gscale, blocks, s4, g, G


def stage_i4(w, name, t, wscale, gscale, asmooth, layout=True):
    """Stage a panel with EXPLICIT quantities -- i.e. what a buggy packer emits.
    `add_gemm_b_int4` derives them itself and so cannot produce these."""
    K, N = t.shape
    lay = gemm_b_layout(TILE_K, TILE_N, MAC[0], MAC[1], dtype="I8")
    kw = {"layout": lay} if layout else {}
    w.add(name, pack_i4(tile_b(t, TILE_K, TILE_N, MAC[0], MAC[1])), "I4",
          "gemm_b", [K, N], **kw)
    w.add(name + WSCALE_SUFFIX, wscale, "F32", "quant_scale", [N])
    if gscale is not None:
        w.add(name + GSCALE_SUFFIX, gscale, "F32", "quant_group",
              [gscale.shape[0], N])
    w.add(name + ASMOOTH_SUFFIX, asmooth, "F32", "quant_smooth", [K])


def writer(group=GROUP, config_group=None):
    return Writer({"arch": "probe", "a_dtype": "i8",
                   "int4_group": group if config_group is None else
                   config_group})


@contextlib.contextmanager
def container(w):
    """Write the container and open it again -- a round trip through the real
    format, not an inspection of the Writer's own buffers."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.npue"
        w.write(str(path))
        r = Reader(str(path))
        try:
            yield r, path
        finally:
            r.close()


def panel2d(r, name):
    """The stored panel widened and de-tiled: a [K, N] int8 matrix.

    `Reader.panel()` is what stage() hands the array (flat, tiled); this adds
    the untile that turns it back into the matrix the gate judges, exactly as
    verify_i8_scheme's `panel()` does for an I8 entry -- where the payload is
    already int8 and only the de-tile is needed.
    """
    e = r.entries[name]
    K, N = e["padded_shape"]
    lay = e["layout"]
    flat = r.panel(name) if e["dtype"] == "I4" else r.raw(name)
    q = untile_b(flat, K, N, lay["tile_k"], lay["tile_n"],
                 lay["mac_s"], lay["mac_t"])
    kl, nl = e["logical_shape"]
    return q[:kl, :nl]


# -- 1. the honest path ------------------------------------------------------

def test_honest():
    print("1. an honestly packed int4 operand")
    src, asmooth = sample(), sample_smooth()
    name = "layer.0.ffn_up"
    w = writer()
    packed_err = add_gemm_b_int4(w, name, src, TILE_K, TILE_N,
                                 asmooth=asmooth, group=GROUP, mac=MAC)
    with container(w) as (r, _):
        e = r.entries[name]
        report(e["dtype"] == "I4", "the payload is stored as I4",
               f"got {e['dtype']}")
        report(e["nbytes"] == SHAPE[0] * SHAPE[1] // 2,
               "two values per byte",
               f"{e['nbytes']} == {SHAPE[0] * SHAPE[1] // 2}")
        report(r.config.get("int4_group") == GROUP,
               "config states int4_group", f"got {r.config.get('int4_group')}")

        # THE PAIRING INVARIANT. `layout` describes the panel the ARRAY
        # consumes, i.e. the widened one, so its hash must be the int8 hash
        # and every existing int8 design must accept this container without
        # being re-exported. The entry's own dtype says how the bytes are
        # stored. If these two ever swapped, an int4 container would refuse
        # every design and an int8 one would silently run int4 bytes.
        i8_hash = layout_hash(gemm_b_layout(TILE_K, TILE_N, MAC[0], MAC[1],
                                            dtype="I8"))
        bf_hash = layout_hash(gemm_b_layout(TILE_K, TILE_N, MAC[0], MAC[1],
                                            dtype="BF16"))
        report(e["layout"]["dtype"] == "I8" and e["layout_hash"] == i8_hash,
               "the LAYOUT is the int8 one, so int8 designs accept it",
               f"dtype {e['layout']['dtype']}, hash {e['layout_hash'][:16]}…")
        report(e["layout_hash"] != bf_hash,
               "...and it is not the bf16 layout")

        q = panel2d(r, name)
        problems, rel = check_i4_operand(
            name, q, r.tensor(name + WSCALE_SUFFIX),
            r.tensor(name + GSCALE_SUFFIX), r.tensor(name + ASMOOTH_SUFFIX),
            src, GROUP)
        expect_clean(problems, "check_i4_operand accepts it")
        report(abs(rel - packed_err) < 1e-12,
               "the gate reproduces the packer's own rel_fro",
               f"gate {rel:.6e} vs packer {packed_err:.6e}")

        # The reader hands back a WEIGHT, not a panel, and the weight is
        # `q * wscale / asmooth` -- the same expression an I8 container's
        # reader evaluates, because widening a nibble is supposed to land in
        # int8's units exactly.
        got = r.tensor(name)
        wsc = r.tensor(name + WSCALE_SUFFIX)
        asm = r.tensor(name + ASMOOTH_SUFFIX)
        report(got.dtype == np.float32
               and np.array_equal(got, dequant_int8(q, wsc, asm)),
               "Reader.tensor dequantises to q*wscale/asmooth")

        rel_src = float(np.linalg.norm(got - src) / np.linalg.norm(src))
        # THE BAR IS A RATIO, not a constant, and there are two reasons. It is
        # `6 * the operand error` because that is how much a LARGER error the
        # source domain costs for both schemes: the reader divides by asmooth,
        # this sample's asmooth spans [0.23, 4.47], and int8 measures 3.6x its
        # own operand error the same way int4 measures 3.3x here. It is not
        # 0.25 like verify_i8_scheme's because int4's step is s4 = max/7 where
        # int8's is max/127 -- roughly 18x coarser, which is the format.
        #
        # What makes it a bar rather than a shrug is that BOTH ways of getting
        # asmooth wrong clear it, by 1.4x and 6x, so a reader that skipped the
        # division fails this line and only this line can fail it.
        wrong_nodiv = float(np.linalg.norm(
            (src * asm[:, None]) - src) / np.linalg.norm(src))
        wrong_mult = float(np.linalg.norm(got * asm[:, None] ** 2 - src)
                           / np.linalg.norm(src))
        bar = 6.0 * packed_err
        report(rel_src < bar,
               "Reader.tensor hands back the SOURCE weights, not the operand",
               f"rel_fro {rel_src:.4e} = {rel_src / packed_err:.1f}x the "
               f"{e_fmt(packed_err)} operand error; not dividing by asmooth "
               f"would give {wrong_nodiv:.3e}, multiplying gives "
               f"{wrong_mult:.3e}")
        report(wrong_nodiv > bar and wrong_mult > bar,
               "...and that bar sits ABOVE both ways of getting it wrong",
               f"bar {bar:.3e} vs {wrong_nodiv:.3e} and {wrong_mult:.3e}")

        # The widening must actually HAPPEN. Packing int4 and then reading back
        # int8's panel unchanged is a file that reports int4 and stores int8 --
        # half the size, all the accuracy, and no check in this container would
        # object to it.
        w8 = Writer({"arch": "probe", "a_dtype": "i8"})
        add_gemm_b_int8(w8, name, src, TILE_K, TILE_N, asmooth=asmooth, mac=MAC)
        with container(w8) as (r8, _):
            q8 = panel2d(r8, name)
            report(not np.array_equal(q, q8),
                   "the int4 panel is NOT the int8 panel it pairs with",
                   f"{int((q != q8).sum())} of {q.size} values differ")
            drift = float(np.linalg.norm(
                (q.astype(np.float32) - q8.astype(np.float32)) * wsc[None, :]
            ) / np.linalg.norm(q8.astype(np.float32) * wsc[None, :]))
            report(drift < 0.25,
                   "...but it is that panel, to within the int4 error",
                   f"dequantised rel_fro {drift:.4e}")


# -- 2. the group sizes ------------------------------------------------------

def test_groups():
    print("\n2. every group size, including the two edges")
    src, asmooth = sample(), sample_smooth()
    K, N = SHAPE
    for g in (32, 16, 64, 0, 256):
        name = "layer.0.qkv"
        w = writer(g)
        add_gemm_b_int4(w, name, src, TILE_K, TILE_N,
                        asmooth=asmooth, group=g, mac=MAC)
        want_g = -(-K // (g or K))
        with container(w) as (r, _):
            gs = r.tensor(name + GSCALE_SUFFIX)
            problems, _ = check_i4_operand(
                name, panel2d(r, name), r.tensor(name + WSCALE_SUFFIX), gs,
                r.tensor(name + ASMOOTH_SUFFIX), src, g)
            where = {32: "the default", 16: "a finer group",
                     64: "coarser than the default",
                     0: "per-channel (one group over all K)",
                     256: "LARGER than K, still one group"}[g]
            expect_clean(problems, f"group={g:<3} {where}")
            report(gs.shape == (want_g, N),
                   f"           .gscale is [{want_g}, {N}] for group {g}",
                   f"got {gs.shape}")
            report(r.config.get("int4_group") == g,
                   f"           config int4_group == {g}",
                   f"got {r.config.get('int4_group')}")


# -- 3. a zero padding column ------------------------------------------------

def test_zero_column():
    print("\n3. a zero padding column stays zero")
    # pack_npue.py pads the fused qkv with np.zeros and the host slices Q/K/V
    # off the front by offset. A pad column that picked up a value through
    # quantisation would put that value inside the next tensor's window.
    src, asmooth = sample(zero_col=PAD_COL), sample_smooth()
    w = writer()
    add_gemm_b_int4(w, "layer.0.qkv", src, TILE_K, TILE_N,
                    asmooth=asmooth, group=GROUP, mac=MAC)
    with container(w) as (r, _):
        q = panel2d(r, "layer.0.qkv")
        wscale = r.tensor("layer.0.qkv" + WSCALE_SUFFIX)
        gscale = r.tensor("layer.0.qkv" + GSCALE_SUFFIX)
        # The dead COLUMN is dead as whole (group, column) BLOCKS, which is
        # check_i4_operand's invariant 5 -- not merely as a flat column, since
        # a group holding it may straddle live rows for a coarse group.
        report(bool(np.all(q[:, PAD_COL] == 0)) and float(wscale[PAD_COL]) == 1.0,
               "an all-zero column packs to zeros with its scale held at 1",
               f"max|q| {int(np.abs(q[:, PAD_COL]).max())}, "
               f"scale {float(wscale[PAD_COL]):.1f}")
        report(bool(np.all(gscale[:, PAD_COL] == 1.0)),
               "and its group factors are 1 (s4 held at 1 too)",
               f"distinct values {np.unique(gscale[:, PAD_COL]).tolist()[:4]}")
        problems, _ = check_i4_operand(
            "layer.0.qkv", q, wscale, gscale,
            r.tensor("layer.0.qkv" + ASMOOTH_SUFFIX), src, GROUP)
        expect_clean(problems, "check_i4_operand accepts it")


# -- 4. unsmoothed -----------------------------------------------------------

def test_no_smoothing():
    print("\n4. an unsmoothed panel is self-consistent, not a broken one")
    # --smooth-alpha 0 writes .asmooth as all ones; the container is then a
    # plain per-group quantisation and the gate must accept it rather than
    # mistake the missing smoothing for a missing tensor. int4 needs this case
    # as much as int8 does: with ones, `.gscale` becomes s4/s8 with both taken
    # from the same matrix, and a fold that skipped it would still be right.
    src = sample()
    w = writer()
    add_gemm_b_int4(w, "layer.0.qkv", src, TILE_K, TILE_N,
                    group=GROUP, mac=MAC)
    with container(w) as (r, _):
        asm = r.tensor("layer.0.qkv" + ASMOOTH_SUFFIX)
        report(bool(np.all(asm == 1.0)), ".asmooth ships as ones when unsmoothed")
        problems, _ = check_i4_operand(
            "layer.0.qkv", panel2d(r, "layer.0.qkv"),
            r.tensor("layer.0.qkv" + WSCALE_SUFFIX),
            r.tensor("layer.0.qkv" + GSCALE_SUFFIX), asm, src, GROUP)
        expect_clean(problems, "check_i4_operand accepts it")


# -- 5. fault injection ------------------------------------------------------

def build_fault(pre_smooth=False, half_wscale=False, per_tensor=False,
                gs=None, coarse_t=False, wrong_quant_asm=False,
                break_pad=False, group=GROUP, config_group=None,
                drop_gscale=False, drop_layout=False):
    """Stage ONE container carrying exactly one fault. Returns (writer, src).

    Each parameter moves a single quantity out from under the others, which is
    the whole difficulty: a fault that also changed `.gscale` would be caught
    by the group rule for the wrong reason and the gate would have proved
    nothing about the quantity under test.
    """
    src = sample(zero_col=PAD_COL)
    asm = sample_smooth()
    quant_asm = asm[::-1].copy() if wrong_quant_asm else asm
    t, wscale, gscale, blocks, s4, g, G = scheme(src, quant_asm, group)
    K, N = src.shape

    if pre_smooth:
        s = np.abs(src).max(axis=0) / 127.0
        wscale = np.where(s > 0, s, np.float32(1.0)).astype(np.float32)
    elif half_wscale:
        wscale = (wscale * np.float32(0.5)).astype(np.float32)
    elif per_tensor:
        v = float(np.abs(src * quant_asm[:, None]).max()) / 127.0
        wscale = np.full(N, max(v, 1e-30), dtype=np.float32)

    if gs == "columns":          # the factor indexed along N instead of K
        gscale = gscale[:, ::-1].copy()
    elif gs == "groups":         # group 0's factor handed to the last group
        gscale = gscale[::-1].copy()
    elif gs == "small":          # a factor that is simply off
        gscale = (gscale * np.float32(0.95)).astype(np.float32)

    if coarse_t:
        # The nibbles rounded at half the scale they claim. `.wscale` and
        # `.gscale` both stay exactly right, so only the payload comparison
        # can see it -- which is the argument for having one that is exact.
        t = np.clip(np.rint(blocks / (s4 * np.float32(0.5))[:, None, :]),
                    -7, 7).astype(np.int8).reshape(G * g, N)[:K]

    if break_pad:
        # A padding column that came back non-zero. Its own group is all-zero,
        # so s4 and s8 are both held at 1 and gscale is 1 -- setting t alone
        # is enough to put a value where the host's offset slice will read it.
        t[:, PAD_COL] = 7

    w = writer(group, config_group=config_group)
    stage_i4(w, "layer.0.qkv", t, wscale,
             None if drop_gscale else gscale, asm, layout=not drop_layout)
    return w, src


def inject(**kw):
    """Open an injected container and return (problems, read_error).

    `problems` is check_i4_operand's list when the reader accepted the file;
    `read_error` is the reader's message when it refused first, and then
    `problems` is []. Which of the two happened is part of what each fault
    tests, which is why there are two different expect_* helpers rather than
    one that would pass for either.
    """
    w, src = build_fault(**kw)
    name = "layer.0.qkv"
    group = kw.get("group", GROUP)
    with container(w) as (r, _):
        try:
            q = panel2d(r, name)
        except Exception as e:                      # noqa: BLE001 - the message is the assertion
            return [], str(e)
        problems, _ = check_i4_operand(
            name, q, r.tensor(name + WSCALE_SUFFIX),
            r.tensor(name + GSCALE_SUFFIX), r.tensor(name + ASMOOTH_SUFFIX),
            src, group)
    return problems, None


def test_faults():
    print("\n5. ten ways a packer can be wrong, each injected on purpose")
    print("   (a passing line here is a line where the GATE FIRED)")

    # 1. the scale taken before the asmooth multiply -- an order slip that
    #    moves the number by a few percent and trips nothing else.
    problems, _ = inject(pre_smooth=True)
    expect_caught(problems, "disagrees with max|W*asmooth|/127",
                  "scale computed before the asmooth multiply")

    # 2. per-tensor. int4 has no invariant of its own against it -- the group
    #    rule would still hold, since gscale is a RATIO and both halves move
    #    together -- so it is the int8 scale rule underneath that catches it.
    problems, _ = inject(per_tensor=True)
    expect_caught(problems, "disagrees with max|W*asmooth|/127",
                  "per-tensor scale instead of per-output-channel")

    # 3. a stored scale too small. The payload then saturates against the
    #    file's own rule, which the scale check takes as read.
    problems, _ = inject(half_wscale=True)
    expect_caught(problems, "disagrees with max|W*asmooth|/127",
                  "a stored scale too small for the weights it scales")

    # 4-6. three ways for the GROUP FACTOR -- the one quantity int4 adds --
    #     to be wrong while everything else stays self-consistent. Each keeps
    #     the shape [.gscale] promises, so the shape check is blind to all
    #     three and only the rule that recomputes s4/s8 can name them.
    problems, _ = inject(gs="columns")
    expect_caught(problems, "gscale disagrees with (max over the group of",
                  "the factor indexed along N instead of K")

    problems, _ = inject(gs="groups")
    expect_caught(problems, "gscale disagrees with (max over the group of",
                  "the factor handed to the wrong group")

    problems, _ = inject(gs="small")
    expect_caught(problems, "gscale disagrees with (max over the group of",
                  "a group factor that is simply off by 5%")

    # 7. the payload rounded at HALF the scale it claims. `.wscale` and
    #    `.gscale` are both exactly right, so every scale rule in the gate is
    #    satisfied and only the exact comparison against the arithmetic finds
    #    it -- this is the fault a tolerance would round away.
    problems, _ = inject(coarse_t=True)
    expect_caught(problems, "int8 values are not rint(rint(W*asmooth/s4) * "
                            "gscale)",
                  "payload rounded below the scale stored beside it")

    # 8. the smoothing vector applied in the WRONG ORDER to the payload while
    #    the file's own `.asmooth` is right. Nothing in the container disagrees
    #    with itself except the payload, which was rounded from another matrix.
    problems, _ = inject(wrong_quant_asm=True)
    expect_caught(problems, "int8 values are not rint(rint(W*asmooth/s4) * "
                            "gscale)",
                  "payload rounded from a different asmooth than the file's")

    # 9. padding that did not stay padding.
    problems, _ = inject(break_pad=True)
    expect_caught(problems, "all-zero (group, column) blocks came back "
                            "non-zero",
                  "a zero padding column that came back non-zero")

    # 10. THE ONE ONLY THE READER CAN CATCH. config says groups of 16 rows and
    #     `.gscale` has 8 rows. Every value in the file is individually
    #     correct for SOME group size; there is simply no single one they all
    #     agree on, and folding at either would weight rows by a neighbour's
    #     scale. The reader refuses before the panel exists.
    problems, err = inject(config_group=16)
    expect_refused(err, "group rows but K=",
                   "config int4_group disagreeing with .gscale's rows",
                   problems)


# -- 6. refuse, or go loud ---------------------------------------------------

def test_refusals():
    print("\n6. refuse, or go loud: the two outcomes a gate must tell apart")

    # Two values per byte is a shape claim, not a truncation. A packer that
    # clipped instead of refusing would drop the high bits of a 127 and turn
    # an int8 payload into plausible int4 garbage.
    try:
        pack_i4(np.array([0, 1, 2], np.int8))
        report(False, "an odd element count is refused", "-> it packed")
    except ValueError as e:
        report("cannot be paired" in str(e),
               "an odd element count is refused", f"-> {str(e)[:70]}")
    try:
        pack_i4(np.array([0, 127], np.int8))
        report(False, "a value outside [-8, 7] is refused", "-> it packed")
    except ValueError as e:
        report("[-8, 7]" in str(e),
               "a value outside [-8, 7] is refused", f"-> {str(e)[:70]}")

    # raw() on an I4 entry would hand back K*N/2 elements for a panel of K*N,
    # silently -- half a panel, of a neighbour's nibbles.
    w = writer()
    add_gemm_b_int4(w, "layer.0.qkv", sample(), TILE_K, TILE_N,
                    group=GROUP, mac=MAC)
    with container(w) as (r, _):
        try:
            r.raw("layer.0.qkv")
            report(False, "Reader.raw refuses an I4 payload", "-> it returned")
        except ValueError as e:
            report("packed two int4" in str(e),
                   "Reader.raw refuses an I4 payload", f"-> {str(e)[:70]}")

    # A payload with no group scale has no way back to an int8 value at all.
    w, _ = build_fault(drop_gscale=True)
    with container(w) as (r, _):
        try:
            r.panel("layer.0.qkv")
            report(False, "an I4 payload with no .gscale raises", "-> it returned")
        except KeyError as e:
            report(".gscale is not in the container" in str(e),
                   "an I4 payload with no .gscale raises", f"-> {str(e)[:70]}")

    # No layout means no order to unpack into: the right count in an unknown
    # sequence. Refused by name rather than inferred.
    w, _ = build_fault(drop_layout=True)
    with container(w) as (r, _):
        try:
            r.panel("layer.0.qkv")
            report(False, "an I4 payload with no layout raises", "-> it returned")
        except ValueError as e:
            report("no block_panel layout" in str(e),
                   "an I4 payload with no layout raises", f"-> {str(e)[:70]}")

    # A negative group is not a group.
    try:
        add_gemm_b_int4(Writer({}), "huge", np.zeros(SHAPE, np.float32),
                        TILE_K, TILE_N, group=-1, mac=MAC)
        report(False, "a negative int4_group is refused", "-> it packed")
    except SystemExit as e:
        report("not a positive row count" in str(e),
               "a negative int4_group is refused", f"-> {str(e)[:70]}")

    # The accumulator is exact only while it cannot overflow, and the guard
    # must be reachable rather than dead behind the tiling check.
    k_overflow = 2 ** 31 // (127 * 127) + 1
    try:
        add_gemm_b_int4(Writer({}), "huge",
                        np.zeros((k_overflow, 1), np.float32),
                        TILE_K, TILE_N, group=GROUP, mac=MAC)
        report(False, "a K that overflows the int32 accumulator is refused",
               f"-> it packed K={k_overflow}")
    except SystemExit as e:
        report("int32 accumulator" in str(e),
               "a K that overflows the int32 accumulator is refused",
               f"-> K={k_overflow}: {str(e)[:58]}")
    except ValueError as e:
        report(False, "a K that overflows the int32 accumulator is refused",
               f"-> tiling raised first, so the guard is unreachable: {e}")

    # And the widening itself: a gate handed the nibbles rather than the panel
    # must say so rather than compare K*N/2 values against a matrix of K*N.
    _, asmooth = sample(), sample_smooth()
    problems, _ = check_i4_operand("layer.0.qkv", pack_i4(
        tile_b(np.zeros(SHAPE, np.int8), TILE_K, TILE_N, MAC[0], MAC[1])),
        np.ones(SHAPE[1], np.float32), np.ones((4, SHAPE[1]), np.float32),
        asmooth, sample(), GROUP)
    expect_caught(problems, "expected int8 -- an I4 container holds nibbles",
                  "a still-packed payload handed to the gate is named")


# -- 7. the C++ decode -------------------------------------------------------

def probe_path(tmp):
    """Compile int4_panel_probe. Returns (path, None) or (None, message).

    The gate's own `needs` says g++ is present (pipeline.py's "cheap" tier
    promises it), so a missing compiler here is a broken promise rather than
    an empty one -- reported as a failure, not skipped, because a parity gate
    that silently skips the half it cannot check is a parity gate.
    """
    cxx = os.environ.get("CXX", "g++")
    if shutil.which(cxx) is None:
        return None, f"{cxx} not found (set CXX)"
    out = Path(tmp) / "int4_panel_probe"
    cmd = [cxx, "-std=c++17", "-O2",
           "-I", str(REPO / "runtime" / "include"),
           str(REPO / "tools" / "verify" / "int4_panel_probe.cpp"),
           str(REPO / "runtime" / "src" / "model.cpp"),
           "-o", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        return None, f"{cxx} failed: {p.stderr.strip()[:400]}"
    return out, None


def run_probe(probe, path, names, bf16_design=False):
    cmd = [str(probe), str(path), str(path) + ".probe"]
    if bf16_design:
        cmd.append("--bf16-design")
    cmd += list(names)
    return subprocess.run(cmd, capture_output=True, text=True)


def test_subtile_table():
    """The (s, t) table against the compiler, not against a comment.

    This is the gate for a defect that passed every other check in the tree.
    The B panel's byte order inside a tile IS the MMAC sub-tile, and the
    sub-tile is a function of the OPERAND DTYPE as well as the board: on npu1
    bf16 is (8, 4) and int8 is (8, 8). Every layer here -- the exporter, the
    packer, the runtime's layout_hash comparison, verify_design_numerics --
    read the same device-keyed table, so an int8 container packed with the
    bf16 pair produced a design whose every product was wrong while the hash
    matched on both sides and every gate agreed with every other gate. Nothing
    in the format can see it, because both halves were consistently wrong.

    So the table is not taken on trust: it is re-measured here, from
    aie.iron.kernels.mm, which is the same call the exporter's geometry makes.
    A toolchain that changes its sub-tile now fails this gate instead of
    shipping.

    Only the LOCAL device's row can be measured -- kernels.mm has no device
    argument, it reports whatever the compiler is configured for -- so the
    other generation is reported as unchecked rather than assumed correct.
    """
    print("\n8. the MMAC sub-tile table, measured rather than trusted")
    try:
        from aie.iron import kernels, str_to_dtype
    except Exception as exc:  # no XRT / no mlir-aie on this machine
        report(False, "aie.iron.kernels.mm is importable",
               f"-> {type(exc).__name__}: {exc}. Source /opt/xilinx/xrt/setup.sh; "
               f"this gate cannot vouch for the sub-tile without it.")
        return

    local = _local_device()
    if local is None:
        report(False, "the local device's bf16 sub-tile matches some known row",
               f"the compiler reports {_measured_bf16()} but MAC_BY_DEVICE "
               f"holds { {d: v['bf16'] for d, v in MAC_BY_DEVICE.items()} }. "
               f"If the bf16 row moved, a toolchain changed under the table -- "
               f"fix npue.MAC_BY_DEVICE from the number above.")
        return

    # Every tile width this project builds. The sub-tile is constant in n, and
    # that is the claim being made, so check more than one.
    for dtype, out, want_py in (("bf16", "f32", "bf16"), ("i8", "i32", "i8")):
        want = MAC_BY_DEVICE[local][want_py]
        got = _measured_mac(dtype, out)
        report(got == want,
               f"{local} + {dtype}: table says {want}, silicon says "
               f"{got if got else sorted(seen)}",
               "" if got == want else
               f"  FIX npue.MAC_BY_DEVICE[{local!r}][{want_py!r}] to {got}. "
               f"A wrong pair here misreads every panel while the layout_hash "
               f"still matches on both sides.")

    others = sorted(d for d in MAC_BY_DEVICE if d != local)
    if others:
        print(f"   --   not measurable here: {', '.join(others)} "
              f"(kernels.mm reports the local device only; those rows are "
              f"documentation until someone runs this gate on that board)")


def _measured_mac(dtype, out):
    """(mac_s, mac_t) the compiler reports for `dtype`, over every tile width
    this project builds. A single answer means the sub-tile really is constant
    in n, which is the assumption the whole table rests on."""
    from aie.iron import kernels, str_to_dtype

    seen = set()
    for n in (16, 32, 48, 64):
        mk = kernels.mm(dim_m=64, dim_k=96, dim_n=n,
                        input_dtype=str_to_dtype(dtype),
                        output_dtype=str_to_dtype(out), b_col_maj=False,
                        c_col_maj=False, use_chess=False,
                        emulate_bf16_mmul_with_bfp16=False, vectorized=True)
        seen.add(tuple(mk.mac_dims[1:]))
    return seen.pop() if len(seen) == 1 else None


def _measured_bf16():
    return _measured_mac("bf16", "f32")


def _local_device():
    """Which row of MAC_BY_DEVICE this machine's compiler is configured for.

    Matched on the bf16 row ALONE, deliberately. Matching on both rows made a
    single wrong row look like an unrecognised device, and the gate then
    complained about the device instead of naming the entry that was wrong --
    which is the one fact the person reading it needs. bf16 is the right row to
    identify by: it is the one every shipped design set was measured against.

    Asked of the compiler rather than the environment: the value that matters
    is the one the exporter bakes into a design, and that is what kernels.mm
    reports.
    """
    bf16 = _measured_bf16()
    if bf16 is None:
        return None
    for dev, by_dtype in MAC_BY_DEVICE.items():
        if by_dtype.get("bf16") == bf16:
            return dev
    return None


def test_cpp():
    print("\n7. the C++ decode, compiled and compared byte for byte")
    print("   (the order npue.fold_i4 and common/int4_panel.hpp each define)")
    with tempfile.TemporaryDirectory() as tmp:
        probe, err = probe_path(tmp)
        if probe is None:
            report(False, "int4_panel_probe compiles", f"-> {err}")
            return
        report(True, "int4_panel_probe compiles")

        # One container per group size, each carrying an int4 GEMM operand
        # AND a plain BF16 tensor, so the non-decode path (a Span straight
        # into the mapping) is held too.
        for g in (32, 16, 0, 256):
            w, _ = build_fault(group=g)
            w.add("layernorm.weight", np.zeros(16, np.uint16), "BF16",
                  "layernorm", [16])
            with container(w) as (r, path):
                names = ["layer.0.qkv", "layernorm.weight"]
                p = run_probe(probe, path, names)
                if p.returncode:
                    report(False, f"C++ agrees with Python at group={g}",
                           f"-> {p.stdout.strip().splitlines()[-1:]}")
                    continue
                same = True
                for i, n in enumerate(names):
                    want = np.ascontiguousarray(r.panel(n)).tobytes()
                    got = (Path(str(path) + f".probe.{i}")).read_bytes()
                    if want != got:
                        j = next((k for k, (a, b) in
                                  enumerate(zip(want, got)) if a != b),
                                 min(len(want), len(got)))
                        report(False, f"C++ agrees with Python at group={g}",
                               f"{n}: first difference at byte {j}")
                        same = False
                        break
                if same:
                    report(True, f"C++ agrees with Python at group={g}",
                           f"{len(names)} operands, byte-identical")

        # THE REFUSALS, which is where a C++ port usually differs from Python:
        # the two sides' error messages and their ORDER of checks. Each fault
        # below is caught by one of them and this section says which.
        cases = [
            (dict(), "--bf16-design", ["layer.0.qkv"],
             "A operand is 2 bytes",
             "I4 staged against a bf16 design"),
            (dict(drop_gscale=True), None, ["layer.0.qkv"],
             ".gscale in the container",
             "I4 payload with no .gscale"),
            (dict(group=32, config_group=None, drop_layout=True), None,
             ["layer.0.qkv"],
             "no usable block_panel layout",
             "I4 payload with no layout"),
            (dict(config_group=16), None, ["layer.0.qkv"],
             "disagree about the group size",
             "config int4_group disagreeing with .gscale"),
        ]
        for kw, flag, names, needle, label in cases:
            w, _ = build_fault(**kw)
            with container(w) as (r, path):
                p = run_probe(probe, path, names, bf16_design=bool(flag))
                out = p.stdout
                hit = p.returncode == 1 and needle in out
                report(hit, f"C++ refuses: {label}",
                       "" if hit else f"-> rc={p.returncode} "
                                      f"{out.strip().splitlines()[-1:]}")

        # A container whose config never mentions int4_group at all -- the
        # case where the group size has to be DERIVED and must not be guessed.
        src, asm = sample(), sample_smooth()
        t, wscale, gscale, *_ = scheme(src, asm, GROUP)
        w = Writer({"arch": "probe", "a_dtype": "i8"})   # no int4_group key
        stage_i4(w, "layer.0.qkv", t, wscale, gscale, asm)
        with container(w) as (r, path):
            p = run_probe(probe, path, ["layer.0.qkv"])
            hit = p.returncode == 1 and "states no int4_group" in p.stdout
            report(hit, "C++ refuses: no int4_group in config",
                   "" if hit else f"-> rc={p.returncode} {p.stdout.strip()}")


def main():
    print("int4 scheme self-check (numpy + a C++17 compiler, no NPU)\n")
    test_honest()
    test_groups()
    test_zero_column()
    test_no_smoothing()
    test_faults()
    test_refusals()
    test_cpp()
    test_subtile_table()
    print()
    if _failures:
        print(f"FAIL -- {len(_failures)} of the lines above did not hold:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("PASS -- the scheme round-trips at every group size, every injected\n"
          "        fault was named by the gate that exists to name it, and the\n"
          "        C++ decode is byte-identical to Python's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
