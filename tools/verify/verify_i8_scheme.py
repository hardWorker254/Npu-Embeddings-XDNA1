# NpuEmbeddings -- the int8 scheme, checked against itself.
# SPDX-License-Identifier: Apache-2.0
#
# WHY THIS EXISTS ALONGSIDE tools/verify/verify_npue.py. That gate answers "is this
# container the right container for this checkpoint", and it answers it by
# running a Python encoder against goldens -- so it needs `reference/`, the
# oracle and the goldens. This one answers the narrower question that needs
# none of that: does the int8 scheme in tools/lib/gemm_i8.py round-trip through
# the container format, and does the gate in that module actually CATCH the
# five ways a packer can be wrong?
#
# The second half is the point. A gate that has only ever been run on a correct
# container is a gate whose sensitivity is unknown: "PASS" on a good file
# proves nothing if the same code would also pass a bad one. So every check is
# run twice -- once on an honestly packed panel, once on a deliberately broken
# one -- and this file FAILS if a broken panel is not reported.
#
# These are the bugs, each of which packs cleanly, runs, and returns embeddings:
#
#   1. the scale computed BEFORE the asmooth multiply (an order slip)
#   2. a per-tensor scale instead of per-output-channel
#   3. asmooth folded on the N axis instead of K -- a shape-valid wrong product
#   4. a scale too small, so the payload saturates at the clip
#   5. a zero padding column that came back non-zero
#
# plus the two outcomes a gate has to tell apart: a refusal (int32 accumulator
# overflow) and a reader bug (a panel handed back with no scale, which is a
# weight 1/127 of the right size and still looks like a number).
#
# Env: numpy only. Usage:
#   python tools/verify/verify_i8_scheme.py

import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

from gemm_i8 import add_gemm_b_int8, check_i8_operand          # noqa: E402
from npue import (ASMOOTH_SUFFIX, WSCALE_SUFFIX, Reader, Writer,             # noqa: E402
                  dequant_int8, gemm_b_layout, layout_hash, tile_b, untile_b)

# npu1's sub-tile, not gemm_i8's (8, 8) MAC_DEFAULT: that is npu2's, and using
# it here would test a byte order this project does not ship.
MAC = (8, 4)
TILE_K, TILE_N = 64, 32
SHAPE = (128, 96)          # K, N -- K a multiple of 64, N of 32, neither square
PAD_COL = 64               # stands in for the fused qkv's N-axis zero padding

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


def sample(k=SHAPE[0], n=SHAPE[1], seed=7, zero_col=None):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((k, n), dtype=np.float32) * np.float32(0.05)
    if zero_col is not None:
        w[:, zero_col] = 0.0
    return w


def e_fmt(x):
    return f"{x:.3e}"


def mult_err(got, src, asmooth):
    """rel_fro the reader WOULD report if it multiplied by asmooth instead of
    dividing.

    Printed beside the real number rather than only asserted, so the gap
    between the right direction and the wrong one is a number on the screen.
    On a real calibrated container that gap was 4.4e-04 against 1.0.
    """
    bad = got * (asmooth[:, None] ** 2)
    return float(np.linalg.norm(bad - src) / np.linalg.norm(src))


def sample_smooth(k=SHAPE[0], seed=11):
    """An asmooth in the range calibration actually produces: positive, spread
    over more than an order of magnitude. A vector of ones would make every
    axis-mixup fault below invisible, which is the reason this file does not
    use one -- and it is not a hypothetical, the packer really does ship ones
    when --smooth-alpha 0 is asked for."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(-1.5, 1.5, size=k)).astype(np.float32)


def stage(w, name, q, wscale, asmooth):
    """Stage a panel with EXPLICIT scales -- i.e. what a buggy packer emits.
    `add_gemm_b_int8` derives both scales itself and so cannot produce these."""
    layout = gemm_b_layout(TILE_K, TILE_N, MAC[0], MAC[1], dtype="I8")
    w.add(name, tile_b(q, TILE_K, TILE_N, MAC[0], MAC[1]), "I8", "gemm_b",
          [q.shape[0], q.shape[1]], layout=layout)
    w.add(name + WSCALE_SUFFIX, wscale, "F32", "quant_scale", [q.shape[1]])
    w.add(name + ASMOOTH_SUFFIX, asmooth, "F32", "quant_smooth", [q.shape[0]])


@contextlib.contextmanager
def container(w):
    """Write the container and open it again -- a round trip through the real
    format, not an inspection of the Writer's own buffers. Yields an OPEN
    Reader; the mapping is closed in the `finally` because np.memmap needs that
    explicitly before the file can be removed on some platforms."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.npue"
        w.write(str(path))
        r = Reader(str(path))
        try:
            yield r
        finally:
            r.close()


def panel(r, name):
    """The stored panel as a [K, N] int8 matrix: de-tiled and cropped, nothing
    else applied. This is what the DMA sees, and what the gate judges."""
    e = r.entries[name]
    K, N = e["padded_shape"]
    q = untile_b(r.raw(name), K, N, e["layout"]["tile_k"], e["layout"]["tile_n"],
                 e["layout"]["mac_s"], e["layout"]["mac_t"])
    kl, nl = e["logical_shape"]
    return q[:kl, :nl]


# -- 1. the honest path ----------------------------------------------------

def test_honest():
    print("1. an honestly packed panel")
    src, asmooth = sample(), sample_smooth()
    w = Writer({"arch": "probe", "a_dtype": "i8"})
    packed_err = add_gemm_b_int8(w, "layer.0.ffn_up", src, TILE_K, TILE_N,
                                 asmooth=asmooth, mac=MAC)
    name = "layer.0.ffn_up"
    with container(w) as r:
        q = panel(r, name)
        wscale = r.tensor(name + WSCALE_SUFFIX)
        got_asm = r.tensor(name + ASMOOTH_SUFFIX)

        problems, rel = check_i8_operand(name, q, wscale, got_asm, src)
        expect_clean(problems, "check_i8_operand accepts it")
        report(abs(rel - packed_err) < 1e-12,
               "the gate reproduces the packer's own rel_fro",
               f"gate {rel:.6e} vs packer {packed_err:.6e}")

        # the reader hands back a WEIGHT, not a panel
        got = r.tensor(name)
        report(got.dtype == np.float32
               and np.array_equal(got, dequant_int8(q, wscale, got_asm)),
               "Reader.tensor dequantises to Wq*wscale/ASMOOTH")

        # THE INVARIANT THAT MATTERS, and the one whose absence let a real bug
        # through: the reader's output must be the SOURCE WEIGHTS, to within
        # the quantisation error. Everything else in this section checks the
        # operand against itself -- panel vs scale vs smoothing -- and a reader
        # that hands back `W * asmooth**2` instead of `W` satisfies every one
        # of those perfectly, because they never mention the source.
        #
        # This is why it must be a non-unit asmooth: multiply-vs-divide is a
        # difference of asmooth**2, so with the packer's all-ones vector the
        # two are the same function and no self-consistency check can tell
        # them apart. That is not hypothetical -- `--smooth-alpha 0` ships
        # ones, so an unsmoothed container would have hidden it too.
        rel = float(np.linalg.norm(got - src) / np.linalg.norm(src))
        report(rel < 0.25,
               "Reader.tensor hands back the SOURCE weights, not the stored "
               "operand",
               f"rel_fro {rel:.4e} against a {e_fmt(packed_err)} quantisation "
               f"error; multiplying by asmooth instead of dividing gives "
               f"{mult_err(got, src, got_asm):.3e}")

        report(np.allclose(got * got_asm[:, None], q * wscale[None, :],
                           rtol=1e-6, atol=0.0),
               "multiplying the smoothing back in recovers the dequantised panel")

        # the operand dtype is part of the layout, which is what makes a
        # mismatched container/design pair refuse instead of running
        e = r.entries[name]
        report(e["layout"]["dtype"] == "I8"
               and e["layout_hash"] == layout_hash(
                   gemm_b_layout(TILE_K, TILE_N, MAC[0], MAC[1], dtype="I8"))
               and e["layout_hash"] != layout_hash(
                   gemm_b_layout(TILE_K, TILE_N, MAC[0], MAC[1], dtype="BF16")),
               "layout_hash carries the operand dtype, so I8 and BF16 cannot pair")


def test_zero_column():
    print("\n2. a zero padding column stays zero")
    # pack_npue.py pads the fused qkv with np.zeros and the host slices Q/K/V off
    # the front by offset. A pad column that picked up a value through
    # quantisation would put that value inside the next tensor's window.
    src, asmooth = sample(zero_col=PAD_COL), sample_smooth()
    w = Writer({"arch": "probe", "a_dtype": "i8"})
    add_gemm_b_int8(w, "layer.0.qkv", src, TILE_K, TILE_N,
                    asmooth=asmooth, mac=MAC)
    with container(w) as r:
        q = panel(r, "layer.0.qkv")
        wscale = r.tensor("layer.0.qkv" + WSCALE_SUFFIX)
        report(bool(np.all(q[:, PAD_COL] == 0)) and float(wscale[PAD_COL]) == 1.0,
               "an all-zero column packs to zeros with its scale held at 1",
               f"max|q| in the column {int(np.abs(q[:, PAD_COL]).max())}, "
               f"scale {float(wscale[PAD_COL]):.1f}")
        problems, _ = check_i8_operand(
            "layer.0.qkv", q, wscale,
            r.tensor("layer.0.qkv" + ASMOOTH_SUFFIX), src)
        expect_clean(problems, "check_i8_operand accepts it")


def test_no_smoothing():
    print("\n3. an unsmoothed panel is self-consistent, not a broken one")
    # --smooth-alpha 0 writes .asmooth as all ones. The container is then a
    # plain per-channel quantisation and the gate must accept it, not mistake
    # the missing smoothing for a missing tensor.
    src = sample()
    w = Writer({"arch": "probe", "a_dtype": "i8"})
    add_gemm_b_int8(w, "layer.0.qkv", src, TILE_K, TILE_N, mac=MAC)
    with container(w) as r:
        asm = r.tensor("layer.0.qkv" + ASMOOTH_SUFFIX)
        report(bool(np.all(asm == 1.0)), ".asmooth ships as ones when unsmoothed")
        problems, _ = check_i8_operand(
            "layer.0.qkv", panel(r, "layer.0.qkv"),
            r.tensor("layer.0.qkv" + WSCALE_SUFFIX), asm, src)
        expect_clean(problems, "check_i8_operand accepts it")


# -- 2. fault injection ----------------------------------------------------

def inject(asm=None, stored_scale=None, q_scale=None, pre_smooth=False,
           halve_stored=False, break_pad=False):
    """Build a container carrying exactly one fault and return the gate's
    verdict on it.

    The two scales are separate parameters on purpose. `stored_scale` is what
    goes into `.wscale` and `q_scale` is what the payload was rounded at, and
    the bug where those two disagree is invisible to every check except the
    payload comparison -- which is the argument for having one.

    `pre_smooth` takes the stored scale from the matrix BEFORE the asmooth
    multiply (an order slip). It cannot be a lambda over the smoothed matrix,
    because the whole point is that it came from a different one.
    """
    src, asmooth = sample(zero_col=PAD_COL), sample_smooth()
    w_eff = src * (asmooth if asm is None else asm)[:, None]

    def rule(m, factor=1.0):
        s = np.abs(m).max(axis=0) / 127.0
        # The dead column is held at 1 BY THE SCHEME. Scaling it too would
        # stage a different fault (a zero scale) and the gate would rightly
        # report that one instead of the one under test.
        return np.where(s > 0, (s * np.float32(factor)).astype(np.float32),
                        np.float32(1.0)).astype(np.float32)

    if pre_smooth:
        s_stored = rule(np.abs(src))
    elif halve_stored:
        s_stored = rule(w_eff, 0.5)
    else:
        s_stored = rule(w_eff) if stored_scale is None else stored_scale(w_eff)
    s_q = rule(w_eff) if q_scale is None else q_scale(w_eff)
    q = np.rint(w_eff / s_q[None, :]).clip(-127, 127).astype(np.int8)
    if break_pad:
        q[0, PAD_COL] = 7                    # padding that did not stay padding
    w = Writer({"arch": "probe", "a_dtype": "i8"})
    stage(w, "layer.0.qkv", q, s_stored, asmooth)
    with container(w) as r:
        return check_i8_operand("layer.0.qkv", panel(r, "layer.0.qkv"),
                                r.tensor("layer.0.qkv" + WSCALE_SUFFIX),
                                r.tensor("layer.0.qkv" + ASMOOTH_SUFFIX), src)


def test_faults():
    print("\n4. seven ways a packer can be wrong, each injected on purpose")
    print("   (a passing line here is a line where the GATE FIRED)")

    # 1. the scale taken before the asmooth multiply -- an order slip that
    #    moves the number by a few percent and trips nothing else
    problems, _ = inject(pre_smooth=True)
    expect_caught(problems, "disagrees with max|W*asmooth|/127",
                  "scale computed before the asmooth multiply")

    # 2. per-tensor. Legal, plausible, and the accuracy claim does not survive
    #    it -- which is why this is a structural check and not a rel_fro bound.
    problems, _ = inject(stored_scale=lambda m: np.full(
        m.shape[1], np.abs(m).max() / 127.0, dtype=np.float32))
    expect_caught(problems, "Per-output-channel symmetric is the scheme",
                  "per-tensor scale instead of per-output-channel")

    # 3. the asmooth vector applied in the wrong ORDER -- an index slip in
    #    slicing the per-call-site factors. Everything the container says about
    #    itself stays self-consistent, so only comparing the payload against
    #    the source sees it.
    problems, _ = inject(asm=sample_smooth()[::-1].copy())
    expect_caught(problems, "int8 values are not rint(W*asmooth/wscale)",
                  "asmooth applied in the wrong order")

    # 4. a stored scale that is simply too small. The payload then saturates
    #    against the file's own rule, which is the scale check's whole subject.
    problems, _ = inject(halve_stored=True)
    expect_caught(problems, "disagrees with max|W*asmooth|/127",
                  "a stored scale too small for the weights it scales")

    # 5. THE INTERESTING ONE, direction A. `.wscale` is right and the payload
    #    was rounded at a SMALLER scale, so the file satisfies its own scale
    #    rule and every live column still reaches 127 -- the clip put it there.
    #    The weights are quietly coarser than the file claims, and the
    #    saturation invariant is blind to this direction BY CONSTRUCTION, which
    #    is why the payload check is a separate check and not part of it.
    problems, _ = inject(q_scale=lambda m: np.where(
        np.abs(m).max(axis=0) > 0,
        (np.abs(m).max(axis=0) / 127.0 * 0.5).astype(np.float32),
        np.float32(1.0)))
    expect_caught(problems, "int8 values are not rint(W*asmooth/wscale)",
                  "payload rounded below the scale stored beside it")
    blind = not any("do not reach 127" in p for p in problems)
    report(blind, "  ...and the saturation invariant is blind to that direction",
           "" if blind else "-> it fired, so the invariant is not direction-free")

    # 6. direction B: rounded at a LARGER scale, so the columns come up short
    #    of 127. Here the payload check and the saturation invariant agree.
    problems, _ = inject(q_scale=lambda m: np.where(
        np.abs(m).max(axis=0) > 0,
        (np.abs(m).max(axis=0) / 127.0 * 2.0).astype(np.float32),
        np.float32(1.0)))
    expect_caught(problems, "int8 values are not rint(W*asmooth/wscale)",
                  "payload rounded above the scale stored beside it")
    expect_caught(problems, "do not reach 127",
                  "...and the saturation invariant names that one too")

    # 7. padding that did not stay padding
    problems, _ = inject(break_pad=True)
    expect_caught(problems, "all-zero columns came back non-zero",
                  "a zero padding column that came back non-zero")


def test_refusals():
    print("\n5. refuse, or go loud: the two outcomes a gate must tell apart")
    # The accumulator is exact only while it cannot overflow; 127*127*K >= 2^31
    # is the line and it is asserted rather than assumed. Checked here because
    # an assertion sitting behind an earlier ValueError is an assertion that
    # never fires -- and the tiling check IS earlier in the function, so this
    # also proves the guard is reachable rather than dead.
    k_overflow = 2 ** 31 // (127 * 127) + 1
    try:
        add_gemm_b_int8(Writer({}), "huge",
                        np.zeros((k_overflow, 1), np.float32),
                        TILE_K, TILE_N, mac=MAC)
        report(False, "a K that overflows the int32 accumulator is refused",
               f"-> it packed K={k_overflow}")
    except SystemExit as e:
        report("int32 accumulator" in str(e),
               "a K that overflows the int32 accumulator is refused",
               f"-> K={k_overflow}: {str(e)[:58]}")
    except ValueError as e:
        report(False, "a K that overflows the int32 accumulator is refused",
               f"-> tiling raised first, so the guard is unreachable: {e}")

    # A panel with no sidecars must RAISE. Returning the raw int8 would hand the
    # caller a weight 1/127 of the right size, and every embedding downstream of
    # it would still look like a plausible number.
    w = Writer({"arch": "probe", "a_dtype": "i8"})
    q = np.zeros(SHAPE, np.int8)
    layout = gemm_b_layout(TILE_K, TILE_N, MAC[0], MAC[1], dtype="I8")
    w.add("layer.0.qkv", tile_b(q, TILE_K, TILE_N, MAC[0], MAC[1]), "I8",
          "gemm_b", list(SHAPE), layout=layout)      # no .wscale, no .asmooth
    with container(w) as r:
        try:
            r.tensor("layer.0.qkv")
            report(False, "an I8 panel with no scales raises", "-> it returned a value")
        except KeyError as e:
            report(".wscale is not in the container" in str(e),
                   "an I8 panel with no scales raises", f"-> {str(e)[:66]}")


def main():
    print("int8 scheme self-check (numpy only)\n")
    test_honest()
    test_zero_column()
    test_no_smoothing()
    test_faults()
    test_refusals()
    print()
    if _failures:
        print(f"FAIL -- {len(_failures)} of the lines above did not hold:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("PASS -- the scheme round-trips, the reader dequantises, and every\n"
          "        injected fault was named by the gate that exists to name it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
