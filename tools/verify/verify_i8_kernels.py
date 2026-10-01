# NpuEmbeddings -- do the int8 host kernels mean the same thing with and without AVX2?
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS IS FOR. `runtime/include/common/host_kernels.hpp` opens by claiming
# "The AVX2 paths are bit-identical to their scalar fallbacks." That was false,
# and it was false in a way nobody could see by reading: the int8 datapath gives
# a slightly different answer depending on whether the host CPU has AVX2. This
# gate builds one probe (runtime/tests/test_int8_host_kernels.cpp) three ways
# from that same header and diffs the bytes.
#
#   A  -mavx2 -mfma                    what ships
#   B  -mno-avx2 -ffp-contract=off     the honest scalar formula
#   C  -mno-avx2                      the same, with the compiler's own FMA
#                                      contraction -- i.e. what a non-AVX2 host
#                                      gets from this source
#
# B vs C is in here because "the scalar answer" is only a meaningful thing to
# compare against if it does not itself depend on the compiler's flags. It does
# not: they agree, which is worth knowing and is asserted.
#
# WHAT IS GATED, and why each one is a gate rather than a print:
#
#   1. The probe's INPUTS are identical across all three builds. A gate whose
#      inputs differ per build is measuring its own generator, and would report
#      a divergence that is really just a different LCG. This is the check that
#      makes every other number below mean anything.
#   2. `quantise_a_int8` IS bit-identical. The vector head clips in float and
#      then rounds; the scalar tail rounds and then clips. For a value already
#      inside +/-127 those orders give the same integer, so they agree, and the
#      file's original claim was right about this one. Gated hard, because this
#      is the half of the int8 arithmetic that decides the payload, and a
#      divergence here would be a real accuracy change.
#   3. `dequantise_c` and `dequant_act_quant` are NOT bit-identical, and the
#      divergence is BOUNDED and ATTRIBUTED. Not "prints a number": the gate
#      rebuilds both orderings in numpy and requires that AVX2 matches
#      round-then-FMA exactly and the scalar build matches round-round-round
#      exactly. An unexplained 1-ULP difference is a different bug and would
#      fail; a difference that grows past 1 ULP fails.
#   4. The alignment precondition refuses what it must and accepts what it may.
#      This is the part a byte diff cannot see, because the failure mode is a
#      SIGSEGV on AVX2 hosts and a correct answer everywhere else.
#
# Env: g++ with AVX2, and numpy. No NPU, no XRT, no checkpoint. Usage:
#   python tools/verify/verify_i8_kernels.py

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "runtime" / "tests" / "test_int8_host_kernels.cpp"

sys.path.insert(0, str(REPO / "tools" / "lib"))
# The bf16 rounding rule, from the one place it is written down. The probe's
# `to_bf16` and the runtime's are the same function, and the gate needs that
# rule to reconstruct what --sim-c-bf16 fed the kernel. Copying the three lines
# here would be a second implementation of a rounding rule in a gate whose
# entire subject is whether two implementations of one rule agree.
from npue import from_bf16_bits, to_bf16_bits                    # noqa: E402

# -ffp-contract=off is what makes build B an honest scalar formula. GCC's default
# is -ffp-contract=fast, which is allowed to fuse `a*b + c` on its own -- so
# without this flag build B would quietly be a third arithmetic variant and the
# whole comparison would be unfalsifiable.
BUILDS = (
    ("avx2", ["-mavx2", "-mfma"]),
    ("scalar", ["-mno-avx2", "-ffp-contract=off"]),
    ("scalar_default", ["-mno-avx2"]),
)

# How each section's bytes are to be read. Kept next to the gate rather than in
# the probe so a section added on one side and not the other is a KeyError here,
# loudly, instead of a silently skipped comparison.
DTYPES = {
    "in.a": np.float32, "in.ias": np.float32, "in.ws": np.float32,
    "in.bias": np.float32, "in.acc": np.int32, "in.cbf16": np.uint16,
    "quantise.q": np.int8, "quantise.sa": np.float32,
    "dequantise_c.i32": np.float32, "dequantise_c.bf16": np.float32,
    "dequantise_c.sim_bf16": np.float32,
    "dequant_act_quant.q": np.int8, "dequant_act_quant.sa": np.float32,
}
# The divergence between the two orderings of one fp32 expression is 1 ULP by
# construction. The budget is 1, expressed as "max ULP apart", because a relative
# tolerance would let a large absolute error through on small values and a large
# relative one through on large values -- and this gate's whole subject is a
# difference of exactly one rounding.
MAX_ULP = 1

_failures = []


def report(ok, label, detail=""):
    print(f"   {'ok  ' if ok else 'FAIL'}  {label}"
          + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def build(tmp, name, flags):
    exe = Path(tmp) / f"probe_{name}"
    cmd = ["g++", "-std=c++17", "-O2", *flags, "-I",
           str(REPO / "runtime" / "include"), str(PROBE), "-o", str(exe)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        print(f"   FAIL  could not build the {name} probe:\n{res.stderr[-1500:]}")
        _failures.append(f"build {name}")
        return None
    return exe


def run(tmp, name, exe):
    blob, index = Path(tmp) / f"{name}.bin", Path(tmp) / f"{name}.idx"
    res = subprocess.run([str(exe), str(blob), str(index)],
                         capture_output=True, text=True)
    if res.returncode:
        print(f"   FAIL  the {name} probe died (rc={res.returncode}). This is "
              f"the alignment\n         precondition failing as a SIGSEGV rather "
              f"than a refusal, which is the\n         bug require_stream_aligned"
              f"() exists to prevent.\n{res.stderr[-600:]}")
        _failures.append(f"run {name}")
        return {}
    raw = blob.read_bytes()
    out = {}
    for line in index.read_text().splitlines():
        name_, off, nbytes = line.split("\t")
        chunk = raw[int(off):int(off) + int(nbytes)]
        if name_.startswith("align."):
            out[name_] = chunk.decode("utf-8", "replace")
        else:
            out[name_] = np.frombuffer(chunk, dtype=DTYPES[name_])
    return out


def fl32(x):
    return x.astype(np.float32)


def fma32(p, w, b):
    """float32 FMA(p, w, b), emulated.

    p is ALREADY rounded to float32, because the vector path computes
    `_mm256_mul_ps(cf, sa)` first and only then contracts the multiply-add. The
    product of two float32 is exact in float64, so one float64 add and one
    round-to-float32 IS an FMA -- modulo the double-rounding that a float64 add
    could in principle introduce, which is why the gate requires an EXACT match
    rather than a close one: if double rounding had bitten, this would fail and
    say so instead of quietly explaining the divergence away.
    """
    return (p.astype(np.float64) * w.astype(np.float64)
            + b.astype(np.float64)).astype(np.float32)


def ulps(a, b):
    """Elementwise ULP distance between two float32 arrays, as int64."""
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    # Negative floats have the sign bit set, so the bit pattern's order is
    # reversed on one side of zero; without this, +x and -y read as adjacent.
    ia = np.where(ia < 0, np.int64(-0x80000000) - ia, ia)
    ib = np.where(ib < 0, np.int64(-0x80000000) - ib, ib)
    return np.abs(ia - ib)


def check_inputs(sections):
    print("\n1. the probe's inputs are the same in every build")
    others = [n for n, _ in BUILDS if n != "avx2"]
    ref = sections["avx2"]
    numeric = [k for k in ref if not k.startswith("align.")]

    bad = [k for k in numeric if k.startswith("in.")
           and not all(np.array_equal(ref[k], sections[b][k]) for b in others)]
    report(not bad, "in.* identical across all three builds",
           "" if not bad else f"-> differs: {bad}")
    mismatched = [k for k in numeric
                  if any(sections[b][k].shape != ref[k].shape for b in others)]
    report(not mismatched, "every section has the same shape in every build",
           "" if not mismatched else f"-> differs: {mismatched}")


def check_scalar_stability(sections):
    print("\n2. the scalar answer does not depend on the compiler's FMA flags")
    a, b = sections["scalar"], sections["scalar_default"]
    diff = [k for k in a if not k.startswith("align.")
            and not np.array_equal(a[k], b[k])]
    report(not diff, "-ffp-contract=off and the compiler default agree",
           "" if not diff else f"-> differ: {diff} -- the scalar reference "
                              f"itself is not one number, so nothing above "
                              f"would mean anything")


def check_quantise(sections):
    print("\n3. quantise_a_int8 IS bit-identical (the payload half of int8)")
    a, b = sections["avx2"], sections["scalar"]
    for key in ("quantise.q", "quantise.sa"):
        same = np.array_equal(a[key], b[key])
        report(same, f"{key} identical",
               "" if same else
               f"-> {int((a[key] != b[key]).sum())} of {a[key].size} differ; the "
               f"vector head clips then rounds, the tail rounds then clips, and "
               f"for a value inside +/-127 that is the same integer -- so this "
               f"must not move")


def c_values(a, key):
    """The float32 values each `dequantise_c` variant actually saw in C.

    Three different ones, and using the wrong one is how this gate first
    reported 448 of 448 unexplained: the sections are not the same input
    re-rounded, they are three different inputs.
      .i32       the accumulator as-is
      .bf16      the pre-baked bf16 transport
      .sim_bf16  the accumulator rounded to bf16 the way --sim-c-bf16 does,
                 which is to_bf16 of the float32 value -- NOT the pre-baked
                 buffer, which carries an extra 1e-4 scaling factor
    """
    acc = a["in.acc"].astype(np.float32)
    if key.endswith(".i32"):
        return acc
    if key.endswith(".bf16"):
        return from_bf16_bits(a["in.cbf16"])
    if key.endswith(".sim_bf16"):
        return from_bf16_bits(to_bf16_bits(acc))
    raise KeyError(key)


def attribute(section, cf, sa, ws, bias, rows, N, key):
    """Rebuild both orderings of the dequantise formula and say which is which."""
    cf = cf.astype(np.float32).reshape(rows, N).astype(np.float64)
    sa_r = np.repeat(sa.astype(np.float64), N).reshape(rows, N)
    ws_r = ws.astype(np.float64)[None, :]
    b_r = bias.astype(np.float64)[None, :]
    p = (cf * sa_r).astype(np.float32)                     # _mm256_mul_ps rounds
    two = (fl32(p * ws_r.astype(np.float32)) + bias[None, :]).astype(np.float32)
    one = fma32(p, ws_r.astype(np.float32), b_r).reshape(-1)
    two = two.reshape(-1)
    out = {}
    for label, ref in (("two_round", two), ("fma", one)):
        for build, sec in (("avx2", section["avx2"][key]),
                           ("scalar", section["scalar"][key])):
            out[(build, label)] = int((sec != ref).sum())
    return out


def check_dequant(sections):
    print("\n4. dequantise_c / dequant_act_quant are NOT bit-identical -- and "
          "the\n   difference is bounded and explained, which is the only "
          "acceptable shape\n   for a divergence between two paths of one "
          "formula")
    a, b = sections["avx2"], sections["scalar"]
    rows, N = 7, 64
    sa = a["quantise.sa"]
    ws, bias = a["in.ws"], a["in.bias"]

    for key in ("dequantise_c.i32", "dequantise_c.bf16", "dequantise_c.sim_bf16"):
        d = a[key] != b[key]
        n = int(d.sum())
        if n == 0:
            report(True, f"{key}: identical after all",
                   "-> the FMA gap closed; update the header's measured numbers")
            continue
        dist = ulps(a[key][d], b[key][d])
        report(int(dist.max()) <= MAX_ULP,
               f"{key}: {n} of {a[key].size} differ, all within {MAX_ULP} ULP",
               f"worst {int(dist.max())} ULP, max relative "
               f"{float((np.abs(a[key][d] - b[key][d]) / np.maximum(np.abs(a[key][d]), 1e-30)).max()):.3e}")
        m = attribute(sections, c_values(a, key), sa, ws, bias, rows, N, key)
        report(m[("avx2", "fma")] == 0 and m[("scalar", "two_round")] == 0,
               f"{key}: attributed -- AVX2 == round-then-FMA, scalar == "
               f"round-round-round",
               f"avx2 vs fma {m[('avx2', 'fma')]} off, avx2 vs two-round "
               f"{m[('avx2', 'two_round')]} off, scalar vs two-round "
               f"{m[('scalar', 'two_round')]} off, scalar vs fma "
               f"{m[('scalar', 'fma')]} off")

    # The one that is not just a rounding difference but a CONSEQUENCE of one.
    d = a["dequant_act_quant.sa"] != b["dequant_act_quant.sa"]
    report(bool(d.any()),
           "dequant_act_quant.sa_next DIFFERS, and that is the finding that "
           "matters",
           f"{int(d.sum())} of {a['dequant_act_quant.sa'].size} activation "
           f"scales move. sa_next is an absmax over the dequantised values, so "
           f"a 1-ULP difference upstream moves the scale and therefore the int8 "
           f"payload of the NEXT operand: an int8 embedding is not bit-"
           f"reproducible across host CPUs. Not an accuracy problem -- 1 ULP is "
           f"four orders below the 2e-03 gate -- a reproducibility one.")
    q = a["dequant_act_quant.q"] != b["dequant_act_quant.q"]
    report(True, f"dequant_act_quant payload: "
                 f"{int(q.sum())} of {a['dequant_act_quant.q'].size} int8 differ",
           "reported, not gated: it is downstream of sa_next, so whether a "
           "moved scale moves a rounded value depends on the data")


def check_alignment(sections):
    print("\n5. the alignment precondition: refuses what it must, accepts what "
          "it may")
    # (section, must_refuse) -- the two halves of the precondition are separate
    # cases because the stride half is the one that was missed: a base pointer
    # off by one element, and a perfectly aligned base whose ROW STRIDE is not a
    # multiple of 32.
    cases = (
        ("align.skewed_base", True, "base pointer off by one int32"),
        ("align.skewed_base.fused", True, "same, into dequant_act_quant"),
        ("align.stride_bad_bf16_n72", True,
         "N=72 over a bf16 C: stride 144 = 4*32+16"),
        ("align.stride_ok_i32_n72", False,
         "N=72 over an int32 C: stride 288 = 9*32, legal"),
    )
    for key, must_refuse, what in cases:
        avx = sections["avx2"].get(key, "<missing>")
        want = "threw:" if must_refuse else "RETURNED"
        report(avx.startswith(want),
               f"AVX2 build, {what}: {'refused' if must_refuse else 'accepted'}",
               "" if avx.startswith(want) else f"-> {avx[:90]!r}")

    # The guard is compiled out without AVX2, because without AVX2 there is no
    # stream load to be misaligned for. Asserting that is what proves the checks
    # above are testing the guard and not something else that happens to throw.
    print("   -- and with -mno-avx2 the guard is compiled out, as it must be:")
    ok = True
    for key, _, what in cases:
        got = sections["scalar"].get(key, "<missing>")
        if not got.startswith("RETURNED"):
            ok = False
            print(f"   FAIL  scalar build, {what}: expected RETURNED, got "
                  f"{got[:70]!r}")
    report(ok, "scalar build accepts all four (no stream load to fault on)",
           "" if ok else "-> so the AVX2 refusals above are the guard, not luck")


def main():
    print("int8 host kernels: AVX2 vs scalar, from one source\n")
    if not PROBE.exists():
        print(f"FAIL -- {PROBE} is missing")
        return 1
    if shutil.which("g++") is None:
        print("FAIL -- g++ is not on PATH; this gate builds the probe itself, "
              "deliberately,\n         so that the two builds differ ONLY in the "
              "flags below")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        sections = {}
        for name, flags in BUILDS:
            exe = build(tmp, name, flags)
            if exe is None:
                print("\nFAIL -- cannot build the probes, so nothing was checked")
                return 1
            sections[name] = run(tmp, name, exe)
        if not all(sections.values()):
            return 1

        check_inputs(sections)
        check_scalar_stability(sections)
        check_quantise(sections)
        check_dequant(sections)
        check_alignment(sections)

    print()
    if _failures:
        print(f"FAIL -- {len(_failures)} of the lines above did not hold:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("PASS -- quantise is bit-identical; the dequant divergence is 1 ULP "
          "and is the\n        FMA, attributed exactly; and the stream-load "
          "precondition refuses exactly\n        the three cases that would "
          "otherwise be a SIGSEGV on AVX2 hosts only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
