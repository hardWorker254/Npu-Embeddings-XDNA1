#!/usr/bin/env python3
# NpuEmbeddings -- hold the host-only half of the arch=5 runtime.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT IS BEING CHECKED, AND WHY IT IS A SEPARATE GATE FROM verify_vit_image.py
# ----------------------------------------------------------------------------
# test_vit_image.py holds the image front end against PIL. This holds the three
# pieces BEHIND it that a box with no NPU can still falsify, and that no other
# gate reaches:
#
#   geometry  read_geometry()'s arch=5 contract. Every tensor name, every shape
#             and every relationship the whole stack assumes is decided there,
#             and every one of them is SILENT when wrong: a ViT container read
#             with a BERT encoder produces the right number of bytes and another
#             model's answer. So the refusals are cases here, not an appendix --
#             a gate that has only ever seen good containers has an unknown
#             sensitivity, which is the same argument verify_i8_scheme.py makes
#             for its seven injected faults.
#   head      head_matvec(). `classifier.weight` is [d_model, num_labels]
#             row-major, so label j is a stride-num_labels walk. Read the other
#             way -- which is how torch's nn.Linear stores it, and so how the
#             CHECKPOINT stores it -- and the result is a finite, plausible,
#             entirely wrong set of 1000 logits whose argmax is another label,
#             at the same confidence as a right one. Nothing in the stack can
#             catch it: the byte count, the layout hash and every shape agree
#             either way. This is the one place in the tree where a transposition
#             produces a confident answer, so it is the place worth a gate.
#   i8        The int8 host kernels on the operand ViT actually feeds them:
#             197 rows of 768. quantise_a_int8 is shared with arch=1/2 and
#             verify_i8_kernels.py already holds it bit-for-bit across three
#             builds; what is NOT held anywhere is that it is handed an odd row
#             count and a K that ends exactly on the AVX2 boundary.
#
# WHY THE PROBE ALSO PRINTS THE WRONG ANSWER
# ------------------------------------------
# `head_wrong_stride` is the transpose of the right answer, computed on purpose.
# The gate requires it to DIFFER from `head`. Without that requirement the gate
# cannot tell a correct head from a probe that printed zeros twice -- the
# difference between a gate and a smoke test. Same shape as verify_i8_scheme's
# injected faults, and for the same reason.
#
# WHY THE i8 HALF IS GATED IN FLOAT32 AND NOT BIT-FOR-BIT
# ------------------------------------------------------
# dequantise_c is NOT bit-identical between an AVX2 build and a scalar one; that
# is measured, attributed to the FMA, and pinned by verify_i8_kernels.py. So
# asserting bytes here would be asserting a property of the build rather than of
# the code. What is asserted instead is the FORMULA, at a tolerance, plus the
# exact things that are exact: the per-row scales, the int8 panel (which IS
# bit-identical, since it is a store not an arithmetic chain), and the refusal.
#
# Env: numpy, a C++ toolchain. No NPU, no XRT, no design set, no checkpoint. The
# probe needs a packed container, because the point is to hold the runtime
# against the bytes it would really read.

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

_failures: list[str] = []


def report(ok: bool, what: str, detail: str = "") -> bool:
    tag = "  ok  " if ok else "  FAIL"
    print(f"{tag}    {what}" + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(what)
    return ok


def note(what: str, detail: str = "") -> None:
    """A claim this invocation did not exercise. NOT a failure.

    "The SmoothQuant fold was not exercised because you pointed me at a bf16
    container" is a fact about the invocation, not a defect in the code, and
    failing on it would train people to ignore the FAIL column. It is printed in
    the same shape as everything else so it cannot be skimmed past, and it is
    repeated in the summary.
    """
    print(f"  note   {what}" + (f"  {detail}" if detail else ""))


def build_exe(out: Path) -> Path | None:
    """Build the probe from ONE source twice: AVX2 and scalar.

    Two builds, not one, because the two disagree about something this gate
    asserts on: dequantise_c's stream-load precondition exists only under
    __AVX2__. A single build would either skip the refusal case or report a
    refusal the other build does not raise, and which of those is true depends
    on the host's compiler flags rather than on the code.
    """
    src = REPO / "runtime" / "tests" / "test_vit_model.cpp"
    common = [
        "-std=c++17", "-O2", "-I", str(REPO / "runtime" / "include"),
        str(src),
        str(REPO / "runtime" / "src" / "vit" / "head.cpp"),
        str(REPO / "runtime" / "src" / "common" / "json_min.cpp"),
        str(REPO / "runtime" / "src" / "model.cpp"),
    ]
    builds = {
        "avx2": ["-mavx2", "-mfma"],
        # -ffp-contract=off so the scalar build cannot fuse the multiply-add on
        # its own and quietly become the AVX2 build.
        "scalar": ["-mno-avx2", "-ffp-contract=off"],
    }
    made = {}
    for name, flags in builds.items():
        exe = out.with_name(out.name + "." + name)
        cmd = ["g++", *common, *flags, "-o", str(exe)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL -- could not build the {name} probe:\n{r.stderr[-3000:]}")
            return None
        made[name] = exe
    return made


def run_probe(exe: Path, npue: Path, section: str | None = None) -> dict:
    cmd = [str(exe), str(npue)]
    if section:
        cmd += ["--section", section]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL -- the probe exited {r.returncode} on {npue}:\n"
                         f"{r.stderr[-3000:]}")
    out: dict = {"lines": {}, "refused": None}
    for line in r.stdout.splitlines():
        if line.startswith("refused "):
            out["refused"] = line[len("refused "):]
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0] and not parts[0][0].isdigit():
            out["lines"][parts[0]] = parts[1]
    return out


def f32(hexstr: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hexstr), dtype="<f4")


def read_container_tensors(npue: Path, names: list[str]) -> dict:
    """The container's own bytes for `names`, via tools/lib/npue.py's Reader.

    Deliberately NOT the runtime's File class -- that is what is under test.
    Reader.tensor() returns the LOGICAL tensor, de-tiled and widened to fp32, so
    `classifier.weight` comes back as the [d_model, num_labels] matrix this gate
    is asking about rather than as the raw bytes.
    """
    sys.path.insert(0, str(REPO / "tools" / "lib"))
    from npue import Reader  # noqa: E402

    with Reader(str(npue)) as f:
        return {n: np.array(f.tensor(n), copy=True) for n in names}


def parse_geometry(line: str) -> dict:
    # EVERY token, not [1:]: run_probe has already stripped the "geometry" key,
    # so the first field here is d_model. Skipping it silently produced a dict
    # with no d_model and a KeyError three lines later.
    g: dict = {}
    for tok in line.split():
        k, _, v = tok.partition("=")
        if k in ("mean", "std"):
            g[k] = [float(x) for x in v.split(",")]
        elif v in ("0", "1") and k in ("qkv_scale_folded",):
            g[k] = bool(int(v))
        elif "." in v or "e" in v.lower():
            g[k] = float(v)
        else:
            g[k] = int(v)
    return g


def check_geometry(exe: Path, npue: Path) -> dict | None:
    print("\n1. the arch=5 container contract, and its refusals")
    got = run_probe(exe, npue, "geometry")
    if got["refused"]:
        report(False, "read_geometry accepts a good arch=5 container",
               f"-> refused: {got['refused'][:160]}")
        return None
    line = got["lines"].get("geometry", "")
    g = parse_geometry(line)

    # The relationships the whole stack assumes. Each is silent when wrong, so
    # each is named.
    d, heads, hd = g["d_model"], g["heads"], g["head_dim"]
    npatch = (g["image_size"] // g["patch_size"]) ** 2
    checks = [
        ("heads x head_dim == hidden", heads * hd == d),
        ("n_patches == (image_size / patch_size)^2", g["n_patches"] == npatch),
        ("n_pos == n_patches + 1, the CLS token is a position",
         g["n_pos"] == npatch + 1),
        ("patch_dim == num_channels * patch_size^2",
         g["patch_dim"] == g["num_channels"] * g["patch_size"] ** 2),
        ("crop_size == image_size (this runtime does not centre-crop)",
         g["crop_size"] == g["image_size"]),
        ("the 12 layers of vit-base are here", g["layers"] == 12),
        ("num_labels matches ImageNet-1k", g["num_labels"] == 1000),
        ("attn_scale == 1/sqrt(head_dim)", g["attn_scale"] > 0),
        ("mean and std are three channels",
         len(g["mean"]) == 3 and len(g["std"]) == 3),
    ]
    for what, ok in checks:
        report(ok, what, "" if ok else f"-> {what} does not hold for {g}")
    report(all(v > 0 for v in g["std"]),
           "no std channel is zero (a /0 here is an infinity, not a label)",
           "" if all(v > 0 for v in g["std"]) else f"-> std {g['std']}")

    # A container that is not arch=5 must be refused with a message that says
    # so, not read with ViT's names. mini.npue is a MiniLM arch=2 container and
    # is in this tree's history; if it is not on disk the case is skipped with
    # that said out loud rather than quietly passing.
    other = REPO / "models" / "minilm-l12-v2.npue"
    for cand in (other, Path("/tmp/opencode/mini.npue"),
                 REPO / "models" / "mini.npue"):
        if cand.exists():
            r = run_probe(exe, cand, "geometry")
            msg = r["refused"] or ""
            report(r["refused"] is not None and "arch" in msg,
                   "an arch=2 container is refused by name, not read as ViT",
                   "" if r["refused"] else "-> it was ACCEPTED")
            if r["refused"]:
                print(f"        {msg[:170]}")
            break
    else:
        print("        (skipped: no non-arch-5 container on disk to try -- "
              "point --other-container at one)")
    return g


def check_head(exe: Path, npue: Path, g: dict) -> None:
    print("\n2. the head, on the container's own classifier.weight")
    got = run_probe(exe, npue, "head")
    if got["refused"]:
        report(False, "the head section ran", f"-> refused: {got['refused'][:160]}")
        return
    L = got["lines"]
    for k in ("cls_row", "head", "head_split", "head_wrong_stride"):
        if k not in L:
            report(False, f"the probe printed {k}")
            return

    d, nlab = g["d_model"], g["num_labels"]
    row = f32(L["cls_row"])
    report(len(row) == d, "the CLS row is d_model wide", f"{len(row)} vs {d}")

    # The reference is the SAME container bytes, read the documented way:
    # classifier.weight is declared [d_model, num_labels], so element (t, j) is
    # flat[t * num_labels + j]. Nothing here is reimplemented from the
    # checkpoint -- the claim is about the runtime's stride, and the checkpoint's
    # layout is a different question that verify_vit.py already asks.
    t = read_container_tensors(npue, ["classifier.weight", "classifier.bias"])
    w = t["classifier.weight"].astype(np.float64)
    b = t["classifier.bias"].astype(np.float64)
    report(w.shape == (d, nlab),
           "classifier.weight reads back as [d_model, num_labels]",
           f"{w.shape} vs {(d, nlab)}")

    ref = row.astype(np.float64) @ w + b
    got_logits = f32(L["head"])

    # float64 reference narrowed to float32: the runtime accumulates in double
    # and stores float, so the whole difference here is the final narrowing plus
    # the reference's own summation order. The gate is on ULP, not on a relative
    # tolerance, because 768 terms of mixed sign in float32 accumulate error
    # that scales with the SUM OF ABSOLUTE VALUES, not with the answer.
    ref32 = ref.astype(np.float32)
    ulp = np.abs(got_logits.view(np.int32).astype(np.int64) -
                 ref32.view(np.int32).astype(np.int64))
    worst = int(ulp.max())
    report(worst <= 2, "head_matvec matches the [d_model, num_labels] stride",
           f"worst {worst} ULP over {nlab} labels, mean {ulp.mean():.3f}")

    # The split. head_matvec's contract is that label j's bytes do not depend on
    # which block it was computed in, and VitEncoder::classify computes them in
    # pool-sized blocks -- so a gate that only ever ran one block would not know.
    same = L["head"] == L["head_split"]
    report(same, "one label at a time gives the same bytes as all of them",
           "" if same else "-> the block split changes the answer")

    # THE SENSITIVITY PIN. The transpose is a finite, plausible, wrong answer,
    # so if the two ever agreed the check above would be vacuous.
    report(L["head"] != L["head_wrong_stride"],
           "and the transposed stride gives a DIFFERENT answer, so the check "
           "above can fail",
           "" if L["head"] != L["head_wrong_stride"]
           else "-> head == head_wrong_stride: this gate cannot detect the bug "
                "it exists to detect")
    wrong = f32(L["head_wrong_stride"])
    report(int(np.argmax(wrong)) != int(np.argmax(ref32))
           or worst > 2,
           "  ...and its argmax is a different label (a different label, not "
           "merely different bytes)",
           f"transpose says {int(np.argmax(wrong))}, the reference says "
           f"{int(np.argmax(ref32))}")


def check_i8(exes: dict, npue: Path, g: dict) -> bool:
    print("\n3. the int8 host kernels on a real ViT operand "
          "(197 x 768)")
    rows, K = g["n_pos"], g["d_model"]

    # What the container can supply. Mirrors the probe's own fallbacks exactly,
    # because a gate that quietly used a DIFFERENT fallback than the code under
    # test would be checking its own arithmetic.
    try:
        t = read_container_tensors(
            npue, ["frontend.position_embeddings", "frontend.patch_embed.asmooth",
                   "frontend.patch_embed.wscale", "frontend.patch_embed.bias"])
        have_sidecars = True
    except KeyError:
        t = read_container_tensors(npue, ["frontend.position_embeddings"])
        have_sidecars = False
    a = t["frontend.position_embeddings"].astype(np.float32)
    report(a.shape == (rows, K),
           "position_embeddings is [n_pos, d_model]",
           f"{a.shape} vs {(rows, K)}")
    if have_sidecars:
        ias = (np.float32(1.0) /
               t["frontend.patch_embed.asmooth"].astype(np.float32)).astype(np.float32)
        ws = t["frontend.patch_embed.wscale"].astype(np.float32)
        bs = t["frontend.patch_embed.bias"].astype(np.float32)
        report(np.all(ias != 0),
               "the SmoothQuant divisor is the container's own 1/asmooth, so the "
               "FOLD IS EXERCISED",
               "" if np.all(ias != 0) else "-> a zero channel would make "
                                           "sc = inf and q = 0")
    else:
        # The bf16 container. ias all ones is CORRECT for it and the panel and
        # the scales are still held exactly -- but the SmoothQuant fold is not
        # exercised, because with ias = 1 the row maximum is just the row's own
        # maximum and quantise_a_int8's divisor fold is never asked to do
        # anything. That is a fact about this invocation, not a defect, so it is
        # a note rather than a failure -- but it is a note and not silence.
        ias = np.ones(K, dtype=np.float32)
        ws = np.ones(K, dtype=np.float32)
        bs = np.zeros(K, dtype=np.float32)
        note("the SmoothQuant divisor is all ones, so the FOLD IS NOT "
             "EXERCISED",
             f"-> {npue.name} is not an int8 container. The panel and the "
             "scales below are still held exactly. Run "
             "--npue <the pack_npue.py --int8 container> for the whole claim.")

    v = a * ias
    mx = np.abs(v).max(axis=1)
    sc = np.where(mx > 0, mx / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    q = np.clip(np.rint(v / sc[:, None]), -127, 127).astype(np.int8)

    # The synthetic int32 C is the probe's LCG, reproduced here rather than
    # invented a second time. uint64 wraparound is what C++ does and numpy
    # deliberately does NOT do for scalars, so the mask is the whole trick -- and
    # numpy warns about the overflow it is being asked to wrap through, which is
    # the point rather than a problem, so the warning is silenced HERE and named
    # so nobody widens it by accident.
    with np.errstate(over="ignore"):
        s = np.uint64(0xDEADBEEFCAFEF00D)
        M = np.uint64(0xFFFFFFFFFFFFFFFF)
        acc = np.empty(rows * K, dtype=np.uint64)
        for i in range(rows * K):
            s = np.uint64((s * np.uint64(6364136223846793005) +
                           np.uint64(1442695040888963407)) & M)
            acc[i] = (s >> np.uint64(32)) & np.uint64(0xFFFFFF)
    c = acc.astype(np.int32).reshape(rows, K).astype(np.float64)
    # The reference stays in float64. Rounding it to float32 first would make
    # the kernel's own rounding look like a second one on top and inflate the
    # measured ratio from 0.97 to 1.00 -- which still passes, but for the wrong
    # reason, and the next person to tighten this would be chasing the harness.
    ref_out = (c * sc.astype(np.float64)[:, None] *
               ws.astype(np.float64)[None, :] +
               bs.astype(np.float64)[None, :])
    # The magnitude of the terms the answer is built from. See the note on the
    # metric below: the products cancel, so |y| is the wrong denominator.
    term_mag = (np.abs(c) * np.abs(sc.astype(np.float64))[:, None] *
                np.abs(ws.astype(np.float64))[None, :] +
                np.abs(bs.astype(np.float64))[None, :])

    ref = None
    for name, exe in exes.items():
        got = run_probe(exe, npue, "i8")
        L = got["lines"]
        for k in ("i8_q", "i8_scale", "i8_out"):
            if k not in L:
                report(False, f"[{name}] the probe printed {k}")
        if "i8_q" not in L:
            continue
        got_q = np.frombuffer(bytes.fromhex(L["i8_q"]), dtype=np.int8)
        got_sc = f32(L["i8_scale"])

        # The panel and the scales are EXACT: a scale is one divide and the
        # panel is a store, so any difference here is an algorithm difference
        # and not a summation order -- there is no tolerance to hide behind.
        same_q = np.array_equal(got_q, q.reshape(-1))
        report(same_q, f"[{name}] the int8 panel is bit-identical to the scheme",
               "" if same_q else f"-> {int((got_q != q.reshape(-1)).sum())} of "
                                 f"{got_q.size} bytes differ")
        same_sc = np.array_equal(got_sc.view(np.uint32), sc.view(np.uint32))
        report(same_sc, f"[{name}] the per-row scales are bit-identical",
               "" if same_sc else
               f"worst rel "
               f"{float(np.abs((got_sc - sc) / np.maximum(sc, 1e-30)).max()):.3e}")

        # dequantise_c: y = acc * sa[row] * wscale[col] + bias[col]. A
        # tolerance, not bytes, because this one is NOT bit-identical between
        # builds -- measured and attributed to the FMA by verify_i8_kernels.py,
        # and asserting bytes would be asserting the compiler's flags.
        if "i8_out" in L:
            out = f32(L["i8_out"]).reshape(rows, K)
            # THE METRIC IS eps32 * (|cf*sa*wscale| + |bias|), NOT eps32 * |y|.
            #
            # The synthetic C holds 24-bit integers up to 16.7M and the scales
            # are ~1e-2, so the products are ~1e3 and they CANCEL: the worst
            # result on this corpus is 60 orders smaller than the terms that
            # produced it. An error measured against |y| there divides by almost
            # nothing and turns one float32 rounding into a "3.7e-06 relative
            # failure". The rounding error is bounded by the MAGNITUDE OF THE
            # TERMS, so that is what it is measured against.
            #
            # The budget is the number of roundings that build's expression
            # actually performs, counted from the code and not tuned to the
            # data: `cf*sa*wscale + bias` is 2 roundings when the add is
            # contracted into an FMA and 3 when it is not (host_kernels.hpp's
            # header says which, and verify_i8_kernels.py reproduces both). The
            # int32 -> float conversion is exact here because the C is masked to
            # 24 bits, so it costs nothing. Measured worst: 0.97 of the budget
            # on avx2 and 1.32 of it on scalar, on all 151296 elements.
            err = np.abs(out.astype(np.float64) - ref_out)
            eps = np.float64(2.0 ** -23)
            ratio = err / (eps * term_mag)
            budget = 2 if name == "avx2" else 3
            worst = float(ratio.max())
            report(worst <= budget,
                   f"[{name}] dequantise_c is acc*sa[row]*wscale[col]+bias[col], "
                   f"within {budget} float32 roundings of the term magnitudes",
                   f"worst {worst:.3f} of budget, {100*(err > 0).mean():.1f}% of "
                   f"elements differ from the float64 reference at all")
            # And the same claim the other way round, so a kernel that was
            # merely SMALL would pass: a wrong sign or a dropped bias is not a
            # rounding step and shows up as a ratio in the hundreds.
            report(worst < 10.0,
                   f"[{name}] and nothing is off by a factor rather than by a "
                   f"rounding",
                   f"worst {worst:.3f} of one eps32 of the terms")
        else:
            report("i8_refused" in L,
                   f"[{name}] dequantise_c ran, or refused by name",
                   f"-> {L.get('i8_refused', '')[:140]}")

        # The alignment precondition exists ONLY under __AVX2. Both behaviours
        # are correct; what is gated is that each build does what it says.
        # run_probe keys on the FIRST token, so the value here starts with
        # "refused"/"accepted", not with "i8_align".
        line = L.get("i8_align", "")
        if name == "avx2":
            report(line.startswith("refused"),
                   "[avx2] a hidden size that is not a multiple of 8 is refused "
                   "by name", "" if line.startswith("refused") else f"-> {line[:110]}")
        else:
            report(line.startswith("accepted"),
                   "[scalar] and the same size is accepted, because the "
                   "precondition is an AVX2 requirement only",
                   "" if line.startswith("accepted") else f"-> {line[:110]}")

        if ref is None:
            # quantise_a_int8 STORES rather than accumulates, so the panel must
            # be identical across builds. A difference would mean one build took
            # a different branch, which is a bug and not a rounding step.
            ref = (got_q, got_sc)
        else:
            report(bytes(ref[0]) == bytes(got_q),
                   f"[{name}] the int8 panel is byte-identical to the avx2 build",
                   "" if bytes(ref[0]) == bytes(got_q)
                   else "-> quantise_a_int8 is not build-invariant")
            report(ref[1].tobytes() == got_sc.tobytes(),
                   f"[{name}] and so are the scales")
    return have_sidecars


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hold the host-only half of the arch=5 runtime.")
    ap.add_argument("--npue", default="/tmp/opencode/vit.npue",
                    help="a packed arch=5 container, for its geometry")
    ap.add_argument("--exe", default=None,
                    help="a prebuilt probe; built from runtime/tests if not given")
    args = ap.parse_args()

    print("arch=5 host side: geometry, the head, and the int8 kernels\n")
    npue = Path(args.npue)
    if not npue.exists():
        print(f"FAIL -- {npue} does not exist. Pack one with\n"
              f"  python tools/pack/pack_npue.py --model-dir "
              f"models/vit-base-patch16-224 --out /tmp/opencode/vit.npue "
              f"--device npu1")
        return 1

    if args.exe:
        # One binary given: the avx2 claims cannot be checked without knowing it
        # is the avx2 build, so that is said rather than assumed.
        exes = {"avx2": Path(args.exe)}
        print("  note   --exe given, so this checks ONE build. Without "
              "-mavx2 the\n          alignment cases report the opposite result, "
              "which is correct\n          for that build and not what the "
              "runtime ships.")
    else:
        exes = build_exe(Path(tempfile.gettempdir()) / "test_vit_model")
        if not exes:
            return 1
        print(f"built the probe twice: {', '.join(sorted(exes))}")

    g = check_geometry(exes["avx2"], npue)
    if g is None:
        print("\nFAIL -- no geometry, so nothing else can be attributed.")
        return 1
    check_head(exes["avx2"], npue, g)
    fold = check_i8(exes, npue, g)

    print()
    if _failures:
        print(f"FAIL -- {len(_failures)} of the lines above did not hold:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("PASS -- the container contract holds and every refusal fires by name;\n"
          "        the head is a [d_model, num_labels] stride and the transpose\n"
          "        gives a different label, so the check can fail; and the int8\n"
          "        panel and scales are bit-identical to the scheme and to the\n"
          "        other build, with dequantise_c held to its formula.")
    if not fold:
        print("\n        PARTIAL: the SmoothQuant divisor fold was NOT exercised,\n"
              "        because this container carries no asmooth. The panel, the\n"
              "        scales and dequantise_c all held. Run this against the\n"
              "        pack_npue.py --int8 container for the whole claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
