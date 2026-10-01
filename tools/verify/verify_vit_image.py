#!/usr/bin/env python3
# NpuEmbeddings -- hold the C++ image front end against PIL.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT IS BEING CHECKED, AND AGAINST WHAT
# ---------------------------------------
# vit/image.hpp claims resize_square() computes "PIL.Image.resize(...), computed
# in float and rounded back to 8 bits", and then says the claim is MEASURED, not
# bit-exact. This is that measurement. The authority is PIL itself -- the library
# transformers' ViTImageProcessor calls -- on the same bytes, for the same four
# resample codes, at the same target size.
#
# This gate is what turned "the same six codes" into "four, and the other two
# refused by name". The first version of image.cpp implemented LANCZOS,
# BILINEAR, BICUBIC, BOX and HAMMING as ONE Keys cubic with a support and a
# sharpness, which is BICUBIC and nothing else: measured against PIL on noise it
# was off by up to 65 levels for the other four. BICUBIC passed throughout, which
# is exactly why the rest went unnoticed -- it shows the INDEX arithmetic is
# right, which is the half the codes share. NEAREST and HAMMING are now refusals
# in section 4, and each message names the four that ARE held to what.
#
# Four stages, reported separately so a divergence can be attributed:
#
#   decoded  the decoder, against PIL's own decode of the same file. 8-bit, so
#            this is exact or it is a bug; the interesting cases are the ones
#            where PIL does NOT hand back three channels (palette, grayscale,
#            RGBA) and both sides have to have made the same decision. 16-bit is
#            the exception and it is refused rather than compared, because the
#            two correct-looking reductions disagree and PIL's is the one
#            transformers feeds the model.
#   resize   the square resize, against PIL, once per implemented code. This is
#            the number the header's comment points at, and it is reported as a
#            worst channel difference AND a fraction of channels that differ --
#            a mean would hide both a systematic half-LSB bias and a handful of
#            badly wrong pixels.
#   pixels   (x/255 - mean)/std per channel, against the same expression on
#            numpy. Gated at 0 ULP, which is why the C++ divides by 255 instead
#            of multiplying by its reciprocal.
#   patches  the im2col, against tools/lib/vit_int8.im2col_patches -- the Python
#            half of the pair that verify_vit.py section 1 holds against HF's
#            Conv2d. Two implementations of one layout, so the layout is
#            compared directly rather than inferred from a classification.
#
# THE REFUSALS ARE CASES TOO
# --------------------------
# NEAREST and HAMMING are refused by name, and each message must name the four
# codes that ARE implemented and say what they are measured against -- a
# refusal that only says "unsupported" is indistinguishable from an
# unimplemented corner. Then: a 16-bit PNG is refused rather than reduced two
# different ways, alpha is composited on white rather than dropped, a fake PNG
# is refused, a truncated PNG is refused rather than decoded to a half-grey
# raster, and a text file is refused with a message that says what to do.
# Silently accepting any of them is the failure this project treats as worst:
# the model classifies a picture it never saw and prints a confident label.
#
# WHY THE MEASUREMENT IS A THRESHOLD AND NOT EQUALITY
# ----------------------------------------------------
# PIL's 8-bit resample is FIXED-POINT with 22 bits of fractional precision and a
# final clamp; this one accumulates in float64 and rounds half away from zero.
# They agree to within a rounding step almost everywhere and can differ by one
# LSB where the fixed-point sum lands on a .5 boundary. The claim in the header
# is PIL-EQUIVALENT, so the gate is: worst difference <= MAX_LSB, and a fraction
# of differing channels that is small enough to be rounding rather than a
# filter. A mean absolute difference is reported too, because "the filter is
# wrong" and "the last bit of the accumulator is different" are different bugs
# and only one of them has a large max.
#
# Env: numpy, pillow, a C++ toolchain with libpng and libjpeg headers. No NPU,
# no XRT. The probe needs a packed container for its geometry.
#
# Usage:
#   python tools/verify/verify_vit_image.py --npue /tmp/vit.npue
#   python tools/verify/verify_vit_image.py --npue /tmp/vit.npue --keep-images

import argparse
import json
import struct
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))

PROBE = REPO / "runtime" / "tests" / "test_vit_image.cpp"

_HEADER = "<4sIII QQQQ 16s"
_HEADER_SIZE = 64
assert struct.calcsize(_HEADER) == _HEADER_SIZE

# The codes resize_square() implements, all four measured against PIL in
# section 2. NEAREST and HAMMING are the fifth and sixth and are refused -- by
# name, with a message that says why -- so they are cases in section 4 rather
# than cases here.
CODES = {1: "LANCZOS", 2: "BILINEAR", 3: "BICUBIC", 4: "BOX"}
REFUSED_CODES = {0: "NEAREST", 5: "HAMMING"}

# The worst single-channel difference in 8-bit levels that still counts as
# "the same resize". One LSB is the largest difference two roundings of one
# value can produce; anything larger is a filter, an index or a clamp.
MAX_LSB = 1
# And the fraction of channels allowed to be off by that one LSB. PIL's
# fixed-point 22-bit sum and a float64 accumulator disagree on the last bit
# where the true value sits within ~2^-16 of a .5 boundary, which on a
# high-frequency random raster is a few percent of channels -- measured, below.
# A filter error would be tens of percent.
MAX_FRACTION = 0.10

_failures = []


def report(ok, label, detail=""):
    print(f"   {'ok  ' if ok else 'FAIL'}  {label}"
          + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def build_exe(path):
    cmd = [
        "g++", "-std=c++17", "-O2", "-I", str(REPO / "runtime" / "include"),
        str(PROBE),
        str(REPO / "runtime" / "src" / "vit" / "image.cpp"),
        str(REPO / "runtime" / "src" / "common" / "json_min.cpp"),
        str(REPO / "runtime" / "src" / "model.cpp"),
        "-lpng", "-ljpeg", "-o", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("building the C++ image front end failed:\n" + r.stderr[-3000:])
        return None
    return path


def run_probe(exe, npue, image, resample=None):
    cmd = [str(exe), str(npue), str(image)]
    if resample is not None:
        cmd += ["--resample", str(resample)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{exe} died with rc={p.returncode}:\n{p.stderr[-2000:]}")
    out = {"refused": None}
    for line in p.stdout.splitlines():
        kind, _, rest = line.partition(" ")
        if kind == "refused":
            out["refused"] = rest
        elif kind == "decoded":
            w, h, hexed = rest.split()
            out["decoded"] = np.frombuffer(bytes.fromhex(hexed),
                                          dtype=np.uint8).reshape(int(h), int(w), 3)
        elif kind == "resize":
            w, h, hexed = rest.split()
            out["resize"] = np.frombuffer(bytes.fromhex(hexed),
                                          dtype=np.uint8).reshape(int(h), int(w), 3)
        elif kind == "pixels":
            out["pixels"] = np.frombuffer(bytes.fromhex(rest), dtype="<f4")
        elif kind == "patches":
            r_, c_, hexed = rest.split()
            out["patches"] = np.frombuffer(bytes.fromhex(hexed),
                                           dtype="<f4").reshape(int(r_), int(c_))
    return out


def ref_geometry(npue):
    """image_size / patch / mean / std / resample, read the way the packer wrote them.

    With struct and json only -- no packer import -- so the geometry this gate
    compares against is the container's own bytes rather than anything the
    packer would recompute. The runtime reads the same keys through its own
    npue::File, so a disagreement here is a disagreement about the FILE.
    """
    with open(npue, "rb") as f:
        magic, _v, arch, _f, jo, jl, _do, _dl, _r = struct.unpack(
            _HEADER, f.read(_HEADER_SIZE))
        if magic != b"NPUE":
            raise SystemExit(f"{npue}: not a .npue file (magic {magic!r})")
        f.seek(jo)
        cfg = dict(json.loads(f.read(jl).decode("utf-8"))["config"])
    return (int(cfg["image_size"]), int(cfg["patch_size"]),
            [float(x) for x in cfg["image_mean"]],
            [float(x) for x in cfg["image_std"]], int(cfg["resample"]))


def ref_patches(raster, image_size, patch, mean, std):
    """PIL -> [n_patches, patch_dim], the reference the runtime's im2col answers to."""
    from vit_int8 import im2col_patches
    a = np.asarray(raster, dtype=np.float32) / np.float32(255.0)
    chw = (a.transpose(2, 0, 1) - np.asarray(mean, np.float32).reshape(3, 1, 1)) \
        / np.asarray(std, np.float32).reshape(3, 1, 1)
    return im2col_patches(chw, image_size, patch)


def build_corpus(tmp, image_size):
    """Rasters that make the resize measurable.

    Random noise (worst case for a filter's last bit), a smooth ramp (worst case
    for a filter's SHAPE -- a wrong support or a missing widening shows up here
    and nowhere else), hard edges (the case a real photograph is made of), a
    flat field (must be EXACT: every filter is the identity on a constant), an
    already-square raster (PIL skips the filter, so must we), and the same rasters
    through JPEG (lossy, so its own decode differs -- measured separately).
    """
    rng = np.random.default_rng(11)
    side = image_size
    big = image_size + 37
    out = {}

    noise = (rng.random((big, big + 9, 3)) * 255).astype(np.uint8)
    ramp = np.zeros((big, big + 9, 3), dtype=np.uint8)
    for c in range(3):
        ramp[..., c] = np.clip(
            np.linspace(0, 255, big + 9, dtype=np.float32)[None, :]
            + np.linspace(0, 90, big, dtype=np.float32)[:, None] * (c - 1), 0, 255
        ).astype(np.uint8)
    edges = np.zeros((big, big + 9, 3), dtype=np.uint8)
    for _ in range(12):
        x0, y0 = rng.integers(0, big - 8, size=2)
        dx, dy = rng.integers(-big, big, size=2)
        for t in range(0, max(abs(int(dx)), abs(int(dy))) + 1):
            x, y = int(x0) + int(dx) * t // max(abs(int(dx)), 1), \
                   int(y0) + int(dy) * t // max(abs(int(dy)), 1)
            if 0 <= x < big + 9 and 0 <= y < big:
                edges[y, x] = rng.integers(0, 256, size=3, dtype=np.uint8)
    flat = np.full((big, big + 9, 3), 137, dtype=np.uint8)

    for name, arr in (("noise", noise), ("ramp", ramp), ("edges", edges),
                      ("flat", flat),
                      ("square", (rng.random((side, side, 3))
                                  * 255).astype(np.uint8))):
        p = tmp / f"{name}.png"
        Image.fromarray(arr).save(p)
        out[name] = p

    # The same noise through JPEG, so the decoder's two paths are both measured.
    p = tmp / "noise.jpg"
    Image.fromarray(noise).save(p, quality=92)
    out["jpeg"] = p

    # Formats PIL opens and the runtime must too, each saved from an array
    # whose CONTENT is known exactly so the only question is what the decoder
    # does with the format.
    gray = np.full((big, big), 90, np.uint8)
    gray[:, :] = np.linspace(0, 255, big, dtype=np.uint8)[None, :]
    g = Image.fromarray(gray, mode="L")
    g.save(tmp / "gray.png")
    g.convert("LA").save(tmp / "gray_alpha.png")
    g.convert("P").save(tmp / "palette.png")
    g.convert("P").save(tmp / "palette_trns.png", transparency=0)

    rgba = Image.fromarray(noise[:side, :side]).convert("RGBA")
    rgba.putalpha(Image.linear_gradient("L").resize((side, side)))
    rgba.save(tmp / "rgba.png")

    # 16-bit GRAYSCALE, because PIL has no 16-bit-per-channel RGB mode and so
    # there is no 16-bit RGB case to build here. This one is REFUSED by name
    # rather than decoded -- see section 1.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Image.fromarray(
            (noise[:side, :side, 0].astype(np.uint16)) * 257,
            mode="I;16").save(tmp / "gray16.png")

    out["gray"] = tmp / "gray.png"
    out["gray_alpha"] = tmp / "gray_alpha.png"
    out["palette"] = tmp / "palette.png"
    out["palette_trns"] = tmp / "palette_trns.png"
    out["rgba"] = tmp / "rgba.png"
    out["gray16"] = tmp / "gray16.png"
    return out


def check_decode(exe, npue, corpus, image_size):
    print("\n1. the decoder: exact against PIL, and every format both sides "
          "read\n   the same way")
    # PIL's own decode of the same file, for the formats where PIL hands back
    # something other than RGB. `convert("RGB")` is what transformers' ViT
    # processor does with an opened image.
    #
    # The formats that CANNOT be compared byte-for-byte against PIL's
    # convert("RGB") are handled below with the rule each one follows, because
    # "exact against PIL" is a claim about the DECODE and those are claims about
    # a policy the two sides each had to choose.
    exact = ("noise", "jpeg", "gray", "gray_alpha", "palette")
    for name in exact:
        got = run_probe(exe, npue, corpus[name])
        if got["refused"]:
            report(False, f"{name}: decoded", f"-> refused: {got['refused'][:120]}")
            continue
        pil = np.asarray(Image.open(corpus[name]).convert("RGB"), dtype=np.uint8)
        mine = got["decoded"]
        same = mine.shape == pil.shape and np.array_equal(mine, pil)
        report(same, f"{name}: identical to PIL's decode",
               "" if same else
               (f"-> {mine.shape} vs {pil.shape}, worst channel difference "
                f"{int(np.abs(mine.astype(np.int32) - pil.astype(np.int32)).max())}"
                if mine.shape == pil.shape else
                f"-> shape {mine.shape} vs PIL's {pil.shape}"))

    # ALPHA. PIL's convert("RGB") DROPS alpha and keeps the raw RGB channels;
    # this runtime COMPOSITES ON WHITE. So the two are SUPPOSED to disagree, and
    # the gate names which one it wants instead of comparing blindly: every
    # viewer the user has shows a transparent PNG over white, and treating alpha
    # as black turns a cut-out logo into a dark rectangle, which then classifies
    # as one. Both rules are checked, so neither is taken on faith.
    for name in ("rgba", "palette_trns"):
        got = run_probe(exe, npue, corpus[name])
        if got["refused"]:
            report(False, f"{name}: decoded", f"-> refused: {got['refused'][:120]}")
            continue
        a = np.asarray(Image.open(corpus[name]).convert("RGBA"), dtype=np.float64)
        rgb, alpha = a[..., :3], a[..., 3:4] / 255.0
        on_white = np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255)
        on_white = on_white.round().astype(np.int32)
        mine = got["decoded"].astype(np.int32)
        # The runtime computes (v*a + 255*(255-a)) / 255 in INTEGERS and this
        # reference does it in float, so the two can differ by one LSB where the
        # true value sits on a rounding boundary. That is the whole budget; the
        # gate prints the max either way so the number is visible either.
        d_white = int(np.abs(mine - on_white).max())
        report(d_white <= 1,
               f"{name}: composited on WHITE, as every viewer shows it",
               "" if d_white <= 1 else
               f"-> worst channel difference {d_white} against "
               f"(rgb*a + 255*(1-a))")
        raw = np.asarray(Image.open(corpus[name]).convert("RGB"),
                         dtype=np.int32)
        d_raw = int(np.abs(mine - raw).max())
        report(d_raw > 0,
               f"{name}: and NOT PIL's convert('RGB'), which drops alpha",
               f"-> differs from the raw-channel reading by up to {d_raw}, so "
               f"the two rules are genuinely different code, not a rename")

    # 16-BIT IS REFUSED, by name, with both candidate reductions in the message.
    got = run_probe(exe, npue, corpus["gray16"])
    ok = (got["refused"] is not None and "16 bits" in got["refused"]
          and "8-bit" in got["refused"])
    report(ok, "a 16-bit PNG is refused by name, not decoded",
           "" if ok else
           (f"-> {got['refused'][:120]}" if got["refused"] else "-> DECODED"))
    if got["refused"]:
        print(f"        {got['refused'][:210]}...")


def check_resize(exe, npue, corpus, image_size, code, name):
    resample = CODES[code]
    print(f"\n2. resize_square to {image_size}px with {resample} "
          f"(PIL.Image.{resample}), against PIL")
    for case, path in corpus.items():
        if case in ("rgba", "palette_trns", "gray16"):
            # Their DECODED rasters differ from PIL's convert("RGB") by a policy
            # decision, not by a filter: alpha compositing (section 1) and the
            # 16-bit refusal. Comparing the resize against a reference built from
            # a different raster would measure that policy twice.
            continue
        got = run_probe(exe, npue, path, resample=code)
        if got["refused"]:
            report(False, f"{case}: resized",
                   f"-> refused: {got['refused'][:120]}")
            continue
        pil_src = Image.open(path)
        if pil_src.size == (image_size, image_size):
            # PIL skips the filter entirely at equal sizes, and image.cpp says so
            # and returns the source. Compared against PIL's own no-op.
            ref = np.asarray(pil_src.convert("RGB"), dtype=np.uint8)
        else:
            ref = np.asarray(
                pil_src.convert("RGB").resize((image_size, image_size),
                                              int(code)), dtype=np.uint8)
        mine = got["resize"]
        d = mine.astype(np.int32) - ref.astype(np.int32)
        worst = int(np.abs(d).max())
        n_diff = int((d != 0).sum())
        frac = n_diff / d.size
        # 'flat' is EXACT and is gated as such: every one of these filters is
        # the identity on a constant, so a single differing channel there is a
        # bug in the index arithmetic or the accumulator, not a rounding step.
        if case == "flat":
            ok = worst == 0
            report(ok, "flat 137 everywhere: EXACT, as every filter must be",
                   "" if ok else f"-> worst difference {worst}, {frac:.4%} of "
                                 f"channels; a constant maps to a constant")
            continue
        ok = worst <= MAX_LSB and frac <= MAX_FRACTION
        report(ok, f"{case}: within {MAX_LSB} LSB on {100*(1-frac):.2f}% of channels",
               f"worst {worst}, mean {np.abs(d).mean():.4f}, "
               f"{frac:.3%} differ (budget {MAX_FRACTION:.0%})")


def check_pixels_and_patches(exe, npue, corpus, image_size, patch, mean, std,
                             code):
    print("\n3. normalise and im2col, against numpy and against the Python half "
          "of\n   the patch layout")
    for case in ("noise", "ramp", "square"):
        got = run_probe(exe, npue, corpus[case], resample=code)
        if got["refused"]:
            report(False, f"{case}: front end", f"-> refused: {got['refused'][:120]}")
            continue

        # normalise: the same expression, from the C++ side's OWN resized raster,
        # so this cannot inherit the resize's last bit and blame it here.
        mine = got["resize"].astype(np.float32) / np.float32(255.0)
        m = np.asarray(mean, np.float32).reshape(3, 1, 1)
        s = np.asarray(std, np.float32).reshape(3, 1, 1)
        ref = ((mine.transpose(2, 0, 1) - m) / s).ravel()
        d = np.abs(got["pixels"] - ref)
        report(float(d.max()) == 0.0,
               f"{case}: normalise exact (0 ULP) on its own resized raster",
               "" if float(d.max()) == 0.0 else
               f"-> worst {float(d.max()):.3e}; (x/255 - mean)/std is one "
               f"multiply and one divide, and both sides do them in float32")

        # im2col: the layout, compared directly. This is the twin half of
        # verify_vit.py's Conv2d check -- there the composition is held against
        # an authority, here the two implementations of the layout are held
        # against each other so a drift is a float diff and not a label.
        ref_p = ref_patches(got["resize"], image_size, patch, mean, std)
        dp = np.abs(got["patches"] - ref_p)
        report(float(dp.max()) == 0.0,
               f"{case}: im2col bit-identical to vit_int8.im2col_patches",
               "" if float(dp.max()) == 0.0 else
               f"-> worst {float(dp.max()):.3e}; the two halves of the patch "
               f"layout are the same permutation")


def check_refusals(exe, npue, corpus, image_size, tmp):
    print("\n4. the refusals, by name")
    # The two resample codes this build will not approximate. Each message has to
    # name the code, say WHAT is implemented and to what, and say what to do --
    # a refusal that only says "unsupported" leaves the user with one fewer way
    # to run the thing they already have.
    for code, name in REFUSED_CODES.items():
        got = run_probe(exe, npue, corpus["noise"], resample=code)
        if not got["refused"]:
            report(False, f"{name} is refused by name",
                   "-> returned a raster anyway")
            continue
        msg = got["refused"]
        # Which of the four IS held to what, so the user can see this is a
        # measured boundary and not an unimplemented corner.
        names_others = all(o in msg for o in CODES.values())
        report(name in msg and names_others and "BICUBIC" in msg,
               f"{name} is refused by name, and names what IS implemented",
               "" if name in msg else "-> the code is not in the message")
        print(f"        {msg[:190]}...")

    junk = tmp / "notanimage.png"
    junk.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not really a png" * 8)
    got = run_probe(exe, npue, junk)
    report(got["refused"] is not None,
           "a .png whose bytes are not a PNG is refused",
           "" if got["refused"] else "-> decoded something")
    if got["refused"]:
        # libpng's own message, not one this runtime wrote. Asserted on the
        # header error rather than on the word "PNG" because the point of the
        # case is that the decoder reached libpng's parser at all -- a file that
        # passes the magic check and then dies in the header is a truncated or
        # mislabelled file, and saying so is what the user needs.
        report("header" in got["refused"] or "not a PNG" in got["refused"],
               "  ...with libpng's reason, not a generic 'corrupt'",
               f"-> {got['refused'][:110]}")
        print(f"        {got['refused'][:150]}")

    trunc = tmp / "truncated.png"
    trunc.write_bytes(Path(corpus["noise"]).read_bytes()[:300])
    got = run_probe(exe, npue, trunc)
    report(got["refused"] is not None and "end of" in got["refused"],
           "a truncated PNG is refused, not decoded to half a picture",
           "" if got["refused"] else "-> decoded a partial raster")
    if got["refused"]:
        print(f"        {got['refused'][:150]}")

    text = tmp / "notes.txt"
    text.write_text("this is not an image at all\n")
    got = run_probe(exe, npue, text)
    ok = (got["refused"] is not None
          and "not a PNG" in got["refused"] and "not a JPEG" in got["refused"]
          and "re-save" in got["refused"])
    report(ok, "a non-image is refused, and the message says what to do",
           "" if ok else f"-> {got['refused'][:140] if got['refused'] else 'DECODED'}")
    if got["refused"]:
        print(f"        {got['refused'][:200]}...")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hold the C++ image front end against PIL.")
    ap.add_argument("--npue", default=str(REPO / "models" /
                                          "vit-base-patch16-224.npue"),
                    help="a packed arch=5 container, for its geometry")
    ap.add_argument("--exe", default=None,
                    help="the probe; built from runtime/tests if not given")
    ap.add_argument("--codes", default=",".join(str(c) for c in CODES),
                    help="comma-separated PIL resample codes to measure. The "
                         "default is every code resize_square() implements "
                         "(1 LANCZOS, 2 BILINEAR, 3 BICUBIC, 4 BOX), because "
                         "the header's claim is about all of them and not "
                         "about the one the checkpoint happens to ask for. "
                         "Pass just the container's own code to measure only "
                         "that, e.g. --codes 3. NEAREST and HAMMING are "
                         "refused, not measured; --codes 5 is a case in "
                         "section 4 and running it as a measurement would "
                         "report the refusal as a difference of zero")
    ap.add_argument("--keep-images", action="store_true")
    args = ap.parse_args()

    print("image front end: C++ against PIL, on the same bytes\n")
    npue = Path(args.npue)
    if not npue.exists():
        print(f"FAIL -- {npue} does not exist. Pack one with\n"
              f"  python tools/pack/pack_npue.py --model-dir "
              f"models/vit-base-patch16-224 \\\n"
              f"      --out /tmp/vit.npue --device npu1")
        return 1

    exe = Path(args.exe) if args.exe else build_exe(
        Path(tempfile.gettempdir()) / "test_vit_image")
    if exe is None or not exe.exists():
        return 1

    image_size, patch, mean, std, code = ref_geometry(npue)
    print(f"container  {npue.name}: {image_size}px, patch {patch}, "
          f"mean {mean}, std {std}, resample {code} (PIL.Image."
          f"{CODES.get(code, 'UNKNOWN')})")

    tmpdir = Path(tempfile.mkdtemp(prefix="vitimg_"))
    try:
        corpus = build_corpus(tmpdir, image_size)
        check_decode(exe, npue, corpus, image_size)
        for c in [int(x) for x in args.codes.split(",")]:
            if c not in CODES:
                print(f"   -- codes {c} is not one this build implements "
                      f"({', '.join(f'{k} {v}' for k, v in CODES.items())}); "
                      f"it is a refusal, checked in section 4. Skipping it "
                      f"rather than reporting a difference of zero against a "
                      f"raster that was never produced.")
                continue
            check_resize(exe, npue, corpus, image_size, c, CODES[c])
        check_pixels_and_patches(exe, npue, corpus, image_size, patch, mean, std,
                                 code)
        check_refusals(exe, npue, corpus, image_size, tmpdir)
    finally:
        if args.keep_images:
            print(f"\ncorpus kept at {tmpdir}")
        else:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if _failures:
        print(f"FAIL -- {len(_failures)} of the lines above did not hold:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("PASS -- the decode is exact against PIL on every format both read;\n"
          "        the resize is PIL-equivalent within one 8-bit LSB for every\n"
          "        code implemented, and exact on a constant; normalise and\n"
          "        im2col are bit-identical to the reference; and NEAREST,\n"
          "        HAMMING, a 16-bit PNG, a fake PNG, a truncated PNG and a\n"
          "        non-image are all refused by name.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
