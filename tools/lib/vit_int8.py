# NpuEmbeddings -- SmoothQuant calibration for a ViT container's activations.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS EXISTS FOR
# --------------------
# The third member of the int8 family, and the reason it is its own file rather
# than a flag on another one. There is now one quantisation SCHEME
# (tools/lib/gemm_i8.py) and three calibrations, because the estimator is the
# same and the STATISTICS are not:
#
#   BERT-family   text    pack_npue.calibrate_smoothing, a text corpus
#   Whisper       audio   whisper_int8.py, log-mel activations
#   ViT           IMAGES  this file, pixel activations
#
# pack_npue.calibrate_smoothing refuses a whisper container by name ("a whisper
# container's activations are log-mel frames and a decoder state"); the refusal
# this file answers is the same shape -- an image is not a sentence and the
# per-channel outliers of a ViT's patch-embedding input are the picture's, not
# the prose's. Measuring one and calling it the other produces a scale set that
# is wrong in a way no operand-level gate can see.
#
#   s_j = max_i |X[i,j]|^alpha / max_n |W[j,n]|^(1-alpha)
#
# and the array computes (X/s) @ (sW), which is X @ W in exact arithmetic.
#
# WHY HOOKS AND NOT A NUMPY ORACLE
# ---------------------------------
# A forward PRE-hook on the nn.Linear of each site sees exactly the tensor the
# array will multiply, so there is nothing to reason about and no module-order
# assumption to get wrong. The patch-embedding site is the interesting one: its
# input is the im2col'd pixel patch matrix, which exists only inside HF's
# Conv2d, and a hook on `projection` is the only place it is visible at all.
#
# THE CORPUS
# -----------
# Default: a synthetic one built here, so a pack needs no external data and two
# packs of the same checkpoint produce the same scales. It is deliberately NOT a
# constant image: a uniform patch has every channel equal, so the per-channel
# maxima collapse to one number and every downstream diagnosis of "the
# smoothing was wrong" becomes indistinguishable from "the corpus was
# degenerate". This moves edges, sweeps level and hue, and adds structured
# high-frequency content, so the maxima are taken over a spread of natural
# image statistics. A deployment should pass its own images: --int8-corpus.
#
# Env: numpy, torch, transformers, PIL -- BUILD TIME ONLY. The shipped runtime
# stays C++; this runs when a container is packed, never when it is served.

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# The GEMM sites the packer asks about, and the torch module whose INPUT is that
# operand. Keys are CONTAINER names, so the packer looks them up by the string
# it already writes.
#
#   patch_embed              <- embeddings.patch_embeddings.projection (the im2col patches)
#   layer.i.qkv              <- attention.attention.query          (layernorm_before's output)
#   layer.i.attn_out         <- attention.output.dense             (the attention output)
#   layer.i.ffn_up           <- intermediate.dense                 (layernorm_after's output)
#   layer.i.ffn_down         <- output.dense                       (the GELU output)
#
# Only query is hooked on the qkv site: q, k and v share one fused A operand,
# and hooking the fused site's first member is what makes the statistics
# describe the operand the array multiplies.
#
# PATCH_EMBED IS NOT HOOK-BACKED, and that is a fact about the arithmetic
# rather than a workaround. The conv's input is [n, 3, 224, 224] and its channel
# axis is 3; the operand the array multiplies is [M, 768], because 768 = 3*16*16
# is the im2col patch width. A per-input-channel scale set measured on the conv
# input has THREE entries for a 768-row operand, and the packer's own check
# refuses it -- which is the correct outcome and the reason that check exists.
# So this site's statistic is measured on the patch matrix the front end
# actually produces, by im2col_patches, i.e. on the tensor the runtime will
# quantise rather than on a tensor that merely feeds it.
OPS: dict[str, str] = {
    "patch_embed":
        "vit.embeddings.patch_embeddings.projection",
    "layer.{i}.qkv": "vit.encoder.layer.{i}.attention.attention.query",
    "layer.{i}.attn_out": "vit.encoder.layer.{i}.attention.output.dense",
    "layer.{i}.ffn_up": "vit.encoder.layer.{i}.intermediate.dense",
    "layer.{i}.ffn_down": "vit.encoder.layer.{i}.output.dense",
}

# The sites whose statistic comes from im2col_patches rather than a hook. Named
# as a list so that adding a site has to choose, instead of defaulting to the
# hook and producing a scale set of the wrong width.
FRONT_END_SITES = ("patch_embed",)


def site_keys(n_layers: int) -> list[tuple[str, str]]:
    """(container name, hook path) for every site, in a stable order."""
    out = [("patch_embed", OPS["patch_embed"])]
    for i in range(n_layers):
        for tmpl, path in OPS.items():
            if tmpl.startswith("layer."):
                out.append((tmpl.format(i=i), path.format(i=i)))
    return out


# ViT-base-patch16-224's own preprocessor_config.json. Read from the checkpoint
# when there is one; these are the values that file carries, named here so a
# corpus built without transformers still produces the tensor the model wants.
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


def read_preprocessor(model_dir: str | Path) -> dict:
    """The checkpoint's OWN image front end. Refused, not defaulted, when absent.

    The mean/std and the resize are part of the model's definition: a container
    that normalised with another set's numbers is a model that classifies
    noise, and nothing downstream can tell. So the file is required, exactly as
    packers/whisper.py requires preprocessor_config.json for the same reason.
    """
    p = Path(model_dir) / "preprocessor_config.json"
    if not p.exists():
        raise SystemExit(
            f"{p} not found. A ViT's image normalisation (size, resample, "
            "mean, std) is part of the model, and a corpus calibrated against "
            "numbers the checkpoint does not use describes a different model. "
            "Run the packer from a checkpoint directory, not from the weights "
            "alone.")
    pre = json.loads(p.read_text(encoding="utf-8"))
    size = pre.get("size", pre.get("image_size", DEFAULT_IMAGE_SIZE))
    # WHICH resize geometry the checkpoint asks for, as a name rather than a
    # number. `{"height": h, "width": w}` is a square resize to h; HF's OTHER
    # form, `{"shortest_edge": s}`, resizes the shortest edge to s and then
    # centre-crops `crop_size` out of the middle, which is a DIFFERENT tensor and
    # not a different quality -- so the two are reported apart instead of being
    # collapsed onto one integer. A caller that cannot do the shortest-edge form
    # refuses; it does not resize to h and call it s.
    crop = pre.get("crop_size", size)
    if isinstance(crop, dict):
        crop = crop.get("height", crop.get("shortest_edge", DEFAULT_IMAGE_SIZE))
    if isinstance(size, dict) and "shortest_edge" in size:
        geom = "shortest_edge_crop"
        side = int(size["shortest_edge"])
    else:
        if isinstance(size, dict):
            if "height" in size and int(size["height"]) != int(size.get("width", -1)):
                raise SystemExit(
                    f"{p}: size is {size}, a non-square resize. This packer's "
                    f"containers hold one square position table sized for "
                    f"image_size x image_size, so a non-square front end cannot "
                    f"be reproduced by them at all -- refusing rather than "
                    f"guessing which side wins.")
            side = int(size.get("height", DEFAULT_IMAGE_SIZE))
        else:
            side = int(size)
        geom = "square"
    return {
        "image_size": side,
        "resize": geom,
        "crop_size": int(crop),
        "mean": tuple(float(x) for x in pre.get("image_mean", DEFAULT_MEAN)),
        "std": tuple(float(x) for x in pre.get("image_std", DEFAULT_STD)),
        "resample": pre.get("resample", 3),   # PIL.Image.BICUBIC
    }


def synth_images(n_images: int = 8, size: int = DEFAULT_IMAGE_SIZE,
                 seed: int = 5) -> list:
    """Deterministic natural-image STATISTICS, as PIL images.

    Not photos and not a flat field. What SmoothQuant is measuring here is the
    per-input-channel MAXIMUM of a patch matrix, and the properties that make a
    maximum interesting are edges and level, so this builds images out of
    those: smooth low-frequency ramps, hard edges at random angles, texture at
    two scales, and a per-image level and hue shift. A uniform colour would give
    a per-channel maximum equal to the colour itself and a scale set that
    happens to be right for that one picture and for nothing else.
    """
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n_images):
        side = size + 32
        base = rng.uniform(0.15, 0.85)
        img = np.zeros((side, side, 3), dtype=np.float32)
        # low-frequency ramps in all three channels, independently
        for ch in range(3):
            ax = rng.uniform(0.2, 1.0, size=3)
            yy = np.linspace(0.0, 1.0, side, dtype=np.float32)[:, None]
            xx = np.linspace(0.0, 1.0, side, dtype=np.float32)[None, :]
            img[..., ch] = np.clip(
                base * (0.35 + 0.65 * (ax[0] * xx + ax[1] * yy + ax[2] * (xx * yy))
                        / (ax.sum() or 1.0)), 0.0, 1.0)
        # texture at two scales, so the patch matrix has a high-frequency tail
        for octave, amp in ((4, 0.22), (16, 0.10)):
            g = rng.standard_normal((octave, octave, 3)).astype(np.float32)
            up = np.kron(g, np.ones((side // octave, side // octave, 1),
                                    dtype=np.float32))
            img = np.clip(img + amp * up[:side, :side], 0.0, 1.0)
        im = Image.fromarray((img * 255).astype(np.uint8))
        # hard edges: the actual per-channel outliers of a real photograph
        d = ImageDraw.Draw(im)
        for _ in range(6):
            x0, y0 = (int(v) for v in rng.integers(0, side - 20, size=2))
            d.line([(x0, y0),
                    (int(x0 + rng.integers(-side, side)),
                     int(y0 + rng.integers(-side, side)))],
                   fill=tuple(int(v) for v in rng.integers(0, 256, size=3)),
                   width=int(rng.integers(1, 9)))
        im = im.resize((size, size), Image.BICUBIC)
        if c % 4 == 3:
            im = im.point(lambda v: int(v * 0.25))    # one dark, low-contrast image
        out.append(im)
    return out


def read_corpus(path: str | Path) -> list:
    """Every image under `path` (png/jpg/jpeg/bmp/webp), opened with PIL."""
    from PIL import Image
    p = Path(path)
    files = sorted(q for q in (p.rglob("*") if p.is_dir() else [p])
                   if q.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp",
                                           ".webp", ".ppm", ".tif", ".tiff"})
    if not files:
        raise SystemExit(f"{path}: no image files (*.png, *.jpg, ...) under it")
    return [Image.open(f).convert("RGB") for f in files]


def preprocess(images: list, image_size: int, mean, std, resample: int = 3):
    """PIL images -> the [n, 3, S, S] float32 tensor transformers' ViT wants.

    Written out rather than delegated, because the calibration corpus must go
    through the SAME arithmetic as the pack's own images and a delegate would
    make that an assumption about a library's default antialias flag. bicubic
    with antialiasing, 1/255, then (x - mean) / std per channel.
    """
    from PIL import Image
    out = np.empty((len(images), 3, image_size, image_size), dtype=np.float32)
    m = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    for i, im in enumerate(images):
        if im.size != (image_size, image_size):
            im = im.resize((image_size, image_size), int(resample))
        a = np.asarray(im, dtype=np.float32) / np.float32(255.0)   # HWC, [0,1]
        out[i] = (a.transpose(2, 0, 1) - m) / s                    # CHW, normalised
    return out


def im2col_patches(pixel, image_size: int, patch: int) -> np.ndarray:
    """[3,S,S] normalised pixels -> [n_patches, 3*patch*patch] fp32.

    THE PIXEL HALF OF THE PATCH LAYOUT, and the twin of
    packers.vit.patch_embed_operand, which is the weight half. They are two
    halves of one fact -- what a "patch" is and in which order its floats are
    laid out -- and they live in different modules because one is the image
    front end and the other is a container's operand. The consequence is
    deliberate and stated here: a drift between them is possible, so the
    composition is what gets held against an authority, not either half.
    tools/verify/verify_vit.py section 1 asserts

        im2col_patches(px) @ patch_embed_operand(W) == Conv2d(W)(px)

    against HF's own convolution, which is the only thing in the tree that
    knows both halves at once.

    The flattening is channel-major outside, row-major inside: (c, i, j) maps to
    (c*patch + i)*patch + j, which is the k index patch_embed_operand builds.
    Written as an explicit loop rather than a reshape because get it backwards
    and every number downstream is a plausible number -- the GEMM stays well
    formed, the attention still normalises, and the model confidently classifies
    noise.

    Rows come out in (patch_row, patch_col) order, which is what a stride-equals-
    -kernel Conv2d emits after `permute(0, 2, 3, 1)`.
    """
    pixel = np.asarray(pixel, dtype=np.float32)
    c, s, s2 = pixel.shape
    if (s, s2) != (image_size, image_size) or image_size % patch:
        raise SystemExit(
            f"image is {s}x{s2}px, expected {image_size}px with patch {patch}: "
            f"not divisible, and this im2col is a permutation, not a strided "
            f"window. A different image size needs a different front end.")
    n = image_size // patch
    out = np.empty((n * n, c * patch * patch), dtype=np.float32)
    for p in range(n):
        for q in range(n):
            blk = pixel[:, p * patch:(p + 1) * patch,
                        q * patch:(q + 1) * patch]            # [c, patch, patch]
            out[p * n + q] = blk.reshape(-1)                  # C order: (c,i,j)
    return out


def load_or_make_corpus(corpus: str | None, n_images: int, image_size: int):
    if corpus:
        imgs = read_corpus(corpus)
        return imgs, f"{corpus} ({len(imgs)} images)"
    imgs = synth_images(n_images, image_size)
    return imgs, f"synthetic, {len(imgs)} images (synth_images)"


def calibrate(model_dir: str | Path, n_layers: int, images: list,
              pixel_values, alpha: float = 0.5, verbose: bool = True):
    """Per-(layer, op) smoothing factors, keyed by CONTAINER operand name.

    `pixel_values` is the [n, 3, S, S] tensor built by `preprocess` -- passed in
    rather than recomputed, so the corpus the packer printed and the corpus the
    scales were measured on are the same object.
    """
    import torch
    from transformers import ViTForImageClassification

    mp = Path(model_dir)
    model = ViTForImageClassification.from_pretrained(
        str(mp), torch_dtype=torch.float32).eval()

    wanted = dict(site_keys(n_layers))
    hook_sites = {k: p for k, p in wanted.items() if k not in FRONT_END_SITES}
    seen: dict[str, str] = {}
    for key, path in hook_sites.items():
        if path in seen:
            raise SystemExit(f"two container keys share the hook {path}: "
                             f"{seen[path]} and {key}. The site list is the "
                             f"place to fix it, not the hooks.")
        seen[path] = key

    amax: dict[str, np.ndarray] = {}

    # The front-end sites, measured on the patch matrix rather than on a hook.
    # Done first and on the SAME pixel_values the model is then run on, so the
    # corpus the two halves saw is identical and no statistic is attributed to
    # an image the rest of the calibration did not see.
    img = int(model.config.image_size)
    pch = int(model.config.patch_size)
    for key in FRONT_END_SITES:
        for px in pixel_values:
            m = im2col_patches(px, img, pch)
            a = np.abs(m).max(axis=0).astype(np.float32)
            amax[key] = a if key not in amax else np.maximum(amax[key], a)

    def hook(path: str):
        def fn(module, inputs):
            x = inputs[0]
            # Axis -1, and it is not a guess: every hooked site is an nn.Linear
            # (see OPS -- the one Conv2d in this model is a FRONT_END_SITE), and
            # a Linear's input is [..., in_features], so the channels ARE the
            # last axis. If a conv is ever hooked, this is the line that has to
            # change, and the packer's channel-count check is what catches it.
            with torch.no_grad():
                v = (x.detach().abs().reshape(-1, x.shape[-1])
                     .amax(dim=0))
            k = seen[path]
            a = v.to(torch.float32).numpy()
            amax[k] = a if k not in amax else np.maximum(amax[k], a)
        return fn

    handles = []
    for name, module in model.named_modules():
        if name in seen:
            handles.append(module.register_forward_pre_hook(hook(name)))
    if len(handles) != len(seen):
        missing = sorted(set(seen) - {n for n, _ in model.named_modules()})
        raise SystemExit(
            f"vit int8: hooked {len(handles)} of {len(seen)} sites; not found: "
            f"{missing[:4]}. The module names in OPS do not match this "
            f"checkpoint's tree; fix OPS, do not ship fewer statistics than "
            f"sites.")

    n_patches = 0
    with torch.no_grad():
        # Chunked so a 12-layer ViT on a 13 GB box does not hold every site's
        # activation at once. amax is a running max, so the chunking does not
        # change the statistic -- which is the property that makes it legal.
        chunk = max(1, min(len(images), 4))
        for i in range(0, len(images), chunk):
            batch = torch.from_numpy(pixel_values[i:i + chunk])
            model(pixel_values=batch)
            n_patches += int(batch.shape[0]) * int(
                (model.config.image_size // model.config.patch_size) ** 2 + 1)
    for h in handles:
        h.remove()

    # Every site the packer will ask about must have a statistic, including the
    # front-end ones. A site that silently produced nothing would reach
    # `factors()` as a zero-length array and then broadcast to nothing at all.
    missing = sorted(set(wanted) - set(amax))
    if missing:
        raise SystemExit(
            f"vit int8: no statistic for {len(missing)} of {len(wanted)} sites: "
            f"{missing[:4]}. A site without a statistic is not a site with a "
            f"default one; fix the site list or the calibration, do not ship "
            f"fewer scales than operands.")

    out = {k: a.astype(np.float32) for k, a in amax.items()}
    if verbose:
        print(f"  vit int8: calibrated on {len(images)} images, "
              f"{n_patches} patch rows, {len(amax)} GEMM sites "
              f"({n_layers} layers + patch_embed), alpha={alpha}")
    return out


def smooth_error(mat: np.ndarray, s: np.ndarray) -> float:
    """Relative error of a smoothed operand, for the report.

    NOT the quantisation error `add_gemm_b_int8` returns: that measures
    Wq*s against W*s, this measures the product's sensitivity to s. Printed so a
    bad s is visible at pack time.
    """
    m = np.asarray(mat, dtype=np.float64)
    sm = m * s[:, None]
    den = float(np.linalg.norm(m))
    if not den:
        return 0.0
    return float(np.linalg.norm(sm - m) / den)


def factors(amax: np.ndarray, w_max: np.ndarray, alpha: float) -> np.ndarray:
    """The SmoothQuant ratio, shared so the packer and any checker agree.

    `w_max` is the per-INPUT-channel maximum of the [K, N] matrix that will be
    quantised -- post-fold, post-transpose, i.e. the rows the array multiplies.
    Keeping this in one function is what stops a calibration and its gate from
    drifting by an eps into a difference nobody can attribute.
    """
    a = np.maximum(np.asarray(amax, np.float64), 1e-8)
    w = np.maximum(np.asarray(w_max, np.float64), 1e-8)
    s = (a ** alpha) / (w ** (1.0 - alpha))
    return np.where(np.isfinite(s) & (s > 0), s, 1.0).astype(np.float32)


if __name__ == "__main__":  # a smoke test, not a gate
    sys.path.insert(0, str(REPO / "tools" / "lib"))
    print("import ok; sites for 12 layers:", len(site_keys(12)))
    for k, p in site_keys(2):
        print(" ", k, "->", p)