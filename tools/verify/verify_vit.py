# NpuEmbeddings -- a ViT container, run in fp32 on the host, held against
# transformers. The gate for arch=5.
# SPDX-License-Identifier: Apache-2.0
#
# WHAT THIS IS, AND WHY IT IS THE FIRST GATE
# ------------------------------------------
# arch=5 exists to test one claim: a new modality is a new FRONT END, not a new
# array. So the first thing that has to be true is that the container, read back
# and computed exactly as the runtime computes it, is the reference model. If
# that is not true then nothing measured on the array afterwards means anything.
#
# So this gate reproduces the runtime's arithmetic rather than a paraphrase of
# it:
#
#   * the patch matrix is built by im2col over the container's OWN patch-embed
#     operand, in the layout packers/vit.py wrote, not by HF's Conv2d -- a
#     harness that built patches with HF and compared them to a container would
#     agree even if the container's ordering were transposed, which is the one
#     bug in this arch that produces a perfectly plausible image of noise;
#   * the fused qkv's folded Q scale is taken from the container's config, so
#     `--no-fold-scale` is a different container and a different expectation;
#   * pre-LN is computed pre-LN, and the LayerNorm is fp64 with eps inside the
#     sqrt, because a pre-LN stack that normalises post-LN still returns a
#     vector -- it returns the wrong one;
#   * attention is fp64 row-softmax with the max subtracted;
#   * the head is applied to row 0 after the final LayerNorm.
#
# WHAT IT DELIBERATELY DOES NOT DO: run the array. There is no design set for
# arch=5 in this tree and no claim about its speed. A container that passes here
# and then fails on hardware is a bug in the dispatch; a container that fails
# here and never reaches hardware is a bug caught for the price of this file.
#
# THE int8 HALF. A separate pass dequantises the container the way
# runtime/include/common/host_kernels.hpp does -- per-row activation scale, int32
# accumulator, rank-1 (sa * wscale) output, activation divided by asmooth before
# quantising -- and reports 1-cos of the LOGITS and the top-1 agreement against
# fp32. Those are MODEL numbers, not operand numbers: the per-operand
# quantisation error the packer prints is the weight half of the cost and
# comparing it to a model tolerance is the category error gemm_i8.py warns about.
#
# Env: numpy, torch, transformers, PIL. Usage:
#   python tools/verify/verify_vit.py --container models/vit-base-patch16-224.npue

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))
sys.path.insert(0, str(REPO / "tools" / "pack"))
sys.path.insert(0, str(REPO / "reference"))

from npue import Reader                                        # noqa: E402
from encoder import erf                                       # noqa: E402
from vit_int8 import im2col_patches, read_preprocessor, synth_images, preprocess
from packers.vit import patch_embed_operand                    # noqa: E402

# Tolerances. The fp32 gate is tight because it is the same arithmetic in a
# different order of operations; the int8 gate is loose because it is a
# different arithmetic, and the number that matters is top-1 agreement, not the
# cosine.
#
# RTOL_INT8 is a PER-OPERAND number and is deliberately not the model tolerance
# below: it says the container's patch_embed panel is within 2% of the
# checkpoint's matrix, which is a statement about one weight matrix. COS_INT8
# is a statement about 1000 logits after 12 layers and 197 positions, and the
# two are not comparable -- gemm_i8.py warns about exactly this substitution.
RTOL_FP32 = 2e-3
RTOL_INT8 = 2e-2
COS_FP32 = 1.0 - 1e-5
COS_INT8 = 1.0 - 5e-2
TOP1_INT8 = 0.99

_failures: list[str] = []


def report(ok, label, detail=""):
    print(f"   {'ok  ' if ok else 'FAIL'}  {label}"
          + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def cos(a, b):
    x = np.asarray(a, np.float64).ravel()
    y = np.asarray(b, np.float64).ravel()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


# -- the image front end, as the container says to do it -------------------
#
# The im2col itself is vit_int8.im2col_patches, NOT a third copy written here.
# It and packers.vit.patch_embed_operand are the two halves of the patch layout
# and they live in different modules (front end vs container operand), so the
# thing worth testing is the COMPOSITION of the two -- section 1 below holds
# `im2col_patches(px) @ patch_embed_operand(W)` against HF's own Conv2d, which
# is the only authority in the tree that knows both halves at once.
make_patches = im2col_patches


# -- the forward pass, fp32, in the container's own layout ------------------

def layernorm(x, gamma, beta, eps):
    """PyTorch LayerNorm over the last axis, in fp64.

    eps goes INSIDE the sqrt, and the variance is the BIASED one (divide by N).
    Both are landmines and both are silent: a post-LN stack hides them in the
    residual, a pre-LN stack propagates them into every later layer.
    """
    x = x.astype(np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    c = x - mu
    var = (c * c).mean(axis=-1, keepdims=True)
    return ((c / np.sqrt(var + eps)) * gamma.astype(np.float64)
            + beta.astype(np.float64)).astype(np.float32)


def gelu(x):
    """Exact erf GELU. NOT the tanh approximation -- see reference/encoder.py.

    The erf comes from the reference oracle rather than being spelled out again
    here: two hand-rolled erf's would be two chances to spell it differently,
    and this one has already been held against torch.
    """
    x64 = x.astype(np.float64)
    return (0.5 * x64 * (1.0 + erf(x64 / math.sqrt(2.0)))).astype(np.float32)


def softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def attention(q, k, v, heads, head_dim, scale):
    """Row-softmax attention over one image's positions, in fp64.

    [M, heads*hd] -> [1, heads, M, hd] before the contractions, which is the
    shape reference/encoder.py uses. Not a plain einsum over the [M, heads, hd]
    form: the contraction indices land on different axes of the two operands
    there, and numpy's einsum remaps by POSITION, so it broadcasts a length-M
    axis against a length-heads one and fails. Making the batch axis explicit
    is the same arithmetic with the axis order stated.

    `scale` is a PARAMETER and is passed as 1.0 when the container folded
    1/sqrt(head_dim) into Q, because applying it a second time is not a rounding
    difference -- it is a 0.125 factor on every score, which moves the softmax
    and returns a confident wrong answer. The caller reads which of the two
    containers it is holding out of config["qkv_scale_folded"] rather than
    assuming, so a --no-fold-scale pack is checked against its own arithmetic
    instead of being silently scored as if it had been folded.
    """
    def split4(t):
        return t.reshape(t.shape[0], heads, head_dim).transpose(1, 0, 2)[None]

    s = split4(q) @ split4(k).transpose(0, 1, 3, 2) * scale
    s = softmax(s)
    ctx = (s @ split4(v))[0].transpose(1, 0, 2)          # [M, heads, hd]
    return s, ctx.reshape(ctx.shape[0], heads * head_dim)


def forward(emb, patches, r, cfg, i8=False):
    """The container's forward pass, in fp32, using only what it stores.

    `emb` is None for the fp32 pass and the container's own dequantised operand
    for the int8 pass, so the two differ in the ARITHMETIC and in nothing else
    -- which is what makes the comparison between them a statement about int8.
    """
    d = int(r.config["hidden"])
    layers = int(r.config["num_layers"])
    heads = int(r.config["num_heads"])
    hd = d // heads
    eps = float(r.config["layer_norm_eps"])
    pos = r.tensor("frontend.position_embeddings")
    cls = r.tensor("frontend.cls_token")
    n_pos, _ = pos.shape
    # 1.0 when the file already carries the scale in Q, which is the default
    # pack. Read out of the config, not assumed: see `attention`.
    qscale = 1.0 if r.config["qkv_scale_folded"] else 1.0 / math.sqrt(hd)

    def operand(name, K, N):
        if emb is None:
            return r.tensor(name)
        q, wscale, asmooth = emb[name]
        return q.astype(np.float32) * wscale[None, :] / asmooth[:, None]

    pe = operand("frontend.patch_embed", patches.shape[1], d)
    # The patch GEMM produces 196 rows and the model has 197 positions: row 0 is
    # the CLS token, which is not a patch and is never multiplied by anything.
    # It is PREPENDED here, in fp32, exactly as the runtime prepends it, and
    # positions land afterwards -- pos[0] on the CLS row and pos[1:] on the
    # patches. Adding the two as if they were the same shape would silently
    # drop the last patch instead.
    h = np.empty((n_pos, d), dtype=np.float32)
    h[0] = cls + pos[0]
    h[1:] = patches @ pe + r.tensor("frontend.patch_embed.bias")[None, :]
    h[1:] += pos[1:]

    for i in range(layers):
        p = f"layer.{i}."
        ln1w, ln1b = r.tensor(p + "ln1.weight"), r.tensor(p + "ln1.bias")
        ln2w, ln2b = r.tensor(p + "ln2.weight"), r.tensor(p + "ln2.bias")
        # PRE-LN. The residual is added AFTER the sub-block and never normalised.
        a = layernorm(h, ln1w, ln1b, eps)
        qkv = a @ operand(p + "qkv", d, 3 * d) + r.tensor(p + "qkv.bias")[None, :]
        q, k, v = (qkv[:, j * d:(j + 1) * d] for j in range(3))
        s, ctx = attention(q, k, v, heads, hd, qscale)
        h = h + ctx @ operand(p + "attn_out", d, d) + r.tensor(
            p + "attn_out.bias")[None, :]
        a = layernorm(h, ln2w, ln2b, eps)
        u = gelu(a @ operand(p + "ffn_up", d, int(r.config["intermediate"]))
                 + r.tensor(p + "ffn_up.bias")[None, :])
        h = h + u @ operand(p + "ffn_down", int(r.config["intermediate"]), d) \
            + r.tensor(p + "ffn_down.bias")[None, :]

    h = layernorm(h, r.tensor("layernorm.weight"), r.tensor("layernorm.bias"),
                  eps)
    # Row 0 is the CLS token: the classifier reads it and nothing else.
    return h[0] @ r.tensor("classifier.weight") + r.tensor("classifier.bias")


def quantised_operands(r):
    """Every I8 panel, de-tiled and paired with its own scales.

    Read back through the container, not from the checkpoint: the question is
    whether the FILE carries what the scheme says it carries, so the file is
    what has to be examined.
    """
    from npue import ASMOOTH_SUFFIX, WSCALE_SUFFIX, untile_b
    out = {}
    for name, e in r.entries.items():
        if e.get("dtype") != "I8":
            continue
        lay = e["layout"]
        K, N = e["padded_shape"]
        q = untile_b(r.raw(name), K, N, lay["tile_k"], lay["tile_n"],
                     lay["mac_s"], lay["mac_t"])[:e["logical_shape"][0],
                                                   :e["logical_shape"][1]]
        out[name] = (q, r.tensor(name + WSCALE_SUFFIX),
                     r.tensor(name + ASMOOTH_SUFFIX))
    return out


def quantise_activation(x, asmooth):
    """`quantise_a_int8` from host_kernels.hpp, in numpy.

    Divide by asmooth FIRST (that is what asmooth is for), then a per-ROW
    absmax, then rint and a clip. The direction is load-bearing: the container
    stores W*asmooth, so an activation that is not pre-divided produces
    X @ W * asmooth^2, which is the bug npue.dequant_int8's docstring records
    as 1-cos 1.045.
    """
    v = x / asmooth[None, :]
    sa = np.abs(v).max(axis=1) / 127.0
    sa = np.where(sa > 0, sa, 1.0).astype(np.float32)
    aq = np.rint(v / sa[:, None]).clip(-127, 127).astype(np.int32)
    return aq, sa


def forward_int8(emb, patches, r, cfg):
    """The same pass with A quantised per row and C dequantised per output.

    Deliberately a SECOND implementation of the container's arithmetic rather
    than a flag on the first: the first is the container's exact fp32 weights,
    and the int8 path's own steps -- pre-divide, per-row scale, int32
    accumulate, rank-1 rescale -- are what the runtime's kernels do and what
    this has to reproduce. Folding them into one parameterised function would
    make it possible for a bug in the fp32 path to hide an int8 bug.
    """
    d = int(r.config["hidden"])
    layers = int(r.config["num_layers"])
    heads = int(r.config["num_heads"])
    hd = d // heads
    eps = float(r.config["layer_norm_eps"])
    inter = int(r.config["intermediate"])
    pos = r.tensor("frontend.position_embeddings")
    cls = r.tensor("frontend.cls_token")
    qscale = 1.0 if r.config["qkv_scale_folded"] else 1.0 / math.sqrt(hd)

    def gemm8(a, name, N):
        """(M,K) x int8 [K,N] -> (M,N) fp32, through the runtime's own steps."""
        q, wscale, asmooth = emb[name]
        aq, sa = quantise_activation(a, asmooth)
        acc = aq.astype(np.int32) @ q.astype(np.int32)     # no rounding anywhere
        bias = r.tensor(name + ".bias")
        return (acc.astype(np.float32) * sa[:, None]
                * wscale[None, :]) + bias[None, :]

    def gemm32(a, w, bias):
        return a @ w + bias[None, :]

    h = np.empty((pos.shape[0], d), dtype=np.float32)
    h[0] = cls + pos[0]
    h[1:] = gemm8(patches, "frontend.patch_embed", d) + pos[1:]
    for i in range(layers):
        p = f"layer.{i}."
        a = layernorm(h, r.tensor(p + "ln1.weight"), r.tensor(p + "ln1.bias"), eps)
        qkv = gemm8(a, p + "qkv", 3 * d)
        q, k, v = (qkv[:, j * d:(j + 1) * d] for j in range(3))
        _, ctx = attention(q, k, v, heads, hd, qscale)
        h = h + gemm8(ctx, p + "attn_out", d)
        a = layernorm(h, r.tensor(p + "ln2.weight"), r.tensor(p + "ln2.bias"), eps)
        u = gelu(gemm8(a, p + "ffn_up", inter))
        h = h + gemm8(u, p + "ffn_down", d)
    h = layernorm(h, r.tensor("layernorm.weight"), r.tensor("layernorm.bias"),
                  eps)
    return gemm32(h[0], r.tensor("classifier.weight"),
                  r.tensor("classifier.bias"))


# -- the reference ----------------------------------------------------------

def reference(model_dir, pixels):
    """transformers' own answer, in fp32, for the same normalised pixels.

    The pixels are passed straight in, not re-derived from a processor, so a
    disagreement is about the CONTAINER and never about a resize filter.
    """
    import torch
    from transformers import ViTForImageClassification
    m = ViTForImageClassification.from_pretrained(
        str(model_dir), torch_dtype=torch.float32).eval()
    with torch.no_grad():
        out = m(pixel_values=torch.from_numpy(pixels)).logits
    return out.numpy().astype(np.float32)


def labels_of(r):
    raw = bytes(r.raw("labels.table")).decode("utf-8")
    return raw.rstrip("\n").split("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--model-dir", default=None,
                    help="the checkpoint the container was packed from "
                         "(default: read source_repo off the container)")
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    r = Reader(args.container)
    cfg = r.config
    arch = cfg["arch"]
    report(arch == "vit_patch16_prenorm_gelu",
           "container arch is vit_patch16_prenorm_gelu", arch)
    model_dir = args.model_dir or (REPO / "models"
                                   / Path(cfg["source_repo"]).name)
    print(f"\ncontainer {args.container}")
    print(f"  a_dtype   {cfg['a_dtype']}   norm_order {cfg['norm_order']}"
          f"   classifier {cfg['classifier']}")

    d = int(cfg["hidden"])
    image = int(cfg["image_size"])
    patch = int(cfg["patch_size"])
    n_patch = int(cfg["n_patches"])
    n_pos = int(cfg["max_seq_len"])
    # Is this the int8 container? Ask the FILE, not config["a_dtype"]: that key
    # is a request the packer records, while the dtypes on the entries are what
    # the bytes are, and comparing a config string against a spelling guessed
    # here is how an int8 container gets silently treated as fp32. (It did: the
    # first run of this gate compared "I8" with "i8", found no match, skipped
    # the entire int8 section, and reported "container is bf16".)
    is_i8 = any(e.get("dtype") == "I8" for e in r.entries.values())
    report(str(cfg["a_dtype"]).lower() == ("i8" if is_i8 else "bf16"),
           "config a_dtype agrees with the dtypes on the entries",
           f"{cfg['a_dtype']} vs {'I8' if is_i8 else 'BF16'}")
    report(n_pos == n_patch + 1,
           "position count is n_patches + 1 (the CLS token)", f"{n_pos}")
    report(int(cfg["num_heads"]) * int(cfg["head_dim"]) == d,
           "heads x head_dim == hidden")

    # -- the im2col layout, held against HF's own Conv2d -------------------
    #
    # The one invariant whose failure is invisible downstream, so it is
    # checked directly rather than through a forward pass.
    print("\n1. the patch-embedding layout")
    import torch
    from transformers import ViTModel
    hf = ViTModel.from_pretrained(str(model_dir),
                                  torch_dtype=torch.float32).eval()
    with torch.no_grad():
        px = torch.zeros(1, 3, image, image)
        px[0, 0] = torch.arange(image, dtype=torch.float32).view(1, image) / image
        px[0, 1] = 1.0 - torch.arange(image, dtype=torch.float32).view(1, image) / image
        conv = hf.embeddings.patch_embeddings.projection
        # [1, oc, 14, 14] -> [196, oc] as (patch_row, patch_col), which is the
        # order im2col_patches enumerates. NOT flatten(1): that is oc-major, so
        # it returns a 196x768 matrix whose rows are output CHANNELS -- which
        # still broadcasts, still has the right shape, and still gives a
        # plausible-looking 1-cos of 1.0 when the two are compared.
        want = conv(px)[0].permute(1, 2, 0).reshape(n_patch, d).numpy()
    got = im2col_patches(px[0].numpy(), image, patch) @ r.tensor(
        "frontend.patch_embed") + r.tensor("frontend.patch_embed.bias")[None, :]
    report(got.shape == (n_patch, d), "patch matrix is [n_patches, hidden]",
           str(got.shape))
    # The tolerance follows the container's own operand dtype, and says so. An
    # int8 panel cannot be held to a bf16 tolerance -- the 2e-3 below IS the
    # quantisation, and demanding more of it would be demanding a bf16 file.
    rel = float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-9))
    rel_tol = RTOL_INT8 if is_i8 else RTOL_FP32
    report(rel <= rel_tol,
           f"the container's patch_embed reproduces HF's Conv2d ({'int8' if is_i8 else 'fp32'} tolerance)",
           f"1-cos {1.0 - cos(got, want):.3e}, max rel {rel:.2e} against "
           f"{rel_tol:.0e}")

    # also: the packer's own layout helper agrees with the runtime's reader,
    # BYTE for BYTE. Compared as uint16 bit patterns, not as floats: a bf16
    # container's rounding error is smaller than the disagreement a wrong row of
    # the tile would produce, so comparing values would pass anyway.
    #
    # This check is bf16-ONLY and says why it does not run on an int8 container:
    # the panel is not the weight, and the weight is not `to_bf16_bits(W)` -- it
    # is W*asmooth quantised. Asserting the bit pattern here would be asserting
    # that the int8 path is the bf16 path. The int8 container's equivalent
    # statements are the tolerance above and the four scheme invariants in
    # section 3.
    from npue import to_bf16_bits, untile_b
    from safetensors_mmap import SafeTensors
    st = SafeTensors(Path(model_dir) / "model.safetensors")
    raw = np.array(patch_embed_operand(
        st.array("vit.embeddings.patch_embeddings.projection.weight")))
    st.close()
    if is_i8:
        print("   --    bit-for-bit de-tile: bf16-only, skipped on an int8 "
              "container (see the comment above)")
    else:
        e = r.entries["frontend.patch_embed"]
        bits = untile_b(r.raw("frontend.patch_embed"), *e["padded_shape"],
                        e["layout"]["tile_k"], e["layout"]["tile_n"],
                        e["layout"]["mac_s"], e["layout"]["mac_t"])
        report(np.array_equal(bits, to_bf16_bits(raw)),
               "Reader de-tiles frontend.patch_embed back to the checkpoint's "
               "matrix, bit for bit")

    # -- the container's own weights against transformers --------------------
    #
    # On a bf16 container this is the fp32 gate. On an int8 container the SAME
    # code runs the fp32 arithmetic over the container's DEQUANTISED weights
    # (Reader.tensor does that for an I8 entry), so it is not the fp32 claim
    # and it is not the int8 claim either -- the int8 claim is section 3, which
    # also quantises the activations. It is reported with the tolerance its
    # operand dtype can actually meet, and named for what it is.
    print(f"\n2. the container's own weights against transformers, "
          f"{args.n_images} images")
    pre = read_preprocessor(model_dir)
    images = synth_images(args.n_images, pre["image_size"], args.seed)
    pixels = preprocess(images, pre["image_size"], pre["mean"],
                        pre["std"], pre["resample"])
    want = reference(model_dir, pixels)

    labels = labels_of(r)
    report(len(labels) == int(cfg["num_labels"]),
           "the label table has one name per id2label entry",
           f"{len(labels)}")

    emb = quantised_operands(r) if is_i8 else None
    cos_tol = COS_INT8 if is_i8 else COS_FP32
    worst_cos, worst_top1, logits32, logits8 = 1.0, 1.0, [], []
    for i in range(len(images)):
        patches = im2col_patches(pixels[i], image, patch)
        got = forward(None, patches, r, cfg)
        c = cos(got, want[i])
        top1 = int(np.argmax(got)) == int(np.argmax(want[i]))
        worst_cos = min(worst_cos, c)
        worst_top1 = min(worst_top1, float(top1))
        logits32.append(got)
        line = (f"   image {i}: 1-cos {1.0 - c:.3e}  "
                f"pred {labels[int(np.argmax(got))]!r} vs "
                f"{labels[int(np.argmax(want[i]))]!r}"
                f"{'  OK' if top1 else '  MISMATCH'}")
        if emb is not None:
            g8 = forward_int8(emb, patches, r, cfg)
            logits8.append(g8)
            line += (f" | int8 1-cos {1.0 - cos(g8, want[i]):.3e}  "
                     f"top1 {'same' if int(np.argmax(g8)) == int(np.argmax(got)) else 'DIFFERS'}")
        print(line)

    report(worst_cos >= cos_tol,
           f"container vs transformers, every image ({'int8 weights' if is_i8 else 'fp32'})",
           f"worst 1-cos {1.0 - worst_cos:.3e} against {1.0 - cos_tol:.0e}")
    report(worst_top1 == 1.0, "top-1 agrees on every image")

    # -- int8 ---------------------------------------------------------------
    if emb is None:
        print("\n3. int8: container is bf16, nothing to check")
    else:
        print(f"\n3. int8, {len(emb)} operands, through the runtime's own steps")
        c8 = min(cos(logits8[i], want[i]) for i in range(len(images)))
        agree = sum(int(np.argmax(logits8[i])) == int(np.argmax(logits32[i]))
                    for i in range(len(images))) / len(images)
        report(c8 >= COS_INT8, "int8 logits vs transformers (MODEL tolerance)",
               f"worst 1-cos {1.0 - c8:.3e} against {1.0 - COS_INT8:.0e}")
        report(agree >= TOP1_INT8, "int8 top-1 agrees with the container's own "
                                  "fp32 arithmetic",
               f"{agree:.0%} against {TOP1_INT8:.0%}")
        # The operand-level check, which is a DIFFERENT statement and must not
        # be compared with the model tolerance above.
        #
        # w_src is the checkpoint's matrix, built with the PACKER'S OWN builders
        # -- not a second transcription of the mapping. That is the whole reason
        # qkv_matrix/linear_matrix/patch_embed_operand are named functions: two
        # hand-written copies of "which way round is a linear stored, and is the
        # attention scale in the Q block" would agree right up until one of them
        # was edited, and the disagreement would then look like a broken
        # container rather than a broken test. It is NOT `r.tensor(name)`:
        # for an I8 entry that is already the DEQUANTISED weight, so feeding it
        # back in would check the file against itself.
        from gemm_i8 import check_i8_operand
        from packers.vit import linear_matrix, qkv_matrix
        from safetensors_mmap import SafeTensors
        st = SafeTensors(Path(model_dir) / "model.safetensors")
        folded = bool(cfg["qkv_scale_folded"])
        qscale = float(cfg["attention_scale"])
        src = {"frontend.patch_embed":
               patch_embed_operand(np.array(
                   st.array(
                       "vit.embeddings.patch_embeddings.projection.weight")))}
        for i in range(int(cfg["num_layers"])):
            p = f"vit.encoder.layer.{i}."
            e = f"layer.{i}."
            src[e + "qkv"] = qkv_matrix(st, p + "attention.attention.",
                                       qscale, folded)
            src[e + "attn_out"] = linear_matrix(
                st, p + "attention.output.dense.weight")
            src[e + "ffn_up"] = linear_matrix(st, p + "intermediate.dense.weight")
            src[e + "ffn_down"] = linear_matrix(st, p + "output.dense.weight")
        st.close()

        missing = sorted(set(emb) - set(src))
        problems = 0
        for name, (q, wscale, asmooth) in sorted(emb.items()):
            if name not in src:
                print(f"   FAIL  {name}: no source matrix to check it against")
                problems += 1
                continue
            probs, _ = check_i8_operand(name, q, wscale, asmooth, src[name])
            if probs:
                print(f"   FAIL  {probs[0]}")
                for extra in probs[1:3]:
                    print(f"         {extra}")
            problems += len(probs)
        if missing:
            print(f"   FAIL  {len(missing)} I8 operands are not in the mapping "
                  f"above: {missing[:3]}. Every int8 operand needs a source.")
            problems += len(missing)
        report(problems == 0,
               "every I8 panel satisfies the four scheme invariants",
               f"{len(emb)} operands" if problems == 0
               else f"{problems} problems")

    r.close()
    print()
    if _failures:
        print(f"FAIL: {len(_failures)} of the checks above")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())