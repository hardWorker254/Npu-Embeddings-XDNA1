# NpuEmbeddings -- pack a google/vit-* image classifier into an arch=5 .npue.
# SPDX-License-Identifier: Apache-2.0
#
# THE POINT OF THIS FILE. Two of the four families in this tree are
# "vector for text" and one is "text for audio". This one answers a THIRD
# question -- "which of a thousand names is this picture" -- and the claim it
# exists to test is that the modality is new and the MACHINE is not. So the
# container below carries no new machinery: every GEMM it holds is one the BERT
# family already had, at the shapes BERT already uses.
#
#   image -> resize/normalise -> im2col into 197 patches of 768 floats
#         -> + cls token, + learned positions
#         -> 12 pre-LN encoder layers: qkv, attn_out, GELU FFN
#         -> final LayerNorm -> take row 0
#         -> classifier -> argmax over 1000 logits
#
# WHAT THE ARRAY RUNS, AND WHAT IT DOES NOT -- stated here because the split is
# the design, not an omission:
#
#   ON THE ARRAY, on the FOUR streams BERT already has:
#     patch_embed   [M,768] x [768,768]  -- this IS the attn_out stream's shape
#     qkv           [M,768] x [768,2304]
#     attn_out      [M,768] x [768,768]
#     ffn_up        [M,768] x [768,3072]
#     ffn_down      [M,3072] x [3072,768]
#
#   ON THE HOST, and each for a measured reason rather than taste:
#     the classifier head  1000 is not divisible by tile_n*cols, so no single
#                         B panel of that width exists on this array; it is one
#                         768x1000 fp32 matvec per image and the head is 0.8%
#                         of the model's arithmetic, so a host GEMM costs less
#                         than the ~150 us a dispatch to ask the array for it.
#     the image front end  resize/normalize/im2col, which is pixels not GEMMs.
#     attention itself     O(seq^2) on the host, exactly as in the BERT path --
#                         the file has no measurement above seq 64 and 197 is
#                         three times that, so this is stated as untested
#                         rather than assumed fast.
#
# THE ONE REAL ARCHITECTURAL DIFFERENCE FROM BERT: PRE-LN.
# BERT here is post-LN (residual then normalise); a ViT layer normalises BEFORE
# each sub-block and adds after. This is why this is arch=5 and not a reuse of
# arch=0's tensor names: a container with BERT's names and ViT's order would be
# read happily by BertEncoder and compute a different model, which is the exact
# failure the arch whitelist in app_state.hpp exists to refuse. The names below
# are ViT's own, so a name collision is impossible and the arch check is the
# only gate that stands between them.
#
# FUSIONS, all exact:
#   1. Q|K|V into one [768, 2304] operand per layer, as in every other arch.
#      All three have a bias here, so the fused bias is the concatenation and
#      nothing is zeroed.
#   2. 1/sqrt(head_dim) folded into Q's weight and Q's bias. A GEMM is linear,
#      so folding before the GEMM and folding after it agree; what folding into
#      the WHOLE fused operand would do is scale K and V too, which is a
#      different model.
#   3. The patch-embedding conv is rewritten as a GEMM, which is exact and not
#      an approximation: Conv2d(3, 768, kernel=16, stride=16, padding=0) over a
#      224x224 image is 196 non-overlapping 16x16 patches, and im2col of a
#      stride-equals-kernel convolution IS the identity -- each output row is
#      one patch, flattened in the same order the weight expects. See
#      `patch_embed_operand` for the layout, which is the one thing here that a
#      reader has to get right.
#   4. GEMM operands pre-tiled to bf16 block_panel at (64, 48). Both tile sizes
#      are the BERT defaults, and they are forced rather than chosen: every K
#      is 768 or 3072 (both multiples of 64) and every N is 768, 2304 or 3072
#      (all multiples of 48*4 = 192), so no operand needs padding.
#
# THE CLASSIFIER IS NOT PRE-TILED, deliberately. It is stored as a plain F32
# [768, 1000] row-major operand under the role `gemm_b_host`, which already
# exists for the CPU-only Gemma path. Pre-tiling it would claim a block_panel
# layout for a matrix the array never reads, and `b_layout_hash` is the
# mechanism this project uses to refuse a container whose operand order and the
# design's disagree -- an unnecessary hash on a host operand is one more thing
# that can be wrong.
#
# Env: numpy only (plus the repo's own npue.py, gemm_i8.py and
# onnx_weights.py); torch and transformers only under --int8.

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

from gemm_i8 import add_gemm_b_int8                              # noqa: E402
from gemm_i4 import add_gemm_b_int4                              # noqa: E402
from npue import (ARCH_VIT_PATCH16_PRELN, MAC_BY_DEVICE,        # noqa: E402
                  MAC_DEFAULT_DEVICE, Writer, gemm_b_layout, layout_hash,
                  mac_for_device, tile_b, to_bf16_bits)
from onnx_weights import MODEL_ONNX, OnnxWeights, model_digest  # noqa: E402

# Only s and t of the MAC geometry affect the B operand's byte order, and the
# board decides them: npu1's MMAC sub-tile is (s=8, t=4) and npu2's is (8, 8).
# `mac` is resolved from --device and threaded into every operand rather than
# fixed here, because a container packed for the other board is not refused by
# anything -- it is merely wrong. `mac_for_device` refuses an unknown board.
MAC_DEFAULT = MAC_BY_DEVICE["npu2"]["bf16"]

# BERT's tile sizes, which are also this family's. See the header: both are
# forced by the geometry, and changing either means repacking every container.
TILE_K, TILE_N = 64, 48

# The B operand's dtype, and with it the layout hash the runtime compares
# against the design's. Switched by --int8 and set once, here, so the writer and
# the printed hash cannot disagree: they read the same constant, and a
# disagreement would be a container whose layout_hash describes the other
# datatype, which nothing except the hardware would notice.
I8_DTYPE = "BF16"


def _read_json(path, what):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{what}: {p} not found. A ViT checkpoint needs "
                         f"config.json, preprocessor_config.json and "
                         f"{MODEL_ONNX}.")
    return json.loads(p.read_text(encoding="utf-8"))


def add_gemm_b(w, name, mat, fold=None, mac=MAC_DEFAULT):
    """Stage a [K, N] GEMM operand: optional scale fold, bf16, pre-tiled.

    The fold is applied in fp32 BEFORE the bf16 rounding, never after: rounding
    first and scaling second would scale the rounding error too, and the error
    is already the dominant term in a bf16 operand.
    """
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if fold is not None:
        mat = mat * np.float32(fold)
    K, N = mat.shape
    if K % TILE_K or N % TILE_N:
        raise SystemExit(
            f"{name}: [{K},{N}] does not tile into ({TILE_K},{TILE_N}). A ViT "
            f"whose hidden size is not a multiple of {TILE_K}, or whose width "
            f"is not a multiple of {TILE_N}, needs this packer's tile sizes "
            f"changed deliberately -- an operand padded to fit would be a "
            f"container whose layout_hash lies about its shape.")
    layout = gemm_b_layout(TILE_K, TILE_N, mac[0], mac[1], dtype=I8_DTYPE)
    bits = to_bf16_bits(mat)
    flat = tile_b(bits, TILE_K, TILE_N, mac[0], mac[1])
    return w.add(name, flat, "BF16", "gemm_b", [K, N], layout=layout)


class _Int8:
    """int8 emission for one container, driven by calibrated activation maxima.

    The scheme is `gemm_i8.add_gemm_b_int8` (per-output-channel symmetric
    scale, int8 x int8 -> int32) and the smoothing factor is SmoothQuant's,
    computed from the calibration's per-input-channel maximum and this
    operand's own rows:

        s_j = max_i |X[i,j]|^alpha / max_n |W[j,n]|^(1-alpha)

    so the runtime computes (X/s) @ (sW), which is X @ W in exact arithmetic.
    The ratio lives in `vit_int8.factors` so a calibration and its gate cannot
    drift apart by an eps.

    The per-operand relative error is collected and printed, because a scale set
    that is wrong for the model is visible here and nowhere else.
    """

    def __init__(self, amax, alpha, mac, layout_dtype, int4_group=None):
        self.amax = amax
        self.alpha = alpha
        self.mac = mac
        self.layout_dtype = layout_dtype
        # None for every int8 pack; an int number of rows per K-group for an
        # int4 pack. It rides on the emitter rather than a module global for
        # the same reason I8_DTYPE does not: one container, one scheme.
        self.int4_group = int4_group
        self.errors: list[tuple[str, float]] = []

    def smoothing(self, key, mat):
        import vit_int8
        a = self.amax[key]
        if a.shape[0] != mat.shape[0]:
            raise SystemExit(
                f"{key}: the calibration measured {a.shape[0]} input channels "
                f"and the operand has {mat.shape[0]}. The hook and the operand "
                f"must be the same tensor; a mismatch here is a per-channel "
                f"scale broadcast onto the wrong axis, which is silent.")
        w = np.abs(np.asarray(mat, dtype=np.float32)).max(axis=1)
        return vit_int8.factors(a, w, self.alpha)

    def emit(self, w, name, mat, key, fold=None):
        if key is None:
            add_gemm_b(w, name, mat, fold=fold, mac=self.mac)
            return
        s = self.smoothing(key, mat)
        # The SAME smoothing either way: int4 is the int8 datapath with a
        # narrower weight, so the activation side of the SmoothQuant identity
        # does not change and the calibration is shared with int8 (gemm_i4.py).
        if self.int4_group is None:
            err = add_gemm_b_int8(w, name, mat, TILE_K, TILE_N, fold=fold,
                                  asmooth=s, mac=self.mac)
        else:
            err = add_gemm_b_int4(w, name, mat, TILE_K, TILE_N, fold=fold,
                                  asmooth=s, group=self.int4_group,
                                  mac=self.mac)
        self.errors.append((name, err))

    def report(self) -> str:
        if not self.errors:
            return "  (no int8 operands)"
        es = [e for _, e in self.errors]
        worst = max(self.errors, key=lambda kv: kv[1])
        return (f"  {'int4' if self.int4_group is not None else 'int8'}: "
                f"{len(self.errors)} operands, quantisation error "
                f"median {float(np.median(es)):.2e} max {worst[1]:.2e} "
                f"({worst[0]})")


def patch_embed_operand(w_proj: np.ndarray) -> np.ndarray:
    """Conv2d(3, H, kernel=P, stride=P) weight -> the [P*C*P, H] GEMM operand.

    THE LAYOUT IS THE WHOLE OF THIS FUNCTION, so it is worth being explicit.

    torch's Conv2d weight is [out_c, in_c, kh, kw] and computes

        out[n, oc, p, q] = sum_{ic, i, j} W[oc, ic, i, j] * x[n, ic, p*P+i, q*P+j]

    A GEMM computes Y[n, m] = sum_k A[n, k] * B[k, m], so we need
    B[k, m] = W[m, k'] with k = (ic, i, j) -- that is, B is W with its first
    axis moved last and the remaining three flattened in C order:

        B[(ic*P + i)*P + j, oc] = W[oc, ic, i, j]

    which is `transpose(1, 2, 3, 0).reshape(K, N)`. Two facts make this exact
    rather than merely plausible:

      * stride == kernel and padding == 0, so the 196 windows do not overlap
        and do not skip: im2col is a permutation, not a sum.
      * patch_size is a multiple of TILE_K's factor chain -- P*P*C = 768 here --
        so no operand needs padding, and the padding rules never come into play.

    A wrong ordering here does not fail: it produces a plausible-looking image
    classification of noise, because the GEMM is perfectly well formed and every
    downstream operation is the one the model does. tools/verify/verify_vit.py
    holds it against HF's own Conv2d path, which is the only thing that can.
    """
    w = np.asarray(w_proj, dtype=np.float32)
    if w.ndim != 4:
        raise SystemExit(
            f"patch-embedding weight is {w.shape}, expected 4 "
            f"[out_c, in_c, kh, kw]. A ViT whose patch projection is not a "
            f"convolution is a different architecture; this packer does not "
            f"guess how it folds into a GEMM.")
    oc, ic, kh, kw = w.shape
    if ic != 3:
        raise SystemExit(
            f"patch-embedding weight has {ic} input channels, expected 3 "
            f"(RGB). A checkpoint on a different channel count needs its own "
            f"front end, not this packer.")
    return np.ascontiguousarray(
        w.transpose(1, 2, 3, 0).reshape(ic * kh * kw, oc), dtype=np.float32)


def linear_matrix(st, key) -> np.ndarray:
    """A torch nn.Linear weight [N, K] as this container's [K, N] operand.

    THE SECOND HALF OF THE LAYOUT STORY, and the reason this exists as a named
    function rather than a `.T` at each of the five call sites: `check_i8_operand`
    in the gate needs the pre-smoothing, post-fold SOURCE matrix of every
    operand, and that matrix is this function's output. Writing the transpose at
    the call sites would give the packer and its gate two definitions of which
    way round a linear is stored, and they would agree until one of them changed.
    """
    return np.ascontiguousarray(st.array(key).T, dtype=np.float32)


def qkv_matrix(st, prefix, scale, fold_scale=True) -> np.ndarray:
    """[d, 3d] fused Q|K|V, with the attention scale in Q's columns ONLY.

    The scale goes into Q's weight and Q's bias and nowhere else: folding it
    into the whole fused operand would scale K and V too, which is a different
    model. A GEMM is linear, so folding before it and folding after it agree --
    what does not agree is folding into Q, K and V together.
    """
    q = linear_matrix(st, prefix + "query.weight")
    if fold_scale:
        q = q * np.float32(scale)
    return np.concatenate([q,
                           linear_matrix(st, prefix + "key.weight"),
                           linear_matrix(st, prefix + "value.weight")],
                          axis=1)                              # [d, 3d]


def _fuse_qkv(w, name, st, prefix, hidden, scale, fold_scale=True,
              mac=MAC_DEFAULT, ctx=None, key=None):
    """Fuse q|k|v into one [d, 3d] operand, scaling into the Q block only.

    Every ViT qkv projection HAS a bias (qkv_bias: true), so the fused bias is
    the plain concatenation -- no zero half, and no refusal to emit one, which
    is the whisper case. The scale goes into Q's columns and Q's bias and
    nowhere else: folding it into the whole operand would scale K and V too,
    which is a different model.
    """
    for suffix in ("query", "key", "value"):
        if prefix + suffix + ".bias" not in st:
            raise SystemExit(
                f"{prefix}{suffix}.bias is missing. This packer writes the "
                f"fused bias as the plain concatenation of the three, which is "
                f"exact only because all three exist; a checkpoint that drops "
                f"one must have that drop handled explicitly rather than "
                f"silently ignored.")
    qb = st.array(prefix + "query.bias").astype(np.float32, copy=True)
    if fold_scale:
        qb *= np.float32(scale)
    fused = qkv_matrix(st, prefix, scale, fold_scale)
    bias = np.concatenate([qb,
                           st.array(prefix + "key.bias").astype(np.float32),
                           st.array(prefix + "value.bias").astype(np.float32)])
    if ctx:
        ctx.emit(w, name, fused, key, fold=None)
    else:
        add_gemm_b(w, name, fused, mac=mac)
    w.add(name + ".bias", bias, "F32", "bias", [3 * hidden])


def _out_proj(w, name, st, prefix, mac=MAC_DEFAULT, ctx=None, key=None):
    ob = st.array(prefix + "dense.bias")
    mat = linear_matrix(st, prefix + "dense.weight")
    if ctx:
        ctx.emit(w, name, mat, key)
    else:
        add_gemm_b(w, name, mat, mac=mac)
    w.add(name + ".bias", ob.astype(np.float32), "F32", "bias",
          [int(ob.shape[0])])


def _ln(w, name, st, wkey, bkey, hidden):
    w.add(name + ".weight", st.array(wkey), "F32", "layernorm", [hidden])
    w.add(name + ".bias", st.array(bkey), "F32", "layernorm", [hidden])


def pack_vit(model_dir, out, fold_scale=True, dry_run=False, device=None,
             int8=False, int4_group=None, int8_alpha=0.5, int8_images=8,
             int8_corpus=None):
    global I8_DTYPE
    if int4_group is not None and not int8:
        # int4 is the int8 datapath with 4-bit weights (gemm_i4.py), so with
        # int8=False there is no quantisation to attach a group size to.
        # Refused rather than dropped: a bf16 container that ignored the flag
        # would exit 0, and the caller would have no way to learn it did
        # nothing -- the accepted-and-ignored shape this packer refuses by
        # name (--max-seq on a ViT, pack_npue.py).
        raise SystemExit(
            f"--int4-group {int4_group} was passed to a pack that does not "
            f"quantise (int8=False); int4 is the int8 datapath with 4-bit "
            f"weights, so it has nothing to attach to. Refusing rather than "
            f"emitting a bf16 container that ignores it.")
    model_dir = Path(model_dir)
    # int8's sub-tile is NOT bf16's on npu1 -- see npue.MAC_BY_DEVICE. The
    # dtype is required by mac_for_device for that reason.
    mac = mac_for_device(device, "I8" if int8 else "BF16")
    cfg = _read_json(model_dir / "config.json", "vit config")
    mt = cfg.get("model_type")
    if mt != "vit":
        raise SystemExit(
            f"{model_dir}/config.json says model_type={mt!r}, not 'vit'")

    d = int(cfg["hidden_size"])
    layers = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    head_dim = d // heads
    if heads * head_dim != d:
        raise SystemExit(
            f"hidden_size {d} is not divisible by {heads} heads. HF uses "
            f"head_dim = hidden_size // num_attention_heads with integer "
            f"division, so a non-dividing pair is a real truncated head and "
            f"the attention scale would be wrong.")
    inter = int(cfg["intermediate_size"])
    patch = int(cfg["patch_size"])
    chan = int(cfg.get("num_channels", 3))
    image = int(cfg.get("image_size", 224))
    n_patch = (image // patch) ** 2
    n_pos = n_patch + 1                      # + the CLS token
    n_labels = len(cfg.get("id2label", {}))
    if n_labels == 0:
        raise SystemExit(
            f"{model_dir}/config.json carries no id2label. The container "
            f"stores the label vocabulary so the runtime can print a NAME and "
            f"not an index, and a classifier that can only answer 437 does not "
            f"answer the question it was asked.")
    if n_pos > 4096:
        raise SystemExit(
            f"{n_pos} positions ({image}px at patch {patch}). The designs "
            f"tile M into 256-row blocks and one dispatch is capped at four of "
            f"them by the shim's DMA descriptors, so above ~1024 positions a "
            f"single image needs more than one dispatch of a different "
            f"geometry. That is a design decision, not a packer default.")
    act = str(cfg.get("hidden_act", "gelu"))
    if act != "gelu":
        raise SystemExit(
            f"hidden_act is {act!r}, expected 'gelu'. The runtime's host GELU "
            f"and its bf16 gelu design are the exact-erf one; a checkpoint "
            f"with another activation needs a different kernel, and packing it "
            f"under this arch would answer with a different model.")
    eps = float(cfg.get("layer_norm_eps", 1e-12))
    scale = 1.0 / math.sqrt(head_dim)
    if not cfg.get("qkv_bias", True):
        raise SystemExit(
            "qkv_bias is false. This packer fuses q|k|v with the plain "
            "concatenation of three biases, which is exact only because all "
            "three exist.")

    st = OnnxWeights(model_dir / MODEL_ONNX)
    src_sha = model_digest(model_dir / MODEL_ONNX)

    # The checkpoint's OWN image front end, read for EVERY pack and not only for
    # an int8 one. Its numbers are part of the model's definition -- a container
    # normalised with another set's mean/std is a model that classifies noise --
    # so they belong in the container's config, where a consumer that is not this
    # Python can read them instead of guessing or hardcoding 0.5.
    #
    # This is the same refusal packers/whisper.py makes about
    # preprocessor_config.json, for the same reason, and it is made ONCE here
    # rather than inside the int8 branch: an int8 pack that calibrated on the
    # right numbers and stored none of them would be a container whose
    # calibration is unreproducible from the container alone.
    import vit_int8
    pre = vit_int8.read_preprocessor(model_dir)
    # The TWO geometries, refused apart. A square resize and a shortest-edge
    # resize plus a centre crop produce different pixel tensors out of the same
    # photograph, so a container cannot claim both -- and a consumer that only
    # implements one must be able to read which one this is.
    if pre["resize"] != "square":
        raise SystemExit(
            f"preprocessor_config.json asks for a '{pre['resize']}' resize "
            f"(shortest edge {pre['image_size']}, centre crop "
            f"{pre['crop_size']}). This runtime's image front end implements the "
            f"square form, and shipping a container that names the other one "
            f"would mean a consumer that could only ever get it wrong. Repack "
            f"from a checkpoint whose preprocessor_config.json has a square "
            f"size, or extend the packer and the runtime together.")
    if pre["crop_size"] != image:
        raise SystemExit(
            f"preprocessor_config.json resizes to {pre['image_size']} and "
            f"centre-crops {pre['crop_size']}, and config.json says image_size "
            f"{image}. The position table has {n_pos} rows, which is what a "
            f"{image}px image at patch {patch} produces, so these cannot both "
            f"be right. Refusing rather than picking one.")
    if len(pre["mean"]) != chan or len(pre["std"]) != chan:
        raise SystemExit(
            f"preprocessor_config.json gives {len(pre['mean'])} means and "
            f"{len(pre['std'])} stds for a {chan}-channel image. Refusing "
            f"rather than broadcasting a length that does not divide.")
    if any(s == 0.0 for s in pre["std"]):
        raise SystemExit(
            f"preprocessor_config.json image_std is {pre['std']}; a zero "
            f"standard deviation divides by zero and would make the normalise "
            f"step's result a NaN on every pixel of that channel.")

    # The int8 calibration, before any operand is written: it needs torch and
    # transformers, which are build-time only. `I8_DTYPE` is switched here
    # rather than inside add_gemm_b so that the layout hash, the writer and the
    # printed report all read one value.
    ictx = None
    if int8:
        I8_DTYPE = "I8"
        images, label = vit_int8.load_or_make_corpus(int8_corpus, int8_images,
                                                      pre["image_size"])
        pixel_values = vit_int8.preprocess(images, pre["image_size"],
                                           pre["mean"], pre["std"],
                                           pre["resample"])
        amax = vit_int8.calibrate(model_dir, layers, images, pixel_values,
                                  alpha=int8_alpha, verbose=False)
        ictx = _Int8(amax, int8_alpha, mac, I8_DTYPE, int4_group)
        print(f"  {'int4' if int4_group is not None else 'int8'}: {label}, "
              f"alpha={int8_alpha}, "
              f"{len(amax)} GEMM sites ({layers} layers + patch_embed)")

    config = {
        "arch": "vit_patch16_prenorm_gelu",
        "kind": "cls",
        "source_repo": f"google/{model_dir.name}",
        "source_sha256": src_sha,
        "num_layers": layers,
        "hidden": d,
        "num_heads": heads,
        "head_dim": head_dim,
        "intermediate": inter,
        "layer_norm_eps": eps,
        "activation": "gelu",
        # The four numbers the design selector and the runtime read. A ViT has
        # no position table of its own to slice, so max_seq_len is the TRUE
        # position count and not a cap the caller chose -- which is what stops a
        # --max-seq flag from being silently meaningless here.
        "max_seq_len": n_pos,
        "num_labels": n_labels,
        "image_size": image,
        "patch_size": patch,
        "num_channels": chan,
        "patch_dim": chan * patch * patch,
        "n_patches": n_patch,
        # -- the image front end, AS DATA. Read from the checkpoint's own
        # preprocessor_config.json above and refused rather than defaulted, so a
        # consumer that is not this Python -- the C++ runtime's PNG/JPEG reader,
        # or anyone writing their own -- has the checkpoint's numbers instead of
        # a hardcoded 0.5 that happens to be right for this one model.
        #
        # `image_resize` names the GEOMETRY, not an algorithm. "square" means
        # resize the image to size x size and nothing else, which is what
        # google/vit-base-patch16-224 does: its preprocessor_config.json has
        # size {"height": 224, "width": 224} AND crop_size {"height": 224,
        # "width": 224}, so HF's resize-then-centre-crop is a resize with a
        # no-op crop. The packer refuses the other form outright rather than
        # shipping a container that names it.
        "image_resize": pre["resize"],
        "crop_size": pre["crop_size"],
        "image_mean": [float(x) for x in pre["mean"]],
        "image_std": [float(x) for x in pre["std"]],
        # PIL.Image's resample code, named by NUMBER because that is what
        # transformers stores: 0 NEAREST, 1 LANCZOS, 2 BILINEAR, 3 BICUBIC,
        # 4 BOX, 5 HAMMING. A consumer that implements a subset refuses on the
        # codes it does not have rather than substituting a filter that is
        # smoother or cheaper.
        "resample": int(pre["resample"]),
        "qkv_n": 3 * d,
        "gated_ffn": False,
        # PRE-LN, stated as data because it is the difference from every other
        # arch here and the runtime branches on it rather than on the arch
        # string alone: the same normalisation-order statement is what makes
        # one encoder correct for both.
        "norm_order": "pre",
        "attention_scale": scale,
        "qkv_scale_folded": bool(fold_scale),
        "classifier": "host",
        # The operand datatype the array is to be built for. "bf16" is the
        # default; "i8" means every GEMM operand is int8 with a
        # per-output-channel scale and a SmoothQuant factor, and the design set
        # must be exported with --int8 or the runtime refuses the pair by name.
        "a_dtype": I8_DTYPE,
        # INT4 ONLY: the K-group size behind every .gscale in this container.
        # Absent for bf16/int8 -- a config that carries it is an int4 config,
        # and the reader cannot fold an int4 panel without it (npue.fold_i4).
        **({"int4_group": int4_group} if int4_group is not None else {}),
        "head_npu": False,
        "pooling": "cls",
        "fusions": [
            "Q|K|V fused into one [768,2304] operand per layer; all three "
            "biases exist in this architecture, so the fused bias is their "
            "concatenation",
            "1/sqrt(head_dim) folded into Q's weight and Q's bias only",
            "the patch-embedding conv rewritten as an exact GEMM: stride "
            "equals kernel, so im2col is a permutation and the conv is one "
            "[768,768] matrix multiply",
            ("patch_embed rides the attn_out stream's shape -- the array "
             "computes it with no new design and no new stream"),
            ("GEMM operands pre-tiled int8 block_panel (64,48) with a "
             "per-output-channel scale and a SmoothQuant factor"
             if int8 else
             "GEMM operands pre-tiled bf16 block_panel (64,48)"),
            "LayerNorm gamma/beta, every bias, the position table, the CLS "
            "token and the label vocabulary kept fp32 / raw",
        ],
        "not_implemented": [
            "the classifier head runs on the HOST: 1000 is not a multiple of "
            "tile_n*cols, so no B panel of that width exists for this array, "
            "and one 768x1000 matvec per image costs less than the ~150 us a "
            "dispatch to ask for it",
            "the image front end (decode, resize, normalise, im2col) is the "
            "caller's, or this runtime's PNG/JPEG reader. The container carries "
            "the geometry, mean, std and resample code as DATA (image_resize, "
            "crop_size, image_mean, image_std, resample), read from the "
            "checkpoint's own preprocessor_config.json and refused rather than "
            "defaulted, so a consumer never has to guess them",
            "attention is O(seq^2) on the host, as in the BERT path. This repo "
            "has no measurement above seq 64 and 197 positions is three times "
            "that, so the throughput is stated as unmeasured rather than "
            "assumed",
            "batching more than one image per request is a host-side loop, not "
            "a batch tier: the design set's M is a multiple of 256 and a "
            "ViT's is 197 padded to 256",
        ],
    }

    w = Writer(config, arch=ARCH_VIT_PATCH16_PRELN)

    # -- front end: position table, CLS token, patch-embedding GEMM ----------
    pos = st.array("vit.embeddings.position_embeddings")
    if pos.shape != (1, n_pos, d):
        raise SystemExit(
            f"position_embeddings is {pos.shape}, expected (1, {n_pos}, {d}) "
            f"for a {image}px image at patch {patch}. The container stores one "
            f"position table and the runtime indexes it by row, so a mismatch "
            f"here is a model that cannot run.")
    w.add("frontend.position_embeddings",
          np.ascontiguousarray(pos[0], dtype=np.float32), "F32",
          "embedding", [n_pos, d])
    cls = st.array("vit.embeddings.cls_token")
    w.add("frontend.cls_token", np.ascontiguousarray(cls.reshape(d),
                                                     dtype=np.float32),
          "F32", "embedding", [d])
    pe = patch_embed_operand(
        st.array("vit.embeddings.patch_embeddings.projection.weight"))
    if pe.shape != (chan * patch * patch, d):
        raise SystemExit(f"patch_embed operand is {pe.shape}, expected "
                         f"{(chan * patch * patch, d)}")
    if ictx:
        ictx.emit(w, "frontend.patch_embed", pe, "patch_embed")
    else:
        add_gemm_b(w, "frontend.patch_embed", pe, mac=mac)
    w.add("frontend.patch_embed.bias",
          st.array("vit.embeddings.patch_embeddings.projection.bias"),
          "F32", "bias", [d])

    # -- encoder stack -------------------------------------------------------
    n_tiled = 1
    for i in range(layers):
        p = f"vit.encoder.layer.{i}."
        e = f"layer.{i}."
        # PRE-LN: normalise first, then the sub-block, then add the residual.
        _ln(w, e + "ln1", st, p + "layernorm_before.weight",
            p + "layernorm_before.bias", d)
        _fuse_qkv(w, e + "qkv", st, p + "attention.attention.", d, scale,
                  fold_scale, mac=mac, ctx=ictx, key=e + "qkv")
        _out_proj(w, e + "attn_out", st, p + "attention.output.", mac=mac,
                  ctx=ictx, key=e + "attn_out")
        _ln(w, e + "ln2", st, p + "layernorm_after.weight",
            p + "layernorm_after.bias", d)
        if ictx:
            ictx.emit(w, e + "ffn_up",
                      linear_matrix(st, p + "intermediate.dense.weight"),
                      e + "ffn_up")
        else:
            add_gemm_b(w, e + "ffn_up",
                       linear_matrix(st, p + "intermediate.dense.weight"),
                       mac=mac)
        w.add(e + "ffn_up.bias", st.array(p + "intermediate.dense.bias"),
              "F32", "bias", [inter])
        if ictx:
            ictx.emit(w, e + "ffn_down",
                      linear_matrix(st, p + "output.dense.weight"),
                      e + "ffn_down")
        else:
            add_gemm_b(w, e + "ffn_down",
                       linear_matrix(st, p + "output.dense.weight"), mac=mac)
        w.add(e + "ffn_down.bias", st.array(p + "output.dense.bias"),
              "F32", "bias", [d])
        n_tiled += 4

    _ln(w, "layernorm", st, "vit.layernorm.weight", "vit.layernorm.bias", d)

    # -- the classifier head, on the host and NOT pre-tiled ------------------
    #
    # Plain row-major F32 under the role that already means "an operand the host
    # reads": no layout, so no layout_hash, so nothing to disagree with a
    # design. See the header for why it is not on the array at all.
    #
    # The checkpoint's head is [labels, hidden] -- torch's nn.Linear layout --
    # and it is TRANSPOSED here, because this container's convention for a
    # [K, N] GEMM operand is K-major and `gemm_b_host` is the same role as
    # `gemm_b` with a different tile. Storing [1000, 768] and declaring
    # [768, 1000] would read back as a well-formed matrix of the right shape
    # containing the wrong numbers, which is why the shape is passed as
    # [d, n_labels] and the data is made to match it.
    head = st.array("classifier.weight")
    if head.shape != (n_labels, d):
        raise SystemExit(
            f"classifier.weight is {head.shape}, expected {(n_labels, d)} for "
            f"{n_labels} labels of width {d}. A checkpoint whose head does not "
            f"match its own id2label would classify into a range it has no "
            f"weights for.")
    head_t = np.ascontiguousarray(head.T, dtype=np.float32)
    w.add("classifier.weight", head_t, "F32", "gemm_b_host", [d, n_labels])
    w.add("classifier.bias",
          st.array("classifier.bias").astype(np.float32), "F32", "bias",
          [n_labels])

    # -- the label vocabulary, so the runtime can print a NAME ---------------
    id2label = cfg["id2label"]
    labels = [id2label[str(i)] for i in range(n_labels)]
    blob = ("\n".join(s.replace("\n", " ") for s in labels) + "\n").encode("utf-8")
    w.add("labels.table", np.frombuffer(blob, dtype=np.uint8), "U8",
          "tokenizer", [len(blob)])

    st.close()

    if dry_run:
        print(f"  would write {len(w.entries)} tensors "
              f"({n_tiled} pre-tiled GEMM operands) to {out}")
        return 0

    info = w.write(out)
    total = Path(out).stat().st_size
    print(f"\n  arch       : 5 vit_patch16_prenorm_gelu  hidden {d}, "
          f"{layers} pre-LN layers, {heads} heads x {head_dim}, patch {patch}, "
          f"{n_pos} positions ({n_patch} patches + CLS), {n_labels} labels")
    print(f"  source     : {src_sha[:16]}...  (google/{model_dir.name})")
    print(f"  tensors    : {len(w.entries)}  ({n_tiled} pre-tiled GEMM operands)")
    print(f"  on array   : patch_embed (rides attn_out), qkv, attn_out, "
          f"ffn_up, ffn_down")
    print(f"  on host    : classifier [{d},{n_labels}], image front end, "
          f"attention")
    print(f"  labels     : {len(labels)} names "
          f"({len(blob) / 1e3:.1f} kB)")
    print(f"  data       : {info['data_length'] / 1e6:.2f} MB")
    print(f"  file       : {out}  ({total / 1e6:.2f} MB)")
    if ictx:
        print(ictx.report())
    print(f"  layout_hash: "
          f"{layout_hash(gemm_b_layout(TILE_K, TILE_N, mac[0], mac[1], dtype=I8_DTYPE))[:16]}..."
          f"  (mac s={mac[0]} t={mac[1]}, {device or MAC_DEFAULT_DEVICE}"
          f"{', i8 operands' if int8 else ''})")
    return 0