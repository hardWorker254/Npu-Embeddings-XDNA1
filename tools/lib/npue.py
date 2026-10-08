# NpuEmbeddings -- the .npue runtime weight container.
#
# Spec lives in docs/04-model/README.md; this is the implementation, and it is
# the reference for the C++ loader in M7. numpy only, so it runs in the iron env.
#
# Layout of the file:
#   [0:64]                   FileHeader, exactly 64 bytes, little-endian
#   [json_offset:+json_len]  UTF-8 JSON: config + tensor directory
#   [data_offset:+data_len]  raw tensor data, every tensor 4096-byte aligned
#
# Design rules, all from docs/04-model and each with a reason:
#
#  * 4096-byte alignment per tensor -- page size, and a multiple of any DMA
#    burst. Costs a few KB on a 21 MB file.
#  * mmap-able: the runtime hands raw pointers to DMA descriptors and never
#    memcpys weights at load. That is why nothing here is compressed and why
#    tile data is stored in exactly the order the DMA will read it.
#  * The layout descriptor is DATA, not code. Retuning to a different tile size
#    regenerates the file rather than editing the loader. `layout_hash` makes a
#    stale file fail loudly instead of producing plausible garbage embeddings.
#  * `source_sha256` of the upstream ONNX model travels with the file, so
#    a golden comparison can assert it ran against the same checkpoint.
#
# SPEC CORRECTION (M4): docs/04-model declared `uint8_t reserved[24]` and
# "exactly 64 bytes", but 4+4+4+4 + 8*4 = 48, so 24 reserved bytes gives 72.
# The header is 64 bytes and reserved is 16. See tasks/0006.

import hashlib
import json
import struct

import numpy as np

MAGIC = b"NPUE"
VERSION = 1

ARCH_BERT_ABS_GELU_POSTLN = 0
# EmbeddingGemma-300M (Gemma3): RMSNorm x/rms*scale (HF's `x/rms*(1+w)`, with
# the `+1` already folded into the scale the ONNX export carries -- see
# encoder_gemma.py's header), MQA + RoPE (theta is PER
# LAYER) + q_norm/k_norm between the projection and RoPE, GeGLU, and four
# RMSNorms per layer rather than BERT's two LayerNorms.
#
# RUNS ON THE ARRAY since tasks/0074-m13-gemma-on-npu/TASK.md -- 97.7% of its
# MACs, on pre-tiled bf16 operands under BERT's tensor names, exactly like
# arch=2 below. Until then it was host-only, because MQA's 1 KV head at
# head_dim 256 makes K and V 256 wide and 256 caps the legal tile_n at 16.
# The fix is ZERO-PADDING the fused Q|K|V from 1280 to 1536 (a multiple of
# tile_n * n_aie_cols = 384): zero columns of B give exactly-zero columns of
# C, so it is exact rather than an approximation, and the host slices Q/K/V
# off the front by offset. See gemma_qkv_blocks() in tools/pack/pack_npue.py.
#
# The HOST-only layout still exists and is still packable (--gemma-host-only),
# but it is now the correctness CONTROL, not the product: it accumulates every
# GEMM in double precision and is tied to reference/encoder_gemma.py at 1-cos
# 5.496e-13, which is what the array path is gated against. A container says
# which of the two it holds in config["gemm_layout"] ("pretiled_bf16" or
# "host"); the runtime READS that rather than inferring it from the arch,
# because the same architecture now has both.
ARCH_GEMMA3_MQA_ROPE_GEGLU = 1

# nomic-embed-text-v1.5 (nomic_bert): RoPE (NeoX-style, theta=1000, applied to
# Q and K only, positions start at 0) + a gated SwiGLU FFN (fc11 is the
# untouched up-path, fc12 gets SiLU: out = fc11(x) * silu(fc12(x)), fused into
# one [hidden, 2*intermediate] ffn_up so the array still sees four GEMMs per
# layer, not five), post-LN, and NO biases anywhere
# (qkv_proj_bias/mlp_fc1_bias/mlp_fc2_bias all False). Every architectural
# fact here was settled empirically, not read off a model card -- see
# tasks/0068-m13-nomic-spike-and-oracle/TASK.md sec 5.
#
# Tensor names and emission order are IDENTICAL to arch=0 (BERT) -- see
# tasks/0069-m13-nomic-arch2-container/TASK.md item 3 -- so
# Encoder::stage_all() and the whole NPU dispatch path work UNCHANGED.
# Every "*.bias" tensor and "embeddings.position" are ZERO-FILLED rather than
# omitted, because the runtime dereferences both unconditionally
# (main.cpp:702, :2889) -- a zero tensor of the right shape is exact (nomic
# has no biases, and RoPE replaces the absolute position table) and costs far
# less than threading nullable branches through the hot path for one new arch.
#
# This arch's GEMM operands are pre-tiled bf16 block_panel, exactly like BERT,
# because nomic's geometry (head_dim 64, every N a multiple of 384, K in
# {768, 3072}) fits the array without any padding at all. arch=1 now does the
# same, but only after padding its fused qkv -- see its note above.
ARCH_NOMIC_ROPE_SWIGLU = 2

# gte-multilingual-base (model_type "new", the NewModel trust_remote_code
# impl; tasks/0134's oracle is the executable spec): RoPE with an NTK-scaled
# frequency set THAT NO SINGLE THETA CAN EXPRESS -- inv_freq_i =
# 160000^(-i/32) / 8^(1/32), derived from NTKScalingRotaryEmbedding's double
# cache build and verified bit-for-bit against a freshly constructed module
# (0134; the constant correction 8^(-1/32) scales even frequency 0). The
# container therefore carries the 32 inv_freq values as data
# ("rope_inv_freq"), and rope_theta/rope_scaling are provenance, not inputs.
# Post-LN, fused-by-the-checkpoint up_gate GLU with exact-erf GELU on the
# GATE half (up first, gate second -- nomic's ordering with GELU for SiLU),
# REAL biases on qkv/attn_out/ffn_down (unlike nomic), bias-free ffn_up,
# CLS pooling, l2_normalize genuinely from the checkpoint (2_Normalize in
# modules.json). Tokenizer is SentencePiece Unigram via the XLMRTOK1 blob
# (T52, tasks/0127) -- a third family beside WordPiece and Gemma's SP-BPE.
#
# Tensor names and emission order are IDENTICAL to arch=0/2 so the packer
# and the whole NPU dispatch path work unchanged; geometry (qkv N=2304,
# attn_out 768, ffn_up 6144, ffn_down K=3072, tile_n 48) is a literal match
# to the shipping nomic design set, b_layout_hash included.
ARCH_GTE_NEW_ROPE_GEGLU = 3

# Whisper (openai/whisper-*): speech-to-text, so this arch carries BOTH halves
# of an encoder-decoder in one container -- there is no text embedding path to
# fall back on, and a decoder that cannot see its encoder is not a model.
#
# What is different from arch=0/2/3, in the order the runtime meets it:
#
#  * NO TOKEN IDS reach the encoder. The input is a 30 s log-mel spectrogram
#    (up to 3000 frames, 80 mel bins for tiny..medium and 128 for large-v3 and
#    large-v3-turbo) that goes through conv1 (mel x d x 3, stride 1) and conv2
#    -- the checkpoint's `conv2` is Whisper's `conv1_pos` -- both with GELU.
#    The absolute position table is FIXED and sinusoidal, not learned per
#    checkpoint in the BERT sense: it is carried as `embeddings.position` so
#    the encoder's add is the same add the BERT path already does.
#  * NO token_type tensor, and no token embedding on the encoder side at all.
#  * k_proj HAS NO BIAS in every one of the 3*layers projections, while
#    q_proj, v_proj and out_proj do. The fused Q|K|V therefore carries a bias
#    whose K segment is ZERO-FILLED: k = x*Wk is then reproduced exactly, not
#    approximated, and the array still sees one GEMM per attention instead of
#    three. The decoder's cross-attention splits differently, because its Q
#    comes from the decoder state and its K|V from the encoder output: two
#    operands, `cross_q` [d,d] and `cross_kv` [d,2d], the latter with a
#    zero-filled K half for the same reason.
#  * pre-LN, GELU, three LayerNorms per decoder layer (self-attn,
#    cross-attn, FFN) against BERT's two.
#  * There is NO proj_out: the checkpoint ties the decoder's output projection
#    to the token embedding, so the logits are `h @ embed_tokens.T`. The
#    container records `tied_embeddings` rather than shipping a second copy of
#    a 51866 x d matrix.
#  * Tiling is (tile_k 64, tile_n 32) for ALL SIX sizes, which is a fact about
#    the geometry rather than a preference: every K is d or 4d and every N is d,
#    2d, 3d or 4d, and d/32 is a multiple of 4 for d in {384, 512, 768, 1024,
#    1280}, so N % (32 * 4 columns) == 0 with no padding anywhere. tile_n 48,
#    which every BERT-family container uses, does NOT divide 5120 (large-v3's
#    FFN) and would force a repack per size.
ARCH_WHISPER_ENC_DEC_GELU = 4

# google/vit-*-224 (ViT): pre-LN, GELU, a learned position table of 197 rows and
# a classification head. It answers a THIRD question -- "which of a thousand
# names is this picture" -- and it exists to test that the modality is new and
# the ARRAY is not.
#
# What it reuses, exactly:
#  * The FOUR GEMM shapes BERT already has: qkv [768,2304], attn_out
#    [768,768], ffn_up [768,3072], ffn_down [3072,768]. No new stream, no new
#    design, no change to tools/export/ -- which is the claim being tested.
#  * The patch-embedding conv rides the attn_out stream's shape: Conv2d(3,768,
#    kernel=16, stride=16) over 224px is 196 non-overlapping patches, and im2col
#    of a stride-equals-kernel convolution is a PERMUTATION, so the conv is one
#    [768,768] GEMM exactly. See packers/vit.py:patch_embed_operand for the
#    layout, which is the one thing in this arch that a wrong answer looks
#    perfectly healthy.
#  * tile_n 48 and tile_k 64 are BERT's, and are forced here too: every K is
#    768 or 3072 (multiples of 64) and every N is 768, 2304 or 3072 (multiples
#    of 48*4 = 192), so nothing needs padding.
#
# What it deliberately does NOT reuse:
#  * arch=0's TENSOR NAMES. This family is pre-LN and BERT's is post-LN; a
#    container carrying BERT's names with ViT's order would be read happily by
#    BertEncoder and compute a different model. The names below are ViT's own
#    (`layer.i.*`, `frontend.*`), so no name collision is possible and the arch
#    whitelist is the only gate standing between the two.
#  * The classifier head is NOT on the array. 1000 is not a multiple of
#    tile_n*cols, so no B panel of that width exists for this generation, and
#    one 768x1000 fp32 matvec per image costs less than the measured ~150 us a
#    dispatch costs to ask the array for it. It is stored as a plain F32
#    `gemm_b_host` -- no layout, therefore no layout_hash, therefore nothing
#    that can disagree with a design.
#
# The `kind` string is "cls": it selects no extra stream set in the exporter, and
# it is what tells the runtime this container is classified rather than
# embedded, which is the difference the `classify` mode exists to express.
ARCH_VIT_PATCH16_PRELN = 5

# arch=6, body pose. The FIRST architecture in this tree whose container holds a
# GRAPH rather than a shape: `pose`'s graph is an explicit list of ops (conv,
# concat, add, maxpool, upsample, detect) because a conv net's structure is not
# recoverable from any set of three numbers the way a transformer's is. The
# runtime walks the list and refuses one whose operand counts do not typecheck --
# see runtime/src/pose/geometry.cpp, which is where that check lives.
#
# The one thing this arch does NOT introduce is a new tensor role: every
# convolution is stored as plain F32 `conv.N.w` / `conv.N.b`, exactly the
# checkpoint's own [Cout, Cin, kh, kw], because the host backend indexes it
# directly. A pre-tiled `gemm_b` panel is added ONLY under --npu, per
# convolution, alongside the F32 copy -- never instead of it.
#
# The `kind` string is "pose": it selects no extra stream set in the exporter
# (the array backend rides the same gemm_rtp directory name as every other
# architecture), and it is what tells the runtime this container is posed rather
# than embedded -- the difference the `pose` mode exists to express.
ARCH_YOLOV8_POSE_C2F_SILU_DFL = 6

# MediaPipe Hands (palm_detection + handpose_estimation, both float): the FIRST
# container in this format that holds TWO networks. Not a convenience -- the
# landmark network consumes a cropped, rotated palm ROI and has no way to find a
# hand in a frame, so a container with only it would answer "no hand here" to
# every image while looking completely healthy.
#
# The `kind` string is "hands". It rides the SAME gemm_rtp directory as every
# other array-backed architecture, exactly as "pose" does; what it adds is a
# second stream set, because MediaPipe's two pyramids ask for five conv shapes
# YOLOv8 never produces (see geometry.py's HANDS_CONV_SHAPES).
#
# Its convolution vocabulary is MobileNet's, not YOLOv8's: PReLU (per-channel)
# and relu6 instead of SiLU, depthwise convolutions instead of dense 3x3, a
# channel-axis Pad, and a bilinear Resize in the FPN neck. Depthwise is the one
# that cannot go to the array -- see ARCH_STRING's packer for why it is a host
# op and not a refusal.
ARCH_MEDIAPIPE_HANDS_PALM_SSD_LM_HEATMAP = 7

# The third MediaPipe family, and the first with a segmentation mask.
#
# Two networks again, for the reason that stops being true if you shorten it:
# the pose landmark net is handed a person's RoI and cannot find a person in a
# frame, so a container with only it answers "no person here" to every image,
# and a container with only the detector stops at a box. What differs from
# hands is WHAT the second network does with its input. MediaPipe Hands crops a
# palm and rotates it by the two points its own detector regresses. MediaPipe
# Pose crops a person and rotates it by `mid_hip -> full_body`, two of the four
# points its detector regresses, then un-rotates every landmark AND a 256x256
# mask on the way out. The rotation is therefore not an implementation detail
# of the crop; it is part of the answer, and getting it backwards produces
# plausible skeletons mirrored about the hip rather than an error.
#
# Its convolution vocabulary is the same MobileNet one hands uses -- relu6,
# depthwise, bilinear Resize -- with THREE additions hands never needed:
# plain `Relu` (which the Act enum already names, so it is a fusion and not a
# new op), a SPATIAL Pad (hands' pad_c appends channels and net.cpp refuses
# any pad that changes the spatial extent), and DepthToSpace, which is genuinely
# new. The spatial Pads do not become ops at all: ONNX Conv pads are zeros, so
# a zero Pad in front of a Conv is folded into that Conv's own `pad` field and
# the arithmetic is identical rather than approximately so. DepthToSpace has no
# such reformulation and is emitted as its own node.
ARCH_MEDIAPIPE_POSE_DET_SSD_LM_REGRESS = 8

FLAG_PRETILED = 1 << 0

# A COMPILATED DESIGN SET, carried inside the container.
#
# WHY IT LIVES HERE AT ALL. A design set is 288-536 KB of instruction streams plus
# a 71 KB xclbin, and producing it needs MLIR-AIE -- a Python toolchain the size of
# a compiler. Without this, `curl`ing one file and running the binary is not enough:
# the reader must also have a working AIE install to build the thing the reader
# dispatches to. So the end user is asked to install a compiler to run a model. With
# this, one .npue is self-sufficient and the toolchain is a BUILD-time dependency,
# which is the only place a compiler belongs.
#
# ONE CONSTANT FOR THE ROLE STRING, because the alternative is the role written as a
# literal at the writer and again at every reader, and a typo there produces a
# container whose blobs are present, correctly sized, and never looked at -- the
# silent-wrong-answer shape this format's design has been bent to avoid elsewhere.
DESIGN_ROLE = "design"
DESIGN_PREFIX = "design/"

HEADER_FORMAT = "<4sIII QQQQ 16s"      # see SPEC CORRECTION above
HEADER_SIZE = 64
assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE

ALIGN = 4096

# dtype tags. BF16 has no numpy dtype, so it travels as raw uint16 and is
# widened on read -- the same convention as tools/lib/onnx_weights.py, which
# is what serves those bytes from the ONNX export.
NP_DTYPE = {"F32": np.dtype("<f4"), "I32": np.dtype("<i4"), "I64": np.dtype("<i8"),
            "U16": np.dtype("<u2"), "BF16": np.dtype("<u2"),
            # I8 carries a per-output-channel symmetrically quantised GEMM
            # operand (tasks/0078). The int8 MMAC datapath is a different one
            # from bf16's -- native (8,8,8) mac_dims, an int32 accumulator
            # with NO rounding in the reduction -- measured at 5.5-7.7x on
            # every production shape (tasks/0077). Scales ride alongside as an
            # F32 "<name>.wscale" tensor; without them the bytes are
            # meaningless, which is why the runtime refuses a container whose
            # operand dtype disagrees with the design's `a_dtype`.
            "I8": np.dtype("<i1"),
            # I4 is the one tag here that is NOT one value per element: two
            # int4 weights per byte, low nibble first (see pack_i4). What
            # sits in the file is a byte, so the numpy dtype is u1 -- and
            # frombuffer() over it yields HALF as many elements as the panel
            # it decodes into, which is why raw() REFUSES an I4 entry rather
            # than returning an array whose length contradicts the shape it
            # was asked for. panel() is the accessor that decodes it.
            #
            # It is weight storage, not a datapath: the array still runs
            # int8 x int8, and the nibbles are widened before stage(). So a
            # container of I4 operands still carries config a_dtype "i8" and
            # still matches an int8 design's b_layout_hash -- the scheme that
            # produces the nibbles and the scales beside them is
            # tools/lib/gemm_i4.py.
            "I4": np.dtype("<u1"),
            # U8 carries opaque bytes -- the tokenizer vocabulary, so a
            # deployed model is ONE file rather than a file plus a
            # vocab.txt that must not get separated from it.
            "U8": np.dtype("<u1")}


def to_bf16_bits(x):
    """fp32 -> bf16 bit pattern (uint16), round-to-nearest-even.

    bf16 is the top 16 bits of fp32, so this is a rounding of the bit pattern.
    RNE and not truncation: truncation biases every weight toward zero, which
    over 10.6M parameters is a systematic error rather than noise.
    """
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def from_bf16_bits(bits):
    """bf16 bit pattern -> fp32. Exact: bf16 is a strict subset of fp32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


# -- int8 GEMM operands -----------------------------------------------------
#
# An I8 operand is three tensors, not one: the panel, its per-output-channel
# weight scale and its per-INPUT-channel SmoothQuant divisor. tools/lib/gemm_i8.py
# writes all three; the C++ loader reads them by these exact names
# (runtime/src/encoders/bert_encoder.cpp builds `name + ".wscale"`), so they
# are named HERE, once, and both sides spell them the same way.
#
# A reader that returns the panel without them hands back a quantised weight
# with no scale applied -- a number about 1e-4 of the real one. That is not a
# rounding error, it is a different model, and every value downstream of it
# still looks plausible.
WSCALE_SUFFIX = ".wscale"
ASMOOTH_SUFFIX = ".asmooth"
# int4 only: the per-(K-group, output-channel) factor that widens an int4
# weight back into the int8 value the array multiplies. It rides as its own
# tensor because the factor varies along BOTH axes and wscale is per-channel.
GSCALE_SUFFIX = ".gscale"

# The `role` a CONVOLUTION weight is stored under -- two spellings, because
# there are two writers and both sets of containers are still on disk.
# conv_quant.add_conv_w writes "conv_w" (arch=6 first, and now arch=7 and
# arch=8); the packers as they stood before convolution weights could be
# quantised wrote "conv", which is what models/mediapipe-hands.npue,
# models/mediapipe-pose.npue and every float variant carry. `role` is not
# otherwise consulted by this module, so this is the one place the two meet.
CONV_W_ROLES = ("conv_w", "conv")


def pack_i4(q):
    """int4 values held in an int8 panel -> packed bytes, LOW nibble first.

        b[i] = (q[2i] & 0xF) | ((q[2i+1] & 0xF) << 4)

    Two per byte, even index in the low nibble. This function and unpack_i4
    are the only definitions of that order -- a container whose two halves
    disagreed would still be the right size, still carry the right layout_hash
    and would produce wrong numbers, which is the failure mode this format
    spends most of its refusals on.

    Values outside [-8, 7] are REFUSED, not truncated: silently dropping the
    high bits of a 127 would turn an int8 payload into plausible int4 garbage.
    """
    flat = np.ascontiguousarray(q, dtype=np.int8).reshape(-1)
    if flat.size % 2:
        raise ValueError(
            f"pack_i4: {flat.size} values cannot be paired; an int4 panel is "
            f"tile_k*tile_n*blocks and both are even, so this is a tiling bug, "
            f"not a shape this format cannot hold")
    if flat.size and (int(flat.min()) < -8 or int(flat.max()) > 7):
        raise ValueError(
            f"pack_i4: values span [{int(flat.min())}, {int(flat.max())}], and "
            f"int4 holds [-8, 7] -- quantise to int4 before packing, or the "
            f"high bits are lost without a trace")
    lo = (flat[0::2].astype(np.int16) & 0x0F).astype(np.uint8)
    hi = (flat[1::2].astype(np.int16) & 0x0F).astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_i4(packed):
    """Packed bytes -> the int8 panel's int4 values, sign-extended from 4 bits.

    The exact inverse of pack_i4; `verify_i4_scheme` round-trips both through
    every value in [-8, 7].
    """
    b = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1)
    lo = (b & 0x0F).astype(np.int16)
    hi = ((b >> 4) & 0x0F).astype(np.int16)
    lo = np.where(lo > 7, lo - 16, lo).astype(np.int8)
    hi = np.where(hi > 7, hi - 16, hi).astype(np.int8)
    out = np.empty(b.size * 2, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out


def fold_i4(t, gscale, group):
    """int4 values in a LOGICAL [K, N] panel -> the int8 bytes the array gets.

        q[k,j] = rint(t[k,j] * gscale[k // group, j])     clipped to +-127

    `t` holds the int4 rounding; `gscale` (gemm_i4.add_gemm_b_int4) is the
    ratio between that rounding's per-group scale and the per-column scale the
    runtime multiplies by, so the product is the weight in units of wscale --
    i.e. the same int8 panel an int8 container would carry, up to the int4
    error that is the point of the format.

    rint is round-half-to-EVEN, and the C++ unpack uses nearbyint under the
    default FE_TONEAREST rounding mode, which is the same rule: the two sides
    must produce byte-identical panels or a gate comparing them measures the
    tie-break, not the packing.

    `group` is config's int4_group: 0 means one group spanning the whole K
    (per-channel int4), a positive N means groups of N rows.
    """
    t = np.asarray(t, dtype=np.int8)
    if t.ndim != 2:
        raise ValueError(f"fold_i4: panel is {t.shape}, expected a [K, N] matrix")
    K, N = t.shape
    gs = np.asarray(gscale, dtype=np.float32)
    if gs.ndim != 2 or gs.shape[1] != N:
        raise ValueError(
            f"fold_i4: gscale is {gs.shape}, expected [ceil(K/group), {N}]")
    g = int(group or K)
    want_g = -(-K // g)
    if gs.shape[0] != want_g:
        raise ValueError(
            f"fold_i4: gscale has {gs.shape[0]} group rows but K={K} with "
            f"group={g} needs {want_g} -- the panel and its scales disagree "
            f"about the group size, and folding at the wrong one would weight "
            f"every row by its neighbour's scale")
    row = (np.arange(K, dtype=np.int64) // g)
    return np.clip(np.rint(t.astype(np.float32) * gs[row, :]), -127, 127
                   ).astype(np.int8)


def dequant_int8(q, wscale, asmooth=None):
    """An int8 B panel -> the fp32 weight to hand an fp32 GEMM.

    The runtime arithmetic, from runtime/include/common/host_kernels.hpp
    (`quantise_a_int8`, `dequantise_c`):

        Aq[i,:] = rint((X[i,:] / asmooth) / sa[i])      sa[i] = max|.| / 127
        Y[i,j] = int32_dot(Aq[i,:], Wq[:,j]) * sa[i] * s[j] + bias[j]

    so the int32 result is a rank-1 product of two scales, and the effective
    weight is `Wq * s`.

    THE SMOOTHQUANT HALF, AND IT IS A DIVISION. The packer stores `W *
    asmooth` (`gemm_i8.add_gemm_b_int8`) because the array divides the
    ACTIVATION by asmooth, and

        (X / asmooth) @ (W * asmooth) == X @ W

    so a caller that pre-divides its activation -- the array -- wants the
    stored operand as it is. A caller that does NOT pre-divide, which is every
    Python reference, wants `W` itself, and

        W == (Wq * s) / asmooth

    DIVIDE. This function used to multiply, which returns `W * asmooth^2`
    instead of `W`, and the error is invisible exactly when it matters least:
    `asmooth` ships even when it is all ones (`gemm_i8.py`), so every synthetic
    check written against an unsmoothed container agreed, and the first
    genuinely calibrated int8 container measured 1-cos 1.045 -- not because
    int8 quantisation is that bad (the weight error is 2.2e-02) but because
    every weight was scaled by the square of a per-channel factor ranging over
    0.42 .. 8.82.

    The int8 path's error is NOT only here: the runtime also quantises A per
    row, and nothing in this function models that. A number produced through
    it is the WEIGHT half of the quantisation cost and a lower bound on the
    total -- tools/verify/verify_npue.py labels it as such rather than calling it the
    datapath.
    """
    out = np.asarray(q).astype(np.float32) * np.asarray(wscale, np.float32)[None, :]
    if asmooth is not None:
        out = out / np.asarray(asmooth, np.float32)[:, None]
    return out


# -- tiling ----------------------------------------------------------------

# The tile_n values this project actually ships a design at, in the order a
# refusal should SUGGEST them. Used ONLY to turn a tiling refusal into a named
# fix: 48 is the BERT-family default, 32 is what every whisper set uses, 64
# divides most hidden sizes, and 16 is the last resort because a smaller tile_n
# means more tiles and a worse DMA pattern -- it is offered, but last. A
# refusal that says "N=1024 does not divide by 48" without saying what does
# leaves the caller to guess, and guessing wrong costs an export.
_TILE_N_CANDIDATES = (48, 32, 64, 16)


def tile_b(b, tile_k, tile_n, s=None, t=None, order="k,n"):
    """Pre-tile a [K, N] GEMM operand into the order the DMA will stream it.

    Absorbs BOTH re-layouts the M2 design did at runtime:

      1. L3->L2: gathering a [tile_k, tile_n] tile out of a row-major [K, N]
         DDR buffer, via TensorTiler2D.step_tiler. This is the strided access
         pattern whose dimension hits the 10-bit (max 1023) DMA BD size field --
         the reason ffn_down (K=1536) could not be expressed at all.
      2. L2->L1: the intrinsic sub-tile order, which the design expressed as
         dims_to_stream=[(k//s, s*n), (n//t, t), (s, n), (t, 1)].

    Output is a flat array of tiles in (kb, nb) order, each tile internally in
    (s, t) order when s and t are given. The runtime then reads whole tiles by
    index, so no access-pattern dimension exceeds K/tile_k or N/tile_n -- 24 and
    32 at MiniLM's largest, comfortably under 1023.
    """
    K, N = b.shape
    if K % tile_k or N % tile_n:
        # NAME THE FIX, and name the shape that failed. This refusal used to be
        # a bare `[{K},{N}] does not tile into ({tile_k},{tile_n})` with no
        # operand name and no hint, which made bge-large-en-v1.5 (hidden=1024)
        # look un-packable when it is packable at --tile-n 32: the packer
        # refused, the caller had to guess a flag, and nothing recorded that a
        # flag was needed. A refusal that cannot be acted on gets worked around
        # by picking a different model.
        why = []
        if K % tile_k:
            why.append(f"K={K} is not a multiple of tile_k={tile_k}")
        if N % tile_n:
            fits = [t for t in _TILE_N_CANDIDATES if N % t == 0]
            hint = (f"; --tile-n {fits[0]} would tile it"
                    if fits else "; no tile_n in "
                    f"{sorted(_TILE_N_CANDIDATES)} divides it either")
            why.append(f"N={N} is not a multiple of tile_n={tile_n}{hint}")
        raise ValueError(
            f"[{K},{N}] does not tile into ({tile_k},{tile_n}): "
            + " and ".join(why) + f". The operand here is {b.shape}, and the "
            f"container and the design must be exported at the SAME tile_n, so "
            f"change it on BOTH sides (pack_npue.py --tile-n and "
            f"export_gemm_rtp.py -n), not one.")

    # [K,N] -> [kb, tile_k, nb, tile_n] -> [kb, nb, ...] or [nb, kb, ...]
    #
    # `order` decides which of kb/nb is major, and it is a PERFORMANCE decision,
    # not a cosmetic one. A core's inner loop walks all K/k k-blocks for one
    # n-block, so with "k,n" those consecutive tiles sit (N/n)*k*n elements
    # apart -- 48 KB at ffn_down -- and the DMA scatters across ~1.1 MB. With
    # "n,k" the same walk is one contiguous run. Measured in M5.
    out = b.reshape(K // tile_k, tile_k, N // tile_n, tile_n)
    if order == "k,n":
        out = out.transpose(0, 2, 1, 3)
    elif order == "n,k":
        out = out.transpose(2, 0, 1, 3)
    else:
        raise ValueError(f"order must be 'k,n' or 'n,k', got {order!r}")

    if s is not None and t is not None:
        if tile_k % s or tile_n % t:
            raise ValueError(f"tile ({tile_k},{tile_n}) not divisible by mac ({s},{t})")
        # within a tile: [tile_k, tile_n] -> [tile_k/s, s, tile_n/t, t]
        #                                 -> [tile_k/s, tile_n/t, s, t]
        out = out.reshape(out.shape[0], out.shape[1],
                          tile_k // s, s, tile_n // t, t).transpose(0, 1, 2, 4, 3, 5)
    return np.ascontiguousarray(out).reshape(-1)


def untile_b(flat, K, N, tile_k, tile_n, s=None, t=None, order="k,n"):
    """Exact inverse of tile_b. The round-trip check is bit-exact, not close."""
    kb, nb = K // tile_k, N // tile_n
    outer = (kb, nb) if order == "k,n" else (nb, kb)
    if s is not None and t is not None:
        x = flat.reshape(*outer, tile_k // s, tile_n // t, s, t)
        x = x.transpose(0, 1, 2, 4, 3, 5).reshape(*outer, tile_k, tile_n)
    else:
        x = flat.reshape(*outer, tile_k, tile_n)
    if order == "n,k":
        x = x.transpose(1, 0, 2, 3)                 # -> [kb, nb, tile_k, tile_n]
    return np.ascontiguousarray(x.transpose(0, 2, 1, 3).reshape(K, N))


def gemm_b_layout(tile_k, tile_n, mac_s=8, mac_t=8, dtype="BF16"):
    """The canonical B layout descriptor. Build it HERE, never inline.

    Every copy of this dict is a chance for two sides to drift, and the drift
    is invisible: `layout_hash` changes, the bytes do not, and the check that
    exists to catch wrong layouts starts reporting a mismatch that is not one.
    That happened -- the upstream seven-design exporter wrote the dict by hand
    and omitted `dtype`, so a correct file failed the check. The packer had it
    twice, too.

    The (8, 8) default is the npu2 sub-tile and NOT a safe default for npu1 --
    see `mac_for_device` and pass what the target generation consumes.
    """
    return {"kind": "block_panel", "tile_k": tile_k, "tile_n": tile_n,
            "order": "k,n,kt,nt", "inner": "s,t",
            "mac_s": mac_s, "mac_t": mac_t, "dtype": dtype}


# The B panel's byte order inside one (tile_k, tile_n) tile is the MMAC
# sub-tile. It depends on the OPERAND DTYPE as well as the board. MEASURED with
# `aie.iron.kernels.mm(...).mac_dims`, which returns (r, s, t), at every tile
# width this project builds (n = 16, 32, 48, 64 -- constant in n):
#
#               npu1 / aie2                 npu2 / aie2p
#     bf16      (4, 8, 4)  -> (8, 4)        (4, 8, 8)  -> (8, 8)
#     i8        (4, 8, 8)  -> (8, 8)        (4, 8, 8)  -> (8, 8)
#
# `tile_b`'s (s, t) is (mac_s, mac_t) here. A container packed with one pair and
# read by a core built for the other has every panel's columns permuted: the
# byte count is right, the shapes agree, both sides derive the same
# `layout_hash` from the same wrong constant, and every product is plausible.
# Nothing in the loader can see it.
#
# WHY THE DTYPE KEY IS NOT OPTIONAL. The int8 row is not a cosmetic extra: on
# npu1 the int8 MMAC's N sub-tile is 8, where bf16's is 4, so a table keyed by
# device alone silently hands an int8 pack the bf16 pair. That shipped a
# bge-small int8 design whose every product was wrong while every check passed
# -- the exporter, the packer, the runtime's layout_hash comparison and
# verify_design_numerics all agreed with each other and all disagreed with the
# hardware, because all four read the same wrong constant. End to end it read as
# 1-cos 8.6e-01 on a model whose bf16 path is 1e-05. `mac_for_device` therefore
# REQUIRES the dtype rather than defaulting it: the bf16 answer is the tempting
# one, and a default is how it got shipped.
#
# The int8 pair is also what an int4 (W4A8) container uses: its B panel is
# widened to int8 before staging, so the bytes the core consumes are int8's.
MAC_BY_DEVICE = {
    "npu1": {"bf16": (8, 4), "i8": (8, 8)},
    "npu2": {"bf16": (8, 8), "i8": (8, 8)},
}
MAC_DEFAULT_DEVICE = "npu1"  # what an unstated target means

# AIE columns per device. A design's N must be a multiple of tile_n * cols, and
# that product is what a B panel has to be padded UP to -- so a packer needs the
# column count and cannot get it from mac_for_device, which knows the sub-tile
# and not the array's width.
#
# The values are the exporter's own fallback columns (FALLBACK_COLS_BY_ARCH in
# tools/export/exporters/common/consts.py: arch 1 -> 4, arch 2 -> 8), repeated
# here rather than imported because tools/pack does not depend on tools/export
# and this table is what makes that true rather than accidental. If one side
# changes, the other has to, and the failure mode if they drift is the WORST
# kind: a panel padded to 32 columns where the core reads 128, which reads past
# the end of the tensor and returns plausible products.
#
# Every model this repository shipped until now had N already a multiple of 128
# -- the embedders' widths are 384, 768, 1536 and 3072 -- so nothing had to pad
# N past tile_n and this table had no reader. YOLOv8-pose is the first network
# with output channels of 16, 32, 48, 51, 64, 96, 128, 192 and 256.
COLS_BY_DEVICE = {"npu1": 4, "npu2": 8}


def cols_for_device(device):
    """AIE columns for `device`, or the default device's when unstated.

    Mirrors mac_for_device's signature on purpose: both are "what does this
    device look like", both take an optional device that falls back to
    MAC_DEFAULT_DEVICE, and both raise on a name neither knows rather than
    guessing a width that would silently over- or under-pad every panel.
    """
    dev = device or MAC_DEFAULT_DEVICE
    if dev not in COLS_BY_DEVICE:
        raise SystemExit(
            f"unknown device {dev!r}: the AIE column count is per-generation "
            f"({', '.join(f'{k}={v}' for k, v in COLS_BY_DEVICE.items())}), and "
            f"guessing one would size every B panel wrongly."
        )
    return COLS_BY_DEVICE[dev]
# npu1, not npu2, and that is a change: every design set and every container
# this repository ships is npu1, so defaulting to npu2 meant an omitted
# --device produced a container whose layout_hash NO design in the tree
# accepts. The failure is loud (the runtime refuses by name), which is what
# kept it from being a wrong-numbers bug, but a default that cannot be used is
# a default that gets passed over. npu2 stays reachable by name for anyone
# building for that board.

# Accepts either spelling: "I8" is the layout dict's, "i8" the CLI's.
_MAC_DTYPE = {"bf16": "bf16", "i8": "i8", "int8": "i8"}


def _mac_dtype(dtype):
    key = _MAC_DTYPE.get(str(dtype).strip().lower())
    if key is None:
        raise SystemExit(
            f"unknown B operand dtype {dtype!r}: the MMAC sub-tile is per "
            f"operand dtype ({', '.join(sorted(set(_MAC_DTYPE)))}). The int8 "
            f"pair is what an int4 container uses too, since it widens to int8.")
    return key


def mac_for_device(device, dtype, mac_s=None, mac_t=None):
    """(mac_s, mac_t) for `device` and B operand `dtype`.

    `dtype` has no default ON PURPOSE -- see MAC_BY_DEVICE. An explicit
    mac_s/mac_t overrides the table.
    """
    if mac_s is not None and mac_t is not None:
        return (int(mac_s), int(mac_t))
    dev = device or MAC_DEFAULT_DEVICE
    if dev not in MAC_BY_DEVICE:
        raise SystemExit(
            f"unknown device {dev!r}: the B panel's sub-tile is per-generation "
            f"({', '.join(f'{k}={v}' for k, v in MAC_BY_DEVICE.items())}). "
            f"Pass one of those, or --mac-s/--mac-t if you know better.")
    by_dtype = MAC_BY_DEVICE[dev]
    dt = _mac_dtype(dtype)
    if dt not in by_dtype:
        raise SystemExit(
            f"no measured MMAC sub-tile for device {dev!r} with B dtype {dt!r}: "
            f"that combination is not in MAC_BY_DEVICE "
            f"({', '.join(sorted(by_dtype))}), and guessing is what the "
            f"layout_hash check cannot see.")
    return by_dtype[dt]


def gemm_b_layout_for_device(device, tile_k, tile_n, dtype="BF16",
                             mac_s=None, mac_t=None):
    s, t = mac_for_device(device, dtype, mac_s, mac_t)
    return gemm_b_layout(tile_k, tile_n, s, t, dtype)


# tile_k and tile_n for the pre-tiled B panels, as of the three conv-only
# packers. ONE pair of constants for all of them, and they were three pairs: pose
# chose 32 because 48 divides none of its channel counts usefully, and mppose and
# hands then copied that 32 rather than deriving it, so a change to one would have
# left the other two writing panels their design could not read.
#
# What actually decides tile_n is `cols`: N pads to tile_n * cols, so the width
# has to divide the widest padded N of the set. For pose (14 shapes, N up to 2304)
# and for mppose (22, N up to 1152) 48 does divide nothing useful and 32 divides
# most. hands' widest padded N is 768, which 48 divides -- but 32 divides it too
# and is what the design set it shipped against was built at, so the choice here is
# "what the existing sets and containers agree on", not a fresh optimisation.
CONV_PANEL_TILE = (64, 32)


def conv_panel(w, device, dtype="BF16", tile=None):
    """One dense convolution's pre-tiled bf16 B panel.

    Returns (tiled_panel, layout, padded_k, padded_n) for a filter of shape
    [Cout, Cin, kh, kw].

    WHY THIS IS IN npue.py AND NOT IN EACH PACKER
    ----------------------------------------------
    Three conv-only packers write these panels -- pose.py, mppose.py, hands.py --
    and the arithmetic here has a wrong answer that is a PLAUSIBLE one. The
    weight is [Cout, Cin, kh, kw] = [N, K] and the panel is [K, N], so
    `w.reshape(K, N)` has the right SIZE, tiles, and matches layout_hash (both
    sides derive that hash from constants, not from the data), and the array then
    multiplies a TRANSPOSED filter: the output has the right shape, every channel
    is a plausible mixture of the right ones, and nothing reports it. pose.py's
    copy said so in a long comment; mppose.py's copy transcribed the comment and
    the arithmetic; hands.py would have been a third, and a third is how the two
    before it drift.

    So the transpose, the K padding to tile_k and the N padding to tile_n*COLS
    live here once. What stays in each packer is the ITERATION -- which
    convolutions are dense, and what their names are -- because that genuinely
    differs per packer: pose has one graph, mppose and hands have two with
    separate index spaces.

    The N padding is to tile_n * cols, NOT to tile_n, and it needs a device for
    that reason alone. `cols_for_device(None)` answers for a device nobody named,
    and a panel built at the wrong width is not a panel the design can read: the
    bytes past the end are whatever follows in the container's data region, and
    every check upstream of the multiply still passes.
    """
    tile_k, tile_n = tile or CONV_PANEL_TILE
    n_mult = tile_n * cols_for_device(device)
    mac_s, mac_t = mac_for_device(device, dtype)
    layout = gemm_b_layout(tile_k, tile_n, mac_s, mac_t, dtype)
    cout, cin, kh, kw = w.shape
    k, n = cin * kh * kw, cout
    pk = ((k + tile_k - 1) // tile_k) * tile_k
    pn = ((n + n_mult - 1) // n_mult) * n_mult
    # Zero padding, which is EXACT and not an approximation: a padded K column is
    # multiplied by nothing and a padded N column is a channel the runtime never
    # reads back. tile_b refuses a shape that does not tile, so the pad happens here.
    b = np.zeros((pk, pn), dtype=np.float32)
    # A TRANSPOSE, not a reshape. See this function's docstring for what a
    # reshape costs.
    b[:k, :n] = w.reshape(n, k).T
    return tile_b(b, tile_k, tile_n, s=mac_s, t=mac_t), layout, pk, pn


def layout_hash(layout):
    """Stable hash of everything that changes how bytes are laid out.

    A file packed with different tile dimensions is not merely suboptimal for a
    kernel expecting others -- it is wrong, silently. The loader compares this
    and refuses rather than producing plausible garbage.
    """
    canonical = json.dumps(layout, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -- writer ----------------------------------------------------------------

class Writer:
    """Accumulates tensors, then writes the container.

    Tensors are staged rather than streamed because the JSON directory carries
    every offset and must be complete before the first data byte is written.
    """

    def __init__(self, config, arch=ARCH_BERT_ABS_GELU_POSTLN, flags=FLAG_PRETILED):
        self.config = dict(config)
        self.arch = arch
        self.flags = flags
        self.entries = []
        self.blobs = []
        self._offset = 0

    def add_design(self, set_name, file_name, blob):
        """One file of a compiled design set, stored as raw bytes.

        `blob` is bytes -- an xclbin is an ELF-ish container and an instruction
        stream is a flat array of 32-bit words, and neither is an ndarray. It is
        stored as U8 with the byte count as the only shape, because there is no
        arithmetic over these bytes and pretending otherwise would invite someone
        to reshape one.

        THE NAME IS `design/<set>/<file>`, and the set name is part of it rather
        than a separate index: a self-sufficient container carries gemm_rtp AND
        gelu AND layernorm AND softmax, the runtime picks between them by name and
        by datapath, and a flat `final.xclbin` in a container that holds four sets
        would be four files of one name.
        """
        name = f"{DESIGN_PREFIX}{set_name}/{file_name}"
        arr = np.frombuffer(blob, dtype=np.uint8)
        return self.add(name, arr, "U8", DESIGN_ROLE, [len(blob)])

    def add(self, name, array, dtype_tag, role, logical_shape,
            layout=None, padded_shape=None):
        arr = np.ascontiguousarray(array, dtype=NP_DTYPE[dtype_tag])
        blob = arr.tobytes()
        entry = {
            "name": name,
            "role": role,
            "dtype": dtype_tag,
            "logical_shape": list(logical_shape),
            "padded_shape": list(padded_shape or logical_shape),
            "offset": self._offset,
            "nbytes": len(blob),
        }
        if layout is not None:
            entry["layout"] = layout
            entry["layout_hash"] = layout_hash(layout)
        self.entries.append(entry)
        self.blobs.append(blob)
        # Pad AFTER each tensor so the next one starts aligned.
        self._offset += len(blob)
        pad = (-self._offset) % ALIGN
        self._offset += pad
        self.blobs.append(b"\0" * pad)
        return entry

    def write(self, path):
        directory = {"config": self.config, "tensors": self.entries}
        js = json.dumps(directory, separators=(",", ":")).encode("utf-8")

        json_offset = HEADER_SIZE
        data_offset = json_offset + len(js)
        data_offset += (-data_offset) % ALIGN          # 4096-align the data blob
        data_length = self._offset

        header = struct.pack(
            HEADER_FORMAT, MAGIC, VERSION, self.arch, self.flags,
            json_offset, len(js), data_offset, data_length, b"\0" * 16,
        )
        with open(path, "wb") as f:
            f.write(header)
            f.write(js)
            f.write(b"\0" * (data_offset - json_offset - len(js)))
            for b in self.blobs:
                f.write(b)
        return {"json_offset": json_offset, "json_length": len(js),
                "data_offset": data_offset, "data_length": data_length,
                "total": data_offset + data_length}


# -- reader ----------------------------------------------------------------

class Reader:
    """Reads a .npue container. Uses np.memmap -- the point of the format is
    that the runtime never copies weights at load."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            head = f.read(HEADER_SIZE)
        (magic, self.version, self.arch, self.flags,
         json_offset, json_length, self.data_offset, self.data_length,
         _reserved) = struct.unpack(HEADER_FORMAT, head)

        if magic != MAGIC:
            raise ValueError(f"{path}: not a .npue file (magic {magic!r})")
        if self.version != VERSION:
            raise ValueError(f"{path}: version {self.version}, expected {VERSION}")

        with open(self.path, "rb") as f:
            f.seek(json_offset)
            directory = json.loads(f.read(json_length).decode("utf-8"))
        self.config = directory["config"]
        self.entries = {e["name"]: e for e in directory["tensors"]}
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def close(self):
        """Release the mapping. Required on Windows before the file can be
        deleted or replaced -- an open memmap holds a lock, and `del` is not
        enough because derived views keep it alive."""
        m = getattr(self._map, "_mmap", None)
        if m is not None:
            m.close()
        self._map = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def design_sets(self):
        """{set_name: {file_name: entry}} for everything this container carries.

        The reader side of DESIGN_ROLE, and it returns ENTRIES rather than bytes
        so a caller can check sizes before deciding whether the set is the one it
        wants. A container that carries no design set yields {} and NOT an error:
        every container packed before this existed is still valid, and the host
        path -- every convolution on the CPU -- needs no design at all.
        """
        out = {}
        for name, e in self.entries.items():
            if e.get("role") != DESIGN_ROLE or not name.startswith(DESIGN_PREFIX):
                continue
            rest = name[len(DESIGN_PREFIX):]
            if "/" not in rest:
                continue
            set_name, file_name = rest.split("/", 1)
            out.setdefault(set_name, {})[file_name] = e
        return out

    def design_bytes(self, set_name, file_name):
        """The exact bytes of one embedded design file, or None if absent.

        None rather than an exception, because "this container does not carry that
        set" is the normal state of every container packed before this feature and
        of every container whose design is still only on disk. The CALLER decides
        whether that is fatal; a reader that raised would make the absence
        indistinguishable from corruption.
        """
        e = self.design_sets().get(set_name, {}).get(file_name)
        if e is None:
            return None
        start = self.data_offset + e["offset"]
        return bytes(self._map[start:start + e["nbytes"]])

    def raw(self, name):
        """The tensor exactly as stored -- tiled, bf16 as uint16.

        NOT the accessor for an I4 entry: those bytes hold two values each, so
        a caller that unpacked them as the stored dtype would get an array half
        the length of the panel it was promised and no error to explain it.
        Use `panel()` for the panel and `tensor()` for the weight.
        """
        e = self.entries[name]
        if e["dtype"] == "I4":
            raise ValueError(
                f"{self.path}: {name} is I4, whose bytes are packed two int4 "
                f"values per byte -- raw() would hand back {e['nbytes']} "
                f"elements for a panel of {e['padded_shape'][0] * e['padded_shape'][1]}. "
                f"panel() gives the int8 panel the array multiplies, tensor() "
                f"the weight itself.")
        start = self.data_offset + e["offset"]
        buf = self._map[start:start + e["nbytes"]]
        return np.frombuffer(buf, dtype=NP_DTYPE[e["dtype"]])

    def panel(self, name):
        """The GEMM B panel as the ARRAY consumes it: flat, in the order
        `layout` describes, one int8 value per element.

        That is raw() for BF16 (uint16, widened by tensor()) and for I8. For I4
        it is the decode -- nibbles unpacked, group scales folded in, tiled
        back -- and the result is byte-for-byte what an I8 container would
        hold at the same shape, up to the int4 error. This is the accessor
        stage() corresponds to; tensor() is what additionally undoes the
        scales, for callers that want the weight rather than the panel.
        """
        e = self.entries[name]
        if e["dtype"] != "I4":
            return self.raw(name)
        if e.get("role") in CONV_W_ROLES:
            raise ValueError(
                f"{self.path}: {name} is an I4 CONVOLUTION weight and has no "
                f"array panel -- the panels these containers carry are the "
                f"separate `.btile` tensors, and the weight itself is the host "
                f"conv engine's. tensor() dequantises it to fp32.")
        K, N = e["padded_shape"]
        lay = e.get("layout")
        if not lay or lay.get("kind") != "block_panel":
            raise ValueError(
                f"{self.path}: {name} is I4 but carries no block_panel layout -- "
                f"the nibbles' positions are what the layout describes, so "
                f"without it there is no order to unpack into. Repack.")
        gs_name = name + GSCALE_SUFFIX
        if gs_name not in self.entries:
            raise KeyError(
                f"{self.path}: {name} is I4 but {gs_name} is not in the "
                f"container -- the int4 weight has no way back to an int8 "
                f"value, so its bytes are not a weight. Repack with "
                f"tools/pack/pack_npue.py --dtype i4.")
        packed = self._map[self.data_offset + e["offset"]:
                           self.data_offset + e["offset"] + e["nbytes"]]
        t = unpack_i4(packed)
        if t.size != K * N:
            raise ValueError(
                f"{self.path}: {name} is I4 with {e['nbytes']} bytes -> "
                f"{t.size} values, but its padded shape {K, N} needs "
                f"{K * N} -- the payload and the panel disagree about their "
                f"size, and half of it would be a neighbour's bytes")
        t = t.reshape(K, N)
        t = untile_b(t, K, N, lay["tile_k"], lay["tile_n"],
                     lay.get("mac_s"), lay.get("mac_t"))
        t = fold_i4(t, self.tensor(gs_name), self.config.get("int4_group", 0))
        return tile_b(t, lay["tile_k"], lay["tile_n"],
                      lay.get("mac_s"), lay.get("mac_t"))

    def payload(self, name):
        """The entry's bytes as uint8, for ANY dtype including I4.

        The accessor `raw()` deliberately refuses: it widens bytes into elements,
        which is wrong for a packed four-bit tensor, and a refusal there is what
        keeps a caller from unpacking half a panel by accident. This one is the
        escape hatch, and it exists for the same reason int4 has a pack/unpack
        pair of its own -- the packed form has to be readable as bytes by
        SOMETHING, and that something is a checker that wants to confirm the
        packing rather than undo it.

        Not a tensor: no dtype, no shape. A caller that wants one of those wants
        `raw()` or `panel()`.
        """
        e = self.entries[name]
        start = self.data_offset + e["offset"]
        return np.asarray(self._map[start:start + e["nbytes"]])

    def _conv_weight(self, name, e):
        """A CONVOLUTION weight, dequantised, at its logical shape.

        WHY THIS BRANCH EXISTS: the general path below reads an I8 or I4 entry
        as a GEMM operand -- TWO sidecars (`.wscale` AND `.asmooth`) and, for
        I4, a `block_panel` layout to untile through. A convolution weight has
        neither: conv_quant writes `.wscale` alone (plus `.gscale` for i4),
        writes NO layout at all, and its array panels are the separate
        `.btile` tensors rather than anything tiled out of the weight. So the
        general path on a quantised conv container raises about a sidecar that
        was never meant to exist, or tilts a weight that was never tiled -- and
        both failures are raised at a container that did exactly what it was
        written to do. That is how tools/verify/verify_hands.py and
        verify_mppose.py came to be unable to open a container they pack.

        Every dtype returns fp32 at `logical_shape`, because that is what a
        conv engine multiplies and what those checkers compare against ORT.

        f32      the bytes are the weights.
        bf16     the bytes are the weights, widened -- exact, and the same
                 widening as a GEMM BF16 operand's (no scale to fold in: the
                 value IS the payload).
        i8       `q[n,k] * wscale[n]`, per output channel, from the file's own
                 `.wscale` rather than one recomputed from the checkpoint --
                 a stored scale one ulp from a recomputation is a real
                 difference in the container and has to be the reader's input,
                 not its own arithmetic.
        i4       `q[n,k] * wscale[n] * gscale[g(k),n]`, left to right, with
                 `g = ceil(K / conv_int4_group)` and the same left-to-right
                 order the C++ reader uses -- `(v * s) * r` and `v * (s * r)`
                 are different float32 numbers, and a checker that quietly
                 swapped them would report a reader bug on weights where two
                 readers agree exactly.

        Missing sidecars raise rather than default: a container whose scale is
        skipped silently is the failure this format's scale pair exists to make
        impossible.
        """
        shp = list(e["logical_shape"])
        dtype = e["dtype"]
        if dtype == "F32":
            return np.ascontiguousarray(self.raw(name).reshape(shp)).copy()
        if dtype == "BF16":
            x = from_bf16_bits(self.raw(name))
            if x.size != int(np.prod(shp)):
                raise ValueError(
                    f"{self.path}: {name} is BF16 with {x.size} values for a "
                    f"{shp} tensor")
            return np.ascontiguousarray(x.reshape(shp)).copy()

        if dtype == "I8":
            q = np.frombuffer(self.payload(name), dtype=np.int8)
            if q.size != int(np.prod(shp)):
                raise ValueError(
                    f"{self.path}: {name} is I8 with {q.size} bytes for a {shp} "
                    f"weight -- one byte per weight, so the payload and the "
                    f"shape disagree about which is wrong")
            N, K = shp[0], int(np.prod(shp[1:]))
            s = self._sidecar(name, WSCALE_SUFFIX, N, e)
            x = (q.reshape(N, K).astype(np.float32) * s[:, None]).reshape(shp)
            return np.ascontiguousarray(x).copy()

        if dtype == "I4":
            N, K = shp[0], int(np.prod(shp[1:]))
            want = N * ((K + 1) // 2)
            if e["nbytes"] != want:
                raise ValueError(
                    f"{self.path}: {name} is I4 with {e['nbytes']} bytes, but "
                    f"[{N}, {K}] packed over the first K columns is {want} -- "
                    f"nbytes is what decides where the NEXT tensor starts, so "
                    f"they cannot disagree without every tensor after this one "
                    f"being read from the wrong offset")
            q = unpack_i4(self.payload(name)).reshape(N, -1)
            if q.shape[1] < K:
                raise ValueError(
                    f"{self.path}: {name} unpacks to {q.shape[1]} values per "
                    f"output channel, fewer than the {K} its shape asks for")
            q = q[:, :K]                 # a whole final byte for an odd K
            s = self._sidecar(name, WSCALE_SUFFIX, N, e)
            g = int(self.config.get("conv_int4_group", 0) or 0) or K
            G = (K + g - 1) // g
            gs = self._sidecar(name, GSCALE_SUFFIX, G * N, e).reshape(G, N)
            # [K, N] by repetition, truncated at K, then TRANSPOSED to [N, K]:
            # the payload was packed over the first K columns exactly, so the
            # tail group's last rows of scales have no weights to scale and must
            # not appear as one -- and the transpose is what makes a [K, N]
            # factor broadcast against an [N, K] payload. Sizes are already
            # pinned above (payload bytes, wscale length, gscale length), so
            # every shape here follows from those three plus K itself.
            gk = np.repeat(gs, g, axis=0)[:K, :].T
            x = q.astype(np.float32) * s[:, None] * gk
            return np.ascontiguousarray(x.reshape(shp)).copy()

        raise ValueError(
            f"{self.path}: {name} is {dtype!r}, which no convolution weight "
            f"unpacking here knows -- the container's own dtype tag is the "
            f"source of this, not the caller's, so this is a packer that wrote "
            f"a tag with no reader for it")

    def _sidecar(self, name, suffix, want, e):
        """One scale tensor of a convolution weight, as fp32, of length `want`."""
        # BASE + SUFFIX, and the base is the weight MINUS its `.w`, because that
        # is what both writers produce: conv_quant adds `.wscale`/`.gscale` to
        # `conv.0` (yielding `conv.0.wscale`, not `conv.0.w.wscale`), and the
        # GEMM side works only because its operand names (`layer.0.qkv`) do not
        # end in `.w` at all. Spelling it here rather than in the caller keeps
        # the one place that knows both conventions from being copied a second
        # time.
        base = name[:-len(".w")] if name.endswith(".w") else name
        sname = base + suffix
        if sname not in self.entries:
            raise KeyError(
                f"{self.path}: {name} is {e['dtype']} but {sname} is not in the "
                f"container -- its bytes carry no scale, so they are not a "
                f"weight. Repack with tools/pack/pack_npue.py.")
        s = self.tensor(sname).reshape(-1)
        if s.size != want:
            raise ValueError(
                f"{self.path}: {sname} has {s.size} entries, and {name} asks "
                f"for {want} -- a scale read with the wrong stride multiplies "
                f"the wrong channel and still produces a tensor of the right "
                f"shape")
        if not np.all(np.isfinite(s)) or np.any(s < 0):
            raise ValueError(
                f"{self.path}: {sname} has a negative or non-finite entry; it "
                f"is multiplied into every weight of its slice")
        return s.astype(np.float32)

    def tensor(self, name):
        """The logical tensor: de-tiled and widened to fp32. For verification
        and for the Python encoder -- the C++ runtime uses panel() instead.

        An I8 or I4 operand comes back DEQUANTISED (see `dequant_int8`): the
        panel alone is not a weight, and a caller that got the raw int8 would
        multiply a number 1e-4 of the right size and report a plausible
        embedding. Missing sidecars raise rather than default, because a
        container whose smoothing is silently skipped is the failure this
        format's whole scale/smooth pair exists to make impossible.
        """
        e = self.entries[name]
        if e.get("role") in CONV_W_ROLES:
            return self._conv_weight(name, e)
        x = self.panel(name)
        if e["dtype"] == "BF16":
            x = from_bf16_bits(x)
        lay = e.get("layout")
        if lay and lay.get("kind") == "block_panel":
            K, N = e["padded_shape"]
            x = untile_b(x, K, N, lay["tile_k"], lay["tile_n"],
                         lay.get("mac_s"), lay.get("mac_t"))
            kl, nl = e["logical_shape"]
            x = x[:kl, :nl]
        else:
            x = x.reshape(e["logical_shape"])
        if e["dtype"] in ("I8", "I4"):
            for suffix in (WSCALE_SUFFIX, ASMOOTH_SUFFIX):
                if name + suffix not in self.entries:
                    raise KeyError(
                        f"{self.path}: {name} is {e['dtype']} but "
                        f"{name + suffix} is not in the container -- its bytes "
                        f"carry no scale, so they are not a weight. Repack with "
                        f"tools/pack/pack_npue.py --dtype "
                        f"{'i4' if e['dtype'] == 'I4' else 'i8'}.")
            x = dequant_int8(x, self.tensor(name + WSCALE_SUFFIX),
                             self.tensor(name + ASMOOTH_SUFFIX))
        # copy(), not ascontiguousarray(): the latter can hand back a view into
        # the mapping, which would keep the file locked and give the caller an
        # array that dies when close() is called.
        return np.ascontiguousarray(x).copy()

    def check_layout(self, name, layout):
        """Refuse a file whose layout is not the one the caller expects."""
        e = self.entries[name]
        want = layout_hash(layout)
        got = e.get("layout_hash")
        if got != want:
            raise ValueError(
                f"{name}: layout_hash mismatch -- file has {got}, caller wants "
                f"{want}. Repack with tools/pack/pack_npue.py.")


# --- one naming rule, three callers ----------------------------------------
# reference/make_goldens.py writes them, tools/export/export_validation.py and
# tools/verify/verify_npue.py read them. Three copies of a naming convention is how
# the three pooling implementations started, so it lives here.

# all-MiniLM-L6-v2's goldens predate the derived scheme and are cited by name
# in tasks/0005 and by six scripts under experiments/. Renaming them would
# falsify a task log, so its historical slug is kept.
LEGACY_GOLDEN_SLUGS = {"all-MiniLM-L6-v2": "minilm_l6"}


def golden_slug(model_name, n_layers):
    """Stem of this model's golden files, without the _s<seq>_* suffix."""
    import os
    name = os.path.basename(str(model_name).rstrip("/\\"))
    if name.endswith(".npue"):
        name = name[:-5]
    return LEGACY_GOLDEN_SLUGS.get(name, f"{name.lower()}_l{n_layers}")


def find_goldens(goldens_dir, source_sha256, seq, load):
    """The goldens for a checkpoint, found by CONTENT rather than by name.

    Goldens belong to a checkpoint; a `.npue` is one packing of it. Two
    containers of the same weights at different tile sizes share goldens, and
    deriving the filename from the container's name made that unexpressible --
    `bge-large-n16.npue` went looking for `bge-large-n16_l24_s64_*`.

    `load` is passed in because reference/npz_io is not importable from
    here without dragging reference/ onto the path of every caller.

    Returns (boundary_path, taps_path). Raises if the match is not exactly one:
    zero means the goldens were never generated, and more than one means two
    checkpoints share a sha256, which is not a thing to guess about.
    """
    from pathlib import Path as _P
    gdir = _P(goldens_dir)
    hits = []
    for cand in sorted(gdir.glob(f"*_s{seq}_boundary.npz")):
        try:
            _, meta = load(cand)
        except Exception:
            continue
        if meta.get("source_sha256") == source_sha256:
            hits.append(cand)
    if len(hits) != 1:
        names = ", ".join(h.name for h in hits) if hits else "none"
        raise FileNotFoundError(
            f"{len(hits)} goldens in {gdir} match checkpoint "
            f"{source_sha256[:16]}... at seq {seq} ({names}). Generate them "
            f"with reference/make_goldens.py --model-dir <dir> --taps")
    return hits[0], hits[0].with_name(hits[0].name.replace("_boundary.", "_taps."))
