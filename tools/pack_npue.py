# NpuEmbeddings -- M4: pack a HuggingFace checkpoint into a pre-tiled .npue.
#
# Applies the five offline fusions from docs/04-model. All are pure weight
# rewrites with zero runtime cost, and each removes work from the hot path:
#
#   1. Fuse Q, K, V into one [384, 1152] B matrix + [1152] bias.
#      One GEMM instead of three, and 3x better weight reuse per DMA.
#   2. Transpose everything to [K, N] so the runtime never transposes.
#      nn.Linear stores [out, in]; the transpose is free offline.
#   3. Convert GEMM operands to bf16, but keep LayerNorm gamma/beta and every
#      bias in fp32 -- they are 384 floats each and numerically sensitive.
#   4. Fold 1/sqrt(head_dim) into the Q weight and bias, so the attention
#      kernel has no scale multiply.
#   5. Pre-slice position_embeddings to the real max sequence length (256 for
#      MiniLM, NOT the 512 in config.json -- docs/04-model, sharp edge 1).
#
# And the point of the milestone: PRE-TILE every GEMM operand, which is both
# the main performance lever now that M2 showed we are data-movement bound, and
# the only way to express ffn_down at all (K=1536 > the 1023 DMA BD limit).
#
# Env: iron (numpy only)
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" tools\pack_npue.py
#   ... --tile-n 32 --out models\minilm_n32.npue

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "reference"))

# gemm_b_layout was factored into npue.py so the descriptor has ONE definition
# (its docstring records two copies drifting apart). The import was never added
# here, so pack_npue.py has not run since that refactor -- the shipped .npue
# predates it and nothing repacked. Found while adding the vocabulary, 0036.
from npue import (ARCH_GEMMA3_MQA_ROPE_GEGLU, ARCH_GTE_NEW_ROPE_GEGLU,  # noqa: E402
                  ARCH_NOMIC_ROPE_SWIGLU, Writer,
                  gemm_b_layout, layout_hash, tile_b, to_bf16_bits)
from safetensors_io import load                              # noqa: E402

# From M2's traced results: mac_dims are (r,s,t) = (4,8,8) for plain bf16 on
# npu2 and (8,8,8) with bfp16 emulation. Only s and t affect the B operand
# layout, and BOTH configurations give s=t=8 -- so the pre-tiled B layout is
# the same either way, and the bf16/bfp16 decision does not force a repack.
MAC_S, MAC_T = 8, 8

# tile_n=48, not M2's winning 32. At 8 columns the design requires
# N % (tile_n * n_cols) == 0, and MiniLM's N dims are 384 / 1152 / 1536:
#   gcd(384/8, 1152/8, 1536/8) = 48, so tile_n must divide 48.
# n=32 fails at 8 columns (1152/256 = 4.5) even though it was M2's best at 4.
# 48 is the largest legal choice, needs zero padding on all three shapes, and
# fits L1: 2*(64*64*2 + 64*48*2 + 64*48*4) = 53,248 < 65,536.
DEFAULT_TILE_K, DEFAULT_TILE_N = 64, 48


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pooling(model_dir):
    """Which pooling the checkpoint actually uses, from its own metadata.

    sentence-transformers records this in 1_Pooling/config.json and
    fetch_model.py already downloads it; until now both packers wrote "mean"
    as a literal, which is right for MiniLM and wrong for every bge model.

    Ambiguity REFUSES rather than picking. A model using max or
    mean_sqrt_len pooling is not something this runtime implements, and
    approximating it with mean would be a silent quality loss.
    """
    p = Path(model_dir) / "1_Pooling" / "config.json"
    if not p.exists():
        raise SystemExit(f"{p} not found -- cannot determine pooling mode. "
                         f"Re-fetch with reference/fetch_model.py.")
    c = json.loads(p.read_text(encoding="utf-8"))
    modes = [k for k, v in c.items()
             if k.startswith("pooling_mode_") and v is True]
    if modes == ["pooling_mode_cls_token"]:
        return "cls"
    if modes == ["pooling_mode_mean_tokens"]:
        return "mean"
    raise SystemExit(f"{p}: this runtime implements cls and mean pooling; "
                     f"the checkpoint asks for {modes or 'nothing'}")


def add_gemm_b(w, name, mat, tile_k, tile_n, fold=None):
    """Stage a [K,N] GEMM operand: optional scale fold, bf16, pre-tile."""
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if fold is not None:
        mat = mat * fold
    K, N = mat.shape
    layout = gemm_b_layout(tile_k, tile_n, MAC_S, MAC_T)
    bits = to_bf16_bits(mat)
    flat = tile_b(bits, tile_k, tile_n, MAC_S, MAC_T)
    return w.add(name, flat, "BF16", "gemm_b", [K, N], layout=layout)


def _n_layers(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    return cfg["num_hidden_layers"]


def calibrate_smoothing(model_dir, alpha=0.5, n_texts=128, max_len=64,
                        corpus_path=None, arch="bert"):
    """SmoothQuant factors for every NPU GEMM, from a calibration corpus.

    Returns {(layer, op): s} where `s` is a per-INPUT-channel vector and the
    identity being exploited is

        X @ W  ==  (X / s) @ (diag(s) W)          for any positive s

    with `s_j = max_i|X[i,j]|^alpha / max_n|W[j,n]|^(1-alpha)`. Dividing the
    activation moves range out of the operand that cannot absorb it (per-token
    scaling is shared by every channel of a row, so one outlier channel sets
    the step size for all of them) into the weights, where each column already
    has its own scale.

    **WHY NOT FOLD IT INTO LAYERNORM.** SmoothQuant's own deployment story is
    to fold `1/s` into the preceding LayerNorm's gamma/beta, which is free.
    That works for PRE-LN decoder architectures. BERT is POST-LN: `ln1`'s
    output feeds `ffn_up` *and* the residual added before `ln2`, so scaling
    gamma/beta scales the residual too and silently changes the model
    (tasks/0078 section 4a -- this was written down wrongly first). The factor
    therefore ships as data and the runtime applies it inside the quantisation
    pass, which already reads every element of A to take its row maximum. One
    extra multiply on values already in registers.

    Calibration is BUILD-TIME Python (CLAUDE.md rule 5 allows that; the shipped
    runtime stays C++). It runs the numpy oracle -- the same
    `reference/encoder.py` the goldens come from -- so the statistics describe
    exactly the tensors the array will see.

    KEYED BY CALL SITE, NOT BY SHAPE. All six layers' `qkv` are [384,1152]; a
    shape-keyed maximum collapses them into one and dividing an early layer by
    a late layer's outlier range drove the embedding to nonsense (measured at
    1-cos 0.39-0.52 before this was found, tasks/0078 section 4b).
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO / "reference"))
    # THE ORACLE MUST MATCH THE ARCHITECTURE, or the statistics describe a
    # model the array will never see (tasks/0081). arch=2 has RoPE instead of
    # absolute positions and a gated SwiGLU whose ffn_up is 2*intermediate
    # wide; running it through BERT's forward pass would produce plausible
    # factors for the wrong activations -- exactly the failure mode the
    # "geometry from the checkpoint" note below already guards for depth.
    if arch == "nomic":
        from encoder_nomic import load_reference, fp32_gemm       # noqa: E402
    elif arch == "gemma":
        from encoder_gemma import load_reference, fp32_gemm        # noqa: E402
    else:
        from encoder import load_reference, fp32_gemm             # noqa: E402
    from transformers import AutoTokenizer                       # noqa: E402

    corpus_path = corpus_path or (REPO / "tasks" / "0074-m13-gemma-on-npu"
                                  / "corpus_520.txt")
    texts = [t for t in pathlib_read_lines(corpus_path) if t][:n_texts]
    if len(texts) < 8:
        raise SystemExit(f"{corpus_path}: only {len(texts)} calibration texts")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    enc = tok(texts, padding="max_length", truncation=True,
              max_length=max_len, return_tensors="np")

    # The four projection GEMMs, and ONLY those: attention's QK^T and A.V run
    # on the host in fp32, and `batched_gemm` hands them to the same callback
    # as 2-D slices, so an ndim test does not separate them. Their widest
    # operand dimension is `seq` (64) or `head_dim`; every projection's
    # narrowest is `hidden`.
    on_npu = lambda b: min(b.shape) >= 128
    # THE ORACLE'S GEMM COUNT IS NOT THE PACKER'S, AND THE DIFFERENCE IS FUSION.
    #
    # Each oracle runs the model as the checkpoint stores it; this packer
    # CONCATENATES operands along N so the array sees four GEMMs per layer
    # whatever the architecture. So the two disagree, differently per arch:
    #
    #   arch 0 (BERT)   4 sites -> 4    qkv already fused upstream
    #   arch 2 (nomic)  5 sites -> 4    fc11 (up) + fc12 (gate) -> ffn_up
    #   arch 1 (Gemma)  7 sites -> 4    q+k+v(+pad) -> qkv, gate+up -> ffn_up
    #
    # Left unhandled the keys shift and every factor from that point on lands
    # on the NEXT tensor -- it packs cleanly and is wrong from layer 0, the
    # same class of bug as tasks/0078 4b's 400x blowup, which was caught only
    # because the number was too implausible to accept. Hence the assert below.
    #
    # Merging is exact: the members of a fused group all consume the SAME
    # activation, so `amax` is identical across them, and the fused operand's
    # per-input-channel weight maximum is the elementwise max of the members'.
    # Taking a max also makes the result independent of visit order.
    if arch == "gemma":
        OPS = ["qkv", "qkv.k", "qkv.v", "attn_out",
               "ffn_up", "ffn_up.up", "ffn_down"]
    elif arch == "nomic":
        OPS = ["qkv", "attn_out", "ffn_up", "ffn_up.gate", "ffn_down"]
    else:
        OPS = ["qkv", "attn_out", "ffn_up", "ffn_down"]
    MERGE = {o: o.split(".")[0] for o in OPS if "." in o}

    n_sites = len(OPS) * _n_layers(model_dir)
    amax, wmax, ctr = {}, {}, [0]
    def collect(a, b):
        if not on_npu(b):
            return fp32_gemm(a, b)
        site = ctr[0]; ctr[0] += 1
        # Anything past the layer stack is a POST-POOL head -- Gemma's dense2
        # and dense3, which the packer deliberately keeps on the host because
        # they run once per sequence rather than once per token (tasks/0074
        # sec 4, ~1% of array time against two more dispatches of fixed cost).
        # They are wide enough to pass the on_npu filter, so they must be
        # excluded by POSITION; they are last in encode order.
        if site >= n_sites:
            return fp32_gemm(a, b)
        key = (site // len(OPS), OPS[site % len(OPS)])
        amax[key] = np.maximum(amax.get(key, 0.0), np.abs(a).max(0))
        wmax[key] = np.abs(b).max(1)
        return fp32_gemm(a, b)

    # GEOMETRY FROM THE CHECKPOINT, not from load_reference's defaults, which
    # are MiniLM's (6 layers, 12 heads). A 24-layer model calibrated as a
    # 6-layer one produces factors for a third of its GEMMs and silently
    # nothing for the rest -- the kind of wrong that packs cleanly.
    _cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    if arch in ("nomic", "gemma"):
        # Both read their own geometry from config.json and take no
        # token_type_ids -- neither architecture has a token-type embedding.
        ref = load_reference(str(model_dir))
        ref.gemm = collect
        ref.encode(enc["input_ids"], enc["attention_mask"])
    else:
        ref = load_reference(str(model_dir),
                             num_layers=_cfg["num_hidden_layers"],
                             num_heads=_cfg["num_attention_heads"],
                             eps=_cfg.get("layer_norm_eps", 1e-12))
        ref.gemm = collect
        ref.encode(enc["input_ids"], enc["attention_mask"],
                   np.zeros_like(enc["input_ids"]))

    # The call-site counter above assumes exactly four NPU GEMMs per layer in
    # (qkv, attn_out, ffn_up, ffn_down) order. If the oracle disagrees, the
    # keys are shifted and every factor lands on the wrong tensor -- packing
    # cleanly, as tasks/0078 4b's 400x blowup did. Assert instead of trusting.
    if ctr[0] < n_sites:
        raise SystemExit(
            f"calibration saw {ctr[0]} NPU GEMM call sites, expected at least "
            f"{n_sites} ({len(OPS)} x {_cfg['num_hidden_layers']} layers) -- "
            f"the oracle's GEMM order does not match this packer's")

    # Fold each fused group's members back into the operand the array receives.
    for i in range(_cfg["num_hidden_layers"]):
        for member, target in MERGE.items():
            if (i, member) not in wmax:
                continue
            wmax[(i, target)] = np.maximum(wmax[(i, target)],
                                           wmax.pop((i, member)))
            amax[(i, target)] = np.maximum(amax[(i, target)],
                                           amax.pop((i, member)))

    out = {}
    for key, aj in amax.items():
        wj = wmax[key]
        s = (np.maximum(aj, 1e-8) ** alpha) / (np.maximum(wj, 1e-8) ** (1 - alpha))
        out[key] = np.where(s > 0, s, 1.0).astype(np.float32)
    print(f"  calibrated on {len(texts)} texts, {ctr[0]} GEMM call sites, "
          f"alpha={alpha}")
    return out


def pathlib_read_lines(p):
    return Path(p).read_text(encoding="utf-8").split("\n")


def add_gemm_b_int8(w, name, mat, tile_k, tile_n, fold=None, asmooth=None):
    """Stage a [K,N] GEMM operand as INT8, per-output-channel symmetric.

    THE SCHEME, and why this one (tasks/0078).

        s[j]     = max_k |W[k,j]| / 127          one scale per output column
        Wq[k,j]  = round(W[k,j] / s[j])          clipped to [-127, 127]

    and at runtime, with A quantised per ROW (per token) by the same rule,

        Y[i,j] = int32_dot(Aq[i,:], Wq[:,j]) * sa[i] * s[j] + bias[j]

    so dequantisation is a rank-1 outer-product scaling of the int32 result --
    one multiply per output element, folded into the pass that already reads C
    and adds the bias.

    **Per CHANNEL, not per tensor.** 2209.13325 documents the +-50-100 outlier
    dimensions in post-LN BERT that make per-tensor quantisation lose real
    accuracy; a per-column scale costs N floats per operand and removes the
    coupling between an outlier column and every other column. Per-ROW
    activation scales do the same job on the other operand, which is why this
    scheme needs no calibration set at all -- both scales are computed from the
    data in front of it.

    -127 not -128: the symmetric range is kept symmetric so that negating a
    weight cannot saturate differently from negating its opposite. It costs one
    representable value and removes an asymmetry that is invisible until it is
    not.

    THE ACCUMULATOR IS EXACT. int8 x int8 -> int32 has no rounding anywhere in
    the reduction (2608.13756's "integer alibi"), so every bit of error in this
    path is in the two roundings above and in the scale multiply -- NOT in the
    K-reduction, unlike bf16. Overflow is the one thing that would break that,
    and it is asserted rather than assumed.
    """
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    if fold is not None:
        mat = mat * fold
    if asmooth is not None:
        # The weight half of the SmoothQuant identity: X @ W == (X/s) @ (sW).
        # Applied BEFORE quantisation so the columns being measured are the
        # ones the array will multiply.
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
    scale = np.abs(mat).max(axis=0) / 127.0          # [N]
    # A genuinely all-zero column would divide by zero. Keep its scale at 1 so
    # the column stays exactly zero rather than becoming NaN.
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    q = np.rint(mat / scale[None, :]).clip(-127, 127).astype(np.int8)

    layout = gemm_b_layout(tile_k, tile_n, MAC_S, MAC_T, dtype="I8")
    flat = tile_b(q, tile_k, tile_n, MAC_S, MAC_T)
    w.add(name, flat, "I8", "gemm_b", [K, N], layout=layout)
    w.add(name + ".wscale", scale, "F32", "quant_scale", [N])
    # Ships even when it is all ones, so the runtime has ONE code path and
    # cannot be handed a container whose smoothing it silently skips.
    w.add(name + ".asmooth",
          np.ones(K, np.float32) if asmooth is None else asmooth,
          "F32", "quant_smooth", [K])
    # Report what the quantisation actually cost on THESE weights, so a bad
    # tensor is visible at pack time rather than at MTEB time.
    deq = q.astype(np.float32) * scale[None, :]
    den = float(np.linalg.norm(mat))
    return float(np.linalg.norm(deq - mat) / den) if den else 0.0


def add_gemm_b_host(w, name, mat):
    """Stage a [K,N] GEMM operand PLAIN: F32, row-major, no tiling.

    Gemma has no NPU kernel yet (tasks/0064-m12-embeddinggemma-arch1-integration): every GEMM in this arch runs on
    the HOST, so there is no DMA descriptor to pre-tile for and no
    layout_hash to check -- `layout=None` (Writer.add's default) is exactly
    that: an entry with no "layout" key, which npue.py's Reader.tensor() and
    the C++ loader both already treat as "not tiled, read [K,N] row-major".
    F32, not bf16: this checkpoint's own weights are F32 on disk (verified in
    tasks/0055 by direct safetensors inspection, contradicting an earlier
    assumption that it shipped bf16 -- see this task's TASK.md), and the
    point of this task is a correctness gate, not a size/speed one; a bf16
    weight path is future work the container format does not block.
    """
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    K, N = mat.shape
    return w.add(name, mat.reshape(-1), "F32", "gemm_b_host", [K, N])


def gemma_qkv_blocks(hidden, head_dim, kv_heads, tile_n, n_cols=8):
    """Where Q, K and V sit inside the padded fused qkv operand, and how wide it is.

    THE GEOMETRIC TRICK THIS MODEL NEEDED (tasks/0074). MQA gives
    `num_key_value_heads = 1`, so K and V are each `head_dim` = 256 wide, and
    256 caps `gcd(N / n_cols)` at 32 across the whole N-set no matter how the
    packer fuses (tasks/0055 checked all four fusion strategies and found the
    same floor in every one). At 8 columns that forces `tile_n` down to 16 or
    32 where every other model this project ships runs at 48, and the cost is
    ~3x the GEMM iterations, per T1's own model.

    Zero-padding the fused operand's N axis removes the floor outright:
    `C = A @ B` with zero columns of B gives exactly-zero columns of C, so the
    host slices Q/K/V off the front and ignores the tail. EXACT -- there is no
    accuracy question to answer, unlike every other lever on this path.

    Padding goes at the END, not between the blocks: Q, K and V keep their
    natural contiguous offsets, so nothing downstream has to know the padding
    exists except the code that sizes the buffer.

    Returned as DATA and written into the container, because the runtime must
    not derive it. `design_fits()` derived qkv's width as `3 * hidden` -- true
    of BERT and nomic, false here (1536, not 2304) -- which is thread T31's
    fail-open one field to the left.
    """
    q_w = hidden                       # num_heads * head_dim
    kv_w = kv_heads * head_dim
    used = q_w + 2 * kv_w
    gran = tile_n * n_cols             # the design's `N % (n * n_aie_cols) == 0`
    padded = ((used + gran - 1) // gran) * gran
    return {
        "n": padded, "used": used, "pad": padded - used,
        "q": [0, q_w], "k": [q_w, q_w + kv_w], "v": [q_w + kv_w, used],
    }


def pack_gemma(model_dir, out, source_repo_override=None, tile_k=None,
               tile_n=None, host_only=False,
               int8=False, smooth_alpha=0.5, smooth_texts=128):
    """Pack an EmbeddingGemma-300M-shaped checkpoint (arch=1).

    Deliberately NOT the BERT path above, reused only via helpers (Writer,
    sha256) -- see tasks/0064-m12-embeddinggemma-arch1-integration/TASK.md for
    why: 4 RMSNorms/layer (not 2 LayerNorms), MQA (num_key_value_heads=1,
    NOT hidden/num_heads), q_norm/k_norm, per-layer RoPE base, separate
    gate/up GeGLU matrices, two post-pool Dense heads, no biases anywhere
    (attention_bias=false, both Dense heads bias=false), no
    token_type_embeddings, no absolute position table (RoPE instead).
    Every architectural fact below is read from reference/encoder_gemma.py
    (tasks/0055, validated 1-cos 1.065e-07 against real HuggingFace) rather
    than re-derived.

    TWO MODES (tasks/0074).

    `host_only=True` reproduces exactly what tasks/0064-0065 shipped: every
    GEMM operand PLAIN (F32, row-major, untiled), for the CPU-only
    `GemmaEncoder`. Kept because that path is verified to 1-cos 5.496e-13 and
    is now this model's discriminating CONTROL, not dead code.

    The default emits the four per-layer GEMM operands PRE-TILED in bf16 under
    BERT's tensor names (`layer.i.qkv` / `.attn_out` / `.ffn_up` / `.ffn_down`,
    each with a zero `.bias`), so the NPU dispatch path -- staging, the layout
    hash check, `Design::stage`, `Encoder::gemm`'s bias add -- works on this
    container unchanged. Same trick and same reason as `pack_nomic` above, with
    two Gemma-specific twists:

      * `qkv` is Q|K|V fused AND zero-padded to a legal N -- see
        gemma_qkv_blocks().
      * `ffn_up` is gate|up fused along N (GeGLU), so the array sees four GEMMs
        per layer rather than five. `config["geglu_halves"]` pins the order;
        the runtime asserts it instead of trusting this constant, exactly as
        `swiglu_halves` does for nomic (tasks/0068 Q2 measured the swapped
        variant at rel_fro 4.022e+00).

    Everything the host still computes -- the four RMSNorms per layer,
    q_norm/k_norm, the embedding table, the final norm and the two post-pool
    Dense heads -- stays F32 and keeps its GEMMA names, because those are the
    tensors whose *placement* differs from BERT and renaming them would make
    two different architectures look alike in the one file a reader checks.

    THE FOLD THAT MUST NOT HAPPEN. Every other model here folds
    1/sqrt(head_dim) into the packed Q block. For Gemma that fold is silently
    ANNIHILATED: `q_norm` is an RMSNorm applied to q after the projection, and
    RMSNorm is scale-invariant -- (s*q)/rms(s*q) == q/rms(q), exactly. The
    scale would vanish with no shape error to notice it. It is therefore
    applied at its reference position (on the attention scores) and asserted
    NOT folded in the container's `fusions` block.
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    src, _ = load(model_dir / "model.safetensors")
    src_sha = sha256(model_dir / "model.safetensors")
    d2, _ = load(model_dir / "2_Dense" / "model.safetensors")
    d3, _ = load(model_dir / "3_Dense" / "model.safetensors")

    L = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    inter = cfg["intermediate_size"]
    swp = cfg.get("_sliding_window_pattern", 6)

    tile_k = DEFAULT_TILE_K if tile_k is None else tile_k
    tile_n = DEFAULT_TILE_N if tile_n is None else tile_n
    qkv = gemma_qkv_blocks(hidden, head_dim, kv_heads, tile_n)

    # ASSERT the geometry the design will demand, here, where the operand is
    # built -- not in the exporter, where a failure costs a compile, and not at
    # dispatch, where it costs a wrong answer. These are gemm_pretiled.py's own
    # `_build_design` assertions (lines 126-129) restated over THIS model's
    # shapes. `hidden` (attn_out N, ffn_down N), 2*inter (ffn_up N) and the
    # padded qkv N must every one of them tile across the columns.
    if not host_only:
        for nm, K, N in (("qkv", hidden, qkv["n"]),
                         ("attn_out", hidden, hidden),
                         ("ffn_up", hidden, 2 * inter),
                         ("ffn_down", inter, hidden)):
            if K % tile_k:
                raise SystemExit(f"{nm}: K={K} does not divide by tile_k={tile_k}")
            if N % (tile_n * 8):
                raise SystemExit(
                    f"{nm}: N={N} does not tile across 8 columns at "
                    f"tile_n={tile_n} (needs a multiple of {tile_n * 8}) -- "
                    f"this is the constraint gemma_qkv_blocks() pads qkv to "
                    f"meet, and it does not hold for this shape")

    mode = "HOST-only GEMMs" if host_only else f"NPU, tile ({tile_k}, {tile_n})"
    print(f"packing {model_dir.name} -> {Path(out).name}  (arch=gemma3, {mode})")
    print(f"  hidden={hidden} heads={heads} kv_heads={kv_heads} head_dim={head_dim} "
          f"layers={L} inter={inter}")
    if not host_only:
        print(f"  qkv fused+padded: N={qkv['n']} "
              f"(q{qkv['q']} k{qkv['k']} v{qkv['v']}, {qkv['pad']} zero cols) "
              f"-- {qkv['pad'] / qkv['n'] * 100:.1f}% of that one GEMM")
        print(f"  ffn_up fused: N={2 * inter} (gate|up), ffn_down K={inter}")

    source_repo = source_repo_override
    if not source_repo:
        ckpt = model_dir / "CHECKPOINT.json"
        if not ckpt.exists():
            raise SystemExit(f"{ckpt} not found and no source_repo given -- "
                             f"refusing to guess which repository these weights "
                             f"came from")
        source_repo = json.loads(ckpt.read_text(encoding="utf-8"))["repo_id"]

    config = {
        "arch": "gemma3_mqa_rope_geglu",
        "a_dtype": "i8" if (int8 and not host_only) else "bf16",
        "model_type": cfg["model_type"],
        "source_repo": source_repo,
        "source_sha256": src_sha,
        "num_layers": L, "hidden": hidden,
        "num_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "intermediate": inter,
        "dense_hidden": int(d2["linear.weight"].shape[0]),
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg["rope_theta"],
        "rope_local_base_freq": cfg["rope_local_base_freq"],
        "sliding_window": cfg["sliding_window"],
        "sliding_window_pattern": swp,
        "query_pre_attn_scalar": cfg["query_pre_attn_scalar"],
        "vocab_size": cfg["vocab_size"],
        # Informational only: unlike BERT's absolute position TABLE (which
        # really is sliced to this many rows), Gemma has no position
        # embedding to pre-slice -- RoPE tables are computed at runtime for
        # whatever sequence length is asked for. This is the HF config's own
        # `max_position_embeddings`, a ceiling, not a packed array size.
        "max_seq_len": cfg["max_position_embeddings"],
        "pooling": "mean_include_prompt", "l2_normalize": True,
        "activation": "gelu_pytorch_tanh",
        "attention_bias": False, "dense_bias": False,
        "not_implemented": [
            "sliding-window mask (exact for seq_len<=512, see "
            "reference/encoder_gemma.py's file header)",
        ],
    }

    # THE TASK-PREFIX TABLE, and why it has to be in the container (0075).
    #
    # EmbeddingGemma's prefixes already ride inside gemma_tokenizer.bin, which
    # is what the runtime tokenizes with -- so for ordinary encoding this table
    # is redundant. It is here for the MEASUREMENT harness:
    # experiments/m8-npu-vs-cpu/run_mteb.py reads the prefix ONCE from the
    # container and applies it to BOTH sides, so the two agree by construction
    # rather than by two literals that can drift. Without it the NPU side would
    # apply its own default inside the exe while the sentence-transformers side
    # applied nothing, and the measured MTEB delta would be the prefix rather
    # than the datapath -- which is the one thing that comparison exists to
    # rule out.
    #
    # Read verbatim from the checkpoint's own config_sentence_transformers
    # .json, but SORTED BY KEY. json_min.hpp (the C++ mirror's parser) is
    # explicit that it does not preserve object key order -- it stores objects
    # in an unordered_map because tokenizer.json's 262k-entry vocab makes
    # anything else expensive -- so a source-order emission could not be
    # mirrored byte for byte. Order is not semantic in a JSON object; content
    # is, and the content is verbatim.
    cst_path = model_dir / "config_sentence_transformers.json"
    if not cst_path.exists():
        raise SystemExit(f"{cst_path} not found -- it carries this model's own "
                         f"task-prefix table; refusing to pack without it "
                         f"rather than inventing prefixes")
    cst = json.loads(cst_path.read_text(encoding="utf-8"))
    prompts = cst.get("prompts") or {}
    if not prompts:
        raise SystemExit(f"{cst_path} has no 'prompts' table")
    config["prompts"] = {k: prompts[k] for k in sorted(prompts)}
    # THIS PROJECT'S choice, not the checkpoint's: its own
    # `default_prompt_name` is null (sentence-transformers applies no prefix
    # unless one is named). "document" is the same default tasks/0061 picked
    # for the tokenizer table, and the two must agree.
    config["prompt_default"] = "document"
    if "document" not in config["prompts"]:
        raise SystemExit("this checkpoint's prompts table has no 'document' "
                         "row -- the project default would name a key that "
                         "does not exist")

    # What the NPU path needs to know and must never re-derive (tasks/0074).
    # `gemm_layout` is the discriminator: "host" containers carry plain F32
    # operands, "pretiled_bf16" ones carry tiled bf16 under BERT names. A
    # runtime that guessed would read the right number of bytes in the wrong
    # order -- tasks/0022's rel_fro 1.186, "a buffer-size check catches a wrong
    # size, never a wrong layout".
    config["gemm_layout"] = "host" if host_only else "pretiled_bf16"
    if not host_only:
        config.update({
            "tile_k": tile_k, "tile_n": tile_n,
            "mac_s": MAC_S, "mac_t": MAC_T,
            "gated_ffn": True,
            "geglu_halves": "gate|up",
            # qkv's width is DATA. It is 1536 here and 3*hidden nowhere.
            "qkv_n": qkv["n"],
            "qkv_blocks": {"q": qkv["q"], "k": qkv["k"], "v": qkv["v"],
                           "pad": [qkv["used"], qkv["n"]]},
            "fusions": {
                "qkv_fused": True,
                "qkv_zero_padded_to_tile": True,
                "transposed_to_kn": True,
                # Stated as FALSE on purpose -- see this function's docstring.
                # q_norm is scale-invariant, so this fold is not merely
                # skipped, it is illegal.
                "qk_scale_folded_into_q": False,
                "qk_scale_folded_into_q_note":
                    "ILLEGAL for this architecture: q_norm (RMSNorm) runs "
                    "after q_proj and is scale-invariant, so a fold into Wq "
                    "would be annihilated and attention would run unscaled "
                    "with no shape error. The scale stays on the scores.",
                "gemm_operands_bf16": True,
                "norms_embeddings_dense_fp32": True,
                "gated_ffn_fused_gate_up": True,
                "biases_zero_filled": True,
            },
        })

    w = Writer(config, arch=ARCH_GEMMA3_MQA_ROPE_GEGLU)

    w.add("embed_tokens.weight", src["embed_tokens.weight"],
          "F32", "embedding", [cfg["vocab_size"], hidden])
    w.add("norm.weight", src["norm.weight"], "F32", "layernorm", [hidden])

    tok_path = model_dir / "gemma_tokenizer.bin"
    if tok_path.exists():
        tb = np.frombuffer(tok_path.read_bytes(), dtype=np.uint8)
        w.add("tokenizer.gemma_table", tb, "U8", "tokenizer", [int(tb.size)])
        print(f"  tokenizer.gemma_table  {tb.size / 1e6:.2f} MB")
    else:
        print(f"  WARNING: {tok_path} not found (generate it with the C++ "
              f"packer, `npuembed --prepare-model`) -- .npue will have no "
              f"tokenizer table")

    # ONE emitter for the four per-layer operands, as in the BERT and nomic
    # paths. Gemma's calibration runs reference/encoder_gemma.py -- arch=1 is
    # RMSNorm x4, MQA and GeGLU, and BERT's forward pass would describe
    # activations the array never sees (tasks/0081).
    #
    # NOTE the padded qkv: gemma_qkv_blocks() appends genuinely all-zero
    # columns to reach a legal tile_n, and add_gemm_b_int8 keeps their scale at
    # 1 rather than dividing by zero -- so a padded column stays exactly zero
    # through quantisation, which is what the host slicing by offset assumes.
    smooth = (calibrate_smoothing(model_dir, alpha=smooth_alpha,
                                  n_texts=smooth_texts, arch="gemma")
              if int8 and not host_only and smooth_alpha > 0 else {})
    qerr = []

    def emit(name, mat, layer, op):
        if not int8 or host_only:
            add_gemm_b(w, name, mat, tile_k, tile_n)
            return
        qerr.append((name, add_gemm_b_int8(w, name, mat, tile_k, tile_n,
                                           asmooth=smooth.get((layer, op)))))
    n_tiled = 0
    for i in range(L):
        p = f"layers.{i}."
        sa = p + "self_attn."

        if host_only:
            add_gemm_b_host(w, f"layer.{i}.q_proj",
                             np.ascontiguousarray(src[sa + "q_proj.weight"].T))
            add_gemm_b_host(w, f"layer.{i}.k_proj",
                             np.ascontiguousarray(src[sa + "k_proj.weight"].T))
            add_gemm_b_host(w, f"layer.{i}.v_proj",
                             np.ascontiguousarray(src[sa + "v_proj.weight"].T))
        else:
            # [Wq (768) | Wk (256) | Wv (256) | zeros (256)] -> [768, 1536].
            # np.zeros, so the padding is EXACTLY zero rather than whatever
            # an uninitialised allocation held: C's padded columns are then
            # exactly zero and the host can slice by offset without masking.
            wq = np.ascontiguousarray(src[sa + "q_proj.weight"].T)
            wk = np.ascontiguousarray(src[sa + "k_proj.weight"].T)
            wv = np.ascontiguousarray(src[sa + "v_proj.weight"].T)
            pad = np.zeros((hidden, qkv["pad"]), dtype=np.float32)
            fused = np.concatenate([wq, wk, wv, pad], axis=1)
            if fused.shape != (hidden, qkv["n"]):
                raise SystemExit(f"fused qkv is {fused.shape}, expected "
                                 f"{(hidden, qkv['n'])}")
            emit(f"layer.{i}.qkv", fused, i, "qkv")
            w.add(f"layer.{i}.qkv.bias",
                  np.zeros(qkv["n"], dtype=np.float32), "F32", "bias",
                  [qkv["n"]])
            n_tiled += 1

        w.add(f"layer.{i}.q_norm.weight", src[sa + "q_norm.weight"],
              "F32", "layernorm", [head_dim])
        w.add(f"layer.{i}.k_norm.weight", src[sa + "k_norm.weight"],
              "F32", "layernorm", [head_dim])

        if host_only:
            add_gemm_b_host(w, f"layer.{i}.o_proj",
                             np.ascontiguousarray(src[sa + "o_proj.weight"].T))
        else:
            emit(f"layer.{i}.attn_out",
                 np.ascontiguousarray(src[sa + "o_proj.weight"].T),
                 i, "attn_out")
            w.add(f"layer.{i}.attn_out.bias",
                  np.zeros(hidden, dtype=np.float32), "F32", "bias", [hidden])
            n_tiled += 1

        w.add(f"layer.{i}.input_layernorm.weight",
              src[p + "input_layernorm.weight"], "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.post_attention_layernorm.weight",
              src[p + "post_attention_layernorm.weight"], "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.pre_feedforward_layernorm.weight",
              src[p + "pre_feedforward_layernorm.weight"], "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.post_feedforward_layernorm.weight",
              src[p + "post_feedforward_layernorm.weight"], "F32", "layernorm", [hidden])

        mp = p + "mlp."
        if host_only:
            add_gemm_b_host(w, f"layer.{i}.gate_proj",
                             np.ascontiguousarray(src[mp + "gate_proj.weight"].T))
            add_gemm_b_host(w, f"layer.{i}.up_proj",
                             np.ascontiguousarray(src[mp + "up_proj.weight"].T))
            add_gemm_b_host(w, f"layer.{i}.down_proj",
                             np.ascontiguousarray(src[mp + "down_proj.weight"].T))
        else:
            # config["geglu_halves"] == "gate|up": lo = cols [0, inter) gets
            # the GELU, hi = cols [inter, 2*inter) does not. ONE GEMM.
            gate = np.ascontiguousarray(src[mp + "gate_proj.weight"].T)
            up = np.ascontiguousarray(src[mp + "up_proj.weight"].T)
            emit(f"layer.{i}.ffn_up",
                 np.concatenate([gate, up], axis=1), i, "ffn_up")
            w.add(f"layer.{i}.ffn_up.bias",
                  np.zeros(2 * inter, dtype=np.float32), "F32", "bias",
                  [2 * inter])
            emit(f"layer.{i}.ffn_down",
                 np.ascontiguousarray(src[mp + "down_proj.weight"].T),
                 i, "ffn_down")
            w.add(f"layer.{i}.ffn_down.bias",
                  np.zeros(hidden, dtype=np.float32), "F32", "bias", [hidden])
            n_tiled += 2

    # The two post-pool Dense heads stay on the HOST in both modes. They run
    # once per SEQUENCE rather than once per token -- 604 MFLOP per batch-128
    # encode against ~700 ms of modelled array time, ~1% (tasks/0074 sec 4) --
    # so two more dispatches would cost more fixed overhead than they save.
    add_gemm_b_host(w, "dense2.weight", np.ascontiguousarray(d2["linear.weight"].T))
    add_gemm_b_host(w, "dense3.weight", np.ascontiguousarray(d3["linear.weight"].T))

    info = w.write(out)
    total = Path(out).stat().st_size
    if not host_only:
        print(f"\n  {'operand':<14} {'[K,N]':>12} {'k-blocks':>9} {'n-blocks':>9} "
              f"{'iters/core':>11}")
        # iters/core is T1/0048's own quantity restated per shape at the
        # production tier, so the packer prints the number the cost model
        # consumes rather than one a reader has to re-derive:
        #   (K/k) * (M/(m*rows)) * (N/(n*cols)),  M = 128*64, m = 64, rows = 4
        for nm, K, N in (("qkv", hidden, qkv["n"]),
                         ("attn_out", hidden, hidden),
                         ("ffn_up", hidden, 2 * inter),
                         ("ffn_down", inter, hidden)):
            kb, nb = K // tile_k, N // tile_n
            iters = (K // tile_k) * (8192 // (64 * 4)) * (N // (tile_n * 8))
            flag = "" if max(kb, nb) < 1024 else "  <-- OVER 1023"
            print(f"  {nm:<14} {str([K, N]):>12} {kb:>9} {nb:>9} "
                  f"{iters:>11}{flag}")
    print(f"\n  tensors    : {len(w.entries)}"
          + ("" if host_only else f"  ({n_tiled} pre-tiled GEMM operands)"))
    print(f"  json       : {info['json_length']} B at {info['json_offset']}")
    print(f"  data       : {info['data_length']/1e6:.2f} MB at {info['data_offset']}")
    print(f"  file       : {total/1e6:.2f} MB")
    print(f"  source     : {src_sha[:16]}...")
    if not host_only:
        print(f"  layout_hash: "
              f"{layout_hash(gemm_b_layout(tile_k, tile_n, MAC_S, MAC_T, dtype='I8' if (int8 and not host_only) else 'BF16'))[:16]}..."
              f"{'  (i8 operands)' if (int8 and not host_only) else ''}")
    return 0


def pack_nomic(model_dir, out, tile_k, tile_n, max_seq, fold_scale,
               int8=False, smooth_alpha=0.5, smooth_texts=128):
    """Pack a nomic-embed-text-v1.5-shaped checkpoint (arch=2).

    Emits the SAME tensor names and the SAME emission order as the BERT
    (arch=0) path above -- tasks/0069-m13-nomic-arch2-container/TASK.md item
    3 -- so Encoder::stage_all() and the whole NPU dispatch path work
    UNCHANGED. Every architectural fact asserted below (post-LN, SiLU on
    fc12 not fc11, no mlp.norm, RoPE NeoX-style on Q/K only starting at
    position 0, theta=1000, three-major Wqkv row order, no biases anywhere)
    was settled EMPIRICALLY against the real checkpoint in tasks/0068 (see
    its TASK.md sec 5), not re-derived here -- this function only implements
    that already-settled architecture and asserts the config facts it
    depends on, so a checkpoint that silently changed underneath it would
    refuse to pack rather than pack wrong.

    Departures from BERT:
      * no absolute position table (RoPE instead) -- zero-filled placeholder
      * no biases anywhere (qkv_proj_bias / mlp_fc1_bias / mlp_fc2_bias all
        False) -- zero-filled placeholders, same rationale
      * gated SwiGLU FFN: fc11 (untouched up-path) and fc12 (SiLU gate) are
        fused into ONE [hidden, 2*intermediate] ffn_up along the N axis, so
        the array still sees four GEMMs per layer, not five --
        out = fc11(x) * silu(fc12(x)) = lo * silu(hi)
      * 1/sqrt(head_dim) is folded into the Q block exactly as the BERT path
        does. This is legal here only because RoPE is a rotation and
        therefore LINEAR in q: rope(s*q) = s*rope(q), so folding the scale
        before the GEMM and before RoPE is exact. tools/verify_npue_nomic.py
        check E proves this numerically rather than assuming it.
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    src, _ = load(model_dir / "model.safetensors")
    src_sha = sha256(model_dir / "model.safetensors")

    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    hidden = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    inter = cfg["intermediate_size"]
    scale = 1.0 / math.sqrt(head_dim)

    if hidden != H * head_dim:
        raise SystemExit(f"hidden={hidden} != num_heads={H} * head_dim={head_dim}")
    if head_dim % 2:
        raise SystemExit(f"head_dim={head_dim} is odd -- RoPE cannot "
                         f"half-split it into rotation pairs")

    # rope_theta: ASSERT, never default. tasks/0068 measured a wrong theta
    # at rel_fro 9.2e-02 on the attention output -- the ONE wrong reading in
    # that whole probe subtle enough to slip past a loose gate (every other
    # wrong reading there was 0.5-5.0).
    theta = cfg["rotary_emb_base"]
    if theta != 1000:
        raise SystemExit(f"rotary_emb_base={theta}, expected 1000 -- "
                         f"refusing to pack against an unverified RoPE base")

    # layer_norm_epsilon and layer_norm_eps are two keys for the same value
    # in this checkpoint's config.json -- read one, assert they agree rather
    # than silently picking one and hoping.
    eps_a, eps_b = cfg["layer_norm_epsilon"], cfg["layer_norm_eps"]
    if eps_a != eps_b:
        raise SystemExit(f"layer_norm_epsilon ({eps_a}) != layer_norm_eps "
                         f"({eps_b}) -- checkpoint is internally inconsistent")
    eps = eps_a

    for flag in ("qkv_proj_bias", "mlp_fc1_bias", "mlp_fc2_bias"):
        if cfg[flag] is not False:
            raise SystemExit(f"{flag}={cfg[flag]!r}, expected False -- this "
                             f"packer zero-fills every bias on the assumption "
                             f"nomic has none; a checkpoint with real biases "
                             f"would be packed WRONG")
    if cfg["prenorm"] is not False:
        raise SystemExit(f"prenorm={cfg['prenorm']!r}, expected False "
                         f"(post-LN block order -- tasks/0068 Q1)")
    if cfg["activation_function"] != "swiglu" or cfg["hidden_act"] != "silu":
        raise SystemExit(f"activation_function={cfg['activation_function']!r} "
                         f"hidden_act={cfg['hidden_act']!r}, expected "
                         f"'swiglu'/'silu'")
    if cfg["rotary_emb_interleaved"] is not False:
        raise SystemExit(f"rotary_emb_interleaved={cfg['rotary_emb_interleaved']!r} "
                         f"-- this packer/runtime assumes NeoX-style RoPE "
                         f"(concat(freqs,freqs), rotate-half) -- tasks/0068")
    if cfg["rotary_emb_fraction"] != 1.0:
        raise SystemExit(f"rotary_emb_fraction={cfg['rotary_emb_fraction']}, "
                         f"expected 1.0 (whole head rotated)")

    print(f"packing {model_dir.name} -> {Path(out).name}  (arch=nomic_bert_rope_swiglu)")
    print(f"  hidden={hidden} heads={H} head_dim={head_dim} layers={L} "
          f"inter={inter} rope_theta={theta}")
    print(f"  tile ({tile_k}, {tile_n}), mac (s={MAC_S}, t={MAC_T}), "
          f"1/sqrt({head_dim}) = {scale:.17g}"
          f"{' folded into Q' if fold_scale else ' NOT folded'}")

    # This project's OWN choice, not the checkpoint's -- labelled as such in
    # the container for the same reason the Gemma table generator
    # (runtime/src/tokenizers/gemma_tokenizer_gen.cpp) labels its table:
    # config_sentence_transformers.json for this checkpoint carries no
    # "prompts" dict at all (verified tasks/0068 sec 4/10), so presenting
    # this table as read-from-the-checkpoint would be a lie in a file other
    # tools read. Prefix strings and token costs measured in tasks/0068 sec 4.
    # How many embedding rows the tokenizer can actually reach. nomic pads
    # vocab_size up to a multiple of 64 (pad_vocab_size_multiple), and those
    # extra rows are NOT zero -- ordinary trained-looking values that no token
    # id can ever select (tasks/0068 sec 5c). Counted, not assumed.
    n_reachable = (len((model_dir / "vocab.txt").read_bytes()
                       .decode("utf-8").splitlines())
                   if (model_dir / "vocab.txt").exists() else 0)

    prompts = {
        "search_document": "search_document: ",
        "search_query": "search_query: ",
        "clustering": "clustering: ",
        "classification": "classification: ",
    }

    config = {
        "arch": "nomic_bert_rope_swiglu",
        "a_dtype": "i8" if int8 else "bf16",
        "model_type": cfg["model_type"],
        "source_repo": json.loads(
            (model_dir / "CHECKPOINT.json").read_text(encoding="utf-8"))["repo_id"],
        "source_sha256": src_sha,
        "num_layers": L, "num_heads": H, "hidden": hidden, "head_dim": head_dim,
        "intermediate": inter,
        "layer_norm_eps": eps,
        "vocab_size": cfg["vocab_size"],
        "max_seq_len": max_seq,
        "pooling": read_pooling(model_dir), "l2_normalize": True,
        "activation": "silu", "gated_ffn": True,
        "swiglu_halves": "fc11_up|fc12_gate",
        "position_embedding_type": "rope",
        "rope_theta": theta,
        "attention_bias": False, "mlp_bias": False,
        "tile_k": tile_k, "tile_n": tile_n, "mac_s": MAC_S, "mac_t": MAC_T,
        "prompts": prompts,
        "prompt_default": "search_document",
        "prompts_source": "npuembeddings, NOT from the checkpoint -- "
                          "config_sentence_transformers.json carries no "
                          "'prompts' dict for this checkpoint, so presenting "
                          "this table as the model's own would be a lie in a "
                          "file other tools read. Same precedent as the "
                          "Gemma table generator "
                          "(runtime/src/tokenizers/gemma_tokenizer_gen.cpp).",
        "l2_normalize_note": "sentence-transformers does NOT L2-normalize "
                             "this model (measured output norm 20.93, "
                             "tasks/0068 sec 5b) -- l2_normalize:true here "
                             "matches THIS RUNTIME's own hardcoded behaviour "
                             "(main.cpp g_l2_normalize) and nomic's own "
                             "documented usage (F.normalize), not "
                             "sentence-transformers' default pipeline for "
                             "this particular model.",
        "fusions": {
            "qkv_fused": True,
            "transposed_to_kn": True,
            "qk_scale_folded_into_q": fold_scale,
            "gemm_operands_bf16": True,
            "biases_and_layernorm_fp32": True,
            "gated_ffn_fused_fc11_fc12": True,
            "position_embeddings_zeroed_rope_instead": True,
        },
        # Copied from the BERT path, this said "pooler.dense" -- which this
        # checkpoint does not have. All 112 of its tensors are consumed
        # (tasks/0068 sec 1), so there is no dead weight to declare, and a
        # claim about a tensor that does not exist is worse than no claim.
        # What IS genuinely not implemented:
        "not_implemented": [
            "Matryoshka truncation (layer_norm(768) -> slice -> normalize is a "
            "different post-processing chain, not just a shorter vector)",
            f"vocab rows {n_reachable}-{cfg['vocab_size'] - 1} are "
            f"pad_vocab_size_multiple padding: non-zero but unreachable from "
            f"the tokenizer (max id {n_reachable - 1}), packed only so "
            f"vocab_size and the tensor agree",
        ],
    }

    w = Writer(config, arch=ARCH_NOMIC_ROPE_SWIGLU)

    # -- embeddings: SAME order as the BERT path, including the odd
    # ln.weight -> tokenizer.vocab -> ln.bias interleaving, which is
    # load-bearing for byte parity with the C++ mirror. --------------------
    w.add("embeddings.word", src["embeddings.word_embeddings.weight"],
          "F32", "embedding", [cfg["vocab_size"], hidden])
    # nomic has NO position table -- RoPE is computed inside attention
    # instead. Zero-filled rather than omitted: Encoder::stage_all() and the
    # --embed path both dereference "embeddings.position" UNCONDITIONALLY
    # (main.cpp:2889), so a zero tensor of the right shape is exact (adds
    # nothing) and keeps that read path untouched.
    w.add("embeddings.position", np.zeros((max_seq, hidden), dtype=np.float32),
          "F32", "embedding", [max_seq, hidden])
    w.add("embeddings.token_type", src["embeddings.token_type_embeddings.weight"],
          "F32", "embedding", [cfg["type_vocab_size"], hidden])
    # emb_ln lives at the TOP LEVEL upstream (not embeddings.LayerNorm, as
    # in BERT) -- tasks/0068 sec 1.
    w.add("embeddings.ln.weight", src["emb_ln.weight"],
          "F32", "layernorm", [hidden])
    vocab_path = model_dir / "vocab.txt"
    if vocab_path.exists():
        vb = np.frombuffer(vocab_path.read_bytes(), dtype=np.uint8)
        w.add("tokenizer.vocab", vb, "U8", "tokenizer", [int(vb.size)])
        print(f"  tokenizer.vocab   {vb.size / 1024:.1f} KB "
              f"({vocab_path.read_bytes().count(chr(10).encode()[0])} lines)")
    else:
        print(f"  WARNING: {vocab_path} not found -- .npue will have no vocab")
    w.add("embeddings.ln.bias", src["emb_ln.bias"],
          "F32", "layernorm", [hidden])

    # ONE emitter for the four per-layer operands, exactly as the BERT path
    # (tasks/0078) -- so bf16 and int8 differ in one place rather than four.
    # The calibration oracle is nomic's OWN: arch=2 has RoPE instead of
    # absolute positions and a gated SwiGLU whose ffn_up is 2*intermediate
    # wide, so BERT's forward pass would describe activations the array never
    # sees (tasks/0081).
    smooth = (calibrate_smoothing(model_dir, alpha=smooth_alpha,
                                  n_texts=smooth_texts, arch="nomic")
              if int8 and smooth_alpha > 0 else {})
    qerr = []

    def emit(name, mat, layer, op):
        if not int8:
            add_gemm_b(w, name, mat, tile_k, tile_n)
            return
        qerr.append((name, add_gemm_b_int8(w, name, mat, tile_k, tile_n,
                                           asmooth=smooth.get((layer, op)))))
    n_tiled = 0
    for i in range(L):
        p = f"encoder.layers.{i}."   # plural upstream, unlike BERT's "layer."
        attn = p + "attn."

        # fused upstream already: Wqkv is [2304,768] three-major
        # [Q(768)|K(768)|V(768)] -- tasks/0068 sec 5 Wqkv row-order check.
        qkv = np.ascontiguousarray(src[attn + "Wqkv.weight"].T)      # [768,2304]
        if fold_scale:
            # RoPE is linear in q, so folding 1/sqrt(head_dim) into the Q
            # block before the GEMM (and before RoPE) is exact -- see
            # tools/verify_npue_nomic.py check E. No qkv bias exists to fold.
            qkv = qkv.copy()
            qkv[:, :hidden] *= scale
        emit(f"layer.{i}.qkv", qkv, i, "qkv")
        w.add(f"layer.{i}.qkv.bias", np.zeros(3 * hidden, dtype=np.float32),
              "F32", "bias", [3 * hidden])
        n_tiled += 1

        emit(f"layer.{i}.attn_out",
             np.ascontiguousarray(src[attn + "out_proj.weight"].T),
             i, "attn_out")
        w.add(f"layer.{i}.attn_out.bias", np.zeros(hidden, dtype=np.float32),
              "F32", "bias", [hidden])
        w.add(f"layer.{i}.ln1.weight", src[p + "norm1.weight"],
              "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.ln1.bias", src[p + "norm1.bias"],
              "F32", "layernorm", [hidden])
        n_tiled += 1

        # gated ffn_up: [fc11 (up, untouched) | fc12 (gate, gets SiLU)]
        # fused along N. Runtime computes out = lo * silu(hi), where
        # lo = cols [0, inter), hi = cols [inter, 2*inter) -- see
        # config["swiglu_halves"]. ONE GEMM, so the array still sees four
        # GEMMs per layer, not five.
        mp = p + "mlp."
        up = np.ascontiguousarray(src[mp + "fc11.weight"].T)         # [768,3072]
        gate = np.ascontiguousarray(src[mp + "fc12.weight"].T)       # [768,3072]
        ffn_up = np.concatenate([up, gate], axis=1)                  # [768,6144]
        emit(f"layer.{i}.ffn_up", ffn_up, i, "ffn_up")
        w.add(f"layer.{i}.ffn_up.bias", np.zeros(2 * inter, dtype=np.float32),
              "F32", "bias", [2 * inter])

        emit(f"layer.{i}.ffn_down",
             np.ascontiguousarray(src[mp + "fc2.weight"].T),
             i, "ffn_down")
        w.add(f"layer.{i}.ffn_down.bias", np.zeros(hidden, dtype=np.float32),
              "F32", "bias", [hidden])
        w.add(f"layer.{i}.ln2.weight", src[p + "norm2.weight"],
              "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.ln2.bias", src[p + "norm2.bias"],
              "F32", "layernorm", [hidden])
        n_tiled += 2

    info = w.write(out)

    print(f"\n  {'operand':<14} {'[K,N]':>12} {'k-blocks':>9} {'n-blocks':>9} "
          f"{'tiles':>7} {'max BD dim':>11}")
    shapes = {"qkv": (hidden, 3 * hidden), "attn_out": (hidden, hidden),
              "ffn_up": (hidden, 2 * inter), "ffn_down": (inter, hidden)}
    for nm, (K, N) in shapes.items():
        kb, nb = K // tile_k, N // tile_n
        flag = "" if max(kb, nb) < 1024 else "  <-- OVER 1023"
        print(f"  {nm:<14} {str([K, N]):>12} {kb:>9} {nb:>9} {kb*nb:>7} "
              f"{max(kb, nb):>11}{flag}")

    total = Path(out).stat().st_size
    print(f"\n  tensors    : {len(w.entries)}  ({n_tiled} pre-tiled GEMM operands)")
    print(f"  json       : {info['json_length']} B at {info['json_offset']}")
    print(f"  data       : {info['data_length']/1e6:.2f} MB at {info['data_offset']}")
    print(f"  file       : {total/1e6:.2f} MB")
    print(f"  source     : {src_sha[:16]}...")
    # The dtype is part of the layout, so printing the bf16 hash over int8
    # tensors reports the intention rather than the value -- the same shape as
    # tasks/0042's `tile (64, 32)` and 0078's banner, both of which cost time.
    print(f"  layout_hash: "
          f"{layout_hash(gemm_b_layout(tile_k, tile_n, MAC_S, MAC_T, dtype='I8' if int8 else 'BF16'))[:16]}..."
          f"{'  (i8 operands)' if int8 else ''}")
    return 0


def pack_gte(model_dir, out, tile_k, tile_n, max_seq, fold_scale, int8=False):
    """Pack a gte-multilingual-base-shaped checkpoint (arch=3, model_type
    "new" -- the NewModel trust_remote_code implementation).

    Emits the SAME tensor names and the SAME emission order as arch=0/2, so
    Encoder::stage_all() and the whole NPU dispatch path work unchanged.
    Every architectural fact asserted below was settled in tasks/0134 by a
    per-layer probe against the repaired fp32 reference (relfro 1e-06 on all
    12 layers, negative controls on the wrong-theta and wrong-half readings)
    -- this function only implements that already-settled architecture and
    asserts the config facts it depends on, so a checkpoint that silently
    changed underneath it refuses to pack rather than packing wrong.

    Departures from the nomic (arch=2) shape it otherwise mirrors:
      * REAL biases on qkv / attn_out / ffn_down (nomic zero-fills all
        three). ffn_up (up_gate_proj) is genuinely bias-free -- zero-filled.
        Folding 1/sqrt(head_dim) into the Q block must therefore scale the
        Q THIRD OF THE BIAS as well: (xW + b)*s == x(Ws) + (bs), and RoPE is
        linear, so this stays exact.
      * the gated FFN arrives ALREADY FUSED upstream: up_gate_proj is one
        [2*inter, hidden] matrix, up rows first, gate rows second -- the
        same [lo|hi] order the runtime's `lo * act(hi)` expects, so no
        concatenation happens here at all. Activation is exact-erf GELU
        (not SiLU) -- recorded in config["activation"], which arch=3's
        runtime reads as DATA rather than hardcoding (tasks/0134 plan).
      * RoPE frequencies are carried as data: config["rope_inv_freq"] holds
        the 32 float32 values inv_freq_i = (theta*factor)^(-i/32) /
        factor^(1/32), duplicated from reference/encoder_gte.py's
        gte_inv_freq() (arch files stand alone -- same choice as
        encoder_nomic.py's copied primitives). A single rope_theta CANNOT
        express this model (0134); rope_theta/rope_scaling are kept as
        provenance only.
      * tokenizer is the XLMRTOK1 Unigram blob (T52, tasks/0127), stored
        whole as "tokenizer.xlmr_table" at the same interleaved position
        arch=0/2 store their vocab.txt -- load-bearing for byte parity with
        the future C++ mirror.
      * embeddings.word is 250,048 x 768 F32 (768 MB): the same
        deliberately-F32 gather as every other arch (pack_nomic's note).
        Rows 250,002..250,047 are padding, unreachable from the tokenizer.
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if int8:
        raise SystemExit(
            "--int8 for arch=3 needs its own calibration oracle "
            "(calibrate_smoothing has no 'gte' arch) -- not implemented in "
            "0.5.0; pack bf16 or extend the oracle first")

    raw, _ = load(model_dir / "model.safetensors")
    src_sha = sha256(model_dir / "model.safetensors")
    # The checkpoint stores F16; every consumer here wants f32 (the bf16
    # pre-tiler and the F32 emitters both). Upcast once, losslessly.
    src = {}
    for k, v in raw.items():
        kk = k[4:] if k.startswith("new.") else k       # strip 'new.'
        src[kk] = v.astype(np.float32) if v.dtype == np.float16 else v

    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    hidden = cfg["hidden_size"]
    head_dim = hidden // H
    inter = cfg["intermediate_size"]
    scale = 1.0 / math.sqrt(head_dim)

    # -- fail-closed assertions: every fact 0134's probe settled -----------
    if cfg["model_type"] != "new":
        raise SystemExit(f"model_type={cfg['model_type']!r}, expected 'new'")
    if cfg["hidden_act"] != "gelu":
        raise SystemExit(f"hidden_act={cfg['hidden_act']!r}, expected 'gelu' "
                         f"(exact erf -- tasks/0134)")
    if cfg["position_embedding_type"] != "rope":
        raise SystemExit(f"position_embedding_type="
                         f"{cfg['position_embedding_type']!r}, expected 'rope'")
    theta = cfg["rope_theta"]
    rs = cfg.get("rope_scaling") or {}
    if theta != 20000 or rs.get("type") != "ntk" or rs.get("factor") != 8.0 \
            or rs.get("mixed_b") is not None:
        raise SystemExit(
            f"rope_theta={theta}, rope_scaling={rs!r} -- expected 20000 / "
            f"ntk / 8.0 / mixed_b None. The baked inv_freq below is derived "
            f"for exactly that configuration (tasks/0134); refusing to pack "
            f"an unverified RoPE against it")
    if cfg["type_vocab_size"] != 1:
        raise SystemExit(f"type_vocab_size={cfg['type_vocab_size']}, expected 1")
    if cfg.get("layer_norm_type", "layer_norm") != "layer_norm":
        raise SystemExit(f"layer_norm_type={cfg['layer_norm_type']!r}")
    if cfg.get("logn_attention_scale"):
        raise SystemExit("logn_attention_scale is set -- 0134's probe "
                         "validated the plain 1/sqrt(head_dim) scale only")
    if not cfg.get("pack_qkv", False):
        raise SystemExit("pack_qkv is false -- this packer reads the fused "
                         "qkv_proj tensor")
    eps = cfg["layer_norm_eps"]

    # The NTK frequency set, float32 exactly as torch computes it --
    # duplicated from reference/encoder_gte.py::gte_inv_freq() (0134:
    # verified bit-for-bit against a freshly constructed module).
    i32 = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = (np.float32(1.0)
                / (np.float32(theta * rs["factor"])
                   ** (i32 / np.float32(head_dim))))
    inv_freq = inv_freq / (np.float32(rs["factor"])
                           ** (np.float32(2.0) / np.float32(head_dim)))
    inv_freq = inv_freq.astype(np.float32)

    tok_blob_path = model_dir / "xlmr_tokenizer.bin"
    if not tok_blob_path.exists():
        raise SystemExit(
            f"{tok_blob_path} not found -- generate it first with the C++ "
            f"packer: `npuembed --prepare-model`")

    print(f"packing {model_dir.name} -> {Path(out).name}  (arch=gte_new_rope_geglu)")
    print(f"  hidden={hidden} heads={H} head_dim={head_dim} layers={L} "
          f"inter={inter} rope=ntk(20000 x 8.0, 32 baked inv_freq)")
    print(f"  tile ({tile_k}, {tile_n}), mac (s={MAC_S}, t={MAC_T}), "
          f"1/sqrt({head_dim}) = {scale:.17g}"
          f"{' folded into Q (weights AND bias)' if fold_scale else ' NOT folded'}")

    config = {
        "arch": "gte_new_rope_geglu",
        "a_dtype": "bf16",
        "model_type": cfg["model_type"],
        "source_repo": json.loads(
            (model_dir / "CHECKPOINT.json").read_text(encoding="utf-8"))["repo_id"],
        "source_sha256": src_sha,
        "num_layers": L, "num_heads": H, "hidden": hidden, "head_dim": head_dim,
        "intermediate": inter,
        "layer_norm_eps": eps,
        "vocab_size": cfg["vocab_size"],
        "max_seq_len": max_seq,
        "pooling": read_pooling(model_dir), "l2_normalize": True,
        "l2_normalize_note": "genuinely the checkpoint's own: modules.json "
                             "lists a 2_Normalize module (unlike nomic, "
                             "where true records this runtime's behaviour).",
        "activation": "gelu", "gated_ffn": True,
        "glu_halves": "up_first|gate_second -- runtime computes "
                      "lo * gelu(hi), same half order as nomic's "
                      "fc11_up|fc12_gate with GELU for SiLU",
        "position_embedding_type": "rope",
        "rope_theta": theta,
        "rope_scaling": {"type": "ntk", "factor": rs["factor"]},
        "rope_inv_freq": [float(x) for x in inv_freq],
        "rope_note": "rope_inv_freq IS the model -- inv_freq_i = "
                     "160000^(-i/32) / 8^(1/32), NOT expressible as any "
                     "single theta (tasks/0134, verified bit-for-bit). "
                     "rope_theta/rope_scaling above are provenance only; a "
                     "consumer that derives frequencies from rope_theta "
                     "alone is wrong by 1.9e-02 relfro at layer 0.",
        "attention_bias": True,
        "mlp_bias": "down_only -- up_gate_proj is genuinely bias-free",
        "tile_k": tile_k, "tile_n": tile_n, "mac_s": MAC_S, "mac_t": MAC_T,
        "fusions": {
            "qkv_fused": True,
            "transposed_to_kn": True,
            "qk_scale_folded_into_q": fold_scale,
            "qk_scale_folded_into_q_bias": fold_scale,
            "gemm_operands_bf16": True,
            "biases_and_layernorm_fp32": True,
            "gated_ffn_fused_upstream": True,
            "position_embeddings_zeroed_rope_instead": True,
        },
        "not_implemented": [
            "int8 datapath (calibrate_smoothing has no 'gte' oracle)",
            f"vocab rows 250002-{cfg['vocab_size'] - 1} are padding: "
            f"unreachable from the tokenizer, packed only so vocab_size and "
            f"the tensor agree",
            "classifier.weight/classifier.bias (a task head this encoder "
            "never runs) are deliberately NOT packed",
        ],
    }

    w = Writer(config, arch=ARCH_GTE_NEW_ROPE_GEGLU)

    # -- embeddings: SAME order as arch=0/2, including the ln.weight ->
    # tokenizer -> ln.bias interleaving (load-bearing for byte parity with
    # the C++ mirror). ----------------------------------------------------
    w.add("embeddings.word", src["embeddings.word_embeddings.weight"],
          "F32", "embedding", [cfg["vocab_size"], hidden])
    w.add("embeddings.position", np.zeros((max_seq, hidden), dtype=np.float32),
          "F32", "embedding", [max_seq, hidden])
    w.add("embeddings.token_type", src["embeddings.token_type_embeddings.weight"],
          "F32", "embedding", [cfg["type_vocab_size"], hidden])
    w.add("embeddings.ln.weight", src["embeddings.LayerNorm.weight"],
          "F32", "layernorm", [hidden])
    tb = np.frombuffer(tok_blob_path.read_bytes(), dtype=np.uint8)
    w.add("tokenizer.xlmr_table", tb, "U8", "tokenizer", [int(tb.size)])
    print(f"  tokenizer.xlmr_table  {tb.size / 1e6:.2f} MB (XLMRTOK1, "
          f"tasks/0127)")
    w.add("embeddings.ln.bias", src["embeddings.LayerNorm.bias"],
          "F32", "layernorm", [hidden])

    n_tiled = 0
    for i in range(L):
        p = f"encoder.layer.{i}."
        at = p + "attention."

        qkv = np.ascontiguousarray(src[at + "qkv_proj.weight"].T)    # [768,2304]
        qkv_b = src[at + "qkv_proj.bias"].copy()                     # [2304]
        if fold_scale:
            # RoPE is linear in q, so folding 1/sqrt(head_dim) into the Q
            # block before the GEMM (and before RoPE) is exact. Unlike
            # nomic, gte HAS a qkv bias, so its Q third scales too:
            # (xW + b)*s == x(Ws) + (bs).
            qkv = qkv.copy()
            qkv[:, :hidden] *= scale
            qkv_b[:hidden] *= scale
        add_gemm_b(w, f"layer.{i}.qkv", qkv, tile_k, tile_n)
        w.add(f"layer.{i}.qkv.bias", qkv_b, "F32", "bias", [3 * hidden])
        n_tiled += 1

        add_gemm_b(w, f"layer.{i}.attn_out",
                   np.ascontiguousarray(src[at + "o_proj.weight"].T),
                   tile_k, tile_n)
        w.add(f"layer.{i}.attn_out.bias", src[at + "o_proj.bias"],
              "F32", "bias", [hidden])
        w.add(f"layer.{i}.ln1.weight", src[p + "attn_ln.weight"],
              "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.ln1.bias", src[p + "attn_ln.bias"],
              "F32", "layernorm", [hidden])
        n_tiled += 1

        # up_gate_proj is already the fused [2*inter, hidden] the runtime
        # wants: transpose to [hidden, 2*inter]; up cols [0, inter), gate
        # cols [inter, 2*inter) -- the lo/hi order of `lo * act(hi)`.
        add_gemm_b(w, f"layer.{i}.ffn_up",
                   np.ascontiguousarray(src[p + "mlp.up_gate_proj.weight"].T),
                   tile_k, tile_n)
        w.add(f"layer.{i}.ffn_up.bias", np.zeros(2 * inter, dtype=np.float32),
              "F32", "bias", [2 * inter])

        add_gemm_b(w, f"layer.{i}.ffn_down",
                   np.ascontiguousarray(src[p + "mlp.down_proj.weight"].T),
                   tile_k, tile_n)
        w.add(f"layer.{i}.ffn_down.bias", src[p + "mlp.down_proj.bias"],
              "F32", "bias", [hidden])
        w.add(f"layer.{i}.ln2.weight", src[p + "mlp_ln.weight"],
              "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.ln2.bias", src[p + "mlp_ln.bias"],
              "F32", "layernorm", [hidden])
        n_tiled += 2

    info = w.write(out)

    print(f"\n  {'operand':<14} {'[K,N]':>12} {'k-blocks':>9} {'n-blocks':>9} "
          f"{'tiles':>7} {'max BD dim':>11}")
    shapes = {"qkv": (hidden, 3 * hidden), "attn_out": (hidden, hidden),
              "ffn_up": (hidden, 2 * inter), "ffn_down": (inter, hidden)}
    for nm, (K, N) in shapes.items():
        kb, nb = K // tile_k, N // tile_n
        flag = "" if max(kb, nb) < 1024 else "  <-- OVER 1023"
        print(f"  {nm:<14} {str([K, N]):>12} {kb:>9} {nb:>9} {kb*nb:>7} "
              f"{max(kb, nb):>11}{flag}")

    total = Path(out).stat().st_size
    print(f"\n  tensors    : {len(w.entries)}  ({n_tiled} pre-tiled GEMM operands)")
    print(f"  json       : {info['json_length']} B at {info['json_offset']}")
    print(f"  data       : {info['data_length']/1e6:.2f} MB at {info['data_offset']}")
    print(f"  file       : {total/1e6:.2f} MB")
    print(f"  source     : {src_sha[:16]}...")
    print(f"  layout_hash: "
          f"{layout_hash(gemm_b_layout(tile_k, tile_n, MAC_S, MAC_T))[:16]}...")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models" / "all-MiniLM-L6-v2"))
    ap.add_argument("--out", default=str(REPO / "models" / "all-MiniLM-L6-v2.npue"))
    ap.add_argument("--tile-k", type=int, default=DEFAULT_TILE_K)
    ap.add_argument("--tile-n", type=int, default=DEFAULT_TILE_N)
    ap.add_argument("--max-seq", type=int, default=256,
                    help="pre-slice position embeddings to this (MiniLM: 256)")
    ap.add_argument("--no-fold-scale", action="store_true",
                    help="do NOT fold 1/sqrt(head_dim) into Q")
    # THE DATAPATH FLAG (tasks/0077, 0078). bf16 stays the default. int8 needs
    # a matching design set (tools/export_gemm_rtp.py --int8); the container's
    # and the design's b_layout_hash differ between the two, so a mismatched
    # pair is refused by the check that already exists.
    ap.add_argument("--int8", action="store_true",
                    help="quantise the four per-layer GEMM operands to int8, "
                         "per output channel, with SmoothQuant. Measured at "
                         "1-cos 1.166e-03 against a 2e-03 gate (tasks/0078) "
                         "and 5.5-7.7x the bf16 datapath (tasks/0077).")
    ap.add_argument("--smooth-alpha", type=float, default=0.5,
                    help="SmoothQuant alpha. 0 disables smoothing, which "
                         "FAILS the gate at 2.864e-03; 0.4 also fails at "
                         "2.356e-03; 0.5 is the measured minimum. Only "
                         "meaningful with --int8.")
    ap.add_argument("--smooth-texts", type=int, default=128,
                    help="calibration corpus size for --int8. More is not "
                         "obviously better: the factors are per-channel "
                         "maxima, so a wider corpus finds larger outliers.")
    ap.add_argument("--gemma-host-only", action="store_true",
                    help="arch=1 only: emit PLAIN F32 row-major GEMM operands "
                         "for the CPU-only GemmaEncoder, as tasks/0064-0065 "
                         "shipped. The default is now the pre-tiled bf16 NPU "
                         "container (tasks/0074); this rebuilds the control.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    # arch=1 branch (tasks/0064-m12-embeddinggemma-arch1-integration): a Gemma3-family checkpoint is a completely
    # different tensor shape and a completely different container -- routed
    # to its own function rather than threaded through the BERT logic below,
    # so the BERT path (in production since M4) is unchanged code, not
    # refactored code. Detected from the checkpoint's OWN config.json
    # (model_type), never assumed from --model-dir's directory name.
    if cfg.get("model_type") == "gemma3_text":
        out = args.out
        if out == str(REPO / "models" / "all-MiniLM-L6-v2.npue"):
            out = str(model_dir.parent / (model_dir.name + ".npue"))
        return pack_gemma(model_dir, out, tile_k=args.tile_k,
                          tile_n=args.tile_n, host_only=args.gemma_host_only,
                          int8=args.int8, smooth_alpha=args.smooth_alpha,
                          smooth_texts=args.smooth_texts)

    # arch=2 branch (tasks/0069-m13-nomic-arch2-container): nomic_bert is
    # RoPE + gated SwiGLU rather than BERT's absolute-position + GELU, so it
    # gets its own function -- same reasoning as the gemma3_text branch
    # above. Unlike Gemma, its GEMM operands DO get pre-tiled (see
    # pack_nomic's docstring), so --tile-k/--tile-n/--max-seq are forwarded.
    if cfg.get("model_type") == "nomic_bert":
        out = args.out
        if out == str(REPO / "models" / "all-MiniLM-L6-v2.npue"):
            out = str(model_dir.parent / (model_dir.name + ".npue"))
        return pack_nomic(model_dir, out, args.tile_k, args.tile_n,
                          args.max_seq, not args.no_fold_scale,
                          int8=args.int8, smooth_alpha=args.smooth_alpha,
                          smooth_texts=args.smooth_texts)

    # arch=3 branch (0.5.0, tasks/0134/0135): model_type "new" is the
    # NewModel family (gte-multilingual-base). Same routing rule as the two
    # branches above: detected from the checkpoint's OWN config.json, never
    # from the directory name.
    if cfg.get("model_type") == "new":
        out = args.out
        if out == str(REPO / "models" / "all-MiniLM-L6-v2.npue"):
            out = str(model_dir.parent / (model_dir.name + ".npue"))
        return pack_gte(model_dir, out, args.tile_k, args.tile_n,
                        args.max_seq, not args.no_fold_scale, int8=args.int8)

    src, _ = load(model_dir / "model.safetensors")
    src_sha = sha256(model_dir / "model.safetensors")

    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    hidden = cfg["hidden_size"]
    head_dim = hidden // H
    scale = 1.0 / math.sqrt(head_dim)

    tk, tn = args.tile_k, args.tile_n
    print(f"packing {model_dir.name} -> {Path(args.out).name}")
    print(f"  tile ({tk}, {tn}), mac (s={MAC_S}, t={MAC_T}), "
          f"1/sqrt({head_dim}) = {scale:.17g}"
          f"{' NOT folded' if args.no_fold_scale else ' folded into Q'}")

    config = {
        "arch": "bert_abs_gelu_postln",
        "source_repo": json.loads(
            (model_dir / "CHECKPOINT.json").read_text(encoding="utf-8"))["repo_id"],
        "source_sha256": src_sha,
        "num_layers": L, "num_heads": H, "hidden": hidden, "head_dim": head_dim,
        "intermediate": cfg["intermediate_size"],
        "layer_norm_eps": cfg["layer_norm_eps"],
        "vocab_size": cfg["vocab_size"],
        "max_seq_len": args.max_seq,
        "pooling": read_pooling(model_dir), "l2_normalize": True,
        "activation": "gelu_erf_exact",
        "tile_k": tk, "tile_n": tn, "mac_s": MAC_S, "mac_t": MAC_T,
        # The operand datapath. Absent in every container packed before
        # tasks/0078, and every one of those is bf16 -- so the runtime reads
        # silence as "bf16" rather than defaulting blindly.
        "a_dtype": "i8" if args.int8 else "bf16",
        "fusions": {
            "qkv_fused": True,
            "transposed_to_kn": True,
            "qk_scale_folded_into_q": not args.no_fold_scale,
            "gemm_operands_bf16": True,
            "biases_and_layernorm_fp32": True,
            "position_embeddings_presliced_to": args.max_seq,
        },
        "not_implemented": ["pooler.dense (unused by sentence-transformers)"],
    }

    w = Writer(config)

    # -- embeddings: gathered, never multiplied, so never tiled --------------
    # Kept fp32 deliberately: a memory-bound gather costs nothing extra in
    # compute, and it removes one source of numerical drift at the input.
    w.add("embeddings.word", src["embeddings.word_embeddings.weight"],
          "F32", "embedding", [cfg["vocab_size"], hidden])
    w.add("embeddings.position",
          src["embeddings.position_embeddings.weight"][:args.max_seq],
          "F32", "embedding", [args.max_seq, hidden])
    w.add("embeddings.token_type", src["embeddings.token_type_embeddings.weight"],
          "F32", "embedding", [cfg["type_vocab_size"], hidden])
    w.add("embeddings.ln.weight", src["embeddings.LayerNorm.weight"],
          "F32", "layernorm", [hidden])
    # The tokenizer vocabulary, verbatim, as bytes. It is not model data and
    # it is not touched by any kernel -- it rides here so that deploying the
    # model is copying ONE file. The C++ tokenizer reads it straight out of
    # the mapping, exactly as the weights are read.
    vocab_path = model_dir / "vocab.txt"
    if vocab_path.exists():
        vb = np.frombuffer(vocab_path.read_bytes(), dtype=np.uint8)
        w.add("tokenizer.vocab", vb, "U8", "tokenizer", [int(vb.size)])
        print(f"  tokenizer.vocab   {vb.size / 1024:.1f} KB "
              f"({vocab_path.read_bytes().count(chr(10).encode()[0])} lines)")
    else:
        print(f"  WARNING: {vocab_path} not found -- .npue will have no vocab")

    w.add("embeddings.ln.bias", src["embeddings.LayerNorm.bias"],
          "F32", "layernorm", [hidden])

    # ONE emitter for the four per-layer operands, so bf16 and int8 differ in
    # exactly one place rather than in four (tasks/0078).
    smooth = calibrate_smoothing(model_dir, alpha=args.smooth_alpha, n_texts=args.smooth_texts)         if args.int8 and args.smooth_alpha > 0 else {}
    qerr = []
    def emit(name, mat, layer, op):
        if not args.int8:
            add_gemm_b(w, name, mat, tk, tn)
            return
        qerr.append((name, add_gemm_b_int8(w, name, mat, tk, tn,
                                           asmooth=smooth.get((layer, op)))))

    n_tiled = 0
    for i in range(L):
        p = f"encoder.layer.{i}."
        sa = p + "attention.self."

        # fusion 1 + 2: concat Q|K|V along `out`, then transpose to [K, N].
        qkv = np.concatenate([src[sa + n + ".weight"]
                              for n in ("query", "key", "value")], axis=0)   # [1152,384]
        qkv_b = np.concatenate([src[sa + n + ".bias"]
                                for n in ("query", "key", "value")], axis=0)  # [1152]
        qkv = np.ascontiguousarray(qkv.T)                                    # [384,1152]

        # fusion 4: fold 1/sqrt(head_dim) into the Q block only -- the first
        # `hidden` columns of the fused N axis, and the matching bias slice.
        if not args.no_fold_scale:
            qkv = qkv.copy()
            qkv[:, :hidden] *= scale
            qkv_b = qkv_b.copy()
            qkv_b[:hidden] *= scale

        emit(f"layer.{i}.qkv", qkv, i, "qkv")
        w.add(f"layer.{i}.qkv.bias", qkv_b, "F32", "bias", [3 * hidden])
        n_tiled += 1

        ao = p + "attention.output."
        emit(f"layer.{i}.attn_out",
             np.ascontiguousarray(src[ao + "dense.weight"].T), i, "attn_out")
        w.add(f"layer.{i}.attn_out.bias", src[ao + "dense.bias"],
              "F32", "bias", [hidden])
        w.add(f"layer.{i}.ln1.weight", src[ao + "LayerNorm.weight"],
              "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.ln1.bias", src[ao + "LayerNorm.bias"],
              "F32", "layernorm", [hidden])
        n_tiled += 1

        emit(f"layer.{i}.ffn_up",
             np.ascontiguousarray(src[p + "intermediate.dense.weight"].T), i, "ffn_up")
        w.add(f"layer.{i}.ffn_up.bias", src[p + "intermediate.dense.bias"],
              "F32", "bias", [cfg["intermediate_size"]])
        emit(f"layer.{i}.ffn_down",
             np.ascontiguousarray(src[p + "output.dense.weight"].T), i, "ffn_down")
        w.add(f"layer.{i}.ffn_down.bias", src[p + "output.dense.bias"],
              "F32", "bias", [hidden])
        w.add(f"layer.{i}.ln2.weight", src[p + "output.LayerNorm.weight"],
              "F32", "layernorm", [hidden])
        w.add(f"layer.{i}.ln2.bias", src[p + "output.LayerNorm.bias"],
              "F32", "layernorm", [hidden])
        n_tiled += 2

    if args.int8 and qerr:
        # SAY WHAT THE QUANTISATION COST, per tensor, at pack time. A bad
        # tensor is then visible here rather than as a puzzling MTEB delta.
        worst = max(qerr, key=lambda t: t[1])
        mean = sum(e for _, e in qerr) / len(qerr)
        print(f"\n  int8: {len(qerr)} operands, weight rel_fro mean "
              f"{mean:.3e}, worst {worst[1]:.3e} ({worst[0]})")
    info = w.write(args.out)

    # What the DMA will actually see, per distinct GEMM shape. The 1023 limit is
    # on a BD size field, so what matters is that no access-pattern dimension
    # reaches it -- with whole tiles read linearly, the dims are tile counts.
    print(f"\n  {'operand':<14} {'[K,N]':>12} {'k-blocks':>9} {'n-blocks':>9} "
          f"{'tiles':>7} {'max BD dim':>11}")
    shapes = {"qkv": (hidden, 3 * hidden), "attn_out": (hidden, hidden),
              "ffn_up": (hidden, cfg["intermediate_size"]),
              "ffn_down": (cfg["intermediate_size"], hidden)}
    for nm, (K, N) in shapes.items():
        kb, nb = K // tk, N // tn
        flag = "" if max(kb, nb) < 1024 else "  <-- OVER 1023"
        print(f"  {nm:<14} {str([K, N]):>12} {kb:>9} {nb:>9} {kb*nb:>7} "
              f"{max(kb, nb):>11}{flag}")
    print(f"  (row-major DDR would need K={cfg['intermediate_size']} as a BD "
          f"dimension for ffn_down -- over 1023, which is why it could not be "
          f"expressed before)")

    total = Path(args.out).stat().st_size
    print(f"\n  tensors    : {len(w.entries)}  ({n_tiled} pre-tiled GEMM operands)")
    print(f"  json       : {info['json_length']} B at {info['json_offset']}")
    print(f"  data       : {info['data_length']/1e6:.2f} MB at {info['data_offset']}")
    print(f"  file       : {total/1e6:.2f} MB")
    print(f"  source     : {src_sha[:16]}...")
    # REPORT THE VALUE, NOT THE INTENTION. This printed the bf16 hash while the
    # tensors carried the I8 one -- the same fail-open shape as tasks/0042's
    # "tile (64, 32)" banner over a tile-48 pack. Build the descriptor the same
    # way the emitter does.
    print(f"  layout_hash: "
          f"{layout_hash(gemm_b_layout(tk, tn, MAC_S, MAC_T, dtype='I8' if args.int8 else 'BF16'))[:16]}..."
          f"  ({'i8' if args.int8 else 'bf16'} operands)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
