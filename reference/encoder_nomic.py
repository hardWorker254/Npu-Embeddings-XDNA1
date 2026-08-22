# NpuEmbeddings -- M13 (tasks/0069, reference half): nomic-embed-text-v1.5
# reference encoder. Pure numpy, no torch -- same discipline as
# reference/encoder.py (M3, MiniLM) and reference/encoder_gemma.py (C1,
# EmbeddingGemma): one operation per line, no fused shortcuts, every numerical
# decision spelled out where it happens and cross-checked against the real
# checkpoint (tasks/0068's fetch_model.py inventory and probe_nomic_arch.py's
# hardware-independent empirical probe), not assumed from the model card.
#
# THIS IS THE arch=2 CANDIDATE. Not yet wired into tools/npue.py, the .npue
# packer, or the C++ runtime -- that is tasks/0069's OTHER half (arch=2
# container, pack_nomic()), owned by a different agent working in tools/ at
# the same time this file was written. See tasks/0068-.../TASK.md for the full
# empirical derivation of every architectural fact asserted below; this file
# treats those findings as SETTLED and does not re-derive them.
#
# Architecture, confirmed from the real checkpoint (nomic-ai/nomic-embed-text-v1.5,
# config.json model_type "nomic_bert") and from tasks/0068's probe_nomic_arch.py,
# which hooked the real (trust_remote_code) forward pass and compared it against
# both the correct numpy reading AND a deliberately wrong "negative control"
# reading for every architectural choice below -- so each fact has a measured
# discriminator on record, not just a code reading:
#
#  * hidden 768, 12 layers, 12 heads, head_dim 64, intermediate 3072,
#    vocab_size 30528 (30522 real rows from vocab.txt + 6 padding rows to a
#    multiple of 64 -- unreachable by the tokenizer, kept only so the
#    checkpoint's vocab_size and the tensor shape agree).
#
#  * Embedding: word_embeddings[input_ids] + token_type_embeddings[0] (added
#    for EVERY position, unconditionally -- confirmed non-trivial:
#    ||token_type_embeddings[0]|| = 2.34, not a zero row, so omitting it is a
#    real, measured error (probe: rel_fro 7.56e-01 without it), not a no-op).
#    Then emb_ln, a LayerNorm at TOP LEVEL (checkpoint tensor names
#    "emb_ln.weight"/"emb_ln.bias" -- NOT "embeddings.LayerNorm.*" the way
#    BERT/MiniLM name it). NO absolute position embeddings are added: the
#    checkpoint has no embeddings.position_embeddings.weight tensor at all;
#    position information comes entirely from RoPE inside attention.
#
#  * Each of the 12 layers is POST-LN, matching config.json's "prenorm": false
#    -- and this direction is not the BERT-obvious default to assume, because
#    nomic's own remote modelling code supports EITHER prenorm=True or False
#    depending on config, and this specific checkpoint is prenorm=False.
#    Confirmed empirically (probe Q1): the post-LN reading matches the real
#    layer-0 output at rel_fro 2.401e-07; the pre-LN reading the same code
#    could equally well have implemented misses by rel_fro 5.031e+00 -- a
#    2.1e+07x gap, i.e. this is not a subtle rounding difference, it is a
#    completely different (and completely wrong) function of the same inputs.
#        h = norm1(attn(h) + h)
#        h = norm2(mlp(h) + h)
#    NOTE there is no separate final top-level norm after layer 12 (unlike
#    Gemma's `model.norm`) -- layer 11's norm2 output IS last_hidden_state.
#
#  * Attention is plain multi-head (NOT MQA/GQA like EmbeddingGemma -- nomic
#    has num_attention_heads == num_key_value_heads == 12, no head repeat).
#    `Wqkv` is ONE fused [2304, 768] weight, THREE-MAJOR row order:
#    rows [0:768] = Q, [768:1536] = K, [1536:2304] = V, each block itself
#    ordered head-major-then-dim (i.e. reshape (seq, 3, heads, head_dim) and
#    index the "3" axis, NOT (seq, heads, 3, head_dim)). Confirmed empirically
#    (probe): three-major matches real attn0 output at rel_fro 2.401e-07 (see
#    Q1 control above, which used the three-major split throughout); the wrong
#    head-major split was probed independently and measured rel_fro 3.35e+00.
#    Attention scale is a plain 1/sqrt(head_dim) = 1/8 -- NOT the tensor's own
#    hidden_size (probe: 1/sqrt(768) measured rel_fro 6.30e-01, and the scale
#    was confirmed IDENTICAL at layer 0 and layer 11, so
#    `scale_attn_by_inverse_layer_idx` (present in config.json, value false)
#    really is dead -- it has zero references in the real modelling code).
#    No projection anywhere in attention (Wqkv, out_proj) carries a bias --
#    confirmed by the checkpoint's own tensor inventory (112 tensors, all
#    five projections `.weight`-only) and by config.json's
#    `qkv_proj_bias: false`.
#
#  * RoPE: theta 1000.0 -- NOT the usual 10000 (probe: theta=10000 gives
#    rel_fro 9.2e-02 on the attention output alone, the SUBTLEST wrong
#    reading tested here, against 0.5-5.0 for every other wrong candidate --
#    a model that "looks fine" with the wrong theta is exactly the trap this
#    project's docs warn about, so theta is asserted from config.json's
#    `rotary_emb_base`/`rope_parameters.rope_theta`, not inferred).
#    Convention is NeoX-style: cos/sin built as concat(freqs, freqs) along the
#    head_dim axis (NOT interleaved GPT-J style -- probe: interleaved measured
#    rel_fro 5.61e-01), combined via rotate_half = concat(-x2, x1). Applied to
#    Q and K ONLY, never V (probe: rotating V too measured rel_fro 5.17e-01).
#    Positions start at 0, no offset (no left-padding in this project's
#    tokenization, so this is exact for our use). `rotary_emb_fraction: 1.0`
#    means the FULL head_dim (64) is rotated, not a fraction of it.
#
#  * MLP is SwiGLU, with THREE separate [3072,768]/[3072,768]/[768,3072]
#    weights (fc11, fc12, fc2 -- NOT the two-matrix GeGLU shape
#    EmbeddingGemma uses):
#        out = fc2( fc11(x) * silu(fc12(x)) )
#    SiLU lands on fc12; fc11 is the untouched "value"/up-path multiplicand.
#    This is the single most dangerous fact in this file -- swapping which
#    projection gets the activation is structurally plausible AND produces a
#    sane-looking (if wrong) embedding, so it was confirmed THREE independent
#    ways in tasks/0068: (a) the numeric probe, real mlp0 output vs candidate
#    A (SiLU on fc12) at rel_fro 1.636e-07 vs candidate B (SiLU on fc11) at
#    4.022e+00 -- a 2.5e7x gap; (b) reading the real remote modelling code's
#    `NomciBertGatedMLP.forward` (`y=fc11(x); gate=fc12(x); y=y*activation(gate)`);
#    (c) HuggingFace's own native-port conversion table, which renames
#    fc11->up_proj (untouched) and fc12->gate_proj (gets the activation) --
#    see transformers/models/nomic_bert/modeling_nomic_bert.py's
#    `NomicBertMLP.forward`: `down_proj(act_fn(gate_proj(x)) * up_proj(x))`.
#    There is NO norm between the gate-multiply and fc2 -- `mlp.norm` is
#    `nn.Identity()` for this checkpoint (config has no `norm_mlp` key, so
#    `getattr(config, "norm_mlp", False)` is False; also confirmed by there
#    being no `*.mlp.norm.*` tensor among the checkpoint's 112). No bias on
#    fc11/fc12/fc2 (config `mlp_fc1_bias`/`mlp_fc2_bias`: false).
#    Activation function is `hidden_act: "silu"` -- plain SiLU, not
#    tanh-approximate GELU (that is EmbeddingGemma's GeGLU, a different model).
#
#  * Pooling: MEAN, attention-mask-weighted, same 1e-9 denominator clamp as
#    encoder.py. sentence-transformers does NOT L2-normalize this checkpoint's
#    output (modules.json lists only Transformer + Pooling, no Normalize --
#    confirmed empirically in tasks/0068: SentenceTransformer.encode() returns
#    vectors of norm ~20.93, matching the raw manual mean-pool to
#    rel_fro 6.37e-08, while a manually-normalized pool differs by 9.52e-01).
#    Because of that, `encode()` below returns BOTH the raw pooled vector and
#    the L2-normalized one, explicitly labelled -- silently picking one would
#    hide exactly the kind of scale mismatch this project has been bitten by
#    before (docs/04-model's own warning about absolute-cosine gates).
#
#  * The task-prefix protocol: nomic REQUIRES a plain-text prefix prepended
#    before [CLS] ("search_document: ", "search_query: ", "clustering: ",
#    "classification: "). `config_sentence_transformers.json` carries NO
#    `prompts` dict for this checkpoint (only a `__version__` block), so this
#    table is THIS PROJECT'S CHOICE, not something read from the checkpoint --
#    exactly like EmbeddingGemma's PROMPTS in encoder_gemma.py. Confirmed
#    (tasks/0068) that `[CLS] + prefix_ids + text_ids + [SEP]` equals
#    whole-string tokenization for all four, i.e. plain string concatenation
#    before the tokenizer call is correct and needs no special-casing.
#
# RoPE cos/sin/rotate_half/apply_rope below are copied (not imported) from
# reference/encoder_gemma.py, which already implements the exact
# concat(freqs,freqs)/rotate-half (NeoX) convention nomic needs, with `base`
# as a parameter -- this file stands alone, matching encoder_gemma.py's own
# choice not to import across arch files.
#
# Env: numpy only (runs in either .venv-ref or the iron env).

import math

import numpy as np

MASK_FILL = np.finfo(np.float32).min


def layer_norm(x, weight, bias, eps=1e-12):
    """PyTorch LayerNorm over the last axis: (x-mean)/sqrt(var+eps)*w + b.

    Same two landmines as encoder.py's layernorm(): eps INSIDE the sqrt, and
    PyTorch's BIASED variance estimator (divide by N, not N-1). Computed in
    fp64 for the same reason -- cheap insurance on a 768-wide reduction.
    """
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=-1, keepdims=True)
    centred = x64 - mean
    var = (centred * centred).mean(axis=-1, keepdims=True)
    normed = centred / np.sqrt(var + eps)
    out = normed * weight.astype(np.float64) + bias.astype(np.float64)
    return out.astype(np.float32)


def silu(x):
    """SiLU / swish: x * sigmoid(x). Computed in fp64, matching every other
    elementwise primitive in this file."""
    x64 = x.astype(np.float64)
    return (x64 / (1.0 + np.exp(-x64))).astype(np.float32)


def softmax(x):
    """Row softmax in fp64 with max subtraction -- same as encoder.py."""
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def fp32_gemm(a, b):
    """The default GEMM: plain fp32. (M,K) x (K,N) -> (M,N)."""
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)


def rope_cos_sin(seq_len, head_dim, base):
    """Standard RoPE cos/sin tables (copied from encoder_gemma.py -- same
    NeoX concat(freqs,freqs) convention, `base` as a free parameter since
    nomic's theta (1000) differs from the usual 10000).

    inv_freq[j] = base ** (-2j/head_dim) for j in [0, head_dim/2).
    freqs[s,j] = s * inv_freq[j]; emb = concat(freqs, freqs) along the last
    axis (NOT interleaved) -- rotate_half convention. attention_scaling is
    1.0 for nomic's "default" rope_type, so it is omitted.
    """
    j = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (base ** (j / head_dim))
    pos = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)                    # [S, head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)       # [S, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    """[..., D] -> concat(-x[D/2:], x[:D/2]) along the last axis."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, cos, sin):
    """q,k: [B,H,S,D]; cos,sin: [S,D] (broadcast over batch and heads)."""
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    q2 = q * cos_b + rotate_half(q) * sin_b
    k2 = k * cos_b + rotate_half(k) * sin_b
    return q2.astype(np.float32), k2.astype(np.float32)


class NomicEmbeddingReference:
    """NomicBertModel forward (post-LN + RoPE + SwiGLU) + mean pooling, numpy.

    `w` is the checkpoint's raw state dict (HF tensor names, as they appear in
    model.safetensors: "embeddings.word_embeddings.weight",
    "embeddings.token_type_embeddings.weight", "emb_ln.{weight,bias}",
    "encoder.layers.{i}.attn.Wqkv.weight", ".attn.out_proj.weight",
    ".mlp.fc11.weight", ".mlp.fc12.weight", ".mlp.fc2.weight",
    ".norm1.{weight,bias}", ".norm2.{weight,bias}" -- note "layers" plural).

    `gemm` is the (M,K)x(K,N)->(M,N) primitive, swappable exactly like
    encoder.py's MiniLMReference and encoder_gemma.py's GemmaEmbeddingReference
    -- the seam a future bf16/NPU precision study would use.
    """

    def __init__(self, w, num_layers=12, hidden=768, num_heads=12, head_dim=64,
                 intermediate=3072, eps=1e-12, rope_theta=1000.0, gemm=None,
                 wrong_swiglu=False):
        self.w = w
        self.gemm = gemm or fp32_gemm
        self.L = num_layers
        self.hidden = hidden
        self.H = num_heads
        self.head_dim = head_dim
        self.inter = intermediate
        self.eps = eps
        self.rope_theta = rope_theta
        self.scale = 1.0 / math.sqrt(head_dim)
        # DISCRIMINATING CONTROL ONLY (check_reference_nomic.py) -- never true
        # in normal use. Swaps which projection gets SiLU, i.e. deliberately
        # implements the WRONG candidate B from tasks/0068's Q2 probe
        # (silu(fc11(x))*fc12(x) instead of the correct fc11(x)*silu(fc12(x))),
        # to prove this oracle is actually sensitive to the thing it asserts.
        self.wrong_swiglu = wrong_swiglu

    # -- primitives --------------------------------------------------------

    def linear(self, x, weight):
        """nn.Linear with bias=False (true for EVERY projection in this
        model): weight stored [out,in], y = x @ W.T."""
        flat = x.reshape(-1, x.shape[-1])
        return self.gemm(flat, np.ascontiguousarray(weight.T)).reshape(*x.shape[:-1], -1)

    def batched_gemm(self, a, b):
        """[B,H,M,K] x [B,H,K,N] -> [B,H,M,N], one self.gemm call per head."""
        B, H = a.shape[0], a.shape[1]
        out = np.empty((B, H, a.shape[2], b.shape[3]), dtype=np.float32)
        for i in range(B):
            for j in range(H):
                out[i, j] = self.gemm(a[i, j], b[i, j])
        return out

    # -- pieces --------------------------------------------------------------

    def embed(self, input_ids, tap):
        word = self.w["embeddings.word_embeddings.weight"][input_ids].astype(np.float32)
        tap("emb.word", word)
        # token_type_embeddings[0] added at EVERY position -- this checkpoint
        # never uses a non-zero token_type_id, but row 0 is genuinely non-zero
        # (||.||=2.34) so this is a real addition, not a documented no-op.
        ttype0 = self.w["embeddings.token_type_embeddings.weight"][0].astype(np.float32)
        summed = (word.astype(np.float64) + ttype0.astype(np.float64)).astype(np.float32)
        tap("emb.sum", summed)
        out = layer_norm(summed, self.w["emb_ln.weight"], self.w["emb_ln.bias"], self.eps)
        tap("emb.ln", out)
        return out

    def attention(self, i, x, add_mask, cos, sin, tap):
        B, S, _ = x.shape
        p = f"encoder.layers.{i}.attn."
        qkv = self.linear(x, self.w[p + "Wqkv.weight"])       # [B,S,2304]
        tap(f"L{i}.qkv", qkv)

        # three-major split: (seq, 3, heads, head_dim), index the "3" axis.
        qkv_r = qkv.reshape(B, S, 3, self.H, self.head_dim)
        q = qkv_r[:, :, 0].transpose(0, 2, 1, 3)               # [B,H,S,hd]
        k = qkv_r[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv_r[:, :, 2].transpose(0, 2, 1, 3)
        tap(f"L{i}.q", q)
        tap(f"L{i}.k", k)
        tap(f"L{i}.v", v)

        q, k = apply_rope(q, k, cos, sin)                      # RoPE on Q,K only
        tap(f"L{i}.q_rope", q)
        tap(f"L{i}.k_rope", k)

        scores = self.batched_gemm(q, np.ascontiguousarray(k.transpose(0, 1, 3, 2)))
        scores = (scores.astype(np.float64) * self.scale).astype(np.float32)
        tap(f"L{i}.scores", scores)

        scores = scores + add_mask
        tap(f"L{i}.scores_masked", scores)

        probs = softmax(scores)
        tap(f"L{i}.probs", probs)

        ctx = self.batched_gemm(probs, v)                      # [B,H,S,hd]
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, S, self.H * self.head_dim)
        tap(f"L{i}.ctx", ctx)

        out = self.linear(ctx, self.w[p + "out_proj.weight"])
        tap(f"L{i}.attn_out", out)
        return out

    def mlp(self, i, x, tap):
        p = f"encoder.layers.{i}.mlp."
        y11 = self.linear(x, self.w[p + "fc11.weight"])       # untouched up-path
        y12 = self.linear(x, self.w[p + "fc12.weight"])       # gated by SiLU
        tap(f"L{i}.fc11", y11)
        tap(f"L{i}.fc12", y12)
        if not self.wrong_swiglu:
            act = silu(y12)
            tap(f"L{i}.silu", act)
            gated = (y11.astype(np.float64) * act.astype(np.float64)).astype(np.float32)
        else:
            act = silu(y11)                                    # WRONG, control only
            tap(f"L{i}.silu", act)
            gated = (act.astype(np.float64) * y12.astype(np.float64)).astype(np.float32)
        tap(f"L{i}.gated", gated)
        down = self.linear(gated, self.w[p + "fc2.weight"])   # mlp.norm is Identity
        tap(f"L{i}.fc2", down)
        return down

    def layer(self, i, x, add_mask, cos, sin, tap):
        """Post-LN sandwich, matching config.json's prenorm=false:
            h = norm1(attn(h) + h)
            h = norm2(mlp(h) + h)
        """
        attn_out = self.attention(i, x, add_mask, cos, sin, tap)
        h = layer_norm(attn_out + x, self.w[f"encoder.layers.{i}.norm1.weight"],
                        self.w[f"encoder.layers.{i}.norm1.bias"], self.eps)
        tap(f"L{i}.norm1", h)

        mlp_out = self.mlp(i, h, tap)
        out = layer_norm(mlp_out + h, self.w[f"encoder.layers.{i}.norm2.weight"],
                          self.w[f"encoder.layers.{i}.norm2.bias"], self.eps)
        tap(f"L{i}.norm2", out)
        return out

    # -- forward -------------------------------------------------------------

    def encode(self, input_ids, attention_mask, taps=None):
        """Returns (pooled_raw, pooled_l2_normalized), BOTH [B, 768].

        sentence-transformers does NOT L2-normalize this checkpoint (see file
        header) -- callers that want the ST-equivalent output use the FIRST
        element; callers that want this project's runtime convention (which
        always normalizes, `g_l2_normalize` hardcoded true in main.cpp) use
        the SECOND. Returning both, explicitly, is deliberate: silently
        picking one is exactly the kind of scale mismatch a cosine-based gate
        would never catch.
        """
        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)
        S = input_ids.shape[1]

        def tap(name, value):
            if taps is not None:
                taps[name] = np.ascontiguousarray(value, dtype=np.float32)

        add_mask = (1.0 - attention_mask[:, None, None, :].astype(np.float32)) * MASK_FILL
        tap("attn_add_mask", add_mask)

        cos, sin = rope_cos_sin(S, self.head_dim, self.rope_theta)

        x = self.embed(input_ids, tap)
        for i in range(self.L):
            x = self.layer(i, x, add_mask, cos, sin, tap)
        # No final top-level norm for nomic -- layer (L-1)'s norm2 output IS
        # last_hidden_state.
        tap("last_hidden_state", x)

        m = attention_mask[:, :, None].astype(np.float32)
        summed = (x * m).sum(axis=1)
        denom = np.clip(m.sum(axis=1), 1e-9, None)
        pooled_raw = summed / denom
        tap("pool.mean_raw", pooled_raw)

        norm = np.clip(np.linalg.norm(pooled_raw, axis=1, keepdims=True), 1e-12, None)
        pooled_normalized = (pooled_raw / norm).astype(np.float32)
        tap("pool.mean_l2normalized", pooled_normalized)

        return pooled_raw.astype(np.float32), pooled_normalized


def load_reference(model_dir):
    """Build the reference from models/nomic-embed-text-v1.5/."""
    import json
    from pathlib import Path

    from safetensors_io import load

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    w, _ = load(model_dir / "model.safetensors")
    rope_theta = cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rotary_emb_base", 1000.0))
    return NomicEmbeddingReference(
        w,
        num_layers=cfg["num_hidden_layers"],
        hidden=cfg["hidden_size"],
        num_heads=cfg["num_attention_heads"],
        head_dim=cfg["head_dim"],
        intermediate=cfg["intermediate_size"],
        eps=cfg["layer_norm_epsilon"],
        rope_theta=float(rope_theta),
    )


# Task-prefix protocol nomic REQUIRES (see file header: config_sentence_
# transformers.json has NO "prompts" dict for this checkpoint -- this table
# is THIS PROJECT'S choice, not the checkpoint's, and must be pinned as data
# in the goldens rather than baked in as a silent constant). Token counts
# (tasks/0068 TASK.md): search_document/search_query 4 tokens, clustering 3,
# classification 2.
PROMPTS = {
    "search_document": "search_document: ",
    "search_query": "search_query: ",
    "clustering": "clustering: ",
    "classification": "classification: ",
}
