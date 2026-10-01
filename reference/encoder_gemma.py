# NpuEmbeddings -- C1 spike: EmbeddingGemma-300M reference encoder.
#
# Pure numpy, fp32, no torch -- same discipline as reference/encoder.py (the M3
# MiniLM oracle): one operation per line, no fused shortcuts, every numerical
# decision spelled out where it happens and cross-checked against the real HF
# source (transformers/models/gemma3/modeling_gemma3.py, installed in
# .venv-ref, transformers 5.15.0) rather than assumed.
#
# This is a RESEARCH SPIKE (tasks/0055), not production code. It is not wired
# into the C++ runtime, the .npue packer, or the tokenizer -- see
# tasks/0055-.../TASK.md for the go/no-go on that (C2-C4 in the research plan).
#
# Architecture, confirmed from the real Gemma3 modeling code (NOT guessed):
#
#  * Embedding: token lookup, then scaled by sqrt(hidden_size). HF stores this
#    as `Gemma3TextScaledWordEmbedding.embed_scale`, a python float computed
#    once and cast to the weight's dtype at multiply time. This checkpoint's
#    weights are F32 on disk (verified: every tensor in
#    unsloth/embeddinggemma-300m's model.safetensors is F32, not BF16), so
#    there is no bf16-downcast-of-the-scale landmine here -- see
#    modeling_gemma3.py's own comment about sqrt(3072) rounding to 55.5 in
#    bf16, which does NOT apply to this checkpoint's dtype.
#
#  * Each of the 24 layers is a SANDWICH: four independent RMSNorms, not two
#    LayerNorms like BERT/MiniLM:
#        residual = x
#        h = input_layernorm(x)
#        h = self_attn(h)                    # MQA + RoPE + q_norm/k_norm, below
#        h = post_attention_layernorm(h)
#        x = residual + h
#        residual = x
#        h = pre_feedforward_layernorm(x)
#        h = mlp(h)                          # GeGLU
#        h = post_feedforward_layernorm(h)
#        x = residual + h
#    Confirmed against Gemma3DecoderLayer.forward line by line.
#
#  * RMSNorm is NOT `x/rms * w` (Llama-style). Gemma3RMSNorm computes
#    `x/rms * (1 + w)` -- the weight is stored zero-centred so a freshly
#    initialised model is the identity. Missing the `1 +` gives a plausible
#    but completely wrong answer (everything scaled by whatever w happens to
#    be near 0, not near 1). Confirmed in Gemma3RMSNorm.forward.
#
#  * Attention is MQA/GQA: 3 query heads, 1 KV head, head_dim 256 (NOT
#    hidden/num_heads=256 by coincidence of arithmetic -- head_dim is an
#    explicit config field and happens to equal hidden_size/num_heads here,
#    but q_proj/k_proj/v_proj shapes are read from head_dim*num_heads and
#    head_dim*num_kv_heads independently). K and V are computed once (256-wide)
#    and REPEATED 3x to match Q's 3 heads before the QK^T / softmax / .V GEMMs
#    -- this is what MQA means computationally, not a fused kernel trick.
#
#  * q_norm/k_norm: an RMSNorm over head_dim (256), applied to Q and K
#    PER-HEAD, AFTER the q/k projection and BEFORE RoPE. This has no BERT
#    analogue and is easy to miss -- Gemma3Attention.forward calls
#    `self.q_norm(query_states)` / `self.k_norm(key_states)` between the
#    reshape-to-heads step and `apply_rotary_pos_emb`.
#
#  * Attention scale is `query_pre_attn_scalar ** -0.5` (config value 256),
#    NOT `head_dim ** -0.5` computed from the tensor shape -- they are equal
#    for this checkpoint (both 256) but are conceptually different knobs, and
#    a model where they differ would need the config value, not the shape.
#
#  * RoPE base frequency is PER LAYER, not global. Layers where
#    `(i+1) % sliding_window_pattern == 0` (sliding_window_pattern=6, so
#    layers 5, 11, 17, 23 zero-indexed) are `full_attention` and use
#    rope_theta=1e6; every other layer is `sliding_attention` and uses
#    rope_local_base_freq=1e4. Confirmed against config.json's own
#    `layer_types` list and configuration_gemma3.py's rope_parameters
#    construction. Getting this wrong desyncs Q/K's rotary phase on 20 of 24
#    layers while looking numerically plausible (RoPE with the wrong base
#    still produces unit-norm rotations).
#
#  * SLIDING-WINDOW MASKING IS NOT IMPLEMENTED -- see the note below. This is
#    the one deliberate simplification in this file, and unlike the framing in
#    the task prompt it is not an approximation for this project's use case,
#    it is EXACT for it. Justification:
#
#      `use_bidirectional_attention=True` means every layer already attends in
#      both directions (no causal mask). The ONLY thing "sliding_attention"
#      layer_type adds on top of that is a window predicate:
#      `abs(q_idx - kv_idx) < sliding_window` (sliding_window=512), ORed with
#      the padding mask (see `_bidirectional_window_overlay` in
#      modeling_gemma3.py). For any input with sequence length <= 512 tokens,
#      `abs(q_idx - kv_idx) < seq_len <= 512` is true for EVERY pair of
#      positions in the sequence, so the window predicate is always satisfied
#      and the sliding-attention mask collapses to the SAME plain
#      padding-only bidirectional mask as the full-attention layers. This file
#      therefore uses one shared additive mask (padding only) for all layers.
#      This is exact, not approximate, for the s<=512 regime, which covers
#      every sentence-embedding workload this project has ever run (all
#      goldens use seq_len 64; sentence-transformers' own default max_seq for
#      this checkpoint is 2048, but typical sentences tokenize far under 512).
#      A genuine sliding-window mask (needed only if seq_len > 512) is NOT
#      implemented here -- flagged explicitly rather than silently wrong.
#
#  * GeGLU FFN: `down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))` -- gate and
#    up are SEPARATE [768,1152] matrices (not fused, unlike MiniLM's Q/K/V),
#    confirmed against the checkpoint's tensor inventory (no fused gate_up
#    tensor exists) and Gemma3MLP.forward.
#
#  * Activation is `gelu_pytorch_tanh` -- the TANH APPROXIMATION, not exact
#    erf like MiniLM/BERT. Using erf here would be the same class of bug M3's
#    encoder.py explicitly warns against for the opposite direction.
#
#  * Final norm: one more RMSNorm (`model.norm`) after all 24 layers, applied
#    to `last_hidden_state` -- not counted as one of the per-layer four.
#
#  * Sentence-transformers head, read from this checkpoint's own
#    modules.json/1_Pooling/2_Dense/3_Dense configs, not assumed:
#    mean pooling (attention-mask-weighted, include_prompt=true i.e. no
#    special-casing of the task-prefix tokens) -> Dense(768->3072, no bias,
#    identity activation) -> Dense(3072->768, no bias, identity activation) ->
#    L2 normalize. Both Dense layers are `nn.Linear(bias=False)`, confirmed by
#    reading their own config.json (`"bias": false`) and their
#    model.safetensors (only a `linear.weight` tensor, no `linear.bias`).
#
# Env: numpy only (runs in either .venv-ref or the iron env).

import math

import numpy as np

MASK_FILL = np.finfo(np.float32).min


def rms_norm(x, weight, eps=1e-6):
    """Gemma3RMSNorm: x/rms(x) * (1 + weight), reduction over the last axis.

    Computed in fp64 for the same reason encoder.py's layernorm() is: it is
    cheap insurance against a mean-of-squares reduction over 256-768 elements,
    and it is what encoder.py already does for the analogous BERT op.
    Real HF computes the norm in x.float() (fp32) regardless of input dtype;
    this checkpoint's activations are fp32 already, so upcasting further to
    fp64 here can only tighten the match, never loosen it.
    """
    x64 = x.astype(np.float64)
    var = (x64 * x64).mean(axis=-1, keepdims=True)
    normed = x64 * (1.0 / np.sqrt(var + eps))
    out = normed * (1.0 + weight.astype(np.float64))
    return out.astype(np.float32)


def gelu_tanh(x):
    """The tanh-approximate GELU (`gelu_pytorch_tanh`), NOT exact erf.

    0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))). Do not substitute the
    erf form used in reference/encoder.py -- that is the correct activation
    for MiniLM/BERT but the WRONG one here; EmbeddingGemma's own config names
    `gelu_pytorch_tanh` explicitly.
    """
    x64 = x.astype(np.float64)
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x64 + 0.044715 * x64 * x64 * x64)
    return (0.5 * x64 * (1.0 + np.tanh(inner))).astype(np.float32)


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
    """Standard RoPE cos/sin tables, matching Gemma3RotaryEmbedding.forward.

    inv_freq[j] = base ** (-2j/head_dim) for j in [0, head_dim/2).
    freqs[s,j] = s * inv_freq[j]; emb = concat(freqs, freqs) along the last
    axis (NOT interleaved) -- this is the "rotate_half" convention, matching
    `rotate_half()` in modeling_gemma3.py, not the interleaved convention some
    other RoPE implementations use. attention_scaling is 1.0 for "default"
    rope_type (both our layer types use it), so it is omitted.
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


class GemmaEmbeddingReference:
    """Gemma3TextModel forward (as used by EmbeddingGemma-300M) + the
    sentence-transformers head (mean pool -> Dense x2 -> normalize), in numpy.

    `w` is the Gemma3TextModel state dict (HF tensor names, no "model."
    prefix -- see the checkpoint inventory in tasks/0055). `dense_w` is
    {"2": [3072,768] weight, "3": [768,3072] weight} from 2_Dense/3_Dense.

    `gemm` is the (M,K)x(K,N)->(M,N) primitive, swappable exactly like
    encoder.py's MiniLMReference -- the seam a future bf16/NPU precision study
    would use.
    """

    def __init__(self, w, dense_w, num_layers=24, hidden=768, num_heads=3,
                 num_kv_heads=1, head_dim=256, intermediate=1152, eps=1e-6,
                 rope_theta_global=1e6, rope_theta_local=1e4,
                 sliding_window=512, sliding_window_pattern=6,
                 query_pre_attn_scalar=256, gemm=None):
        self.w = w
        self.dense_w = dense_w
        self.gemm = gemm or fp32_gemm
        self.L = num_layers
        self.hidden = hidden
        self.H = num_heads
        self.KVH = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.inter = intermediate
        self.eps = eps
        self.rope_theta_global = rope_theta_global
        self.rope_theta_local = rope_theta_local
        self.sliding_window = sliding_window          # unused -- see file header
        self.sliding_window_pattern = sliding_window_pattern
        self.scale = query_pre_attn_scalar ** -0.5
        self.embed_scale = math.sqrt(hidden)

    def is_full_attention_layer(self, i):
        # config.json's own construction: layer_types[i] == "full_attention"
        # iff (i+1) % sliding_window_pattern == 0.
        return (i + 1) % self.sliding_window_pattern == 0

    # -- primitives --------------------------------------------------------

    def linear(self, x, weight):
        """nn.Linear with bias=False: weight stored [out,in], y = x @ W.T."""
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
        word = self.w["embed_tokens.weight"][input_ids].astype(np.float32)
        tap("emb.word", word)
        scaled = (word.astype(np.float64) * self.embed_scale).astype(np.float32)
        tap("emb.scaled", scaled)
        return scaled

    def attention(self, i, x, add_mask, cos, sin, tap):
        B, S, _ = x.shape
        p = f"layers.{i}.self_attn."
        q = self.linear(x, self.w[p + "q_proj.weight"])       # [B,S,H*hd]
        k = self.linear(x, self.w[p + "k_proj.weight"])       # [B,S,KVH*hd]
        v = self.linear(x, self.w[p + "v_proj.weight"])
        tap(f"L{i}.q_proj", q)
        tap(f"L{i}.k_proj", k)
        tap(f"L{i}.v_proj", v)

        def heads(t, nh):
            return t.reshape(B, S, nh, self.head_dim).transpose(0, 2, 1, 3)
        q, k, v = heads(q, self.H), heads(k, self.KVH), heads(v, self.KVH)

        # q_norm / k_norm: RMSNorm over head_dim, BEFORE RoPE.
        q = rms_norm(q, self.w[p + "q_norm.weight"], self.eps)
        k = rms_norm(k, self.w[p + "k_norm.weight"], self.eps)
        tap(f"L{i}.q_normed", q)
        tap(f"L{i}.k_normed", k)

        q, k = apply_rope(q, k, cos, sin)
        tap(f"L{i}.q_rope", q)
        tap(f"L{i}.k_rope", k)

        # MQA: repeat the single KV head n_rep times to match Q's heads.
        k_rep = np.repeat(k, self.n_rep, axis=1)               # [B,H,S,hd]
        v_rep = np.repeat(v, self.n_rep, axis=1)

        scores = self.batched_gemm(q, np.ascontiguousarray(k_rep.transpose(0, 1, 3, 2)))
        scores = (scores.astype(np.float64) * self.scale).astype(np.float32)
        tap(f"L{i}.scores", scores)

        scores = scores + add_mask
        tap(f"L{i}.scores_masked", scores)

        probs = softmax(scores)
        tap(f"L{i}.probs", probs)

        ctx = self.batched_gemm(probs, v_rep)                  # [B,H,S,hd]
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, S, self.H * self.head_dim)
        tap(f"L{i}.ctx", ctx)

        out = self.linear(ctx, self.w[p + "o_proj.weight"])
        tap(f"L{i}.attn_out", out)
        return out

    def mlp(self, i, x, tap):
        p = f"layers.{i}.mlp."
        gate = self.linear(x, self.w[p + "gate_proj.weight"])
        up = self.linear(x, self.w[p + "up_proj.weight"])
        tap(f"L{i}.gate", gate)
        tap(f"L{i}.up", up)
        act = gelu_tanh(gate)
        tap(f"L{i}.act", act)
        prod = (act.astype(np.float64) * up.astype(np.float64)).astype(np.float32)
        tap(f"L{i}.geglu", prod)
        down = self.linear(prod, self.w[p + "down_proj.weight"])
        tap(f"L{i}.down", down)
        return down

    def layer(self, i, x, add_mask, cos, sin, tap):
        residual = x
        h = rms_norm(x, self.w[f"layers.{i}.input_layernorm.weight"], self.eps)
        tap(f"L{i}.ln_in", h)
        h = self.attention(i, h, add_mask, cos, sin, tap)
        h = rms_norm(h, self.w[f"layers.{i}.post_attention_layernorm.weight"], self.eps)
        tap(f"L{i}.ln_post_attn", h)
        x = residual + h
        tap(f"L{i}.resid1", x)

        residual = x
        h = rms_norm(x, self.w[f"layers.{i}.pre_feedforward_layernorm.weight"], self.eps)
        tap(f"L{i}.ln_pre_ffn", h)
        h = self.mlp(i, h, tap)
        h = rms_norm(h, self.w[f"layers.{i}.post_feedforward_layernorm.weight"], self.eps)
        tap(f"L{i}.ln_post_ffn", h)
        x = residual + h
        tap(f"L{i}.resid2", x)
        return x

    # -- forward -------------------------------------------------------------

    def encode(self, input_ids, attention_mask, taps=None):
        """Returns the L2-normalized sentence embedding [B, 768]."""
        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)
        S = input_ids.shape[1]

        def tap(name, value):
            if taps is not None:
                taps[name] = np.ascontiguousarray(value, dtype=np.float32)

        # Padding-only bidirectional mask, shared by every layer -- see the
        # file header for why this is exact (not approximate) at S <= 512.
        add_mask = (1.0 - attention_mask[:, None, None, :].astype(np.float32)) * MASK_FILL
        tap("attn_add_mask", add_mask)

        cos_g, sin_g = rope_cos_sin(S, self.head_dim, self.rope_theta_global)
        cos_l, sin_l = rope_cos_sin(S, self.head_dim, self.rope_theta_local)

        x = self.embed(input_ids, tap)
        for i in range(self.L):
            cos, sin = (cos_g, sin_g) if self.is_full_attention_layer(i) else (cos_l, sin_l)
            x = self.layer(i, x, add_mask, cos, sin, tap)

        x = rms_norm(x, self.w["norm.weight"], self.eps)
        tap("last_hidden_state", x)

        # Mean pooling, include_prompt=true -> plain attention-mask-weighted
        # mean, no special-casing of prefix tokens. Same 1e-9 denominator
        # clamp as encoder.py, matching sentence-transformers' own Pooling.
        m = attention_mask[:, :, None].astype(np.float32)
        summed = (x * m).sum(axis=1)
        denom = np.clip(m.sum(axis=1), 1e-9, None)
        pooled = summed / denom
        tap("pool.mean", pooled)

        # Dense(768->3072) -> Dense(3072->768), both bias=False, identity act.
        d2 = self.gemm(pooled, np.ascontiguousarray(self.dense_w["2"].T))
        tap("dense2", d2)
        d3 = self.gemm(d2, np.ascontiguousarray(self.dense_w["3"].T))
        tap("dense3", d3)

        norm = np.clip(np.linalg.norm(d3, axis=1, keepdims=True), 1e-12, None)
        out = (d3 / norm).astype(np.float32)
        tap("out.embedding", out)
        return out


def load_reference(model_dir):
    """Build the reference from models/embeddinggemma-300m/."""
    import json
    from pathlib import Path

    from safetensors_io import load

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    w, _ = load(model_dir / "model.safetensors")
    d2, _ = load(model_dir / "2_Dense" / "model.safetensors")
    d3, _ = load(model_dir / "3_Dense" / "model.safetensors")
    dense_w = {"2": d2["linear.weight"], "3": d3["linear.weight"]}
    return GemmaEmbeddingReference(
        w, dense_w,
        num_layers=cfg["num_hidden_layers"],
        hidden=cfg["hidden_size"],
        num_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate=cfg["intermediate_size"],
        eps=cfg["rms_norm_eps"],
        rope_theta_global=cfg["rope_theta"],
        rope_theta_local=cfg["rope_local_base_freq"],
        sliding_window=cfg["sliding_window"],
        sliding_window_pattern=cfg.get("_sliding_window_pattern", 6),
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
    )


# Task-prefix protocol EmbeddingGemma expects (config_sentence_transformers.json
# "prompts"). Applying the wrong one (or none) changes the embedding -- this is
# part of the model's contract, not an optional nicety.
PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "sts": "task: sentence similarity | query: ",
    "clustering": "task: clustering | query: ",
    "classification": "task: classification | query: ",
}
