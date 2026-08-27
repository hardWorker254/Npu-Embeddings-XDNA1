# NpuEmbeddings -- 0.5.0 (tasks/0134, reference half): gte-multilingual-base
# reference encoder. Pure numpy, no torch -- same discipline as
# reference/encoder.py (MiniLM), encoder_gemma.py and encoder_nomic.py: one
# operation per line, every numerical decision spelled out where it happens
# and cross-checked against the real checkpoint via
# tasks/0134-gte-oracle/probe_gte_arch.py (per-layer comparison against the
# trust_remote_code NewModel, WITH negative controls), not assumed from the
# model card.
#
# THIS IS THE arch=3 CANDIDATE. Not yet wired into tools/npue.py, the packer
# or the C++ runtime -- that is later 0.5.0 work. Architecture, read verbatim
# out of Alibaba-NLP/new-impl `modeling.py` (fetched 2026-08-27) and confirmed
# by the probe:
#
#  * hidden 768, 12 layers, 12 heads, head_dim 64, intermediate 3072,
#    vocab_size 250,048 (250,002 real XLM-R pieces + padding rows),
#    type_vocab_size 1, layer_norm_eps 1e-12, cls pooling.
#
#  * Embedding: word_embeddings[input_ids] + token_type_embeddings[0] (added
#    unconditionally -- type_vocab_size is 1 so every token is type 0), then
#    `embeddings.LayerNorm`. NO absolute position table: with
#    position_embedding_type == "rope" the table is never created; position
#    information enters entirely through RoPE inside attention.
#
#  * Each layer is POST-LN, exactly nomic's shape:
#        h = attn_ln(h + attn(h))
#        h = mlp_ln(h + mlp(h))
#    and there is NO final norm after layer 12 -- `NewModel.forward` returns
#    the last layer's `mlp_ln` output directly.
#
#  * Attention: `qkv_proj` is ONE fused [2304, 768] weight WITH bias (unlike
#    nomic, which is bias-free -- `pack_qkv: true`, `bias=True` read from
#    `NewAttention.__init__`), split THREE-MAJOR:
#    `.split(all_head_size, dim=-1)` gives rows [0:768]=Q, [768:1536]=K,
#    [1536:2304]=V, each then reshaped (heads, head_dim). `o_proj` also
#    carries a bias. Scale is 1/sqrt(head_dim) = 1/8 (the config's
#    `logn_attention_scale` is false, so the log-n branch is dead).
#
#  * RoPE -- THE TRAP THIS FILE EXISTS TO DEFUSE (tasks/0068 measured that a
#    wrong theta is *silently* wrong). The config says
#    `rope_theta: 20000, rope_scaling: {type: ntk, factor: 8.0}` and the
#    obvious readings are ALL WRONG:
#      - plain base 20000:      wrong (the NTK cache overwrites it);
#      - plain base 160000:     wrong (misses the constant correction);
#      - any single "effective theta": IMPOSSIBLE -- the correction is a
#        constant factor, not a power law.
#    What `NTKScalingRotaryEmbedding` actually does, traced through its
#    constructor: the parent builds a plain-base cache for 8192 positions;
#    the child then calls `_set_cos_sin_cache(8192 * 8)`, and because
#    `self.max_position_embeddings` still reads 8192, `seq_len (65536) >
#    8192` takes the NTK branch, which REGISTERS a new inv_freq that every
#    later forward (any seq_len <= 65536, i.e. all of ours) uses:
#        base     = 20000 * 8 = 160000            # mixed_b is None
#        inv_freq = 1 / base**(2i/64)             # i = 0..31
#        inv_freq = inv_freq / 8**(2/64)          # the constant correction
#    So even frequency i=0 is scaled: inv_freq[0] = 8**(-1/32) = 0.93714...,
#    not 1.0. This file computes exactly that, in float32 like torch does.
#    Convention is NeoX (concat(freqs, freqs), rotate_half = concat(-x2, x1)),
#    applied to Q and K only, positions from 0.
#
#  * MLP is a GATED GLU with TWO matrices (Gemma's shape, nomic's ordering,
#    GELU's activation):
#        up_gate  = up_gate_proj(x)               # [768 -> 6144], NO bias
#        up, gate = split(up_gate, 3072)          # up FIRST, gate SECOND
#        h        = gelu(gate) * up               # activation on the GATE
#        out      = down_proj(h)                  # [3072 -> 768], WITH bias
#    `hidden_act: "gelu"` is torch's default exact (erf) GELU, NOT the tanh
#    approximation Gemma uses.
#
#  * Pooling: CLS (1_Pooling/config.json: pooling_mode_cls_token). encode()
#    returns both the raw CLS vector and the L2-normalized one, explicitly
#    labelled, same as encoder_nomic.py and for the same reason.
#
# Env: numpy only (runs in either .venv-ref or the iron env).

import math

import numpy as np


# --- primitives -- copied (not imported) from encoder_nomic.py, matching its
# --- own choice that arch files stand alone.

def layer_norm(x, weight, bias, eps=1e-12):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def gelu_exact(x):
    # torch.nn.functional.gelu default: 0.5 * x * (1 + erf(x / sqrt(2))).
    # NOT the tanh approximation -- that is Gemma's gelu_pytorch_tanh.
    from scipy.special import erf as _erf  # scipy present in .venv-ref
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def gelu_exact_no_scipy(x):
    # math.erf fallback so the iron env (no scipy) can run this file too.
    v = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + v(x / math.sqrt(2.0)))


def softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def fp32_gemm(a, b):
    return a.astype(np.float32) @ b.astype(np.float32)


def gte_inv_freq(head_dim=64, base=20000.0, factor=8.0):
    """The NTK frequencies gte actually runs with -- see the header. float32
    throughout, matching torch.arange(...).float() in the real module."""
    i = np.arange(0, head_dim, 2, dtype=np.float32)
    scaled_base = np.float32(base * factor)
    inv_freq = np.float32(1.0) / (scaled_base ** (i / np.float32(head_dim)))
    inv_freq = inv_freq / np.float32(factor) ** (np.float32(2.0) / np.float32(head_dim))
    return inv_freq.astype(np.float32)


def rope_cos_sin(seq_len, head_dim=64, base=20000.0, factor=8.0):
    inv_freq = gte_inv_freq(head_dim, base, factor)
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, cos, sin):
    # q, k: [seq, heads, head_dim]; cos/sin: [seq, head_dim]
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]
    q2 = q * cos_b + rotate_half(q) * sin_b
    k2 = k * cos_b + rotate_half(k) * sin_b
    return q2, k2


class GteEncoder:
    """weights: dict name -> np.float32 array, state-dict names WITHOUT the
    leading 'new.' (the probe strips it)."""

    def __init__(self, w, num_layers=12, hidden=768, num_heads=12,
                 head_dim=64, intermediate=3072, eps=1e-12,
                 rope_theta=20000.0, rope_factor=8.0):
        self.w = w
        self.L = num_layers
        self.hidden = hidden
        self.heads = num_heads
        self.hd = head_dim
        self.inter = intermediate
        self.eps = eps
        self.theta = rope_theta
        self.factor = rope_factor
        self._taps = None
        try:
            import scipy  # noqa: F401
            self.gelu = gelu_exact
        except ImportError:
            self.gelu = gelu_exact_no_scipy

    def linear(self, x, weight, bias=None):
        y = fp32_gemm(x, weight.T)
        if bias is not None:
            y = y + bias
        return y

    def _tap(self, name, value):
        # Mirrors encoder_nomic.py's taps mechanism: fixture generation
        # (tools/export_validation.py) needs the same intermediates the C++
        # runtime checks itself against. No-op unless encode(..., taps=)
        # was given a dict.
        if self._taps is not None:
            self._taps[name] = np.ascontiguousarray(value, dtype=np.float32)

    def embed(self, input_ids):
        w = self.w
        h = w["embeddings.word_embeddings.weight"][input_ids]
        # type_vocab_size == 1: every token is type 0, added unconditionally.
        h = h + w["embeddings.token_type_embeddings.weight"][0]
        self._tap("emb.sum", h)
        h = layer_norm(h, w["embeddings.LayerNorm.weight"],
                       w["embeddings.LayerNorm.bias"], self.eps)
        self._tap("emb.ln", h)
        return h.astype(np.float32)

    def attention(self, h, cos, sin, mask):
        w = self.w
        p = self.pfx
        qkv = self.linear(h, w[p + "attention.qkv_proj.weight"],
                          w[p + "attention.qkv_proj.bias"])
        if self.pfx == "encoder.layer.0.":
            # UNFOLDED qkv (the oracle's weights carry no scale fold);
            # export_validation.py applies the container's fold itself.
            self._tap("L0.qkv", qkv)
        S = h.shape[0]
        # THREE-MAJOR split, then (heads, head_dim) -- see header.
        q = qkv[:, 0 * self.hidden:1 * self.hidden].reshape(S, self.heads, self.hd)
        k = qkv[:, 1 * self.hidden:2 * self.hidden].reshape(S, self.heads, self.hd)
        v = qkv[:, 2 * self.hidden:3 * self.hidden].reshape(S, self.heads, self.hd)
        q, k = apply_rope(q, k, cos, sin)
        q = q.transpose(1, 0, 2)                 # [heads, S, hd]
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        scores = np.einsum("hqd,hkd->hqk", q, k) / math.sqrt(self.hd)
        scores = scores + mask[None, None, :]
        probs = softmax(scores)
        ctx = np.einsum("hqk,hkd->hqd", probs, v)
        ctx = ctx.transpose(1, 0, 2).reshape(S, self.hidden)
        return self.linear(ctx, w[p + "attention.o_proj.weight"],
                           w[p + "attention.o_proj.bias"])

    def mlp(self, h):
        w = self.w
        p = self.pfx
        up_gate = self.linear(h, w[p + "mlp.up_gate_proj.weight"])  # no bias
        up = up_gate[:, :self.inter]
        gate = up_gate[:, self.inter:]
        act = self.gelu(gate) * up               # GELU on the GATE half
        return self.linear(act, w[p + "mlp.down_proj.weight"],
                           w[p + "mlp.down_proj.bias"])

    def layer(self, h, i, cos, sin, mask):
        w = self.w
        self.pfx = f"encoder.layer.{i}."
        p = self.pfx
        a = self.attention(h, cos, sin, mask)
        h = layer_norm(h + a, w[p + "attn_ln.weight"], w[p + "attn_ln.bias"],
                       self.eps)
        m = self.mlp(h)
        h = layer_norm(h + m, w[p + "mlp_ln.weight"], w[p + "mlp_ln.bias"],
                       self.eps)
        return h

    def encode(self, input_ids, attention_mask=None, taps=None):
        """input_ids: [seq] int array (one sequence). Returns dict with
        last_hidden, cls_raw, cls_normalized -- pick explicitly. `taps`,
        if a dict, receives emb.sum / emb.ln / L0.qkv (per sequence)."""
        self._taps = taps
        input_ids = np.asarray(input_ids)
        S = len(input_ids)
        if attention_mask is None:
            attention_mask = np.ones(S, dtype=np.float32)
        # additive mask, -inf on padding
        mask = np.where(attention_mask > 0, 0.0, -1e9).astype(np.float32)
        cos, sin = rope_cos_sin(S, self.hd, self.theta, self.factor)
        h = self.embed(input_ids)
        for i in range(self.L):
            h = self.layer(h, i, cos, sin, mask)
        cls = h[0]
        return {
            "last_hidden": h,
            "cls_raw": cls,
            "cls_normalized": cls / np.linalg.norm(cls),
        }
