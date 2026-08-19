# NpuEmbeddings -- M3: the reference encoder. Pure numpy, fp32, no torch.
#
# This is the ORACLE. Every NPU kernel in M5 is validated against the taps this
# produces, so it is written for obviousness over speed: one operation per line,
# no fused shortcuts, every numerical decision from docs/04-model/README.md
# ("Numerical landmines") spelled out where it happens.
#
# Deliberate design choices:
#
#  * numpy only -- it must run in the iron env, which stays clean of pip
#    installs. safetensors is read by our own reference/safetensors_io.py.
#  * QKV is computed FUSED, as one [384, 1152] GEMM, because that is the shape
#    M4 will bake and M5 will dispatch. The oracle should have the same seams as
#    the implementation, or the goldens validate a different program.
#  * 1/sqrt(head_dim) is NOT folded into Q here. M4 folds it offline; keeping the
#    reference canonical is what lets us prove the fold is exact.
#  * The pooler (pooler.dense) is not implemented. sentence-transformers never
#    calls it -- 147,840 dead params. See docs/04-model.
#
# Env: numpy only (runs in either .venv-ref or the iron env).

import math

import numpy as np

try:                                    # scipy is present in .venv-ref, absent in iron
    from scipy.special import erf as _erf

    def erf(x):
        return _erf(x)
except ImportError:
    _erf_uf = np.frompyfunc(math.erf, 1, 1)

    def erf(x):
        # math.erf is the C library erf: exact, and identical to what torch
        # computes. Slower than scipy but this is an oracle, not a kernel.
        return _erf_uf(x.astype(np.float64)).astype(np.float64)


# Applied to masked positions before softmax. HF uses
# (1.0 - mask) * finfo(dtype).min -- a large FINITE negative, never -inf, or a
# fully-masked row produces NaN instead of a uniform distribution.
MASK_FILL = np.finfo(np.float32).min


def layernorm(x, gamma, beta, eps=1e-12):
    """PyTorch LayerNorm over the last axis.

    Two landmines, both from docs/04-model:
      * eps goes INSIDE the sqrt: x / sqrt(var + eps), not sqrt(var) + eps.
      * PyTorch uses the BIASED variance estimator (divide by N, not N-1).
    Computed in fp64 because post-LN BERT has hidden dims carrying +/-50-100
    while the rest sit near +/-1; the mean/variance of that is worth protecting.
    """
    x = x.astype(np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    var = (centred * centred).mean(axis=-1, keepdims=True)     # biased: / N
    normed = centred / np.sqrt(var + eps)
    return (normed * gamma.astype(np.float64) + beta.astype(np.float64)).astype(np.float32)


def gelu(x):
    """Exact erf GELU: 0.5 * x * (1 + erf(x / sqrt(2))).

    NOT the tanh approximation. Substituting tanh introduces a ~1e-3
    systematically biased error that later shows up as an unexplainable
    relative-Frobenius floor. See docs/04-model.
    """
    x64 = x.astype(np.float64)
    return (0.5 * x64 * (1.0 + erf(x64 / math.sqrt(2.0)))).astype(np.float32)


def softmax(x):
    """Row softmax in fp64 with max subtraction."""
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def fp32_gemm(a, b):
    """The default GEMM: plain fp32. Signature is (M,K) x (K,N) -> (M,N)."""
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)


class MiniLMReference:
    """BertModel forward + mean pooling + L2 normalize, in numpy.

    `w` is the raw safetensors state dict (fp32 numpy arrays, HF names).

    `gemm` is the (M,K) x (K,N) -> (M,N) primitive used for every matmul that
    will run on the NPU -- the six per-layer GEMMs, including QK^T and A.V.
    Swapping it is how precision_study.py asks "what does bf16 or bfp16 cost on
    real activations?" without a second copy of the model. Everything NOT routed
    through it (bias add, LayerNorm, softmax, GELU, pooling) stays fp32 on
    purpose: docs/04-model requires LN and softmax in fp32, and biases stay fp32
    in the .npue format.
    """

    def __init__(self, w, num_layers=6, num_heads=12, eps=1e-12, gemm=None,
                 qkv_w=None, qkv_b=None, qk_scale=None, gelu_fn=None,
                 ln_fn=None, softmax_fn=None, pooling="mean"):
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling {pooling!r}: expected 'mean' or 'cls'")
        # MiniLM pools mean, bge pools CLS. Defaulting is safe only because
        # load_reference() below always passes the checkpoint's own answer.
        self.pooling = pooling
        self.w = w
        self.gemm = gemm or fp32_gemm
        # M6 hook: the activation, like the GEMM, is swappable, so the same
        # forward pass can run ops on the NPU. Defaults to the exact-erf oracle.
        self.gelu_fn = gelu_fn or gelu
        # Same idea as gelu_fn: the reduction ops are swappable so the one
        # forward pass can run them on the array. Signatures match the module
        # functions they default to.
        self.ln_fn = ln_fn or layernorm
        self.softmax_fn = softmax_fn or softmax
        self.L = num_layers
        self.H = num_heads
        self.eps = eps
        self.hidden = w["embeddings.word_embeddings.weight"].shape[1]
        self.head_dim = self.hidden // self.H
        # M4 hooks. A .npue file arrives with Q,K,V already fused and with
        # 1/sqrt(head_dim) already folded into Q, so it supplies its own fused
        # arrays and sets qk_scale=1.0 -- running the identical forward pass off
        # packed weights is what makes the fusions verifiable rather than
        # merely round-trippable.
        self.qk_scale = math.sqrt(self.head_dim) if qk_scale is None else qk_scale
        if qkv_w is not None:
            self.qkv_w, self.qkv_b = qkv_w, qkv_b
            return
        # Fuse Q,K,V once at construction -- this is the M4 offline fusion,
        # done here so the taps have the shapes the NPU will actually see.
        self.qkv_w = {}
        self.qkv_b = {}
        for i in range(self.L):
            p = f"encoder.layer.{i}.attention.self."
            self.qkv_w[i] = np.concatenate(
                [w[p + n + ".weight"] for n in ("query", "key", "value")], axis=0
            )                                                   # [1152, 384]
            self.qkv_b[i] = np.concatenate(
                [w[p + n + ".bias"] for n in ("query", "key", "value")], axis=0
            )                                                   # [1152]

    # -- primitives --------------------------------------------------------

    def linear(self, x, weight, bias):
        """nn.Linear: weight is stored [out, in] and computes y = x @ W.T + b.

        We want [K, N] for the NPU, so the transpose is a free offline op in M4.
        Here it is explicit, and the GEMM goes through self.gemm on a flattened
        [B*S, K] matrix -- the exact 2D shape the NPU kernel receives.
        """
        flat = x.reshape(-1, x.shape[-1])
        y = self.gemm(flat, np.ascontiguousarray(weight.T)).reshape(*x.shape[:-1], -1)
        return y + bias.astype(np.float32) if bias is not None else y

    def batched_gemm(self, a, b):
        """[B,H,M,K] x [B,H,K,N] -> [B,H,M,N], one self.gemm call per head.

        Looped rather than using numpy broadcasting because each head IS a
        separate NPU dispatch, and a precision model must see the same
        blocking the hardware would.
        """
        B, H = a.shape[0], a.shape[1]
        out = np.empty((B, H, a.shape[2], b.shape[3]), dtype=np.float32)
        for i in range(B):
            for j in range(H):
                out[i, j] = self.gemm(a[i, j], b[i, j])
        return out

    # -- pieces ------------------------------------------------------------

    def embed(self, input_ids, token_type_ids, tap):
        w = self.w
        word = w["embeddings.word_embeddings.weight"][input_ids]
        S = input_ids.shape[1]
        # position_ids are just arange -- the stored embeddings.position_ids
        # tensor is a buffer, not a parameter. docs/04-model, sharp edge 3.
        pos = w["embeddings.position_embeddings.weight"][:S][None, :, :]
        typ = w["embeddings.token_type_embeddings.weight"][token_type_ids]
        tap("emb.word", word)
        tap("emb.pos", np.broadcast_to(pos, word.shape))
        tap("emb.token_type", typ)
        summed = (word + pos + typ).astype(np.float32)
        tap("emb.sum", summed)
        out = self.ln_fn(summed, w["embeddings.LayerNorm.weight"],
                        w["embeddings.LayerNorm.bias"], self.eps)
        tap("emb.ln", out)
        return out

    def attention(self, i, x, add_mask, tap):
        B, S, _ = x.shape
        qkv = self.linear(x, self.qkv_w[i], self.qkv_b[i])      # [B,S,1152]
        tap(f"L{i}.qkv", qkv)

        q, k, v = np.split(qkv, 3, axis=-1)                     # 3 x [B,S,384]
        # [B,S,384] -> [B,H,S,head_dim]
        def heads(t):
            return t.reshape(B, S, self.H, self.head_dim).transpose(0, 2, 1, 3)
        q, k, v = heads(q), heads(k), heads(v)

        # Scale by 1/sqrt(head_dim). M4 folds this into the Q weight+bias;
        # keeping it explicit here is what makes that fold verifiable.
        scores = self.batched_gemm(q, np.ascontiguousarray(k.transpose(0, 1, 3, 2)))
        scores = scores / self.qk_scale         # == 1.0 when M4 folded it into Q
        tap(f"L{i}.scores", scores)

        scores = scores + add_mask                              # fp32, finite min
        tap(f"L{i}.scores_masked", scores)

        probs = self.softmax_fn(scores)
        tap(f"L{i}.probs", probs)

        ctx = self.batched_gemm(probs, v)                       # [B,H,S,hd]
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, S, self.hidden)
        tap(f"L{i}.ctx", ctx)

        p = f"encoder.layer.{i}.attention.output."
        proj = self.linear(ctx, self.w[p + "dense.weight"], self.w[p + "dense.bias"])
        tap(f"L{i}.attn_proj", proj)

        # post-LN: normalize AFTER the residual add
        out = self.ln_fn(proj + x, self.w[p + "LayerNorm.weight"],
                        self.w[p + "LayerNorm.bias"], self.eps)
        tap(f"L{i}.ln1", out)
        return out

    def ffn(self, i, x, tap):
        pre = f"encoder.layer.{i}."
        up = self.linear(x, self.w[pre + "intermediate.dense.weight"],
                         self.w[pre + "intermediate.dense.bias"])    # [B,S,1536]
        tap(f"L{i}.ffn_up", up)
        act = self.gelu_fn(up)
        tap(f"L{i}.gelu", act)
        down = self.linear(act, self.w[pre + "output.dense.weight"],
                           self.w[pre + "output.dense.bias"])        # [B,S,384]
        tap(f"L{i}.ffn_down", down)
        out = self.ln_fn(down + x, self.w[pre + "output.LayerNorm.weight"],
                        self.w[pre + "output.LayerNorm.bias"], self.eps)
        tap(f"L{i}.ln2", out)
        return out

    # -- forward -----------------------------------------------------------

    def encode(self, input_ids, attention_mask, token_type_ids=None, taps=None):
        """Returns the L2-normalized mean-pooled sentence embedding [B, 384].

        `taps`, if a dict, is filled with every named intermediate.
        """
        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)

        def tap(name, value):
            if taps is not None:
                taps[name] = np.ascontiguousarray(value, dtype=np.float32)

        # HF's extended attention mask: [B,S] -> [B,1,1,S] additive, fp32.
        add_mask = (1.0 - attention_mask[:, None, None, :].astype(np.float32)) * MASK_FILL
        tap("attn_add_mask", add_mask)

        x = self.embed(input_ids, token_type_ids, tap)
        for i in range(self.L):
            x = self.attention(i, x, add_mask, tap)
            x = self.ffn(i, x, tap)
        tap("last_hidden_state", x)

        if self.pooling == "cls":
            # [CLS] is position 0 by construction; sentence-transformers takes
            # it without consulting the mask, and so do we.
            pooled = x[:, 0, :]
        else:
            # Mean pooling over non-pad tokens. The denominator is clamped at
            # 1e-9 to match sentence-transformers exactly (docs/04-model).
            m = attention_mask[:, :, None].astype(np.float32)
            summed = (x * m).sum(axis=1)
            denom = np.clip(m.sum(axis=1), 1e-9, None)
            pooled = summed / denom
        # The tap keeps its name across modes so golden files stay structurally
        # identical between models; pool.mode says which produced it, so a
        # golden cannot be read as the wrong kind.
        tap("pool.mean", pooled)

        norm = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        out = (pooled / norm).astype(np.float32)
        tap("out.embedding", out)
        return out


def load_reference(model_dir, num_layers=6, num_heads=12, eps=1e-12):
    """Build the reference from a checkpoint directory (models/<name>/)."""
    import json
    from pathlib import Path

    from safetensors_io import load

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    w, _ = load(model_dir / "model.safetensors")
    return MiniLMReference(
        w,
        num_layers=cfg.get("num_hidden_layers", num_layers),
        num_heads=cfg.get("num_attention_heads", num_heads),
        eps=cfg.get("layer_norm_eps", eps),
        pooling=read_pooling(model_dir),
    )


def read_pooling(model_dir):
    """Which pooling this checkpoint uses, from its own metadata.

    The same 1_Pooling/config.json both packers read, so the oracle and the
    .npue cannot disagree about it. Ambiguity refuses rather than guessing:
    max and mean_sqrt_len pooling are not implemented here, and approximating
    either with mean would be a silent quality loss.
    """
    import json
    from pathlib import Path

    p = Path(model_dir) / "1_Pooling" / "config.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- cannot tell whether this checkpoint pools by "
            f"mean or by CLS. Re-fetch with reference/fetch_model.py.")
    c = json.loads(p.read_text(encoding="utf-8"))
    modes = [k for k, v in c.items()
             if k.startswith("pooling_mode_") and v is True]
    if modes == ["pooling_mode_cls_token"]:
        return "cls"
    if modes == ["pooling_mode_mean_tokens"]:
        return "mean"
    raise ValueError(f"{p}: this reference implements cls and mean pooling; "
                     f"the checkpoint asks for {modes or 'nothing'}")
