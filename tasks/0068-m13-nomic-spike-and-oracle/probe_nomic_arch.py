"""
tasks/0068-m13-nomic-spike-and-oracle/probe_nomic_arch.py

Empirical arch spike for nomic-ai/nomic-embed-text-v1.5 (arch=2 candidate).

Answers, EMPIRICALLY (numpy fp64 reference vs a real forward-hooked HF model),
not by reading code alone:

  Q1. Is the live block ordering post-LN or pre-LN?
  Q2. Which projection (fc11 or fc12) does SiLU land on?
  Q3. Is the gated-MLP's internal `self.norm` an identity for this checkpoint?

Plus the "ALSO REPORT" cheap facts (RoPE convention, Wqkv row ordering,
token_type_embeddings, padding rows, attention scale, sentence-transformers
config contents, L2 normalization).

Every claim below is backed by a printed number. Do not trust prose alone.
"""

import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = "models/nomic-embed-text-v1.5"
SAFETENSORS_PATH = f"{MODEL_DIR}/model.safetensors"

torch.manual_seed(0)
np.random.seed(0)


def hr(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def relfro(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def maxabs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def report(name, a, b):
    ma = maxabs(a, b)
    rf = relfro(a, b)
    print(f"  {name:45s} max_abs={ma:.6e}  rel_fro={rf:.6e}")
    return ma, rf


# ---------------------------------------------------------------------------
# 0. Load everything
# ---------------------------------------------------------------------------
hr("0. LOAD: tokenizer, real HF model (trust_remote_code, the ORIGINAL nomic "
   "modeling_hf_nomic_bert.py), and raw safetensors weights (independent read)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("tokenizer class:", type(tokenizer).__name__)
print("tokenizer vocab_size (len of vocab.txt as loaded):", tokenizer.vocab_size)

with open(f"{MODEL_DIR}/config.json") as f:
    cfg_json = json.load(f)
print("config prenorm:", cfg_json["prenorm"])
print("config activation_function:", cfg_json["activation_function"], " hidden_act:", cfg_json["hidden_act"])
print("config rotary_emb_base:", cfg_json["rotary_emb_base"], " rotary_emb_fraction:", cfg_json["rotary_emb_fraction"],
      " rotary_emb_interleaved:", cfg_json["rotary_emb_interleaved"])
print("config head_dim:", cfg_json["head_dim"], " num_attention_heads:", cfg_json["num_attention_heads"])
print("config layer_norm_epsilon:", cfg_json["layer_norm_epsilon"])
print("config scale_attn_weights:", cfg_json["scale_attn_weights"],
      " scale_attn_by_inverse_layer_idx:", cfg_json["scale_attn_by_inverse_layer_idx"])
print("config type_vocab_size:", cfg_json["type_vocab_size"], " vocab_size:", cfg_json["vocab_size"])
print("config mlp_fc1_bias:", cfg_json["mlp_fc1_bias"], " mlp_fc2_bias:", cfg_json["mlp_fc2_bias"],
      " qkv_proj_bias:", cfg_json["qkv_proj_bias"])
print("config has 'norm_mlp' key:", "norm_mlp" in cfg_json, "(absence => getattr default False => nn.Identity)")

model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
model.eval()
print("\nreal model class:", type(model).__name__)
print("real model module file:", sys.modules[type(model).__module__].__file__)
print("real model.encoder.layers[0].mlp class:", type(model.encoder.layers[0].mlp).__name__)
print("real model.encoder.layers[0].mlp.norm:", model.encoder.layers[0].mlp.norm)
print("real model.encoder.layers[0].attn.norm_factor (attention scale denominator):",
      float(model.encoder.layers[0].attn.norm_factor))
print("real model.encoder.layers[11].attn.norm_factor (layer 11, checking layer-idx dependence):",
      float(model.encoder.layers[11].attn.norm_factor))
print("real model.encoder.layers[0].attn.rotary_emb.base:", model.encoder.layers[0].attn.rotary_emb.base)
print("real model.encoder.layers[0].attn.rotary_emb.interleaved:", model.encoder.layers[0].attn.rotary_emb.interleaved)
print("real model.encoder.layers[0].attn.rotary_emb.dim:", model.encoder.layers[0].attn.rotary_emb.dim,
      "(head_dim=64, rotary_emb_fraction=1.0 => full head rotated)")
print("real model.embeddings has position_embeddings attr:",
      hasattr(model.embeddings, "position_embeddings"),
      "(max_position_embeddings forced to 0 when rotary_emb_fraction>0, per source)")

# Raw safetensors, independent of the transformers loader / any key-renaming.
raw = {}
with safe_open(SAFETENSORS_PATH, framework="pt") as f:
    for k in f.keys():
        raw[k] = f.get_tensor(k).to(torch.float64).numpy()
print(f"\nraw safetensors: {len(raw)} tensors loaded independently via safetensors.safe_open")
assert len(raw) == 112, f"expected 112 tensors, got {len(raw)}"
for k in ["encoder.layers.0.attn.Wqkv.weight", "encoder.layers.0.attn.out_proj.weight",
          "encoder.layers.0.mlp.fc11.weight", "encoder.layers.0.mlp.fc12.weight",
          "encoder.layers.0.mlp.fc2.weight", "encoder.layers.0.norm1.weight",
          "encoder.layers.0.norm2.weight", "embeddings.word_embeddings.weight",
          "embeddings.token_type_embeddings.weight", "emb_ln.weight"]:
    print(f"  {k:45s} shape={raw[k].shape}")

H = 768
NHEAD = 12
HD = 64
EPS = 1e-12
THETA = 1000.0


# ---------------------------------------------------------------------------
# 1. Real forward pass with hooks capturing layer-0 boundary tensors
# ---------------------------------------------------------------------------
hr("1. REAL FORWARD PASS with hooks on layers[0], layers[0].attn, layers[0].mlp, embeddings")

SENTENCE = "search_document: The quick brown fox jumps over the lazy dog near the river bank."
enc = tokenizer(SENTENCE, return_tensors="pt")
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]
print("sentence:", repr(SENTENCE))
print("input_ids:", input_ids.tolist())
print("seq_len:", input_ids.shape[1], " attention_mask all ones:", bool(attention_mask.all()))

captured = {}


def save(name):
    def _hook(module, inp, out):
        captured[name] = (inp, out)
    return _hook


h_emb = model.embeddings.register_forward_hook(save("embeddings"))
h_layer0 = model.encoder.layers[0].register_forward_hook(save("layer0"))
h_attn0 = model.encoder.layers[0].attn.register_forward_hook(save("attn0"))
h_mlp0 = model.encoder.layers[0].mlp.register_forward_hook(save("mlp0"))

with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask)

for h in (h_emb, h_layer0, h_attn0, h_mlp0):
    h.remove()

# embeddings module: forward(input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None)
emb_out_real = captured["embeddings"][1].detach().to(torch.float64).numpy()[0]  # (seq, 768)
print("captured embeddings module output (word_emb+token_type_emb, pre emb_ln):", emb_out_real.shape)

# layer0: forward(hidden_states, hidden_states2, residual, attention_mask, position_ids, ...)
layer0_in_args = captured["layer0"][0]
h_in_real = layer0_in_args[0].detach().to(torch.float64).numpy()[0]  # (seq, 768) -- input to encoder.layers[0]
layer0_out = captured["layer0"][1]
h_out_real = layer0_out[0].detach().to(torch.float64).numpy()[0]  # (seq, 768) -- output of encoder.layers[0]
print("captured layer0 input hidden_states (= emb_ln(embeddings)):", h_in_real.shape)
print("captured layer0 output hidden_states:", h_out_real.shape)

# attn0: forward(hidden_states, attention_mask, position_ids, ...) -> attn_output (post out_proj, pre residual/norm1)
attn0_in_args = captured["attn0"][0]
attn_in_real = attn0_in_args[0].detach().to(torch.float64).numpy()[0]
attn_out_real = captured["attn0"][1].detach().to(torch.float64).numpy()[0]
print("captured attn0 input (should equal layer0 input):", attn_in_real.shape,
      " max_abs vs h_in_real:", maxabs(attn_in_real, h_in_real))
print("captured attn0 output (post out_proj, pre residual+norm1):", attn_out_real.shape)

# mlp0: forward(x) -> down_proj output (pre residual/norm2). x = post_attention_layernorm(h)
mlp0_in_args = captured["mlp0"][0]
mlp_in_real = mlp0_in_args[0].detach().to(torch.float64).numpy()[0]
mlp_out_real = captured["mlp0"][1].detach().to(torch.float64).numpy()[0]
print("captured mlp0 input (= norm1(attn_out + h_in)):", mlp_in_real.shape)
print("captured mlp0 output (down_proj result, pre residual+norm2):", mlp_out_real.shape)

# ground-truth RoPE cos/sin cache the real model actually used (buffers populated post-forward)
cos_real = model.encoder.layers[0].attn.rotary_emb._cos_cached.detach().to(torch.float64).numpy()
sin_real = model.encoder.layers[0].attn.rotary_emb._sin_cached.detach().to(torch.float64).numpy()
print("captured real rotary_emb cos/sin cache shape:", cos_real.shape, sin_real.shape,
      "(seq_len, head_dim/2)")

seq_len = input_ids.shape[1]


# ---------------------------------------------------------------------------
# 2. numpy fp64 primitives
# ---------------------------------------------------------------------------
def layernorm_np(x, w, b, eps=EPS):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def silu_np(x):
    return x / (1.0 + np.exp(-x))


def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def build_rope_cos_sin(seq_len, head_dim, theta):
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))  # (32,)
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)  # (seq, 32)
    return np.cos(freqs), np.sin(freqs)  # each (seq, head_dim/2)


def rotate_half_neox(x):
    # x: (..., d). NeoX-style: split into first/second half, [-x2, x1].
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def rotate_half_gptj(x):
    # GPT-J-style (interleaved): rotate even/odd pairs.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = -x2
    out[..., 1::2] = x1
    return out


def apply_rope(x, cos_half, sin_half, interleaved):
    # x: (heads, seq, head_dim); cos_half/sin_half: (seq, head_dim/2)
    if not interleaved:
        cos = np.concatenate([cos_half, cos_half], axis=-1)  # (seq, head_dim)
        sin = np.concatenate([sin_half, sin_half], axis=-1)
        rot = rotate_half_neox(x)
    else:
        # repeat each element twice: d -> (d 2)
        cos = np.repeat(cos_half, 2, axis=-1)
        sin = np.repeat(sin_half, 2, axis=-1)
        rot = rotate_half_gptj(x)
    return x * cos[None, :, :] + rot * sin[None, :, :]


def attention_numpy(h_in, Wqkv, Wo, *, wqkv_split="three_major", rope_interleaved=False,
                     rope_theta=THETA, scale=None, rotate_v=False):
    """
    h_in: (seq, 768) fp64
    Wqkv: (2304, 768) fp64 raw weight (nn.Linear weight, out_features x in_features)
    Wo:   (768, 768)
    Returns attn output after out_proj, pre-residual. (seq, 768)
    """
    seq = h_in.shape[0]
    qkv = h_in @ Wqkv.T  # (seq, 2304)

    if wqkv_split == "three_major":
        # rows: [0:768]=Q (h-major,d-minor within block), [768:1536]=K, [1536:2304]=V
        qkv_r = qkv.reshape(seq, 3, NHEAD, HD)  # three-major: matches einops "(three h d)"
        q = qkv_r[:, 0].transpose(1, 0, 2)  # (heads, seq, hd)
        k = qkv_r[:, 1].transpose(1, 0, 2)
        v = qkv_r[:, 2].transpose(1, 0, 2)
    elif wqkv_split == "head_major":
        # WRONG-candidate split: rows grouped per head as [head0:Q,K,V | head1:Q,K,V | ...]
        qkv_r = qkv.reshape(seq, NHEAD, 3, HD)
        q = qkv_r[:, :, 0].transpose(1, 0, 2)
        k = qkv_r[:, :, 1].transpose(1, 0, 2)
        v = qkv_r[:, :, 2].transpose(1, 0, 2)
    else:
        raise ValueError(wqkv_split)

    cos_half, sin_half = build_rope_cos_sin(seq, HD, rope_theta)
    q = apply_rope(q, cos_half, sin_half, rope_interleaved)
    k = apply_rope(k, cos_half, sin_half, rope_interleaved)
    if rotate_v:
        v = apply_rope(v, cos_half, sin_half, rope_interleaved)

    if scale is None:
        scale = 1.0 / math.sqrt(HD)
    scores = np.einsum("hqd,hkd->hqk", q, k) * scale
    probs = softmax_np(scores, axis=-1)
    ctx = np.einsum("hqk,hkd->hqd", probs, v)  # (heads, seq, hd)
    ctx = ctx.transpose(1, 0, 2).reshape(seq, NHEAD * HD)  # (seq, 768), h-major d-minor -> matches out_proj input layout
    return ctx @ Wo.T


def mlp_numpy(x, Wfc11, Wfc12, Wfc2, *, silu_on="fc12"):
    """
    x: (seq, 768). Wfc11/Wfc12: (3072, 768). Wfc2: (768, 3072).
    silu_on='fc12' => Candidate A: fc11(x) * silu(fc12(x))   [matches HF conversion_mapping.py]
    silu_on='fc11' => Candidate B: silu(fc11(x)) * fc12(x)
    """
    y11 = x @ Wfc11.T
    y12 = x @ Wfc12.T
    if silu_on == "fc12":
        gated = y11 * silu_np(y12)
    elif silu_on == "fc11":
        gated = silu_np(y11) * y12
    else:
        raise ValueError(silu_on)
    return gated @ Wfc2.T


# ---------------------------------------------------------------------------
# 3. RoPE cos/sin sanity: numpy build vs real model's cached buffers
# ---------------------------------------------------------------------------
hr("3. RoPE cos/sin table: numpy build (theta=1000, dim=64) vs real model's own cache")
cos_np, sin_np = build_rope_cos_sin(seq_len, HD, THETA)
report("cos (theta=1000)", cos_np, cos_real)
report("sin (theta=1000)", sin_np, sin_real)
cos_wrong, sin_wrong = build_rope_cos_sin(seq_len, HD, 10000.0)
print("  NEGATIVE CONTROL theta=10000 (wrong):")
report("cos (theta=10000, WRONG)", cos_wrong, cos_real)


# ---------------------------------------------------------------------------
# 4. Attention: correct config vs each wrong candidate, all against the REAL
#    captured attn0 output (post out_proj, pre-residual).
# ---------------------------------------------------------------------------
hr("4. ATTENTION discriminators (real captured attn0 output is ground truth)")

Wqkv0 = raw["encoder.layers.0.attn.Wqkv.weight"]
Wo0 = raw["encoder.layers.0.attn.out_proj.weight"]

print("\n[correct config: three-major split, non-interleaved RoPE, scale=1/sqrt(64), RoPE on Q,K only]")
attn_correct = attention_numpy(h_in_real, Wqkv0, Wo0, wqkv_split="three_major",
                                rope_interleaved=False, scale=1.0 / math.sqrt(HD), rotate_v=False)
report("attn_out (ALL CORRECT)", attn_correct, attn_out_real)

print("\n[NEGATIVE CONTROL: head-major Wqkv split instead of three-major]")
attn_wrong_split = attention_numpy(h_in_real, Wqkv0, Wo0, wqkv_split="head_major",
                                    rope_interleaved=False, scale=1.0 / math.sqrt(HD), rotate_v=False)
report("attn_out (WRONG: head-major split)", attn_wrong_split, attn_out_real)

print("\n[NEGATIVE CONTROL: interleaved (GPT-J style) RoPE instead of NeoX-style concat(freqs,freqs)]")
attn_wrong_rope = attention_numpy(h_in_real, Wqkv0, Wo0, wqkv_split="three_major",
                                   rope_interleaved=True, scale=1.0 / math.sqrt(HD), rotate_v=False)
report("attn_out (WRONG: interleaved RoPE)", attn_wrong_rope, attn_out_real)

print("\n[NEGATIVE CONTROL: RoPE also applied to V]")
attn_wrong_v = attention_numpy(h_in_real, Wqkv0, Wo0, wqkv_split="three_major",
                                rope_interleaved=False, scale=1.0 / math.sqrt(HD), rotate_v=True)
report("attn_out (WRONG: RoPE applied to V too)", attn_wrong_v, attn_out_real)

print("\n[NEGATIVE CONTROL: wrong attention scale 1/sqrt(768) instead of 1/sqrt(64)]")
attn_wrong_scale = attention_numpy(h_in_real, Wqkv0, Wo0, wqkv_split="three_major",
                                    rope_interleaved=False, scale=1.0 / math.sqrt(H), rotate_v=False)
report("attn_out (WRONG: scale=1/sqrt(768))", attn_wrong_scale, attn_out_real)

print("\n[NEGATIVE CONTROL: wrong theta=10000 instead of 1000]")
attn_wrong_theta = attention_numpy(h_in_real, Wqkv0, Wo0, wqkv_split="three_major",
                                    rope_interleaved=False, rope_theta=10000.0,
                                    scale=1.0 / math.sqrt(HD), rotate_v=False)
report("attn_out (WRONG: theta=10000)", attn_wrong_theta, attn_out_real)


# ---------------------------------------------------------------------------
# 5. Q2: which projection gets SiLU?  (against real captured mlp0 output)
# ---------------------------------------------------------------------------
hr("5. Q2 -- SiLU placement: fc11(x)*silu(fc12(x)) [Cand A] vs silu(fc11(x))*fc12(x) [Cand B]")

Wfc11_0 = raw["encoder.layers.0.mlp.fc11.weight"]
Wfc12_0 = raw["encoder.layers.0.mlp.fc12.weight"]
Wfc2_0 = raw["encoder.layers.0.mlp.fc2.weight"]

mlp_candA = mlp_numpy(mlp_in_real, Wfc11_0, Wfc12_0, Wfc2_0, silu_on="fc12")  # out = fc11(x) * silu(fc12(x))
mlp_candB = mlp_numpy(mlp_in_real, Wfc11_0, Wfc12_0, Wfc2_0, silu_on="fc11")  # out = silu(fc11(x)) * fc12(x)

ma_A, rf_A = report("Candidate A: fc11(x) * silu(fc12(x))", mlp_candA, mlp_out_real)
ma_B, rf_B = report("Candidate B: silu(fc11(x)) * fc12(x)  [WRONG]", mlp_candB, mlp_out_real)

print(f"\n  ==> Q2 ANSWER: {'Candidate A (SiLU on fc12, fc11 is the untouched value/up path)' if rf_A < rf_B else 'Candidate B'}"
      f"   (A rel_fro={rf_A:.3e} vs B rel_fro={rf_B:.3e}, ratio B/A = {rf_B / max(rf_A, 1e-300):.3e})")
print("  (this ALSO empirically answers Q3: mlp_numpy has NO norm between the gate-multiply and fc2 -- "
        "if a real LayerNorm were applied there for this checkpoint, Candidate A could not match to fp32 precision.)")


# ---------------------------------------------------------------------------
# 6. Q1: post-LN vs pre-LN, full independent numpy layer, against real layer0 output
# ---------------------------------------------------------------------------
hr("6. Q1 -- block ordering: post-LN [Cand A] vs pre-LN [Cand B], full independent numpy layer")

norm1_w = raw["encoder.layers.0.norm1.weight"]
norm1_b = raw["encoder.layers.0.norm1.bias"]
norm2_w = raw["encoder.layers.0.norm2.weight"]
norm2_b = raw["encoder.layers.0.norm2.bias"]


def layer_numpy_post_ln(h_in):
    """Candidate A: h = norm1(attn(h) + h); h = norm2(mlp(h) + h)"""
    attn_out = attention_numpy(h_in, Wqkv0, Wo0, wqkv_split="three_major", rope_interleaved=False,
                                scale=1.0 / math.sqrt(HD), rotate_v=False)
    h_mid = layernorm_np(attn_out + h_in, norm1_w, norm1_b)
    mlp_out = mlp_numpy(h_mid, Wfc11_0, Wfc12_0, Wfc2_0, silu_on="fc12")
    h_out = layernorm_np(mlp_out + h_mid, norm2_w, norm2_b)
    return h_out


def layer_numpy_pre_ln(h_in):
    """Candidate B: h = h + attn(norm1(h)); h = h + mlp(norm2(h))"""
    h_normed = layernorm_np(h_in, norm1_w, norm1_b)
    attn_out = attention_numpy(h_normed, Wqkv0, Wo0, wqkv_split="three_major", rope_interleaved=False,
                                scale=1.0 / math.sqrt(HD), rotate_v=False)
    h_mid = h_in + attn_out
    h_mid_normed = layernorm_np(h_mid, norm2_w, norm2_b)
    mlp_out = mlp_numpy(h_mid_normed, Wfc11_0, Wfc12_0, Wfc2_0, silu_on="fc12")
    h_out = h_mid + mlp_out
    return h_out


h_out_candA = layer_numpy_post_ln(h_in_real)
h_out_candB = layer_numpy_pre_ln(h_in_real)

ma_qA, rf_qA = report("Candidate A: post-LN  (h=norm1(attn(h)+h); h=norm2(mlp(h)+h))", h_out_candA, h_out_real)
ma_qB, rf_qB = report("Candidate B: pre-LN   (h=h+attn(norm1(h)); h=h+mlp(norm2(h)))  [WRONG]", h_out_candB, h_out_real)

print(f"\n  ==> Q1 ANSWER: {'Candidate A (post-LN, matches config prenorm=false)' if rf_qA < rf_qB else 'Candidate B (pre-LN)'}"
      f"   (A rel_fro={rf_qA:.3e} vs B rel_fro={rf_qB:.3e}, ratio B/A = {rf_qB / max(rf_qA, 1e-300):.3e})")


# ---------------------------------------------------------------------------
# 7. Q3 direct: does inserting a REAL LayerNorm inside the MLP (as if norm_layer=True)
#    make things WORSE?  Constructive negative control using the layer's own norm2
#    weights as a stand-in "what if it wasn't identity" probe.
# ---------------------------------------------------------------------------
hr("7. Q3 -- explicit negative control: MLP with a LayerNorm forced between gate-multiply and fc2")

y11 = mlp_in_real @ Wfc11_0.T
y12 = mlp_in_real @ Wfc12_0.T
gated = y11 * silu_np(y12)
# Insert a plausible LayerNorm (reusing norm2's affine params, since MLP has none of its own
# in the 112-tensor checkpoint -- there is no encoder.layers.0.mlp.norm.* tensor at all).
gated_normed = layernorm_np(gated, np.ones(gated.shape[-1]), np.zeros(gated.shape[-1]))  # unit-affine LN, no spare params exist (gated is 3072-wide, no learned 3072-wide norm exists in the checkpoint either)
mlp_out_forced_norm = gated_normed @ Wfc2_0.T
ma_fn, rf_fn = report("MLP with LayerNorm forced before fc2 [WRONG if norm were real]", mlp_out_forced_norm, mlp_out_real)
print("  (no 'encoder.layers.N.mlp.norm.*' tensor exists in the 112-tensor checkpoint at all --")
print("   confirmed by key search below -- so even a unit-affine LN has no learned params to be exactly right,")
print("   and the true Identity (Candidate A of Q2, section 5) already matched near fp32 precision.)")
norm_keys = [k for k in raw if ".mlp.norm" in k]
print(f"  keys matching '.mlp.norm' in the 112-tensor checkpoint: {norm_keys} (expect: [])")
assert norm_keys == []


# ---------------------------------------------------------------------------
# 8. token_type_embeddings: is row 0 actually added?
# ---------------------------------------------------------------------------
hr("8. token_type_embeddings row 0 addition")

word_emb = raw["embeddings.word_embeddings.weight"]
tok_type_emb = raw["embeddings.token_type_embeddings.weight"]
ids = input_ids[0].numpy()

emb_with_ttype = word_emb[ids] + tok_type_emb[0][None, :]
emb_without_ttype = word_emb[ids]

report("embeddings WITH token_type_embeddings[0] added", emb_with_ttype, emb_out_real)
report("embeddings WITHOUT token_type_embeddings[0]  [WRONG if it should be added]",
       emb_without_ttype, emb_out_real)
print(f"  ||token_type_embeddings[0]|| = {np.linalg.norm(tok_type_emb[0]):.6f} "
      f"(non-zero row, so 'without' really differs)")


# ---------------------------------------------------------------------------
# 9. Padding vocab rows (30528 vs 30522)
# ---------------------------------------------------------------------------
hr("9. Padding vocab rows: 30528 (padded) vs vocab.txt's 30522 real rows")

with open(f"{MODEL_DIR}/vocab.txt", encoding="utf-8") as f:
    vocab_lines = [l.rstrip("\n") for l in f]
print("vocab.txt line count:", len(vocab_lines))
print("config.json vocab_size:", cfg_json["vocab_size"], " pad_vocab_size_multiple:", cfg_json["pad_vocab_size_multiple"])
print("word_embeddings.weight rows:", word_emb.shape[0])
pad_rows = list(range(len(vocab_lines), word_emb.shape[0]))
print("padding row indices:", pad_rows)
for r in pad_rows:
    print(f"  row {r}: ||w|| = {np.linalg.norm(word_emb[r]):.6e}, "
          f"max_abs = {np.max(np.abs(word_emb[r])):.6e}")
max_id_from_tokenizer = tokenizer.vocab_size - 1
print(f"tokenizer.vocab_size - 1 = {max_id_from_tokenizer} "
      f"(highest id the tokenizer's own vocab table can emit)")
# Confirm the tokenizer literally cannot address the pad rows: get the tokenizer's full vocab dict
# and check no id >= 30522 exists.
full_vocab = tokenizer.get_vocab()
max_tok_id = max(full_vocab.values())
print(f"max id present anywhere in tokenizer.get_vocab(): {max_tok_id} "
      f"(pad rows start at {len(vocab_lines)}; unreachable iff {max_tok_id} < {len(vocab_lines)}: "
      f"{max_tok_id < len(vocab_lines)})")


# ---------------------------------------------------------------------------
# 10. config_sentence_transformers.json / sentence_bert_config.json verbatim
# ---------------------------------------------------------------------------
hr("10. Sentence-transformers side-config files, verbatim")

with open(f"{MODEL_DIR}/config_sentence_transformers.json") as f:
    cst = json.load(f)
print("config_sentence_transformers.json:")
print(json.dumps(cst, indent=2))
print("  contains 'prompts' key:", "prompts" in cst)

with open(f"{MODEL_DIR}/sentence_bert_config.json") as f:
    sbc = json.load(f)
print("\nsentence_bert_config.json:")
print(json.dumps(sbc, indent=2))

with open(f"{MODEL_DIR}/modules.json") as f:
    modules = json.load(f)
print("\nmodules.json:")
print(json.dumps(modules, indent=2))
print("  module types present:", [m["type"] for m in modules])
print("  'sentence_transformers.models.Normalize' present:",
      any("Normalize" in m["type"] for m in modules))


# ---------------------------------------------------------------------------
# 11. L2 normalization: does the real sentence-transformers pipeline apply it?
# ---------------------------------------------------------------------------
hr("11. L2 normalization -- manual mean-pool (numpy, from the real hidden states) vs "
   "sentence_transformers.SentenceTransformer(...).encode(...)")

last_hidden_real = out.last_hidden_state.detach().to(torch.float64).numpy()[0]  # (seq, 768)
mask = attention_mask[0].numpy().astype(np.float64)  # (seq,)
manual_pooled = (last_hidden_real * mask[:, None]).sum(axis=0) / mask.sum()
print("manual mean-pooled embedding norm (NO explicit L2 norm applied):", np.linalg.norm(manual_pooled))

try:
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(MODEL_DIR, trust_remote_code=True)
    st_vec = st_model.encode(SENTENCE, convert_to_numpy=True).astype(np.float64)
    print("SentenceTransformer.encode() output norm:", np.linalg.norm(st_vec))
    print("cosine(manual_pooled, st_vec):",
          float(np.dot(manual_pooled, st_vec) / (np.linalg.norm(manual_pooled) * np.linalg.norm(st_vec))))
    manual_normalized = manual_pooled / np.linalg.norm(manual_pooled)
    report("manual_pooled/||manual_pooled|| vs ST.encode() output", manual_normalized, st_vec)
    report("manual_pooled (RAW, unnormalized) vs ST.encode() output [should be far off IF ST normalizes]",
           manual_pooled, st_vec)
except Exception as e:
    print("SentenceTransformer path failed:", repr(e))


# ---------------------------------------------------------------------------
# 12. Cross-check: native transformers port (no trust_remote_code) gives the
#     same pooled embedding as the original remote code, as an independent sanity check.
# ---------------------------------------------------------------------------
hr("12. Cross-check vs the NATIVE transformers nomic_bert port (no trust_remote_code)")
try:
    native_model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=False)
    native_model.eval()
    with torch.no_grad():
        native_out = native_model(input_ids=input_ids, attention_mask=attention_mask)
    native_hidden = native_out.last_hidden_state.detach().to(torch.float64).numpy()[0]
    report("native transformers last_hidden_state vs remote-code last_hidden_state", native_hidden, last_hidden_real)
    print("(the native port is HF's own from-scratch reimplementation, driven by the WeightRenaming/"
          "WeightConverter table in transformers/conversion_mapping.py -- agreement here is an INDEPENDENT "
          "confirmation of Q1/Q2/Q3, not a restatement of the same code path)")
except Exception as e:
    print("native cross-check failed:", repr(e))


hr("DONE")
print("All sections printed above. See report summary in the task's chat reply.")
