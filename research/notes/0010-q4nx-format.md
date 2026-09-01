# 0010 — FastFlowLM's `q4nx` weight format, solved against ground truth

- **Date** 2026-08-28
- **Context** Work in the sibling repo `../LLMNpuTest` ("OpenFFLM"), whose
  question is whether we can write the kernels FastFlowLM keeps closed. FLM's
  orchestration is MIT, but `gemm.dll`, `dequant.dll`, `lm_head.dll`, `mha.dll`,
  `q4_npu_eXpress.dll` and its two `.xclbin` blobs are not, and every public
  header is pimpl. Reading the weight container is the first thing an open
  replacement needs. This note belongs here rather than there because it is a
  durable finding about somebody else's format, and `../LLMNpuTest` is kept to
  code.
- **Relevant to** [`0001`](0001-aie-kernel-pitfalls.md) (the int4 deferral in
  `docs/00-overview.md:107` — *"there is no native int4 MAC on AIE2P; it is a
  storage format requiring in-core dequant"* — is exactly what this format is)

## The container

`.q4nx` is a plain safetensors file: `u64` header length, JSON header, data.
For `FastFlowLM/Qwen3.5-0.8B-NPU2` the header is 40,888 bytes with 357 entries,
so the whole index is reachable in one 64 KB HTTP range request against a 1.1 GB
file, and any single tensor in a second.

Weight tensors are **pre-tiled**, shaped `[N/32][K/256][bytes]`. One tile is 32
output rows × 256 K, stored as two row-blocks of 16. Two forms:

| form | bytes/tile | layout | dequant |
|---|---:|---|---|
| q4 | 5120 | `[512 B bf16 d][512 B bf16 m][4096 B nibbles]` | `w = code·d + m` |
| q8 | 8704 | `[512 B bf16 scale][8192 B int8]` | `w = code·scale` |

With `rb` = row-block (0..1), `r` = row within block (0..15), `k` = 0..255,
`kb = k / 32`:

```
metadata index  = kb*32 + rb*16 + r          256 entries per plane
weight index i  = rb*4096 + k*16 + r         8192 weights
q4 byte, nibble = i >> 1, low nibble when i is even
```

**The row index is the fastest axis.** That is the whole point of the layout: 16
consecutive weights are 16 different rows at one `k`, so they need 16 different
scales — and those 16 scales are constant for an entire k-block. A core loads
them once per `(rb, kb)`, splats them, and the inner loop is a mask, a shift, a
zip and one mul-add. No gather, no per-group metadata reload. It is the
transpose of what a naive `[row][k]` layout would give you, and it is why the
dequant is nearly free.

FLM's own `modules/dequant.hpp` names the two entry points
`generate_dequant_q4_1_seq` and `generate_dequant_q80_packed_in_q4nx_seq`, which
is where the Q4_1 reading (scale **and minimum**, not scale and a fixed zero
point) comes from.

## Why this is not a guess

`model.layers.N.linear_attn.ssm_alpha_proj` is stored **twice in the same file**
— once quantised (`[1, 4, 8704]`, q8) and once as `.bf16.weight`
(`BF16 [16, 1024]`). That is exact ground truth for 67 KB of download.

Method: compute the true int8 codes from the bf16 twin under each candidate
scale assignment, then require **exact agreement across all four tiles** at every
one of the 8192 code positions — chance ≈ 256⁻⁴ per position. Under the mapping
above, **8192/8192 matched**, and reconstruction lands at max abs err 9.77e-04
against a scale/2 of ~7e-04, i.e. one quantisation step. Row-block 1 came back
byte-identical to row-block 0, which is the N=16 tensor padded to the 32-row
tile — itself a prediction of the layout that held.

Three earlier hypotheses were **wrong** and are worth recording, because each
looked reasonable: per-group interleaved `{d, m, 16 B}` GGUF-style blocks (gives
`nan`/1e38 when read as bf16); metadata appended at the tile tail (same); and
metadata at the tail of the whole tensor. The size arithmetic —
5120 = 256×20 = 32×8×20 — is satisfied by all of them, which is exactly why size
arithmetic alone is not evidence. What actually located the structure was a byte
profile showing content repeating with period 4096 and a distinct first 512
bytes.

The q4 form then followed from three independent checks on a real tile
(`layers.0.mlp.up_proj`): `d` everywhere positive, `m` everywhere negative,
`mean(m/d) = -7.48` (the Q4_1 signature for a symmetric weight distribution), and
every group's codes spanning 0..15. Dequantised, the tile is mean −6.3e-05,
std 8.7e-03 — an ordinary projection weight distribution.

## What is in the file, and what is not quantised

For Qwen3.5-0.8B (24 layers, hidden 1024, hybrid
`6 × (3 × Gated DeltaNet → 1 × Gated Attention)`):

- `model.embed_tokens.weight` stays **BF16** `[248320, 1024]` — 508 MB, **46% of
  the 1.1 GB file**. The vocabulary, not the transformer, dominates this model.
- `lm_head.weight` is **q8, not q4** `[7760, 4, 8704]` → N=248320, K=1024.
- `self_attn.q_proj` `[128, 4, 5120]` → N=4096 = 2 × (8 heads × 256): **q and the
  output gate are one fused projection** (`attn_output_gate: true`), which is why
  full-attention layers have no separate `self_attn.gate_proj` and
  linear-attention layers do.
- `linear_attn.qkv_proj` `[192, 4, 5120]` → N=6144 = 3 × 16 × 128.
- The SSM projections (`ssm_out_proj`, `ssm_alpha_proj`, `ssm_beta_proj`) are q8;
  everything else is q4.
- `config.json` carries FLM-private keys alongside the HF ones: `flm_version`,
  `addr_qk`, `addr_kv`, and xclbin names for the vision path.

## Four conversion differences, all of them silent

Reading the bytes is not the whole job. Loading the decoded weights into
transformers' own `Qwen3_5ForCausalLM` turned up four places where FLM's
convention differs from the reference implementation, and **not one of them
raises anything**: shapes match, `load_state_dict` is happy, the model runs.

1. **`q_proj` grouping.** `attn_output_gate` makes it 2x wide.
   transformers views it as `(..., n_heads, head_dim*2)` and chunks the last
   axis, so rows are per head: `[q_h0 | g_h0 | q_h1 | g_h1 | ...]`. FLM stores
   whole halves: `[all 2048 query | all 2048 gate]`. Cosine 0.138 un-permuted,
   0.9974 permuted.
2. **RMSNorm's `+1`.** `Qwen3NextRMSNorm` computes `x_norm * (1 + w)`, so its
   weights are stored centred on zero. FLM folded the 1 in — its
   `input_layernorm` measures mean **1.41**, `model.norm` **4.31**. Subtract it.
   Not `linear_attn.norm`: that is `Qwen3NextRMSNormGated`, plain `w * x`, and it
   measures 0.94, already right.
3. **`A_log`.** transformers keeps `A_log` and computes `-A_log.exp()`. All 288
   of FLM's `ssm_a` values are negative; `A_log` would be positive ~94% of the
   time. FLM stores `-exp(A_log)` evaluated. `A_log = log(-ssm_a)` matches the
   upstream checkpoint **exactly**.
4. **`conv1d`.** `[kernel, channels]` in the file, `[channels, 1, kernel]` in
   torch.

The names mislead too: `self_attn.gate_proj` on a **linear**-attention layer is
the delta-net output gate (`in_proj_z`), not an attention gate.

## The nibble bit, and how it was actually settled

Filed as T53 on the reasoning that a forward pass would settle it because a row
swap "is not subtle". That reasoning was wrong, and the way it was wrong is the
point of this section.

Run both ways, **both were degenerate** — repeated `.` one way, repeated `!` the
other — because difference 1 above was also present. A broken model is broken the
same way for every reason. The end-to-end check could not discriminate, and would
have been read as evidence against the nibble hypothesis if trusted.

What settled it was **the upstream bf16 checkpoint** — `Qwen/Qwen3.5-0.8B`,
1.7 GB, the thing FLM quantised — diffed tensor by tensor:

| `LOW_NIBBLE_IS_EVEN` | cosine, `layers.0.mlp.up_proj` |
|---|---:|
| **True** | **+0.997465** (the int4 floor) |
| False | +0.029775 |

The same diff found `q_proj` in one line. *An aggregate signal cannot localise,
and with two faults present it cannot even detect.* Same shape as
[T51](../OPEN-THREADS.md#t51): every instrument reported a central tendency.

## Confirmed

- `../LLMNpuTest/designs/q4nx_unpack` dequantises a real q4 tile on one AIE2P
  core, bit-exact against numpy on **8192/8192 bf16 values**.
- `../LLMNpuTest/reference/check_weights.py`: all **150** quantised tensors at or
  above the quantisation floor against the upstream checkpoint, worst 0.996116.
- `../LLMNpuTest/chat.py` generates coherent text from the q4nx weights on CPU
  at ~4 tok/s, via transformers' own model. That is the oracle every NPU kernel
  gets checked against.

So the int4 deferral in `docs/00-overview.md:107` is right about the hardware —
there is no int4 MAC — and wrong as a blocker: in-core dequant costs a mask, a
`to_float` with a non-zero binary-point shift, a zip and a mul-add.

One toolchain note earned there, of the same family as
[`0001`](0001-aie-kernel-pitfalls.md): **`aie::downshift` on a `uint8` vector
lowers to `srs_to_v64uint8`, which Peano marks deprecated and the build promotes
to a hard error.** Mask the high nibble instead and let `aie::to_float`'s shift
argument — the position of the binary point — do the divide by 16.
