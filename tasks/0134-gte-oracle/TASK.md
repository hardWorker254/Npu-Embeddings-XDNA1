# 0134 — The gte-multilingual-base oracle: validated per-layer at 1e-06, the NTK theta derived bit-for-bit — and the fp32 reference itself was broken as loaded

**Date**: 2026-08-27
**Goal**: the arch-3 reference oracle (`reference/encoder_gte.py`), validated
the 0068 way — per-layer against the real `trust_remote_code` model, with
negative controls so every architectural fact has a measured discriminator.
This is the file every downstream arch-3 claim (packer parity, hardware
1-cos, goldens) will be checked against, so it comes first.

## Result

`reference/encoder_gte.py` (numpy only) matches the repaired fp32 reference
at **7e-08 (embeddings) to 1.9e-06 (layer 12) relfro on every one of the 12
layers**, CLS at 1.8e-06 — fp32 noise level, same order as nomic's oracle
(2.4e-07). Two texts, 18 and 19 tokens. Full output in [`raw.txt`](raw.txt);
probe in [`probe_gte_arch.py`](probe_gte_arch.py).

```
.venv-ref\Scripts\python tasks\0134-gte-oracle\probe_gte_arch.py
```

## The RoPE theta — the trap, defused with a measurement

Config says `rope_theta: 20000, rope_scaling: {type: ntk, factor: 8.0}`. All
three obvious readings are wrong, and the truth is not expressible as any
single theta:

* Tracing `NTKScalingRotaryEmbedding.__init__` (code fetched verbatim from
  `Alibaba-NLP/new-impl/modeling.py`): the parent builds a plain-base cache;
  the child then rebuilds at `8192 × 8` positions, and because
  `self.max_position_embeddings` still reads 8192, that second build takes
  the NTK branch and **registers a new inv_freq every later forward uses —
  at all positions, including 0..63**:
  `inv_freq_i = 160000^(-i/32) / 8^(1/32)` — a scaled base **and** a
  constant correction (`8^(-1/32) = 0.937`, so even frequency 0 is not 1.0).
* **Verified bit-for-bit**: a freshly constructed module's buffer equals
  `encoder_gte.gte_inv_freq()` at **max|diff| = 0.0** (float32 all the way,
  matching torch).
* **Negative controls, layer 0**: plain theta=20000 misses by relfro
  **1.9e-02**; theta=160000 without the correction by **8.9e-03** — against
  the oracle's 9.6e-07. Note how subtle both wrong readings are: 1e-02 is
  exactly the "looks fine, is wrong" regime 0068 warned about. The
  gelu-on-the-wrong-half control misses by **0.78** (structurally wrong).

**Consequence for the container**: bake the 32 `inv_freq` values (or the
two-step formula) — a single `rope_theta` field cannot represent this model.

## The reference itself was broken as loaded — two landmines, recorded

1. **transformers 5.15 + this remote code = uninitialised rotary buffers.**
   v5 instantiates custom modules on the meta device; `persistent=False`
   buffers computed in `__init__` are materialised as raw memory. The loaded
   model's `inv_freq` read run-to-run garbage (`[0.13, 1e-42, 0, 0, ...]`
   one run, different junk the next) and the cos/sin caches with it. The
   probe repairs the reference by constructing the rotary module fresh on
   CPU — the author's own code, run as intended. **Any fp32 reference for
   this model built on this env without the repair is silently wrong at
   every position.** (The failure is *loud* only if the model's derived-
   position path runs — it indexes with garbage and throws; with explicit
   `position_ids` it would be silent.)
2. **`torch_dtype: float16` in the checkpoint config is honoured by v5's
   `from_pretrained`.** The first probe run compared against an fp16 model
   and read ~1e-03 per layer — not an oracle error, a reference error. Load
   with `torch_dtype=torch.float32` for any reference use.

Both matter beyond this task: MTEB / sentence-transformers runs of gte on
this env hit the same code path. The gate task must confirm its reference
pipeline carries both repairs.

## Architecture facts the oracle encodes (each probe-confirmed via the per-layer match)

* Embedding: `word + token_type[0]` (type_vocab_size 1, unconditional) →
  `embeddings.LayerNorm`; **no position table** (RoPE only).
* Post-LN, nomic's exact shape: `h = attn_ln(h + attn(h));
  h = mlp_ln(h + mlp(h))`; **no final norm** — layer 12's `mlp_ln` output is
  `last_hidden_state`.
* Attention: fused `qkv_proj [2304, 768]` **with bias** (unlike nomic),
  **three-major** split, heads 12 × head_dim 64, scale 1/8, `o_proj` with
  bias, RoPE on Q/K only, NeoX rotate_half.
* MLP: `up_gate_proj [6144, 768]` **bias-free**, split up-first/gate-second,
  **exact (erf) GELU on the gate half**, `down_proj [3072, 768]` with bias.
* Pooling: CLS; `encode()` returns raw and normalized, labelled.
* State-dict names: `new.`-prefixed; the probe strips the prefix. 136
  tensors; the checkpoint also carries `classifier.*` heads (unused,
  UNEXPECTED at load, harmless).

## Commands run

```
curl -sL -o models/gte-multilingual-base/model.safetensors \
  https://huggingface.co/Alibaba-NLP/gte-multilingual-base/resolve/main/model.safetensors   # 610,753,338 B
.venv-ref\Scripts\python tasks\0134-gte-oracle\probe_gte_arch.py     # full transcript: raw.txt
```

(The verbatim `modeling.py` excerpts the derivation rests on were fetched
from `Alibaba-NLP/new-impl` on 2026-08-27; the HF module cache pinned
revision `40ced75c3017eb27626c9d4ea981bde21a2662f4`.)

## Problems hit

* First probe run crashed on the remote code's derived-position path
  (garbage index) — that crash is what surfaced landmine 1.
* First patched rerun still compared against fp16 — landmine 2. Both are in
  the probe as permanent comments so the next reader cannot re-trip them.

## What is NOT done here

The packer (`pack_gte`), the C++ mirror, `main.cpp` wiring, goldens and
gates — later 0.5.0 tasks. This file plus
[`0127`](../0127-t52-unigram-generator/TASK.md)'s tokenizer are the two
executable specs they build against.
