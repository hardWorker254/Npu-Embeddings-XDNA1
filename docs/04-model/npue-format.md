# The `.npue` runtime weight container — v1

**Implemented and verified in M4** ([`tasks/0006`](../../tasks/0006-m4-npue-pretiling/TASK.md)).
Reference implementation: [`tools/npue.py`](../../tools/npue.py). This document is
what the C++ loader in M7 must implement.

> ### Corrected by M5 — read this first
>
> [`tasks/0007`](../../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md) took this
> layout to hardware and **refuted two of the claims below**:
>
> 1. **"`ffn_down` cannot be expressed without pre-tiling" is false for the
>    whole-array design.** That BD failure belongs to the *single-core* design,
>    where B is walked as one full column strip so the k-blocks collapse into a
>    single `size=1536`. The whole-array design keeps k-blocks as their own
>    dimension (`<size=24, stride=24576>`) and K never appears as a size.
>    Verified by running `ffn_down` row-major on the unmodified M2 design.
> 2. **Pre-tiling is not a performance win here.** Per-core it is equal at best
>    and −11% on average with up to 22% run-to-run spread (row-major: 0.2%);
>    end-to-end at 8 columns it is a ±9% wash across all four MiniLM GEMMs.
>    Measured against the same design with the same BD dimension count and the
>    same sizes — only the strides differ.
>
> **What survives:** `tile_n = 48` is confirmed necessary and workable at 8
> columns (n=32 is impossible for three of the four shapes), the offline
> fusions are free, the round-trip is bit-exact, and baking the sub-tile order
> into the file costs nothing. The container is worth having for the fusions,
> bf16 conversion, fp32 biases, `mmap` and the sha256 pin — not for the DMA
> reasons originally given.

Input is HuggingFace **safetensors**. Output is this: a pre-tiled, pre-fused,
`mmap`-able container that the runtime hands to DMA descriptors without touching.

## Why a custom format

The runtime must not transpose, must not convert dtypes, must not concatenate,
and must not re-tile. Every one of those is a pure function of the weights, so
every one belongs offline. What is left at load time is `mmap` and pointer
arithmetic.

There was believed to be a second, harder reason: the DMA BD size field is 10
bits (max 1023), and a row-major `[1536, 384]` `ffn_down` walked as one column
strip needs `size = 1536`. **M5 showed that applies to the single-core design
only** — see the correction above. The remaining reasons are the ones at the top
of this section, and they are enough on their own.

**GGUF was considered and rejected**: we need neither llama.cpp's quant zoo nor
its metadata system. Two of its ideas are worth stealing and are here — explicit
alignment padding, and room to embed the tokenizer vocabulary in the same file.

## File layout

```
[0 : 64]                       FileHeader, exactly 64 bytes, little-endian
[json_offset : +json_length]   UTF-8 JSON: config + tensor directory
[data_offset : +data_length]   raw tensor data, each tensor 4096-byte aligned
```

```c
struct FileHeader {           // exactly 64 bytes at offset 0
  char     magic[4];          // "NPUE"
  uint32_t version;           // 1
  uint32_t arch;              // 0 = BERT_ABS_GELU_POSTLN
                               // 1 = GEMMA3_MQA_ROPE_GEGLU (host-only, M12)
                               // 2 = NOMIC_ROPE_SWIGLU (M13, see below)
  uint32_t flags;             // bit0 = pre-tiled
  uint64_t json_offset;       // 64
  uint64_t json_length;
  uint64_t data_offset;       // 4096-aligned
  uint64_t data_length;
  uint8_t  reserved[16];      // zero
};
```

> **Correction to the M0 draft.** The planned spec in
> [`README.md`](README.md) declared `reserved[24]` while also requiring exactly
> 64 bytes. Those are inconsistent: `4 + 4 + 4 + 4 + 8·4 = 48`, so reserved must
> be **16**, not 24. The 64-byte total is what matters (it keeps the JSON start
> cache-line aligned), so reserved was shortened. Found by asserting
> `struct.calcsize(...) == 64` rather than trusting the draft.

## Directory entry

```json
{ "name": "layer.0.qkv", "role": "gemm_b", "dtype": "BF16",
  "logical_shape": [384, 1152], "padded_shape": [384, 1152],
  "layout": {"kind": "block_panel", "tile_k": 64, "tile_n": 48,
             "order": "k,n,kt,nt", "inner": "s,t",
             "mac_s": 8, "mac_t": 8, "dtype": "BF16"},
  "layout_hash": "94266693ea31aa67...",
  "offset": 0, "nbytes": 884736 }
```

`offset` is relative to `data_offset`. Roles, as actually used by the three
packers in `tools/pack_npue.py`:

| role | meaning |
|---|---|
| `gemm_b` | a pre-tiled `block_panel` GEMM operand (`layout`/`layout_hash` present) — the DMA reads it straight into the array. arch 0 and arch 2 only. |
| `gemm_b_host` | a GEMM operand stored PLAIN — F32, row-major `[K,N]`, `layout=None` — for an arch with no NPU kernel yet. arch 1 (Gemma) only; every one of its GEMMs runs on the host. |
| `bias` | fp32, per-op. For arch 2 these are ZERO-FILLED placeholders (see below), not real biases. |
| `layernorm` | fp32 γ/β (or just γ for RMSNorm, arch 1). |
| `embedding` | gathered, never multiplied, so never tiled. For arch 2, `embeddings.position` is a ZERO-FILLED placeholder (RoPE replaces it). |
| `tokenizer` | opaque `U8` bytes — `vocab.txt` (arch 0/2, WordPiece) or the generated Gemma table (arch 1) — riding in the same file so deployment is one file, not a file plus a sidecar the tokenizer must not be separated from. |

**`layout_hash` is a sha256 over the canonical JSON of the layout object.** A
file packed for different tile dimensions is not merely suboptimal for a kernel
expecting others — it is *silently wrong*. The loader compares and refuses.

## Pre-tiled `gemm_b` layout

`kind: "block_panel"` stores a `[K, N]` operand as a linear sequence of
`[tile_k, tile_n]` tiles in `(kb, nb)` order — `k` major — and, when `inner` is
`"s,t"`, permutes each tile internally into the MAC intrinsic's sub-tile order.

This absorbs **both** re-layouts the M2 design performed at runtime:

| stage | M2 did this at runtime | now |
|---|---|---|
| L3→L2 | `TensorTiler2D.step_tiler((K,N),(k,n),…)` gathering a tile out of row-major DDR | a contiguous read |
| L2→L1 | `dims_to_stream=[(k//s, s*n), (n//t, t), (s, n), (t, 1)]` | baked into the file |

Consequence — the access-pattern dimensions become *tile counts*, not extents:

| operand | `[K, N]` | k-blocks | n-blocks | max BD dim |
|---|---|---|---|---|
| `qkv` | `[384, 1152]` | 6 | 24 | 24 |
| `attn_out` | `[384, 384]` | 6 | 8 | 8 |
| `ffn_up` | `[384, 1536]` | 6 | 32 | 32 |
| `ffn_down` | `[1536, 384]` | 24 | 8 | **24** |

`ffn_down` needed `K = 1536` as a BD dimension **in the single-core design**. It
needs 24 now. The whole-array design already needed only 24 — its `step_tiler`
keeps k-blocks as their own dimension. M5 measured both descriptors side by side
and found identical dimension counts, identical sizes, and identical total
length; only the strides differ.

### Choosing `tile_n`

`mac_s = mac_t = 8` for **both** bf16 configurations on npu2 — `mac_dims` are
`(4,8,8)` plain and `(8,8,8)` under bfp16 emulation, and only `s` and `t` touch
the B layout. **So the bf16-vs-bfp16 decision does not force a repack.**

`tile_n = 48`, not M2's winning 32. The whole-array design requires
`N % (tile_n · n_cols) == 0`, and MiniLM's `N` dims are 384 / 1152 / 1536:

| n_cols | gcd(N/n_cols) | legal `tile_n` (multiples of 8) |
|---|---|---|
| 4 | 96 | 8, 16, 24, 32, 48, 96 |
| 8 | 48 | 8, 16, 24, **48** |

**M2's `tile_n = 32` is illegal at 8 columns** — `1152 / (32·8) = 4.5`. 48 is the
largest legal choice, needs **zero padding** on all three shapes, and fits L1:
`2·(64·64·2 + 64·48·2 + 64·48·4) = 53,248 < 65,536`.

This is arithmetic against documented constraints, **not yet a hardware result**.
M5 compiles and traces it.

## Offline fusions baked in

All five from [`README.md`](README.md), all pure weight rewrites:

1. **Q, K, V fused** into one `[384, 1152]` B matrix + `[1152]` bias.
2. **Transposed to `[K, N]`** — `nn.Linear` stores `[out, in]`.
3. **GEMM operands to bf16**, round-to-nearest-even. LayerNorm γ/β and every
   bias stay **fp32** — 384 floats each and numerically sensitive.
4. **`1/√head_dim` folded into the Q block** of the fused weight and bias, so
   the attention kernel has no scale multiply. Measured free (below).
5. **`position_embeddings` pre-sliced to 256** — MiniLM's real
   `max_seq_length`, not the 512 in `config.json`.

The **embedding table is stored un-tiled and fp32**: it is gathered, never
multiplied, so tiling would only hurt, and fp32 removes a source of numerical
drift at the input. It is 46.9 MB of the 68.8 MB file.

The **pooler is not stored at all** — 147,840 params sentence-transformers never
calls.

## `arch = 2` — `nomic_bert_rope_swiglu` (M13)

`tools/pack_npue.py::pack_nomic()`, dispatched on the checkpoint's own
`config.json` `model_type == "nomic_bert"` (never on directory name). Written
for `nomic-embed-text-v1.5` — the first arch on this project whose GEMM
operands are pre-tiled **and** whose block structure differs from BERT's:
RoPE (NeoX-style, θ=1000, applied to Q/K only) instead of absolute position
embeddings, and a gated SwiGLU FFN instead of GELU. Every architectural fact
below was settled empirically against the real checkpoint, not read off a
model card — see
[`tasks/0068`](../../tasks/0068-m13-nomic-spike-and-oracle/TASK.md) sec 5.

**The load-bearing decision: it emits the SAME tensor names, in the SAME
order, as `arch = 0`** — `embeddings.word`, `embeddings.position`,
`embeddings.token_type`, `embeddings.ln.{weight,bias}` (with `tokenizer.vocab`
interleaved between `.weight` and `.bias`, exactly as arch 0 does — load-
bearing for byte parity with the C++ mirror), `layer.i.{qkv,attn_out,
ffn_up,ffn_down}` (+ `.bias` each) and `layer.i.{ln1,ln2}.{weight,bias}`. That
means `Encoder::stage_all()` and the whole NPU dispatch path work
**unchanged** — a new arch, zero new runtime code for weight loading. GEMM
operands are pre-tiled `block_panel` bf16 exactly like arch 0 (unlike arch 1
Gemma, which is host-only): nomic's geometry (head_dim 64, every `N` a
multiple of 384, `K ∈ {768, 3072}`) fits the array, and packing it at the
project's default `(tile_k, tile_n) = (64, 48)` reproduces the **exact same
`layout_hash`** (`94266693ea31aa67…`) as MiniLM/bge-small/bge-base — the
architectural claim ("nomic fits the same array shape BERT does") visible as
a constant.

Departures from arch 0, and how each is represented:

- **No absolute position table** — RoPE is computed inside attention at
  runtime instead. `embeddings.position` is kept as a tensor of the same
  `[max_seq, hidden]` shape but **filled with zeros**, rather than omitted.
- **No biases anywhere** (`qkv_proj_bias` / `mlp_fc1_bias` / `mlp_fc2_bias`
  all `false` in the checkpoint's own config, and every projection appears
  in the checkpoint as `.weight` only). Every `layer.i.*.bias` tensor is
  **zero-filled**, same shape as arch 0's real biases.

  *Why zero-fill instead of omit*: `Encoder::gemm()` dereferences `bias`
  unconditionally (`main.cpp:1063`), `stage_all()` does
  `model.raw(name + ".bias")` unconditionally (`:702`), and the `--embed`
  path reads `embeddings.position` unconditionally (`:2889`). A zero tensor
  of the right shape is **exact** (adds nothing to the sum it participates
  in) and keeps that whole read path untouched — cheaper than threading
  nullable branches through the hot path for one new arch. The risk is that
  a zero tensor looks exactly like a bug, which is why `tools/
  verify_npue_nomic.py` check D asserts every one of them is zero **and**
  runs the identical check against an arch-0 container as a discriminating
  control (which must find them non-zero) — a check that cannot fail proves
  nothing.
- **Gated SwiGLU FFN, fused into one GEMM.** nomic's FFN is
  `out = fc11(x) * silu(fc12(x))` — `fc11` is the untouched up-path, `fc12`
  is the SiLU gate (confirmed three independent ways in tasks/0068 sec 5: a
  numeric discriminator at `rel_fro` 1.636e-07 vs 4.022e+00 for the swapped
  candidate, the original remote-code source, and HuggingFace's own native
  port's weight-conversion table). Both `[intermediate, hidden]` matrices are
  transposed to `[hidden, intermediate]` and **concatenated along the N
  axis** into one `[hidden, 2·intermediate]` `layer.i.ffn_up`: columns
  `[0, intermediate)` are `fc11` (up), columns
  `[intermediate, 2·intermediate)` are `fc12` (gate). The config records this
  convention explicitly as data — `"swiglu_halves": "fc11_up|fc12_gate"` —
  so it is not a constant someone has to remember. One GEMM, so the array
  still sees **four** GEMMs per layer, not five.
- **`1/√head_dim` folded into the Q block**, exactly as arch 0 does — but
  this is only legal here because RoPE is a rotation and therefore *linear*
  in `q`: `rope(s·q) = s·rope(q)`, so folding the scale before the GEMM (and
  before RoPE, which runs strictly after it) is exact. Not just asserted:
  `tools/verify_npue_nomic.py` check E proves it numerically — comparing
  `rope(scale·q)` against `scale·rope(q)` through the actual NeoX-style RoPE
  this arch uses — and measured **`rel_fro = 0.0`** (exact, not merely
  within fp32 round-off).
- **The prefix/prompt table is this project's own choice, not the
  checkpoint's.** `config_sentence_transformers.json` for this checkpoint
  carries no `prompts` dict at all, so the container's `"prompts"` /
  `"prompt_default"` keys are accompanied by a `"prompts_source"` string
  saying so explicitly — the same precedent as
  `tools/gen_gemma_tokenizer_table.py:63-77`. `tools/verify_npue_nomic.py`
  check F asserts the label is present, not just the table.
- **`rope_theta` is asserted from the checkpoint, never defaulted.** A wrong
  theta measured `rel_fro` 9.2e-02 on the attention output in tasks/0068 —
  subtle enough to slip past a loose gate where every other wrong reading in
  that probe was 0.5–5.0. `pack_nomic()` refuses to pack if the checkpoint's
  `rotary_emb_base != 1000`.

`sentence-transformers` measures this checkpoint's own `.encode()` output at
norm **20.93**, i.e. it does **not** L2-normalize nomic (no `Normalize` module
in `modules.json`) — the container still sets `l2_normalize: true` because
that is this **runtime's** own hardcoded behaviour and matches nomic's own
documented usage (`F.normalize`), not sentence-transformers' default pipeline
for this particular model. The config carries an `l2_normalize_note` saying
so, so the flag is not mistaken for a checkpoint fact.

Verified by `tools/verify_npue_nomic.py` (not `tools/verify_npue.py`, which is
arch-0-only) — see [`tasks/0069`](../../tasks/0069-m13-nomic-arch2-container/TASK.md)
for the run and every gate's number, all PASS: spec, bit-exact round-trip
(including the fused `ffn_up`, where "bit-exact" also proves both halves
survived in the right order), the layout guard, the zero-fill assertions with
a live discriminating control, the RoPE-fold proof, and the config facts.
Goldens comparison (`reference/encoder_nomic.py` + `goldens_nomic/`) is a
separate, later task; the verifier is structured so that check can be added
without touching what is already here.

## Verification (the M4 gate, arch = 0)

`tools/verify_npue.py`, five checks:

| check | result |
|---|---|
| **A** header 64 B, magic, version, every tensor 4096-aligned | pass; overhead **0.223%** (150 KB on 68.8 MB) |
| **B** de-tile == bf16 of the fused source, **bit-exact** | **0 differing of 10,616,832** bf16 elements |
| **C** a wrong `layout_hash` must raise | `tile_n 48 → 64` refused |
| **D** the encoder run off packed weights vs the M3 goldens | `1-cos = 1.17e-05`, **0.92×** M3's end-to-end bf16 |
| **E** what the scale fold costs, in isolation | 1.17e-05 folded vs 1.28e-05 at runtime — **free** |

D is the check the others exist to support: it is the only one that catches a
fusion which round-trips perfectly but is mathematically wrong — the scale folded
into K instead of Q, a transpose applied twice, Q/K/V concatenated in the wrong
order.

## Sizes

| | |
|---|---|
| encoder weights, bf16 | 21.3 MB |
| embedding table, fp32 | 46.9 MB |
| alignment + JSON | 150 KB (0.223%) |
| **total** | **68.79 MB** |

## Not yet in v1

- Tokenizer vocabulary embedded in the file (GGUF's good idea; M7).
- int8/W8A8 tensors — deliberately deferred.
- A `[M, K]` activation layout: activations are runtime data, not weights.
