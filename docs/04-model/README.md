# The model: all-MiniLM-L6-v2

## Choice and rationale

**First target: `sentence-transformers/all-MiniLM-L6-v2`.**
**Design constraint: `BAAI/bge-small-en-v1.5` must run on the same compiled
designs.** It does, and it now has. What it is *not* is a byte-identical
drop-in: it has **12 layers** and pools **CLS**. Both are read from the
container and `1_Pooling/config.json` rather than compiled in.

The two checkpoints have **byte-for-byte identical tensor names and shapes**. The only
differences are layer count (6 vs 12) and pooling (mean vs CLS). So the same pre-tiling
script and the same kernels run both, and moving to bge buys **+5.9 MTEB
(56.26 → 62.17)** — beating gte-small and e5-small-v2 — for a config change and zero
new code. That is an unusually good deal and it is an explicit design goal, not an
afterthought.

Why MiniLM first:

- **Smallest correct target.** 6 layers, 22.7M params, 45 MB fp32 / 21 MB bf16.
- **Vanilla BERT throughout.** Absolute learned positions (a plain add, no RoPE
  rotation kernel), exact erf-GELU (no gating, no second FFN matrix), post-LayerNorm,
  WordPiece with a plain 30522-line vocab. Nothing exotic can go wrong.
- **No prompt-prefix protocol.** e5 requires `query:`/`passage:` and nomic requires
  `search_query:`/`search_document:`; getting those wrong silently degrades quality.
  MiniLM needs none.
- **Many independent oracles.** 258M downloads, Apache-2.0, with reference
  implementations in PyTorch, ONNX, OpenVINO, Rust/candle, ggml, Go and JS.

Rejected for v1: **model2vec/potion** (a static lookup table — no transformer to
accelerate, already 20–30k sent/s on one CPU core; keep as a speed *floor* baseline);
**jina-v2-small** (only 4 layers, but ALiBi + GEGLU + `trust_remote_code`);
**nomic / ModernBERT** (friendlier dims, but RoPE/SwiGLU or alternating local-global
attention with two RoPE thetas — right complexity for v2, not v1).

## Architecture

| | MiniLM-L6-v2 | bge-small-en-v1.5 |
|---|---|---|
| `model_type` | `bert` | `bert` |
| Layers | **6** | 12 |
| Hidden | **384** | 384 |
| FFN | **1536** | 1536 |
| Heads | 12 | 12 |
| head_dim | **32** | 32 |
| Vocab | 30522 | 30522 |
| max_position_embeddings | 512 | 512 |
| Params | 22.7M | 33.4M |
| Positions | absolute learned | absolute learned |
| Activation | **exact erf GELU** | same |
| Norm | LayerNorm, **post-LN**, **eps 1e-12** | same |
| QKV bias | yes | yes |
| Tokenizer | WordPiece, lowercase, strip accents, CJK split | same |
| **Pooling** | **mean** | **CLS** |
| L2 normalize | yes | yes |
| Effective max seq | **256** (see below) | 512 |
| MTEB avg | 56.26 | **62.17** |

Verified `config.json` (fetched live):

```json
{ "architectures": ["BertModel"], "hidden_act": "gelu", "hidden_size": 384,
  "intermediate_size": 1536, "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512, "model_type": "bert",
  "num_attention_heads": 12, "num_hidden_layers": 6, "pad_token_id": 0,
  "position_embedding_type": "absolute", "type_vocab_size": 2, "vocab_size": 30522 }
```

## Sharp edges — read before writing the loader

1. **MiniLM truncates at 256, not 512.** `sentence_bert_config.json` says
   `"max_seq_length": 256` while `config.json` says `max_position_embeddings: 512`.
   Supporting 512 would silently differ from the reference. bge/e5/gte use 512.
2. **`do_lower_case` is a trap.** `tokenizer_config.json` has
   `"do_lower_case": true, "strip_accents": null`. `strip_accents: null` means
   *inherit from `do_lower_case`* → **accents ARE stripped**. Meanwhile
   `sentence_bert_config.json` has `"do_lower_case": false`, which is a
   sentence-transformers-level flag that is **not applied**. The tokenizer config wins.
3. **`embeddings.position_ids`** (`I64 [1,512]`) is a **buffer, not a parameter**. Ignore it.
4. **`pooler.dense.{weight,bias}`** (`[384,384]` + `[384]`) exists in the checkpoint and
   is **completely unused** by sentence-transformers. 147,840 dead params. Do not implement it.
5. **`nn.Linear` stores `weight` as `[out, in]`** and computes `y = x @ Wᵀ + b`. So
   `intermediate.dense.weight` of shape `[1536, 384]` is `[out, in]`. We want `[K, N]`,
   so **the transpose is a free offline operation**.
6. **Vocab 30522 is not a multiple of 64 — and that is fine.** The embedding is a
   gather, never a GEMM. No padding needed.

## Tensor inventory

**104 tensors** for MiniLM-L6 (6 + 6×16 + 2); 200 for bge-small. No `bert.` prefix —
these are bare `BertModel` checkpoints. (A `BertForSequenceClassification` checkpoint
*would* have the prefix; handle both.)

Embeddings:
```
embeddings.word_embeddings.weight        F32 [30522, 384]
embeddings.position_embeddings.weight    F32 [512, 384]
embeddings.token_type_embeddings.weight  F32 [2, 384]
embeddings.LayerNorm.{weight,bias}       F32 [384]
embeddings.position_ids                  I64 [1, 512]      <- BUFFER, IGNORE
```

Per layer, `encoder.layer.{N}.` — exactly 16 tensors:
```
attention.self.{query,key,value}.weight   F32 [384, 384]
attention.self.{query,key,value}.bias     F32 [384]
attention.output.dense.weight             F32 [384, 384]     # O projection
attention.output.dense.bias               F32 [384]
attention.output.LayerNorm.{weight,bias}  F32 [384]          # post-attn LN
intermediate.dense.weight                 F32 [1536, 384]    # FFN up
intermediate.dense.bias                   F32 [1536]
output.dense.weight                       F32 [384, 1536]    # FFN down
output.dense.bias                         F32 [384]
output.LayerNorm.{weight,bias}            F32 [384]          # post-FFN LN
```

## Tiling analysis

Every dimension that matters is a clean multiple of 64:

| dim | value | ÷64 | ÷128 | verdict |
|---|---|:--:|:--:|---|
| hidden | 384 | **6** | **3** | clean |
| FFN | 1536 | 24 | 12 | very clean (also ÷512 = 3) |
| fused QKV out | 1152 | 18 | 9 | clean |
| **head_dim** | **32** | **0.5** | ✗ | **the only rough edge** |
| num_heads | 12 | — | — | 4 cols × 3 heads |
| vocab | 30522 | ✗ | ✗ | irrelevant — gather only |

**head_dim=32 appears in only two of six GEMMs per layer** — as the reduction (K) dim
of QKᵀ, and the output (N) dim of A·V. The other four (fused QKV `[S,384]×[384,1152]`,
O proj `[S,384]×[384,384]`, FFN up `[S,384]×[384,1536]`, FFN down `[S,1536]×[1536,384]`)
never see it. **So the entire projection/FFN pipeline runs on a uniform 64×64×64 bf16
tile with zero padding.**

At intrinsic level 32 is fine (2× the bf16 native K=8, 4× the native N=8). The cost is
at tile level: K=32 under a 64-deep K tile wastes half the accumulate depth.

FLOP split for MiniLM-L6 (2·M·K·N convention) — linear layers are 21.2 MFLOP/token:

| seq | linear GFLOP | attention GFLOP | attention share |
|---|---|---|---|
| 128 | 2.72 | 0.151 | **5.3%** |
| 256 | 5.44 | 0.604 | **10.0%** |
| 512 | 10.87 | 2.42 | **18.2%** |

**A worst-case 2× inefficiency on attention costs ~5% at seq 128.** Acceptable, and
recoverable later with a K=32/N=32 specialised kernel or by folding two heads into one
64-deep tile. This independently confirms Zen-Attention's finding that attention is
not where encoder time goes.

**Padding plan:** `head_dim=32` → pad to 64 or write a specialised kernel (≤5–18% at
stake); `num_heads=12` → map as 4 columns × 3 heads (8 columns gives an unbalanced
8+4); `seq_len` → bucket to {64, 128, 256}; batch → pad to 4 or 8 for DMA efficiency.

## Runtime weight format (`.npue`) — planned for M4

> **Implemented and verified in M4.** The section below is the original design
> sketch, kept for the reasoning. The authoritative spec is
> **[`npue-format.md`](npue-format.md)** — read that one before writing a loader.
> Two things changed: `reserved` is **16 bytes, not 24** (the draft's fields sum
> to 48, so 24 would make the header 72 rather than the required 64), and
> `tile_n` is **48, not 64** (M2's tiling is illegal at 8 columns for MiniLM's
> N dims). Both are explained there.

Input format is HuggingFace **safetensors** (8-byte LE header length, JSON header, raw
data buffer; `data_offsets` relative to the data buffer, little-endian, row-major).
Runtime format is our own pre-tiled container:

```c
struct FileHeader {           // exactly 64 bytes at offset 0
  char     magic[4];          // "NPUE"
  uint32_t version;           // 1
  uint32_t arch;              // 0 = BERT_ABS_GELU_POSTLN
  uint32_t flags;             // bit0 = pre-tiled
  uint64_t json_offset;       // UTF-8 JSON: config + tensor directory
  uint64_t json_length;
  uint64_t data_offset;       // 4096-aligned
  uint64_t data_length;
  uint8_t  reserved[24];
};
```

Directory entry:
```json
{ "name": "layer.0.qkv", "role": "gemm_b", "dtype": "BF16",
  "logical_shape": [384, 1152], "padded_shape": [384, 1152],
  "layout": {"kind":"block_panel","tile_k":64,"tile_n":64,"order":"k,n,kt,nt"},
  "offset": 0, "nbytes": 884736 }
```

Design rules and why:

- **Every tensor 4096-byte aligned** — page size, and comfortably a multiple of any
  DMA burst. Costs a few KB on a 21 MB file.
- **`mmap` the whole file** (`CreateFileMappingW` / `MapViewOfFile`) and hand raw
  pointers to DMA descriptors. Never `memcpy` weights at load; load should be ~0 ms.
- **The layout descriptor is data, not code.** When we retune to 32×64 tiles we
  regenerate the file rather than editing the loader. A `layout_hash` makes a stale
  file fail loudly instead of producing garbage embeddings.
- **Store `source_sha256`** of the upstream `model.safetensors` so results are always
  reproducible and the golden harness can assert it compared the same checkpoint.
- **Embedding table stored separately and un-tiled**, row-major `[30522, 384]`. It is
  gathered, never multiplied — tiling would only hurt. Consider keeping it **fp32**
  (47.7 MB), since a memory-bound gather costs nothing extra in compute and it removes
  one source of numerical drift at the input.

**Offline fusions to bake in** (pure weight rewrites, zero runtime cost):

1. **Fuse Q, K, V** into one `[384, 1152]` B-matrix + `[1152]` bias — one GEMM instead
   of three, 3× better weight reuse per DMA.
2. **Transpose** everything to `[K, N]` so the runtime never transposes.
3. **Convert to bf16**, but keep **LayerNorm gamma/beta and all biases in fp32** —
   they are tiny (384 floats) and numerically sensitive.
4. **Fold `1/√head_dim` = `1/√32` = 0.17677669529663687 into the Q weight and bias**, so
   the attention kernel has no scale multiply.
5. Pre-slice `position_embeddings` to the max bucket if we never exceed it.

Resulting sizes: encoder 10.65M params → **21.3 MB bf16**; embeddings 11.92M params →
23.8 MB bf16 or 47.7 MB fp32.

**GGUF was considered and rejected** for the model file — we do not need llama.cpp's
quant zoo or its metadata system. But two of its ideas are worth stealing: **embed the
tokenizer vocab in the same file**, and **explicit alignment padding**.

## Tokenizer

Two stages, both exactly specified. **We write our own** (~500–700 LOC) — see
`docs/00-overview.md` for why vendoring costs more than it saves.

**Stage A — BasicTokenizer** (`do_lower_case=true`, `tokenize_chinese_chars=true`):
clean text (drop cp 0 and 0xFFFD, drop control chars but treat `\t\n\r` as whitespace,
collapse whitespace); insert spaces around CJK codepoints (ranges `0x4E00–0x9FFF`,
`0x3400–0x4DBF`, `0x20000–0x2A6DF`, `0x2A700–0x2B73F`, `0x2B740–0x2B81F`,
`0x2B820–0x2CEAF`, `0xF900–0xFAFF`, `0x2F800–0x2FA1F` — note this **excludes**
Hiragana/Katakana/Hangul); whitespace split; lowercase then NFD-normalize and **drop
every codepoint of category `Mn`**; split around punctuation (ASCII 33–47, 58–64,
91–96, 123–126, plus any category starting with `P`), keeping punctuation as tokens.

**Stage B — WordPiece** (`unk_token="[UNK]"`, `max_input_chars_per_word=100`):
greedy longest-match-first with `##` continuation prefix. **The subtle bit: if any
position fails, the ENTIRE word becomes a single `[UNK]`** — you do not emit the
pieces found so far.

**Post-processing:** `[CLS] … [SEP]`, `token_type_ids = 0`, `attention_mask` 1/0,
truncate to `max_seq_length - 2` content tokens.

Verified special IDs: `[PAD]=0`, `[unused0..98]=1..99`, `[UNK]=100`, `[CLS]=101`,
`[SEP]=102`, `[MASK]=103`.

## Numerical landmines

Recorded here because each one produces a *plausible but wrong* result:

- **Post-LN BERT has enormous outlier dimensions** — specific hidden dims carry ±50–100
  while the rest sit near ±1 (the phenomenon behind LLM.int8 and *Outlier Suppression*,
  arXiv 2209.13325). Consequences: never judge by max-abs error alone, use relative;
  **keep LayerNorm in fp32** (bf16 mean/variance over 384 elements with a ±100 outlier
  loses everything); and per-tensor int8 activation quantisation would destroy accuracy.
- **Softmax must be fp32** with max-subtraction.
- **The attention mask:** HF adds `(1.0 - mask) * finfo(dtype).min`. Apply the mask
  **in fp32 before softmax** with a large *finite* negative. Using `-inf` gives NaN if a
  row is fully masked.
- **GELU is exact erf** (`0.5x(1+erf(x/√2))`), **not** the tanh approximation. Using
  tanh introduces a ~1e-3 systematically-biased error that shows up as an unexplainable
  relative-Frobenius floor.
- **LayerNorm eps is 1e-12, applied inside the sqrt** (`x/sqrt(var+eps)`), and PyTorch
  uses the **biased** variance estimator (divide by N).
- **Mean-pool denominator** uses `clamp(min=1e-9)`. Match it.

## Measured precision (M3)

What the NPU's number formats actually cost *this model's embedding*, measured end to
end against the fp32 oracle in `reference/encoder.py`. Source:
[`tasks/0005`](../../tasks/0005-m3-python-reference/TASK.md),
`reference/goldens/precision_study.json`.

### Per-GEMM, measured on hardware (M5)

One GEMM, layer 0 QKV `256×384×1152`, real post-LayerNorm activations against real
trained weights, on the device. Source:
[`tasks/0008`](../../tasks/0008-m5-bfp16-real-data/TASK.md).

| inputs | bf16 + fp32 accum | **bfp16** | median within-block max/min |
|---|---|---|---|
| uniform [0,1) (synthetic) | 1.88e-07 | **1.06e-02** | 10.8 |
| **real activations × real weights** | 1.72e-07 | **9.02e-03** | 18.2 |

**The input distribution barely matters: 0.85×.** Post-LN outliers really do widen
the within-block dynamic range (18.2 vs 10.8, exactly as the outlier phenomenon above
predicts) — but on this datapath that does not turn into error.

> **Superseded.** This section previously claimed real activations were **6.0× worse**
> than uniform for block floating point, and that bfp16 cost `1-cos ≈ 1.8e-02` end to
> end. Both came from a simulation whose mantissa width was fitted on *uniform* data
> and then applied to real data. Hardware refuted the first, and refitting on the real
> measurement corrects the second by ~10×.

### End to end

| | bf16 + fp32 accum | bfp16 + fp32 accum |
|---|---|---|
| worst `1 - cos` vs HuggingFace | **1.3e-05** | **1.8e-03** |
| max shift in sentence-similarity matrix | 5.9e-04 | 6.5e-03 |
| `last_hidden_state` rel. Frobenius | 5.7e-03 | 6.0e-02 |

**bf16 with fp32 accumulation is free** — nothing downstream can resolve 1.3e-05, and
it is the safe default.

**bfp16 is a real tradeoff, not a disqualification.** It is ~140× worse than bf16 and
its error still compounds with depth, but a 6.5e-03 similarity shift buys **5.5×
throughput**. Whether that costs more than the 0.3 MTEB-point budget is an M8 question
that needs MTEB, not more GEMMs.

> The bfp16 end-to-end figures come from a **simulation** whose mantissa width (7 bits
> per element) is fitted to the hardware measurement **on real data**. The fit is good
> to ~1.4×, so treat them as estimates. The bf16 figures reproduce hardware to 1.00×.
> Neither is an NPU performance claim; see
> [`../05-measurement/`](../05-measurement/README.md).

## Performance targets

Theoretical ceiling ~5,100 seq/s (2.87 GFLOP/sequence at seq 128 against 14.71 TFLOPS bf16).

| tier | throughput | meaning |
|---|---|---|
| Table stakes | **>250 seq/s** | matches an i9-13900K at far less power |
| Good | **>800 seq/s** | ~15% of peak; beats any laptop CPU |
| Excellent | **>1600 seq/s** | ~30% of peak |
| Also required | **<5 W package**, CPU stays free | the point of an NPU |

> ### The real bar is parity, not a multiple
>
> The tiers above rank throughput, and that ranking is misleading about what
> makes this worth doing. **Throughput parity with the CPU is already a win** if
> the work leaves the CPU and costs less energy — everything above parity is
> bonus.
>
> Measured ([`tasks/0019`](../../tasks/0019-offload-and-energy/TASK.md)), per
> encode-layer-equivalent at M=4096:
>
> | | wall | CPU consumed | cores busy |
> |---|---|---|---|
> | CPU (numpy/BLAS, 12 threads) | 21.06 ms | **335.08 ms** | 11–17 |
> | **NPU** | **6.38 ms** | **2.11 ms** | **0.16–0.64** |
>
> **159× less CPU, for 3.3× less wall time.** The CPU path holds 11–17 hardware
> threads at full clock; the NPU path leaves the machine essentially free.
>
> **Energy remains unmeasured.** CPU-seconds is not joules, and the NPU draws
> power of its own. Windows' `Power Meter` and `Energy Meter` counter sets exist
> on this machine but expose no instances, so a real number needs external
> instrumentation. This is the last claim in this section with no evidence
> behind it.
>
> For context on the throughput side, measured against
> `sentence-transformers` on the same machine
> ([`tasks/0018`](../../tasks/0018-npu-vs-cpu/TASK.md)): CPU reaches 710 seq/s at
> batch 128, and the validated cost model projects ~1,300 seq/s for a fused NPU
> encode. The margin is small single digits, not orders of magnitude — which is
> exactly why the offload number above is the one that matters.

> **The batching constraint.** 21.3 MB of bf16 weights over ~120 GB/s takes 0.18 ms —
> **equal to the theoretical compute time for one sequence.** At batch 1 we are
> memory-bound and the NPU's FLOPs are irrelevant. Always report batch-1 latency *and*
> large-batch throughput; they tell different stories.
