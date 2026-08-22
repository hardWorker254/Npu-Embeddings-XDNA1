# 0055 — EmbeddingGemma-300M: C1 spike (numpy reference, geometry, go/no-go)

- **Date** 2026-08-20
- **Milestone** M10 (research plan Del C, C1 only)
- **Status** done

## Goal

Execute **C1 only** from the approved research plan
(`~/.claude/plans/lag-en-plan-for-velvety-hollerith.md`, Del C): a pure-Python,
no-NPU spike to decide whether integrating EmbeddingGemma-300M (C2-C4 — a new
SentencePiece tokenizer, `arch=1` runtime branch, MQA-aware packer changes,
quantization) is worth building. Deliverables per the plan: a numpy fp32
reference encoder validated against HuggingFace, goldens pinned to the
checkpoint sha256, a geometry note on `tile_n` legality using this repo's real
formulas, and a decision document with a rough seq/s prior. **C2-C4 are
explicitly out of scope** and were not attempted — no tokenizer, no
`runtime/src/`, no `.npue` packer, no NPU/IRON/Peano work of any kind.

## Context

The plan's Del C already recorded verified facts about EmbeddingGemma-300M
(architecture, license, FastFlowLM's own coverage, gap analysis) before this
task started — see the plan file for that background; this task does not
re-derive it. What was still open: does the actual HF modeling code match the
plan's summary in every numerical detail (norm topology, RoPE base per layer,
q_norm/k_norm, attention scale), does a from-scratch numpy oracle reproduce
HuggingFace, and does the geometry risk the plan flagged (MQA forces a small
`tile_n`) actually cost what a naive prior would guess.

This project's `reference/encoder.py` (M3, MiniLM oracle) is the pattern this
task follows structurally — same discipline (one op per line, no fused
shortcuts, every landmine documented at the point it applies), same
two-oracle validation method (raw HF model exposing hidden_states, plus the
real `sentence-transformers` pipeline as an independent check).

## What was done

### 1. Confirmed the architecture against the real HF source, not the plan's summary

Read `transformers/models/gemma3/modeling_gemma3.py` and
`configuration_gemma3.py` directly (installed in `.venv-ref`, transformers
5.15.0) rather than trusting the plan's config.json-derived summary. Found
several details the plan's summary did not capture, all load-bearing:

- **RMSNorm is `x/rms * (1 + weight)`, not `x/rms * weight`** — Gemma stores
  the norm weight zero-centred. Missing the `1 +` gives a plausible-looking
  but completely wrong forward pass.
- **`q_norm`/`k_norm`**: an RMSNorm over `head_dim` (256), applied to Q and K
  **per head, after projection, before RoPE**. No BERT analogue; easy to miss
  entirely, and the checkpoint carries the weights (`layers.N.self_attn.
  q_norm.weight` / `k_norm.weight`, confirmed in the tensor inventory below).
- **Attention scale is `query_pre_attn_scalar ** -0.5`** (config value 256),
  a distinct config knob from `head_dim` even though they are numerically
  equal here (both 256).
- **RoPE base frequency is per-layer**: layers where
  `(i+1) % sliding_window_pattern == 0` (i.e. 0-indexed 5, 11, 17, 23) are
  `full_attention` and use `rope_theta=1e6`; every other layer is
  `sliding_attention` and uses `rope_local_base_freq=1e4`. Confirmed against
  `config.json`'s own `layer_types` list and `configuration_gemma3.py`'s
  `rope_parameters` construction — not inferred.
- **Each layer is a four-RMSNorm sandwich**: `input_layernorm` →
  self-attention → `post_attention_layernorm` → residual add →
  `pre_feedforward_layernorm` → MLP → `post_feedforward_layernorm` →
  residual add. Confirmed line-by-line against `Gemma3DecoderLayer.forward`.
- **Sliding-window masking is EXACT to skip, not an approximation, for
  sequences ≤ 512 tokens.** `use_bidirectional_attention=True` means every
  layer already attends both directions (no causal mask); the ONLY thing
  `sliding_attention` layer_type adds is the predicate
  `abs(q_idx - kv_idx) < sliding_window` (512), ORed with the padding mask.
  For any input with `seq_len <= 512`, that predicate is true for every pair
  of positions in the sequence, so it collapses to the same plain
  padding-only mask as the full-attention layers. This project's goldens use
  seq_len 64; the checkpoint's own `sentence_bert_config.json` sets
  `max_seq_length: 2048` but no realistic sentence-embedding input reaches
  512 tokens. A genuine sliding-window mask (needed only past 512 tokens) is
  **not implemented** — flagged explicitly, not silently skipped.
- **GeGLU uses two SEPARATE matrices** (`gate_proj`, `up_proj`, both
  `[1152,768]`), not a fused gate+up tensor — confirmed against the
  checkpoint's own tensor inventory (no such fused tensor exists).
- Activation is `gelu_pytorch_tanh` (the **tanh approximation**), not the
  exact-erf GELU MiniLM/BERT use — the opposite landmine from the one M3's
  `encoder.py` warns about.

Full detail and reasoning: `reference/encoder_gemma.py`'s module docstring
(deliberately verbose — every decision is justified at the point it is used,
same style as `reference/encoder.py`).

### 2. Checkpoint access

`google/embeddinggemma-300m` is gated (Gemma Terms of Use, needs an accepted
license + `HF_TOKEN`). `HF_TOKEN` was not set in this environment, so this
spike used the ungated community mirror `unsloth/embeddinggemma-300m`, per
the plan's decision ("Gemma = spike først... gated nedlasting = HF-token i
hub.cpp" — i.e. the HF_TOKEN wiring is explicitly C3 scope, not this spike's).
`reference/fetch_model_gemma.py` prefers the official gated repo automatically
whenever `HF_TOKEN` is set, and prints a warning when it falls back to the
mirror — so this is not a silent, permanent dependency on a third-party repo.
**This is a real finding for the go/no-go, not just plumbing**: any C3
integration needs `HF_TOKEN` support in `hub.cpp` (already scoped there by
the plan) and, separately, someone with an Anthropic/user HF account must
accept the Gemma license once, manually — this cannot be automated.

Verified the mirror's `config.json` matches the structural facts this task's
encoder depends on (`reference/fetch_model_gemma.py`'s `EXPECT_CONFIG`
dict) — model_type, hidden_size, layer count, head counts, head_dim,
activation, rms_norm_eps, RoPE bases, sliding window, and
`query_pre_attn_scalar`, all asserted, not assumed.

Checkpoint's own tensor inventory (`model.safetensors`, verified by direct
`safe_open` inspection, not the config alone): 314 tensors, **all F32 on
disk** (not BF16) — so there is no bf16-downcast-of-the-embed-scale landmine
the HF source code comments about (`modeling_gemma3.py`'s own comment: "Gemma3
downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5" —
does not apply to this checkpoint's F32 weights). No biases anywhere
(`attention_bias: false`, both Dense heads `bias: false`) — confirmed both
from each module's own `config.json` and from the safetensors files
containing only a `linear.weight` tensor.

### 3. Wrote the numpy reference encoder

`reference/encoder_gemma.py` — `GemmaEmbeddingReference`, same shape as
`MiniLMReference`: swappable `gemm` primitive (the seam a future precision
study would use), `tap()` hook at every boundary, RMSNorm/softmax computed in
fp64 for the same cheap-insurance reason `encoder.py`'s LayerNorm is.

Implements: embed → ×√768 scale → 24 × [input_norm → MQA attention
(q/k/v proj → q_norm/k_norm → RoPE(per-layer base) → repeat_kv ×3 →
scaled-dot-product → o_proj) → post_attn_norm → residual → pre_ffn_norm →
GeGLU (`down_proj(act(gate_proj(x)) * up_proj(x))`) → post_ffn_norm →
residual] → final RMSNorm → masked mean pool → Dense(768→3072, no bias) →
Dense(3072→768, no bias) → L2 normalize.

`reference/corpus_gemma.py` re-exports `reference/corpus.py`'s four sentences
(ASCII / accented Latin / CJK / long-OOV) with `SEQ_LEN=64`, same bucket as
MiniLM — tokenized with the "document" task prefix the four sentences run
14/25/24/30 tokens, so 64 gives comfortable padding room.

Task-prefix table (`encoder_gemma.PROMPTS`) copied verbatim from the
checkpoint's own `config_sentence_transformers.json` `"prompts"` dict —
`"query": "task: search result | query: "`,
`"document": "title: none | text: "`, etc. — not retyped from memory or from
FastFlowLM's copy (the plan flagged FastFlowLM's own prefix table as
reusable, but reading it from the checkpoint directly is strictly more
trustworthy and no harder).

### 4. Validated against HuggingFace and sentence-transformers

`reference/make_goldens_gemma.py` runs the real `Gemma3TextModel` (via
`AutoModel`) with `output_hidden_states=True`, and separately the real
`SentenceTransformer` pipeline, on the 4-sentence corpus with the "document"
prefix, seq_len 64.

**Indexing trap found and resolved during this step**: with 24 layers,
`output_hidden_states` returns 25 entries. `hs[0]` is the scaled embedding
(before layer 0). `hs[i+1]` for `i` in `0..22` is the **raw** output of layer
`i` (this file's `L{i}.resid2` tap, pre-final-norm). But `hs[24]` (the very
last entry) is **bit-identical to `last_hidden_state`** (`max abs diff =
0.0`, asserted in the script) — i.e. it is **post** the final `model.norm`,
not layer 23's raw output. Layer 23's raw `resid2` therefore has no direct HF
tap and is validated only transitively (same situation as `encoder.py`'s
`L{i}.ln1` for BERT — a wrong value there cannot leave `last_hidden_state`
correct). Documented in the script's header and the golden file's own
metadata `note` field so this does not have to be re-discovered.

Also asserted (not just printed, matching `make_goldens.py`'s discipline):
hand-computed mean-pool → Dense → Dense → normalize agrees with the real
`SentenceTransformer.encode()` pipeline to **max abs diff 1.318e-07** — proof
the prefix text, the pooling mode, and both Dense heads are wired correctly,
independent of the numpy reference encoder entirely.

`reference/check_reference_gemma.py` then runs `encoder_gemma.py` against the
saved goldens (`reference/goldens_gemma/embeddinggemma-300m_l24_s64_boundary.
safetensors`) and reports rel_fro per boundary plus final cosine, exactly like
`check_reference.py`. **Confirmed to run in the numpy-only `iron` env too**
(`C:\dev\mlir-aie\ironenv\Scripts\python.exe`), not just `.venv-ref` — same
cross-environment discipline as the MiniLM oracle, even though nothing in
this spike touches the NPU.

## Commands

```powershell
# Fetch + pin the checkpoint (ungated mirror; prefers google/embeddinggemma-300m if HF_TOKEN is set)
& .\.venv-ref\Scripts\python.exe reference\fetch_model_gemma.py

# Generate goldens (boundary, committed) and the full tap dump (gitignored)
& .\.venv-ref\Scripts\python.exe reference\make_goldens_gemma.py
& .\.venv-ref\Scripts\python.exe reference\make_goldens_gemma.py --taps

# Validate the numpy reference against the goldens (runs in EITHER env)
& .\.venv-ref\Scripts\python.exe reference\check_reference_gemma.py
& "C:\dev\mlir-aie\ironenv\Scripts\python.exe" reference\check_reference_gemma.py
```

## Result

### Accuracy

`check_reference_gemma.py`, 30 boundary comparisons across 24 layers plus the
sentence-embedding head:

| tensor | worst rel_fro across all layers | shape |
|---|---:|---|
| `emb.scaled` | 4.352e-08 | (4,64,768) |
| `L{i}.resid2`, i=0..22 (23 tensors) | 8.286e-07 (L17) | (4,64,768) |
| `last_hidden_state` | 1.389e-06 | (4,64,768) |
| `pool.mean` | 7.012e-07 | (4,768) |
| `dense2` | 6.029e-07 | (4,3072) |
| `dense3` | 9.377e-07 | (4,768) |
| `out.embedding` vs HF | 9.656e-07 | (4,768) |
| `out.embedding` vs sentence-transformers | 9.933e-07 | (4,768) |

**Final cosine: `1-cos` max 1.065e-07 vs raw HuggingFace, 2.110e-08 vs the
real sentence-transformers pipeline.** As tight as M3's MiniLM oracle
(2.2e-08), despite 24 RMSNorm/RoPE/MQA/GeGLU layers against MiniLM's 6
LayerNorm/GELU ones and a materially different numerical path (RMSNorm vs
LayerNorm, RoPE vs no positional rotation, tanh-GELU vs erf-GELU). All 30
comparisons PASS at `TOL_RELFRO=2e-6, TOL_COSINE=2e-7` — tolerances set from
the measured worst case with headroom, not guessed a priori (see the
tolerance comment in `check_reference_gemma.py`).

**This is the strongest finding of the spike**: the architecture is
completely reproducible from first principles by reading the real HF source,
with no residual mystery error. There is no numerical landmine hiding in the
part of the model this task could check.

### Geometry

Computed the real N-set for four offline-fusion strategies (fused/unfused
QKV × fused/unfused gate+up) and applied this repo's own tiling-legality rule
from `docs/04-model/npue-format.md` (`N % (tile_n · n_cols) == 0`, `tile_n` a
multiple of `mac_t=8`, largest legal value = largest multiple of 8 dividing
`gcd(N/n_cols)` across every GEMM shape in the model):

| fusion strategy | N-set | legal `tile_n` @ 4 cols | legal `tile_n` @ 8 cols |
|---|---|---:|---:|
| A: fused QKV (1280), unfused gate/up (1152 each) | {1280,768,1152,768} | 32 | **16** |
| B: unfused Q/K/V (768/256/256), unfused gate/up | {768,256,256,768,1152,768} | 32 | 16 |
| C: fused QKV, fused gate_up (2304) | {1280,768,2304,768} | 64* | 32 |
| D: unfused Q/K/V, fused gate_up (2304) | {768,256,256,768,2304,768} | 64* | 32 |

`*` **`tile_n=64` at `tile_k=64` fails the L1 budget** (65,536 B, the exact
64 KB the L1 discussion in `CLAUDE.md` says is 1 KB over the real 63 KB — the
budget arithmetic and its own trap both reproduce independently here) — so
even fusing gate+up buys nothing at `tile_k=64` unless `tile_k` also shrinks.
**Every fusion strategy is bottlenecked the same way, by MQA's K/V width
(1 KV head × head_dim 256 = 256), exactly as the plan flagged** — fusing or
not fusing QKV/gate-up does not move the bottleneck, because 256 alone caps
`gcd` at 32 regardless of what else is in the N-set.

L1 budget (`2·(tile_k²·2 + tile_k·tile_n·2 + tile_k·tile_n·4) < 64512`,
`tile_k=64`):

| `tile_n` | bytes | verdict |
|---:|---:|---|
| 16 | 28,672 | OK |
| 32 | 40,960 | OK (same number bge-large uses in production, tasks/0042) |
| 48 | 53,248 | OK (same number MiniLM/bge-base/bge-small use) |
| 64 | 65,536 | **EXCEEDS** (1,024 B over) |

DMA BD dimension limit (max 1023): every k-block/n-block count for every
Gemma shape at every `tile_n` above is under 100 (worst case: `qkv` at
`tile_n=16`, 8 cols → 80 n-blocks). **Not a binding constraint at all** —
unlike MiniLM's historical single-core `ffn_down` issue, this model's
geometry never comes close to the BD limit.

**Verdict: `tile_n = 16` at 8 columns, `tile_n = 32` at 4 columns — matching
the plan's prior exactly, now with the arithmetic and the L1/BD checks
behind it, and the mechanism identified (MQA's 256-wide K/V, not FFN
shape or QKV fusion choice).**

### Performance prior (NOT a measurement — no NPU/IRON/hardware work was done in this task)

Applied [T1/0048's cost model](../../research/OPEN-THREADS.md#t1)
(`t = 573 µs + 4.72 µs × iterations`, `iterations = k-blocks × n-blocks-per-
column`) to every Gemma GEMM shape at `tile_n=16`/8 cols, and to bge-base's
real shapes at `tile_n=48`/8 cols (bge-base: same hidden=768, 12 layers,
measured 181.2-209.1 seq/s on hardware in tasks/0051-0052):

| | k-blocks·n-blocks/col summed per layer | total (×layers) | 0048-model dispatch time |
|---|---:|---:|---:|
| bge-base (12 layers, tile_n=48) | 288 | 3,456 | 43.82 ms |
| EmbeddingGemma (24 layers, tile_n=16) | 516 | 12,384 | 127.21 ms |

Ratio: **3.58× more GEMM iterations, 2.90× more dispatch time** (the fixed
573 µs/dispatch term dilutes the raw iteration ratio). Isolating causes: a
hypothetical (illegal) `tile_n=48` for Gemma would need only 172
iterations/layer — so the **tile_n tax alone is ~3.0×**, partially offset by
GeGLU's genuinely lighter per-layer FFN width (2×1152 vs bge-base's 1×3072,
**0.60×**) — net 1.79× from shape+geometry combined before the dispatch-count
difference, ×2 layers = the observed ratios above.

**PRIOR (explicitly not a trace, not a wall-clock claim, not an NPU
performance result under this project's Rule 1): roughly 62-72 seq/s**
(181.2/2.90 = 62.4, 209.1/2.90 = 72.0), against bge-base's measured 181-209.
This almost certainly overstates achievable throughput, because it counts
GEMM dispatch time only — it does not include the NEW host-side work this
architecture needs that BERT-style models this repo ships do not: 4×
RMSNorm/layer (vs BERT's 2× LayerNorm), RoPE application to Q/K, and GeGLU's
elementwise `act(gate)·up` multiply. All of that would sit on the host under
the plan's own C3 sketch ("RMSNorm/RoPE/GeGLU-multiply som AVX2-vertsstier
først"), same as this project's existing LayerNorm/softmax/GELU. Given
0044's finding that host eltwise + the transport it forces already costs
33% of a BERT-model encode, a model with roughly 2× the elementwise
op-count per layer (4 norms vs 2, plus RoPE, plus the extra GeGLU multiply)
should be expected to cost proportionally more of that, not less.

## Problems hit

- **Golden-file naming ambiguity for `hidden_states` semantics** (symptom:
  before checking, it was not obvious whether `hs[i+1]` for the LAST layer
  would be pre- or post-final-norm) — **cause**: the newer `transformers`
  `@capture_outputs`/`_can_record_outputs` mechanism is not documented at the
  level of detail `check_reference.py`'s BERT-era assumption relied on —
  **fix**: tested empirically (`torch.allclose` / exact `max abs diff`
  between `hs[-1]` and `last_hidden_state`) rather than assumed, and asserted
  the finding in `make_goldens_gemma.py` so a future `transformers` upgrade
  that changes this behavior fails loudly instead of silently corrupting the
  L23 boundary comparison (which is skipped anyway, but a future version
  changing `hs[i]` semantics for OTHER indices would otherwise go unnoticed).
- **`UnicodeEncodeError` on Windows stdout with CJK sentences** — same trap
  `make_goldens.py`'s own comment already documents (`cp1252` default
  console encoding). Worked around the same way: `sys.stdout.reconfigure
  (encoding="utf-8", errors="replace")` at the top of every script that
  prints the corpus.
- **`shutil.copy` to `/tmp/...` failed on Windows** (this environment's Bash
  tool is Git Bash over a Windows filesystem; `/tmp` does not exist) — used
  the session's scratchpad directory instead for exploratory config
  downloads. Not a project-code bug, just an environment note for next time.
- No blockers. HF was reachable, the mirror had everything needed, `.venv-ref`
  already had `transformers 5.15.0` (supports `gemma3_text` out of the box)
  and `sentence-transformers 5.7.0` — no `pip install` was needed into
  `.venv-ref`, and `ironenv`/the iron conda env were never touched.

## Go/no-go recommendation: **conditional go, at reduced priority — not urgent**

**Recommendation for the user's decision, not a unilateral call** (matching
this project's convention that architecture-cost tradeoffs like this are the
user's to make, e.g. T23's bfp16-vs-accuracy precedent):

**The accuracy case is fully closed and clean** — the numpy oracle is a
genuine, first-principles match to HuggingFace (1-cos 2.1e-08 vs
sentence-transformers), so C2-C4 would not be starting from an uncertain
reference. Nothing here argues against building it on correctness grounds.

**The performance case is the reason to deprioritize, not cancel.** The
~62-72 seq/s prior (already likely optimistic, see above) is roughly
**2.5-3× slower than bge-base** on hardware this project already ships and
has fully productionized (181-209 seq/s), for a model whose primary
advantage over bge-base is Matryoshka truncation and a larger/more modern
tokenizer vocabulary — not raw embedding quality on this project's own MTEB
harness (untested here; would need running). The bottleneck is architectural
(MQA's 256-wide K/V), not a tuning parameter this project can push on the
way it pushed lane count or bfp16 — there is no lever in this repo's toolkit
that fixes a `tile_n=16` floor short of a different attention head geometry,
which is not this project's model to redesign.

**Refined cost list for C2-C4** (the plan's own estimate, refined with what
this spike learned):

| item | plan's estimate | this spike's refinement |
|---|---|---|
| SentencePiece Unigram + metaspace + byte-fallback tokenizer | ~600-900 LOC | Unchanged — not touched this spike. Note: also needs BOS/EOS handling and the task-prefix table (small, already extracted to `encoder_gemma.PROMPTS`) baked into the tokenizer's contract, not just the vocabulary. |
| Runtime `arch=1` branch, `Encoder::run()` | new branch at main.cpp:1217-1275 | Also needs: 4 RMSNorms/layer (not 2 LayerNorms) on the host, RoPE application (new — no existing kernel/host code does this), GeGLU's extra elementwise multiply, and Dense-head matmuls post-pool. All host-side per the plan's own C3 sketch; roughly double the elementwise op surface area per layer that MiniLM/bge needed, compounding 0044's 33%-of-encode transport finding. |
| Packer changes (MQA-aware QKV fusion, scale fold, BF16-safetensors read) | both packers, byte-identical via `verify_pack_parity` | Confirmed the K/V-width bottleneck this spike found (256) is INDEPENDENT of fusion choice — the packer can fuse QKV or not without changing the tile_n ceiling, so this part of the plan's design freedom is real, not illusory. |
| hub.cpp `HF_TOKEN` support | Authorization: Bearer, fail-closed | Confirmed necessary in practice, not hypothetical — `unsloth/embeddinggemma-300m` worked for this research spike, but the plan is explicit that production should not depend on a third-party mirror, and this spike agrees: the mirror could disappear or drift out of sync with the official repo with no warning. |
| Expected geometry payoff | "tile_n 16 @ 8 cols... alternatives to price: padding, host K/V" | This spike priced it: **no alternative in this repo's existing toolkit removes the bottleneck** — KV-padding (padding the 1 KV head's 256-wide output to look like a wider N) would waste MAC cycles on padding rather than fix `tile_n`, and doing K/V-projection on the host removes only 2 of 5 GEMM shapes per layer (QKV's K/V portion), leaving `attn_out`/`ffn_gate`/`ffn_up`/`ffn_down` still bound by the SAME `tile_n=16` (they don't touch K/V's N at all) — so host K/V does not change the tile_n verdict, it only removes one already-small GEMM (the K/V halves of QKV) from the array's plate. Not evaluated further; out of scope for a Python-only spike. |

**Suggested path if the user wants to proceed**: given the ~2.5-3× speed gap
and that this project's stated interest is close-to-metal NPU programming
(CLAUDE.md, "Start by reading docs/00-overview.md" / "the point is to
genuinely understand the AIE array"), EmbeddingGemma is a reasonable target
for a *future* session once the phase-fusion work (T3/T28, currently the
project's own top priority per `research/OPEN-THREADS.md`) has landed and
reduced the fixed per-dispatch cost that dominates the 2.90× ratio above — at
that point the same architecture could be re-priced and might close much of
the gap, since the fixed-cost term is exactly what fusion attacks.
Re-running this spike's geometry arithmetic after T28 lands would take
minutes, not a new spike.

## Artifacts

- `reference/encoder_gemma.py` — the numpy fp32 reference encoder (committed).
- `reference/corpus_gemma.py` — golden corpus (re-exports `corpus.py`, sets
  `SEQ_LEN=64`) (committed).
- `reference/fetch_model_gemma.py` — fetch + pin script (committed).
- `reference/make_goldens_gemma.py` — golden generator (committed).
- `reference/check_reference_gemma.py` — validation gate (committed).
- `reference/goldens_gemma/embeddinggemma-300m_l24_s64_boundary.safetensors`
  (19.8 MB, committed) — the boundary contract, same role as
  `reference/goldens/minilm_l6_s64_boundary.safetensors`.
- `reference/goldens_gemma/embeddinggemma-300m_l24_s64_taps.safetensors`
  (381.6 MB, **gitignored**, regenerate with `--taps`) — full intermediate
  dump, same role as MiniLM's `_taps` file.
- `models/embeddinggemma-300m/` — the fetched checkpoint (gitignored except
  `CHECKPOINT.json`, same convention as every other model directory). Source:
  `unsloth/embeddinggemma-300m`, sha256
  `cbf5a78393b6a033e0b8a63a57549964f7ed5c6fbeb4ba0694214f36123f2fd2`.
- `research/OPEN-THREADS.md` — new thread **T29** recording this spike's
  findings and the open go/no-go decision.
- `.gitignore` — one new line for `reference/goldens_gemma/*_taps.safetensors`.

## Next

**Nothing in C2-C4 was started or should be started without the user's
explicit go-ahead**, per the plan's own framing of C1 as a decision point.
If the user says go: start with the tokenizer (C2), since it is the one
piece every other C3 component depends on and has no dependency on the
performance question above. If the user wants to wait: T28 (phase fusion) is
already this project's own top-priority open thread independent of Gemma, so
"wait for T28, re-price" costs nothing extra to defer.
