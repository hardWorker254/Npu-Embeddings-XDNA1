# 0069 — M13: nomic `arch=2` container

**Date:** 2026-08-21
**Status:** DONE (container + gate) AND DONE (reference oracle + goldens).
Both halves of this task landed the same day, by two agents working
concurrently in this same task directory under a file-ownership split
(`tools/`+`docs/` here vs. `reference/` there). See the **"Reference oracle
and goldens"** section below for the second half's own goal, commands,
results and problems hit — written by that agent, in its own voice, appended
rather than merged into the section above so neither half's account is
edited by the other.

## Goal

Add `arch = 2` to the `.npue` container format and write the nomic packer, per
`tasks/0068`'s empirically-settled architecture (post-LN, SiLU on `fc12` not
`fc11`, no `mlp.norm`, RoPE NeoX-style on Q/K only starting at position 0,
theta=1000, three-major Wqkv row order, no biases anywhere). Nothing here was
re-derived — every fact below is asserted from `tasks/0068`'s findings, and
the packer additionally asserts the underlying config facts itself so a
checkpoint that silently changed would refuse to pack rather than pack wrong.

## Files touched (per the file-ownership split for this task)

- `tools/npue.py` — added `ARCH_NOMIC_ROPE_SWIGLU = 2`
- `tools/pack_npue.py` — added `pack_nomic()` + `main()` dispatch on
  `config.json`'s `model_type == "nomic_bert"`
- `tools/verify_npue_nomic.py` — new, the arch=2 gate
- `docs/04-model/npue-format.md` — documented `arch=2`, fixed the role table
  (it predated `gemm_b_host` and `tokenizer`, M12/M7 additions it had never
  been updated for), documented the gated-`ffn_up` convention
- `tasks/0069-m13-nomic-arch2-container/` — this log, `pack_npue.txt`,
  `verify_npue_nomic.txt`

Not touched: `reference/`, `runtime/`, `experiments/` (owned by a concurrent
agent).

## Design decisions (all specified by the launch instructions, not invented
## here — recorded so the reasoning is in one place)

1. **Same tensor names and emission order as arch=0.** `embeddings.word` →
   `embeddings.position` → `embeddings.token_type` → `embeddings.ln.weight`
   → `tokenizer.vocab` → `embeddings.ln.bias` → per-layer
   `{qkv, qkv.bias, attn_out, attn_out.bias, ln1.weight, ln1.bias, ffn_up,
   ffn_up.bias, ffn_down, ffn_down.bias, ln2.weight, ln2.bias}`. This is the
   load-bearing decision: `Encoder::stage_all()` and the whole NPU dispatch
   path in `runtime/` work **unchanged** for a genuinely new architecture.

2. **Zero-filled `*.bias` and `embeddings.position`.** `main.cpp` dereferences
   `bias` unconditionally in `Encoder::gemm()` (`:1063`), `stage_all()` reads
   `model.raw(name + ".bias")` unconditionally (`:702`), and `--embed` reads
   `embeddings.position` unconditionally (`:2889`). nomic has neither
   (confirmed in `tasks/0068`: all five projections are `.weight`-only in the
   112-tensor checkpoint, and `embeddings.position_embeddings.weight` is
   absent entirely). A zero tensor of the right shape is exact — adds nothing
   to a sum it participates in — and is far cheaper than threading nullable
   branches through the hot path for one new arch. Costs 786,432 B
   (`256 · 768 · 4`, the `embeddings.position` placeholder alone at the
   default `--max-seq 256`) plus 479,232 B of per-layer biases across 12
   layers — under 1.3 MB total on a 322 MB file.

   The risk: a zero tensor looks exactly like a bug. Paired with
   `verify_npue_nomic.py` check D, which asserts every one is zero **and**
   runs the identical check against an arch=0 container (`bge-base-en-v1.5`)
   as a discriminating control, which must find them non-zero — a check that
   cannot fail proves nothing.

3. **Gated `ffn_up` fusion order `[fc11 | fc12]` along N.** Columns
   `[0, 3072)` are `fc11` (untouched up-path), `[3072, 6144)` are `fc12` (gets
   SiLU). `config["swiglu_halves"] = "fc11_up|fc12_gate"` records the
   convention as data, not a constant someone has to remember. One GEMM, so
   the array sees four GEMMs per layer, not five.

4. **`1/√head_dim` folded into the Q block, in the same code shape as the
   BERT path** (`qkv[:, :hidden] *= scale`, float32 array in-place). No qkv
   bias exists to fold. **Verified legal, not just assumed**: RoPE is a
   per-position rotation and therefore linear in `q`, so
   `rope(s·q) = s·rope(q)` and folding before the GEMM (and before RoPE,
   which runs strictly after it in this arch) is exact. Proved numerically in
   `verify_npue_nomic.py` check E — see Results below,
   **`rel_fro = 0.000e+00`**, exact rather than merely within fp32
   round-off.

5. **`rope_theta` asserted, never defaulted.** `pack_nomic()` raises
   `SystemExit` if `cfg["rotary_emb_base"] != 1000` — `tasks/0068` measured a
   wrong theta at `rel_fro` 9.2e-02 on the attention output, the one wrong
   reading in that whole probe subtle enough to slip past a loose gate (every
   other wrong reading there was 0.5–5.0).

6. **Prefix/prompt table labelled as this project's own choice.**
   `config_sentence_transformers.json` for this checkpoint carries no
   `prompts` dict at all (confirmed in `tasks/0068` sec 4/10), so
   `config["prompts_source"]` says so explicitly — the precedent set by
   `tools/gen_gemma_tokenizer_table.py:63-77`. `verify_npue_nomic.py` check F
   asserts the label text is present, not just the table.

7. **Other config asserts added defensively**, beyond what the instructions
   listed, because `pack_nomic()` reads several config facts it depends on
   and a silently-changed checkpoint should refuse rather than pack wrong:
   `qkv_proj_bias`/`mlp_fc1_bias`/`mlp_fc2_bias` all `False`, `prenorm`
   `False`, `activation_function == "swiglu"` and `hidden_act == "silu"`,
   `rotary_emb_interleaved == False` (NeoX not GPT-J layout),
   `rotary_emb_fraction == 1.0` (whole head rotated), `head_dim` even
   (RoPE needs a half-split), `hidden == num_heads * head_dim`, and
   `layer_norm_epsilon == layer_norm_eps` (both keys exist in this
   checkpoint's config for the same value — read one, assert they agree
   rather than picking one and hoping).

## Commands run, in order

```powershell
& .\.venv-ref\Scripts\python.exe tools\pack_npue.py `
    --model-dir models\nomic-embed-text-v1.5 --out models\nomic-embed-text-v1.5.npue
& .\.venv-ref\Scripts\python.exe tools\verify_npue_nomic.py --model nomic-embed-text-v1.5
```

Full output saved verbatim: `pack_npue.txt`, `verify_npue_nomic.txt`.

## Results

**Container**: 322.08 MB, 150 tensors (48 pre-tiled GEMM operands: 4 shapes
× 12 layers), tile `(64, 48)`, mac `(s=8, t=8)`.

**`layout_hash` — the architectural claim, visible as a constant**:
`94266693ea31aa67…` — **identical** to MiniLM/bge-small/bge-base's shared
hash. Checked directly against all four `.npue` files' `layer.0.qkv` entries
(not just eyeballed): all four print the exact same 65-hex-char digest. This
is exactly what `tasks/0068`'s architecture claim ("nomic fits the same array
shape BERT does") should look like as a number.

**GEMM shapes** (nomic's real N-set differs from bge-base's — the fail-open
`tasks/0068` sec 7 found in `design_fits()`, scheduled for `tasks/0070`, is
about exactly this):

| operand | `[K,N]` | k-blocks | n-blocks | tiles | max BD dim |
|---|---|---:|---:|---:|---:|
| `qkv` | `[768, 2304]` | 12 | 48 | 576 | 48 |
| `attn_out` | `[768, 768]` | 12 | 16 | 192 | 16 |
| `ffn_up` (fused fc11\|fc12) | `[768, 6144]` | 12 | 128 | 1536 | 128 |
| `ffn_down` | `[3072, 768]` | 48 | 16 | 768 | 48 |

**Every gate in `tools/verify_npue_nomic.py` — PASS**:

| gate | result |
|---|---|
| A. spec | header 64 B, magic, version 1, `arch==2`, `flags&1`, reserved zero, `data_offset` 4096-aligned, size, all 150 tensors 4096-aligned. Overhead 145.9 KB on 322.1 MB (0.046%). |
| B. round-trip | **48 operands, 113,246,208 bf16 elements, 0 differing.** Of which `ffn_up`: 12 operands, 56,623,104 bf16 elements — the fused fc11\|fc12 check, proving both halves survived in the right order. |
| C. layout guard | matching layout accepted; `tile_n 48 → 64` refused. |
| D. zero-fill | arch=2: 49 tensors (`embeddings.position` + 4×12 biases), **every one exactly zero**. Control (`bge-base-en-v1.5.npue`, arch=0): same 49-tensor-shaped check, **non-zero as expected** — the discriminating control did not come back "also zero", so the check is falsifiable and passed. |
| E. RoPE-fold proof | `rope(scale·q)` vs `scale·rope(q)`, seq_len=64, head_dim=64, theta=1000: **`rel_fro = 0.000e+00`**. Exact, not merely within the 1e-6 fp32-round-off bound the gate checks against. |
| F. config | `rope_theta` present and equals the checkpoint's `rotary_emb_base` (1000); `swiglu_halves == "fc11_up|fc12_gate"`; `gated_ffn is True`; `position_embedding_type == "rope"`; `prompts` table present with 4 entries; `prompts_source` labels it as ours, not the checkpoint's. |

**Overall: PASS** — spec conformant, round-trip bit-exact, layout guarded,
zero-fills verified against a live control, scale fold proved exact through
RoPE, config facts present and correct.

## Problems hit

None. The packer and verifier both worked on the first real run against the
actual checkpoint (`models/nomic-embed-text-v1.5/`, already fetched and
pinned by `tasks/0068`) — every gate passed without needing a second attempt.
The only surprise was a pleasant one: check E's `rel_fro` came back exactly
`0.0`, not merely small, for this particular seed/seq_len/head_dim
combination — reported as measured rather than rounded up to "exact" by
assertion.

## The design set, T31, and the fail-open this task nearly created

Done in the same session, outside the two agents' file ownership. The two items
below were scheduled for `tasks/0070`; they were pulled forward because the
third one made them urgent.

### The N=6144 design built, first attempt

`tools/export_gemm_rtp.py` gained `--intermediate` and `--gated-ffn`.
`shapes_for()` had `4 * hidden` hardcoded — true of every model this project
ships, and a property of those *checkpoints* rather than of the architecture, so
it was an assumption wearing a constant's clothes. A gated FFN breaks it twice:
`ffn_up` emits both halves (`N = 2·intermediate`) while `ffn_down` still consumes
one (`K = intermediate`). BERT defaults are unchanged, checked at hidden 384 and
768 before building anything.

```powershell
python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 `
       --hidden 768 --intermediate 3072 --gated-ffn -n 48 `
       --out runtime\artifacts_nomic
```

**ONE xclbin, 16 streams, all 15 identity checks 64–75 differing bytes** —
inside the 0029 UUID-only budget. So the N=6144 stream shares the same static
configuration as the other three, at a width this project has never built
before. `b_layout_hash` is `94266693ea31aa67…`, the same constant MiniLM,
bge-small and bge-base share.

This only works because `tasks/0068` fixed the C-drain guard: `64·4·6144 =
1,572,864 > 2^20`, so nomic's `ffn_up` is the first shape in this project's
history to actually execute that guarded path.

### T31 fixed: `design_fits()` now matches geometry, not just K

Rewritten to match the **streams' `(op, K, N)`** against the container's
`hidden`, `intermediate` and `gated_ffn`. It reuses `parse_streams()`, which
every `design.json` has carried since 0032 — so it closes the hole on design
sets exported long before the geometry keys existed, **with no re-export**.
`design.json` still gained explicit `hidden`/`intermediate`/`gated_ffn` keys,
because a design that states its own geometry is better than one that has to be
inferred from its streams.

Falsification test, which is the part that matters — a fix that only ever says
"yes" proves nothing:

| | `artifacts_nomic` present | `artifacts_nomic` moved away |
|---|---|---|
| nomic | **ready** | **no design** |
| bge-base | ready | ready |

With `artifacts_base` still on disk (bge-base stays `ready`), nomic correctly
finds nothing. The old K-only predicate would have said `ready` and dispatched
an `ffn_up` stream built for half the output width.

It also fixed a fail-open nobody had filed: `print_catalog` called
`pick_artifacts` unconditionally, so **`embeddinggemma-300m` reported "ready"
whenever any hidden-768 design happened to be present**, despite arch=1 having
no NPU kernel at all. There is now a `cpu` state for it.

### The fail-open this task nearly shipped, caught before commit

With the container packed and the design built, `list` said **`ready`** for
nomic — and it was telling the truth about designs while being badly wrong about
what would happen next. `Encoder::run()` is a BERT forward pass: absolute
positions, plain GELU FFN, no rotary anything. arch=2 deliberately reuses BERT's
tensor names and shapes — that is the whole reason the packer and dispatch path
are free — so `serve nomic-embed-text-v1.5` would have read every tensor
happily, **run the wrong model, and returned embeddings nothing downstream could
question.**

Guard added in `set_model_shape()`, written as a **whitelist of what is
implemented** rather than a blacklist of what is not. The arch=1 diversion
already in `main.cpp` names the one arch it redirects, so anything unrecognised
falls through to BERT — which is precisely how this would have shipped.

```
error: container architecture 'nomic_bert_rope_swiglu' has no encoder in this
build. The NPU GEMM designs for it may well be present -- the tensor names and
shapes are shared with BERT on purpose -- but running it through the BERT
encoder would silently return embeddings for the wrong model. Refusing.
```
exit code 2. `list` reports `no encoder`, and `encoder_implemented()` is the
single source both the table and the dispatch refusal read, so they cannot
drift — a table saying "ready" while dispatch throws is its own kind of lie,
just a politer one.

### Regression: the four shipping models, after touching `main.cpp`

| model | design set | `rel_fro` vs HF golden | |
|---|---|---:|---|
| all-MiniLM-L6-v2 | artifacts_b128il | 4.473e-03 | PASS |
| bge-small-en-v1.5 | artifacts_b128il | 3.789e-03 | PASS |
| bge-base-en-v1.5 | artifacts_base | **4.297e-03** | PASS |
| bge-large-en-v1.5 | artifacts_large | 3.763e-03 | PASS |

bge-base's 4.297e-03 is **identical to the figure recorded in `tasks/0051`** —
a reproduction, not merely a pass.

## What is deliberately NOT here

- **The nomic forward pass.** No RoPE-in-fused-QKV kernel, no `swiglu_cpu`, no
  `Encoder::run()` branches. That is `tasks/0070`, and until it lands the arch
  guard above is what stands between a packed container and a wrong answer.
- **A `hub.cpp` catalogue row.** Deliberate: nothing should be fetchable by name
  before it can be executed.
- **Any throughput number.** Nothing has run on the array yet.

## Next

`tasks/0070` — the runtime: `apply_rope_qkv()` (in place inside the fused
`[B, S, 3·hidden]` buffer, avoiding the repack the Gemma path does),
`swiglu_cpu()`, and two config-driven branches in `Encoder::run()`. Then the
hardware `1-cos` gate against the goldens below, the e2e gate, and a
bit-identical regression on all four shipping models — `Encoder::run()` is
shared, and `tasks/0038`'s lesson is that bit-identical proves correctness and
says nothing about performance.

---

## Reference oracle and goldens (the OTHER half of this task)

**Status: DONE.** `reference/encoder_nomic.py` (the numpy fp64-internal
oracle), `reference/corpus_nomic.py`, `reference/make_goldens_nomic.py`,
`reference/check_reference_nomic.py`, and the committed
`reference/goldens_nomic/nomic-embed-text-v1.5_l12_s64_boundary.safetensors`
(11.0 MB, 18 tensors). Owned files only — no edits made to `tools/`,
`runtime/`, `experiments/`, or the BERT/Gemma `reference/` files.

### Goal

Write the numpy oracle and goldens for nomic-embed-text-v1.5, following
`reference/encoder_gemma.py`'s template and `tasks/0068`'s already-settled
architecture (§5/§5b/§5c there) — post-LN, RoPE theta=1000 NeoX-style on Q/K
only, SwiGLU with SiLU on `fc12`, three-major `Wqkv`, no biases anywhere, mean
pooling with NO L2-normalize by sentence-transformers, and the
`search_document:`-style prefix table labelled as this project's own choice
(the checkpoint carries no `prompts` dict).

### What was done

`reference/encoder_nomic.py` mirrors `encoder_gemma.py`'s shape: a
`NomicEmbeddingReference` class taking the raw safetensors state dict, a
swappable `gemm=` primitive, `layer_norm`/`silu`/`softmax` computed in fp64
internally (matching `layer_norm`'s biased-variance/eps-inside-sqrt
convention from `encoder.py`), and `rope_cos_sin`/`rotate_half`/`apply_rope`
copied verbatim from `encoder_gemma.py` (not imported across arch files, per
that file's own precedent) with `base` as a free parameter for nomic's
theta=1000. `encode()` deliberately returns **both** the raw pooled vector
and the L2-normalized one, explicitly labelled (`pool.mean_raw` /
`pool.mean_l2normalized`), rather than silently picking one — this
checkpoint's sentence-transformers pipeline does NOT normalize (no
`Normalize` module in `modules.json`), which this project's own runtime
convention (`g_l2_normalize` hardcoded true in `main.cpp`) papers over but a
goldens file must not.

**Two independent oracles**, gated against each other before either was
trusted (`make_goldens_nomic.py`):

1. The **native** `transformers.models.nomic_bert` port — plain
   `AutoModel.from_pretrained(model_dir)`, **no** `trust_remote_code`.
   Confirmed empirically first (`probe_hidden_states.py`, this directory):
   it resolves by default (transformers 5.15.0 has it registered against
   `model_type: "nomic_bert"`), and its `output_hidden_states=True` gives 13
   entries for 12 layers with `hs[-1]` bit-identical to `last_hidden_state`
   (`max_abs=0.0` against the remote code's `last_hidden_state` too, on the
   real 2-sentence padded probe batch — independently reproducing
   `tasks/0068`'s single-sentence section-12 finding). This is the primary
   source for the per-layer boundary taps, since the ORIGINAL remote code
   (`modeling_hf_nomic_bert.py`) has **no `output_hidden_states` support at
   all** — read directly from the cached remote source
   (`NomicBertEncoder.forward` just loops and returns the final
   `hidden_states`, no per-layer collection) — so it could not have served
   this role even with hooks re-added, without patching code this project
   does not own.
2. The **original remote code**, forced via
   `SentenceTransformer(model_dir, trust_remote_code=True)` — a genuinely
   different codebase (hand-maintained vs. code-generated from HuggingFace's
   `modular_nomic_bert.py`; `get_extended_attention_mask` vs
   `create_bidirectional_mask`), driving its own tokenization, prompt
   handling and pooling.

Measured agreement on the real 4-sentence, 64-token, `search_document:`
-prefixed corpus: **max abs diff 1.356e-06** (gate at 2e-5) — both oracles
trusted before either was written to the goldens file.

### GATE result (`check_reference_nomic.py`)

Every boundary tensor from `emb.ln` through `L11.norm2`, `last_hidden_state`
and `pool.mean_raw`, both against the native-HF golden and against the
sentence-transformers (remote-code) golden:

| tensor | rel_fro | |
|---|---:|---|
| emb.ln | 7.717e-08 | ok |
| L0.norm2 … L10.norm2 | 2.657e-07 – 3.924e-07 | ok (tight band, no formula-bug jump) |
| L11.norm2 = last_hidden_state | 1.038e-06 | ok |
| pool.mean_raw vs native HF | 7.575e-07 | ok |
| pool.mean_raw vs sentence-transformers | 7.770e-07 | ok |

**Final embedding `1-cos`: 3.573e-13** against both oracles (task's ≤1e-6 bar,
beaten by 6 orders of magnitude — tighter than this project's own BERT
oracle, 9.9e-07, and close to the Gemma oracle's 1.065e-07). Error grows with
depth in a tight, not-quite-monotonic band (2.6e-07–3.9e-07 across layers
0–10, one step up to 1.0e-06 at layer 11) — the accumulation-order signature
this project's docs describe, not a formula bug: no layer jumps out of band.

**Discriminating control, run through the same comparison code as the real
gate** (not a separate one-off script): the identical goldens scored against
`NomicEmbeddingReference(..., rope_theta=10000.0)` and against
`NomicEmbeddingReference(..., wrong_swiglu=True)` (SiLU on `fc11` instead of
`fc12`):

| control | `last_hidden_state` rel_fro | pooled 1-cos |
|---|---:|---:|
| correct oracle | 1.038e-06 | 3.573e-13 |
| RoPE theta=10000 (wrong) | 2.254e-01 | 1.876e-02 |
| SwiGLU silu(fc11)\*fc12 (wrong) | 1.291e+00 | 8.879e-01 |

Both wrong configurations land 5–6 orders of magnitude worse — the oracle is
demonstrably sensitive to the two facts `tasks/0068` flagged as the most
dangerous to get backwards (Q2's SiLU placement, and RoPE's theta being the
one wrong reading subtle enough to slip past a loose gate).

### Commands, in order

```powershell
# Confirm native-port output_hidden_states support/indexing before writing
# make_goldens_nomic.py around it -- the remote code has none, so this
# determines which oracle can even serve as the boundary-tap source.
& .\.venv-ref\Scripts\python.exe tasks\0069-m13-nomic-arch2-container\probe_hidden_states.py
#   -> native resolves with plain AutoModel.from_pretrained (no
#      trust_remote_code); 13 hidden_states for 12 layers; hs[-1] ==
#      last_hidden_state; native vs remote max_abs 0.0.

& .\.venv-ref\Scripts\python.exe reference\make_goldens_nomic.py
#   -> oracle cross-check PASS (max abs diff 1.356e-06); wrote
#      reference\goldens_nomic\nomic-embed-text-v1.5_l12_s64_boundary.safetensors

& "C:\Users\vegar\.conda\envs\iron\python.exe" reference\check_reference_nomic.py
#   -> PASS, all 16 comparisons within rel_fro 2e-6, cosine within 2e-7;
#      discriminating control PASS (both wrong configs >>1000x worse)

# --taps run, to prove the full intermediate-dump path works (gitignored,
# 305.2 MB, 223 tensors) -- .gitignore extended with the same
# *_taps.safetensors pattern make_goldens.py/make_goldens_gemma.py already use.
& .\.venv-ref\Scripts\python.exe reference\make_goldens_nomic.py --taps
```

Full output saved verbatim: `make_goldens_nomic.txt`, `check_reference_nomic.txt`,
`probe_hidden_states.txt`.

### Problems hit

**The original remote code cannot produce per-layer boundary goldens at all**
— `NomicBertEncoder.forward` (the real `modeling_hf_nomic_bert.py`, cached
under `~/.cache/huggingface/hub/models--nomic-ai--nomic-bert-2048/`) has no
`output_hidden_states` parameter and does not collect per-layer states in its
loop; it can only ever return the final `hidden_states`. This is why the
native transformers port had to be the primary source for boundary taps
rather than a convenience choice — read directly from the installed
`transformers/models/nomic_bert/modeling_nomic_bert.py` before writing
`make_goldens_nomic.py`, not assumed. The native port's `_can_record_outputs
= {"hidden_states": NomicBertLayer}` mechanism (HF v5's generic
`capture_outputs` decorator) gives exactly the per-layer collection needed,
confirmed to agree with the remote code bit-identically on `last_hidden_state`
before being trusted.

No other problems: both scripts passed on the first real run against the
actual checkpoint.

### Artifacts

- `reference/encoder_nomic.py`, `reference/corpus_nomic.py`,
  `reference/make_goldens_nomic.py`, `reference/check_reference_nomic.py`
- `reference/goldens_nomic/nomic-embed-text-v1.5_l12_s64_boundary.safetensors`
  (committed, 11.0 MB) — `_taps.safetensors` gitignored, regenerate with
  `--taps`
- `tasks/0069-m13-nomic-arch2-container/probe_hidden_states.py`/`.txt`,
  `make_goldens_nomic.txt`, `check_reference_nomic.txt`
