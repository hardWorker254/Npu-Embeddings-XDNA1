# 0068 — M13: nomic-embed-text-v1.5 spike, and the oracle

**Date:** 2026-08-21
**Status:** IN PROGRESS

## Goal

Add a fifth model for release 0.3.0, and make it the first genuinely NEW
architecture that runs **on the array**. Establish — by reading and running the
real checkpoint, not by trusting a model card — exactly how
`nomic-ai/nomic-embed-text-v1.5` differs from this project's `arch=0` BERT path,
and build a numpy oracle plus goldens for it.

## Context

`tasks/0051` chose bge-base by writing down four measured requirements and
finding the one common embedder that met all of them. In the same paragraph it
**rejected nomic by name** — *"nomic (RoPE + SwiGLU)"* — and
`docs/04-model/README.md:32-33` filed it as *"friendlier dims, but RoPE/SwiGLU
… right complexity for **v2**, not v1"*. This task is v2.

The reason to spend the effort here rather than take one of 0051's named
drop-ins (`e5-base` / `gte-base` / `arctic-embed-m`, all geometrically identical
to bge-base) is that nomic is the first new architecture that **fits the array**.
`arch=1` (EmbeddingGemma, tasks/0055–0067) is host-only: MQA with `head_dim 256`
floors `tile_n` at 16 and exceeds `kMaxHeadVecs = 16`, so tasks/0064 ran every
op on the CPU at ~7.9 s/sentence. nomic has `head_dim 64`, every N a multiple of
384, and the same K set {768, 3072} as bge-base.

Licence: **Apache-2.0** (the release requirement was MIT/Apache, not GPL).

Prior art on this exact SKU: `huggingface.co/amd/NPU-Nomic-embed-text-v1.5-ryzen-strix-cpp`,
an ONNX runtime app. `research/prior-art.md:60-69` already flags their licence as
self-contradictory — read for architecture, never vendor (CLAUDE.md rule 4).

## What was done

### 1. Fetch and pin

`reference/fetch_model.py` was run against nomic **before** being taught about
it, deliberately: its BERT-only structural check downloads first and asserts
second, so the failure list is a free, mechanical enumeration of every way the
checkpoint departs from BERT. That output is kept in `fetch_model.txt` and is
the raw material for everything below.

```
  model.safetensors     : 546.9 MB
  sha256                : 9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c
  tensors in checkpoint : 112
    required by arch    : 197
  geometry              : hidden 768, 12 heads x 64, ffn 3072, vocab 30528, 2048 positions

FAIL -- checkpoint does not match docs/04-model/README.md:
  - config.model_type: expected 'bert', got 'nomic_bert'
  - config.position_embedding_type: expected 'absolute', got None
  - config.hidden_act: expected 'gelu', got 'silu'
  - missing tensor embeddings.position_embeddings.weight
  ... (188 more)
```

**Three of the seven open architecture questions were answered by the tensor
inventory alone, with no experiment needed:**

| question | answer | evidence |
|---|---|---|
| Is `attn.out_proj` biased? | **No** — and neither is anything else | all five projections appear as `.weight` only; `Wqkv`, `out_proj`, `fc11`, `fc12`, `fc2` |
| Are absolute position embeddings present? | **No — the tensor is absent entirely** | `embeddings.position_embeddings.weight` missing; RoPE is the live path |
| Where is the embedding LayerNorm? | `emb_ln.{weight,bias}` at **top level**, not `embeddings.LayerNorm` | listed under "unexpected tensors" |

Count check: 4 embedding-level tensors + 12 layers × 9 = **112**, exactly the
number in the checkpoint. Nothing unaccounted for.

That "no biases anywhere" finding matters beyond bookkeeping: it makes the
planned zero-filled-bias approach *correct* rather than merely convenient.
`Encoder::gemm()` dereferences `bias` unconditionally (`main.cpp:1063`) and
`stage_all()` does `model.raw(name + ".bias")` unconditionally (`:702`), so
packing zeros keeps the fused bias-add readback untouched and adds exactly zero.

### 2. `fetch_model.py` made architecture-aware

`EXPECT_CONFIG` became a dict keyed by the checkpoint's own `model_type`, and
the tensor inventory was split into `embeddings_bert`/`per_layer_bert` and
`embeddings_nomic`/`per_layer_nomic` behind an `INVENTORY` table (note
`encoder.layers.` plural upstream against BERT's `encoder.layer.`).

An **unknown `model_type` is now refused** rather than checked against BERT's
inventory. Without an inventory there is no way to distinguish "this checkpoint
is fine" from "we do not know what to look for", and the second must never print
OK.

The nomic config assertions are the departures themselves, so each is asserted
rather than read: `prenorm: False`, `rotary_emb_fraction: 1.0`,
`rotary_emb_interleaved: False`, `rotary_emb_base: 1000` (**not** the usual
10000), `activation_function: "swiglu"`, `hidden_act: "silu"`,
`qkv_proj_bias`/`mlp_fc1_bias`/`mlp_fc2_bias` all `False`,
`layer_norm_epsilon: 1e-12`. **All of these are now confirmed against the real
downloaded checkpoint, not against the model card.**

`embeddings_nomic` demands the *absence* of a position table. That is the point:
a nomic checkpoint that shipped one would mean the rotary branch is not live,
which would silently change every vector.

Also added: `config_sentence_transformers.json` to `ALLOW`, and an odd-`head_dim`
refusal for the nomic arch (RoPE rotates `(x1, x2)` pairs and cannot half-split
an odd head).

After: **112 required / 112 present, OK.** All four existing BERT models
re-checked and still OK (197/200, 101/104, 389/392, 197/200 — the surplus is the
unused pooler plus `position_ids`, both in `IGNORABLE`).

### 3. Tokenizer — can the existing C++ WordPiece serve nomic unchanged?

**Yes.** Probe: `probe_nomic_tokenizer.py` / `.txt`.

- **`vocab.txt` is byte-identical across all five models** — sha256
  `07eced375cec144d…`, 30,522 lines, for nomic, MiniLM, bge-small, bge-base and
  bge-large alike. Verified independently of the probe.
- `tokenizer.json`, the authoritative file, matches bge-base field for field:
  `BertNormalizer` (`clean_text`, `handle_chinese_chars`, `strip_accents: null`,
  `lowercase: true`), `BertPreTokenizer`, `WordPiece` with `##` and
  `max_input_chars_per_word: 100`, and the `[CLS] … [SEP]` post-processor.
- 45/45 identical id sequences between nomic's and bge-base's HF tokenizers over
  the adversarial corpus (accents, combining marks, CJK, Greek final sigma,
  emoji, empty, whitespace-only, >100-char word). Since `tasks/0036` proved the
  C++ tokenizer byte-identical to bge-base's HF tokenizer on 6,826/6,826 texts,
  it matches nomic **by transitivity on this corpus**.
- **Still ungated, stated as such:** the real C++ binary has never been run
  against nomic, because no `.npue` exists yet. `tools/verify_tokenizer.py`'s
  full corpus through `npuembed --tokenize` is deferred to the packing task.

`sentence_bert_config.json` says `do_lower_case: false` where bge-base says
`true`. This is **not** a risk: MiniLM also says `false`, so it is an
already-shipping configuration, and the flag is a sentence-transformers-level
extra `.lower()` that our runtime never applies — we drive the raw tokenizer,
whose `BertNormalizer` lowercases regardless.

### 4. The prefix protocol

nomic *requires* a task prefix, and `docs/04-model/README.md:24` already names
getting this wrong as a silent quality regression.

**`config_sentence_transformers.json` carries NO `prompts` dict** — it holds only
a `__version__` block. So, exactly as with the Gemma tokenizer table
(`gen_gemma_tokenizer_table.py:63-77`), the prefix table is **this project's
choice and must be labelled as such in the container**, not presented as
something read from the checkpoint.

Prefixes are plain text prepended before `[CLS]` — confirmed, not assumed:
`[CLS] + prefix_ids + text_ids + [SEP]` equals whole-string tokenization for all
four. Cost against the 62 usable slots of a 64-token sequence:

| prefix | ids | tokens |
|---|---|---|
| `search_document: ` | 3945, 1035, 6254, 1024 | 4 |
| `search_query: ` | 3945, 1035, 23032, 1024 | 4 |
| `clustering: ` | 9324, 2075, 1024 | 3 |
| `classification: ` | 5579, 1024 | 2 |

The goldens must pin the prefix text explicitly — the oracle and the runtime
using the *same wrong* prefix would pass every numeric gate we have. MTEB is the
only gate that catches it.

### 5. The three architecture questions, settled empirically

Probe: `probe_nomic_arch.py` / `.txt`. Method: hook the real `layers[0]` (and
`layers[0].mlp`) to capture its true input and output, reimplement it in fp64
numpy from the safetensors weights, and compare **both** candidate readings.
Reading the code is not an answer on its own — each question carries a
discriminating negative control, because an experiment where both candidates
pass proves nothing.

| Q | answer | correct reading | wrong reading | ratio |
|---|---|---|---|---|
| **Q1** block ordering | **post-LN** `h=norm1(attn(h)+h); h=norm2(mlp(h)+h)` | `rel_fro` **2.401e-07** | pre-LN 5.031e+00 | 2.1e+07 |
| **Q2** SiLU placement | **`out = fc11(x) * silu(fc12(x))`** — SiLU on **fc12**; fc11 is the untouched up-path | `rel_fro` **1.636e-07** | silu(fc11)*fc12 4.022e+00 | 2.5e+07 |
| **Q3** `mlp.norm` | **`Identity()`** | (implied by Q2 matching at fp32) | forced LayerNorm 7.154e+00 | — |

Q3's root cause in source: `norm_layer = getattr(config, "norm_mlp", False)`, and
this config has no `norm_mlp` key → `nn.Identity()`. Consistent with there being
no `*.mlp.norm.*` tensor among the 112.

**Q2 was the dangerous one**, as expected: swapping the two projections is
structurally plausible and would still produce sane-looking embeddings. It is now
confirmed three independent ways — the numeric test, the remote code
(`NomciBertGatedMLP.forward`: `y=fc11(x); gate=fc12(x); y=y*activation(gate)`),
and HuggingFace's own native-port conversion table, which maps
`fc11→up_proj` (untouched) and `fc12→gate_proj` (gets `act_fn`).

Everything else, each with its own control:

| property | confirmed | control that fails |
|---|---|---|
| RoPE theta | **1000** (1.07e-07) | theta=10000 → 4.36e-01 on the tables |
| RoPE layout | NeoX `concat(freqs,freqs)` / rotate-half | interleaved GPT-J → 5.61e-01 |
| RoPE targets | **Q and K only** | rotating V too → 5.17e-01 |
| RoPE positions | start at 0, no offset | — |
| `Wqkv` row order | **three-major `[Q(768)|K(768)|V(768)]`** | head-major split → 3.35e+00 |
| attention scale | **1/sqrt(64)**, identical at layers 0 and 11 | 1/sqrt(768) → 6.30e-01 |
| `token_type[0]` | **is added** (2.64e-08); ‖row0‖ = 2.34, not zero | omitting it → 7.56e-01 |

`scale_attn_by_inverse_layer_idx` has **zero references** anywhere in the
modelling file — a dead config field, not a behaviour we need.

**A warning worth carrying forward:** the wrong-theta control on the attention
output lands at `rel_fro` **9.2e-02**, against 0.5–5.0 for every other wrong
reading. RoPE's theta is the one parameter here subtle enough to slip past a
loose gate while still being wrong. Assert it, do not infer it.

**Independent cross-check:** HuggingFace's new native `transformers.models.nomic_bert`
port (no `trust_remote_code`, driven by their own weight-conversion table)
reproduces the remote code's `last_hidden_state` **bit-identically**
(`max_abs = 0.0`). Two code-independent implementations agreeing is stronger
evidence for Q1–Q3 than either alone.

### 5b. sentence-transformers does NOT L2-normalise nomic

`modules.json` lists only `Transformer` + `Pooling` — no `Normalize`. Measured:
`SentenceTransformer.encode()` returns vectors of norm **20.93**, and they match
the raw manual mean-pool to `rel_fro` 6.37e-08, while the *normalised* manual
pool differs by 9.52e-01.

Consequences, worked through rather than assumed:

- **The e2e gate is unaffected.** `verify_embed_e2e.py` scores `1-cos`, which is
  invariant to scaling, so our L2-normalised output still compares correctly
  against an unnormalised reference. Same for MTEB's cosine tasks.
- **Our runtime should still normalise.** `g_l2_normalize` is hardcoded `true`
  (`main.cpp:90`) and nomic's own documented usage calls `F.normalize`
  explicitly. So the shipping behaviour is right — but note it is right by
  coincidence, since the container's `l2_normalize` key is written and never
  read. A model that genuinely wanted unnormalised output would be silently
  normalised.
- **Matryoshka is out of scope here.** The documented truncation is
  `layer_norm(768) → slice → normalize`, which is a different post-processing
  chain, not just a shorter vector. Not exposed in this milestone.

### 5c. The six padding rows

`vocab_size` is 30528 while `vocab.txt` has 30522 lines
(`pad_vocab_size_multiple: 64`). Rows 30522–30527 are **not zero** — norms
≈0.81–0.85, ordinary trained-looking values — but they are **unreachable**: the
maximum id in `tokenizer.get_vocab()` is 30521. The packer keeps the full
[30528, 768] table so the config's `vocab_size` and the tensor agree; the 18 KB
of unreachable rows is the cost of that consistency.

### 6. THE BIG ONE: a silent correctness bug in the shipping design generator,
### and every shipped model sits ONE STEP from it

The probe that was only supposed to answer "does N=6144 compile?" answered
something much more important. Probe: `probe_n6144.py` (+ `_part2`, `_part3`) /
`probe_n6144.txt`.

**N=6144 compiles fine. It computes the wrong answer.**

| shape | compiles | `rel_fro` | |
|---|---|---|---|
| A: M=8192, K=768, **N=6144** (nomic ffn_up) | yes | **7.074e-01** | FAIL |
| B: M=8192, K=3072, N=768 (positive control) | yes | 8.173e-07 | PASS — harness is sound |
| C: M=1024, K=768, N=6144 | yes | 7.067e-01 | FAIL |

Not rounding: 28 of 32 row-bands have max abs error > 1.0, where bf16 noise alone
would be ~1e-2. Identical corruption at cols 4 and 8, at tile_n 16/32/48, at
M=1024 and 8192 — always ≈0.708. That invariance is what pointed away from a
hardware limit. BD size, stride range and DMA channel counts were all checked and
are fine; L1 is 53,248 B of 63 KB exactly as trap 3 predicts, and N does not
enter that formula.

**Root cause**, in `experiments/m5-pretiled-gemm/gemm_pretiled.py::_build_design`.
The tasks/0030 guard that fixed the C-drain DMA stride wall:

```python
tb_n_rows = min(tb_max_n_rows // 2, M // m // n_aie_rows)     # line ~503
if m * n_aie_rows * N > 2**20:                                 # line ~511
    tb_n_rows = 1
...
C_tiles = TensorTiler2D.step_tiler(..., tile_group_repeats=(tb_n_rows, ...))   # ~557  USES it
...
current_tb_n_rows = min([tb_max_n_rows // 2,                   # ~592  IGNORES it
                         M // m // n_aie_rows - row_base])
```

When the guard fires, each C-drain tap covers **1** row-block while the A/B fill
loop still streams up to **2**. Drains and fills desynchronise and most of the
output is stale. The design compiles because the stride fix genuinely works —
only the fill/drain accounting was left half-wired.

**The boundary is exact.** The guard fires at `N > 2**20 / (m·n_aie_rows)` =
**N > 4096** at m=64, n_aie_rows=4:

| N | tile_n | `rel_fro` | |
|---|---|---|---|
| 3840 | 48 | 3.291e-07 | PASS |
| **4096** | 32 | **3.283e-07** | PASS — **bge-large's real production shape** |
| 4224 | 48 | 7.083e-01 | FAIL |
| 6144 | 48 | 7.074e-01 | FAIL |

`64 · 4 · 4096 = 1,048,576` — **exactly 2^20** — and the test is a strict `>`.
So **the guard has never fired in any design this project has shipped.** The fix
written in tasks/0030 for the `hidden >= 1536` wall has never executed in
production, and half of it is wrong. bge-large, our largest model, sits precisely
one step below the threshold.

The file's own comment at line ~504 even records the near-miss without drawing
the conclusion: *"the DMA stride field is 20 bits ([1:1048576], INCLUSIVE —
measured: N=4096 at exactly 2^20 builds)"*. It was measured **at** the boundary,
so the code path beyond it was never exercised.

This also **corrects `docs/CURRENT_STATUS.md`'s "known walls" table**, which still
lists `hidden >= 1536` as an open build wall. It is not a build wall any more —
it builds, and returns wrong numbers, which is strictly worse.

**Decision: fix the root cause, keep the fallback documented.**
The fallback (split the gated `ffn_up` into two N=3072 dispatches over the
existing, proven bge-base design set — 6144 = 2 × 3072, and N=3072 gives
`64·4·3072 = 786,432 < 2^20`) is real and would unblock nomic today at the cost
of 12 extra dispatches per encode. But shipping a release while knowingly leaving
a silent-corruption path in the production design generator is not a trade worth
making, and the fix is the same work either way the moment anyone builds wider.
The fallback stays on record as the escape hatch if the fix does not land clean.

Note the fix is **not** the one-line change it appears to be: `row_base` steps by
`tb_max_n_rows // 2` and the `tb` loop is bounded by `tb_max_n_rows`, so if
`tb_n_rows` becomes 1 the stepping no longer matches the drain taps either. The
whole row-block walk has to be driven by the guarded value, with the M=256
single-row-block tail case (the NPUE-M5 comment at line ~499, which exists
because that case was broken once already) still intact.

### 6b. The fix, and its four gates

```python
# was
for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
    ...
        row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
        current_tb_n_rows = min([tb_max_n_rows // 2, M // m // n_aie_rows - row_base])

# now
tb_step = 2 * tb_n_rows
for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_step)):
    ...
        row_base = tb * tb_step + pingpong * tb_n_rows
        current_tb_n_rows = min([tb_n_rows, M // m // n_aie_rows - row_base])
```

The whole walk is now driven by `tb_n_rows` — the same value that sizes the
C-drain tap. **It is provably a no-op below the guard threshold**: at the
historical unguarded `tb_n_rows = 2`, `tb_step` is exactly `4 = tb_max_n_rows`
and both expressions reduce to the originals, algebraically.

| gate | result |
|---|---|
| **1. N=6144 fixed** | **7.075854e-01 FAIL → 3.283356e-07 PASS.** Also M=1024/N=6144 3.286067e-07, N=4224 3.282708e-07 |
| **2. below-threshold unchanged** | K=3072/N=768 8.168e-07 (was 8.173e-07); N=3840 3.288e-07 (was 3.291e-07); N=4096/tile_n=32 3.284e-07 (was 3.283e-07) |
| **3. small-M tail** | M=256 (1 row block) 3.288944e-07; M=1024 3.286852e-07 — both PASS |
| **4. production identity** | fixed vs unfixed, same toolchain: **74 differing bytes**, inside the 0029 UUID-only budget (≤80) |

Note on gate 2: the numbers **reproduce to 3 significant figures but are not
identical**, because each hardware run draws fresh random operands. Run-to-run
`rel_fro` scatter of ~0.1% is expected and is *not* evidence of a design change.
The design identity is established by gate 4 — the xclbin comparison — not by
these numbers.

**Gate 4 had to be measured twice, and the honest version is the interesting
one.** The scratch rebuild does **not** match the shipped
`runtime/artifacts_base/gemm_rtp/final.xclbin`: 125,791 vs 122,334 bytes, a
3,457-byte size mismatch far outside the UUID budget. Rebuilding with the
**unmodified original** `gemm_pretiled.py` under today's toolchain reproduces the
*same* 3,457-byte divergence — so it is pre-existing and caused by the mlir-aie
1.3.4 → 1.4.x upgrade (tasks/0058), not by this fix. Isolating the fix by
comparing fixed against unfixed *under the same toolchain* gives the 74 bytes
above.

**A standing fact that falls out of this, worth recording:** the shipped
`artifacts_base` xclbin **cannot be reproduced byte-for-byte from today's
toolchain**. tasks/0059 established that this does not affect correctness — a
shipped `.xclbin` is a static binary XRT loads regardless of what built it, and
all four models reproduced their exact `1-cos` after the upgrade. But
"regenerating the release rebuilds a different binary than the one on disk" is a
real property of the repository as it stands, and it was not previously written
down.

**Trap 7c bit again, and was caught:** the first fixed build appeared not to work
because a stale JIT cache entry was served. Purge before believing a rebuild.

### 7. A fail-open found in `design_fits()` — verified, not hypothesised

`pick_artifacts()` chooses a design set by calling `design_fits()`
(`runtime/src/main.cpp:1363`), which asks one question: **does `hidden` appear
as a `"K"` anywhere in `design.json`?** Its own comment states the failure mode
it exists to prevent:

> *"Checked rather than assumed because a design built for another width has the
> same filenames and loads fine — it would simply compute the wrong thing."*

nomic is a checkpoint where that check passes and the conclusion is still wrong.
Compared against the shipping `runtime/artifacts_base` (bge-base's set):

| stream | artifacts_base | nomic needs | |
|---|---|---|---|
| `qkv` | K=768, N=2304 | K=768, N=2304 | same |
| `attn_out` | K=768, N=768 | K=768, N=768 | same |
| **`ffn_up`** | K=768, **N=3072** | K=768, **N=6144** | **differs** |
| `ffn_down` | K=3072, N=768 | K=3072, N=768 | same |

nomic's K set is `{768, 3072}` — **identical to bge-base's** — so `design_fits`
returns true and `pick_artifacts` hands nomic bge-base's design. The runtime
would then dispatch an `ffn_up` stream built for half the output width. No error,
no warning; the gated FFN would silently lose its gate half.

The check is sound for every model shipped so far only because all of them have
`N = 4·hidden`; nomic is the first model whose N-set differs at the same K. Fix
(scheduled for `tasks/0070`, with the design work): match the **streams'
`(op, K, N)`** against the container's `hidden`, `intermediate` and `gated_ffn`,
not `hidden` against a bare `"K"`. Then re-run `list` for all four shipping
models and confirm each still resolves to the set it resolved to before.

Recorded here rather than in 0070 because it was found and confirmed here, and
because it is the tenth instance of this repo's recurring pattern: a guard whose
comment names the right danger while its predicate is weaker than the comment.

## Commands, in order

```powershell
# 1. Fetch. Deliberately run BEFORE fetch_model.py knew about nomic, so its
#    BERT-only check enumerates every departure mechanically.
& .\.venv-ref\Scripts\python.exe reference\fetch_model.py --model nomic-ai/nomic-embed-text-v1.5 --layers 12
#    -> FAIL, 112 tensors present / 197 required.  Output: fetch_model.txt

# 2. (edit reference/fetch_model.py: arch-aware EXPECT_CONFIG + INVENTORY)

# 3. Re-check nomic, then regression-check all four shipping models.
& .\.venv-ref\Scripts\python.exe reference\fetch_model.py --model nomic-ai/nomic-embed-text-v1.5 --layers 12
& .\.venv-ref\Scripts\python.exe reference\fetch_model.py --model BAAI/bge-base-en-v1.5 --layers 12
& .\.venv-ref\Scripts\python.exe reference\fetch_model.py --model sentence-transformers/all-MiniLM-L6-v2 --layers 6
& .\.venv-ref\Scripts\python.exe reference\fetch_model.py --model BAAI/bge-large-en-v1.5 --layers 24
& .\.venv-ref\Scripts\python.exe reference\fetch_model.py --model BAAI/bge-small-en-v1.5 --layers 12
#    -> all five OK

# 4. Tokenizer compatibility probe
& .\.venv-ref\Scripts\python.exe tasks\0068-m13-nomic-spike-and-oracle\probe_nomic_tokenizer.py

# 5. Architecture probe. nomic's remote modelling code needs einops; installed
#    into .venv-ref ONLY -- never into C:\dev\mlir-aie\ironenv, which is the
#    toolchain that took the most work to get running.
& .\.venv-ref\Scripts\python.exe -m pip install einops
& .\.venv-ref\Scripts\python.exe tasks\0068-m13-nomic-spike-and-oracle\probe_nomic_arch.py

# 6. Does N=6144 compile? (iron env, dot-sourced -- every step below needs this)
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python tasks\0068-m13-nomic-spike-and-oracle\probe_n6144.py
python tasks\0068-m13-nomic-spike-and-oracle\probe_n6144_part2.py   # boundary
python tasks\0068-m13-nomic-spike-and-oracle\probe_n6144_part3.py   # controls

# 7. (fix experiments/m5-pretiled-gemm/gemm_pretiled.py: drive the fill/drain
#     walk from the guarded tb_n_rows)

# 8. Re-verify the fix: guard-fired shapes, below-threshold shapes, small-M tail.
#    PURGE the JIT cache first -- trap 7c served a stale entry and made the
#    first fixed build look broken.
python tasks\0068-m13-nomic-spike-and-oracle\verify_fix_tb_n_rows.py

# 9. Production identity. Build into SCRATCH dirs, never over artifacts_base.
python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 `
       --hidden 768 -n 48 `
       --out tasks\0068-m13-nomic-spike-and-oracle\scratch_export_base
#    then `git stash` the gemm_pretiled.py fix and repeat into
#    scratch_export_base_unfixed, and compare the two final.xclbin files.
#    -> 74 differing bytes, inside the 0029 UUID budget.
#    The scratch build trees were DELETED afterwards: the evidence is the
#    comparison in fix_tb_n_rows.txt, not 1.2 MB of regenerable binaries.
```

## Problems hit

**`fetch_model.py` rewrites `CHECKPOINT.json` with CRLF where the repo stores
LF.** Symptom: `models/bge-small-en-v1.5/CHECKPOINT.json` showed as modified
with an empty `git diff`. Cause: `Path.write_text()` on Windows translates `\n`
to `\r\n`; git's autocrlf hides it from `diff` but not from `status`. The
content and the pinned sha256 (`3c9f3166…`, matching `hub.cpp`'s catalogue) were
unchanged, so the file was restored with `git checkout --`.

Worth recording because `hub.cpp:497-502` claims to write this file
**"byte-identically to `reference/fetch_model.py`'s `json.dumps(indent=2)` with
no trailing newline"** — and if Python emits CRLF on Windows while the C++ emits
LF, that claim is not true. Nothing hashes `CHECKPOINT.json`, so this is
cosmetic today; it is logged because a byte-parity claim that is quietly false is
the kind of thing this repo has been bitten by before.

## Artifacts

- `fetch_model.txt` — the deliberate BERT-check failure, the full departure list
- `probe_nomic_tokenizer.py` / `.txt` — tokenizer compatibility
- `probe_nomic_arch.py` / `.txt` — the three empirical architecture questions
- `probe_n6144.py` / `.txt` — go/no-go on the widest GEMM stream this project
  has attempted

## Next

`tasks/0069` — the `arch=2` container: `ARCH_NOMIC_ROPE_SWIGLU = 2`,
`pack_nomic()` fusing `fc11`+`fc12` into one `[768, 6144]` `ffn_up`, zero-filled
bias and position tensors with an explicit non-zero/zero assertion, and the
prefix table labelled as this project's choice.
