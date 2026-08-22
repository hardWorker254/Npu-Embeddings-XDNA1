# 0071 — M13: making nomic-embed-text-v1.5 shippable

**Date:** 2026-08-21
**Status:** DONE

## Goal

`tasks/0070` got nomic running on the array (hardware `1-cos` 2.599e-05, e2e
2.401e-05) but left three things between it and shipping in 0.3.0:

1. `--prefix` on the CLI, fail-visible (the container carries the answer;
   `npuembed` was not reading it).
2. A C++ mirror of `tools/pack_npue.py`'s `pack_nomic()`, so a fresh clone
   can pack nomic without Python (matching what tasks/0065 already did for
   Gemma).
3. A catalogue row in `runtime/src/hub.cpp`, so `npuembeddings serve
   nomic-embed-text-v1.5` on a machine that has never seen this repo's
   `models/` fetches, verifies, packs and runs it.

Scope: `runtime/` (all of it), `tools/verify_pack_parity.py`,
`tools/verify_embed_e2e.py`. Explicitly NOT touched: `reference/`,
`experiments/`, `tools/pack_npue.py`, `tools/npue.py`, `docs/`.

## 1. `--prefix`

The container already carries a `prompts` table (`search_document` /
`search_query` / `clustering` / `classification`) and a `prompt_default` of
`"search_document"`, written by `pack_nomic()` and labelled `prompts_source`
as this project's OWN choice, not the checkpoint's (`config_sentence_
transformers.json` for this checkpoint has no `prompts` dict at all).

**Reading it**: `.npue`'s JSON directory stores an object-valued config key
as raw JSON text (`npue::File::config_string()` returns
`{"search_document":"search_document: ",...}` verbatim for a nested value,
never parses it). `set_model_shape()` in `runtime/src/main.cpp` now parses
that text with the existing `npue::json` DOM parser
(`runtime/include/json_min.hpp`, already linked into the `npuembed` target
for `gemma_tokenizer_gen.cpp` — no new CMake wiring needed) into two new
file-scope globals, `g_prompts` (name → prefix text) and `g_prompt_default`,
both cleared and re-populated on every `set_model_shape()` call, same
discipline as `g_rope`/`g_gated_ffn`. A container with no `"prompts"` key
leaves `g_prompts` empty via a try/catch around `config_string("prompts")` —
`g_prompts.empty()` IS the "this model has no prefix concept" signal, not a
second bool that could drift from it. A `"prompts"` key that parses to zero
entries, or a `prompt_default` that is not one of its own keys, both THROW —
a malformed container refuses rather than silently running prefix-less.

**Applying it**: `resolve_prefix(argc, argv)` (new, in `main.cpp`) returns
`""` immediately for `g_prompts.empty()` (BERT models — completely
unaffected, verified below). Otherwise it reads `--prefix NAME` from argv,
falls back to `g_prompt_default` when absent, looks `NAME` up in
`g_prompts`, and **always prints which prefix it chose, on stderr**,
distinguishing an explicit `--prefix` from the container default:

```
  prefix     'search_document' -> "search_document: " (container default -- no --prefix given)
```

An unknown `--prefix` name throws, listing the real options (sorted). Called
exactly once, inside `make_service()` (shared by `--embed` and `--serve`,
called once each), so it fires only when text is actually about to be
embedded — not on every invocation of the binary (a golden-check run,
`--bench`, a probe). `EmbedService` gained a `prefix_text` member; `chunk()`
prepends it to the raw text before `tok.encode()`, the same place
`verify_embed_e2e.py`'s former workaround did it.

**HTTP `/v1/embeddings`**: the OpenAI shape has no per-request prefix field,
and adding a non-standard one breaks that compatibility contract. Chose a
**server-wide default set at `serve` startup** (whatever `--prefix`/the
container default resolved to) over a per-request field — simpler, and one
process serves one model with one intended usage. Made visible two ways:
the startup NOTE line (in addition to the `prefix` line `resolve_prefix()`
already prints), and a `"prefix"` key in `GET /health`'s JSON response,
present only when the model has a prefix concept at all — a BERT model's
`/health` is byte-identical to before.

**Subcommand form**: `npuembeddings embed/serve <model> ... --prefix NAME`
is scanned in the subcommand→flag translation block and passed through to
the flag form unchanged (absent when not given, so nothing new is forced
onto the flag-form default path).

### Verification

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime
cmake --build build --config Release
```

- **Default (container's own default), banner check**:
  `npuembed.exe . --model nomic-embed-text-v1.5 --artifacts artifacts_nomic --threads 16 --embed <in.txt> <out.f32>`
  → stderr: `prefix     'search_document' -> "search_document: " (container default -- no --prefix given)`
- **Explicit `--prefix search_query`**: stderr `prefix     'search_query' -> "search_query: "` (no "container default" suffix); output embeddings differ from the default-prefix run (`out_default.f32` vs `out_query.f32`, confirmed NOT byte-equal — the prefix genuinely changes what gets encoded).
- **Unknown `--prefix bogus`**: exit code 2,
  `error: --prefix 'bogus' is not one of this model's task prefixes: [classification, clustering, search_document, search_query]`
- **BERT model, `--prefix bogus`**: `npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --threads 16 --prefix bogus --embed <in.txt> <out.f32>` — no error, no prefix banner line, and the output is **byte-identical** (sha256
  `f7f724c19a26f7797556a46f60c12a39056b7c259c149cb94009b8f696969594`) to the
  same run with no `--prefix` at all.
- **`--serve` + `/health`**: started `npuembed.exe . --model nomic-embed-text-v1.5 --artifacts artifacts_nomic --threads 16 --serve 8099`;
  `curl http://127.0.0.1:8099/health` →
  `{"status":"ok","model":"nomic-embed-text-v1.5-npu","backend":"amd-xdna2-npu","prefix":"search_document: "}`;
  `POST /v1/embeddings {"input":"hello"}` returned a real embedding.
- **Subcommand passthrough**: `npuembeddings.exe embed nomic-embed-text-v1.5 <in.txt> <out.f32> --prefix search_query` produced output byte-identical to the flag-form `--prefix search_query` run.

## 2. The C++ packer mirror (`prepare_model_nomic`)

`runtime/include/npue_pack.hpp` / `runtime/src/npue_pack.cpp` gained
`prepare_model_nomic()`, mirroring `tools/pack_npue.py`'s `pack_nomic()`
tensor-for-tensor and byte-for-byte:

- Same assertions before packing anything: `qkv_proj_bias`/`mlp_fc1_bias`/
  `mlp_fc2_bias` all false, `prenorm` false, `activation_function=="swiglu"`
  + `hidden_act=="silu"`, `rotary_emb_interleaved` false,
  `rotary_emb_fraction==1.0`, `rotary_emb_base==1000`,
  `layer_norm_epsilon==layer_norm_eps`, `head_dim` even,
  `hidden==heads*head_dim`.
- **`layer_norm_eps` (`1e-12`) and `rotary_emb_base` (`1000`) are copied
  VERBATIM from `config.json`'s own literal text**, not reparsed and
  reformatted — the same trick `prepare_model_gemma()`'s `cfg_raw()` uses,
  and for the same reason: Python's `json.dumps` happens to reproduce both
  literals unchanged (checked directly:
  `json.dumps(1e-12) == "1e-12"`, `json.dumps(1000) == "1000"` against this
  checkpoint's actual parsed values), so re-deriving them risked drifting
  from Python's float printer for zero benefit.
- **The fp32 (not double) scale fold**: `const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));`
  — exactly `prepare_model()`'s existing pattern for BERT's qkv fold
  (`npue_pack.cpp`'s own comment there names the 1-ULP bug this avoids).
  nomic's qkv fold reuses the EXISTING `add_gemm_b()` helper's `fold`/
  `fold_cols` parameters (`add_gemm_b(w, tag+"qkv", get(attn+"Wqkv.weight"), tile_k, tile_n, layout_json, layout_hash, scale, hidden)`)
  rather than writing new fold logic — nomic's Wqkv is already ONE fused
  `[2304,768]` checkpoint tensor (unlike BERT's three separate Q/K/V
  matrices), so no manual 3-way concatenation is needed here at all.
- **`fc11|fc12` fusion order**: a new helper, `add_gemm_b_concat2()`,
  transposes both `[inter,hidden]` checkpoint tensors independently into
  the SAME `[hidden, 2*inter]` buffer at their own column offset (`fc11` at
  `[0,inter)`, `fc12` at `[inter,2*inter)`), then tiles once — mirrors
  `np.concatenate([up, gate], axis=1)` exactly. Order matters: swapping it
  is the wrong candidate tasks/0068 measured at `rel_fro` 4.022e+00.
- **Zero-filled biases and position table**: `add_zero_bias()` and an
  explicit `std::vector<float> zpos(...)` — nomic has no biases anywhere and
  no absolute-position table (RoPE instead), but `Encoder::stage_all()`
  dereferences `embeddings.position` unconditionally, so the zero tensor
  keeps that read path untouched, exactly as `pack_nomic()`'s docstring
  explains.
- **The odd emission order**: `embeddings.ln.weight` → `tokenizer.vocab` →
  `embeddings.ln.bias`, reproduced exactly (not reordered to something more
  "natural") because the parity gate is byte equality, not tolerance.
- **`n_reachable`** (for the `not_implemented` message's vocab-padding
  note) is computed the same way Python's `str.splitlines()` does for this
  specific plain-LF-ASCII input (`count_lines()`, documented as NOT a
  general `splitlines()` reimplementation).
- Config JSON built as a hand-concatenated `std::string`, key order and
  every string copied character-for-character from `pack_nomic()`'s dict
  literal (including the `prompts` table, `prompt_default`,
  `prompts_source` and `l2_normalize_note` explanatory strings) — this is
  what the byte-parity gate below actually exercises.
- Wired into `main.cpp`'s `--prepare-model` dispatch: reads `config.json`'s
  `model_type` (same detection main.cpp already uses for `gemma3_text`) and
  routes to `prepare_model_nomic()` for `"nomic_bert"`, placed AFTER the
  existing tile_k/tile_n/layout/pooling/source_repo resolution (which nomic
  shares unchanged with the BERT path) and BEFORE the generic
  `prepare_model()` call.

### Verification — byte parity

```powershell
& .\.venv-ref\Scripts\python.exe tools\verify_pack_parity.py `
    --model-dir models\nomic-embed-text-v1.5 --tile-n 48
```

```
  tile_n 48, model nomic-embed-text-v1.5 (arch=2 nomic_bert)
  pack_npue.py    322.08 MB, json 32882 B at 64, data 322.05 MB at 36864, v1 arch2 flags1
  --prepare-model 322.08 MB, json 32882 B at 64, data 322.05 MB at 36864, v1 arch2 flags1

  python : 5bcb931a908887d30f30a6ea87e7052fdcaf4e60c1221565644d208ebf8f387b
  c++    : 5bcb931a908887d30f30a6ea87e7052fdcaf4e60c1221565644d208ebf8f387b

PASS -- byte-identical
```

**PASSED on the first attempt** (no differing-byte diagnosis was ever
needed). `tools/verify_pack_parity.py` itself gained an `is_nomic` flag
purely for diagnostic labelling (`" (arch=2 nomic_bert)"` in the printed
header) — no behavioural change was needed, because nomic already falls
through the existing "not gemma → needs `model.safetensors` +
`vocab.txt`" branch correctly, and both packers already dispatch on
`config.json`'s own `model_type` internally rather than needing a
different CLI shape from this script. Re-ran the existing MiniLM case
afterward to confirm the change is a no-op there: still
`PASS -- byte-identical`.

## 3. The catalogue row

`runtime/src/hub.cpp`'s `table()` gained:

```cpp
{"nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5",
 "9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c",
 "mean", 768, 12, 12, 3072, 48, 546.9,
 "RoPE + gated SwiGLU (arch=2); same array designs as bge-base; "
 "needs --prefix (...)", /*gated=*/false, /*gemma=*/false,
 /*gated_ffn=*/true},
```

**sha256 re-derived locally**, independently of the value the launch brief
supplied, with plain `hashlib.sha256` against
`models/nomic-embed-text-v1.5/model.safetensors` (546,938,168 bytes):
`9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c` —
**matches**, and also matches `models/nomic-embed-text-v1.5/CHECKPOINT.json`
(written by `reference/fetch_model.py` when this checkpoint was first
brought up in tasks/0068).

`kFiles[]` (the BERT fetch list: `model.safetensors`, `vocab.txt`,
`config.json`, `1_Pooling/config.json`) needed **no changes** — confirmed
by inspecting `models/nomic-embed-text-v1.5/` directly, all four files
present. `verify_config()`'s generic `hidden_size`/`num_hidden_layers`/
`num_attention_heads`/`intermediate_size` cross-check against `config.json`
and the `1_Pooling/config.json` pooling check also needed no changes —
confirmed by reading nomic's actual `config.json` (768/12/12/3072, matches
the catalogue row) and `1_Pooling/config.json`
(`pooling_mode_mean_tokens: true`, matches `"mean"`).

**`gated_ffn=true` is load-bearing, not decorative**: without it,
`design_fits()`/`pick_artifacts()` cannot distinguish nomic's `{768,3072}`
K-set from bge-base's identical one (tasks/0069 thread T31) — confirmed by
`npuembeddings list` reporting `nomic-embed-text-v1.5` as `ready` against
`artifacts_nomic`'s actual gated design set, not merely "some 768/3072
design exists."

**The packer dispatch inside `ensure_model()`** reads `config.json`'s own
`model_type` (a new tiny `json_str()` scanner, same "flat, machine-
generated" reasoning as the existing `json_int()`) rather than branching on
the catalogue's `gated_ffn` bit — `hub.hpp`'s own comment on that field
warns against inferring architecture identity from it, since a future
non-nomic gated-FFN BERT-family model would otherwise silently inherit
nomic's packer.

### Verification — the real gate: cold-path fetch, pack, and byte identity

```powershell
mv models\nomic-embed-text-v1.5.npue tasks\0071-m13-nomic-shippable\...bak   # aside
mv models\nomic-embed-text-v1.5     tasks\0071-m13-nomic-shippable\...bak   # aside, whole checkpoint dir too
.\runtime\build\npuembeddings.exe embed nomic-embed-text-v1.5 in.txt out.f32
```

Ran the **genuinely cold** case per tasks/0051's lesson (a warmed-up test
proves the layout works, not that the search is right): moved the ENTIRE
`models/nomic-embed-text-v1.5/` checkpoint directory aside too, not just
the `.npue`, so this exercised a real network fetch (546.9 MB from
`huggingface.co`, ~60 s), not a re-pack of an already-downloaded
checkpoint. Output confirmed every stage ran: `Fetching it from nomic-ai/
nomic-embed-text-v1.5` → per-file `get`/progress-percent lines for all four
files → `hash model.safetensors` → `ok 9e7d262b1fe5ea35...` → `pack
nomic-embed-text-v1.5.npue` → `ready ...` → then straight into the encode
(`prefix 'search_document' -> ... (container default)`, `PASS`-shaped
banner, `wrote ... [2, 768] fp32`) — the whole chain: fetch → sha256 pin
check → config cross-check → pack → design pick (`gated_ffn=true` routed it
to `artifacts_nomic` correctly) → encode, no manual step between them.

**Byte identity, cold-built vs the container moved aside**:

```
cold-built : 5bcb931a908887d30f30a6ea87e7052fdcaf4e60c1221565644d208ebf8f387b
moved-aside: 5bcb931a908887d30f30a6ea87e7052fdcaf4e60c1221565644d208ebf8f387b
IDENTICAL
```

Also re-ran the hardware golden-check on the freshly cold-built container
(not the moved-aside one) to make sure the fetch→pack cycle did not somehow
produce a container that packs identically but fails differently at
runtime: `rel_fro` **6.119e-03**, `1-cos` **2.599e-05** — unchanged.

Large scratch backups (`...dir.bak` 523 MB, `...npue.bak` 308 MB) were
deleted after the byte-identity check passed; they were never meant to be
kept.

## Removing the `verify_embed_e2e.py` workaround

Per the launch brief: now that `npuembed --embed` applies its own prefix,
the harness stops prepending it to the runtime's side. Changed:

- The file handed to `npuembed --embed` now contains the **raw, unprefixed**
  texts (previously it contained the prefix-prepended texts, the
  workaround).
- The `npuembed` subprocess call gains `--prefix <prompt_default>` when the
  model is nomic (absent for every other model — no behavioural change
  there).
- The REFERENCE side (`sentence-transformers`, which has no `--prefix`
  concept) still gets the prefix manually prepended, unchanged from before
  — this is not a workaround anymore, it is just how you use ST directly.

### Re-run

```powershell
& .\.venv-ref\Scripts\python.exe tools\verify_embed_e2e.py `
    --model nomic-embed-text-v1.5 --artifacts artifacts_nomic --threads 24 `
    --out tasks\0071-m13-nomic-shippable\verify_embed_e2e_nomic.json
```

```
worst 1-cos 2.401e-05   (tolerance 2e-03)
pairwise-similarity error over 78 pairs:
  mean 7.569e-04   p99 2.633e-03   max 2.633e-03
top-10 neighbour overlap: 1.0000
PASS -- text in, vector out, matches the reference
```

**The number did not move at all**: 2.401e-05, bit-for-bit the same figure
tasks/0070 recorded with the harness-side workaround. This is the expected
result — `--prefix` and the harness workaround construct the exact same
input to the tokenizer, just from two different places now — and it is
reassuring rather than uninteresting: it confirms `resolve_prefix()` reads
the SAME `prompts[prompt_default]` text the harness independently reads
from the same container.

Re-ran MiniLM through the same (changed) script as a regression check on
the harness itself: `worst 1-cos` 2.644e-05, `PASS`, unaffected (no
`--prefix` is ever passed for a non-nomic model, and the input-writing
change is a no-op when `ref_texts == texts`).

## Hard regression gate — the four shipping BERT models, plus nomic

```powershell
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2   --artifacts artifacts_b128il --threads 16
.\runtime\build\npuembed.exe . --model bge-small-en-v1.5  --artifacts artifacts_b128il --threads 16
.\runtime\build\npuembed.exe . --model bge-base-en-v1.5   --artifacts artifacts_base   --threads 16
.\runtime\build\npuembed.exe . --model bge-large-en-v1.5  --artifacts artifacts_large  --threads 16
.\runtime\build\npuembed.exe . --model nomic-embed-text-v1.5 --artifacts artifacts_nomic --threads 16
```

| model | `rel_fro` | expected | |
|---|---:|---:|---|
| all-MiniLM-L6-v2 | 4.473e-03 | 4.473e-03 | IDENTICAL |
| bge-small-en-v1.5 | 3.789e-03 | 3.789e-03 | IDENTICAL |
| bge-base-en-v1.5 | 4.297e-03 | 4.297e-03 | IDENTICAL |
| bge-large-en-v1.5 | 3.763e-03 | 3.763e-03 | IDENTICAL |
| nomic-embed-text-v1.5 | 6.119e-03 | 6.119e-03 | IDENTICAL |

All five reproduced exactly, run after all `main.cpp`/`npue_pack.cpp`/
`hub.cpp` changes were in place (the regression run for the BERT four was
done right after the `--prefix` wiring landed, before the packer/catalogue
work; nomic's number was re-confirmed a second time after the cold-path
fetch/pack test, see above — unchanged both times).

**`--embed` bit-identity, MiniLM, with vs without a (no-op) `--prefix`**:
sha256 `f7f724c19a26f7797556a46f60c12a39056b7c259c149cb94009b8f696969594`,
identical both ways.

Full captures: `regress_minilm.txt`, `regress_bgesmall.txt`,
`regress_bgebase.txt`, `regress_bgelarge.txt`, `regress_nomic.txt`,
`coldpath.txt`, `serve_nomic.txt`, `verify_pack_parity_nomic.txt` (+ `2`,
after the diagnostic-label change), `verify_embed_e2e_nomic.txt`/`.json`.

## What was NOT touched, per the launch brief

`reference/`, `experiments/`, `tools/pack_npue.py`, `tools/npue.py`,
`docs/` — confirmed by `git status`/`git diff` below only showing files
under `runtime/`, `tools/verify_pack_parity.py`,
`tools/verify_embed_e2e.py`, and this task directory. `--bench` /
throughput measurement was explicitly out of scope (tasks/0073, needs an
idle machine) and was not run.

## Files touched

- `runtime/include/npue_pack.hpp` — `prepare_model_nomic()` declaration.
- `runtime/src/npue_pack.cpp` — `prepare_model_nomic()`,
  `add_gemm_b_concat2()`, `count_lines()`.
- `runtime/src/main.cpp` — `#include "json_min.hpp"`, `#include <map>`,
  `g_prompts`/`g_prompt_default` globals, `resolve_prefix()`,
  `set_model_shape()`'s prompts-table parsing, `EmbedService::prefix_text`
  + its use in `chunk()`, `make_service()` wiring, `--serve`'s startup NOTE
  and `/health` `"prefix"` key, `print_usage()`'s `--prefix` entry, the
  subcommand `--prefix` passthrough, and the `--prepare-model` dispatch
  branch for `nomic_bert`.
- `runtime/src/hub.cpp` — the catalogue row, `json_str()`, the
  `prepare_model_nomic()` dispatch branch in `ensure_model()`.
- `tools/verify_pack_parity.py` — `is_nomic` diagnostic label (no
  behavioural change).
- `tools/verify_embed_e2e.py` — `--prefix` passed to the `npuembed`
  subprocess for nomic instead of prepending to the runtime's input file;
  reference-side prepending unchanged.

## What surprised me

- **Nothing failed on the first attempt** for either the packer parity gate
  or the cold-path fetch. Given tasks/0070's `swiglu_cpu()` in-place race
  (a bug that passed its own narrower gate and needed a second, harder gate
  to catch), I went in expecting to find at least one byte-parity
  discrepancy or an `ensure_model()` file-list gap. Neither existed —
  nomic's fetch list and config-cross-check keys turned out to already be
  identical to the BERT family's, and the packer mirror's careful
  attention to the specific traps named in the brief (verbatim float
  literals, fp32 scale fold, fusion order, emission order) was apparently
  sufficient on its own.
- **`resolve_prefix()`'s placement inside the `make_service()` lambda**
  (rather than unconditionally in `set_model_shape()` or right after model
  load) was a deliberate choice to keep the prefix banner from appearing on
  every invocation of the binary (golden validate, `--bench`, every probe)
  — worth stating explicitly since it means "prints on stderr" is scoped to
  "when this process is actually about to embed text," not "whenever a
  container with a prompts table is opened."
