# 0136 — arch=3 wired through the runtime: gte-multilingual-base encodes on the NPU at 2.2–3.7e-04, nomic bit-identical throughout

**Date**: 2026-08-27
**Goal**: wire `gte_new_rope_geglu` (arch=3) through `runtime/src/main.cpp` —
the container [`0135`](../0135-gte-container/TASK.md) packed, on the tokenizer
[`0133`](../0133-t52-cpp-port/TASK.md) built, validated end to end on the NPU
against the oracle [`0134`](../0134-gte-oracle/TASK.md) wrote. Plus the golden
files (`make_goldens_gte.py` / `check_reference_gte.py`). `hub.cpp`,
`npue_pack.cpp` and the catalogue are deliberately untouched — a later task
owns them.

## Results, headline

| check | result |
|---|---|
| nomic bit-identity across all edits | **PASS** — sha256 `ee1734216fe25c2f...` before AND after (full hashes below) |
| gte NPU vs numpy oracle, 8 varied texts | **1-cos 2.248e-04 … 3.683e-04** — the expected bfp16-datapath decade (bge-base measures ~2.3e-04 median) |
| C++ tokenizer path vs AutoTokenizer | **8/8 byte-exact**, ids incl. padding, via `--tokenize` |
| semantic gate (`verify_semantics.py`) | **PASS** — gate 24/24, probe 12/12, control fails-as-required ([report](semantic_gate_gte.json)) |
| goldens + oracle gate (`check_reference_gte.py`) | **PASS** — 17/17 within rel_fro 5e-06, worst valid-rows 2.0e-06; 3 discriminating controls all >>1000× worse |
| 0110 truncation contract on arch=3 | refuses at 402 tokens naming index+count; `--allow-truncation` warns and cuts |
| promptless contract on arch=3 | `--prefix` → refused; serve `prompt_name` → HTTP 400 `invalid_request_error`; plain POST OK |

## What changed (all in `runtime/src/main.cpp` unless noted)

**Step 0 — the pre-existing single-source defect, fixed FIRST** (`:587-595`).
The dispatch-time arch refusal in `set_model_shape()` restated
`encoder_implemented()`'s two string literals inline while its comment claimed
that function was "the single source" — so arch=3 would have landed in one
list and been refused by the other. It calls the real list now
(`!encoder_implemented(arch) || arch == "gemma3_mqa_rope_geglu"` — the gemma
term stays because arch=1 IS implemented, but by `run_gemma_mode()`, and a
gemma container reaching this path means the diversion was bypassed).
Verified: `grep -n '"bert_abs_gelu_postln"' runtime/src/main.cpp` → exactly
one hit (`:238`, the whitelist). `"nomic_bert_rope_swiglu"` similarly appears
in the whitelist once; its other two hits (`:623`, `:635` error text) are the
arch=2 per-arch *config* branch, not a second implemented-list.

**arch 3 in the whitelist** — `encoder_implemented()` `:249`.

**Geometry/config** — `set_model_shape()`:
* `:436-465` the gemma-diversion comment area unchanged; new arch=3 branch at
  `:658-706`: requires `position_embedding_type == "rope"`, `gated_ffn`,
  `glu_halves` beginning `up_first|gate_second` (prefix match — the container
  key is marker + prose), and **`rope_inv_freq` (REQUIRED — a container
  without it refuses, never falls back to `rope_theta`**, which 0134 measured
  wrong by 1.9e-02 relfro at layer 0). Parsed with `npue::json` (the same
  raw-JSON-through-`config_string` pattern the prompts table uses), length
  checked against `head_dim/2`.
* New globals `:145` `g_rope_inv_freq` and `:152-153`
  `enum class GatedAct { Silu, GeluErf }` / `g_gated_act`; both reset
  unconditionally at the top of `set_model_shape()` (`:654-656`) — the same
  no-leak discipline as `g_rope`/`g_gated_ffn`.

**Activation became data** — `:708-730`. The container's `"activation"` key
(written since 0068 for nomic, load-bearing for the first time per 0135) is
read for every gated non-gemma arch: `"silu"` → SiLU, `"gelu"` → **exact erf
GELU** (`gelu_erf_exact()` `:160-165` — `std::erf` in double, rounded once,
matching the numpy oracle's float64 erf; NOT `gelu8`'s polynomial, NOT
Gemma's tanh). Missing key on arch=2 defaults to `"silu"` (containers may
predate the read; the shipped ones do carry it); missing on arch=3 refuses;
an unknown value refuses on either. Dispatched in all three gated-activation
sites, each as a new arm ahead of byte-for-byte untouched SiLU code:
`swiglu_cpu` `:1968`, `dequant_act_bf16` (the fused bf16/bfp16 epilogue —
the arm that actually runs for gte) `:2133`, and the int8 fused lambda
`:2284` (no int8 gte container exists — `pack_npue.py --int8` refuses arch=3
— but the arm must not silently run SiLU if one arrives).

**RoPE from `rope_inv_freq`** — `apply_rope_qkv()` `:2800-2824`. When
`g_rope_inv_freq` is non-empty the cos/sin tables are built from it directly
(double angle, round-once, same NeoX concat(freqs,freqs) layout
`gemma_rope_tables()` emits); arch=2 keeps the theta path bit-for-bit. The
1/√64 scale is already folded into Q weights AND Q bias by the packer
(exact — RoPE is linear), so `qk()` needed nothing.

**Tokenizer** — `AnyTokenizer` facade `:387-423` + `load_tokenizer()`
`:425-478`, member swap at `:6829`. Decision, per the brief's
smaller-cleaner-change question: the facade lives in main.cpp rather than
giving `XlmrTokenizer` a WordPiece-shaped `encode()`, because max_len /
padding / truncation are this runtime's *policy* (0110's contract runs on the
`Encoded` fields), not a property of the Unigram algorithm —
`tokenizer_xlmr.cpp` stays the line-for-line port of its Python reference.
And it beats branching at call sites because the `encode()` calls sit inside
`EmbedService::chunk()` and the `--tokenize` loop — a branch at each is the
drift shape Step 0 just removed. Semantics: arch=3 loads
`tokenizer.xlmr_table` via `from_table_bytes`; specials checked against
XLM-R's <s>=0, </s>=2, <pad>=1 rather than assumed; truncation keeps
`<s> + first (max_len-2) pieces + </s>` (HF's longest_first under the
`<s> A </s>` post-processor — verified 8/8 against AutoTokenizer including
padding); `n_tokens_full`/`truncated` feed `check_truncation()` unchanged.
Everything else (arch 0/2 vocab path, pre-0036 loose-file fallback) behaves
exactly as before behind the same interface.

**What needed NO change, verified rather than assumed**: `embeddings.position`
is packed as zeros so the `wv+pv+w_typ` gather is gte's `word+token_type[0]`
exactly; biases are staged unconditionally and the container's are real
(0135); LN eps is hardcoded 1e-12 in the AVX2 kernels and gte's
`layer_norm_eps` IS 1e-12; CLS pooling and `l2_normalize:true` ride the
existing data-driven paths; the promptless path refuses `--prefix` and serve's
`prompt_name` (both exercised live, table above).

**New files**: `reference/corpus_gte.py` (re-exports the shared 4-sentence
corpus, SEQ_LEN 64), `reference/make_goldens_gte.py` (0116 `--seq` flag
included; filename carries `_s<seq>_`), `reference/check_reference_gte.py`,
`reference/goldens_gte/gte-multilingual-base_l12_s64_boundary.safetensors`
(11.1 MB, 19 tensors).

**Artifacts stored in THIS directory** (rule 6 — `<scratch>` in the commands
below was the session scratchpad; the inputs and outputs are kept here):
[`nomic_in.txt`](nomic_in.txt) / [`nomic_before.f32`](nomic_before.f32) /
[`nomic_after.f32`](nomic_after.f32) (the bit-identity pair),
[`gte_in.txt`](gte_in.txt) / [`gte_out.f32`](gte_out.f32) /
[`gte_ref_compare.py`](gte_ref_compare.py) (the 1-cos table's producer),
[`probe_st_disagree.py`](probe_st_disagree.py) (problem 1's probe),
[`semantic_gate_gte.json`](semantic_gate_gte.json).

## Exact commands

```powershell
# baseline BEFORE any edit (binary from 0133's build), run twice -- deterministic
cd runtime
.\build\npuembed.exe .. --model nomic-embed-text-v1.5 --artifacts artifacts_nomic_bfp16 `
  --embed <scratch>\nomic_in.txt <scratch>\nomic_before.f32 --threads 24 --prefix search_document

# build
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL && cmake --build build --config Release'
#   -> [1/2] main.cpp.obj, [2/2] npuembed.exe, zero warnings

# (a) nomic bit-identity AFTER
.\build\npuembed.exe .. --model nomic-embed-text-v1.5 --artifacts artifacts_nomic_bfp16 `
  --embed <scratch>\nomic_in.txt <scratch>\nomic_after.f32 --threads 24 --prefix search_document
Get-FileHash -Algorithm SHA256 <scratch>\nomic_before.f32, <scratch>\nomic_after.f32

# (b) gte e2e on the NPU (8 texts: 4 EN + de/no/ja/fr one-liners, file-only per corpus hygiene)
.\build\npuembed.exe .. --model gte-multilingual-base --artifacts artifacts_nomic_bfp16 `
  --embed <scratch>\gte_in.txt <scratch>\gte_out.f32 --threads 24
..\.venv-ref\Scripts\python.exe <scratch>\gte_ref_compare.py   # AutoTokenizer + reference/encoder_gte.py per text

# tokenizer diff (padded ids, both sides)
.\build\npuembed.exe .. --model gte-multilingual-base --tokenize <scratch>\gte_in.txt 64 > <scratch>\gte_tok_cpp.txt

# truncation contract + prompts contract
.\build\npuembed.exe .. --model gte-multilingual-base --artifacts artifacts_nomic_bfp16 --embed <scratch>\gte_long.txt out.f32 --threads 24                      # -> ERROR, exit 2
.\build\npuembed.exe .. ... --allow-truncation                                                                                                                  # -> WARNING, exit 0
.\build\npuembed.exe .. ... --prefix search_query                                                                                                               # -> refused, exit 2
.\build\npuembed.exe .. --model gte-multilingual-base --artifacts artifacts_nomic_bfp16 --serve 18742 --threads 24   # /health ok; prompt_name -> 400; plain POST -> dim 768

# semantic gate (report into THIS task dir -- see problems)
python tools\verify_semantics.py --models gte-multilingual-base --exe runtime\build\npuembed.exe --artifacts artifacts_nomic_bfp16 --threads 24 --out tasks\0136-gte-runtime\semantic_gate_gte.json

# goldens + oracle gate
.venv-ref\Scripts\python.exe reference\make_goldens_gte.py
"C:\Users\vegar\.conda\envs\iron\python.exe" reference\check_reference_gte.py
```

## The numbers

**Nomic bit-identity** (4 texts, `search_document`, 24 threads, bfp16 set):

```
before  ee1734216fe25c2f0bc2633974f29c20785b6646358276377b555a1d9438fd29
after   ee1734216fe25c2f0bc2633974f29c20785b6646358276377b555a1d9438fd29
```

(Also run twice pre-edit — run-to-run deterministic, so the hash is a real
instrument.)

**gte NPU (bfp16 datapath, seq 64, batch tier 16) vs `encoder_gte.py` (fp32
numpy), CLS + L2, 1-cos per text** — texts in the session scratchpad's
`gte_in.txt`, EN×4 then de/no/ja/fr:

| text | tokens | 1-cos |
|---|---:|---:|
| text_0 (EN, NPU) | 18 | 2.558e-04 |
| text_1 (EN, RoPE) | 17 | 2.865e-04 |
| text_2 (EN, ship) | 20 | 2.248e-04 |
| text_3 (EN, garden) | 21 | 2.667e-04 |
| text_4 (de) | 15 | 3.671e-04 |
| text_5 (no) | 18 | 3.277e-04 |
| text_6 (ja) | 15 | 2.384e-04 |
| text_7 (fr) | 14 | 3.683e-04 |

Worst 3.683e-04 — same decade as bge-base's bfp16 ~2.3e-04 median, as the
brief predicted. Per T51's lesson this is 8 varied sentences, not a tail
study; the tail instrument for gte belongs to the gates task.

**Oracle gate** (`check_reference_gte.py`, iron env, numpy only): 17/17
comparisons pass; valid-rows rel_fro 8.8e-08 (emb.ln) → 2.0e-06 (worst
layer), pooled 1-cos 8.0e-08 vs repaired HF / 1.3e-07 vs
sentence-transformers. Discriminating controls, same goldens: plain
theta=20000 → 7.2e-02 relfro, theta=160000-no-correction → 1.8e-02,
GELU-on-up-half → 1.03 — all ≥4 decades above the correct oracle.

## Problems hit, honestly

1. **A THIRD uninitialised-buffer landmine in the gte reference, beyond
   0134's two.** `make_goldens_gte.py`'s first run failed its own
   cross-pipeline gate: SentenceTransformer disagreed with the manual
   pipeline by **9.0e-02** max-abs on normalized CLS — after both 0134
   repairs were applied. Cause: the same transformers-v5 meta-device
   mechanism also materialises `embeddings.position_ids`
   (`modeling.py:308`, `persistent=False`) as garbage. 0134 never saw it
   because its probe passes explicit `position_ids`; any forward WITHOUT
   them — which is what SentenceTransformer does — indexes the rope cache
   with that garbage: out-of-bounds throws (the loud half 0134 recorded),
   **in-bounds silently applies wrong positions** (the 9.0e-02). Probe
   transcript: an explicit-positions padded batch matched unpadded singles
   at 6e-06 max-abs while the derived-positions path crashed with
   `IndexError: index 2933899067392 ... size 64`. Repair (now in
   `make_goldens_gte.py::repair_rotary`, applied to BOTH oracles):
   re-register the buffer exactly as the author's `__init__` does. After
   it: pipelines agree at **6.7e-08**. Consequence for the gates task:
   *any* MTEB / sentence-transformers run of gte on this env is silently
   position-scrambled without this repair — it must be carried there too.
2. **`verify_semantics.py --out` defaults into tasks/0121 and I clobbered
   the six-model report.** The first gte gate run overwrote
   `tasks/0121-semantic-gate/semantic_gate.json` (uncommitted) with a
   one-model report. Restored by re-running the default six-model sweep
   (all PASS again, same protocol) and re-running gte with `--out` into
   this task's directory. Lesson: pass `--out` when the run is not the
   canonical catalogue sweep.
3. **HF and the numpy oracle genuinely diverge on PADDED rows — up to
   1e-03 by layer 11 — while agreeing at 1.4e-06 on every valid row.**
   First `check_reference_gte.py` run failed 9 tensors on whole-tensor
   rel_fro (the nomic pattern this file mirrors compares whole tensors and
   happens to pass — nomic's two implementations agree on padded rows,
   gte's do not). Measured split: valid rows 1.355e-06, padded rows up to
   1.090e-03, CLS unaffected (1.9e-06). Padded rows are don't-cares — CLS
   pooling reads row 0, the runtime masks them out of mean pooling, HF's
   consumers never read them — so the gate now scores valid rows and
   REPORTS the padded max|d| beside it, ungated, with the reasoning in
   `compare()`'s docstring. Not chased further: which of the two
   implementations' padded-row arithmetic "is right" is a question with no
   consumer.
4. **`--model <relative-path>.npue` resolves against the CWD, not against
   root** (`resolve_model_path()` tries `std::ifstream(want)` first), so
   the brief's `--model models\gte-multilingual-base.npue` form fails from
   `runtime/`. Used the name form (`--model gte-multilingual-base`), which
   resolves through `discover_models(root)` as every script here does.
5. **Left for the hub/gates task, on purpose**: `list` shows gte as
   "no design" because a locally-packed (non-catalogue) model is required
   to be plain-bf16 (`pick_artifacts(..., "bf16")`, tasks/0104's safe
   default) while the only fitting set is bfp16 — so every run here names
   `--artifacts artifacts_nomic_bfp16` explicitly, which wins over
   adoption entirely. The catalogue entry (with `datapath: "bfp16"` if
   adopted), MTEB, and the C++ packer mirror are that task's scope, as is
   re-checking `tools/verify_tokenizer_xlmr.py --cli` semantics against
   the new `--tokenize` mode if it ever drives the exe instead.

## What is NOT done here

`hub.cpp` routing/catalogue entry, `npue_pack.cpp` (C++ packer mirror,
which must call `generate_xlmr_tokenizer_table()`), MTEB gate, int8
calibration for gte, T51-style tail study, and the register/docs updates —
per the brief, later tasks own all of these. T52's last open item
("the arch-3 wiring") is now built and validated; closing the thread is the
register-owner's call, not this log's.
