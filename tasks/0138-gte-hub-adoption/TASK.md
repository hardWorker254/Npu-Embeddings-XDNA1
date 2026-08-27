# 0138 — gte becomes a first-class catalogue citizen: hub routing, the 7th built-in with pins, the bfp16 adoption record, and a byte-identical C++ packer mirror

**Date**: 2026-08-27
**Goal**: the pieces [`0136`](../0136-gte-runtime/TASK.md) problem 5 deliberately
left for this task — `hub.cpp` routing + the built-in catalogue row (with the
0137-gated **bfp16 adoption**, per the coordinator's decision on
[`0137`](../0137-gte-gates/TASK.md)'s numbers), and `prepare_model_gte`, the
C++ mirror of `pack_gte` ([`0135`](../0135-gte-container/TASK.md)), held to the
existing packers' parity bar.

## Results, headline

| check | result |
|---|---|
| C++ vs Python container parity | **WHOLE-FILE byte-identical** — sha256 `4d59e2c4a5e6f2b9...` both sides (full-file identity is the standing bar `tools/verify_pack_parity.py` holds the other mirrors to; the brief's minimum was data-region only) |
| `tools/npue_data_hash.py` (data region) | **exit 0** — `ea6ef66ddb7efae8...` both sides |
| C++ tokenizer-generation fallback (cached bin absent) | regenerated `xlmr_tokenizer.bin` sha256 `15bc6dd63d6b3c54...` (0133's pinned hash), container again whole-file `4d59e2c4...` |
| `list` | `gte-multilingual-base  ready` in the catalogue table; resolution note names **artifacts_nomic_bfp16** (datapath record, not alphabetical luck) |
| `embed` WITHOUT `--artifacts` | works end to end on the NPU, status line `datapath bfp16-emulated MMAC, C as bf16`; output **bit-identical** to the explicit `--artifacts artifacts_nomic_bfp16` form AND to 0136's stored `gte_out.f32` (`ae25544c22f903fa...` all three) |
| C++-packed container encodes | `--model <scratch>\gte_cpp.npue` embed output bit-identical to the same `ae25544c...` — the C++ bytes produce the exact vectors 0136 validated at 1-cos 2.2–3.7e-04 against the numpy oracle |
| semantic gate, all SEVEN models | **PASS** — six at their 0137 v6 shape (bge-small's two known probe misses included), gte 30/30 gate + 15/15 probe, resolved via catalogue |
| tail gate, all SEVEN models | **PASS** — gte reproduces 0137's baseline exactly (median 3.419e-04, p99 5.160e-04, max 7.361e-04, 2.2×) |
| nomic bit-identity regression | `ee1734216fe25c2f...` — identical to 0136's recorded before/after hash |
| `warn_if_unpinned` | did NOT fire for gte (pin present) — every embed transcript above is warning-free |

## What changed

### `runtime/include/hub.hpp`
* `CatalogEntry` grows **`bool gte`** (between `qkv_n` and `datapath`, so every
  existing positional initialiser is untouched): selects the third fetch file
  list and mirrors exactly how `gemma` is kept separate from `gated` and
  `gated_ffn` — a future model sharing one property must not inherit the rest.
* The `qkv_n` comment now notes gte states 2304 explicitly (== 3*768, so 0
  would behave identically in `design_fits()`; the 0138 brief fixed the
  geometry fields per the container).

### `runtime/src/hub.cpp`
* **The 7th built-in row**: `gte-multilingual-base` /
  `Alibaba-NLP/gte-multilingual-base`, sha256 pin
  `f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0`
  (**verified against the local `models/gte-multilingual-base/model.safetensors`
  before hardcoding** — `sha256sum` matched, as did all four secondary pins
  below), `cls`, hidden 768, 12 layers, 12 heads, ffn 3072, tile_n 48,
  `gated_ffn=true`, `qkv_n=2304`, `gte=true`, 582.5 MB download.
  **Pinning-scheme note, as the brief asked**: the table's scheme pins exactly
  ONE file per row — `model.safetensors`, the only hash `ensure_model()`
  enforces. Followed. The other four hashes from 0127 were re-verified locally
  (`tokenizer.json f59925fc…`, `tokenizer_config.json 24cebbf2…`,
  `special_tokens_map.json 8c785abe…`, `config.json 711bdc81…`) and are
  recorded in the row's comment for traceability, NOT enforced by the fetch
  path — that difference is the scheme's, not this row's.
* **Adoption record**: `"gte-multilingual-base"` appended to `kAdoptedBfp16`
  — the 0104 discipline exactly: an explicit allowlist entry, never an
  inherited default (bge-small remains the counter-example row), with the
  0137 gate verdict in the adjacent comment (MTEB +0.06 / worst −0.06).
* **`kFilesGte`** (third `Want[]`): model.safetensors, config.json,
  tokenizer.json, tokenizer_config.json, special_tokens_map.json,
  1_Pooling/config.json, modules.json. Subdirectory fetch needs nothing new —
  `download()` creates parent directories, and the BERT set already fetches
  `1_Pooling/config.json` the same way.
* **`probe_repo`**: `e.gte = (mt == "new")`;
  `e.gated_ffn = e.gemma || e.gte || (mt == "nomic_bert")`. The existing
  `tile_n` search needs no change for this geometry: qkv_used = 3·768 = 2304,
  and {2304, 768, 6144, 768} all divide 48·8 — it lands on 48, same as nomic.
* **`load_user_catalog` / `add_to_user_catalog`** carry the `gte` bit, so an
  `add`-ed gte finetune round-trips through `models/catalog.json`.
* **`ensure_model`**: fetch list branches `gemma → gte → default`; the pack
  dispatch gains a `model_type == "new"` branch (the checkpoint's OWN
  config.json, same rule as the nomic branch, never a catalogue bit) calling
  `prepare_model_gte(..., max_seq=64)`. 64, not the BERT/nomic 256, because
  the Python-packed reference this mirror is held byte-identical to was packed
  `--max-seq 64` (0135), the shipped designs and goldens for this model are
  seq-64, and under RoPE the position table is zeros so max_seq only caps
  request length.

### `runtime/include/npue_pack.hpp` + `runtime/src/npue_pack.cpp`
* **`prepare_model_gte`** — the faithful port of `pack_gte`:
  * Same emission order, including the parity-anchoring
    `embeddings.ln.weight → tokenizer.xlmr_table → embeddings.ln.bias`
    interleaving, then per layer qkv(+bias)/attn_out(+bias)/ln1/ffn_up(+zero
    bias)/ffn_down(+bias)/ln2. `w.write(..., arch=3)`.
  * Every fail-closed assert ported in `pack_gte`'s order: model_type "new",
    hidden_act "gelu", position_embedding_type "rope", rope_theta 20000 +
    rope_scaling {ntk, 8.0, mixed_b None} (read SCOPED to the rope_scaling
    object — a bare `"type"` search is unsafe in a file with several `*_type`
    keys), type_vocab_size 1, layer_norm_type, logn_attention_scale,
    pack_qkv. The nomic packer's own bias assert is untouched — arch 3 has
    its own checks and REAL biases.
  * The **Q-third scale fold hits both weight and bias** via the existing
    `add_gemm_b(fold=scale, fold_cols=hidden)` plus an explicit bias-copy
    scale — and for head_dim 64 the scale is 0.125, a power of two, so the
    fold is exact in any evaluation order.
  * **F16 needed no new code**: `read_safetensors()` has widened F16/BF16 to
    F32 at read time since 0076 (`add` met a F16 finetune first). The `get`
    lambda mirrors Python's `new.` prefix strip by lookup
    (`"new." + name` first) rather than by rebuilding the map.
  * **`rope_inv_freq` is COMPUTED, not transcribed** — the same float32
    arithmetic as `pack_gte`/numpy (`1.0f / powf(160000.0f, 2i/64.0f)` then
    `/ powf(8.0f, 2.0f/64.0f)`). Probed FIRST, before any wiring: MSVC's
    `powf` reproduces numpy's float32 power **bit-for-bit on all 32 values**
    (float32 bit patterns `3f6fe4b9 … 370ee449` identical both sides — the
    probe source and transcript are in the session scratchpad, the identity
    is re-proven by the container hash). That licenses computing them; the
    fallback the brief allowed (transcribe/round-from-double) was not needed.
  * **`py_double_repr()`** — the one place the JSON must FORMAT a float
    rather than copy source text verbatim (`cfg_raw` copies "1e-12",
    "20000", "8.0", "250048" as the gemma/nomic mirrors do). Both sides
    print the shortest round-trip decimal (CPython `repr`; MSVC
    `std::to_chars`, whose [charconv] minimality guarantee makes the digit
    sequences agree by construction), so only the dressing is reimplemented:
    CPython's fixed-vs-scientific threshold (fixed iff the decimal point
    lands in (−4, 16]), the trailing `.0` on fixed integers, none on a
    bare scientific mantissa, sign + ≥2 exponent digits. Verified by the
    whole-file hash, which contains all 32 formatted values.
  * **Tokenizer blob**: cached `models/<dir>/xlmr_tokenizer.bin` when
    present, else `generate_xlmr_tokenizer_table()` (0133's C++ generator)
    runs and writes the cache back — the exact `prepare_model_gemma` shape,
    closing the fresh-clone gap for arch 3.

### `runtime/src/main.cpp`
* One addition only: the `--prepare-model` dispatch gains the
  `model_type == "new"` branch beside nomic's (same
  checkpoint-decides-never-directory-name rule), `max_seq` 64 as above. The
  encode paths are untouched — which the nomic hash regression then verified
  rather than assumed.

## Exact commands

```powershell
# 0. pins verified against the local checkout BEFORE hardcoding (bash sha256sum)
cd models\gte-multilingual-base ; sha256sum model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json config.json
#   -> all five match the 0127/0138-brief values

# 1. the powf probe (scratchpad rope_probe.cpp): 32 float32 bit patterns vs
#    np.float32(container values).view(uint32) -- identical, 32/32

# 2. build
cd runtime
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL && cmake --build build --config Release'
#   -> [4/4] npuembed.exe (+ npuembeddings.exe copy); the two hub.cpp getenv
#      C4996 notes are the pre-existing HF_TOKEN reads, not new code

# 3. parity: C++ pack to scratch (the committed models\gte-multilingual-base.npue is never overwritten)
.\runtime\build\npuembed.exe . --prepare-model models\gte-multilingual-base <scratch>\gte_cpp.npue
Get-FileHash -Algorithm SHA256 models\gte-multilingual-base.npue, <scratch>\gte_cpp.npue
#   -> 4D59E2C4A5E6F2B946AF5DF51384EC5A20BAB9A74C290449B74DA06960407DD6  BOTH (whole file)
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\npue_data_hash.py models\gte-multilingual-base.npue <scratch>\gte_cpp.npue
#   -> ea6ef66ddb7efae8ac97b6acef44330a881ed1e5b5f0838a4b222d0138383492  both, exit 0

# 4. the C++ generator fallback: hide the cached bin, pack again
Move-Item models\gte-multilingual-base\xlmr_tokenizer.bin <scratch>\xlmr_tokenizer.bin.orig
.\runtime\build\npuembed.exe . --prepare-model models\gte-multilingual-base <scratch>\gte_cpp_gen.npue
#   -> "generated tokenizer.xlmr_table (no cached xlmr_tokenizer.bin found)"
#   -> regenerated bin sha256 15bc6dd63d6b3c54... == the moved-out original (0133's pin)
#   -> container again 4D59E2C4... whole-file identical

# 5. list (catalogue resolution)
.\runtime\build\npuembeddings.exe list --root .
#   -> gte-multilingual-base  ready  12  768  cls  1001 MB
#   -> stderr note for its row: "3 design sets serve hidden 768; using artifacts_nomic_bfp16"

# 6. embed via the catalogue (NO --artifacts) vs explicit vs 0136's stored output
cd runtime
.\build\npuembeddings.exe embed gte-multilingual-base ..\tasks\0136-gte-runtime\gte_in.txt <scratch>\gte_cat_out.f32 --root .. --threads 24
.\build\npuembeddings.exe embed gte-multilingual-base ..\tasks\0136-gte-runtime\gte_in.txt <scratch>\gte_art_out.f32 --root .. --artifacts artifacts_nomic_bfp16 --threads 24
#   status line both: "datapath   bfp16-emulated MMAC, C as bf16"; no unpinned warning
Get-FileHash <scratch>\gte_cat_out.f32, <scratch>\gte_art_out.f32, ..\tasks\0136-gte-runtime\gte_out.f32
#   -> AE25544C22F903FA6CA4AD3415A8A129F95D6262D4CDEC2748D6211423F3839C  ALL THREE

# 7. the C++-packed bytes encode (golden-by-bit-identity)
.\build\npuembed.exe .. --model <scratch>\gte_cpp.npue --artifacts artifacts_nomic_bfp16 --embed ..\tasks\0136-gte-runtime\gte_in.txt <scratch>\gte_cppnpue_out.f32 --threads 24
#   -> AE25544C... again -- the exact vectors 0136 validated against the
#      numpy oracle (1-cos 2.248e-04..3.683e-04), from the C++-packed container

# 8. gates, all seven, gte through the CATALOGUE (--out per 0136 problem 2 --
#    the canonical 0121/0132 reports are untouched)
& "...\iron\python.exe" tools\verify_semantics.py --models all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5 bge-large-en-v1.5 nomic-embed-text-v1.5 embeddinggemma-300m gte-multilingual-base --out tasks\0138-gte-hub-adoption\semantic_gate_seven.json
#   -> PASS x7; gte 30/30 gate, 15/15 probe, margin +0.087, datapath bfp16
& "...\iron\python.exe" tools\verify_tail.py --models <same seven> --out tasks\0138-gte-hub-adoption\tail_gate_seven.json
#   -> PASS x7; gte median 3.419e-04 / p99 5.160e-04 / max 7.361e-04 (0137's baseline, reproduced)

# 9. nomic bit-identity regression (0136's instrument)
.\build\npuembed.exe .. --model nomic-embed-text-v1.5 --artifacts artifacts_nomic_bfp16 --embed ..\tasks\0136-gte-runtime\nomic_in.txt <scratch>\nomic_0138.f32 --threads 24 --prefix search_document
#   -> EE1734216FE25C2F0BC2633974F29C20785B6646358276377B555A1D9438FD29 == 0136's hash
```

## The parity route, recorded (per the brief's fallback clause)

The route taken was **compute-and-format**, not transcribe: the 32
`rope_inv_freq` values are computed in C++ with the same float32 arithmetic
as numpy (MSVC `powf` bit-identical on all 32 — measured first, in a
standalone probe, before any packer code was written), and formatted with a
CPython-`repr`-compatible printer. The result meets the SAME bar as
`prepare_model_nomic` vs `pack_nomic`: **full-file byte identity**, JSON
included — the brief's "data region + semantically-equal JSON" concession was
not needed.

## Problems hit, honestly

1. **The first probe run burned 4 minutes on shell quoting.** `cmd /c '"…
   vcvars64.bat" >NUL && …'` through the bash tool mangles into an
   interactive `cmd` (banner, prompt, nothing runs, timeout); the same line
   through PowerShell works. All MSVC steps here went through PowerShell.
   `vcvars64.bat` also prints a (harmless, pre-existing) `'vswhere.exe' is
   not recognized` line on this machine; the builds succeed regardless.
2. **A hash split across two comment lines was transcribed wrong on the
   first edit** — `tokenizer_config.json`'s pin briefly read
   `…a326586546f946…` for `…a326586f946…` in the hub.cpp comment (a doc
   comment, not enforced code, but a wrong pin in a comment is exactly how a
   future session re-pins wrongly). Caught by re-deriving the split before
   building; fixed in the next edit. Lesson: never hand-wrap a hash — split
   it with a tool or keep it on one line.
3. **`verify_semantics.py`/`verify_tail.py` still default to the SIX-model
   BUILTIN list.** This task ran all seven explicitly with `--out` into this
   directory (0136 problem 2's precedent). Whether gte joins those tools'
   BUILTIN defaults — and whether the canonical 0121/0132 reports are
   re-cut as seven-model sweeps — is release scope, the coordinator's call;
   deliberately not edited here.

## Artifacts in this directory

`semantic_gate_seven.json`, `tail_gate_seven.json`. The scratch containers
(`gte_cpp.npue`, `gte_cpp_gen.npue`, both whole-file `4d59e2c4…`) were
deleted after verification — their identity to the committed
`models/gte-multilingual-base.npue` is the recorded evidence, and 2 GB of
byte-duplicates of a committed file is not an artifact. The probe
(`rope_probe.cpp` + transcript) and all `.f32` comparison outputs are in the
session scratchpad; every hash they produced is quoted above.

## What is NOT done here

The register/docs updates (T44/T52 closure wording, CURRENT_STATUS,
`tasks/README.md` index row — per the brief, not this task's files), MTEB
promotion of the multilingual STS baseline (0.6.0, per the user's decision),
int8 calibration for gte, and the seven-model BUILTIN question above.
