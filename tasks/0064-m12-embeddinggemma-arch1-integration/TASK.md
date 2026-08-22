# 0064 — EmbeddingGemma-300M: arch=1 integration (packer + CPU-only runtime + hub.cpp gating)

- **Date** 2026-08-20
- **Milestone** M12 (research plan Del C, C3/C4 slice — the runtime + packer
  integration the plan folded into "C3 Runtime + container")
- **Status** done for the packer (Python side) and the CPU-only runtime path,
  verified end to end; explicitly NOT done for the C++ packer mirror
  (`npue_pack.cpp`) and the HTTP `/v1/embeddings` endpoint — see "What was
  NOT done" below.

## Goal

Take EmbeddingGemma-300M from "three verified, standalone, unwired pieces"
(tasks/0055 numpy reference, 0061 tokenizer, 0063 host kernels) to an actual
`.npue` container plus a C++ runtime path that runs a real forward pass and
is checked against `reference/encoder_gemma.py` on real sentences — the
actual end-to-end correctness gate this project uses for every other model,
now for this one. Per the task brief and this project's own precedent for
BERT ("eltwise lives on the host"): since no Gemma-specific NPU kernel or
design exists yet, **every op runs on the CPU**, including every GEMM.

## Context

- `reference/encoder_gemma.py` (tasks/0055) is the ground-truth architecture
  reference, validated `1-cos` 1.065e-07 against real HuggingFace.
- `runtime/src/tokenizer_gemma.cpp`/`.hpp` (tasks/0061) — SentencePiece BPE,
  1,925/1,925 byte-identical to HuggingFace.
- `runtime/src/gemma_kernels.cpp`/`.hpp` (tasks/0063) — RMSNorm, RoPE, GeGLU,
  36/36 records PASS against real tapped intermediates.
- None of the three were wired into `main.cpp`, `hub.cpp`, or any `.npue`
  packer before this task. `research/OPEN-THREADS.md` T29 tracked this as
  open with the user's explicit go-ahead ("Vi kjører Gemma").
- The checkpoint on disk (`models/embeddinggemma-300m/`, from the ungated
  `unsloth/embeddinggemma-300m` mirror — `google/embeddinggemma-300m` is
  gated and `HF_TOKEN` was not set in this session either, same situation as
  0055) — verified directly against the real `model.safetensors` header
  (below) rather than assumed.

## Correction to the task brief, checked before writing any packer code

The brief said "EmbeddingGemma ships bf16 weights, not F32" and asked both
packers to handle BF16-safetensors reading as new work. **Read the actual
checkpoint's safetensors header directly** (`struct.unpack` on the file, not
`transformers`) rather than trusting either claim:

```
total tensors: 314
dtypes: {'F32'}
```

Every one of 314 tensors in `model.safetensors`, plus both Dense heads'
`linear.weight`, is **F32**, matching `config.json`'s own `"dtype":
"float32"` and reproducing tasks/0055's identical finding from a different
angle (there: reading via `safetensors_io.load`; here: reading the raw
header). **This checkpoint needs no BF16-safetensors-reading work at all** —
`reference/safetensors_io.py`'s `load()` already auto-widens BF16 to fp32
losslessly if a future checkpoint ships it, so the capability exists without
needing to be exercised here. Recorded so a future session does not
re-investigate this from scratch.

## What was done

### 1. Packer (`tools/pack_npue.py`) — arch=1, Python side, byte-verified

Added `ARCH_GEMMA3_MQA_ROPE_GEGLU = 1` to `tools/npue.py` and a new
`pack_gemma()` function in `tools/pack_npue.py`, dispatched from `main()` by
reading the checkpoint's own `config.json` (`model_type == "gemma3_text"`) —
**never** guessed from a directory name. The existing BERT `main()` body is
completely unreached for a Gemma checkpoint; nothing in the BERT path was
refactored or touched.

**Load-bearing design decision, made explicit rather than inherited from the
BERT packer**: GEMM operands are stored **PLAIN** — F32, row-major `[K,N]`,
no `block_panel` tiling, no `layout_hash`. The entire `tile_n`/L1-budget/
DMA-BD-limit machinery `docs/04-model/npue-format.md` describes exists to
serve the NPU's DMA engine, and there is no NPU kernel for this arch — every
GEMM runs on the host, so nothing here ever becomes a DMA descriptor. This
also means **the MQA-fused-QKV geometry problem tasks/0055 spent real effort
pricing (tile_n=16 at 8 columns, floored by the 256-wide K/V) never
arises**: it was a consequence of needing to satisfy `N % (tile_n·n_cols) ==
0` for an NPU B-operand layout, which a plain row-major host tensor has no
analogue of. Q/K/V are packed **unfused** (three separate `[hidden,*]`
matrices, matching the checkpoint's own tensor layout with only a transpose
applied) rather than concatenated — there was no performance reason to fuse
them once tiling is off the table, and keeping them separate is strictly
less code.

Tensor naming and shapes (all transposed from HF's `[out,in]` nn.Linear
convention to `[in,out]=[K,N]`, matching `y = x @ W` for a host GEMM; no
biases anywhere — `attention_bias: false`, both Dense heads `bias: false`,
confirmed from the checkpoint's own config, not assumed):

| tensor | shape `[K,N]` | role |
|---|---|---|
| `embed_tokens.weight` | `[262144,768]` | embedding (untouched, gathered not multiplied) |
| `norm.weight` | `[768]` | layernorm (final RMSNorm) |
| `layer.{i}.q_proj` | `[768,768]` | gemm_b_host |
| `layer.{i}.k_proj`, `.v_proj` | `[768,256]` | gemm_b_host (MQA: 1 KV head × head_dim 256) |
| `layer.{i}.o_proj` | `[768,768]` | gemm_b_host |
| `layer.{i}.q_norm.weight`, `.k_norm.weight` | `[256]` | layernorm (per-head RMSNorm) |
| `layer.{i}.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight` | `[768]` × 4 | layernorm |
| `layer.{i}.gate_proj`, `.up_proj` | `[768,1152]` | gemm_b_host (separate GeGLU matrices) |
| `layer.{i}.down_proj` | `[1152,768]` | gemm_b_host |
| `dense2.weight` | `[768,3072]` | gemm_b_host (post-pool Dense 1) |
| `dense3.weight` | `[3072,768]` | gemm_b_host (post-pool Dense 2) |
| `tokenizer.gemma_table` | raw bytes | tokenizer (0061's `gemma_tokenizer.bin`, embedded so the container is one file) |

317 tensors total, config carries `arch: "gemma3_mqa_rope_geglu"`,
`model_type`, `num_key_value_heads`, `head_dim`, `rms_norm_eps`,
`rope_theta`/`rope_local_base_freq`, `sliding_window_pattern`,
`query_pre_attn_scalar`, `dense_hidden` — everything the runtime needs to
reconstruct the architecture without a compiled-in constant.

**Packed**: `models/embeddinggemma-300m.npue`, 1,239.65 MB (dominated by F32
weights at no compression — a real cost of the "host-only, correctness
first" decision; a future bf16 host-weight path is flagged as future work,
not attempted here since the task's stated point was correctness).

**Round-trip verified** (`tools/verify_pack_gemma.py`, new): every one of
317 packed tensors compared **bit-exact** against the value read directly
out of the source `model.safetensors`/`2_Dense`/`3_Dense` (not a
re-derivation — the check reads the checkpoint files itself, independently
of whatever the packer computed) — **317 checks, 0 failures**. The tokenizer
table's raw bytes were also compared byte-for-byte against
`gemma_tokenizer.bin` on disk.

### 2. CPU-only runtime forward pass (`runtime/include/gemma_encode.hpp` + `runtime/src/gemma_encode.cpp`)

New, standalone (no XRT dependency — same discipline as tasks/0061/0063),
`GemmaEncoder` class implementing the full `reference/encoder_gemma.py`
forward pass: embed (×√768) → 24×[4 RMSNorm + MQA attention (Q/K/V proj →
q_norm/k_norm → RoPE per-layer-base → MQA dot-product attention, KVH=1 so
every head reuses the same K/V with no `repeat_kv` materialisation needed →
o_proj) → GeGLU FFN (gate/up proj → `gemma_kernels.cpp`'s `geglu_cpu` →
down_proj)] → final RMSNorm → masked mean pool (`include_prompt=true`) →
Dense(768→3072) → Dense(3072→768) → L2-normalize. One sequence at a time
(no batching) — deliberately: attention never mixes across sequences in a
correctly-masked batched implementation either (each row's mask only zeroes
its own padding), so batch=1 gives bit-for-bit the same per-text result a
batched version would, at the cost of not sharing host GEMM calls across
texts. Speed was explicitly not the goal this task (see brief).

**Precision decisions, each copied from `reference/encoder_gemma.py` rather
than re-derived** (documented in the file's own header): a new `gemm_f32()`
helper accumulates every GEMM in **double precision**, rounded to float32 on
output — not bit-identical to the reference's own `fp32_gemm` (opaque BLAS
accumulation), but strictly *more* accurate for any reduction length, so it
cannot be the source of any measured gap. Softmax runs in double precision
end to end (matching the reference's own upcast). The `q_norm`/`k_norm`
RMSNorm calls the same `rms_norm_cpu` from tasks/0063 that was already
verified bit-exact against real tapped intermediates. RoPE is applied via a
per-head extract-rotate-writeback (q's natural GEMM-output layout is
`[S,H,head_dim]`, s-major; `apply_rope_cpu`'s documented contract needs
h-major so position recovers as `row%seq_len` — `gemma_kernels.hpp`'s own
comment names this exact caller obligation).

**One assumption made explicit and guarded, not silently relied on**: the
attention loop assumes `num_key_value_heads == 1` (true for
EmbeddingGemma-300M) to avoid writing a general `repeat_kv` — `encode_one`
throws immediately if a future checkpoint has KVH > 1, rather than silently
computing the wrong thing.

**Standalone verification CLI** (`runtime/src/gemma_encode_cli.cpp`, built
by hand with `cl.exe`, same precedent as `tokenizer_gemma_cli.cpp` /
`gemma_kernels_test.cpp` — fast iteration, isolated from any concurrent NPU
session): `gemma_encode_cli.exe <model.npue> <texts.txt> <out.f32> [max_len]
[prefix]`.

**The actual correctness gate** (`tools/verify_gemma_cpu_encode.py`, new):
tokenizes the SAME real sentences with the SAME real HuggingFace tokenizer
tasks/0061 already proved byte-identical to `tokenizer_gemma.cpp`
(1,925/1,925 — so this is not a second thing to trust), runs
`reference/encoder_gemma.py`'s numpy oracle independently, and compares
`1-cos` against the C++ output. Not a device read-back and not a
self-comparison — the reference computation never touches anything the C++
side produced.

### 3. Wired into `main.cpp`'s production `Encoder::run()` path

An `arch=1` branch, resolved **early** in `main()` (right after
`--prepare-model` handling, before the `--artifacts` design-set resolution
that THROWS if no BERT NPU design is installed under `root` — a
precondition Gemma does not share, since it has no NPU design at all). The
model is peeked at (`resolve_model_path` + `npue::File::config_string
("arch")`) and, if it names the Gemma arch, control passes to a new
`run_gemma_mode()` function and never reaches the BERT `Encoder` construction
or any `npu::Design` object.

**This is a separate code path, not a branch inside `Encoder::run()`** —
`Encoder` holds `npu::Design&` references for seven NPU designs that a
Gemma container has no design to construct in the first place; forcing this
into that constructor would mean fabricating throwaway design objects rather
than genuinely sharing code. The BERT path (`Encoder::run()`,
`set_model_shape()`, the whole `--artifacts`/tier/pipeline machinery) is
**completely unmodified** — verified by re-running the existing MiniLM
golden-vector validation after every rebuild (below).

`run_gemma_mode()` supports `--embed <in.txt> [out.f32]` (`--max-len`,
`--prefix` optional) and a bare-invocation demo encode. **`--serve` FAILS
LOUDLY** with a clear "not implemented" error rather than silently running
the demo and exiting 0 — there is no `/v1/embeddings` HTTP endpoint wired up
for this arch (that code lives entirely inside the BERT-only section of
`main()`, unreached from this early branch), and doing nothing while
claiming success is exactly the fail-open shape this project's own history
warns about repeatedly.

`CMakeLists.txt` gained three new source files (`tokenizer_gemma.cpp`,
`gemma_kernels.cpp`, `gemma_encode.cpp`) — all XRT-free, added only so
`main.cpp` can call them.

### 4. `hub.cpp` — `HF_TOKEN` support and a `gated` field, infrastructure only

Added `bool gated` to `CatalogEntry` (`hub.hpp`) and threaded an optional
`bearer_token` through `download()` (sent as `Authorization: Bearer <token>`
via `WinHttpAddRequestHeaders`). `ensure_model()` now reads `HF_TOKEN` from
the environment for any `gated` catalogue entry and **fails closed** with a
clear two-step message (accept the licence, set `HF_TOKEN`) if it is unset —
it does **not** fall back to an ungated mirror in production code; that
remains a research-only shortcut (`reference/fetch_model_gemma.py`, tasks/
0055) the brief explicitly said not to ship.

**Deliberately NOT done**: no `embeddinggemma-300m` row was added to
`hub.cpp`'s `table()`. Two independent reasons, both real blockers rather
than time-saving: (1) no session, including this one, has ever held
`HF_TOKEN`, so there is no *verified* sha256 for the official
`google/embeddinggemma-300m` repo to pin — CLAUDE.md rule 6 ("a number
without a traceable artifact is not a result") applies to a checksum pin as
much as to a performance figure, and a fabricated one is worse than none.
(2) `ensure_model()`'s final step unconditionally calls the **BERT** C++
`prepare_model()` — with no C++ Gemma packer built this session (see below),
adding a catalogue row would make `npuembeddings serve embeddinggemma-300m`
on a clean machine download 1.2 GB, pass the (currently absent) checksum
check, and then **crash inside the wrong packer**. That is a worse failure
mode than the honest "not in the catalogue" `discover_models()` already
shows. The mechanism (gated flag, bearer token, fail-closed HF_TOKEN check)
is real, additive, does not touch the BERT catalogue, and is ready for
whichever future session both holds `HF_TOKEN` and builds the C++ Gemma
packer.

## Commands

```powershell
# 1. Pack (Python), then round-trip verify against the source checkpoint
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\pack_npue.py `
    --model-dir models\embeddinggemma-300m --out models\embeddinggemma-300m.npue
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_pack_gemma.py

# 2. Standalone CPU-encode CLI (fast iteration, no CMake, no XRT)
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cd runtime\build_gemma_encode
cmd /c "call `"$vcvars`" >nul && cl.exe /nologo /std:c++17 /EHsc /O2 /arch:AVX2 /I ..\include ..\src\npue.cpp ..\src\tokenizer_gemma.cpp ..\src\gemma_kernels.cpp ..\src\gemma_encode.cpp ..\src\gemma_encode_cli.cpp /Fe:gemma_encode_cli.exe"
.\gemma_encode_cli.exe ..\..\models\embeddinggemma-300m.npue texts_corpus.txt out_corpus.f32 64 document

# 3. The actual correctness gate: reference/encoder_gemma.py vs the C++ output
& "C:\Users\vegar\Documents\GitHub\NpuEmbeddings\.venv-ref\Scripts\python.exe" tools\verify_gemma_cpu_encode.py `
    --texts runtime\build_gemma_encode\texts_corpus.txt --cpu-out runtime\build_gemma_encode\out_corpus.f32

# 4. Full production rebuild (npuembed.exe), MSVC env via vcvars64.bat through PowerShell
#    (Bash tool's cmd /c nesting silently no-ops here -- tasks/0059's documented harness quirk)
cd runtime
cmd /c "call `"$vcvars`" && cmake --build build --config Release --target npuembed"

# 5. Production arch=1 path, and a regression check that the BERT path is untouched
.\build\npuembed.exe .. --model embeddinggemma-300m --embed build_gemma_encode\texts_corpus.txt build_gemma_encode\out_corpus_prod.f32
.\build\npuembed.exe .. --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --threads 4
```

## Result

**Packer round-trip: 317/317 tensors bit-exact** against the source
checkpoint (`tasks/0064-.../verify_pack_gemma.txt`).

**CPU encode correctness, two independent 4-sentence corpora, `document`
prefix, max_len 64** — the real end-to-end gate, `reference/encoder_gemma.py`
vs the C++ `GemmaEncoder` output, tokenized with the real HuggingFace
tokenizer (`tasks/0064-.../verify_gemma_cpu_encode_adhoc.txt`,
`..._corpus.txt`):

| corpus | worst `1-cos` | worst `rel_fro` |
|---|---:|---:|
| ad hoc (ASCII / accented / CJK / technical) | **4.969e-12** | 3.153e-06 |
| `reference/corpus_gemma.py` (this project's own established corpus) | **5.496e-13** | 1.048e-06 |

**This is dramatically tighter than every other model's gate in this
project** (MiniLM's own bf16 NPU path sits at `1-cos` ~1e-05; the numpy
oracle itself is 1.065e-07 against real HuggingFace) — expected, and not a
red flag: this path is host fp64-accumulated GEMMs with no bf16 rounding
anywhere, so the only error source left is float32 storage/RMSNorm/RoPE
rounding, all already independently verified at the float32-ULP floor in
tasks/0063. The gap is essentially machine precision.

**Production wiring confirmed working and bit-identical to the standalone
CLI**: `npuembed.exe .. --model embeddinggemma-300m --embed ...` and
`npuembeddings embed embeddinggemma-300m ...` (subcommand form) both produce
output files `cmp`-identical to `gemma_encode_cli.exe`'s. `--serve` on this
model refuses loudly with a clear message instead of silently doing nothing.

**BERT regression check, after every rebuild**: `all-MiniLM-L6-v2` on real
hardware (`--artifacts artifacts_b128il`) still reports `worst 1-cos vs
HuggingFace 1.086e-05`, **PASS** — identical to the value recorded in
tasks/0059, confirming the arch=1 addition changed nothing in the BERT path.

**Speed**: ~7.9 s/sentence at max_len=64 on one thread, no batching, no
AVX2 in the GEMM loop — explicitly not optimised (task's own priority
order). Not an NPU number and not claimed as one.

## Problems hit

- **The task brief's BF16-safetensors claim was wrong for this checkpoint** —
  see the correction section above. Caught by reading the actual safetensors
  header before writing any dtype-handling code, rather than trusting either
  the brief or memory of what "usually" ships.
- **The Bash tool's `cmd /c "vcvars64.bat && cmake --build ..."` silently
  no-ops** (prints only the `cmd.exe` banner, no build output, looks like a
  hang) — this is `tasks/0059`'s already-documented harness quirk, not a new
  finding. Fixed the same way: run the identical command through the
  **PowerShell** tool instead, which streams full Ninja output.
- **`UnicodeEncodeError` on Windows console for the CJK sentence** in
  `verify_gemma_cpu_encode.py` — the same `cp1252`-default trap every
  corpus-printing script in this repo already documents; fixed with
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.
- **`discover_models()`/`print_catalog()`'s "state" column is cosmetically
  wrong for arch=1 models**: it reports `embeddinggemma-300m` as "ready"
  because `pick_artifacts()` happens to find bge-base's NPU design (which
  also serves `hidden=768`) — harmless in effect (the Gemma path never reads
  `art`), but the label implies an NPU design is being used, which is false.
  Not fixed this session; a one-line note for whoever touches `print_catalog`
  next.
- No blockers otherwise. The checkpoint, `.venv-ref`, `ironenv`, and MSVC
  were all already in place from tonight's earlier tasks.

## What was NOT done (explicit, per the task's own scoping)

- **The C++ packer mirror (`runtime/src/npue_pack.cpp`) has no `arch=1`
  support**, and `tools/verify_pack_parity.py` was not extended to cover it.
  `--prepare-model` on a Gemma checkpoint through the C++ side would still
  run the BERT packing logic and fail (most likely a `std::map::at` /
  `KeyError`-shaped exception hunting for `encoder.layer.N...` tensor names
  that do not exist in a Gemma checkpoint) — loud, not silent, but not
  implemented. This is the single largest remaining piece: mirroring
  `pack_gemma()`'s ~150 lines of Python into C++, then getting
  `verify_pack_parity.py` to byte-match it, the same rigor the BERT path
  already has.
- **No `/v1/embeddings` HTTP endpoint** for arch=1 — `--serve` refuses
  loudly (see above) rather than working.
- **No `hub.cpp` catalogue entry** for `embeddinggemma-300m` — blocked on
  both `HF_TOKEN` (nobody has held it yet) and the C++ packer (above). The
  gated-fetch *mechanism* (bearer token, fail-closed check) is real and
  built, but has never been exercised against the actual gated endpoint —
  there is no way to test it without `HF_TOKEN`.
- **No batching, no AVX2, no threading** in the CPU GEMM path — correctness
  was the stated priority; ~7.9 s/sentence is unoptimised by design.
- **No MTEB run** on this arch's output — accuracy against HuggingFace is
  established (`1-cos` ~1e-12), but downstream embedding-quality regression
  (the metric this project actually gates release decisions on, per
  `tasks/0035`) has not been checked for Gemma at all yet.
- **`print_catalog`'s cosmetic "ready" mislabel** (above) — not fixed.

## Artifacts

- `tools/pack_npue.py` — `pack_gemma()` added, `main()` branches on
  `model_type` (committed).
- `tools/npue.py` — `ARCH_GEMMA3_MQA_ROPE_GEGLU = 1` added (committed).
- `tools/verify_pack_gemma.py` — round-trip verification, new (committed).
- `tools/verify_gemma_cpu_encode.py` — the actual correctness gate, new
  (committed).
- `runtime/include/gemma_encode.hpp`, `runtime/src/gemma_encode.cpp` — the
  CPU-only forward pass, new (committed).
- `runtime/src/gemma_encode_cli.cpp` — standalone verification CLI, new
  (committed).
- `runtime/src/main.cpp` — `#include "gemma_encode.hpp"`, `run_gemma_mode()`,
  and the early arch=1 dispatch in `main()` (modified; BERT path untouched).
- `runtime/CMakeLists.txt` — three new XRT-free source files added
  (modified).
- `runtime/include/hub.hpp`, `runtime/src/hub.cpp` — `gated` field,
  `bearer_token` param on `download()`, fail-closed `HF_TOKEN` check in
  `ensure_model()` (modified; no BERT catalogue entries changed).
- `models/embeddinggemma-300m.npue` — the packed container, 1,239.65 MB
  (gitignored, regenerate with the command above).
- `runtime/build_gemma_encode/` — standalone CLI + test corpora + output
  vectors (gitignored, new `.gitignore`-worthy directory — not added to
  `.gitignore` explicitly this session since `models/build_gemma_*`-shaped
  patterns already exist for the sibling directories; verify before next
  commit).
- `tasks/0064-m12-embeddinggemma-arch1-integration/verify_pack_gemma.txt`,
  `verify_gemma_cpu_encode_adhoc.txt`, `verify_gemma_cpu_encode_corpus.txt`,
  `texts_adhoc.txt`, `texts_corpus.txt` — verification output and the exact
  corpora used (committed).
- `research/OPEN-THREADS.md` — T29 updated with this task's result.
- `tasks/README.md` — index row added.

## Next

**If continuing toward a fully production Gemma path**: (1) the C++ packer
mirror + `verify_pack_parity.py` extension is the largest remaining
correctness-adjacent piece — until it exists, `--prepare-model` on a Gemma
checkpoint only works through Python; (2) once someone holds `HF_TOKEN` and
has accepted the licence at `google/embeddinggemma-300m`, fetch it, verify
its sha256 against the ungated mirror's weights (they should be identical
modulo repo metadata — worth checking, not assuming), and only then add the
`hub.cpp` catalogue row; (3) an MTEB run to establish whether this arch's
CPU-only embeddings are competitive, now that raw numerical correctness is
this tightly closed; (4) speed — batching and AVX2 in `gemm_f32`/the
attention loop are straightforward and unexplored, though `tasks/0055`'s
~62-72 seq/s *NPU* prior does not apply here at all (this path has no NPU
component whatsoever, so that estimate describes a different, hypothetical
future design, not this one).

**The state of the world after this task**: EmbeddingGemma-300M can be
packed from a real checkpoint (Python), loaded, tokenized, and run through a
full 24-layer forward pass entirely in the production C++ binary
(`npuembeddings embed embeddinggemma-300m <file> [out]`), producing
embeddings verified to `1-cos` ~1e-12 against the numpy reference oracle
which is itself verified to `1-cos` ~1e-07 against real HuggingFace. Nothing
touches the NPU. The BERT models are unaffected, confirmed by rebuild and
regression check.
