# tools/ — build-time tooling

Python used before anything touches the NPU: packing the model container,
exporting the IRON designs, and verifying both. Nothing here ships — Python is
build-time and prototyping only.

Runs in the **iron** environment, except the gates marked `.venv-ref` below.
Everything is invoked from the repository root.

## Start here

```sh
python tools/pipeline.py          # "I want to ___" -> the command for it
python tools/pipeline.py list     # every tool, and what it needs first
python tools/pipeline.py check    # what THIS machine is missing
```

`pipeline.py` is the way in and it is stdlib-only, so it answers even in an
interpreter that cannot import numpy. It runs no tool of its own: `run <tool>`
checks the requirements, prints the command, and execs the real script with
your arguments, so the tools keep owning their own flags.

| If you… | then run | where |
|---|---|---|
| want a model to **run** | `pack_npue` → `export_gemm_rtp` → build the runtime | `BUILD.md` §2.2–2.4 |
| changed a **weight or a fusion** | `pack_npue`, then `verify_npue`, then `verify_pack_parity` | `BUILD.md` §2.5 |
| changed a **kernel or the dataflow** | `export_gemm_rtp`, then `verify_design_numerics` (needs the NPU) | `BUILD.md` §2.3, §2.5 |
| want **int8 (W8A8)** instead of bf16 | `pack_npue --int8`, `export_gemm_rtp --int8`, then `verify_i8_scheme`, `verify_i8_kernels`, `verify_npue` | [int8](#int8-w8a8-containers) below |
| changed the **Whisper** path | `verify_whisper_tokenizer`, `verify_whisper_features`, `verify_whisper_model` | `BUILD.md` §2.5 |
| are **about to ship** | `python tools/pipeline.py gates --only release` | `BUILD.md` §2.5–2.6 |
| touched anything under **export/** | `parity_exporters` | [parity](#parity_exporterspy) below |

The one rule worth knowing: **`export_gemm_rtp.py --target X` and
`pack_npue.py --device Y` must be given the same NPU generation.** A container
packed for the other one is not refused by the layout hash — it was derived from
the same wrong constant on both sides — it is merely wrong everywhere, with
plausible numbers. `--arch 1` is npu1, `--arch 2` is npu2.

## Layout

Six buckets, one per question, and a file goes in the bucket its question
belongs to:

| bucket | the question it answers | entry points |
|---|---|---|
| `pack/` | where does the container come from? | `pack_npue.py`, `packers/` |
| `export/` | where do the designs come from? | `export_gemm_rtp.py`, `export_eltwise.py`, `export_validation.py`, `exporters/` |
| `verify/` | does it actually work? | every `verify_*.py`, `parity_exporters.py` |
| `gen/` | what generated code does a committed file need? | `gen_*.py`, `make_tail_reference.py` |
| `lib/` | shared modules — imported, not launched (three have a `__main__`; see below) | `npue.py`, `gemm_i8.py`, `gemm_pretiled.py`, `npu_ops.py`, … |
| `research/` | what was tried and did not ship? | `gemm_pretiled_research.py` |
| `data/` | input files, not tools | `npu_targets.json`, `semantic_corpus.json` |

Two conventions hold across all of them:

- **`lib/` is on `sys.path`, flat.** Every script does
  `sys.path.insert(0, ... "lib")` and then `import npue` — not
  `from lib import npue`. That is deliberate: `aie.iron.jit` reads kernel
  annotations at decoration time, and the exporters' history is full of
  flat-name imports that a package rename would have to touch one at a time.
  `lib/` documents that these are libraries; it is not a Python package.
- **Entry points are `bucket/name.py`; everything else is a module under it.**
  A new tool is a new `verify_foo.py`, not a new flag on an existing one.

## Inventory

### pack/ — the container

| File | Role | Invoked by |
|---|---|---|
| `pack/pack_npue.py` | HuggingFace checkpoint → pre-tiled, pre-fused `.npue`. `--int8` switches the four per-layer GEMM operands to int8. | `BUILD.md` §2.2; `verify_npue.py`, `verify_pack_parity.py`. |
| `pack/packers/whisper.py` | The arch=4 packer: a Whisper checkpoint, with a conv frontend, a positional-embedding table, a second (decoder) stack and a tokenizer table. Split out of `pack_npue.py` rather than appended to it. | Imported by `pack_npue.py`. |

### export/ — the designs

| File | Role | Invoked by |
|---|---|---|
| `export/export_gemm_rtp.py` | Entry point for the unified `gemm_rtp` design set (`final.xclbin` + instruction streams) into `runtime/artifacts_npu*`. A shim; the code is `export/exporters/gemm_rtp/`. | `BUILD.md` §2.3; fork `README.md`. |
| `export/export_eltwise.py` | Entry point for the `gelu` / `layernorm` / `softmax` design sets, `--extra-ops` picking which. A shim; the code is `export/exporters/eltwise/`. | `BUILD.md` §2.3; `export_gemm_rtp.py --npu-extra-ops CODES`. |
| `export/exporters/` | The exporter implementation, split per concern: `common/` holds what both exporters share (the `npu_targets.json` reader, JIT-cache identity, the output layout, argument validation), `gemm_rtp/` and `eltwise/` hold their own geometry, build and CLI. | Imported by the two entry points. |
| `export/export_validation.py` | Dumps the runtime's golden check vectors to `runtime/artifacts/validation/`. | `BUILD.md` §2.2. |

### verify/ — the gates

`pipeline.py gates` runs these in tiers; `BUILD.md` §2.5 is the canonical
commented list, and each row below says which tier it is in.

| File | Role | Invoked by |
|---|---|---|
| `verify/verify_npue.py` | `.npue` gate: spec conformance, bit-exact round-trip, stale-layout guard, goldens. Handles both weight schemes: bf16 against the bf16 of the fused source, int8 against `rint(W*asmooth/wscale)` plus the scale, saturation and dead-column invariants, and refuses a container whose `a_dtype` disagrees with its own panels. | `BUILD.md` §2.5. |
| `verify/verify_pack_parity.py` | The Python and C++ packers must agree byte for byte. | `BUILD.md` §2.5. |
| `verify/verify_i8_scheme.py` | The int8 scheme against itself, numpy only and no `reference/`: the round trip, the reader's dequantisation, and **seven injected packing faults** each of which the gate must name. A gate that has only ever seen a correct container is a gate whose sensitivity is unknown. | `BUILD.md` §2.5. |
| `verify/verify_i8_kernels.py` | The runtime's int8 host kernels against themselves: builds `runtime/tests/test_int8_host_kernels.cpp` three ways from one source (AVX2+FMA, scalar, scalar with the compiler's FMA contraction) and diffs the bytes. Pins that `quantise_a_int8` **is** bit-identical, that `dequantise_c` is **not** (1 ULP, attributed exactly to the FMA), and that the stream-load alignment precondition refuses exactly the three cases that would otherwise be a SIGSEGV on AVX2 hosts only. Needs `g++`; no NPU, no XRT, no checkpoint. | `BUILD.md` §2.5. |
| `verify/verify_design_numerics.py` | Feeds random matrices through an exported design set's own instruction streams and compares C against numpy; `--npue`/`--tensor` stages real container operands verbatim. The only check that a design *computes* what it claims to, and the only one that spans design, file and hardware at once; needs an NPU. | Run by hand after any `export_gemm_rtp.py` change, and per model when its design set is first built. |
| `verify/verify_whisper_tokenizer.py` | Three-way gate on the Whisper tokenizer: C++, the reference, and HuggingFace, over an adversarial corpus (Cyrillic, CJK, emoji, CRLF, long runs, contractions). Exact ids, no tolerance: a transcript is text. | Run by hand; `BUILD.md` §2.5. |
| `verify/verify_whisper_features.py` | Holds the C++ audio front end against `transformers`: the WAV reader and the ffmpeg path against Python's `wave`, the log-mel spectrogram against `WhisperFeatureExtractor`, the slaney filter bank on its own, and the two convolutions against torch with the container's own weights. The corpus includes a clip shorter than 30 s and one longer, because the padding and the cut are the whole story. | Run by hand; `BUILD.md` §2.5. |
| `verify/verify_whisper_model.py` | Holds the two C++ NPU stacks against `transformers`: the encoder's output hidden states, then teacher-forced decoder steps one position at a time (state after the final LayerNorm, logits, and top-1), then the greedy argmax chain and the text those ids decode to. Also 16 workers vs 1 byte-for-byte, and two refusals. Gated on `1-cos`, not `max|d|`: the designs' bf16 operands put the encoder 1.7e-04 from `transformers` in cosine and 12 in absolute value on 15 elements of 576000, and a float32 replay of the same stack prices that gap exactly. | Run by hand; `BUILD.md` §2.5. Needs the NPU, the container, the design set and the checkpoint. |
| `verify/verify_whisper_cli.py` | Holds the `transcribe` CLI and `POST /v1/audio/transcriptions` against transformers: the long-form window schedule (30 s windows, 5 s stride each side, so windows start 20 s apart), each window's own text against `generate()`, the merged text against transformers' own `_find_longest_common_sequence` (imported, not reimplemented), and ten refusals. Starts the server itself. | Run by hand; `BUILD.md` §2.5. Needs a built runtime. |
| `verify/verify_whisper.py` | Word error rate of the C++ transcription path against a **human** transcript, with `transformers`' WER on the same audio printed beside it — the only Whisper gate with an opinion of its own; the other two only compare against another program. No audio is committed: the corpus is an argument (`--audio` + `--ref`, or `--corpus DIR`). Normalises to lowercase words with punctuation dropped, and reports substitutions / deletions / insertions separately, because a WER of 0.12 means something different when it is 12 wrong words than 12 missing ones. | Run by hand with a corpus; `BUILD.md` §2.5. Needs a built runtime. |
| `verify/verify_semantics.py` | Near/far ordering gate — no reference, no tolerance. Stdlib only, so it runs against a cold release. | Run by hand / release gate. |
| `verify/verify_tail.py` | Per-model, per-datapath p99 tail gate. Stdlib only. | Run by hand / release gate. |
| `verify/verify_endpoint.py` | Drives `--serve` with the official OpenAI client and checks the numbers. Needs `.venv-ref`. | `BUILD.md` §2.5. |
| `verify/verify_npue_nomic.py` | arch=2 gate for the nomic `.npue`. | Run by hand. |
| `verify/verify_vit.py` | arch=5 gate for `vit-base-patch16-224.npue`, both weight schemes. bf16: the pool + attention + pre-LN stack and the host head, on `1-cos` and top-1. int8: the same plus all four scheme invariants on every I8 panel and top-1 agreement with fp32. Holds `im2col_patches(px) @ patch_embed_operand(W)` against HF's own `Conv2d` — the one place in the tree that knows the pixel half of the patch layout and the weight half at the same time, so a drift between them is a float diff and not a confident label. | Run by hand after any `packers/vit.py` or `vit_int8.py` change. Needs the checkpoint and a container; no NPU. |
| `verify/verify_vit_image.py` | The C++ image front end against PIL — the library `transformers`' own image processor calls — on the same bytes. The decode is **exact** on every format both sides read (grayscale, gray+alpha, palette, JPEG), alpha is composited on white rather than dropped as `convert("RGB")` drops it, and the resize is measured **per resample code** rather than for the one the checkpoint happens to ask for. That is what turned "six codes" into four: the first version spelled LANCZOS, BILINEAR, BICUBIC, BOX and HAMMING as one Keys cubic, correct for BICUBIC alone and off by 65 levels for the rest — and BICUBIC passing throughout is exactly why the rest went unnoticed. NEAREST, HAMMING, a 16-bit PNG, a fake PNG, a truncated PNG and a text file are all cases here, each refused by name with a message that names what *is* implemented. | `BUILD.md` §2.5. Needs `g++` with libpng and libjpeg headers and a packed container; no NPU. |
| `verify/verify_vit_model.py` | The **host-only** half of arch=5: `read_geometry`'s contract, the classifier head, and the int8 host kernels. Builds `runtime/tests/test_vit_model.cpp` twice from one source (AVX2 and scalar) because the two disagree about `dequantise_c`'s stream-load precondition. The head case is the one worth the gate: `classifier.weight` is `[d_model, num_labels]` row-major, the checkpoint holds it transposed, and reading it the checkpoint's way yields a finite, plausible, entirely wrong 1000 logits whose argmax is another label — with every shape, byte count and layout hash agreeing either way. The probe therefore prints **both**, and the gate requires them to differ, so the check is known to be able to fail. `dequantise_c` is held to `eps32 × (sum of term magnitudes)`, not to a relative error on the result, because the products cancel. | `BUILD.md` §2.5. Needs `g++`, numpy and a packed container; no NPU, no XRT, no design set, no checkpoint. Pass the `--int8` container for the SmoothQuant fold to be exercised at all. |
| `verify/parity_exporters.py` | Diffs the split `exporters/` against the monoliths in git: same stdout, stderr and exit code on 17 documented invocations, plus 26 flag-presence checks. The split is committed now, so the reference side is resolved FROM HISTORY per tool — the newest revision at which that tool's entry point was still a monolith rather than a shim onto `exporters/`, which is not the same question as "when did the package appear" (one commit in this history is a shim whose package was never committed). It prints the revision it used. `--rev` overrides both. | Run by hand after touching anything under `export/exporters/`. |

### gen/ — generators for committed files

These write a file that is **committed**, so they are run when that file's
inputs change, not as part of a build.

| File | Role | Invoked by |
|---|---|---|
| `gen/gen_tokenizer_tables.py` | Emits `runtime/include/tokenizers/bert_unicode_tables.hpp` from Python's `unicodedata`. | `BUILD.md` §2.2. |
| `gen/gen_xlmr_unicode_tables.py` | Emits `runtime/include/tokenizers/xlmr_unicode_tables.hpp` the same way. | Run by hand when the XLM-R tokenizer changes. |
| `gen/gen_whisper_unicode_tables.py` | Emits `runtime/include/tokenizers/whisper_unicode_tables.hpp`: `\p{L}`, `\p{N}`, `\s` as range tables. `std::regex` cannot run Whisper's pre-tokeniser pattern, so the categories are computed where Python's `unicodedata` *is* the definition. | Run by hand when Python's Unicode data moves. |
| `gen/make_tail_reference.py` | Generates the fp32 reference JSON `verify_tail.py` gates against. Needs torch. | Run by hand in `.venv-ref`; `verify_tail.py` points at its output. |

### lib/ — shared modules

Imported by name, not launched — with three exceptions that are also runnable
in their own right and are marked below: `*_tokenizer_ref.py` (an executable
specification is a program as well as a reference) and `whisper_int8.py` (whose
`__main__` is a smoke test).

| File | Role | Imported by |
|---|---|---|
| `lib/npue.py` | The `.npue` container: header, JSON directory, tiling, reader, writer. Reference the C++ loader must match. Also the `I8` dtype's meaning: `Reader.tensor()` on an int8 operand returns `Wq*wscale*asmooth`, the weight the array effectively multiplies, because the raw panel is not a weight. | `pack_npue.py`, `export_gemm_rtp.py`, `gemm_pretiled_research.py` and the `verify_*` gates. |
| `lib/gemm_i8.py` | The int8 scheme, shared by both packers: per-output-channel symmetric weights, per-row activations, an exact int32 accumulator, SmoothQuant folded into the weights rather than into LayerNorm (BERT is post-LN). `check_i8_operand` is its gate — four invariants, each with its own message. | `pack_npue.py`, `packers/whisper.py`, `verify_npue.py`, `verify_i8_scheme.py`. |
| `lib/gemm_pretiled.py` | The production GEMM design library (`pretiled_array()`). | `export_gemm_rtp.py`; research driver in `gemm_pretiled_research.py`. |
| `lib/npu_ops.py` | The op vocabulary shared with the runtime: code (`gelu`, `layn`, `softm`), the design directory each code means, and the flag names (`--npu-extra-ops` on both sides — it builds the design at export time and selects it at run time; `--extra-ops` is `export_eltwise`'s own spelling). Written twice on purpose — once here, once in `runtime/include/common/npu_ops_flag.hpp` — because the exporter is Python and the runtime's parser is C++; each file points at the other, and a typo is a refusal by name on both sides. | Imported by both exporters; change it in both places at once. |
| `lib/safetensors_mmap.py` | Memory-maps a safetensors checkpoint, so packing reads tensors without loading the file. | `pack_npue.py`, `packers/whisper.py`. |
| `lib/toolchain_provenance.py` | Writes the `toolchain.json` sidecar next to each `design.json`. | `export_gemm_rtp.py`. |
| `lib/whisper_bpe.py` | The Whisper tokenizer's binary table: the single definition of its layout and of GPT-2's byte-level alphabet. | `packers/whisper.py`, `whisper_tokenizer_ref.py`. |
| `lib/whisper_int8.py` | SmoothQuant calibration for a Whisper container's activations — the same estimator `pack_npue.py` runs over text, run over audio instead. Its `__main__` is a smoke test, not a gate. | `packers/whisper.py`. |
| `lib/whisper_tokenizer_ref.py` | Pure-Python executable spec for the C++ Whisper tokenizer, over the packed table blob. Runnable. | `verify_whisper_tokenizer.py`. |
| `lib/xlmr_tokenizer_ref.py` | Pure-Python executable spec for the C++ XLM-R tokenizer. Runnable; imported by nothing. | Reference for the C++ port; run by hand. |

### research/ — tried, measured, not shipped

| File | Role | Invoked by |
|---|---|---|
| `research/gemm_pretiled_research.py` | Research driver for `gemm_pretiled.py`: presets, traces, wall-clock benchmarks, `--arch`/`--dev` selection. Writes its traces to `research/artifacts/`. | Run by hand. |

### data/ — inputs, not tools

| File | Role | Read by |
|---|---|---|
| `data/npu_targets.json` | Model geometry and per-arch defaults: the exporters' source of truth for every `--target`. Closed key sets, so a typo in a geometry field fails here rather than exporting a design for a shape nobody asked for. | `export/exporters/common/targets.py`; the runtime's own error messages name it by path. |
| `data/semantic_corpus.json` | The shared near/far corpus. | `verify_semantics.py` and `make_tail_reference.py`. |

## Run

The three commands that cover most of it:

```sh
python tools/pack/pack_npue.py             # -> models/all-MiniLM-L6-v2.npue
python tools/verify/verify_npue.py         # bit-exact round trip, layout guard, goldens
python tools/export/export_validation.py   # golden check vectors for the runtime
```

Retuning the tiling repacks instead of editing the loader:

```sh
python tools/pack/pack_npue.py --tile-n 32 --out models/minilm_n32.npue
```

The `.npue` is gitignored — a deterministic derivative of the sha256-pinned
checkpoint, which travels inside the file as `source_sha256`.

## Why the weights are transformed offline

The runtime must not transpose, convert dtypes, concatenate or re-tile — all are
pure functions of the weights. What is left at load time is `mmap` and pointer
arithmetic. The layout descriptor is **data, not code**: to try different tile
dimensions, repack rather than editing a loader. `layout_hash` makes a stale file
fail loudly instead of producing plausible garbage embeddings.

## int8 (W8A8) containers

An int8 container is **not** a smaller bf16 container; it is a different MMAC
datapath, and a container and a design set have to agree about which one they
are. `layout_hash` covers the operand dtype, so a mismatched pair is refused
rather than run — but the export has to be redone, and the design set lands in
the **same directory** as the bf16 one (`export/exporters/common/paths.py`), so
exporting int8 overwrites the bf16 set. Keep two checkouts or two `--out` roots
if you want both.

```sh
# 1. the container (needs a calibration corpus and the numpy oracle in reference/)
python tools/pack/pack_npue.py --int8 --out models/all-MiniLM-L6-v2.i8.npue

# 2. the design set, built for THAT container's operand width
python tools/export/export_gemm_rtp.py --target all-MiniLM-L6-v2 --arch 1 --int8 --out runtime

# 3. the gates
python tools/verify/verify_i8_scheme.py   # the scheme and the gate that checks it
python tools/verify/verify_i8_kernels.py  # the runtime's int8 kernels, AVX2 vs scalar
python tools/verify/verify_npue.py         # spec, round trip, layout, goldens
```

**What int8 covers: the four per-layer GEMM operands** (`qkv`, `attn_out`,
`ffn_up`, `ffn_down`) and nothing else. The embedding tables stay fp32,
attention's `QK^T`/`A·V` run on the host in fp32, the elementwise design sets
stay bf16 kernels, and Whisper's convolutions are built from fp32 weights on
purpose.

**What it buys: container size, and only that.** The layer operands halve
exactly; the scales cost 1.5% of them; the embedding tables do not move. It
buys no throughput, because the bottleneck is the ~150 µs per-dispatch cost and
the dispatch count does not depend on operand width. The "5.5–7.7×" in
`pack_npue.py --help` is the MMAC datapath measured against bf16's
(`tasks/0077`), not an end-to-end figure.

**What it costs: accuracy, and the gate for it is not the bf16 one.** The
common 2e-3 end-to-end gate is not reachable for int8 — `bge-large-en-v1.5`
measures 2.968e-3 at MTEB −0.05. `verify_npue.py` therefore gates an int8
container on `INT8_WEIGHT_ONLY_LIMIT`, and prints what that number is: the
**weight half** of the quantisation only, because the reference encoder does not
model the per-row activation quantisation the array also does. For the full
int8 error, run the design's own instruction streams:
`verify_design_numerics.py --npue`, which needs an NPU.

## Design export

`export_gemm_rtp.py` builds each (shape × batch tier) design, then refusing
unless every design shares one static configuration (they must differ only in
UUID metadata). The commands for this fork's two served models are in
[`README.md`](../README.md)'s `Operations:` block.

The two exporter entry points are thin. `export/exporters/common/` is the shared
half —
`targets.py` reads `npu_targets.json`, `cache.py` decides which cached entry is
this design and which are stale, `paths.py` owns the one output convention, and
`validate.py` holds the checks that do not depend on which exporter runs. Neither
`export/exporters/` package may use `from __future__ import annotations`:
`aie.iron.jit` reads kernel annotations at decoration time to tell
`CompileTime[T]` from a runtime scalar, and PEP 563 stringified annotations make
it reject a defaulted parameter with a `TypeError` raised from inside the
decorator.

Spec: [`docs/04-model/npue-format.md`](../docs/04-model/npue-format.md).

## The B operand's byte order depends on the generation

`layout_hash` is what makes a stale container fail loudly, and it covers the
`mac_s`/`mac_t` of the B panel. Those two numbers are **not** the same on both
boards. Measured, not assumed — `aie.iron.kernels.mm(...).mac_dims` returns
`(r, s, t)` and the B tile interior is stored in `(s, t)` order:

| device | `mac_dims` | B panel order |
|---|---|---|
| `npu2` (aie2p) | `(8, 8, 8)` | `(s=8, t=8)` |
| `npu1` (aie2) | `(4, 8, 4)` | `(s=8, t=4)` |

`npue.gemm_b_layout`'s `(8, 8)` default is npu2's, and the packers used to pass
that default straight through, so a container packed for `npu2` was read by an
`npu1` design with the columns of every 64×32 panel permuted. Nothing detected
it: the bytes are the right size, the shapes agree, both sides computed the
same `layout_hash` from the same wrong constant, and every number that came out
was plausible. It is per-device now — `pack_npue.py --device`, and the exporter
takes the pair from `--arch` — so the hash compares like with like and a
cross-generation container is refused instead of misread.

`verify_design_numerics.py` is what saw it, in one dispatch per stream, before
any model was involved. It prints the mismatch by name:

```
LAYOUT MISMATCH: b_layout records mac (s=8, t=8); device npu1 consumes
(s=8, t=4). A container packed as recorded is misread by this design.
```
