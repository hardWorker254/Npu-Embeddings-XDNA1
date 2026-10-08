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
| want **int4 (W4A8)** — half the weight bytes, still on the int8 array | `pack_npue --dtype i4 --int4-group 32`, then `verify_i4_scheme`, `verify_npue`; run it with an **int8** design set, no re-export | [int4](#int4-w4a8-containers) below |
| want the **operand dtype stated** | `pack_npue --dtype {f32,bf16,i8,i4}` — `bf16` is the default, `i8` is `--int8` spelled the newer way, `i4` is the four per-layer GEMM operands stored as packed 4-bit weights over the same int8 datapath (it takes `--int4-group`), `f32` is the CPU control container (arch=1 only). fp16 is refused by name: the array's datapath is bf16 or int8, so a container claiming one could not be run. | `pack_npue --help` |
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
| `pack/pack_npue.py` | HuggingFace checkpoint → pre-tiled, pre-fused `.npue`. `--dtype {f32,bf16,i8,i4}` names the operand dtype: `bf16` (default) the pre-tiled panel the array runs, `i8` the four per-layer GEMM operands as per-channel int8 with SmoothQuant, `i4` the same four operands stored as packed 4-bit weights with per-group scales (takes `--int4-group`, default 32; `0` is one group over the whole K), `f32` plain row-major operands for the CPU control container (arch=1 only). `--int8` and `--gemma-host-only` are `i8` and `f32` spelled the older way; two spellings that disagree are refused rather than ranked. | `BUILD.md` §2.2; `verify_npue.py`, `verify_pack_parity.py`. |
| `pack/packers/whisper.py` | The arch=4 packer: a Whisper checkpoint, with a conv frontend, a positional-embedding table, a second (decoder) stack and a tokenizer table. Split out of `pack_npue.py` rather than appended to it. | Imported by `pack_npue.py`. |

### export/ — the designs

| File | Role | Invoked by |
|---|---|---|
| `export/export_gemm_rtp.py` | Entry point for the unified `gemm_rtp` design set (`final.xclbin` + instruction streams) into `runtime/artifacts_npu*`. A shim; the code is `export/exporters/gemm_rtp/`. | `BUILD.md` §2.3; fork `README.md`. |
| `export/export_eltwise.py` | Entry point for the `gelu` / `layernorm` / `softmax` design sets, `--extra-ops` picking which. A shim; the code is `export/exporters/eltwise/`. | `BUILD.md` §2.3. The GEMM exporter builds these too, from the registry — see `NPU_OPS.md`. |
| `export/exporters/` | The exporter implementation, split per concern: `common/` holds what both exporters share (the `npu_targets.json` reader, JIT-cache identity, the output layout, argument validation), `gemm_rtp/` and `eltwise/` hold their own geometry, build and CLI. | Imported by the two entry points. |
| `export/export_validation.py` | Dumps the runtime's golden check vectors to `runtime/artifacts/validation/`. | `BUILD.md` §2.2. |

### verify/ — the gates

`pipeline.py gates` runs these in tiers; `BUILD.md` §2.5 is the canonical
commented list, and each row below says which tier it is in.

| File | Role | Invoked by |
|---|---|---|
| `verify/verify_npue.py` | `.npue` gate: spec conformance, bit-exact round-trip, stale-layout guard, goldens. Handles all three weight schemes: bf16 against the bf16 of the fused source, int8 against `rint(W*asmooth/wscale)` plus the scale, saturation and dead-column invariants, int4 against the same int8 scale rule plus the group and payload rules, each with its own tail limit — and refuses a container whose `a_dtype` disagrees with its own panels. | `BUILD.md` §2.5. |
| `verify/verify_pack_parity.py` | The Python and C++ packers must agree byte for byte. | `BUILD.md` §2.5. |
| `verify/verify_i8_scheme.py` | The int8 scheme against itself, numpy only and no `reference/`: the round trip, the reader's dequantisation, and **seven injected packing faults** each of which the gate must name. A gate that has only ever seen a correct container is a gate whose sensitivity is unknown. | `BUILD.md` §2.5. |
| `verify/verify_i8_kernels.py` | The runtime's int8 host kernels against themselves: builds `runtime/tests/test_int8_host_kernels.cpp` three ways from one source (AVX2+FMA, scalar, scalar with the compiler's FMA contraction) and diffs the bytes. Pins that `quantise_a_int8` **is** bit-identical, that `dequantise_c` is **not** (1 ULP, attributed exactly to the FMA), and that the stream-load alignment precondition refuses exactly the three cases that would otherwise be a SIGSEGV on AVX2 hosts only. Needs `g++`; no NPU, no XRT, no checkpoint. | `BUILD.md` §2.5. |
| `verify/verify_i4_scheme.py` | The int4 scheme against itself, **and against C++**: builds `verify/int4_panel_probe.cpp` from `runtime/src/model.cpp` and requires `common/int4_panel.hpp`'s decode to equal `npue.fold_i4()` **byte for byte**, over group sizes 32/16/64/0/256 (0 = per-channel, 256 = larger than K) plus a bf16 operand for the non-decode path, then runs five C++ refusals. Sections 1–6 are the `verify_i8_scheme` question applied to the new scheme: the honest round trip, every group size, a zero padding column, an unsmoothed panel, **ten injected faults** each of which the gate must name, and the refusals. The point of the byte comparison is that int4 is the first format where the runtime does arithmetic the writer also did — a wrong nibble index, a wrong group row, a wrong sign extension or a tie broken the other way all still produce a right-sized, right-hashed panel of plausible numbers that no tolerance and no self-consistency check sees. Needs numpy and `g++`; no NPU, no checkpoint, no `reference/`. | `BUILD.md` §2.5. |
| `verify/verify_design_numerics.py` | Feeds random matrices through an exported design set's own instruction streams and compares C against numpy; `--npue`/`--tensor` stages real container operands verbatim. The only check that a design *computes* what it claims to, and the only one that spans design, file and hardware at once; needs an NPU. | Run by hand after any `export_gemm_rtp.py` change, and per model when its design set is first built. |
| `verify/verify_npu_op_matrix.py` | The op registry gate, 9 codes × 7 architectures = 63 cells: the registry's own shape (five statuses, one **pinned tally**, `STREAM_ONLY` exactly the six codes with no directory of their own), all four **generated** documents regenerated from the sources and compared to the files on disk, all 162 model cells against their kind's row (the two gated-FFN cells excepted, by name), the runtime's own `npu_op_table()` against the Python one, and then **all 63 cells run against the binary** — a cell that says `honours` and does not dispatch fails it. Fixture rows that have no container print why and skip rather than quietly counting as passes. | `pipeline.py gates` (npu tier); `BUILD.md` §2.5. |
| `verify/verify_whisper_tokenizer.py` | Three-way gate on the Whisper tokenizer: C++, the reference, and HuggingFace, over an adversarial corpus (Cyrillic, CJK, emoji, CRLF, long runs, contractions). Exact ids, no tolerance: a transcript is text. | Run by hand; `BUILD.md` §2.5. |
| `verify/verify_whisper_features.py` | Holds the C++ audio front end against `transformers`: the WAV reader and the ffmpeg path against Python's `wave`, the log-mel spectrogram against `WhisperFeatureExtractor`, the slaney filter bank on its own, and the two convolutions against torch with the container's own weights. The corpus includes a clip shorter than 30 s and one longer, because the padding and the cut are the whole story. | Run by hand; `BUILD.md` §2.5. |
| `verify/verify_whisper_model.py` | Holds the two C++ NPU stacks against `transformers`: the encoder's output hidden states, then teacher-forced decoder steps one position at a time (state after the final LayerNorm, logits, and top-1), then the greedy argmax chain and the text those ids decode to. Also 16 workers vs 1 byte-for-byte, and two refusals. Gated on `1-cos`, not `max|d|`: the designs' bf16 operands put the encoder 1.7e-04 from `transformers` in cosine and 12 in absolute value on 15 elements of 576000, and a float32 replay of the same stack prices that gap exactly. | Run by hand; `BUILD.md` §2.5. Needs the NPU, the container, the design set and the checkpoint. |
| `verify/verify_whisper_cli.py` | Holds the `transcribe` CLI and `POST /v1/audio/transcriptions` against transformers: the long-form window schedule (30 s windows, 5 s stride each side, so windows start 20 s apart), each window's own text against `generate()`, the merged text against transformers' own `_find_longest_common_sequence` (imported, not reimplemented), and ten refusals. Starts the server itself. | Run by hand; `BUILD.md` §2.5. Needs a built runtime. |
| `verify/verify_whisper.py` | Word error rate of the C++ transcription path against a **human** transcript, with `transformers`' WER on the same audio printed beside it — the only Whisper gate with an opinion of its own; the other two only compare against another program. No audio is committed: the corpus is an argument (`--audio` + `--ref`, or `--corpus DIR`). Normalises to lowercase words with punctuation dropped, and reports substitutions / deletions / insertions separately, because a WER of 0.12 means something different when it is 12 wrong words than 12 missing ones. | Run by hand with a corpus; `BUILD.md` §2.5. Needs a built runtime. |
| `verify/verify_semantics.py` | Near/far ordering gate — no reference, no tolerance. Stdlib only, so it runs against a cold release. | Run by hand / release gate. |
| `verify/verify_tail.py` | Per-model, per-datapath p99 tail gate. Stdlib only. | Run by hand / release gate. |
| `verify/verify_endpoint.py` | Drives `--serve` with the official OpenAI client and checks the numbers. Needs `.venv-ref`. | `BUILD.md` §2.5. |
| `verify/verify_serve_dispatch.py` | The **dispatch** gate for `serve`: one verb, four endpoints, and the container's arch picks which. Starts each of the four servers itself, requires `/health` to name the right `kind` (a client picks its parser from it), POSTs to that architecture's own path and requires the expected keys, asks for the other three and requires a 404 that names where to go, and for the two image endpoints runs the refusal list plus a check that a per-request threshold does not survive into the next request. It also compares the endpoint's answer against the CLI's, since both call one emitter. Separate from `verify_endpoint.py` because that gate is the embedding endpoint with a real client, and three of the four are not that. Stdlib only. | `BUILD.md` §2.5. Needs a built runtime and the containers named at the top of the file; skips (loudly) an architecture with no container on disk. |
| `verify/verify_npue_nomic.py` | arch=2 gate for the nomic `.npue`. | Run by hand. |
| `verify/verify_vit.py` | arch=5 gate for `vit-base-patch16-224.npue`, both weight schemes. bf16: the pool + attention + pre-LN stack and the host head, on `1-cos` and top-1. int8: the same plus all four scheme invariants on every I8 panel and top-1 agreement with fp32. Holds `im2col_patches(px) @ patch_embed_operand(W)` against HF's own `Conv2d` — the one place in the tree that knows the pixel half of the patch layout and the weight half at the same time, so a drift between them is a float diff and not a confident label. **Section 4 runs the C++ binary** and holds its answer against transformers, asserting the *confidence* and not only the top-1: the other three sections are numpy and were all green while the shipped runtime divided each activation by `asmooth` where `run_i8` documents `1/asmooth`, which is `X @ W * asmooth²` and a confidently wrong label. It finds the design set by the container's `source_repo` and its own operand dtype, so a hand-named container still runs. | Run by hand after any `packers/vit.py`, `vit_int8.py` **or `NpuGemm`/ViT dispatch** change. Needs the checkpoint and a container; section 4 additionally needs the built binary. `transformers` **4.44.2** — this export and `vit_int8.py` both name tensors after the pre-5.x ViT module tree. No NPU. |
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
| `lib/npue.py` | The `.npue` container: header, JSON directory, tiling, reader, writer. Reference the C++ loader must match. Also the `I8` dtype's meaning: `Reader.tensor()` on an int8 operand returns `Wq*wscale*asmooth`, the weight the array effectively multiplies, because the raw panel is not a weight. And `I4`: `pack_i4`/`unpack_i4` (low nibble first, values outside [-8, 7] refused rather than truncated), `fold_i4` (the group factor back in, `rint` = round-half-to-even), `GSCALE_SUFFIX`, and the accessor split — `raw()` **refuses** an I4 payload, `panel()` is the int8 panel the array multiplies, `tensor()` is the weight. | `pack_npue.py`, `export_gemm_rtp.py`, `gemm_pretiled_research.py` and the `verify_*` gates. |
| `lib/gemm_i8.py` | The int8 scheme, shared by both packers: per-output-channel symmetric weights, per-row activations, an exact int32 accumulator, SmoothQuant folded into the weights rather than into LayerNorm (BERT is post-LN). `check_i8_operand` is its gate — four invariants, each with its own message. | `pack_npue.py`, `packers/whisper.py`, `verify_npue.py`, `verify_i8_scheme.py`. |
| `lib/gemm_i4.py` | The int4 scheme, **weight-only** W4A8: the int8 scheme's `.wscale` rule kept unchanged, a per-(group, column) `s4 = max\|M\|/7` for the nibbles, and `.gscale` = `s4/wscale` as the sidecar stage() folds in while it still knows where each nibble sits. `add_gemm_b_int4` is the emitter, `check_i4_operand` its gate — five invariants. The layout dict is written as `dtype: "I8"` on purpose, because `layout` describes the panel the ARRAY consumes after widening and its hash must equal the int8 designs' `b_layout_hash`; the entry's own `dtype: "I4"` is what says how the bytes are stored. `emit_gemm_b` in `pack_npue.py` is the one place all three schemes branch. | `pack_npue.py`, `packers/vit.py`, `packers/whisper.py`, `verify_npue.py`, `verify_i4_scheme.py`. |
| `lib/gemm_pretiled.py` | The production GEMM design library (`pretiled_array()`). | `export_gemm_rtp.py`; research driver in `gemm_pretiled_research.py`. |
| `lib/npu_ops.py` | The op vocabulary, and THE REGISTRY of what each architecture can honour (9 codes × 7 architectures = 63 cells, each with a status and a reason). The code (`gelu`, `layn`, `softm`, …) and the design directory it means are one row; the exporter reads `buildable_codes()` to decide what to compile, with no flag of its own, and the runtime's `--npu-ops` only selects among what was built. `NPU_OPS.md` is generated from this file by `tools/gen_npu_ops_doc.py` and checked by `tools/verify/verify_npu_op_matrix.py`. `--extra-ops` is `export_eltwise`'s own spelling and stays; `--npu-ops` and `--npu-extra-ops` are refused by name there. Written twice on purpose — once here, once in `runtime/include/common/npu_ops_flag.hpp` — because the exporter is Python and the runtime's parser is C++; each file points at the other, and a typo is a refusal by name on both sides. | Imported by both exporters; change it in both places at once. |
| `lib/onnx_weights.py` | The ONNX weight reader: walks the protobuf wire format itself (no protobuf runtime), recovers each tensor's checkpoint name and orientation from how the graph actually uses it, maps its external-data side files, and defines `model_digest()` — the one source digest. Memory-maps, so packing reads tensors without holding the file. | `pack_npue.py`, `packers/whisper.py`, `packers/vit.py`, `reference/onnx_io.py`, the `verify_*` gates. |
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

## int4 (W4A8) containers

int4 is a **storage** format, not a datapath. The array has no 4-bit MAC: the
payload is stored two values per byte, widened to an int8 panel at `stage()`
time, and every other part of the stack — the design set, the instruction
streams, the dispatch, the container's own `a_dtype: "i8"` — is the int8 one,
unchanged. That is what the `layout` dict says: it is written with
`dtype: "I8"` because `layout` describes the panel the **array** consumes after
the widening, so its hash equals the int8 designs' `b_layout_hash` and an int4
container runs on an existing int8 design set with **no re-export**. The
entry's own `dtype: "I4"` is what says how the bytes are stored. Two facts, two
places, deliberately not one.

The scheme (`tools/lib/gemm_i4.py`), with `M = W·fold·asmooth`:

```
wscale[j]  = max_k |M[k,j]| / 127      the INT8 rule, kept unchanged
s4[g,j]    = max_{k in g} |M[k,j]| / 7  one scale per (group of rows, column)
t[k,j]     = clip(rint(M[k,j] / s4[g,j]), -7, 7)   the stored nibble
gscale[g,j]= s4[g,j] / wscale[j]        ≤ 127/7, written as `.gscale`
q[k,j]     = clip(rint(t[k,j] · gscale[k // group, j]), -127, 127)   ← the array
```

`.wscale` keeps the int8 rule rather than taking the int4 one because with
`gscale` in front of it `|t · gscale| ≤ 7 · 127/7 = 127` by construction, so
`q` lands inside int8's range without a clip that could bite — and
`check_i4_operand` then verifies `.wscale` against the *same* bit-exact rule
`check_i8_operand` does, so both schemes' containers are held by one invariant
rather than two similar ones. Folding `gscale` in at **pack** time would mean
writing int8, i.e. not packing int4 at all; folding it at read time needs the
group, which is config's `int4_group`, written **only** by an i4 pack.

```sh
# 1. the container (group 32 by default; 0 = one group over the whole K)
python tools/pack/pack_npue.py --dtype i4 --int4-group 32 \
    --out models/all-MiniLM-L6-v2.i4.npue

# 2. the gates — numpy and a C++ compiler, no NPU and no checkpoint
python tools/verify/verify_i4_scheme.py   # scheme, ten faults, C++ parity
python tools/verify/verify_npue.py        # spec, round trip, layout, goldens

# 3. run it against an INT8 design set (see below: no re-export needed)
```

**What it covers: the same four per-layer GEMM operands** as int8 (`qkv`,
`attn_out`, `ffn_up`, `ffn_down`) and nothing else — the embedding tables,
attention's host-side fp32 work and the elementwise design sets are untouched.

**What it buys.** On MiniLM the operands drop 10.6 MB → 5.3 MB, the `.gscale`
sidecar costs 1.3 MB, and the container goes 58.68 MB (int8) → 54.73 MB, or
69.02 MB (bf16) → 54.73 MB. Like int8 it buys no throughput: the bottleneck is
the per-dispatch cost, and widening two nibbles to an int8 value is not on the
critical path.

**What it costs: accuracy, with its own limit.** `verify_npue.py` gates an i4
container on `INT4_WEIGHT_ONLY_LIMIT`, a number the file states and shows the
measurement table for — it is a **weight-half** limit like int8's, and it is an
order of magnitude looser because an int4 step is ~18× an int8 one. Measured
weight-half `1-cos` at `--device npu1`, `--int4-group` 32 unless said
otherwise:

| model | int8 | int4 |
|---|---|---|
| `all-MiniLM-L6-v2` | 4.383e-04 | 3.676e-02 (g16 2.724e-02, g0 1.008e-01) |
| `bge-small-en-v1.5` | — | 2.505e-02 |
| `bge-large-en-v1.5` (`--tile-n 32`) | — | 1.905e-02 |

**Group size is the knob.** Smaller groups track a row-varying weight more
closely and shrink the error (32 → 16 moved MiniLM 3.676e-02 → 2.724e-02); the
cost is `.gscale` rows, `ceil(K/group)` × N floats — 4× the sidecar at group 8.
`--int4-group 0` is per-channel, one group over all of K, and measures worst of
the three on every model tried.

**Refusals worth knowing about.** `--dtype i4` without `--int8` is refused
rather than accepted-and-ignored; `pack_gte` refuses the pair outright because
its smoothing calibration has no `gte` oracle; a packer that takes
`--int4-group` without `i4` names the flag and stops; `Reader.raw()` refuses an
I4 payload (it would hand back half a panel of a neighbour's nibbles); and the
runtime refuses an I4 panel staged against a non-int8 design, a payload with no
`.gscale`, no layout, or a `int4_group` that `.gscale`'s rows contradict.

The format's failure mode, and what `verify_i4_scheme.py` section 7 exists for:
a wrong nibble index, a wrong group row, a sign extended from the wrong bit or a
tie broken the other way all still produce a **right-sized, right-hashed** panel
of plausible-looking numbers. So the gate compiles
`runtime/include/common/int4_panel.hpp` and requires its decode to equal
`npue.fold_i4()`'s byte for byte.

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

## The B operand's byte order depends on the generation AND the operand dtype

`layout_hash` is what makes a stale container fail loudly, and it covers the
`mac_s`/`mac_t` of the B panel. Those two numbers are **not** the same on both
boards, and — less obviously — not the same for both operand dtypes on the same
board. Measured, not assumed — `aie.iron.kernels.mm(...).mac_dims` returns
`(r, s, t)` and the B tile interior is stored in `(s, t)` order:

| device | operand | `mac_dims` | B panel order |
|---|---|---|---|
| `npu2` (aie2p) | bf16, i8 | `(8, 8, 8)` | `(s=8, t=8)` |
| `npu1` (aie2) | bf16 | `(4, 8, 4)` | `(s=8, t=4)` |
| `npu1` (aie2) | i8 | `(4, 8, 8)` | `(s=8, t=8)` |

`npue.gemm_b_layout`'s `(8, 8)` default is npu2's, and the packers used to pass
that default straight through, so a container packed for `npu2` was read by an
`npu1` design with the columns of every 64×32 panel permuted. Nothing detected
it: the bytes are the right size, the shapes agree, both sides computed the
same `layout_hash` from the same wrong constant, and every number that came out
was plausible. It is per-device now — `pack_npue.py --device`, and the exporter
takes the pair from `--arch` — so the hash compares like with like and a
cross-generation container is refused instead of misread.

### The dtype half, and why it was missed

The fix above made the table **per device** and stopped there. That is right for
bf16 and wrong in general, and the wrongness is invisible for the same reason the
original bug was: the int8 row of a device-keyed table is a *guess* that every
layer reads, so when the guess is wrong every layer is wrong identically and all
of them agree. An int8 container on `npu1` was therefore tiled `(s=8, t=4)` —
bf16's pair — while the int8 MMAC consumed `(8, 8)`. Measured end to end on
bge-small: `1 - cos` **8.6e-01**, against 6.3e-04 once the pair was right. Not
one check objected. The exporter, the packer, the runtime's `layout_hash`
comparison and `verify_design_numerics` all read the same constant, so
`layout_hash` matched a layout the hardware does not use — the check did its job
on the value it was given and that value was wrong.

Two things follow, and both are now enforced rather than documented:

- **`mac_for_device(device, dtype)` takes the dtype as a required argument.** The
  bf16 answer is the tempting one and a default is how the int8 half got it. The
  table lives in `npue.MAC_BY_DEVICE` only; `exporters/common/consts.py` used to
  keep `MAC_BY_ARCH` beside it, and `verify_design_numerics.py` a third copy, and
  `npue_pack.cpp` a fourth — a table edited in one place and read in four is
  exactly how a wrong value comes to look like a settled fact.
- **`verify_i4_scheme.py` section 8 re-measures the table** from
  `aie.iron.kernels.mm` on every run, over four tile widths, and fails by name
  if the table and the compiler disagree. It cannot see the `npu2` row on an
  `npu1` machine and says so rather than implying it checked.

`verify_design_numerics.py` is what saw the original, in one dispatch per stream,
before any model was involved. It prints the mismatch by name:

```
LAYOUT MISMATCH: b_layout records mac (s=8, t=8); device npu1 consumes
(s=8, t=4). A container packed as recorded is misread by this design.
```
