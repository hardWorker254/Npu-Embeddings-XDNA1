# tools/ — build-time tooling

Python used before anything touches the NPU: packing the model container,
exporting the IRON designs, and verifying both. Nothing here ships — Python is
build-time and prototyping only.

Runs in the **iron** environment, except the gates marked `.venv-ref` below.
Everything is invoked from the repository root.

## Inventory

| File | Role | Invoked by |
|---|---|---|
| `npue.py` | The `.npue` container: header, JSON directory, tiling, reader, writer. Reference the C++ loader must match. | Imported by `pack_npue.py`, `export_gemm_rtp.py`, `gemm_pretiled_research.py` and the `verify_*` gates. |
| `pack_npue.py` | HuggingFace checkpoint → pre-tiled, pre-fused `.npue`. | `BUILD.md` §2.2; `verify_npue.py`, `verify_pack_parity.py`. |
| `export_gemm_rtp.py` | Entry point for the unified `gemm_rtp` design set (`final.xclbin` + instruction streams) into `runtime/artifacts_npu*`. A shim; the code is `exporters/gemm_rtp/`. | `BUILD.md` §2.3; fork `README.md`. |
| `export_eltwise.py` | Entry point for the `gelu` / `layernorm` / `softmax` design sets, `--extra-ops` picking which. A shim; the code is `exporters/eltwise/`. | `BUILD.md` §2.3; `export_gemm_rtp.py --npu-extra-ops CODES`. |
| `exporters/` | The exporter implementation, split per concern: `common/` holds what both exporters share (the `npu_targets.json` reader, JIT-cache identity, the output layout, argument validation), `gemm_rtp/` and `eltwise/` hold their own geometry, build and CLI. | Imported by the two entry points. |
| `gemm_pretiled.py` | The production GEMM design library (`pretiled_array()`). | Imported by `export_gemm_rtp.py`; research driver in `gemm_pretiled_research.py`. |
| `gemm_pretiled_research.py` | Research driver for that library: presets, traces, wall-clock benchmarks, `--arch`/`--dev` selection. | Run by hand. |
| `toolchain_provenance.py` | Writes the `toolchain.json` sidecar next to each `design.json`. | Imported by `export_gemm_rtp.py`. |
| `export_validation.py` | Dumps the runtime's golden check vectors to `runtime/artifacts/validation/`. | `BUILD.md` §2.2. |
| `gen_tokenizer_tables.py` | Emits `runtime/include/tokenizers/bert_unicode_tables.hpp` from Python's `unicodedata`. | `BUILD.md` §2.2. |
| `gen_xlmr_unicode_tables.py` | Emits `runtime/include/tokenizers/xlmr_unicode_tables.hpp` the same way. | Run by hand when the XLM-R tokenizer changes. |
| `make_tail_reference.py` | Generates the fp32 reference JSON `verify_tail.py` gates against. Needs torch. | Run by hand in `.venv-ref`; `verify_tail.py` points at its output. |
| `xlmr_tokenizer_ref.py` | Pure-Python executable spec for the C++ XLM-R tokenizer. | Reference for the C++ port; run by hand. |
| `whisper_bpe.py` | The Whisper tokenizer's binary table: the single definition of its layout and of GPT-2's byte-level alphabet. | Imported by `packers/whisper.py`, `whisper_tokenizer_ref.py`. |
| `whisper_tokenizer_ref.py` | Pure-Python executable spec for the C++ Whisper tokenizer, over the packed table blob. | Reference for `verify_whisper_tokenizer.py`. |
| `gen_whisper_unicode_tables.py` | Emits `runtime/include/tokenizers/whisper_unicode_tables.hpp`: `\p{L}`, `\p{N}`, `\s` as range tables. `std::regex` cannot run Whisper's pre-tokeniser pattern, so the categories are computed where Python's `unicodedata` *is* the definition. | Run by hand when Python's Unicode data moves. |
| `verify_whisper_tokenizer.py` | Three-way gate on the Whisper tokenizer: C++, the reference, and HuggingFace, over an adversarial corpus (Cyrillic, CJK, emoji, CRLF, long runs, contractions). Exact ids, no tolerance: a transcript is text. | Run by hand; `BUILD.md` §2.5. |
| `verify_whisper_features.py` | Holds the C++ audio front end against `transformers`: the WAV reader and the ffmpeg path against Python's `wave`, the log-mel spectrogram against `WhisperFeatureExtractor`, the slaney filter bank on its own, and the two convolutions against torch with the container's own weights. The corpus includes a clip shorter than 30 s and one longer, because the padding and the cut are the whole story. | Run by hand; `BUILD.md` §2.5. |
| `verify_whisper_model.py` | Holds the two C++ NPU stacks against `transformers`: the encoder's output hidden states, then teacher-forced decoder steps one position at a time (state after the final LayerNorm, logits, and top-1), then the greedy argmax chain and the text those ids decode to. Also 16 workers vs 1 byte-for-byte, and two refusals. Gated on `1-cos`, not `max|d|`: the designs' bf16 operands put the encoder 1.7e-04 from `transformers` in cosine and 12 in absolute value on 15 elements of 576000, and a float32 replay of the same stack prices that gap exactly. | Run by hand; `BUILD.md` §2.5. Needs the NPU, the container, the design set and the checkpoint. |
| `npu_ops.py` | The op vocabulary shared with the runtime: code (`gelu`, `layn`, `softm`), the design directory each code means, and the flag names (`--npu-extra-ops` on both sides -- it builds the design at export time and selects it at run time; `--extra-ops` is export_eltwise's own spelling). Written twice on purpose — once here, once in `runtime/include/common/npu_ops_flag.hpp` — because the exporter is Python and the runtime's parser is C++; each file points at the other, and a typo is a refusal by name on both sides. | Imported by both exporters; change it in both places at once. |
| `verify_whisper_cli.py` | Holds the `transcribe` CLI and `POST /v1/audio/transcriptions` against transformers: the long-form window schedule (30 s windows, 5 s stride each side, so windows start 20 s apart), each window's own text against `generate()`, the merged text against transformers' own `_find_longest_common_sequence` (imported, not reimplemented), and ten refusals. Starts the server itself. | Run by hand; `BUILD.md` §2.5. Needs a built runtime. |
| `verify_whisper.py` | Word error rate of the C++ transcription path against a **human** transcript, with `transformers`' WER on the same audio printed beside it — the only Whisper gate with an opinion of its own; the other two only compare against another program. No audio is committed: the corpus is an argument (`--audio` + `--ref`, or `--corpus DIR`). Normalises to lowercase words with punctuation dropped, and reports substitutions / deletions / insertions separately, because a WER of 0.12 means something different when it is 12 wrong words than 12 missing ones. | Run by hand with a corpus; `BUILD.md` §2.5. Needs a built runtime. |
| `verify_npue.py` | `.npue` gate: spec conformance, bit-exact round-trip, stale-layout guard, goldens. | `BUILD.md` §2.5. |
| `verify_npue_nomic.py` | arch=2 gate for the nomic `.npue`. | Run by hand. |
| `verify_pack_parity.py` | The Python and C++ packers must agree byte for byte. | `BUILD.md` §2.5. |
| `verify_endpoint.py` | Drives `--serve` with the official OpenAI client and checks the numbers. Needs `.venv-ref`. | `BUILD.md` §2.5. |
| `verify_semantics.py` | Near/far ordering gate — no reference, no tolerance. Stdlib only, so it runs against a cold release. | Run by hand / release gate. |
| `verify_tail.py` | Per-model, per-datapath p99 tail gate. Stdlib only. | Run by hand / release gate. |
| `verify_design_numerics.py` | Feeds random matrices through an exported design set's own instruction streams and compares C against numpy; `--npue`/`--tensor` stages real container operands verbatim. The only check that a design *computes* what it claims to, and the only one that spans design, file and hardware at once; needs an NPU. | Run by hand after any `export_gemm_rtp.py` change, and per model when its design set is first built. |
| `parity_exporters.py` | Diffs the split `exporters/` against the monoliths in git: same stdout, stderr and exit code on 17 documented invocations, plus 26 flag-presence checks. The split is committed now, so the reference side is resolved FROM HISTORY per tool — the newest revision at which that tool's entry point was still a monolith rather than a shim onto `exporters/`, which is not the same question as "when did the package appear" (one commit in this history is a shim whose package was never committed). It prints the revision it used. `--rev` overrides both. | Run by hand after touching anything under `exporters/`. |
| `semantic_corpus.json` | The shared near/far corpus (data, not a tool). | Read by `verify_semantics.py` and `make_tail_reference.py`. |

## Why the weights are transformed offline

The runtime must not transpose, convert dtypes, concatenate or re-tile — all are
pure functions of the weights. What is left at load time is `mmap` and pointer
arithmetic. The layout descriptor is **data, not code**: to try different tile
dimensions, repack rather than editing a loader. `layout_hash` makes a stale file
fail loudly instead of producing plausible garbage embeddings.

## Run

```sh
python tools/pack_npue.py             # -> models/all-MiniLM-L6-v2.npue
python tools/verify_npue.py           # bit-exact round trip, layout guard, goldens
python tools/export_validation.py     # golden check vectors for the runtime
```

Retuning the tiling repacks instead of editing the loader:

```sh
python tools/pack_npue.py --tile-n 32 --out models/minilm_n32.npue
```

The `.npue` is gitignored — a deterministic derivative of the sha256-pinned
checkpoint, which travels inside the file as `source_sha256`.

## Design export

`export_gemm_rtp.py` builds each (shape × batch tier) design, then refusing
unless every design shares one static configuration (they must differ only in
UUID metadata). The commands for this fork's two served models are in
[`README.md`](../README.md)'s `Operations:` block.

The two exporter entry points are thin. `exporters/common/` is the shared half —
`targets.py` reads `npu_targets.json`, `cache.py` decides which cached entry is
this design and which are stale, `paths.py` owns the one output convention, and
`validate.py` holds the checks that do not depend on which exporter runs. Neither
`exporters/` package may use `from __future__ import annotations`: `aie.iron.jit`
reads kernel annotations at decoration time to tell `CompileTime[T]` from a
runtime scalar, and PEP 563 stringified annotations make it reject a defaulted
parameter with a `TypeError` raised from inside the decorator.

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
