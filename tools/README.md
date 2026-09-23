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
| `export_gemm_rtp.py` | Builds the unified `gemm_rtp` design set (`final.xclbin` + instruction streams) into `runtime/artifacts_npu*`. | `BUILD.md` §2.3; fork `README.md`. |
| `gemm_pretiled.py` | The production GEMM design library (`pretiled_array()`). | Imported by `export_gemm_rtp.py`; research driver in `gemm_pretiled_research.py`. |
| `gemm_pretiled_research.py` | Research driver for that library: presets, traces, wall-clock benchmarks, `--arch`/`--dev` selection. | Run by hand. |
| `toolchain_provenance.py` | Writes the `toolchain.json` sidecar next to each `design.json`. | Imported by `export_gemm_rtp.py`. |
| `export_validation.py` | Dumps the runtime's golden check vectors to `runtime/artifacts/validation/`. | `BUILD.md` §2.2. |
| `gen_tokenizer_tables.py` | Emits `runtime/include/tokenizers/bert_unicode_tables.hpp` from Python's `unicodedata`. | `BUILD.md` §2.2. |
| `gen_xlmr_unicode_tables.py` | Emits `runtime/include/tokenizers/xlmr_unicode_tables.hpp` the same way. | Run by hand when the XLM-R tokenizer changes. |
| `make_tail_reference.py` | Generates the fp32 reference JSON `verify_tail.py` gates against. Needs torch. | Run by hand in `.venv-ref`; `verify_tail.py` points at its output. |
| `xlmr_tokenizer_ref.py` | Pure-Python executable spec for the C++ XLM-R tokenizer. | Reference for the C++ port; run by hand. |
| `verify_npue.py` | `.npue` gate: spec conformance, bit-exact round-trip, stale-layout guard, goldens. | `BUILD.md` §2.5. |
| `verify_npue_nomic.py` | arch=2 gate for the nomic `.npue`. | Run by hand. |
| `verify_pack_parity.py` | The Python and C++ packers must agree byte for byte. | `BUILD.md` §2.5. |
| `verify_endpoint.py` | Drives `--serve` with the official OpenAI client and checks the numbers. Needs `.venv-ref`. | `BUILD.md` §2.5. |
| `verify_semantics.py` | Near/far ordering gate — no reference, no tolerance. Stdlib only, so it runs against a cold release. | Run by hand / release gate. |
| `verify_tail.py` | Per-model, per-datapath p99 tail gate. Stdlib only. | Run by hand / release gate. |
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

Spec: [`docs/04-model/npue-format.md`](../docs/04-model/npue-format.md).
