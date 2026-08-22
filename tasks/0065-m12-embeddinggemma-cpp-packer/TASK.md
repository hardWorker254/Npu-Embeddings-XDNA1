# 0065 — EmbeddingGemma-300M: the C++ packer mirror (arch=1 for `npue_pack.cpp`)

- **Date** 2026-08-20
- **Milestone** M12 (closes the single largest gap tasks/0064 left open)
- **Status** done — byte-identical parity confirmed, correctness gate
  re-verified against the fresh C++-packed container. No NPU/IRON work; pure
  C++ file-format code, verified by round-trip and byte-diff, not hardware.

## Goal

Task 0064 wired EmbeddingGemma-300M into the runtime end to end on the CPU
host path, and added `pack_gemma()` to `tools/pack_npue.py` (the Python
packer) for `arch=1` containers, verified 317/317 tensors bit-exact. It
explicitly left `runtime/src/npue_pack.cpp` (the C++ mirror, exposed as
`npuembed.exe --prepare-model`) with **no `arch=1` support** — the project's
long-standing invariant that both packers produce byte-identical `.npue`
output (`tools/verify_pack_parity.py`, the existing gate for the BERT arch)
did not hold for Gemma. This task closes that gap: port `pack_gemma()` into
`npue_pack.cpp`, following the exact structural pattern the BERT path in both
files already uses, and prove byte parity the same way the BERT path is
proven — not by inspection, by `sha256`.

## Context read before writing any code

- `tasks/0064-m12-embeddinggemma-arch1-integration/TASK.md` in full — the
  primary reference for what `pack_gemma()` does and why.
- `tools/pack_npue.py` in full, especially `pack_gemma()` (lines 127–260) and
  `add_gemm_b_host()` (108–124): every GEMM operand is stored PLAIN — F32,
  row-major `[K,N]`, no `block_panel` tiling, no `layout_hash` — because there
  is no NPU kernel for this arch, so nothing here ever becomes a DMA
  descriptor. This makes the Gemma path structurally *simpler* than BERT's:
  no `tile_k`/`tile_n`/MAC-sub-tile geometry to replicate, just a transpose
  and a raw copy.
- `tools/npue.py`'s `Writer`/`ARCH_GEMMA3_MQA_ROPE_GEGLU = 1` and the
  `.npue` header/JSON-directory format.
- `runtime/src/npue_pack.cpp` and `runtime/include/npue_pack.hpp` in full —
  the existing BERT `prepare_model()`, its `Writer` class (already
  general-purpose: `add()` takes an optional empty `layout_json`/`layout_hash`
  and omits the `"layout"` JSON key when they're empty — exactly what a
  plain, untiled tensor needs, with **zero changes**), `read_safetensors()`
  (already F32-only, already refuses other dtypes loudly), `Sha256`, `slurp`,
  `tile_b` (BERT-only, unused by the new path).
- `tools/verify_pack_parity.py` — the existing byte-diff gate, generic enough
  (invokes both packers as subprocesses, compares `sha256`) that it needed
  only a precondition-file fix, not new comparison logic.
- `runtime/src/main.cpp`'s `--prepare-model` CLI block (~line 1738) and the
  existing arch=1 runtime-dispatch block (~line 1828) that reads
  `config_string("arch")` from the packed container — confirms the packed
  header's numeric `arch` field is *not* read back by the runtime dispatch
  (only the JSON config's `"arch"` string is), but byte parity with Python
  still requires writing header `arch=1`, since `Writer::write()` serialises
  it into the fixed 64-byte header.

## What was done

### 1. `runtime/include/npue_pack.hpp` / `runtime/src/npue_pack.cpp`

- `Writer::write()` gained a third parameter, `uint32_t arch = 0` — the BERT
  call site (`w.write(out, cj.str())`) is unchanged (default keeps arch=0);
  a new Gemma call site passes `arch=1` explicitly. This was the only change
  needed to the container header path; the JSON-directory writer and the
  aligned-blob writer were already dtype/layout-agnostic.
- New `static void add_gemm_b_host(Writer &w, name, const Tensor &t)`,
  placed next to the existing `add_gemm_b()`: transposes the checkpoint's
  `[out,in]` `nn.Linear` weight to `[in,out]=[K,N]`, stores it as raw F32
  with role `"gemm_b_host"`, no tiling, no `layout` key — the direct C++
  mirror of `pack_npue.py`'s `add_gemm_b_host()`.
- New `void prepare_model_gemma(model_dir, out, source_repo, log)`, declared
  in `npue_pack.hpp`, defined at the end of `npue_pack.cpp`: reads
  `model.safetensors`, `config.json`, `2_Dense/model.safetensors`,
  `3_Dense/model.safetensors` and (if present) `gemma_tokenizer.bin` from
  `model_dir`, and writes the 317-tensor container in the exact tensor order
  `pack_gemma()` uses (`embed_tokens.weight`, `norm.weight`,
  `tokenizer.gemma_table`, then per layer `q_proj/k_proj/v_proj/q_norm/
  k_norm/o_proj/{4 layernorms}/gate_proj/up_proj/down_proj`, then
  `dense2.weight`/`dense3.weight`).
- Config-field extraction: three small lambdas —`cfg_int`/`cfg_str` (mirroring
  the BERT path's existing `cfg_int` pattern) and a new `cfg_raw(key)` that
  returns a config.json numeric field's **literal source text**, unparsed,
  rather than reparsing and reformatting it. This matters for
  `rms_norm_eps` (`1e-06`), `rope_theta` (`1000000.0`) and
  `rope_local_base_freq` (`10000.0`): Python's `json.dumps` only reproduces a
  float byte-for-byte when the source text is already its own
  shortest-round-trip form, which is true for every numeric field in this
  checkpoint's `config.json` (checked directly: `1e-06`, `1000000.0`,
  `10000.0`, `512`, `256`, `262144`, `2048` all round-trip unchanged through
  Python's `json` module — see "Verification" below). Copying the literal
  substring sidesteps having to reimplement Python's float-to-shortest-string
  algorithm in C++ for a value that never needs one.
- The output config JSON string is built as a flat sequence of `+=` in the
  exact key order and formatting of `pack_gemma()`'s Python dict literal
  (`json.dumps(..., separators=(",", ":"))` preserves insertion order).
- `main()`'s `--prepare-model` block in `runtime/src/main.cpp` gained an
  early branch, inserted right after computing the default `out` path and
  before the BERT-only `tile_k`/`tile_n`/pooling logic: it peeks
  `config.json`'s `model_type` (via the existing `npue::http::
  json_field_string` helper, already used elsewhere in this file for the
  same kind of flat-JSON read), and if it is `"gemma3_text"`, resolves
  `source_repo` the same way the BERT path does (`--source-repo` flag, else
  `CHECKPOINT.json`'s `repo_id`), calls `npue::prepare_model_gemma(...)`, and
  returns — mirroring `tools/pack_npue.py`'s `main()`, which makes the exact
  same dispatch decision the exact same way. The BERT path below this block
  is **completely unmodified**.

### 2. `tools/verify_pack_parity.py`

The existing gate was generic enough (spawn both packers as subprocesses,
`sha256`-compare the outputs) that no new comparison logic was needed — only
its precondition file check assumed every checkpoint has `vocab.txt`, which
Gemma checkpoints don't (the tokenizer travels as `gemma_tokenizer.bin`
instead, per tasks/0061). Fixed by reading `config.json`'s own `model_type`
(the same signal both packers use to dispatch) and checking for
`vocab.txt` only on a non-Gemma checkpoint, or for `2_Dense/model.safetensors`
+ `3_Dense/model.safetensors` on a Gemma one. Everything else in the script —
argument passing (`--tile-n` is harmless-but-unused on the Gemma path in both
packers), the `sha256` comparison, the byte-level diff-diagnosis on failure —
needed no changes.

## Commands

```powershell
# Build (PowerShell, not the Bash tool's cmd /c nesting -- tasks/0059's
# documented harness quirk: it silently no-ops and prints only the cmd.exe
# banner)
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cd runtime
cmd /c "call `"$vcvars`" && cmake --build build --config Release --target npuembed"

# Pack via the C++ packer into a NEW path -- never touches the Python
# packer's output from tasks/0064
.\build\npuembed.exe --prepare-model ..\models\embeddinggemma-300m `
    ..\models\embeddinggemma-300m.cpp_test.npue

# The real gate: extended verify_pack_parity.py, byte-identical or not
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_pack_parity.py `
    --model-dir models\embeddinggemma-300m
# regression check: BERT path unaffected
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_pack_parity.py

# Correctness, not just parity: re-run task 0064's own correctness gate
# against the FRESH C++-packed container
.\runtime\build_gemma_encode\gemma_encode_cli.exe `
    models\embeddinggemma-300m.cpp_test.npue `
    runtime\build_gemma_encode\texts_corpus.txt out_cpp_packer_verify.f32 64 document
& "C:\Users\vegar\Documents\GitHub\NpuEmbeddings\.venv-ref\Scripts\python.exe" `
    tools\verify_gemma_cpu_encode.py --texts runtime\build_gemma_encode\texts_corpus.txt `
    --cpu-out runtime\build_gemma_encode\out_cpp_packer_verify.f32
```

## Result

**Byte parity: PASS, byte-identical.** `tools/verify_pack_parity.py
--model-dir models/embeddinggemma-300m` (extended, this task) reports both
packers produce a `sha256`-identical 1,239.65 MB, 317-tensor container:

```
python : d163731455b2527d88a67a9ebee431ead2bde9bda482736313f571638f495acb
c++    : d163731455b2527d88a67a9ebee431ead2bde9bda482736313f571638f495acb

PASS -- byte-identical
```

Confirmed independently outside the harness too: `sha256sum` on the fresh
`models/embeddinggemma-300m.cpp_test.npue` (C++, this task) against
`models/embeddinggemma-300m.npue` (Python, tasks/0064's own artifact, never
touched by this task) — same digest, same 1,239,646,208-byte size
(`sha256_byte_diff.txt`).

**BERT regression check: PASS, byte-identical (unchanged).** `tools/
verify_pack_parity.py` with no `--model-dir` (defaults to all-MiniLM-L6-v2)
still reports byte-identical 69.02 MB containers after the `Writer::write()`
signature change — the new `arch` parameter's default (0) preserves the BERT
call site exactly (`verify_pack_parity_bert_regression.txt`).

**Correctness, re-verified against the fresh C++-packed container** (not
assumed from parity): `gemma_encode_cli.exe` against
`embeddinggemma-300m.cpp_test.npue`, then `tools/verify_gemma_cpu_encode.py`
against `reference/encoder_gemma.py`, on both of tasks/0064's own corpora:

| corpus | worst `1-cos` | task 0064's own figure (Python-packed container) |
|---|---:|---:|
| `texts_corpus.txt` (4 sentences) | **5.495604e-13** | 5.496e-13 |
| `texts_adhoc.txt` (ASCII/accented/CJK/technical) | **4.969358e-12** | 4.969e-12 |

Identical to the digit tasks/0064 recorded. Also confirmed directly:
`gemma_encode_cli.exe`'s raw output file against the C++-packed container
`cmp`-matches its output against the original Python-packed container
bit-for-bit (both reproduce tasks/0064's own `out_corpus.f32` exactly) — the
byte-identical container trivially reproduces the byte-identical encode, and
this step confirmed that rather than assuming it.

## Problems hit

- **None that survived to the final version.** The design ported cleanly:
  the existing `Writer` class needed exactly one new parameter
  (`Writer::write`'s `arch`), `read_safetensors()` and `Sha256` needed no
  changes at all, and the plain/untiled storage discipline
  (`add_gemm_b_host`) is *less* code than the tiled BERT `add_gemm_b` it sits
  next to.
- **The one real risk was float formatting**, not tiling: Python's
  `json.dumps` re-serialises `rms_norm_eps`/`rope_theta`/
  `rope_local_base_freq` from parsed Python floats, and a C++ reformat of the
  same doubles could plausibly disagree in the last digit or the exponent
  form (`1e-06` vs `1e-6` vs `1.000000e-06`) without any test catching it
  except a byte-diff. Sidestepped by copying the literal config.json
  substring rather than reparsing-and-reformatting, after confirming by
  direct `json.dumps(json.loads(x))` round-trip in Python that this
  checkpoint's specific literals (`1e-06`, `1000000.0`, `10000.0`) are
  already in canonical form. This is a checkpoint-specific verification, not
  a general guarantee — flagged below.
- **`verify_pack_parity.py`'s file precondition check** (`vocab.txt`
  required unconditionally) would have rejected the Gemma checkpoint outright
  before either packer ran. Found immediately on first invocation (`missing
  models\embeddinggemma-300m\vocab.txt`), fixed by branching the required-file
  set on `config.json`'s own `model_type`, the same signal both packers use.

## Known limitation, stated rather than hidden

`cfg_raw()`'s literal-copy strategy for `rms_norm_eps`/`rope_theta`/
`rope_local_base_freq` is provably correct for *this* checkpoint's exact
`config.json` text (verified by direct round-trip through Python's `json`
module, not assumed) but is not a general float-formatting implementation —
a future Gemma-family checkpoint whose `config.json` writes one of these
fields in a non-canonical form (e.g. trailing zeros, a different exponent
style) would produce a byte-for-byte parity **failure** rather than a silent
wrong answer, since `verify_pack_parity.py` would catch it immediately. This
is the deliberately narrower, safer failure mode: loud non-parity over silent
wrong formatting.

## Artifacts

- `runtime/include/npue_pack.hpp` — `prepare_model_gemma()` declared
  (modified).
- `runtime/src/npue_pack.cpp` — `Writer::write()` gained an `arch` parameter;
  `add_gemm_b_host()` and `prepare_model_gemma()` added; BERT `prepare_model()`
  untouched (modified).
- `runtime/src/main.cpp` — early `model_type`-dispatch branch inside the
  `--prepare-model` CLI handler; BERT branch below it unmodified (modified).
- `tools/verify_pack_parity.py` — precondition file check now branches on
  `config.json`'s `model_type` (modified).
- `tasks/0065-m12-embeddinggemma-cpp-packer/verify_pack_parity_gemma.txt`,
  `verify_pack_parity_bert_regression.txt`,
  `verify_gemma_cpu_encode_cpp_packed_corpus.txt`,
  `verify_gemma_cpu_encode_cpp_packed_adhoc.txt`, `sha256_byte_diff.txt`,
  `texts_corpus.txt`, `texts_adhoc.txt` — verification output and the exact
  corpora used (committed).
- `models/embeddinggemma-300m.cpp_test.npue` — the fresh C++-packed
  container, 1,239.65 MB, byte-identical to `models/embeddinggemma-300m.npue`
  (gitignored via the repo's blanket `*.npue` rule; not committed, not
  needed since it is reproducible from the command above and never diverges
  from the tracked Python-packed one).
- `research/OPEN-THREADS.md` — T29 updated with this task's result.
- `tasks/README.md` — index row added.

## Next

The C++ packer mirror gap from tasks/0064 is now **fully closed**: both
packers produce byte-identical `.npue` containers for both architectures
they support, verified by the same `sha256`-based gate
(`verify_pack_parity.py`) either project uses. What remains toward a fully
production Gemma path is unchanged from tasks/0064's own "Next" section and
was out of scope here: `HF_TOKEN`-gated fetch of the official checkpoint and
a `hub.cpp` catalogue row, an MTEB run, and CPU-side speed work
(batching/AVX2/threading). None of those are blocked on the packer anymore.
