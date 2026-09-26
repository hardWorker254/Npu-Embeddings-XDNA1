# NpuEmbeddings

Sentence embeddings for BERT-family models on an AMD NPU, with the host CPU as
a fallback path. Runs `serve` (an OpenAI-shaped HTTP endpoint) and `embed` (a
file in, a file out).

This tree targets **XDNA1 / `npu1` on Linux**. The generation is selected with
`--dev`; the design set records which one it was built for and a set built for
the other generation is refused rather than loaded.

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [How a request is executed](#how-a-request-is-executed)
- [Where each operation runs](#where-each-operation-runs)
- [Command-line reference](#command-line-reference)
- [Models](#models)
- [Building design sets](#building-design-sets)
- [Accuracy](#accuracy)
- [Known defects](#known-defects)
- [Performance notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)

---

## Requirements

**Runtime** (building and running the C++ binary):

- Linux with an AMD NPU exposed at `/dev/accel0` and the `amdxdna` driver
- XRT, e.g. `/opt/xilinx/xrt` — found via `XRT_ROOT`, `XILINX_XRT`, or the
  default `/opt/xilinx/xrt`
- A C++17 compiler, CMake ≥ 3.20, `libcurl` (used to download model weights)

**Artifacts** (one-time, per model *and* per NPU generation):

- Python with MLIR-AIE / IRON
- **Peano** (the `llvm-aie` package) — required. `kernels/*.cc` are compiled as
  AIE *external functions*, so aiecc cannot build a design without Peano's
  `clang++`. A missing Peano surfaces as
  `RuntimeError: Invalid Peano install directory: peano_not_found`.

Check the device:

```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine          # expect RyzenAI-npu1
ls /dev/accel0
```

Environment for the artifact build (bundle it in a local helper script if you
like — this is what such a script should contain):

```bash
source /opt/xilinx/xrt/setup.sh
export XRT_INCLUDE_DIR=/opt/xilinx/xrt/include
export PEANO_INSTALL_DIR="$HOME/.local/lib/python3.14/site-packages/llvm-aie"
```

The runtime build reads `XRT_ROOT`/`XILINX_XRT`; `XILINX_XRT` pointing at a
Windows path does not break it, because `CMakeLists.txt` accepts either and
falls back to `/opt/xilinx/xrt` on Linux.

---

## Quick start

Three steps. Steps 1 and 2 are needed once per machine; step 1 is additionally
needed once per model.

### 1. Build the design set

A *design set* is a compiled `xclbin` plus its instruction streams, specialised
for one model geometry, one NPU generation and one batch tiering. Nothing runs
without it.

```bash
# unified GEMM set — this is the mandatory one
python tools/export_gemm_rtp.py --target all-MiniLM-L6-v2 --arch 1 --out runtime
```

This writes `runtime/all-MiniLM-L6-v2/artifacts_npu1/gemm_rtp/`.

Add `--npu-eltwise` to also build `gelu/`, `layernorm/` and `softmax/` beside
it. Only needed for `--npu-eltwise` at run time; see
[Where each operation runs](#where-each-operation-runs) for why you probably
do not want it.

Preview what would be built, without invoking the toolchain:

```bash
python tools/export_gemm_rtp.py --target all-MiniLM-L6-v2 --arch 1 --out runtime --dry-run
```

### 2. Build the runtime

```bash
cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build -j"$(nproc)"
```

Produces `runtime/build/npuembeddings` (and a copy named `npuembed`).

### 3. Run it

```bash
# what is installed, and what can run
runtime/build/npuembeddings list

# OpenAI-shaped endpoint on 127.0.0.1:8080
runtime/build/npuembeddings serve all-MiniLM-L6-v2

# batch: one text per line -> raw fp32 matrix
runtime/build/npuembeddings embed all-MiniLM-L6-v2 texts.txt out.f32
```

`serve` downloads and verifies the weights on first use. Test it:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["hello world"]}'
```

Models with a prompt table (`nomic-embed-text-v1.5`, `gte-multilingual-base`)
require `"prompt_name"` in the body; `GET /health` lists the accepted values.
A request without it is a 400 — a wrongly-prefixed vector is correctly shaped
and correctly normed, so nothing downstream could detect it.

---

## How a request is executed

```
texts
  │  tokenize (WordPiece / XLM-R Unigram / SentencePiece)
  │  embed: word + position + token_type
  ▼
┌─ per encoder layer (6 for MiniLM, 24 for bge-large) ──────────────┐
│  LayerNorm                                                host    │
│  QKV projection                                           NPU    │
│  RoPE (where the model uses it)                            host    │
│  Q·Kᵀ + attention mask                                    host    │
│  softmax                                                  host*   │
│  A·V                                                      host    │
│  output projection                                        NPU    │
│  residual + LayerNorm                                      host*   │
│  FFN up (+ GeGLU/SwiGLU)                                   NPU     │
│  GELU                                                      host*   │
│  FFN down                                                 NPU    │
│  residual + LayerNorm                                      host*   │
└────────────────────────────────────────────────────────────────────┘
  │  mean-pool over unmasked tokens, then L2-normalise
  ▼
embedding
```

`*` = on the array instead when `--npu-eltwise` is given.

The array does the four GEMMs per layer; attention and the elementwise ops run
on the host by default because they were **measured faster on the host** — the
elementwise designs pay a fixed per-dispatch cost that a 384-wide row-wise op
does not amortise. `--npu-eltwise` exists to make that an explicit,
measurable choice, not because it is the default.

**`--pipeline N`** splits one request into right-sized chunks and runs `N` of
them concurrently, each in its own thread with its own host-side buffers and
its own device A/C slots. `serve` and `embed` pass `--pipeline 4`; a later
`--pipeline` on the command line wins, so `--pipeline 1` disables pipelining.
The NPU itself serialises dispatches, so the win is overlapping *host* work with
array work, not more array throughput.

---

## Where each operation runs

| Operation | Default | With `--npu-eltwise` |
|---|---|---|
| QKV / attention-out / FFN-up / FFN-down GEMM | array | array |
| LayerNorm | host | array |
| softmax | host | array |
| GELU | host | array |

`--host-ln`, `--host-sm`, `--host-gelu` force an individual op back onto the
host even when `--npu-eltwise` is given.

`--npu-eltwise` requires all three sibling design sets (`gelu/`, `layernorm/`,
`softmax/`) next to `gemm_rtp/`. A missing one is **refused by name** — it never
silently falls back to the host, because a flag whose whole point is "put this
on the array" must not quietly not do that.

It also costs four `hw_context`s instead of one. Before allocating any, the
runtime asks `xrt-smi` how many contexts the device allows and refuses by name
if the total would not fit. `--allow-contention` overrides; a throughput number
from a contended run is not an NPU performance claim.

---

## Command-line reference

`npuembeddings --help` is authoritative and kept in sync with this section.

### Subcommands

| Command | What it does |
|---|---|
| `list` | every model this build can run, and which are installed |
| `serve <model>` | OpenAI-shaped `POST /v1/embeddings`; downloads the model if needed |
| `embed <model> <in.txt> [out.f32]` | embed a text file, one text per line |
| `add <org/model> [<sha256>]` | register a model this build does not know (a finetune) |
| `tokenize` | tokenizer round-trip, for debugging |

### Options for `serve` / `embed`

| Flag | Meaning |
|---|---|
| `--port N` | listen port (default 8080) |
| `--bind ADDR` | interface (default `127.0.0.1`, localhost only) |
| `--threads N` | host thread budget (`serve`/`embed` pass 24) |
| `--pipeline N` | concurrent encode lanes (`serve`/`embed` pass 4) |
| `--artifacts DIR` | override the design set |
| `--npu-eltwise` | run GELU, LayerNorm and softmax on the array |
| `--host-ln` / `--host-sm` / `--host-gelu` | force one op back to the host |
| `--dev npu1\|npu2` | NPU generation; a design built for the other is refused |
| `--root DIR` | override where models/ and designs live |
| `--token VALUE` | HuggingFace token for a gated model (else `$HF_TOKEN`) |
| `--prefix NAME` | task prefix, `embed` only; required for models with a prompt table |
| `--allow-truncation` | embed the first `seq` tokens instead of refusing |
| `--allow-contention` | proceed despite other processes holding NPU contexts |

`serve` **rejects** `--prefix`: its prompt is per request, via `"prompt_name"`
in the body.

### Why the safe defaults are the way they are

- **Truncation is an error, not a warning.** A truncated text still returns a
  correctly shaped, correctly normed vector, so nothing downstream can tell the
  answer is wrong — and inputs sharing a preamble truncate to *identical*
  vectors. Without `--allow-truncation` such an input is an error naming its
  real token count.
- **This build runs at the sequence length its design was exported for**
  (`--seq` to the exporter, default 64).
- **`--bind` defaults to localhost.** There is no authentication on this
  endpoint.

---

## Models

From `npuembeddings list` on this machine. `npu_targets.json` is the exporter's
source of truth for geometry; the catalogue in `src/common/hub.cpp` is the
runtime's.

| Model | Layers | Hidden | Pooling | Size | Notes |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 6 | 384 | mean | 91 MB | smallest and fastest; head_dim 32 keeps attention off the array |
| `bge-small-en-v1.5` | 12 | 384 | cls | 134 MB | MiniLM's width at twice the depth |
| `bge-base-en-v1.5` | 12 | 768 | cls | 438 MB | best geometric fit for this NPU: head_dim 64, every N a multiple of 384 |
| `bge-large-en-v1.5` | 24 | 1024 | cls | 1340 MB | highest quality; N=1024 forces `tile_n 32`, 24 layers cost dispatches |
| `embeddinggemma-300m` | 24 | 768 | mean | 1155 MB | MQA + RoPE + GeGLU on the array (4 GEMMs/layer); gated |
| `nomic-embed-text-v1.5` | 12 | 768 | mean | 547 MB | RoPE + gated SwiGLU; needs `--prefix` |
| `gte-multilingual-base` | 12 | 768 | cls | 582 MB | multilingual, XLM-R tokenizer; NTK RoPE + gated GeGLU |

`list` also reports a *state* per model — `ready`, `available`, `cpu`,
`no design`, `no encoder` — which says whether `serve` can actually run it here.

Registering a model this build does not know:

```bash
npuembeddings add org/finetune-name <sha256>
```

Without the sha256 the weights are **not** verified; that is allowed and warned
about on every run.

---

## Building design sets

### `tools/npu_targets.json`

Single source of truth for model geometry, per-generation defaults, and the
stream list per `kind`. Inspect it:

```bash
python tools/export_gemm_rtp.py --list-targets
```

Per-generation defaults:

| arch | device | cols | batch | tiers |
|---|---|---|---|---|
| 1 | `npu1` | 4 | 16 | 4, 8, 16 |
| 2 | `npu2` | 8 | 128 | 4, 16, 32, 128 |

### Exporter flags worth knowing

| Flag | Meaning |
|---|---|
| `--target NAME` | take the model's geometry from `npu_targets.json` |
| `--list-targets` | print the catalogue and exit |
| `--dry-run` | print the resolved argv, call no toolchain |
| `--arch 1\|2\|all` | NPU generation |
| `--out DIR` | artifact root (usually `runtime`) |
| `--batches a,b,c` | batch tiers to emit an instruction stream for |
| `--npu-eltwise` | also build `gelu/`, `layernorm/`, `softmax/` |
| `--elt-cols N` | array columns for the eltwise designs (LayerNorm/softmax cap at 2) |
| `--seq N` | sequence length to build for |
| `--cache-root DIR` | IRON JIT cache (default `~/.npu/cache`) |

### Two output layouts

`--target` puts designs under a per-model subdirectory:

```
runtime/<model>/artifacts_npu<N>/gemm_rtp/
```

Without `--target` the older flat layout is used:

```
runtime/artifacts_npu<N>/gemm_rtp/
```

The runtime searches both (`artifacts_candidates()` in
`include/common/design_selection.hpp`), so either works. Manual flags override
values from the config.

### A design set is not portable between machines

`design.json` records the `arch`, `device` and the toolchain that built it. A
set built for `npu2` is **refused** by `npu1`, not loaded and misread. Rebuild
per generation.

### Rebuilding is idempotent, artifacts are gitignored

`export_eltwise.py` purges matching JIT-cache entries before rebuilding and then
requires exactly one candidate, because a cache hit does not restamp a directory
and neither the symbol nor the buffer size distinguishes two column counts.

`final.xclbin` is not byte-reproducible (the container embeds build paths), so
a rebuild legitimately changes its hash while `insts.bin` may not change at all
— a change in the *core program* moves the xclbin and leaves the host-side BDO
stream alone.

---

## Accuracy

The runtime compares its output against HuggingFace goldens on every default run
(no `--bench`, no `--serve`, no `--embed`), per row, with no Python in the
process. The gate is `worst 1 - cos ≤ 2e-3` across **all** rows.

Recorded results for the shipped datapatbs, from
[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — measured on the project's
reference machine, **not** on this one:

| model | datapath | worst `1 - cos` | p99 tail |
|---|---|---:|---:|
| `all-MiniLM-L6-v2` | bfp16 | 3.406e-04 | 9.7e-04 |
| `bge-small-en-v1.5` | bf16 | 8.348e-06 | 1.2e-05 |
| `bge-base-en-v1.5` | bfp16 | 2.284e-04 | 3.7e-04 |
| `bge-large-en-v1.5` | bfp16 | 2.626e-04 | 6.6e-03 (waived) |
| `nomic-embed-text-v1.5` | bfp16 | 1.402e-03 | 1.6e-03 |
| `gte-multilingual-base` | bfp16 | 4.608e-04 | 5.2e-04 |
| `embeddinggemma-300m` | bfp16 | differential | 4.2e-04 |

`embeddinggemma-300m` is arch=1 and has no HuggingFace golden, so it is checked
*differentially* against the bf16 NPU encode instead.

**What this gate is and is not.** `1 - cos` is a fidelity check on the
arithmetic, not a quality gate. On the int8 containers `bge-large-en-v1.5`
measured `2.968e-03` — a FAIL against the `2e-3` gate — while passing MTEB at
mean −0.05. It is still the check that caught every real bug it was pointed at,
precisely because it is sensitive to what MTEB averages away. Quality claims
come from MTEB; this is the regression alarm.

Notes on the check itself:

- The goldens are batch 4. Larger batches are tiled by **rotating** which base
  sequence lands in which row, with the expected output rotated identically.
  Plain tiling would make every physical copy byte-identical, and a row-indexing
  or cross-row aliasing bug would then be invisible to a content comparison.
- Every output row is compared, not just the first four.
- A non-finite check runs alongside the tolerance check. `std::max(0.0, NaN)`
  returns `0.0`, so a NaN-producing kernel would otherwise score a *perfect*
  `1 - cos` and pass a test whose failure mode is a perfect score.
- The fixtures are matched by **checkpoint sha256**, not by model name, so a
  renamed directory cannot silently validate against the wrong goldens.
- `--pipeline > 1` additionally requires every lane to reproduce lane 0
  **bitwise**; a disagreement is reported as cross-lane corruption.

The array LayerNorm path is verified against the *host* path rather than against
goldens directly — see [Known defects](#known-defects).

---

## Known defects

### LayerNorm on the array used the previous layer's parameters — fixed

**Symptom.** With `--npu-eltwise`, the LayerNorm result was not reproducible and
was not correct. Consecutive requests for the same input answered differently
(cos ≈ 0.38 against ≈ 0.68), and mixing batch tiers inside one request made it
worse. The host LayerNorm path — the default — was never affected.

**Root cause.** A `defect class` in the IRON program, not in the C++ runtime.
In `tools/export_eltwise.py`, the LayerNorm worker acquired its gamma|beta
object once and **never released it**:

```python
def core_fn(a, pm, c, ln):
    ep = pm.acquire(1)          # a and c are balanced; pm was not
    for _ in range_(per_core):
        ...
    pm.release(1)               # <-- the fix
```

`sequence()` issues exactly one `fill(P, ...)` per program run, so a run consumed
one object and returned none. The L1-forwarded params buffer (`depth=1`) was
therefore still occupied when the program ended, and the next dispatch's
`acquire` could be served by the previous run's leftover instead of waiting for
its own fill. **Every array LayerNorm computed with the previous site's
gamma/beta.**

GELU and softmax were never affected: their workers have no params fifo at all,
which is exactly what the isolation measurements showed (12 runs per
configuration, LayerNorm on the array was the only broken one).

**Effect on accuracy.** Against the host path, before and after:

| | cos |
|---|---|
| before | 0.8987 |
| after | **0.9999** |

**Fix and verification.** One line, plus a rebuild of the design set. Verified:
12 runs per configuration across every batch-tier mix, all producing a single
bit-identical answer, and `--pipeline 1` and `--pipeline 4` now agree bitwise on
all of them.

**A second, independent defect fixed alongside it.** The `--pipeline` lanes
shared one `npu::Design` per operation, and the eltwise designs — unlike the
GEMM design — gave their lanes no private input/output buffers, while
`layer_norm()` and `eltwise()` took no dispatch mutex at all. Lanes overwrote
each other's rows. Each lane now gets its own A and C slots and the whole
bind → sync → dispatch → sync_from window is taken under the same `npu_mu` that
`gemm()` already used. This was a real race, but on its own it accounted for
only part of the damage (10/12 → 6/12 distinct answers); the IRON imbalance
above was the rest.

**Consequence for artifacts.** The fix lives in the exporter, and design sets
are gitignored, so the corrected `layernorm/final.xclbin` is **not** in this
repository. Anyone building from source must re-run the exporter; a stale
artifact directory still carries the bug.

### Roadmap items not yet done

Listed under [Roadmap](#roadmap).

---

## Performance notes

**Measure before believing any number, and measure on this machine.** The
throughput table in [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) was
recorded on the project's reference machine for the reference NPU; it is a
baseline to compare against, not a claim about your hardware. The mechanisms
below are the ones that actually decide the number.

- **The array is shared.** Wall clock is never an NPU performance claim on its
  own. A leftover process holding an `Active` `hw_context` was measured reading
  221 seq/s against a true 694 on the same binary, minutes apart. `xrt-smi
  examine -r all` is the tool that knows who holds what; the runtime refuses a
  timed run when the array is not exclusively its own, unless you pass
  `--allow-contention`.
- **Per-dispatch cost is fixed and not small** (~150 µs measured). A design set
  is loaded once and kept precisely to avoid paying it per dispatch — which is
  also why the elementwise ops default to the host, and why `--npu-eltwise` can
  *lower* throughput rather than raise it.
- **Batch tiers are right-sized, not padded.** A request is split into the
  largest tier that fits; the design set carries an instruction stream per tier.
- **`--npu-eltwise` costs four `hw_context`s** instead of one, against a
  measured budget of six on this driver. XRT exposes no usable-count query, so
  the budget is a labelled constant and the source of the number is printed
  beside it rather than hardcoded silently.
- **`--pipeline N` divides the host thread budget.** `--threads 8 --pipeline 4`
  gives each lane 2. Requesting more lanes than there are chunks in a request
  buys nothing.

The timed path prints its own breakdown — wall, host threads busy, per-phase
costs (bf16 conversion, sync to device, dispatch + wait, sync from device, read
out + bias, host attention), and per-design wait time — so a regression can be
attributed to a phase rather than guessed at.

---

## Troubleshooting

**`DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22)`** — the device's
`hw_context` budget is full: another NPU process, or an earlier run of yours
that was killed before releasing its contexts. `xrt-smi examine -r all` lists
the holders. Close them, or pass `--allow-contention` if the contention is
intended.

**`Invalid Peano install directory: peano_not_found`** — Peano is not
installed. `kernels/*.cc` compile as AIE external functions and need Peano's
`clang++`. Set `PEANO_INSTALL_DIR` to the `llvm-aie` package directory.

**`Unsupported device: npu`** from the exporters — the NPU tensor backend fell
back to CPU-only. Put XRT's Python bindings on the path and select the XRT
backend:

```bash
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH
export NPU_RUNTIME=xrt
```

**`--npu-eltwise asks for <op> on the array, but <dir>/design.json does not
exist`** — the sibling eltwise design sets were not built. Re-run the exporter
with `--npu-eltwise`, or drop the flag.

**`--artifacts '<name>': no design set found`** — the path resolved to a
directory with no `gemm_rtp/design.json` and no `qkv/design.json`. The error
lists every directory that was tried.

**`this design set records no sequence length`** — the `design.json` predates
the field. Re-export it.

**Throughput collapsed between two runs of the same command** — check for a
leftover NPU process first (`xrt-smi examine -r all`), then `--bo-mode`, then
whether the two runs used the same batch tiering. A timed run prints its
`bo-mode`, tiers and alignment for exactly this reason.

**`--pipeline N` results are slower than `--pipeline 1`** — possible when the
request is smaller than `N` chunks, so lanes idle; and `--pipeline` divides the
host thread budget, so `--threads 8 --pipeline 4` gives each lane 2.

---

## Repository layout

```
tools/                     Python: exporters, packer, verifiers (no C++ at run time)
  export_gemm_rtp.py       unified GEMM design set, per arch and batch tier
  export_eltwise.py        gelu / layernorm / softmax design sets
  npu_targets.json         model geometry + per-arch defaults (exporter's source of truth)
  pack_npue.py             build the .npue container
  verify_*.py              correctness gates
kernels/                   AIE device code (C++), compiled by aiecc via Peano
runtime/
  include/runtime/         device, design, model container, encoder, pool, run_*
  include/encoders/        BertEncoder (BERT family)
  include/embed_models/    model wrappers and the loader registry
  include/tokenizers/      WordPiece, XLM-R Unigram, SentencePiece, Whisper
  include/server/          HTTP server, embedding service
  include/cli/             argument parsing and subcommand dispatch
  include/common/          app state, host kernels, JSON, hub, packer helpers
  include/whisper/         Whisper model wrapper + the speech-to-text engine
  src/                     implementations, mirroring include/
                           (cli/ common/ embed_models/ encoders/ server/
                            tokenizers/ whisper/ + the top-level units)
docs/                      research notes and the running status log
models/                    vendored checkpoints (large)
```

**Design rules the tree follows**, so a change does not violate them by accident:

- `main.cpp` is thin: parse, construct, run. No encode logic.
- The C++ runtime compiles nothing. Every xclbin and instruction stream comes
  from `tools/`.
- `Design` owns one xclbin, its instruction streams and its buffers; the xclbin
  is loaded once and kept.
- **Shared mutable state on a `Design` is either per-lane or under `npu_mu`.**
  `--pipeline` runs several encoders against the *same* `Design` objects, so
  every dispatch site holds `npu_mu` across bind → sync-to → dispatch →
  sync-from, and every buffer a lane writes concurrently has a lane-private
  slot. `gemm()`, `eltwise()` and `layer_norm()` all do this.
- **`Model` never builds encoders.** A `BertEncoder` cannot be constructed
  without seven live `npu::Design &`, and those only exist after the device is
  opened and a design set is selected — so `Runtime` owns encoder construction,
  because it is the component that already does both. `Model` exposes geometry
  and the tokenizer only, and there is deliberately no `make_encoder()`; if a
  design-aware context is ever needed it belongs on `Runtime`, not `Model`.
- **The `Encoder` interface is text-in only** (`encode`, `hidden`, `seq`). The
  raw pre-embedded forward pass that the benchmark and golden-check paths need
  is a concrete second method on `BertEncoder`,
  `run(const std::vector<float> &emb_in)` — not on the base interface.
- **Model types plug in through a registry**, not a switch: a loader declares
  which container `arch` it handles, and `load_model()` asks each in turn.
  Adding an architecture is a new loader plus a new tokenizer, with no existing
  code edited.
- Fail loudly by name. A missing design, a wrong generation, a full context
  budget and an unpinned prefix are all errors, never silent fallbacks.

---

## Roadmap

Carried over from the deleted planning notes:

1. **End-to-end verification of the per-model layout.** Walk every model in
   `npu_targets.json` through export → serve → compare against goldens, and make
   it a gate rather than a manual pass.
2. **`tools/verify_targets.py`.** Assert that `npu_targets.json` and the
   catalogue in `src/common/hub.cpp` agree. The catalogue already holds the
   authoritative geometry; two copies of it will drift.
3. **Whisper (STT) extension point.** `npu_targets.json` has a `kinds` table
   with an `stt` entry whose `exporter` is `null` and whose `streams` are empty.
   `--target` should fail with a clear "no exporter for kind=stt" rather than
   something incidental.

---

## Further reading

| Document | What is in it |
|---|---|
| `docs/CURRENT_STATUS.md` | the running status log: measured numbers, rejected hypotheses, known walls |
| `docs/00-overview.md` | system overview and design |
| `docs/01-hardware/` … `docs/06-performance.md` | hardware, toolchain, kernels, model, measurement methodology, performance |
| `docs/history.md` | chronological history of the project |
| `tools/README.md` | exporter and packer reference |

`docs/CURRENT_STATUS.md` is the place to look for *why* something is the way it
is; this README is the place to look for *how to run it*.
