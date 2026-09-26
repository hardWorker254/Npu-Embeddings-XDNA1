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
- [Speech to text](#speech-to-text)
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

Add `--npu-extra-ops CODES` to also build the elementwise designs beside it:
`gelu`, `layn` (LayerNorm), `softm` (softmax), comma-separated, and only the
ones you name. Build only what the runtime's `--npu-ops` will ask for — each is
a compile, an xclbin, and one more `hw_context`. See
[Where each operation runs](#where-each-operation-runs) for why you probably do
not want any of them.

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

`*` = on the array instead when its op is named in `--npu-ops`.

The array does the four GEMMs per layer; attention and the elementwise ops run
on the host by default because they were **measured faster on the host** — the
elementwise designs pay a fixed per-dispatch cost that a 384-wide row-wise op
does not amortise. `--npu-ops` exists to make that an explicit, measurable
choice, not because it is the default.

**`--pipeline N`** splits one request into right-sized chunks and runs `N` of
them concurrently, each in its own thread with its own host-side buffers and
its own device A/C slots. `serve` and `embed` pass `--pipeline 4`; a later
`--pipeline` on the command line wins, so `--pipeline 1` disables pipelining.
The NPU itself serialises dispatches, so the win is overlapping *host* work with
array work, not more array throughput.

---

## Where each operation runs

| Operation | Default | On the array when |
|---|---|---|
| QKV / attention-out / FFN-up / FFN-down GEMM | array | always |
| LayerNorm | host | `--npu-ops layn` |
| softmax | host | `--npu-ops softm` |
| GELU | host | `--npu-ops gelu` |

One flag, and an op is on the host exactly when it is **not** listed — so
`--npu-ops layn,softm` is "LayerNorm and softmax on the array, GELU on the
host", and there is no inverse flag to drift against it. This replaces
`--npu-eltwise` (all three or none) and `--host-ln/--host-sm/--host-gelu`; the
old names are now **refused by name**, so a stale command line cannot look like
it worked.

`--npu-ops` needs one sibling design set per named op next to `gemm_rtp/`
(`layernorm/`, `softmax/`, `gelu/`), built by
`tools/export_gemm_rtp.py --npu-extra-ops CODES`. A missing one is **refused by
name** — it never silently falls back to the host, because a flag whose whole
point is "put this on the array" must not quietly not do that. Each op also
costs one more `hw_context` out of six.

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
| `transcribe <model> <audio.wav>` | transcribe 16 kHz audio with a Whisper model; the transcript alone on stdout |
| `add <org/model> [<sha256>]` | register a model this build does not know (a finetune) |
| `tokenize` | tokenizer round-trip, for debugging |

`serve` and `transcribe` are the same command for two different architectures:
a Whisper container is served by the STT mode off the same flags, answers
`POST /v1/audio/transcriptions` instead of `/v1/embeddings`, and refuses
`--npu-ops` rather than ignoring it. The mode is picked by the container's
`arch`, never by a flag.

### Options for `serve` / `embed`

| Flag | Meaning |
|---|---|
| `--port N` | listen port (default 8080) |
| `--bind ADDR` | interface (default `127.0.0.1`, localhost only) |
| `--threads N` | host thread budget (`serve`/`embed` pass 24) |
| `--pipeline N` | concurrent encode lanes (`serve`/`embed` pass 4) |
| `--artifacts DIR` | override the design set |
| `--npu-ops CODES` | send the named ops to the array: `gelu`, `layn`, `softm` |
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

## Speech to text

An `openai/whisper-*` container is a **speech-to-text** model, not an embedder:
it answers "text for audio", it has no pooling mode, and it runs through a mode
of its own. `npuembeddings list` shows those rows as `stt`.

### What a user has to download

**Only `model.safetensors`.** Everything else a Whisper container is built from
is committed: `config.json` (geometry), `preprocessor_config.json` (the audio
constants), `generation_config.json` (the decoding policy) and the byte-level
BPE table as `vocab.json` + `merges.txt` + `added_tokens.json` — 1.5 MB per
model against a 3 GB checkpoint, identical for everyone.

Two of those cannot be guessed, and that is why they are in the repository
rather than in the download: a `generation_config.json` without its
`suppress_tokens` list yields a container that transcribes *differently* from
`transformers` while looking perfectly healthy, and a
`preprocessor_config.json` without its constants yields a spectrogram of the
wrong audio. The packer refuses a checkpoint that lacks either, which is a
good refusal — but it would leave anyone holding the weights with a dead end.

So the flow is: download the weights, drop them in `models/<name>/`, and run
the three commands below.

### Three steps, once per model

```bash
# 1. the container. --max-seq is NOT needed and should not be passed: the
#    default is per family, and for whisper it is the checkpoint's own
#    max_source_positions (1500). The embedder default (256) writes a container
#    whose position table is shorter than one audio window, and the packer
#    refuses that by name rather than writing it.
python tools/pack_npue.py --model-dir models/whisper-base \
    --out models/whisper-base.npue --device npu1

# 2. the design sets -- BOTH, and one invocation writes both, because a Whisper
#    target resolves to an encoder pass and a decoder pass (their M differs by
#    8x, so one xclbin cannot serve both).
python tools/export_gemm_rtp.py --target whisper-base --arch 1 --out runtime

# 3. transcribe.
./runtime/build/npuembeddings transcribe whisper-base recording.wav
```

**`--max-seq` has two defaults, and that is the point.** An embedder's position
table is its own `max_position_embeddings` (256 for MiniLM). A whisper encoder
always sees `max_source_positions` rows — 1500, one 30 s window — so a whisper
container sliced shorter cannot transcribe anything and refuses at the first
request. Omit the flag and each family gets its own default; pass a value the
model cannot use and the packer says so, naming the number.

`generation_config.json` is **required** by the packer: it carries the
checkpoint's decoding policy (`suppress_tokens`, `begin_suppress_tokens`), which
decides which token a greedy step picks. Without it a raw argmax is not the
reference implementation's step, and a container packed without the policy is
refused at run time rather than quietly answering something else.

### What runs where

| | on the NPU | on the host |
|---|---|---|
| audio → log-mel, 3001 FFTs of 400 points | | yes |
| conv1, conv2, GELU, the permute into 1500 rows | | yes |
| every GEMM: qkv, attention-out, ffn-up, ffn-down, cross-q, cross-K\|V | **yes** | |
| LayerNorm, GELU, softmax, attention | | yes |
| the tied-embedding logit projection and argmax | | yes |
| token ids → text, and the long-form merge | | yes |

Only GEMMs are on the array. The elementwise ops could be (see
[Where each operation runs](#where-each-operation-runs)) but these design sets
declare no eltwise streams, and a row-wise op of this width is measured faster
on the host. The two convolutions are host work and grow as d², which is what
dominates the runtime on `whisper-large-v3`.

### Long form

Audio longer than 30 s is transcribed in 30 s windows with a 5 s stride on
**each** side, so consecutive windows start 20 s apart, and the texts are merged
by the same longest-common-sequence over token ids that `transformers` uses
without timestamps. A clip shorter than 30 s is zero-padded, and one longer than
30 s is cut.

### Options

`--language` names the language the decoder is primed with; there is no
detection, and an assumed language is printed as `ASSUMED` on stderr and in
`/health`. A code the model's tokenizer has no token for is refused by name.
`--convert` ingests through ffmpeg. `--max-new N` caps generated tokens per
window, `--chunk-seconds` / `--stride-seconds` move the window schedule (a window
longer than the model's position table allows is refused), and `--json` prints
the OpenAI-shaped object with one segment per window.

`tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` are
**not** needed and are not committed: nothing in the packing path opens them.
If you clone the upstream repository wholesale you will have them anyway; the
commit list is the minimum, and the network fetch list matches it.

### The endpoint

```
POST /v1/audio/transcriptions        multipart/form-data: file, model,
                                     language, task, response_format
```

`json`, `verbose_json` and `text` are answered; word and segment timings are
refused with a 400 that says why (no timestamps are decoded), as are a
non-zero `temperature` (this build is greedy only) and a `prompt` (not
implemented, and accepting one while ignoring it would return a transcription of
the audio alone).

### The NPU's power mode

XRT exposes a performance mode, and it is **not** on by default:

```bash
xrt-smi validate                  # "Power Mode: default" -- no admin needed
sudo xrt-smi validate             # "Power Mode: performance"
```

On this device (RyzenAI-npu1, firmware 1.5.5.391, XRT 2.26.0) it makes no
measurable difference: `validate` reports **100.0 us latency in both modes** and
**33679 vs 33720 op/s throughput** — a 0.1% difference, inside the run-to-run
noise of the benchmark itself. So it is not where the time goes on this box, and
chasing it before the host-side passes is the wrong order. It is documented here
because "the NPU is in default power mode" is a real answer to "why is this
slow", and because setting it requires `sudo`, which no run in this repository
does on your behalf.

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
| `--npu-extra-ops CODES` | also build the named elementwise designs |
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

**Symptom.** With the LayerNorm design loaded (`--npu-ops layn`), the LayerNorm
result was not reproducible and
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

### Containers were packed in the other board's byte order — fixed

**Symptom.** On npu1, every GEMM was numerically wrong, and nothing said so.
The byte count was right, the shapes were right, and the layout hash matched on
both sides of the only check that exists for this — because the design exporter
and the model packer derived it from the same constant. The embeddings were
unit-length, stable, and wrong: measured against numpy, the operand was read
with the columns of every 64×32 panel permuted (`1 - cos` ≈ 0.87, where the gate
is 2e-3).

**Root cause.** The B panel's byte order inside a tile *is* the MMAC sub-tile,
and the sub-tile is not the same on both generations. Measured with
`aie.iron.kernels.mm(...).mac_dims`, which returns `(r, s, t)`:

| device | `mac_dims` | B panel order |
|---|---|---|
| `npu2` (aie2p) | `(8, 8, 8)` | `(s=8, t=8)` |
| `npu1` (aie2) | `(4, 8, 4)` | `(s=8, t=4)` |

`npue.gemm_b_layout` defaulted to `(8, 8)` — correct for npu2 — and the packers
passed that default straight through, so every container was written in npu2's
order. The comment above the constant recorded the (correct) observation that
plain bf16 and bfp16-emulated agree, which is why it was never questioned: the
datapath does not change the sub-tile, and the *board* does.

**Fix.** The sub-tile is resolved from the target device in one place
(`npue.MAC_BY_DEVICE`, mirrored in `exporters/common/consts.py` and
`npue_pack.cpp`), threaded into every operand by both packers
(`pack_npue.py --device`, `packers/whisper.py`) and taken by every C++
`prepare_model*`, and printed in every build log. The exporter picks the pair
from `--arch`, so `design.json`'s `b_layout_hash` now means what it says: a
container packed for the other generation hashes differently and the runtime's
existing check refuses it.

**Verification.** `tools/verify_design_numerics.py`, added for this: it feeds
random matrices through the exported instruction streams and compares C with
numpy. Before the fix, 37 of 37 whisper and MiniLM streams failed at
`1 - cos` ≈ 0.87; after it, all 37 pass at `≤ 8.4e-06`, and the same check with
the real containers staged verbatim (`--npue ... --tensor ...`) passes too.
`verify_pack_parity.py --device` now compares the two packers for **both**
generations, 78/78 tensors byte-identical each, and the npu2 output is
bit-identical to what shipped before the change.

**Consequence for artifacts.** Both halves of the pair are derived, not stored:
the containers are gitignored and the design sets are gitignored. Both were
regenerated for npu1 here, but anyone building from source must pack **and**
export for the generation they intend to run on, or the runtime will now refuse
the pair it is given — loudly, which is the point.

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
  also why the elementwise ops default to the host, and why `--npu-ops` can
  *lower* throughput rather than raise it.
- **Batch tiers are right-sized, not padded.** A request is split into the
  largest tier that fits; the design set carries an instruction stream per tier.
- **`--npu-ops` costs one `hw_context` per named op** on top of the unified
  one, so all three is four, against a
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

**`--npu-ops <code> asks for it on the array, but <dir>/design.json does not
exist`** — that sibling design set was not built. Re-run the exporter with
`--npu-extra-ops <code>`, or drop the code from the list.

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
  include/whisper/         Whisper: geometry, the NPU encoder and decoder
                           stacks, the audio front end, the transcription
                           session
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
3. **Whisper (STT) design sets for the sizes that have none.** The `stt` kind
   in `npu_targets.json` exports `gemm_rtp` and `gemm_rtp_dec`, and whisper-tiny
   has both. Nothing else does: `transcribe` on whisper-base or larger is
   refused by name until its two sets are exported, and that refusal is the
   right one — a Whisper that transcribes from the encoder stack alone is not a
   degraded mode, it is nothing.

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
