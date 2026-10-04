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
- [What to move to the NPU](#what-to-move-to-the-npu)
- [Command-line reference](#command-line-reference)
- [Speech to text](#speech-to-text)
- [Image classification](#image-classification)
- [Body pose](#body-pose)
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
python tools/export/export_gemm_rtp.py --target all-MiniLM-L6-v2 --arch 1 --out runtime
```

This writes `runtime/artifacts/all-MiniLM-L6-v2/artifacts_npu1/gemm_rtp/` **and
every other design the target can honour** — `gelu/`, `layernorm/`, `softmax/`
for a BERT-family text embedder, one command, no flag. The exporter prints the
list it chose and, for each code it did *not* build, the reason. There is no
op-selection flag on the exporter at all: `--npu-ops`, `--npu-extra-ops` and
`--npu-eltwise` are refused there by name, each with its own message. Which
designs an architecture can honour is a table, not a flag:
[**NPU_OPS.md**](NPU_OPS.md) — all 40 code × architecture cells with a status and
a reason, generated from `tools/lib/npu_ops.py`, the same registry the exporter
reads.

The build is deliberately not free: each sibling design is a compile, an xclbin
and one more `hw_context` at run time, so the set is built once per target and
shared. See [Where each operation runs](#where-each-operation-runs) for why you
probably do not want to *ask* for any of them at run time.

Preview what would be built, without invoking the toolchain:

```bash
python tools/export/export_gemm_rtp.py --target all-MiniLM-L6-v2 --arch 1 --out runtime --dry-run
```

### 2. Build the runtime

```bash
cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build -j"$(nproc)"
```

Produces `runtime/build/npuembeddings` — one executable, one name.

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

| Operation | Default | On the array when | Costs |
|---|---|---|---|
| QKV / attention-out / FFN-up / FFN-down / cross-Q / cross-K\|V GEMM | array | always | — |
| conv1, conv2 (Whisper) | host | `--npu-ops conv` | no context: the encoder set's own `[rows, d, d]` stream |
| attention, as QK^T and softmax·V GEMMs | host | `--npu-ops attn` | no context: two more streams in the same sets |
| the mel filter bank, as a GEMM | host | `--npu-ops mproj` | no context: one more stream in the encoder set |
| the 400-point transform, as a GEMM | host | `--npu-ops fft` | no context: one more stream in the encoder set |
| the tied-embedding logit projection | host | `--npu-ops logit` | no context: eight chunk streams in the decoder set; 39 MB of staged panels |
| LayerNorm | host | `--npu-ops layn` | one `hw_context` (`layernorm/`) |
| softmax | host | `--npu-ops softm` | one `hw_context` (`softmax/`) |
| GELU | host | `--npu-ops gelu` | one `hw_context` (`gelu/`) — exact erf for Whisper, the degree-8 `poly` fit (2.49e-3 relative) for everything else; see the caveat below |

The five GEMM-shaped codes (`conv`, `attn`, `mproj`, `fft`, `logit`) need no
sibling design set: each one is a stream inside a set that already exists, so
they cost no `hw_context` and the runtime refuses a set that does not carry
them **by name** rather than answering from the host. The three elementwise
ones are whole xclbins of their own and cost one context each, so all three is
five contexts of the six npu1 allows.

### Which architecture honours which code

**One flag, eight codes, one parser — and the set a given architecture can
honour is a property of the MODEL, not of the flag.** There are no
architecture-specific NPU flags anywhere in the tree; the GEMMs are on the array
everywhere and unconditionally, with no flag, because there is no GEMM code.

All 40 cells (5 architectures × 8 codes) were measured by running every code
against every container on this machine with a real invocation of each mode, not
read off the source. They now live in a registry — `tools/lib/npu_ops.py` — which
the exporter reads to decide what to build, and
[**NPU_OPS.md**](NPU_OPS.md) is generated from it with a reason per cell.
`tools/verify/verify_npu_op_matrix.py` runs all 40 cells against the binary and
fails if the runtime and the table disagree — every architecture row has a
fixture now, pose included, so nothing is skipped for want of a container.
**One flat table would hide the thing worth knowing: the empty cells are four
different situations, and only one of them is permanent.**

| | BERT | gemma-300m | whisper | ViT | pose |
|---|---|---|---|---|---|
| `gelu` | ✓ | △ | ✓ | ✓ | — |
| `layn` | ✓ | ◇ | ✓ | ✓ | — |
| `softm` | ✓ | ✓ | ✓ | ✓ | — |
| `conv` | — | — | ✓ | ⊕ | ✓ |
| `attn` | ✓ | ✓ | ✓ | ✓ | — |
| `mproj` | — | — | ✓ | — | — |
| `fft` | — | — | ✓ | — | — |
| `logit` | — | — | ✓ | ◇ | — |
| **counts** | 4 ✓ | 2 ✓ | 8 ✓ | 5 ✓ | 1 ✓ |

* **✓ — runs on the array now** (19 cells).
* **⊕ — already on the array, with nothing for the code to select** (1 cell:
  `ViT conv`), explained below.
* **— — the model has no such operation** (17 cells). Permanent, and correct.
* **△ ◇ — the model has the operation and this board cannot compute it** (3
  cells; `◇` takes two of them), explained per symbol below.
  19 + 1 + 17 + 3 = 40.

There is no longer an "unimplemented" symbol in this table, and that is the
change worth noticing: the five cells that used to read **▢** — `attn` for BERT,
gemma and ViT, `softm` for gemma and ViT — are ✓ now, each with a measured reason
in NPU_OPS.md. The counts row is codes on the array (19 honoured + 1 already
there = 20), not a count of ticks in one column.

The 17 permanent cells, by reason:

| cells | why |
|---|---|
| `conv`, `mproj`, `fft` on BERT, gemma (6); `mproj`, `fft` on ViT (2) | these name Whisper's **audio front end**, its **mel filter bank** and its **400-point transform**. A text or image encoder has none of them. |
| `logit` on BERT, gemma (2) | `logit` names Whisper's **tied token embedding used as the logit matrix** — `decoder.cpp:196`, "the checkpoint has no proj_out: the logit matrix is the tied token". An embedder stops at its pooling head and has no such tensor: `bert_encoder.cpp` and `gemma_npu_encoder.cpp` contain **zero** occurrences of `logit` or `vocab`. |
| all 7 non-`conv` codes on pose (7) | `runtime/include/pose/net.hpp` exposes exactly one weighted op, `conv(..., bool silu, ...)`, plus `concat/slice/add/maxpool/upsample/head`. Its SiLU is fused into the convolution epilogue. There is **no normalization op at all**: `tools/pack/packers/pose.py:91` is `GRAPH_OPS = {Conv, Mul, Sigmoid, Add, Concat, Split, MaxPool, Resize}` and `BatchNormalization` is absent from it, so a graph carrying BN would be *refused by name* — and `grep BatchNormalization` over `tools/` and `docs/` returns nothing, because ultralytics folds BN into the conv weights at export. No attention, no vocabulary. Nothing for a code to name. |

The four special cells:

**⊕ `ViT conv` — the model HAS a convolution, and it is already on the array.**
ViT's patch embedding is `Conv2d(3, 768, kernel=16, stride=16)`, im2col'd to
`[197, 768] @ [768, 768]`. That K and N are exactly `attn_out`'s shape, so it is
dispatched on `attn_out`'s instruction slot — unconditionally, with no flag, as
one of the 49 GEMMs (`runtime/src/vit/encoder.cpp:267`,
`g_.run(streams_.attn_out, ...)`). The cell carries ⊕ rather than a tick because
**the work is already done**: `conv` names Whisper's `conv1`/`conv2` streams,
which a ViT's design set does not carry, and honouring it would dispatch nothing
new. This is the one cell that is better as it stands than as a tick — and asking
for it is **refused**, with exactly that as the reason, because a code that
selects nothing is the failure this project treats as worst.

**⚠ Caveat on `gelu`, `ViT` and `gemm_rtp` rows — the activation is the `poly`
fit, not the exact erf.** Only `kind: stt` gets the exact-erf kernel; a BERT or a
ViT keeps the exporter's degree-8 `poly` default, which is **2.49e-3 relative
from the erf the host computes**. Measured end to end on bge-base with `--npu-ops
gelu`, relfro 6.1e-03, and on ViT under 0.011 in label confidence — small enough
that the runtime accepts the code rather than refusing it, and recorded in both
cells' reasons in NPU_OPS.md. It is not fixed here because it cannot be: fixing it
means compiling a second kernel variant, which needs the toolchain this machine
cannot run.

**△ `gemma gelu` — the activation is fused, so there is no pass to move.**
gemma's FFN is gated (GeGLU); the activation is computed *inside* the gated path
while the two halves are being multiplied, rather than as a separate pass over a
finished tensor. `gelu`'s whole design is one `hw_context` doing a standalone
elementwise pass, so honouring it would print "on the ARRAY" while moving
nothing. The exporter refuses for the same reason, and now by REGISTRY rather
than by a special case: `gelu`'s cell for gemma says `blocked`, and a gated
FFN's activation is part of the gated path between `ffn_up` and `ffn_down` rather
than a separate pass (nomic and gte drop `gelu` for the same structural reason,
and their rows say so).

**◇ `gemma layn` — would need a NINTH code, and is deliberately not done.**
gemma uses **RMSNorm**, not LayerNorm: no mean pass, no beta, its own
`rms_norm_eps`. `kernels/layernorm.cc` is parameterised only by
`-DLN_COLS/-DLN_EPS/-DLN_ROWS`, so RMSNorm is a different kernel body, a new
design kind and a new code. That would make the code set *larger* and *less*
uniform — three of the four transformer architectures would have a norm code and
one would not — so `layn` stays refused there rather than quietly meaning two
different operations.

**◇ `ViT logit` — the model has it, the array cannot tile it.**
The refusal message says it outright: the head is `[768, 1000]` and
**1000 is not a multiple of `tile_n × cols = 48 × 4 = 192`**, so no legal B
panel of that width exists on this array — it is not missing, it does not exist
on this board. The same numbers are in `runtime/include/vit/encoder.hpp:37` and
the header of `tools/pack/packers/vit.py`. (The head is also 0.8% of the image
cost as a host matvec, so the point is moot for speed either way.)

#### The 5 cells that were unimplemented work — done, and measured

All five of them computed the operation on the host with no host/array choice,
and the gap was never a missing kernel: it was a missing branch in a host path
this repository already shared between architectures.

```cpp
// runtime/src/encoders/bert_encoder.cpp, BertEncoder::run — the old shape
qk(qkvbuf, scores);                          // host, no choice
if (host_sm) softmax_cpu(scores);
else eltwise(softmax_, slots_sm, ...);       // softm was already array-capable
av(scores, qkvbuf, ctx);                     // host, no choice
```

That one `if` was exactly why `softm` worked for BERT while `attn` did not:
softmax is a standalone pass over a finished tensor and needed only the eltwise
design, whereas the two GEMMs that bracket it had **no array branch at all**.
`gemma_npu_encoder.cpp` was `attention(qkvbuf, ctx)` with no alternative either,
and ViT does not even have its own attention — `vit/encoder.cpp` calls
**`npue::whisper::attention(qkv_all.data(), ...)`**, the same shared function,
the same signature. The host path was already literally shared between two
architectures; what was missing everywhere was only the branch inside it.

That branch now exists in one place — `NpuAttention`, reached from
`BertEncoder::attention_npu()`, from gemma's `GemmaNpuEncoder::attention()` and
from ViT through the shared `whisper::attention()` — and the exporter's
`if model_spec.get("kind") == "stt":` gate in front of `ns.attn_streams` is
replaced by a consult of this same registry, so the streams are built for any
kind that honours `attn`. `--npu-ops attn` moves QK^T and scores·V to the array
and leaves the softmax wherever `softm` says; `--npu-ops softm` moves the
softmax alone. **Three per-model details the array branch had to be taught, all
still true and all spelled out in the per-cell reasons:**

* BERT's and ViT's additive **padding mask** (Whisper's encoder has none) rides
  the call — one mask row per sequence, added to the scores before the softmax.
* gemma's **RoPE** is applied to the qkv buffer before `attention()` runs, so
  the array's A operand is already post-RoPE and nothing has to move it.
* gemma's **MQA** forces K|V width and head count to be separate parameters
  from `d_model`, and its **sliding window** is a band rather than a suffix, so
  `n_kv` is the window (512, padded to 576) and a longer sequence is
  **refused** rather than silently computed as full attention.

**The geometry caveat is resolved.** `n_kv` is no longer read as
`model_spec.get("frames", 1500)`; it comes from the container's own geometry,
which is why BERT exports `attn_qk` for its 256 positions (padded to 384), ViT
for its fixed 197 (also 384) and gemma for its 512 window rather than the 2048
the container is packed to. `kinds.cls` carries `attn_qk`/`attn_av` now. All
three sets exported and ran, so none of that paragraph is a prediction any
more.

**The speed caveat is confirmed rather than refuted — every one of these cells
is slower than the host it replaces:**

| container | code | host | array | ratio |
|---|---|---|---|---|
| bge-base, 15 texts | `attn` | 0.16 s | 1.38 s | 8.6× slower |
| bge-base, 15 texts | `softm` | 0.16 s | 1.70 s | 10.6× slower |
| ViT, bus.jpg encoder | `attn` | 0.244 s | 0.349 s | 1.43× slower |
| ViT, bus.jpg encoder | `softm` | 0.244 s | 0.822 s | 3.4× slower |
| gemma-300m, attention column | `attn` | 21 ms | 358 ms | 17× slower |
| gemma-300m, attention column | `softm` | 21 ms | 1497 ms | 71× slower |
| whisper, 3 s window | `attn` | 0.94 s | 4.32 s | 4.6× slower |

And the one combination nobody would guess: `attn,softm` together is
**100.90 s** on bge-base against 1.38 s for `attn` alone, **6.685 s** on ViT
against 0.349 s, and **18.7 s** on gemma against 0.358 s. The cause is one line
long — `NpuAttention` calls the softmax once per head per chunk while every
call fills the softmax design's whole 12288-row capacity — and it is why that
combination is printed as a measurement rather than offered as a default.

Correctness was held against the host on the **whole tensor**, not on the label
that would have agreed anyway: bge-base `attn` relfro 1.58e-02 / cos 0.999876
and `softm` 1.15e-02 / 0.999933; ViT's full 1000-way probability row relfro
4.384e-03 with `attn` and 1.959e-02 with `softm`, top-5 and label identical;
gemma relfro 5.735e-03 / 6.227e-03. One trap that measurement caught rather
than review: the softmax design is compiled for the **padded** key count,
because that is the row width `NpuAttention` hands it, while BERT's own host
`qk_impl` wrote rows `seq` wide — and `elt_chunks`' only guard (`n % cols != 0`)
happens to pass anyway. `--npu-ops softm` alone on bge-base came back at relfro
**7.08e-01** for exactly that reason and is fixed by laying the rows out at the
design's width with `-1e30` past the live ones. The default path with no
`--npu-ops` at all is byte-identical to what it was before any of this existed —
checked by comparing the output files, not by inspection.

**Which of them is worth asking for is a measurement, not a preference** — see
[What to move to the NPU](#what-to-move-to-the-npu).

One flag, and an op is on the host exactly when it is **not** listed — so
`--npu-ops layn,softm` is "LayerNorm and softmax on the array, GELU on the
host", and there is no inverse flag to drift against it. This replaces
`--npu-extra-ops` (the longer spelling of this same flag), `--npu-eltwise` (all
three or none) and `--host-ln/--host-sm/--host-gelu`; the old names are now
**refused by name**, so a stale command line cannot look like it worked.

`--npu-ops` only SELECTS among designs that were already built. Each named op
needs its sibling design set next to `gemm_rtp/` (`layernorm/`, `softmax/`,
`gelu/`), and the exporter builds all of them with no flag, so the command you
already have for the GEMM set is the whole command. A missing one is **refused by
name** — it never silently falls back to the host, because a flag whose whole
point is "put this on the array" must not quietly not do that. Each op also
costs one more `hw_context` out of six.

Which codes a given architecture honours is a table with a reason per cell, not
a flag to guess at: [**NPU_OPS.md**](NPU_OPS.md). Of the 40 cells, 19 run on the
array today, 1 is already dispatched without a code (and the code is refused,
because there is nothing for it to select), 3 cannot be moved on this board for a
stated reason, and 17 do not exist in that model at all.

That table is per ARCHITECTURE. For the question you actually have — what can
*this* model name do — there is a second generated table,
[**NPU_MODELS.md**](NPU_MODELS.md): 16 models × 8 codes, with a column for what
the exporter compiles for each one and the geometry it takes. It differs from the
first in exactly two places, and both are model facts: `gemma`'s encoder, and the
**gated FFN** of `nomic` and `gte`, which has no standalone activation pass, so
their `gelu` is not built even though their kind honours it.

`conv` is one of the five codes with no sibling design set, and that is
deliberate: it is a speech-to-text op (Whisper's two audio convolutions) which
runs on the encoder set's own `[rows, d, d]` stream, so there is nothing to build
and an exporter asked for it refuses the code by name. On an embedder it is
refused too.

It also costs four `hw_context`s instead of one. Before allocating any, the
runtime asks `xrt-smi` how many contexts the device allows and refuses by name
if the total would not fit. `--allow-contention` overrides; a throughput number
from a contended run is not an NPU performance claim.

---

## What to move to the NPU

Three different questions have three different answers, and they are not the
same set of operations. Everything below is **measured on this machine**, on
`whisper-tiny` (d=384, 4+4 layers, 6 heads), a 3 s 440 Hz tone, 16 host
threads, the same binary, ~0.35 s of which is setup. `CPU` is the summed user
time of all threads, which is the number that matters when the CPU is the scarce
resource; `wall` is what a caller waits.

| `--npu-ops` | wall, s | CPU, s | encoder, s | decoder, s | dispatches |
|---|---:|---:|---:|---:|---:|
| *(nothing — everything on the host)* | 0.94 | 5.54 | 0.28 | 0.07 | 192 |
| `conv` | **0.66** | **3.81** | 0.30 | 0.07 | 207 |
| `conv,mproj,fft` | **0.65** | **3.49** | 0.30 | 0.07 | 219 |
| `conv,mproj,logit` | 0.77 | 3.94 | 0.28 | 0.07 | 253 |
| `layn` | 1.05 | 5.57 | 0.36 | 0.28 | 284 |
| `gelu` | 1.22 | 5.04 | 0.43 | 0.33 | 224 |
| `conv,mproj,layn,gelu,logit` | 1.56 | 3.62 | 0.52 | 0.53 | 377 |
| `conv,attn,softm` | 4.32 | 2.46 | 1.01 | 2.88 | 1143 |
| `conv,attn,mproj,fft,logit,layn,softm,gelu` | 4.90 | **1.05** | 1.21 | 3.35 | 1319 |

Run-to-run spread on this box is a few percent on wall and up to 10% on CPU, so
read the ratios, not the last digit.

### The same question asked of a ViT

The table above is Whisper's, because Whisper is where all eight codes are
reachable. For a classifier only two of them are, and the answer for those two
is the same one the Whisper table already gives — **the work per dispatch has to
beat the dispatch**, and a ViT's elementwise passes are too small for it.

Measured on `vit-base-patch16-224.npue` bf16, `bus.jpg` (810×1080), 16 host
threads, best of 8 after a warm-up run. Wall is front end + encoder, which is
the whole per-image cost minus the 0.9 ms head:

| `--npu-ops` | wall, s | vs host | label | p | elt dispatches |
|---|---:|---:|---|---:|---:|
| *(nothing)* | **0.248** | 1.00× | minibus | 0.629 | 0 |
| `gelu` | 0.343 | 1.38× slower | minibus | 0.635 | 12 |
| `layn` | 0.377 | 1.52× slower | minibus | 0.631 | 25 |
| `layn,gelu` | 0.449 | 1.81× slower | minibus | 0.640 | 37 |

The **labels agree in all four rows** and the confidences agree to within 0.011
absolute — that spread is the bf16 datapath, not the schedule, and it is the
same order as the int8 row in the table above. So the flag is honest: it puts
the work where it was asked to and returns the same answer.

The dispatch counts are **measured per image**, not derived from the model's
layer count: 25 LayerNorm sites (2 per layer plus the final one) and 12 GELU
blocks, and they are reported as `elt_dispatches` in the `classify --json`
object — a separate field from `dispatches`, which is the 49 GEMMs and stays 49.

So the honest summary for a classifier is the same as for Whisper's `layn` and
`gelu`: **both codes work, both are slower, and the default stays on the host.**
They are implemented because the flag is then uniform across the architectures
that have a choice to make — not because they are worth asking for.

### 1. Maximum performance: `conv`, and only `conv`

`conv` is the one op that wins on **both** axes, and the reason is structural:
its work is d² MACs, so the host's cost grows with the square of the width while
the array's grows with the dispatch count, which is constant. Per 30 s window on
16 host workers: 0.11 s → 0.02 s at d=384, 1.37 s → 0.07 s at d=1280. The
measured 0.94 → 0.66 s wall and 5.54 → 3.81 s CPU on whisper-tiny is that same
ratio on a short window, and it gets **better** with width, not worse.

The GEMMs are already there — they are the model, and they are what the array is
for. Everything else is host work whose arithmetic is small next to its
dispatch count.

### 2. Balance: `conv,mproj,logit` — or just `conv`

`mproj` is 48M MAC per window, ~5 ms of host, and 6 dispatches: free at this
width, and it removes the front end's largest host loop. `logit` is neutral on
wall (the encoder and decoder times do not move) because the host matvec and the
eight dispatches cost about the same — it buys 39 MB of staged bf16 panels and
back, which is worth it only when the CPU is the bottleneck.

`layn` and `gelu` are the counter-example, and the reason is the same in both:
**the work per dispatch has to beat the dispatch**. A `layn` dispatch covers 512
rows of 384 columns and the host does the same 196k elements in 0.08 s across 16
threads — the array's per-dispatch cost and its bf16 round trip are the same
order. Worse in the decoder, where one LayerNorm is **one row**: 0.07 → 0.28 s,
four times worse, because 511 of the 512 rows are padding.

### 3. Maximum CPU saving at acceptable performance: everything

All eight codes: **1.05 s of CPU against 5.54 s** — 5.3x less — for 4.90 s of
wall against 0.94 s. That is the honest trade: the CPU stops doing the
conversions, the fp64 work and the panel tiling, and the array does the
dispatches instead.

The middle point is `conv,mproj,layn,gelu,logit`: 1.56 s wall (1.7x slower) for
3.62 s CPU (35% less), which is the set to pick when the transcript may be 60%
later and a core has to be freed for something else. `conv,attn,softm` is the
opposite end: it costs 4.6x the wall time and saves 56% of the CPU.

### What attention costs, and why

Attention is the largest piece of arithmetic in the model (184 GFLOP per
window) and the **worst** thing to move at this width: 0.28 → 1.01 s for the
encoder and 0.07 → 2.88 s for the decoder. The dispatch count is the obvious
part (heads × chunks × 2 × layers, 144 per layer for whisper-tiny) but it is not
the whole cost. The B operand of a score is an **activation** — the K and V
rows of the fused qkv tensor — not a weight, so it cannot be staged once per
session: it is re-tiled from scratch for every head of every layer, 196k
elements of `tile_b_panel` each, and that host loop is what the measurement is
mostly made of. A GEMM whose B is a weight is a one-off stage; a GEMM whose B
is an activation pays for the tiling on every layer.

The FFT is the other end of the same trade, and the only one that is not merely
slower: `dft400` is the direct transform as a GEMM against a precomputed
bf16 twiddle matrix, so it has a **noise floor** of about (bf16 eps)² times the
loudest bin. The loud bins agree with the host to 0.2%, and the quiet mel
channels — the ones the floor would otherwise clamp — read 0.3 where the host
reads 3e-5. It is on the array because it can be, and off by default because
that is a different spectrogram; fixing it is an fp32 radix-400 kernel, which
is research rather than wiring.

### The rule this gives

1. Measure the **work per dispatch** against the ~150 µs fixed cost. An op
   whose dispatch carries less arithmetic than that is host work.
2. Count the **B operand**. A weight stages once; an activation is re-tiled per
   layer, and the tiling is on the host no matter where the multiply happens.
3. Check the **padding**. A design computes all of its rows; a row-wise op
   asked to do one row of work pays for the whole buffer in both directions.
4. Look at the **host's own parallelism** before calling a host loop slow. 16
   AVX2 threads beat one AIE column on anything but a big GEMM.

---

## Command-line reference

`npuembeddings --help` is authoritative and kept in sync with this section.

### Subcommands

| Command | What it does |
|---|---|
| `list` | every model this build can run, and which are installed |
| `serve <model>` | the HTTP endpoint for whatever the container is; downloads the model if needed |
| `embed <model> <in.txt> [out.f32]` | embed a text file, one text per line |
| `transcribe <model> <audio.wav>` | transcribe 16 kHz audio with a Whisper model; the transcript alone on stdout |
| `classify <model> <image>` | image classification (arch=5); `--top-k` prints the runners-up to stderr |
| `pose <model> <image...>` | body pose (arch=6); MediaPipe-shaped JSON on stdout, `--text` for a human summary |
| `add <org/model> [<sha256>]` | register a model this build does not know (a finetune) |
| `tokenize` | tokenizer round-trip, for debugging |

### One verb, four endpoints

`serve` is the only verb that opens a socket, and the container's `arch` picks
which endpoint answers:

| arch | model | endpoint |
|---|---|---|
| 1, 2, 3 | BERT-family text | `POST /v1/embeddings` |
| 4 | Whisper | `POST /v1/audio/transcriptions` |
| 5 | ViT classifier | `POST /v1/classify` |
| 6 | YOLOv8-pose | `POST /v1/pose` |

The modes are dispatched on `arch` in `runtime/src/runtime.cpp` before `--serve`
is ever read, so no two of them can meet and no flag chooses between them. A path
that belongs to another architecture is a 404 that names the right one — never
another endpoint's answer. A code the mode cannot honour -- `--npu-ops conv` on a
classifier, `--npu-ops layn` on gemma -- is **refused by name**, never ignored and
never quietly run on the host; the reason is in
[NPU_OPS.md](NPU_OPS.md) and the refusal repeats it.

This was not always one verb: pose arrived with a `pose-server` subcommand and
the classifier refused `--serve` outright with "there is no endpoint for it". The
refusal was honest and it was also an inconsistency — `serve` is the verb, so an
architecture that refuses it is one a user discovers by typing.

### Options for `serve` / `embed`

| Flag | Meaning |
|---|---|
| `--port N` | listen port (default 8080) |
| `--bind ADDR` | interface (default `127.0.0.1`, localhost only) |
| `--threads N` | host thread budget (`serve`/`embed` pass 24) |
| `--pipeline N` | concurrent encode lanes (`serve`/`embed` pass 4) |
| `--artifacts DIR` | override the design set |
| `--npu-ops CODES` | send the named ops to the array: `gelu`, `layn`, `softm`, and `conv` (Whisper's conv1/conv2, a `transcribe`-only code). The exporter's flag is the same string and builds them |
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

**Only the ONNX export.** Everything else a Whisper container is built from
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

So the flow is: download the export, drop it into `models/<name>/onnx/` — the
graph, plus any side file it names, under that file's own basename — and run
the three commands below.

### Three steps, once per model

```bash
# 1. the container. --max-seq is NOT needed and should not be passed: the
#    default is per family, and for whisper it is the checkpoint's own
#    max_source_positions (1500). The embedder default (256) writes a container
#    whose position table is shorter than one audio window, and the packer
#    refuses that by name rather than writing it.
python tools/pack/pack_npue.py --model-dir models/whisper-base \
    --out models/whisper-base.npue --device npu1

# 2. the design sets -- BOTH, and one invocation writes both, because a Whisper
#    target resolves to an encoder pass and a decoder pass (their M differs by
#    8x, so one xclbin cannot serve both).
python tools/export/export_gemm_rtp.py --target whisper-base --arch 1 --out runtime

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
| the 3001 transforms of 400 points | `--npu-ops fft` | yes (the default, fp64) |
| the slaney mel bank | `--npu-ops mproj` | yes (the default) |
| conv1, conv2 | `--npu-ops conv` | yes (the default) |
| the GELU between them, and the permute into 1500 rows | `--npu-ops gelu` (that one only) | yes (the default) |
| every GEMM: qkv, attention-out, ffn-up, ffn-down, cross-q, cross-K\|V | **yes** | |
| attention: QK^T, softmax, softmax·V | `--npu-ops attn,softm` | yes (the default) |
| LayerNorm (both stacks) | `--npu-ops layn` | yes (the default) |
| GELU in the FFN | `--npu-ops gelu` | yes (the default) |
| the tied-embedding logit projection | `--npu-ops logit` | yes (the default) |
| argmax and the two suppression lists | | yes, always |
| token ids → text, and the long-form merge | | yes, always |

Two rows are unconditional and stay that way: the argmax is a search over a
vector and the suppression policy is a table walk, and **decode + merge** is
BPE longest-common-sequence over ids. None of them is a matrix, so no design
expresses them; that is not a scheduling choice left open.

The two convolutions are host work by default and grow as d², which is what
dominates the runtime on `whisper-large-v3` (2.8 s of a 19 s window on 16 host
workers). `--npu-ops conv` puts them on the array: a 3-tap convolution is a
GEMM over an im2col'ed `(input channel, tap)` K axis, so conv1 rides in one
dispatch of the encoder set's own `[rows, d, d]` stream (attn_out's shape) and
conv2 in three, accumulated in fp32 on the host. It needs **no new design set** —
the same instruction stream the encoder's attention-out projection uses — and it
costs no extra `hw_context`. Measured per 30 s window, 16 host workers:

| model | host conv | `--npu-ops conv` | dispatches added |
|---|---|---|---|
| whisper-tiny (d=384) | 0.11 s | 0.02 s | 15 |
| whisper-large-v3-turbo (d=1280) | 1.37 s | 0.07 s | 15 |
| whisper-large-v3 (d=1280, 128 mel) | 1.42 s | 0.15 s | 15 |

The output is bf16 rather than fp32, which is the design's C precision: the
tensor matches `transformers` to 1-cos 1.5e-6 and 1.4e-2 max-abs at whisper-tiny
(`tools/verify/verify_whisper_model.py` checks both, the array path at its own max-abs
tolerance), and the encoder fed from it still matches at the same 1-cos as the
host-fed one. A design set with no `[rows, d, d]` stream cannot run them, and
asking for it there is **refused by name** rather than answered from the host.

Everything else that can be a matrix now can be one: attention, the mel bank,
the transform and the logit projection each got a stream, and the elementwise
ops got their own designs. Whether any of them is **worth** asking for is a
different question, and it has a measured answer:
[What to move to the NPU](#what-to-move-to-the-npu).

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

## Image classification

A `google/vit-base-patch16-224` container is an **image classifier**, not an
embedder: it answers "label for this picture", it has no pooling mode a caller
can ask for, and it runs through a mode of its own. `npuembeddings list` shows
that row as `cls`.

It is the same ViT that any embedding ViT would be — a 224x224 image becomes
196 patches of 16x16 px plus one `[CLS]` token, so 197 tokens of width 768,
through 12 pre-LayerNorm transformer layers. What makes it a classifier is the
last step: the `[CLS]` vector is multiplied by a stored `[768, 1000]` matrix and
the argmax is a class index. This build ships **only** that; the 768-number
embedding is not exposed, and adding it is a head change, not a new design set.

The 1000 classes are ImageNet-1k and nothing else. The model must answer with
one of them even for a picture that belongs to none, so on an out-of-distribution
input it returns a confident wrong label rather than a refusal. Every accuracy
claim below is therefore about ImageNet-distribution images.

### What a user has to download

**Only the ONNX export.** It goes at
`models/vit-base-patch16-224/onnx/model.onnx` (plus a side file if the graph
names one, beside it under that basename). One small file is committed:
`preprocessor_config.json`, because its `image_size`, resample code and mean/std
*are* the front end — the packer refuses a checkpoint without it, and no other
file in the tree carries those numbers, so a wrong mean yields a container that
classifies smoothly and wrongly. 160 bytes, identical for everyone. Geometry is
deliberately **not** committed the way whisper's `config.json` is: an embedder's
geometry has one owner, the checkpoint, which is re-read at pack time and pinned
by `CHECKPOINT.json`.

### Four commands, and the third one is the one people miss

```bash
# 1. the container. Do NOT pass --max-seq: it is REFUSED for this family by
#    name, because a ViT's position count is (224/16)^2 + 1 = 197, fixed by the
#    image size rather than by the caller.
python tools/pack/pack_npue.py --model-dir models/vit-base-patch16-224 \
    --out models/vit-base-patch16-224.npue --device npu1
#    add --int8 for the 90 MB container (from 175 MB)

# 2. the design set for the container you packed.
python tools/export/export_gemm_rtp.py --target vit-base-patch16-224 --arch 1

# 3. ONLY if you packed --int8: a SECOND design set, exported --int8.
python tools/export/export_gemm_rtp.py --target vit-base-patch16-224-i8 \
    --arch 1 --int8 --out runtime

# 4. classify.
./runtime/build/npuembeddings classify vit-base-patch16-224 photo.png
```

**An int8 container needs its own design set, and this is not a detail.** The
two containers have *identical* GEMM shapes and identical tiling, but not
identical `mac_s`/`mac_t`: on npu1 the int8 MMAC's sub-tile is `(8, 8)` where
bf16's is `(8, 4)`, so the int8 layout reads `tile_k 64, tile_n 48, mac_s 8,
mac_t 8`. `b_layout_hash` covers both the dtype and the sub-tile, so the int8
container asks for hash `177088d6...` and the bf16 one for `52a4adad...`, and a
design set serves exactly one of them. Point an int8 container at the bf16 set
and the run refuses by name rather than staging mismatched operands — see
`design_selection.hpp`, tasks/0080.

There is no `mv` step. The exporter writes
`runtime/artifacts/<model>/artifacts_npu<arch>/`, the runtime's by-name candidate
list proposes exactly that path, and the two disagree nowhere — see
`design_selection.hpp` and `tools/lib/design_sets.py`, which state the layout
once each on the reading and finding side. (There used to be a `mv` here, and a
paragraph explaining it: the exporter appended `<model>/artifacts_npu<arch>` to
`--out`, landing three levels deep, while the fallback scan looked two levels
down, so an int8 set had to be *moved* to `runtime/<model>-i8/gemm_rtp/` before
anything would find it. The int8 set is now a directory of its own —
`runtime/artifacts/<model>-i8/artifacts_npu<arch>/` — because one directory holds
one design set and the two datapaths hash differently.)

**`--max-seq` is refused for this family, by name.** It is refused rather than
quietly honoured because there is no value to honour — 197 comes from arithmetic
on the checkpoint's own `image_size` and `patch_size`, not from an argument —
and the refusal names that number instead of leaving you to work it out. It used
to be accepted and dropped, exiting 0 while doing nothing, which is the
fail-open shape this repository treats as a defect elsewhere. Whisper is the
family where the flag is genuinely dangerous — a table too short for one audio
window is refused there too, and the argument is wired through and checked
rather than discarded.

`--arch 1` is `npu1`. `--arch 2` targets a different board (8 AIE columns) and
cannot be built or run on a one-NPU machine.

### The classify endpoint

`npuembeddings serve vit-base-patch16-224` answers `POST /v1/classify`, and the
answer is **the same bytes** `classify --json` prints — both call
`npue::vit::prediction_json`.

```bash
./runtime/build/npuembeddings serve models/vit-base-patch16-224.npue --port 8080

curl -s -F image=@photo.jpg localhost:8080/v1/classify
# {"image": "photo.jpg", "label": 654, "name": "minibus", "p": 0.629018188,
#  "front_end_s": 0.020288158, "encoder_s": 0.252298374,
#  "head_s": 0.000773930999, "dispatches": 49}

curl -s -F image=@photo.jpg -F top_k=3 localhost:8080/v1/classify
# ... "top_k": [{"label": 654, "p": 0.629018189},
#               {"label": 874, "p": 0.177649605},
#               {"label": 829, "p": 0.079892135}] ...

curl -s localhost:8080/v1/labels     # the whole 1000-name vocabulary
```

`top_k` is the number of labels asked for, and it is the only per-request knob:
there is no per-request model selection and no batch, because one image is one
request. `top_k` above the vocabulary is **capped rather than refused** —
`top_k=5000` on a 1000-label model is a request for everything, not a mistake —
and `top_k` below 1 or not an integer is a 400, because answering "the top 0
labels" with an empty array looks like a model that recognised nothing.

`GET /v1/labels` returns all 1000 names and not a page of them: a partial list
would make some ids silently unnameable. It exists so a client can turn a `label`
into a `name` without shipping the container's `labels.table` itself.

One field is named for what it is rather than for what a reader wants:
`/health` reports `dispatches_so_far`, not `dispatches_per_image`. It is the
encoder's running counter, so it reads 0 on a freshly started server and 49 after
one request; calling it "per image" would report 0 for a model that in fact costs
49 dispatches.

### What runs where

| | on the NPU | on the host |
|---|---|---|
| patch embedding | yes — it rides `attn_out`'s stream | — |
| 12 x qkv / attn_out / ffn | yes, 49 dispatches | — |
| image front end | — | decode, PIL-equivalent resize to 224, `(x/255 - mean)/std`, im2col |
| LayerNorm (25 sites) | `--npu-ops layn` | fp32, with the container's own epsilon (the default) |
| GELU | `--npu-ops gelu` | yes (the default) |
| attention: QK^T, softmax, scores·V | `--npu-ops attn,softm` | the whole pass, `197x197 x 12 heads` (the default) |
| classifier head | — | 768x1000 matvec |

The patch embedding is free: its shape `[196, 768] x [768, 768]` **is**
`attn_out`'s shape, so it dispatches on that stream and costs no new design and
no new stream. The head is on the host because 1000 is not a multiple of
`tile_n * cols`, so no legal B panel of that width exists on this array — one
host matvec costs less than the ~150 us a dispatch would.

### Measured on npu1

RyzenAI-npu1, firmware 1.5.5.391, XRT 2.26.0. 25 samples per container over 5
images, alternating, warm-up discarded:

| container | staged on the array | median | mean | range | dispatches |
|---|---|---|---|---|---|
| bf16 | 166.8 MB | 0.280 s | 0.2748 s | 0.25-0.30 s | 49 |
| int8 | 85.2 MB | 0.190 s | 0.1864 s | 0.17-0.20 s | 49 |

The distributions do not overlap (worst int8 0.20 beats best bf16 0.25), so
**for this model the operand width does buy speed, not only memory** — 1.47x,
32% less wall time per image. That is the opposite of what the Whisper branch
of `pack_npue.py` says, and it is measured here rather than assumed: the design
is fixed at M=1024 while the model has 197 rows, so the array runs a full-size
design either way and i8 operands halve the bytes it has to move. The
host/NPU split inside those numbers was **not** measured; the M=1024 explanation
is the reason it is consistent, not a measurement of it.

#### The int8 row above was measured on a container the runtime computed wrongly

Worth stating plainly, because the table does not show it and the numbers are
otherwise fine: those int8 timings were taken while `VitEncoder::gemm1()` was
dividing each activation by `asmooth` where `NpuGemm::run_i8()` documents
`1/asmooth`. The product came out as `X @ W * asmooth²`, the logits collapsed
towards uniform, and the classifier answered **confidently and wrongly** —
`bus.jpg` came back `water jug` at p=0.019 where fp32 says `minibus` at
p=0.629. The speed was real; the arithmetic was not.

What makes it worth writing down is that every gate was green. `verify_vit.py`'s
sections 1-3 check the container's bytes, the patch layout, the panel invariants
and the int8 scheme's own arithmetic in numpy — and all of that was, and still
is, correct. The mistake lived in one line of C++ that no section executed.
`gemm1()` was a ViT-only wrapper around `NpuGemm::run()` that branched on the
container's dtype itself, and Whisper's encoder — which goes through `run()` and
therefore picks the scales up from `stage_operand`'s `OpScale` — was never
affected. The wrapper is deleted; all five ViT call sites say `run()` now, and
the dtype is decided in the one place that already decided it.

The gate now has a fourth section that runs the **C++ binary** and holds its
answer against `transformers`, asserting the *confidence* and not only the
top-1: on easy images a collapsed logit vector can still land on the right
label. Reintroducing the bug fails it 4 images out of 4 with a worst p-ratio of
102 against a gate of 2.0; fixed, the worst ratio is 1.52. `bus.jpg` through
`serve` now reads `minibus` at p=0.636 against bf16's 0.629.

Calibration is worth one line of its own, because it is a separate knob: the
default corpus is `synth_images`, eight synthetic ramp-and-edge pictures. On the
bus it is enough — top-1 still `minibus` — but quantisation error on
`layer.11.ffn_down` is 1.95e-02 against 1.48e-02 for a 22-image corpus of real
photographs. SmoothQuant is only as good as what it was shown, so a deployment
should pass `--int8-corpus <dir>`.

#### Reproducing the int8 container

The ViT int8 path needs `transformers` 4.x: this tree's ViT code and its ONNX
export both name tensors after the pre-5.x module tree
(`vit.encoder.layer.0.attention.attention.query`), and transformers 5 renamed
`encoder.layer` to `layers` and folded `attention.attention` into
`attention.q_proj`. Under 5.x the calibration refuses by name rather than
hooking the wrong modules, which is the right refusal — but it means
`python tools/pack/pack_npue.py --model-dir models/vit-base-patch16-224 --dtype
i8` needs `transformers==4.44.2` on a Python new enough to have its wheels.
(Other families in this tree are on transformers 5; the ViT export is not.)

### What has NOT been measured

Read this before quoting the table above as a result.

- **No top-1 accuracy on ImageNet data.** Every image run in this repository so
  far has been a PNG icon, a JPEG logo or a flat grey square — all outside the
  training distribution. The runtime has never been checked against
  `transformers` on the *dispatched* path, so no logit parity is claimed. A
  transposed panel or a drifted `im2col` would print smooth, confident, entirely
  wrong labels; `verify_vit_model.py` exists because that is the failure mode
  that does not announce itself, and it holds the host half of it.
- **Attention on the array is measured, and it is slower.** The flag exists and
  works — 0.349 s of encoder against 0.244 s on the host for `--npu-ops attn`,
  0.822 s for `--npu-ops softm`, and 6.685 s for both at once — but that is one
  image, one machine and one run, not a throughput curve. What is still not
  predicted is any *other* shape: nothing here extrapolates to a different
  position count, a different batch or a loaded device.
- **`npu2` is untested.** No second board was available.
- **Still no capacity or scaling curve.** The requests-per-second figures below
  are `n` sequential requests from ONE client on one machine, which bounds an
  endpoint from above and says nothing about where it saturates. No percentile
  distribution, no keep-alive accounting, and no more than two clients at once.

### Throughput, measured

`n = 5` sequential requests, one client, warm, on this machine. The warm-up
request is not counted. These are single-stream figures: every request pays the
whole model, which is what "how fast is this endpoint" means.

| endpoint | container | s/request | requests/second |
|---|---|---|---|
| `POST /v1/embeddings` | `all-MiniLM-L6-v2` (bf16) | 0.014 | **73.6** |
| `POST /v1/classify` | `vit-base-patch16-224` (bf16) | 0.261 | **3.8** |
| `POST /v1/pose` | `p_i8`, host convolutions | 0.314 | **3.2** |
| `POST /v1/pose` | `p_i8`, `--npu-ops conv` | 0.434 | **2.3** |
| `POST /v1/audio/transcriptions` | `whisper-tiny`, `jfk.wav` (11 s) | 0.849 | **1.2** |

The pose line is the array one **being slower**, which is the same result the
per-layer measurement reaches from the other end: 436 dispatches at a ~660 us
cost each against a 258 ms host network. The two figures bracket it from either
end and they agree — the array path is ~1.38x slower end to end.

Two clients at once, on all four: wall time is within a few percent of two
sequential requests, and **each client gets the answer its own input gets when
sent alone**. That second half is the claim worth having. "Both answered" is
what a server with one shared output buffer also does whenever the two requests
happen to agree; the check sends two different images (or two different texts)
and compares each answer against that input's solo answer, so a swapped or
interleaved result fails. They are served one at a time by design — one
`Session`, one pool — so the concurrency figure is about the queue and about
correctness, not about scaling.

---

## Body pose

A YOLOv8-pose container is a **detector**: it finds people in a photograph and
puts 17 COCO keypoints on each one. It is not an embedder and not a classifier,
and it runs through a mode of its own.

The network is 73 convolutions, 4.59 GMAC at 640x640, ending in a head that
produces `[56, 8400]` — four box values, one class score and 51 keypoint values
for each of 8400 candidate cells across three scales (strides 8, 16 and 32). The
runtime collapses all of that into one canonical tensor and decodes it: NMS,
thresholding, and the inverse letterbox. The 17-point skeleton and its 12 edges
live in the **runtime**, not the container, because they are a property of COCO
rather than of this checkpoint.

**CPU by default.** `--npu-ops conv` moves the convolutions to the array
and nothing else, exactly as it does for the other modes.

### Four commands

```bash
# 1. the container. --pose-onnx takes a single ONNX file, because a pose
#    checkpoint is handed over as one file rather than as a --model-dir with a
#    config.json. Do NOT pass --dtype bf16: it is refused by name, since there
#    is no bf16 convolution-weight path in this runtime.
python tools/pack/pack_npue.py --pose-onnx yolov8n-pose.onnx \
    --out models/yolov8n-pose.npue --device npu1
#    --dtype i8 takes the container from 22.8 MiB to 13.7, --dtype i4 to 12.8-13.9
#    (or to 24.8 at --int4-group 1, which is wider than fp32; see below).
#    --npu additionally stages the pre-tiled array panels; without it the
#    container runs on the CPU and --npu-ops conv refuses it by name.

# 2. run it. One image per argument; the result on stdout, the status block on
#    stderr.
./runtime/build/npuembeddings pose models/yolov8n-pose.npue photo.jpg --text

# 3. the machine-readable form: every box, every joint, the skeleton, and the
#    letterbox transform, so a client can map its own annotations through the
#    same pipeline the model saw.
./runtime/build/npuembeddings pose models/yolov8n-pose.npue photo.jpg --json

# 4. every graph node's fp32 output, for diffing against an independent
#    interpreter. This is how the 116-node parity below was established.
./runtime/build/npuembeddings pose models/yolov8n-pose.npue photo.jpg \
    --pose-dump /tmp/rt.bin
python tools/verify/diff_pose_dump.py yolov8n-pose.onnx \
    models/yolov8n-pose.npue photo.jpg --dump /tmp/rt.bin --head /tmp/ref.bin
```

### The array path, in three more commands

The container must be packed with `--npu` to carry the pre-tiled panels, and the
design set has to exist before the runtime will touch the array. Without both,
`--npu-ops conv` refuses *by name* rather than quietly running on the host —
which is the whole point of a flag that moves work.

```bash
# 5. pack with the panels. Without --npu the container has no gemm_b entries and
#    the array path refuses it.
python tools/pack/pack_npue.py --pose-onnx yolov8n-pose.onnx \
    --out models/yolov8n-pose.npue --device npu1 --npu

# 6. build the design set: 14 streams, one per distinct padded convolution
#    shape. The exporter flag is -n (short), and 32 is the N-tiling multiple on
#    npu1 -- a design's N must be a multiple of tile_n*cols = 128.
python tools/export/export_gemm_rtp.py --target yolov8n-pose --arch 1 \
    -n 32 --batch 1
#    writes runtime/artifacts/yolov8n-pose/artifacts_npu1/gemm_rtp/

# 7. run it. --artifacts names that set; without it this run would silently be
#    the host path again and any comparison against it would be trivially equal.
./runtime/build/npuembeddings pose models/yolov8n-pose.npue photo.jpg \
    --npu-ops conv --artifacts runtime/artifacts/yolov8n-pose/artifacts_npu1 \
    --text
```

The status block prints both backends' numbers every run, so the comparison is
per image and not a claim in a comment. Measured on this network the array is
**1.4x slower than the host** — see below for the three structural reasons and
the per-layer numbers behind them.

### The pose endpoint

`npuembeddings serve <model> --port 8080` serves the same model over HTTP at
`POST /v1/pose`, and the answer is **the same bytes** the CLI prints — both call
`npue::pose::result_json`, so the endpoint and `--json` cannot drift.

```bash
./runtime/build/npuembeddings serve models/yolov8n-pose.npue --port 8080

curl -s localhost:8080/health
# {"status":"ok","model":"yolov8n-pose",...,"kind":"pose","engine":"host",
#  "input_size":640,"conf":0.25,"iou":0.7,"keypoint":0.5,"max_det":300,
#  "skeleton":"nose,left_eye,..."}

curl -s -F image=@photo.jpg localhost:8080/v1/pose
curl -s -F image=@photo.jpg -F conf=0.4 -F iou=0.6 localhost:8080/v1/pose
```

`POST /v1/pose` takes `multipart/form-data` with one `image` part (PNG or JPEG,
decided by magic bytes, not by the filename) and optional `conf`, `iou`, `kpt`,
`max_det`. Also `GET /v1/models`.

The verb is `serve`, the same one every architecture uses — see
[One verb, four endpoints](#one-verb-four-endpoints). What keeps the URL space
honest is not a second verb but the second **path**: `/v1/embeddings` is
embeddings and `/v1/pose` is pose, and neither answers for the other. A pose
container asked for `/v1/embeddings` gets a 404 that names `/v1/pose` and lists
all four endpoints, not landmarks under the wrong name.

What it refuses, and why each refusal is there rather than a fallback:

| request | answer |
|---|---|
| not multipart | 400, naming the `Content-Type` it got — an image is binary and this cannot tell a JPEG from a long JSON string |
| no `image` part, or an empty one | 400 — there is no picture to look at |
| two `image` parts | 400 — ambiguous, and taking the first would be a silent answer to an ambiguous question |
| `conf`/`iou`/`kpt` not a number in range | 400 — a fallback to the default would answer at thresholds the client did not ask for, and the detection count is the product of them |
| `max_det` not an integer in 1..10000 | 400, same reason |
| upload over 64 MB | 413 |
| a file that is neither PNG nor JPEG | 400 with the front end's own refusal by name |
| unknown path | 404, naming the path and saying what this model serves |
| `GET /v1/pose` | 400 — the verb |

A 4xx is always the caller's request and 500 is always ours; a client that
retries on 500 must not retry "conf was not a number".

Per-request thresholds apply to **that request only** — the session's defaults
are restored immediately afterwards, so one request's `--conf` cannot become the
next request's default. Measured: `conf` 0.1 / 0.25 / 0.9 on `bus.jpg` gives
5 / 3 / 0 people, `max_det=1` gives 1, and the request after it is back to 3.

Not implemented, and named rather than silently absent: streaming, batch, world
landmarks, segmentation masks, per-request model selection, concurrent request
overlap. One request is one image on one process; a client that wants a second
model starts a second process, which is the same rule the speech endpoint
follows and for the same reason — one `hw_context` per model is the budget this
hardware has.

### The Python facade

`python/npue_pose.py` is a MediaPipe-shaped API over the same binary. It has no
numerics of its own: it runs `npuembeddings pose` (or POSTs to `npuembeddings serve`)
and parses that JSON, so there is exactly one place a keypoint is computed.

```python
import sys; sys.path.insert(0, "python")
from npue_pose import PoseLandmarker, PoseLandmarkerOptions

with PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        container="models/yolov8n-pose.npue", backend="http")) as lm:
    r = lm.detect("photo.jpg")            # path, bytes, file, ndarray or PIL
    for person in r.pose_landmarks:       # one list of 17 per person
        nose = person[0]                  # .name == "nose"
        print(nose.x, nose.y, nose.visibility)
```

Two backends: `cli` (default — one process per `detect`, nothing needs to be
running) and `http` (a `serve` child started on first use and stopped by
`close()`; that is what a video loop wants, and it is why the second detect on
`bus.jpg` costs 0.30 s rather than a second model load). Both return the same
object — verified identical landmark-for-landmark.

### The webcam demo

`examples/pose_webcam.py` draws the skeleton over a live camera with OpenCV — a
thin drawing loop over the facade above, which invents none of the numbers it
draws.

```console
$ pip install opencv-python          # already in requirements.txt
$ python examples/pose_webcam.py
camera 0: 1280x720 at 10 fps reported
q/ESC quit   s save   space pause   c colours   -/+ thickness
input 640px, letterbox {'scale': 0.5, 'pad_x': 0, 'pad_y': 140}, backend host
```

`q`/ESC quit, `s` saves a frame to `out/`, space pauses, `c` cycles the palette,
`-`/`+` change the line thickness. Other flags: `--camera N`, `--image PATH` to
run on one still (no camera needed), `--no-window` to write every frame to
`--out` instead of opening a window, `--conf`/`--kpt`/`--max-det`, and `--array`
for the convolutions on the NPU.

Measured on this machine, i8 container, 16 threads, one 1280×720 camera frame at
640 px: **0.73 s per frame, 1.36 fps.** The 0.59 s of that is the model (the
same 0.589 s the CLI reports on `bus.jpg`); the rest is the PNG the facade
encodes to upload the frame, the `cvtColor`, and the drawing. At that rate a loop
is a check that the detection works, not a preview — which is the point of it.
`--array` is refused on a container with no design set on disk, and says so.

Two things the drawing does that a naive version gets wrong, both visible in the
output rather than argued here:

- **BGR → RGB before the model.** `cv2` hands back BGR and the runtime's front
  end reads an array as RGB. Handing it BGR trains the model on swapped
  channels; the pose mostly still lands, which is exactly what makes it worth
  doing deliberately.
- **A bone is drawn only when both its joints cleared `--kpt`.** Drawing the
  surviving half would join a wrist to a shoulder across a gap the model said
  was not there, and the eye reads that as a straight arm rather than as a
  low-confidence one. Bone colour is blended with the mean confidence of its two
  joints, because the informative end of a limb is its middle and a dot per
  joint cannot show that.

And one measured behaviour worth knowing before you point this at a room: a
person **mostly out of frame still scores high** — 0.85 for a sitter at the left
edge with head and shoulders cropped, surviving `--conf 0.7` — and the skeleton
drawn is the visible part, with the absent bones left absent rather than
extrapolated. That is why the default `--conf` stays at YOLO's 0.25 and is not
raised to make a demo look tidy.

Two differences from MediaPipe are structural, and both are stated in the module
rather than discovered later:

- **No `pose_world_landmarks`.** MediaPipe's are metric hip-origin coordinates
  from a second head. YOLOv8-pose has no such head; the attribute does not
  exist. `NormalizedLandmark.z` is always `0.0` and `has_depth` is False, which
  is more useful than a plausible-looking number.
- **`pose_boxes` and `pose_scores` exist**, which MediaPipe does not have —
  YOLOv8-pose detects, so it has both. Anything that reads only
  `pose_landmarks` works unchanged.

`detect_for_video(image, timestamp_ms)` exists for call-shape parity and does
**not** use anything temporal: the model has no tracker and no memory, so the
answer for frame 900 comes from frame 900. The timestamp is echoed into
`PoseResult.timestamp_ms` and read by nothing else.

One alias is exact and one is refused, because they are different things:
`min_pose_detection_confidence` → `--conf`, `min_pose_presence_confidence` →
`--kpt` (this model's per-joint score *is* MediaPipe's presence score), and
`min_tracking_confidence` → **nothing**; passing it is ignored and
`PoseLandmarkerOptions.ignored_options` says so, rather than being turned into a
threshold that does something else.

MediaPipe's `min_pose_presence_confidence`/`min_tracking_confidence`/`min_pose_detection_confidence`
triple, a `PoseResult` that iterates as its people, and `NormalizedLandmark`
unpacking as `x, y, z` are all honoured. It is a facade, not a claim of
compatibility: **no `mediapipe` import is needed, and none would work** — this is
a different runtime with a different model behind the same names.

### What has been verified

The CPU path is **correct end to end**, against an independent NumPy ONNX
interpreter (`tools/verify/verify_pose.py`) that shares no code with the
runtime:

- **116 of 116 graph nodes agree.** Worst relative difference 6.5e-06, on the
  first image. The canonical head tensor agrees to 7.4e-07 overall.
- **The array path produces the same network.** With
  `--npu-ops conv --artifacts ...`, all 117 nodes of the `--pose-dump` agree
  with the host path's, on the same metric `diff_pose_dump.py` uses
  (`max|a-b| / max(1, max|b|)`): median 1.7e-02, worst 6.4e-02 at the c=51 40x40
  keypoint branch, 1.3e-02 at the head. That is bf16 — the array's datapath has no
  fp32 multiplier — and it is *not* a small number, so it is worth being precise
  about what it does and does not establish: the detections are the same three
  people with boxes within 3 px and keypoints within 10 px, and every node's
  scale is right, but an intermediate tensor can be 6% off. On the CPU path the
  same comparison is 6.5e-06, so the 1e-2 band is the datapath and not the graph.
  Getting here took three bugs that a shape check could not see: a *permuted*
  panel written as `w.reshape(K, N)` where a transpose was meant, 58 of 72
  convolutions reading another layer's B panel because the staging was keyed by
  stream name, and a per-chunk transpose that took its destination channel stride
  from the chunk's row count instead of the tensor's own `OH*OW` — which left
  channel 0 right and 15 of 16 channels wrong at every chunk.
- **The detections are identical on `bus.jpg`** — same three people, same scores to
  three decimals, same boxes and all 17 keypoints, against the reference.
- **The quantised networks are still tracking.** f32, i8 and i4 were each run
  node by node against the f32 container's dump: node −1, the graph input, is
  **bit-identical** in all three, so every difference downstream is the weights and
  not the image. The i8 relative error per node sits in a 2e-2 band, peaks at
  7.2e-02 at node 101, and falls to 3.3e-03 at the head; i4 group 8 runs an order
  of magnitude higher and peaks at 4.2e-01. No NaN or Inf in either. A dequantiser
  reading the wrong axis would show ~1.0 at node 0, not 0.02.
- **Six damaged containers are refused by name**, each with the actual
  inconsistency in the message: an unknown `conv_weight_dtype`, a `conv_int4_group`
  on an i8 container, a missing `conv_weight_dtype`, a `wscale` shortened to 8
  bytes against 16 output channels, an i4 payload of 100 bytes where `[16, 27]`
  needs 224, and a `gscale` of 4 bytes where the group needs 64. All exit non-zero
  with an empty stdout.

That last one is the claim that matters and it took five runtime bugs to reach.
Each was invisible to a shape check and to an end-to-end comparison, which is
why the node-by-node dump exists. Recorded here because the shape of the
failures is the argument for the tool: a weight transpose read with the right
stride and the wrong meaning produces a network that is confidently wrong.

### Measured, on both paths

One image (`bus.jpg`, 810x1080), i8 container, 16 threads, this machine. Both
columns are the same run with and without `--npu-ops conv --artifacts`, so
the difference is the backend and nothing else.

| stage | host (default) | array |
|---|---|---|
| front end (decode, letterbox, normalise) | 23 → **5.5** | 24 → **6.1** |
| **network** | **166** | **311** |
| — of which the multiply | 76 | 162 *(on the device)* |
| — of which im2col | 46 | 51 |
| — of which the `[M,N]` → NCHW transpose, SiLU included | 15 | 32 |
| — of which the per-convolution weight transpose | 0 *(built at load)* | — |
| — of which widening A to the panel's K | — | 46 |
| — the 43 non-convolution nodes | 20 | 19 |
| — neither: allocation and per-node overhead | 9 | 4 |
| decode (NMS, boxes, JSON) | 0.0 | 0.0 |
| **total** | **0.19 s** | **0.34 s** |

The front-end row is the only one that moved, and it moved because of a measured
fix rather than a re-baseline: see the letterbox paragraph below for the before
and after, the method, and the gate.

The network rows are **left at the numbers that session measured** rather than
replaced by today's. This machine's network time moves about 20% between
sessions and by ±20% inside one: re-measured today the same CLI gives a median
network of 178 ms across ten runs (145-188), and twelve consecutive requests to
one long-lived server give 146 ms (135-193). The 166 in the table sits inside both
of those ranges, so swapping in any of them would be picking a point in the noise
and calling it a result. The front end, by contrast, moved from 23-24 ms to under
7 — a change several times this machine's own run-to-run spread, which is why that
row is the one that changed.

Medians of six consecutive frames of the same image in one process, because a
single frame on this machine is worth 10 ms of noise and the changes below moved
the number by less than that.

Every row is a printed bucket, not a subtraction: the host split is timed
directly, and the array's multiply is timed around `npu_->gemm` rather than
recovered as "whatever `t_array` has left over". That distinction is not
cosmetic — the leftover *was* wrong until it was measured, and it was wrong in
both directions at different times. `t_array` starts before the weight transpose
and before im2col, so subtracting only the two host spans reported 66 ms of
"device GEMM" that was mostly those two; and `t_array_transpose` used to start
before the `gemm` call, so its 175 ms was 157 ms of device multiply and about
20 ms of transposing. A table built on either would have pointed the reader at
the wrong bottleneck.

#### The letterbox was 21 ms of serial scalar code

The front end is `decode`, then a PIL-equivalent bilinear **letterbox** to
640px, then `(x/255 - mean)/std`. The letterbox was the whole cost, and the way
that was established was by removing it rather than by reading it: the same
810x1080 frame, fed in already at 640x640, where `resize_to` returns itself
unchanged and every other step runs identically.

| input | front end |
|---|---|
| 810x1080, resize present | **24.8 ms** |
| 640x640, resize the identity | **3.7 ms** |

So ~21 of the 23 ms was one bilinear resize. `runtime/src/vit/image.cpp`
referenced the thread pool **zero** times, and its resampler had two separate
problems:

- **The weights were recomputed per line.** They are a function of the output
  index and the axis geometry alone — they never read a pixel — so the 1080-line
  horizontal pass evaluated the same 640 weight vectors 1080 times over, about
  1.1 M evaluations of the index arithmetic and the kernel body that could not
  have differed between lines. They are now built once per pass into an
  `AxisPlan`.
- **The passes were serial.** With the taps hoisted, a line is a short dot
  product and lines are independent, so both passes go to the pool. The vertical
  pass is split over **output rows, not source columns**: consecutive columns are
  3 bytes apart inside one output row, so ~21 columns share a 64-byte line and
  any 21 of them are likely to be on 21 different workers.

The normalise loop went the same way (one task per channel-row), and the front
end now writes the network's input tensor directly instead of returning a vector
that the caller `std::copy`'d — and that tensor is kept between frames, which is
worth another ~2 ms, since a fresh one costs 1.1 ms of value-initialising 4.9 MB
that is immediately overwritten plus ~0.9 ms of first-touch page faults.

| | before | after |
|---|---|---|
| CLI, 10 runs, median | 23-24 ms | **5.5 ms** |
| 12 consecutive requests to one server | — | **3.5 ms** |
| resize alone, isolated | ~21 ms | **2.1-3.1 ms** |

The server figure is lower than the CLI's because the CLI is a fresh process per
frame and pays the one-time allocation every time.

**None of this may change a pixel**, and that is gated rather than asserted.
`tools/verify/verify_vit_image.py` section 2 now runs every corpus case through
all four implemented filters twice — once serial, once on an 8-thread pool — and
requires the two rasters to be **byte-identical** before comparing either to PIL.
`AxisPlan` is what makes that true: the taps are computed by the same expressions
from the same inputs, and every output element still sums the same taps in the
same order and divides by the same `ww` (a division, not a multiply by a
precomputed reciprocal — those differ in the last bit, and this function's
contract with PIL is measured to one LSB). All four filters and every corpus case
pass; the PIL comparison itself is unchanged and still within one LSB.

`Pool::run` also grew a guard: a thread already inside a generation runs the work
serially and returns. `resize_to` taking a pool makes nested use reachable from
inside `par_items`, and the failure mode of getting that wrong is a hang in a
webcam demo rather than a wrong number.

Summed per convolution rather than per stage, the 72 convolutions that actually
run take **137 ms on the host and 289 ms on the array**, in **436 dispatches at
650-695 us each** — 72, not 73, because the head's `[1,16,1,1]` DFL convolution is
folded analytically into the decoder and is never dispatched as a GEMM. The
array's own GEMM is 372 us of those; the remaining 290 are im2col, the A repack
and the transpose, all on the host.

**The two paths are not interchangeable, and the reason is now measured rather
than inferred.** The device multiply is 162 ms where the host's own blocked
fp32 GEMM is 76 ms of the same 4.59 GMAC, so the array is *slower at the
multiply*, not faster — 74 GMAC/s against 60, on work the design has already
padded 2.6x, so per useful MAC it is about 4x worse. Whatever the array buys, it
is not arithmetic throughput on this graph.

torch, fp32, 16 threads, on the same graph: **20.5 ms**, so the host path is
**8.1x slower than torch**, and its split says why: it is mostly not the
multiply. The multiply is 76 of 166 ms and is 2x off oneDNN-class code, and
im2col is 46 and moves 18 GB per image to feed it. The non-convolution nodes are
20 and **do not scale in either direction** — 18 ms on one thread against 20 on
sixteen — which is what a bandwidth-bound span looks like and not what a
barrier-bound one looks like. Raising the parallel threshold to 2^20, so that
only the largest joins stay parallel, measured 172 ms of network against 166, so
that is not the lever either. Closing the rest of the gap means not
materialising im2col at all, which is a change to the GEMM's inner loop and not
to the weights.

**The array is 1.9x slower than the host on this network.** Three structural
reasons account for the difference, and the first is now the measured one:

- **The device multiply is slower than the host's GEMM**, 162 ms against 76, for
  the 2.6x of arithmetic padding below.
- **2.6x arithmetic padding waste.** A design's `N` must be a multiple of
  `tile_n * AIE columns` = 128 on npu1, and most of this network's convolutions
  have `N <= 64`. 4.59 GMAC of useful work is dispatched as 11.95 GMAC.
- **436 dispatches, 100 of them for the stem.** A dispatch is `M <= 1024` rows and
  the stem has `M = 102400`. The rest follow: 5 convolutions at M=25600 (25 each),
  20 at 6400 (7 each), 25 at 1600 (2 each), 21 at 400 (1 each).
- **The per-dispatch host work is 129 ms to the device's 162**, which is the one
  that would have to change to fix the others.

The measured per-layer picture says an oracle would not save it either: **19 of
the 72 convolutions are faster on the array than on the host**, and an oracle that
put each one on its faster side measured **146.4 ms against the host's 150.2** —
2.6% for choosing per layer. So `--npu-ops conv` is honoured rather than
refused: the choice belongs to the operator, the flag makes it measurable, and
the status block prints both sides' numbers so the claim is checkable per run
rather than argued in a comment.

The status block prints both of these splits, so every number above is
reproducible rather than asserted.

### Quantised weights

`--dtype i8` and `--dtype i4` store the convolution weights narrow and
dequantise once, at load, into the same fp32 buffer the GEMM already reads. So
this **buys container bytes and nothing else on the CPU path** — the arithmetic
is the same fp32 arithmetic. It is not claimed to be faster, and it is not
measured to be.

ONE IMAGE ONLY — the whole right-hand half of this table is `bus.jpg` and three
detections in it (`tools/verify/verify_pose_quant.py`). Two sections below repeat
it over more files, and they disagree in a way that matters.

| dtype | file | worst rel-Fro | score Δ | box Δ | keypoint Δ | people |
|---|---|---|---|---|---|---|
| f32 | 22.8 MiB | — | — | — | — | 3 |
| **i8** | **13.7 MiB** | 0.034 | **0.001** | **1.9 px** | **5.8 px** | **3** |
| i4 group 8 | 13.9 MiB | 0.095 | 0.050 | 5.6 px | 50.9 px | 3 |
| i4 group 16 | 13.1 MiB | 0.120 | 0.077 | 28.5 px | 54.3 px | **8** |
| i4 group 32 | 12.8 MiB | 0.150 | 0.206 | 26.1 px | 71.3 px | **8** |
| i4 group 1 | 24.8 MiB | 5.8e-08 | 0.000 | 0.0 px | 0.0 px | 3 |

Four things in that table are worth reading twice.

**The sizes are much closer together than the weights are, and the reason is the
9.9 MiB of array panels.** Every container in that row carries the same
`gemm_b` panels, 9.875 MiB, because the array's datapath is bf16 whatever the
host's weights are: 24% of the f32 container, 76% of the i8 one. `conv_w` itself
goes 12.53 → 3.13 MiB (i8) → 1.57 MiB (i4). So the i4 payload saves 1.57 MiB and
its group scales give 1.57 MiB back, which is why **i4 group 8 is 0.2 MiB larger
than i8** and why i4 buys anything at all only at group 32 or above.

**i8 is the quantisation to use.** 0.034 rel-Fro, a 1.9 px box, the same three
people *in this photograph*. Not "the same three people" in general — see below,
where i8 loses one of them on a set of unrelated photographs.

**There is a cliff between i4 group 8 and group 16.** Eight keeps three
detections; sixteen finds eight in a photograph of three. Both containers are
valid, the smaller one is past the cliff, and the relative error on the weights
moves only from 0.095 to 0.120 across it. A weight-error number cannot express
that, which is why the detector comparison exists as a separate tool.

**i4 with a group of one is lossless and bigger than fp32.** One scale per weight
means the nibbles are exact — the group scale is `|w|/7`, so the product
round-trips to fp32 — and the detections are identical to fp32's down to the last
digit. The file is 24.8 MiB against fp32's 22.8, because four bits per weight
plus four bytes of scale per weight is more than the four bytes it replaced. A
group too small to lose anything saves nothing, and those are one fact rather
than two.

### Over more files, and then over other scenes

The table above would support "i8 is indistinguishable from fp32", and on
`bus.jpg` it is. Repeated it is not, and the correction is worth more than the
table was. Two sets, because they answer different questions:

**(a) five treatments of one scene** — grayscale, RGBA, a 37x23 thumbnail, a
1300x1300 crop, a quality-20 JPEG. This exercises the front end, the letterbox
and the decode path, and it is *one scene*.

**(b) nine photographs** — `bus.jpg` plus eight unrelated stock photographs of
indoor and outdoor scenes with 1 to 4 people each, which is the first scene
diversity in this tree.

| set | container | matched | lost | **gained** | mean score Δ | worst | mean box | worst | mean keypoint | worst | flips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) 5 files | **i8** | 13 | **0** | **3** | 0.0055 | 0.053 | 2.1 px | 3.3 px | 5.7 px | 18.5 px | 1 |
| (a) 5 files | i4 g8 | 13 | **0** | 1 | 0.036 | 0.129 | 6.1 px | 14.1 px | 36.5 px | 94.6 px | 26 |
| (b) 9 photos | **i8** | 26 | **1** | **1** | 0.0093 | 0.039 | 1.9 px | 12.8 px | 7.2 px | 42.4 px | 3 |
| (b) 9 photos | i4 g8 | 22 | **5** | 0 | 0.078 | 0.348 | 11.9 px | 78.9 px | 25.1 px | 74.8 px | 18 |
| (b) 9 photos | i4 g16 | 20 | **7** | **14** | 0.139 | 0.436 | 17.1 px | 54.9 px | 30.6 px | 68.6 px | 22 |
| (b) 9 photos | i4 g1 | 27 | **0** | **0** | 0.0000 | 0.000 | 0.0 px | 0.0 px | 0.0 px | 0.0 px | 0 |

"Gained" is a detection the quantised container reports and the fp32 container
does not, at the default `--conf 0.25`; "lost" is the opposite. Matching is box
IoU ≥ 0.5.

Three things follow.

**Scene diversity changes the answer, and not in i8's favour.** On the five
treatments of one scene i8 loses nothing. On nine photographs it **loses one
detection** and i4 group 8 loses **five**. Those are not people who were never
there; they are people fp32 finds at the default threshold and the quantised
container does not. **This is why the earlier "i8 is indistinguishable from fp32"
was a statement about one photograph.**

**The extra detections are not ordered by weight error.** In set (a) i8 is the far
more accurate container on every axis — 0.034 rel-Fro against 0.095, 5.7 px
against 36.5 px of keypoint error — and it adds *three* times as many detections.
A borderline score crosses 0.25 because of which way the rounding fell, and a
container that is uniformly closer to fp32 is not uniformly less likely to push
one particular cell over a threshold. **A weight-error number does not predict
this**, which is the same reason the i4 cliff needs a tool of its own.

**The keypoint flips are the other failure, and they are worse than the box
column.** i4 group 8 flips 26 of 221 keypoint visibility decisions in set (a)
against i8's 1. That is a number about *pose* rather than about *detection*, and
the detection column does not show it.

What set (b) still does **not** cover: nine photographs is nine scenes, not a
benchmark. No crowd over ten people, no night-time or heavy-occlusion frame, no
child or wheelchair, nothing at all in another language's signage, and the eight
stock photographs were not chosen to be hard. A number from this table is a
measurement, not a threshold, and **no accuracy requirement has been decided for
a quantised pose container** — `verify_pose_quant.py` exits 0 on every accuracy it
measures, including i4 group 16.

### What has NOT been measured

- **No i8 or i4 array datapath.** Quantised weights are for the CPU path. On
  the array an int8 MMAC is the reason quantisation pays, and that is unbuilt.
  What the array reads today is bf16, which is why every container in the table
  above carries the same 9.875 MiB of panels whatever its host weights are.
- **No load test on the pose server.** The refusals, the per-request thresholds
  and the CLI/server agreement above were checked one request at a time. The
  server is single-threaded by construction (one `Session`, one pool). It has now
  been run with two clients at once — each got the answer its own image gets
  alone, on both backends — and timed at 3.2 req/s host / 2.3 req/s array, but
  there is still no capacity curve and no scaling data past two clients.
- **The Python facade has no test suite.** It has been exercised by hand against
  `bus.jpg` — both backends, all five input types, the refusals, and
  landmark-for-landmark agreement with the CLI — which is not the same as a gate
  that fails when it should. It is also stdlib-only, so it is not covered by the
  gates that check `requirements.txt`.
- **`npu2`.** No second board was available, as everywhere else in this README.

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

### `tools/data/npu_targets.json`

Single source of truth for model geometry, per-generation defaults, and the
stream list per `kind`. Inspect it:

```bash
python tools/export/export_gemm_rtp.py --list-targets
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
| `--elt-cols N` | array columns for the eltwise designs (LayerNorm/softmax cap at 2) |
| `--seq N` | sequence length to build for |
| `--cache-root DIR` | IRON JIT cache (default `~/.npu/cache`) |

**There is no op flag, and that is the point.** One command builds the GEMM set
plus every sibling design the target can honour — `gelu/`, `layernorm/`,
`softmax/` for a BERT-family text embedder — and prints the list it chose plus a
reason for each code it skipped. `--npu-ops`, `--npu-extra-ops` and
`--npu-eltwise` are refused there by name, each with its own message, because the
set a model can use is a fact about the model and not something a command line
should be able to get wrong. See [NPU_OPS.md](NPU_OPS.md) for what "can honour"
means per architecture, and `tools/lib/npu_ops.py` for the registry the exporter
reads.

### Two output layouts

`--target` puts designs under a per-model subdirectory:

```
runtime/artifacts/<model>/artifacts_npu<N>/gemm_rtp/
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
In `tools/export/export_eltwise.py`, the LayerNorm worker acquired its gamma|beta
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

| device | operand | `mac_dims` | B panel order |
|---|---|---|---|
| `npu2` (aie2p) | bf16, i8 | `(8, 8, 8)` | `(s=8, t=8)` |
| `npu1` (aie2) | bf16 | `(4, 8, 4)` | `(s=8, t=4)` |
| `npu1` (aie2) | i8 | `(4, 8, 8)` | `(s=8, t=8)` |

`npue.gemm_b_layout` defaulted to `(8, 8)` — correct for npu2 — and the packers
passed that default straight through, so every container was written in npu2's
order. The comment above the constant recorded the (correct) observation that
plain bf16 and bfp16-emulated agree, and the table was then keyed by device
alone. That generalisation — *the datapath does not change the sub-tile, and the
board does* — is true of bf16 and false in general, and it is what let the int8
datapath ship broken for years afterwards: on npu1 the int8 MMAC's N sub-tile is
8, where bf16's is 4, so an int8 container was tiled for 4. See
[the int8 datapath section](#the-int8-datapath-read-a-whole-model-wrong) below
for that one, which is the same failure mode reached by a different route.

**Fix.** The sub-tile is resolved from the target device *and* the operand dtype
in one place (`npue.MAC_BY_DEVICE`), threaded into every operand by both packers
(`pack_npue.py --device`, `packers/whisper.py`) and taken by every C++
`prepare_model*`, and printed in every build log. `mac_for_device` takes the
dtype as a required argument rather than defaulting it, because the bf16 answer
is the tempting one and a default is how the int8 half got it. The exporter
picks the pair from `--arch` and the operand dtype, so `design.json`'s
`b_layout_hash` now means what it says: a container packed for the other
generation hashes differently and the runtime's existing check refuses it.
`verify_i4_scheme.py` section 8 re-measures the table against the compiler on
every run, so a toolchain that moves its sub-tile fails a cheap gate instead of
shipping.

**Verification.** `tools/verify/verify_design_numerics.py`, added for this: it feeds
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
`--npu-ops <code>`, or drop the code from the list.

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
tools/                     Python: packer, exporters, gates (no C++ at run time)
  pipeline.py              the way in: "I want to ..." -> the command for it
  pack/pack_npue.py        build the .npue container
  export/export_gemm_rtp.py  unified GEMM design set, per arch and batch tier
  export/export_eltwise.py   gelu / layernorm / softmax design sets
  verify/verify_*.py       correctness gates
  gen/, research/          generators for committed files; experiments
  lib/                     shared modules, imported not run
  data/npu_targets.json    model geometry + per-arch defaults (exporter's source of truth)
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
  include/pose/            YOLOv8-pose: geometry, the image front end, the op
                           graph's convolutions, the head, the decode, the one
                           JSON emitter both the CLI and the server call
  src/                     implementations, mirroring include/
                           (cli/ common/ embed_models/ encoders/ pose/ server/
                            tokenizers/ whisper/ + the top-level units)
python/                   the runtime's optional Python face
  npue_pose.py             MediaPipe-shaped PoseLandmarker over the CLI or the
                           HTTP endpoint. Stdlib only -- no numpy, no torch --
                           and it holds no numerics of its own
examples/
  pose_webcam.py           draw your skeleton from a webcam with OpenCV. The only
                           file in the tree that imports cv2, which is why cv2 is
                           in requirements.txt and nothing else here needs it
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
| [`tools/README.md`](tools/README.md) | exporter and packer reference |
| [`STATE.md`](STATE.md) | the session handoff: what is built, what is measured, what is not |
| [`PROVENANCE.md`](PROVENANCE.md) | what `tasks/NNNN` and `docs/…` in the source comments refer to |
| [`THIRD-PARTY.md`](THIRD-PARTY.md) | licences and provenance of vendored and generated material |

**The `docs/` tree is not in this repository.** A set of design and status
documents under `docs/` — `CURRENT_STATUS.md`, `00-overview.md`,
`01-hardware/` through `06-performance.md`, `04-model/npue-format.md`,
`history.md` — is cited in roughly 45 places in the source, and those citations
are where the *reasoning* behind a number lives rather than the number. This
README is the complete guide to *how to run* the project; it is not a substitute
for the documents that record *why* it is built this way. `PROVENANCE.md` lists
every path the tree cites and what each one was, so a dangling reference is
recognisable as one rather than as a typo.

Comments citing `tasks/NNNN` name entries in the project's task log, which is
also not in this repository. The numbers are stable identifiers; see
`PROVENANCE.md`.
