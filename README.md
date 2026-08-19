# NpuEmbeddings

**Sentence embeddings on the AMD Ryzen AI NPU, from hand-written AIE kernels,
on native Windows, with a C++ runtime.**

Text in, vectors out — one executable, one model file, no Python at runtime.
It serves an OpenAI-shaped `/v1/embeddings` endpoint, so anything that already
talks to an embeddings API can point at it instead.

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "another sentence"]}'
```

---

## What this is

An implementation of **BERT-family sentence encoders** that runs their matrix
multiplies on the **XDNA2 NPU** in AMD Ryzen AI processors (Strix Point). The
kernels are written directly against the AI Engine array using
[MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie) — not through ONNX
Runtime, not through a vendor overlay — and the shipped runtime is C++ and XRT.

Four models are supported today. **`bge-base-en-v1.5` is the one whose
geometry fits this NPU best** — head_dim 64, and every layer width a multiple
of 384 — so it is the sensible default; `all-MiniLM-L6-v2` is the fastest and
`bge-large-en-v1.5` the most accurate.

It is also a **learning project**, and that is visible in the repository. Every
result has a task log recording what was measured, on which command, and what
went wrong on the way. The failures are kept deliberately; several of them are
more useful than the successes.

| | |
|---|---|
| Models | all-MiniLM-L6-v2, bge-small/base/large-en-v1.5 — 384 to 1024 hidden |
| Sequence length | 64 tokens (fixed at build time) |
| Hardware | AMD Ryzen AI, XDNA2 / Strix Point (developed on a Ryzen AI 9 HX 370) |
| OS | Windows, native — no WSL |
| Precision | bf16 with fp32 accumulation on the NPU; fp32 on the host |
| License | Apache-2.0 |

**What actually runs where.** The GEMMs — 24 per encode on a 6-layer model — run on the NPU as
instruction streams over a single design. LayerNorm, softmax and GELU run on
the host in fp32 — that is not a shortcut but a measured decision: at hidden
384 those ops are too small to earn an NPU dispatch, and moving them to the
host made the model both *faster and more accurate*. Attention's per-head
matmuls are on the host too (`head_dim = 32` does not tile the array).

## What it solves

A sentence-embedding workload is a good fit for an NPU and a poor fit for the
CPU it shares a package with: it is a fixed, small model applied to a large
number of short inputs. The point is to **get that work off the CPU cores**
without losing throughput or quality.

Measured on a Ryzen AI 9 HX 370, against `sentence-transformers` on the same
machine at the same sequence length:

Measured **interleaved** — all three encoders round-robin in one session, the
same statistic on every side — because wall clock on a shared machine drifts
enough that numbers taken minutes apart compare the machine, not the code
([`0040`](tasks/0040-m9-honest-cpu-baseline/TASK.md)).

Throughput at batch 128, seq 64, in sequences per second:

| model | hidden | layers | **NPU** | torch | ONNX Runtime | **NPU / best CPU** |
|---|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 384 | 6 | **877** | 489 | 234 | **1.79×** |
| bge-small-en-v1.5 | 384 | 12 | **445** | 290 | 134 | **1.53×** |
| bge-large-en-v1.5 | 1024 | 24 | **52.8** | 25.1 | 11.4 | **2.11×** |

`bge-base-en-v1.5` (hidden 768, 12 layers) was added later and runs at
**181 seq/s**; its CPU side has not been re-measured in the same session, so
no ratio is quoted for it. It is the model whose geometry fits this NPU best —
head_dim 64, and every layer width a multiple of 384
([`0051`](tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md)).

| | NPU | CPU (`sentence-transformers`, 12 threads) |
|---|---|---|
| CPU cores occupied | **~5.3–6.0** | 12 |
| Energy per 1000 sequences (MiniLM) | **44.0 J** | 85.3 J |
| Latency, single text | 15 ms | — |

ONNX Runtime is included because `research/prior-art.md` prescribes it as the
primary CPU baseline. It is **half** the speed of torch on all three models --
attention never fuses in the exported graph -- so torch with SDPA is the harder
opponent and the ratio is taken against it.

So: **1.5–2.1× the throughput on under half the cores**, at 1.94× less energy
per sequence, and **3.2–4.2× more throughput per core**.

**The advantage grows with model width and shrinks with depth**, and the three
rows separate the two: bge-small holds width fixed and doubles the layers
(1.79 → 1.53, because our cost is per dispatch and the CPU has no such term),
while bge-large's 1024-wide layers take it to 2.11×. That is the prediction
[`0027`](tasks/0027-m7-width-hypothesis/TASK.md) made from a synthetic sweep,
now measured on real models.

And the embeddings are the same embeddings:

| check | result |
|---|---|
| Worst `1 − cos` vs the HuggingFace reference | **1.086e-05** |
| MTEB, 5 tasks (STS, classification, clustering), delta vs CPU | **+0.04 points** |
| Tokenizer, vs HuggingFace over 6,826 texts | **6,826 / 6,826 exact** |
| Top-10 nearest-neighbour overlap vs reference | **99.0%** |

Every figure above is reproducible from this repository —
see [`tools/verify_*.py`](tools/) and the task logs.

### What it does not solve

Being explicit, because the honest limits are part of the point:

- **Four models, one sequence length.** all-MiniLM-L6-v2 and bge-small-en-v1.5
  share one compiled design set, bge-base-en-v1.5 has its own (hidden 768), and
  bge-large-en-v1.5 needs a third because its layer widths make the production
  tile size illegal. All are compiled for **seq 64**, and longer inputs are
  truncated. The model is named explicitly — `npuembeddings serve <model>` —
  so a script breaks loudly rather than silently changing which model it ran.
- **One machine architecture.** XDNA2 on native Windows. Not XDNA1, not Linux,
  not other NPUs.
- **The server is a localhost server.** No TLS, no authentication, one request
  at a time. Put a reverse proxy in front of it, or use it in-process.
- **The NPU is shared.** Other processes using it will affect your throughput.

## How to use it

### 1. Download

Grab the latest `npuembeddings-*-win-x64.zip` from
[**Releases**](../../releases) and unzip it anywhere — it is about 300 KB:

```
npuembeddings.exe         the runtime            0.7 MB
artifacts_b128il/         compiled NPU design, hidden 384
artifacts_base/           compiled NPU design, hidden 768
list-models.cmd           what this build can run
serve.cmd                 start the endpoint
embed.cmd                 embed a text file
manifest.json             sha256 of every file, and the verified figures
```

No installer, no Python, nothing written outside the folder.

### 2. Pick a model

```cmd
npuembeddings list
```

```
  model                state     layers hidden  pooling      size  notes
  all-MiniLM-L6-v2     available      6    384     mean  91 MB dl  smallest and fastest
  bge-small-en-v1.5    available     12    384      cls 134 MB dl  +2.99 MTEB points over MiniLM
  bge-base-en-v1.5     available     12    768      cls 438 MB dl  best geometric fit for this NPU
  bge-large-en-v1.5    available     24   1024      cls 1340 MB dl highest quality, slowest
```

### 3. Run the endpoint

```cmd
npuembeddings serve bge-base-en-v1.5
```

**The weights are not redistributed here**, so the first run fetches them:

```
  bge-base-en-v1.5 is not installed. Fetching it from BAAI/bge-base-en-v1.5.
  438.0 MB of checkpoint, verified against a checksum built into this executable.
  ...
  hash  model.safetensors
        ok  c7c1988aae201f80...
  pack  bge-base-en-v1.5.npue
serving http://127.0.0.1:8080/v1/embeddings   (model bge-base-en-v1.5-npu, seq 64)
```

It downloads from HuggingFace, **checks the checksum against the exact
checkpoint this build was verified against**, cross-checks the model's own
config against what this build expects, and packs it into
`models/<name>.npue` — weights pre-tiled into the order the NPU's DMA reads
them, plus the tokenizer vocabulary, in one file. A checksum or config
mismatch **stops**, rather than running weights nobody verified.

All of that happens inside `npuembeddings.exe`: no Python, no `curl`, no
download script. (Before 0.2.0 it *was* a batch script that fetched a binary
and compared a hardcoded hash — which is indistinguishable from a dropper, so
antivirus flagged it. Same checks, no script.) The packer is verified
byte-identical to the reference implementation
([`tools/verify_pack_parity.py`](tools/verify_pack_parity.py)) — two
implementations of one binary layout is a real risk, so it is tested rather
than trusted.

Use it from the official OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
r = client.embeddings.create(
    model="bge-base-en-v1.5-npu",
    input=["a man is playing a guitar", "someone plays guitar at a concert"],
)
print(len(r.data[0].embedding))     # 768
```

Supported: `POST /v1/embeddings` (`input` as a string or array of strings,
`encoding_format` `float` or `base64`), `GET /v1/models`, `GET /health`.
Requests carrying token-id arrays instead of text are rejected rather than
guessed at.

**Right-sizing.** The design carries instruction streams for several batch
sizes, so a 3-text request runs a 4-sequence encode rather than padding to
128. That is why a single text takes 15 ms and 512 texts take 558 ms.

### 4. Or embed a file directly

```cmd
npuembeddings embed bge-base-en-v1.5 texts.txt out.f32
```

One text per line in; `out.f32` is `[n_texts, hidden]` little-endian fp32,
L2-normalised, in input order — `hidden` is the model's width, shown by
`npuembeddings list`.

```python
import numpy as np
v = np.fromfile("out.f32", dtype=np.float32).reshape(-1, 768)   # bge-base
```

### Useful flags

Run it with no arguments to get this list plus the model table:

```
npuembeddings                       what this is, and what it can run
npuembeddings list
npuembeddings serve <model> [--port N] [--bind ADDR]
npuembeddings embed <model> <in.txt> [out.f32]

  --port N          listen port (default 8080)
  --bind ADDR       interface (default 127.0.0.1, localhost only)
  --threads N       host thread budget (default 24 for serve/embed)
  --pipeline N      concurrent encode lanes (default 2)
  --artifacts DIR   override the design set
  --root DIR        override where models/ and the design live
```

The flag form is unchanged and still carries the probes and benchmarks:

```
npuembeddings <root> --model NAME --artifacts DIR --serve [port]
```

## What else you need

### To run a release

| | |
|---|---|
| **A Ryzen AI machine** | XDNA2 NPU — Strix Point (Ryzen AI 300 series) or newer |
| **The AMD NPU driver + XRT** | Ships with the [AMD Ryzen AI Software](https://ryzenai.docs.amd.com/en/latest/inst.html) install. The runtime loads `xrt_coreutil.dll` from it. |
| **MSVC 2015–2022 redistributable** | [Microsoft download](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) — most machines already have it |

Check the NPU is present and idle:

```cmd
C:\Windows\System32\AMD\xrt-smi.exe examine
```

> `XILINX_XRT` must **not** be set in your environment. It breaks Windows XRT
> builds. Use `XRT_ROOT` if you need to point at an install.

### To build from source

You additionally need **MLIR-AIE (IRON)**, which is what compiles the AIE
kernels into the `.xclbin` the runtime loads, plus the Peano LLVM-AIE compiler
and a C++ toolchain. Installation is involved and is documented upstream — we
link to it rather than duplicating it.

**→ See [BUILD.md](BUILD.md)** for the full build, the model preparation steps,
and the verification suite.

### The model

Not shipped, by choice — see step 3 above. `npuembeddings serve <model>`
fetches, verifies and prepares it on first use, and building from source does
the same thing through [BUILD.md](BUILD.md)'s pipeline.

Redistributing it would have been permitted, since the model is Apache-2.0. It
would also have been the worse deal: a 66 MB blob in a stranger's zip that
nobody can practically check against the original, going stale the moment
upstream changes, and hiding the model card. A checksummed download from the
source is both smaller and more honest.

## How it works, briefly

```
text ──► WordPiece tokenizer ──► embedding gather ──┐
                                                     │  (C++, host)
        ┌────────────────────────────────────────────┘
        ▼
   6 × encoder layer
     ├── QKV / attn-out / FFN-up / FFN-down GEMMs ──► NPU  (bf16, fp32 accum)
     ├── attention per head ──────────────────────►  host (AVX2)
     └── LayerNorm / softmax / GELU ──────────────►  host (fp32, AVX2)
        │
        ▼
   mean pool ──► L2 normalise ──► 384-dim vector
```

Two design decisions carried most of the performance, and both came out of
measurement rather than intuition:

1. **One xclbin, many instruction streams.** Changing which design the NPU is
   configured for costs 2.2–2.6 ms — far more than a dispatch. By moving the
   only shape-dependent values into runtime parameters, all four GEMM shapes
   *and* all four batch sizes share one static design and one hardware context,
   so an encode pays **zero** switch cost. That alone took the encode from 305
   to 611 seq/s.
2. **Two encode lanes over one design.** The NPU serialises dispatches
   regardless, so one lane's host work overlaps another's NPU work — measured
   1.49× — for 833 seq/s.

The full reasoning, including the routes that did *not* work, is in
[`tasks/`](tasks/README.md).

## About this repository

This is the public subset of a larger working repository, assembled by
[`tools/sync_public_repo.py`](tools/sync_public_repo.py). What is not here is
an indexed literature review: summaries of the published work on AIE
programming that the project read as it went, and the papers themselves. Those
summaries were written to be complete enough to use instead of the papers,
which makes them exactly the thing not to republish.

Everything else ships, including the task logs that mention them. So when a
document refers to `research/papers/` or `OthersResarch/`, it is describing
that private material — the papers themselves are cited by arXiv id and linked
to arxiv.org wherever a claim depends on one.

## Repository layout

```
runtime/        the product: C++ + XRT, no Python
  include/      npue reader, XRT dispatch, tokenizer, http
  src/          main.cpp is the encoder, the server and the probes
experiments/    IRON designs and kernels (Python at build time only)
  m5-eltwise/   GELU, LayerNorm, softmax kernels (.cc) and their designs
  m5-pretiled-gemm/  the production GEMM design
tools/          build-time: pack the model, export designs, verify everything
docs/           how it works, and the measurement rules
tasks/          what happened, day by day, failures included
reference/      the numpy oracle everything is validated against
```

## Acknowledgements

Built on [MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie) and
[XRT](https://github.com/Xilinx/XRT).

Sixteen papers, theses and write-ups shaped this project's architecture, and
three of them changed the design outright.
[`docs/literature.md`](docs/literature.md) records **which source led to which
decision** — including the three places our own measurements ended up
disagreeing with published guidance.

### Related work

[jyatesdotdev/npu-embeddings](https://github.com/jyatesdotdev/npu-embeddings)
takes the same idea — all-MiniLM on XDNA2 via MLIR-AIE — down a different
path: INT8 rather than bf16, Linux rather than native Windows, and Strix Halo
rather than Strix Point. Worth reading alongside this one; the two make
different trades and neither is the obvious answer.

### Licensing

This repository is **Apache-2.0**. Five files began as MLIR-AIE examples and
remain **Apache-2.0 WITH LLVM-exception**, keeping their original copyright
headers and stating what was changed — one of them is a near-verbatim copy of
AMD's GELU kernel, vendored on purpose as a control experiment.

[`THIRD-PARTY.md`](THIRD-PARTY.md) lists them, and it is *generated* by
[`tools/audit_third_party.py`](tools/audit_third_party.py) rather than
maintained by hand: attribution rots as files are rewritten, so the audit
measures the longest contiguous run of shared code against an mlir-aie
checkout and fails if anything substantial lacks a header. Everything else
here is original work.

This is an independent implementation and is not affiliated with or endorsed
by AMD.
