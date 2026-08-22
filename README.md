# NpuEmbeddings

Sentence embeddings on the **XDNA2 NPU** in AMD Ryzen AI processors, on native
Windows. The AI Engine kernels are written directly against the array with
[MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie) — not through ONNX
Runtime, not through a vendor overlay — and the shipped runtime is C++ and XRT.

Six models, three architectures. An OpenAI-compatible endpoint, one executable,
no Python at runtime.

**The point is not to beat the CPU.** It is to get embedding work *off* the CPU
cores — a background job behind a search index should not take the machine
hostage — without losing throughput or quality.

This is a learning project, openly. Every number has a task log with the exact
command and the stored artifact behind it, several conclusions in here were
overturned by later measurements and the reversals are kept in place, and the
open questions are written down rather than tidied away. **Issues, ideas and
code are all very welcome** — see [Contributing](#contributing).

---

## Install

Grab the latest `npuembeddings-*-win-x64.zip` from
[**Releases**](../../releases) and unzip it anywhere. No installer, no Python,
nothing written outside the folder.

You need:

| | |
|---|---|
| **A Ryzen AI machine** | XDNA2 NPU — Strix Point (Ryzen AI 300 series) or newer |
| **AMD NPU driver + XRT** | ships with [AMD Ryzen AI Software](https://ryzenai.docs.amd.com/en/latest/inst.html); the runtime loads `xrt_coreutil.dll` from it |
| **MSVC 2015–2022 redistributable** | [download](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) — most machines already have it |

Check the NPU is present and idle:

```cmd
C:\Windows\System32\AMD\xrt-smi.exe examine
```

> `XILINX_XRT` must **not** be set in your environment — it breaks Windows XRT
> builds. Use `XRT_ROOT` if you need to point at an install.

**Building from source** additionally needs MLIR-AIE (IRON), the Peano LLVM-AIE
compiler and a C++ toolchain. → **[BUILD.md](BUILD.md)**

## Run

```cmd
npuembeddings list
npuembeddings serve bge-base-en-v1.5
```

The first run fetches the weights — **they are not redistributed here** —
verifies them against a checksum built into the executable, cross-checks the
model's own `config.json`, and packs everything into one `models/<name>.npue`.
A checksum or config mismatch **stops** rather than running weights nobody
verified. All of it happens inside the executable: no `curl`, no download
script.

Then use any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
r = client.embeddings.create(
    model="bge-base-en-v1.5-npu",
    input=["a man is playing a guitar", "someone plays guitar at a concert"],
)
print(len(r.data[0].embedding))     # 768
```

Supported: `POST /v1/embeddings` (`input` as string or array, `encoding_format`
`float` or `base64`), `GET /v1/models`, `GET /health`. Requests carrying
token-id arrays instead of text are rejected rather than guessed at.

Or embed a file directly — one text per line in, `[n_texts, hidden]`
little-endian fp32 out, L2-normalised, in input order:

```cmd
npuembeddings embed bge-base-en-v1.5 texts.txt out.f32
```

```python
import numpy as np
v = np.fromfile("out.f32", dtype=np.float32).reshape(-1, 768)
```

### Which model

| model | hidden | layers | notes |
|---|---:|---:|---|
| `all-MiniLM-L6-v2` | 384 | 6 | smallest and fastest |
| `bge-small-en-v1.5` | 384 | 12 | MiniLM's width, twice the depth |
| `bge-base-en-v1.5` | 768 | 12 | **the geometry that fits this NPU best** — a good default |
| `bge-large-en-v1.5` | 1024 | 24 | highest quality, slowest |
| `nomic-embed-text-v1.5` | 768 | 12 | RoPE + gated SwiGLU; **needs `--prefix`** |
| `embeddinggemma-300m` | 768 | 24 | host-only, no NPU kernel yet; gated, needs `HF_TOKEN` |

`nomic-embed-text-v1.5` requires a task prefix — `--prefix search_document` for
documents, `--prefix search_query` for queries. Getting it wrong costs retrieval
quality and no similarity check can detect it, so the runtime always prints
which prefix it applied and refuses an unknown one.

### Flags

```
npuembeddings                       what this is, and what it can run
npuembeddings list
npuembeddings serve <model> [--port N] [--bind ADDR] [--prefix NAME]
npuembeddings embed <model> <in.txt> [out.f32] [--prefix NAME]

  --port N          listen port (default 8080)
  --bind ADDR       interface (default 127.0.0.1, localhost only)
  --threads N       host thread budget (default 24)
  --pipeline N      concurrent encode lanes (default 4)
  --prefix NAME     task prefix, for models that use one
  --artifacts DIR   override the design set
  --root DIR        override where models/ and the design live
  --token VALUE     HuggingFace token for gated repos (or set HF_TOKEN)
```

A single text takes ~15 ms: the design carries instruction streams for several
batch sizes, so a 3-text request runs a 4-sequence encode rather than padding
to 128.

## Performance

Ryzen AI 9 HX 370, batch 128, sequence 64. All three encoders measured
**interleaved** in one session — round-robin in one process, same statistic on
every side — because wall clock on a shared machine drifts enough that numbers
taken minutes apart compare the machine, not the code.

| model | **NPU** | torch | ONNX Runtime | **NPU / best CPU** | energy |
|---|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | **942** | 509 | 245 | **1.85×** | **3.65×** better |
| bge-small-en-v1.5 | **494** | 309 | 134 | **1.60×** | **3.00×** |
| bge-base-en-v1.5 | **211** | 84.5 | 41.3 | **2.49×** | **3.33×** |
| bge-large-en-v1.5 | **60.7** | 24.6 | 11.2 | **2.47×** | **3.28×** |
| nomic-embed-text-v1.5 | **165** | 73.0 | 44.0 | **2.26×** | **3.75×** |

sequences per second; energy is joules per 1000 sequences, measured by the
differential RAPL method.

**The number worth looking at is not in that table.** Measured on an idle
machine, the same models run at 951 / 494 / 211 / 60.8 / 166 seq/s — so the NPU
column above, taken while torch and ONNX Runtime saturated all twelve cores,
is **within 1.5% of the idle figure on every model**. The work really is off
the CPU.

**And the embeddings are the same embeddings.** Every shipping model is gated
against a from-scratch numpy oracle, end to end against `sentence-transformers`,
and on MTEB with both sides run in one session:

| model | worst `1 − cos` | top-10 overlap | MTEB Δ |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 2.644e-05 | 1.0000 | +0.03 |
| bge-small-en-v1.5 | 3.022e-05 | 1.0000 | −0.03 |
| bge-base-en-v1.5 | 2.613e-05 | 1.0000 | +0.02 |
| bge-large-en-v1.5 | 3.801e-04 | 1.0000 | +0.05 |
| nomic-embed-text-v1.5 | 2.401e-05 | 1.0000 | +0.09 |

The `1 − cos` gate is 2e-03, so every model sits two orders of magnitude inside
it; the MTEB gate is ±0.5 points against the same checkpoint on CPU. Three of
those five MTEB runs were the first that model had ever had.

**→ [docs/06-performance.md](docs/06-performance.md) has the caveats**, and
they matter more than the table: how the numbers were taken, which of them
reproduce and which do not, and what we could not explain. The short version is
that NPU throughput and accuracy are solid, and the CPU ratio is indicative to
about ±20% because the CPU side moves more between sessions than the difference
anyone is arguing about.

## Contributing

**Issues, ideas and code are the most useful things anyone can bring.** Open an
issue for anything — a question, a result that looks wrong, a machine where it
does not work, a direction worth trying.

If you want somewhere to start,
[`research/OPEN-THREADS.md`](research/OPEN-THREADS.md) is every question this
project has written down and not answered, with a status, ordered by what it
would change. Threads leave that file only by being answered, retired or
superseded — never by being quietly forgotten. Some need this exact hardware;
several do not:

- **The array runs at ~100% of the *fp32* vector datapath's limit while the
  MMAC unit sits idle** (T16). The remaining multi-× levers are all datapath
  changes — bfp16 emulation, or int8 with its native `(8,8,8)` MAC dims — and
  both are accuracy decisions rather than engineering ones.
- **Layer fusion** (T28). Our cost is per dispatch and the CPU has no such
  term, which is why the advantage shrinks with depth. The mechanism is built
  at small scale and stuck at production scale.
- **The on-device golden gate is blind to an entire bug class** (T32) — it
  tiles one batch-4 corpus, so a bug reading the *wrong row* cannot change the
  answer. It passed a real threaded data race this month. Three cheap fixes are
  listed; none is done.

If you disagree with a measurement, the task logs record the command and the
artifact for every number, so disagreements can be settled rather than argued.

## Where things live

```
runtime/        the product: C++ + XRT, no Python
experiments/    IRON designs and AIE kernels (Python at build time only)
tools/          pack the model, export designs, verify everything
docs/           how it works, and the measurement rules
tasks/          what happened, day by day, failures included
reference/      the numpy oracle everything is validated against
research/       open questions, prior art
```

- **[docs/00-overview.md](docs/00-overview.md)** — how it works and why it is
  built this way
- **[docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md)** — what runs today, what
  does not, what was tried and failed
- **[docs/05-measurement/](docs/05-measurement/README.md)** — the measurement
  doctrine, including why wall clock is never an NPU performance claim
- **[tasks/](tasks/README.md)** — the day-by-day log; the failures are the
  valuable part

This repository is the public subset of a larger working one, assembled by
[`tools/sync_public_repo.py`](tools/sync_public_repo.py). What is not here is an
indexed literature review — summaries written to be usable *instead of* the
papers, which makes them exactly the thing not to republish. Everything else
ships, so a document referring to `research/papers/` is describing that private
material; the papers themselves are cited by arXiv id.

## Related work

[jyatesdotdev/npu-embeddings](https://github.com/jyatesdotdev/npu-embeddings)
takes the same idea down a different path: INT8 rather than bf16, Linux rather
than native Windows, Strix Halo rather than Strix Point. Worth reading alongside
this one; the two make different trades and neither is the obvious answer.

## Licence

**Apache-2.0.** Five files began as MLIR-AIE examples and remain Apache-2.0
WITH LLVM-exception, keeping their original headers and stating what changed.
[`THIRD-PARTY.md`](THIRD-PARTY.md) lists them and is *generated* by
[`tools/audit_third_party.py`](tools/audit_third_party.py) rather than
maintained by hand — attribution rots as files are rewritten, so the audit
measures shared code against an mlir-aie checkout and fails if anything
substantial lacks a header.

This is an independent implementation, not affiliated with or endorsed by AMD.
