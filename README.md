# FORK DETAILS

I have added support for NPU1 and Linux. Some command are not working, but ./runtime/build/npuembeddings.exe serve works. NPU1 can only load 1 .xclbin
Operations:
QKV projection NPU
Attention output NPU
FFN up NPU
FFN down NPU
LayerNorm CPU
Softmax CPU
GELU CPU

In the beginning you need to compile artifacts:

source /opt/xilinx/xrt/setup.sh # Or your cutom path
export PEANO_INSTALL_DIR=/home/prof/.local/lib/python3.14/site-packages/llvm-aie # Or your path to llvm-aie
export XRT_INCLUDE_DIR=/opt/xilinx/xrt/include

python tools/export_gemm_rtp.py --batch 16 --batches 4,8,16 --cols 4     --hidden 768 --intermediate 3072 --gated-ffn --qkv-n 2304     --emulate-bfp16 --c-bf16     --out runtime/artifacts_npu1 # For gte-multilimgual-base

python tools/export_gemm_rtp.py --batch 16 --batches 4,8,16 --cols 4     --emulate-bfp16 --c-bf16 --out runtime/artifacts_npu1 # For all-MiniLM-L6-v2

cd runtime
cmake --build build --config Release

#Run the model
./runtime/build/npuembeddings.exe serve gte-multilingual-base --root . --artifacts runtime/artifacts_npu1 --threads 16 --pipeline 4


You will see something like this:
NpuEmbeddings C++ runtime -- full encode
  bo-mode    host_only (data-buffer allocation)
  model      gte-multilingual-base: 150 tensors, 1000.91 MB, checkpoint f5a35a10faa54da7
  shape      Alibaba-NLP/gte-multilingual-base: 12 layers, hidden 768, 12 heads x 64, ffn 3072, CLS pooling
  designs    ONE xclbin, 12 streams (3 batch tiers), one hw_context
  datapath   bfp16-emulated MMAC, C as bf16
  toolchain  mlir_aie 1.4.2, peano 22.0.0.2026090701+3e93bf7b, mlir-aie HEAD unavailable
  shape      batch 16 x seq 64  (M = 1024)
  tiers      4, 8, 16  (requests are right-sized, not padded)
  gelu       on the HOST (fp32) -- 12 fewer NPU dispatches
  softmax    on the HOST (fp32) -- 12 fewer NPU dispatches
  layernorm  on the HOST (fp32) -- 25 fewer NPU dispatches
  bo-align   last data buffer aligned to 16384 B
  weights    226.49 MB staged on the device once, not per call
  pipeline   4 concurrent encodes of 16, one NPU mutex, 4 host threads per lane
  tokenizer  250002 tokens, from the .npue

  serving http://127.0.0.1:8080/v1/embeddings   (model gte-multilingual-base-npu, seq 64)
  POST {"input": "text" | ["a","b"], "encoding_format": "float"|"base64"}


# NpuEmbeddings

Sentence embeddings on the **XDNA2 NPU** in AMD Ryzen AI processors, on native
Windows. The AI Engine kernels are written directly against the array with
[MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie) — not through ONNX
Runtime, not through a vendor overlay — and the shipped runtime is C++ and XRT.

Seven models, four architectures — including a multilingual one. An
OpenAI-compatible endpoint, one executable, no Python at runtime.

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
token-id arrays instead of text are rejected rather than guessed at, and so is
any input longer than the design's sequence length — see
[Input length](#input-length--read-this-before-embedding-documents), which you
will hit on the first real document.

Or embed a file directly — one text per line in, `[n_texts, hidden]`
little-endian fp32 out, L2-normalised, in input order:

```cmd
npuembeddings embed bge-base-en-v1.5 texts.txt out.f32
```

```python
import numpy as np
v = np.fromfile("out.f32", dtype=np.float32).reshape(-1, 768)
```

### Input length — read this before embedding documents

**The shipped designs run at a sequence length of 64 tokens, and an input
longer than that is refused rather than shortened.**

```
$ curl ... -d '{"input": "<a 123-token document>"}'
HTTP 400
{"error":{"message":"input 0 is 123 tokens, but this runtime is running at
sequence length 64. Split the text into shorter pieces, or run a design
exported for a longer sequence (tools/export_gemm_rtp.py --seq N)...",
"type":"invalid_request_error"}}
```

That refusal is deliberate, and it is newer than the cap. Until `tasks/0110`
the runtime **quietly embedded the first 64 tokens instead**, which is a much
worse failure than an error: a truncated text still returns a
correctly-shaped, correctly-normed, perfectly deterministic vector, so nothing
downstream can tell that the answer is wrong. The pathological case is
documents that share a preamble — cut them all at the same point and their
vectors become **byte-identical**, and retrieval between them degrades to a
coin flip wearing a similarity score. We measured exactly that on real
municipal records before fixing it.

So: **chunk your text on the caller's side**, embed the pieces, and mean-pool
plus renormalise if you need one vector per document. Roughly 64 tokens is
~140–160 characters of English or Norwegian prose; measure against your own
corpus rather than trusting that ratio.

`--allow-truncation` restores the old cut-and-continue behaviour if you
genuinely want it. It warns once per run on stderr and its vectors mean "the
first 64 tokens of this text", not "this text".

**Why 64, and what longer costs — now measured.** Sequence length is not a
hardware limit and it is not baked into the kernels. It enters a design only as
`M = batch × seq`, and the instruction streams know just `M`, `K` and `N`, so
`batch 128 × seq 64` and `batch 16 × seq 512` are the same `M = 8192` and the
same array work. `tools/export_gemm_rtp.py --seq N` builds a long-sequence
design; the seq-256 and seq-512 sets serve both `nomic-embed-text-v1.5` and
`gte-multilingual-base` (they share their array geometry exactly, which is
why the multilingual model cost zero new NPU designs).

They were exported and measured. **Across an 8× sequence range the array does
not move: −2.0% on identical `M` and identical dispatch counts.** Everything
longer sequences cost is host-side, because attention runs on the host at
O(seq²) per sequence — O(seq) per token — while array work at constant `M` is
flat:

| nomic, constant `M = 8192` | seq 64 | seq 256 | seq 512 |
|---|---:|---:|---:|
| array, share of wall clock | **76.3%** | 62.0% | **46.6%** |
| host attention, share of wall | 16.5% | 33.6% | **52.1%** |
| cost per token | 1.000× | 1.221× | **1.605×** |

**Host attention overtakes the whole array at seq ≈ 470.** So the widely-quoted
"attention is only 2–5% of the work" is a **seq-64 number and must not be
quoted above it** — that is the single most useful thing to know before asking
for long sequences.

Two practical caveats if you build one. The 256 cap on the BERT-family models
is the container's `max_seq_len` **config field**, not the position table, so a
repack is needed above 256 like everything else. And a long-sequence design is
a **bad instrument for short requests**: tier selection rounds up, so the
smallest tier's token footprint is what bites — measured at **5.8×** for a
single short text on a seq-512 design. Run the design that matches your
traffic.

### Which model

| model | hidden | layers | notes |
|---|---:|---:|---|
| `all-MiniLM-L6-v2` | 384 | 6 | smallest and fastest |
| `bge-small-en-v1.5` | 384 | 12 | MiniLM's width, twice the depth |
| `bge-base-en-v1.5` | 768 | 12 | **the geometry that fits this NPU best** — a good default |
| `bge-large-en-v1.5` | 1024 | 24 | highest quality, slowest |
| `nomic-embed-text-v1.5` | 768 | 12 | RoPE + gated SwiGLU; **needs a task prompt** |
| `embeddinggemma-300m` | 768 | 24 | MQA + RoPE + GeGLU; needs a task prompt; gated, needs `HF_TOKEN` |
| `gte-multilingual-base` | 768 | 12 | **multilingual** (XLM-R vocab, 70+ languages); RoPE + gated GELU; no prompt |

### Task prompts

`nomic-embed-text-v1.5` and `embeddinggemma-300m` need a **task prompt** —
`search_document` for documents and `search_query` for queries on nomic, one of
14 sentence-transformers names on Gemma. Getting it wrong costs retrieval
quality and **no similarity check can detect it**: a wrongly-prompted embedding
comes back correctly shaped, correctly normed and deterministic. So the runtime
never picks one for you.

**Serving, it is a per-request field.** A RAG deployment needs queries and
documents embedded differently in the same session, so the choice cannot be a
process-wide flag:

```jsonc
POST /v1/embeddings
{ "input": ["what is XDNA2?"], "prompt_name": "search_query" }

// omit it on a model that has prompts and you get, instead of a guess:
// 400 {"error":{"message":"this model requires 'prompt_name'; valid names:
//      [classification, clustering, search_document, search_query], ...",
//      "type":"invalid_request_error"}}
```

`GET /health` lists the names, so a client never has to provoke the 400 to
learn them. `"prompt_name": ""` means no prompt at all. Passing the field to a
model that has no prompts (the four BERT models, and `gte-multilingual-base`)
is also a 400 — a client
sweeping one config across the whole catalogue should break loudly rather than
quietly return unprompted vectors.

OpenAI's own `/v1/embeddings` has no such field, because its embedding models
are symmetric. Among servers that host asymmetric models the name is not
standardised — Cohere requires `input_type`, vLLM accepts `input_type`, and
HuggingFace TEI uses `prompt_name` on its native route. This project uses
`prompt_name` because EmbeddingGemma's table holds sentence-transformers names
like `STS` and `BitextMining`, which are not "input types".

**Batch (`embed`) still takes `--prefix NAME`**, because a file of texts has no
per-request anything. It is **required** there for a model with prompts — there
is no default — and `--prefix ""` means none. `serve` rejects the flag.

### Flags

```
npuembeddings                       what this is, and what it can run
npuembeddings list
npuembeddings serve <model> [--port N] [--bind ADDR]
npuembeddings embed <model> <in.txt> [out.f32] [--prefix NAME]

  --port N          listen port (default 8080)
  --bind ADDR       interface (default 127.0.0.1, localhost only)
  --threads N       host thread budget (default 24)
  --pipeline N      concurrent encode lanes (default 4)
  --prefix NAME     task prompt, for `embed` only and REQUIRED for a
                    model that has one. `serve` rejects it -- send
                    "prompt_name" per request instead
  --allow-truncation
                    embed the first 64 tokens of an over-long input
                    instead of refusing it. Off by default; see
                    "Input length" above for why that default is
                    the safe one
  --artifacts DIR   override the design set
  --root DIR        override where models/ and the design live
  --token VALUE     HuggingFace token for gated repos (or set HF_TOKEN)
```

A single text takes ~15 ms: the design carries instruction streams for several
batch sizes, so a 3-text request runs a 4-sequence encode rather than padding
to 128.

## Performance

Ryzen AI 9 HX 370, sequence 64. One whole-catalogue sweep, one machine state,
one protocol — because a row-by-row patchwork misleads even when no row lies.
CPU comparisons are measured **interleaved in one session**, round-robin in one
process, because wall clock on a shared machine drifts enough that numbers taken
minutes apart compare the machine rather than the code.

| model | datapath | **NPU** | vs 0.4.0 ¹ | **NPU / best CPU** | energy ² | worst `1 − cos` | **p99 tail** ³ | MTEB Δ mean / worst ⁴ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | bfp16 | **1461** | 1.00× | **1.855×** | 3.42× better | 3.406e-04 | 9.7e-04 | +0.12 / −0.07 |
| `bge-small-en-v1.5` | bf16 | **630** | 1.00× | 1.589× | 2.64× | 8.348e-06 | 1.2e-05 | −0.10 / **−0.5010** |
| `bge-base-en-v1.5` | bfp16 | **324** | 1.00× | **2.713×** | 4.52× | 2.284e-04 | 3.7e-04 | −0.06 / −0.19 |
| `bge-large-en-v1.5` | bfp16 | **95.2** | 1.00× | 2.607× | 4.89× | 2.626e-04 | **6.6e-03** ⁵ | +0.13 / −0.01 |
| `nomic-embed-text-v1.5` | bfp16 | **262** | 1.00× | **3.494×** | 6.55× | 1.402e-03 | 1.6e-03 | +0.01 / −0.25 |
| `embeddinggemma-300m` | bfp16 | **181** | 1.01× | 1.271× | 3.72× ⁶ | PASS ⁷ | 4.2e-04 | +0.16 / −0.02 |
| `gte-multilingual-base` | bfp16 | **255** | new | — ⁸ | — ⁸ | 4.608e-04 | 5.2e-04 | **+0.06 / −0.06** ⁹ |

¹ against 0.4.0's sweep, **same harness, same stage, same statistic**. The six
carried-over models reproducing 0.4.0 to within ±1% is itself the release's
regression check passing: 0.5.0 changed no array code for them, and the column
says so. ² joules per 1000 sequences, differential RAPL, **re-measured this
sweep**; the ratio moves with the CPU side, which drifts more between sessions
than the NPU side does (nomic read 4.01× in 0.4.0 and 6.55× here — treat the
column as indicative, not as a constant of nature). ³ **new in 0.5.0**: p99 of
per-text `1 − cos` over 224 varied inputs, single words included — the gate
that sees what a mean hides. ⁴ from the symmetric gate runs (bfp16 adoption for
the six, task 0137 for gte); still valid because everything since is
bit-identical, verified by hash. ⁵ bge-large's tail is real, ~100× its median,
carried as a register-linked waiver (T51) rather than rounded away — its
median-regime accuracy is the 2.626e-04 column. ⁶ carried from 0.4.0: this
sweep's gemma NPU energy arm produced a degenerate differential (Δt ≈ 0
between the low and high runs — the harness limitation is recorded in the task
log) and a wrong number reported confidently is worse than a carried one
labelled. ⁷ arch 1 has no HF-reference golden; its gate is differential
against the host-only control. ⁸ **deliberately absent**: both need a CPU
reference arm, and gte's `trust_remote_code` model is unusable without buffer
repairs (an unrepaired run is silently position-scrambled) — so the NPU figure
stands alone and labelled, per this project's measurement rules. ⁹ the
cleanest MTEB verdict in the catalogue, clustering cell positive; multilingual
STS evidence (29 language subsets within [−0.30, +0.20]) is in the release
note.

Sequences per second, end to end, wall clock — **that is a throughput figure,
not an NPU kernel performance claim**; per-kernel numbers in this project come
from hardware traces only. The `1 − cos` gate is 2e-03, the tail gate is p99
against a per-model ceiling, and the MTEB gate is `|mean| ≤ 0.5` with no task
worse than −0.5.

**The `datapath` column is read from the runtime's own status line**, not
restated from a table — a design and the intention behind it are different
things, and only one of them is evidence.

**Rows worth stopping on.** `bge-small-en-v1.5` stays on plain bf16: it failed
the bfp16 accuracy gate at **−0.5010** against a −0.5 line, bit-reproducibly,
and we did not round that away. `bge-large-en-v1.5` carries the catalogue's
one real accuracy tail — same five single words atop the tail on two
arithmetically unrelated datapaths, so it is a property of the model × input,
not of the number format (task 0129). And `gte-multilingual-base` runs on the
**same NPU design set as nomic** — its geometry matches bit for bit, so the
seventh model cost zero new array designs; every cost was host-side (a third
tokenizer family and a fourth encoder architecture).

**And the memory system is measured now, not inferred.** The bandwidth roof is
**45.5 GB/s on the array's own clock** (trace-derived dispatch time agreeing
with wall clock to 0.4%), four instruments concur, and the mem-tile↔L1 leg is
two-thirds idle while the cores wait on locks — the DRAM leg is the one that
binds. B-reuse, the biggest priced lever, re-priced on those measurements at
1.72× array / 1.31× end-to-end and is parked behind a funded C-join re-plumb.

**→ [docs/06-performance.md](docs/06-performance.md) has the caveats**, and
they matter more than the table: how the numbers were taken, which reproduce
and which do not, and what we could not explain.

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

- **A byte-level BPE tokenizer**
  ([T43](research/OPEN-THREADS.md#t43)) — this is the gate on nearly every
  encoder released since 2024. We ship WordPiece, Gemma's SentencePiece BPE
  and (since 0.5.0) XLM-R's SentencePiece Unigram, but nothing byte-level,
  which blocks the whole ModernBERT, Qwen and Mistral family at once. The
  Unigram build is the template: a table generator, a C++ port, and a
  byte-exact test against HuggingFace — it went from filed to 343/343 in a
  day. **This one needs no NPU.**
- **Which encoder joins the catalogue next**
  ([T44](research/OPEN-THREADS.md#t44)) — four candidates are already priced
  against this project's own geometry gates, with the arithmetic shown.
- **Fold attention onto the array for long-sequence designs**
  ([T42](research/OPEN-THREADS.md#t42)) — filed with a price and an explicit
  trigger. The geometry blocker is gone (mem-tile padding is proven exact on
  all 8 columns); what is not settled is whether it is worth it.
- **Does the NPU have a case for LLM *decode* at all?**
  ([T55](research/OPEN-THREADS.md#t55)) — a side workstream took the same
  hardware knowledge to a decoder (`granite-4.2-3B`, a W4A16 GEMV kernel
  written from scratch) and got a **negative**: a 24-thread AVX2 CPU baseline
  is 2.4× faster, at 89% of its own memory bandwidth. Decode is bandwidth-bound
  and both devices sustain ~47 GB/s. The two cases left are **energy** and
  **prefill**, and neither has been measured. None of that code is in this
  repository, but the question and the numbers are.

Those are filed with a price and a trigger rather than as open questions
about the hardware — the register ran to 20 threads a fortnight ago and 51 of
the 60 are now closed, each by a measurement or a build rather than by a
decision to stop caring. **How** they were settled is in
[`research/CLOSED-THREADS.md`](research/CLOSED-THREADS.md), verbatim, because
the refuted claims and the measurements that killed them are the valuable part.

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
