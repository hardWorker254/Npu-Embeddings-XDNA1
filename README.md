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
design, and 0.4.0 ships two for nomic (`--seq 256`, `--seq 512`).

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
model that has no prompts (the four BERT models) is also a 400 — a client
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

| model | datapath | **NPU** | vs 0.3.0 ¹ | **NPU / best CPU** | energy ² | worst `1 − cos` | MTEB Δ mean / worst ³ |
|---|---|---:|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | bfp16 | **1459** | 1.55× | **1.876×** | **3.10×** better | 3.406e-04 | +0.12 / −0.07 |
| `bge-small-en-v1.5` | bf16 | **630** | 1.28× | 1.561× | 2.73× | 8.348e-06 | −0.10 / **−0.5010** |
| `bge-base-en-v1.5` | bfp16 | **324** | 1.54× | **2.714×** | 4.17× | 2.284e-04 | −0.06 / −0.19 |
| `bge-large-en-v1.5` | bfp16 | **94.8** | 1.56× | 2.633× | **4.79×** | 2.626e-04 | +0.13 / −0.01 |
| `nomic-embed-text-v1.5` | bfp16 | **259** | 1.57× | **3.447×** | 4.01× | 1.402e-03 | +0.01 / −0.25 |
| `embeddinggemma-300m` | bfp16 | **108** | new ⁴ | 1.303× | 3.72× | 1.749e-04 ⁵ | +0.16 / −0.02 |

¹ against 0.3.0's published figure, **same harness, same stage, same statistic** —
it is the one release-over-release comparison in this table that is like for
like. ² joules per 1000 sequences, by the differential RAPL method; carried from
the run that measured it, not re-measured here. ³ from the gate run that decided
bfp16 adoption per model; still valid because everything since has been
bit-identical. ⁴ `embeddinggemma-300m` ran host-only in 0.3.0, so there is no NPU
figure to compare against. ⁵ a **differential** against the host-only path, not a
comparison against HuggingFace — arch 1 has no reference golden, which makes this
a weaker claim than the other five.

Sequences per second, end to end, wall clock — **that is a throughput figure,
not an NPU kernel performance claim**; per-kernel numbers in this project come
from hardware traces only. Energy is joules per 1000 sequences by the
differential RAPL method. The `1 − cos` gate is 2e-03 and the MTEB gate is
`|mean| ≤ 0.5` with no task worse than −0.5.

**The `datapath` column is read from the runtime's own status line**, not
restated from a table — a design and the intention behind it are different
things, and only one of them is evidence.

**Two rows are worth stopping on.** `bge-small-en-v1.5` is the one model still on
plain bf16: it failed the bfp16 accuracy gate at **−0.5010** against a −0.5 line,
bit-reproducibly, so it did not get the faster datapath. We did not round that
away to tidy the column. And `embeddinggemma-300m`'s `1 − cos` is a
**differential** against the host-only path rather than against HuggingFace,
because arch 1 has no reference golden — a weaker claim than the other five, and
labelled as one.

**How much of this reproduces.** The sweep was run twice, twenty minutes apart
on the same idle machine. **NPU throughput repeated to within 0.2% on every
model**; the CPU baselines moved 3–4%. That is the ±20% caveat below, measured
rather than asserted — and it is why the ratio column, not the NPU column, is
the one to treat as indicative.

Two contributions to the gain are separable and were each measured in one
session: the bfp16 datapath adoption was **+7.3% to +20.5%** sweep-against-sweep
(with `bge-small`, whose datapath did not change, flat at −2.2% as the control),
and the host-side epilogue fusion on top of it was **1.153×–1.340×**,
bit-identical.

**→ [docs/06-performance.md](docs/06-performance.md) has the caveats**, and they
matter more than the table: how the numbers were taken, which reproduce and which
do not, and what we could not explain. The short version is that NPU throughput
and accuracy are solid, and the CPU ratio is indicative to about ±20% because the
CPU side moves more between sessions than the difference anyone is arguing about.

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
  encoder released since 2024. We ship WordPiece and Gemma's SentencePiece BPE
  and nothing byte-level, which blocks the whole ModernBERT, Qwen and Mistral
  family at once. Gemma's merge machinery is reusable; the new part is a GPT-2
  pretokenizer. **This one needs no NPU** — it is a tokenizer and a byte-exact
  test against HuggingFace.
- **Which encoder joins the catalogue next**
  ([T44](research/OPEN-THREADS.md#t44)) — four candidates are already priced
  against this project's own geometry gates, with the arithmetic shown.
- **Fold attention onto the array for long-sequence designs**
  ([T42](research/OPEN-THREADS.md#t42)) — filed with a price and an explicit
  trigger. The geometry blocker is gone (mem-tile padding is proven exact on
  all 8 columns); what is not settled is whether it is worth it.
All three are filed with a price and a trigger rather than as open questions
about the hardware — the register ran to 20 threads a week ago and 43 of the 46
are now closed, each by a measurement or a build rather than by a decision to
stop caring. **How** they were settled is in
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
