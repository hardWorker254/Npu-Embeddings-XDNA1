# NpuEmbeddings — overview

Run **embedding models** (encoder-only transformers — BERT-style: all-MiniLM, bge-small,
e5, gte) very fast on the **AMD Ryzen AI NPU**, from hand-written AIE kernels, on
**native Windows**, with a C++ runtime.

Conceptually: *"FastFlowLM, but only embeddings."*

> **Where the project stands today** — what works, what does not, what was
> tried and failed, where code lives, and how to build it end to end:
> [`CURRENT_STATUS.md`](CURRENT_STATUS.md).

## Why

**The gap is real.** No open, from-scratch, hand-tiled C++ BERT engine exists on XDNA.
The closest prior art is AMD's ONNX-Runtime-VitisAI demo and FastFlowLM's *closed*
embedding support (which is a Gemma decoder, not a BERT encoder). AMD's own recommended
ONNX path has an open bug — `RyzenAI-SW` #312, *"BERT model hangs during VitisAI
compilation on Ryzen AI 9 HX 370"* — **on the exact SKU this project targets**.

**It is a learning project.** The goal is genuinely understanding the AIE array, not
assembling prebuilt parts. Consequently **documentation and traceability are
first-class deliverables**, not overhead. If a thing was learned and not written down,
the task is not finished.

## Where things live

| Path | What |
|---|---|
| [`01-hardware/`](01-hardware/README.md) | XDNA2 / AIE2P: array shape, memory, MACs/cycle, DMA constraints, realistic performance ceilings |
| [`02-toolchain/`](02-toolchain/README.md) | The native-Windows build flow: IRON, Peano, `aiecc`, XRT, the cache, and every Windows gotcha |
| [`03-kernels/`](03-kernels/README.md) | Per-kernel design notes and trace results |
| [`04-model/`](04-model/README.md) | all-MiniLM-L6-v2: architecture, tiling analysis, weight format, tokenizer, numerical landmines |
| [`05-measurement/`](05-measurement/README.md) | **How we are allowed to claim performance.** Read before quoting any number |
| [`../reference/`](../reference/README.md) | The fp32 numpy oracle and the golden vectors every kernel is validated against |
| [`../tools/`](../tools/README.md) | Build-time tooling: the `.npue` pre-tiled weight packer and its verifier |
| [`literature.md`](literature.md) | What we read and what we did with it — sources, and the decisions they drove |
| [`../research/notes/`](../research/notes/README.md) | Our own findings: kernel pitfalls, the silent arch fallback, the switch-cost model |
| [`../tasks/`](../tasks/README.md) | Dated log of every unit of work, with the exact commands run |

## Ground rules

These are project constraints, not preferences. Each has a reason.

1. **Native Windows, no WSL.** Consequence: **Peano** is the only kernel compiler;
   `xchesscc` needs Vitis, which has no Windows distribution.
2. **We build everything here.** FastFlowLM ships prebuilt kernel DLLs; we generate our
   own `.xclbin` and instruction streams from source, reproducibly.
3. **The product is C/C++.** Python is permitted at *build time* (IRON has no C++
   frontend) and for prototyping — never at runtime.
4. **Python prototyping is a legitimate milestone**, not a detour. A working Python
   encoder is worth banking before the C++ rewrite.
5. **Wall-clock time is never an NPU performance claim.** The NPU is shared and can be
   contended. All NPU numbers come from hardware traces or static instruction counts.
   See [`05-measurement/`](05-measurement/README.md).
6. **Every task is traceable** in [`../tasks/`](../tasks/README.md), including failures,
   with the exact commands run — the log alone should permit a rebuild from scratch.
7. **Never vendor FastFlowLM binaries.** Its `src/lib/**` and `src/xclbins/**` are
   closed and its own installer terms forbid redistribution. We read that repo for
   architecture only, and vendored nothing from it.

## What runs where

```
text ──► WordPiece tokenizer ──► embedding gather ──┐
                                                     │  (C++, host)
        ┌────────────────────────────────────────────┘
        ▼
   N × encoder layer
     ├── QKV / attn-out / FFN-up / FFN-down GEMMs ──► NPU  (bf16, fp32 accum)
     ├── attention per head ──────────────────────►  host (AVX2)
     └── LayerNorm / softmax / GELU ──────────────►  host (fp32, AVX2)
        │
        ▼
   pool ──► L2 normalise ──► vector
```

The elementwise operators live on the host because each measured *faster and
more accurate* there than as an NPU dispatch
([`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md)) — a design switch
costs ~2.3 ms, so no elementwise op at h=384 earns one. That is a statement
about dispatch economics, not about the array.

## Key decisions and why

Two of these carried most of the throughput, and both came out of measurement
rather than intuition:

| Decision | Rationale |
|---|---|
| **One xclbin, many instruction streams** | Changing which design the NPU is configured for costs **2.2–2.6 ms** — far more than a dispatch, and the encoder used to pay it on all 49 of them. Moving the only shape-dependent values into runtime parameters lets all four GEMM shapes *and* all four batch tiers share one static design and one hardware context, so an encode pays **zero** switch cost. 305 → 611 seq/s on its own. → [`0029`](../tasks/0029-m7-one-xclbin-probe/TASK.md), [`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md) |
| **Several encode lanes over one design** | The NPU serialises dispatches regardless, so one lane's host work overlaps another's NPU work. Measured 1.49× at two lanes; the default is now four. → [`0033`](../tasks/0033-m7-pipelined-lanes/TASK.md), [`0052`](../tasks/0052-m10-research-night/TASK.md) |
| **IRON Python at build time only** | MLIR-AIE has no C++ design frontend. `aiecc.exe` accepts hand-written `.mlir`, but nothing generates it from C++. Fighting this would stall the project before any embedding runs. |
| **One validated kernel first** | Debugging a multi-kernel dataflow with no measured baseline is where these projects die. |
| **bf16 first, int8 second** | bf16 is the AIE2P native multiply, needs no calibration, and is empirically safe (<0.22% max error vs fp32, measured twice independently). int8 adds calibration and scale plumbing on top of kernel bring-up, so failures become hard to attribute. |
| **int4 deferred indefinitely** | **There is no native int4 MAC on AIE2P** — it is a storage format requiring in-core dequant. High risk on a 22M-param encoder with no published MTEB evidence. |
| **all-MiniLM-L6-v2 first** | Smallest vanilla-BERT target, and bge-small-en-v1.5 reuses every compiled design unchanged. **Not** a byte-identical drop-in, as this table claimed until it was tried: bge-small has **12 layers, not 6**, and pools **CLS, not mean**. The GEMM shapes are identical, so the xclbins and the `.npue` tiling carry over; the layer count and the pooling are data the runtime reads. Measured at `1-cos` 8.348e-06 and **0.506x** MiniLM's throughput, which is what twice the layers costs. |
| **Standalone Apache-2.0 tool** | FastFlowLM's kernels are closed binaries, so they cannot be contributed as source today. Apache-2.0 keeps an explicit patent grant, which matters since those kernels are patent-pending. Approach AMD later with a working artifact. |
| **Write our own WordPiece tokenizer** | Every vendorable option costs more than it saves: `tokenizers-cpp` drags a **Rust toolchain** into a build whose selling point is a lean native Windows binary; llama.cpp's WPM has a known NFD bug that **silently drops accented characters**; sentencepiece is the wrong algorithm. The algorithm is ~500–700 LOC and exactness is directly testable against HF. |

## Roadmap

| M | Milestone | Gate |
|---|---|---|
| **M0** | Scaffold, research index, foundational docs | Repo explains itself |
| **M1** | Hello NPU — stock design runs natively, **with a trace** | Non-empty `trace.txt` → cycle counts |
| **M2** | bf16 GEMM at MiniLM shapes, traced | Cycles + vector utilisation vs the 14.7 TOPS ceiling |
| **M3** | Python reference encoder + golden vectors | Layer-by-layer oracle exists |
| **M4** | Offline weight pre-tiling → `.npue` | Format spec + round-trip verified |
| **M5** | Encoder ops onto the NPU, one at a time | Each validated against M3 goldens and traced |
| **M6** | Full encode path in Python | **Bankable — real value even if C++ never happens** |
| **M7** | C++ runtime | Pure C++ + XRT, no Python at runtime |
| **M8** | Benchmark and compare | CPU-vs-NPU under protocol; MTEB within 0.3 pts |

Deliberately deferred: int8/W8A8, int4, attention micro-optimisation
(measured at 1.4% end-to-end for BERT), and any FastFlowLM PR.

## The three findings that shaped the architecture

From the published work this project draws on (cited inline below):

**F1 — Per-dispatch overhead dominates, not kernel throughput.** TileFuse measured the
NPU *losing* to the iGPU at 256-token prompts. MLIR-AIR got 2.24× purely by fusing five
dispatches into one. → Batch, one resident `.xclbin`, fuse whole layers.

**F2 — Batching is mandatory, for bandwidth reasons.** 21.3 MB of weights over
~120 GB/s = 0.18 ms, which *equals* the theoretical compute time for one sequence.

**F3 — Attention is not the bottleneck for encoders.** Zen-Attention (AMD) measured
BERT: full attention folding bought 1.4% end-to-end. Our own FLOP split agrees —
attention is 5.3% of work at seq 128.
