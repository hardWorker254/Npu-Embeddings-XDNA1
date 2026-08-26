# 0084 — the host ISA was never measured, and neither was our own spread

- **Date** 2026-08-23
- **Status** done — one negative result, one correction to 0082's numbers, one
  re-measured breakdown

## Goal

**User:** *"hva tenker du videre?"* — asked after the host fusions landed.

Answering it honestly required re-measuring first: **every priority claim in
0081–0082 rests on a breakdown taken BEFORE the fusions**, and the fusions
removed ~40% of host time. The number I had been quoting was invalidated by my
own work.

---

## 1. The breakdown, re-measured

`bge-large-en-v1.5.int8n64`, single lane, 2 encodes of 128 sequences:

| | before fusion | after fusion |
|---|---:|---:|
| **wall** | 1465.73 ms | **1087.57 ms** |
| dispatch + wait | 432.44 (30.4%) | 439.75 (**40.4%**) |
| host attention | 207.96 (14.6%) | 207.40 (**19.1%**) |
| read out + dequantise | 260.28 (18.3%) | 189.13 (17.4%) |
| host LayerNorm | 37.00 (2.6%) | 157.84 (14.5%) |
| quantise A | 140.76 (9.9%) | **20.09 (1.8%)** |
| residual / pooling | 183.50 (12.9%) | **8.74 (0.8%)** |
| host GELU | 108.79 (7.6%) | **absorbed** |

LayerNorm's line grew because it now does four jobs in one pass — that is the
fusion showing up where it should. What matters is the ranking, and it
inverted: **the array is the largest single item again at 40.4%**, and
**attention is the largest thing nobody has ever touched**, at 19.1%.

---

## 2. Attention cannot be tuned, only moved

51.5 GFLOP per encode at 103.7 ms = **497 GFLOP/s**, against an AVX2 FMA peak
of ~568 GFLOP/s on the 10.1 cores the run reported busy. **88% of peak.**

So the 19.1% is not slack.

> **CORRECTION, same day.** This first said *"`qk()` carries two intrinsics and
> `av()` has none — both are compiler-vectorised"*. **Wrong, and it was a
> grep artefact**: `qk()` and `av()` are three-line dispatchers on `head_dim`;
> the work is in `qk_impl<NV>` and `av_impl<NV>`, and **both are hand-written
> `_mm256_*` intrinsics** — `qk_impl` holds Q in `NV` registers across the j
> loop and reduces with `hsum256`, `av_impl` broadcasts `a[j]` and FMAs into
> `NV` accumulators. See §3, which the mistake invalidates.

---

## 3. The host ISA: measured, and it buys nothing

`runtime/CMakeLists.txt` had said `/arch:AVX2` since the runtime was written,
justified as *"Strix Point is Zen 5, so AVX2 is a floor, not an assumption"*.
True, and a long way below the ceiling. CPUID on this machine:

```
  AVX512F 1   AVX512DQ 1   AVX512BW 1   AVX512VL 1
  AVX512_VNNI 1   AVX512_BF16 1   AVX_VNNI 1
```

**The project has been meticulous about the NPU's ISA and never once queried
the host's** — while the host is ~60% of an encode.

Rebuilt at `/arch:AVX512`. Correctness first: all five gates reproduce their
numbers **exactly** — bf16 MiniLM 1.086e-05 and bge-large 8.432e-06; int8
MiniLM 1.161e-03, bge-large 2.968e-03, nomic 1.098e-03.

Throughput, three runs per arm:

| model | AVX2 | AVX-512 | |
|---|---|---|---:|
| `bge-large-en-v1.5.int8n64` | 143.8 / 144.6 / 144.3 | 144.0 / 144.1 / 143.9 | **1.00×** |
| `all-MiniLM-L6-v2.int8` | 1676.3 / 1674.4 / 1673.3 | 1689.9 / 1698.7 / 1703.8 | 1.014× |

**Nothing.** Attention specifically moved 207.40 → 203.91 ms, 1.7%.

> **AND THE EXPLANATION I GAVE FOR THAT WAS WRONG.** It read: *"consistent with
> Zen 5 mobile double-pumping a 256-bit datapath"* — a claim about the
> hardware. It is nothing of the sort. `/arch:AVX512` cannot widen a
> hand-written `_mm256_fmadd_ps`; those stay 256-bit whatever the flag says.
> The flag only moves **auto-vectorised** code, and per §2's correction
> attention has none. So this measurement says **the flag cannot reach the hot
> kernels**, not that the hardware cannot go wider.
>
> **Attention at 512 bits is therefore UNTESTED, not tested-and-failed** — and
> it is a better prospect than this task first concluded: 19.1% of the encode,
> hand-written at half the machine's vector width, and `av_impl`'s `NV`
> accumulators would also benefit from AVX-512's 32 architectural registers
> against AVX2's 16. Testing it means writing `_mm512_*` variants of
> `qk_impl`/`av_impl`, not setting a flag.
>
> This is the second time in two days I have read a conclusion off a grep
> instead of the code. The first cost an hour re-deriving 0062; this one nearly
> retired a live lever.

**Kept as a build option rather than reverted.** `-DNPUE_HOST_ARCH=AVX512` puts
the experiment one flag away on a machine with a full-width datapath, and the
shipped default stays AVX2 because that is what measured equal here.

**What this does NOT test is VNNI.** `vpdpbusd` is a different instruction —
four int8 MACs per lane with int32 accumulate — and no compiler reaches for it
from fp32 source. Given attention is the one host kernel that is compute-bound
rather than bandwidth-bound, that is the live half of this idea and it needs
hand-written intrinsics plus an accuracy argument for int8 attention scores.

---

## 4. A correction to 0082's numbers

0082's fused/unfused A/B was **one run per arm**. Re-run with three:

| model | unfused | fused | gain | 0082 published |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2.int8` | 1195.3 | 1672.0 | **1.399×** | 1.394× |
| `bge-base-en-v1.5.int8` | 288.2 | 405.3 | **1.406×** | 1.418× |
| `nomic-embed-text-v1.5.int8` | 216.1 | 317.1 | **1.467×** | 1.482× |
| `bge-large-en-v1.5.int8n64` | 103.2 | 143.6 | **1.391×** | 1.428× |

**The ratios were sound to within 3%** — both arms drift together, which is why
a same-session A/B survives being single-run. **The absolutes were not**:
0082's fused figures run ~4% low (MiniLM 1599.6 against 1672.0).

And 0082's own caveat was itself measured badly. It claimed *"run-to-run spread
is ~4%"* from two bge-large readings, 141.4 and 135.7. With three runs per arm
on a quiet machine the within-arm spread is **under 0.5%** — the 135.7 was
taken while something else was on the machine, and I generalised one contended
reading into a property of the instrument. The right caveat is
[T18](../../research/OPEN-THREADS.md#t18)'s, which is about `--probe-streams`
disagreeing with `--bench`, not about `--bench` disagreeing with itself.

---

## 5. Where this says to go next

Ranked by the re-measured breakdown, not the pre-fusion one:

1. **The array, 40.4%** — the largest single item again.
   [T28](../../research/OPEN-THREADS.md)'s tile-to-tile streaming is the lever
   that removes shim bytes rather than moving them faster, and
   [`0083`](../0083-m13-join-port-budget/TASK.md) established the port budget
   allows 5 sources and that production geometry fits a streaming relay on
   paper.
2. **Attention, 19.1%** — untouched, at 88% of AVX2 peak. Two host-side ideas,
   both needing hand-written intrinsics rather than a flag: **AVX-512** (the
   kernels are 256-bit by hand, so the width is genuinely untested) and
   **VNNI int8** (`vpdpbusd`, four MACs per lane, plus an accuracy argument for
   quantised attention scores).
3. **The C drain, 17.4%** — halved once by narrowing to bf16 (0080) and still
   third-largest.

And a process note worth more than any of them: **the breakdown has been
invalidated twice in two days by work done against it.** A priority list
derived from a measurement has to be re-derived when the measurement moves,
and this project's habit of quoting a task-log number weeks later is exactly
how that goes wrong. It is the same failure as the stale register in
[`0083`](../0083-m13-join-port-budget/TASK.md) §1, in a different medium.

---

## 6. Commands

```powershell
cmake -S runtime -B runtime\build -DNPUE_HOST_ARCH=AVX512   # or AVX2
cmake --build runtime\build --config Release

.\runtime\build\npuembed.exe . --model bge-large-en-v1.5.int8n64 `
    --artifacts artifacts_int8c_large_n64 --threads 24 --bench 2
```
