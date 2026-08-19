# 0044 — M9: an optimisation sweep of the IRON toolchain and four outside repos

**Question.** We have been programming this NPU for nine milestones against one
slice of the IRON API — `ObjectFifo` with `.split()/.join()`, `Worker`,
`Runtime`, `@iron.jit`. **What is in `C:\dev\mlir-aie` that we have never
used**, and what have four other people building on the same silicon done that
we have not thought of?

This is a **reading and pricing** task, not a build task. Its output is a list
of levers with a mechanism, a price, and a link to the constraint each one
attacks. Where a lever is *closed* — checked and worth nothing — that is
recorded too, because the next person will otherwise check it again.

Depends on nothing; informs [`0043`](../0043-m9-attention-geometry/TASK.md)
(unfinished) and the fused-layer direction.

---

## Method

```powershell
# the four outside repos, read-only, gitignored (externalrepos/)
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\externalrepos
git clone --depth 1 https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm.git
git clone --depth 1 https://github.com/Scottcjn/npu-linux-kit.git
git clone --depth 1 https://github.com/drakosha/whisper-xdna.git
git clone --depth 1 https://github.com/midhatn/phoenix-sdr-dsp.git
```

The IRON half was done by reading `programming_guide/` top to bottom, then
**grepping our own tree for every API name the guide mentions** — the point
being to find the features we have never typed, rather than to re-read the ones
we use:

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
foreach ($k in "consumer_obj_type","aie_stream","CascadeFlow","put_mcd",
               "TileDma","Bd(","PacketFlow","pad_dimensions",
               "disable_synchronization","delegate_tile","init_values",
               "AIE_LOOP_UNROLL") {
  "$((Select-String -Path experiments\*\*.py,tools\*.py,reference\*.py `
       -Pattern $k -SimpleMatch).Count)  $k"
}
```

**Every one of those came back 0.**

---

## Progress log

- **Step 1 — cloned the four repos, gitignored them.** `externalrepos/` added
  to `.gitignore` with the same reasoning as `../FastFlowLM`: read for
  architecture, never vendored.
- **Step 2 — read `programming_guide/` §2a–2h and §4a–4c.** Sections **2g**
  (data movement without ObjectFifos) and **2h** (advanced ObjectFifo +
  cross-tile Buffer) are the ones we have never touched, and 2h did not exist
  when this project started.
- **Step 3 — grepped the whole IRON Python surface** for what the guide does
  not document, notably the `ObjectFifo.__init__` keyword list.
- **Step 4 — read the four outside repos.** *(below)*

---

## Findings

*(filled in as they are established; see the note that this task produces)*

---

## Finding 0 (unplanned) — a stale `npuembed.exe` was holding the array, and nothing in our harness would have told us

Before pricing anything I took a fresh `--bench` so the arithmetic would rest on
today's machine rather than a doc figure. It read **221.4 seq/s** for MiniLM at
batch 128, against the **618 seq/s single-lane** figure in `CLAUDE.md`.

```powershell
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 `
    --artifacts artifacts_b128il --bench 10 --threads 24
#   wall 578.25 ms -> 221.4 seq/s ; wait (hardware) 15,541 us per dispatch
```

2.8× is far too big for noise, so the next command was the machine, not the code:

```powershell
C:\Windows\System32\AMD\xrt-smi.exe examine -r all
```

```
|63804               |144        |7233        |0           |0    |Normal   |
|npuembed.exe        |Active     |7232        |0           |     |N/A      |
|1032 MB             |608 KB     |            |            |     |N/A      |
```

**A second `npuembed.exe` from an earlier session still held an `Active`
hw_context with 1,032 MB resident** — that is bge-large's 1023 MB, so it is a
leftover `--serve` or `--bench` on the large model. Eight
`WorkloadsSessionHost.exe` contexts (Windows Studio Effects) were also present,
all `Idle`.

### Why this is a finding and not just an accident

`CLAUDE.md` rule 1 already says wall clock is never an NPU performance claim
because the array is shared. What this shows is that **the rule's usual mitigation
does not cover this case**:

- *"Interleave against the CPU"* ([`0040`](../0040-m9-honest-cpu-baseline/TASK.md))
  fixes drift that hits both sides. A resident NPU context hits **only the NPU
  side**, so interleaving makes the *ratio* wrong, confidently.
- The contention is **ours**. Every existing caution is about "other processes";
  the process here is a previous run of the same binary, which is exactly what a
  long measurement session produces.
- Nothing reports it. `--bench` prints bo-mode, model, tiers, alignment and
  staging — every input except the one that moved the number by 2.8×.

`0040` ended with *"record machine state by tool"*, having found a hand-rolled
battery check that reported "ON BATTERY" for a machine with no battery. This is
the same lesson one level up: the tool that knows is `xrt-smi`, it is installed
at `C:\Windows\System32\AMD\xrt-smi.exe`, and we were not asking it.

**Proposed gate:** before any timed run, parse `xrt-smi examine -r all` and
**refuse to report a throughput** if any hw_context other than our own is
`Active` — the same fail-closed shape as `npu_utilisation.ps1`'s refusal to
report a percentage from too few samples. Listing them and continuing is not
enough; a warning in a scrollback is how the eight fail-opens in `CLAUDE.md`
all began.

**Consequence for this task:** every wall-clock number taken before the
process was killed is contended and is labelled as such. The task's conclusions
are about *mechanisms and prices*, which do not depend on it.

> **Superseded by Part 2 below.** The process was killed, everything was
> re-measured on an idle array, and the guard this section proposes was built
> and verified. The contended numbers are kept because the *comparison* between
> them is the finding: contention did not merely add noise, it **flattered the
> current architecture** by diluting every host-side term.

---

## What came out

The durable output is [`research/notes/0007`](../../research/notes/0007-unused-iron-surface.md).
This section records the *sequence*, including the parts that were wrong.

### The grep that started it

Twelve IRON API names appear in `programming_guide/` and **zero** times in our
tree. Verified present in the *installed* wheel, not just the source checkout —
a distinction that mattered, because `C:\dev\mlir-aie` is tagged `v1.3.4` while
carrying 2026-dated guide sections, so "documented upstream" and "available to
us" are different questions:

```powershell
cd C:\dev\mlir-aie
.\ironenv\Scripts\python.exe -c "import inspect; from aie.iron.dataflow.objectfifo import ObjectFifo; print(list(inspect.signature(ObjectFifo.__init__).parameters)); import aie.iron as I; print([n for n in ['CascadeFlow','TileDma','DmaChannel','Bd','Flow','Lock','PacketFlow','Acquire','Release'] if hasattr(I,n)])"
```

All present. Guide **§2h** (advanced ObjectFifo + cross-tile `Buffer`) did not
exist when this project started; **§2g** (hand-wired DMA) we had simply never
needed.

### Two claims that had to be corrected before shipping

Both were caught by checking rather than by reading, and both would have been
wrong in a way that wasted someone's day.

**1. `pad_dimensions` is mem-tile only.** The first draft said N could be padded
"in the DMA" as if that were unconditional. `DMABDOp::verify()` in
`AIEDialect.cpp` says otherwise:

```cpp
if (!parentTile.isMemTile())
  return emitOpError() << "Padding is only supported by memtile dma bds.";
```

It also requires an n-d access pattern and 32-bit-word-aligned inner padding.
The lever survives — the mem tile is where the column's B slice lands anyway —
but padding is one-directional, so the **output** side needs a strided
`dims_to_stream` to drop the pad columns again. That is where a first attempt
would fail, and the draft would not have warned anyone.

**2. The centred-polynomial-basis claim does not transfer, and I nearly shipped
it as the headline.** whisper-xdna measured **2.5x faster at equal accuracy**
from re-centring their GELU polynomial to `u` in `[-1,1]`, and our `gelu_poly.cc`
is visibly a monomial Horner with coefficients spanning 7.2e-05 to 4.9e-01 —
four orders of magnitude, exactly the shape they describe.

Tested instead of assumed, in float32 Horner, the way the kernel evaluates:

```powershell
& "C:\Users\vegar\.conda\envs\iron\python.exe" -c "<fit c(u)=GELU(u)-max(u,0) on [0,4], monomial vs centred, degrees 3..10, evaluate in float32>"
```

| degree | monomial | centred |
|---:|---:|---:|
| 7 | 5.913e-04 | 5.872e-04 |
| **8 (ours)** | **3.613e-04** | 3.613e-04 |
| 10 | 3.728e-05 | 1.562e-05 |

**Identical to four significant figures until degree 10.** Their fix targets
**bf16 coefficient** precision; ours are `fp32` literals evaluated in `fp32`
(`0016` established AIE `vector<float>` is full IEEE fp32), so the conditioning
that motivated the change is absent. A real finding, borrowed honestly, that
happens not to apply — and the only way to know was to run it.

The same experiment did produce something smaller and real: degree 8 sits at
3.6e-04, **7x below the 2.465e-03 bf16 output floor**, and degree 7 is still
4.2x below it. **Degree 8 -> 7 is free**, one Horner step of eight.

### The finding that changes an architecture decision

Not from any repo — from our own `--bench` output, read differently.
[`0032`](../0032-m7-one-xclbin-production/TASK.md) moved LayerNorm, softmax and
GELU to the host because each measured faster and more accurate than its NPU
dispatch. True. But that comparison priced **the operator**, and the operator is
not what it costs:

| | share of encode |
|---|---:|
| host gelu + softmax + layernorm | **3.6%** |
| `read out + bias` + bf16 convert + syncs | **22.0%** |

`read out + bias` alone is 14.3%, and it is 679 MB of fp32 C per MiniLM encode
at batch 128, read out of write-combined XRT memory at ~8 GB/s. It exists
*only* because the next operator runs on the host. The eltwise-on-array question
was closed on the wrong ledger and should be reopened on the right one — with
the 16 KB program-memory wall respected as **one operator per core**, not the
three-op universal worker `0032` correctly killed.

### The outside repos, ranked by what they were worth

| repo | worth |
|---|---|
| [`whisper-xdna`](https://github.com/drakosha/whisper-xdna) | **high.** Same problem shape (an encoder), same discipline. Default rounding mode is `floor`; fused attention built, correct, and still slower, with the reason; `pyxrt.runlist` worth -1400 ms; chaining without host round-trip worth 1.73x; `AIE_LOOP_UNROLL_FULL` **14% slower**, independently confirming note `0006`'s inference. |
| [`phoenix-sdr-dsp`](https://github.com/midhatn/phoenix-sdr-dsp) | **one fact.** Windows-native like us, pinned to mlir-aie **v1.4.1** where we are on 1.3.4, plus the trap that the rolling wheel channel silently resolves back to 1.3.4. |
| [`hawkpoint-npu-llm`](https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm) | **low.** Its own performance doc declines to attribute its 274 ms/token to any phase. Its `qwen_decoder_layer_bf16.cc` has a 128-lane **scalar float** reduction and an index `switch` in the hot loop — our note `0001` trap 5 and AMD's own anti-pattern. Read for architecture, not technique. |
| [`npu-linux-kit`](https://github.com/Scottcjn/npu-linux-kit) | **none yet.** Its embeddings module is "candidate, not built". |

---

## Loose ends deliberately not closed

- **The compression experiment was not run.** `basic/dma_compression` sizes its
  shim TAP to the *compressed* byte count (2944 for `arange`), so feeding it a
  real weight tile means finding that count first or the consumer DMA stalls.
  One session's work, decisive either way; the note says what to measure.
- **`0043` is still unfinished** (its Results section is empty) and this task
  does not finish it. What it does is remove the reason 0043 gave for stopping,
  and hand it someone else's measured warning about what is on the other side.
- **Nothing here was built.** Every lever is priced and none is implemented;
  that was the task.

---

## Sweeping our own parked leads (the other half of the brief)

`grep -rn --include=TASK.md -iE "(deferred|not measured|untested|open risk|worth
trying)" tasks/` over everything from 0020 on. Most parked items had since been
picked up — 0027's *"LayerNorm was never rewritten and still opens three fifos
per core"* was closed by 0030's mem-tile broadcast, 0030's §6a was closed by
0033's pipelined lanes. **One was not, and it is the same thing this task
arrived at independently:**

```
research/notes/0005-expert-review-tests.md:24
  | §6b | Device-resident intermediates remove t_conv/sync | DEFERRED with cause
        | Same prerequisite; t_conv+syncs ~= 70 ms at batch 128 remain the prize |
```

and then, a task later:

```
tasks/0032-m7-one-xclbin-production/TASK.md:143
  "... anymore -- the blocker 0030 deferred it on is gone."
```

**§6b has been open and unblocked for two tasks and nobody picked it up.** That
matters more than the finding: the deferral was correct and well-reasoned, the
unblocking was noticed and written down, and the item still fell through,
because "deferred with cause" has no expiry and nothing re-reads it when the
cause expires.

What this task adds to it is that **the prize was under-scoped by about 2x**.
§6b counted `t_conv` + syncs (7.7% today). It did not count `read out + bias`
(14.3%), which is the largest single term — and which **grew because of the very
decision 0032 made**: moving eltwise to the host means every GEMM result now
crosses the bus. The estimate was made before the thing that made it bigger.

Still genuinely open elsewhere, found in the same sweep and *not* pursued here:
`aie::exp2` remains unmeasured (0020), and whether hw_context partition width is
settable at context creation was never tested (0025).

---

## Files touched

| file | why |
|---|---|
| `research/notes/0007-unused-iron-surface.md` | the durable output |
| `research/notes/README.md` | index row |
| `research/prior-art.md` | new §7, the four community projects |
| `CLAUDE.md` | trap 2b (default rounding is `floor`), trap 1 gains the burst-length consequence, Current state |
| `tasks/README.md` | index row + a note that 0038-0043 were never indexed |
| `.gitignore` | `externalrepos/` |

Nothing in `experiments/`, `runtime/` or `tools/` was modified. No design was
built and no kernel was changed.

---

## Part 2 — the machine was cleared, and the finding got bigger

The stale process was killed. Everything below is on an idle array, three runs.

### The numbers contention was hiding

| | contended | **idle** | |
|---|---:|---:|---|
| wall | 578.25 ms | **185.23 ms** | |
| throughput | 221.4 seq/s | **691.0 seq/s** | 3.1x |
| hardware wait / dispatch | 15,541 us | **3,029 us** | 5.1x |
| submit / dispatch | 103 us | **66 us** | |

Three idle runs: **694.0, 687.5, 691.0 seq/s** — 0.9% spread, every share within
0.2 points. And `--pipeline 2` gives **907.5 seq/s**, against the 833 recorded in
[`0033`](../0033-m7-pipelined-lanes/TASK.md) and the 877 in
[`0040`](../0040-m9-honest-cpu-baseline/TASK.md).

### Contention was flattering the current architecture

This is the part worth keeping. The transport claim did not survive the
re-measurement unchanged — **it got stronger**:

| | contended | **idle** |
|---|---:|---:|
| `dispatch + wait` | 65.0% | **40.3%** |
| host gelu + softmax + layernorm | 3.6% | **7.5%** |
| `read out + bias` | 14.3% | **18.8%** |
| bf16 convert | 4.7% | **9.0%** |
| syncs | 3.0% | **5.2%** |
| **transport total** | **22.0%** | **33.0%** |

A five-times-slower NPU diluted every host term. Take it away and the array
shrinks to 40% of the encode while `read out + bias` **rises** to 18.8%. **The
faster the array gets, the larger this term becomes** — which is the opposite of
how a cost you intend to optimise away later should behave.

In production (`--pipeline 2`) the per-lane line says it outright:

```
p1 host work  142.33 ms  50.5%  (conv 22.7  bias 70.7  attn 29.2  elt 19.7)
```

Host work is half the wall clock per lane, and the C readback is **70.7 ms of
it — 3.6x all three eltwise operators combined (19.7 ms)**. The thing 0032 moved
to the host is the smallest line in the budget that move created.

`research/notes/0007` Part 4 was rewritten against these numbers.

### The ninth fail-open, closed

`--bench` now **refuses to run** when the array is not ours.

- `runtime/include/npu_contention.hpp`, `runtime/src/npu_contention.cpp` —
  shells out to `xrt-smi examine -r all`, parses the HW Contexts table, and
  compares each row's PID against `GetCurrentProcessId()`.
- `runtime/src/main.cpp` — the check runs after the banner and before any
  timing; a refusal exits **2**. `--allow-contention` downgrades it to a loud
  warning that says the number is not an NPU performance claim.

**It fails closed on three distinct things**, which is the whole point:

| situation | behaviour |
|---|---|
| a foreign `Active` hw_context | refuse, listing pid / process / status / memory |
| `xrt-smi` not found | refuse — *an absent data source is not a negative reading* (0040) |
| `xrt-smi` ran but no rows parsed | refuse — a format change must not read as "nothing is running" |

Verified on hardware, both directions:

```powershell
# idle
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --bench 10 --threads 24
#   npu        exclusive -- 9 hw_context(s), none Active but ours (C:\Windows\System32\AMD\xrt-smi.exe)
#   wall 185.23 ms -> 691.0 seq/s

# with a second npuembed deliberately holding the array
#   !! ANOTHER PROCESS IS USING THE NPU -- 2 Active hw_context(s) that are not ours:
#        pid 66000    npuembed.exe             Active   177 MB
#   Refusing to report a throughput. ...
# exit code 2
```

**The third failure mode caught a bug in the guard itself on its first run.**
The table is indented six spaces under `HW Contexts:`, and the first version of
`cells()` tested `line.front() == '|'`, so it parsed exactly zero rows. Because
"the tool ran but I could not read it" was written as a refusal rather than as
an empty result, it printed *"the output format may have changed"* instead of
`exclusive -- 0 hw_context(s)`. A fail-open would have said the machine was
idle, forever, on every machine.

And the test itself reproduced the original sin: `kill` on the background shell
job left **two** `npuembed.exe` holding contexts. They were found with
`tasklist`, killed properly, and the final measurement re-taken. The guard is
not a substitute for cleaning up; it is what tells you that you did not.

---

## Part 3 — the rounding mode, tested on our own kernels. It was one line, and it was the whole implementation error.

Note 0007 ranked `set_rounding(conv_even)` first on the "what to do" list on the
strength of someone else's measurement and a header file. That is not a result.
This is.

### The hypothesis was already written down here, 28 tasks ago

[`0015`](../0015-m5-gelu-polynomial/TASK.md) measured the GELU kernel at
**4.312e-03** against the golden, while a numpy model of the *identical
polynomial* sat at 1.923e-03 — an implementation gap of **3.886e-03**, which it
noted was "2⁻⁸ — bf16 territory". It inferred:

> **Inference: `aie::vector<float>` arithmetic on AIE2P is not IEEE fp32.**

[`0016`](../0016-m5-fp32-probe/TASK.md) built a probe and **refuted that**:
`aie::vector<float>` carries ~24 mantissa bits on both add and multiply. And
then it wrote, under *What this leaves open*:

> Untested suspects: `aie::abs`, `aie::min`, `aie::max` on float, **or the
> rounding mode of the fp32 → bf16 conversion. Truncation instead of
> round-to-nearest would fit the magnitude, but that is a hypothesis, not a
> finding, and it is not chased here.**

**The right hypothesis, named precisely, and never run.** It sat for 28 tasks
while three kernels shipped with the error in them.

### Method

One new `.cc` per kernel, differing from the original by exactly one line
(`aie::set_rounding(aie::rounding_mode::conv_even)` before the impl call), wired
into the existing harnesses as a new variant. **Separate files, not a flag**,
because `CLAUDE.md` trap 7c and five previous incidents say an edit-in-place A/B
is how this toolchain serves a stale build.

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python experiments\m5-eltwise\gelu_kernel.py      --kernel poly    --cols 1
python experiments\m5-eltwise\gelu_kernel.py      --kernel polyrne --cols 1
python experiments\m5-eltwise\softmax_kernel.py   --variant poly     --stack 0x2000 --cols 1
python experiments\m5-eltwise\softmax_kernel.py   --variant poly_rne --stack 0x2000 --cols 1
python experiments\m5-eltwise\layernorm_kernel.py --variant base --cols 1
python experiments\m5-eltwise\layernorm_kernel.py --variant rne  --cols 1
```

Every harness already reports a **built-in control**: `CPU model vs golden`, the
design limit, which cannot depend on the rounding mode. It is **bit-identical
across every pair below**, which is what makes these A/Bs rather than two runs.

### Result — all three kernels, one line each

| kernel | vs golden, `floor` | vs golden, **`conv_even`** | | control (design limit) |
|---|---:|---:|---:|---:|
| GELU (`gelu_poly`) | 4.312e-03 | **2.494e-03** | **1.73x** | 1.923e-03 both |
| softmax (`poly`) | 4.278e-03 | **3.325e-03** | **1.29x** | 3.225e-03 both |
| LayerNorm (`base`) | 3.326e-03 | **2.059e-03** | **1.62x** | 2.058e-03 both |

And the row that says what actually happened — **NPU against a CPU model of the
identical formula**, i.e. implementation error with the design's own error
removed:

| kernel | `floor` | **`conv_even`** | |
|---|---:|---:|---:|
| GELU | 3.886e-03 | **1.556e-03** | 2.50x |
| softmax | 3.424e-03 | **1.481e-03** | 2.31x |
| **LayerNorm** | 3.659e-03 | **3.967e-05** | **92x** |

**LayerNorm's implementation error was not reduced, it was removed.** At
3.967e-05 the kernel is numerically indistinguishable from numpy running the
same formula, and its end-to-end 2.059e-03 is the bf16 floor of 2.058e-03 to
four significant figures.

GELU lands on **2.494e-03** — which is the number `gelu_poly.cc`'s own header
has claimed since it was written: *"Degree 8 with R = 4 lands at 2.494e-03 on
the real L0.ffn_up activations, which is 1.01x the bf16 floor."* The kernel's
design-time prediction and its hardware measurement disagreed by 1.73x for
nine milestones, and the entire discrepancy was an unset control register.

### The mechanism, confirmed independently of any error metric

Softmax rows must sum to 1, which gives a signature that no rel_fro can fake.
Under `floor` every element rounds **down**, so a row sum can only come in low:

```
floor      row sums: min 0.994581  max 1.000000  worst |1-sum| 5.419e-03
conv_even  row sums: min 0.997272  max 1.002485  worst |1-sum| 2.728e-03
```

**No row exceeds 1.0 under `floor` — max is exactly 1.000000.** Under
`conv_even` the rows straddle 1.0 and the worst deviation halves (ratio 1.986),
which is what a one-sided error becoming symmetric looks like. This is the
mechanism observed directly, not inferred from a magnitude.

### whisper-xdna's warning did not transfer, as predicted

They measured `conv_even` **breaking** their softmax (cosine 0.879) because the
stock `getExpBf16` LUT is calibrated for the default mode. Note 0007 §3.1
predicted this would not apply to us, since
[`0030`](../0030-m7-expert-review-tests/TASK.md) replaced that table with our own
`exp2_poly`. Ours improves, 4.278e-03 → 3.325e-03. **The prediction was made
before the measurement and it held.**

### Two traps hit while building the A/B

1. **The include-order trap, exactly as `CLAUDE.md` records it.** `softmax_rne.cc`
   does `#include "softmax.cc"` to wrap the impl, and **`aie_kernels` ships its
   own `softmax.cc`** — so it silently compiled the upstream file and failed with
   *"use of undeclared identifier 'softmax_impl'"*. Fixed by
   `include.insert(0, HERE/"kernels")` in all three harnesses. The GELU A/B had
   worked *by accident* before this (no upstream `gelu_poly.cc` to collide with,
   and the cache dir happened to hold ours), so **both GELU runs were repeated
   after the fix** and reproduce to four significant figures.
2. **`layernorm.cc` puts `layernorm_impl` INSIDE its
   `#ifndef NPUE_ELTWISE_IMPL_ONLY` block**, unlike `gelu_poly.cc` and
   `softmax.cc`, so the guard that makes a wrapper file clean for those two
   strips the function here. Peano's message named the survivor: *"did you mean
   'layernorm_il4_impl'?"* — that one lives outside the guard.

### What was deliberately NOT changed

**The shipped kernels still default to `floor`.** The `*_rne.cc` files are new
variants, selectable by `--kernel polyrne`, `--variant poly_rne`,
`--variant rne`, and the originals are untouched.

That is a considered choice, not an oversight. Production runs eltwise on the
**host** ([`0032`](../0032-m7-one-xclbin-production/TASK.md)), so none of these
kernels executes today and there is no end-to-end gate to validate a swap
against; and `set_rounding` writes **core-wide** state, so in any design that
puts two kernels on one core the setting leaks between them, which needs
deciding at design time rather than inherited from whichever kernel ran first.
Keeping both halves also keeps 0015's and 0030's numbers reproducible.

**When eltwise returns to the array, `conv_even` is the default and this is the
evidence.** It is free, it is measured on all three kernels, and it is the
difference between a kernel that hits its design limit and one that misses it by
1.3-1.7x.

### The il4 variants agree exactly, and one self-inflicted detour

The row-interleaved kernels from [`0031`](../0031-m7-eltwise-ilp/TASK.md) are a
different instruction *schedule* of the same arithmetic, so they are a free
consistency check on the whole result. They agree to four significant figures:

| | `floor` | **`conv_even`** |
|---|---:|---:|
| softmax `poly` | 4.278e-03 | **3.325e-03** |
| softmax `poly_il4` | 4.278e-03 | **3.325e-03** |
| LayerNorm `rne` | — | **2.059e-03** |
| LayerNorm `il4_rne` | — | **2.059e-03** |

Row sums move identically too (`floor` max exactly 1.000000 → `conv_even`
1.002485 in both softmax variants).

**The detour was mine and is not a finding.** `--variant poly_il4_rne --stack
0x2000` timed out (`ERT_CMD_STATE_TIMEOUT`), which looked like the new variant
breaking something. It is not: **the baseline `poly_il4` times out at 0x2000
too**, and 0x4000 fixes both.
[`0031`](../0031-m7-eltwise-ilp/TASK.md) had already written this down — *"at
stack 0x2000 the design timed out on the array instead of corrupting. 0x4000
fixed it. The exporter now sets 0x4000 for `poly_il4`"* — and its Commands
section even lists `--stack 0x4000` for exactly this invocation. I took the
0x2000 figure from `CLAUDE.md`'s summary line, which is about the *non*-il4
poly softmax, and did not check the task log for the variant I was actually
running. Twenty minutes, and the only reason it did not become a wrong
conclusion is that running the baseline control is the first move here.

**Method note:** the control that saved it — "does the *unmodified* kernel do
this too?" — is the same one [`0008`](../0008-m5-bfp16-real-data/TASK.md) and
whisper-xdna's int8 decode both credit. A new variant failing is not evidence
about the variant until the baseline has been run in the same session.

### An A/B that overwrote its own control

Found in `git status` at the end, not by anything failing: after the sweep,
`artifacts/softmax_kernel.json` and `artifacts/layernorm_L0_ln1.json` showed as
**modified**. Both harnesses write one fixed filename regardless of variant, so
each `_rne` run had silently replaced the baseline artifact it was being
compared against. `gelu_kernel.py` was already right — it writes
`gelu_kernel_{args.kernel}.json` — which is why only two of the three drifted.

Rule 6 says a number without a traceable artifact is not a result; this is the
version where the artifact exists and quietly holds the *other* arm's numbers.
Both harnesses now key by variant, the clobbered baselines were restored from
git, and **all four measurements were re-run from scratch** — reproducing
4.278e-03 / 3.325e-03 / 3.326e-03 / 2.059e-03 exactly. Every number in Part 3 has
now been produced at least twice, in separate sessions, with the artifact
written to its own file.

---

## Files touched (Parts 2 and 3)

| file | why |
|---|---|
| `runtime/include/npu_contention.hpp`, `runtime/src/npu_contention.cpp` | **new** — the `xrt-smi` exclusivity gate |
| `runtime/src/main.cpp`, `runtime/CMakeLists.txt` | wire the gate into `--bench`; `--allow-contention` |
| `experiments/m5-eltwise/kernels/{gelu_poly,softmax,layernorm}_rne.cc` | **new** — one line different each |
| `experiments/m5-eltwise/{gelu,softmax,layernorm}_kernel.py` | `_rne` variants; our include dir first; artifacts keyed by variant |
| `experiments/m5-eltwise/artifacts/*_{poly,polyrne,poly_rne,base,rne}.json` | **new** — both arms of every A/B |
| `CLAUDE.md` | trap 2b rewritten with the measured table; Current state |
| `research/notes/0007-unused-iron-surface.md` | §3.1 and Part 4 rewritten against measurements |

The production kernels (`gelu_poly.cc`, `softmax.cc`, `layernorm.cc`) and the
GEMM path are **unmodified**. `verify_embed_e2e.py` passes after all of it:
worst `1-cos` **2.644e-05** against a 2e-03 tolerance, top-10 neighbour overlap
**1.0000**.
