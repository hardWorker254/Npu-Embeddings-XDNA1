# 0047 — M9: cascade does not create DMA headroom, it moves it

**Question.** [`0046`](../0046-m9-b-reuse-asymmetric/TASK.md) closed B-reuse with
a census: the production GEMM has **zero spare input channels** — all 32 core
tiles at 2/2, five of eight mem tiles at 6/6 — and the mem-tile arithmetic
`A(1) + B(1) + C(4 core rows) = 6` said the **C join** spends the budget. It
proposed `CascadeFlow` as the fix, because a column could then return C through
one core instead of four.

**Is that true?** Answered without writing a kernel: upstream ships a cascade
matmul, so build it and count.

---

## Method

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\dev\mlir-aie\programming_examples\basic\matrix_multiplication\cascade
python cascade.py -M 512 -K 512 -N 512 -m 32 -k 32 -n 32 `
    --n-aie-cols 4 --dtype_in bf16 --dtype_out f32
#   -> PASS!   (runs on our hardware, our dtypes)

cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python tools\count_dma_channels.py <cache-dir>          # cascade
python <scratchpad>\probe_asym.py --mode off --cols 4   # ours, SAME width
python tools\count_dma_channels.py <cache-dir>          # ours
```

The control matters: 0046's census was at 8 columns and upstream's cascade caps
at 4, so both were rebuilt at **4 columns** before any comparison.

`kernels.cascade_mm` supports **(bf16, f32)** — our exact combination — and the
design builds, runs and verifies on this machine unmodified.

---

## Result

**4 columns, bf16 in / fp32 out, both designs:**

| | mem tile **in** | mem tile **out** | core **in** | core **out** |
|---|---|---|---|---|
| whole-array (ours) | **6/6 FULL** | 3/6 | **2/2 FULL** | 1/2 |
| cascade (upstream) | **3/6** | **6/6 FULL** | **2/2 FULL** | 0–1/2 |

**The prediction was right and incomplete.** Cascade takes mem-tile inputs from
6/6 down to **3/6** — exactly the `A(1) + B(1) + C(1)` that 0046 predicted, with
the C join collapsing from four returns to one. Three of the four cores in every
column have **zero DMA outputs at all**: they hand their accumulator north over
the cascade and never touch a stream.

But the mem-tile **output** side goes 3/6 → **6/6**, because B now has to be
delivered to the four rows as four *different* K-slices rather than broadcast.

**So cascade does not create headroom. It trades three inputs for three
outputs.**

### That trade is still the one worth having

The failure 0046 hit was, verbatim:

```
error: 'aie.tile' op number of input DMA channel exceeded!
```

**Inputs** were the exhausted side. Cascade frees exactly those, and B-reuse's
staging adds no mem-tile *output* — the fan-out to the rows is the same, just
fed from a resident buffer instead of refetched from DDR. So the combination is
plausible in a way neither half was alone. Plausible, not proven: that is a
build, and this task is a probe.

### One structural fact that falls out and is worth more than the rest

**Every core tile is at 2/2 inputs in BOTH designs.** That is not a property of
our dataflow — it is a property of the problem. A GEMM core needs A and B, a
core has exactly two input channels, and there is no third.

Anything that wants to hand a core a third stream — a bias vector, LayerNorm
parameters, a fused activation's coefficients — **cannot**, without giving up A
or B. It arrives independently at the workaround whisper-xdna documents for
fused attention: *"a compute tile has 2 input DMA channels and Q+K+V needs
three, so K and V ship as a single packed object."* Same wall, same fix.
[`0020`](../0020-m5-layernorm-kernel/TASK.md) already hit it here — γ and β had
to be packed into one buffer — and it is now measured rather than remembered.

---

## What this does not say

**Upstream's cascade kernel is not usable as-is.** It measured **3.19 GFLOPS**
on this run, which is two to three orders below our vectorised whole-array
design, and the upstream README says why: *"The cascade kernel is currently
scalar-only and the design is single-buffered (`fifo_depth=1` to avoid CDO
program-memory blowup), giving a structurally lower performance ceiling."*

So the cascade **dataflow** is what we would want and the cascade **kernel** is
not. `kernels.cascade_mm` exists with `put_only` / `get_only` / `put_get` /
`zero` bindings, so the work is vectorising a microkernel, not inventing one.

**And our shapes constrain it.** `cascade.py` asserts
`K % (k * n_aie_rows) == 0`, i.e. K must divide by `4k`:

| model | K | k=64 | k=32 |
|---|---|---|---|
| MiniLM / bge-small | 384 | 1.5 ✗ | 3 ✓ |
| MiniLM / bge-small | 1536 | 6 ✓ | 12 ✓ |
| bge-large | 1024 | 4 ✓ | 8 ✓ |
| bge-large | 4096 | 16 ✓ | 32 ✓ |

At h=384 the K-split forces **k = 32**, which changes the L1 budget and the
`.npue` B layout. bge-large works at k = 64 unchanged — **the wide model is the
easier target for cascade**, which is the same direction
[`0027`](../0027-m7-width-hypothesis/TASK.md) found for everything else.

---

## Status and cost of the next step

**Cascade is confirmed as the mechanism that unblocks B-reuse, and priced as a
milestone rather than a change.** To claim
[`0010`](../0010-m5-b-reuse-and-cost-model/TASK.md)'s 1.26–1.68× the work is:

1. a **vectorised** `cascade_mm`-shaped microkernel (upstream's is scalar);
2. the K-split dataflow in `gemm_pretiled.py`, with `k = 32` at h=384;
3. B staged in L2 with `repeat_count`, now that inputs have room;
4. re-validating the `.npue` layout, since `k` is part of `gemm_b_layout`.

That is not a session. It is, however, now the **only** identified route to the
GEMM time that dominates the encode — 39% at h=384, 61% at h=1024 — and the
census says why every cheaper route was blocked.

`tools/count_dma_channels.py` is the durable output of 0046+0047: it answers
"is there room for one more stream?" in a second, and both tasks were spent
learning that the answer had been "no" all along.
