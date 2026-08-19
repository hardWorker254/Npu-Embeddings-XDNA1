# Open threads

Every question this project has written down and not answered, in one place,
with a status.

**Why this exists.** [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) found
that the expert review's **§6b** had been *"deferred with cause"*, then
explicitly unblocked by [`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md),
and then sat untouched for two more tasks — because nothing re-reads a deferral
when its cause expires. [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md)
Part 3 found the same shape again: [`0016`](../tasks/0016-m5-fp32-probe/TASK.md)
wrote down the *correct* hypothesis for a mystery and left it unchased for **28
tasks**.

A `TASK.md` is a diary entry — it is written once and never revisited. That is
the right property for a diary and the wrong one for an open question. This file
is the register that gets revisited.

**Rules.** A thread is added the moment a task says "untested", "not measured",
"deferred" or "open question". It leaves only by being **ANSWERED** (with a
pointer to where), **RETIRED** (with a reason), or **SUPERSEDED**. Nothing is
removed silently, and a stale "open" in an old task is not evidence that the
thread is still open — this file is.

Status: **OPEN** · **ANSWERED** · **RETIRED** · **BLOCKED**

---

## Live, ordered by what they would change

### T1 — What *is* the GEMM's 3 ms? · **ANSWERED 2026-08-19**
**The count of tile iterations, not bytes.**
[`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md): `ffn_up` and
`ffn_down` have identical MACs and differ **1.50× in bytes**, and measure
**4,196 vs 4,273 µs** — 1.8% apart, in the wrong direction. `GMAC/ms` is flat at
1.08–1.15; `GB/s` spreads 17.7–27.0. Reproduced on the `--c-bf16` set. Fit:
`t = 573 µs + 4.72 µs × iterations`, ≤2.3% residual, against 0010's traffic
model being 50% out on the discriminating pair.

Left a successor, [T16](#t16--why-is-a-k-block-iteration-43-the-arithmetic-in-it--answered-2026-08-19),
answered the same day — the "4.3×" compared production against the
bfp16-emulated datapath's trace. Note therefore that this thread's own framing
("compute-shaped, but not compute") is half-superseded: it IS compute, on the
fp32 vector datapath.

### T2 — Claim B-reuse via cascade · **RETIRED 2026-08-19**
**Retired by [T1](#t1--what-is-the-gemms-3-ms--answered-2026-08-19).** B-reuse
removes *bytes*; bytes are not the constraint. The 1.26–1.68× that
[`0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md) priced was priced with
the model 0048 refutes, and the cascade milestone
[`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) scoped existed only to
free channels *for* B-reuse.

The channel census from [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) and
[`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) keeps its value — it is
about what the array can express, not about B.

### T3 — Device-resident intermediates (expert review §6b) · **OPEN**
Raised in [note 0005](notes/0005-expert-review-tests.md) §6b, deferred *with
cause*, unblocked by [`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md),
re-priced by [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) Part 4 at
**33% of the encode** — against §6b's original ~70 ms estimate, which never
counted the C readback. Needs the one-operator-per-core design that 0032's
16 KB program-memory wall dictates.

### T4 — Finish [`0043`](../tasks/0043-m9-attention-geometry/TASK.md) · **OPEN**
Its Results section is empty. The four artifact sets are exported and the
commands are recorded; only the measurements are missing.
[note 0007](notes/0007-unused-iron-surface.md) §1.1 additionally reopens its
`cols ≤ 4` conclusion via mem-tile `pad_dimensions`, and §3.4 supplies
whisper-xdna's warning that fused attention was built elsewhere, was correct,
and still lost.

### T5 — Does DMA compression pay on real weights? · **OPEN** · cheap, decisive
[note 0007](notes/0007-unused-iron-surface.md) §2. Lossless, unused by
mlir-aie's own passes for the compute-tile path, and the only published ratio is
**1.39× on `arange`**. One run of `basic/dma_compression`'s `cmp_only` with a
real `.npue` tile answers it permanently.

### T6 — Does `aie_stream=` free a DMA channel? · **OPEN** · one probe
[`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) closed B-reuse on channel
exhaustion. `aie_stream=(end, port)` makes a producer wire-only with no L1
buffer ([note 0007](notes/0007-unused-iron-surface.md) §1.6); whether it also
costs no channel is unknown and directly relevant.

### T7 — `gelu_poly.cc` narrows through an emulated fp32 multiply · **OPEN** · free
[`0045`](../tasks/0045-m9-bf16-gemm-epilogue/TASK.md) measured
`aie::mul(v, 1.0f).to_vector<bfloat16>()` at **34 `vmul.f`/`vadd.f` per 64
elements** against **0** for `accum::from_vector`. `gelu_poly.cc` uses the
expensive form in two places, and
[`0026`](../tasks/0026-m7-closing-on-cpu/TASK.md) called that kernel "at the
machine's fp32 vector limit" — a limit measured with avoidable emulated ops in
it. Dormant while eltwise runs on the host.

### T8 — `gelu_poly` degree 8 → 7 · **OPEN** · ~9% of that kernel
[note 0007](notes/0007-unused-iron-surface.md) §3.2: degree 8 sits at 3.6e-04
against a 2.465e-03 bf16 floor, degree 7 at 5.9e-04 — still 4.2× inside.
One Horner step of eight. Dormant with T7.

### T9 — `xrt::runlist` · **OPEN** · 0.9% today
[note 0007](notes/0007-unused-iron-surface.md) §3.3. Present in our XRT, unused
by our runtime, and worth having the moment dispatch count rises and dispatch
size falls — i.e. after T3 or T4.

### T10 — `aie::exp2` never measured · **OPEN** · small
[`0020`](../tasks/0020-m5-layernorm-kernel/TASK.md). Superseded in practice by
our own `exp2_poly` ([`0030`](../tasks/0030-m7-expert-review-tests/TASK.md)),
but the comparison was never made.

### T11 — Is the hw_context partition width settable at creation? · **OPEN** · low value
[note 0004](notes/0004-context-switch-cost.md) §1 and
[`0025`](../tasks/0025-m7-batching-and-crossover/TASK.md). Decided in the
driver's create-hwctx call, not at XRT level. Low value now that production is
one xclbin at a fixed width.

### T12 — Larger L2 megatiles · **OPEN** · re-pointed by T1
[`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md) §3, citing
[2602.06063](https://arxiv.org/abs/2602.06063)'s 5.9 → 13.7 TOPS from megatile size alone.
Was filed as a *bandwidth* lever, which T1 has now retired. If the megatile
result is real it must be acting through **iteration count** (bigger tiles) —
i.e. it is [T17](#t17--bigger-l1-tiles-which-need-more-l1-than-a-core-has--open--the-successor-lever)
in L2 clothing, and the paper is worth re-reading with that question.

### T13 — Explain the pre-tiled instability · **OPEN** · "or stop caring"
[`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md) §4: best-case pre-tiled
runs match row-major exactly, so it is an intermittent stall, not a ceiling.
Pre-tiling was refuted as a lever, so this only matters if it is a symptom of
something else.

### T14 — Per-core: operand prep or accumulator dependency? · **ANSWERED 2026-08-19**
**Neither — the datapath choice.** Folded into T16 and answered with it
([`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md)): the plain-bf16
inner loop is near-perfectly packed (a `vmac.f` in essentially every VLIW
bundle, operand shuffles dual-issued alongside), so operand prep costs ~nothing
and the 32-MACs/cycle ceiling is the fp32 vector datapath itself. 0003's static
prediction (~28) was right all along.

### T15 — `tasks/README.md` has no rows for 0038–0043 · **OPEN** · documentation debt
Noted in [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md). Deliberately not
back-filled from `CLAUDE.md` summaries, because an index entry written from a
summary rather than from the task is the kind of second-hand claim this repo
does not keep. Needs reading the six tasks.

### T16 — Why is a k-block iteration 4.3× the arithmetic in it? · **ANSWERED 2026-08-19**
**It is not — the 145-MACs/cycle baseline was the bfp16-EMULATED datapath.**
[`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md): every row of 0007's
"148.9–149.9 traced" artifact carries `emulate_bfp16: true` and the 1.04e-02
error signature; M2's 137–142 are the same path, and M2's own plain-bf16 figure
was always **25.0**. The production datapath traced today at 4 columns:
**7,813-cycle k-block window of which 6,144 = exactly 768 MMAC steps × 8
`vmac.f` — the fp32 datapath at its hard 32 MACs/cycle limit** — plus 1,669
non-vector in-window and an 84-cycle gap. LOCK/STREAM_STALL ≈ 0. At the
documented 1.808 GHz that is 4.37 µs, 93% of 0048's fitted 4.72 µs (0048's
"5,900 cycles" used an implicit wrong clock). **The GEMM is compute-bound on
the fp32 datapath at ~100% code quality; only the 22% non-vector share is
amortisable by any tile-geometry lever.** Under emulation the same design is
the opposite — 39% vector busy, lock-stall gaps, DMA-bound — which is what
M2's "starved, not slow" actually described, and why bytes measured free in
0048.

### T17 — Bigger L1 tiles, which need more L1 than a core has · **OPEN** · re-priced DOWN by T16
Was "the successor lever" under the assumption that per-iteration cost is fixed.
[T16](#t16--why-is-a-k-block-iteration-43-the-arithmetic-in-it--answered-2026-08-19)
measured it: **78% of the iteration is `vmac.f` compute that scales with
m·k·n**, so bigger tiles amortise only the ~1,753 non-vector cycles — **≤1.29×
with all overhead gone, ~1.06× for (64,64,64)**. The L1 wall itself is
unchanged: `(64,64,48)` costs 53,248 B of 63 KB, every legal larger geometry
overflows, and cross-tile `Buffer`
([note 0007](notes/0007-unused-iron-surface.md) §1.2) remains the one way past
it — now priced as a small lever, not the lever.

### T18 — `--probe-streams` and `--bench` disagree by up to 10% on array time · **OPEN**
[`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md). Mean per-dispatch:
probe 3,328 µs vs bench 3,028 µs (fp32 C); 3,038 vs 2,999 (bf16 C). Both on an
idle array with the gate green. The probe dispatches back to back with no buffer
syncs; bench syncs around each dispatch. Until it is explained, **neither number
should be quoted as "the" array time** — though 0048's conclusion is a
within-run comparison and does not depend on it.

### T19 — Stationary-B single buffering, to make `k` bigger · **OPEN** · re-priced to ≈1.08× by T16
Falls straight out of [T1](#t1--what-is-the-gemms-3-ms--answered-2026-08-19):
iterations go as `1/(m·k·n)`, so raising `k` cuts them directly. **`k` is already
maximal under today's budget** — everything is double-buffered, giving
`4mk + 4kn + 8mn`, and at `(64, ?, 48)`:

| k | today (all double-buffered) | Stationary-B (`B` single) | 384 % k | 1536 % k |
|---:|---:|---:|---:|---:|
| **64** | 53,248 **OK** | 47,104 OK | yes | yes |
| 88 | 64,000 OK | 55,552 OK | no | no |
| **96** | 67,584 **over** | **58,368 OK** | **yes** | **yes** |
| 128 | 81,920 over | 69,632 over | yes | yes |

So `k = 96` is **illegal today and legal under the Stationary-B budget**
`2mk·T_in + kn·T_in + 2mn·T_out` that `CLAUDE.md` trap 3 records from
ICPP '25 — the paper that says Stationary-B
beats the Stationary-C algorithm our stock IRON matmul uses.

Using [`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)'s fit, `k` 64 → 96
took `ffn_up` from **4,198 → 3,267 µs, 1.28×** — but that assumed the
per-iteration cost is fixed.

**T16 answered the condition, and it fails**
([`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md)): 78% of the
iteration is compute that scales with `m·k·n`, so `k = 96` amortises only the
~1,753 non-vector cycles — **≈1.08×, not 1.28×**. Still positive, and it
stacks with nothing else on the ledger, but it now has to beat its own costs:
single-buffering B risks losing fetch/compute overlap (which would eat the
8% directly), and `k` is part of `gemm_b_layout`, so the `.npue` repacks.

**A cheaper mechanism for the same 1.08×, 2026-08-19**: [ATB
(2511.16041)](https://arxiv.org/abs/2511.16041) reaches the L1 relief by shrinking **A's
buffered M** (its lifetime is one C-row accumulation) instead of
single-buffering B — everything stays double-buffered, so the overlap risk
above disappears. At our (64,64,48), ρ=2 on A frees 8,192 B. If this thread is
ever built, build it ATB-shaped, not Stationary-B-shaped.

### T20 — int8 · **OPEN** · promoted by T16: a datapath lever, and those are now the only big ones
[note 0007](notes/0007-unused-iron-surface.md) §3.5. `_MM_MAC_DIMS["aie2p"]`
gives int8 **(8, 8, 8)** against bf16's **(4, 8, 8)** — twice the `r` — and
whisper-xdna measured **1.33×**, not the 2× the geometry suggests.
[T16](#t16--why-is-a-k-block-iteration-43-the-arithmetic-in-it--answered-2026-08-19)
sharpened why this matters: the production GEMM runs the fp32 vector datapath
at its 32-MACs/cycle limit while the MMAC unit sits idle, so **changing
datapath is the only identified multi-× lever left on array time** (the other
route is bfp16 emulation, [T23](#t23--the-gemm-datapath-itself-bfp16-emulation-is-29-of-array-gemm-time--open--an-accuracy-decision)).
int8 would cut *cycles per iteration*, not iteration count. Accuracy is an
MTEB question, and [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) is the
precedent for how to settle it.

### T21 — One tile geometry for four shapes · **OPEN** · a tension, not yet a lever
whisper-xdna measured **2.5× and 4.6×** swings from per-shape tile sizes
([note 0007](notes/0007-unused-iron-surface.md) §3.5). Our M2/M5 finding that
per-core cost is flat to 0.4% across the four MiniLM shapes is not contradicted —
ours are all fat, theirs were not — but the one-xclbin architecture *requires* a
single geometry, and [T1](#t1--what-is-the-gemms-3-ms--answered-2026-08-19) has
just made geometry the thing that matters. Unmeasured here.

### T22 — mlir-aie 1.4.1 · **OPEN** · housekeeping
[note 0007](notes/0007-unused-iron-surface.md) §3.6. We are on 1.3.4;
phoenix-sdr-dsp pins 1.4.1 and documents that the rolling wheel channel silently
resolves back to 1.3.4. Not urgent — every feature note 0007 lists is already in
our 1.3.4 — but the gap will widen.

### T24 — 0048's iteration fit missed bge-base by 27% · **OPEN** · bounds a model we quote
[`0051`](../tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md). Predicted ~230
seq/s from `t = 573 µs + 4.72 µs × iterations`, measured **181.2**. The fit was
made on h=384 dispatches and extrapolated across a width doubling it was never
tested at, and it models array time only while the prediction was of end-to-end
throughput. **Treat the fit as valid at h=384 and unvalidated elsewhere** until
someone runs `--probe-streams` on the h=768 design, which would separate the
array term from the host term and cost one command. Note bge-base is **74.1%
NPU** against MiniLM's ~40%, so its host term is the *smaller* one — the miss is
most likely in the array term, i.e. in the fit itself.

### T25 — bge-base has no MTEB gate and no interleaved CPU ratio · **OPEN** · cheap
[`0051`](../tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md) validated it to
`1-cos` 1.353e-05 and end-to-end 2.613e-05 with top-10 overlap 1.0000, which is
the correctness bar. It does **not** have the two things
[`0035`](../tasks/0035-m8-mteb-gate/TASK.md) and
[`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md) established as the bars
for *quality* and for *speed claims*: five MTEB tasks against the CPU on the
same checkpoint, and a round-robin interleaved throughput ratio. Until both
exist, the README quotes bge-base's seq/s **without** a ratio, deliberately.

### T23 — The GEMM datapath itself: bfp16 emulation is 2.9× of array GEMM time · **OPEN** · an accuracy decision, not ours to make
[`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md). On the same design,
same shape, the emulated path costs 2,684 cycles per k-block against plain
bf16's 7,897 — **2.9× of array GEMM time**, worth roughly **1.35× end to end
on MiniLM (39.4% wait) and 1.66× on bge-large (60.8% wait)**. `--emulate-bfp16`
was retired when it was worth +2.2% end to end
([`0026`](../tasks/0026-m7-closing-on-cpu/TASK.md)) and failed the 1-cos gate at
3.470e-03; **the pricing half of that retirement is superseded, the accuracy
half stands.** T16 also showed the emulated design is DMA-bound (39% vector
busy, lock-stall gaps), so under emulation the retired byte-levers (B-reuse,
T2) would partially revive. Reopening this means running MTEB on the bfp16
path per [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) — flagged for the user;
per the 0045 precedent, datapath accuracy decisions are made by the user on
MTEB evidence, not unilaterally. [T20](#t20--int8--open--promoted-by-t16-a-datapath-lever-and-those-are-now-the-only-big-ones)
(int8) is the same decision shape with different numbers.

**Upside re-priced 2026-08-19 by [ATB](https://arxiv.org/abs/2511.16041)** (UCLA+AMD, our
SKU, our toolchain, web-indexed): with asymmetric tile buffering and a
hand-optimised microkernel, BFP16 GEMM reaches **24.3 TFLOPS** on this silicon
— ~8× our production array rate, of which 2.88× is microkernel work alone
(stock 0.32 → 0.92 TFLOPS/core). The 2.9× above is what the *stock* emulated
path buys; it is the floor of this decision, not the ceiling.

---

## Closed

| thread | status | where |
|---|---|---|
| Is `aie::vector<float>` really IEEE fp32? | **ANSWERED** — yes, ~24 mantissa bits | [`0016`](../tasks/0016-m5-fp32-probe/TASK.md), refuting [`0015`](../tasks/0015-m5-gelu-polynomial/TASK.md) |
| What carries GELU's 3.886e-03 implementation error, if not fp32 precision? | **ANSWERED** — the default `floor` rounding mode | [`0044`](../tasks/0044-m9-optimisation-sweep/TASK.md) Part 3, chasing [`0016`](../tasks/0016-m5-fp32-probe/TASK.md)'s own hypothesis after 28 tasks |
| Can B reuse be expressed with `consumer_obj_type`? | **ANSWERED** — no; no spare DMA channel exists | [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) |
| Does cascade free the channels B-reuse needs? | **ANSWERED** — frees inputs 6/6→3/6, costs outputs 3/6→6/6 | [`0047`](../tasks/0047-m9-cascade-channel-probe/TASK.md) |
| LayerNorm still opens three fifos per core | **ANSWERED** — params broadcast from the mem tile, 8 columns | [`0030`](../tasks/0030-m7-expert-review-tests/TASK.md) |
| M6 speed not measured | **ANSWERED** — deliberately deferred to M7, then measured | [`0023`](../tasks/0023-m7-full-cpp-encode/TASK.md) onward |
| Is bge-small a byte-identical drop-in? | **ANSWERED** — no; 12 layers and CLS pooling are data, not constants | [`0039`](../tasks/0039-m9-bge-small/TASK.md) |
| `pack_npue.py` had not run in months | **ANSWERED** — broken import found and fixed | [`0036`](../tasks/0036-m8-tokenizer/TASK.md) |
| Is the centred polynomial basis worth 2.5×? | **RETIRED** — measured, worth nothing at fp32 | [note 0007](notes/0007-unused-iron-surface.md) §3.2 |
| `AIE_LOOP_UNROLL_FULL` | **RETIRED** — 14% slower on straight vector loops | [note 0007](notes/0007-unused-iron-surface.md) §1.9 |
| `burst_length` tuning | **RETIRED** — already maximal by default | [note 0007](notes/0007-unused-iron-surface.md) §2 |
| MTEB on the bf16-C datapath | **RETIRED** — datapath decided as bf16 in / fp32 out, 2026-08-19 | [`0045`](../tasks/0045-m9-bf16-gemm-epilogue/TASK.md) |
| `--emulate-bfp16` | **RETIRED** — fails accuracy; closed by the MTEB gate | [`0035`](../tasks/0035-m8-mteb-gate/TASK.md) |
| Pre-tiling as a performance lever | **RETIRED** — a wash under isolation | [`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md), [`0008`](../tasks/0008-m5-bfp16-real-data/TASK.md) |
