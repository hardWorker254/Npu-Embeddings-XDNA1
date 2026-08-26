# Research directions

**What this file is.** [`OPEN-THREADS.md`](OPEN-THREADS.md) is the register of
questions this project has *already asked* and not answered. This file is the
other half: **which directions are worth taking next, and why**, ranked, with
what each would cost and what would make it fail.

Written 2026-08-22, after 0074–0076, against a working premise the user stated
directly: *"repoets største verdi nå er enorme mengde tekst som andre kan leke
seg med"* — the biggest value here is now the volume of written-down work
others can use.

**I think that premise is right, and sharper than it sounds.** The array
performance is good but not unique; AMD, STEEL and ARIES all report bigger
numbers on this silicon. What no one else has published is **a continuous,
dated, adversarial record of an XDNA2 project being wrong in public** — ~80
tasks, ~35 threads, and a dozen substantial claims that were believed,
measured, and killed, each with the measurement that killed it. That is the
scarce artifact. The top two directions below follow from taking it seriously.

---

## Tier 1 — make what already exists legible

These produce no new measurements. They are the highest ratio of value to risk
in the repository right now, and neither needs the NPU.

### R1. `docs/refuted.md` — the killed claims, as a first-class document

**The claim.** This project's most transferable output is its list of things
that were plausible, believed, acted on, and then falsified. Nobody publishes
these; papers report what worked. The material already exists but is scattered
across `docs/history.md`, task logs and thread updates, where it is only
findable by someone who already knows it is there.

Candidates, each already measured:

| believed | killed by | what replaced it |
|---|---|---|
| pre-tiling B is a performance lever | [`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md), −11% mean | it is a wash; keep it for the fusions |
| the GEMM is bandwidth-bound (`t = 150 µs + traffic/33 GB/s`) | [`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md) — `ffn_up`/`ffn_down` have identical MACs, 1.5× the bytes, and 1.8% the difference | bound by **iteration count** |
| ~4,500 cycles per iteration are unaccounted for | [`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md) — the 145 MACs/cycle baseline was traced with `--emulate-bfp16` ON | there was no gap; ~100% of the fp32 datapath |
| B-reuse is a 1.26–1.68× lever | [`0046`](../tasks/0046-m9-b-reuse-asymmetric/TASK.md) DMA channels, then 0048 retired the premise | out of channels *and* the bytes were free |
| ONNX Runtime is the stronger CPU baseline | [`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md) — 234 vs torch's 489 seq/s | torch, interleaved |
| bfp16 is an accuracy disqualifier | [`0008`](../tasks/0008-m5-bfp16-real-data/TASK.md) on hardware; then [`0052`](../tasks/0052-m10-research-night/TASK.md) inverted it again | still open ([T26](OPEN-THREADS.md#t26)) |
| MQA's `tile_n=16` rules EmbeddingGemma out | [`0074`](../tasks/0074-m13-gemma-on-npu/TASK.md) — pad the fused qkv | 2.7× more than the floor implied |
| parity needs hidden ≈ 1300–2000 | [`0027`](../tasks/0027-m7-width-hypothesis/TASK.md) withdrew its own projection | no projection; the per-core claim stands |

**Why now.** 0075 just moved 1,027 lines of this into `docs/history.md`, so it
is in one place for the first time. Turning it into a *structured* document is
now mostly editing.

**What makes it fail.** Writing it as a victory lap. The value is in stating
**why each was plausible** — most were reasonable inferences from real data —
and what *class* of error it was. A list of mistakes with no mechanism teaches
nothing.

**Cost:** one session, no hardware.

---

### R2. A methodology note: how to measure an NPU without fooling yourself

**The claim.** This project has independently rediscovered the same family of
measurement failures at least nine times, and the patterns generalise well
beyond XDNA2:

- **Contention that hits one side makes a *ratio* confidently wrong, not
  merely noisy.** Interleaving fixes drift that hits both sides; it cannot fix
  a resident foreign context or a runaway `find` burning one core. Both
  happened here, in opposite directions.
- **An absent data source is not a negative reading.** A hand-rolled check
  reported "ON BATTERY" for a machine with no battery, because `Win32_Battery`
  returned nothing and the `else` branch fired.
- **Lifetime CPU is not current CPU.** `Get-Process | Sort CPU` ranks a process
  that finished an hour ago above one burning a core right now. Only a delta
  over a window separates them.
- **A status line that reports the *intention* rather than the *value* is a
  fail-open.** The packer printed `tile (64, 32)` while tiling at 48.
- **Any script writing a result to a constant path is an A/B waiting to
  overwrite its own baseline.** Three instances, one of them found while fixing
  another.
- **A golden fixture that tiles ONE corpus is structurally blind to wrong-row
  bugs.** A threaded race PASSED the gate and failed a 13-sentence encode at
  1-cos 0.44 ([T32](OPEN-THREADS.md#t32)).
- **Bit-identical output verifies correctness only.** Genericity that replaces
  a compile-time constant with a runtime variable is a performance change by
  construction — a 2× attention regression shipped inside a commit verified
  bit-identical.

**Why now.** These are currently expressed as ~9 separate warnings inside
CLAUDE.md and task logs. As one note they are a contribution; as scattered
warnings they are project lore.

**Cost:** one session, no hardware. Pairs naturally with R1.

---

## Tier 2 — the open technical questions worth real effort

### R3. True layer fusion ([T28](OPEN-THREADS.md#t28)) — the largest unclaimed lever

**Status:** built at GROUP=2 with real compute at every stage
([`0062`](../tasks/0062-m11-t28-hierarchical-merge/TASK.md)), `rel_fro`
1.726e-03 against an independent fp64 reference. Blocked from production scale
because a real second-stage matmul needs L1 for a resident weight *and* an
accumulator on top of the gathered hops.

**Why it is still the biggest thing.** Four independent external measurements
on this silicon point at it, and they are not small: AMD cut 15 dispatches per
layer to 3 on our SKU; STEEL measured fused attention at **22.8×** over the
layer-by-layer IRON equivalent; ARIES beat the vendor overlay 1.24× *with
scalar, unvectorised code* purely by handing intermediates between adjacent
tiles' L1; Estévez hit 95% of peak on all 32 tiles with a design that
dispatches once and moves no data.

**The one identified way to give a worker more L1** is cross-tile `Buffer`
([note 0007](notes/0007-unused-iron-surface.md) §1.2) — a core reading its
neighbours' L1, which is the API for what ARIES did. Untried.

**What makes it fail.** [note 0007](notes/0007-unused-iron-surface.md) §3.4
carries whisper-xdna's warning that fused attention was built elsewhere, was
correct, and still lost. Fusion is not automatically a win; it has to beat a
dispatch path that is already at ~100% of the fp32 datapath.

**Cost:** several sessions. Highest risk, highest ceiling.

---

### R4. The bfp16 accuracy inversion ([T26](OPEN-THREADS.md#t26)) — an unexplained result

**The finding.** bfp16 + fp32-C reproduces history at `1-cos` 2.395e-03 (FAIL).
bfp16 + **bf16-C** measures **6.6× more accurate**, passes every gate, and MTEB
confirms it independently. **Nobody knows why.** The standing hypothesis is
that the fp32-C fifo path re-quantises C partials at every k-block under
emulation while the bf16-C accumulator-`Buffer` path does not.

**Why this is the most interesting open question in the repository.** It is a
genuine anomaly on real silicon, the mechanism is testable, and the answer
re-prices a **2.9× array-GEMM lever** that is currently parked on accuracy
grounds. "More precision in the accumulator made the answer worse" is the kind
of result that is either a bug in the toolchain, a real property of the
emulation path, or a mistake in our own measurement — and all three are worth
knowing.

**How to attack it.** A k-block sweep: if the hypothesis holds, the fp32-C
error should grow with the number of k-blocks and the bf16-C error should not.
That is one design, two builds, one afternoon — and it either confirms the
mechanism or kills it cleanly.

**Cost:** one to two sessions. **Best value-to-effort ratio of anything in
Tier 2, and I would do this one first.**

---

### R5. A cost model that works across widths

**The claim.** [`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)'s
`t = 573 µs + 4.72 µs × iterations` was fitted at h=384. It missed bge-base by
**27%** ([`0051`](../tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md)) and then
predicted Gemma's array rate correctly at h=768
([`0074`](../tasks/0074-m13-gemma-on-npu/TASK.md), 176 measured against a
140–162 prior). Three widths and two architectures now have real data.

**Why it matters more than it looks.** The model is what made 0074 *cheap*: the
padding trick was priced at 2.7× before a single line was written, which is why
it was worth building. A model with an explicit width term would let the next
"is model X worth supporting?" question be answered in an hour instead of a
week — and that question is now recurring, because `add` exists.

**What makes it fail.** Overfitting three points. The honest output may be "the
fixed term and the per-iteration term each depend on width, here are the two
curves, here is the residual" rather than a single formula.

**Cost:** one session, mostly re-analysis of stored traces.

---

### R6. int8 ([T20](OPEN-THREADS.md#t20)) — the last untried datapath

`aie2p` has native `(8,8,8)` mac dims for int8. It has never been tried here.
Embedding models are among the most quantisation-tolerant transformers, MTEB is
already wired as the accuracy gate, and [T16](OPEN-THREADS.md#t16) established
that **datapath changes are the only multi-× array levers left**.

**What makes it fail.** Calibration is real work, and this project has no
quantisation infrastructure at all. Budget for that, not for the kernel.

---

## Tier 3 — smaller, and cheap

- **[T34](OPEN-THREADS.md#t34) §3, the host side.** Gemma's array and host are
  at 48.9%/46.0% — the most balanced encode here. `--c-bf16` halves the C
  readback (11.5%) and host attention (14.7%) is an unoptimised loop. **Note
  the tension:** lanes are worth 1.54× on this model *because* the two sides
  are balanced, so speeding the host partly cannibalises the overlap. Measure,
  do not assume.
- **[T5](OPEN-THREADS.md#t5), DMA compression.** One run of
  `basic/dma_compression`'s `cmp_only` with a real `.npue` tile answers it
  permanently. The only published ratio is 1.39× on `arange`.
- **[T35](OPEN-THREADS.md#t35), a finetune with added tokens.** The cheapest
  test in this file and it has not been done.
- **Matryoshka truncation quality.** EmbeddingGemma supports it, the container
  declares it `not_implemented`, and nothing has measured what truncating to
  512/256/128 costs on MTEB. Answerable entirely on the host.

---

## Revision after the August 2026 paper batch (2026-08-22)

Seventeen off-platform papers were added the same day this file was written, deliberately
chosen for ideas rather than for our hardware
(INDEX.md, triage). Three of them change the
ranking above, and one adds a direction that was not here at all.

### R0 — NEW, and it is now the most urgent thing: **is our CPU baseline soft?**

[2608.18182](https://arxiv.org/abs/2608.18182) (Intel) reports **up to 5.8× end-to-end** on
**BERT-family encoders** — our exact model family — from SmoothQuant INT8, **upstreamed into
PyTorch and TorchAO**, with "negligible, in some cases no measurable" accuracy loss.

Every CPU ratio this project publishes (1.53×–2.49× interleaved) is against an **FP32** torch
baseline. If a user runs INT8 SmoothQuant torch on a modern CPU, those ratios could narrow
sharply or invert. Their 5.8× is a Xeon number and leans on AMX, which Zen 5 does not have —
so it will not transfer whole — but even a fraction of it matters.

**This is a measurement, not an argument, and it is one afternoon in `.venv-ref`:**
SmoothQuant one BERT model through TorchAO, run it through `compare_three.py`'s existing
interleaved protocol, report the ratio. Two reasons it is urgent rather than merely
interesting:

1. **It is more honest.** An INT8 CPU baseline against our bf16 array is closer to
   like-for-like than FP32-vs-bf16 ever was.
2. **Better we find it than a reader does.** This project's whole claim to credibility is
   that it reports its own refutations first.

Cost: one session, no NPU. **Do this before the 0.4.0 numbers are published.**

### R6 (int8) is promoted to **the top array lever** — and I under-sold it first time round

My initial reading of this batch framed int8 as a *threat* (a faster CPU baseline). That was
half the story and the less interesting half. **int8 runs on the NPU, on a datapath that is
currently idle**, and the full case is now in [T20](OPEN-THREADS.md#t20). In brief:

**The datapath.** `_MM_MAC_DIMS["aie2p"]` gives int8 **(8,8,8)** against bf16's **(4,8,8)**,
verified in our installed wheel. Attainable GEMM on this SKU is **38 TOPS int8 against
14.71 TOPS bf16** — 2.58×. whisper-xdna measured only 1.33×, and
[note 0007](notes/0007-unused-iron-surface.md) §3.5 attributed the shortfall to a movement
bound — **but that note predates 0048 and 0049**, which showed our plain-bf16 path is *not*
movement-bound and runs at ~100% of the **fp32 vector** datapath **while the MMAC unit sits
idle**. int8 moves the work onto the idle unit. Our own reason for expecting the low end of
1.33×–2.6× no longer holds.

**The accuracy blocker had a name and now has an answer.** 2209.13325 (already indexed)
documents the ±50–100 outlier dimensions in post-LN BERT that **rule out per-tensor int8
activations** — which is exactly what SmoothQuant was built to fix, and
[2608.18182](https://arxiv.org/abs/2608.18182) shows it doing so on BERT/DistilBERT/XLM-RoBERTa with
negligible loss, upstream in PyTorch/TorchAO.

**The implementation bug is pre-located.** [2608.13756](https://arxiv.org/abs/2608.13756): the int32
accumulator is provably exact and order-independent, so the danger is entirely **scale
application and output rounding** — our SRS/narrowing step, where trap 2b already caught this
project once. Mitigation supplied: power-of-two scales.

**And it compounds with a tile lever T19 could not reach.** int8 halves the operand bytes, so
the L1 budget `2·(m·k·in + k·n·in + m·n·out) < 64512` opens up: **(64,96,48) fits
double-buffered at 46,080 B where bf16 needs 67,584 and does not.** T19 closed *negative* for
exactly that reason — k=96's vector cycles scaled perfectly (1.5×, still 8 cyc/MMAC) but B
had to be single-buffered and the exposed fill grew the inter-window gap 84 → 1,193 cycles.
**k=96 divides every K in this project's model set** (384, 768, 1152, 1536, 3072), so the
k-block count would drop 1.5× across the board.

Fewer iterations *and* more MACs per cycle. Both halves are priors; the second rests on an
inference about why T19 failed rather than a re-run.

**The gate on all of it is one build:** does `gemm_pretiled.py` compile with
`dtype_in_str="i8"` and an int32 accumulator? Every design this project has ever built is
bf16. That probe costs an afternoon and separates the whole story from a fantasy — **do it
before any of the rest.**

### R3 (fusion) gains a third, and the most controlled, external data point

[2608.08730](https://arxiv.org/abs/2608.08730) cut WebGPU dispatches **876 → 564** and gained **+53%
throughput using the same shaders**, explicitly ruling out kernel quality and memory
bandwidth. Alongside AMD's 15→3 and STEEL's 22.8×, that is three independent arrivals — and
this one isolates the *mechanism* T28 would exploit better than either.

### NEW, cheap, and slightly uncomfortable: check our own 573 µs

The same paper finds that **measuring one operation in isolation overestimates per-dispatch
cost by 20×**, because it conflates dispatch with synchronisation (497 µs measured = 450 µs
sync + 24–71 µs dispatch). Their fix is a *sequential-dispatch* method: dispatch N, sync once.

[`0048`](../tasks/0048-m9-what-is-the-gemm-time/TASK.md)'s fixed term is **573 µs**, and it
is 55 ms of EmbeddingGemma's modelled 707 ms. Our fit came from real production dispatches in
sequence rather than one op in isolation, so I expect it survives — but this project has been
caught by exactly this class of conflation before
([`0049`](../tasks/0049-m9-t16-iteration-anatomy/TASK.md): the "missing 4,500 cycles" were a
baseline traced in the wrong mode). **Checking is cheaper than assuming.**

### R4's framing widens: precision as a coordinate, not a mode

[TileMix](https://arxiv.org/abs/2608.17336) routes FP16/INT8 **per score-tile group** inside one fused
kernel, with connectivity untouched. [T23](OPEN-THREADS.md#t23) and
[T26](OPEN-THREADS.md#t26) have only ever asked *which* precision. Our GEMM is already
decomposed into exactly the tiles such a decision would range over. Their shared-state
warning transfers directly and is a trap we would otherwise hit: path-specific rounding
propagates through the **normaliser**, not just the product, so a naive per-tile scheme
produces plausible wrong numbers rather than an obvious failure.

### A practice worth stealing: pre-registration

[2608.13756](https://arxiv.org/abs/2608.13756) pre-registered its per-layer predictions before measuring,
and reports 196/196 and 252/252 against that list. This project already writes priors before
building — 0074 predicted 140–162 seq/s and measured 176 — but informally, inside the task
log, where "roughly right" is hard to score afterwards. Writing the prediction into
[OPEN-THREADS.md](OPEN-THREADS.md) **before** the build costs nothing and turns a hunch into
a scored claim. Given that this repository's stated value is its record of being wrong in
public, a scored prediction list is the natural next form of that.

---

## Suggested further searches

What is conspicuously *missing* from 33 indexed documents, in rough order of value:

1. **Encoder-specific NPU work.** [Zen-Attention](https://arxiv.org/abs/2508.17593) is still the only
   paper here that benchmarks BERT on XDNA2. Everything else is decoder LLMs. Search:
   *BERT / sentence embeddings / retrieval encoders on NPU, XDNA, Hexagon, Apple ANE,
   Intel NPU* — including the competing platforms, since the architectural question
   (dispatch-bound short sequences) is shared.
2. **Block-floating-point numerics, specifically where the error comes from.**
   [T26](OPEN-THREADS.md#t26)'s inversion is unexplained and nothing in the index addresses
   MX/bfp accumulation error directly. Search: *microscaling MX formats, block floating
   point accumulation error, shared-exponent block size, OCP MX*.
3. **Cross-tile / neighbour-L1 dataflow on AIE.** [T28](OPEN-THREADS.md#t28)'s blocker is L1
   capacity and the one identified route is cross-tile `Buffer`
   ([note 0007](notes/0007-unused-iron-surface.md) §1.2), which
   ARIES did in spirit but not in IRON. Search: *AIE cascade,
   shared L1, neighbour tile buffer, MLIR-AIE memtile bypass*.
4. **Honest CPU baselines for embedding models.** We have one paper on INT8 BERT and it is
   from Intel. Search: *sentence-transformers CPU throughput, ONNX Runtime vs PyTorch
   encoder inference, AVX512-VNNI BERT* — specifically for numbers on **AMD Zen**, which is
   what we actually run against.
5. **Matryoshka / truncated embedding quality.** Nothing indexed. It is a Tier-3 item above
   and entirely host-side, so it is answerable here without any of this.

## What I would actually do next

*(revised after the August 2026 batch — the original ranking is above and unchanged in its
reasoning; this is what I would actually schedule.)*

1. **R0 — measure an INT8 CPU baseline.** One afternoon, no NPU, and it decides whether the
   ratios in `docs/06-performance.md` are still true. Do it **before** publishing 0.4.0's
   numbers, because the alternative is a reader finding it.
2. **R4 — the bfp16 inversion** ([T26](OPEN-THREADS.md#t26)). One k-block sweep either
   confirms the mechanism or kills it, and the answer re-prices a 2.9× lever.
3. **R1 and R2 alongside both**, because they need no hardware and, if the premise at the top
   of this file is right, they *are* the product.
4. **R6 — int8 ON THE ARRAY** ([T20](OPEN-THREADS.md#t20)). Not conditional on R0 — it is
   the top *array* lever in its own right, and the only multi-× one left that does not
   require fusion. Start with the one-build probe (`dtype_in_str="i8"`, int32 accumulator);
   if it compiles, the rest of the case is already assembled. R0 only changes how *urgent*
   it is.
5. **R3 — fusion** ([T28](OPEN-THREADS.md#t28)) when there is a stretch of time, with
   whisper-xdna's warning in hand that fused attention was built elsewhere, was correct, and
   still lost.

And the small one that should just be done: **check our own 573 µs** with the
sequential-dispatch method. It costs an hour and it is exactly the class of conflation this
project has already been caught by once.
