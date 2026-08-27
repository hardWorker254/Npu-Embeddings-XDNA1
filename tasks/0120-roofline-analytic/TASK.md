# 0120 — Roofline, step A: the analytic figure, built from geometry alone

**Question.** The user asked to start on roofline analysis for the NPU. Step A
of the three the session scoped: **place all 24 shipped production GEMM
dispatches on a roofline using nothing but geometry the design files already
record — no hardware, no new measurement — then check whether the figure
reproduces the diagnoses this project reached the hard way.**

**Answer: it does, on all three datapaths, and the reproduction is exact enough
to be worth trusting on the cases we have NOT measured.** The figure also
prices the levers, and finding one it *refuses* mattered as much as the ones it
approves. Two bugs fell out of building it — one in the runtime's own
`--probe-streams` byte accounting, one in the thread register.

**Nothing in this task ran on hardware.** No `iron_env.ps1`, no XRT, no
`xrt-smi`. Every y-coordinate that is a measurement is *copied* from an earlier
task with its provenance attached, and every y-coordinate that is not is
labelled a prediction.

---

## 1. Why this chip needs a different roofline than the textbook one

**There is not one compute ceiling. There are three, and which one applies is a
property of the design, not of the silicon.**

| ceiling | MACs/cyc/core | array, 32 cores @ 1.808 GHz | who runs under it |
|---|---:|---:|---|
| fp32 vector unit — plain bf16 | 32 | 1 851 GMAC/s | `bge-small-en-v1.5` |
| MMAC via bfp16 emulation | 256 | 14 811 GMAC/s | the other five, since [`0104`](../0104-adopt-bfp16-per-model/TASK.md) |
| MMAC int8 | 512 | 29 623 GMAC/s | `--int8` containers |

Plain bf16 rides the **fp32 vector unit while the MMAC sits idle** — that is
[`0049`](../0049-m9-t16-iteration-anatomy/TASK.md)'s finding, and 32 is a hard
limit, not a quality of our kernel. So `emulate_bfp16` in `design.json` does not
move a dispatch *along* a roofline; it moves it to **a different roofline**, 8×
higher. A single-roof plot of this catalogue would be wrong for five of six
models.

The figure also carries each ceiling as this repo has actually **traced** it in
isolation (0049 §1): 28.9 MACs/cyc/core plain bf16, 146.7 emulated. The gap
between the solid and dashed line of the same colour is microkernel quality; the
gap between a dashed line and a measured square is everything else.

**Bandwidth is a band, not a line.** Four numbers from four methods —
60 GB/s (Zen-Attention, AMD), <40 (Gemma3 team on Krackan), 33
([`0010`](../0010-m5-b-reuse-and-cost-model/TASK.md)'s marginal fit, R² 0.902),
28 ([`0080`](../0080-m13-int8-traffic-bound/TASK.md)'s, R² 0.987). All four are
drawn. Collapsing them to one would have been the dishonest move, and the spread
turns out to bracket the measurements neatly (§3).

**The rule-1 problem lives entirely on the y-axis.** x is `2·M·K·N / bytes` —
pure arithmetic over `M`, `K`, `N`, `tile_n`, `cols` and two dtypes, all of them
fields in `design.json`. Nothing about it can be contended by another process.
y is work ÷ time, and *time* is what rule 1 forbids taking from the wall clock.
Step A therefore plots predictions on y and marks measured points as a separate
series, each carrying the caveat [`0109`](../0109-fused-ratio-energy/TASK.md)
wrote for itself: `wait (hardware)` is *"trace-adjacent … still reported as a
host-observed duration, not a hardware trace."* Replacing those y values with
traced cycles is step B.

## 2. The x-axis, and the one place it is a modelling choice

DDR bytes per dispatch, using the accounting [`0010`](../0010-m5-b-reuse-and-cost-model/TASK.md)
fitted and both [`0048`](../0048-m9-what-is-the-gemm-time/TASK.md) and 0080
re-used:

```
A is re-streamed once per n-block group  ->  max(1, N / (tile_n * cols))
B is re-streamed once per row block      ->  M / (tile_m * 4)      = 32 today
C is written once
```

The `×32` on B is not a modelling choice we are free to make. B-reuse would
remove it and [`0046`](../0046-m9-b-reuse-asymmetric/TASK.md) established it
cannot be built at 8 columns — every core tile is at 2/2 input DMA channels,
five of eight mem tiles at 6/6. The formula describes the design that ships.

**Where the bytes are, on the five bfp16 designs as shipped:**

| model | A | **B** | C | total, one layer |
|---|---:|---:|---:|---:|
| `minilm_bfp16` | 31% | **46%** | 23% | 245 MB |
| `base_bfp16` | 35% | **52%** | 13% | 868 MB |
| `gemma_bfp16` | 34% | **51%** | 16% | 560 MB |
| `nomic_bfp16` | 34% | **52%** | 14% | 1 170 MB |
| `large_bfp16` | 46% | **46%** | 9% | 1 762 MB |
| `small_bf16` (plain bf16, fp32 C) | 25% | 37% | 37% | 302 MB |

C is down to 9–23% because the project already narrowed it. **B is now the
largest single term on every dispatch of every model.**

## 3. Does the figure reproduce what we already know?

Three independent tests, one per datapath. This is the whole point of step A: a
roofline that cannot re-derive the answers we paid for is not worth extending.

### 3a. Plain bf16 — the discriminating pair lands flat ✅

0048's control: `ffn_up` and `ffn_down` have **identical MACs** and differ
**1.50× in bytes**, so their AI differs 1.50×.

| shape | AI | measured GMAC/s | % of the 1 851 ceiling |
|---|---:|---:|---:|
| `ffn_up` | 85.3 | 1 250 | 67.5% |
| `ffn_down` | **128.0** | 1 236 | 66.7% |

**1.50× apart on x, 1.1% apart on y.** On a roofline that is only possible on
the flat roof — a bandwidth-bound pair would have to differ by 1.50× on y too.
The figure reads 0048's conclusion straight off the geometry. The three large
shapes sit at 65–68% of the ceiling, i.e. 72–75% of the *traced* microkernel
line, which is 0049's anatomy (77.8% INSTR_VECTOR, 21.1% loop bookkeeping) seen
from outside. `attn_out` is the exception at 48.2%, and §6 explains why.

### 3b. int8 — the same pair separates, in the byte direction ✅

Same two shapes, same M/K/N, one datapath over ([`0097`](../0097-t18-t21-t4-measurements/TASK.md) t21):

| shape | AI | measured GMAC/s | % of the 29 623 ceiling |
|---|---:|---:|---:|
| `ffn_up` | 170.7 | 3 140 | 10.6% |
| `ffn_down` | **256.0** | **4 057** | 13.7% |

y now **rises with x** — and the compute ceiling has become irrelevant at 10–14%
occupancy. That is 0080's "int8 put the GEMM back in the traffic-bound regime",
recovered without re-running anything.

### 3c. bfp16, the datapath that actually ships — a bandwidth roof at ~44 GB/s ✅

Four models, one aggregate point each (0109's `wait (hardware)`, four shapes
summed):

| model | AI | GMAC/s | % of ceiling | **implied GB/s** |
|---|---:|---:|---:|---:|
| MiniLM | 118.2 | 2 440 | 16.5% | **41.3** |
| bge-base | 133.6 | 3 127 | 21.1% | **46.8** |
| bge-large | 117.0 | 2 623 | 17.7% | **44.8** |
| nomic | 132.1 | 3 043 | 20.5% | **46.1** |

Apply the runtime's own instruction — *"if `GMAC/ms` is flat the design is
compute-bound; if `GB/s` is flat it is traffic-bound"* — across **models**
rather than across shapes. GMAC/s spreads **28.2%**; implied bandwidth spreads
**13.3%**, i.e. ±6% around 44 GB/s, across four architectures whose weights span
10×. Bandwidth is the tighter invariant, and 44 GB/s sits inside the documented
40–60 band.

`bge-small`, the one model that did *not* move to bfp16, reads 25.0 GB/s and
64.7% of its (much lower) ceiling — in family with every other plain-bf16 row
and nowhere near the bfp16 cluster. **The datapath decision is visible in the
figure as a change of regime, not a change of speed.**

### 3d. And the headline the three tests produce together

> The bfp16 adoption raised the compute ceiling **8.0×** (1 851 → 14 811
> GMAC/s) and bought **2.04×** on MiniLM — 1 194 GMAC/s aggregated over
> t18's four plain-bf16 dispatches against 2 440 for the shipped bfp16
> design. The dispatches landed on the bandwidth roof at AI ≈ 120 while the
> ridge point sits at **741**, so **production runs 5–7× to the left of its
> own ridge** and four fifths of the new ceiling is unreachable without
> raising AI.

## 4. Two bugs, found by making the numbers agree

### 4a. `--probe-streams` counts bytes with two constants that are not constant

`runtime/src/main.cpp:6484` computes the traffic above with **`2` hardcoded as
the A/B element size** and **`48.0 * 8.0` hardcoded as `tile_n * cols`**. Both
are `design.json` fields, and both are wrong for designs we ship:

* **int8** operands are 1 byte. Every int8 `GB/s` this project has printed is
  inflated — on the one published table (0097 t21) by **1.57× to 1.85×**, and
  *differentially*, because C is counted correctly and C's share varies by
  shape. That column read 56.5 / 35.9 / 57.4 / **58.5**, which looks flat for
  three of four shapes; corrected it is 35.9 / 22.8 / 36.8 / **31.7**, where
  `ffn_down` sits 14% below its sibling.
* **`tile_n`** is 32 on shipped bf16 `bge-large` and 64 on int8 `bge-large`
  ([`0081`](../0081-m13-int8-everywhere/TASK.md)). At 32 the A term is
  under-counted by 1.5×.

The **conclusion** t21 drew survives — GMAC/s still spreads 2.1× against
bandwidth's 1.16×, so int8 is still traffic-bound — but the printed number was
wrong and the flatness it displayed was partly an artifact. Not fixed in this
task: `tools/roofline.py` computes it correctly and the divergence is recorded
here rather than silently patched, so the two can be diffed. *(Not audited: every
other published table. Only t21 was checked line by line.)*
→ [T47](../../research/OPEN-THREADS.md#t47).

### 4b. T2 is annotated REOPENED and filed in `CLOSED-THREADS.md`

[T2](../../research/CLOSED-THREADS.md#t2) (B-reuse) was retired 2026-08-19 on
"bytes are not the constraint", then carries an in-place block: *"REOPENED for
the int8 datapath, 2026-08-22."* It never moved back to `OPEN-THREADS.md`, and
`CLAUDE.md` states there are three live threads.

[T27](../../research/CLOSED-THREADS.md#t27) is the sharper case. It is closed
"**ANSWERED for int8, which replaced bfp16 as the fast datapath**", and its
reasoning rests on the sentence *"bfp16 was never adopted (T23 is an accuracy
decision nobody took)"*. That was true when written. Tasks
[`0103`](../0103-t23-bfp16-all-models/TASK.md)/[`0104`](../0104-adopt-bfp16-per-model/TASK.md)
then took the decision, and **0.4.0 ships bfp16 on five of six models with int8
behind a flag** — the exact reverse of the premise. The levers T27 priced were
priced on int8 traffic, whose composition differs from bfp16's (§2).

This is rule 3's failure mode with a new shape: not a live thread left unread,
but a **closed thread whose closing premise expired**. `tools/check_register.py`
cannot catch it — it matches task *titles* against threads, and no task is
titled about T27. Filed rather than silently repaired: the substance is
[T48](../../research/OPEN-THREADS.md#t48), the mechanism is
[T49](../../research/OPEN-THREADS.md#t49), and both closed threads now carry a
pointer forward instead of being edited.

## 5. What the figure prices

On a bandwidth-bound dispatch, array time is proportional to bytes and nothing
else, so a lever's value is a byte ratio and needs no hardware to compute. End
to end it is then multiplied by 0109's measured array share.

**Lever 1 — B streamed once (B-reuse).** The largest term, replicated 32×:

| model | traffic now | with reuse | AI | array | **end to end** |
|---|---:|---:|---:|---:|---:|
| MiniLM | 245 MB | 136 MB | 118 → 214 | 1.81× | **1.18×** |
| bge-base | 868 | 429 | 134 → 270 | 2.02× | **1.31×** |
| nomic | 1 170 | 585 | 132 → 264 | 2.00× | **1.34×** |
| bge-large | 1 762 | 981 | 117 → 210 | 1.80× | **1.34×** |
| bge-small | 302 | 192 | 96 → 151 | **1.00×** | **1.00×** |

`bge-small` prices at exactly 1.00× because it is on the flat roof — the same
reason 0048 retired this lever in the first place. **The retirement was correct
for the datapath of the day and is wrong for the one that ships.** It remains
channel-blocked (0046); this is a price, not a plan.
→ [T48](../../research/OPEN-THREADS.md#t48).

**Lever 2 — narrowing `bge-small`'s fp32 C.** 19% fewer bytes, priced at
**1.00×**. [`0045`](../0045-m9-bf16-gemm-epilogue/TASK.md) measured ~0% of array
time. The figure agrees, and that agreement is worth more than the lever.

**Lever 3 — `tile_n` on `bge-large`, and the figure refusing it.** bge-large has
the highest A share of any model (46%) and A re-streams `N/(tile_n·cols)` times,
so raising `tile_n` from 32 looks free. It is not, and the two candidates fail
for *different* reasons:

| `tile_n` | N divisible by `n·cols`? | L1 (trap 3) | traffic |
|---:|---|---|---|
| 48 | **no** — 1024 mod 384 ≠ 0 | 53 248 B, fits | (1.18× if it were legal) |
| 64 | yes — 1024/3072/4096 all divide 512 | **65 536 B of 64 512 — over** | (1.30×) |

The 65 536 B is not a new derivation: it is the number `gemm_pretiled.py:355`
already carries in a comment, and it survives even though C is now bf16, because
L1 holds the fp32 accumulator `Buffer` **and** the narrowed bf16 output tile.
Worth stating because `CLAUDE.md`'s own account of why bf16 bge-large ships at
32 cites only int8's 1-byte *operands* — the `out` term halving under bf16 C is
the obvious follow-up thought, and it does not rescue the tile.

## 6. What step A cannot do — stated, not hidden

1. **A roofline has no fixed-cost term, and this machine has a large one.**
   0010 fitted 150 µs, 0048 573 µs, 0080 627 µs. It shows up as one consistent
   artifact: **`attn_out` — the smallest dispatch — reads low on every datapath
   we have measured**, at 64–78% of its same-datapath sibling's occupancy
   (48.2% vs 65.1% plain bf16; 51.1% vs 65.8% with bf16 C; 6.6% vs 10.3% int8).
   The same deficit in three datapaths is a per-dispatch constant, not a
   property of the shape's arithmetic intensity. It also explains why MiniLM is
   the lowest of the four bfp16 aggregates: at 1 485 µs/dispatch a ~150 µs
   fixed cost is 10% of the reading, against 1.5% at bge-large's 9 825.
2. **Every measured y is wall-clock-derived** (§1). Steps B and C exist to fix
   that; nothing in §3 should be quoted as a traced number.
3. **One DDR-level roofline only.** The memtile↔L1 and L1↔register legs are not
   modelled, and a design can be bound there while looking healthy here.
4. **`embeddinggemma-300m` has no aggregate point** — it has no `--bench` mode
   (0109 §2), so its array share was never measured. Left blank rather than
   borrowed from a model with a different geometry.

## 7. Commands

No hardware, no `iron_env.ps1`. Interpreter is the conda `iron` base;
`matplotlib` exists in none of this repo's environments, so the figure is
emitted as hand-written SVG.

```powershell
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\roofline.py `
    --designs "dist/npuembeddings-0.4.0/artifacts_*/gemm_rtp" `
    --measured tasks\0120-roofline-analytic\measured.json `
    --out tasks\0120-roofline-analytic\roofline

& "C:\Users\vegar\.conda\envs\iron\python.exe" tasks\0120-roofline-analytic\levers.py
```

## 8. Artifacts

| file | what |
|---|---|
| `tools/roofline.py` | the model; reads `design.json`, emits table + JSON + SVG |
| `measured.json` | every landmark re-plotted, each with its source task and the rule-1 caveat |
| `roofline.json` | 24 dispatches + 17 measured points, all intermediate byte terms kept |
| `roofline.svg` | the figure |
| `levers.py`, `raw.txt` | the pricing in §5, and the verbatim output of both commands |

## 9. What step B should measure

The figure's weakest joint is the y-axis, and the cheapest fix is the one the
project already owns: **traced cycles at 4 columns** (trap 7 — 8 columns cannot
be core-traced). Three asks, in value order:

1. **A traced point on the bfp16 datapath at production tile width.** §3c rests
   on four host-observed aggregates. One traced dispatch converts the ~44 GB/s
   roof from an inference into a measurement.
2. **Separate the fixed cost from the slope.** Sweep `M` on one shape and one
   datapath; the intercept is the term §6.1 says the roofline cannot see, and
   subtracting it is what makes `attn_out` plottable.
3. **The mem-tile leg.** `PortEvent(CoreEvent.PORT_RUNNING_0, …)` is already
   in the tracing surface (`docs/05-measurement`), so an L2↔L1 roofline is a
   second figure rather than a second instrument.

## 10. Problems hit

* `matplotlib` is in no environment this repo owns (checked `.venv-ref` and the
  conda `iron` base). Rather than install into either — the fragility that bit
  `.venv-ref` on 2026-08-23 — the plotter writes SVG directly. It is also
  diffable, which a PNG is not.
* A bash heredoc could not carry the tool's source (unterminated-quote parse
  error at an unrelated line); written with the file tool instead. Noted only
  because the failure mode was misleading.
* The first draft plotted analytic points **on** their compute ceiling, which is
  true but useless — it shows where a dispatch *could* run, not where the model
  says it *will*. Changed to `min(ceiling, AI × 40 GB/s / 2)`, taking the
  documented band's lower edge as a stated constant so the circles are a
  prediction rather than a fit.

## 11. Threads filed

Five, all on 2026-08-27, all from this task. The register went from three live
threads to eight — which is what a new instrument does: it does not answer
questions so much as give previously uncoordinated ones a coordinate system.

| thread | what it owes |
|---|---|
| [T45](../../research/OPEN-THREADS.md#t45) | the y-axis is wall clock; one traced bfp16 dispatch, and an M-sweep intercept for the fixed cost §6.1 says the model cannot see |
| [T46](../../research/OPEN-THREADS.md#t46) | only the DRAM leg is drawn; memtile↔L1 is a second figure off the same trace, and is *not* urgent, because 0049 already accounts for today's gap |
| [T47](../../research/OPEN-THREADS.md#t47) | `--probe-streams`' byte accounting, and the audit of every published `GB/s` that this task did not do |
| [T48](../../research/OPEN-THREADS.md#t48) | B-reuse re-priced on the shipping datapath, and the two expired premises that retired it |
| [T49](../../research/OPEN-THREADS.md#t49) | a closed thread whose closing premise expired — the third distinct failure mode this register has produced |

[T2](../../research/CLOSED-THREADS.md#t2) and
[T27](../../research/CLOSED-THREADS.md#t27) were **not edited**. Each gained a
pointer forward, because rule 3b's whole point is that the reasoning which was
sound for its own configuration is the valuable part.

`python tools/check_register.py` → `register OK`, 8 open / 43 closed.
