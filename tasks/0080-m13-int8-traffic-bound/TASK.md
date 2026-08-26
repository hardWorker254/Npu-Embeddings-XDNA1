# 0080 — int8 put the GEMM back in the traffic-bound regime, and 0048's retired byte levers came back with it

**Goal (user, 2026-08-22):** *"Les nøye og se om det er noe vi overser. Kvantifisering og
konvertering gjør vi TIDLIGERE - altså ikke runtime."* — continuing from
[`0079`](../0079-m13-int8-why-only-1.1x/TASK.md), which answered *"why is 1.1× all we get?"*

Headline: **the cost model that governs the GEMM depends on the datapath, and int8 changed
it.** Under bf16 the GEMM is iteration-bound ([`0048`](../0048-m9-what-is-the-gemm-time/TASK.md));
under int8 it is traffic-bound, and [`0010`](../0010-m5-b-reuse-and-cost-model/TASK.md)'s model,
which 0048 superseded for production shapes, fits it at R² 0.987. Acting on that took MiniLM
from 1104.2 to **1204.7 seq/s** and bge-large from 86.3 to **93.1**, at **better** accuracy.

---

## 1. Three hypotheses, two of them mine and wrong

0079 closed by naming the C drain as the gap between int8's 7× arithmetic and its 1.79×
dispatch, and listed "narrow C" as lever #1. That was reasoning, not measurement. Testing it
killed two hypotheses before the right one survived.

### 1a. The core trace says the array is NOT drain-stalled — wrong instrument

T1's discriminating pair — `ffn_up` and `ffn_down`, **identical MACs**, 4× the C bytes —
re-run on the int8 datapath at 4 columns, traced:

| dtype | ffn_up | ffn_down | ratio | MACs/cyc |
|---|---:|---:|---:|---:|
| bf16 | 6681.8 cyc | 7518.3 cyc | 0.889 | 29.4 / 26.2 |
| int8 | 1044.7 cyc | 1057.2 cyc | **0.988** | 188.2 / 186.0 |

Dead heat. The core does not stall on the drain under either datapath.

**But the core trace cannot see this.** The shim DMA that drains C to host memory runs after
the last compute window; the trace records core cycles, and `Encoder::gemm()` waits on the
dispatch. The arithmetic exposes it: int8's traced core time for `ffn_up` at production
geometry is 768 iterations × 1044.7 cyc = **444 µs** against a measured 1727 µs/dispatch,
while bf16's is 2838 µs against 3084. **bf16 spends nearly all of its dispatch computing;
int8 spends a quarter of it.**

### 1b. k=128 — a real geometric opening that buys nothing

int8 halves the operand bytes, so trap 3's `2·(m·k·in + k·n·in + m·n·out) < 64512` has room
the bf16 geometry never had:

| geometry | bf16 | int8 |
|---|---:|---:|
| (64, 64, 48) — shipped | 53,248 ok | 38,912 ok |
| **(64, 128, 48)** | 81,920 **OVER** | **53,248 ok** |
| (64, 192, 48) | 110,592 OVER | 67,584 OVER |

`k=128` is illegal for bf16 and legal for int8, both production K values divide it, and it
**halves the k-block iterations** (768 → 384). Under 0048's model that is close to 2×.

Built and measured. **Bit-exact at both k** (rel_fro 0.000e+00 — integers, so the gate is
equality, not a tolerance), and the four production dispatches came out **0.989×**: no
change at all. Halving the iterations bought nothing, which by itself refutes the iteration
model for this datapath — the tile geometry was inherited from bf16's budget, and it turns
out not to matter.

### 1c. Fitting both models over the four production shapes

M=8192, 8 columns, 9 dispatches per point with one outlier dropped at each end (0007's rule
that a single number is not a measurement here):

| datapath | traffic model (0010) | iteration model (0048) |
|---|---|---|
| bf16 | R² 0.709, worst **20.1%** | **R² 0.994, worst 2.6%** |
| int8 | **R² 0.987, worst 5.3%** | R² 0.568, worst **36.4%** |

**The two models swap places with the datapath.** int8: `t ≈ 627 µs + traffic / ~28 GB/s`,
which is in family with 0010's 33 GB/s and inside the 40–60 GB/s band this project has always
cited for what reaches the NPU.

**This is the finding, and it un-retires a whole class of levers.** 0048 retired B-reuse and
scoped-out [`0047`](../0047-m9-cascade-channel-probe/TASK.md)'s cascade milestone on the
grounds that "bytes are free". Bytes are free *on the bf16 datapath*. On int8 they are the
whole cost. Note this is the same regime [`0052`](../0052-m10-research-night/TASK.md)
found for emulated bfp16 (T27) — anything that makes the MACs cheap enough moves the binding
constraint to transport.

---

## 2. What the traffic model says to attack

Traffic per int8 dispatch, by 0010's accounting (A re-streamed per n-block group, B per row
block, C once):

| shape | A | B | **C** | total | C share |
|---|---:|---:|---:|---:|---:|
| qkv | 9.4 | 14.2 | **37.7** | 61.3 MB | **61.6%** |
| attn_out | 3.1 | 4.7 | **12.6** | 20.4 MB | **61.7%** |
| ffn_up | 12.6 | 18.9 | **50.3** | 81.8 MB | **61.5%** |
| ffn_down | 12.6 | 18.9 | **12.6** | 44.0 MB | 28.6% |

C is 61% of the traffic on three of the four shapes, because **int32 C is 4 bytes exactly as
fp32 was** — int8 halved both operands and not one byte of the result.

---

## 3. Measure before building: `--sim-c-bf16`

Narrowing C needs an AIE kernel, a design change, an exporter change and a runtime branch.
The accuracy question does not: rounding the accumulator in the *host* dequantiser reproduces
exactly what such a design would emit. `npuembed --sim-c-bf16` does that (int32 → fp32 →
bf16, round-to-nearest-even per trap 2b) and changes nothing else.

| model | int32 C | simulated bf16 C |
|---|---:|---:|
| all-MiniLM-L6-v2.int8 | 1.178e-03 | **1.161e-03** |
| bge-large-en-v1.5.int8 | 4.475e-03 | **4.436e-03** |

**Both marginally better**, i.e. free. The mechanism is that the int32 accumulator's low bits
sit below the int8 quantisation noise already in the operands, so 8 mantissa bits discard
nothing that was signal. Contrast [`0045`](../0045-m9-bf16-gemm-epilogue/TASK.md), where the
same narrowing on the **bf16** datapath cost a real 1.38–1.52× on 1-cos — there the operands
were not already quantised, so the accumulator's low bits still carried information.

**The simulation was faithful to four digits**: the built design measures 1.161e-03, the
number the simulation predicted. That is the cheapest validation in this task and worth
repeating as a habit.

---

## 4. Why the core CAN do this, when it could not apply the scale

This is the part that makes the lever reachable, and it is worth stating because the obvious
version of the idea is blocked.

Applying the per-column `wscale[j]` on the core — folding the dequantisation into the array —
needs a **third input stream**. CLAUDE.md trap 3b: every core tile is already 2/2 in, which is
the wall that forced [`0020`](../0020-m5-layernorm-kernel/TASK.md) to pack γ+β into a single
buffer and that [`0046`](../0046-m9-b-reuse-asymmetric/TASK.md) measured as having zero
spare capacity. Not available.

**But int32 → bf16 is a pure format conversion with no operand.** It needs nothing streamed
in. The host still applies the rank-1 `sa[i] · wscale[j] + bias[j]` exactly as before, over
half the bytes. The transport narrows; the arithmetic does not move.

---

## 5. Built

* **`experiments/m5-eltwise/kernels/narrow_i32_bf16.cc`** — the int32 sibling of 0045's
  `narrow_f32_bf16.cc`. `aie::to_float<bfloat16>(v, 0)` converts int32 → bf16 in one
  operation on Gen2, so there is no widen-then-narrow; `conv_even` rounding for trap 2b's
  reason; two independent chains against the single store unit. Three entry points
  (m·n = 1024 / 2048 / 3072), mirroring the fp32 kernel exactly.
* **`gemm_pretiled.py`** — `c_bf16` accepts an int32 accumulator, and picks source file and
  entry point by accumulator dtype. L1 is unchanged in both cases: the single-buffered
  accumulator costs `m·n·4` and the halved C fifo saves `2·m·n·2`, cancelling exactly
  (38,912 B for int8 operands either way).
* **`export_gemm_rtp.py`** — **`--int8 --c-bf16` now compose.** The previous refusal
  ("different answers to the same question") was wrong: `--int8` picks the MMAC datapath and
  the accumulator, `--c-bf16` picks the transport width of an already-reduced result. Cache
  markers already carry both dtypes, so the two int8 variants cannot shadow each other.
* **`runtime/src/main.cpp`** — a fourth dequantisation arm for `int8 + bf16 C`: one 32-byte
  streaming load carries 16 bf16 against 8 int32, same instruction count for twice the
  elements. Plus `--sim-c-bf16`, which **refuses on a bf16-C design** (it would round twice)
  and on any non-int8 design.

### And a live bug this uncovered: `serve`/`embed` could not run an int8 model at all

Adding a second int8 design set made me check how one gets *chosen*, and the answer was
wrong in a way that predates this task. `design_fits()` matched on `(op, K, N)` — and **an
int8 container and a bf16 container of the same model have identical `(op, K, N)` on every
stream.** So `pick_artifacts` handed `all-MiniLM-L6-v2.int8` a bf16 design set, which then
died at stage time on the layout hash:

```
error: layer.0.qkv: layout mismatch -- design qkv wants 94266693ea31aa67...,
       file has 177088d6bc9feb74...  The bytes would be the right size and the wrong order.
```

Fail-closed, so nothing silent — but `npuembeddings embed all-MiniLM-L6-v2.int8 …` simply did
not work, and only the flag form with an explicit `--artifacts` did. Note the near-miss:
this is the **fifth** time a selector in this repo has matched on too little (0051's
directory names, 0071's `design_fits` K-only match, T31's `qkv_n`, 0073's MTEB path), and
each time the fix is the same shape — match on data the artifact itself carries.

**Fixed by passing the container's own `b_layout_hash` into the match**, so the choice is a
fact about the data rather than about which directory sorts first. Selection now narrows
8 candidates → 2 for an int8 container and 8 → 6 for a bf16 one, and the encode runs.

**Where several sets still fit, the runtime says so** rather than picking silently:

```
note: 2 design sets serve hidden 384; using artifacts_int8_mini
      also fits: artifacts_int8c_mini  (--artifacts to choose)
```

That residual ambiguity is genuine — the i32-C and bf16-C int8 sets carry the *same*
`b_layout_hash`, because C's width is not part of B's layout. A release should carry one set
per width; a source tree legitimately carries eight.

**Regression check after touching shared selection code** — all three bf16 models reproduce
their exact historical numbers: MiniLM `rel_fro` 4.473e-03 / `1-cos` 1.086e-05, bge-large
3.763e-03 / 8.432e-06, nomic 6.119e-03 / 2.599e-05.

### Verification

| check | result |
|---|---|
| kernel vs exact int32 → RNE bf16, [2048,384,1536] | **rel_fro 0.000e+00, bit-exact** |
| bf16 rounding alone, vs exact int32 | 1.656e-03 (8 mantissa bits, as expected) |
| MiniLM 1-cos vs HuggingFace | **1.161e-03 PASS**, = the simulated prediction |
| bge-large 1-cos | 4.436e-03 (still FAIL — see §7) |
| `--embed` pipeline 1 vs 4 | **bit-identical** |
| xclbin identity across 16 streams | 66–78 differing bytes |
| **M8 MTEB gate, MiniLM** | **mean −0.04, worst −0.14 — PASS** |
| bf16-C container + int8 design (and the reverse) | refuses on `b_layout_hash` |

### The MTEB gate, which is the authority [`0035`](../0035-m8-mteb-gate/TASK.md) established

`run_mteb.py --model all-MiniLM-L6-v2.int8 --cpu-model all-MiniLM-L6-v2
--artifacts artifacts_int8c_mini`, both sides one session, five tasks at seq 64:

| task | CPU (fp32) | NPU (int8, bf16 C) | delta |
|---|---:|---:|---:|
| STSBenchmark | 82.03 | 81.90 | −0.14 |
| SICK-R | 77.58 | 77.56 | −0.02 |
| STS12 | 72.37 | 72.29 | −0.07 |
| Banking77Classification | 80.05 | 80.00 | −0.05 |
| TwentyNewsgroupsClustering | 45.81 | **45.89** | **+0.08** |
| **mean** | | | **−0.04** |

Against [`0078`](../0078-m13-int8-accuracy/TASK.md)'s int32-C int8 at mean −0.03 / worst
−0.13: **indistinguishable.** Narrowing C costs nothing on the measure that decides, which is
what 1-cos already said and what makes the flag safe to prefer.

---

## 6. What it buys

Four production dispatches, M=8192, 8 columns, both arms `rtp=True` so the only difference is
C's transport width:

| shape | traffic i32 → bf16 | µs i32 | µs bf16 |
|---|---|---:|---:|
| qkv | 61.3 → 42.5 MB | 2611 | 1726 |
| attn_out | 20.4 → 14.2 MB | 1738 | 1244 |
| ffn_up | 81.8 → 56.6 MB | 3445 | 2655 |
| ffn_down | 44.0 → 37.7 MB | 2044 | 1757 |
| **sum** | 207.5 → 151.0 MB (−27%) | **9839** | **7383 = 1.333×** |

End to end, and against the bf16 baselines 0079 recorded:

| model | bf16 | int8 (i32 C) | **int8 (bf16 C)** | vs bf16 |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2, 1 lane | 689.9 | 996.6 | **1081.7** | 1.57× |
| all-MiniLM-L6-v2, 4 lanes | 985.3 | 1104.2 | **1204.7** | **1.22×** |
| bge-large-en-v1.5, 4 lanes | 60.1 | 86.3 | **93.1** | **1.55×** |

**bge-large gains less from the narrowing (+7.9%) than MiniLM (+9.1%), and the traffic model
says why**: bge-large runs `tile_n = 32`, so `N/(n·cols)` re-streams A **16×** where MiniLM's
48 re-streams it 4×. C is only ~1/3 of bge-large's traffic against MiniLM's 61%, so there is
less of it to remove. A model's tile geometry now decides which byte lever pays on it.

---

## 7. What this does NOT fix, stated plainly

* **bge-large int8 still fails the 1-cos gate.** 4.436e-03 against 2e-03. Narrowing C is
  orthogonal to that — 0079 swept α on hardware and the best is 3.127e-03 at α = 0.5, and
  **the container currently on disk is the α = 0.3 one** (its 4.475e-03 matches that row
  exactly). Repacking at α = 0.5 and running MTEB is the measurement that would settle
  bge-large, and neither has been done.
* **The lanes still barely overlap.** Single-lane int8 is 1081.7 against 4 lanes' 1204.7 —
  lanes buy 1.11× here against bf16's 1.43×, because a smaller NPU share leaves less to hide
  host work behind. The host is the wall on MiniLM now.
* **Fusing quantisation into LayerNorm is still unbuilt** (0079 §3). It remains the right
  shape — LN already reduces per row, so a third per-row statistic is nearly free and LN
  could emit fp32 for the residual and int8 for the GEMM in one pass — but at ~3–4 ms of a
  ~118 ms encode it is now smaller than what was just taken.
* **`add` and int8 still do not compose** (0079 §5), unchanged.

---

## 8. Levers this reopens

Because bytes stopped being free on the int8 datapath, the following are no longer closed and
should be re-priced rather than re-refuted:

1. **B-reuse.** 0010 priced it at 1.26–1.68×; 0048 retired it on "bytes are free"; 0046 showed
   it is blocked by DMA *channels*, not capacity. B is 23–43% of int8 traffic. The blocker is
   still real, so this is a channel problem, not a value problem.
2. **The cascade milestone** ([`0047`](../0047-m9-cascade-channel-probe/TASK.md)), scoped out
   for B-reuse, on the same grounds.
3. **`tile_n` as a traffic parameter.** A re-streaming goes as `N/(n·cols)`, and §6 shows that
   term dominating bge-large. Larger `tile_n` cuts A traffic directly, and int8's L1 headroom
   is exactly what makes a bigger tile legal — the direction k=128 failed in, tried on the
   axis the traffic model actually names.

Recorded in [T20](../../research/OPEN-THREADS.md) and [T1](../../research/OPEN-THREADS.md).

---

## 9. Commands

```powershell
# the discriminating pair, traced, both datapaths (scratch scripts, §1a/§1c)
. C:\dev\mlir-aie\iron_env.ps1

# the kernel, validated against exact int32 -> RNE bf16
#   -> rel_fro 0.000e+00, bit-exact

# design sets
python tools\export_gemm_rtp.py --int8 --c-bf16 --out runtime\artifacts_int8c_mini `
    --hidden 384 --intermediate 1536 --cols 8 --batches 4,16,32,128
python tools\export_gemm_rtp.py --int8 --c-bf16 --out runtime\artifacts_int8c_large `
    --hidden 1024 --intermediate 4096 --cols 8 --batches 128 -n 32

# accuracy + throughput
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2.int8 `
    --artifacts artifacts_int8c_mini --threads 24
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2.int8 `
    --artifacts artifacts_int8c_mini --threads 24 --pipeline 4 --bench 3

# price a narrowed-C design WITHOUT building one
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2.int8 `
    --artifacts artifacts_int8_mini --threads 24 --sim-c-bf16
```
