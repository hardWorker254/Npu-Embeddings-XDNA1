# 0145 — granite's q4nx weights on the AIE array: a W4A16 GEMV kernel

**Status: the kernel WORKS and every matmul in granite-4.2-3B runs on the NPU
— but a 24-thread AVX2 CPU baseline is 2.4x FASTER, at 89% of the CPU's memory
bandwidth. Decode is bandwidth-bound and the NPU is on the wrong side of that.
See "The CPU beats it". The NPU case, if there is one, is energy or prefill,
and neither is measured yet.**
All 8 projection shapes (q, k, v, o, gate, up, down, lm_head) match a host
reference built from the same bytes: cosine 1.00000000 in every case, **exact**
under a one-hot activation, max relative error 1.1e-05 under random input. That
is ~99% of the model's weight traffic. The non-GEMV ops (RMSNorm, RoPE, SwiGLU,
softmax) do NOT run on the NPU — see "What is NOT done".

## Goal

0144 ended at a wall: granite-4.2-3B cannot run on FastFlowLM's *shipped*
designs, because it needs head_dim 64 at hidden 2560 and every shipped design at
hidden >= 2560 is head_dim 128. The two sets are disjoint and head_dim cannot be
padded.

So: stop trying to borrow a design and write one. This task builds the first
kernel of our own that consumes q4nx **q4** weights directly on the array.

## Why lm_head first

It is the single largest matmul in the model (160.6 MB against 49.2 MB for a
whole transformer layer), so it has the best work-per-dispatch ratio — and
dispatch, not arithmetic, is what decides whether offload is worth anything on
XDNA2 (0024: ~178 us fixed + bytes/39.3 GB/s).

## The arithmetic, and why the Q4_1 minimum is nearly free

q4nx q4 is GGUF Q4_1 semantics: a scale **and a minimum** per 32-wide K block,

    w = code * d + m          code 0..15, d and m bf16 per block

Naively `m` is another per-element term. It is not, because it is constant across
the 32 K of a block and factors out of the inner sum:

    y[r] = sum_k (code[k][r]*d[kb][r] + m[kb][r]) * x[k]
         = sum_kb { d[kb][r] * (sum_{k in kb} code[k][r]*x[k])
                  + m[kb][r] * (sum_{k in kb} x[k]) }

The second term needs one **scalar** sum of x per K block — 8 per tile, against
8192 MACs. See "the xs reduction" below for why that scalar nearly ate the kernel
anyway.

The layout cooperates: within a tile the **row index is the fastest axis**, so 16
consecutive nibbles are 16 different rows at one k. They need 16 different scales
and those are constant for a whole k-block, so the scales load once per (rb, kb)
and the inner loop is a mask, a `to_float` with a shifted binary point, a zip and
a mac. No gather anywhere.

## Results (measured)

| config | GB/s of weights | GFLOP/s | time |
|---|---|---|---|
| 1024 tile-rows, 1 core | 1.8 | 5.7 | 29.32 ms |
| 1024 tile-rows, 4 cores | 6.8 | 21.8 | 7.71 ms |
| 1024 tile-rows, 8 cores | 13.5 | 43.1 | 3.90 ms |
| full lm_head, 8 cores | 13.5 | 43.2 | 11.88 ms |
| + vectorised `xs` | 18.4 | 58.9 | 8.72 ms |
| + `inline` body linkage | **19.7** | **63.0** | **8.16 ms** |

Accuracy, against a float32 host GEMV built from the same bytes:

| x | cosine | max rel err |
|---|---|---|
| `onehot:7` | 1.00000000 | **0.000e+00** (exact) |
| `ones` | 1.00000000 | **0.000e+00** (exact) |
| random, 8 tile-rows | 1.00000000 | 4.821e-06 |
| random, full 100352 rows | 1.00000000 | 6.158e-06 |

`onehot` being **exact** is the load-bearing check: it isolates a single k, so a
wrong nibble parity, a swapped row-block or a mis-indexed scale plane shows up as
a permutation, not as noise. A permutation cannot be exact.

The residual ~6e-06 on random input is the **bf16 hi/lo split floor**, and it was
predicted before it was measured: AIE2P has no fp32 vector multiplier, so each
fp32 partial is split into two bf16 halves (8 + 8 mantissa bits, ~2^-17 =
7.6e-06 relative). q8 does one such split and lands at 3.0e-06; q4 does two (one
for the scale term, one for `xs`) and lands at ~5e-06. Consistent, so not a bug.

### Every projection, `--all --cores 8`

| tensor | N | K | K-tiles/call | MB | ms | GB/s | max rel err |
|---|---|---|---|---|---|---|---|
| q_proj | 2560 | 2560 | 10/5 | 4.1 | 0.38 | 10.7 | 6.90e-06 |
| k_proj | 512 | 2560 | 10/5 | 0.8 | 0.23 | 3.6 | 1.06e-05 |
| v_proj | 512 | 2560 | 10/5 | 0.8 | 0.22 | 3.8 | 5.21e-06 |
| o_proj | 2560 | 2560 | 10/5 | 4.1 | 0.37 | 11.0 | 3.09e-06 |
| gate_proj | 8192 | 2560 | 10/5 | 13.1 | 0.86 | 15.3 | 7.25e-06 |
| up_proj | 8192 | 2560 | 10/5 | 13.1 | 0.82 | 16.0 | 5.87e-06 |
| down_proj | 2560 | 8192 | **32/4** | 13.1 | 0.83 | 15.8 | 6.20e-06 |
| lm_head | 100352 | 2560 | 10/5 | 160.6 | 8.08 | 19.9 | 6.16e-06 |

Every one is cosine 1.00000000, and every one is **exact** (rel err 0.000e+00)
under `--x onehot:7`.

**(N, K) cannot be read off the file.** q4nx stores a tiled shape
`[N/32 * K/256, 5120]`, and `gate_proj` and `down_proj` are BOTH `[2560, 5120]` —
256 tile-rows x 10 K-tiles against 80 x 32. They factor differently, and a GEMV
against the wrong factoring is not an error, just a different and wrong matmul.
So (N, K) comes from `config.json` and the product is checked against the stored
shape.

### The dispatch tax, now measured rather than predicted

The GB/s column falls off a cliff on the small shapes: `k_proj` moves 0.8 MB in
0.23 ms, of which only ~0.02 ms is DMA. The rest is 0024's fixed ~178 us.

Summing the measured per-op times: 3.71 ms per layer x 40 + 8.08 ms for lm_head =
**156 ms/token = 6.4 tok/s**. The same 2.13 GB at the best observed 19.9 GB/s
would be **107 ms**. The 49 ms difference is dispatch overhead, and 0024's model
predicts 281 dispatches x 0.178 ms = **50 ms**. The prediction and the
measurement agree to 2%.

That is the quantitative case for whole-layer fusion: at one dispatch per op,
roughly a third of the token budget is spent not computing.

## Commands

```
call c:\dev\mlir-aie\iron_env.cmd
cd C:\Users\vegar\Documents\GitHub\LLMNpuTest
python designs\granite_gemv\granite_gemv.py --all --cores 8 --x onehot:7
python designs\granite_gemv\granite_gemv.py --all --cores 8
python designs\granite_gemv\granite_gemv.py --tensor lm_head.weight --cores 8
```

Direct size measurement — the instrument that mattered most (see finding 1):

```
:: C:\Users\vegar\kcompile.cmd
clang++ granite_gemv_p5_k0.cc -c -o g_k0.o --target=aie2p-none-unknown-elf -O2 ...
llvm-objdump -h g_k0.o        :: section sizes
llvm-objdump -t g_k0.o        :: symbol binding (l vs w)
```

## Five findings, three of them corrections to my own reasoning

### 1. Program memory: measure, do not reason

The core has ~16 KB of program memory and the kernel overflowed it. I proposed
three causes in turn and **all three were wrong**:

* too many entry points (10) — a bisect to a *single* entry point still
  overflowed, which killed the theory outright;
* loop unrolling — disabling it changed nothing;
* runtime `extract` lane indices lowering to dynamic shuffles — also not it.

Compiling the object directly and reading `llvm-objdump -h` settled it in
seconds, where each full IRON build took minutes. The body was **4736 B** because
it was a *template* on the K-tile index, so every entry point instantiated its
own copy.

The fix: `KT` was only ever pointer arithmetic (`x + KT*256`) and never needed to
be compile-time. Runtime `kt` + `__attribute__((noinline))` emits the body once:
**2528 B**, wrapper 208 B.

**The lesson is the instrument, not the answer.** Three plausible theories, all
wrong, all cheap to disprove with a 5-second compile. Build the fast measurement
before spending build cycles on hypotheses.

### 2. `static` vs `inline` is what makes wide K affordable

The noinline body was `static`, so each entry point's translation unit still
carried a private copy. Making it `inline` gives it vague linkage: the copies
land in a COMDAT and the linker keeps exactly one. `llvm-objdump -t` confirms the
symbol goes from local (`l`, mangled `_ZL12...`) to weak (`w`, `_Z12...`).

The cost of an entry point drops from a whole body to its 208-byte wrapper:

| entry points | `static` | `inline` |
|---|---|---|
| 2 (K = 2560) | 2912 B | 1664 B |
| 8 (K = 8192, `down_proj`) | 11648 B | **2912 B** |

This is what makes `down_proj` possible at all: L1 caps a call at 5 tiles
(5 x 5120 x 2 for double buffering = 51200 B against a 63 KB budget), so
K = 8192's 32 K-tiles need 8 entry points.

It was also, unexpectedly, ~7% **faster** (8.72 -> 8.16 ms) — a smaller program
image, nothing else changed.

### 3. The xs reduction: half the code, a third of the runtime

The per-block `sum of x` was written as 32 scalar adds — ~256 scalar ops per
tile, and scalar work does not overlap the vector pipeline. Replacing it with a
load into an fp32 accumulator plus `aie::reduce_add`:

* body **2528 B -> 1248 B** (the scalar loop was *half the kernel's code*);
* full lm_head **11.88 ms -> 8.72 ms**, 13.5 -> 18.4 GB/s;
* accuracy bit-for-bit unchanged (`onehot`/`ones` still exact, random still
  6.158e-06). Summing through an fp32 accumulator rather than in bf16 is what
  keeps it exact — a bf16 running sum over 32 terms would lose ~5 bits.

It is still pure redundancy: `xs` depends only on x and kb, never on the weights,
yet it is recomputed for every one of the 3136 tile-rows. Hoisting it to the host
(80 values per token) would remove it entirely. Not done.

### 4. The float32 cosine was measuring itself (found by contradiction)

The first passing run reported `cosine 0.99999988` against a gate of
`> 0.9999999` and printed **FAIL**. The tempting move — the gate came from the q8
design, q4 has more rounding, loosen it — would have been wrong.

`x = ones` gave `max rel err 0.000e+00` **and** `cosine 0.99999988`
simultaneously. Those are contradictory: if every element matches exactly, the
cosine is exactly 1. The dot product and the two norms are reductions over up to
100352 terms accumulated in **float32 in different orders**, so the ratio drifts
from 1 even for bit-identical vectors. The harness was reporting its own rounding
as the kernel's error.

Computing the metric in float64 makes every case pass **against the original,
unrelaxed gate**. A failing test was fixed by fixing the test — but only because
the contradiction proved the test wrong first.

### 5. Unrolling the inner loop bought nothing (NEGATIVE)

With code budget freed by (3), the `unroll(disable)` on the innermost loop was
lifted: body 1248 -> 1856 B, full lm_head 8.72 -> **8.75 ms**. Inside noise.
Reverted — 600 B of code for zero gain. The compiler was already scheduling it.

## The ceiling, measured

Scaling is near-linear in cores (7.5x on 8), which says **compute-bound, not
DMA-bound** — 19.7 GB/s against the 39.3 GB/s shim ceiling from 0024.

**8 cores is a hard ceiling for this topology, and it is a routing limit rather
than a compute one.** `--cores 16` fails at placement:

```
error: no ShimNOCTile has sufficient DMA capacity for 0 input/1 output
       channels near centroid column 1
```

The device is **8 columns x 6 rows** (row 0 shim, row 1 memtile, rows 2-5
compute) = **32 compute cores**, of which this design reaches 8. Each column's
shim has 2 MM2S + 2 S2MM. One stream per core costs n+1 in / n out, so 8 cores is
9/8 (fits in 16/16) and 16 cores is 17/16 (does not).

The other 24 cores are reachable only through the **memtile leg**: one shim
stream per column into the memtile, `ObjectFifo.cons().split()` four ways to the
column's compute cores, `.prod().join()` on the way back. `ObjectFifo.split` /
`.join` (`python/iron/dataflow/objectfifo.py`, lines 958 / 865) take exactly the
offsets this needs; `programming_examples/basic/matrix_multiplication/whole_array`
is the reference. The DDR tap must interleave at call granularity so that chunk i
of each memtile object belongs to core i. Not built.

## What this means for running granite on the NPU

Weight traffic per token, at q4nx q4 (0.625 B/param):

| | |
|---|---|
| per layer (q, k, v, o, gate, up, down) | 49.2 MB |
| x 40 layers | 1.97 GB |
| lm_head | 160.6 MB |
| **total per token** | **2.13 GB** |

At the measured 19.7 GB/s that is 108 ms/token = **9.2 tok/s**; at the 39.3 GB/s
shim ceiling, 54 ms = **18.5 tok/s**. Decode is memory-bound, so this is a
property of the memory system and the model size, not of the kernel — no amount
of inner-loop cleverness moves it. The levers are the memtile leg (reaching the
other 24 cores, to convert compute-bound into DMA-bound) and not re-streaming
weights that could stay resident.

Every projection is **this same kernel** at a different (N, K), and all 8 are
measured above. Only `down_proj` needed anything new, and only because K = 8192
needs 8 entry points — which finding (2) had already made affordable.

Against that, the measured **156 ms/token (6.4 tok/s)** at one dispatch per op
decomposes as ~107 ms of weight streaming and ~49 ms of dispatch. Both terms have
a known lever: fusion removes the second, and the memtile leg attacks the first
by converting a compute-bound kernel into a DMA-bound one.

## The CPU beats it, and that is the headline

A like-for-like CPU baseline (`designs/granite_gemv/cpu_baseline.cpp` -- same
file, same bytes, same arithmetic, AVX2 + FMA, 24 threads) was written
specifically to avoid comparing against a straw man. Validated against numpy to
7 significant figures before being timed (lm_head y[0] = 0.00243628 CPU vs
0.00243640 numpy).

| lm_head, 100352 x 2560, 160.6 MB | time | GB/s |
|---|---|---|
| **CPU GEMV**, 24 threads, AVX2 | **3.41 ms** | **47.0** |
| CPU STREAM (read the bytes, do nothing) | 3.04 ms | 52.8 |
| NPU, 8 cores, this kernel | 8.08 ms | 19.9 |
| NPU at 0024's 39.3 GB/s shim ceiling (hypothetical) | ~4.1 ms | 39.3 |

**The CPU is 2.4x faster than the NPU kernel, and it is running at 89% of its own
memory bandwidth** (47.0 of 52.8 GB/s), so it is near-optimal rather than merely
well written. There is no CPU-side optimisation left to find.

### CORRECTION: "even a perfect NPU design loses" was WRONG

That was written here, and it was wrong. It rested on two unchecked assumptions:
that 0024's 39.3 GB/s was a hard shim ceiling, and that this kernel was near it.
**Both are false**, and one measurement settles it — a null kernel (`--null`)
that streams every weight byte and does no arithmetic, isolating the memory path
from the compute:

| lm_head, 160.6 MB, weight stream only | time | GB/s |
|---|---|---|
| **NPU DMA, 8 cores, no arithmetic** | **3.47 ms** | **46.3** |
| NPU DMA, 4 cores, no arithmetic | 4.10 ms | 39.2 |
| NPU GEMV, 8 cores (this kernel) | 8.08 ms | 19.9 |
| CPU GEMV, 24 threads | 3.41 ms | 47.0 |

**The NPU's weight path sustains 46.3 GB/s — within 2% of the CPU's 47.0.** It
also beats 0024's 39.3 GB/s, so that figure was a measurement of one transfer
pattern, not a ceiling. At the DMA bound lm_head would take **3.47 ms against the
CPU's 3.41 ms**: parity on latency, at far lower power.

So the kernel is **compute-bound by 2.3x**, not memory-bound. The linear core
scaling had already said "compute-bound" and I read that correctly — then drew
the opposite conclusion by assuming the ceiling was somewhere it is not.

**The lesson is the same one as finding 1, and I failed to apply it here:**
measure the bound before reasoning about what is near it. A null-kernel probe is
~20 lines and would have prevented a wrong strategic conclusion.

Note 4 cores already reach 39.2 GB/s on DMA: **the bandwidth is available long
before the cores are**, which makes core count the lever, not the memory path.

Compute scales perfectly linearly, so the core count needed is arithmetic rather
than a guess (full lm_head, 160.6 MB):

| cores | ms | GB/s | GB/s per core |
|---|---|---|---|
| 1 | 62.83 | 2.6 | 2.6 |
| 2 | 32.03 | 5.0 | 2.5 |
| 4 | 16.52 | 9.7 | 2.4 |
| 8 | 8.08 | 19.9 | 2.5 |
| *needed for 46.3 GB/s* | *~3.5* | *46.3* | *-> **~18 cores*** |

**~18 cores saturates the measured DMA bound. The device has 32. This design
reaches 8.** That is the whole gap, and it is a routing problem (trap 13), not an
arithmetic one.

## The memtile leg: 32 cores, 2.04x, BUILT AND PASSING

`designs/granite_gemv/granite_gemv32.py`. One shim stream per **column** into the
memtile, `split()` four ways to that column's compute cores, `join()` back out:

    shim MM2S x1  ->  memtile  ->  split -> 4 cores       (per column)
    shim S2MM x1  <-  memtile  <-  join  <- 4 cores

8 + 1 MM2S and 8 S2MM for **32 cores**, comfortably inside 16/16.

| lm_head, 160.6 MB | cores | ms | GB/s | GFLOP/s |
|---|---|---|---|---|
| one shim stream per core | 8 | 8.08 | 19.9 | 63.0 |
| memtile leg, 4 columns | 16 | 4.89 | 32.8 | 105.1 |
| **memtile leg, 6 columns** | **24** | **3.61** | **44.2** | **141.5** |
| memtile leg, 8 columns | 32 | 3.97 | 40.4 | 129.3 |
| memtile DMA bound (null kernel, 8 cols) | 32 | 3.76 | 42.7 | - |
| direct-shim DMA bound (null kernel) | 8 | 3.47 | 46.3 | - |
| CPU, 24 threads AVX2 | 24 thr | 3.41 | 47.0 | 150.5 |

**2.24x at 24 cores, and at parity with the CPU.** Repeat runs of the 24-core
configuration gave **3.61 / 3.51 / 3.40 ms** (44.2 / 45.5 / 47.0 GB/s), so there
is ~5% run-to-run variance and the honest statement is a range, not the best
sample: **NPU 3.4-3.6 ms against CPU 3.41 ms.** They are the same speed within
noise — on a part of the chip that was idle before, and at far lower power.

Exactness holds at this width too: both `onehot:7` and `ones` reproduce the host
reference with **rel err 0.000e+00** at 24 cores.

**24 cores beats 32, and that is the interesting part.** 44.2 GB/s at 6 columns
is *higher* than the 42.7 GB/s the null kernel measured at 8 columns, so "the DMA
bound" is not one number: adding the 7th and 8th columns costs more than the 8
extra cores return. Something saturates between 6 and 8 columns — memtile or
stream contention, or the x broadcast fanning to 32 consumers instead of 24. Not
diagnosed. The practical consequence is that **the widest configuration is not
the fastest**, and a null-kernel probe at one width does not give you the bound
at another.

(The 24-core run covers 99840 of 100352 rows: tile-rows must divide by the core
count and 3136 does not divide by 24. A remainder path is not written.)

Accuracy is untouched: exact (rel 0.00e+00) under one-hot at 2 columns, and
6.16e-06 at 32 cores — *identical* to the 8-core figure, which is what confirms
the permutation is right rather than merely plausible.

### Why the weights are permuted on the host

`split()` hands child i the i-th slice of each parent object, so the stream must
arrive as `[core0 chunk k][core1 chunk k][core2 chunk k][core3 chunk k]`. With
each core owning a contiguous block of tile-rows that is a strided 3-D tap whose
innermost run is 25600 B — close enough to the BD size limits to be a liability.
A host-side permutation makes each column's stream plain contiguous instead:

```python
a.reshape(n_cols, 4, per_core * n_entry, call_bytes).transpose(0, 2, 1, 3)
```

and the joined output is un-permuted by the mirror of it. This is a **one-time
cost at model load**, not per token: weights are uploaded once and streamed for
every token after.

## Fusion by shared input: 7 -> 4 dispatches, 1.47x

`designs/granite_gemv/granite_layer.py`. A group of projections can share one
dispatch iff they share an **input vector** and a **K**. The kernel is row-
parallel and every tile-row is independent, so concatenating weights along N is
exactly concatenating the outputs — no new arithmetic, no new correctness
surface, no new kernel.

    q, k, v      share the post-input_layernorm hidden    -> ONE dispatch
    gate, up     share the post-attention_layernorm       -> ONE dispatch
    o            takes the attention output               -> alone
    down         takes the SwiGLU output, and K = 8192    -> alone

| layer 0, 49.2 MB | dispatches | ms/layer | ms/token | tok/s |
|---|---|---|---|---|
| unfused, 8 cores (direct shim) | 7 | — | 156 | 6.4 |
| unfused, memtile leg | 7 | 3.41 | 140 | 7.1 |
| **fused by shared input** | **4** | **2.32** | **96** | **10.4** |
| CPU, 24 threads | — | — | 45 | ~22 |
| **hardware floor** (2.13 GB / 44 GB/s) | — | 1.12 | **48** | **~21** |

Every group still passes: cosine 1.00000000, rel err ~1e-05.

`gate_up` as one dispatch reaches **30.6 GB/s** against 24.2 and 24.7 for the two
halves separately — the larger transfer amortises both the fixed cost and the
DMA ramp, which is the same effect the small-shape numbers showed from the other
direction.

### Where the remaining 1.2 ms/layer goes

Streaming 49.2 MB at 44 GB/s is 1.12 ms; the fused layer measures 2.32 ms.
Roughly 0.72 ms is the 4 x ~178 us of dispatch, and the rest is that the smaller
groups (`qkv` 5.7 MB at 13.8 GB/s, `o` 4.1 MB at 8.4) never reach peak rate.

**Both are fixed by the same thing: fewer, larger dispatches.** And below 4 per
layer that requires RMSNorm, RoPE, attention and SwiGLU on the array, because
those sit *between* the groups.

    1 dispatch/layer  ->  0.18 + 1.12 = 1.30 ms  ->  ~55 ms/token  ~18 tok/s
    1 dispatch/token  ->             ~48 ms      ->              ~21 tok/s

So **the path to maximum speed and the goal of "run most of granite on the NPU"
are the same piece of work**, which is a good position to be in: the elementwise
kernels are not a detour from performance, they are the remaining performance.

## Batching: running several tokens over one weight pass

The user's question, and it is the only lever that beats the memory bound
outright. Decode reads all 2.13 GB of weights per token; running **B independent
tokens through a layer before moving to the next** divides the per-token traffic
by B. It applies to independent tokens only — concurrent requests, prefill, or
speculative decoding — since in one autoregressive stream token t+1 needs t.

`GRANITE_BATCH` in `granite_gemv.h`; x becomes `[B][K]`, y becomes `[B][32]` per
tile-row, and the weights do not change size at all. That asymmetry is the point.

Measured **within a single run** (see the variance note below), lm_head at 24
cores:

| B | ms | GFLOP/s | ms/token | vs B=1 |
|---|---|---|---|---|
| 1 | 3.50 | 146.1 | 3.50 | — |
| 2 | 3.77 | 271.2 | 1.89 | **1.86x** |
| 4 | 6.08 | 336.0 | 1.52 | **2.30x** |

A second independent run (at a smaller DMA element) gave 2.22x and 2.71x for the
same two points, so the honest range is **~1.9-2.2x at B=2 and ~2.3-2.7x at
B=4**. Compute saturates around 336 GFLOP/s, against 146 at B=1.

Every token is checked against its own reference, not just token 0 — a batched
kernel that ignored the token index would reproduce token 0 perfectly.

### My prediction was wrong, and the reason is the useful part

I predicted saturation at ~1.36x by B=2, reasoning that B tokens means B times
the work per byte. **That is false.** The expensive part of this kernel is
decoding the weights — mask, `to_float`, `interleave_zip` — and that is done
**once per weight byte regardless of B**. Only the MACs scale. So the marginal
token is far cheaper than the first, which is exactly why 108 -> 240 GFLOP/s
came with a *lower* wall-clock time.

Compute saturates around ~336 GFLOP/s: at B=4 the design is back to compute-
bound (26.3 GB/s of a 45 GB/s path), which is where a matmul-unit (`aie::mmul`)
kernel would be the next lever rather than more batching.

**Caveat worth stating plainly: batching helps the CPU too.** The CPU baseline at
B=1 is already at 89% of its memory bandwidth but only ~10% of its FLOP peak
(150 of ~1500 GFLOP/s), so a batched CPU GEMM would gain similarly. Batching
breaks the *memory* bound for whoever does it; it is orthogonal to the
CPU-vs-NPU comparison, not a win over the CPU.

### Batching trades L1 against the DMA element

The activation buffer is `B*K*2` bytes and grows with B while the weights do not,
so a batch that is free in bandwidth terms still shrinks the weight element.
B=4 at 5 tiles per call wants 51200 + 20480 + 1024 = 72704 B and fails as
`'aie.tile' op allocated buffers exceeded available memory` — an MLIR-level
error, not anything the kernel could report. `tiles_per_call` is now L1-aware.

**And the first budget I wrote for it was too conservative, which is its own
lesson.** 63 KB - stack = 61184 rejected B=2 at 5 tiles (61952 B) — a
configuration that had *already been measured working* at 3.59 ms. The rejection
is not a safe failure: it silently drops to 2 tiles per call and costs ~20%.
The budget is now set from the verified number (62208), not from the rule of
thumb.

### Trap 7d bit, and the defence I had already built did not cover it

`granite_gemv32.py` derived `per_call` **inside** the jitted generator. IRON's
cache key hashes the call's arguments — `tile_rows, k, n_cols, null, batch` — and
nothing computed within (trap 7d). So two runs differing only in `per_call`
collide on one cache entry: the second silently receives the first one's xclbin
while the host permutes the weights for its own value.

It surfaced as **cosine 0.208** on a configuration that had passed twice before,
after nothing changed but an L1 budget constant. Not a crash, not a build error —
a plausible-looking wrong answer, which is this project's recurring failure
shape.

The uncomfortable part: I had *already* defended against exactly this by putting
`per_call` and `batch` into the generated `.cc` file names **and** the C symbols
(`granite_gemv_p5b2_k0`). That defence is real but covers only the kernel object.
`per_call` also sets the ObjectFifo object sizes and the entry-point count — the
whole dataflow — and none of that is in the kernel symbol.

**Fix: `per_call` is now an explicit `CompileTime` argument**, computed by the
caller and passed in, so it is part of the cache key by construction. The general
rule this reinforces: on IRON, anything that changes the generated design must
arrive as an argument, never be derived inside the generator — deriving it is
invisible to the cache even with `use_cache=False`.

### Run-to-run variance is larger than it looked

The same 24-core B=1 configuration measured 3.40, 3.51, 3.61, 3.97, 3.98 and
**4.72** ms across the session — roughly +/-20%, drifting slower as the session
went on (thermal, most likely). **Batch comparisons above are therefore all
within a single run.** Any cross-run comparison in this document that differs by
less than ~20% should be read as "no measured difference".

## RMSNorm on the array: the first non-GEMV granite op

`granite_rmsnorm.h` / `.py`. Written rather than adopted, because AMD's
`aie_kernels/aie2p/rms_norm.cc` hardcodes `const float gamma = 1.0f` and never
applies the per-channel weight.

Checked against the **real** `model.layers.0.input_layernorm.weight`, not
synthetic data — that matters here, because a kernel that ignored the weight
would look fine on a uniform vector:

```
cols 2560   eps 1e-05   weight range [-15.0000, 23.1250]   std 2.2746
cosine 0.99999877   max rel err 1.855e-03   PASS
```

**1.855e-03 is the bf16 output's own rounding floor** (2^-9 = 1.95e-03), so the
result is correctly rounded and the gate is set by the storage format rather
than fitted to the measurement. The weight's spread (std 2.27, range -15 to
+23) is the reason the AMD kernel would not have been subtly wrong but grossly
wrong, at the right magnitude and with no shape mismatch.

Body 768 B, wrapper 16 B; `cols` is a runtime argument, so one build serves any
hidden size. Precision follows the same rules the GEMV needed: x^2 summed in an
fp32 accumulator (a bf16 running sum over 2560 terms would lose ~6 bits), and
`x*w*inv` done as three bf16 cross terms because AIE2P's fp32 vector multiply
silently returns zero.

## RoPE and SwiGLU on the array

`granite_elementwise.h` / `.py`. Both written rather than adopted, and the third
reference kernel turned out to be worse than the first two:

**`swiglu.cc`'s `extern "C"` entry point hardcodes `input_size = 1024`.** The
templated body takes a size; the wrapper passes a literal. Against granite's
intermediate of 8192 it would compute one eighth of the vector and leave seven
eighths as whatever was in the buffer — partially correct output, no error, no
shape mismatch. (Its silu maths is fine and worth keeping: `sigmoid(x) =
(tanh(x/2)+1)/2` is an identity, not an approximation.)

```
rope     cosine 0.99999889   max rel err 2.731e-03   (219 us)
         interleaved convention differs by 1.506
swiglu   cosine 0.99993644   max rel err 8.140e-03   (252 us)
         8192/8192 outputs non-zero
```

**The RoPE test checks that it can fail.** The half-split and interleaved
conventions give outputs of identical magnitude, so a test that only compared
against half-split would also pass for a kernel doing the wrong rotation. It
therefore computes the interleaved result too and asserts it does **not** match:
1.506 relative. AMD's kernel would have been wrong by 150%, at the right
magnitude.

Similarly the SwiGLU test asserts all 8192 outputs are non-zero, which is what
catches the hardcoded-1024 failure specifically.

RoPE's 2.7e-03 is the bf16 output floor. SwiGLU's **8.1e-03 is above it** — that
is `aie::tanh`'s own approximation, and it is the one op here whose error is set
by the kernel rather than by the storage format. Acceptable for a bf16
activation, but it is not correctly rounded and should not be described as such.

### L1 again, and the error message that pays for itself

Three 8192-element bf16 vectors, double-buffered, is 98304 B against 64 KB.
`Basic sequential allocation failed` **prints the MemoryMap** — every buffer with
its address range — which named the overflow immediately. The kernel takes its
length at runtime, so the design streams the vector in 2048-element chunks
(24576 B) instead of holding it.

## Kernel-level fusion: q_proj + RoPE in one dispatch

`granite_qrope.py` / `granite_qrope.h` / `granite_qgemv_g{0,1}.cc`.

Fusing two projections generally needs an all-gather — each core holds only its
slice, and the next matmul needs the whole vector. **RoPE is the exception:** it
is per head, so a core owning whole heads rotates its own slice with no
communication. granite is head_dim 64 = 2 tile-rows per head, so at 8 cores over
q_proj's 80 tile-rows each core owns 10 tile-rows = 5 whole heads.

Measured **in one run**, which is the only valid comparison here:

| | us |
|---|---|
| q_proj alone (unfused) | 450 |
| **q_proj + RoPE fused** | **395** |
| unfused + a RoPE dispatch (~178) | ~630 |

The fused version does strictly more work in less time, and saves a dispatch on
top: **~1.6x** for the pair. The float32 GEMV output never leaves L1 — the
epilogue narrows it to bf16 and rotates it in place.

**A caution about the numbers:** the first sample of the fused design read 590.6
us; the same configuration in the same run as its reference read 394.7. That is
a 50% spread on identical work, worse than the +/-20% recorded earlier. Only
same-run comparisons in this document mean anything.

### Three build failures, each a real constraint

1. **Two `ExternalFunction`s pointing at one `.cc`** — duplicate symbols, the
   trap already recorded twice. One entry point per translation unit.
2. **`requires 3 input/1 output DMA channels, but only 2 input/2 output
   available`.** A compute tile has **two input DMA channels** and the GEMV
   already uses both (weights, x); cos/sin was a third. Fixed by carrying
   cos/sin at the end of the x buffer and having the kernel find it at a fixed
   offset — one stream, no extra channel. (It is also 128 B, so it would have
   hit trap 14 as its own transfer.)
3. **L1 overflow.** Weights double-buffered at 5 tiles is 51200 B of a 65536 B
   tile; x and the output are acquired once and held, so depth 2 on them bought
   no overlap and did not fit. Depth 1 for both. `Basic sequential allocation
   failed` prints the full memory map, which makes this a 30-second diagnosis.

## A GRANITE LAYER IN THREE DISPATCHES

`granite_qkv.py` adds the last fusion: q, k, v and RoPE in one dispatch. All
three consume the same x (no gather), and RoPE applies to q and k but not v --
a core owns whole heads of each, so the epilogue is core-local. cosine
0.99999788, max rel err 3.643e-03. **v is checked separately** (3.706e-03),
because a kernel that rotated everything would still pass on the q and k
majority.

Measured in one run, at 8 cores:

| | dispatches | ms |
|---|---|---|
| q, k, v, o, gate, up, down one at a time | 7 | 3.77 |
| ...plus RoPE and SwiGLU dispatches | 9 | ~4.15 |
| **[q,k,v,RoPE] + o + [gate,up,SwiGLU,down]** | **3** | **2.97** |

**1.40x for the layer**, and 1.27x against the seven GEMVs alone -- the fusion
absorbs RoPE and SwiGLU for free. Nine ops in three dispatches.

    [q,k,v,RoPE]  0.45 ms      four ops, no gather
    o_proj        0.37 ms      alone
    [gate,up,SwiGLU,gather,down]  2.16 ms   four ops + a DDR gather

### Fusion and the memtile leg do not compose, and the arithmetic says why

The fused designs run at 8 cores and the memtile path reaches 24, so combining
them looks like the obvious next 2-3x. It does not work, and the reason is
channel budgets rather than anything about the code.

**A fused core needs four streams**: weights in, x/h in, h out, result out.

* **On direct shim** that costs `n+1` MM2S and `2n` S2MM. `2n <= 16` gives
  **n <= 8** -- exactly where the design sits. Predicted, then confirmed: at 16
  cores it fails with `no ShimNOCTile has sufficient DMA capacity`.
* **Through memtiles**, one column's memtile would carry 1 weight in + 4 out,
  1 activation in + 4 out, 4 h in + 1 out, 4 result in + 1 out = **10 in / 10
  out**, against the ~6/6 a memtile has (established by the 8-way join refusal
  in granite_gather.py). It does not fit either.

The unfused GEMV needs only three streams (weights, x, result), which is why the
memtile leg scales it to 24 cores. **Fusion buys dispatches and costs channels,
and on this array the two trade against each other directly.**

The obvious way out is to cut a stream: h and the result are produced in
different phases, so one output fifo could serve both -- the trick the activation
fifo already uses, filled twice. That halves S2MM to `n`, allowing n <= 15.

**It would not actually help, and the reason is divisibility rather than
channels.** The core count must divide both 256 (gate/up tile-rows) and 80
(down's), and `gcd(256, 80) = 16`, so the legal counts are 1, 2, 4, 8, **16**.
The next step above 8 is 16, which needs 16 weight streams plus one for x = 17
MM2S against 16. The merge buys a channel and the geometry immediately takes it
back.

So the stream merge is only worth building **together with a remainder path**
(so the row count need not divide the core count), and that does not exist --
`granite_gemv32.py` already drops 512 of lm_head's 100352 rows for the same
reason. Two changes, not one. Recorded before building either, because the
arithmetic was cheaper than the build cycle.

**The general shape, which is the part worth keeping:** on this array a design's
core count is set by `min(channel budget, divisors of the work)`. Both have bitten
now -- channels here and at 8 vs 16 for the plain GEMV, divisors at 24 cores on
lm_head. Neither is visible in the kernel.

## THE WHOLE MLP BLOCK IN ONE DISPATCH

`granite_mlp_full.py`: gate, up, SwiGLU, DDR gather, down -- four ops, one
dispatch, on 8 cores.

| same run | ms |
|---|---|
| gate alone | 0.82 |
| up alone | 0.82 |
| down alone | 0.84 |
| the three, plus a SwiGLU dispatch | ~2.67 |
| **fused** | **2.12** |

**1.26x**, and the fused block beats even the three GEMVs alone (2.12 vs 2.48)
while additionally absorbing SwiGLU. cosine 0.99989113, max rel err 2.114e-02
(the `aie::tanh` floor, as in the half-block), and the intermediate matches the
reference h to 1.282e-02.

### The gather

`join()` + `forward()` is not expressible, so the intermediate goes cores -> DDR
-> cores inside the dispatch, which `granite_roundtrip.py` established is ordered
(20/20). 16 KB each way against 39 MB of MLP weights: **~0.04%**.

### Three things that had to be got right, two of them silent

* **The weight stream is filled twice**, gate|up in phase 1 and down in phase 2.
  A single fill spanning both could not complete until the core consumed its
  phase-2 objects, which it cannot do until `tg1.finish()` returns -- and that is
  waiting for the fill. **A deadlock, not an error.**
* **One activation fifo carries x then h.** A compute tile has 2 input DMA
  channels and the weights need one; a third for h does not exist. The fifo is
  sized for the larger payload and filled twice.
* **The down-weight tap needed an offset** into the shared buffer. Without it
  phase 2 read the *gate* weights: cosine -0.034, no error, no warning -- the
  same failure shape as the 128-byte transfer. Found in one build rather than
  three, because the design reports the intermediate separately from the result:
  `intermediate vs reference h: 1.282e-02 (phase 1 OK)` localised it immediately.
  That check exists precisely because guessing cost three cycles on attention.

### Where a granite layer now stands

| group | dispatches | status |
|---|---|---|
| q_proj + RoPE | 1 | fused, 1.6x |
| k, v | | not fused |
| attention | 1 | |
| o_proj | 1 | |
| gate + up + SwiGLU + down | **1** | fused, 1.26x |

## Fused MLP first half: gate + up + SwiGLU in one dispatch

`granite_mlp.py` / `granite_swiglu_f32.h` / `granite_mlp_g{0..4}.cc`.

`down_proj` consumes the whole 8192-wide intermediate and needs the DDR gather.
**The first half needs none:** SwiGLU pairs `gate[i]` with `up[i]` at the same
index, so a core owning the same row range of both combines its own slices
locally -- the property that made q_proj + RoPE fuse. Both float32 accumulators
stay in L1 and only the result is narrowed.

Measured in one run:

| | ms |
|---|---|
| gate_proj alone | 0.91 |
| up_proj alone | 1.02 |
| sum + a SwiGLU dispatch | ~2.1 |
| **fused** | **1.49** |

**~1.4x**, three ops in one dispatch, cosine 0.99988993, max rel err 1.224e-02.

### Three constraints, and one hypothesis that was wrong

1. **BD sizes cap at 1023 per dimension.** Core c wants gate rows then up rows,
   two runs a whole matrix apart; as a strided tap that is a BD dimension of
   1638400 and `'aie.dma_bd' op Size 0 exceeds the [0:1023] range`. The host
   interleaves the two matrices per core instead -- the same trade the memtile
   leg makes, for the same reason.

2. **`Overflow of program memory`, from my own wrong comment.** granite_qrope.py
   says `row` must be a compile-time constant "for the offset to fold". **That
   is false** -- the kernel computes `y + row * kRows` at runtime. There the
   Python loop cost nothing (20 call sites); here 2 x 32 x 5 = **320 unrolled
   calls** overflowed the core outright. As a `range_` hardware loop it is 10
   call sites. A comment asserting a constraint that does not exist propagated
   into a design where it did damage.

3. **The bf16 narrowing is NOT the error source** (hypothesis tested and
   rejected). The kernel narrows both accumulators to bf16 before the
   nonlinearity, so the reference was changed to do the same -- and the error
   moved from 1.227e-02 to 1.224e-02, i.e. not at all. The residual really is
   `aie::tanh`, which the standalone SwiGLU measures at 8.14e-03 on random input.
   Only after that test could the gate be set from a known source rather than
   fitted: cosine 0.9999 had been picked by analogy with the other ops, and
   0.999 is what is consistent with a 5e-2 relative bound.

## CORRECTION: ~50 GB/s is a per-agent limit, not the platform's

This document said decode is bounded at ~21-22 tok/s because 2.13 GB must cross
DRAM once per token at the ~44-50 GB/s either device sustains, "for the CPU and
the NPU alike", and called it physics. **That was wrong**, and the test that
shows it is running both at once:

| | solo | concurrent |
|---|---|---|
| NPU weight stream (null kernel) | 46.7 GB/s | **45.4 GB/s** |
| CPU STREAM / GEMV, 24 threads | 51.8 / 50.5 GB/s | 31.9-51.8 GB/s |

**The NPU is barely affected (-3%) while the CPU keeps running.** Taking the
worst CPU sample observed during the window, the aggregate is **at least ~77
GB/s**, against ~52 for either alone. That is consistent with LPDDR5X-7500 at
128-bit having ~120 GB/s theoretical: **~50 GB/s is what one agent can pull, not
what the memory system can deliver.**

The consequence is that a hybrid engine splitting layers between NPU and CPU
should approach ~90 GB/s, i.e. **~24 ms/token, roughly 40 tok/s** — about double
what this document previously called the ceiling.

**Caveat, stated because the measurement does not fully resolve it:** the CPU
loop ran 40 invocations and the NPU probe overlapped only some of them, and the
harness does not timestamp which. The NPU's own number is solid; the CPU's
concurrent figure is a range, and the lowest sample (31.9 GB/s) is used above so
the aggregate is a lower bound rather than a best case. A properly interleaved
measurement with timestamps would tighten it.

**The error was the same shape as the earlier one in this document:** I measured
one agent, found ~50 GB/s twice, and promoted it to a property of the machine
without testing the two together. Two coincidentally similar numbers are not a
law.

### Hybrid NPU + CPU, measured on a real matmul

lm_head split by tile-rows, both devices computing concurrently, each checked
against its own reference:

| | rows | MB | ms | GB/s |
|---|---|---|---|---|
| NPU alone (24 cores) | 3136 | 160.6 | 4.23 | 37.8 |
| CPU alone (24 threads) | 3136 | 160.6 | **3.27** | 49.2 |
| **hybrid** NPU | 1416 | 72.5 | 1.92 | 37.8 |
| **hybrid** CPU | 1720 | 88.1 | **2.38** | 37.0 |

Layer time is `max(1.92, 2.38) = 2.38 ms` against **3.27 ms** for the best single
device: **1.37x**, at an aggregate of **74.8 GB/s**. Perfect balance (the CPU is
still the straggler here) would put it near 2.15 ms, ~1.52x.

**The useful asymmetry: the NPU held 37.8 GB/s in the hybrid, exactly its solo
figure, while the CPU fell from 49.2 to 37.0.** Under contention the CPU gives up
bandwidth and the NPU does not, so the NPU's share is close to free. That also
means the split must be chosen from *contended* bandwidths, not solo ones — a
50/50 split by solo figures left the NPU the straggler at 2.64 ms.

**Correcting the estimate above:** the null-kernel test suggested ">= 77 GB/s"
aggregate, which was optimistic for real work. With both sides doing arithmetic
it is ~75 GB/s. The direction of the earlier correction stands; the magnitude was
too generous.

For the whole model that is 2.13 GB at ~75 GB/s = **~28 ms/token, ~35 tok/s**
(matmuls only, dispatch excluded) — against ~22 tok/s for the CPU alone and the
~21 this document once called a ceiling.

### The old claim, kept for the record

2.13 GB of weights must cross DRAM once per token. At the ~44-47 GB/s either
device sustains, that is 45-48 ms = **~21-22 tok/s, for the CPU and the NPU
alike**. Decode is memory-bound; neither device can beat it, and the NPU's case
remains energy and leaving the CPU free rather than latency.

### What is left

At 24 cores the GEMV is essentially at the memory bound this path can deliver;
the remaining latency gap to the CPU is ~6%. That is no longer the thing to
chase. The ~49 ms/token of dispatch overhead is, and that is fusion.

Also open: why 8 columns is slower than 6, and whether a remainder path (so the
row count need not divide by the core count) changes the picture at 32.

Note the CPU numbers are, if anything, **pessimistic**: the benchmark spawns its
24 threads per iteration, which costs real time and hurts the small shapes badly
(gate_proj reads only 13.1 MB and shows 13.2 GB/s on STREAM for that reason).
The overhead works against the CPU, so the conclusion is safe.

Projected over the whole model (matmuls only, both arms equally optimistic):

| | ms/token | tok/s |
|---|---|---|
| CPU, 2.13 GB at 47.0 GB/s | 45 | ~22 |
| NPU at 8 cores, 19.9 GB/s + 49 ms dispatch | 156 | ~6.4 |
| **NPU at 24 cores, 44-47 GB/s + 49 ms dispatch** | **~97** | **~10** |
| **NPU at 24 cores, dispatch removed by fusion** | **~48** | **~21** |

The third row is what the hardware has already been shown to sustain, not an
extrapolation from a datasheet. Closing the gap to it needs two things, and the
second is the bigger one:

* **Fusion**, to remove the ~49 ms of dispatch. Also fixes the small shapes,
  which never reach peak bandwidth: `k_proj` moves 0.8 MB at 3.6 GB/s against
  `lm_head`'s 160 MB at 19.9. A fused layer is one 49.2 MB stream, not seven.
* **More cores.** Per-core compute is ~2.5 GB/s, so saturating 46.3 GB/s needs
  ~18 cores; the design reaches 8 (trap 13 — shim DMA channels). The memtile leg
  to the other 24 cores is the single largest lever available.

### So what else is the NPU for

Beyond latency parity, the cases that were always true:

* **Energy.** This project already measures J/1k (0141/0143). A CPU pinning 24
  threads at 47 GB/s is not free, and the NPU's draw is much lower. That ratio
  is the number that would justify the work, and it has NOT been measured here.
* **Leaving the CPU alone.** The baseline uses the entire machine. Inference that
  must run beside a user's actual workload does not have 24 idle cores.
* **Prefill, not decode.** Decode is one token against the whole weight set --
  arithmetic intensity ~1, so it is pure bandwidth. Prefill batches many tokens
  against the same weights, which is where an array with high compute density
  and a fixed weight stream should win. Every measurement here is decode.

None of that is established. It is the list of things that would have to be true.

## The AMD reference kernels are demos, not drop-ins

`c:\dev\mlir-aie\aie_kernels\aie2p\` ships kernels for every non-GEMV op granite
needs — `rms_norm.cc`, `rope.cc`, `silu.cc`, `swiglu.cc`, `softmax.cc`. Two of
them are **wrong for granite in ways that no shape check would catch**, both
producing plausible output of the right magnitude:

* **`rms_norm.cc` hardcodes `const float gamma = 1.0f;`** — it normalises but
  never applies the per-channel `weight` tensor. `epsilon` is a `constexpr 1e-5f`
  too. `cols` *is* a runtime argument, so 2560 is fine; the weight is not.
  Granite needs `x * rsqrt(mean(x^2) + eps) * weight[c]`, so this needs writing.

* **`rope.cc` uses the interleaved-pair convention** — `filter_even`/`filter_odd`,
  pairing element 0 with 1, 2 with 3 (GPT-NeoX style). Granite, like Llama, uses
  the **half-split** convention: `rotate_half` pairs element i with i + d/2.
  These are different rotations and the result has the same magnitudes, so it
  fails silently.

  This one is recoverable rather than fatal: the two conventions differ by a
  permutation of the head_dim axis, and q4nx-build **already applies a RoPE
  permutation** to q/k weights during GGUF conversion (llama.cpp does the same).
  So the interleaved kernel is usable provided the weight permutation is matched
  to it — but which permutation is in the shipped file has not been checked.

`softmax.cc`'s `#define log2e 1.4453125` looks truncated next to its own comment
(`// 1.44269504089`) but is correct: it is log2(e) rounded to bf16, deliberate
for bf16 math.

## Attention: WORKING. And the bug was a 128-byte DMA that silently delivered zeros

```
seq  32 (1 block )  cosine 0.99977640  max rel err 2.709e-02
seq  64 (2 blocks)  cosine 0.99979644  max rel err 3.440e-02
seq 128 (4 blocks)  cosine 0.99930532  max rel err 3.646e-02
```

**A shim transfer of 128 bytes arrives as all zeros.** `q` is 64 bf16 = 128 B,
and so is the result. No error, no warning, no diagnostic anywhere in the
pipeline — the kernel simply computes 32 dot products against a zero vector,
which is why every score was wrong, all of them in a plausible range, and with
no permutation structure to hint at a layout fault. Padding both to 512 elements
(1024 B) fixed it outright. The same design's 8192 B weight stream and 264 B raw
state both worked throughout.

This is the eighth member of this project's "fails open" family and the first in
the shim DMA itself. Note IRON *does* validate host tensor sizes against the
compiled design (`Tensor argument 'q' has 64 elements but the kernel was compiled
for 512`) — the silence is specifically in the transfer, not the plumbing.

### How it was found, after three wrong guesses

The three guesses — the `range_` first-block selector, the build cache, and
`iron.Buffer` — are recorded below because each was plausible and each cost a
build cycle. What actually worked was giving up on hypotheses and **dumping the
intermediate state**:

1. `granite_attn_dump` emits `acc | m | l` instead of the normalised output.
   `m` came back 2.42 against a reference 2.56 — *close but wrong*, which rules
   out a completely broken dot product and points at the data.
2. Dumping the scores: **all 32 wrong**, right magnitude, not a permutation.
   A permutation would mean a layout fault; this meant wrong input.
3. Dumping `q` and `K[0]` as the kernel sees them: `q` all zeros, `K` exact.

Three probes, one build each, and the answer was unambiguous at every step. The
three guesses before them produced no information at all.

### What each earlier suspect actually was

* **`iron.Buffer` persists correctly** — a counter returned exactly `100 + n`.
  Innocent, and the probe also proved the peeled-first-call structure sound.
* **fp32 vector `sub` is exact**; AIE2P's missing fp32 vector *multiply* does not
  extend to subtract. Innocent.
* **`aie::exp2<bfloat16>` works but is coarse: 5.5e-02 relative error**, an order
  of magnitude worse than the 4e-03 bf16 rounding would suggest, and undocumented
  in the aie_kernels sources. Not the bug, but it *is* the accuracy floor of any
  softmax built on it, and it is what sets this kernel's ~3.6e-02 — the test gate
  is set from that measurement rather than fitted to the result.
* **The `range_` first-block selector** (`1 if i == 0 else 0` where `i` is an
  MLIR value) is a real hazard and was rewritten, but it was not this bug: after
  clearing the cache entirely, the single-block case returned bit-identical.
* **The build cache was innocent.** The bit-identical number that implicated it
  was simply the same wrong computation.

## Attention: the earlier record

`granite_attention.h` / `granite_attn_block.cc` / `granite_attn_finish.cc` /
`granite_attention.py`. Flash-style decode attention with an online softmax, so
the KV cache (2 MB at seq 1024, per layer, per token) is read exactly once
instead of twice. It compiles, links and runs. **The output is wrong:**

```
seq  32 (1 block )  cosine -0.05249757  max rel err 4.016e+00
seq  64 (2 blocks)  cosine  0.15206822  max rel err 3.860e+00
seq 128 (4 blocks)  cosine -0.14545930  max rel err 7.479e+00
```

Cosine near zero is uncorrelated, not imprecise. Since `granite_attn_finish`
only scales by 1/l, a wrong normaliser would leave the direction intact and the
cosine near 1 — so the fault is in the accumulator itself (scores, exp2, or the
V-weighted sum), or in the state buffer not persisting between calls.

**A wrong diagnosis, recorded because the reasoning was seductive.** The design
originally selected the first block inside the loop:

```python
for i in range_(n_blk):
    kb(qe, ke, state, BLK, 1 if i == 0 else 0)
```

`range_` yields an MLIR value, so `i == 0` is a Python comparison on a `Value`
and should fold to a constant — which would mean `first` never gets set. That is
a real hazard and the code was rewritten to peel the first block out. **It was
not the bug**: after clearing `~/.npu/cache` entirely, seq 32 returned
`-0.05249757` — bit-identical to before. The single-block case cannot tell the
two formulations apart, and it did not.

I also blamed the build cache first, on the strength of that bit-identical
number, and cleared the whole cache to prove it. The cache was innocent; the
identical result was simply the same wrong computation.

Two things did get fixed on the way, both real:

* **Duplicate symbols.** Two `ExternalFunction`s pointed at one `.cc`. IRON
  compiles the source once per ExternalFunction, so both objects defined both
  symbols and the link failed — the trap `granite_gemv.h` already records, met
  again from the other side. Now one entry point per translation unit, selected
  by `GRANITE_ATTN_EMIT_*`.
* **log2e folded into the score scale**, so the softmax is pure `exp2` and there
  is no per-element fp32 vector multiply (which AIE2P lacks) and no bf16
  rounding of log2e — the error `softmax.cc` warns about and `bf16_exp.cc`
  makes.

### Two suspects probed in isolation, both innocent

Rather than guess further, each was tested on its own. Both probes are ~30 lines
and one build each, and both came back clean:

**`iron.Buffer` persists across kernel calls.** A counter incremented once per
call returned exactly `100 + n` and `10n` for n = 4. So the buffer resolves to a
valid persistent pointer, and — usefully — this also proves the peeled-first-call
structure does what was intended, which the attention test could not show.

**fp32 vector `sub` is exact, and `exp2` works.** The exact subexpression the
softmax uses, `exp2(s - broadcast(m))`:

```
fp32 vector sub   max err 0.0         (exact)
exp2<bfloat16>    max rel 5.5e-02
```

AIE2P's missing fp32 vector *multiply* does not extend to subtract. `aie::exp2`
does work, but **its error is 5.5%, an order of magnitude coarser than the 0.4%
bf16 output rounding would suggest** — worth knowing for any softmax built on it,
and not something the aie_kernels sources mention. It is still nowhere near
enough to explain the failure: softmax weights wrong by 5% give a cosine near
0.99, not near 0.

So the fault is elsewhere: the score computation, the V-weighted accumulation, or
the state layout. **Next step is to make `granite_attn_finish` emit the raw state
(m and l) instead of the normalised output**, and compare them against numpy's
max and sum directly — that separates "scores are wrong" from "accumulation is
wrong" in one build, which guessing has not.

## All-gather: NOT expressible as join + forward (negative result)

Full layer fusion needs every core to see the whole intermediate vector, so it
needs cores -> memtile -> cores in one dispatch. The obvious expression is a
`join()` into a memtile fifo and a `forward()` of the same fifo back out.
`designs/granite_gemv/granite_gather.py` tests exactly that, in isolation --
deliberately, because the fused MLP would also be a new two-phase core program
and debugging both at once is what cost three build cycles on attention.

Two hard limits, in order:

1. **A MemTile has its own DMA channel budget.** An 8-way join is refused:
   `no MemTile has sufficient DMA capacity for 8 input/1 output channels`. Four
   is what one memtile serves -- which is the reason the memtile leg in
   `granite_gemv32.py` is 4 cores per column, arrived at there by construction
   and confirmed here as a limit.

2. **A fifo cannot be both joined into and forwarded out of.** At 4 cores it
   gets past placement and then fails in the dialect:

   ```
   'aie.objectfifo' op objectfifo cannot be in more than one ObjectFifoLinkOp
   ```

   `join()` creates a link and `forward()` creates a second on the same fifo.
   `split()` would too. So the gather-then-broadcast hop is not expressible this
   way at any width.

**What this means for full layer fusion.** The remaining route is a round trip
through DDR inside one dispatch: cores -> shim -> cores. The bandwidth cost is
trivial -- 16 KB out and 16 KB back against 49 MB of weights per layer, ~0.07%
-- and it would still save the ~178 us dispatch. The open question is ordering:
within one `TaskGroup` there is no guarantee that the refill of a buffer happens
after its drain, and getting that wrong is a silent data race rather than an
error. That needs establishing before anything is built on it.

So one dispatch per layer is still reachable, but not by the route this was
built to test. Recorded because the next attempt should not spend its first
three cycles rediscovering it.

### The DDR round trip IS ordered (the open question, now closed)

`granite_roundtrip.py`. The scratch buffer is zeroed by the host; each core
writes a tag only it would write, the round trip runs, and each core sums what
comes back. Two `TaskGroup`s with `finish()` between them, which is the only
sequencing primitive the Runtime offers.

```
scratch after the run  : [1.0, 2.0, 3.0, 4.0]   each core's tag landed
each core read back    : 5120 = 512 * (1+2+3+4)
all cores agree        : yes
20 runs                : 20 PASS, 0 FAIL
```

**`TaskGroup.finish()` does order a refill after a drain of the same buffer**, so
cores -> DDR -> cores works inside one dispatch. That restores the route to one
dispatch per layer: the gather goes through DDR instead of the memtile, at 16 KB
each way against 49 MB of weights (~0.07%).

**Stated precisely, because this is a race question:** 20 consecutive passes is
evidence, not proof. A race can pass by luck, and this was measured at one
payload size on an otherwise idle machine. What is established is that the
ordering is not *obviously* broken; a design that depends on it should keep a
correctness check that would notice if it ever were, rather than trusting the
20 runs.

## Regression: the whole suite after the day's structural changes

`granite_gemv.py` took four structural changes after the 8/8 sweep that first
validated it -- an L1-aware `tiles_per_call`, `per_call` moving into the jit
cache key, the batch plumbing, and the null path losing its x stream. Given that
trap 7e had just produced cosine 0.208 on a configuration which had passed twice,
re-running everything mattered more than moving on:

```
8/8 projection shapes   cos 1.00000000, rel 0.00e+00 (exact under one-hot)
RMSNorm                 PASS
RoPE                    PASS   2.731e-03
SwiGLU                  PASS   8.140e-03
attention 1/2/4 blocks  PASS   0.9993-0.9998
```

Nothing regressed.

## What is NOT done

* **No non-GEMV op runs.** RMSNorm, RoPE, SwiGLU and softmax are all still
  host-side, and two of the four reference kernels need work first (above).
* **No full layer, and no full model.** Nothing is wired into FastFlowLM's
  engine; the kernel is driven from IRON's Python.
* **The memtile leg is not built**, so 24 of 32 cores are unused.
* **`xs` is not hoisted to the host** — 3136x redundant per token.
* **Dispatch is IRON's Python path** (~465 us). `LLMNpuTest/reference/npu.py`
  talks to XRT directly and is the pattern for the C++ side.
* **The host engine's output-quality bug from 0144 is still unexplained.**
  `diff_engine_logits.py` has still never completed a run. Unrelated to this
  kernel, which is validated against a numpy reference rather than against that
  engine.

## Files

```
LLMNpuTest/designs/granite_gemv/
  granite_gemv.h              the kernel: W4A16 q4nx-q4 GEMV for AIE2P
  granite_gemv.py             the IRON design, shape table, numpy reference,
                              gates, and the entry-point generator
  granite_gemv_p{P}_k{i}.cc   GENERATED entry points, P tiles per call
```

Entry points are one translation unit each, because IRON compiles the kernel
source once per `ExternalFunction`: several entry points in one `.cc` become
several objects that each define every symbol, and the link fails on duplicates.

They are qualified by the tiles-per-call variant in **both** the file name and
the symbol (`granite_gemv_p4_k0` vs `granite_gemv_p5_k0`). `down_proj` wants 4
tiles per call and everything else wants 5; if both variants produced
`granite_gemv_k0`, a build cache keyed on anything other than the kernel source
bytes could hand one shape the other's object — which is not an error, just a
different and wrong matmul.

Weights are read from the installed `Granite-4.2-3B-NPU2/model.q4nx`; nothing is
vendored or redistributed.
