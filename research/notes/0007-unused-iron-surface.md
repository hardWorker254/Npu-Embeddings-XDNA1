# 0007 — the IRON surface we never typed, and what four outside repos already measured

*Written 2026-08-19, [`tasks/0044`](../../tasks/0044-m9-optimisation-sweep/TASK.md).
Everything marked **verified** was checked against the installed toolchain or
our own hardware today; everything marked **theirs** is someone else's measured
number, attributed, and not yet reproduced here.*

---

## Why this note exists

Nine milestones of this project used one slice of IRON: `ObjectFifo` with
`.split()` / `.join()` / `.forward()`, `Worker`, `Runtime`, `@iron.jit`, plus
`TensorTiler2D` and `CompileTime`. That slice was enough to get a full encoder
onto the array, so nothing ever forced a second look at the API.

The method here was deliberately inverted: instead of re-reading the features we
use, **grep our own tree for every API name the programming guide mentions and
keep the ones that come back zero.** Twelve did.

Separately, four people are building on the same silicon in public
(`prior-art.md` has the context). Two of them have measured
things we have argued about.

---

## Part 1 — present in IRON, never typed by us

**Verified**: all of these exist in the *installed* `mlir_aie` 1.3.4 wheel, not
only in the source checkout. The distinction matters — the checkout at
`C:\dev\mlir-aie` is tagged `v1.3.4` but carries 2026-dated guide sections, so
"documented" and "available" needed separating:

```powershell
cd C:\dev\mlir-aie
.\ironenv\Scripts\python.exe -c "import inspect; from aie.iron.dataflow.objectfifo import ObjectFifo; print(list(inspect.signature(ObjectFifo.__init__).parameters)); import aie.iron as I; print([n for n in ['CascadeFlow','TileDma','DmaChannel','Bd','Flow','Lock','PacketFlow','Acquire','Release'] if hasattr(I, n)])"
```

Every name below came back present. Ordered by which of our constraints it
attacks.

### 1. `pad_dimensions=` — the attention geometry wall is a padding problem

[`0043`](../../tasks/0043-m9-attention-geometry/TASK.md) established a
*structural* result: attention's per-head GEMM is `[64,64] x [64,64]`, the
design requires `N % (n * cols) == 0`, and the bf16 microkernel requires
`n % 16 == 0`. So `n * cols` must divide 64 with `n >= 16`, hence **`cols <= 4`**
— a design that can express attention can use at most half the array.

That derivation takes `N = 64` as given. **It is not given.** AIE-ML DMAs apply
constant padding at the buffer-descriptor level (`bd_pad_layout<before, after>`
in `AIEOps.td`, surfaced as `ObjectFifo(..., pad_dimensions=...)`), so the
per-column N slice can be padded 8 -> 16 *in the DMA*, with no host-side buffer
and no change to the weights. Then `n = 16, cols = 8` is legal and the wall is
gone.

**Read the verifier before designing around this.** `AIEDialect.cpp`'s
`DMABDOp::verify()` imposes three restrictions that the guide does not mention:

```cpp
if (!dims.has_value())        // padding requires an n-d access pattern
  return emitOpError() << "Padding requires n-d data layouts ...";
if (!parentTile.isMemTile())
  return emitOpError() << "Padding is only supported by memtile dma bds.";
// ... and inner-most pad-before/after must land on 32-bit word boundaries
```

**Padding is mem-tile only** — not shim, not core. That happens to be the right
place for this: in the whole-array design B already goes L3 -> mem tile -> split
across the column's four cores, so the mem tile pads the column's real 8-wide
slice to 16 on the way in. It also means the trick is one-directional — padding
adds columns, it cannot remove them — so **C comes back 16-wide per column and
the output side needs a strided `dims_to_stream` that takes 8 of every 16.**
That is an ordinary access pattern, not a second padding feature, but it is a
second thing to get right and it is where a first attempt will fail.

The price is arithmetic, and it is small: the padded half of every QK^T and A.V
tile is multiplied by zero and discarded, so attention's FLOPs double — and
**F3** prices attention at **5.3% of encoder work at seq 128**, so the
encoder-wide cost is ~5%, bought against **2x the columns for every operator in
the design**.

This does not make fused attention a good idea by itself (see §3.4 — someone
built it and it lost). It removes the reason 0043 gave for not being able to
try.

### 2. Cross-tile `Buffer` — the 63 KB L1 budget is per *worker*, not per *design*

Guide §2h: a `Buffer` pinned to a **different tile** than the `Worker` it is
passed to no longer raises. AIE core tiles read their north/south/east/west
neighbours' L1 directly over shared-memory paths, and `Program.resolve()` now
wires that up. Upstream's `ml/magika/group2.py` spreads four large lookup tables
across a compute tile and its three neighbours "so the kernel can read all four
without DMA round-trips".

`CLAUDE.md` trap 3 — *budget L1 before compiling,
`2*(m*k*in + k*n*in + m*n*out) < 64512`* — has priced every tile decision this
project has made, including `tile_n = 32` for bge-large
([`0042`](../../tasks/0042-m9-bge-large/TASK.md)) and the geometry table in 0043.
It remains exactly right for the *streaming* operands. It does not bind for
**resident, read-only** data, which can now live next door.

This is also the IRON API for what ARIES did
(`fpga25-aries.md`): beat the vendor overlay 1.24x
with scalar, unvectorised code on fewer cores, purely by handing intermediates
between adjacent tiles' L1. We indexed that paper in
[`0028`](../../tasks/0028-research-index-nine-new-papers/TASK.md) and recorded it
as evidence for fusion without noticing the mechanism had landed in the API.

### 3. `CascadeFlow` — a core-to-core accumulator path that costs zero descriptors

`aie.cascade_flow(src.tile, dst.tile)`, driven by `put_mcd` / `get_scd`. Each
compute tile has one cascade input (from N or W) and one output (to S or E);
shim and mem tiles have none.

Two properties make this ours to want:

- It is **not a DMA**. No `aie.lock`, no `aie.dma_bd`, no stream-switch route.
  [`0024`](../../tasks/0024-m7-dispatch-cost-anatomy/TASK.md) priced a design
  switch at **~25 us + 7.2 us per lock**; a cascade edge adds nothing to that
  bill. Every other way of moving a partial sum between cores does.
- It carries the **accumulator**, not the rounded result. Upstream's own matmul
  README says so: "the AI Engine also provides a higher-precision cascade data
  path, which can be used to accumulate results between cores".

`programming_examples/basic/matrix_multiplication/cascade/` splits the K
reduction across the four rows of a column. It is explicitly **scalar-only and
single-buffered** upstream, so it is slower than the vectorised whole-array
design *today* — but the kernel is the part we would write anyway, and the
dataflow is the one the ICPP Stationary-B algorithm wants
(`icpp25-aie-mmm-models.md`).

### 4. `consumer_obj_type=` — the B-reuse blocker was a fifo-shape problem

[`0010`](../../tasks/0010-m5-b-reuse-and-cost-model/TASK.md) priced B reuse at
**1.26x (M=512) to 1.68x (M=4096)** and then could not build it, for two
reasons: ObjectFifo depth maps 1:1 onto mem-tile buffer descriptors (ceiling 6
tiles at 4 columns, 4 at 8, against a slice wanting 48/24), and the
one-big-object workaround hit *number of input DMA channel exceeded* on a
**core** tile.

`consumer_obj_type=` is precisely "one big producer transfer, many small
consumer acquires, in **one** fifo": the producer sends `obj_type`-sized chunks,
the consumer receives `consumer_obj_type`-sized chunks, producer count an
integer multiple of consumer count. The guide's stated use case is "a single DMA
fan-out serving consumers that want to walk the data in smaller chunks (e.g.
weight broadcast feeding a row of compute tiles that each acquire a sub-slice),
**without paying for two separate fifos and a join**".

That is the shape 0010 wanted and could not express. It does not automatically
follow that the descriptor ceiling moves — that has to be built and counted —
but the workaround that failed was a *different* workaround.

### The capacity was never the problem — checked, 2026-08-19

Worth stating because the obvious question is "is there room to stage B at all?"
There is, and the answer differs by model. Per column: mem tile **512 KB**
(`BaseNPU2TargetModel::getMemTileSize()`), minus A double-buffered and the C
join, leaves ~400 KB (MiniLM geometry) / ~432 KB (bge-large geometry):

| model | B slice per column | fits in L2? |
|---|---:|---|
| MiniLM / bge-small `qkv` | 108 KB | **yes**, 3.7x headroom |
| MiniLM / bge-small `ffn_up` / `ffn_down` | 144 KB | **yes**, 2.8x headroom |
| bge-large `attn_out` | 256 KB | yes |
| bge-large `qkv` | 768 KB | **no** — 56% stageable |
| bge-large `ffn_up` / `ffn_down` | 1,024 KB | **no** — 42% stageable |

So at h=384 the **entire** column slice is resident-able and B could be streamed
from DDR **once instead of 32 times** (`M/m/rows` = 8192/64/4). At h=1024 three
of four shapes overflow, and the answer there is either partial staging or the
**Stationary-B** algorithm the ICPP paper describes
(`icpp25-aie-mmm-models.md`), whose budget
`2mk·T_in + kn·T_in + 2mn·T_out` is what fits a larger K chunk in the same space.

**And note where the shim is not.** A shim tile has a DMA engine and a stream
switch and **no data memory at all** — `AIEDialect.cpp` rejects the idea
outright with *"Shim tiles cannot have an allocation scheme"*, and
`getMemTileSize()` is a mem-tile property. The staging shelf is L2, one row up
from the door.

**And the same constraint has a second unused dial: per-endpoint depth.** Guide
§2a — *"the producer and each consumer have their own working resource pool …
the user does however have the possibility to manually choose the depth of these
pools"* — and IRON exposes it as `of.prod(depth=)` / `of.cons(depth=)`, so a
fifo can be, say, `[2, 3]` rather than 3 everywhere. Since 0010 established that
**depth maps 1:1 onto mem-tile buffer descriptors**, a symmetric depth spends
descriptors on the endpoint that does not need them. We call `.prod()` /
`.cons()` **81 times across our designs and pass a depth in 2 of them**, so 97%
of our endpoints take whatever the fifo-wide default was.

The guide's own caveat: this is for producers and consumers *on different
tiles*, and its documented purpose is the broadcast-with-skip-connection
dependency, not descriptor thrift. Ours would be a second use of it.

### 5. `disable_synchronization=True` + `delegate_tile=` — lockless intra-core chaining

`disable_synchronization` skips lock generation for an ObjectFifo entirely;
`delegate_tile` puts the fifo's buffers on a neighbouring tile's memory module.
Upstream uses them together in `ml/mobilenet/bottleneck/regular.py`, commented
**"self-loop fifos (no synchronization — single core)"**, to chain several
operators inside one worker.

Two things follow for us. It is the idiom for **chaining ops within a core**,
which is what layer fusion is at the smallest scale. And every lock it removes
is 7.2 us off every design switch, by our own 0024 model — relevant again the
moment we have more than the one static design production runs today.

### 6, 7, 8 — smaller, same family

- **`aie_stream=(end, port)`** — marks a fifo wire-only: no L1 buffer is
  allocated and the kernel emits with `aie::stream::put_ms()`. Saves L1 and a
  descriptor on any producer whose output is consumed immediately.
- **`init_values=`** — static per-buffer initial values baked into the design
  image. LayerNorm's gamma/beta and every polynomial coefficient table are
  small, constant, and currently arrive by DMA. Baked in, they cost no transfer
  and no descriptor.
- **`TileDma` / `DmaChannel` / `Bd` / `Lock` / `Flow` / `Acquire` / `Release`**
  (guide §2g) — hand-wired DMA inside a normal `@iron.jit` design. This is the
  only way to *minimise descriptor count directly*, and descriptor count is the
  quantity our own switch-cost model is denominated in. The guide never mentions
  switch cost; the connection is ours.

### 9. Loop hints — close this one

Note [`0006`](0006-peano-loop-hints.md) found that `AIE_PREPARE_FOR_PIPELINING`
compiles to nothing under Peano (8 uses in our kernels), that
`AIE_LOOP_MIN_ITERATION_COUNT` does bind, and that we have never used the upper
bound or either unroll. It declined to add `AIE_LOOP_UNROLL_FULL` on the grounds
that our loops lack the index-dependent `switch` that AMD's -47% precedent came
from — an inference, explicitly labelled as one.

**whisper-xdna measured it** (theirs): on their straight vector loops,
`AIE_LOOP_UNROLL_FULL` was **14% slower**, with the same explanation — "a
straight vector loop with compile-time constants has nothing to unroll and only
gains register pressure". Two independent arrivals at the same conclusion, one
by reasoning and one by measurement. **Treat the unroll hints as closed for
kernels of this shape.**

---

## Part 2 — checked and worth nothing (recorded so nobody checks again)

### `burst_length` is already maximal — but the arch fallback caps it

`aie.dma_bd` carries a `burst_length` attribute and `aiex.py` documents "if 0,
defaults to the highest available value". `getShimBurstLength()` in
`AIEDialect.cpp` takes `max_element` over the target model's table when the
value is 0, and `BaseNPU2TargetModel::getShimBurstEncodingsAndLengths()` returns
`{(0,64), (1,128), (2,256), (3,512)}`. **Verified: we already get 512-byte shim
bursts, and there is nothing above it to ask for.**

One thing falls out that is *not* nothing. `AIE2TargetModel` returns
`{(0,64), (1,128), (2,256)}` — **no 512**. So note
[`0002`](0002-iron-silent-arch-fallback.md)'s silent fallback to `aie2` does not
only halve the bf16 `mac_dims` and neuter the bfp16 flag; it also **halves the
maximum shim burst length**, on a design that is data-movement bound. That is a
third consequence of the same silent failure, and it was not on the list.

### DMA compression exists, is lossless, and almost certainly will not pay

`programming_examples/basic/dma_compression/` is a 2026 silicon probe of
`Enable_Compression` on compute-tile and memtile DMAs, on **npu2 among others**.
It establishes that the feature works, that it is losslessly invertible on
non-trivial data, that enabling it needs *two* register writes (per-BD bit 31
and per-channel bit 4), and that **mlir-aie's own passes plumb neither for the
compute-tile path** — you flip them from the runtime sequence via
`npu_maskwrite32` inside `Runtime.inline_ops`, or from the core with peano's
`write_tm`.

We are bandwidth-bound, so this looks like a headline. It probably is not: the
only ratio reported is **1.39x on `arange`**, which is a sequence of int32s with
mostly-zero high bytes. AIE-ML DMA compression is a zero/sparsity scheme; dense
bf16 weights drawn from a trained distribution have no zeros to find.

It is worth **one cheap decisive experiment** rather than an argument: run the
existing `cmp_only` config with a real `.npue` weight tile as the input tensor
and read the compressed byte count off the shim TAP. If it is not meaningfully
below 1.0x on real weights, close it permanently. If it is, it applies to the
21 MB (MiniLM) / 604 MB (bge-large) that stream every dispatch.

Also from that page, unrelated but real: a **VLIW-bundle hazard**
([mlir-aie #2346](https://github.com/Xilinx/mlir-aie/issues/2346)) where Peano
packs consecutive `st.tm` ops into one bundle and the second store may issue
before the first reaches the processor bus. Only matters if we ever write tile
registers from a kernel.

---

## Part 3 — from the outside repos

Cloned to `externalrepos/` (gitignored, read-only, never vendored).
[`whisper-xdna`](https://github.com/drakosha/whisper-xdna) is the substantive
one: the Whisper **encoder** on XDNA1, which is the same problem shape as ours —
bf16 GEMMs with fp32 accumulate, a polynomial GELU, a host/device split, and a
CPU baseline it has to beat. Its `HISTORY.md` is measured the way our `tasks/`
are. [`hawkpoint-npu-llm`](https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm)
is a decoder-side project whose own performance doc declines to attribute its
274 ms/token to any phase; its kernels use a scalar float reduction over 128
lanes and an index `switch` in the hot loop, i.e. our
[`0001`](0001-aie-kernel-pitfalls.md) trap 5 and AMD's own anti-pattern, so it is
read for architecture and not for kernel technique.
[`npu-linux-kit`](https://github.com/Scottcjn/npu-linux-kit) marks its
embeddings module "candidate, not built".
[`phoenix-sdr-dsp`](https://github.com/midhatn/phoenix-sdr-dsp) is
Windows-native on Phoenix and contributes one fact below.

### 3.1 The default AIE rounding mode is `floor` — **verified, and we never set it**

Not "round to nearest". `aie_api/aie.hpp` says, twice, "the default
`aie::rounding_mode::floor`", and `aie_types.hpp` defines `floor` as **"always
round towards negative infinity"**. Every bf16 SRS in every kernel we have
written uses it.

```powershell
# ours: zero hits
Select-String -Path experiments\m5-eltwise\kernels\*.cc -Pattern "set_rounding"
```

That is a **systematic downward bias, not noise** — which is exactly the
signature whisper-xdna used to find it (theirs): "94% of the deviations shrank
the magnitude, and noise is symmetric while a systematic shift is not". They
measured `aie::set_rounding(aie::rounding_mode::conv_even)` — one line, zero
wall-time cost — at end-to-end cosine **0.99411 -> 0.99462**.

**This is the one borrowed claim that was run on our own hardware, and it is
larger here than it was for them.** One extra `.cc` per kernel, differing by
exactly that line, through the existing M5 harnesses
([`0044`](../../tasks/0044-m9-optimisation-sweep/TASK.md) Part 3):

| kernel | vs golden, `floor` | vs golden, **`conv_even`** | | implementation error alone |
|---|---:|---:|---:|---|
| GELU | 4.312e-03 | **2.494e-03** | 1.73x | 3.886e-03 -> 1.556e-03 (2.50x) |
| softmax | 4.278e-03 | **3.325e-03** | 1.29x | 3.424e-03 -> 1.481e-03 (2.31x) |
| LayerNorm | 3.326e-03 | **2.059e-03** | 1.62x | 3.659e-03 -> **3.967e-05 (92x)** |

"Implementation error alone" is the NPU against a numpy model of the *identical*
formula. **LayerNorm's did not shrink, it vanished** — at 3.967e-05 the kernel
is indistinguishable from numpy, and its 2.059e-03 against the golden is the
bf16 floor of 2.058e-03. Each harness's `CPU model vs golden` control is
**bit-identical across every pair**, which is what makes these A/Bs.

**This closes a mystery this project opened and then dropped.**
[`0015`](../../tasks/0015-m5-gelu-polynomial/TASK.md) saw the 3.886e-03 gap and
inferred *"`aie::vector<float>` arithmetic on AIE2P is not IEEE fp32"*;
[`0016`](../../tasks/0016-m5-fp32-probe/TASK.md) refuted that with a direct probe
(~24 mantissa bits), and then wrote down the correct hypothesis and left it:
*"or the rounding mode of the fp32 -> bf16 conversion. Truncation instead of
round-to-nearest would fit the magnitude, but that is a hypothesis, not a
finding, and it is not chased here."* It went unchased for 28 tasks. GELU now
measures **2.494e-03 — the exact figure `gelu_poly.cc`'s own header has claimed
all along** as its design prediction.

The mechanism is confirmed without reference to any error metric, because
softmax rows must sum to 1: under `floor` the sums are **min 0.994581, max
exactly 1.000000** — no row can exceed it when every element rounds down — and
under `conv_even` they straddle it (0.997272 / 1.002485) with the worst
deviation halving.

Their caveat does **not** transfer, and the prediction was made before the
measurement: it *breaks* their softmax (cosine 0.879) because the stock
`getExpBf16` lookup table is calibrated for the default mode.
[`0030`](../../tasks/0030-m7-expert-review-tests/TASK.md) replaced our softmax's
table with our own `exp2_poly`, so we have no mode-calibrated LUT to break.
Ours improved.

**Production is unchanged and the shipped kernels still default to `floor`** —
eltwise runs on the host, so none of these kernels executes today, there is no
end-to-end gate to validate a swap against, and `set_rounding` is *core-wide*
state that leaks between kernels sharing a core. The `*_rne.cc` variants are the
evidence; `conv_even` becomes the default when eltwise returns to the array.

### 3.2 The centred-basis claim — tested against our kernel, and it does not transfer

This one arrived looking like the biggest kernel win in the whole sweep, and
measuring it is what made it small. Recorded in full because the *reason* it
does not transfer is the useful part.

**Theirs.** Five GELU kernels in, they found that in the monomial form
`a0 + a1*s + a2*s^2 + ...` the coefficients span orders of magnitude and the
terms very nearly cancel — the same catastrophic cancellation that killed their
tanh kernel, just hidden. Mapping each piece to `u` in `[-1, 1]` keeps the state
O(1) and the coefficients O(0.1), and **plain bf16 coefficients then become
sufficient** (0.00681 against 0.00663 for exact ones) where the monomial form
had needed hi/lo bf16 coefficient *pairs*. Three degree-4 pieces held the
accuracy that had needed five: **12 Horner steps instead of 20, 2.5x faster at
equal accuracy, by changing coordinates rather than optimising code.**

**Ours** (`gelu_poly.cc`) is `max(x,0) + c(min(|x|,4))` with `c` even, a single
degree-8 monomial Horner — 8 fused steps, no pieces, no select, no branch. So
the shape looks identical to the thing they fixed. It is not, and the difference
is one line of the kernel's own comment: *"Horner, fp32 throughout, rounded
exactly once on the store below."* The coefficients are `fp32` literals
broadcast into `aie::vector<float,16>`, and
[`0016`](../../tasks/0016-m5-fp32-probe/TASK.md) established that AIE
`vector<float>` is full IEEE fp32.

**Their fix targets bf16 coefficient precision. We do not have bf16
coefficients.** Measured rather than argued — least-squares fit of `c` on
`[0,4]`, evaluated in float32 Horner exactly as the kernel does, monomial `u`
against centred `t = (u-2)/2`:

| degree | monomial, `u` in [0,4] | centred, `t` in [-1,1] | max abs coefficient, mono / centred |
|---:|---:|---:|---:|
| 5 | 7.324e-03 | 7.324e-03 | 6.435e-01 / 2.463e-01 |
| 6 | 3.332e-03 | 3.332e-03 | 5.449e-01 / 3.249e-01 |
| 7 | 5.913e-04 | 5.872e-04 | 4.990e-01 / 3.249e-01 |
| **8 (ours)** | **3.613e-04** | 3.613e-04 | 4.916e-01 / 3.025e-01 |
| 9 | 1.020e-04 | 1.003e-04 | 4.977e-01 / 3.025e-01 |
| 10 | 3.728e-05 | **1.562e-05** | 5.006e-01 / 2.903e-01 |

**Centring changes nothing until degree 10**, where it is worth 2.4x — and by
then the error is 150x below the floor that matters, so it is unreachable.
Conditioning is simply not binding at fp32.

What *does* fall out is a smaller, real result. The bf16 output floor is
**2.465e-03** ([`0015`](../../tasks/0015-m5-gelu-polynomial/TASK.md)), and our
degree-8 approximation error is **3.6e-04 — seven times below it**. Degree 7
(5.9e-04) is still 4.2x below; degree 6 (3.3e-03) breaks it. So **the kernel can
drop from degree 8 to degree 7 for free, and no further**: one Horner step of
eight. By [`0026`](../../tasks/0026-m7-closing-on-cpu/TASK.md)'s fit
`t ~ 2174 us + 941 us per Horner step`, that is ~940 us per call, ~9% of the
kernel — worth taking when eltwise next runs on the array, and not worth a
redesign.

**The one thing that does transfer lands on this exact kernel.** "Rounded
exactly once on the store below" is a single bf16 SRS, and by §3.1 it currently
rounds toward negative infinity. Our best eltwise kernel has exactly one
rounding in it and it is the biased one.

Same reasoning applies to `exp2_poly.h`; it has not been re-fitted here.

### 3.3 `xrt::runlist` — present in our XRT, unused by our runtime

**Verified**: `C:\Xilinx\XRT\include\xrt\experimental\xrt_kernel.h` declares
`xrt::runlist`, "a list of `xrt::run` objects such that they can be executed
atomically in the order they are added"; `grep -rn runlist runtime/` in our tree
returns nothing. `npu_device.cpp::dispatch_only()` builds one `xrt::run` and
calls `r.wait()`, once per dispatch.

The precondition is `hw_context` identity — "a runlist is associated with a
specific hwctx, and all run objects added must be created in that hwctx". After
[`0032`](../../tasks/0032-m7-one-xclbin-production/TASK.md) we have **exactly
one** hw_context for the whole encoder, so we satisfy it trivially. whisper-xdna
had to work for it and still got **-1400 ms of 8569** by wiring it into their
attention chain (20 submit+wait pairs per layer).

**It is small for us today and that should be said plainly.** Measured this
afternoon on an idle machine, our submit is **66 us per dispatch**, 1.58 ms
against a 184 ms encode — about **0.9%**. The reason is that our dispatches are
enormous (M = 8192) and theirs were
tiny. It becomes worth having exactly when the dispatch count goes up and the
dispatch size goes down: **attention on the array, or eltwise back on the
array**, both of which are the directions everything else in this note points at.
Recorded now so it is not rediscovered then.

### 3.4 Fused attention, built by someone else, correct, and still slower

Their `attn_fused.cc` implements QK -> softmax -> AV as one kernel over 16 tiles,
one head across the array. It is correct (rel L2 **0.0088** against exact softmax
attention, versus 0.0068 for the three-overlay path it replaces) and it collapses
their design set from 6 contexts to 4. **It still loses: 1151 ms against
1016 ms.**

Three sub-results that de-risk our 0043 whichever way it goes:

- **The accumulators must be fp32.** With bf16 accumulators rel L2 is **0.120**,
  and the dominant term is the *running softmax sum*, not the output accumulator
  — its error scales the entire row through the final division. fp32 fixes the
  accuracy and **eats the entire transport saving**. That is the whole reason the
  fused version loses.
- **A compute tile has 2 input DMA channels and Q + K + V needs 3.** They ship K
  and V as a single packed object. This is our `CLAUDE.md` trap 3b arriving from
  the other side, with the workaround attached.
- **Tail masking after softmax**, not as a `-inf` sentinel before it.

Their non-fused **chained** path is the one that won: QK writes bf16 straight
into a sub-buffer that softmax normalises in place and AV reads back, with no
`sync` between stages — **8.7 s against 15.1 s, 1.73x**. They note this works
even across differing hw_contexts; we have one, so it is strictly easier here.

### 3.5 Two numbers that price things we have argued about

- **Tile size is per-shape, and the swing is large** (theirs): `1536x64x1536`
  wants 32/64/64 and `1536x1536x64` wants 64/64/16, "a 2.5x and 4.6x difference
  respectively"; the skinny attention shape reaches **386 GFLOPS against 1761**
  for a square one. Our M2 finding — per-core cost flat to 0.4% across four
  MiniLM shapes — is not contradicted: our four shapes are all fat, and theirs
  are not. But it prices a real tension in the one-xclbin architecture, which
  *requires* one tile geometry for every shape. If attention ever joins the
  design, it brings the skinny shape with it.
- **int8 is 1.33x, not 2x** (theirs), on the projection and MLP GEMMs. Ours would
  have the geometry for it: **verified**, `_MM_MAC_DIMS["aie2p"]` gives int8
  `(8, 8, 8)` against bf16's `(4, 8, 8)`. The gap between 2x on paper and 1.33x
  measured is the same movement bound we have measured everywhere else.

### 3.6 One toolchain fact from phoenix-sdr-dsp

They pin **mlir-aie v1.4.1**; we are on **1.3.4**. They also document the trap
that the rolling `latest-wheels-4` channel resolves an untagged checkout back to
1.3.4 — worth knowing before anyone upgrades. Not urgent: every feature in Part 1
is already in our installed 1.3.4.

---

## Part 4 — the reframing this task actually produced

Everything above is a lever. This is a **correction to how we priced an
architecture decision**, and it comes from our own bench output.

**This is not a new idea and should not be presented as one.** The expert review
raised it as **§6b, "device-resident intermediates"**, and
[`0030`](../../tasks/0030-m7-expert-review-tests/TASK.md) deferred it *with
cause* — it needed §4's architecture, which did not exist yet — while pricing it
at **"t_conv + syncs ≈ 70 ms at batch 128"**
([note 0005](0005-expert-review-tests.md)).
[`0032`](../../tasks/0032-m7-one-xclbin-production/TASK.md) then noted, in
passing, that "the blocker 0030 deferred it on is gone." It has been open and
unblocked since, and nothing has picked it up.

Two things are new, and both make it bigger:

**1. The §6b prize was under-scoped by roughly a factor of two.** It counted
`t_conv` and the syncs. It did not count **`read out + bias`**, which is the
single largest term — 18.8% of the encode against the 14.2% that §6b did count.

**2. That term grew *because of* the 0032 decision.** 0032 put LayerNorm,
softmax and GELU on the host because each was "measured faster **and** more
accurate than its NPU dispatch". That is true, and the mechanism was the 2.3 ms
design switch each NPU eltwise dispatch cost. But moving them to the host means
**every** GEMM result now crosses the bus, so the readback that §6b did not count
is exactly what the move added.

**0032 priced the operator. It did not price the transport the operator
forces.**

MiniLM, batch 128, seq 64, **on an idle machine** (the stale hw_context of
`tasks/0044` Finding 0 was killed first; two runs agreed to 0.9% on wall and to
0.2 points on every share):

```powershell
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 `
    --artifacts artifacts_b128il --bench 10 --threads 24
#   wall 184.45 ms -> 694.0 seq/s   (repeat: 186.19 ms -> 687.5)
```

| | ms | share |
|---|---:|---:|
| host gelu + softmax + layernorm | 13.8 | **7.5%** |
| | | |
| `read out + bias` (device->host C) | 34.7 | **18.8%** |
| bf16 convert (both ways) | 16.6 | **9.0%** |
| sync to + from device | 9.6 | **5.2%** |
| **transport total** | **60.9** | **33.0%** |

The three host operators cost 7.5%. The host<->device movement of intermediates
**that exists only because the next operator runs on the host** costs **33%** —
against the **14.2%** (`t_conv` + syncs) that §6b's estimate covered.

**Contention flattered the old architecture.** The first draft of this table was
taken with a foreign hw_context resident and read 22% / 3.6%; the hardware wait
was 15,541 µs per dispatch instead of 3,024, so every host term was diluted by an
NPU that was five times slower than it should have been. On an idle machine the
NPU shrinks and the transport does not — `dispatch + wait` falls from 65.0% to
40.3% while `read out + bias` *rises* from 14.3% to 18.8%. **The faster the array
gets, the larger this term becomes**, which is the opposite of how a cost one
plans to optimise away later should behave.

In the **production** configuration it is starker still, because the per-lane
breakdown names it directly:

```powershell
... --bench 10 --threads 24 --pipeline 2
#   wall 282.10 ms -> 907.5 seq/s
#   p1 host work 142.33 ms  50.5%  (conv 22.7  bias 70.7  attn 29.2  elt 19.7)
#   p2 host work 146.24 ms  51.8%  (conv 22.8  bias 74.4  attn 30.4  elt 18.6)
```

Host work is **half the wall clock per lane**, and `bias` — the C readback — is
**70.7 ms of it, 3.6x the cost of all three eltwise operators combined (19.7 ms)**.
The thing 0032 moved to the host is the *smallest* line in the budget it created.

`read out + bias` is not accounting noise; `main.cpp` reads the fp32 C buffer out
of an XRT `bo` with `_mm256_stream_load_si256` because ordinary loads measured
~2 GB/s on that write-combined memory. Per MiniLM encode at batch 128 that is
**679 MB of fp32 C** — 37.7 + 12.6 + 50.3 + 12.6 MB per layer, six layers — read
at about 8 GB/s.

Two consequences:

1. **The eltwise-on-array question is not settled**, because it was never asked
   this way. In the one-xclbin architecture an eltwise dispatch costs no design
   switch (that is the whole point of 0029/0032), and what it *saves* is not its
   own runtime but the round trip on either side of it. The 0032 comparison
   should be re-run with the transport on the ledger. The obstacle that remains
   is real and different: a core runs one program and 16 KB of program memory
   holds at most two of our kernels
   ([`0032`](../../tasks/0032-m7-one-xclbin-production/TASK.md)), so a mixed
   design needs **one operator per core** across different columns/rows — not the
   abandoned three-op universal worker.
2. **There is a cheaper, independent version** that needs no eltwise kernel at
   all: give the GEMM a **bias + fp32->bf16 epilogue** and let C leave the array
   as bf16. 679 MB becomes 340 MB, and most of the 4.7% host convert disappears
   with it — call it 10% of the encode. The mechanism already exists: 0030 built
   `epilogue="gelu"` as a second kernel in the worker and measured the fused form
   **10x more accurate** than the separate path. The accuracy question is
   specific and testable: today the host eltwise sees fp32 input, and this would
   hand it bf16 — one extra rounding, measurable against the goldens, on a path
   whose current `1-cos` is 8.348e-06 (bge-small) against an MTEB gate of -0.03.

Note that this does **not** contradict `CLAUDE.md` trap 2. Trap 2 is about the
*accumulator*: `output_dtype=bf16` re-rounds at every K step, 7.4e-3 against
1.21e-07. An epilogue converts **once, after the full K reduction**, with the
accumulation still in fp32.

---

## What to do with this, in order

1. ~~**`set_rounding(conv_even)`**~~ — **DONE and measured** (§3.1): 1.73x / 1.29x
   / 1.62x on GELU, softmax and LayerNorm against the goldens, and LayerNorm's
   implementation error fell 92x to essentially zero. `*_rne.cc` variants exist;
   flip them to default when eltwise returns to the array.
2. **Bias + bf16 epilogue on the GEMM** (§4.2). No new dataflow, uses a mechanism
   0030 already built, ~10% of the encode, one testable accuracy question.
3. **Re-price eltwise-on-array with transport on the ledger** (§4.1), as
   one-operator-per-core across columns, not one worker with three programs.
4. **`pad_dimensions` on N for attention** (§1.1) — reopens 0043 at 8 columns
   instead of 4, at ~5% of encoder FLOPs. Read §3.4 first: someone has already
   built the fused version and it lost, for a reason (fp32 accumulators) that
   applies to us identically.
5. **The compression experiment** (§2) — one run, decisive either way.
6. **`consumer_obj_type` against the 0010 B-reuse ceiling** (§1.4) — priced at
   1.26-1.68x by our own cost model, and the thing that blocked it was a
   different workaround.
7. **`xrt::runlist`** (§3.3) — worth 0.9% today, worth having the moment 3 or 4
   lands.
8. **`gelu_poly` degree 8 -> 7** (§3.2) — ~9% of that kernel, only matters if 3
   happens.

Closed, do not revisit: `burst_length` (§2), `AIE_LOOP_UNROLL_FULL` (§1.9),
**centred polynomial basis** (§3.2 — measured, worth nothing at fp32).
