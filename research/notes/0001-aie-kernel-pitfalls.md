# 0001 — AIE kernel pitfalls found by disassembly

*First observed during [M1](../../tasks/0002-m1-hello-npu/TASK.md), 2026-08-16.*

A running list of things that silently destroy AIE kernel performance and are
**invisible in the source** but obvious in the emitted object. Check these before
blaming the design.

```powershell
$P   = $env:PEANO_INSTALL_DIR
$obj = "<jit-cache-dir>\<kernel>.o"

& "$P\bin\llvm-nm.exe" $obj | Select-String '__div|__mul|__udiv|__float'   # must be empty
& "$P\bin\llvm-nm.exe" --print-size $obj                                    # spills / size
& "$P\bin\llvm-objdump.exe" -d --no-show-raw-insn $obj                      # loop body
```

## 1. Scalar float arithmetic becomes a function call

**Measured cost: 1,617×.** The scalar SAXPY variant ran 541,662 cycles against 335 for
the vectorised one — 132 cycles per element.

The cause was not narrowness. `llvm-nm` showed `U __mulsf3`, and the loop contained a
call instruction per element:

```
      6c:  lda.s16  r0, [p7, dj0]
      70:  lda.s16  r3, [p0, dj0]
      76:  jl  #0x0                  <-- call to __mulsf3, once per element
```

**Rule:** never use scalar `float`/`bfloat16` arithmetic in a kernel body. Use
`aie::vector` / `aie::accum` and the `aie_api` operations. A stray scalar `a * x` in a
tail/epilogue is enough to dominate the kernel.

**Symptom to watch for:** an undefined symbol beginning `__` in `llvm-nm` output.
`__mulsf3`, `__divsf3`, `__floatsisf` and friends are all software-float routines.

## 2. Integer division emits `__divsi3`

Called out by AMD's own `skills/aie-kernel-opt/SKILL.md` as a priority lever. AIE has
no integer divide. Any `/` or `%` by a non-power-of-two runtime value becomes a library
call in the inner loop.

**Rule:** make divisors compile-time constants (ideally powers of two) so the compiler
turns them into shifts. Our M1 kernel is clean — verified no `__div` symbols.

## 3. `nop` padding means the loop is issue-limited, not compute-limited

The M1 vectorised loop body is 5 VLIW bundles carrying only 8 real vector ops across
~25 slots:

```
.LBB0_1:
  90:  nopa ; vldb x4       ; nops ; nopxm ; nopv
  a0:  nopa ; vldb x5       ; nops ; nopxm ; vadd.f
  b0:  nopa ; nopb          ; vst  ; nopxm ; nopv
  c0:  vlda ; nopb          ; vst  ; nopxm ; vmul.f
  d0:  vlda ; nopb          ; nops ; nopxm ; nopv
```

This is what `get_vector_time()` reported as **0.382** — 62% of the window is not
issuing vector work.

For a memory-bound elementwise kernel that is expected and not worth fixing. For GEMM
it is the pathology to avoid: it is exactly why Rösti
([2504.03083](https://arxiv.org/abs/2504.03083)) computes **four independent output tiles in
four accumulator registers**, so four back-to-back VMACs fill the slots instead of
stalling on a RAW hazard against a single accumulator (4-cycle result latency).

**Rule:** in a GEMM microkernel, count the `nop`s in the ZOL body. Their presence is
the signal to add accumulator independence.

## 4. Cross-checking cycles without hardware

The trace and the disassembly must agree. For M1:

```
loop count from  `add.nc lc, r2, #-0x3`  with r2 = 0x40  ->  61
5 bundles x 61 iterations                                ->  305 cycles
+ prologue/epilogue                                      ->  ~335
measured                                                     335
```

Because AIE cores never stall (no cache, no OoO, no branch prediction, fixed
latencies), bundle count × iterations *is* the execution time for a compute-bound
kernel. When the two disagree, the kernel is waiting on data — which is itself the
finding.

Note the `-0x3`: three iterations are peeled into the prologue/epilogue for software
pipelining, so the ZOL trip count is `N/lanes - 3`, not `N/lanes`. Don't be surprised
by the off-by-three.

## 5. `trace.txt` can be silently empty

`trace_512x512x512.txt` in the user's earlier experiments is **0 bytes** — the run
completed, produced no trace, and said nothing. Almost certainly trace-buffer overflow
(`trace_size=524288` was not enough for 512³).

**Rule:** always assert `trace.txt` is non-empty before trusting any conclusion. Our
`experiments/m1-hello-npu/saxpy.py` checks this and warns explicitly. A missing
measurement must never be mistaken for a good one.

---

## Addendum, 2026-08-28 — three more, from the OpenFFLM lm_head GEMV

Found building `../LLMNpuTest/designs/lm_head`, a W8A16 GEMV over
248320 x 1024. Same toolchain: mlir-aie 1.4.2.dev16+g7e00b57, Peano
21.0.0.2026080301, NPU2/AIE2P.

### AIE2P has no fp32 vector multiplier, and aie_api returns zero rather than say so

`aie::mul(vector<float, N>, vector<float, N>)` **compiles clean and produces
zero**. No warning, no diagnostic, no error at any stage of the pipeline.

This is the most expensive kind of bug this project keeps meeting — the same
shape as trap 1 (silent arch fallback) and trap 5b (dropped pragmas). It cost
five hardware iterations to localise, and the localisation only worked because
the kernel was instrumented to dump an intermediate: the MAC loop feeding the
multiply was provably correct, so the fault had to be in the two lines after it.

The natural formulation of a scaled GEMV walks straight into it. Quantised
weights are `code * scale` with the scale constant across a group, so the
efficient order is to accumulate `sum(code * x)` in fp32 and apply the scale once
per group — which needs exactly `fp32 vector * fp32 vector`.

**The fix, and it costs nothing.** Split the fp32 partial into two bf16 halves
and scale both:

```cpp
aie::vector<bfloat16, N> hi = part.to_vector<bfloat16>();
aie::vector<bfloat16, N> lo = aie::sub(part, hi).to_vector<bfloat16>();
acc = aie::mac(acc, hi, scale);
acc = aie::mac(acc, lo, scale);
```

Both products are bf16 x bf16, which is native, and 8 + 8 mantissa bits land
exactly in the fp32 accumulator. Measured against a host fp32 reference over
248320 outputs: **cosine 1.00000000, max relative error 3.0e-06**. Rounding the
partial to bf16 once instead — the obvious cheaper fix — measured 1.7e-03, which
is the 2^-9 floor and is *not* improved by summing more terms: the error and the
sum both grow as sqrt(N).

### IRON compiles the kernel source once per ExternalFunction

Four entry points in one `.cc` gives four objects that each define all four
symbols; `ld.lld` fails on duplicates. One header plus one translation unit per
entry point. Worth knowing before designing a kernel family around compile-time
specialisation.

### There are 16 shim MM2S channels on the whole device

A design that gives every core its own stream for something *shared* — here the
activation vector, identical for all cores — runs out at 16 cores:
`no ShimNOCTile has sufficient DMA capacity`. Broadcasting from one ObjectFifo
with N `.cons()` handles fixes it and is the right design anyway. Past 8 weight
streams, further scaling needs shim -> memtile -> cores rather than more shim
channels.

The lm_head GEMV runs 8 cores at **37.7 GB/s** — 83% of the 45.5 GB/s roof
[`0130`](../../tasks/0130-t45-traced-roof/TASK.md) measured — and the placer put
all 8 on **two columns**, 4 rows each. That is worth a second look rather than a
victory lap: if two columns' shim channels already reach 83% of a roof that was
measured device-wide, either the roof is not column-limited in the way the
memcpy microbenchmark suggests, or this design is being served from somewhere
other than where it looks. Filed nothing yet; it wants a measurement, not a
guess.

---

## Addendum, 2026-08-29 — dispatch cost, and why a hybrid split cannot win

Measured in `../LLMNpuTest/designs/lm_head/dispatch_probe.py`: sweep one design
over ten sizes, fit `t = fixed + bytes / bandwidth`, read the intercept.

```
npu timebase   t = 177.9 us + bytes / 39.3 GB/s      (R2 0.9995)
hw_context switch                     563 us         (median of 40, alternating
                                                      two contexts vs repeating one)
IRON Python dispatch, on top          ~465 us        (e2e intercept minus npu)
```

This is the number that decides an architecture, and it is worth stating next to
the work it is being spent on. For Qwen3.5-0.8B:

| | bytes | matmuls | work per dispatch |
|---|---:|---:|---:|
| `lm_head` | 270.2 MB | 1 | **7166 us** |
| all 24 layers | 328.4 MB | 186 | **47 us** |

**Issuing a layer matmul costs 3x what running it costs**, and 12x that again if
the design has to switch. A hybrid that keeps norms, RoPE, attention and the
recurrence on the host must return to the host between matmul groups; the most
that can be fused is about four groups per layer, so ~97 dispatches per token.
At 563 us that is **54.6 ms/token of switching against 8.7 ms of work.**

Two `hw_context`s coexist on one device without complaint — that was worth
checking and it is not the constraint. Alternating between them is.

So the conclusion is architectural, not about kernel quality: either the whole
layer goes on the array in one dispatch, or every shape shares one xclbin with
per-shape instruction streams in a single context. This project already built the
second of those ([`0032`](../../tasks/0032-m7-one-xclbin-production/TASK.md),
`tools/export_gemm_rtp.py`: 16 streams, one hw_context, zero design switches per
encode) — and [`0024`](../../tasks/0024-m7-dispatch-cost-anatomy/TASK.md) priced the switch
it avoids, at ~55 us + ~286 us per column. The 563 us measured here on a 2-column
design is consistent with that. The finding
here is that for *LLM decode* the same lever is not an optimisation but a
precondition, because the per-op work is an order of magnitude smaller than an
encoder's.

FastFlowLM's shape is the same answer from the other direction: a monolithic
`layer.xclbin` behind a `gen_layer_seq` entry point, i.e. the whole layer, one
dispatch. TileFuse reports the matching negative result — decode stays
iGPU-dominated on XDNA2 because millisecond-scale dispatch swamps microsecond-scale
GEMV.

**What it was worth anyway.** `lm_head` is the one projection large enough to pay
for its own dispatch, and moving only that one operation took the reference chat
from **4.42 to 5.62 tok/s** (three runs each, same harness) — 1.27x, with the array
doing 34% of the arithmetic in 5% of the wall clock.

### Two host-side traps, both of which cost a debugging cycle

- **XRT teardown order.** Python's collector frees the device before its buffer
  objects; the process dies with an access violation *after* every result has been
  computed and printed. Release explicitly in dependency order.
- **`nn.Module` already has a `cpu()` method.** Storing a fallback submodule as
  `self.cpu` shadows it, and the failure surfaces as a `TypeError` from torch's
  call machinery rather than anywhere near the assignment.
