# 0086 — attention at 512 bits: two negatives, and a microbenchmark that lied

- **Date** 2026-08-23
- **Status** done — both 512-bit paths built and verified, neither pays; the
  reason reframes what VNNI would be for

## Goal

**User, item 2 of three:** *"VNNI på attention — den eneste vertsspaken igjen på
den ene kjernen som er compute-bundet. `vpdpbusd` gjør fire int8-MAC per lane;
ingen kompilator finner den fra fp32-kilde."*

The premise — attention is the compute-bound host kernel — came from
[`0084`](../0084-m13-host-isa-and-repeats/TASK.md): 497 GFLOP/s against a
~568 GFLOP/s AVX2 FMA peak, **88% of peak**. This task tests it, and the premise
does not survive.

---

## 1. Attention is two different problems, and they are 50/50

`t_attn` covered both halves. Split:

| model | QK^T | A·V | total |
|---|---:|---:|---:|
| `bge-large-en-v1.5.int8n64` | 98.7 ms | 105.2 ms | 203.8 ms |
| `bge-base-en-v1.5.int8` | 35.8 | 34.5 | 70.3 |
| `all-MiniLM-L6-v2.int8` | 10.5 | 9.3 | 19.8 |

They differ in kind. **QK^T** holds Q in registers across the j loop and ends
each `(i, j)` in a horizontal reduction. **A·V** accumulates per *output
element* and never reduces horizontally — it streams `scores` and `V`.

*(The first version of this counter summed to 1.5× its own parent, because
`reset_timers()` zeroes `t_attn` and I had not added the two new ones — so they
carried the warm-up encode too. A counter that is not reset measures a
different window from everything printed beside it.)*

---

## 2. A·V at 512 bits: bit-identical, and worth nothing

Done first *because* it can be bit-identical: each accumulator lane owns one
output element and sums over j in the same order at any width, so widening
changes how many lanes ride in a register and never which numbers are added.
Verified — MiniLM bf16 1.086e-05 and bge-large int8 2.968e-03 both reproduce
exactly.

**Attention 207.40 → 207.83 ms.** Nothing.

That negative closes a loose end from 0084. It guessed *"Zen 5 mobile
double-pumps a 256-bit datapath"* to explain why `/arch:AVX512` did nothing,
then retracted the guess because the flag cannot reach hand-written `_mm256_*`
intrinsics — so the flag experiment proved nothing about the hardware. **This
one is hand-written at 512 bits and still gains nothing**, which is evidence for
the retracted hypothesis rather than a repeat of the flag test.

---

## 3. QK^T at 512 bits: 1.33× in a microbenchmark, SLOWER in situ

The microbenchmark, bge-large's geometry, 20,000 reps:

| inner loop | time | vs today |
|---|---:|---:|
| AVX2 fp32 (ships) | 2.1 ms | 1.00× |
| **AVX-512 fp32** | 1.6 ms | **1.33×** |
| AVX-512 VNNI int8 | 1.0 ms | **2.16×** |

The 1.33× is not the width — it is `_mm512_reduce_add_ps` replacing `hsum256`'s
four-instruction shuffle chain, once per `(i, j)` pair. A·V has no reduction and
gained nothing, which is the same story from the other side.

Built it. **`qk` went 98.7 → 106.6 ms — slower.**

**The microbenchmark measured the wrong thing, and the way it was wrong is the
lesson.** It held `q` and `k` in two small arrays and reused them 20,000 times,
so everything was L1-resident and the loop was pure arithmetic. The real kernel
streams K out of `qkvbuf` with `3 · hidden` floats between consecutive rows. An
inner-loop microbenchmark with hot data does not predict a kernel that streams,
and this one was off by enough to invert the sign of the result.

Accuracy moved as the reassociation requires — summing `head_dim` floats as
lanes of 16 associates differently from lanes of 8, so this half **cannot** be
bit-identical and was gated on `1-cos` instead:

| model | AVX2 | 512-bit QK^T |
|---|---:|---:|
| MiniLM bf16 | 1.086e-05 | 1.093e-05 |
| bge-large bf16 | 8.432e-06 | 9.130e-06 |
| bge-large int8 | 2.968e-03 | **2.636e-03** |
| nomic int8 | 1.098e-03 | 1.189e-03 |

bge-large int8 came out *better*. That is the reassociation landing luckier on
one model, not a systematic improvement, and it is recorded as such.

---

## 4. What this does to the VNNI premise

0084's "88% of AVX2 FMA peak" said attention is compute-bound. **Two independent
widenings say it is not**: A·V gains nothing and QK^T loses. Both halves are
**memory-bound in situ** — the FLOP/s figure is high because the loop issues
FMAs while waiting, not because the FMAs are the constraint.

**So VNNI's value on attention is not the arithmetic.** `vpdpbusd` does four
MACs per 32-bit lane, and this task says that is not what is missing. What int8
Q/K/V would actually buy is **four times fewer bytes**, which is a *traffic*
lever — the same one that made the int8 GEMM and both host fusions pay.

Sizing it against this task's own measurements, per bge-large layer:

| | fp32 today | int8 Q/K/V |
|---|---:|---:|
| QK^T reads Q + K | 67.1 MB | 16.8 MB |
| A·V reads scores + V | 67.1 MB | 41.9 MB |

≈2× less traffic for QK^T and ≈1.33× for A·V, so **~6% of the encode** if the
quantisation itself is free. It is not free as a separate pass — but the qkv
GEMM's dequantiser already touches Q and K, so it can emit them int8 the way
[`0082`](../0082-m13-fused-ffn-epilogue/TASK.md) made LayerNorm emit `ffn_up`'s
operand. For arch 0 that is clean; arch 1 and 2 rotate Q and K with RoPE *after*
the GEMM, so the quantisation has to follow the rotation.

**Not built.** The premise this task was given is refuted, the replacement
premise is measured, and the design that follows from it is a different piece of
work from the one asked for.

---

## 5. Also fixed

`runtime/CMakeLists.txt` had `NPUE_HOST_ARCH` defaulting to **AVX512** while
0084's own text said *"the shipped default stays AVX2"* — a default disagreeing
with its own justification, introduced by me the same day. Now AVX2, which is
also what measures faster. Both 512-bit paths stay in the source behind
`__AVX512F__`: they are the evidence, and they are compiled out because they
lose.

Verified the shipped AVX2 build reproduces [`0085`](../0085-m13-release-sweep/TASK.md):
`1-cos` exactly (2.968e-03, 1.161e-03), throughput within 1.4%.

---

## 6. Commands

```powershell
cmake -S runtime -B runtime\build -DNPUE_HOST_ARCH=AVX512   # to re-test
cmake --build runtime\build --config Release

.\runtime\build\npuembed.exe . --model bge-large-en-v1.5.int8n64 `
    --artifacts artifacts_int8c_large_n64 --threads 24 --bench 2
#   -> host attention ... (qk 98.7  av 105.2)
```
