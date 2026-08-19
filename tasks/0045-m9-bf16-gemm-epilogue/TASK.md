# 0045 — M9: let C leave the array as bf16

**Goal.** Halve the largest single host cost in the encode by narrowing the GEMM
result to bf16 **on the array**, after the fp32 K-reduction, instead of shipping
fp32 to the host and narrowing there.

Comes out of [`0044`](../0044-m9-optimisation-sweep/TASK.md) Part 2, which
measured on an idle machine:

| | ms | share of a 185 ms MiniLM encode |
|---|---:|---:|
| `read out + bias` (device→host C) | 34.7 | **18.8%** |
| `sync from device` | 5.5 | 3.0% |
| host gelu + softmax + layernorm | 13.8 | 7.5% |

`read out + bias` is **679 MB of fp32 C per encode** (37.7 + 12.6 + 50.3 +
12.6 MB per layer × 6 layers) read out of write-combined XRT memory at
19.6 GB/s. Halving the bytes is the whole idea.

---

## Why this and not the other levers

[`0044`](../0044-m9-optimisation-sweep/TASK.md) listed seven. This one was
picked because it is the only one that needs **no new dataflow**: the epilogue
mechanism already exists ([`0030`](../0030-m7-expert-review-tests/TASK.md) built
`epilogue="gelu"` as a second kernel in the GEMM worker), the `.npue` does not
change, B is not repacked, and the tile geometry is untouched.

The bigger lever — eltwise back on the array, attacking the whole 33% — is
milestone-scale and needs the one-operator-per-core design that 0032's program
memory wall dictates. This is the part of the prize that can be taken now.

---

## What is actually being changed

Today, per GEMM dispatch:

```
core:  zero(C_l1 fp32) → K/k × matmul(A,B → C_l1 fp32) → release
DMA:   C_l1 fp32 → memtile → shim → host        (4 bytes/element)
host:  streaming-load fp32, add bias, hand fp32 to eltwise
```

After:

```
core:  zero(acc fp32) → K/k × matmul(A,B → acc fp32) → narrow(acc → C_l1 bf16)
DMA:   C_l1 bf16 → memtile → shim → host        (2 bytes/element)
host:  streaming-load bf16, widen, add bias, hand fp32 to eltwise
```

**`CLAUDE.md` trap 2 is respected and this is the crux.** Trap 2 forbids
`output_dtype=bf16` on the matmul kernel because that re-rounds the accumulator
at **every K step** (7.4e-3 against 1.21e-07). Here the accumulation stays fp32
end to end and the narrowing happens **once, after the full K reduction** — the
same single rounding the host performs today, moved upstream of the eltwise.

### L1 budget — unchanged, which is what makes this cheap

The accumulator is a core-local `Buffer` and needs **no double buffering** (it
is filled and drained inside one iteration), while the C fifo it feeds halves.
Those exactly cancel:

| | today | after |
|---|---:|---:|
| A (bf16, ×2) | 2·m·k·2 | 2·m·k·2 |
| B (bf16, ×2) | 2·k·n·2 | 2·k·n·2 |
| C fifo | 2·m·n·**4** | 2·m·n·**2** |
| acc (fp32, ×1) | — | 1·m·n·**4** |

MiniLM `(64,64,48)`: **53,248 B both ways** — the exact figure `CLAUDE.md`
records for the current design. bge-large `(64,64,32)`: **40,960 B both ways**.
No geometry has to move.

---

## The two things that could make this not worth it

Written down before measuring, so the result can contradict them.

**1. Accuracy — one extra rounding, in a new place.** Today the host eltwise
sees **fp32** C and rounds to bf16 only after LayerNorm/GELU/softmax have run.
After this change eltwise sees **bf16** C. That is one additional rounding per
GEMM, on the input side of every elementwise op. Production `1-cos` is
**1.086e-05** (MiniLM) and **8.348e-06** (bge-small), against an MTEB gate of
±0.5 points that measured **+0.04**. The budget is large but the change is real
and must be measured, not assumed.

*Mitigation available:* the narrowing kernel is new, so it gets
`aie::set_rounding(conv_even)` from the first line —
[`0044`](../0044-m9-optimisation-sweep/TASK.md) Part 3 measured the default
`floor` as a systematic bias worth 1.3–1.7× on every eltwise kernel. This is the
first kernel in the project written with the correct mode.

**2. Speed — the narrow is serial work on the core's critical path.** It runs
after the K loop, so it cannot overlap with the matmul. For MiniLM's 3,072-element
tile that is roughly 400 cycles against ~8,400 for a 6-block K reduction — call
it **5% on the GEMM**, which is 40% of the encode, so **~2% back** against the
~10% saved. Expected net positive, but the GEMM cost must be read off `--bench`
rather than assumed.

If either of these lands badly the change is revertible: it is a new flag, and
the fp32 path stays.

---

## Plan

1. `narrow_f32_bf16.cc` — the epilogue kernel. fp32 in-place tile → bf16 tile,
   `set_rounding(conv_even)`.
2. `gemm_pretiled.py` — `c_bf16=True`: fp32 accumulator `Buffer`, bf16 C fifos,
   narrow before release. Must work on the **`rtp=True`** path, which is what
   production runs.
3. `export_gemm_rtp.py` — `--c-bf16`; C tensor bf16; `buffers[2]` halves;
   record the C dtype in `design.json` so the runtime cannot guess wrong.
4. `runtime/` — read bf16 C, widen in the same AVX2 streaming loop that already
   adds bias.
5. Validate: `export_validation.py` / goldens for numerics, `verify_embed_e2e.py`
   for the product, `--bench` for the speed, all three models.

---

## Progress log

*(updated as it goes)*

- **Step 0 — plan written, budget checked on paper.** L1 is unchanged at both
  production geometries, which is why this was picked over the alternatives.
- **Step 1 — `narrow_f32_bf16.cc` written and verified statically.** Compiled
  standalone with Peano and disassembled before wiring anything into a design,
  because a kernel that compiles is not the same as a kernel that is cheap.

### The idiom this project documents is the expensive one

`aie::to_vector<T>()` is defined on **accumulators**, not vectors — the direct
form is rejected outright:

```
error: no matching function for call to 'to_vector'
  note: candidate template ignored: constraints not satisfied
        [with TR = bfloat16, T = aie::vector<float, 16>]
```

`gelu_poly.cc` solves that by multiplying by 1.0f to get an accumulator, and
says so in a comment: *"vector::to_vector<T>() does not exist; the AIE API
exposes it on accumulators, so multiply by 1.0, which is exact."* Exact, yes.
**Free, no** — fp32 multiply on aie2p is *emulated*, which the indexed papers
already told us ([`0028`](../0028-research-index-nine-new-papers/TASK.md):
"fp32 is emulated and compute-bound 8:1"). Measured on the two forms, 64
elements, `-O2`, `llvm-objdump`:

| form | `vmul.f`/`vadd.f` | total instrs | narrowing |
|---|---:|---:|---|
| `aie::mul(v, 1.0f).to_vector<bfloat16>()` | **34** | 59 | via the emulated multiply |
| `accum.from_vector(v); acc.to_vector<bfloat16>()` | **0** | 17 | **`vst.conv.bf16.fp32` — free in the store** |

The hardware has a store-with-conversion instruction. Asking for it through a
multiply makes the compiler emulate an fp32 multiply it did not need.

Final kernel, disassembled:

```
emulated fp32 (vmul.f/vadd.f): 0
vst.conv.bf16.fp32:            6
crrnd set (set_rounding):      1      <- mov crrnd, #0xc
zero-overhead loop body:       2 lines
```

**Two lines in the ZOL is the floor** — one load, one store-with-conversion.
At 3,072 elements that is ~192 cycles against ~8,400 for a 6-block K reduction:
**~2.3% on the GEMM**, against the ~5% the plan budgeted. Object is 1,792 bytes,
which matters because a core has 16 KB of program memory and must still hold
the matmul ([`0032`](../0032-m7-one-xclbin-production/TASK.md)).

`set_rounding(conv_even)` compiles to a single `mov crrnd, #0xc` — once per
call, as intended, not once per element.

**Worth following up separately, not here:** `gelu_poly.cc` uses the
multiply-by-1.0f idiom in two places, including its one store. If the narrowing
one is replaceable there too it is free emulated-op savings in a kernel
[`0026`](../0026-m7-closing-on-cpu/TASK.md) called "at the machine's fp32 vector
limit" — a limit that was measured with 34 avoidable ops per 64 elements in it.
- **Step 2 — `gemm_pretiled.py` gained `c_bf16=True`, built and validated on
  hardware in isolation.**

The split that makes it safe: `dtype_out` stays the **accumulator** type and is
asserted fp32; a new `dtype_c` is what the C fifos and the L3 buffer carry. The
matmul kernel never sees bf16. Three guards fail loudly rather than silently
building the wrong thing — accumulator must be fp32, `rtp` must be on (the only
worker wired for it), and `epilogue="gelu"` is rejected because both own the
post-K step.

```
core_fn:  elem_out = out_c.acquire(1)      # bf16 slot, acquired where it always was
          zero(acc)                        # core-local fp32 Buffer
          K/k x  matmul(a, b, acc)         # fp32 accumulation, untouched
          narrow(acc, elem_out)            # ONE round, conv_even
          out_c.release(1)
```

### The probe failed on both arms first, which is the point of having two

The first run gave `rel_fro 1.384` — uncorrelated. The **control arm**
(`--cbf16 0`, the shipping fp32 path through the same harness) gave *the same*
1.384, which says the probe is wrong and the design is not implicated. It was:
I fed B through `tile_b(B_float32, k, n)` without the `(s, t)` sub-tile order
and without the `uint16` view, while the design is built `inner_st=True` and
expects that order baked in. The shipped convention is
`tile_b(B.view(np.uint16), k, n, s, t, order="k,n")` plus an `untile_b`
round-trip assert, and the probe now does exactly that.

Twenty minutes, and no wrong conclusion, because the control was in the probe
from the first run rather than added after a surprise.

### Result — the narrowing does exactly one correct rounding

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
python <scratchpad>\probe_cbf16.py --cbf16 0     # control
python <scratchpad>\probe_cbf16.py --cbf16 1     # new path
```

512x384x1152, tile (64,64,48), 8 columns, `mac_dims (4,8,8)`:

| | fp32 C (control) | **bf16 C** |
|---|---:|---:|
| rel_fro vs the model that arm claims | 2.015e-07 | **2.792e-05** |
| rel_fro vs a pure fp32 matmul | 2.015e-07 | **1.502e-03** |
| bit-exact vs the model | 15.86% | **589,807 / 589,824 = 100.00%** |

**Only 17 elements of 589,824 differ from "accumulate in fp32, round once at the
end"** — and those are expected: the array reduces K in blocks across cores, so
its fp32 sum differs from numpy's in the last ulp and occasionally lands on the
other side of a rounding boundary.

The 1.502e-03 against pure fp32 is **not error, it is the bf16 format floor** —
rounding a value to bf16 costs ~2⁻⁹ relative, and that is what we measure. The
number that would indicate a real problem is **7.4e-3**, which is what
`CLAUDE.md` trap 2 records for re-rounding at every K step. We are 5x below it,
which is the positive evidence that the accumulator really is fp32 all the way
through the reduction.
- **Step 3 — exporter (`--c-bf16`), and a cache collision that this change
  created.**

The C element type is now part of the design's cache identity, and it had to
become part of it in a stronger way than expected.

### `--c-bf16` made two GEMM shapes indistinguishable in the cache

`markers_for()` identified a cached design by three loose strings —
`memref<M*K xbf16>`, `memref<K*N xbf16>`, `memref<M*N xf32>` — asking only
whether each appeared *somewhere* in the module. That worked because the `xf32`
suffix on C kept the sets apart. **With bf16 C it does not:**

```
ffn_up   [8192,  384, 1536]  ->  M*K=3145728   K*N=589824   M*N=12582912
ffn_down [8192, 1536,  384]  ->  M*K=12582912  K*N=589824   M*N=3145728
```

Same three numbers, and now all three are `bf16`. The two shapes became
identical to the matcher, and `purge()` for `ffn_down` **deleted `ffn_up`'s
build** in the middle of the export:

```
FileNotFoundError: ...\.npu\cache\2005d3a796739557c0f6f167\final.xclbin
```

**That it crashed is luck.** `purge` runs before each build, so the victim
happened to be the one already finished; had `ffn_up` been second it would have
found `ffn_down`'s directory, passed every identity check (the xclbins ARE
identical — that is the architecture), and shipped **`ffn_down`'s instruction
stream under `ffn_up`'s name**. That is the fifth fail-open
([`0030`](../0030-m7-expert-review-tests/TASK.md)) with a new cause.

Fixed by matching the **ordered** signature instead, which binds each size to an
argument position:

```
aie.runtime_sequence(%arg0: memref<3145728xbf16>, %arg1: memref<442368xbf16>,
                     %arg2: memref<9437184xbf16>)
```

A and B and C cannot trade places whatever their element types are. This is
strictly better than the old form for the fp32 path too.

`design.json` gains `"c_dtype"`, and the runtime **reads** it —
`DesignInfo::c_elem_bytes`, an unknown spelling throws. A missing field means an
export predating it, and every one of those is fp32, so 4 is the correct reading
of silence rather than a default hiding a gap. `--probe-rtp` reads C as fp32
directly and now **refuses** on a bf16 artifact rather than printing plausible
wrong sums.

Export is clean and the one-xclbin architecture survives:

```
identity qkv@b128 vs attn_out@b128  69 differing bytes  OK
identity qkv@b128 vs ffn_up@b128    66 differing bytes  OK
identity qkv@b128 vs ffn_down@b128  71 differing bytes  OK
C buffer 50,331,648 -> 25,165,824 bytes (2.0x)
```

- **Step 4 — runtime reads bf16 C.** One extra branch in `Encoder::gemm`, both
  arms using `_mm256_stream_load_si256` against the write-combined bo. The bf16
  arm gets **16 elements per 32-byte streaming load against 8** — same
  instruction count, half the traffic — then widens to fp32 and adds bias
  exactly as before.

---

## Results — MiniLM, batch 128, idle array

`--bench 5`, the contention gate reporting `exclusive` on both runs.

| | fp32 C | **bf16 C** | |
|---|---:|---:|---|
| **throughput** | 693.5 seq/s | **727.2 seq/s** | **+4.9%** |
| wall | 184.58 ms | 176.03 ms | −8.6 ms |
| `read out + bias` | 34.57 ms (18.7%) | **27.75 ms (15.8%)** | −6.8 ms |
| `sync from device` | 5.73 ms | **4.63 ms** | −1.1 ms |
| **`wait` (hardware)** | 3,028 µs | **2,999 µs** | **unchanged** |

**The epilogue is free.** Per-dispatch hardware time did not move — 2,999
against 3,028 µs, which is inside run-to-run noise and if anything lower. The
plan budgeted ~5% and the static analysis said ~2.3%; the answer is that it
disappears into the DMA shadow entirely.

### Accuracy — one extra rounding, and it costs what one rounding costs

| | fp32 C | **bf16 C** | tolerance |
|---|---:|---:|---:|
| worst `1-cos` vs HuggingFace | 1.086e-05 | **1.498e-05** | 2e-03 |
| `rel_fro` vs HF golden | 4.473e-03 | 4.994e-03 | — |
| e2e product worst `1-cos` | 2.644e-05 | **3.918e-05** | 2e-03 |
| e2e pairwise similarity, mean | 3.493e-04 | 3.618e-04 | — |
| **top-10 neighbour overlap** | 1.0000 | **1.0000** | — |

1.38× on the golden `1-cos` and 1.48× end to end — the size of one bf16
rounding, arriving exactly where the plan said it would (the eltwise ops now see
bf16 input instead of fp32). Both arms PASS, **133× inside the tolerance**, and
**the retrieval-relevant metric does not move at all**: neighbour overlap stays
1.0000.

### The saving is real but it is half of what was projected, and here is why

The plan said ~10% and measured +4.9%. `read out + bias` fell 34.57 → 27.75 ms,
a 20% cut, not the 50% that halving the bytes suggests.

**Because only the read halved.** The loop reads C out of write-combined device
memory and writes fp32 into a normal-memory `std::vector`. Per encode:

| | fp32 C | bf16 C |
|---|---:|---:|
| read from WC bo | 679 MB | **340 MB** |
| write to host `out` | 679 MB | 679 MB |

The write side is untouched, because everything downstream — attention,
LayerNorm, GELU, the residuals — consumes fp32. Halving *that* means keeping
host activations in bf16, which is a different and much larger change.

So the honest accounting is that this took the device→host half of the transport
and left the host-side half alone. **+4.9% for one flag, at 1.38× on a number
133× inside tolerance, with the GEMM unchanged.**

---

## All three models

Every arm on an idle array with the contention gate reporting `exclusive`.
Accuracy is the runtime's own golden check against HuggingFace; tolerance
2e-03.

| model | fp32 C | **bf16 C** | gain | `read out + bias` | `wait`/dispatch |
|---|---:|---:|---:|---|---|
| MiniLM-L6 | 693.5 | **727.2** | **+4.9%** | 34.6 → 27.8 ms (−20%) | 3,028 → 2,999 µs |
| bge-small | 347.9 | **365.3** | **+5.0%** | 70.2 → 56.6 ms (−19%) | 3,031 → 3,008 µs |
| bge-large | 42.6 | **43.9** | **+3.1%** | 388.5 → 310.1 ms (−20%) | 19,003 → 18,984 µs |

| model | `1-cos` fp32 | `1-cos` **bf16** | ratio | headroom to gate |
|---|---:|---:|---:|---:|
| MiniLM-L6 | 1.086e-05 | 1.498e-05 | 1.38× | **133×** |
| bge-small | 8.348e-06 | 1.232e-05 | 1.48× | 162× |
| bge-large | 8.432e-06 | 1.281e-05 | 1.52× | 156× |

Three things this table says that one model could not:

1. **The readback saving is a flat ~20%, everywhere.** Same mechanism, same
   proportion, at three widths and three depths.
2. **The GEMM is untouched at every geometry** — the largest deviation is 29 µs
   on 3,028, and bge-large's 19 ms dispatches move by 19 µs. The narrowing
   epilogue really is free, not merely cheap at one shape.
3. **The throughput gain shrinks as the array gets busier.** bge-large spends
   **60.8%** of its encode in `wait` against MiniLM's 39.4%, so the host-side
   share this lever attacks is smaller there. It is a *host* optimisation and it
   pays best where the array is least loaded — the opposite of where the
   fusion levers pay.

`top-10 neighbour overlap` stays **1.0000** on the end-to-end product check for
both arms.

---

## Same artifact-clobbering bug, third occurrence

`verify_embed_e2e.py` wrote one fixed filename regardless of `--model` and
`--artifacts`, so running the two arms in sequence left only the second one's
JSON. Identical to the two harnesses [`0044`](../0044-m9-optimisation-sweep/TASK.md)
fixed hours earlier. Now keyed by model and artifact set; the baseline was
restored from git and re-run.

Three occurrences of one pattern is not three accidents. **Any script that
writes a result artifact to a constant path is an A/B waiting to erase its own
control**, and this repo had at least three.

---

## Verdict, and what is deliberately NOT done

**Shipped as a flag, not as the default.** `--c-bf16` on the exporter,
`c_dtype` in `design.json`, and the runtime reads it. The fp32 path is
untouched and remains what `runtime/artifacts*` default to.

Making it the default is an **accuracy** decision, and this project has a
standing rule for those: [`0035`](../0035-m8-mteb-gate/TASK.md) settled the
bf16-vs-bfp16 question on **MTEB**, not on `1-cos`. The numbers here are
encouraging — 1.4–1.5× on a quantity already 133–162× inside tolerance, with
neighbour overlap unmoved — but "encouraging" is what the bfp16 path looked like
before MTEB was run, and the correct next step is
`experiments/m8-npu-vs-cpu` on the bf16-C artifacts. Until that exists, the
default stays fp32.

**Not attempted here:** the host-side half of the transport. C is read at half
the bytes and still *written* as 679 MB of fp32, because attention, LayerNorm,
GELU and the residuals all consume fp32. Halving that means bf16 host
activations — a different and much larger change, and the honest reason this
delivered +4.9% instead of the ~10% the plan projected.

---

## Regression: the fp32 default path, rebuilt from scratch

`gemm_pretiled.py`, `export_gemm_rtp.py` and the runtime all changed, so the
shipping path was re-exported end to end rather than assumed intact — the
cache-marker rewrite in particular touches every fp32 build.

```powershell
python tools\export_gemm_rtp.py --batch 128 --cols 8 --out runtime\artifacts_regress_f32
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_regress_f32 --threads 24
```

| | recorded | rebuilt |
|---|---|---|
| `1-cos` vs HuggingFace | 1.086e-05 | **1.086e-05** |
| `read out + bias` | 18.7% | 18.7% |
| `c_dtype` / C buffer | f32 / 50,331,648 | f32 / 50,331,648 |
| one-xclbin identity | OK | 68–70 differing bytes, OK |

Throughput 680.6 seq/s, inside the 680–694 spread of the day's other fp32 runs.
Nothing regressed.

---

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# designs
python tools\export_gemm_rtp.py --c-bf16 --batch 128 --cols 8 `
    --out runtime\artifacts_cbf16
python tools\export_gemm_rtp.py --c-bf16 --hidden 1024 -n 32 --batch 128 --cols 8 `
    --out runtime\artifacts_large_cbf16

# speed + accuracy, per model, both arms
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_cbf16 `
    --bench 5 --threads 24
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_cbf16 `
    --threads 24                      # golden 1-cos
& .\.venv-ref\Scripts\python.exe tools\verify_embed_e2e.py --artifacts artifacts_cbf16
```

## Files

| file | why |
|---|---|
| `experiments/m5-eltwise/kernels/narrow_f32_bf16.cc` | **new** — the epilogue, 2-line ZOL, `conv_even` |
| `experiments/m5-pretiled-gemm/gemm_pretiled.py` | `c_bf16=`: fp32 accumulator `Buffer`, bf16 C fifos, three guards |
| `tools/export_gemm_rtp.py` | `--c-bf16`, `c_dtype` in the manifest, **ordered** cache markers |
| `runtime/include/npu_device.hpp`, `runtime/src/npu_device.cpp` | `c_elem_bytes`, read from the manifest, unknown spelling throws |
| `runtime/src/main.cpp` | bf16 readback arm; `--probe-rtp` refuses a bf16 artifact |
| `tools/verify_embed_e2e.py` | artifact keyed by model + artifact set |

## Next

1. **MTEB on the bf16-C artifacts** — the gate for making it the default,
   per [`0035`](../0035-m8-mteb-gate/TASK.md). Everything else is measured.
2. The host-side half: bf16 activations across attention/LN/GELU would halve the
   679 MB *write* this task left alone. Much larger, and it overlaps with the
   eltwise-on-array question [`0044`](../0044-m9-optimisation-sweep/TASK.md)
   Part 4 reopened.
3. `gelu_poly.cc` still narrows through the emulated multiply-by-1.0f. Same
   `accum::from_vector` fix, free, measured here at 34 avoidable ops per 64
   elements.

---

## Decision, 2026-08-19

**The datapath stays bf16 in, fp32 out.** `--c-bf16` is kept, measured and
documented, but the default does not change and MTEB is not being run on it for
now. Recorded in `CLAUDE.md` so it is a standing contract rather than an
unexamined default.

Consequence for what comes next: `read out + bias` stays at ~19% of the encode
and the 33% transport share stands. The remaining levers therefore have to
attack either **`wait`** (39% of a MiniLM encode, 61% of a bge-large one) or the
host round trip as a whole — not the C element type.
