# 0095 — T6 (`aie_stream=` channel freedom) and T10 (`aie::exp2` vs `exp2_poly`), both closed

- **Date** 2026-08-23
- **Milestone** research
- **Status** done — **T6 ANSWERED** (frees a channel, but not the one B-reuse needs) · **T10 ANSWERED** (`exp2_poly` wins on cost and accuracy; confirms the standing production choice)

## Goal

Two small, independent open threads, run concurrently with other agents also on
the NPU — kept to compile-time and static evidence per instructions (compiler
messages, `input_with_addresses.mlir` channel counts, `llvm-objdump`
disassembly), no wall-clock timing.

- **T6** — does `aie_stream=(end, port)` (note 0007 §1.6) free a DMA channel,
  the way 0047 showed cascade does?
- **T10** — `aie::exp2<bfloat16>` was never measured against our own
  `exp2_poly`, on cost and accuracy. Softmax uses `exp2_poly` today
  ([`0030`](../0030-m7-expert-review-tests/TASK.md) §5a); the comparison that
  justifies it was never actually run.

## Context

Read first, per the brief: [note 0007](../../research/notes/0007-unused-iron-surface.md)
§1.6, [`0046`](../0046-m9-b-reuse-asymmetric/TASK.md) (closed B-reuse on mem-tile
channel exhaustion), [`0047`](../0047-m9-cascade-channel-probe/TASK.md) (cascade
trades 3 mem-tile inputs for 3 outputs — the method this task copies), and for
T10: [`0020`](../0020-m5-layernorm-kernel/TASK.md), [`0021`](../0021-m5-softmax-and-full-model/TASK.md)
(the original "works standalone, corrupts composed" failure), and
[`0030`](../0030-m7-expert-review-tests/TASK.md) (where `exp2_poly` came back).

Reading 0030 turned up something the brief asked to watch for: **the
`docs/CURRENT_STATUS.md` "unresolved bug" is stale.** 0030 §5a already
diagnosed and fixed it. See "The never-diagnosed bug: it was diagnosed" below.

## What was done

### T6 — build a design once with `aie_stream`, once without, and diff the channels

`aie_stream` is a real, documented, buildable IRON feature — one line in
`aie/iron/dataflow/objectfifo.py`:

```python
aie_stream (tuple[int, int] | None, optional): Mark the fifo as a direct
    AIE-stream connection by stamping the ``aie_stream`` / ``aie_stream_port``
    attributes ``(end, port)`` on the underlying ``aie.objectfifo`` op. Use with
    kernels that emit on the wire via ``put_ms()`` instead of going through an L1
    buffer.
```

and `AIEOps.td` names what `end` means:

```
// aie_stream==0 means enable aie stream port on producer tile
// aie_stream==1 means enable aie stream port on consumer tile
// aie_stream==2 means enable aie stream ports on producer and consumer tiles
```

`programming_examples/ml/magika/group2.py` is the one shipped example that
uses it: a single compute tile (col=1, row=3) whose kernel calls `put_ms()` to
push its output straight onto the wire; the output `ObjectFifo` is built with
`aie_stream=(0, 0)` (`end=0` = producer = the compute tile).

**Compile-only, no hardware.** `run_design_cli`'s `--xclbin-path`/`--insts-path`
branch calls `.compile(...)` without ever calling the `@iron.jit` function, so
this needed no NPU time and produced `input_with_addresses.mlir` directly
(`<outdir>/stream.prj/`).

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\dev\mlir-aie\programming_examples\ml\magika
python group2.py --dev npu --xclbin-path <out>\stream.xclbin --insts-path <out>\stream.insts.bin
```

(`--dev npu`, not `npu2`: the shipped `group2.cc` kernel uses `mac_4x8_8x4`,
an aie2-only intrinsic — it does not compile for `aie2p` at all, confirmed by a
first attempt with `--dev npu2` that failed with `use of undeclared identifier
'mac_4x8_8x4'`. Channel topology does not depend on which AIE generation, so
this does not weaken the probe — see Problems hit.)

A byte-identical copy with `aie_stream=(0, 0)` deleted (`artifacts/t6_group2_nostream.py`)
is the control.

**Result — `tools/count_dma_channels.py` on both:**

| | mem tile in/6 | mem tile out/6 | core (1,3) in/2 | core (1,3) out/2 |
|---|---:|---:|---:|---:|
| WITH `aie_stream=(0,0)` | 1 | 1 | 1 | **0** |
| WITHOUT | 1 | 1 | 1 | **1** |

**Yes — it frees a channel.** The core's output channel goes from 1/2 to 0/2.
Full census: [`artifacts/t6_dma_channel_census.txt`](artifacts/t6_dma_channel_census.txt).

**The raw MLIR is even more direct than the census.** In the no-stream build,
core (1,3)'s `aie.mem` region has both a `dma_start(S2MM, ...)` (input) and a
`dma_start(MM2S, ...)` (output) block. In the `aie_stream` build, **the `MM2S`
block is entirely absent** — not present-and-empty, gone:

```
=== core (1,3) DMA region, WITH aie_stream ===
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3) ...
      ...
    ^bb3:  // pred: ^bb0
      aie.end
    }

=== core (1,3) DMA region, WITHOUT aie_stream ===
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3) ...
      ...
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6) ...
      ...
    }
```

Full diff: [`artifacts/t6_mlir_evidence.txt`](artifacts/t6_mlir_evidence.txt).

**But the channel is freed only on the marked (core) endpoint, not on the
other end of the link.** The shim's DMA allocation for the same fifo
(`aie.shim_dma_allocation @of_dout_L1L3_shim_alloc(..., S2MM, 0)`) is
byte-identical in both builds — same channel index, same everything. A shim
tile has no program counter; it cannot execute `get_ss()`/`put_ms()`, so it
cannot give up its DMA channel regardless of what the other end does. The same
evidence file shows this directly.

**Does the freed channel have to come from a core?** Tried marking the
*consumer* side of an input fifo (`aie_stream=(1, 0)`, `end=1`) while the
consumer's kernel still called ordinary `.acquire()`/`.release()`
(`artifacts/t6_probe_in_stream.py`, based on `exp2_probe.py`, compile-only on
`npu2`). It fails at compile time, immediately and unambiguously:

```
aie.mlir:14:14: error: 'aie.objectfifo.acquire' op cannot acquire from objectfifo stream port
        %1 = aie.objectfifo.acquire @ein(Consume, 1) : ...
```

Full log: [`artifacts/t6_consumer_side_rejection.txt`](artifacts/t6_consumer_side_rejection.txt).
So `aie_stream` is not a free flag on an existing design — the endpoint it
marks must have its kernel rewritten to `put_ms()`/`get_ss()`, and IRON refuses
the mismatch outright rather than doing something silently wrong (a good trait
of the verifier, and a real engineering cost of using the feature).

**What this means for B-reuse.** 0046's census found the production GEMM's
bottleneck is **mem-tile input** channels (`A(1) + B(1) + C(4 core rows) = 6/6`
on five of eight mem tiles), not core channels — 0047's census even showed
every core sits at 2/2 **input** and only 1/2 **output**, i.e. cores already
had *output* headroom before this task. `aie_stream` frees a channel
**only on a tile with an executing core** — a mem tile or shim tile is pure
DMA/switch fabric with no program counter, so it can never be the marked
endpoint, and the census evidence above (shim allocation unchanged) confirms
this by example. The exhausted resource in 0046/0047 is a mem-tile *input*.
`aie_stream` cannot touch it. **T6 does not reopen B-reuse** — not built, per
instructions, and the arithmetic above is why it would not have been worth
building.

### T10 — `aie::exp2` vs `exp2_poly`, cost and accuracy

**1. Cost, statically.** `softmax.cc`'s `softmax_impl<kUsePoly>` template holds
both forms as the only difference between two otherwise-identical functions —
same clamps, same two reductions, same normalize pass — so compiling it once
gives a true apples-to-apples pair in the same translation unit, same flags.
Compile-only build (`artifacts/build_softmax.py`, based on `softmax_kernel.py`'s
`sm_array`, `variant="lib"` — both symbols are emitted regardless of which one
the MLIR calls, since both live in the same `#ifndef NPUE_ELTWISE_IMPL_ONLY`
block):

```powershell
python build_softmax.py <out>   # sm_array.specialize(...).compile(xclbin_path=..., inst_path=...)
llvm-objdump.exe -t   <out>\sm.prj\softmax_bf16.o     # symbol/section sizes
llvm-objdump.exe -d   <out>\sm.prj\softmax_bf16.o     # full disassembly
```

Symbol table (`artifacts/t10_objdump_evidence.txt`):

| function | section size | static instructions |
|---|---:|---:|
| `softmax_impl<false>` (`aie::exp2`) | 0x610 = 1,552 B | 291 |
| `softmax_impl<true>` (`exp2_poly`) | 0xe70 = 3,696 B | 765 |

**`aie::exp2` is not a library call — it is a single native AIE2P hardware
instruction, inlined.** No `bl`/`call`/`jal` anywhere in the `aie::exp2` path
(checked explicitly), and there is exactly one `vexp2` in the whole function:

```
420:  vsel.32  x7, x7, x10, r16
41a:  vmov     bmhh4, x7
420:  vexp2    wl4, bmhh4
424:  vmul.f   dm0, x4, x2, r4
```

That one instruction covers the entire 64-wide row (`SM_COLS`) — the source
calls `aie::exp2<bfloat16>` once per 16-lane sub-vector inside a
compile-time-constant `SM_VECS=4` loop, and the disassembly shows that whole
inner loop fully unrolled (only 2 `jnz` in the function, both accounted for —
one is the `rows > 0` entry guard, the other is the outer per-row loop's own
back-edge; no branch anywhere inside a single row), yet only one `vexp2`
appears. The AIE2P `vexp2` instruction operates wide enough to absorb what the
source expresses as four separate 16-lane calls.

`exp2_poly` has no such instruction to reach for. Its cost is the full 7-step
Horner chain plus the IEEE exponent-field construction, and — unlike
`aie::exp2` — Peano evidently could not fuse its four 16-lane sub-vector
Horner chains into one wide op (`gelu_poly`'s `SM_EXP2_STEP`-style
interleaving macro exists precisely because it doesn't, per `softmax.cc`'s own
comment on `softmax_il4_impl`). Mnemonic histogram, arithmetic-instruction
subset only:

| | `aie::exp2` path | `exp2_poly` path |
|---|---:|---:|
| `vmul.f` + `vadd.f` + `vmsc.f` | 22+16+7 = 45 | 98+82+31 = **211** |
| `vsrs.2x` / `vups.2x` / `vups.4x` (extra fp32↔bf16 shuttling exp2_poly needs and exp2 does not) | 0 | 5+3+3 = 11 |

Full histograms and the vexp2 context:
[`artifacts/t10_objdump_evidence.txt`](artifacts/t10_objdump_evidence.txt),
raw disassembly: [`artifacts/t10_softmax_impl_aie_exp2.asm`](artifacts/t10_softmax_impl_aie_exp2.asm),
[`artifacts/t10_softmax_impl_exp2_poly.asm`](artifacts/t10_softmax_impl_exp2_poly.asm).

**2. Accuracy, in numpy.** `exp2_poly`'s coefficients (`EXP2_C0..C7`, from
`exp2_poly.h`) were reproduced bit-for-bit in `artifacts/t10_numpy_model.py`:
the same trunc/Horner/exponent-field recurrence, run twice — once in float64
throughout (isolates the polynomial's own approximation error from any bf16
rounding) and once through the kernel's real bf16 datapath (bf16-in → fp32
Horner → **one** bf16 store, matching `softmax.cc` exactly), under both the
AIE **default rounding mode `floor`** (CLAUDE.md trap 2b — never overridden in
this kernel) and `conv_even`. Domain: the dense `[-120, 0]` ramp
[`0021`](../0021-m5-softmax-and-full-model/TASK.md)'s own probe used, plus a
second heavy-tailed sample standing in for real pre-softmax `d·log2e` values
(most mass near 0, matching `docs/04-model`'s description of masked/clamped
attention scores).

```
=== dense ramp [-120,0] (N=100000) ===
  algorithm only (fp64 Horner)            : max 2.010e-08  rms 1.057e-08
  full bf16 datapath, floor (AIE default) : max 7.109e-03  rms 1.730e-03
  full bf16 datapath, conv_even           : max 3.702e-03  rms 9.193e-04
  floor/conv_even max-error ratio         : 1.920x

=== realistic pre-softmax d*log2e (N=100000) ===
  algorithm only (fp64 Horner)            : max 2.013e-08  rms 1.050e-08
  full bf16 datapath, floor (AIE default) : max 7.109e-03  rms 2.933e-03
  full bf16 datapath, conv_even           : max 3.702e-03  rms 1.370e-03
  floor/conv_even max-error ratio         : 1.920x
```

Full output: [`artifacts/t10_numpy_model_output.txt`](artifacts/t10_numpy_model_output.txt).

Three things fall out:

- **The polynomial itself is not the error.** Algorithm-only (fp64) error is
  2.0e-08 — five orders below the bf16 grid. Essentially all of `exp2_poly`'s
  measured error is the *single* bf16 store at the end, exactly as the header
  comment claims ("degree 7 gives 1.75e-07 ... four orders below the bf16
  grid").
- **Under the AIE's actual default (`floor`), this numpy model lands at
  7.109e-03 max relative error — closely reproducing
  [`0021`](../0021-m5-softmax-and-full-model/TASK.md)'s own hardware-measured
  6.7e-03** (`exp2_probe`, isolated ramp test, same `[-120,0]` domain). That
  agreement is the validation that this numpy model is measuring the same
  thing the silicon does, not a different formula that happens to also be
  small.
- **`conv_even` would be 1.92x better** (3.702e-03 vs 7.109e-03) — the same
  systematic-downward-bias signature note 0007 §3.1 already measured on
  GELU (1.73x), softmax-as-a-whole (1.29x) and LayerNorm (1.62x). This is a
  fifth, independent data point for the same mechanism, on the one kernel
  piece note 0007 explicitly flagged as "not yet re-fitted."

`aie::exp2` has no public formula to reproduce in numpy — it is a hardware
primitive, not source we have. Its accuracy against the identical class of
reference (exact fp64 `2**x`) is the number [`0021`](../0021-m5-softmax-and-full-model/TASK.md)
already measured on real hardware: **1.711e-02**, "against a CPU model of the
same [softmax] formula" — i.e. measured the same way this task's numpy numbers
were derived, just on hardware instead of in numpy because there is no source
to run in numpy.

**3. The decision.**

| | static cost (whole-function instrs) | accuracy vs fp64 (`[-120,0]`, floor) |
|---|---:|---:|
| `aie::exp2` (hardware) | 291 (1 `vexp2` + 44 other arithmetic) | **1.711e-02** (hardware-measured, 0021) |
| `exp2_poly` (software) | 765 (211 arithmetic + exponent construction) | **7.109e-03** (numpy, this task; 6.7e-03 measured, 0021) |

`exp2_poly` is **2.4x more accurate** than `aie::exp2` (4.6x under `conv_even`,
untested on hardware) at **2.6x the instruction count** of the exp2-bearing
kernel — not the whole design; passes 1 and 3 (the two reductions and the
normalize) are identical between the variants and untouched by this choice.
**Softmax should keep using `exp2_poly`** — which is exactly what
[`0030`](../0030-m7-expert-review-tests/TASK.md) already shipped. T10 does not
change production; it supplies the comparison that decision was missing, with
fp64-referenced numbers on both sides for the first time.

### The never-diagnosed bug: it was diagnosed

`docs/CURRENT_STATUS.md` §3 still read, going into this task:

> `exp2_poly` works standalone (6.7e-03) but corrupts when composed into
> softmax — row sums went to zero ... Never diagnosed.

That paragraph is 25 tasks stale. [`0030`](../0030-m7-expert-review-tests/TASK.md)
§5a tested exactly this ("the exp2_poly bug is the worker stack") as one of the
external review's ten claims, and **confirmed it**: a 2×2 of (stack 0xD00 /
0x2000) × (`aie::exp2` / `exp2_poly`) reproduces the 0021 corruption
(384 non-finite values, the same signature) only at the small stack, and
passes at 4.278e-03 with `stack=0x2000`. The narrowing (the other candidate
cause the review raised) was ruled out in the same test: correct narrowing
with the old 0xD00 stack still fails. Cause: **the worker stack was too small
— the same 0xD00 the 4-chain GELU kernel independently overran — and a stack
overrun corrupts silently rather than faulting**, exactly the class of bug
`softmax.cc`'s own comment now documents in place. `exp2_poly` has been
**production softmax** since 0030, not a reverted experiment: `softmax_bf16`
(the `aie::exp2` path) and `softmax_poly_bf16` (`exp2_poly`, `stack=0x2000`)
both still exist as separate symbols in `softmax.cc`'s template, but the
exported design uses the poly path.

Fixed in [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md) §3 as part of
this task (durable-truth correction, not a register edit — that file is not one
of the three protected files).

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1

# T6 -- control (fails on npu2, the shipped kernel is aie2-only)
cd C:\dev\mlir-aie\programming_examples\ml\magika
python group2.py --dev npu2 --xclbin-path <out> --insts-path <out>   # FAILS: mac_4x8_8x4 undeclared
python group2.py --dev npu  --xclbin-path <out>\stream.xclbin --insts-path <out>\stream.insts.bin   # OK

# T6 -- no-stream control (aie_stream=(0,0) deleted, artifacts/t6_group2_nostream.py)
python <that copy> --dev npu --xclbin-path <out>\nostream.xclbin --insts-path <out>\nostream.insts.bin

cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python tools\count_dma_channels.py <out>\stream.prj
python tools\count_dma_channels.py <out>\nostream.prj

# T6 -- consumer-side rejection probe (artifacts/t6_probe_in_stream.py)
python probe_in_stream.py off      <out>\off        # control, OK
python probe_in_stream.py instream <out>\instream    # FAILS: cannot acquire from objectfifo stream port

# T10 -- compile-only softmax.cc, both symbols in one object (artifacts/build_softmax.py)
python build_softmax.py <out>
llvm-objdump.exe -t <out>\sm.prj\softmax_bf16.o
llvm-objdump.exe -d --no-show-raw-insn <out>\sm.prj\softmax_bf16.o

# T10 -- numpy accuracy model (artifacts/t10_numpy_model.py)
python artifacts\t10_numpy_model.py
```

## Result

- **T6 ANSWERED.** `aie_stream=` does free a DMA channel — proven two
  independent ways (channel census, and the `MM2S` `dma_start` block vanishing
  from the MLIR outright) — but only on the endpoint it marks, and only a
  tile with an executing core can be that endpoint (verified: marking a
  mem/shim-side consumer is meaningless since it has no program counter;
  marking a core-side consumer that still uses ordinary `.acquire()` is
  rejected at compile time). The exhausted resource in the production GEMM
  (0046/0047) is **mem-tile input** channels, which `aie_stream` cannot touch.
  **Does not reopen B-reuse.**
- **T10 ANSWERED.** `aie::exp2` is one native inlined hardware instruction
  (`vexp2`, no call) covering a full 64-wide row, cheap (291 instructions,
  whole function) and coarse (1.711e-02 vs fp64, hardware-measured). `exp2_poly`
  is a fully-unrolled degree-7 Horner polynomial plus manual IEEE exponent
  construction, more expensive (765 instructions) and 2.4x more accurate
  (7.109e-03 numpy / 6.7e-03 hardware). Confirms the standing production
  choice ([`0030`](../0030-m7-expert-review-tests/TASK.md)) rather than
  changing it.
- **Found and fixed while reading 0030 for T10**: `docs/CURRENT_STATUS.md` §3's
  "exp2_poly ... never diagnosed" was stale — 0030 diagnosed and fixed it
  (worker stack size). Corrected in place.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| `group2.py --dev npu2`: `use of undeclared identifier 'mac_4x8_8x4'` | The shipped `group2.cc` kernel targets aie2 (npu1) only — `mac_4x8_8x4` does not exist on aie2p | Built with `--dev npu` instead. Channel topology (core = 2 in/2 out, mem tile = 6 in/6 out) does not depend on AIE generation, so this does not weaken the probe — confirmed identical budgets in `count_dma_channels.py`'s own header comment, which is generation-agnostic |
| `probe_in_stream.py --dev npu` (T10's sibling script, same kernel family as `exp2_probe.cc`): `ld.lld: error: undefined symbol: mul_elem_16_conf` | `exp2_probe.cc`'s `aie::mul` on bf16 needs an aie2p-specific compiled helper not present in the aie2 (`npu`) library set — the reverse problem from `group2.cc` | Built with `--dev npu2` instead (works for `exp2_probe.cc`, unlike `group2.cc`) |
| `ObjectFifo(..., aie_stream=(1, 0))` consumed by ordinary `a.acquire()` | `aie_stream` requires the marked endpoint's kernel to use `put_ms()`/`get_ss()`, not the acquire/release L1-buffer API | Expected and recorded as the answer to "is it a free flag" — not a bug to fix, the compile-time rejection *is* the evidence |
| `ObjectFifoHandle.forward()` has no `aie_stream=` parameter | Only `ObjectFifo.__init__` accepts it; a mem-tile-forwarded fifo cannot be marked this way through `forward()` | Not needed for either probe (both used a direct producer/consumer `ObjectFifo`, no `forward()` hop) — noted here since it would matter for a real GEMM C-join, which does go through `.join()`/`.forward()`-shaped mem-tile fan-in |

## Artifacts

All under `tasks/0095-t6-t10-probes/artifacts/`:

- `t6_group2_nostream.py`, `t6_group2_upstream_reference.py` — the aie_stream A/B
- `t6_dma_channel_census.txt` — `count_dma_channels.py` on both builds
- `t6_mlir_evidence.txt` — raw `aie.mlir` / `input_with_addresses.mlir` diffs (the `MM2S` block vanishing, the unchanged shim allocation)
- `t6_probe_in_stream.py`, `t6_consumer_side_rejection.txt` — the consumer-side rejection probe and its compiler error
- `t10_objdump_evidence.txt` — symbol sizes, full mnemonic histograms, the `vexp2` context, the no-call check
- `t10_softmax_impl_aie_exp2.asm`, `t10_softmax_impl_exp2_poly.asm` — full disassembly of both functions
- `t10_numpy_model.py`, `t10_numpy_model_output.txt` — the numpy accuracy model and its output

Not checked in (build byproducts, regenerable from the commands above):
`.xclbin`/`.insts.bin`/`.prj` directories under the session scratchpad.

Docs touched: [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md) §3 (the
stale "never diagnosed" paragraph, corrected with a pointer to 0030 and this
task).

## Next

Neither thread has a follow-on. T6 is closed as answered-negative for the
purpose it was raised for (B-reuse); the mechanism itself (compute-tile
channel freedom for a kernel already written against `put_ms()`/`get_ss()`,
e.g. `group2`-shaped designs) remains available if a future design needs it.
T10 is closed with the production choice confirmed; `conv_even`'s 1.92x is
recorded next to note 0007 §3.1's other four instances of the same finding but
not applied (eltwise runs on the host today, per CLAUDE.md — this is dormant
until it doesn't).

## Proposed register update

*(`research/OPEN-THREADS.md`, `research/CLOSED-THREADS.md` and
`tasks/README.md` are being edited by other agents concurrently — not touched
here. Paste-ready text below.)*

### T6 — move from OPEN-THREADS.md to CLOSED-THREADS.md, status ANSWERED

```markdown
### T6 — Does `aie_stream=` free a DMA channel? · **ANSWERED 2026-08-23** · [`0095`](TASK.md)
[`0046`](../0046-m9-b-reuse-asymmetric/TASK.md) closed B-reuse on channel exhaustion. `aie_stream=(end, port)` makes a producer wire-only with no L1 buffer ([note 0007](../../research/notes/0007-unused-iron-surface.md) §1.6); whether it also costs no channel was unknown and directly relevant.

**Yes, but not the channel B-reuse needs.** Built `programming_examples/ml/magika/group2.py` (upstream's shipped `aie_stream` example) with and without `aie_stream=(0,0)`, compile-only, and diffed `input_with_addresses.mlir`: the marked tile's `aie.dma_start(MM2S, ...)` block is absent outright (not merely idle) — `count_dma_channels.py` reads core (1,3) output going 1/2 → 0/2. The *other* endpoint of the same link (a shim tile, no program counter) keeps its DMA channel unchanged in both builds — proven by identical `aie.shim_dma_allocation` lines. A second probe tried marking the consumer side of an ordinary `.acquire()`-based fifo (`aie_stream=(1,0)`) and got an immediate compiler rejection: `'aie.objectfifo.acquire' op cannot acquire from objectfifo stream port` — the marked endpoint's kernel must be rewritten to `put_ms()`/`get_ss()`.

**Conclusion: `aie_stream` frees a channel only on a tile with an executing core.** Mem and shim tiles are pure DMA/switch fabric with no program counter and can never be the marked endpoint (confirmed: marking one side of a link never changes the *other* side's channel count, by construction — there is nothing on a mem/shim tile that could run `put_ms`/`get_ss`). 0046/0047's census found the production GEMM's exhausted resource is specifically **mem-tile input** channels (the four-core C join), while cores themselves sit at 2/2 input but only 1/2 *output* — headroom on the wrong tile type and the wrong direction. **B-reuse stays closed**; not rebuilt, per the arithmetic above. → [`0095`](TASK.md)
```

### T10 — move from OPEN-THREADS.md to CLOSED-THREADS.md, status ANSWERED

```markdown
### T10 — `aie::exp2` never measured · **ANSWERED 2026-08-23** · [`0095`](TASK.md)
[`0020`](../0020-m5-layernorm-kernel/TASK.md). Superseded in practice by our own `exp2_poly` ([`0030`](../0030-m7-expert-review-tests/TASK.md)), but the comparison was never made.

**Measured on both axes, and it confirms the standing choice.** Cost, statically (`llvm-objdump` on `softmax_impl<false/true>`, same translation unit, same flags): `aie::exp2` is one native inlined AIE2P instruction (`vexp2`, no call/branch-and-link anywhere) covering the whole 64-element row — 291 instructions / 1,552 B for the whole function. `exp2_poly` has no such instruction to reach for and costs a fully-unrolled degree-7 Horner chain plus manual IEEE exponent construction — 765 instructions / 3,696 B, 2.6x. Accuracy, in numpy (coefficients reproduced bit-for-bit from `exp2_poly.h`, against exact fp64 `2**x`, over softmax's actual `[-120,0]` domain, AIE's actual default rounding mode `floor`): `exp2_poly` measures max relative error **7.109e-03**, closely reproducing [`0021`](../0021-m5-softmax-and-full-model/TASK.md)'s own hardware figure (6.7e-03) — the algorithm alone (float64, no bf16 rounding) is 2.0e-08, so essentially all of that error is the kernel's one bf16 store. `conv_even` would be **1.92x** better (3.702e-03) — a fifth instance of note 0007 §3.1's systematic-floor-bias finding, on the one piece that note explicitly flagged as not yet re-fitted. `aie::exp2` has no public formula to model in numpy; its hardware-measured figure against the same fp64 reference ([`0021`](../0021-m5-softmax-and-full-model/TASK.md)) is **1.711e-02** — 2.4x worse than `exp2_poly`. **Decision: keep `exp2_poly`**, which is what [`0030`](../0030-m7-expert-review-tests/TASK.md) already shipped — T10 supplies the comparison that decision was missing rather than changing it. → [`0095`](TASK.md)
```

### `tasks/README.md` index row

```markdown
| [0095](0095-t6-t10-probes/TASK.md) | **T6 and T10 closed by compiler/objdump evidence, no wall clock.** T6: `aie_stream=` genuinely frees a DMA channel (an `MM2S` `dma_start` block vanishes outright from `input_with_addresses.mlir`, core output 1/2→0/2) but only on a tile with an executing core — a mem/shim endpoint's channel is untouched regardless (identical `aie.shim_dma_allocation` in both builds), and marking a core-side consumer that still uses ordinary `.acquire()` is rejected at compile time (`cannot acquire from objectfifo stream port`). Since 0046/0047's exhausted resource is specifically mem-tile *input* (the four-core C join), **B-reuse stays closed**. T10: `aie::exp2` is one inlined native AIE2P instruction (`vexp2`, no call) covering a full 64-wide row, 291 instructions/whole function, 1.711e-02 vs fp64 (hardware-measured, 0021); `exp2_poly` is a fully-unrolled degree-7 Horner chain, 765 instructions, **2.4x more accurate** (7.109e-03 numpy / 6.7e-03 hardware) — a numpy model reproducing 0021's own hardware number confirms the standing production choice ([0030](0030-m7-expert-review-tests/TASK.md)) rather than changing it; `conv_even` would be a further 1.92x, dormant since eltwise runs on the host. Also found and fixed a stale `docs/CURRENT_STATUS.md` claim: the "exp2_poly ... never diagnosed" bug **was** diagnosed and fixed in 0030 (worker stack size) | research | done |
```
