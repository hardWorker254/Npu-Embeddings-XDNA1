# 0003 — M2: bf16 GEMM on one AIE core, traced

- **Date** 2026-08-16
- **Milestone** M2
- **Status** done (single-core). Multi-core deferred — see "Next".

## Goal

A trustworthy bf16 GEMM cycle baseline at MiniLM's shapes, measured from hardware
traces, plus the `--emulate-bf16-mmul-with-bfp16` A/B as a single-variable change.

## Context

From [0002](../0002-m1-hello-npu/TASK.md) the measurement chain is proven. M2 moves
from a toy elementwise kernel to the operation that actually dominates an encoder.

**Why single-core first**, deviating from the plan's "use `whole_array`":
`whole_array.py` — the canonical multi-core GEMM — **has no trace support whatsoever**
(`grep -c trace whole_array.py` → 0; no `trace_config` parameter, no worker marked
`trace=1`). It cannot answer a cycles question. The single-core design *does* trace,
and is the same design the pre-existing int16 traces came from, so bf16-vs-int16 stays
a clean single-variable comparison. Adding trace support to `whole_array` is now an
explicit follow-up.

## What was done

Wrote `experiments/m2-bf16-gemm/gemm_single_core.py`, derived from the mlir-aie
single-core example but parameterised over shape, tile, input dtype, **output dtype**,
and the bfp16 flag, with correctness checking, trace capture, and a JSON result record
per run.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m2-bf16-gemm

python gemm_single_core.py --preset square -n 32                    # native bf16->f32
python gemm_single_core.py --preset square -n 32 --emulate-bfp16    # the A/B
foreach ($p in @("qkv","proj","ffn_up")) {
  python gemm_single_core.py --preset $p -n 32 --emulate-bfp16
}
python gemm_single_core.py -M 256 -K 960  -N 384 -n 32 --emulate-bfp16   # BD limit probe
python gemm_single_core.py -M 256 -K 1024 -N 384 -n 32 --emulate-bfp16   # ...fails

# Second signal
$P = "C:\dev\mlir-aie\ironenv\Lib\site-packages\llvm-aie"
& "$P\bin\llvm-objdump.exe" -d --no-show-raw-insn `
    C:\Users\vegar\.npu\cache\69ac2e15202ca13ac86ec967\matmul_bf16_bf16_d8991b41.o
```

## Results

All bf16 → **f32**, tile 64×64×32, single core, from hardware traces:

| shape | variant | avg cycles | MACs/cycle | % of 256 peak | rel-Frobenius |
|---|---|---|---|---|---|
| 256×256×256 | native `(4,8,8)` | 5250.9 | 25.0 | 9.8% | **1.21e-07** |
| 256×256×256 | **bfp16 `(8,8,8)`** | **954.7** | **137.3** | **53.6%** | 1.04e-02 |
| 256×384×1152 (qkv) | bfp16 | 952.6 | 137.6 | 53.7% | 1.04e-02 |
| 256×384×384 (proj) | bfp16 | 952.3 | 137.6 | 53.8% | 1.04e-02 |
| 256×384×1536 (ffn_up) | bfp16 | 953.7 | 137.4 | 53.7% | 1.04e-02 |
| 256×960×384 | bfp16 | 949.8 | 138.0 | 53.9% | 1.04e-02 |

### Finding 1 — the bfp16 flag is worth 5.5×, not the 2× predicted

`--emulate-bf16-mmul-with-bfp16` changes `mac_dims` from `(4,8,8)` to `(8,8,8)`, which
by geometry alone suggested 2×. Measured: **5.5×** (5250.9 → 954.7 cycles), taking
bf16 GEMM from **9.8% to 53.6% of peak**.

**This single compile flag is worth more than any tiling work we had planned.** For
context, MLIR-AIR reports ~48–50% of bf16 peak for a well-compiled *multi-core* matmul;
we are at 53.6% on **one core**.

### Finding 2 — it costs ~10⁵× accuracy, and that decision is not yet made

| | rel-Frobenius |
|---|---|
| native bf16→f32 | **1.21e-07** (essentially exact) |
| bfp16 emulation | **1.04e-02** |

bfp16 shares one exponent across each 8-element block, so this is inherent, not a bug.
At ~1% error per GEMM, six layers deep, this plausibly exceeds our documented ≤5e-3
end-to-end embedding budget. **Do not adopt bfp16 by default yet** — it needs an
MTEB-level decision in M8, not a guess now. Both paths must stay selectable.

*Caveat on the number:* inputs are `iron.rand` = uniform [0,1), whose exponents vary
widely; real activations are roughly normal and may behave differently under block
floating point. Re-measure against real activations in M5.

### Finding 3 — fp32 accumulation was the entire correctness problem

First runs used `output_dtype = bfloat16` and reported 7.4e-3 error, marginally failing
tolerance. That was **100% the accumulator**, not the kernel: with bf16 out, the L1
accumulator tile is bf16, so each of the K/k accumulation steps re-rounds to 8 mantissa
bits. Switching to f32 out took the same kernel to **1.21e-07** — a 6×10⁴ improvement
from a dtype, with no algorithmic change.

Every paper specifies bf16 in / fp32 accumulate ([2504.03083](https://arxiv.org/abs/2504.03083),
[2607.11211](https://arxiv.org/abs/2607.11211)). Now confirmed on our own hardware.

### Finding 4 — per-tile cycles are shape-independent

Across four different MiniLM shapes — N from 384 to 1536, K from 384 to 960 — per-tile
cost stayed within **949.8–953.7 cycles (0.4% spread)**. Only the invocation count
changes.

This is direct empirical support for Rösti's minimal-reconfiguration strategy: **pick
one tile size, use it for every GEMM shape in the model, and vary only the shim DMAs
and loop bounds.** We do not need per-shape tuning.

### Finding 5 — the L1 budget rules out 64³ with fp32 output

`m=k=n=64` with f32 out failed to compile with an opaque
`'aie.tile' op Basic sequential allocation also failed`. The arithmetic explains it —
ObjectFifos are double-buffered:

```
bf16 out : 2*(64*64*2 + 64*64*2 + 64*64*2) = 49152 B = 48 KB   fits
f32  out : 2*(64*64*2 + 64*64*2 + 64*64*4) = 65536 B = 64 KB   exactly L1 -> fails
```

fp32 accumulation doubles the C tile. We use **64×64×32** (40 KB), which fits with room
to spare. The script now pre-flights this and prints the budget rather than letting
aiecc fail obscurely.

### Finding 6 — the DMA BD size limit bites MiniLM directly

`ffn_down` (K=1536) **cannot be compiled at all** with the naive B access pattern:

```
error: 'aie.dma_bd' op Size 1 exceeds the [0:1023] range.
aie.dma_bd(... [<size=1,stride=0>, <size=12,stride=32>,
                <size=1536,stride=384>, <size=32,stride=1>])
```

Boundary pinned by bisection: **K=960 works, K=1024 fails.** The BD size field is
10 bits, so K ≤ 1023.

This is exactly TileFuse's design challenge #3 (*"DMA stride-register capacity limits
breaking large MLP matrices"*), hit in practice. **Consequence: one of MiniLM's four
per-layer GEMMs is unrepresentable this way, so offline weight pre-tiling (M4) is a
hard requirement rather than an optimisation.** TileFuse's fix — interleaved
column-major ordering so each column's tiles are contiguous — is the known remedy.

### Finding 7 (from M1's second signal) — `aie::mmul` is not one instruction

The disassembly of `matmul_bf16_bf16_*.o` shows the inner loop `.LBB0_3 → .L_LEnd0` is
**~36 bundles containing ~32 `vmac.f`**, interleaved with `vextbcst` /
`vextbcstshfl` / `vshuffle` operand preparation. So `aie::mmul<4,8,8>` is decomposed
into many vector MACs, and an "intrinsics per cycle" efficiency model is meaningless.

At ~1 `vmac.f`/cycle × ~32 MACs each ≈ 28 MACs/cycle predicted, against 25.0 measured —
the two signals agree, and confirm the native bf16 path is issue-bound. Efficiency is
therefore reported as **MACs/cycle against the datapath peak**, which is defensible.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| **`mac_dims` = `(4,8,4)` on NPU2, bfp16 flag a no-op** | `_detect_arch()` uses `get_current_device(probe_runtime=False)`, which needs an **explicitly set** device; the handler silently falls back to `'aie2'` | `iron.set_current_device(from_name("npu1", n_cols=None))` before any `kernels.mm()`. **Verified**: before (4,8,4)/(4,8,4), after (4,8,8)/(8,8,8). See [note 0002](../../research/notes/0002-iron-silent-arch-fallback.md) |
| `ModuleNotFoundError: aie.helpers.dialects.ext` | Guessed import path | `from aie.iron.controlflow import range_` |
| `iron.tensor(bf16_array)` raises casting to uint32 | `iron.tensor()` cannot ingest an `ml_dtypes` bfloat16 array | Build on device with `iron.rand` / `iron.randint`, read `.numpy()` back for the reference |
| Correctness failed at 7.4e-3 | bf16 accumulator (finding 3) | `output_dtype=float32` |
| Opaque `Basic sequential allocation also failed` | L1 overflow (finding 5) | Pre-flight L1 budget check with a specific message |
| `ffn_down` fails to compile | DMA BD 10-bit size limit (finding 6) | **Not solved.** Needs offline pre-tiling in M4 |
| First `ffn_down` attempt errored differently | Trace buffer too small | Raised `--trace-size`; the underlying BD error then surfaced |

## Artifacts

`experiments/m2-bf16-gemm/artifacts/` — per run: `trace_*.txt`, `trace_*.json`,
`mlir_*.mlir` (the `input_with_addresses.mlir` needed to re-parse), and `result_*.json`
with shape, tile, dtypes, mac_dims, cycles, MACs/cycle, efficiency and error.

## Next

**Immediate follow-ups, in order:**

1. **Add trace support to a multi-core GEMM.** `whole_array.py` has none. Either patch
   our own copy (add `trace_config` + `trace=1` on one worker) or extend the
   single-core design to a worker grid. Without this we cannot measure the 8×4 array,
   and single-core numbers do not extrapolate.
2. **Decide bfp16 on evidence, not preference.** Carry both paths to M5, then let an
   MTEB subset decide in M8. Re-measure the error against realistic activation
   distributions, not uniform [0,1).
3. **M4 pre-tiling is now load-bearing** — it is the only way to express `ffn_down`
   (K=1536 > BD limit 1023), not merely a ~50% optimisation.
4. Sweep tile shapes (the 40 KB L1 budget leaves headroom; 64×64×32 was chosen to fit,
   not tuned) — but per finding 4, expect this to be a small effect.

**Open question worth an experiment:** at 53.6% of peak on a single core, is the
remaining gap operand preparation (`vextbcst`/`vshuffle` occupying slots) or accumulator
dependency? Rösti's four-independent-accumulator technique targets the latter; the
disassembly already shows `dm0..dm4` in use, so mm.cc may already do this.
