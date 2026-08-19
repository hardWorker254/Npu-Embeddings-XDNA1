# 0049 — M9: T16 answered — the "missing 4,500 cycles" never existed; the baseline was the bfp16-emulated datapath

**Question** ([`OPEN-THREADS`](../../research/OPEN-THREADS.md) T16, successor to
[`0048`](../0048-m9-what-is-the-gemm-time/TASK.md)'s T1). A k-block is 196,608
MACs = 1,356 cycles at the "145 MACs/cycle/core M2 and M5 traced in isolation";
production spends ~4.72 µs per iteration. Where do the other ~4,500 cycles go —
fifo acquire/release, DMA stall, or the microkernel being slower in situ?

**Answer: nowhere, because they do not exist. The 145-MACs/cycle baseline was
traced with `--emulate-bfp16` ON — the 5.5× datapath M8 closed out for
accuracy. The production datapath (plain bf16) traces at 25–30 MACs/cycle in
isolation too, and the trace shows the core spending 78% of every iteration
executing back-to-back `vmac.f` at the fp32 datapath's hard limit of 32
MACs/cycle. The GEMM is not starved and not overhead-bound: it is
compute-bound on the wrong datapath.**

---

## 1. The baseline was mislabelled, and the record proves it

`experiments/m5-pretiled-gemm/artifacts/traced_all_shapes_c4.json` — the file
behind [`0007`](../0007-m5-pretiled-gemm-on-npu/TASK.md)'s "148.9–149.9
MACs/cycle" headline — carries `emulate_bfp16: true` and
`rel_frobenius: 1.04e-02` (the bfp16 error signature) on **every** row. M2's
137–142 figures are the same path — its own headline says plain bf16 is
**25.0** MACs/cycle. Both numbers have coexisted since M2; 0048 and T16
compared production (emulation off since M8) against the emulated trace.

Reproduced today, same script, same toolchain, same shape — the pair is stable:

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd <repo>\experiments\m5-pretiled-gemm
python gemm_pretiled.py --preset ffn_up --cols 4 -n 48 -M 512 --trace-size 1048576 --emulate-bfp16
python gemm_pretiled.py --preset ffn_up --cols 4 -n 48 -M 512 --trace-size 1048576
python gemm_pretiled.py --preset ffn_up --cols 4 -n 48       --trace-size 1048576   # M=256, canonical
```

| datapath | avg cycles/invocation | MACs/cyc/core | rel_fro |
|---|---:|---:|---|
| bfp16 emulation (`--emulate-bfp16`) | 1,339.9 | **146.7** | 1.04e-02 |
| plain bf16 (production) | 6,806.3 | **28.9** | 1.89e-07 |

## 2. Anatomy of a production k-block iteration (traced)

`analyze_trace.py` (this directory) decomposes each INSTR_EVENT_0→1 window and
the gaps between windows, integrating INSTR_VECTOR / MEMORY_STALL / LOCK_STALL /
STREAM_STALL overlap. Canonical run: ffn_up M=256, 4 cols, tile (64,64,48),
plain bf16 → `anatomy_bf16_M256.txt`.

55 windows captured = 47 matmul k-blocks + 8 zero-kernel calls (8 C-tiles per
core at this shape — the 583-cycle population is the zero kernel, exactly 8 of
them). Steady-state matmul k-block:

| component | cycles | share |
|---|---:|---:|
| INSTR_VECTOR (median) | **6,144** | 77.8% |
| in-window non-vector (loop bookkeeping, prologue) | 1,669 | 21.1% |
| in-window MEMORY_STALL (median, subset of above) | 198 | 2.5% |
| in-window LOCK_STALL / STREAM_STALL | 0 | 0% |
| gap to next window (acquire/release; median) | 84 | 1.1% |
| **total per iteration** | **≈7,897** | 100% |

**6,144 is not a round coincidence: it is exactly (64·64·48)/(4·8·8) = 768
MMAC steps × 8 cycles.** Each `aie::mmul<4,8,8,bf16,f32>` lowers to 8 `vmac.f`
— 32 MACs per instruction on the fp32 vector datapath — confirming
[`0003`](../0003-m2-bf16-gemm/TASK.md)'s static analysis ("~32 MACs each, ~28
MACs/cycle predicted, 25.0 measured") in situ. Disassembly of the built kernel
(`llvm-objdump -d matmul_bf16_f32_c894e098.o`, JIT cache): 64 `vmac.f`, inner
loop near-perfectly packed — a `vmac.f` in essentially every VLIW bundle,
dual-issued with `vextbcst`/`vshuffle`/`vlda`/`vldb`. **There is no code-quality
gap: the kernel runs the fp32 datapath at ~100% during its vector window.**

### The same design under bfp16 emulation is DMA-bound

`anatomy_bfp16_M512.txt`: median window 1,469 cycles with only 576 vector
(39% busy), and the gaps (median 1,215) are dominated by LOCK_STALL — the
MMAC datapath outruns the feed. **M2's "the array is starved, not slow" was an
emulated-datapath finding. Under plain bf16 the truth is the opposite: the core
computes 78% of the time and the DMA idles in its shadow.** That is also
*why* bytes measured free in 0048.

## 3. The account closes at the documented clock

At the 1.808 GHz [`docs/01-hardware`](../../docs/01-hardware/README.md)
records, 7,897 cycles = **4.37 µs — 93% of 0048's fitted 4.72 µs marginal
per-iteration cost.** (0048's "≈5,900 cycles" converted at an implicit
~1.25 GHz; the µs facts stand, the cycle figure was wrong.) The remaining ~7%
is unexplained and belongs with T18's probe-vs-bench ±10% family: this trace is
4 columns, the fit is 8, and DVFS is unobservable.

## 4. What this re-prices

Only the 1,753 non-vector cycles per iteration are amortisable; the 6,144
vector cycles scale with m·k·n and follow the work to any tile size.

- **T16 — ANSWERED** (this task). **T14 — answered with it**: the per-core gap
  to "peak" is the datapath choice, not operand prep; in-window overhead is 21%.
- **T19 (Stationary-B, k=96) collapses from 1.28× to ≈1.08×**: per 1.5
  old-iterations, 1.5×6,144 vector + one 1,753 overhead = 10,969 vs 11,846.
  The condition T19 itself stated ("assumes the 4,544 cycles are fixed per
  iteration") is now measured: **78% of them are not.**
- **T17 (bigger tiles via cross-tile L1) caps at ≤1.29×** (all overhead
  eliminated), realistically ~1.06× for (64,64,64). No longer "the lever".
- **The lever is the datapath.** On the same design the emulated path costs
  2,684 cycles/iteration against plain bf16's 7,897 — **2.9× of array GEMM
  time**, priced at ~1.35× end-to-end for MiniLM (39.4% wait) and ~1.66× for
  bge-large (60.8% wait) — gated on accuracy (bfp16 1-cos 3.47e-03 failed the
  2e-03 gate in 0026; MTEB per 0035 is the authority if this is reopened).
  int8 (T20, native (8,8,8), 512 MACs/cyc peak) is the other datapath route.

## Problems hit

- The plain-bf16 M=512 trace dropped event packets; my E0→next-E1 pairing then
  produced 2×-length windows and negative gaps (`anatomy_bf16_M512.txt` —
  kept, do not quote). The M=256 trace is clean and canonical.
- `run_one`'s printed "avg cycles" mixes zero-kernel windows (583 cyc) into the
  matmul average, deflating it ~6%: 6,539 printed vs 7,813 median matmul
  window at M=256. Window-level histograms, not the mean, for any future claim.

## Artifacts

- `analyze_trace.py`, `anatomy_bf16_M256.txt` (canonical),
  `anatomy_bfp16_M512.txt`, `anatomy_bf16_M512.txt` (dropped-packet example)
- Traces + parsed JSON under `experiments/m5-pretiled-gemm/artifacts/`:
  `trace_pretiled_kn_st_4c_bf16_f32_256x384x1536_t64x64x48.{txt,json}`,
  `..._512x384x1536...` and `..._bfp16_512x384x1536...` variants
- 0007's `traced_all_shapes_c4.json` — the mislabelled baseline, `emulate_bfp16: true` on every row
