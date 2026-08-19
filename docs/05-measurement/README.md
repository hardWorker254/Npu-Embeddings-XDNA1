# Measurement doctrine

**The rule: wall-clock time is never a valid NPU performance claim on this project.**

The NPU is a shared resource. Another process — Ryzen AI SW, a browser using WinML,
FastFlowLM — can hold it or contend for it, and the driver serialises access. A
wall-clock number therefore measures *"how busy was the machine"* at least as much as
*"how good is the kernel"*. Worse, it fails silently: you get a plausible number that
is simply wrong, and you tune against noise.

So every performance claim about NPU execution comes from **hardware traces** or
**static instruction counts**. Both are immune to contention.

## The two signals

We deliberately keep two independent signals. If they disagree, something is wrong,
and finding out what is more valuable than either number alone.

### Signal 1 — hardware event trace (primary)

The AIE array emits timestamped events. Kernels bracket their work with `event0()` at
entry and `event1()` at exit; pairing them gives **exact cycle counts** per invocation.
Every kernel in `aie_kernels/` already does this.

### Signal 2 — static instruction count (cross-check, and sometimes enough)

> **Stronger than a cross-check for compute-bound kernels.** AIE cores do not
> stall — no cache, no dynamic scheduling, no branch misprediction — so the hot
> loop's instruction count ÷ core clock **is** its execution time. That is not
> the source-level paper-compute modelling this document warns about elsewhere:
> counting instructions in *emitted assembly* counts what runs, while modelling
> from C guesses what the compiler did (and has mispredicted by 5–300×).
>
> It matters here more than it would elsewhere, because the production
> 8-column design **cannot be traced at all** (see below). A method needing
> only the `.o` is not blocked by routing.
>
> `llvm-objdump -d -r build/X.o` — the relocations mark the zero-overhead
> loop's bounds. Count instructions **and `nop`s**: a `nop` is a VLIW slot the
> compiler could not fill, i.e. visible evidence the loop is not packed tight.
> → [note 0006](../../research/notes/0006-peano-loop-hints.md)


**AIE cores do not stall.** No caches, no out-of-order execution, no branch prediction,
fixed instruction latencies. Therefore, for a compute-bound kernel:

```
instructions in the emitted loop body ÷ core clock ≈ execution time
```

This needs no hardware at all — just `llvm-objdump` on the `.o`. It is the method
AMD's own `skills/aie-kernel-opt/SKILL.md` prescribes, and it is how Rösti verified
100% vector-unit utilisation (by confirming the absence of compiler-inserted no-ops).

## Getting a trace

### 1. Enable it in the design

```python
worker = Worker(core_fn, fn_args=[...], trace=1)
...
rt = Runtime()
with rt.sequence(...) as (a_in, c_out):
    rt.enable_trace(trace_size, workers=[worker])
```

`Runtime.enable_trace(trace_size=None, workers=None, ddr_id=4, coretile_events=None,
coremem_events=None, memtile_events=None, shimtile_events=None, egress_shim_col=0)`

- `ddr_id=4` → XRT `group_id(7)`, a dedicated trace buffer. `ddr_id=-1` appends the
  trace after the last output tensor instead.
- **Up to 8 events per tile type.** Defaults include `INSTR_EVENT_0`, `INSTR_EVENT_1`,
  `INSTR_VECTOR`, `MEMORY_STALL`, `STREAM_STALL`, `LOCK_STALL`, `ACTIVE`, `DISABLED`.
- Port monitoring: `PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0,
  master=True)`.

### 2. Run and parse

```powershell
python ..\..\..\utils\run_example.py trace          # does everything below

# ...or explicitly:
.\target.exe -x build\final.xclbin -i build\insts.bin -k MLIR_AIE -t 65536

python C:\dev\mlir-aie\python\utils\trace\parse.py `
    --input  trace.txt `
    --mlir   C:\Users\vegar\.npu\cache\<hash>\input_with_addresses.mlir `
    --output trace.json

python C:\dev\mlir-aie\python\utils\trace\get_trace_summary.py --input trace.json
```

`get_trace_summary.py` prints, per traced core:

```
<core name>
Total number of full kernel invocations is N
First/Min/Avg/Max cycles is a/b/c/d
```

**Report Min and Avg, not First.** The first invocation includes cold-start effects.

### 3. Visualise

Open `trace.json` at <https://ui.perfetto.dev> (W/S zoom, A/D pan). This is how you
*see* whether DMA and compute actually overlap, whether ping-pong buffering is working,
and where port contention is.

### From Python, without the Makefile

```python
from aie.utils.trace import TraceConfig
from aie.utils.trace.utils import print_cycles_summary, get_vector_time

cfg = TraceConfig(trace_size=65536)
my_design(a_t, b_t, c_t, M=512, K=512, N=512, trace_config=cfg)
cfg.trace_to_json(cfg.physical_mlir_path, "trace.json")   # path auto-populated
print_cycles_summary("trace.json")
print(get_vector_time("trace.json"))    # fraction of the window in INSTR_VECTOR
```

`get_vector_time()` is **vector-unit utilisation** — the single most useful
efficiency number for a compute kernel.

### Trace gotchas

| Symptom | Cause |
|---|---|
| Parse produces garbage / "invalid tile" | `--mlir` pointed at the **source** MLIR. It must be **`input_with_addresses.mlir`** from the cache dir — the lowered register writes only exist there. **This is the #1 trace mistake.** |
| Wrong tiles reported | `colshift` mismatch. **npu2 = 0**, npu1 = 1. Auto-detected; override with `--colshift`. |
| `trace.txt` empty or all zeros | Wrong `ddr_id` / buffer. With `ddr_id=-1` the trace is appended after the last output — size that buffer accordingly and do **not** also allocate `bo_trace` at `group_id(7)`. |
| No cycle summary | Kernel doesn't call `event0()`/`event1()`. |
| Too few events to form a packet | Very simple core. Add a ShimTile to the traced tiles. |
| `TraceConfig` produces nothing | `TraceConfig.ddr_id` must match `enable_trace()`. |

## Static instruction counting

```powershell
$P = $env:PEANO_INSTALL_DIR

# Instructions and no-ops in the zero-overhead-loop body
& $P\bin\llvm-objdump.exe -d -r build\kernel.o

# Count vector ops
& $P\bin\llvm-objdump.exe -d build\kernel.o | Select-String -Pattern '\bv(lda|ldb|mac|mul|st)\b' -AllMatches

# Spill detection (stack frame size)
& $P\bin\llvm-nm.exe --print-size build\kernel.o

# Integer division must NOT appear — __divsi3 is a performance disaster
& $P\bin\llvm-nm.exe build\kernel.o | Select-String '__div'      # must be empty

# Emit assembly directly
& $P\bin\clang++.exe -O2 -std=c++20 -I$env:MLIR_AIE_INSTALL_DIR\include `
    --target=aie2p-none-unknown-elf -S kernel.cc -o kernel.s
```

**No-ops in the loop body mean the vector unit is idle**, usually from a RAW hazard on
the accumulator. The fix is more independent accumulators (Rösti used four).

## Standing policy: the production array cannot be traced

**A fully-packed 8-column design cannot be core-traced.** Adding a single trace flow to
a 32-core `whole_array` GEMM exhausts routing — `Unable to find a legal routing` on
circuit-switched routing, and `max number of packet IDs reached` with
`--packet-sw-objFifos`. An exhaustive search over 32 `(trace_col, egress_shim_col)`
combinations found nothing. See [`tasks/0004`](../../tasks/0004-m2-multicore-gemm/TASK.md).

Traceable widths are **2 and 4 columns only**, and only with specific egress settings:

| cols | cores | traceable | `(trace_col, egress_shim_col)` |
|---|---|---|---|
| 1 | 4 | no | — |
| 2 | 8 | yes | `(1, 1)` |
| 4 | 16 | **yes** | `(0, 0)` |
| 8 | 32 | **no** | — |

And we cannot simply use 4 columns in production: at M=4096 it delivers 1.78 TFLOP/s
against 8 columns' 2.56 — **8 columns is worth 1.44×**.

**Therefore measurement is permanently two-track:**

| Quantity | How | Where |
|---|---|---|
| per-core cycles, vector utilisation, kernel efficiency | **hardware trace** | **4 columns** |
| end-to-end throughput, dispatch cost, TFLOP/s | **wall clock**, labelled | **8 columns**, NPU quiesced |

Extrapolating per-core cycles from 4 columns to 8 is legitimate **because per-core cost
is empirically flat**: 137.3 (1 core) → 142.0 (8 cores) → 141.7 (16 cores) MACs/cycle.
State the extrapolation explicitly whenever you rely on it.

> The two tracks will disagree, and that disagreement is a *result*, not an error. At
> 512³ on 32 cores the traced compute time is 16.4 µs while measured NPU time is 243 µs
> — **compute is ~7% of the total.** Reporting only the trace would claim 99.8% scaling;
> reporting only wall clock would claim 40%. Both numbers are needed to see that the
> array is starved rather than slow.

## What wall clock *is* for

Wall clock is legitimate — and necessary — for the **host side**, because that is
where it is the actual quantity of interest:

- Dispatch / submission overhead (finding **F1** is entirely a wall-clock story)
- XRT buffer sync and copy cost
- Tokenisation, weight loading, pooling on the CPU
- **End-to-end sentences/second**, which is what a user experiences

Rules when using it:

1. **Always label it as host-side or end-to-end.** Never present it as kernel performance.
2. **Median of ≥100 runs after ≥20 warmup.** Report the spread.
3. Note whether anything else was using the NPU.
4. `aie.utils.benchmark.run_iters(fn, *args, warmup, iters)` returns a `BenchmarkResult`
   with separate `e2e` and `npu` statistics — use it rather than hand-rolling.

There is also **`C:\Xilinx\XRT\nputrace_tool\nputrace.bat`** — an ETW-based
*driver-level* tracer (run as Administrator: `nputrace start [1-5]`, run the app,
`nputrace stop`, `nputrace view`). It captures submission/driver events, **not** AIE
core cycles. It is the right tool for measuring dispatch overhead specifically.

## CPU-vs-NPU comparison protocol

Required for any claim of the form "the NPU is N× faster".

1. **Quiesce the NPU.** Stop FastFlowLM, any Ryzen AI / WinML process, and anything
   else holding a hardware context. Verify with `xrt-smi examine`.
2. **Record the machine state** *by tool, not by hand* — `compare_three.py`
   writes it into every result file. A hand-rolled `Win32_Battery` check
   reported **"ON BATTERY" for a machine with no battery**, because an absent
   WMI class made the comparison never run and the `else` branch fire. Use
   `[System.Windows.Forms.SystemInformation]::PowerStatus`. The fields:
   power plan, on mains or battery
   (Rösti measured 145 → 255 GFLOP/s on mains vs 95 → 111 on battery — it matters
   enormously), background load, and whether the iGPU is busy.
3. **Same input, same tokenisation, same sequence length bucket, same batch size.**
   Feed pre-tokenised IDs where possible so tokenizer differences do not leak in.
3b. **INTERLEAVE, and use the same statistic on every side.** Run the encoders
   round-robin in one session so whatever the machine is doing, it does to all
   of them. This session measured sentence-transformers at 710, 662.9, 580.3,
   518.5 and 489.4 seq/s at the same batch on the same machine; a ratio between
   numbers taken minutes apart measures the machine's mood. Never compare a
   best-of-N against a mean — that buys the best-of side its whole spread.
   `experiments/m8-npu-vs-cpu/compare_three.py`.
3c. **Report steady state, not the ramp.** torch went 469 → 686 seq/s over five
   rounds while the NPU held to ±1%. Report the second half of the rounds as
   well as all of them, and take the ratio from the second half — that is the
   half that favours the CPU.
4. **Report both** batch-1 latency and large-batch throughput. Finding F2 says these
   tell different stories, and reporting only the flattering one is misleading.
5. **Report power.** The NPU's case is efficiency as much as speed — the Gemma3 team
   measured ~4.5 W NPU vs ~54 W iGPU. A 1.2× speedup at one tenth the power is a win;
   saying only "1.2×" hides that.
6. **Include the CPU baseline configuration**: thread count, ORT graph optimisation
   level, fp32 vs int8.

## Reporting numbers

Every performance figure that appears in a `TASK.md` or in `docs/` must be traceable
to a stored artifact — a `trace.json`, a `get_trace_summary.py` output, or a captured
command log in the task folder. **A number without a traceable source is not a result.**

Express kernel results as:

- **Cycles** (min / avg / max) per invocation
- **Vector-unit utilisation** from `get_vector_time()`
- **% of the 14.7 TOPS bf16 attainable ceiling** (not the 50 TOPS marketing figure)
- **Achieved DRAM read bandwidth**, since we are bandwidth-bound — see
  [`../01-hardware/`](../01-hardware/README.md)

## Watching the NPU from Windows — and why its memory never moves

**There is no NPU memory to watch.** XDNA2 has no device-local DRAM. The only
memory physically on the array is SRAM: **32 × 64 KB L1 = 2 MB** and
**8 × 512 KB mem tiles = 4 MB** ([`docs/01-hardware`](../01-hardware/README.md)).
Weights, activations and instruction streams all live in **system DRAM**,
pinned and mapped into the NPU's address space through the IOMMU. An XRT buffer
object is host memory. So a tool that reports "NPU memory" would be reporting
our own process's working set, and none of them do:

- Windows exposes **no NPU counter set at all** — nothing matching NPU, Neural
  or Compute Accelerator in `Get-Counter -ListSet *`.
- The NPU is not a `Win32_VideoController`; the only one present is the
  Radeon 890M. It appears as `NPU Compute Accelerator Device`,
  `PCI\VEN_1022&DEV_17F0`.
- `xrt-smi` on Windows (`C:\Windows\System32\AMD\xrt-smi.exe`) has exactly two
  reports, `aie-partitions` and `all`. Neither is memory.

Measured, batch 128, two lanes, `artifacts_b128il`:

| | |
|---|---|
| process peak working set | **607.0 MB** |
| process private bytes | 581.8 MB |
| XRT buffers (A 25.2 + B 1.2 + C 50.3 per lane, ×2) | 153.4 MB |
| host scratch in `Encoder::run` (×2 lanes) | 302.0 MB |
| mapped `.npue` | 69.0 MB |
| weights staged on the device once | 21.2 MB |

That accounts for ~546 MB of the 582 MB private, the rest being allocator
overhead and thread stacks. **Nothing is missing and nothing is hidden — the
"device" memory is the process memory.**

### Busy percentage *is* observable

The NPU is an **MCDM** device, and MCDM shares the GPU counter infrastructure,
so NPU busy time arrives as a GPU engine counter scoped to a pid:

```powershell
.\tools\npu_utilisation.ps1 -Exe build\npuembed.exe -WorkingDirectory runtime `
  -Arguments ".. --artifacts artifacts_b128il --threads 24 --pipeline 2 --bench 40"
```

The instance carries `luid_0x00000000_0x000170c6`, matching the NPU's BDF
`[00c6:00:01.1]` from `xrt-smi examine`, and our runtime issues no graphics
work, so nothing else can produce it. It agrees with our own accounting:

| | counter | runtime's own `NPU dispatch+wait` |
|---|---|---|
| 2 lanes, batch 128 | **50.5%** mean (52.1% peak) | **52.3%** |

### Memory IS attributed — under Shared, never Dedicated

`\GPU Process Memory(pid_*)` does report our buffers. At batch 128, two lanes:

| counter | ours | FastFlowLM `serve gemma3:1b` |
|---|---:|---:|
| Total Committed | 229.4 MB | 1234.3 MB |
| **Shared Usage** | **229.4 MB** | **1234.3 MB** |
| Local Usage | 229.4 MB | 1234.3 MB |
| **Dedicated Usage** | **0** | **0** |
| process working set | 607.0 MB | 2090.5 MB |

Ours tracks the XRT buffer count — 157.4 MB at one lane against 229.4 at two.

**A hypothesis tested and refuted.** We allocate data buffers
`XRT_BO_FLAGS_HOST_ONLY` while FastFlowLM uses `xrt::ext::bo(device, size)`
with 1 MB alignment and no host-only flag (`src/include/buffer.hpp`), so the
obvious guess was that their allocations are device-class and ours are not,
and that this is why theirs is visible in Task Manager. **It is not**: measured
side by side, both land in Shared/Local and **neither reports a single byte of
Dedicated**. There is no dedicated pool on this device for either of us to
allocate from.

The difference is **magnitude, not mechanism**: a 1B-parameter LLM carries
~1.2 GB of weights and KV cache; a 22M-parameter embedding model carries 21 MB
of weights and 153 MB of I/O buffers. 229 MB on a 30 GB machine is easy to
miss; 1.2 GB is not.

**This is not an NPU performance claim** (rule 1). It is a busy fraction of
wall time — good for *"is the array actually being used"* and for
cross-checking our accounting, useless for kernel quality.

### The trap: too few samples reads as 0%

`Get-Counter "\GPU Engine(*)"` enumerates every engine instance on the machine
and costs **seconds** per call — a 13-second run returned **one** sample, and
an earlier attempt returned **0.0%** for a configuration that was in fact
running at 37–39%. A zero from an unsampled counter is indistinguishable from
a zero from an idle device.

Two fixes, both in `tools/npu_utilisation.ps1`: filter on the instance name
**inside the counter path** (`\GPU Engine(pid_1234*)\...`), which is fast
enough to sample properly, and **refuse to print a percentage** below a
minimum sample count rather than reporting the average of nothing.
