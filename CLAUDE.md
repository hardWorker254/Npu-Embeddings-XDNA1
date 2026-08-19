# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

**NpuEmbeddings** — run encoder-only embedding models (BERT-style: all-MiniLM,
bge-small, e5, gte) very fast on the **AMD Ryzen AI NPU (XDNA2)**, from hand-written
AIE kernels, on **native Windows**, with a C++ runtime. *"FastFlowLM, but only embeddings."*

It is a **learning project**. The user's interest is close-to-metal programming. The
point is to genuinely understand the AIE array — **documentation and traceability are
first-class deliverables, not overhead.**

**Start by reading [`docs/00-overview.md`](docs/00-overview.md).**
**For where the project stands right now — what works, what does not, what was
tried and failed, and how to build and run the whole thing — see
[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).**

## Non-negotiable rules

1. **Wall-clock time is NEVER an NPU performance claim.** The NPU is shared and can be
   contended by other processes, so wall clock silently measures machine business
   instead of kernel quality. All NPU numbers come from **hardware traces**
   (`parse.py` → `get_trace_summary.py`) or **static instruction counts**
   (`llvm-objdump`). Wall clock is valid *only* for host-side/dispatch cost and
   end-to-end throughput, always labelled as such.
   → [`docs/05-measurement/`](docs/05-measurement/README.md)

2. **Never read the PDFs in `OthersResarch/`.** They are already indexed. Check
   `research/papers/manifest.json` first: if the file
   is listed, read the linked summary in `research/papers/` instead — the summaries are
   written to be genuine substitutes. Only a PDF **not** in the manifest (or whose
   sha256 changed) should be opened, and then it gets summarised and added.
   → `research/README.md`

3. **Every open question goes in [`research/OPEN-THREADS.md`](research/OPEN-THREADS.md),
   and that file is the authority on what is still open.** A `TASK.md` is a
   diary entry — written once, never revisited — which is the wrong property for
   a question. Twice now a correctly-stated open question has sat unread for
   more than 20 tasks after the thing blocking it went away
   ([`0044`](tasks/0044-m9-optimisation-sweep/TASK.md) Part 3, §6b in
   [note 0005](research/notes/0005-expert-review-tests.md)). A stale "untested"
   in an old task is **not** evidence a thread is live; the register is. Threads
   leave it only as ANSWERED, RETIRED or SUPERSEDED, with a pointer.

3b. **Every unit of work gets a `tasks/NNNN-slug/TASK.md`** recording goal, what was
   done, **the exact commands run**, results, and problems hit. **Failures are the
   valuable part — never delete or rewrite them.** The stated bar is that the task log
   alone should permit rebuilding the solution from scratch.
   → [`tasks/README.md`](tasks/README.md)

4. **Never vendor or redistribute anything from `../FastFlowLM/src/lib/**` or
   `src/xclbins/**`.** Those are closed binaries and its installer terms forbid
   redistribution. We read that repo for architecture and conventions only. This repo
   is Apache-2.0 and independently written.

5. **No Python at runtime.** Python is for build-time design generation (IRON has no
   C++ frontend) and for prototyping. The shipped product is C++ + XRT.

6. **A number without a traceable artifact is not a result.** Any figure in `docs/` or
   a `TASK.md` must point at a stored `trace.json` or command output.

## Environment

Everything is already installed. **Verify, don't reinstall.**

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1          # MUST be dot-sourced
```

| | |
|---|---|
| NPU | Ryzen AI 9 HX 370, Strix Point, **XDNA2 / AIE2P / npu2**, 8 cols × 4 rows = 32 tiles |
| Target | **`aie2p`**, `NPU2=1` (set by `iron_env.ps1`) |
| IRON | `C:\dev\mlir-aie\ironenv` — mlir-aie 1.3.4, Peano 21.0.0, **Python 3.13.15** |
| XRT | **`C:\Xilinx\XRT`** (2.21.0) — *not* `C:\Program Files\...` |
| Compiler | **Peano only.** No `xchesscc` (needs Vitis, no Windows build) |
| MSVC | VS Community 2026, toolset 14.51; cmake/ninja/clang on PATH |
| `make` | `C:\msys64\mingw64\bin\mingw32-make.exe` (GNU Make 4.4.1) |

**Traps that will cost an hour each:**

- **`XILINX_XRT` must stay unset** — it "poisons Windows builds" per `iron_setup.py`.
  Use `XRT_ROOT`. Ryzen AI SW 1.7.0 is also installed and may leak it into a shell.
- **The example Makefiles do not work natively** even with MinGW make — they assume
  POSIX and have WSL hooks. Use `python utils\run_example.py build|run|trace`, or run
  the `@iron.jit` script directly.
- **Tracing: `--mlir` must be `input_with_addresses.mlir`** from the build cache
  (`C:\Users\vegar\.npu\cache\<hash>\`), not the source MLIR. Most common trace mistake.
- **`colshift = 0`** for npu2 (npu1 uses 1).
- **CMake: `project()` must precede `find_package(XRT)`** or linking silently
  downgrades to static (mlir-aie #3048).
- `aiecc.py` is now a shim over the C++ `aiecc.exe`; `aiecc.run()` is deprecated.

Full detail: [`docs/02-toolchain/`](docs/02-toolchain/README.md)

## Python environments — keep them separate

| Env | Role |
|---|---|
| `C:\dev\mlir-aie\ironenv` | **IRON build only.** Do not install into it. |
| `C:\Users\vegar\.conda\envs\iron` | Py 3.13.15 base; torch 2.10.0+cpu, numpy, onnxruntime |
| `.venv-ref` (repo root, **created in M3**) | venv off the conda `iron` interpreter with `--system-site-packages`, so torch/numpy are inherited not duplicated. Adds transformers, safetensors, sentence-transformers, huggingface_hub. **mteb deferred to M8.** Recreate: `& "C:\Users\vegar\.conda\envs\iron\python.exe" -m venv --system-site-packages .venv-ref` |

Golden data crosses env boundaries **as files** (`.safetensors`), never as imports.
A `pip install` accident must not break the toolchain that took the most work to get running.

## Traps that have already cost us time

Read these before writing an IRON design — each was found the hard way and each is
silent.

1. **Set the device explicitly, or you compile for the wrong NPU.**
   ```python
   iron.set_current_device(from_name("npu2", n_cols=None))   # BEFORE any kernels.*
   ```
   Without it, `_detect_arch()` silently falls back to `aie2` (NPU1): bf16 `mac_dims`
   become `(4,8,4)` instead of `(4,8,8)`, and `emulate_bf16_mmul_with_bfp16` — worth
   **5.5×** — becomes a **no-op**. No error. `iron.get_current_device()` still says
   NPU2. Also note `from_name("npu2")` defaults to **1 column**; pass `n_cols=None`.
   **Third consequence, found in [`0044`](tasks/0044-m9-optimisation-sweep/TASK.md):**
   it also **halves the maximum shim DMA burst** — `BaseNPU2TargetModel`
   offers {64, 128, 256, **512**} bytes and `AIE2TargetModel` only {64, 128, 256},
   and `burst_length = 0` means "take the largest available". On a design that
   is data-movement bound, the fallback silently costs bandwidth as well as MACs.
   → [note 0002](research/notes/0002-iron-silent-arch-fallback.md)
2. **Always accumulate in fp32** (`output_dtype=np.float32`). bf16 output re-rounds at
   every K step: 7.4e-3 error vs **1.21e-07** with f32.
2b. **The default AIE rounding mode is `floor`, and it was the entire
   implementation error of all three eltwise kernels.** `aie_api/aie.hpp` says
   so twice; `aie_types.hpp` defines `floor` as *"always round towards negative
   infinity"*. We had never called `aie::set_rounding`, so every bf16 SRS this
   project ever executed carried a **systematic downward bias, not symmetric
   noise**. Measured on hardware ([`0044`](tasks/0044-m9-optimisation-sweep/TASK.md)
   Part 3), one line per kernel, each harness's *CPU-model-vs-golden* control
   bit-identical across the pair:

   | kernel | vs golden | implementation error alone |
   |---|---|---|
   | GELU | 4.312e-03 → **2.494e-03** (1.73×) | 3.886e-03 → 1.556e-03 |
   | softmax | 4.278e-03 → **3.325e-03** (1.29×) | 3.424e-03 → 1.481e-03 |
   | LayerNorm | 3.326e-03 → **2.059e-03** (1.62×) | 3.659e-03 → **3.967e-05 (92×)** |

   All three now sit at their bf16 design limit. GELU lands on **2.494e-03 —
   the exact figure `gelu_poly.cc`'s own header has predicted since it was
   written.** Mechanism confirmed independently of any error metric: under
   `floor` softmax row sums are min 0.994581 / **max exactly 1.000000** (no row
   can exceed 1 when every element rounds down); under `conv_even` they straddle
   it. This closes a mystery [`0015`](tasks/0015-m5-gelu-polynomial/TASK.md)
   opened by misattributing the gap to `aie::vector<float>` not being IEEE fp32,
   which [`0016`](tasks/0016-m5-fp32-probe/TASK.md) refuted while naming the
   right hypothesis and leaving it unchased for 28 tasks.
   **Shipped kernels still default to `floor` on purpose** — eltwise runs on the
   host, so none of them executes today, and `set_rounding` is *core-wide* state
   that leaks between kernels sharing a core. The `*_rne.cc` variants hold the
   evidence; `conv_even` becomes the default when eltwise returns to the array.
   → [note 0007](research/notes/0007-unused-iron-surface.md) §3.1
3. **Budget L1 before compiling.** `2*(m*k*in + k*n*in + m*n*out) < 64512`. Exceeding it
   gives the opaque `'aie.tile' op Basic sequential allocation also failed`.
   **The limit is 63 KB, not 64** — 1 KB of the 64 KB DMEM is reserved for the program
   stack (two independent sources: ICPP '25,
   Steinert). Note also that this
   inequality is the *Stationary C* form (C resident, A and B streamed, everything
   double-buffered) — the algorithm the stock IRON matmul uses. Stationary B's budget is
   `2mk·T_in + kn·T_in + 2mn·T_out`, which fits a larger `k_local` in the same space.
3b. **Budget ports and registers too, not just bytes.** Per core: **2 in / 2 out** DMA
   streams (so a kernel can compute `C = AB` but *not* `C = AB + C`), **24 × 256-bit
   vector registers**, and **40 × 256-bit accumulator registers = 5 × 2048-bit**. AIE-MLv2
   documentation says 8 accumulators; `aie2p` has **5**. With bf16's `C_v = 8` that caps a
   kernel at **5 independent MMAC accumulators**. Per mem tile: **6 in / 6 out ports**,
   48 across the array. A shimNOC DMA has **≤6 S2MM channels**, so a flat 32-way join is
   inexpressible — join hierarchically through the mem tiles.
   → INDEX.md constants table
4. **DMA BD size field is 10 bits (max 1023).** Any access-pattern dimension ≥1024
   fails to compile. MiniLM's `ffn_down` (K=1536) hits this **in the single-core
   design**, where B is walked as one column strip so the k-blocks collapse into a
   single `size=1536`. The **whole-array design does not** — its `step_tiler` keeps
   k-blocks as their own dimension (`<size=24, stride=24576>`) and K never appears
   as a size. Corrected in [`tasks/0007`](tasks/0007-m5-pretiled-gemm-on-npu/TASK.md);
   whether a shape hits the limit depends on the *access pattern*, not on K.
5. **Never use scalar float math in a kernel** — it lowers to `__mulsf3` calls, measured
   at 1,617× slower. → [note 0001](research/notes/0001-aie-kernel-pitfalls.md)
5b. **Loop hints: `chess_*` are ignored by Peano, and so is
   `AIE_PREPARE_FOR_PIPELINING`.** `chess_prepare_for_pipelining` and
   `chess_loop_range(...)` are xchesscc directives; Peano does not error on
   them, it drops them, so copied example code *looks* tuned and is not. AMD
   measured a relu kernel dropping **47% → 26% vectorisation** from exactly
   this. We are Peano-only. → [NPUEval](https://arxiv.org/abs/2507.14403)

   **The portable `AIE_*` wrappers are not a blanket fix.** Peano compiles
   under the header's `__AIECC__` branch (the aie2p driver defines it itself —
   verified with a `#error` probe), and there
   **`AIE_PREPARE_FOR_PIPELINING` is defined as nothing**. It is the only
   pipelining hint our kernels carry, in 8 places. What *does* survive Peano:
   `AIE_LOOP_MIN_ITERATION_COUNT` / `AIE_LOOP_MAX_ITERATION_COUNT` /
   `AIE_LOOP_RANGE` / `AIE_LOOP_UNROLL` / `AIE_LOOP_UNROLL_FULL`, all real
   `clang loop` pragmas. We use the min bound and none of the others.
   And they only bind when the trip count is a **compile-time constant** —
   which is the mechanism behind `rtp=True`'s measured +1.6%
   ([`0030`](tasks/0030-m7-expert-review-tests/TASK.md)).
   → [note 0006](research/notes/0006-peano-loop-hints.md)
6. **Assert `trace.txt` is non-empty** before believing any measurement.
6b. **Never write device tensors through `.numpy()`.** `Tensor.numpy()` syncs
   *from* the device and returns the host buffer; writing into that array never
   syncs back, so the NPU keeps using stale data.
   ```python
   A.numpy()[:] = values    # WRONG -- host only, silently ignored by the NPU
   A[:] = values            # RIGHT -- Tensor.__setitem__ syncs both ways
   ```
   **The first dispatch in a process is correct either way**, which is what makes
   this so easy to ship. And `.numpy()` keeps reporting the values you wrote, so
   a read-back "confirms" it landed.
6c. **Never validate a kernel against a device read-back.**
   `ref = A.numpy() @ B.numpy()` re-syncs from the device, so it agrees with
   whatever the device actually used and passes while measuring nothing.
   Reference against the values you *intended*, and assert the device matches:
   ```python
   ref = A_np @ B_np
   assert np.array_equal(A.numpy(), A_np), "A did not reach the device"
   ```
   → [note 0003](research/notes/0003-two-designs-per-process.md),
   [`tasks/0009`](tasks/0009-m5-sync-misdiagnosis/TASK.md)
7b. **Switching design costs far more than dispatching one**, and it scales with
   **descriptors, not columns**: **~25 µs + 7.2 µs per `aie.lock`** (≈ 49 µs + 5.8 µs
   per `aie.dma_bd`), i.e. 89 µs for a trivial passthrough up to 2.4 ms for an
   8-column GEMM. It is the switch itself — the *same* xclbin in two contexts costs
   the same as two different designs — and it is **not eviction**: `xrt-smi` shows
   every context `Active` with `Suspensions = 0` throughout.
   **Data-movement optimisation and switch optimisation are the same budget**: every
   descriptor added to feed the array better is paid again at each switch.
   Consequence: **wider is slower** while every dispatch changes design (8 columns is
   33% slower end to end than 2; 1 and 2 tie), so `tools/export_xclbin.py` defaults to
   `--cols 2`. Re-measure once batching or fusion lowers the switch count.
   → [note 0004](research/notes/0004-context-switch-cost.md),
   [`tasks/0024`](tasks/0024-m7-dispatch-cost-anatomy/TASK.md)
7c. **Never identify a build artifact by mtime.** A JIT *cache hit* does not restamp
   the directory, so "newest wins" silently returns a design you did not ask for —
   this produced a 2-column GELU when 1 was requested, and four identical xclbins in
   [`0022`](tasks/0022-m7-cpp-runtime/TASK.md). Match on contents. And note that
   **`aie.mlir` has no tile coordinates** — it is pre-placement
   (`aie.logical_tile<CoreTile>(?, ?)`), so counting columns there returns 0 for
   every design and the check fails open. Use `input_with_addresses.mlir`.
7. **A fully-packed 8-column design cannot be core-traced** — adding one trace flow
   exhausts routing (`Unable to find a legal routing`, or `max number of packet IDs
   reached` with `--packet-sw-objFifos`). Traceable widths are 2 cols at
   `(trace_col=1, egress_shim_col=1)` and 4 cols at `(0,0)`. Measure **per-core cycles
   at 4 columns (traced)** and **throughput at 8 columns (wall clock)**.
   → [`docs/05-measurement/`](docs/05-measurement/README.md)

## Current state

**Update 2026-08-19 (tasks/0051): FOUR models — and bge-base-en-v1.5 is the
one whose geometry actually fits XDNA2. Also: model fetching now happens
INSIDE the executable, and `get-model.cmd` is gone.**

- **Why bge-base fits**: every N a multiple of 384 (qkv 2304, attn_out 768,
  ffn_up 3072, ffn_down 768) so **tile_n stays 48** where bge-large is forced
  to 32; **head_dim 64** where MiniLM and bge-small have the untileable 32;
  wide rather than deep (0027/0042); WordPiece + post-LN + absolute positions,
  so zero runtime work. It packs to the **same layout_hash as MiniLM and
  bge-small** — the architectural claim as a constant.
- **It validated first attempt**: round-trip bit-exact, `1-cos` **1.353e-05**
  on hardware, **2.613e-05** end to end, top-10 neighbour overlap 1.0000.
  ONE xclbin, 16 streams, 8 columns, identity 64–69 bytes.
- **181.2 seq/s (pipelined, 5.50 cores) against my own ~230 prediction — a
  27% miss, recorded as one.** 0048's iteration fit was made at h=384 and
  extrapolated across a width doubling it was never tested at. What it did
  reveal: **the NPU is 74.1% of wall clock here against MiniLM's ~40%** — the
  most array-bound model we run, so **host levers (T3) buy less on it and
  datapath levers (T23/T20) buy more.** No CPU ratio is claimed; 0040's
  interleaving rule stands and the CPU side was not re-measured.
- **`get-model.cmd` is deleted.** It ran `curl` and then compared a
  `certutil -hashfile` digest against a hardcoded constant — the literal
  behavioural signature of a dropper, so SmartScreen and AV heuristics
  flagged it and a security warning was the first thing a new user saw.
- **New CLI**: `npuembeddings list` / `serve <model>` / `embed <model> <file>`
  / `help`. Fetching is **WinHTTP inside the exe** (`runtime/src/hub.cpp`),
  verified against a catalogue of pinned sha256s using the packer's own
  `sha256_file` — exposed, not reimplemented, because the Python side already
  has four copies of that function. Subcommands are **translated into the
  existing flag form**, so the ~2,700 lines below are untouched and there is
  no second dispatch path to drift.
- **Fail-closed and tested**: wrong weights refuse (exit 2, both digests
  printed); right weights with a wrong `config.json` refuse (the catalogue's
  geometry is cross-checked against the download before packing); an
  interrupted download never promotes its `.part`. Containers built in the
  release come out **byte-identical** to the repo's.
- **Designs are now chosen by reading which K they serve**, not by directory
  name (`pick_artifacts`), and a release carries one design set per width.
  The build emits `npuembeddings.exe` beside `npuembed.exe` so every existing
  script and task log keeps working.
- **`default_root` bug, reported by the user and worth remembering**: it
  searched *upwards* before checking the executable's own directory, so a
  release staged inside the repo (`dist\npuembeddings-0.2.0\`) climbed to the
  repository root and served **the repo's models while claiming to be
  self-contained**. The earlier test passed only because it unzipped outside
  the tree — it proved the layout worked, not that the search was right. The
  exe's own directory now wins whenever it holds a design or `models/`.
  **Then I reintroduced the very bug the function's comment warns about**, by
  folding the design check into the same upward loop: from `runtime\build\`
  the parent `runtime\` has designs but no models, so it beat the repo root.
  The two searches must each run to completion before the other starts. Now
  verified on all four layouts, not the one that was reported.

**Update 2026-08-19 (tasks/0050): T23's ceiling has an external measurement on
our exact SKU.** The ATB paper ([2511.16041](https://arxiv.org/abs/2511.16041),
UCLA+AMD, IRON/MLIR-AIE, web-indexed) demonstrates **24.3 TFLOPS BFP16 GEMM**
by buffering A's M dimension smaller than C's (ρ ≥ 1) — ~8× our production
array rate, with 2.88× of it from microkernel hand-optimisation alone
(0.32 → 0.92 TFLOPS/core ≈ 100% of MMAC peak). All on the MMAC datapath:
nothing for today's plain-bf16 path, but it re-prices the T23/T20 datapath
decision (0049's 2.9× is the floor) and gives T19 an overlap-safe mechanism.

**Update 2026-08-19 (tasks/0049): T16 ANSWERED — THE "MISSING 4,500 CYCLES"
NEVER EXISTED. The 145-MACs/cycle baseline was traced with `--emulate-bfp16`
ON; the production datapath (plain bf16) has always traced at 25–30. The GEMM
is compute-bound at ~100% of the fp32 vector datapath's 32 MACs/cycle/core
limit.** → [`research/OPEN-THREADS.md`](research/OPEN-THREADS.md) T16, T23

- **The record proves the mislabelling**: every row of 0007's "148.9–149.9
  traced" artifact carries `emulate_bfp16: true` and rel_fro 1.04e-02, and
  M2's own plain-bf16 line says 25.0. Reproduced today as a pair, same script:
  emulation ON 1,340 cyc / 146.7 MACs/cyc / 1.04e-02; OFF 6,806 cyc /
  28.9 MACs/cyc / 1.89e-07.
- **Anatomy of a production k-block** (traced at 4 cols, ffn_up): 7,813-cycle
  window, of which **6,144 = exactly 768 MMAC steps × 8 `vmac.f`** — each
  `aie::mmul<4,8,8,bf16,f32>` lowers to 8 fp32-datapath MACs at 32 MACs/instr,
  confirming 0003's static analysis in situ. 1,669 non-vector in-window, 84-cyc
  gaps, LOCK/STREAM_STALL ≈ 0. Disassembly: the inner loop is near-perfectly
  packed. **No code-quality gap exists; only the 22% non-vector share is
  amortisable by any tile-geometry lever.**
- **The account closes at the documented 1.808 GHz**: 7,897 cyc = 4.37 µs, 93%
  of 0048's fitted 4.72 µs. (0048's "≈5,900 cycles" converted at an implicit
  wrong clock; its µs facts stand.)
- **Under emulation the same design is DMA-bound** (39% vector busy, lock-stall
  gaps). M2's "the array is starved, not slow" was an emulated-datapath
  finding — under plain bf16 the DMA idles in the compute's shadow, which is
  *why* 0048 measured bytes as free.
- **Re-priced by this**: T19 Stationary-B k=96 collapses 1.28× → **≈1.08×**;
  T17 bigger-tiles caps at ≤1.29× (realistic ~1.06×) — no longer "the lever".
  T14 answered (datapath, not operand prep). **The only multi-× array levers
  left are datapath changes**: bfp16 emulation is 2.9× of array GEMM time
  (≈1.35× e2e MiniLM, ≈1.66× bge-large) but failed the 1-cos gate at 3.47e-03
  — reopening it is an MTEB accuracy decision for the user (T23), as is int8
  (T20, native (8,8,8)).
- **Measurement note**: `run_one`'s printed avg mixes 583-cycle zero-kernel
  windows into the matmul mean (~6% deflation); use window-level histograms.
  The M=512 plain-bf16 trace dropped packets and its pairing is corrupt —
  the M=256 trace is canonical (`tasks/0049`, `analyze_trace.py`).

**Update 2026-08-19 (tasks/0048): THE GEMM IS NOT BANDWIDTH-BOUND. It is bound
by the NUMBER of tile iterations.** This overturns the cost model every
data-movement decision since M5 has rested on, and retires two levers.
→ [`research/OPEN-THREADS.md`](research/OPEN-THREADS.md) T1

- **The discriminating pair was in the production shapes all along.** `ffn_up`
  and `ffn_down` have **identical MACs** (4.83 GMAC) and differ **1.50× in
  bytes**. They measure **4,196 vs 4,273 µs** — 1.8% apart, and the one moving
  *more* data is the faster one. `GMAC/ms` is flat at 1.08–1.15 across the three
  large shapes; `GB/s` spreads 17.7–27.0. Reproduced independently on the
  `--c-bf16` set, where 27% more bytes on `ffn_up` costs **0.3% less** time.
- **`t = 573 µs + 4.72 µs × k-block iterations per core`**, ≤2.3% residual on
  four points — against [`0010`](tasks/0010-m5-b-reuse-and-cost-model/TASK.md)'s
  `150.4 µs + traffic/33 GB/s`, which is 18% out on the mean and **50% out on
  the discriminating pair**. Treat the traffic model as **superseded for
  production shapes**; it was fitted at M=512 and never tested where it is used.
- **RETIRED: B-reuse** and the cascade milestone
  [`0047`](tasks/0047-m9-cascade-channel-probe/TASK.md) scoped for it. B-reuse
  removes bytes; bytes are free. 0010's 1.26–1.68× was priced with the refuted
  model. (The channel census from 0046/0047 keeps its value — it is about what
  the array can express.)
- **It also explains [`0045`](tasks/0045-m9-bf16-gemm-epilogue/TASK.md)**:
  narrowing C bought +4.9% end to end and ~0% of array time, because it removed
  bytes.
- **The successor question is now the top thread.** A k-block is 196,608 MACs =
  **1,356 cycles** at the 145 MACs/cycle/core M2 and M5 *traced in isolation*.
  Production spends **~5,900**. The array runs at **28–33 MACs/cycle/core
  against 145 traced**, and ~4,500 cycles per iteration are unaccounted for.
  Traceable at 4 columns (trap 7 forbids 8).
- **Tile SIZE is the lever now**, since iterations go as `1/(m·k·n)` — and it is
  hard-blocked: `(64,64,48)` costs **53,248 B of the 63 KB L1 budget** and every
  legal larger geometry overflows. That reframes **cross-tile `Buffer`**
  ([note 0007](research/notes/0007-unused-iron-surface.md) §1.2) from a
  curiosity into the one identified way to give a worker more L1.

**Update 2026-08-19 (tasks/0046–0047): B-reuse is out of DMA channels, not out
of space — and cascade moves the pressure rather than removing it.** Two probes,
nothing built, one lever closed and one milestone priced.
→ [`tools/count_dma_channels.py`](tools/count_dma_channels.py) (new)

- **The shipping GEMM has ZERO spare input channels.** Census of the
  post-placement MLIR at 8 columns: **all 32 core tiles at 2/2 in**, five of
  eight mem tiles at **6/6 in**. The mem-tile arithmetic is exact —
  `A(1) + B(1) + C(4 core rows) = 6` — so **the C join spends the budget**.
  B-reuse ([`0010`](tasks/0010-m5-b-reuse-and-cost-model/TASK.md), priced
  1.26–1.68×) is blocked by this, not by capacity: the column B slice is
  108–144 KB of a 512 KB mem tile.
- **`consumer_obj_type` cannot rescue it.** `repeat_count` is *"unavailable for
  shim tiles"* so an L3-fed fifo cannot replay, and `forward()`/`split()` do not
  expose `consumer_obj_type` at all.
- **Two things the repo believed are wrong.** `0010` and `gemm_pretiled.py` say
  the mega workaround fails on a **core** tile and needs "a core-side redesign";
  the failing op is `row = 1`, a **mem tile**.
- **Cascade trades 3 inputs for 3 outputs.** Built upstream's cascade matmul —
  it runs here with bf16→f32 and PASSes. At the same 4 columns: ours 6/6 in and
  3/6 out; cascade **3/6 in and 6/6 out**, with three of four cores per column
  having **no DMA output at all**. Since *inputs* were the exhausted side, this
  is still the route — but it is a milestone: upstream's kernel is scalar-only
  (**3.19 GFLOPS**), and `K % (4k) == 0` forces **k = 32** at h=384 (bge-large
  is fine at k = 64, so the wide model is again the easier target).
- **A CORE CANNOT TAKE A THIRD INPUT STREAM.** Every core tile is 2/2 in under
  *both* dataflows — A and B, and there is no room for a bias vector, LayerNorm
  params or fused-activation coefficients. This is why
  [`0020`](tasks/0020-m5-layernorm-kernel/TASK.md) had to pack γ+β into one
  buffer, and it is the same wall whisper-xdna hits packing K+V for fused
  attention. Treat it as a design constant.

**Update 2026-08-19 (tasks/0045): the GEMM can emit bf16 C — `--c-bf16`,
+4.9% and the epilogue is free.** The first lever from 0044 built and measured.
The matmul still accumulates in **fp32 for the whole K reduction** (trap 2
intact); a core-local fp32 `Buffer` holds the accumulator and a new
`narrow_f32_bf16` kernel converts **once** into the bf16 fifo object the DMA
drains. L1 is unchanged — the single-buffered accumulator costs exactly what the
halved C fifo saves, 53,248 B at (64,64,48) either way.

| model | fp32 C | **bf16 C** | gain | `1-cos` fp32 → bf16 |
|---|---:|---:|---:|---|
| MiniLM-L6 | 693.5 | **727.2** | +4.9% | 1.086e-05 → 1.498e-05 |
| bge-small | 347.9 | **365.3** | +5.0% | 8.348e-06 → 1.232e-05 |
| bge-large | 42.6 | **43.9** | +3.1% | 8.432e-06 → 1.281e-05 |

- **The epilogue costs nothing measurable**: per-dispatch `wait` moves by ≤29 µs
  of 3,028 (and 19 µs of 19,003 on bge-large). Static analysis said 2.3%; the
  DMA shadow absorbs it.
- **`read out + bias` falls ~20% at every geometry**, not the 50% halving the
  bytes suggests — because only the **read** halved. C is still *written* as
  679 MB of fp32 into host memory, since attention/LN/GELU/residuals all consume
  fp32. That is why this is +4.9% and not the ~10% projected.
- **It pays least where the array is busiest**: bge-large is 60.8% `wait`
  against MiniLM's 39.4%, so the host share this attacks is smaller. A host
  lever, the opposite of the fusion levers.
- **Accuracy costs exactly one rounding** — 1.38–1.52× on `1-cos`, still
  **133–162× inside** the 2e-03 gate, and **top-10 neighbour overlap is
  unchanged at 1.0000**.
- **DECIDED 2026-08-19: the datapath stays bf16 in, fp32 out.** `--c-bf16`
  remains available and measured, but it is **not** the default and MTEB is not
  being run on it for now. Treat "16 in, 32 out" as the standing contract when
  designing anything downstream; revisit only if a mode selection is explicitly
  asked for. (Making it the default would be an accuracy decision, and
  [`0035`](tasks/0035-m8-mteb-gate/TASK.md) established MTEB as the gate for
  those, not `1-cos`.)
- **`--c-bf16` collapsed the cache-marker namespace and nearly shipped the wrong
  stream.** `markers_for()` matched three positionless memref strings; with A,
  B and C all bf16, `ffn_up` [8192,384,1536] and `ffn_down` [8192,1536,384] have
  the *same three sizes* and became indistinguishable, so `purge()` for one
  deleted the other's build. It crashed only by luck of ordering. Now matched on
  the **ordered `aie.runtime_sequence(%arg0…%arg1…%arg2…)` signature**, which is
  strictly better for the fp32 path too. Sixth cache fail-open.
- **Third instance of one bug class:** `verify_embed_e2e.py` wrote a constant
  artifact path, so an A/B erased its own control — same as the two eltwise
  harnesses 0044 fixed. **Any script writing a result to a constant path is an
  A/B waiting to overwrite its baseline.**

**Update 2026-08-19 (tasks/0044): an optimisation sweep — twelve IRON features
we have never typed, and one architecture decision priced on the wrong ledger.**
Nothing was built; everything is priced.
→ [note 0007](research/notes/0007-unused-iron-surface.md)

- **Host eltwise costs 7.5% of an encode; the transport it forces costs 33%.**
  [`0032`](tasks/0032-m7-one-xclbin-production/TASK.md) moved LayerNorm, softmax
  and GELU to the host because each measured faster *and* more accurate than its
  NPU dispatch. True — but that priced **the operator**, not the `read out +
  bias` (18.8%) + bf16 convert (9.0%) + syncs (5.2%) that exist only because the
  next operator is on the host. `read out + bias` is **679 MB of fp32 C per
  MiniLM encode at batch 128**, read out of write-combined XRT memory at
  19.6 GB/s. In production (`--pipeline 2`) host work is **half the wall clock
  per lane** and that readback is **70.7 ms of it — 3.6× all three eltwise
  operators combined (19.7 ms)**. This was already raised as the expert review's
  **§6b**, priced at "t_conv + syncs ≈ 70 ms", **deferred with cause**, and then
  explicitly unblocked by 0032 — and nobody picked it up. Its estimate was ~2×
  low because it never counted the readback, which *grew* as a result of the
  0032 move. Reopen it — with the 16 KB program-memory wall respected as **one
  operator per core**, not the three-op universal worker 0032 killed. The cheap
  independent half: a **bias + fp32→bf16 GEMM epilogue** halves that 679 MB
  using the mechanism 0030 already built.
  **And note which way contention biased this**: the first reading of that table
  was taken with a foreign hw_context resident and said 22% / 3.6%. On an idle
  array the NPU shrinks from 65.0% to 40.3% of the encode while the readback
  *rises* from 14.3% to 18.8%. **The faster the array gets, the bigger this term
  is.**
- **Unused and available** (all verified in the installed 1.3.4 wheel, not just
  the source checkout): `pad_dimensions` (mem-tile only — it reopens
  [`0043`](tasks/0043-m9-attention-geometry/TASK.md)'s `cols ≤ 4` wall at ~5% of
  encoder FLOPs), **cross-tile `Buffer`** (a core reads its neighbours' L1, so
  trap 3's 63 KB is per *worker*, not per design — and it is the API for what
  ARIES did), **`CascadeFlow`** (core-to-core accumulator path costing **zero**
  `aie.lock`/`aie.dma_bd`, i.e. zero switch cost by our own 0024 model),
  `consumer_obj_type` (the fifo shape [`0010`](tasks/0010-m5-b-reuse-and-cost-model/TASK.md)
  wanted for B reuse and could not express), `disable_synchronization` +
  `delegate_tile`, `aie_stream`, `init_values`, and hand-wired
  `TileDma`/`Bd`/`Lock` (§2g).
- **`xrt::runlist` is in our XRT and unused by our runtime.** Worth only ~0.4%
  today (submit is 94–103 µs against a 578 ms encode) because our dispatches are
  huge; worth having the moment attention or eltwise returns to the array.
- **The rounding mode was the whole implementation error** (trap 2b above):
  `set_rounding(conv_even)` is one line and it moved GELU 1.73×, softmax 1.29×
  and LayerNorm 1.62× against the goldens, taking LayerNorm's error against a
  numpy model of the same formula from 3.659e-03 to **3.967e-05**. All three
  kernels now hit their bf16 design limit. Found by *running* the hypothesis
  [`0016`](tasks/0016-m5-fp32-probe/TASK.md) wrote down and left for 28 tasks.
- **Closed, do not revisit:** `burst_length` (already maximal by default),
  `AIE_LOOP_UNROLL_FULL` (whisper-xdna measured **14% slower** on a straight
  vector loop, independently confirming note 0006's inference), and the
  **centred polynomial basis** — a borrowed 2.5× claim, tested against
  `gelu_poly.cc` and **refuted at fp32**: identical error to four significant
  figures until degree 10, because their fix targets *bf16 coefficients* and
  ours are fp32. That test did show degree 8 → 7 is free (3.6e-04 against a
  2.465e-03 bf16 floor).
- **NINTH FAIL-OPEN, FOUND AND CLOSED: a stale `npuembed.exe` was holding an
  Active hw_context with 1,032 MB, and `--bench` had no way to say so.** It read
  **221.4 seq/s against a true 691.0** (3.1×), with per-dispatch hardware wait
  15,541 µs against 3,029. Rule 1's usual mitigation does not cover this:
  interleaving against the CPU ([`0040`](tasks/0040-m9-honest-cpu-baseline/TASK.md))
  corrects drift that hits *both* sides, while a resident NPU context hits only
  ours — so it makes the **ratio** wrong, confidently. And the contending
  process was **ours**, which a long measurement session produces by itself.
  **`--bench` now refuses to run** (`runtime/src/npu_contention.cpp`, exit 2)
  unless `xrt-smi examine -r all` shows no foreign `Active` context. It fails
  closed on three things — a foreign context, `xrt-smi` missing, and a table it
  cannot parse — because *an absent data source is not a negative reading*
  (0040). `--allow-contention` overrides it and says in the output that the
  number is not an NPU performance claim.
  **The third failure mode caught a bug in the guard on its first run** (the
  table is indented, so the first parser matched zero rows); a fail-open would
  have reported "idle" forever on every machine.
- **Throughput on an idle array, three runs, 0.9% spread: MiniLM 691.0 seq/s
  single-lane and 907.5 with `--pipeline 2`** — against the 618 / 833 recorded
  in [`0033`](tasks/0033-m7-pipelined-lanes/TASK.md). The CPU side was **not**
  re-measured, so no new ratio is claimed here; `0040`'s rule stands that the
  defensible quantity is the interleaved ratio, not either side alone.

**Update 2026-08-18 (tasks/0038–0039): TWO models, and the runtime reads them
rather than assuming them.** `--model` picks between `models/*.npue`; the table
is built from the containers, and selection becomes **required the moment there
is more than one**, so a script breaks loudly instead of silently changing which
model it measured.

- **bge-small-en-v1.5 runs on the NPU**, on unchanged xclbins — identical GEMM
  shapes, so only the dispatch count changes. `1-cos` **8.348e-06**, MTEB gate
  **−0.03**, **+2.99 MTEB points over MiniLM** on the same five tasks, at
  **450.0 seq/s against 888.7** (0.506×, exactly what twice the layers costs).
- **MiniLM is now 892.2 seq/s** (was 844.8), and `--embed` output is
  **bit-identical** through the whole refactor.
- **A 2× regression shipped inside a commit verified bit-identical.** Making
  attention generic in `head_dim` turned a compile-time vector count into a
  runtime one; attention went 29 → 58 ms and wall clock moved 829 → 828, so
  only the `--bench` breakdown showed it. Fixed with `qk_impl<NV>`/`av_impl<NV>`.
  **Bit-identical verifies correctness only** — genericity that replaces a
  constant with a variable is a performance change by construction.
- **Seven fail-opens are now eight.** Pooling, `source_repo` and the golden
  filenames were literals; the fixture `source_sha256` was carried and never
  read (it fired in production on its first real encounter); `verify_npue`'s
  1-cos gate existed only relative to MiniLM and skipped entirely for a new
  model; the endpoint gated on absolute cosine, which is a property of the
  training objective rather than of quality, and now gates on **separation**.

**Update 2026-08-18 (tasks/0042): THREE models, and the width prediction is
confirmed on real ones.** bge-large runs at **hidden 1024, 24 layers,
head_dim 64** -- `1-cos` **8.432e-06**, 604 MB of weights staged, ONE xclbin
with 4 streams, first attempt.

| model | h | layers | NPU | torch | ORT | **NPU / strongest CPU** |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM-L6 | 384 | 6 | 877.0 | 489.4 | 234.3 | **1.792×** |
| bge-small | 384 | 12 | 444.6 | 290.0 | 134.0 | **1.533×** |
| bge-large | 1024 | 24 | 52.8 | 25.1 | 11.4 | **2.106×** |

**Width helps, depth hurts, and bge-small separates them.** Same width, twice
the layers: 1.792 → 1.533, because our cost is per dispatch and the CPU has no
such term. Width 384 → 1024 takes it to 2.106× — the direction
[`0027`](tasks/0027-m7-width-hypothesis/TASK.md) predicted from a synthetic
sweep, now on a real model. Per core: **4.2× better**, up from 3.2×.

**`tile_n` is a parameter, not a constant.** bge-large's N in
{1024, 3072, 4096} makes 48 illegal; 32 is the largest legal value that fits
L1 (40,960 B of 63 KB). The C++ packer's *frozen* layout hash is gone,
replaced by `npue::gemm_b_layout()` — verified by reproducing the frozen
constant byte for byte. `verify_pack_parity --tile-n` covers both sizes.

**Memory was never the blocker**: `--probe-bo` allocates 8 GB in 1.76 s and a
single 3 GB buffer, against bge-large's 1023 MB need.

**And a banner lied.** After wiring `--tile-n` the packer printed
`tile (64, 32)` and tiled at 48 — the flag reached the layout descriptor and
not the `prepare_model` call. Caught only because 48 is illegal at these
shapes. A status line that reports the intention rather than the value is the
same fail-open as all the others.

**The CPU comparison is now interleaved, and ONNX Runtime is not the stronger
opponent** (tasks/0040). `prior-art.md` prescribes ORT as *the* baseline; it
measures **234.3 seq/s against torch's 489.4**, and attention never fuses under
either attention implementation. Measured round-robin in one session, same
statistic on every side, steady state:

| | NPU | torch | ORT | ratio |
|---|---:|---:|---:|---:|
| MiniLM | **877.0** | 489.4 | 234.3 | **1.792×** |
| bge-small | **444.6** | 290.0 | 134.0 | **1.533×** |

**Interleave, always.** sentence-transformers measured 710, 662.9, 580.3, 518.5
and 489.4 seq/s at the same batch on the same machine in one session, and torch
ramps 469 → 686 *within* a run while the NPU holds to ±1%. The published
"1.17× of 710" compared best-of-5 against a mean, minutes apart, and is
superseded. Best-of-5 is worth only 3.4%, so the statistic does not explain the
change — the machine does, and the defensible quantity is the ratio.
**And record machine state by tool**: a hand-rolled check reported "ON BATTERY"
for a machine with no battery, because `Win32_Battery` returns nothing there and
the `else` branch fired. An absent data source is not a negative reading.

**bge-small shows F1 again**: twice the layers costs us **0.507×** and the CPU
only **0.593×**, because our cost is per dispatch and 12 layers is 96 of them.
That gap is the size of the fusion prize.

**NPU memory never moves, and that is correct.** XDNA2 has no device-local
DRAM — 2 MB of L1 and 4 MB of mem tiles is all the memory on the array, and
every buffer is pinned host RAM. Windows exposes no NPU counter set and
`xrt-smi` has no memory report. Busy percentage *is* observable through the
MCDM compute engine and agrees with our own accounting (50.5% vs 52.3%);
`tools/npu_utilisation.ps1` measures it and **refuses to report a percentage
from too few samples**, because an unsampled counter reads as 0%.
→ [`docs/05-measurement/`](docs/05-measurement/README.md)

**Update 2026-08-18 (tasks/0031–0033): 833 seq/s pipelined / 618 single-lane,
1-cos 1.086e-05 — CPU WALL PARITY PASSED (1.17× of the CPU's 710).**
Production run: `npuembed .. --artifacts artifacts_b128il --threads 24
--pipeline 2` (two encode lanes over one design, NPU under one mutex,
lanes verified bit-identical). **And energy is now measured, not argued:
1.94× better per sequence than the CPU (tasks/0034) — via the Windows
`\Energy Meter` RAPL counters, which DO have instances on this machine.
The old "no instances, needs external instrumentation" note was checking
`\Power Meter`, a different counter set. Harness: `tools/measure_energy.ps1`
and `tools/energy_compare.ps1`. **And the M8 ACCURACY GATE IS PASSED**
(tasks/0035): five MTEB tasks, mean delta **+0.04 points** against the CPU
on the same checkpoint at the same sequence length — the bf16 + fp32
accumulate path costs nothing downstream, and `--emulate-bfp16` is closed
out. MTEB runs through a file bridge (`npuembed --encode-file` +
`experiments/m8-npu-vs-cpu/npu_encoder.py`). **And the TOKENIZER is done
(tasks/0036): WordPiece in C++, 6,826/6,826 texts byte-identical to
HuggingFace, Unicode tables generated by `tools/gen_tokenizer_tables.py`.
`npuembed --embed file.txt out.f32` is text in, vectors out, one process.
The vocabulary now rides inside the .npue as a U8 tensor, so the product
is ONE file plus one exe. Note: `pack_npue.py` was found broken (missing
`gemm_b_layout` import) and fixed -- it had not run since that refactor.** The production architecture is now: NPU = pure GEMM — all
four shapes as instruction streams over ONE xclbin in ONE hw_context (zero
design switches, mechanism proven in 0029/0030, productionised by
`tools/export_gemm_rtp.py` + the `gemm_rtp` runtime mode) — and LayerNorm,
softmax and GELU on the host in fp32 (`--host-ln/-sm/-gelu`, forced on in
unified mode), each measured faster AND more accurate than its NPU dispatch.
The paragraphs below describe the state as of 0027 and remain accurate as
history; where they disagree with this paragraph on what production does
TODAY, this paragraph wins. Key correction to the eltwise story: a 2.3 ms
design switch means **no elementwise op at h=384 earns an NPU dispatch**; the
il4 kernels (0031) stay in the tree for the future fused design. Also
measured: a core's 16 KB program memory holds at most two of our kernels —
the three-op universal worker is dead (0032).

**M0–M6 complete, and the model runs fully on the NPU.** Every op the array can
express is on it — 24 GEMMs, 6 GELU, 13 LayerNorm, 6 softmax per encode — validated
end to end against HuggingFace at **1-cos 3.4e-04**. Only attention's per-head GEMMs
(`head_dim = 32` does not tile) and elementwise glue remain on the host.

**M7 gate passed too**: a C++ runtime mmaps the `.npue`, dispatches through XRT with
no Python in the process, and reproduces the Python path's `rel_fro 1.507e-03`
*identically* ([`tasks/0022`](tasks/0022-m7-cpp-runtime/TASK.md)). Weights go from the
mapped file to a DMA descriptor untouched. What remains for a complete C++ encoder is
the other kernels, the orchestration, and the WordPiece tokenizer.

**And it can now be measured properly.** The C++ encode runs at **42.4 seq/s using
0.20 CPU cores** — **60× the Python path**, whose 8.4 ms per dispatch of glue is gone
([`tasks/0023`](tasks/0023-m7-full-cpp-encode/TASK.md)). Against
`sentence-transformers` on CPU (267.6 seq/s at 12–17 cores) that is 6.3× slower in
wall clock and **~10× better per core** — which on this project's own criterion
(parity is enough; offload and energy are the point) already qualifies.

**And then the breakdown was taken apart, and it said something else.** 0023 read
the 1.50 ms per dispatch as host-side bf16 conversion and memcpy. Removing all of it —
21.2 MB of per-dispatch weight copies staged once, conversions vectorised to AVX2 —
bought **9%**, and the instrumentation that proved it wrong found the real cost:
**changing which design the array is configured for**, paid on all 49 dispatches
([`tasks/0024`](tasks/0024-m7-dispatch-cost-anatomy/TASK.md),
[note 0004](research/notes/0004-context-switch-cost.md)). Host work is now 2.5 ms of a
76 ms encode. **52.4 seq/s at 0.33 cores**, `1-cos` unchanged at 3.430e-04.

The switch costs **~25 µs + 7.2 µs per lock in the design** — 89 µs for a trivial
passthrough, 2.4 ms for an 8-column GEMM. Three hypotheses were killed to get there:
it is not reconfiguration-by-difference (the *same* xclbin in two contexts costs the
same), not resident-context pressure (2 loaded ≡ 7 loaded), and not eviction (`xrt-smi`
shows `Suspensions = 0` on every context throughout). **Data movement and switching are
one budget** — each descriptor added to feed the array better is paid again at every
switch — which is a trade the indexed papers do not describe, because they measure
kernels in isolation rather than a 49-switch pipeline.

**Then batching cashed it in: 72.2 seq/s** ([`tasks/0025`](tasks/0025-m7-batching-and-crossover/TASK.md)).
The switch bill is fixed per encode, so a larger M amortises it directly — 1.35× at
constant width, **1.70× over 0023** — with `1-cos` identical at 3.430e-04 for batch 4 and
batch 16. The width default became a *prediction* rather than a preference: fitting
`t = 140 µs + 2.43·M/cols` (whose 140 µs independently reproduces M2's ~144) puts the
4-column crossover at **M ≈ 947**, and at M = 1024 two and four columns measure a dead
heat. Use `--cols 2` below ~batch 15 and `--cols 4` above it.

**Two processes scale 1.46× — and identically at 1, 2 and 4 columns.** Spatial
partitioning would need narrow designs to scale better; they do not. Full serialisation
on the array predicts `1/0.72 = 1.39×`, so the rest is host overlap. The array runs one
design at a time, measured rather than inferred.

**Then six routes were measured against CPU parity, and it stops at 2.9× short**
([`tasks/0026`](tasks/0026-m7-closing-on-cpu/TASK.md)). **251.3 seq/s at batch 128 on
1.58 cores**, against the CPU's 710 on twelve — **3.2× better per core**, `1-cos`
unchanged at 3.397e-04. What worked: AVX2+threaded host attention (1.39×), eltwise
designs 1→2 columns (1.36×), a 4-chain GELU kernel (1.17×), batching, and threading
the host conversions. What did not: a 4× larger GELU tile (no change at all) and
`--emulate-bfp16` (+2.2% and it **fails** accuracy at 3.470e-03).

**The ceiling is quantified.** Eltwise is 209 ms of the 347 ms NPU path, and it is
compute bound, not data bound: a passthrough moving the same bytes over the same
columns is **15× faster** than GELU. A degree-2 speed probe gives
`t ≈ 2174 µs + 941 µs per Horner step` — linear in degree **with four independent
chains already running**, so it is throughput, not latency, and ~3 cycles per native
op. **The kernel is at the machine's fp32 vector limit.** Even reducing eltwise to
*zero* leaves 427 seq/s against 710. Parity needs a different decomposition — bf16
eltwise arithmetic (an M8 accuracy decision) or work that stays in the MAC datapath
where the 14.7 TOPS actually live — not more optimisation of this one.

> **Corrections from [`0027`](tasks/0027-m7-width-hypothesis/TASK.md).**
> **(1)** The ceiling above overreached. The degree probe bounds ONE CORE, and
> compute-bound work scales with cores: eltwise measures **1.98× per column
> doubling**. The blocker was never the arithmetic — it was that the design
> would not *compile* wider, because each core opened its own shim stream.
> Rewritten with ObjectFifo `.split()`/`.join()` through the mem tile (the way
> the GEMM already does it), **GELU runs at 8 columns: 3.87× alone, 1.13× on the
> full encode**, `1-cos` unchanged. LayerNorm and softmax still need the same
> rewrite — they are 95 ms of the 209 ms of eltwise at batch 128.
> **(2)** "Work that stays in the MAC datapath" is now quantitative and tested.
> A BERT layer has **4h GELU elements against 12h² MACs — a 1 : 3h ratio**, so
> the elementwise share falls as **1/h** regardless of implementation. Measured:
> GEMM grows 3.72× and GELU 1.96× when h goes 384 → 768 (predicted 4× and 2×),
> and the elementwise share drops **29.7% → 18.1%**. **MiniLM at h=384 is near
> the worst case for this machine**, measured rather than asserted.
> **(3) No parity-h number.** An earlier projection said h ≈ 1300–2000; it is
> **withdrawn**. It assumed CPU time grows 4× per doubling, but the CPU runs the
> same encoder and has the same 1 : 3h structure, so its total also grows
> sub-quadratically — and without a measured CPU split into GEMM and elementwise
> time there is no basis for the number. Also, our **GEMM rate gain is nearly
> spent**: 0.639 → 0.688 TFLOP/s (+7.7%) from h=384→768, already 72–88% of the
> 2-column-normalised ceiling, so later doublings give GEMM ×4 not ×3.72. The
> claim that stands without projection is the per-core one: **3.2× better per
> core**, and 1 : 3h says it only strengthens with width.

**And then step 0 of the one-xclbin architecture confirmed it is real**
([`tasks/0029`](tasks/0029-m7-one-xclbin-probe/TASK.md)). `final.xclbin` is the static
configuration and `insts.bin` the per-dispatch runtime sequence, so operations sharing
a static design are just instruction streams over one hw_context. Measured: two
sequences over the same design produce xclbins differing **only in UUID metadata**; a
foreign instruction stream through another build's xclbin is **exact on all 6.3 M
elements**; and alternating two streams in one context costs **alone-price** (500 µs
vs 497) while the two-context control still pays 479 µs — 0024's lock model predicted
486, a 1.5% cross-check. **The 49 switches per encode (~60 ms at batch 128) are
removable without fusion.** What remains: RTP-ify the GEMM kernel's two loop counts so
the four shapes share one ELF, then unify. Risks: RTP maturity in IRON, and
`range_(rtp)` possibly pipelining worse than a compile-time bound.

**An external review was then tested claim by claim** ([`tasks/0030`](tasks/0030-m7-expert-review-tests/TASK.md),
scoreboard in [note 0005](research/notes/0005-expert-review-tests.md)): 7 of 10 claims
confirmed on hardware, 1 refuted by an existing measurement, 2 deferred with a measured
pricing argument. Production changes that came out of it: **softmax now runs our
`exp2_poly`** (the 0021 mystery was the worker stack — 0xD00 corrupts silently, 0x2000
fixes; accuracy 4× better), **all three eltwise designs run at 8 columns** (LayerNorm's
params broadcast from the mem tile), and the exporter **purges ambiguous cache
candidates before every build** (fifth fail-open: bfp16 GEMM builds shadowed bf16 ones
by mtime and the encode silently regressed to exactly 0026's bfp16 number). Net:
**298.5 seq/s at `1-cos` 2.469e-04** — both the best the project has produced. Proven
but not yet productionised: `rtp=True` (one xclbin serves many GEMM shapes at zero
switch cost, +1.6% overhead) and `epilogue="gelu"` (fused `gelu(A@B+bias)`, 10× more
accurate than the separate path). The naive one-xclbin integration prices to a wash —
the pipelined block-fusion form (§4) is the remaining large lever, and h ≥ 1536 now
*builds* after the stride cap (correctness untested; no goldens).

**Nine new documents indexed** ([`0028`](tasks/0028-research-index-nine-new-papers/TASK.md)),
bringing `OthersResarch/` to 16. Four of them add measurements that all point at layer
fusion (see F1 below). Three change ground rules here: the L1 budget is **63 KB not 64**,
`aie2p` has **5 accumulator registers not 8**, and **Peano silently discards `chess_*`
pragmas**. Two more give us numbers we had been inferring: the **mem-tile port budget**
(6 in / 6 out, 48 total — the limit behind our failed B-reuse and wide-eltwise designs),
and an independent confirmation on the same silicon that **fp32 is emulated and
compute-bound 8:1**, which strengthens the bf16-eltwise direction for M8. Two open-source
artifacts are now directly relevant: **STEEL**, a FlashAttention for XDNA2 written in IRON
and merged into `github.com/amd/iron`, and the **Stationary-B GEMM** algorithm, which the
ICPP paper shows beats the Stationary-C algorithm our stock IRON matmul uses.

- **M0** — scaffold, research index, foundational docs. → [`tasks/0001`](tasks/0001-scaffold-and-research-index/TASK.md)
- **M1 — gate PASSED.** The full native-Windows chain works: IRON design → Peano →
  `aiecc` → xclbin → NPU → trace → cycle counts. bf16 SAXPY: **335 cycles vectorised vs
  541,662 scalar (1,617×)**. Both measurement signals agree (5-bundle ZOL × 61
  iterations + prologue ≈ 335). → [`tasks/0002`](tasks/0002-m1-hello-npu/TASK.md)
- **M2 (single-core)** — bf16→f32 GEMM at MiniLM shapes, traced.
  → [`tasks/0003`](tasks/0003-m2-bf16-gemm/TASK.md)

- **M2 (multi-core)** — whole-array GEMM with tracing added by us.
  → [`tasks/0004`](tasks/0004-m2-multicore-gemm/TASK.md)
- **M3 — gate PASSED.** Pure-numpy fp32 oracle matches HuggingFace to
  **rel_fro ≤ 9.9e-07** at every layer boundary (`1-cos = 2.2e-08` on the final
  embedding), verified against two independent oracles, running in the **iron env with
  numpy only**. → [`tasks/0005`](tasks/0005-m3-python-reference/TASK.md)
- **M4 — gate PASSED.** `.npue` pre-tiled weight container: round-trip **bit-exact**
  (0 of 10,616,832 bf16 elements differ), fusions cost nothing beyond bf16 (0.92× M3's
  baseline). → [`tasks/0006`](tasks/0006-m4-npue-pretiling/TASK.md),
  spec in [`docs/04-model/npue-format.md`](docs/04-model/npue-format.md)
- **M5 (first task)** — pre-tiled GEMM on hardware. `tile_n=48` confirmed at 8 columns
  for all four MiniLM GEMMs; **pre-tiling itself refuted as a performance lever**, and
  M4's `ffn_down` BD claim corrected.
  → [`tasks/0007`](tasks/0007-m5-pretiled-gemm-on-npu/TASK.md)

**M2 headlines:**

- **`--emulate-bf16-mmul-with-bfp16` is worth 5.5×** (25.0 → 137.3 MACs/cycle per core;
  9.8% → 53.6% of peak) but costs ~10⁵× accuracy (1.21e-07 → 1.04e-02). Both paths stay
  selectable until MTEB decides in M8.
- **fp32 accumulation is mandatory** — bf16 output re-rounds every K step.
- **Per-core cost is flat and shape-independent**: 137.3 (1 core) → 142.0 (8) → 141.7
  (16) MACs/cycle, and 0.4% spread across four MiniLM shapes. One tile size serves the
  whole model.
- **But compute is only ~7% of NPU time.** At 512³ on 32 cores, traced compute is
  16.4 µs while measured NPU time is 243 µs. Core scaling is 99.8%; wall-clock scaling
  is 40%. **The array is starved, not slow.**
- **~144 µs fixed cost per dispatch**, and an asymptote of ~3.1 TFLOP/s at K=N=512 even
  as M→∞ — ~5× below compute peak. We are **data-movement bound**.

**M3 headlines:**

- **The oracle exists and is trustworthy.** `reference/encoder.py` (numpy, fp32) matches
  HF at every layer; error grows monotonically with depth (4.7e-07 → 9.9e-07), the
  signature of accumulation order, not of a formula bug. Goldens are in
  `reference/goldens/`, pinned to the checkpoint sha256.
- **bf16 + fp32 accumulate is effectively free on real data**: worst `1-cos` **1.3e-05**,
  sentence-similarity shift ≤ 5.9e-04.
- ~~**bfp16 is a real accuracy risk**~~ and ~~**real activations are 6.0× worse for
  block FP**~~ — **both refuted on hardware in M5**, see below. They were simulation
  artifacts. → [`tasks/0008`](tasks/0008-m5-bfp16-real-data/TASK.md)

**M4 headlines** (two of them corrected by M5 — see below):

- ~~**`ffn_down` is now expressible.**~~ **Refuted.** The whole-array design already
  expressed it; the BD failure was single-core only.
  → [`tasks/0007`](tasks/0007-m5-pretiled-gemm-on-npu/TASK.md)
- **M2's winning `tile_n = 32` is illegal at 8 columns.** `N % (tile_n·n_cols) == 0`
  and `1152/(32·8) = 4.5`. Legal set at 8 cols is {8,16,24,48}; **48** is the largest,
  needs zero padding, and fits L1 (53,248 < 65,536). It would have compiled fine at 4
  columns — exactly where M2 tested.
- **The format is stable across the bf16/bfp16 decision**: only `s` and `t` affect the
  B layout, and both are 8 for both paths. No repack either way.
- **The offline pipeline is free**: round-trip bit-exact, and packed weights on the
  real datapath land at **0.92×** M3's bf16 baseline. Folding `1/√32` into Q costs
  nothing measurable (0.914×, indistinguishable from 1.0 on 4 sentences).
- **Spec bug found in the M0 draft**: `reserved[24]` made the header 72 bytes while the
  same paragraph required 64. It is **16**.

**M5 (first task) headlines** — [`tasks/0007`](tasks/0007-m5-pretiled-gemm-on-npu/TASK.md):

- **`tile_n = 48` confirmed on hardware.** All four MiniLM GEMMs compile, run and
  trace at **8 columns**; n=32 is impossible for three of the four. Traced at 4
  columns: **148.9–149.9 MACs/cycle** (58–59% of peak) on qkv/proj/ffn_up, 140.9 on
  ffn_down — better than M2's 141.7 at n=32.
- **Pre-tiling is NOT a performance win.** Per-core: equal at best, **−11% mean** with
  up to **22% run-to-run spread** (row-major: 0.2%). End-to-end at 8 columns: a **±9%
  wash** across all four shapes. Both descriptors have identical dimension counts,
  identical sizes and identical total length — only strides differ.
- **The cost is in the L3→L2 access pattern, not the sub-tile reorder.** Isolated:
  125.7 vs 126.4. Baking the sub-tile order into the file really is free.
- **Tile order (`k,n` vs `n,k`) changes nothing** — 118.9 vs 118.5. The locality
  hypothesis was wrong.
- **Pre-tiled's *best* runs match row-major exactly** (149.0), and the instability
  scales with B size. It is an intermittent stall, not a throughput ceiling.

**M5 (second task) headlines** — [`tasks/0008`](tasks/0008-m5-bfp16-real-data/TASK.md):

- **bfp16 on real data measured on hardware at last: the distribution barely matters.**
  Real activations × real weights gives **9.02e-03** against **1.06e-02** for uniform in
  the same session — **0.85×**. M3's simulated prediction of 6.0× worse is **refuted**.
  The outlier mechanism is real (within-block range 18.2 vs 10.8) but does not produce
  error on this datapath.
- **Refitting the model on real hardware data gives 7 bits/element, not 5**, and the
  end-to-end cost drops ~10×: `1-cos` **1.8e-03** (was 1.8e-02), similarity shift
  **6.5e-03** (was 1.4e-02). **bfp16 is a genuine 5.5× tradeoff for M8 to decide**, not
  a disqualification. bf16 + fp32 accumulate stays the safe default at `1-cos` 1.3e-05.
- **A silent result-corrupting trap found** — see traps 6b/6c above. It was caught by a
  control with a known correct value, and then **misdiagnosed** as a driver bug before
  a falsifying test showed it was our own missing host→device sync.
  → [`tasks/0009`](tasks/0009-m5-sync-misdiagnosis/TASK.md)
- **0007's pre-tiling conclusion re-verified** under process isolation with repeats: a
  wash (+0.7%, −2.6%). A single isolated run had suggested +12.8%; that was noise.
  Pre-tiled spread is 9–17% against row-major's 2–3% in wall clock too.

**M5 (third task) headlines** — [`tasks/0010`](tasks/0010-m5-b-reuse-and-cost-model/TASK.md):

- **A validated cost model: `t = 150.4 µs + traffic / 33.0 GB/s`** (R² 0.90 on the fit;
  **±1.4%** when validated on an M-sweep it was not fitted to). Traffic counts A
  re-streamed per n-block group, B per row block, and C once.
- **We are dispatch-bound, not bandwidth-bound.** The alarming "8.6–19 GB/s effective
  bandwidth" is the 150 µs fixed cost folded into an average. The *marginal* rate is
  **33 GB/s**, inside the expected 40–60 band. At M=512 the fixed cost is **40–73%** of
  runtime. This independently reproduces M2's ~144 µs — **and explains why pre-tiling
  was a wash**: it optimises something that is not the binding constraint.
- **Batching is the cheapest win available: 2.05×** (1.85 → 3.79 TFLOP/s for `ffn_down`
  from M=512 to M=4096) with no code change at all.
- **B reuse is blocked by two hardware limits**, at two levels: ObjectFifo depth maps
  1:1 to mem-tile buffer descriptors (ceiling **6 tiles at 4 cols, 4 at 8**, against a
  slice needing 48/24), and the one-big-object workaround hits *number of input DMA
  channel exceeded* on a **core** tile. It needs a core-side redesign, not a fifo
  change — and the model prices it at **1.26× (M=512) to 1.68× (M=4096)**.

**M5 (fourth task) — GATE PASSED** — [`tasks/0011`](tasks/0011-m5-first-op-validated/TASK.md):

- **The first real encoder op ran on the NPU and was validated against the goldens.**
  Layer 0 QKV, `[256,384]×[384,1152]`, weights straight out of `.npue`, activations
  from the M3 golden: **rel_fro 1.507e-03**, worst per-row `1-cos` **2.6e-06**. bfp16
  on the same op: 9.898e-03, independently reproducing [`0008`](tasks/0008-m5-bfp16-real-data/TASK.md)'s
  9.02e-03 through a different script.
- **The `1/√32` fold is confirmed correct**: Q/K/V blocks all sit at the same error
  level (1.38 / 1.44 / 2.01e-03). A fold into the wrong block would be O(1) off.
- The full chain now runs end to end on real data: HuggingFace → M3 goldens → M4
  `.npue` → NPU → oracle.

**M5 (kernels) and M6 (full encode) headlines:**

- **M5 GATE** — every per-layer GEMM validated against the goldens on hardware:
  `qkv` 1.5e-03, `attn_out` 2.4e-03, `ffn_up` 1.4e-03, `ffn_down` 2.1e-03 (bf16).
  → [`tasks/0011`](tasks/0011-m5-first-op-validated/TASK.md),
  [`0012`](tasks/0012-m5-all-layer-gemms/TASK.md)
- **bfp16 damage is concentrated in `ffn_down`** (`1-cos` 8.9e-03, 34–120× the other
  three) because its input is **post-GELU** — heavy within-block dynamic-range tail.
  Mixed precision is priced at 2.21× against 5.5×, so it is only worth it if `ffn_down`
  alone breaks the MTEB budget.
- **Write our own kernels.** IRON's GELU lands at 1.33e-02; the error is `aie::tanh`
  itself (~1% accurate), not the LUT or the tanh formula. Our polynomial kernel —
  `max(x,0) + poly(min(|x|,4))`, exploiting that `GELU(x)-max(x,0)` is **even** — gets
  **4.31e-03 with no transcendental call at all**.
  → [`tasks/0013`](tasks/0013-m5-first-eltwise-kernel/TASK.md)–[`0015`](tasks/0015-m5-gelu-polynomial/TASK.md)
- **AIE `vector<float>` is full IEEE fp32** (24-bit mantissa, add and multiply),
  measured directly. fp32 LayerNorm and softmax remain achievable.
  → [`tasks/0016`](tasks/0016-m5-fp32-probe/TASK.md)
- **M6 GATE — a full encode runs on the NPU.** 24 GEMMs + 6 GELU dispatched per
  encode; `1-cos` **2.05e-05** vs HuggingFace (1.09e-05 with host GELU), bfp16
  2.35e-03. **M3's simulation predicted both**, to 1.3× and 1.08% respectively.
  → [`tasks/0017`](tasks/0017-m6-full-encode/TASK.md)

**Next: M7 — C++ runtime.** Carry forward:

1. **The 150 µs per dispatch is the whole performance story**, and a full encode now
   pays it **30 times**. Fusing whole layers is the lever, and the op count is real
   rather than hypothetical.
2. **Batch.** 2.05× is available with no code change.
3. **LayerNorm and softmax kernels** — no precision obstacle remains, they simply have
   not been written. Both must be fp32 per `docs/04-model`.
4. **Attention on the array** needs `head_dim = 32` handling: pad to 64, or fold two
   heads into one 64-deep tile. Worth 5.3% of FLOPs, so it is not urgent.
5. **M8's accuracy half can start now** — the bfp16-vs-bf16 decision needs MTEB, and
   both configurations produce embeddings today.
6. **Stop optimising B layout.** Pre-tiling is a wash, confirmed under isolation.

**Measurement notes added in M5:**

- A single per-core number is not a measurement on the pre-tiled path (up to 22%
  spread). Always `--repeat` and report mean/range. The same holds for wall clock:
  a one-shot isolated bench suggested +12.8% that three repeats erased.
- **Every measurement sweep needs a control with a known correct value.** Trap 6b
  produces plausible wrong numbers silently, and a sweep that only measures the
  quantity of interest cannot detect it.

Roadmap: M0 scaffold → M1 hello-NPU → M2 traced bf16 GEMM → M3 Python reference +
goldens → M4 offline weight pre-tiling → M5 encoder ops on NPU → **M6 full Python
encode (bankable)** → M7 C++ runtime → M8 benchmark.

## Target model

**all-MiniLM-L6-v2** and **bge-small-en-v1.5**, both running. Same tensor names,
same GEMM shapes, so the compiled designs and the `.npue` tiling are shared --
but bge-small has **12 layers, not 6** and pools **CLS, not mean**, so it is not
the byte-identical drop-in earlier docs claimed. Those two facts are data the
runtime reads (container config, `1_Pooling/config.json`), not constants.

6 layers, hidden 384, FFN 1536, 12 heads × head_dim 32, vocab 30522, absolute learned
positions, exact-erf GELU, post-LayerNorm (eps 1e-12), mean pooling, L2 normalized.

384/1152/1536 all divide 64 cleanly; only `head_dim=32` is awkward, and attention is
just 5.3% of work at seq 128, so it barely matters.

Full analysis incl. numerical landmines: [`docs/04-model/`](docs/04-model/README.md)

## The three findings that drive the design

**F1 — Per-dispatch overhead dominates, not kernel throughput.** The NPU has been
measured *losing* to the iGPU at 256-token prompts. Fusing five dispatches into one
gave 2.24×. → Batch, keep one resident `.xclbin`, fuse whole layers, target **one
dispatch per encoder layer**.

> **Sharpened in [`0024`](tasks/0024-m7-dispatch-cost-anatomy/TASK.md):** the
> expensive thing is not the dispatch (~150 µs) but **changing design between
> dispatches** (~55 µs + 286 µs/column, so 630 µs–2.4 ms). Our encoder changes design
> on all 49 of its dispatches. This is why "fuse whole layers" is the lever — the
> price of *not* fusing is now measured. Note that operator-major reordering cannot
> substitute: layer *L*+1 depends on layer *L*, so the dispatches are a chain, not a
> set. Batching and fusion are the two levers that survive that.
>
> **Four more independent measurements, indexed [`0028`](tasks/0028-research-index-nine-new-papers/TASK.md).**
> AMD, on *our* SKU, cut **15 dispatches per layer to 3** by merging the pre-attention
> (norm + QKV + RoPE) and post-attention (out-proj + FFN) blocks
> ([2606.07586](https://arxiv.org/abs/2606.07586)). STEEL measured fused attention at
> **22.8×** over the layer-by-layer IRON equivalent, crediting one design load instead of
> several plus never materialising intermediates off-chip
> ([2607.09385](https://arxiv.org/abs/2607.09385)). ARIES beat the vendor NPU overlay
> **1.24× with scalar, unvectorised code on fewer cores**, purely by handing intermediates
> between adjacent tiles' L1 (ARIES). And Estévez hit
> **95% of peak on all 32 tiles** with a design that dispatches once and moves no data
> (peak TOPS) — the control experiment showing
> the array itself is not what limits us. **Fusing whole encoder layers is the
> best-evidenced unclaimed lever we have.**

**F2 — Batching is mandatory.** 21.3 MB of bf16 weights over ~120 GB/s = 0.18 ms,
which *equals* the theoretical compute time for one sequence. At batch 1 we are
memory-bound and the FLOPs are irrelevant.

**F3 — Attention is not the encoder bottleneck.** AMD measured BERT: full attention
folding bought **1.4%** end-to-end. Effort goes to projection/FFN GEMMs and the
dispatch path.

**Plan against 14.7 TOPS bf16 attainable, not 50 TOPS marketing.** We are
**bandwidth-bound** (~40–60 GB/s reaches the NPU) — optimise measured bandwidth
utilisation, not TOPS.

## Working style

- Prefer reading [`docs/`](docs/00-overview.md) and
  `research/` over re-deriving. That is what they are for.
- When something is learned, write it down in `docs/` (durable truth) and record the
  session in `tasks/` (what happened that day). Keep the two distinct.
- Update this file when the current state or the ground rules change.
- Read-only reference trees, do not modify: `C:\dev\mlir-aie\`,
  `C:\Users\vegar\Documents\GitHub\FastFlowLM\`.
