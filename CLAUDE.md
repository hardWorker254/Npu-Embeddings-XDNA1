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

   **It is two files now (2026-08-23).** `OPEN-THREADS.md` holds only what is
   open — **1 thread, 163 lines as of 2026-08-26**, down from 20/765 when the
   split was made. Answered, retired and superseded threads live
   in [`research/CLOSED-THREADS.md`](research/CLOSED-THREADS.md), **verbatim**,
   because rule 3b's point is that the refuted claims and the measurements that
   killed them are the valuable part. They were 64% of the register, and a file
   nobody finishes cannot be an authority.

   **And something checks it now: `python tools/check_register.py`.** There was
   a *third* instance of the failure above on 2026-08-23, and the first where
   the register misled its own reader — T28 ran to 308 lines without ever
   naming `tasks/0062`, which had built and passed the thing T28 still called
   open, and an hour went into re-deriving it. The check that catches this is
   cheap: **a task whose title is about a thread, that the thread does not link
   back**. Filtering on the task's title rather than its body takes it from 41
   hits to 1, and it would have caught 0062. Run it before a release; it exits
   non-zero.

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
| IRON | `C:\dev\mlir-aie\ironenv` — mlir-aie 1.4.x (tracks upstream `main`, currently `1.4.2.dev16+g7e00b57`; wheel-based install, not a source build — see [`.claude/skills/update-mlir-aie`](.claude/skills/update-mlir-aie/SKILL.md) before touching this), Peano 21.0.0.2026080301, **Python 3.13.15** |
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
| `C:\Users\vegar\.conda\envs\iron` | Py 3.13.15 base. Its **own** `Lib\site-packages` holds almost nothing; numpy and onnxruntime come from the **user** site-packages (`%APPDATA%\Roaming\Python\Python313`) — which matters, see below. |
| `.venv-ref` (repo root, **created in M3**) | venv off the conda `iron` interpreter with `--system-site-packages`. Adds transformers, safetensors, sentence-transformers, huggingface_hub, mteb — **and torch, which it now OWNS rather than inherits.** Recreate: `& "C:\Users\vegar\.conda\envs\iron\python.exe" -m venv --system-site-packages .venv-ref`, then `.venv-ref\Scripts\python -m pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu` |

Golden data crosses env boundaries **as files** (`.safetensors`), never as imports.
A `pip install` accident must not break the toolchain that took the most work to get running.

**The inheritance is fragile and has already failed once (2026-08-23).** The
release sweep died three stages in with `ModuleNotFoundError: No module named
'torch'` from `.venv-ref` — an interpreter that had run MTEB successfully two
hours earlier. `--system-site-packages` inherits the base's own
`Lib\site-packages` and nothing else, and that directory had been emptied.

**What made it hard to see:** bare `python` still imported numpy and
onnxruntime, because those live in the **user** site-packages, which a venv
deliberately ignores. So `python -c "import numpy"` succeeded while `.venv-ref`
had nothing — two different search paths, one of them invisible from the other.
When a venv cannot find a package the base "has", check
`site.getusersitepackages()` before anything else.

torch is now installed into `.venv-ref` itself: 114 MB of duplication against a
dependency on a directory nothing in this repo owns. `ironenv` was verified
untouched throughout — it is the toolchain, and it is the thing not to fix.

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

   **Trap 2b may have a further depth — a strong hypothesis, NOT YET
   CONFIRMED ON HARDWARE.** Reading `mm.cc` / `mmul_bf16_bf16.hpp` /
   `aie2p_srs.h` found that the emulated bf16 matmul's own
   `swap_rounding(conv_even)`/`set_rounding` pair around its k-loop calls the
   **ambient** `to_v64bfp16ebs8`, not the `_conf` variant that actually
   saves/sets/restores `crrnd` — so on the read evidence the save/restore
   looks like dead code an optimiser can remove (compiled object: zero
   `rnd`/`round` instructions in the matmul object; the *same* toolchain
   correctly emits `mov crrnd, #0xc` for `narrow_f32_bf16.cc`'s explicit
   `set_rounding` call, and never restores it). If so, every A/B-tile bfp16
   quantisation inside the emulated matmul runs under whatever rounding mode
   a **prior kernel on that physical core** left behind, not the `conv_even`
   the source asks for — which would make T26's still-open 6.6× anomaly
   (bf16-C outperforming fp32-C) a rounding-state leak between kernels
   sharing a core, one level deeper than 2b's known cross-kernel leak. Two
   hardware ablations are proposed and neither has run.
   → [T26](research/OPEN-THREADS.md#t26), [`0098`](tasks/0098-t26-kernel-source/TASK.md)
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

   **The shim DMA has neither compression nor padding hardware, and padding
   exists ONLY on the mem tile.** Read straight out of the vendor's own
   register tables, `xaie2pgbl_reginit.c` — the three `XAie_DmaMod` structs at
   lines 1658 / 1896 / 2149:

   | | `.Compression` | `.Padding` |
   |---|---|---|
   | `Aie2PMemTileDmaMod` | **AVAILABLE** | **AVAILABLE** (1667) |
   | `Aie2PTileDmaMod` (compute tile) | **AVAILABLE** (1904) | UNAVAILABLE (1905) |
   | `Aie2PShimDmaMod` | UNAVAILABLE (2157) | UNAVAILABLE (2158) |

   These are **hardware capability flags on this exact chip**, not mlir-aie
   omissions, so no software reaches around them. Note the two features do
   *not* have the same footprint: compression reaches both on-die DMA kinds,
   padding **only** the mem tile — which independently corroborates note
   0007 §1.1's correction that `pad_dimensions` is mem-tile-only, and is what
   keeps [T38](research/OPEN-THREADS.md#t38) alive. Since 100% of DRAM
   traffic crosses the shim, neither compression nor padding can ever reach
   the leg a bandwidth-bound cost model prices — confirmed for compression,
   worth at most **5.8%** of the four production GEMM dispatches even
   granted for free on real weight bytes (a ratio never actually measured;
   the attempt was blocked by an `iron.jit` caching bug, trap 7d below).
   → [T5, CLOSED-THREADS.md](research/CLOSED-THREADS.md#t5),
   [`0090`](tasks/0090-t5-dma-compression/TASK.md)

   **`aie::accum<accfloat,N>` on aie2p is a plain 32-bit-per-lane
   accumulator** (`accum_native_types.hpp`'s `AccumClass::FP, Bits=32`), not
   an extended-precision intermediate — relevant when budgeting the 5
   accumulator registers above. → [`0098`](tasks/0098-t26-kernel-source/TASK.md)
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
7d. **`iron.jit`'s cache key never inspects a generator's own module
   globals.** `_create_function_cache_key()` hashes the call's
   positional/keyword arguments plus `(device, full_elf)` — nothing in the
   key derivation reads the generator function's `__globals__`. A
   module-level constant a generator reads via `LOAD_GLOBAL` (not a
   `CompileTime[T]` kwarg) is therefore **invisible to the cache even with
   `use_cache=False`**, while `.as_mlir()` (which bypasses the cache
   entirely) *does* show the change — so the generator text looks edited
   and the compiled binary silently is not. This is the **sixth** instance
   of this project's "stale binary fails open" class (0030, 0053, 0054,
   0083, 0087) and the first found inside IRON itself rather than in this
   project's own marker logic. → [T5](research/CLOSED-THREADS.md#t5),
   [`0090`](tasks/0090-t5-dma-compression/TASK.md),
   [`0092`](tasks/0092-t28-relay-bf16-output/TASK.md) Part 4 addendum
7. **A fully-packed 8-column design cannot be core-traced** — adding one trace flow
   exhausts routing (`Unable to find a legal routing`, or `max number of packet IDs
   reached` with `--packet-sw-objFifos`). Traceable widths are 2 cols at
   `(trace_col=1, egress_shim_col=1)` and 4 cols at `(0,0)`. Measure **per-core cycles
   at 4 columns (traced)** and **throughput at 8 columns (wall clock)**.
   → [`docs/05-measurement/`](docs/05-measurement/README.md)
8. **IRON's declared argument types are cosmetic at two separate boundaries,
   and neither is checked against what actually runs.**
   - **A `Kernel`/`ExternalFunction`'s `arg_types` shapes only the MLIR
     call-site declaration the caller sees — never the linked object**,
     which is compiled straight from the C++ source's own signature.
     Declaring a Buffer bf16 on the Python side does not make a
     `float*`-signatured kernel treat it as bf16; it silently
     double-writes. Found via objdump: a 2,048 B buffer overflowed by
     4,096 B of `float` vector stores, from a kernel whose Python-side
     `arg_types` said bf16.
   - **A design's host-facing argument dtype must independently track the
     internal ObjectFifo pipeline's dtype** — nothing checks that
     `Runtime(sequence, [A_ty, B_ty, Y_ty, ...])`'s declared types agree
     with what the pipeline actually produces. A hardcoded fp32 `Y_ty`
     against a bf16 pipeline configured the shim DMA to move 4,096 B
     against a host buffer XRT had actually allocated at 2,048 B: the
     mem-tile side signals completion after 2,048 B, the shim side waits
     forever for bytes that will never arrive. **Compiles clean, hangs
     with no diagnostic** — the same fail-open shape as traps 6b/6c, one
     layer further out, at the design's own I/O boundary rather than
     inside the pipeline.

   Lesson from finding the second one: when every on-chip mechanism
   bisects clean, check the compiled design's outer I/O signature against
   what the host actually allocates, not just what the internal pipeline
   declares. → [`0092`](tasks/0092-t28-relay-bf16-output/TASK.md) Parts 1
   and 4

## Current state

**Six models run end to end on the NPU, in C++, no Python in the process.**
M0–M9 and M12–M13 are done: the tokenizers ship, the MTEB gate passes, energy
is measured, and the runtime reads its geometry from the container rather than
having it compiled in.

**The datapath is now a per-model decision, and the runtime states which one it
ran.** Five of six models run the **bfp16-emulated MMAC with bf16 C**;
`bge-small-en-v1.5` alone stays on plain bf16, having failed the MTEB gate at
−0.5010 against the −0.5 line, bit-reproducibly. The user's rule is *adopt
wherever MTEB passes*
([`0103`](tasks/0103-t23-bfp16-all-models/TASK.md)–[`0104`](tasks/0104-adopt-bfp16-per-model/TASK.md)).
Two guards exist because that decision created the need for them, and both
follow the same discipline — **report the value you read, never the intention**:

* `design.json` records `emulate_bfp16`, and `design_fits()` **refuses** a
  datapath the model was not adopted for. Without it bge-small and MiniLM, which
  share a geometry and a `b_layout_hash`, were separated only by alphabetical
  sort order.
* `toolchain.json` beside it records which mlir-aie, Peano and git HEAD built
  the design ([`0106`](tasks/0106-toolchain-provenance/TASK.md), T39). A design
  predating either field reads **`UNRECORDED`**, never a guess.

**And the host lever is largely spent, so the array is the larger piece again.**
[`0108`](tasks/0108-fuse-epilogue-bfp16/TASK.md) ported the host-side epilogue
fusion to bf16/bfp16 (it had been int8-only, gated on `a_elem_bytes == 1`) for
**1.153–1.340×**, bit-identical. Array time did not move — `wait (hardware)`
within −0.1% to +2.2% — but its **share** of wall clock rose on every model
(bge-large 46.4% → 56.7%), and the array-infinite ceiling with it (1.87× →
**2.31×**) ([`0109`](tasks/0109-fused-ratio-energy/TASK.md)). **F1 still holds,
but the cheap host-side half of it is now collected.**

| `--model` | arch | hidden | layers | notes |
|---|---:|---:|---:|---|
| `all-MiniLM-L6-v2` | 0 | 384 | 6 | smallest; head_dim 32 keeps attention off the array |
| `bge-small-en-v1.5` | 0 | 384 | 12 | MiniLM's width, twice the depth |
| `bge-base-en-v1.5` | 0 | 768 | 12 | the best geometric fit for XDNA2 |
| `bge-large-en-v1.5` | 0 | 1024 | 24 | highest quality; `tile_n` 32 at bf16, 64 at int8 |
| `nomic-embed-text-v1.5` | 2 | 768 | 12 | RoPE + gated SwiGLU; needs a task prompt |
| `embeddinggemma-300m` | 1 | 768 | 24 | MQA + RoPE + GeGLU; gated repo, needs a token |

**Those six are built in; more can be added without recompiling.**
`npuembeddings add <org/model> [<sha256>]` reads a HuggingFace repository's own
`config.json`, derives its geometry, computes the largest legal `tile_n`,
checks an installed design actually serves it, and writes
`models/catalog.json` — a **user** catalogue merged *after* the built-ins and
forbidden from shadowing them (their pins are the reason `serve bge-base` is
safe). Finetunes of the six are the intended case, and they need no new design.
**Omitting the sha256 is allowed and means the weights are never verified** —
warned at add, at fetch, and on **every run**. → [`0076`](tasks/0076-m13-add-model/TASK.md)

**int8 is a second datapath, behind a flag, and bf16 is untouched.**
`aie2p` has native int8 `mac_dims` (8,8,8) against bf16's (4,8,8), and the bf16
path runs on the **fp32 vector** unit while the MMAC unit sits idle
([`0049`](tasks/0049-m9-t16-iteration-anatomy/TASK.md)). Measured on all four
production shapes, pre-tiled, traced: **5.5–7.7× the bf16 datapath, bit-exact**
([`0077`](tasks/0077-m13-int8-gate/TASK.md)). Accuracy needs SmoothQuant —
naive W8A8 fails the 2e-03 gate at 2.864e-03; with statically-calibrated
smoothing at α=0.5 it passes at **1.417e-03 on hardware**
([`0078`](tasks/0078-m13-int8-accuracy/TASK.md)).

**End to end, the 0.4.0 release sweep measures 1.71× (MiniLM) to 2.39×
(bge-large)** ([`0085`](tasks/0085-m13-release-sweep/TASK.md); full table in
`docs/CURRENT_STATUS.md`, not reproduced here per rules 1/6). That supersedes
**1.10×**, which was [`0079`](tasks/0079-m13-int8-why-only-1.1x/TASK.md)'s
first measurement and genuinely the catalogue's worst case at the time
(bge-large was already 1.44× then, because its array share of wall clock was
87% against MiniLM's 61%).

0079 also measured, and **refuted**, its own headline explanation —
*"the int8 quantisation pass costs more than the bf16 conversion it
replaces."* Measured: quantise/convert A is **1.04×** bf16's conversion cost,
essentially the same. The real reason the array's 5.5–7.7× arithmetic gain
(0077) reached the encode as only 1.10–1.44× is the **C drain**: int32 C is
still 4 bytes, exactly like fp32, so int8 moves no fewer output bytes despite
7× faster MACs — at MiniLM's 679 MB of C per encode that alone explains most
of the per-dispatch time 0048's arithmetic-only cost model left unaccounted
for.

The follow-on work closed that gap. [`0080`](tasks/0080-m13-int8-traffic-bound/TASK.md)–[`0082`](tasks/0082-m13-fused-ffn-epilogue/TASK.md)
narrowed C to bf16 on-core and fused the host's two multi-pass epilogue
chains (dequantise→activate→quantise, and add→norm→quantise) into one
L1-resident pass per row — verified byte-for-byte identical to the unfused
path — removing 3.45–4.29× of host memory traffic at a measured **~60 GB/s**;
[`0081`](tasks/0081-m13-int8-everywhere/TASK.md) found bge-large had been
shipping at half its legal `tile_n` (see below). The value int8 delivers was
never in question — **the array stops being the bottleneck, after which the
host side and fusion are the whole game** — but 0079's first measurement
caught it before that follow-on work existed.

Select it with `--int8` on both `tools/export_gemm_rtp.py` and
`tools/pack_npue.py`. The dtype is **data**: `a_dtype` in `design.json` and in
the container, and the two carry different `b_layout_hash` values, so a
mismatched pair is refused rather than read as garbage.

**`arch` is a field in the `.npue` header, not a description.** It names which
forward pass the runtime must run — 0 = BERT (absolute positions, GELU,
post-LN), 1 = Gemma3 (RMSNorm ×4, MQA, per-layer RoPE, GeGLU), 2 = nomic
(RoPE, SwiGLU, post-LN). Arch 1 and 2 deliberately reuse **BERT's tensor
names** so the packer and the whole NPU dispatch path work unchanged — which
means names cannot tell them apart, and running one through the wrong encoder
would return plausible embeddings for the wrong model. So `arch` is a
**whitelist** in `encoder_implemented()` / `set_model_shape()`: an
unimplemented architecture refuses rather than guesses.

**An input that does not fit is now an ERROR, not a shorter input.** Until
[`0110`](tasks/0110-refuse-silent-truncation/TASK.md) both tokenizers cut at
`max_len` and said nothing, and `n_tokens` was capped by construction — so a
truncated text was indistinguishable from one that fitted, right through to
`usage.prompt_tokens`. It is the fail-open class of traps 6b/6c/7c/7d with the
worst blast radius yet, because what is wrong is the **answer**: a truncated
text still returns a correctly shaped, correctly normed, deterministic vector,
and two documents sharing an administrative preamble truncate to
**byte-identical** vectors (measured, 0110). The runtime now refuses, naming
the input's index and its real token count, and `serve` answers **HTTP 400
`invalid_request_error`**. `--allow-truncation` restores the old behaviour and
warns once per run. `--tokenize` deliberately still truncates — it exists to
diff against HuggingFace, which truncates too — and reports a count on stderr.

**The task prompt is a per-REQUEST field, and there is no default left.**
Until [`0118`](tasks/0118-prompt-name-per-request/TASK.md) the prompt was
resolved once per process from `--prefix`, so one `serve` could answer only one
kind of query — a RAG deployment needing `search_query` *and* `search_document`
had to run two servers, each holding an `hw_context` on a shared NPU. It is
`"prompt_name"` in the POST body now, **required** for a model that has a
prompts table, answered with **HTTP 400 listing the valid names** when missing,
unknown, or not a string; `GET /health` advertises `prompt_names` so a client
never has to provoke the 400 to discover them. `"prompt_name": ""` means no
prompt; passing the field to a model that has none is also a 400, so a client
sweeping one config across the catalogue breaks loudly.

The same task removed **the last silent default in the runtime**:
EmbeddingGemma's `--prefix` defaulted to `"document"`, one of *fourteen*
prompts, on a checkpoint that names no default at all. `--prefix` survives on
`embed`/`--bench` (a file of texts has no per-request anything) but is required
there too, and `serve` now **rejects** it rather than ignoring it.
`prompt_default` stays in the container as advisory metadata — harnesses use it
to choose which prompt to exercise — and nothing applies it. Bit-identical
output on all three model families for the same effective prompt.

**And `seq` is an export parameter now, not a constant.** `SEQ = 64` became
`tools/export_gemm_rtp.py --seq`. The reason it is cheap: seq is **not
compiled into anything**. It enters only as `M = batch * seq`, the instruction
streams know just `M`/`K`/`N`, and the runtime inverts the split at load
(`batch = design.M / design.seq`) — so `batch 128 × seq 64` and
`batch 16 × seq 512` are the same M and the same array work.

**Measured at seq 256 ([`0112`](tasks/0112-t40-seq256-nomic/TASK.md)), and the
array really does not move: −0.9% on the same M.** Per token, seq 256 costs
**1.221×**, and every bit of that is host-side — attention **2.487×** (a pure
O(seq²) term predicts 4×; longer rows amortise the AVX2 loop), softmax 1.31×,
while the per-token buckets stay flat. The balance inverts: array share of wall
76.3% → 62.0%, attention 16.5% → **33.6%**, and `attn` overtakes `bias` as the
largest host bucket.

**And measured again at 512 ([`0113`](tasks/0113-t40-seq512-close/TASK.md)),
which is where the answer lives.** Across an **8× sequence range the array does
not move: −2.0%**, on identical `M` and identical dispatch counts. Attention
scales 2.49× for 4× (64→256, sublinear — longer rows amortise the AVX2 loop)
then **2.04× for 2× (256→512, textbook linear)**: the small-seq discount is
spent by 256. Per token seq 512 costs **1.605×**, and:

> **Host attention overtakes the whole array at seq ≈ 470** (nomic, constant
> `M = 8192`). At 512 attention alone is 52.1% of wall clock against the
> array's 46.6%.

So **F3's "2–5% of the work" is a seq-64 number and must not be quoted above
it.** Whether folding attention onto the array is *worth* it is
[T42](research/OPEN-THREADS.md#t42)'s question —
[`0114`](tasks/0114-t38-pad-dimensions-probe/TASK.md) removed the geometry
blocker by proving mem-tile `pad_dimensions` works **exactly, on all 8
columns**, so 0043's `cols ≤ 4` is a property of a design that declines to pad,
not of the hardware.

**Three traps found along the way.** The 256 cap is the container's
`max_seq_len` **config field**, not the position table, so nomic needs a repack
above 256 like everything else. The **validation goldens are seq-shaped**, so
`--bench` refuses a new seq until they are regenerated — now a flag rather than
an edit ([`0116`](tasks/0116-t41-golden-seq-flag/TASK.md): `--seq` on all three
golden makers). And a long-sequence design is a **bad instrument for short
requests**: `use_tier()` rounds up, so the smallest tier's token footprint is
what bites (2048 slots against seq 64's 256) — measured at **5.8× for a single
text**. → [T40, closed](research/CLOSED-THREADS.md#t40)

**The production architecture.** The NPU does **pure GEMM** — four shapes per
layer as instruction streams over **ONE xclbin in ONE hw_context**, so an
encode performs zero design switches. LayerNorm/RMSNorm, softmax, GELU/GeGLU,
RoPE and attention all run on the host in fp32, each measured faster *and* more
accurate than its NPU dispatch at these widths (tasks/0032). The datapath
contract is **bf16 in, fp32 out**; `--c-bf16` exists, is measured, and is not
the default.

**Where the numbers are.** Throughput, accuracy, CPU ratios and energy live in
[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) and
[`docs/06-performance.md`](docs/06-performance.md), from a whole-catalogue
sweep run in one session with one protocol (`tools/release_benchmark.ps1`).
**Do not quote a number from a task log or from `docs/history.md` as if it were
current** — those are dated, and several are superseded.

**Where the open questions are.**
[`research/OPEN-THREADS.md`](research/OPEN-THREADS.md) is the authority, per
rule 3 — and it now holds *only* what is open. A stale "untested" in an old
task is not evidence that a thread is live; the register is. **How** something
was settled is in [`research/CLOSED-THREADS.md`](research/CLOSED-THREADS.md),
and `tools/check_register.py` checks the two against the task logs.

**As of 2026-08-26 there are THREE live threads**, and all three are design
tasks with a price and an explicit trigger rather than open questions about the
hardware:
[T42](research/OPEN-THREADS.md#t42) — fold attention onto the array for
*long-sequence* designs; [T43](research/OPEN-THREADS.md#t43) — the missing
byte-level BPE tokenizer, which is the gate on the whole ModernBERT/Qwen/Mistral
generation of encoders; and [T44](research/OPEN-THREADS.md#t44) — which encoder
joins the catalogue next, with four candidates already priced against this
project's own geometry gates. The other 43 are closed. The next piece of work
still comes from `docs/CURRENT_STATUS.md` or from the user as often as from
here.

**Where the history is.** [`docs/history.md`](docs/history.md) holds the dated
update blocks that used to live here — 1,027 lines of them, moved verbatim in
0075 because they were 76% of a file loaded into every session and mostly
superseded. They are kept, not deleted, because the refuted claims are in them
(the bandwidth cost model, the "missing 4,500 cycles", B-reuse, the pre-tiling
win) each with the measurement that killed it — rule 3b. Read them as history.

**When you finish a piece of work**, update this section only if the *ground
rules or the current truth* changed. The session's diary goes in `tasks/`, the
durable facts in `docs/`, the open questions in the register. Adding another
dated block here is what produced the 1,027 lines.

## What geometry the array actually wants

There is no single "target model" any more — six run, listed above. What
matters when a *seventh* is proposed is whether its shapes tile, and that
question has a fixed answer, checked in `gemm_pretiled.py`'s own assertions:

```
M % (m · n_aie_rows) == 0     M = batch·seq, m = 64, rows = 4  -> M % 256 == 0
K % k == 0                                    k = 64
N % (n · n_aie_cols) == 0     n = 48, cols = 8  -> every N a multiple of 384
2·(m·k·2 + k·n·2 + m·n·4) < 64512              the 63 KB L1 budget
```

`tile_n` is the parameter that moves, and **it now depends on the datapath as
well as the width**: **48** for hidden 384/768 (MiniLM, bge-small, bge-base,
nomic, Gemma) at either dtype — N there never divides 512, so 64 is illegal
regardless. bge-large (N ∈ {1024, 3072, 4096}) is the case where the L1
budget itself is datapath-dependent, since `in` in trap 3's inequality halves
for int8: at **bf16**, `(64,64,64)` needs 65,536 B and is **illegal**, so
bf16 bge-large ships at **32**. At **int8**, the same tile needs only
49,152 B of the 63 KB budget, and 1024/3072/4096 all divide 64·8 = 512, so
**int8 bge-large ships at `tile_n = 64`** — free, no padding needed, bit-exact
against exact int32 accumulation, and **1.366× faster** on the shapes it
touches (`tools/pack_npue.py --int8 --tile-n 64`,
[`0081`](tasks/0081-m13-int8-everywhere/TASK.md) §1).

**When a width does not divide, pad it — do not lower `tile_n`.** MQA gave
EmbeddingGemma a 256-wide K/V that capped `tile_n` at 16 across the whole
design, a ~3× iteration tax. Appending zero columns to the fused Q|K|V (1280 →
1536) removes the cap for **4.4% of the iterations it saves**, and it is
*exact*: zero columns of B give exactly-zero columns of C. B is pre-tiled
offline, so the padding costs the packer nothing.
→ [`docs/04-model/npue-format.md`](docs/04-model/npue-format.md) `arch = 1`

**What is awkward and why it is tolerated:** `head_dim` 32 (MiniLM, bge-small)
and 256 (Gemma) both fail to tile for the attention GEMMs, so attention runs on
the host in every model. F3 prices that at ~2–5% of work, and
[`0043`](tasks/0043-m9-attention-geometry/TASK.md) has not found it worth the
fight.

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
>
> **CLAIMED AND CLOSED, 2026-08-26 — and not the way this paragraph expected.**
> The two threads that carried the on-array version of it (device-resident
> intermediates, the pipelined relay) were built to correctness at production
> tile width ([`0092`](tasks/0092-t28-relay-bf16-output/TASK.md)), priced twice,
> and **RETIRED** ([`0117`](tasks/0117-t3-t28-repricing-retire/TASK.md)). What
> collected the prize instead was **fusion on the host**: the multi-pass GEMM
> epilogues, fused into one L1-resident pass per row, for a measured
> **1.153–1.340×** bit-identical ([`0108`](tasks/0108-fuse-epilogue-bfp16/TASK.md)).
> Re-priced afterwards, the relay's remaining ceiling is **1.095×–1.160×**, and
> even an upper bound crediting it with the *entire* transport bucket is
> **1.242×** on bge-large — below what the host rewrite already banked. The
> paragraph above is kept because its evidence is real and its *direction* was
> right; what it got wrong is which side of the PCIe boundary the fusion had to
> happen on. Reopen only if the host side grows back to dominate — at seq 512 it
> does, but there the dominant host work is attention, i.e.
> [T42](research/OPEN-THREADS.md#t42), not this.

**F2 — Batching is mandatory.** 21.3 MB of bf16 weights over ~120 GB/s = 0.18 ms,
which *equals* the theoretical compute time for one sequence. At batch 1 we are
memory-bound and the FLOPs are irrelevant.

**F3 — Attention is not the encoder bottleneck — AT seq 64.** AMD measured BERT:
full attention folding bought **1.4%** end-to-end. Effort goes to projection/FFN
GEMMs and the dispatch path.

> **The premise is sequence-dependent, and the qualifier is not optional
> ([`0113`](tasks/0113-t40-seq512-close/TASK.md)).** Attention is O(seq²) per
> sequence, so O(seq) per token, while array work at constant `M` is flat.
> Measured on nomic at identical token counts — host attention as a share of
> wall clock **16.5% / 33.6% / 52.1%** at seq **64 / 256 / 512**, against an
> array share falling 76.3% → 62.0% → **46.6%**. **Host attention overtakes the
> whole array at seq ≈ 470.** F3's *conclusion* (leave attention on the host)
> may well survive — that is [T42](research/OPEN-THREADS.md#t42)'s question, and
> [`0114`](tasks/0114-t38-pad-dimensions-probe/TASK.md) removed the geometry
> blocker by proving mem-tile padding works on all 8 columns. But **"2–5% of the
> work" is a seq-64 number and must not be quoted above it.**

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
