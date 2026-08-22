# CURRENT STATUS

*Last updated: 2026-08-21, after [`tasks/0073`](../tasks/0073-m13-release-benchmarks/TASK.md).*

> **Six models, and a second architecture on the array.**
> [`0068`](../tasks/0068-m13-nomic-spike-and-oracle/TASK.md)–[`0071`](../tasks/0071-m13-nomic-shippable/TASK.md)
> added **nomic-embed-text-v1.5** (Apache-2.0) as `arch=2`: RoPE instead of
> absolute positions, a gated SwiGLU FFN instead of GELU, and the first
> genuinely new architecture that runs **on the NPU** rather than host-only the
> way `arch=1` (EmbeddingGemma) does. It needs a task prefix — `--prefix`, and
> the runtime prints which one it applied, because a wrong prefix is a quality
> regression no `1-cos` gate can see.
>
> Getting there found a **silent correctness bug** in the shipping design
> generator: above `N = 4096` a design compiled and returned wrong numbers, and
> bge-large's production `N = 4096` sits at *exactly* the threshold of a strict
> `>`, so the guarded path had never once executed. See §"Known walls".

> **The shipped CLI is now subcommands**: `npuembeddings list`,
> `npuembeddings serve <model>`, `npuembeddings embed <model> <file>`, and a
> bare invocation prints help plus the model table. Models are fetched and
> checksum-verified **inside the executable** (`runtime/src/hub.cpp`); the
> `get-model.cmd` batch script is gone, because `curl` plus a hardcoded
> `certutil` digest is the behavioural signature of a dropper and antivirus
> treated it as one. The flag form below is unchanged and still carries every
> probe and benchmark.

A single place to answer: **what works, what does not, what was tried and
failed, where everything lives, and how to build and run it.**

This file is a snapshot and will go stale. The durable documents are
[`docs/`](00-overview.md) (how things work) and [`tasks/`](../tasks/README.md)
(what happened on a given day, including the failures). Where they disagree with
this file, they are right and this file is old.

---

## 1. Where the project is

**Five models run end to end on the NPU, in C++, with no Python in the
process, all validated against HuggingFace**, plus one (EmbeddingGemma) that
runs entirely on the host because its geometry does not fit the array. M0-M8
are done: the tokenizer ships, the MTEB gate passes, and energy is measured.
M9 made the runtime model-driven; M12 added `arch=1`; M13 added `arch=2`.

Throughput below is from the **whole-catalogue sweep**
([`0073`](../tasks/0073-m13-release-benchmarks/TASK.md)) — one session, one
machine state, one protocol, `--threads 24 --pipeline 4`, mean of three runs,
contention guard on. It is **end-to-end throughput and not an NPU kernel
performance claim** (rule 1). Earlier tables here mixed sessions and lane
counts and were not comparable to themselves.

| `--model` | arch | hidden | layers | pooling | `rel_fro` vs HF | seq/s |
|---|---|---:|---:|---|---:|---:|
| `all-MiniLM-L6-v2` | 0 | 384 | 6 | mean | 4.473e-03 | **951** |
| `bge-small-en-v1.5` | 0 | 384 | 12 | CLS | 3.789e-03 | **494** |
| `bge-base-en-v1.5` | 0 | 768 | 12 | CLS | 4.297e-03 | **211** |
| `bge-large-en-v1.5` | 0 | 1024 | 24 | CLS | 3.763e-03 | **60.8** |
| `nomic-embed-text-v1.5` | **2** | 768 | 12 | mean | 6.119e-03 | **166** |
| `embeddinggemma-300m` | **1** | 768 | 24 | mean | host-only | ~0.13 |

**Interleaved against the CPU**, one session: 1.85× / 1.60× / 2.49× / 2.47× /
2.26×, and energy 3.65× / 3.00× / 3.33× / 3.28× / 3.75× better per sequence.
The caveats are in [`06-performance.md`](06-performance.md) and they matter:
the CPU side moves ~24% between sessions, so treat a ratio as indicative to
about ±20%.

**The number that matters is not the ratio.** Those NPU figures were taken with
torch and ONNX Runtime saturating all twelve cores; measured idle the same
models give 951 / 494 / 211 / 60.8 / 166 — **within 1.5% on every model.** The
work is genuinely off the CPU.

`nomic-embed-text-v1.5` was predicted at **150–160 seq/s** before it was
measured, from one argument — its gated `ffn_up` emits both halves, so it does
~1.33× bge-base's per-layer GEMM work at the same depth. It landed at **166**,
so the prior was **4% low**: 211 / 1.33 = 159 against a measured 166. A near
miss rather than a hit, and recorded as one. `bge-base-en-v1.5` remains **the
model whose geometry fits this NPU best**.

`bge-base-en-v1.5` was added in [`0051`](../tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md)
and is **the model whose geometry fits this NPU best**: head_dim 64, and every
layer width a multiple of 384, so `tile_n` stays 48 where bge-large is forced
down to 32. It is also the most array-bound model we run -- 74.1% of wall
clock is NPU against MiniLM's ~40%.

`--model` is **required as soon as more than one container is installed** --
picking one silently is how this project's fail-open bugs have always looked.
Geometry, depth, pooling and tile size all come from the `.npue`; none of them
is a constant in the binary.

```
worst 1 - cos vs HuggingFace          1.086e-05        (tolerance 2e-03)
```

**The product is complete and every claim is measured on hardware.** One
`.npue` (69 MB, weights + vocabulary) and one `npuembed.exe` take text in and
return embeddings, with no Python in the process
([`0036`](../tasks/0036-m8-tokenizer/TASK.md)):
faithful to fp32 (`1-cos` 1.086e-05), **downstream quality preserved**
(MTEB +0.04 points, [`0035`](../tasks/0035-m8-mteb-gate/TASK.md)), **faster
than the CPU** (interleaved: 877 vs torch's 489 and ONNX Runtime's 234,
[`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md))
and **1.94× lower energy per sequence** ([`0034`](../tasks/0034-m8-energy/TASK.md)),
on ~5.3 cores against twelve. The tokenizer shipped in
[`0036`](../tasks/0036-m8-tokenizer/TASK.md).

**And the advantage grows with width, shrinks with depth**
([`0042`](../tasks/0042-m9-bge-large/TASK.md)). Same width, twice the layers:
1.792 -> 1.533, because our cost is per dispatch and the CPU has no such term.
Width 384 -> 1024: **2.106x**, and **4.2x per core**. That is
[`0027`](../tasks/0027-m7-width-hypothesis/TASK.md)'s prediction, made from a
synthetic sweep, confirmed on real models.

**Attention still cannot share a design with the projections, and now we know
why structurally** ([`0043`](../tasks/0043-m9-attention-geometry/TASK.md)).
Attention is `[64,64] x [64,64]`, so `N % (n * cols) == 0` forces `n * cols` to
divide 64, while the AIE microkernel asserts `n % 16 == 0`. **Therefore
cols <= 4**: any design that can express attention uses at most half the array,
with an eighth of the output tile.

Since 0030 the architecture changed shape (tasks/0031–0032): the NPU is now a
**pure GEMM engine** — four shapes as instruction streams over **one xclbin in
one hw_context, zero design switches** — while LayerNorm, softmax and GELU run
on the host in fp32 (the same polynomials the NPU kernels used, minus the bf16
round trip). Every op that moved got FASTER and MORE ACCURATE; 1.086e-05 is
exactly what M6 predicted for a host-GELU pipeline.

### Throughput, honestly

Against `sentence-transformers` on the same machine's CPU (12 threads,
[`tasks/0018`](../tasks/0018-npu-vs-cpu/TASK.md)), **at matched batch**:

| config | NPU seq/s | NPU cores | CPU seq/s | wall ratio | per core | J/1000 seq |
|---|---|---|---|---|---|---|
| single lane, batch 128 | 604–618 | ~3.5 | 710.0 | 0.86× | ~3.0× better | 53.5 (1.59×) |
| **pipelined 2 lanes** | **833** | ~5.3 | 710.0 | **1.17×** | 2.7× better | **44.0 (1.94×)** |

> **Superseded by [`0040`](../tasks/0040-m9-honest-cpu-baseline/TASK.md).** That
> table compares the CPU's **best-of-5** against the NPU's **mean**, and the two
> sides were measured minutes apart. Re-measured **interleaved**, same statistic
> on every side, steady state: **NPU 877.0, torch 489.4, ONNX Runtime 234.3
> seq/s — 1.792×.** The 710 is not reproducible today and the gap is not fully
> explained by the statistic (worth ~3.4%) or by background load (~6%); what is
> defensible is the interleaved ratio. The NPU is also far the more
> reproducible side: ±1.09× against the CPU's ±1.28× within a single run.

**Both halves of the project's own goal are now measured**: faster than the CPU
in wall clock, and **1.94× less energy per sequence** ([`0034`](../tasks/0034-m8-energy/TASK.md)).

**CPU wall parity is PASSED** (tasks/0033): two encodes pipelined over the one
unified design overlap host and NPU work 1.49×. (Older per-batch rows predate
0031–0033; re-measure small batches before citing them.)

Progress within M7: **42.4 → 833 seq/s, 19.6×** — and accuracy improved 31×
along the way (3.4e-04 → 1.086e-05).

---

## 2. What works

| | status | evidence |
|---|---|---|
| Native-Windows toolchain (IRON → Peano → aiecc → xclbin → NPU → trace) | ✅ | [`0002`](../tasks/0002-m1-hello-npu/TASK.md) |
| bf16 GEMM on the whole array, traced | ✅ | [`0003`](../tasks/0003-m2-bf16-gemm/TASK.md), [`0004`](../tasks/0004-m2-multicore-gemm/TASK.md) |
| numpy fp32 oracle matching HF at every layer (≤ 9.9e-07) | ✅ | [`0005`](../tasks/0005-m3-python-reference/TASK.md) |
| `.npue` pre-tiled weight container, bit-exact round trip | ✅ | [`0006`](../tasks/0006-m4-npue-pretiling/TASK.md) |
| All four per-layer GEMMs validated on hardware | ✅ | [`0011`](../tasks/0011-m5-first-op-validated/TASK.md), [`0012`](../tasks/0012-m5-all-layer-gemms/TASK.md) |
| Own GELU kernel (no transcendental call) | ✅ | [`0015`](../tasks/0015-m5-gelu-polynomial/TASK.md) |
| Own LayerNorm and softmax kernels | ✅ | [`0020`](../tasks/0020-m5-layernorm-kernel/TASK.md), [`0021`](../tasks/0021-m5-softmax-and-full-model/TASK.md) |
| Full encode in Python on the NPU | ✅ | [`0017`](../tasks/0017-m6-full-encode/TASK.md) |
| **Full encode in C++, no Python** | ✅ | [`0022`](../tasks/0022-m7-cpp-runtime/TASK.md), [`0023`](../tasks/0023-m7-full-cpp-encode/TASK.md) |
| Batching, arbitrary batch from one flag | ✅ | [`0025`](../tasks/0025-m7-batching-and-crossover/TASK.md) |
| Validated cost models (dispatch, switch, width crossover) | ✅ | [`0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md), [`0024`](../tasks/0024-m7-dispatch-cost-anatomy/TASK.md) |

**On the NPU per encode:** 24 GEMMs, 6 GELU, 13 LayerNorm, 6 softmax = 49
dispatches.
**Still on the host:** embedding gather, attention's per-head GEMMs, bias adds,
residual adds, pooling, L2 normalise.

---

## 3. What does NOT work, or was not achieved

### Not achieved

- **CPU parity.** ~2.9× short at every batch. See §5 for why and what would be
  needed. [`0026`](../tasks/0026-m7-closing-on-cpu/TASK.md)
- **Attention on the array.** `head_dim = 32` fails the whole-array design's
  `M % (m·4) == 0`. Needs padding to 64 or folding two heads into one 64-deep
  tile. Worth ~4% of the encode now that the host version is vectorised.
- ~~**A tokenizer.**~~ **DONE** ([`0036`](../tasks/0036-m8-tokenizer/TASK.md)):
  WordPiece in C++, **6,826/6,826 texts byte-identical to HuggingFace**, with
  the Unicode tables generated from `unicodedata`. `npuembed --embed file.txt`
  takes plain text and returns vectors — one process, no Python. The
  vocabulary now lives inside the `.npue`, so the product is ONE file plus one
  executable.
- ~~**M8 / MTEB.**~~ **GATE PASSED** ([`0035`](../tasks/0035-m8-mteb-gate/TASK.md)):
  five tasks, mean delta **+0.04 points** against the CPU running the same
  checkpoint at the same sequence length, worst task −0.01. bf16 + fp32
  accumulate confirmed as production; `--emulate-bfp16` closed out for good.
- ~~**Energy measurement.**~~ **DONE** ([`0034`](../tasks/0034-m8-energy/TASK.md)):
  **1.94× better energy per sequence than the CPU** (44.0 vs 85.3 J per 1000
  sequences), measured with the Windows **`\Energy Meter`** RAPL counters — the
  earlier "no instances" finding was against `\Power Meter`, the wrong counter
  set. No external instrumentation needed. A control experiment proves the
  package meter sees the NPU block (≈5.2 W under saturation).

### Known walls (things that will not build)

| wall | message | status |
|---|---|---|
| LayerNorm / softmax above 2 columns | `no ShimNOCTile has sufficient DMA capacity` | **Fixable, and the constraint is documented.** A shimNOC DMA has **≤ 6 S2MM channels**, so a flat 32-way join is inexpressible — it must be joined *hierarchically through the mem tiles* (INDEX.md constants). That is exactly what `.split()`/`.join()` does, and it is why GELU now runs at 8 columns. LayerNorm still opens **three** fifos per core and softmax two; neither has been rewritten. [`0027`](../tasks/0027-m7-width-hypothesis/TASK.md) |
| ~~`hidden ≥ 1536` in the width sweep~~ | ~~`'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range`~~ | **CLOSED, and it was worse than a wall.** tasks/0030 fixed the stride by forcing `tb_n_rows = 1` above `m·n_aie_rows·N > 2^20`, so it *builds* — but the fill/drain walk kept stepping by a hardcoded `tb_max_n_rows // 2` and never read the guarded value, so above the threshold it **compiled and returned wrong numbers** (`rel_fro` 7.07e-01, 28/32 row-bands wrong). Never observed because bge-large's N=4096 sits at **exactly** 2^20 and the guard is a strict `>`, so it had never fired in a shipped design. Found and fixed in [`0068`](../tasks/0068-m13-nomic-spike-and-oracle/TASK.md) §6/§6b when nomic's N=6144 crossed it. → [T30](../research/OPEN-THREADS.md) |
| 8-column design + core trace | `Unable to find a legal routing` | Known. Trace at 2 or 4 columns, throughput at 8 |
| B reuse in L2 | `no space for this BD` | ObjectFifo depth maps 1:1 to mem-tile BDs; ceiling is 6 tiles at 4 cols, 4 at 8, against a slice needing 24–48. Needs a core-side redesign. [`0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md) |

### Unresolved bug

- **`exp2_poly` works standalone (6.7e-03) but corrupts when composed into
  softmax** — row sums went to zero, output `[0,-120,0,-120,...]`, which is fp32
  being read as bf16 pairs. Reverted to `aie::exp2<bfloat16>`. Never diagnosed.
  [`0021`](../tasks/0021-m5-softmax-and-full-model/TASK.md)

---

## 4. Things that were tried and did NOT work

Kept because the negative results are load-bearing — several of them stopped us
optimising the wrong thing.

| tried | result |
|---|---|
| **Pre-tiling B as a performance lever** | **Refuted.** A wash (±9%), with 9–22% run-to-run spread against row-major's 2–3%. It optimises the L3→L2 access pattern, which is not the binding constraint. Kept anyway because it lets the runtime hand mapped bytes straight to DMA. [`0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md) |
| **Tile order `k,n` vs `n,k`** | No difference (118.9 vs 118.5). The locality hypothesis was wrong |
| **`--emulate-bfp16` (worth 5.5× on GEMM compute)** | **+2.2% end to end, and it fails accuracy** (3.470e-03 vs the 2e-03 tolerance). That it buys so little *is* the diagnosis: GEMM compute is not the bottleneck |
| **GELU tile 1024 → 4096** (4× fewer DMA transactions) | **No change at all** (16.4 ms both ways). This is what first showed the cost was not in the data movement |
| **More resident contexts / fewer resident contexts** | No effect. 2 loaded ≡ 7 loaded (1509 vs 1519 µs) |
| **Spatial partitioning across processes** | Two processes scale 1.46× *regardless of design width* (1, 2, 4 columns all the same). Width-independence rules out spatial partitioning; it is host overlap |
| **Simulated bfp16 accuracy predictions from M3** | **Refuted on hardware.** M3 predicted real activations would be 6.0× worse than uniform for block FP; measured 0.85×. Refit on hardware data gave 7 bits/element, not 5 |
| **"Two designs per process corrupt the NPU"** | **Our own bug**, publicly retracted before filing upstream. `A.numpy()[:] = x` writes the host buffer only. [`0009`](../tasks/0009-m5-sync-misdiagnosis/TASK.md), [note 0003](../research/notes/0003-two-designs-per-process.md) |
| **"AIE `float` is not real fp32"** | Wrong inference, killed by a direct probe: full 24-bit mantissa on add and multiply. [`0016`](../tasks/0016-m5-fp32-probe/TASK.md) |
| **Ceiling claim in 0026** ("eltwise is at the machine's fp32 limit, no more optimisation") | **Overreached.** The degree probe bounds *one core*; compute-bound work scales with cores. Eltwise measures 1.98× per column doubling |
| **"The placer will not do 4 or 8 columns"** | **Misattributed.** The failing frame was `ln_array`, not `gelu_array`. GELU runs at 8 columns: 3.87× alone, 1.13× end to end |
| **Parity at h ≈ 1300–2000** | **Withdrawn.** It assumed CPU time grows 4× per doubling, but the CPU runs the same encoder and has the same 1:3h structure, so its total also grows sub-quadratically |

### Four bugs that "failed open"

A recurring class worth its own list: a check that could not tell, and was read
as "fine".

1. **[`0022`](../tasks/0022-m7-cpp-runtime/TASK.md)** — mtime picked the same
   xclbin for all four designs. *A buffer-size check catches a wrong size, never
   a wrong layout* (`rel_fro` 1.186).
2. **[`0024`](../tasks/0024-m7-dispatch-cost-anatomy/TASK.md)** — column counter
   read `aie.mlir`, which is **pre-placement and has no tile coordinates**, so
   it returned 0 for every design.
3. **[`0025`](../tasks/0025-m7-batching-and-crossover/TASK.md)** — cache matcher
   pinned symbol and width but not **size**, so a batch-4 export got the
   batch-16 GELU. `1-cos` 8.651e-02.
4. **[`0026`](../tasks/0026-m7-closing-on-cpu/TASK.md)** — `std::max(0.0, NaN)`
   returns `0.0`, so a NaN-producing kernel scored a **perfect 0.000e+00 and
   PASSED**. *A tolerance test whose failure mode is a perfect score is not a
   test.*

All four are now fail-closed, and the fixes were verified against
**known-bad** artifacts, not only good ones.

---

## 5. Why CPU parity was not reached

At batch 128, `dispatch + wait` is **347 ms of a 509 ms encode**:

| | per call | calls | total |
|---|---|---|---|
| GELU | ~19 ms | 6 | **114 ms** |
| LayerNorm | ~4.2 ms | 13 | 55 ms |
| softmax | ~6.6 ms | 6 | 40 ms |
| 4 GEMMs | — | 24 | 77 ms |
| design switches | — | 49 | ~60 ms |

**Elementwise is 209 of 347 ms**, and it is compute bound, not data bound: a
passthrough moving the *same* bytes over the *same* columns takes **626 µs**
against GELU's **9703 µs** at batch 64.

Two structural findings explain the rest:

**(a) Switching design is the most expensive single operation in the stack.**
~25 µs + 7.2 µs per `aie.lock` — 89 µs for a trivial passthrough, 2.4 ms for an
8-column GEMM, 10–17× a dispatch. It is not reconfiguration-by-difference (the
*same* xclbin in two contexts costs the same), not residency, not eviction
(`Suspensions = 0` throughout). **Data movement and switching are one budget:
every descriptor added to feed the array better is paid again at each switch.**
[note 0004](../research/notes/0004-context-switch-cost.md)

**(b) MiniLM is structurally the wrong shape for this machine.** Per layer and
token: **4h GELU elements against 12h² MACs — a 1:3h ratio**, so the elementwise
share falls as **1/h** regardless of implementation. Measured: GEMM grows 3.72×
and GELU 1.96× when h goes 384 → 768 (predicted 4× and 2×), and the elementwise
share drops **29.7% → 18.1%**. At h = 384 we are near the worst case.

**No parity-h number is claimed.** The crossover lies somewhere inside the range
of widths people use, above h = 384; where exactly is unmeasured, and the GEMM
rate gain is nearly spent (0.639 → 0.688 TFLOP/s, already 72–88% of the
2-column-normalised ceiling).

---

## 6. Where everything lives

```
NpuEmbeddings/
├── CLAUDE.md                 ground rules, traps, current state (read first)
├── docs/                     durable truth — how things work
│   ├── 00-overview.md        start here
│   ├── 01-hardware/          XDNA2 array, limits, bandwidth
│   ├── 02-toolchain/         IRON, Peano, aiecc, XRT on native Windows
│   ├── 03-kernels/           kernel authoring
│   ├── 04-model/             MiniLM analysis, .npue format spec
│   ├── 05-measurement/       the measurement doctrine
│   └── CURRENT_STATUS.md     this file
├── tasks/                    what happened, day by day, failures included
│   └── 0001…0027/TASK.md     27 tasks; README.md is the index
├── research/notes/           0001 kernel pitfalls, 0002 arch fallback,
│                             0003 the retracted sync claim, 0004 switch cost,
│                             0005 the expert-review scoreboard
├── reference/                the ORACLE
│   ├── encoder.py            numpy fp32 MiniLM with swappable hooks
│   ├── encode_npu.py         the Python NPU path (M6)
│   ├── make_goldens.py       generates goldens/ from HuggingFace
│   └── goldens/              per-layer taps, pinned to a checkpoint sha256
├── experiments/
│   ├── m1-hello-npu/         SAXPY, first traced kernel
│   ├── m2-bf16-gemm/         GEMM bring-up
│   ├── m5-pretiled-gemm/     gemm_pretiled.py — THE production GEMM design
│   ├── m5-eltwise/           gelu_kernel.py, layernorm_kernel.py,
│   │   └── kernels/*.cc      softmax_kernel.py + the AIE C++ kernels
│   ├── m7-switch-cost/       the passthrough control design
│   └── m8-npu-vs-cpu/        the CPU baseline
├── tools/                    BUILD-TIME Python (never at runtime)
│   ├── pack_npue.py          HuggingFace → .npue
│   ├── npue.py               reader/writer + gemm_b_layout()/layout_hash()
│   ├── verify_npue.py        round-trip check
│   ├── export_xclbin.py      IRON designs → runtime/artifacts*/
│   └── export_validation.py  golden check vectors → artifacts/validation/
├── runtime/                  THE PRODUCT — C++ + XRT, no Python
│   ├── CMakeLists.txt
│   ├── include/npue.hpp      mmap reader for .npue
│   ├── include/npu_device.hpp  Design: one xclbin, staged buffers, dispatch
│   └── src/main.cpp          the encoder, the benchmark, the probes
├── models/                   checkpoint + all-MiniLM-L6-v2.npue (68.77 MB)
```

**Everything under `runtime/artifacts*/` is a build artifact and is
gitignored.** Regenerate it; never commit it.

---

## 7. How to build the whole thing

### Prerequisites (already installed — verify, do not reinstall)

| | |
|---|---|
| IRON | `C:\dev\mlir-aie` — mlir-aie 1.3.4, Peano 21.0.0, Python 3.13.15 |
| XRT | `C:\Xilinx\XRT` (2.21.0) — **not** under Program Files |
| `xrt-smi` | `C:\Windows\System32\AMD\xrt-smi.exe` (not on PATH) |
| MSVC | VS Community 2026, toolset 14.51 |
| Boost | `C:/dev/boost_1_88_0` (fallback only) |

> **`XILINX_XRT` must stay unset** — it poisons Windows builds. Use `XRT_ROOT`.

### Step 0 — environment (every shell)

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1          # MUST be dot-sourced
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
```

### Step 1 — weights and goldens (once)

```powershell
python reference\fetch_model.py        # HuggingFace checkpoint -> models/
python reference\make_goldens.py       # per-layer goldens from HF
python tools\pack_npue.py              # -> models/all-MiniLM-L6-v2.npue
python tools\verify_npue.py            # bit-exact round trip
python tools\export_validation.py      # -> runtime/artifacts/validation/
```

### Step 2 — compile the NPU designs

One command produces all seven xclbins. Pick width and batch deliberately —
they interact (§8):

```powershell
# the current best full-encode configuration
python tools\export_xclbin.py --cols 8 --elt-cols 2 --batch 128 `
                              --out runtime\artifacts_b128

# small batch: narrow GEMMs win, because switching dominates
python tools\export_xclbin.py --cols 2 --elt-cols 1 --batch 4 `
                              --out runtime\artifacts_b4
```

Useful flags: `--gelu-tile {1024,4096}`, `--gelu-variant {poly,probe2}`,
`--hidden H` (width sweep), `--emulate-bfp16` (faster, fails accuracy).

> The validation vectors live in `runtime/artifacts/validation/` and are read
> from there regardless of `--artifacts`. If you build into a fresh directory,
> copy `validation/` into it or leave the default `artifacts/` in place.

### Step 3 — build the C++ runtime

```powershell
cd runtime
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

> **`project()` must precede `find_package(XRT)`** or linking silently
> downgrades to static (mlir-aie #3048). **`/Zc:__cplusplus` is required**, not
> cosmetic — without it XRT demands Boost. `/arch:AVX2` enables the vectorised
> host paths.

### Step 4 — validate, then measure

```powershell
# correctness against HuggingFace (do this first, always)
.\build\npuembed.exe .. --artifacts artifacts_b128 --threads 16

# throughput
.\build\npuembed.exe .. --artifacts artifacts_b128 --threads 16 --bench 5
```

`--threads` controls the **host attention and conversion pool only**; default 1
keeps the low-core-count claim intact, so turning it up is an explicit trade of
cores for wall clock.

---

## 8. How to measure, and the knobs that matter

### The probes built into the runtime

```powershell
.\build\npuembed.exe .. --probe                     # repeat one design vs alternate two
.\build\npuembed.exe .. --probe-pair                # 2 resident contexts vs 7
.\build\npuembed.exe .. --probe-ctx                 # SAME xclbin in two contexts
.\build\npuembed.exe .. --probe-design <dir>        # one design: alone, A<->A', switch
C:\Windows\System32\AMD\xrt-smi.exe examine --report aie-partitions
```

`--bench` prints a five-way split of the NPU path (convert / sync-in /
dispatch+wait / sync-out / bias) plus a per-design table. Use it before
optimising anything: **twice in M7 the obvious explanation was wrong and the
breakdown found the real one.**

### Choosing width and batch

Design switching costs ~25 µs + 7.2 µs per lock, so **wider designs compute
faster and switch slower**. The crossover moves with M:

| GEMM columns | M = 256 (batch 4) | M = 1024 (batch 16) |
|---|---|---|
| 1 | 51.9 | 62.9 |
| 2 | **53.6** | **72.2** |
| 4 | 46.1 | **72.0** |

Predicted crossovers from `t = 140 µs + 2.43·M/cols`: 2 over 1 at M ≈ 237,
**4 over 2 at M ≈ 947**, 8 over 4 at M ≈ 3793. The M ≈ 947 prediction was tested
and confirmed. Rule of thumb: **`--cols 2` below ~batch 15, `--cols 4` above,
`--cols 8` at batch 64+**. `--elt-cols 2` always (until LayerNorm and softmax
are rewritten).

### The measurement rules that are not negotiable

1. **Wall clock is never an NPU kernel claim.** Hardware traces or static
   instruction counts. Wall clock is valid only for host/dispatch cost and
   end-to-end throughput, always labelled.
2. **Every sweep needs a control with a known correct value.** A sweep that only
   measures the quantity of interest cannot detect a silent corruption.
3. **Never a single run.** Pre-tiled designs show up to 22% spread; a one-shot
   bench once suggested +12.8% that three repeats erased.
4. **Verify a check against a known-bad artifact**, not only a good one.

---

## 9. Next steps, in priority order

0. **The one-xclbin architecture** ([`0029`](../tasks/0029-m7-one-xclbin-probe/TASK.md)):
   step 0 is confirmed — two instruction streams over one context alternate at
   alone-cost, so the 49 switches per encode (~60 ms at batch 128) are removable
   without fusion. Next: RTP-ify the GEMM loop counts so the four shapes share
   one ELF, unify, then re-measure the width crossover (with switches gone, the
   8-column designs should win outright).
1. **`.split()`/`.join()` for LayerNorm and softmax.** GELU proved the pattern —
   3.87× alone, 1.13× end to end. These two are 95 ms of the 209 ms of eltwise
   at batch 128. Highest value, lowest risk, and the code to copy already exists
   in `gelu_kernel.py`.
2. **Fuse layers.** The only way to *remove* switches rather than amortise them.
   Correctly priced now: fusing GELU into `ffn_up` saves the movement and one
   switch (~626 µs per call), **not** the arithmetic.
3. **M8 / MTEB.** Both bf16 and bfp16 configurations produce embeddings today;
   the accuracy decision needs the benchmark. bf16 + fp32 accumulate is the safe
   default at `1-cos` 3.4e-04.
4. **A wider model.** `docs/04-model` already designed for bge-small as a
   drop-in weight swap; **bge-large (h = 1024)** is the interesting one, given
   §5(b). ~~but the stride wall at h ≥ 1536 blocks testing much beyond it.~~
   *(Superseded: that wall is gone — see the Known walls table. It was fixed in
   tasks/0030 and the fix was half-wired until [`0068`](../tasks/0068-m13-nomic-spike-and-oracle/TASK.md).)*
5. **WordPiece tokenizer**, to close the last gap to a standalone product.
6. **Attention on the array** (`head_dim = 32` → pad to 64 or fold two heads).
7. **Energy**, which needs external instrumentation.

---

## 10. The traps that will cost you an hour each

The full list is in [`CLAUDE.md`](../CLAUDE.md); these are the ones that bite
hardest.

- **Set the device explicitly** or IRON silently compiles for NPU1 and
  `--emulate-bfp16` becomes a no-op — worth 5.5×, no error.
  `iron.set_current_device(from_name("npu2", n_cols=None))`
- **Never write device tensors through `.numpy()`.** `A[:] = x`, not
  `A.numpy()[:] = x`. The first dispatch in a process is correct either way,
  which is what makes it so easy to ship.
- **Never validate against a device read-back** — it agrees with whatever the
  device used and passes while measuring nothing.
- **Never identify a build artifact by mtime.** A JIT *cache hit* does not
  restamp the directory.
- **`aie.mlir` has no tile coordinates** — it is pre-placement. Use
  `input_with_addresses.mlir` (the same file tracing needs).
- **A JIT cache can serve a stale object after a `.cc` edit** with identical
  size. Delete the cache directory.
- **Budget L1 before compiling**: `2·(m·k·in + k·n·in + m·n·out) < 64512`. The
  limit is **63 KB, not 64** — 1 KB of DMEM is reserved for the program stack.
- **Budget ports and registers, not only bytes.** Per core: **2 in / 2 out** DMA
  streams (so a kernel can compute `C = AB` but *not* `C = AB + C`), 24 vector
  and **5** accumulator registers. Per mem tile: 6 in / 6 out. Per shimNOC:
  **≤ 6 S2MM channels** — which is the wall the eltwise designs hit.
- **Never scalar float math in a kernel** — 1,617× slower, measured.
- **Watch the worker stack.** Four interleaved chains in GELU overran
  `stack_size=0xD00`. It does not fault; it **corrupts**, and produced NaN.
