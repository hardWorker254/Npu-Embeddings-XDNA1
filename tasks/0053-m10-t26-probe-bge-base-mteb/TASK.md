# 0053 — M10: T26 probe, bge-base MTEB gate, register maintenance

- **Date** 2026-08-20
- **Milestone** M10 (post-0052 research-night follow-up)
- **Status** complete

## Goal

Execute Del A ("Kveld 1") of the approved research plan
(`~/.claude/plans/lag-en-plan-for-velvety-hollerith.md`): pure measurement,
no new architecture.

1. **A1** — probe T26 (why bfp16+bf16-C is 6.6x more accurate than
   bfp16+fp32-C) with a controlled K=384-vs-K=64xN, fp32-C-vs-bf16-C matrix,
   plus disassembly of both kernel variants.
2. **A2** — bge-base-en-v1.5 MTEB gate (T25): CPU baseline, NPU plain-bf16,
   NPU bfp16+bf16-C, plus the interleaved CPU-ratio half.
3. **A3** — register maintenance: file the missing phase-fusion thread, update
   T23/T25/T26, write this TASK.md.

## Context

[`0052`](../0052-m10-research-night/TASK.md) found bfp16-emulated GEMM +
bf16-C transport measures **6.6x more accurate** than bfp16 + fp32-C
(1-cos 3.615e-04 vs 2.395e-03), which is backwards — adding a rounding step
should never improve accuracy. It named a hypothesis (fp32-C fifo path
re-quantises C partials at every k-block boundary, 6x at K=384) and left it
untested, filed as T26. Separately, bge-base-en-v1.5
([`0051`](../0051-m9-bge-base-and-in-exe-fetch/TASK.md)) has no MTEB gate and
no interleaved CPU-ratio measurement (T25), and the pipelined block-fusion
architecture (0030 §4, the largest remaining host-side lever) has never had
an OPEN-THREADS entry despite being live since 0030.

## What was done

### A1 — the T26 probe

Wrote `experiments/m5-pretiled-gemm/t26_probe.py`, a standalone driver
(rather than CLI-plumbing `gemm_pretiled.py` itself, since the K-split
host-accumulation logic needs data slicing that doesn't fit a flag) that
imports `pretiled_array`/`tile_b` from `gemm_pretiled.py` directly. Design:
one small legal shape (M=256, N=192, tile 64x64x48, 4 columns) built two
ways for the SAME logical GEMM:

- **"full"** — one dispatch, K = K_FULL (384 or 1536), tile k=64, so the
  core's inner loop walks K_FULL/64 k-blocks (6 or 24) inside ONE
  `zero()...matmul()xN...release()` sequence.
- **"split"** — K_FULL/64 SEPARATE dispatches, each K=64 (exactly one
  k-block, no internal boundary at all), partial C's summed on the host in
  fp64.

Crossed with `{emulate+fp32C, emulate+bf16C, plain+fp32C}` (the bonus pair
the plan asked for). Same random Gaussian A/B data throughout (sliced for
`split`), same fp64 reference computed from the *actual bf16 values written
to the device* (CLAUDE.md trap 6c — never validate against a device
read-back of the true pre-quantisation numbers).

**First run had a bug that would have invalidated everything**: `markers_for`
was called with the TILE dims (64,64,48) instead of the per-call GEMM shape
(M,K,N differ between "full" and "split"), so the cache-marker text it built
never matched any real `aie.mlir`, and `purge()` silently deleted zero
candidates on every call — the "purge before every build" safety net
CLAUDE.md and 0030's fifth fail-open exist specifically to prevent was
running as a no-op. Found by inspecting the `purged N candidate(s)` log line
during the disassembly step (0 candidates when >=1 was expected), fixed by
passing the real M,K,N through (see `t26_probe.py`'s `markers_for`
docstring, which records this as the bug it is). Re-ran; results were
bit-identical to the unpurged run, which is not evidence the bug was
harmless — it means this session's cache did not happen to contain a
colliding entry, not that the mechanism is safe. Left the doc comment in the
source rather than deleting it, per `tasks/README.md`.

Env: `C:\dev\mlir-aie\iron_env.ps1` dot-sourced, ironenv Python 3.13.15.

### A2 — bge-base-en-v1.5 MTEB gate + interleaved CPU ratio

Ran `experiments/m8-npu-vs-cpu/run_mteb.py` for the CPU baseline together
with the NPU **plain-bf16** column (`--artifacts artifacts_base`, default
`--sides cpu,npu`) — that run self-computes the gate since both sides land in
the same `results` dict. Then, to add the **bfp16+bf16C** NPU column without
re-running the ~15-minute CPU side a second time, ran `run_mteb.py --sides
npu --artifacts artifacts_base_bfp_cbf16` (NPU only) and merged its per-task
scores against the CPU scores already recorded by the first run in a short
inline script, writing a matching `_gate.json` artifact. Confirmed first that
`runtime/artifacts_base_bfp_cbf16/` exists (built by 0052 for its
`--probe-streams` measurement) but carries **only the batch-128 instruction
tier** (`insts_{qkv,attn_out,ffn_up,ffn_down}_b128.bin`; `artifacts_base` has
b4/b16/b32/b128 for all four ops) — read `runtime/src/main.cpp`'s
`use_tier()` (~line 585) to confirm this is not a blocker: with a single tier
in the ladder, every request just pads up to that tier's batch (128 seq),
correct but with padding waste on smaller batches. Confirmed the NPU
contention guard was clear (`xrt-smi examine -r all`, found at
`C:\Windows\System32\AMD\xrt-smi.exe` — not on PATH, not at either of the
`C:\Xilinx\XRT\bin\` paths the guard also tries) before every NPU run, and
waited for each prior Python process to fully exit (not just its output file
to appear — `run_mteb.py`'s CPU-side torch/HF cleanup lingers for tens of
seconds after the "wrote ..." line prints, which the plain-bf16 log's own
timestamps don't show since stdout is block-buffered to a redirected file
and only flushes at process exit) before starting the next one, to avoid two
processes contending the same physical NPU.

Then ran `experiments/m8-npu-vs-cpu/compare_three.py` (0040's interleaved
protocol) for the plain-bf16 artifact set only — the bfp16+bf16C interleaved
ratio was **not** measured this session (left for later, alongside T27).
`bge-base-en-v1.5.onnx` did not exist yet (only MiniLM/bge-small/bge-large
had cached exports); `compare_three.py` exported it on first use.

## Commands

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m5-pretiled-gemm
python t26_probe.py --out artifacts\t26_probe.json
python t26_probe.py --k-full 1536 --no-disasm --out artifacts\t26_probe_k1536.json
```

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
& ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py `
    --model bge-base-en-v1.5 --artifacts artifacts_base --pipeline 2 `
    --out experiments\m8-npu-vs-cpu\artifacts\mteb_bge_base.json
& ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py `
    --model bge-base-en-v1.5 --artifacts artifacts_base_bfp_cbf16 --sides npu `
    --pipeline 2 --out experiments\m8-npu-vs-cpu\artifacts\mteb_bge_base_bfp_cbf16.json
& ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\compare_three.py `
    --model bge-base-en-v1.5 --batch 128 --rounds 8 --artifacts artifacts_base `
    --pipeline 2 --out experiments\m8-npu-vs-cpu\artifacts\compare_bge_base.json
```

## Result — A1

### Accuracy matrix, K=384 (6 k-blocks), M=256, N=192, 4 columns

| cell | full (1 dispatch) rel_fro | split (6 dispatches, host-summed) rel_fro | full/split ratio |
|---|---:|---:|---:|
| emulate + fp32-C | 1.471527e-02 | 1.471527e-02 | **1.000x (flat)** |
| emulate + bf16-C | 1.480471e-02 | 1.073492e-02 | 1.379x |
| plain + fp32-C (today's shipping datapath) | 1.377949e-07 | 5.203924e-08 | 2.648x |

### Accuracy matrix, K=1536 (24 k-blocks, ffn_down's real K), same M/N/cols

| cell | full rel_fro | split (24-way) rel_fro | full/split ratio |
|---|---:|---:|---:|
| emulate + fp32-C | 1.529620e-02 | 1.529620e-02 | **1.000x (flat)** |
| emulate + bf16-C | 1.539419e-02 | 9.877524e-03 | 1.559x |
| plain + fp32-C | 3.274301e-07 | 5.248258e-08 | 6.239x |

Full JSON: `experiments/m5-pretiled-gemm/artifacts/t26_probe.json`,
`t26_probe_k1536.json`.

### Disassembly (K=384, rtp=True both sides, so only `c_bf16` differs)

`objdump_{fp32C_rtp,bf16C_rtp}_matmul_bf16_f32_333c4d33.txt` — **byte-for-byte
identical** disassembly (md5 differs only because the printed file path
includes the cache directory name; the instruction stream is the same
`matmul_bf16_f32_333c4d33.o`, same hash suffix, in both builds). Compared
with `diff`; the only line that differs across the whole file is the header
`... file format elf32-aie` path.

`objdump_{fp32C_rtp,bf16C_rtp}_main_core_0_2.txt` (the per-core wrapper) —
differs only in: stack-frame offsets (different local layout), the
accumulator relocation target (`C_L1L2_0_0_buff_0/1` for fp32-C vs
`cacc_0_0` for bf16-C), and one extra basic block for bf16-C that calls
`narrow_3072_f32_bf16` right before `rel #0x36` releases the C fifo slot.
**No extra instructions appear inside the k-loop itself** — the
`acq/lda/movxm/jl matmul_bf16_f32` sequence that runs once per k-block is
structurally identical in both builds, just pointed at a different
accumulator address.

`narrow_f32_bf16.cc` (source, not rebuilt this task) explicitly documents
`aie::set_rounding(aie::rounding_mode::conv_even)` and calls itself *"the
first kernel in the project written with the mode set"* — every other
kernel, including the built-in `kernels.mm()` matmul, defaults to AIE's
`floor` (CLAUDE.md trap 2b / 0044's systematic-downward-bias finding).

### Conclusion: original T26 hypothesis REFUTED; anomaly does not reproduce in isolation; new hypothesis proposed

1. **The "k-block-boundary re-quantisation" hypothesis is refuted, on two
   independent grounds.** (a) The matmul kernel object is bit-identical
   between the fp32-C and bf16-C builds — there is only one matmul kernel,
   used the same way regardless of where its output accumulator lives, so no
   extra conversion instruction exists in either path's k-loop to be the
   mechanism. (b) Numerically, if boundary-crossing degraded the fp32-C
   path, its full/split ratio should grow with k-block count (6 -> 24
   blocks); it does not move AT ALL — 1.000x flat at both K=384 and K=1536.
2. **The anomaly itself does not reproduce on an isolated synthetic-data
   GEMM.** At "full" K (one dispatch, all k-blocks), bf16-C is statistically
   TIED with fp32-C — actually a hair WORSE — at both K=384 (1.480e-02 vs
   1.472e-02) and K=1536 (1.539e-02 vs 1.530e-02). Production's 6.6x
   improvement (0052, full MiniLM encode, real weights, 8 columns) is not a
   property of an isolated bfp16-emulated GEMM with random Gaussian data at
   these tile dims.
3. **A smaller, genuinely real effect WAS found, in "split" mode only**:
   bf16-C beats fp32-C by 27-36% when each dispatch does exactly ONE
   k-block (no boundary at all, by construction) and the results are summed
   on the host. This cannot be a "boundary crossed N times" effect — N=1
   here — so it must be intrinsic to the accumulator-Buffer +
   conv_even-narrow structure itself, not to K-block count.
4. **New leading hypothesis, not tested this task**: rounding-mode
   asymmetry between the two paths' eventual fp32->bf16 conversions. The
   bf16-C path narrows ON-CORE with `conv_even` (unbiased) exactly once.
   The fp32-C path's output stays fp32 out of the array; SOMEWHERE
   downstream (in production: feeding the next layer's GEMM, which needs
   bf16 input) it must still be narrowed to bf16 eventually, and if that
   narrowing uses the AIE default `floor` (biased) instead of `conv_even`,
   the difference would compound across MiniLM's 24 chained GEMMs in a way
   a single isolated GEMM cannot show — consistent with this probe's
   isolated single-GEMM effect topping out at 36% against production's
   6.6x, and consistent with the object-code evidence ruling out a
   matmul-kernel-internal mechanism. **Not yet confirmed**: would need
   either a chained multi-layer version of this probe, or tracing exactly
   where/how fp32-C's output gets narrowed to bf16 for the next layer in
   the production pipeline.
5. Also not ruled out: real model weight/activation distributions (vs this
   probe's synthetic Gaussian) may matter — 0008 already established that
   real vs uniform data changes plain-bfp16 error by 0.85x, not the 6x a
   simulated prediction assumed, so distribution-dependence of bfp16 error
   is an established mechanism class here, just not this specific one.

## Result — A2

### MTEB, five tasks, seq 64, bge-base-en-v1.5

| task | CPU | NPU plain-bf16 | Δ plain | NPU bfp16+bf16C | Δ bfp16+bf16C |
|---|---:|---:|---:|---:|---:|
| STSBenchmark | 86.418 | 86.422 | +0.004 | 86.407 | −0.011 |
| SICK-R | 80.301 | 80.299 | −0.002 | 80.286 | −0.016 |
| STS12 | 78.028 | 78.027 | −0.002 | 77.984 | −0.045 |
| Banking77Classification | 83.984 | 83.981 | −0.003 | 83.925 | −0.058 |
| TwentyNewsgroupsClustering | 50.576 | 50.695 | +0.119 | 50.383 | −0.193 |
| **mean / worst** | | | **+0.023 / −0.003** | | **−0.065 / −0.193** |

Both **PASS** (gate: \|mean\| ≤ 0.5 points AND worst ≥ −0.5 points). Neither
datapath is more fragile on bge-base than on MiniLM (0052 §7: bfp16+bf16C
mean +0.16, worst −0.06) — bge-base's worst (−0.193) is comfortably inside
that envelope, on an independent geometry (h=768, 12 layers, CLS pooling, N
set that still forces `tile_n=48`, but a completely different real-weight
distribution).

### Interleaved CPU ratio, plain bf16 only (0040 protocol)

Mains power, `Online / NoSystemBattery / Balanced`, 8 rounds, steady = mean
of the last 4:

| side | mean seq/s | steady seq/s | range |
|---|---:|---:|---:|
| torch (strongest CPU) | 110.7 | **111.2** | 108.2–113.0 |
| ONNX Runtime | 55.3 | 55.4 | 54.8–55.8 |
| **NPU** | 181.6 | **181.5** | 180.3–183.1 |

**NPU / strongest CPU = 1.633×.** Not yet measured: the bfp16+bf16C
interleaved ratio (only its MTEB gate was run — the throughput side needs a
full artifact export for that path, deferred alongside T27).

## Problems hit

1. **The `markers_for` tile-dims-not-shape bug** (above) — the purge
   safety net silently did nothing on the first run. Caught by reading the
   `purged N candidate(s)` log line rather than trusting the numbers, per
   M5's "always have a control with a known value" rule (here: the log line
   itself was the control — 0 purged when the shape had definitely been
   built before was the giveaway). Re-ran after the fix; numbers were
   unchanged, which is reassuring but not proof the bug was harmless in
   general — recorded rather than glossed over.
2. **A Python `SyntaxError: name 'K_FULL' is used prior to global
   declaration`** when adding `--k-full` as a CLI override: `argparse`'s
   `default=K_FULL` referenced the module global before the function's
   later `global K_FULL` statement, which Python's compiler treats as an
   error for the whole function body, not just the point of use. Fixed by
   defaulting the flag to `None` and assigning the global only if the flag
   was passed.
3. **`xrt-smi` is not on PATH** — lives at
   `C:\Windows\System32\AMD\xrt-smi.exe`, not `C:\Xilinx\XRT\bin\` (that
   directory has no `xrt-smi.exe` at all, only `xclbinutil`,
   `aiebu-asm/dump`, `xrt-runner`). Matches `npu_contention.cpp`'s own
   search order, just noting it since two of the three paths it tries
   don't exist on this machine.
4. **`run_mteb.py --sides npu` cannot self-compute a gate.** Its gate loop
   reads `results.get("cpu", {})`; with only `npu` requested, that is always
   `{}`, so every task prints `--` and `"pass": null` — not a crash, just a
   silently-empty gate table (the process even exits 1, which reads like
   failure but is only "no gate computed"). Worked around by writing a small
   inline merge script that pulls the CPU scores already recorded by the
   plain-bf16 run (same model, same seq 64, same tokenization) rather than
   paying for a second ~8-minute CPU pass; the merge — and the reason it is
   valid to merge across two separate process invocations — is recorded in
   `mteb_bge_base_bfp_cbf16_gate.json`'s own `"note"` field. Not fixed in
   `run_mteb.py` itself this session; a real, minor gap in the harness for
   whoever runs a `--sides npu`-only comparison next.
5. **`compare_three.py` crashes in its own final `print`, after the artifact
   is already written.** `out.relative_to(REPO)` (line 191) raises
   `ValueError: '...' is not in the subpath of '...'` on this machine because
   `out` is left as the *relative* `Path` from `--out` while `REPO` is
   absolute (`HERE.parent.parent`), and `Path.relative_to` requires both
   sides to agree on absolute-vs-relative — it does not resolve either
   operand first. Cosmetic only: `out.write_text(...)` (line 175) already
   ran, so `experiments/m8-npu-vs-cpu/artifacts/compare_bge_base.json` is
   complete and correct; only the confirmation `print` and the process exit
   code (1 instead of 0) are wrong. Not fixed this session — a one-line fix
   (`out.resolve().relative_to(REPO)`) for later.
6. **A dead NPU process outlived its own completed output by tens of
   seconds.** After the plain-bf16 MTEB run's JSON was already fully written
   (confirmed complete and correct), its Python process (torch/HF cleanup,
   ~2.8 GB resident) was still alive for a noticeable interval afterward.
   Started the bfp16+bf16C run only once `tasklist`/`Get-CimInstance
   Win32_Process` confirmed the PID had actually exited, not just once the
   output file existed — the file appearing is not proof the process (and
   therefore its hold on the NPU) is gone.

## Artifacts

- `experiments/m5-pretiled-gemm/t26_probe.py` (new)
- `experiments/m5-pretiled-gemm/artifacts/t26_probe.json` (K=384 matrix + disasm log)
- `experiments/m5-pretiled-gemm/artifacts/t26_probe_k1536.json` (K=1536 matrix)
- `experiments/m5-pretiled-gemm/artifacts/objdump_{fp32C_rtp,bf16C_rtp}_*.txt` (34 files: 16 core wrappers x 2 builds + 1 shared matmul kernel dump x 2 + 1 narrow kernel dump)
- `experiments/m8-npu-vs-cpu/artifacts/mteb_bge_base.json` (CPU + NPU plain-bf16, self-computed gate, PASS)
- `experiments/m8-npu-vs-cpu/artifacts/mteb_bge_base_bfp_cbf16.json` (NPU bfp16+bf16C only, five task scores)
- `experiments/m8-npu-vs-cpu/artifacts/mteb_bge_base_bfp_cbf16_gate.json` (merged gate: bfp16+bf16C NPU column vs the plain run's CPU column, PASS)
- `experiments/m8-npu-vs-cpu/artifacts/compare_bge_base.json` (interleaved plain-bf16 vs torch vs ORT, 8 rounds, ratio 1.633×)

## Next

`research/OPEN-THREADS.md` status changes this session: **T25 ANSWERED**
(both MTEB gate and interleaved ratio landed); **T23** updated with
bge-base's accuracy evidence (still PASS on a second geometry) but its
throughput case remains MiniLM-only; **T28 filed** (the phase-fusion registry
gap CLAUDE.md rule 3 warns about — 0030 §4 had never had a thread); **T26**
left OPEN, updated with the original hypothesis's refutation and the new
rounding-mode-asymmetry hypothesis, which points at T3 (device-resident
intermediates) as the place a chained multi-layer version of the probe, or a
production trace of where fp32-C's bf16 narrowing actually happens, would
need to look.

Left for a follow-up night: the bfp16+bf16C **interleaved throughput ratio**
for bge-base (only its MTEB gate ran this session — T27's traffic-bound
question is still argued from MiniLM alone); a chained/production-trace test
of T26's rounding-mode hypothesis; and Del B (phase fusion) / Del C
(EmbeddingGemma-300M spike) from the approved plan, neither attempted this
session.
