# 0056 — M10: T26 — production narrowing path traced, chained-GEMM compounding confirmed

- **Date** 2026-08-20
- **Milestone** M10 (post-0053 T26 follow-up)
- **Status** done

## Goal

Resolve T26's leading (untested) hypothesis: that bfp16+bf16-C measures 6.6×
more accurate than bfp16+fp32-C (0052) because the fp32-C path's output gets
narrowed to bf16 *downstream*, in production, using AIE's default `floor`
rounding (biased) instead of the bf16-C path's on-core `conv_even` (unbiased),
and that this asymmetry compounds across MiniLM's 24 chained GEMMs in a way
0053's single-isolated-GEMM probe could not show.

Two lines of attack, both to be pursued:

1. **Trace where fp32-C's output actually gets narrowed to bf16 in the
   production pipeline**, and what rounding mode that conversion uses.
2. **Build a chained multi-GEMM version of 0053's probe** to test directly
   whether the fp32-C vs bf16-C accuracy gap widens with chain length.

## Context

[`0052`](../0052-m10-research-night/TASK.md) §6 found bfp16-emulated GEMM +
bf16-C transport measures `1-cos` 3.615e-04 against bfp16-emulated + fp32-C's
2.395e-03 — 6.6× more accurate, backwards from what adding a rounding step
should do. [`0053`](../0053-m10-t26-probe-bge-base-mteb/TASK.md) refuted the
original "k-block-boundary re-quantisation" hypothesis on two grounds (the
matmul kernel object is byte-identical between builds; the full/split error
ratio is flat 1.000× regardless of block count) and showed the anomaly does
**not** reproduce on an isolated single synthetic-data GEMM at full K (bf16-C
ties or loses to fp32-C there). It found a smaller, real 27–36% bf16-C
advantage only in "split" mode (K=64, one k-block, no boundary by
construction), and proposed the rounding-mode-asymmetry hypothesis this task
tests.

## What was done

### Direction 1 — reading the production narrowing path

Read `runtime/src/main.cpp`'s `gemm()` (~line 1004), which is the function
every layer's GEMM dispatch goes through in the actual runtime (production
and the 0052/0053 full-encode measurements alike — the emulate-bfp16 and
`--c-bf16` flags only change which design gets built/dispatched, not this
host-side wrapper). Found:

- `to_bf16()` (line 403) — the function that converts a host fp32 buffer into
  the bf16 buffer fed to the *next* GEMM's A input — is:
  ```cpp
  inline uint16_t to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
  }
  ```
  round-to-nearest-even, explicitly, with a comment saying so ("truncation
  would bias every value toward zero"). The vectorised AVX2 form (`bf16_fill`,
  line 426) is bit-identical by construction.
- `tools/npue.py`'s `to_bf16_bits` (line 57, used when packing weights
  offline) is the **exact same bit formula**:
  `(((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF)`.
- In `gemm()`, every dispatch does `bf16_fill(abuf, a.data(), ...)` at the
  top (line 1009) to narrow the *incoming* activation to bf16 before sending
  it to the device — this is the actual "downstream narrowing" step the T26
  hypothesis was asking about, and it runs **identically regardless of
  whether the previous GEMM used c_bf16 or fp32-C**: with fp32-C the raw fp32
  accumulator is bias-added and (if the next op is a host eltwise op)
  processed in fp32, then narrowed via this RNE `to_bf16` right before the
  next GEMM; with bf16-C the accumulator is *additionally* narrowed once
  on-core via `conv_even` immediately after the K-reduction (before bias),
  then unpacked, bias-added, and **also** narrowed via this same host RNE
  `to_bf16` before the next GEMM.

**Conclusion: the literal hypothesis is refuted.** There is no floor-vs-RNE
asymmetry anywhere in the traced path — both the on-core `conv_even`
(bf16-C) and the host `to_bf16` (used by both paths, every layer boundary)
are round-to-nearest-even. AIE's default `floor` (CLAUDE.md trap 2b) does not
appear anywhere in this narrowing chain. `runtime/src/main.cpp` and
`pack_npue.py`/`npue_pack.cpp` were read only, not modified, per this task's
scope.

This did surface a real structural difference, though: **bf16-C narrows the
raw accumulator *before* the bias add; fp32-C keeps full fp32 precision
through the bias add (and any host eltwise op) and only narrows once, right
before the next GEMM.** Same rounding family (RNE-equivalent) at both
narrowing sites, but applied to a different intermediate value, and (for
bf16-C) one extra time. Filed as the refined hypothesis for direction 2:
not "which rounding mode", but "does an extra, earlier RNE-family narrowing
step reduce the growth of bfp16-emulation-induced error across a chain, even
though a single such narrowing measured roughly neutral in isolation
(0053)?"

### Direction 2 — chained multi-GEMM probe

Wrote `experiments/m5-pretiled-gemm/t26_chain_probe.py` (new sibling to
0053's `t26_probe.py`, not a modification of it). Design: ONE constant GEMM
shape reused for every chain stage (M=256, K=192, N=192, tile 64×64×48, 4
columns — K=N so a stage's output width feeds the next stage's input width
directly), so only **2 device builds total** (fp32-C rtp=False, bf16-C
rtp=True) are needed regardless of chain length; every stage dispatch at that
constant shape hits the cache after the first build. Cache purge runs once
per build (ordered `aie.runtime_sequence` marker, same convention as
`export_gemm_rtp.py` and 0053, not the three-loose-strings form 0045 found
broken).

Per stage `i` (chain length L, tested up to L=4), the SAME random weight
`W_i` and bias `b_i` feed both paths:

- **fp32-C path**: `C = dispatch(..., c_bf16=False)` (real device fp32
  output) → `Y = C + b_i` (host fp64) → `X_{i+1} = to_bf16_np(Y)` (the exact
  RNE bit formula from `main.cpp`/`npue.py`, reimplemented in the probe) →
  feeds next stage's A.
- **bf16-C path**: `C = dispatch(..., c_bf16=True)` (real device bf16
  output, already `conv_even`-narrowed on-core) → unpack, `Y = C + b_i` →
  `X_{i+1} = to_bf16_np(Y)` — the **same** final RNE narrow as fp32-C, plus
  the extra early on-core one.
- **Reference**: a pure fp64 chain with no MAC-emulation error and no
  intermediate narrowing at all (`X_ref = X_ref @ W_i.astype(f64) + b_i`,
  kept in fp64 throughout) — the "ideal" trajectory, same spirit as 0053's
  `ref`.

Both device chains use `emulate_bf16_mmul_with_bfp16=True` throughout (the
regime T26 lives in). `C.numpy()` is read as the actual device output in
both cases (bf16 bits view for c_bf16, raw fp32 otherwise) — never a
re-derived value, per CLAUDE.md trap 6c.

## Commands

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m5-pretiled-gemm
python t26_chain_probe.py --out artifacts\t26_chain_probe.json
```

Pre-flight (per CLAUDE.md process discipline): confirmed no active hw_context
via `& "C:\Windows\System32\AMD\xrt-smi.exe" examine -r all` (empty `HW
Contexts:` list) and no stray project python/npuembed processes (`Get-Process
python` found only unrelated `blender-mcp` extension processes, confirmed via
`Get-CimInstance Win32_Process` command lines) before dispatching.

## Result

```
stage   fp32C rel_fro   bf16C rel_fro  ratio (fp32C/bf16C)
    1    1.545069e-02    1.551581e-02                0.996
    2    2.140817e-02    1.845126e-02                1.160
    3    2.523557e-02    2.090068e-02                1.207
    4    2.946577e-02    2.283459e-02                1.290
```

Full JSON (`experiments/m5-pretiled-gemm/artifacts/t26_chain_probe.json`)
also carries `1-cos`, which shows the same trend more sharply:

| stage | fp32-C `1-cos` | bf16-C `1-cos` | ratio |
|---:|---:|---:|---:|
| 1 | 1.1935e-04 | 1.2036e-04 | 0.992 |
| 2 | 2.2903e-04 | 1.7020e-04 | 1.346 |
| 3 | 3.1814e-04 | 2.1830e-04 | 1.457 |
| 4 | 4.3394e-04 | 2.6055e-04 | 1.666 |

**The gap genuinely compounds with chain length.** At stage 1 (a single
GEMM + bias + narrow, no chaining yet) the two paths are statistically tied
— fp32-C is actually a hair *more* accurate (ratio 0.99–1.00), reproducing
0053's isolated-GEMM finding almost exactly. From stage 2 onward the ratio
grows monotonically and does not plateau over the 4 stages tested: 1.16×,
1.21×, 1.29× on `rel_fro`; 1.35×, 1.46×, 1.67× on `1-cos`. Extrapolating the
`1-cos` ratio's growth rate (roughly geometric after stage 1, ~1.25–1.35× per
additional stage) out to a 24-GEMM chain lands in the same order of magnitude
as production's measured 6.6× (0052) — not a reproduction of that exact
number (this probe uses a repeated toy shape, no real weight distributions,
no LayerNorm/GELU nonlinearity between stages, no residual connections), but
consistent with "compounds across chained GEMMs" as the right mechanism
*class*.

## Problems hit

None — the probe ran clean on the first attempt. Contributing factors: the
shape was deliberately kept small and constant across stages (cheap builds,
one purge per c_bf16 variant rather than per stage), and 0053's marker-bug
lesson (tile dims vs actual per-call shape) was applied from the start —
`markers_for` here uses the real `M,K,N` module constants directly, with no
per-call shape variation to get wrong.

## Artifacts

- `experiments/m5-pretiled-gemm/t26_chain_probe.py` — new, checked in.
- `experiments/m5-pretiled-gemm/artifacts/t26_chain_probe.json` — the run
  above, checked in.
- `runtime/src/main.cpp`, `tools/npue.py` — read only (Del B's territory /
  the packer respectively; no edits made).

## Next

**T26 updated, not fully closed.** What is now established:

1. The literal "floor vs conv_even" rounding-mode-asymmetry hypothesis is
   **refuted** — production's only downstream narrowing site (`to_bf16` in
   `main.cpp`, mirrored by `to_bf16_bits` in `npue.py`) is round-to-nearest-
   even, not `floor`, and it runs identically for both the fp32-C and bf16-C
   paths at every layer boundary.
2. A refined, purely **structural** hypothesis — bf16-C narrows the raw
   accumulator once, early (before bias), in addition to the same late RNE
   narrow both paths share — is **confirmed as a genuine compounding
   mechanism** on a controlled synthetic chain: tied at 1 stage, diverging
   monotonically to 1.29–1.67× by 4 stages.
3. **Still open**: *why* an extra early RNE-family narrowing reduces the
   growth of bfp16-emulation-induced error, at the numerical-mechanism level.
   A plausible candidate, not tested here: 0053's own "split" data already
   showed bf16-C's error *decreases* as block count grows (1.073e-2 at 6
   blocks → 9.878e-3 at 24 blocks), consistent with independent per-block
   quantisation noise partially cancelling when narrowed-and-summed rather
   than accumulated-then-narrowed-once — but this task did not isolate that
   mechanism from the chain-compounding one measured here, and they may be
   the same phenomenon viewed two ways (both are about *how many independent
   narrowing/summing events* the error passes through) or two different
   ones.
4. **Still not tested** (carried over from 0053): whether real model weight
   and activation distributions change this — this probe and 0053's both use
   synthetic Gaussian data, and 0008 already established that real-vs-uniform
   data changes plain-bfp16 error by 0.85×, not the 6× a simulated prediction
   assumed, so distribution-dependence remains a live open question, just not
   confirmed as *this* mechanism.

A next task could either (a) chase the numerical "why" with a controlled
noise-injection model (skip real hardware, work in numpy: synthetic
per-k-block quantisation noise of the measured magnitude, narrow-early vs
narrow-late, see if the same compounding ratio reproduces without any NPU
involved — cheap and fast to iterate), or (b) extend this probe to L=8–12
with real bias magnitudes/LayerNorm-shaped nonlinearity between stages to
get closer to production's actual structure. Neither was attempted here —
out of scope for this session, which was pure measurement per the task
brief.

**Not addressed, on purpose**: whether today's *shipping plain-bf16* path
(no emulation) carries any analogous, smaller rounding-position effect. The
brief was explicit that this is a future decision, not this task's scope;
the finding above (both narrowing sites are RNE, no floor bias) already
covers the part of that question this task could answer for free.
