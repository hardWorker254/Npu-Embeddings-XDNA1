# 0052 — M10: the 0.3.0 research night

**Brief** (user, 2026-08-20 01:17): research and testing toward visible 0.3.0
progress. Named directions: (1) force geometry across tiles, (2) use acquire
more efficiently, (3) run two phases in parallel where B does not depend on A
— or build them together in one run. Free to test ideas until 03:30.

Everything below was measured tonight; commands inline. Times are wall-clock
labelled, traces are traces, per docs/05-measurement.

---

## 1. T24 ANSWERED — the 0048 fit holds at h=768; the 27% miss was the host term

```powershell
.\build\npuembed.exe .. --model bge-base-en-v1.5 --artifacts artifacts_base --probe-streams
```

| stream | GMAC | MB | µs | GMAC/ms | GB/s |
|---|---:|---:|---:|---:|---:|
| qkv | 14.50 | 264.2 | 10,888 | 1.33 | 24.3 |
| attn_out | 4.83 | 88.1 | 4,221 | 1.14 | 20.9 |
| **ffn_up** | **19.33** | **352.3** | **14,365** | 1.35 | 24.5 |
| **ffn_down** | **19.33** | **276.8** | **14,409** | 1.34 | 19.2 |

The discriminating pair reproduces at h=768: identical MACs, 1.27× bytes,
**0.3% apart**. Marginal per-iteration (14,365−573)/3,072 = **4.49 µs vs the
h=384 fit's 4.72 (−5%)**. And the 0051 account closes: predicted array time
for two pipelined lanes 2×526.6 = 1,053 ms; measured NPU share of the
pipelined wall = 1,047 ms. **The 27% throughput miss was entirely the host +
overlap term the prediction ignored; the array fit was fine.**

## 2. Lane sweep — pipeline 2 was leaving 19% on the table (user idea 3)

The NPU serialises dispatches, so lanes only hide *host* work behind array
work. bge-base (74% array-bound) was expected to plateau early; it does not:

| lanes | bge-base seq/s | NPU busy | MiniLM seq/s | NPU busy |
|---:|---:|---:|---:|---:|
| 1 | 132.4 | — | — | — |
| 2 | 175.4 | 72.8% | 892.7 | 53.9% |
| 3 | 196.4 | 83.4% | — | — |
| 4 | **209.1** | 87.1% | **962.6** | 62.7% |
| 5 | 210.3 | 87.9% | — | — |

**bge-base +19% and MiniLM +7.8% by raising the default from 2 to 4 lanes.**
0033's "three lanes plateau" was measured on MiniLM before batch tiers and
bf16-C existed; it does not describe today's runtime. Latency per group grows
with lanes (a 4-lane group is ~2.4 s of work on bge-base), so a server default
of 4 trades tail latency for throughput — flag it rather than bury it.

## 3. THE HEADLINE — bfp16 passes the 0035 MTEB gate criterion on MiniLM (T23)

Steps: `--emulate-bfp16` added to `tools/export_gemm_rtp.py` (research flag,
documented as such in its help text; bfp16 and bf16 builds are ambiguous in
`aie.mlir`, so correctness rests on the 0030 purge-before-build discipline —
noted in the flag's comment). Exported a unified bfp16 set, validated, ran the
0035 MTEB bridge on it.

**End to end accuracy on today's architecture: `1-cos` 2.395e-03** — against
0026's 3.470e-03 (host eltwise is fp32 since 0032, which is where the
improvement comes from). Still 20% over the 2e-03 gate. But 0035 established
MTEB as the authority, and MTEB says:

| task | CPU fp32 (0035) | NPU bf16 (0035) | NPU bfp16 (tonight) | bfp16 Δ (pts) |
|---|---:|---:|---:|---:|
| STSBenchmark | 0.8203 | 0.8204 | 0.8207 | +0.04 |
| SICK-R | 0.7758 | 0.7759 | 0.7725 | **−0.33** |
| STS12 | 0.7237 | 0.7236 | 0.7220 | −0.17 |
| Banking77 | 0.8005 | 0.8005 | 0.8013 | +0.08 |
| TwentyNews | 0.4581 | 0.4598 | 0.4624 | +0.43 |

**Mean delta +0.01 points** (the 0035 gate passed bf16 at +0.04), worst single
task −0.33 (SICK-R, 0.4% relative), and the largest positive delta is
clustering, where run-to-run spread across our three models spans ±0.25.
Fine print that must travel with this: NPU side ran at `--pipeline 1`
(lanes are bit-identical per 0033, so this should not matter, but it is a
difference from 0035's protocol); CPU column reused from 0035, same
checkpoint, same seq 64.

### The bfp16 array is TRAFFIC-bound — the retired byte-levers revive on it

```
probe-streams, bfp16:      µs     GMAC/ms   GB/s     vs plain bf16
  qkv                    1,840      1.97     46.2      1.83x
  attn_out                 798      1.51     35.5      1.85x
  ffn_up                 2,890      1.67     39.2      1.45x
  ffn_down               2,136      2.26     35.3      2.00x
```

**1.74× on total array GEMM time** (13,314 → 7,664 µs/layer) — not 0049's
per-iteration 2.9×, because the datapath change moves the binding constraint:
GMAC/ms is no longer flat, GB/s sits at 35–46, and **ffn_up is now SLOWER than
ffn_down in proportion to its bytes**. Exactly what 0049's traced lock-stalls
predicted. Consequence: on a bfp16 datapath, B-reuse (T2), bf16-C, and
ATB-style buffering (2511.16041) all become live again.

### But e2e barely moves until the host follows

bfp16 at pipeline 4: **982.1 seq/s vs plain 962.6 (+2%)**, NPU busy only 49%.
The array got 1.74× faster and the encode is now host-walled. The datapath
decision is therefore ALSO a host-architecture decision: it pays 1.35–1.66×
only combined with readback reduction (bf16-C halves the read; T3 removes it).

## 4. k=96 via single-buffered B — Stationary-B EXPRESSES and PASSES (user ideas 1+2)

`gemm_pretiled.py` gained `--b-depth` (L1 depth of the B fifo; `forward()`
accepts `depth=`). With `--b-depth 1` the L1 budget becomes the Stationary-B
form `2mk + kn + 2mn` = 58,368 B at (64,96,48) — **k=96 is legal, compiles,
runs and matches**: relfro 1.89e-07, identical to the k=64 baseline.

First trace (taken while MTEB held the array, and with dropped E1 packets —
9 of 74 windows unpaired): in-window LOCK_STALL mean ~3.9k cycles, which would
be the exposed B-fill the single buffer risks — but the trace is DOUBLY
contaminated and is not evidence. Re-measured clean below (§6).

## 5. Files touched

- `tools/export_gemm_rtp.py` — `--emulate-bfp16` research flag
- `experiments/m5-pretiled-gemm/gemm_pretiled.py` — `--b-depth` /
  `b_l1_depth` through the JIT wrapper and `_build_design`; the L1 check now
  uses the depth-aware budget; tags carry `_bd1`
- `runtime/artifacts_bfp_rtp/`, `runtime/artifacts_bfp_cbf16/` — research
  artifact sets (bfp16, bfp16+bf16-C)
- `experiments/m8-npu-vs-cpu/artifacts/mteb_bfp16.json` — the T23 measurement

## 6. bfp16 + bf16-C: the combination is anomalously accurate, and that is UNEXPLAINED

```
artifacts_bfp_cbf16 (emulate-bfp16 + c-bf16):
  1-cos vs HF        3.615e-04     <- PASSES the 2e-03 gate outright
  probe-streams      1,453 / 680 / 2,149 / 1,765 µs  = 6,047 µs/layer
  array GEMM         2.20x vs plain bf16 (13,314 -> 6,047)
  bench p=4          1,046.2 seq/s  (NPU 46.9% busy)
```

For reference: plain bf16 fp32-C 1.086e-05; plain bf16 bf16-C 1.498e-05
(1.38× — one extra rounding, sane); bfp16 fp32-C **2.395e-03** (matches M6's
historical 2.35e-03, sane); bfp16 bf16-C **3.615e-04 — 6.6× BETTER than
bfp16 with MORE precise C transport.** Both workers accumulate fp32 in L1
through the same matmul kernel; the only structural difference is fifo-object
accumulator vs core-local `Buffer` + `narrow_f32_bf16`. No mechanism I can
name explains an *improvement*. Filed as an open thread — verify first that
it reproduces (different fixture, e2e corpus, MTEB), THEN hunt the mechanism.
The suspicious direction: if something in the fp32-C bfp16 path *degrades*
partial sums (e.g. the emulated kernel re-quantising the reloaded C partials
through the bfp16 datapath on every k-block, which the Buffer path might
avoid by keeping the accumulator hot), then 3.6e-04 is the true bfp16 floor
and 2.4e-03 was carrying an extra, removable error all along — which would
ALSO explain part of 0026's verdict against bfp16.

## 7. MTEB on the combination — passes by a wide margin

`--sides npu --artifacts artifacts_bfp_cbf16 --pipeline 1`, CPU column reused
from 0035 (same checkpoint, same seq 64):

| task | CPU fp32 | NPU bf16 (0035) | **NPU bfp16+bf16C** | Δ (pts) |
|---|---:|---:|---:|---:|
| STSBenchmark | 0.8203 | 0.8204 | 0.8199 | −0.04 |
| SICK-R | 0.7758 | 0.7759 | 0.7761 | +0.03 |
| STS12 | 0.7237 | 0.7236 | 0.7231 | −0.06 |
| Banking77 | 0.8005 | 0.8005 | 0.8006 | +0.01 |
| TwentyNews | 0.4581 | 0.4598 | 0.4670 | +0.89 |

**Mean +0.16 points, worst single task −0.06** — every non-clustering task
within ±0.06 of the fp32 CPU. Independently, `verify_embed_e2e` on real text:
worst `1-cos` **9.701e-04**, top-10 neighbour overlap **0.9923** (gate 0.98),
PASS. And the kernel's identity is proven, not assumed: the built object
disassembles with `vconv.bfp16ebs8.fp32` — the block-floating-point datapath.

**Every gate this project has now passes on bfp16+bf16C.** Whether it becomes
a default is the user's call (0035/0045 precedent), but the decision now has
measurements on both sides instead of a stale FAIL from 0026.

## 8. k=96 / Stationary-B VERDICT: negative, measured (T19 closed)

Clean traces, idle array, ffn_up M=256 at 4 columns:

| config | window (median) | vector | gap (median) | cyc/MAC |
|---|---:|---:|---:|---:|
| k=64, B depth 2 (0049 baseline) | 7,813 | 6,144 | 84 | 0.0402 |
| k=64, B depth 1 | 2 E1 packets dropped, pairing partly corrupt; printed avg +6% | | LOCK_STALL total 44,800 vs baseline's 11,072 — real, idle array | — |
| **k=96, B depth 1** | **11,265** | **9,216** (= 1.5×6,144 exactly) | **1,193** | **0.0422** |

The compute scales perfectly (vector = exactly 1.5×, still 8 cyc/MMAC at the
32 MACs/cyc limit) and the fixed overhead amortises as predicted — but
single-buffering B exposes its L2→L1 fill, and the exposure lands in the
inter-window gaps: **84 → 1,193 cycles**. Net: **k=96 Stationary-B is 5.2%
WORSE per MAC than the shipping (64,64,48)**. relfro 1.89e-07 both — the
variant is numerically exact, just slower.

T19's own risk note ("single-buffering B risks losing fetch/compute overlap,
which would eat the 8% directly") is confirmed and exceeded: the overlap loss
(~1,100 cyc/iteration) is roughly double the amortisation gain (~590). Only
the ATB form (shrink A's M, keep everything double-buffered — 2511.16041)
remains open, and 0049's ≤1.29× ceiling still caps it.

## 9. Production change: serve/embed default is now 4 lanes

One line in `main.cpp` (subcommand defaults only; the flag form is
untouched): pipeline 2 → 4, from §2's sweep. Verified: `npuembeddings embed`
reports "4 concurrent encodes". New model bests: **MiniLM 962.6 seq/s,
bge-base 209.1** (bench 3, idle array, threads 24).

## 10. bge-base on the combination — the anomaly reproduces at h=768

`artifacts_base_bfp_cbf16` (bfp16 + bf16-C, hidden 768, tier 128):

- **`1-cos` 2.237e-04, PASS** — a second model, a second width, same
  anomalously-good accuracy class as MiniLM's 3.6e-04. T26's mystery is not a
  MiniLM fixture artifact.
- **238.0 seq/s at pipeline 4** against plain-bf16's 209.1 (+13.8%) and the
  0.2.0 configuration's 175.4 (**+35.7% in one night**). NPU busy fell
  87% → 59%: the 2×-faster array put the wall back on the host, exactly the
  T27 shape.
- Not yet run on it: MTEB (T25 covers bge-base generally) and e2e verify.

## 11. What did NOT get done tonight

- **Cross-tile `Buffer` for streamed operands** (user idea 1 in its literal
  form, T17): not attempted — the k=96 result makes its ceiling (≤1.29×,
  realistically ~1.06×) even less attractive against its build cost.
- **True phase overlap / fused qkv+attn_out dispatch** (user idea 3 beyond
  lanes): not attempted; the lane sweep captured the cheap version of the
  same idea. The 0030 §4 pipelined block-fusion form remains the identified
  big lever on the host side.
- bge-base MTEB (T25) and the bfp16-anomaly mechanism (new thread below).

## Files and artifacts

- `tools/export_gemm_rtp.py` (+`--emulate-bfp16`),
  `experiments/m5-pretiled-gemm/gemm_pretiled.py` (+`--b-depth`),
  `runtime/src/main.cpp` (lane default)
- `runtime/artifacts_bfp_rtp/`, `runtime/artifacts_bfp_cbf16/`
- `experiments/m8-npu-vs-cpu/artifacts/mteb_bfp16.json`, `mteb_bfp16_cbf16.json`
- traces: `experiments/m5-pretiled-gemm/artifacts/trace_pretiled_kn_st_bd1_*`
- `tasks/0036-m8-tokenizer/verify_embed_e2e_all-MiniLM-L6-v2_artifacts_bfp_cbf16.json`
