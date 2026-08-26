# 0079 — "why is 1.1× all we get?" — the array is 7× faster and the encode is not

**Goal (user, 2026-08-22):** *"Alle modeller har int8-kjøring. Jeg lurer på om vi skal hente
ned andre modeller da? … Du avgjør litt her, men jeg synes det er rart at 1.1x er alt vi får
ut av det."*

The scepticism was correct. **1.10× was the catalogue's worst case**, and the chain from
"5.5–7.7× arithmetic" to "1.10× encode" is now measured at every link rather than inferred.

---

## 1. The headline: it scales with the model

| model | bf16 | int8 | gain | array share of wall (bf16) |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 (6 layers, h=384) | 985.3 | 1090.0 | **1.10×** | 61% |
| **bge-large-en-v1.5 (24 layers, h=1024)** | **60.1** | **86.3** | **1.44×** | **87%** |

Exactly the direction [`0077`](../0077-m13-int8-gate/TASK.md) §4c's Amdahl bound predicted,
and it is why MiniLM was the wrong model to judge the lever on: its array was only 61% of
wall clock before int8 touched anything.

---

## 2. Where the other 5× goes — measured, single lane, MiniLM

| | bf16 | int8 | ratio |
|---|---:|---:|---:|
| quantise / convert A | 16.58 ms | 15.95 ms | 1.04× |
| **dispatch + wait** | **74.03 ms (3084 µs each)** | **41.46 ms (1727 µs each)** | **1.79×** |
| read out + bias / dequantise | 34.61 ms | 30.66 ms | 1.13× |
| NPU path total | 134.70 ms | 96.67 ms | 1.39× |
| wall | 185.53 ms | 172.85 ms | 1.07× |

**The arithmetic really is 5.5–7.7× faster** (0077, traced). **The dispatch is only 1.79×**,
and the difference is not mysterious:

[`0048`](../0048-m9-what-is-the-gemm-time/TASK.md)'s model predicts bf16's per-dispatch mean
at **3292 µs** against a measured **3084 µs** — excellent. Applying 7× to the *variable* term
only (the fixed 573 µs does not care how fast the MACs are) predicts int8 at **961 µs**.
Measured **1727 µs**. The missing ~766 µs/dispatch is **the C drain**: MiniLM writes
**679 MB of C per encode** (CLAUDE.md's own figure), and int32 C is 4 bytes exactly as fp32
was, so int8 does not move a single byte less. At 0010's marginal 33 GB/s that is ~858 µs
spread over 24 dispatches — which accounts for the gap.

**So the binding constraint moved.** It was arithmetic; it is now the C drain plus the fixed
dispatch cost. That is the whole answer to "why only 1.1×".

**The fix already exists in this repository.**
[`0045`](../0045-m9-bf16-gemm-epilogue/TASK.md) built core-side C narrowing (`--c-bf16`) and
measured **+4.9%** when C was a smaller share of a bigger total. For int8 the shape is even
better: the core could apply the per-column `wscale[j]` and emit **bf16** C, halving the
drain, leaving the host only the per-row `sa[i]`. Not built.

---

## 3. "Do the quantisation earlier" — tested, and half of it is impossible

The user's follow-up: *"Kvantifisering og konvertering gjør vi TIDLIGERE - altså ikke
runtime."*

Weights already are quantised offline. What remains is the **activation** side, and A is
inherently runtime data. The question is whether the *scale* can be static, because that
would remove the per-row reduction and make dequantisation a single per-column FMA whose
vector is known at pack time — and it would let the quantisation fold into LayerNorm's
`gamma`/`beta` as a second output at **zero extra arithmetic**.

**Measured: it does not work.**

| activation scale | worst 1-cos (MiniLM, α=0.5) | |
|---|---:|---|
| per row (per token) | 1.166e-03 | PASS |
| **static, calibrated per call site** | **6.693e-02** | **FAIL, 57× worse** |

The mechanism is structural rather than marginal: at seq 64 most rows are **padding**, and
sentence rows vary in magnitude. One scale for the whole tensor is set by the largest row and
quantises the rest toward zero.

**What survives of the idea**, and it is worth doing: the per-row absmax is a *reduction over
a row that LayerNorm already reduces over*. LN computes a per-row mean and variance; a third
per-row statistic in the same pass is nearly free, and LN could write **two** outputs — fp32
for the residual, int8 for the GEMM. That removes a full re-read of the activation tensor
(12.6 MB per site at batch 128) without touching accuracy. The post-LN residual objection
from [`0078`](../0078-m13-int8-accuracy/TASK.md) §4a does not apply to a *second output*,
only to rescaling the one that feeds the residual.

Two smaller "earlier" wins already taken: `1/asmooth` is reciprocated once per GEMM instead
of dividing twice per element (`conv` 109.9 → 47.0 ms), and both host passes are now AVX2.

---

## 4. bge-large int8 FAILS the accuracy gate, and α cannot save it

Swept on hardware, repacking each time:

| α | worst 1-cos | weight rel_fro (mean) |
|---:|---:|---:|
| 0.3 | 4.475e-03 | 9.351e-03 |
| 0.4 | 3.537e-03 | 1.027e-02 |
| **0.5** | **3.127e-03** | 1.171e-02 |
| 0.65 | 1.892e-02 | 1.524e-02 |
| 0.8 | 1.979e-01 | 2.012e-02 |

Minimum at α=0.5 and it **misses the 2e-03 gate by 1.56×**. The curve is informative: pushing
more range into the weights (higher α) hurts *more* here than on MiniLM, because bge-large's
weights are already the harder operand (`layer.21.ffn_down` at 2.238e-02 against MiniLM's
worst 1.781e-02). **24 layers accumulate what 6 layers absorb.**

So the honest position is: **int8 passes on MiniLM (MTEB −0.03) and fails 1-cos on
bge-large.** Note the precedent cuts both ways — bfp16 failed 1-cos at 3.47e-03, which is
where bge-large int8 sits, and [`0035`](../0035-m8-mteb-gate/TASK.md) makes MTEB the
authority. **MTEB on bge-large int8 has not been run**, and it is the measurement that would
settle it.

---

## 5. Should we download pre-quantised models? **No.**

The linked search (`base_model:quantized:nomic-ai/nomic-embed-text-v1.5`) lists ONNX and GGUF
exports. Three reasons not to:

1. **We already quantise, and it costs 0.03 MTEB points.** There is no accuracy left to buy.
2. **Their scheme would become ours.** ONNX int8 is usually *asymmetric* with zero-points;
   our GEMM computes `sum(aq·wq)` with no zero-point correction term, so adopting it means
   real kernel work (`−zp_a·Σwq` per column) for nothing.
3. **We would take on an ONNX/GGUF parser** in a packer that reads safetensors.

**What IS worth downloading is different architectures and finetunes**, and
[`0076`](../0076-m13-add-model/TASK.md) already built `npuembeddings add` for that. One gap
worth naming: `add` and int8 do not compose yet — `hub.cpp` packs through the C++
`prepare_model`, which is bf16-only, because int8 needs the SmoothQuant calibration pass that
runs the numpy oracle.

---

## 6. What this changes about where to spend effort

int8 did what it was supposed to: **the array stopped being the bottleneck.** On MiniLM the
host is now 60% of wall clock and the NPU 46%, and the two barely overlap (they sum to ~106%
of wall). The levers that matter now are, in order:

1. **Narrow C** — 0045's mechanism, aimed at what is now the floor inside the dispatch.
2. **Fuse quantisation into LayerNorm/GELU** — §3, removes a full tensor re-read per site.
3. **Lane overlap** — 106% of wall means almost nothing is overlapping; that is scheduling,
   not arithmetic.
4. **Fewer dispatches** ([T28](../../research/OPEN-THREADS.md#t28)) — the 573 µs fixed term is
   now a *larger* share of a smaller total.
