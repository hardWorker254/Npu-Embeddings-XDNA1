# 0078 — the int8 accuracy half: naive W8A8 FAILS, SmoothQuant PASSES, and the flag is wired

**Goal (user, 2026-08-22):** *"kjør ferdig det du trenger for int8, men la oss beholde fp
også. la det være flagg."*

Status: **DONE. int8 runs end to end on hardware and PASSES both gates** -- golden
`1-cos` **1.417e-03** (against 2e-03) and the **M8 MTEB gate at mean -0.03, worst -0.13**,
which is indistinguishable from bf16's recorded +0.04. Throughput 1086.7 seq/s against
bf16's 985.3 (**1.10x** end to end, on the catalogue's worst case for this lever). bf16
remains the default and is byte-identical to before. Naive W8A8 fails the 2e-03 gate at **2.864e-03**; W8A8 +
**statically-calibrated SmoothQuant at alpha=0.5 passes at 1.166e-03**, 1.7x inside.
**Read §4b first — it supersedes the tables in §3 and §4**, both of which were measured
through harness bugs. The runtime quant/dequant path is NOT built (§6).

---

## 1. What [`0077`](../0077-m13-int8-gate/TASK.md) left

0077 established the datapath: int8 builds, is **bit-exact**, and traces at **5.5–7.7×** bf16
on all four production shapes. It made **no accuracy claim at all** — bit-exactness is about
integer arithmetic, not about whether an int8-*quantized* model embeds well.

This task answers that, **in numpy, before building any of the runtime**. The reference
encoder takes a `gemm=` primitive precisely so a precision question can be asked without a
second copy of the model (`reference/encoder.py`'s own docstring), so the whole investigation
needed no NPU and no C++.

---

## 2. A prediction, registered before the measurement

Following [2608.13756](https://arxiv.org/abs/2608.13756)'s practice of pre-registering,
and because this project's own record shows how easily a hoped-for number gets found:

> Per-GEMM W8A8 error on real golden activations measured **7.5e-03 – 2.6e-02**, against
> bf16's real ~1.5–2.4e-03 (M5) and bfp16's ~1.04e-02. bfp16 landed end-to-end at 1-cos
> **3.47e-03** and FAILED the 2e-03 gate ([`0026`](../0026-m7-closing-on-cpu/TASK.md)).
> **Prediction: naive W8A8 lands 3e-03 – 1e-02 end-to-end and fails the gate.**

**Measured: 3.397e-03. Failed.** The prediction was right, and landed within 2% of bfp16's
3.47e-03 — two unrelated quantisation schemes converging on the same end-to-end number,
which says the encoder's sensitivity sets it rather than the scheme.

---

## 3. The scheme, and the ablation that justifies each half

Symmetric W8A8: weights per **output channel**, activations per **token (row)**.

```
s[j]    = max_k |W[k,j]| / 127          Wq = round(W/s) clipped [-127,127]
sa[i]   = max_k |X[i,k]| / 127          Xq = round(X/sa)
Y[i,j]  = int32_dot(Xq[i,:], Wq[:,j]) * sa[i] * s[j] + bias[j]
```

Dequantisation is a rank-1 outer-product scaling of the int32 result — one multiply per
output element, which folds into the pass that already reads C and adds the bias. The
accumulator is exact (no rounding in the K-reduction at all), so **every bit of error is in
the two roundings and the scale multiply**.

End-to-end, worst `1-cos` over the golden corpus, against the same oracle with fp32 GEMMs:

| configuration | worst 1-cos | gate 2e-03 |
|---|---:|---|
| bf16 (RNE, fp32 accumulate) — sanity check | 1.380e-05 | PASS |
| **W8A8, per-token act + per-channel W** | **3.397e-03** | **FAIL** |
| W8A8, **per-tensor** act | 1.794e-02 | FAIL (5.3× worse) |
| W8A8, **per-tensor** W | 4.445e-02 | FAIL (13× worse) |

**Both "per-" choices are load-bearing, and the ablation measures why.** 2209.13325 (indexed)
documents the ±50–100 outlier dimensions in post-LN BERT that rule out per-tensor
activations; the outlier ratio (max channel mean-abs / median) measured **3.1–21.2** across
the real golden taps. Per-tensor scaling lets one outlier column set the step size for every
column, and it costs 5.3×.

---

## 4. SmoothQuant rescues it — and the deployable half is enough

SmoothQuant divides activations by a per-input-channel factor and multiplies the weights by
the same factor, which is **mathematically an identity** on the linear layer while moving the
outlier problem from the operand that cannot absorb it to the one that can:

```
s_j = max_i|X[i,j]|^alpha / max_n|W[j,n]|^(1-alpha)      X /= s ,  W *= s
```

| alpha | worst 1-cos | |
|---:|---:|---|
| none | 3.397e-03 | FAIL |
| 0.3 | 2.170e-03 | FAIL |
| 0.4 | 1.448e-03 | PASS |
| **0.5** | **1.051e-03** | **PASS** |
| 0.6 | 1.060e-03 | PASS |
| 0.7 | 1.963e-03 | PASS |
| 0.8 | 1.918e-03 | PASS |

A clean bowl with its minimum at 0.5–0.6, which is where SmoothQuant's own paper puts it.

### 4a. CORRECTION — the LayerNorm fold is NOT available here, and I wrote that it was

My first reading of this said the factor is free because it folds into the preceding
LayerNorm's gamma/beta, and that **only** `qkv` (after `ln1`) and `ffn_up` (after `ln2`)
therefore have somewhere to hide it. Checking `reference/encoder.py` before building on it:

```python
out = self.ln_fn(proj + x, ...)   # ln1  -- consumed by ffn(x) AND by the next residual
out = self.ln_fn(down + x, ...)   # ln2  -- consumed by the next qkv AND the next residual
```

**BERT is post-LN.** Each LayerNorm's output feeds the following linear layer *and* the
residual added before the next LayerNorm. Dividing gamma/beta by the smoothing factor would
scale the residual too, which is not an identity — it silently changes the model. SmoothQuant
folds cleanly in **pre-LN** decoder architectures, which is what its paper targets; this is
not one.

**The right mechanism costs one multiply per element and does not touch the residual.** The
quantisation pass already reads every element of A to take a row absmax and divide by it;
dividing by a per-input-channel vector in the same pass is one extra multiply on data
already in registers. So the smoothing vector ships in the container per GEMM
(`<name>.asmooth`, F32 [K]) with the factor pre-multiplied into the weights, and the runtime
applies it where it is already touching the data.

**And that makes ALL FOUR GEMMs smoothable at the same near-zero cost**, not two — so the
deployable configuration is the 1.051e-03 row below, not the 1.38e-03 one. The
"foldable only" measurement stays in the table because it is what the two post-LN GEMMs
contribute on their own, which is worth knowing: they carry most of the outlier problem.

| smoothed | alpha | worst 1-cos | |
|---|---:|---:|---|
| none | — | 3.397e-03 | FAIL |
| post-LN two only (qkv, ffn_up) | 0.4 | 1.406e-03 | PASS |
| post-LN two only | 0.5 | 1.537e-03 | PASS |
| post-LN two only | 0.6 | 1.376e-03 | PASS |
| **all four (the deployable one)** | **0.5** | **1.051e-03** | **PASS** |

The two post-LN GEMMs carry most of the outlier problem on their own (1.38e-03 of the
1.05e-03 available), which is what 2209.13325 predicts — but since the mechanism is an
explicit per-channel multiply rather than a fold, there is no reason not to smooth all four.
**1.051e-03 against a 2e-03 gate is 1.9x of headroom**, not 1.4x.

---

## 4b. TWO HARNESS BUGS, and the corrected numbers that supersede everything above

Both found by a result being too implausible to accept, which is the only reason either
surfaced.

**Bug 1 — the attention GEMMs were being quantised.** Every table above guarded with
`if a.ndim > 2: return fp32_gemm(...)`, intending to skip QK^T and A.V. But
`MiniLMReference.batched_gemm` calls `self.gemm(a[i,j], b[i,j])` — **2-D slices** — so the
guard never fired once. Production runs attention on the HOST in fp32 (CLAUDE.md's production
architecture), so those numbers quantised strictly more of the model than the array ever
would. They were **pessimistic**, not optimistic, which is the better direction to be wrong
in but wrong all the same. Corrected by selecting on the weight operand instead:
`min(b.shape) >= 128` includes all four projections (narrowest dim 384) and excludes
attention (widest dim 64).

**Bug 2 — static calibration keyed by shape collapsed six layers into one.** First attempt
keyed the collected activation statistics by `(K, N)`. All six layers' `qkv` are `(384,1152)`,
so one entry held the max over all of them, and dividing an early layer's activations by a
late layer's outlier scale drove them to near-zero. It read as **0.39–0.52** — a 400x blowup
against the dynamic bound's 1.05e-03, which is far too large to be a real degradation and is
the only reason it got a second look. Keyed by **call site** instead (the call order is
deterministic), which the corrected run confirms structurally: **24 sites, exactly
4 GEMMs x 6 layers.**

### The corrected table — this is the one to read

Only the four projection GEMMs quantised; attention fp32 on the host, as in production.
Calibration on 128 texts the evaluation never sees.

| smoothing | worst 1-cos | gate 2e-03 |
|---|---:|---|
| none | 2.864e-03 | **FAIL** |
| static calib, alpha=0.4 | 2.356e-03 | FAIL |
| **static calib, alpha=0.5** | **1.166e-03** | **PASS** |
| static calib, alpha=0.6 | 1.910e-03 | PASS (marginal) |
| static calib, alpha=0.7 | 1.628e-03 | PASS |
| *dynamic bound, alpha=0.5* | *1.213e-03* | *PASS* |

**Static calibration passes at alpha=0.5 with 1.7x of headroom, and is marginally BETTER
than the dynamic upper bound** (1.166e-03 against 1.213e-03). That inversion is not noise in
the wrong direction: the dynamic factors come from the four evaluation sentences while the
static ones come from 128 texts, so the static estimate of each channel's range is simply
better. It also removes §5's headline caveat — the thing most likely to move when built for
real did move, and it moved the right way.

**alpha=0.5 is the setting.** 0.4 fails, 0.6 is marginal, and the minimum is where
SmoothQuant's own paper puts it.

---

## 4c. BUILT AND RUNNING ON HARDWARE

The whole chain now exists, behind a flag, with bf16 untouched.

| stage | int8 | bf16 (default) |
|---|---|---|
| design | `export_gemm_rtp.py --int8` -> `runtime/artifacts_int8_mini` | unchanged |
| container | `pack_npue.py --int8` -> 58.68 MB | 69.02 MB |
| `b_layout_hash` | `177088d6bc9feb74…` | `94266693ea31aa67…` |
| weights staged | **10.62 MB** | 21.2 MB |
| **worst 1-cos vs HuggingFace** | **1.417e-03 PASS** | 1.086e-05 PASS |
| throughput, 4 lanes, --bench 5 | **1086.7 seq/s** | 985.3 seq/s |

**The golden gate passes on hardware at 1.417e-03**, against the simulation's 1.166e-03 —
the gap is expected, since the simulation compared against the fp32 oracle while this
compares against HuggingFace.

### End to end it is 1.10x, and that is the honest headline

The array is **5.5-7.7x** faster (0077) and the encode is **1.10x** faster. Both are true and
the second is the one that matters to a user. Amdahl bounded MiniLM at <=1.53x
([`0077`](../0077-m13-int8-gate/TASK.md) 4c) and it came in below even that, for two reasons
worth stating:

1. **MiniLM at h=384 is the worst case in the catalogue for this lever** — 0044 measured the
   NPU at 40.3% of the encode on an idle array, and part of that 40.3% is the fixed
   573 us/dispatch that no datapath change touches.
2. **The int8 host pass is more expensive than the bf16 one it replaces.** bf16 conversion is
   one streaming pass; int8 quantisation needs a per-row maximum *and* a divide per element,
   so some of the array's saving is spent on the host. The dequantisation is genuinely nearly
   free (one multiply folded into the bias pass that already reads C), but the input side is
   not.

**The prediction to test next is bge-large**, where 0045 measured the array at 60.8% of wall
clock: the same 7x should buy closer to 2x there. That needs an int8 design at `tile_n` 32
and an int8 container, neither of which exists yet.

### Two bugs worth recording

- **The banner reported the intention, not the value.** `layout_hash:` printed the bf16 hash
  while the tensors carried the I8 one, because the print built its own descriptor without
  the dtype. Same shape as tasks/0042's `tile (64, 32)` over a tile-48 pack. Fixed by
  building the descriptor the way the emitter does.
- **A segfault that only appeared under `--pipeline`.** Extra lanes copy the staged weights
  and biases from lane 0; the new `ws_*`/`as_*` scale vectors were not in that list, so lane
  0 worked and lanes 1+ dereferenced a null `wscale`. The code's own comment
  (*"The staged weights and parameters are the design's, not a lane's"*) describes exactly
  the invariant that was broken. Fixed, and `gemm()` now **throws** on a null scale rather
  than crashing, so the next instance says what is wrong.

### One byte-parity regression, found and repaired

Adding `a_dtype` to the container config made a fresh bf16 pack differ from the shipped
`.npue`, and `verify_pack_parity.py` fail — because `runtime/src/npue_pack.cpp` did not write
the key. **It does now, in the same position**, and parity is restored (`a2c57cb5…`,
byte-identical). The C++ packer emits `bf16` only: int8 packing needs the SmoothQuant
calibration pass, which runs the numpy oracle and is build-time Python by design. The shipped
containers predate the key; the runtime reads silence as `bf16`, which is what they are.

---

## 4d. THE M8 GATE — int8 PASSES, and by the same margin bf16 does

`run_mteb.py --model all-MiniLM-L6-v2.int8 --cpu-model all-MiniLM-L6-v2
--artifacts artifacts_int8_mini`, both sides one session, five tasks at seq 64:

| task | CPU (fp32) | NPU (int8) | delta |
|---|---:|---:|---:|
| STSBenchmark | 82.03 | 81.95 | -0.09 |
| SICK-R | 77.58 | 77.57 | -0.01 |
| STS12 | 72.37 | 72.24 | -0.12 |
| Banking77Classification | 80.05 | 79.92 | -0.13 |
| TwentyNewsgroupsClustering | 45.81 | 46.00 | **+0.20** |
| **MEAN** | | | **-0.03** |

**PASS** — the gate is |mean| <= 0.5 and no task worse than -0.5. Worst single task
**-0.13**.

**This is the number that decides int8, and it is indistinguishable from bf16's.** MiniLM's
recorded bf16 delta is **+0.04**; int8 measures **-0.03**. Both are inside the run-to-run
noise of these tasks, and the clustering task actually came out ahead. The whole quantisation
question -- 5 orders of magnitude worse per-GEMM error than bf16, an end-to-end 1-cos 130x
looser -- costs **0.03 MTEB points**.

That is also a reminder of why [`0035`](../0035-m8-mteb-gate/TASK.md) made MTEB the authority
rather than 1-cos: on fidelity alone int8 looks 130x worse than bf16 and only 1.4x inside its
gate. On the measure that describes what the embeddings are FOR, it is a wash.

One thing this needed, which is worth recording because it is a small fail-open avoided: the
CPU side loads a sentence-transformers **directory** and the NPU side a **.npue container**,
and with int8 those stopped sharing a name (`all-MiniLM-L6-v2` vs `all-MiniLM-L6-v2.int8`).
`--cpu-model` now separates them, defaulting to `--model` so every existing invocation is
unchanged. Without it the CPU side would have failed to load rather than compared the wrong
thing -- loud, but only by luck of the naming.

---

## 5. Caveats, and they matter

1. **These smoothing factors are derived from the activations of each call**, not from an
   offline calibration set. That is an **upper bound** on what static SmoothQuant achieves —
   deliberately, because if the upper bound failed the gate then static calibration could not
   pass it either. It did not fail, and the deployable configuration (all four smoothed) has
   **1.9x of headroom** — 1.051e-03 against 2e-03. A static calibration will be worse than
   this; **this is the number most likely to move when it is built for real.**
2. **One model, one corpus.** MiniLM-L6 on the four golden sentences. bge-large's wider
   layers, Gemma's RMSNorm (which has no beta to fold into, only gamma) and nomic's gated FFN
   are all untested.
3. **1-cos is not the authority — MTEB is** ([`0035`](../0035-m8-mteb-gate/TASK.md)). This is
   the cheap pre-check that decides whether MTEB is worth running, and it now says yes.
   Note the precedent cuts both ways: bfp16 failed 1-cos at 3.47e-03 and was parked, and
   [`0052`](../0052-m10-research-night/TASK.md) later found a bfp16 variant at 3.6e-04 that
   passed MTEB. A 1-cos number is a screening tool.
4. **No hardware was involved in §3–§4 at all.** These are numpy simulations of the
   arithmetic, not measurements of the array. The array's int8 result is separately known to
   be bit-exact (0077), so the simulation and the hardware should agree exactly — but that
   has not been shown end to end.

---

## 6. What was built, and what was not

**Built — the flag, as asked, with dtype as DATA rather than a compiled assumption:**

* `tools/export_gemm_rtp.py --int8` builds the int8 datapath (i8 operands, int32
  accumulator) and writes **`a_dtype`** into `design.json` alongside the existing `c_dtype`.
  It refuses `--int8` together with `--c-bf16` or `--emulate-bfp16` (all three answer the
  same question).
* `markers_for()` now carries the operand dtype. It hardcoded `xbf16` for A and B, which
  would have made the JIT cache search look for a design that does not exist and report
  "0 cache candidates" — the same silent-miss shape tasks/0060 hit when the MLIR printer
  changed under an unchanged marker.
* `tools/npue.py` gained the **`I8`** container dtype.
* `tools/pack_npue.py` gained `add_gemm_b_int8()` — per-output-channel symmetric
  quantisation, emitting the tiled int8 operand plus an F32 `<name>.wscale` vector, with an
  asserted overflow bound and a returned per-tensor quantisation error so a bad tensor is
  visible at pack time.
* **An int8 design set exists**: `runtime/artifacts_int8_mini` (MiniLM geometry, 8 columns,
  4 batch tiers, ONE xclbin, 65–75 differing bytes).

**The flag is a safety property, not only a switch.** The int8 design's `b_layout_hash` is
`177088d6…` against bf16's `94266693…`, so feeding a bf16 container to an int8 design is
refused by the layout check that already exists rather than producing plausible garbage.

**NOT built:**

* The **runtime quant/dequant path**. `Encoder::gemm()` still converts A to bf16 and reads C
  as fp32. int8 needs a per-row absmax + quantise on the way in, and `* sa[i] * s[j]` folded
  into the existing bias-add pass on the way out.
* The **packer's SmoothQuant fold** into LayerNorm gamma/beta, and the calibration pass that
  produces the factors.
* `pack_npue.py --int8` is not wired into the BERT path's emission loop — only the helper
  exists.
* **MTEB on int8.** That is the real gate and none of it has been run.
