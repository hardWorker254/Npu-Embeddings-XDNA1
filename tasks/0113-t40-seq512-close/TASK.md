# 0113 — T40 closed: host attention overtakes the array at seq ≈ 470

- **Date** 2026-08-25
- **Milestone** research (T40)
- **Status** done — T40 **ANSWERED**. seq 512 measured on hardware. The array
  still does not move (−2.0% across an 8× sequence range); attention scales
  **exactly linearly per token above 256** and passes the array between 256 and
  512; per token seq 512 costs **1.605×**. The tier question is answered too,
  and it is about *small* requests: a single short text costs **5.8×** on a
  seq-512 design.

## Goal

Close [T40](../../research/CLOSED-THREADS.md#t40)'s three remaining items after
[`0112`](../0112-t40-seq256-nomic/TASK.md) measured seq 256:

1. **seq 512**, which 0112 found needs a repack (the 256 cap is the container's
   `max_seq_len` config field, not the position table — so nomic is capped too).
2. **The tier table**: `batch = M / seq` shrinks it, and whether real request
   shapes still fill a tier was untouched.
3. The thread's own decision rule, which 0112 satisfied the *first* half of:
   > If host attention has not overtaken the array by 256, 512 is worth a
   > repack; if it has, the answer is that windowing on the caller's side is
   > the right design and this thread retires.

   At 256 it had not (attention 33.6% of wall against the array's 62.0%). So
   512 was worth the repack, by the thread's own rule.

## Context

Read: [`0112`](../0112-t40-seq256-nomic/TASK.md) in full (the method this
repeats, and the two corrections it made to T40's filing),
[`0110`](../0110-refuse-silent-truncation/TASK.md) (why `--seq` is cheap),
CLAUDE.md F3 and rule 1.

## Commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# 1. the repack 0112 found to be necessary -- a SEPARATE container, so the
#    shipped one is never touched
.\.venv-ref\Scripts\python.exe tools\pack_npue.py --model-dir models\nomic-embed-text-v1.5 --out models\nomic-embed-text-v1.5.s512.npue --max-seq 512

# 2. the design
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python tools\export_gemm_rtp.py --batches 4,16 --batch 16 --cols 8 --hidden 768 --intermediate 3072 --gated-ffn -n 48 --seq 512 --emulate-bfp16 --c-bf16 --out runtime\artifacts_nomic_bfp16_seq512

# 3. goldens -- SEQ_LEN raised to 512 in reference/corpus_nomic.py, then restored
.\.venv-ref\Scripts\python.exe reference\make_goldens_nomic.py --taps
.\.venv-ref\Scripts\python.exe tools\export_validation.py --model nomic-embed-text-v1.5.s512 --seq 512

# 4. the bench, 3 runs
.\runtime\build\npuembed.exe . --model nomic-embed-text-v1.5.s512 --artifacts artifacts_nomic_bfp16_seq512 --threads 24 --pipeline 4 --bench 5

# 5. the tier probe -- 1, 4 and 64 texts through each design, single lane
.\runtime\build\npuembed.exe . --model <m> --artifacts <a> --prefix search_document --embed tasks\0113-t40-seq512-close\corpus_<n>.txt <out> --threads 24 --pipeline 1
```

**Note the improvement over 0112's procedure**: `export_validation.py --model
nomic-embed-text-v1.5.s512` writes to `runtime/artifacts/validation/nomic-embed-text-v1.5.s512`,
a directory of its own. 0112 had to back up and restore the shipped seq-64
fixtures because it reused the model name; giving the long-sequence container
its own name removes that hazard entirely. The only thing still shared is
`reference/corpus_nomic.py`'s `SEQ_LEN`, which was raised to 512 and restored
(`git diff` empty).

The export was again a cache hit on both tiers — at seq 512, batches 4 and 16
give `M` 2048 and 8192, which the shipped seq-64 export had already compiled.
`M` is the JIT cache key; `seq` is not in it. All seven identity checks passed.
The repack produced the same `layout_hash 94266693ea31aa67…` as the shipped
container, which is what makes the two designs comparable at all.

## Contention record — and an honest gap in it

Runs 1 and 2 of the seq-512 bench sit in a window where the foreign
`WorkloadsSessionHost.exe` contexts **did** submit: sampled `16360:14 …
17144:3` before the session and `16360:26 … 17144:5` after, so **14 foreign
submissions occurred at unknown times** across that block. That is a smaller
claim than 0111/0112 could make, and it is stated rather than smoothed over.

So a **third run was taken, bracketed by samples immediately before and after**,
both reading `9028:0 16360:26 16756:0 17336:4 17144:5 14008:0 14008:0` —
identical, zero growth during the run:

| run | wall ms | NPU ms | attn ms | contention |
|---|---:|---:|---:|---|
| 1 | 3300.27 | 1535.90 | 1731.5 | window had 14 foreign submissions |
| 2 | 3297.74 | 1529.18 | 1723.5 | window had 14 foreign submissions |
| **3** | **3304.17** | **1547.10** | **1706.7** | **bracketed clean, zero growth** |

The bracketed run reproduces the other two to within 0.6% (wall), 1.2% (NPU)
and 1.4% (attn), which is what retires the concern — not an argument that the
submissions were harmless, but a measurement taken while there were none.

## Result — the whole seq curve

All three points encode **exactly 163,840 tokens** per bench
(4×128×64 = 4×32×256 = 4×16×512), which is what makes them comparable at all.
seq 64 and 256 are the means of two runs each from
[`0112`](../0112-t40-seq256-nomic/TASK.md); seq 512 is the mean of the three
above.

| | seq 64 | seq 256 | seq 512 |
|---|---:|---:|---:|
| batch (same M = 8192) | 128 | 32 | 16 |
| **tokens/s** | **79,671** | **65,271** | **49,637** |
| **cost per token** | **1.000×** | **1.221×** | **1.605×** |
| **NPU dispatch+wait, ms** | 1568.8 | 1554.5 | **1537.4** |
| NPU share of wall | 76.3% | 62.0% | **46.6%** |
| **host attn, ms** | 339.1 | 843.5 | **1720.6** |
| attn, relative to seq 64 | 1.00× | 2.49× | **5.07×** |
| **attn share of wall** | 16.5% | 33.6% | **52.1%** |
| host elt, ms | 202.4 | 265.5 | 337.8 |
| host bias, ms | 539.7 | 487.5 | 403.7 |
| host conv, ms | 32.0 | 33.1 | 22.2 |

### 1. The array does not move, across an 8× range

**1568.8 → 1554.5 → 1537.4 ms: −2.0% end to end**, on identical `M` and
identical 192 dispatches per group. 0110 derived this by reading the code; two
tasks have now measured it at three sequence lengths. `seq` really is a
host-side slicing convention over a fixed row count.

### 2. Attention scales linearly per token above 256 — the discount is exhausted

| interval | seq factor | attn factor |
|---|---:|---:|
| 64 → 256 | 4× | **2.49×** (sublinear) |
| 256 → 512 | 2× | **2.04×** (linear) |

Attention is O(seq²) per sequence and therefore O(seq) per token, so 2.04× for a
2× sequence is the textbook answer. The **2.49×-not-4×** discount between 64 and
256 was longer rows amortising the AVX2 loop's per-row overheads, and it is a
small-seq effect that is **spent by 256**.

**This corrects 0112's own extrapolation, which was mine.** 0112 projected
"attention near 2,100 ms at 512" by assuming the sublinear factor persisted.
Measured: **1,720.6 ms — 22% below that projection**, because the assumption was
wrong in the direction that flattered the extrapolation's *shape* while
overshooting its value. The lesson is the ordinary one about extrapolating a
scaling law from a single interval, and it is exactly why the register asked for
the measurement rather than accepting the projection.

### 3. The crossover: seq ≈ 470

At 512, host attention (1720.6 ms) **exceeds** the whole array (1537.4 ms).
Solving in the linear regime — `attn(seq) ≈ 843.5 × seq/256 = 3.295 · seq` ms
against a flat ~1,540 ms of array time:

> **Host attention overtakes the array at seq ≈ 470**, on nomic, at constant
> `M = 8192`.

That is T40's central question, with a number on it. **F3's premise does not
survive it**: CLAUDE.md prices attention at 2–5% of the work, which is a seq-64
figure (and, per [`0109`](../0109-fused-ratio-energy/TASK.md), a bf16-era one).
At 512 attention alone is 52.1% of wall clock. F3's *conclusion* — leave
attention on the host — is a separate question that still runs into the
head_dim-64 tiling problem [T38](../../research/OPEN-THREADS.md#t38) is about,
and nothing here touches it.

Note also that the two per-token buckets *fall* as seq grows: `bias` 539.7 →
403.7 and `conv` 32.0 → 22.2 on identical row counts, i.e. fewer, longer
contiguous runs are cheaper per byte. That is a real (if small) argument in
favour of long sequences, and it is the only one this task found.

## Result — the tier table, item 2

`plan()` is greedy over the tier ladder and `use_tier()` rounds a job **up** to
the smallest tier that fits, so the ladder's *span* is not what matters — the
**smallest tier's token footprint** is. In token terms the two ladders are:

| design | tiers (sequences) | tiers (token slots) | smallest job |
|---|---|---|---|
| shipped, seq 64 | 4, 16, 32, 128 | 256, 1024, 2048, 8192 | **256 slots** |
| seq 512 | 4, 16 | 2048, 8192 | **2048 slots** |

8× more padded work for the smallest possible request. Measured rather than
argued — same texts, single lane, both designs:

| request | seq 64 | seq 512 | penalty |
|---|---:|---:|---:|
| 1 text | 0.05 s | 0.29 s | **5.8×** |
| 4 texts | 0.05 s | 0.30 s | **6.0×** |
| 64 texts | 0.36 s | 4.14 s | **11.5×** |

The 1- and 4-text rows are the padding effect alone (both round to the same
4-sequence tier, 256 vs 2048 token slots) and land at 5.8–6.0× against the 8×
the slot count predicts — the gap being the same per-token cheapening seen in
`bias`/`conv`. The 64-text row is larger because it stacks the padding penalty
on top of the genuine 8× token count and attention's 5× per-token growth.

**So the tier answer is: a long-sequence design is a bad instrument for short
inputs, and it is the smallest tier, not the ladder's span, that makes it one.**
Windowing on the caller's side — the design the thread anticipated — is right
for a *mixed* workload, but for the reason measured here rather than the one
predicted: not because attention makes long sequences unaffordable (1.605× per
token is affordable), but because a seq-512 design charges a 1-token request for
2,048 tokens.

## What T40 now says, in one line

Long sequences work, cost **1.605× per token at 512**, leave the array
untouched, and put **attention past the array at seq ≈ 470** — while making
short requests 5.8× more expensive. Nothing here is blocked, and nothing is
left unmeasured on this thread.

## Problems hit

1. **The contention record for the first two seq-512 runs is weaker than
   0111/0112's**, as described above. Resolved by a third bracketed run rather
   than by argument, but the gap is real and is why that run exists.
2. **`SEQ_LEN` in `reference/corpus_nomic.py` is still a module constant**, so
   every new sequence length needs an edit-and-restore around the golden
   regeneration. 0112 filed this as the same hardcoded-seq class 0110 removed
   from `export_gemm_rtp.py`; this task hit it a second time and still did not
   fix it — it wants a `--seq` flag threaded through
   `make_goldens_nomic.py`/`corpus_nomic.py`, which is a change to the
   *reference* pipeline and out of scope for a measurement task. Filed as
   [T41](../../research/OPEN-THREADS.md#t41).

## Artifacts

* `corpus_1.txt`, `corpus_4.txt`, `corpus_64.txt` — the tier probe inputs
* `models/nomic-embed-text-v1.5.s512.npue` — the repacked container (322.87 MB,
  same `layout_hash` as the shipped one)
* `runtime/artifacts_nomic_bfp16_seq512/` — the design, with `toolchain.json`
* `runtime/artifacts/validation/nomic-embed-text-v1.5.s512/` — its own fixtures,
  which is why nothing shipped had to be backed up this time
* `reference/goldens_nomic/nomic-embed-text-v1.5_l12_s512_{boundary,taps}.safetensors`
  (88.1 MB / 4026.6 MB), two-oracle check passed at 1.356e-06

## Next

T40 closes. The follow-on it leaves is [T41](../../research/OPEN-THREADS.md#t41)
(the golden pipeline's hardcoded `SEQ_LEN`), which is a tooling debt rather than
an open question about the hardware.
