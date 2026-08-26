# 0112 — T40 measured at seq 256: the array does not move, host attention 2.49×, and 1.22× per token

- **Date** 2026-08-25
- **Milestone** research (T40)
- **Status** done — T40 **PARTLY ANSWERED**. seq 256 runs today on nomic, costs
  **1.221× per token** end to end, and **every bit of that cost is host-side**:
  array time is unchanged to within 0.9% on the same `M`. Attention becomes the
  **largest host bucket** and the array/host balance inverts. seq **512** is
  still unmeasured and does need a repack — including for nomic, correcting
  what this thread said.

## Goal

[T40](../../research/OPEN-THREADS.md#t40) was filed by
[`0110`](../0110-refuse-silent-truncation/TASK.md), which made long sequences
*expressible* (`tools/export_gemm_rtp.py --seq`) without making any claim about
what they cost. The thread named its own cheapest decisive experiment:

> export nomic at `--seq 256 --batch 32` (same M = 8192, no repack needed since
> nomic has no position table), trace it, and compare *host* time per token
> against the shipped seq-64 build.

This is that experiment. The question behind it is F3: CLAUDE.md prices host
attention at **2–5% of the work** — measured at seq 64 — and attention is
O(seq²) while array work at constant `M` is not, so the ratio F3 rests on has to
invert somewhere. Nobody had found where.

## Context

Read: [`0110`](../0110-refuse-silent-truncation/TASK.md) (what `--seq` is and
why it is cheap), [`0109`](../0109-fused-ratio-energy/TASK.md) (the array/host
balance on the fused build — the baseline this moves), CLAUDE.md F3 and rule 1,
[`0104`](../0104-adopt-bfp16-per-model/TASK.md) (nomic's exact shipped export
line, reproduced below with only `--seq`/`--batch` changed).

## Contention record (rule 1)

Same session as [`0111`](../0111-t13-pretiled-stability-at-scale/TASK.md).
Foreign `hw_context` submission counts sampled before, during and after, all
identical — seven parked `WorkloadsSessionHost.exe` contexts, **zero
submissions**. The runtime's own `npu exclusive -- 8 hw_context(s), none Active
but ours` line appears in every bench output below.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# the seq-256 design -- 0104's nomic line, with --seq 256 and the tiers rescaled
python tools\export_gemm_rtp.py --batches 4,32 --batch 32 --cols 8 --hidden 768 --intermediate 3072 --gated-ffn -n 48 --seq 256 --emulate-bfp16 --c-bf16 --out runtime\artifacts_nomic_bfp16_seq256

# seq-256 goldens (see "the friction nobody knew about" below)
.\.venv-ref\Scripts\python.exe reference\make_goldens_nomic.py --taps
.\.venv-ref\Scripts\python.exe tools\export_validation.py --model nomic-embed-text-v1.5 --seq 256

# the A/B, two runs per arm
.\runtime\build\npuembed.exe . --model nomic-embed-text-v1.5 --artifacts artifacts_nomic_bfp16        --threads 24 --pipeline 4 --bench 5
.\runtime\build\npuembed.exe . --model nomic-embed-text-v1.5 --artifacts artifacts_nomic_bfp16_seq256 --threads 24 --pipeline 4 --bench 5
```

The export was clean: 8 streams (4 shapes × 2 tiers), all seven
xclbin-identity-mod-UUID checks passed, `toolchain.json` written. Note it was
also **fast**, and for the reason T40 predicted: at seq 256 the tiers 4 and 32
give `M` 1024 and 8192, both of which the shipped seq-64 export had already
compiled, so the instruction streams were cache hits. `M` is the cache key;
`seq` is not in it.

## Result — the headline

Both arms encode **exactly 163,840 tokens** per bench (5 groups × 4 lanes ×
batch × seq), which is what makes them comparable: 4×128×64 = 4×32×256.

| | seq 64 (batch 128) | seq 256 (batch 32) | ratio |
|---|---:|---:|---:|
| wall, ms | 2022.3 / 2091.7 | 2577.9 / 2445.8 | |
| sequences/s | 253.2 / 244.8 | 49.7 / 52.3 | 0.205× |
| **tokens/s** | **81,015 / 78,327** | **63,555 / 66,987** | **0.819×** |
| **NPU dispatch+wait, ms** | **1555.8 / 1581.8** | **1544.7 / 1564.3** | **0.991×** |
| NPU share of wall | 76.9% / 75.6% | 59.9% / 64.0% | |
| host attn, ms (lane p1) | 337.5 / 340.7 | 857.0 / 829.9 | **2.487×** |
| host elt, ms | 194.9 / 209.8 | 269.7 / 261.2 | 1.312× |
| host bias, ms | 504.7 / 574.6 | 499.4 / 475.6 | 0.903× |
| host conv, ms | 29.3 / 34.6 | 34.3 / 31.8 | 1.034× |

**Per token, seq 256 costs 1.221×.** That is the whole answer to "what does a
long-sequence design cost".

### 1. The array does not move — measured, not argued

**1554.5 ms against 1568.8 ms, −0.9%.** 0110 established by *reading* that
`seq` enters only as `M = batch·seq` and that the instruction streams know only
`M`, `K`, `N`. This is that claim on hardware: same `M`, same array work,
4× the sequence length. The 192 dispatches per group are identical too.

### 2. Host attention is 2.49×, not 4×

Attention is O(seq²) per sequence, so per *token* it is O(seq) — 4× predicted
for a 4× sequence at constant token count. Measured **2.487×**. The gap is real
and in our favour: longer rows amortise the per-row overheads of the AVX2 loop,
so the constant improves even as the order grows. It does **not** rescue the
scaling — it discounts it by about a third.

`elt` (softmax over the attention scores, also O(seq²)) moves **1.312×**, and
the two per-token buckets behave exactly as they should: `bias` (the C readout,
per row) and `conv` (bf16 conversion, per row) are **flat** — 0.90× and 1.03×
on identical row counts.

### 3. The balance inverts, and attention becomes the biggest host bucket

| | seq 64 | seq 256 |
|---|---:|---:|
| array share of wall | **76.3%** | **62.0%** |
| lane host work share | 55.4% | 65.4% |
| attention as share of wall | 16.5% | **33.6%** |
| attention as share of lane host work | 30.4% | **52.0%** |
| largest host bucket | `bias` (539.7 ms) | **`attn` (843.5 ms)** |

At seq 64 attention is the *second* host bucket and a sixth of wall clock. At
256 it is **the** host bucket and a third of wall clock.

**What this does to F3.** CLAUDE.md prices attention folding at 1.4%
end-to-end, from AMD's BERT measurement, and [`0109`](../0109-fused-ratio-energy/TASK.md)
already flagged that number as bf16-era. It is a seq-64 number as well. At seq
256 attention is 33.6% of wall clock on this model — F3's *conclusion* (leave
attention on the host) may well survive, since folding it onto the array runs
into the head_dim-64 tiling question [T38](../../research/OPEN-THREADS.md#t38)
is about, but F3's *premise* — that attention is a rounding error — does not
survive past seq 64.

**Extrapolating to 512, labelled as extrapolation.** If the same sublinear
factor holds (2.49× per 4× of seq), attention at 512 lands near 2,100 ms
against an unchanged ~1,555 ms of array time, i.e. attention alone exceeding
the whole array. Nothing here measures that, and the extrapolation rests on one
interval.

## The friction nobody knew about: goldens are seq-shaped

T40 said nomic needs no repack, and for the **container** that is true. It is
not true for the **validation fixtures**, and the bench refuses without them:

```
error: ./runtime/artifacts/validation/nomic-embed-text-v1.5/emb_sum.f32:
       expected 3145728 bytes, found 786432
```

The fixtures are sized `kGoldenBatch × seq × hidden`, so a seq-256 design wants
1024 rows where the shipped seq-64 fixtures hold 256. Regenerating them needs
the torch oracle **and** an edit to a module constant:

```python
# reference/corpus_nomic.py:18
SEQ_LEN = 64
```

— a hardcoded sequence length of exactly the kind
[`0110`](../0110-refuse-silent-truncation/TASK.md) removed from
`export_gemm_rtp.py`, one layer further out in the reference pipeline. It was
edited to 256 for this task and **restored**; the seq-64 fixtures were copied to
`validation-seq64-backup/` in this directory first and copied back afterwards,
verified by re-reading `validation.json` (`rows 256`) and by the seq-64 bench
re-running clean after the restore. `git diff reference/corpus_nomic.py` is
empty.

Worth knowing: the golden regeneration itself passed its own two-oracle check at
seq 256 (`max abs diff 1.356e-06`, native transformers vs SentenceTransformer
with `trust_remote_code`), so nothing about seq 256 troubles the reference path
— it just has to be *run*.

## A correction to T40's own item 2

T40 says the 256-position cap applies to BERT-family models and that *"nomic
(RoPE) has no such table"*. The table is not what enforces the cap:

```cpp
// runtime/src/main.cpp:474, 597
g_max_positions = m.config_int("max_seq_len");
if (seq > g_max_positions) throw ...
```

It is a **config field**, and nomic's container carries `max_seq_len: 256` like
every other. seq 256 works only because it sits exactly *at* the limit. **seq
512 needs a repack for nomic too** — a cheap one, since `pack_npue.py:970`
writes nomic's position table as zeros (`np.zeros((max_seq, hidden))`, a
placeholder RoPE never reads), but a repack nonetheless. The cheap experiment
was nomic; the cheap *next* experiment is not free.

## What is still open

1. **seq 512 is unmeasured**, and needs `pack_npue.py --max-seq 512` first.
2. **The tier table.** This export used tiers 4 and 32 (`M` 1024 and 8192)
   because batch 32 is the largest tier at seq 256 and `--batch` must be the
   largest. That is an 8× span against the shipped design's 32× (4–128).
   Whether real request shapes still land in a tier is unexamined — T40's third
   item, untouched here.
3. **No trace.** These are `--bench` numbers: wall clock end-to-end plus the
   runtime's own host/NPU split, which is what the question needed (host time
   per token). No per-core trace was taken, and none of the above is a
   kernel-cycle claim.

## Artifacts

* `corpus_2080.txt` — the 520-sentence corpus ×4, used for the `--embed` probe
  that established the seq-256 design encodes correctly before benching
  (2080 texts, 8.72 s)
* `validation-seq64-backup/` — the shipped seq-64 fixtures, copied before the
  regeneration and restored after
* `runtime/artifacts_nomic_bfp16_seq256/` — the design itself, with its
  `toolchain.json`
* `reference/goldens_nomic/nomic-embed-text-v1.5_l12_s256_{boundary,taps}.safetensors`
  — the seq-256 goldens (44.1 MB / 1560.3 MB), kept: they are what a repeat of
  this measurement needs, and they do not collide with the seq-64 pair

## Next

T40 stays open on items 1–3 above. The thread's own framing was right about the
cheap experiment and wrong about the repack; both are recorded in the register.
