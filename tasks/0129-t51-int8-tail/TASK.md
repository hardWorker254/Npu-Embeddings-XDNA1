# 0129 — T51 step 1: the tail is the input's, not the tile's — int8 at tile_n 64 reproduces the same worst words as bfp16 at tile_n 32

**Date**: 2026-08-27
**Goal**: T51's cheapest discriminator — run 0122's 224 texts through the
already-shipping **int8 bge-large at `tile_n = 64`** (no build), which changes
the tile geometry AND the datapath while holding depth (24) and width (1024).
0122 left three candidate discriminators standing: depth, width, `tile_n` 32
(bge-large is the only model not at 48).

## Commands run

```
.venv-ref\Scripts\python tasks\0129-t51-int8-tail\1_int8_tail.py    # distributions, both arms
.venv-ref\Scripts\python tasks\0129-t51-int8-tail\2_rank_correlation.py  # per-text correlation
```

Corpus is 0122's byte for byte (180 vocab-drawn single words + 36 semantic-
corpus sentences + 8 five-word phrases); reference is fp32 `AutoModel` CLS,
shared by both arms; the bfp16 arm re-ran in the same session as control.
Full output in [`raw.txt`](raw.txt); per-text error vectors saved as
`e_int8n64.npy` / `e_bfp16.npy`.

## Results

| arm | median | p99 | max | over 2e-03 | max/median |
|---|---:|---:|---:|---:|---:|
| int8 `tile_n=64` | 2.212e-03 | 1.872e-02 | **1.082e-01** | 138/224 (61.6%) | 48.9× |
| bfp16 `tile_n=32` (control) | 2.179e-04 | 6.626e-03 | 2.183e-02 | 5/224 (2.2%) | **100.2×** |

The control reproduces 0122's distribution to the digit (median 2.18e-04, max
2.18e-02, same five violators) — no environment drift.

**The worst five are the SAME WORDS on both arms**: `newsletter` (1.08e-01
int8 / 2.18e-02 bfp16), `contact`, `sermons`, `cinema`, `messages`. Across
all 224 texts: **Pearson on log-errors 0.834**, Spearman 0.551 (0.533 on the
180 words alone), **top-10 overlap 8/10** (`cinema, contact, donation,
essays, messages, movies, newsletter, sermons`).

## What is now excluded, and what stands

* **`tile_n` is exonerated.** 32 → 64 changes the entire B-layout and
  iteration structure; the same words sit at the top of the tail.
* **The number format is exonerated as the cause.** bfp16 block-float and
  SmoothQuant-calibrated W8A8 share no mechanism at the arithmetic level,
  yet they rank the same inputs worst (log-Pearson 0.834). The tail is a
  property of **(model × input)**: bge-large's forward pass on these
  single-word inputs amplifies *any* low-precision perturbation. int8's
  larger absolute level (its median already sits above the 2e-03 gate,
  consistent with 0085's known int8 bge-large 1-cos FAIL at 2.968e-03 —
  measured mean here 3.297e-03) scales the same underlying sensitivity.
* **Still standing as the discriminator vs bge-base**: depth (24 vs 12),
  width (1024 vs 768), or something about bge-large's learned
  representations of degenerate one-word inputs. Only T51 step 2 (layer-wise
  divergence) separates those.
* Observation recorded, not pursued: the shared worst words read like web
  boilerplate/navigation vocabulary (`newsletter`, `contact`, `donation`,
  `messages`) — possibly tokens whose training-data contexts are unusually
  uniform. A hypothesis for step 2 to check against layer profiles, nothing
  more.

## Consequence for the tail gate (T51 step 3)

The gate must be **per-model and per-datapath on a p99**, not a max: int8
bge-large's *median* is already over 2e-03 (its adoption verdict rests on
MTEB, per 0085), so a max-based gate would tell us nothing new about it,
while bfp16 bge-large's p99 (6.6e-03) against its median (2.2e-04) is
exactly the signal the sweep's existing instruments average away.

## Status

**T51 stays OPEN**: step 1 done (this task), step 3 (the p99 gate in the
release sweep) pending, step 2 (mechanism) optional and now better-scoped —
it should profile depth, not tiling.
