# 0140 — gte at long sequences: the NTK reading verified where it matters, and the seq-256/512 design sets serve the new model unchanged

**Date**: 2026-08-27
**Goal**: Phase 8 of the 0.5.0 plan — long sequences for gte by reusing the
nomic seq-256/512 design sets (same `b_layout_hash`), with the one check
that only long positions can perform: the NTK RoPE scaling exists *for*
long context, so a wrong reading grows with position and the 18-token probe
of 0134 could not have caught it.

## 1. The s512 container and the seq goldens

Following the nomic precedent exactly (`nomic-embed-text-v1.5.s512.npue` —
a separate container, so the main one and the C++ hub pack stay
byte-identical at `max_seq 64`):

```
.venv-ref\Scripts\python tools\pack_npue.py --model-dir models\gte-multilingual-base --out models\gte-multilingual-base.s512.npue --max-seq 512
.venv-ref\Scripts\python reference\make_goldens_gte.py --seq 256   # two-oracle agreement 6.706e-08, PASS
.venv-ref\Scripts\python reference\make_goldens_gte.py --seq 512   # same figure, PASS
```

1,002.32 MB, same `layout_hash 94266693…`; goldens at
`reference/goldens_gte/gte-multilingual-base_l12_s{256,512}_boundary.safetensors`
(0116's `--seq` mechanism, byte-coexisting with the s64 golden).

## 2. NTK at long positions — the discriminator GROWS as predicted

[`1_ntk_long_positions.py`](1_ntk_long_positions.py) (output in
[`ntk_long.txt`](ntk_long.txt)): a 512-token input through the repaired
fp32 reference vs the numpy oracle:

```
tokens: 512
last_hidden relfro 1.157e-05      CLS relfro 3.060e-06
layer0 right relfro 9.945e-07     control theta=20000-plain relfro 4.205e-02
```

The oracle holds fp32-noise agreement out to position 511, and the
wrong-theta control's layer-0 miss **grew from 1.9e-02 at 18 tokens (0134)
to 4.2e-02 at 512** — rotation angle is position × frequency, so a theta
error compounds with length, and this is the regime where it would have
bitten. The NTK reading is now verified at both ends of the range it
serves.

## 3. Hardware: both long-seq design sets serve gte in the bfp16 decade

[`2_hw_seq512.py`](2_hw_seq512.py) (output in [`hw_seq.txt`](hw_seq.txt)):
the s512 container on `artifacts_nomic_bfp16_seq512` and `_seq256`, four
texts per set **constructed to fit** each design's sequence length
(tokenize-while-repeating — see problems), English long/medium/short plus a
Norwegian long text, each compared against the numpy oracle's normalized
CLS:

```
seq512 (batch 16 x seq 512): 497 tok 4.34e-04 | 227 tok 1.88e-04 | 17 tok 2.16e-04 | 506 tok (no) 3.19e-04
seq256 (batch 32 x seq 256): 227 tok 1.88e-04 | 92 tok 2.46e-04 | 17 tok 2.16e-04 | 254 tok (no) 2.90e-04
```

All in the datapath's measured decade (0136's seq-64 range was
2.2–3.7e-04), the status line reports the adopted bfp16 datapath on both
sets, and the 227-token text reproduces **identically (1.881e-04) across
the two design sets** — per-input determinism across designs, for free.

The 0113 caveat carries over verbatim: a long-sequence design is a bad
instrument for short requests (`use_tier()` rounds up; seq 512's smallest
tier holds 4×512 = 2,048 slots). The seq-64 sets remain the default; the
long-seq sets are for long inputs.

## Problems hit

* First harness leaned on `--max-len` to truncate — but `embed` **refuses**
  oversize inputs by design (0110), naming the 902-token input and the
  design's length. The refusal is the feature working; the harness now
  fits texts to `seq - 2` tokens by construction. (Also two patch-script
  string-match failures from line-wrapping — resolved with smaller
  anchors; recorded because half-applied patches are how stale-harness
  bugs happen.)

## Status

Phase 8 done. T42 (attention on the array for long sequences) remains
priced, not built — nothing here changes its trigger, but gte is now the
second model the long-seq sets serve, which is the demand signal T42's
trigger watches.
