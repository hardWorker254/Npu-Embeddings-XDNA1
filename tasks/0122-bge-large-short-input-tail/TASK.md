# 0122 — bge-large has a 100× tail on single-word inputs, and every gate we own averages it away

**How this started.** The user obtained embedding vectors for the word `"test"`
from an online service — `bge-small-en-v1.5` first, then `bge-large-en-v1.5` —
and asked us to compare. That is the first check this project has ever run
against a reference with **no shared code lineage with us at all**, and it
immediately found something three existing gates cannot see.

**Result.**

> **bge-large's `1-cos` against a full-precision reference is 2.18e-04 at the
> median and 2.18e-02 at the worst — a 100× spread — and 5 of 224 texts exceed
> the project's own 2e-03 tolerance. The entire tail lives in SINGLE-WORD
> inputs. Sentences and phrases are within 1.24× of the median.
> `bge-base-en-v1.5`, on the identical bfp16 datapath, has a max/median of 3.6×
> and zero violations.**

Deterministic, reproduced across two independent reference implementations and
three batch compositions. **Two mechanisms were tested and refuted; the cause is
unknown.**

---

## 1. The external reference, and what it could and could not settle

The user's vectors are quoted to **three decimals**, which puts a floor under
any comparison: rounding alone permits ±0.0005 per component, and a uniform
rounding error has mean |e| = 0.00025. So this can only place a **ceiling** on
our error.

| model | datapath | cosine | `1-cos` | mean \|Δ\| | rounding floor |
|---|---|---:|---:|---:|---:|
| `bge-small-en-v1.5` | bf16 MMAC, C fp32 | 0.999976 | **2.38e-05** | 0.00029 | 0.00025 |
| `bge-large-en-v1.5` | bfp16 MMAC, C bf16 | 0.997870 | **2.13e-03** | 0.00162 | 0.00025 |

**bge-small sits exactly on the floor.** Rounding a unit vector of 384
components to three decimals predicts `1-cos ≈ 1.6e-05` on its own; we measure
2.38e-05, leaving ~0.8e-05 for our own error — which agrees with the 8.348e-06
`CURRENT_STATUS` records for it. Two independent corroborations arrived free:
the reference's own L2 norm is **1.000176**, the signature of a rounded unit
vector, so the service normalises as we do; and min/max agree to the third
decimal.

**bge-large does not.** Its mean |Δ| is 6.5× the rounding floor and its max is
15× it. That is beyond what quoting can explain, so it had to be re-measured
against something sharper.

## 2. Confirmed against full precision — the reference was right, we are off

`tools/verify_embed_e2e.py` (the repo's own harness, `.venv-ref`,
sentence-transformers at fp32) on the single word `test`:

```
    #        1-cos   text
    0    2.234e-03   'test'
  worst 1-cos 2.234e-03   (tolerance 2e-03)
```

**2.234e-03 against our own 2e-03 gate**, and within 5% of the 2.130e-03
measured against the user's 3-decimal vectors — the residual being exactly the
quoting. The external reference is correct.

Three control questions, each its own run:

| question | answer |
|---|---|
| Is it short inputs generally? | **No.** MiniLM 1.8×, bge-small 1.2×, bge-base **1.0×**, bge-large **10.4×** (short vs normal sentence) |
| Does it depend on the batch? | **No.** 2.234e-03 alone, with 1 companion, and with 36 — identical to four figures |
| Is it length? | **No.** `rain` 4.20e-04 and `mechanic` 4.44e-04 are the same 3 tokens as `test` at 2.23e-03 |

That last one is what turned this from "short inputs are lossy" into "something
input-specific is happening", and it is what made the mechanism worth chasing.

## 3. Two mechanisms, tested and refuted

**3a. Cancellation — REFUTED.** bf16/bfp16 rounding lands on the pooled vector
as a roughly constant *absolute* perturbation while `1-cos` is *relative*, so an
unusually short pre-normalisation CLS vector would buy a larger angle for the
same noise. Prediction: error ∝ 1/‖CLS‖².

Measured (`3_refute_cancellation.py`), 13 texts: **‖CLS‖ is 16.5–19.1 for every
one**, and `test` has the **largest** norm at 19.101. `corr(log‖CLS‖, log 1-cos)
= +0.111` where the hypothesis requires ≈ −1. Dividing by ‖CLS‖² makes the
spread *worse* (15.4× against 13.8×).

> **The first attempt at this probe was VOID and is kept as
> `2_void_cancellation_probe.py`.** It asked sentence-transformers for
> unnormalised vectors with `normalize_embeddings=False` and got ‖v‖ = 1.000 for
> all 13 — because bge-large's ST config carries a **Normalize module in the
> pipeline** and the flag governs only the final step. The output looked like a
> clean refutation and measured nothing; the giveaway was the derived column
> being bit-identical to the input column. `AutoModel` +
> `last_hidden_state[:, 0]` is the correct instrument.

**3b. Block-floating-point outliers — REFUTED.** The adopted datapath is
`--emulate-bf16-mmul-with-bfp16`, and bfp16 shares one exponent across a block
of **8** (`to_v64bfp16ebs8`), so a single outlier channel costs the other seven
their mantissa bits. This is the classic block-FP failure mode and the same
phenomenon SmoothQuant exists to fix for int8
([`0078`](../0078-m13-int8-accuracy/TASK.md)).

Measured (`4_refute_block_fp.py`) as the hardware sees it — mean over aligned
groups of 8 of `max|x| / rms|x|`, over all 25 hidden states, real tokens only:

| | range across 13 texts | `test` | corr with log `1-cos` |
|---|---|---|---|
| crest factor, all tokens | 1.8787 – 1.9087 | 1.8943 | **−0.000** |
| crest factor, CLS row | 1.8737 – 1.9021 | 1.8846 | −0.203 |
| `max|h|` | 17.6 – 18.0 | 17.7 | −0.275 |

**No input statistic predicts the error.** That is itself the finding, and it is
what redirected the investigation from *cause* to *distribution*.

## 4. What it actually is: a heavy tail, single words, bge-large only

224 texts — 180 single words drawn deterministically from the model's own vocab
(`sorted(vocab)[::stride]`, so they are not hand-picked), 36 corpus sentences, 8
five-word phrases — against fp32 `AutoModel` CLS:

| `bge-large-en-v1.5` | n | median | max | over 2e-03 |
|---|---:|---:|---:|---:|
| sentences | 36 | 2.00e-04 | **2.47e-04** | 0 |
| 5-word phrases | 8 | 1.89e-04 | 2.60e-04 | 0 |
| **single words** | 180 | 2.27e-04 | **2.18e-02** | **5** |
| all | 224 | 2.18e-04 | 2.18e-02 | 5 (2.2%) |

`mean 4.52e-04 · p90 3.69e-04 · p99 6.63e-03 · max/median 100.2×`

| `bge-base-en-v1.5` | n | median | max | over 2e-03 |
|---|---:|---:|---:|---:|
| all | 224 | 1.72e-04 | 6.13e-04 | **0** |

`p99 3.71e-04 · max/median 3.6×`

The five over the line: `newsletter` **2.183e-02**, `sermons` 1.195e-02,
`contact` 7.940e-03, `cinema` 2.227e-03, `messages` 2.222e-03. **`test` is only
sixth**, at 2.234e-03 — the input that started this is not close to the worst.

**Reproduced independently.** The five were re-run on their own through
`tools/verify_embed_e2e.py`, which uses sentence-transformers rather than the
`AutoModel` path my script uses — two different reference implementations, a
different batch composition, and the numbers agree to four significant figures.
It is not a harness bug.

## 5. Why no gate we own can see this

| gate | on bge-large | why it misses |
|---|---|---|
| `1-cos`, `CURRENT_STATUS` | **2.626e-04 — PASS** | a mean over a small golden corpus of sentences; it is the *median*, and the tail is not in the corpus |
| MTEB delta | **+0.13 / −0.01 — PASS** | averages thousands of pairs; the docs already say `1-cos` is "sensitive to what MTEB averages away" |
| semantic gate, [`0121`](../0121-semantic-gate/TASK.md) | **24/24 — PASS** | a ranking over 36 sentences; the tail is not in sentences, and the gate has no absolute error term by design |

**All three report a central tendency and none looks at a tail.** That is the
part of this finding with the longest reach — it is a methodology gap, not one
bad number.

## 6. How much it matters

For anything phrase-length or longer, **bge-large is fine**: max 2.60e-04,
within 1.24× of its own median, nowhere near the gate. The failure regime is
narrow.

It is not academic, though. `newsletter` and `contact` are exactly what a search
box receives, and in a retrieval deployment the document side is sentences (safe)
while the **query** side is where one-word inputs live. Worst case is cosine
0.978 — an angle of about 12°, enough to reorder nearest neighbours.

**Not measured**: whether that reordering actually changes a retrieval result.
That would need an MTEB retrieval task, which
[`0035`](../0035-m8-mteb-gate/TASK.md) deliberately excluded from the gate.

## 7. Commands

```powershell
& "C:\Users\vegar\.conda\envs\iron\python.exe" tasks\0122-bge-large-short-input-tail\1_external_reference.py
& ".\.venv-ref\Scripts\python.exe" tasks\0122-bge-large-short-input-tail\3_refute_cancellation.py
& ".\.venv-ref\Scripts\python.exe" tasks\0122-bge-large-short-input-tail\4_refute_block_fp.py
& ".\.venv-ref\Scripts\python.exe" tasks\0122-bge-large-short-input-tail\5_distribution.py

# the confirmation against the repo's own harness
& ".\.venv-ref\Scripts\python.exe" tools\verify_embed_e2e.py --model bge-large-en-v1.5 `
      --artifacts artifacts_large_bfp16 --corpus worst.txt
```

All scripts take `Path.cwd()` as the repo root, so run them from the repo root.
Steps 3–5 need `.venv-ref`; step 1 needs only the stdlib plus the runtime.

## 8. Artifacts

| file | what |
|---|---|
| `ref_bge_small_test.txt`, `ref_bge_large_test.txt` | **the user's externally-obtained vectors**, verbatim — the only third-party reference this project has |
| `1_external_reference.py` | §1 |
| `2_void_cancellation_probe.py` | the probe that measured nothing, kept per rule 3b |
| `3_refute_cancellation.py`, `4_refute_block_fp.py` | §3 |
| `5_distribution.py` | §4 |
| `raw.txt` | verbatim output of all four |

## 9. Problems hit

* **The void probe (§3a) produced a confident-looking refutation of the right
  hypothesis with the wrong instrument.** What caught it was the derived column
  being bit-identical to the input column — not a failure, an *absence of
  variation*. Worth remembering: a normalisation flag on a library that also has
  a normalisation *module* controls only one of the two.
* `tools/verify_embed_e2e.py` **crashes on a single-text corpus**:
  `np.triu_indices(1, k=1)` is empty and `.max()` on a zero-size array raises
  `ValueError`. It prints the `1-cos` first, so the number survives, but the
  harness has evidently never been run on one text. Not fixed here.
* The `--models`/top-8 print limit in that harness hides rows on larger corpora,
  which is why §4 uses its own comparison.

## 10. Filed

[T51](../../research/OPEN-THREADS.md#t51). Not fixed: the mechanism is unknown
and the fix is unclear, so this is filed with a measured regime and two
eliminated hypotheses rather than a plan. `CURRENT_STATUS.md`'s `1-cos` column
now says what it is — a central tendency, not a bound.
