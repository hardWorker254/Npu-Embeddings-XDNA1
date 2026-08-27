# 0132 — T51 step 3: the p99 tail gate, as a shipping tool

**Date**: 2026-08-27
**Goal**: the release sweep needs a gate that can SEE a tail. Every accuracy
instrument this project owned reported a central tendency, and bge-large
passed all of them while carrying a measured 100× max/median tail on
single-word inputs ([`0122`](../0122-bge-large-short-input-tail/TASK.md),
[T51](../../research/OPEN-THREADS.md#t51)). Design was decided by
[`0129`](../0129-t51-int8-tail/TASK.md), not taste: **per-model, per-datapath,
gated on p99** — a max-gate is uninformative where the median already fails
(int8 bge-large), and p99/median ≈ 30× is exactly what the averaging
instruments miss.

**Result: built, baselined, wired into the sweep. All six models pass, and
the gate demonstrably fails when it should.**

## What was built

* **`tools/make_tail_reference.py`** (.venv-ref) — fp32 reference unit
  vectors for all six built-ins over 0129's corpus **byte for byte** (180
  single words drawn deterministically from bge-base's vocab, the 36
  semantic-corpus sentences, 8 five-word phrases; one shared corpus, so
  cross-model numbers compare models, not corpora). Reference lineages mirror
  what already exists rather than inventing a fourth: AutoModel with the
  pooling the checkpoint's own `1_Pooling/config.json` declares (CLS for
  bge-*, mean for MiniLM — 0129's loop), sentence-transformers for nomic
  (trust_remote_code, container `prompt_default` = `search_document` prefix
  prepended by hand, verify_embed_e2e.py's approach verbatim) and for
  embeddinggemma (`prompt_name="document"`, the prompt every gemma harness
  pins since 0118). Writes `reference/tail/<model>.f32` + `<model>.json`
  (texts, group boundaries, pooling, prompt name AND prefix string, versions,
  a `baseline` slot). Files across the env boundary, never imports.
* **`tools/verify_tail.py`** — **stdlib only**, mirroring
  `verify_semantics.py`'s structure (`--exe`/`--root` for a cold dist zip,
  the shipped `embed <model> <in> <out> --root <dir>` CLI form, datapath
  scraped from the runtime's own status line, byte-count-checked f32 reads,
  numpy-compatible linear-interpolation percentiles so the figures are
  directly comparable with 0122/0129's). Gate: **FAIL if p99 >
  `baseline.p99_ceiling`**; max and max/median are reported on every run and
  never gated. `--write-baseline` measures p99 and records
  `{p99_measured, p99_ceiling = max(2e-3, 2 × p99), measured_date, datapath}`
  into the reference JSON — a ratchet, tight where the model is tight, honest
  where it is not. An **unbaselined** model FAILS rather than skips (the
  fail-open shape of 6b/6c/7c/7d/0110, refused); a **datapath mismatch**
  against the baseline FAILS rather than comparing across baselines (0129:
  int8 and bfp16 bge-large differ 10× at the median).
* **`tools/release_benchmark.ps1`** — new `tail` stage (skippable as `-Skip
  tail`), one invocation for the whole catalogue after the per-model loop;
  logs to `tail.txt`, artifact to `tail_gate.json`, `tail_pass` set in
  `sweep.json` only when the stage ran. On FAIL it says explicitly that it
  will NOT re-baseline — that is a deliberate act.

## Commands run

```
.venv-ref\Scripts\python tools\make_tail_reference.py            # references, all six
C:\Users\vegar\.conda\envs\iron\python.exe tools\verify_tail.py --write-baseline
C:\Users\vegar\.conda\envs\iron\python.exe tools\verify_tail.py  # the gate; exit 0
```

Raw output: [`raw.txt`](raw.txt); machine-readable results incl. every
baseline block: [`tail_gate.json`](tail_gate.json). The gate run reproduced
the baseline run's numbers to the digit (same corpus, bit-stable encode, per
0122).

## The measured baselines (224 texts, `1-cos` vs fp32 reference)

| model | datapath | median | p90 | p99 | max | max/med | ceiling |
|---|---|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | bfp16 | 3.426e-04 | 5.114e-04 | 9.652e-04 | 1.847e-03 | 5.4× | 2.000e-03 |
| bge-small-en-v1.5 | bf16 | 6.845e-06 | 8.322e-06 | 1.162e-05 | 1.301e-05 | 1.9× | 2.000e-03 |
| bge-base-en-v1.5 | bfp16 | 1.716e-04 | 2.242e-04 | 3.711e-04 | 6.132e-04 | 3.6× | 2.000e-03 |
| **bge-large-en-v1.5** | bfp16 | 2.179e-04 | 3.687e-04 | **6.626e-03** | **2.183e-02** | **100.2×** | **1.325e-02** ¹ |
| nomic-embed-text-v1.5 | bfp16 | 1.081e-03 | 1.329e-03 | 1.579e-03 | 1.657e-03 | 1.5× | 3.157e-03 ² |
| embeddinggemma-300m | bfp16 | 1.578e-04 | 2.455e-04 | 4.227e-04 | 1.163e-03 | 7.4× | 2.000e-03 |

¹ **bge-large's wide ceiling is a documented, register-linked waiver, not an
endorsement** — its baseline block carries a `note` pointing at
[T51](../../research/OPEN-THREADS.md#t51). Do not tighten it by hand; close
T51 first.
² nomic's ceiling exceeded the 2e-03 floor via the ratchet's 2× arm, but this
is a **level, not a tail**: median 1.05–1.38e-03 across all three groups,
max/median 1.5× — the flattest distribution in the table. Consistent with its
recorded release-sweep `1-cos` of 1.402e-03 (0085 / CURRENT_STATUS).

**Cross-checks that the references are the same instrument 0122/0129 used**:
bge-large reproduces 0122 to the digit — median 2.179e-04, p99 6.626e-03,
max 2.183e-02, max/median 100.2×, worst five `newsletter` 2.18e-02 /
`sermons` 1.19e-02 / `contact` 7.94e-03 / `cinema` 2.23e-03 / `messages`
2.22e-03. bge-base's max 6.13e-04 (`adjective`) matches 0122's quoted
bge-base max exactly.

## The gate can fail (negative control)

A copy of bge-base's reference with its ceiling tightened to 1e-06 FAILS with
exit 1 ([`negative_control.txt`](negative_control.txt)):

```
C:\Users\vegar\.conda\envs\iron\python.exe tools\verify_tail.py \
    --models bge-base-en-v1.5 --reference <scratch copy> --out <scratch>
  bge-base-en-v1.5   FAIL -- p99 3.711e-04 > ceiling 1.000e-06 (measured 3.711e-04 on 2026-08-27)
exit=1
```

(First attempt at this control reported PASS/exit 0 — because my JSON-editing
one-liner had a syntax error and never tightened the ceiling, so the control
ran against an unmodified copy. A control that passes is not evidence the
gate works until you have verified the control itself did what it claims;
0121 §5 made the same point about its rotation control.)

## How the sweep invokes it

```powershell
.\tools\release_benchmark.ps1                 # tail stage included
.\tools\release_benchmark.ps1 -Skip tail      # without it
```

One invocation for the whole catalogue after the per-model loop; log
`tail.txt`, artifact `tail_gate.json` beside `sweep.json`, and `tail_pass` in
`sweep.json` only when the stage actually ran (an unset key cannot misreport
a skipped stage). Against a cold dist zip:

```powershell
python tools\verify_tail.py --exe dist\...\npuembeddings.exe --root dist\...
```

(the references live in the repo's `reference/tail/`; `--reference` points
elsewhere if needed — a dist zip does not carry them.)

## Problems hit

* **The negative-control-that-wasn't**, above. Cost minutes, kept because the
  shape of the mistake — trusting a control's verdict without checking the
  control ran — is this project's recurring fail-open class.
* **nomic's reference generation touches the network even with a local
  checkpoint**: `trust_remote_code` resolves `auto_map` entries pointing at
  `nomic-ai/nomic-bert-2048` for the modelling code (HF warned about
  unauthenticated requests; the code was cached and the run succeeded). The
  gate itself is offline — only regeneration has this dependency, same as
  verify_embed_e2e.py always had.
* **Worst-word observation, recorded not pursued**: MiniLM's worst input is
  `fucking` at 1.847e-03 — within 8% of its 2e-03 ceiling, the closest any
  passing model sits. If MiniLM ever fails this gate first, look there before
  suspecting the datapath.

## Files

```
tools/make_tail_reference.py                 generator (.venv-ref)
tools/verify_tail.py                         the gate (stdlib only)
tools/release_benchmark.ps1                  + tail stage
reference/tail/<model>.{f32,json}            6 references + baselines
tasks/0132-t51-tail-gate/{raw.txt, tail_gate.json, negative_control.txt}
```

## Status

T51 step 3 **done**: the sweep now has an instrument that sees a tail, and it
would have caught bge-large on the day it was adopted. Steps 1 (done, 0129)
and 2 (mechanism, optional) unchanged; T51 itself stays OPEN — this gate
*waives* bge-large's tail with a pointer at the register, it does not explain
it. Register/CLAUDE.md updates are the coordinator's.
