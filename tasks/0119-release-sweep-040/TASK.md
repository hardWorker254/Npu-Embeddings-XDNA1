# 0119 — cutting 0.4.0: the sweep, the README, and 130 broken links

**Date**: 2026-08-26
**Threads**: none opened or closed. Follows the `release` skill.

## Goal

Prepare release 0.4.0: re-measure what it ships, bring the documents in line
with it, write the release note, and build the bundle.

Scope was set by the user after the cost of each stage was put on the table:
**re-run accuracy, throughput and interleaved on all six models; carry MTEB and
energy forward** from [`0103`](../0103-t23-bfp16-all-models/TASK.md) and
[`0109`](../0109-fused-ratio-energy/TASK.md). The reason MTEB can be carried is
specific rather than convenient: the one change since those runs
([`0118`](../0118-prompt-name-per-request/TASK.md)) is **bit-identical**,
verified with `cmp` on all three model families, so a six-model MTEB pass —
"hours", per [`0085`](../0085-m13-release-sweep/TASK.md) — would reproduce the
same numbers at the cost of an evening. Also set: **build the zip locally, do
not publish.** No tag, no `gh release`, no push of the public repo.

## What was done

### 1. The sweep

```powershell
.\tools\release_benchmark.ps1 -Skip mteb,energy -OutDir "tasks\0119-release-sweep-040"
```

**It refused three times before it ran**, and each refusal named a different
process: `clion64` + `msedge` at ~0.8 cores, then a single windowless Edge
background host (`--no-startup-window`) at 0.71, then **Task Manager** at 0.31 —
the tool opened to check whether the machine was idle was the last thing making
it not idle.

Worth writing down because the arithmetic was questioned and the answer is not
"the guard is right, trust it": the machine is 12 cores / 24 threads, so the
0.71-core process **was** essentially the whole 2.5% mean the harness reported.
The gate is not a "is the machine busy" test — the mean passed every time. It is
a per-process tripwire at 0.2 cores, and it is set there because the damage is
to a **ratio**: contention slows the CPU side while the NPU path offloads to the
array. At 0.3 cores of 24 the NPU/CPU column would have come out ~1.3% high —
inside the ±20% the README already claims. So the cost of honouring it was
minutes of waiting and the cost of overriding it would have been an artifact
that labels itself indefensible in a release table. `-AllowCpuContention` was
never used.

Ran clean at **cpu 2.2% / 2.3%, `cpu_quiet: true`, `cpu_contenders: []`**.

**And the sweep immediately found a regression introduced by
[`0118`](../0118-prompt-name-per-request/TASK.md) — mine.** EmbeddingGemma's
accuracy and throughput stages invoke the runtime with no `--prefix`, which
0118 made a refusal:

```
error: this model has task prefixes and one must be named: pass --prefix with
one of [BitextMining, ..., document, query] ...
```

0118's plan listed the callers to update and checked `release_benchmark.ps1`
for `--prefix`; finding none, it concluded the file needed no change. **That
inference was backwards** — the file needed one *added*, not preserved. Worse,
the coupling is tighter than a missing flag: `tasks/0074`'s stored control
vectors `out_cpu.f32` were produced under the old silent `"document"` default,
so **the prefix is part of the control**. Running that stage under any other
prompt would compare two different questions and report the difference as a
datapath error. It is pinned explicitly now, with that reason in the comment.
This is the coupling 0118 existed to expose, exposed by something refusing
instead of guessing.

Re-run after the fix, restricted to the six models that actually ship (the int8
rows are not in the bundle and their `interleaved` stage has a separate,
pre-existing harness bug: it derives a checkpoint directory from a container
variant name, so it looks for `models/embeddinggemma-300m.int8/1_Pooling/`,
which does not exist). Exit 0.

**Results** — one session, idle machine, `--pipeline 4`, `--threads 24`:

| model | datapath | worst `1 − cos` | NPU seq/s | torch | ort | NPU / best CPU |
|---|---|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | bfp16 | 3.406e-04 | **1459.4** | 777.9 | 311.6 | 1.876× |
| `bge-small-en-v1.5` | bf16 | 8.348e-06 | **630.2** | 403.7 | 173.1 | 1.561× |
| `bge-base-en-v1.5` | bfp16 | 2.284e-04 | **324.0** | 119.4 | 59.8 | 2.714× |
| `bge-large-en-v1.5` | bfp16 | 2.626e-04 | **94.8** | 36.0 | 18.1 | 2.633× |
| `nomic-embed-text-v1.5` | bfp16 | 1.402e-03 | **258.9** | 75.1 | 45.9 | 3.447× |
| `embeddinggemma-300m` | bfp16 | 1.749e-04 * | **107.9** | 82.8 | — | 1.303× |

`*` differential against the host-only path; arch 1 has no HuggingFace golden.
Every `datapath` reading is taken from the runtime's own status line.

**Two things this table proves that were not the reason for running it.**

1. **Every accuracy figure reproduces 0105/0108 to the digit** — 3.406e-04,
   8.348e-06, 2.284e-04, 2.626e-04, 1.402e-03. That is an independent
   confirmation that 0118 moved nothing, by a different route than the `cmp`
   checks in 0118 itself.
2. **The sweep ran twice, twenty minutes apart.** NPU throughput repeated to
   within **0.2%** on all five BERT-family models; the CPU baselines moved
   3–4%. The README's "±20%, because the CPU side moves more between sessions
   than the difference anyone is arguing about" is now measured rather than
   asserted, and it points the right way.

### 2. README

Four claims were true of 0.3.0 and not of this build. All four are things a user
reads before anything else:

| was | is |
|---|---|
| performance table pre-bfp16, pre-fusion, **no `embeddinggemma-300m` row** | the 0.4.0 table, six rows, with a `datapath` column |
| model table: gemma "host-only, no NPU kernel yet" | it has run on the array since [`0074`](../0074-m13-gemma-on-npu/TASK.md) |
| "**We have not exported or measured one**" (long sequences) | [`0112`](../0112-t40-seq256-nomic/TASK.md)/[`0113`](../0113-t40-seq512-close/TASK.md) exported two and measured across an 8× range |
| Contributing points at T16, T28, T32 | all three closed; now T42/T43/T44, flagging which needs no NPU |

The new performance table carries a **`vs 0.3.0`** column, and it is the one
like-for-like comparison in it: 0.3.0's published figures came from this
harness's own `interleaved` stage, which is the same stage, same statistic and
same round-robin protocol this sweep ran. That is worth stating because the
first draft of this section said the opposite — that the protocols were not
comparable — which was true of the *throughput* stage (‑‑bench against a corpus
encode) and not of the one 0.3.0 actually quoted. 1.28× to 1.57×, plus
EmbeddingGemma, which has no 0.3.0 NPU figure because it did not have one.

It also explains rather than hides the two rows that need it — bge-small on the
slower datapath because it **failed** at −0.5010, and gemma's differential
accuracy figure — and states the reproducibility measured above rather than
only asserting the ±20% caveat.

### 3. `sync_public_repo.py`, run as the gate it is

The skill records that this was once chained with `;`, failed, and a commit
with a dead link went out anyway. Run properly it exited 1 — **and had been
failing for some time.**

| class | count | why |
|---|---:|---|
| link to excluded material surviving the rewriter | 1 | **hard FAIL.** an arXiv-id link in `docs/history.md` pointing into `research/papers/`, invisible to the rewriter because it was *also* relative-wrong: written without a way up, so from `docs/` it resolved under `docs/` and never matched the excluded prefix |
| root-relative from a subdirectory | 68 | [`0075`](../0075-m13-arch1-measurement-harness/TASK.md) moved 1,027 lines out of `CLAUDE.md` verbatim; `CLAUDE.md` sits at the repo root, so **every relative link in them broke on arrival** |
| doubled prefix, missing way up, wrong depth | ~60 | `../tasks/X` from inside `tasks/`; `notes/0007-…` from a task log; a skill three levels deep reaching for `../../` |
| **quoted `tasks/README.md` rows** | **15** | deliberately not fixed — see Problems |

130 repaired. `check_register.py` still passes.

### 4. Release note

`dist/RELEASE-NOTES-0.4.0.md`, written from the task logs rather than from
`CLAUDE.md`'s summaries, per the skill. Structure follows 0.3.0's. The headline
is that all six models are on the array and **two silent wrong answers became
loud errors** — 0110's truncation refusal and 0118's required task prompt share
a shape, and it is the worst shape a numerical library can have.

## Problems hit

1. **The first link-repair pass guessed, and guessed wrong.** Seven links named
   a task by NUMBER with a slug matching no directory —
   a link labelled 0032 whose path named an
   `eltwise-on-host` slug, when the real directory for that number is
   `0032-m7-one-xclbin-production`. A rule that redirects on the number alone
   "fixed" all seven, and it is a **guess about whether the author meant the
   number or the topic**. Reverted whole, then each of the seven checked against
   the *content* of the task it points at: 0032's §2 really is "eltwise ops must
   EARN the dispatch"; 0039 really is where the cross-model golden guard came
   from; 0042 really is where the third model landed. The number was right every
   time — **but that is now a finding rather than an assumption**, and the
   difference is the whole point.

2. **The quoted-text guard was wrong in a way that fixed only half a link
   pair.** It counted backticks in a 200-character window to detect a code
   span; this project's house style backticks link *labels*
   , which makes the count odd for the wrong reason. Symptom:
   a line carrying two adjacent task links, one
   already correct and one root-relative, had the broken half skipped because
   the correct half had put the backtick count out. Code spans are found as spans now.

3. **15 links are left broken on purpose.** They are `tasks/README.md` index
   rows quoted verbatim into the task log that added them.
   The link is correct **for its real home** and broken where it is quoted, so a
   reader of the log does get a 404. Two honest repairs exist — fence the row as
   source, since that is what it is, or rewrite the target and lose the verbatim
   quote — and choosing between them is the author's call, not a script's. The
   sync therefore still exits 1, which blocks the public push and nothing else.

4. **A deliberate placeholder nearly got "repaired".** `tasks/0089` writes
   a bullet template with a placeholder thread
   id in the label and a bare `CLOSED-THREADS.md` target, to describe the
   *format* of a register entry. `research/CLOSED-THREADS.md` exists, so a naive rule
   rewrites it — replacing a description with a different falsehood. This is the
   trap the skill warns about, met from the other direction.

## State at the end

| | |
|---|---|
| sweep | **clean**, `tasks/0119-release-sweep-040/sweep.json`, cpu 2.3%, no contenders |
| README, CLAUDE.md, CURRENT_STATUS, register | updated, numbers from this sweep |
| release note | `dist/RELEASE-NOTES-0.4.0.md` |
| `check_register.py` | **OK** |
| zip | **built and cold-tested**, `dist/npuembeddings-0.4.0-win-x64.zip`, 0.73 MB, six design sets |
| `sync_public_repo.py` | exits 1 on 15 quoted `tasks/README.md` rows -- blocks the public push, nothing else |
| tag / GitHub release / public push | not done, by instruction |

### The cold test

Unzipped to `%TEMP%\shiptest040`, run as somebody who has never seen the source:
help and catalogue, `list`, then `embed all-MiniLM-L6-v2` -- which fetched,
verified, packed and ran on the array, reporting `bfp16-emulated MMAC, C as
bf16` and the toolchain that built the design.

**Root resolution, all four layouts** -- the check that exists because
[`0051`](../0051-m9-bge-base-and-in-exe-fetch/TASK.md) found a release staged
*inside* the tree climbing to the repo root and serving the repository's models
while claiming to be self-contained:

| layout | root printed |
|---|---|
| source tree | the repo |
| **staged inside the repo** | `dist/npuembeddings-0.4.0` -- **not** the repo |
| unzipped elsewhere | `%TEMP%\shiptest040` |
| `--root` explicit | the path given |

**The container check did not match, and that is the interesting part.** The
skill's rule is that the container a release builds must equal the one that was
validated. It did not:

```
cold-built : A2C57CB5BBB6365...   validated : A64D53DC8AD0E56...
```

Same byte count, eight days apart. Decomposed: the **tensor data region is
byte-identical** (sha256 `dc2661b0…` both sides), and the whole difference is
**17 bytes of JSON** -- the cold container carries an `a_dtype` key that the
repo's, packed 2026-08-18, predates ([`0104`](../0104-adopt-bfp16-per-model/TASK.md)
added it). Total size is unchanged because the JSON is padded to the
4096-aligned data offset.

So this is not a mispack, and the numbers measured against the repo container
transfer to what a user gets, because the weights are bit-identical. But the
skill's check as written -- compare whole-file hashes -- **cannot tell those two
cases apart**, and would have failed a correct release. The check that means
what it intends is on the data region.

## What was NOT done

- **The public repo was not synced or pushed**, and `sync_public_repo.py` still
  exits 1 on the 15 quoted index rows. That gate blocks a push and nothing else;
  the zip does not depend on it.
- **No tag, no `gh release`.** Publishing puts files on the internet under the
  user's name and is theirs to trigger.
- **MTEB and energy were not re-measured**, by decision recorded above. Both are
  carried with their source named.
- **The int8 rows were dropped from the final sweep** rather than fixed. Their
  `interleaved` harness bug is real and pre-existing, and int8 is not in the
  bundle.
- **The repo's own `models/*.npue` were not repacked** to carry `a_dtype`. They
  are correct as they are -- absence reads as the bf16 default, which is what
  they are -- and repacking six containers to make a hash comparison tidier is
  not a reason to touch shipped data.
