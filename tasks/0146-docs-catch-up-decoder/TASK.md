# 0146 — the documents catch up to the decoder workstream, and the public repo is pushed

**Status: DONE.** No hardware was run and no number in this log is new — every
figure quoted here is carried from
[`0143`](../0143-gte-cpu-energy/TASK.md)–[`0145`](../0145-granite-npu-gemv/TASK.md)
with a pointer. This is a documentation and register pass, run to the release
skill's discipline **without cutting a release**: no version bump, no zip, no
release note, no re-validation sweep.

## Goal

The user's words: *"vi skal ikke ha en ny release, men vi må pushe en del
informasjon som er skrevet inn her de siste dagene til `repo/` og videre til
remote"*, then *"gjør som om det var release"* — apply the release discipline to
the documents, stop short of the release itself.

Three days of work (0143, 0144, 0145) had landed as **task logs and traps only**.
`docs/CURRENT_STATUS.md` and `README.md` had not moved since 0.5.0, and
`research/OPEN-THREADS.md` was **byte-identical to its 0.5.0 state** — which is
exactly the failure rule 3 exists to prevent, one register sweep after the rule
grew a checker.

## What had actually changed since 0.5.0

```
$ git diff --stat 839e6a4..HEAD | tail -1
44 files changed, 3902 insertions(+), 16 deletions(-)
```

Substantively: `tasks/0143` (gte CPU/energy harness repaired, measurement not
taken), `tasks/0144` (granite → q4nx, T53/T54 answered), `tasks/0145` (a W4A16
GEMV kernel on the array), `research/notes/0010-q4nx-format.md`, seven new traps
in `CLAUDE.md` (9–15), an extended `research/notes/0001`, a `commit-msg` hook
and the `.gitattributes` rule that keeps it LF.

**No runtime code, no design, no model.** The shipped product is bit-for-bit
what 0.5.0 shipped, which is why this is not a release.

## Part 1 — the register (rule 3)

**Four threads filed**, all from work that stated its own gaps and had nowhere
to put them:

| | from | what is open |
|---|---|---|
| [T55](../../research/OPEN-THREADS.md#t55) | 0145 | Is there an NPU case for **decode** at all? The CPU is 2.4× faster at 89% of its own bandwidth; the array's weight path is at parity (46.3 vs 47.0 GB/s), so the kernel is compute-bound by 2.3×. The two cases left — **energy** and **prefill** — are both unmeasured |
| [T56](../../research/OPEN-THREADS.md#t56) | 0145 | The memtile leg (24 cores, 2.04×) and layer fusion (3 dispatches, 1.40×) each pay and **do not compose**; channel budgets and divisibility, not code. Also: why 8 columns measured slower than 6 |
| [T57](../../research/OPEN-THREADS.md#t57) | 0143 | gte's `NPU/best CPU` and `J/1k` cells are still empty. The harness is **repaired and verified**; only the interleaved run is missing. Trigger: the next whole-catalogue sweep |
| [T58](../../research/OPEN-THREADS.md#t58) | 0144/0145 | The granite host engine's output-quality bug, still with no mechanism; `diff_engine_logits.py` has never completed a run |

They are filed in the same register as the encoder threads deliberately — they
are questions about *this* hardware, and rule 3 has no second file for questions
that arrive from a different direction. The live count moves **5 → 9**.

```
$ python tools\check_register.py            # before
note: 9 of 144 task logs are referenced from no thread, doc or note:
  0014 0037 0041 0086 0100 0119 0140 0142 0143
note: OPEN-THREADS.md    595 lines,  5 threads
register OK

$ python tools\check_register.py            # after
note: 8 of 144 task logs are referenced from no thread, doc or note:
  0014 0037 0041 0086 0100 0119 0140 0142
note: OPEN-THREADS.md    757 lines,  9 threads
register OK
```

**The checker earned its keep in the small way it was designed for**: 0143 was
in the unreferenced list, and filing T57 is what took it out. A task nothing
points at is a task nobody will re-read.

## Part 2 — the three documents that drift

* **`docs/CURRENT_STATUS.md`** — an amendment block under the 0.5.0 header
  saying plainly that **nothing about the shipped product changed**, and a new
  **§11, "A second workstream: a decoder on the same array"**, which states up
  front that it is a research direction, that none of its code is in this
  repository (`../LLMNpuTest`, `../q4nx-build`), and that nothing is vendored
  from FastFlowLM (rule 4). §9's "five live threads" → nine, split by
  workstream so the encoder priority order is not diluted by it.
* **`CLAUDE.md`** — the thread count, and one new paragraph in *Current state*.
  It leads with what works (all 8 projection shapes at cosine 1.00000000, a
  layer in three dispatches, 24 of 32 cores) and then with the **negative**,
  because that is the headline: the CPU is 2.4× faster, and the null-kernel
  probe is what makes that a useful fact rather than a discouraging one.
* **`README.md`** — the register counts (51 of 60 closed), and T55 added to the
  contribution list. It is written as a question with numbers attached, not as
  an achievement: *"Does the NPU have a case for LLM decode at all?"* — the
  answer we have is a negative, and someone reading the README should be able
  to see that before they clone anything.

## Part 3 — the release skill's one unguarded step (0143's leftover)

0143 ended by naming a thing it had not fixed: `SKILL.md`'s push block was
`cd repo; git add -A; git commit; git push origin main` — no check of which
branch `repo/` is on, and no warning that a rejected push must not be answered
with `git pull`. Both had already cost 0143 real time (a whole release pushed to
a `v0.4.0` branch; then a 23-file conflict storm from merging a pre-rewrite
history back into a rewritten one).

It was deferred to "0.6.0's release-skill pass". This *is* a release-skill pass,
so it was fixed here rather than deferred again: a `git rev-parse --abbrev-ref
HEAD` guard, the two-line rewrite check (`git diff --stat` empty + `git cherry`
all `-`) that tells a rewrite apart from a genuine divergence, and two new
checklist rows.

## Part 4 — sync and push

```
$ python tools\sync_public_repo.py
PASS -- 1879 files in repo, no dead links
EXIT=0
$ git -C repo rev-parse --abbrev-ref HEAD
main
```

The sync is a gate, and it passed first time — the new material's links (four
new thread anchors, §11's pointers into `tasks/` and `research/notes/`) all
resolve. Its two advisory reports are unchanged in kind: 51 files over 0.5 MB
(golden tensors) and 60 prose mentions of excluded material.

**One thing worth recording about the diff.** `git status` in `repo/` showed
**280 modified files** and `git diff --stat` showed **11**. The difference is
the `.gitattributes` added in 0143 (`* text=auto`): the sync copies working-tree
files with CRLF and git normalises them away, so the other 269 are line-ending
noise with no content change. Checking `diff --stat` rather than `status` is
what makes that visible in one line instead of a review of 280 files.

## What was NOT done, deliberately

* **No release.** No version bump, no `make_release.ps1`, no zip, no cold test,
  no `dist/RELEASE-NOTES-*.md`. 0.5.0 remains the current release.
* **No hardware run.** Every figure in the updated documents is carried from
  0143–0145 with a pointer; nothing was re-measured, and nothing needed to be,
  because no shipped artifact changed.
* **The gte CPU/energy cells stay empty.** Filling them from an older,
  differently-measured run would break rule 1 and 0040's interleaving
  requirement. They are [T57](../../research/OPEN-THREADS.md#t57) now instead.
