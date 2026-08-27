# 0142 — Cutting 0.5.0: the cold zip fetches, verifies, C++-packs and runs the seventh model by itself, and the data region proves it

**Date**: 2026-08-27
**Goal**: the release mechanics for 0.5.0, per the `release` skill's order —
prove what ships (the sweep, [`0141`](../0141-release-sweep-050/TASK.md)),
update the documents, regenerate the public repository, build the zip and
cold-test it as a stranger would.

## Documents

README (seven-model table, 0.5.0 performance section with the p99-tail
column and the measured-roof paragraph), `docs/CURRENT_STATUS.md` (0.5.0
table above the 0.4.0 one now marked historical; §9 rewritten — the old
§9 ranked eight priorities that were ALL closed), `CLAUDE.md` current-state
(seven models, arch-3 row), `dist/RELEASE-NOTES-0.5.0.md` written from the
task logs. One self-inflicted defect caught and fixed mid-pass: the §9
rewrite truncated §10 off the file; restored from HEAD before commit.

## Public repository

```
python tools\sync_public_repo.py     # PASS -- 1841 files, no dead links
cd repo && git add -A && git commit  # committed locally; PUSH LEFT FOR THE USER
```

The 0.4.0-era failures (15 quoted rows, 52 dangling links) had self-resolved
— they were links into then-uncommitted task directories, exactly as
predicted in the 0.5.0 plan.

## The zip, and the cold test

```
.\tools\make_release.ps1 -Version 0.5.0 -Artifacts artifacts_minilm_bfp16,artifacts_small_bf16,artifacts_base_bfp16,artifacts_large_bfp16,artifacts_nomic_bfp16,artifacts_gemma_bfp16
# dist\npuembeddings-0.5.0-win-x64.zip  (0.82 MB) -- SAME six design sets as
# 0.4.0: the seventh model needs no new array design, which is the release.
```

Cold test, unzipped to `%TEMP%\shiptest050`:

* `list` shows all seven models with the correct cold root; gte's row reads
  "same array designs as bge-base/nomic".
* **`embed gte-multilingual-base` from nothing**: fetched 582.5 MB + the
  five-file gte set (the new `kFilesGte` path), verified against the
  built-in pins, **packed in C++** (`prepare_model_gte`, including the
  C++-side XLMRTOK1 generation), and ran on the NPU — exit 0, vectors out.
  Took four attempts: HuggingFace cut the unauthenticated stream three
  times at 207/278/344 MB ("short read", loud, exit 2 — the refusal
  behaving exactly as designed) before the fourth ran clean. Recorded
  because a release note that says "first run fetches the weights" should
  know what a throttled afternoon looks like.
* **The data-region check, on the container the cold zip built itself**:
  `python tools/npue_data_hash.py models\gte-multilingual-base.npue <cold>\models\gte-multilingual-base.npue`
  → `ea6ef66ddb7efae8…` on both sides, **exit 0**. The cold C++ pack equals
  the validated Python pack in every tensor byte.
* The other six containers **seeded from the repo's `models/`** (the user's
  call, and the right one — they are the already-verified bytes, and
  re-downloading 3.7 GB to re-prove a fetch path 0.4.0 already proved adds
  risk, not evidence).
* **Semantic gate against the cold zip**: PASS, all seven models
  (`--exe <cold> --root <cold>`, `--out` to a scratch path).
* **Tail gate against the cold zip**: PASS, every model's p99 under its
  recorded ceiling. (Its default output path clobbered the canonical
  `tasks/0132` report — restored from git; `verify_tail.py` wants the same
  `--out` flag its sibling grew in 0136's identical incident.)
* **Root in all four layouts**: source tree (all session), staged inside
  the repo (`dist\npuembeddings-0.5.0` prints its own dir — the 0051
  climb-to-repo-root bug absent), unzipped outside, and explicit `--root`.

## Left for the user, deliberately

The public-repo **push**, the git **tag**, and `gh release create` with the
zip — outward-facing steps per the skill's §6. Everything up to them is
built and verified.

## Checklist state

Every item of the skill's checklist is green except the three user-gated
ones above. `1 − cos` figures recorded per model (0141's table); no CPU
ratio quoted that was not interleaved (gte's is absent and says why);
register check green throughout; the note names what was not measured
(gemma's energy differential, discarded; gte's CPU columns).
