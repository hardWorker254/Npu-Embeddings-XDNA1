# 0143 — the gte CPU/energy row, and the session links in the public repo

**Status: INTERRUPTED, deliberately. The harness work is DONE and verified;
the measurement is NOT. Nothing below claims a gte CPU ratio or a gte energy
figure, because neither was reached.** Resume instructions at the bottom.

## Goal

Two holes left over from 0.5.0 (0141/0142):

1. **`gte-multilingual-base` has no `NPU/best CPU` and no `J/1k better` cell**
   in `docs/CURRENT_STATUS.md`'s 0.5.0 table. 0141 opted the row out of the
   `interleaved` and `energy` stages, correctly: both need a CPU reference
   arm, and this checkpoint's arm was broken.
2. **The commits carry `Claude-Session:` links, in a public repository.**
   A URL that resolves for exactly one person, published to everyone else.

## Part 1 — the gte CPU reference arm (DONE, verified)

### Why it was opted out

`transformers` 5.15 instantiates `trust_remote_code` modules on the meta
device, so every `persistent=False` buffer comes back as **uninitialised
memory**. On this checkpoint that is three things (0134 L1/L2, extended in
0136): rotary `inv_freq` and its cos/sin caches, the config's own
`torch_dtype: float16` being honoured silently, and — the dangerous one —
`embeddings.position_ids`, which is exactly what `SentenceTransformer`'s
derived-position path reads. In-bounds garbage there is **silent**: it encodes
the WRONG positions and returns a plausible vector.

### What changed

`repair_rotary` is **imported** from the one place it is written down
(`reference/make_goldens_gte.py`) rather than restated — a second copy is a
second thing that can fall out of date with the checkpoint.

* `experiments/m8-npu-vs-cpu/compare_three.py` — `is_gte` from the
  container's own `arch` field (`gte_new_rope_geglu`), then
  `model_kwargs={"torch_dtype": torch.float32}` at construction and
  `repair_rotary(st[0].auto_model)` after. Prints that it did.
* `experiments/m8-npu-vs-cpu/energy_cpu_load.py` — same, and the container
  read moved ABOVE the `SentenceTransformer(...)` call, since the load now
  depends on it.
* `tools/release_benchmark.ps1` — the gte row's `interleaved = $false;
  energy = $false` opt-outs removed, with the reason they existed and the
  reason they no longer do kept in the comment.

The ORT third arm still declines for this checkpoint (`AutoModel` without
`trust_remote_code` cannot load `model_type: "new"`), so gte's CPU baseline
is torch alone — the same as nomic's, through the existing "degrade, do not
crash" path, which says so in the log rather than dropping the side silently.

### The gate, with its negative control

Probe: load the arm exactly as the patched harness does, encode
`corpus_gte.SENTENCES`, compare against `st.pool.cls_norm` in
`reference/goldens_gte/gte-multilingual-base_l12_s64_boundary.safetensors`.

```
arch: gte_new_rope_geglu  is_gte: True
RESULT repaired=False: FAILED -- IndexError: index 6300247457792 is out of bounds
                                 for dimension 0 with size 34
RESULT repaired=True:   max-abs 2.980e-08   1-cos 0.000e+00
```

The repaired arm reproduces the stored golden. The unrepaired one failed
**loudly** this time (out-of-bounds index); 0136 measured the in-bounds,
silent version of the same fault at 9.0e-02. Both are the same defect.

## Part 2 — the measurement (NOT DONE)

The whole catalogue's `interleaved` + `energy` stages were re-run in ONE
session rather than gte alone, because a ratio taken in a different session
than the rows beside it is not comparable — `docs/06-performance.md` puts the
CPU side's between-session movement at ~24%, and 0141 saw an energy ratio move
60%.

```powershell
.\tools\release_benchmark.ps1 `
  -Models all-MiniLM-L6-v2,bge-small-en-v1.5,bge-base-en-v1.5,bge-large-en-v1.5,nomic-embed-text-v1.5,embeddinggemma-300m,gte-multilingual-base `
  -Skip accuracy,throughput,mteb,tail -OutDir 'tasks\0143-gte-cpu-energy'
```

**Run 1 REFUSED at the quiet gate** — `clion64` at ~2.97 cores. The guard was
right; nothing was measured. Re-run after it settled (2.8% mean).

**Run 2 was KILLED by the operator part-way through bge-large's energy stage.**
It never reached nomic, gemma or **gte** — so the row this task exists to fill
is still empty, and no number here may be quoted as a result.

What run 2 did produce, kept as artifacts and NOT as results:

| model | interleaved | energy |
|---|---|---|
| all-MiniLM-L6-v2 | complete (torch 497.4, npu 1390.6 seq/s steady) | complete |
| bge-small-en-v1.5 | complete (torch 262.9, npu 613.5) | complete |
| bge-base-en-v1.5 | complete (torch 71.6, npu 304.0) | complete |
| bge-large-en-v1.5 | **REFUSED** — NPU contention guard: a foreign process held an `Active` hw_context | partial, and the `npu-single-60` arm reported **idle drift 44.8% UNSTABLE -- DISCARD** |
| nomic, gemma, **gte** | not reached | not reached |

Two things in that table are worth carrying forward rather than re-deriving:

1. **bge-large's interleaved stage refused on NPU contention while its energy
   stage ran anyway.** The energy stage does not apply the same guard. On a
   run where one stage has already declared the machine contended, that is a
   gap worth closing — it is the same "contention hits one side only" reasoning
   the CPU-quiet gate is built on.
2. **The `idle drift 44.8% UNSTABLE` discard is the energy harness working**,
   and it sits next to 0141's still-open gemma defect (Δt ≈ 0 between the
   low and high runs, ratio 14,554×, left for 0.6.0's harness pass).

## Part 3 — the session links (DONE locally, NOT PUSHED)

`Claude-Session: https://claude.ai/code/session_...` is appended by the agent
harness's own commit template, so it is not something this repository chose,
and no prose here would stop the next session doing it again.

* **History rewritten** in `repo/` (the public repo): `git filter-branch
  --msg-filter 'grep -v "^Claude-Session: https://claude.ai/code/session_"'`
  over `--all`. 7 of 10 commits carried it; `HEAD`'s history now carries
  none. `Co-Authored-By:` is deliberately KEPT — attribution is honest, a
  private session URL is not. Backups: `refs/original/*` and the
  `pre-session-strip-backup` tag.
* **Prevention**: `tools/git-hooks/commit-msg`, installed in both
  repositories via `core.hooksPath` (`tools/git-hooks` here,
  `../tools/git-hooks` in `repo/`, which resolves to the same file because
  `repo/` lives inside this working tree). Verified with a probe commit in
  `repo/` — the trailer was dropped, `Co-Authored-By` survived, probe reset.
* **NOT PUSHED.** The rewrite needs `git push --force origin main`, and
  `--force` on tags v0.2.0 / v0.3.0 / v0.4.0, which now point at rewritten
  commits. v0.1.0 is unaffected (`d7f9059`, no trailer). The user approved
  the force push; it was left for the session that finishes the release.

### The bug found while in there — the 0.5.0 public commits were not on `main`

`repo/`'s checked-out branch was **`v0.4.0`**, a BRANCH sharing its name with
the `v0.4.0` TAG, and both 0.5.0 commits sat on it. `main` was still at
`release: 0.4.0`. The release skill's `cd repo; git add -A; git commit; git
push origin main` would therefore have **published nothing** of 0.5.0.

What kept it invisible is worth stating plainly: `git log main..v0.4.0`
resolves the ambiguous name to the **tag**, which is where `main` already was,
so the range came back **empty** — the check that would have caught it reported
"nothing to push" instead of "two commits on the wrong branch". `git branch -v`
shows it immediately; a range query does not.

Fixed: `main` fast-forwarded onto the two commits (lossless — it was a strict
ancestor), stray branch deleted.

## Also done

`main` in THIS repository fast-forwarded to `feat/first` (212 commits, `main`
was a strict ancestor). Work commits to `main` from here.

## To resume

1. Machine idle — **close CLion/PyCharm**, both tripped or came near the
   quiet gate. Check no foreign process holds an NPU `hw_context` either;
   that is what refused bge-large.
2. Re-run the run-2 command above into a FRESH `-OutDir`. This directory's
   `sweep.json` was never written — the run died before the merge step — and
   the per-model files here are a partial run that must not be merged into a
   complete one.
3. Then, and only then: fill gte's two cells in `docs/CURRENT_STATUS.md`'s
   0.5.0 table, replacing the `— (see below)` markers AND the paragraph under
   it that explains why they are absent. `docs/06-performance.md` and the
   0.5.0 release note carry the same claim.
4. ~~Push the public repo~~ — DONE 2026-08-28, see Part 3b. Remaining: drop
   `refs/original/*` and `pre-session-strip-backup` from `repo/` when the
   pre-rewrite SHAs are no longer wanted.

## Part 3b — the push, and the conflict storm it caused (2026-08-28)

**Symptom:** `git push origin main` in `repo/`, then a merge with **23 files in
conflict** — `both modified` on almost every file the project owns, including
files neither side had touched in months.

**Cause: none of those conflicts were real.** The rewrite of Part 3 changed
every commit SHA from `05345d7` onward, so the remote's `main` and the local
`main` were two *content-identical* histories with no shared commit after
`05345d7`. The plain push was correctly rejected as non-fast-forward; the
`git pull` that followed then asked git to **merge the old history back into
the rewritten one**. Git compared the two branches against `05345d7` — the last
real ancestor — and every change made since, on both sides, showed up as both
sides having independently added the same thing. A duplicated history conflicts
with itself on the whole tree.

The give-away that it was not a content problem, and the check to run before
touching anything:

```
$ git diff --stat f38a640 ea8d9f5     # rewritten tip vs pre-rewrite tip
(empty)
$ git cherry -v main origin/main      # every remote commit, by patch-id
- 5977961 feat: bge-base-en-v1.5 ...  # '-' on all six = already in main
```

Empty diff and six `-` marks: the rewrite lost nothing, and the remote held
nothing local `main` did not already have. **A rewrite is finished by
force-pushing it, never by merging the pre-rewrite branch back in** — the merge
recreates exactly the commits the rewrite removed.

**What was done**

```
git merge --abort
git push --force-with-lease=refs/heads/main:7ac37a4 origin main
git push --force origin refs/tags/v0.2.0 refs/tags/v0.3.0 refs/tags/v0.4.0
```

`--force-with-lease` pinned to the SHA the remote was expected to hold, so the
push would have refused rather than overwritten if anything had reached
`origin/main` in the meantime. Verified after: `origin/main = f38a640`, three
tags moved to their rewritten commits, `v0.1.0` untouched, and
**`git log main --grep Claude-Session` returns nothing**. The 0.5.0 commits are
public for the first time — Part 3's `v0.4.0`-branch bug had kept them off
`main` entirely.

The pre-rewrite history is still in `refs/original/*` and the
`pre-session-strip-backup` tag, **locally only** (`git ls-remote` confirms
neither is on the remote). They are the reason `git log --all` still shows the
duplicated graph in this clone; deleting them is safe once nobody wants the
old SHAs back, and is the last cleanup step.

**One thing left unguarded:** `.claude/skills/release/SKILL.md:105` still says
`cd repo; git add -A; git commit; git push origin main`, with no check of which
branch `repo/` is on (Part 3's other bug) and no note that a push rejected as
non-fast-forward must **never** be answered with `git pull`. Both belong in
0.6.0's release-skill pass.
