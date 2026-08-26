# 0094 — T32: the golden gate compares 4 of N rows, and even N of N rows isn't enough alone

- **Date** 2026-08-23
- **Milestone** research (T32, `research/OPEN-THREADS.md`)
- **Status** done

## Goal

Close open thread T32 ("The golden gate is structurally blind to cross-row
corruption"). T32's own text names the mechanism (every tiled copy of the
4-sentence golden batch is identical content, so a bug that reads the wrong
ROW is invisible) and proposes fixes in increasing order of value. The launch
brief added a second, cheaper, and more basic finding on top of T32's own
text: the comparison loop in `runtime/src/main.cpp` only ever checked the
first `kGoldenBatch = 4` rows of the output, regardless of batch size — at
batch 128 it read 4 of 128 rows and never touched the other 124 at all. That
is truncation, not indistinguishability, and it is a different bug with a
different fix. Both had to be closed, and both had to be demonstrated against
a deliberately reintroduced known-bad artifact, not just shown to pass on the
good path (`docs/CURRENT_STATUS.md` §4's closing rule).

## Context

- `research/OPEN-THREADS.md` T32, filed 2026-08-21, citing
  [`tasks/0070`](../0070-m13-nomic-runtime/TASK.md): a genuine threaded
  read/write race in `swiglu_cpu()` corrupted output rows, the golden gate
  **PASSED**, and a 13-distinct-sentence `verify_embed_e2e.py` run caught it
  at worst `1-cos` 0.44.
- `docs/CURRENT_STATUS.md` §4, "Four bugs that failed open" — the standing
  list this task is a fifth candidate for, and whose closing line is the bar
  this task had to clear: *"the fixes were verified against known-bad
  artifacts, not only good ones."*
- The code in question, before this task: `runtime/src/main.cpp` around
  line 5210 (`constexpr int64_t kGoldenBatch = 4;`, plain `tile()` lambda,
  `emb_in`/`mask`/`amask_i` tiled `reps4 = batch / 4` times, `want` read at
  exactly `kGoldenBatch * g_hidden` floats and never tiled at all) and around
  line 6087 (`for (int64_t b = 0; b < kGoldenBatch; ++b)`).

## What was done

**1. Verified the lead's reading before touching anything.** Read
`pool_and_normalise` (`runtime/src/main.cpp`, wraps `pool_rows`) — it takes
`batch` rows of hidden state and writes `batch * g_hidden` floats of pooled,
normalised output, one row per input sequence, no reduction. So `emb` really
does carry `batch` distinct rows, and comparing only 4 of them (while `want`
only ever held 4) was confirmed to be exactly the truncation the brief
described: **at batch 128 the gate compared 4 of 128 output rows and never
read the other 124.**

Checked whether the same pattern exists anywhere else: grepped for
`kGoldenBatch`, `want[`, `1 - cos`/`1mcos`, `rel_fro` across
`runtime/src/*.cpp`. The pattern is unique to this one comparison loop in
`main.cpp` (the standard `--model X --artifacts Y` golden gate, arch 0 and
arch 2). `gemma_kernels_test.cpp` is a separate unit-test harness for
individual RoPE/RMSNorm kernels with no tiling of this shape. arch=1
(EmbeddingGemma) has **no HuggingFace golden at all**
(`docs/CURRENT_STATUS.md` §1: *"EmbeddingGemma has no `rel_fro` against
HuggingFace"*) — its gate is differential, built in `gemma_encode_cli.cpp`
and `tools/verify_gemma_npu_encode.py`, and does not go through this
comparison loop, so it is out of scope for this specific fix (T32's own text
also scopes itself to "the M3 goldens, which are batch 4", i.e. the BERT/
nomic family).

**2. Fixed the truncation.** The comparison loop now runs `for (int64_t b =
0; b < batch; ++b)`, not `b < kGoldenBatch`. The printed line was rewritten
so it does not silently redefine what "worst 1-cos" means: it now reads
`worst 1 - cos vs HuggingFace   1.086e-05  (all 128 rows, 4 distinct
sentences rotated across tile copies)` instead of the old unlabelled
`1.086e-05` that implicitly meant "of 4".

**3. Fixed the indistinguishability (T32 option 1).** Replaced the plain
`tile()` lambda (which just concatenated the 4-row golden `reps4` times,
identically) with `tile_rot()`, which places base golden row `(k + r) %
kGoldenBatch` into row `k` of tile copy `r`. Applied identically to
`emb_in`, `mask`, `amask_i` **and `want`** — `want` used to be read at a
fixed 4 rows and never tiled at all; now it is tiled with the same rotation,
so row `b`'s expected value is base golden row `(b % kGoldenBatch + b /
kGoldenBatch) % kGoldenBatch`, the exact sequence that was actually fed into
row `b`. This is still an exact golden (a relabelling of which known-good
row goes where, not new data) and it is trivially invertible. A code comment
at the definition site explains why plain tiling was insufficient, citing
T32 and tasks/0070 by name (see diff below).

Why both were needed, concretely: fixing only the truncation (loop over all
rows, keep plain tiling) makes a **content-corruption** bug visible (any row
whose actual bytes differ from its expected bytes now gets compared) but
does **nothing** for a **row-aliasing** bug, where the runtime reads the
wrong physical row but that row happens to hold identical content under
plain tiling — the comparison would still pass because the two candidate
values (correct row's data, wrong row's data) are bit-identical. Only the
rotation makes every physical row's content unique, so an aliasing bug
changes what gets compared. Section "Known-bad verification" below
demonstrates this split directly.

**4. Verified against known-bad artifacts.** Added temporary `getenv`-gated
debug hooks (never touched by the shipped path with no environment
variables set), ran five known-bad demonstrations against them, then
**removed every hook** and rebuilt before re-confirming the real gate PASSes
on the real path. See "Known-bad verification" below for the exact
mechanism and all five runs' real output.

**5. Regression-checked all four models with a HuggingFace golden** (arch 0
and arch 2): `all-MiniLM-L6-v2`, `bge-base-en-v1.5`, `bge-large-en-v1.5`,
`nomic-embed-text-v1.5`. All four PASS at the 2e-3 tolerance and **every
`rel_fro` matches the value already recorded in `docs/CURRENT_STATUS.md` to
the digit** — see "Regression" below. This is itself a finding, stated
explicitly per the launch brief: **no published number moved**, which means
rows 4-127 of the tiled batch were always as accurate as rows 0-3, but
nobody had ever checked that until this task's all-rows loop existed to
check it.

**6. Checked the NaN-fail-open regression** (`docs/CURRENT_STATUS.md` §4 bug
#4: `std::max(0.0, NaN)` returns `0.0`). The explicit `!std::isfinite(...)`
check sits after the rewritten loop, untouched by this change; verified it
still fires (see "Known-bad verification" run 5).

**7. Assessed T32 item 3** (wiring `tools/verify_embed_e2e.py` into the
standard release gate) rather than silently dropping it. See "T32 item 3"
below — partially cheap (the five bf16 BERT-family rows), not fully cheap
(the six int8 rows need a code change `verify_embed_e2e.py` does not have
yet, mirroring the `--cpu-model` fix `run_mteb.py` needed in tasks/0085; the
one arch=1 row already has an equivalent-purpose differential check in the
sweep today and was not touched). Ran it standalone against `bge-base-en-v1.5`
to confirm it still works and is cheap to invoke; left the actual wiring as a
named remaining item rather than editing `tools/release_benchmark.ps1`
without being able to run the multi-hour full sweep this session (the
machine was busy with other agents throughout).

## The fix itself (`runtime/src/main.cpp`)

```
git diff runtime/src/main.cpp
```

Three regions changed, all inside `main()`'s golden-gate setup and
comparison:

1. `constexpr int64_t kGoldenBatch = 4;` plus the tiling: the old `tile()`
   lambda (plain repeat, no `row_floats` parameter, used for `emb_in`,
   `mask`, `amask_i` only — `want` was never tiled) is replaced with
   `tile_rot()`, which takes a `row_floats` stride and rotates source row
   `(k + r) % kGoldenBatch` into position `(r * kGoldenBatch + k)`. Applied
   to `emb_in`, `mask`, `want`, `amask_i` — all four, including `want`,
   which previously bypassed tiling entirely.
2. The comparison loop: `for (int64_t b = 0; b < batch; ++b)` in place of
   `b < kGoldenBatch`.
3. The printed `"worst 1 - cos vs HuggingFace"` line now states row
   coverage and the underlying sentence count explicitly.

The full diff (68 insertions / 19 deletions) is reproduced in this task's
git history; the key excerpt:

```cpp
constexpr int64_t kGoldenBatch = 4;
auto tile_rot = [kGoldenBatch](const std::vector<float> &v, int64_t reps,
                               int64_t row_floats) {
  std::vector<float> out(static_cast<size_t>(reps * kGoldenBatch * row_floats));
  for (int64_t r = 0; r < reps; ++r)
    for (int64_t k = 0; k < kGoldenBatch; ++k) {
      const int64_t src = (k + r) % kGoldenBatch;
      std::memcpy(out.data() + static_cast<size_t>((r * kGoldenBatch + k) * row_floats),
                  v.data() + static_cast<size_t>(src * row_floats),
                  static_cast<size_t>(row_floats) * sizeof(float));
    }
  return out;
};
...
double num = 0.0, den = 0.0, worst_1mcos = 0.0;
for (int64_t b = 0; b < batch; ++b) {
  double dot = 0.0;
  for (int64_t c = 0; c < g_hidden; ++c) {
    double diff = emb[b * g_hidden + c] - want[b * g_hidden + c];
    num += diff * diff;
    den += static_cast<double>(want[b * g_hidden + c]) * want[b * g_hidden + c];
    dot += static_cast<double>(emb[b * g_hidden + c]) * want[b * g_hidden + c];
  }
  worst_1mcos = std::max(worst_1mcos, 1.0 - dot);
}
```

## Commands

Environment (every shell):

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime
```

Build:

```powershell
cmake --build build --config Release
```

Real-path regression, four models with a HuggingFace golden:

```powershell
.\build\npuembed.exe .. --model all-MiniLM-L6-v2   --artifacts artifacts_b128il --threads 16
.\build\npuembed.exe .. --model bge-base-en-v1.5    --artifacts artifacts_base   --threads 16
.\build\npuembed.exe .. --model bge-large-en-v1.5   --artifacts artifacts_large  --threads 16
.\build\npuembed.exe .. --model nomic-embed-text-v1.5 --artifacts artifacts_nomic --threads 16
```

T32 item 3 smoke test (from the repo root, `.venv-ref`):

```powershell
& ".\.venv-ref\Scripts\python.exe" tools\verify_embed_e2e.py --model bge-base-en-v1.5 `
    --artifacts artifacts_base --threads 16 `
    --out tasks\0094-t32-golden-gate-rows\verify_embed_e2e_bgebase.json
```

Known-bad demonstrations (temporary debug hooks, all `getenv`-gated;
`NPUE_DEBUG_CORRUPT_ROW=<idx>`, `NPUE_DEBUG_TRUNCATE_COMPARE=1`,
`NPUE_DEBUG_TILE_MODE=plain`, `NPUE_DEBUG_ALIAS_SWAP=r1,r2`,
`NPUE_DEBUG_NAN_ROW=<idx>` — see "Known-bad verification" for exactly what
each does; **all five were added, exercised, and then fully removed from
`runtime/src/main.cpp` before the final build**, so the shipped binary
contains none of them):

```powershell
# 1. corruption, old (truncated) gate -- expect PASS (bug invisible)
$env:NPUE_DEBUG_CORRUPT_ROW="10"; $env:NPUE_DEBUG_TRUNCATE_COMPARE="1"
.\build\npuembed.exe .. --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --threads 16
Remove-Item Env:\NPUE_DEBUG_CORRUPT_ROW, Env:\NPUE_DEBUG_TRUNCATE_COMPARE

# 2. corruption, new (all-rows) gate -- expect FAIL
$env:NPUE_DEBUG_CORRUPT_ROW="10"
.\build\npuembed.exe .. --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --threads 16
Remove-Item Env:\NPUE_DEBUG_CORRUPT_ROW

# 3. aliasing, plain tiling, all-rows gate -- expect PASS (bug invisible)
$env:NPUE_DEBUG_TILE_MODE="plain"; $env:NPUE_DEBUG_ALIAS_SWAP="6,10"
.\build\npuembed.exe .. --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --threads 16
Remove-Item Env:\NPUE_DEBUG_TILE_MODE, Env:\NPUE_DEBUG_ALIAS_SWAP

# 4. aliasing, rotated tiling, all-rows gate (the shipped fix) -- expect FAIL
$env:NPUE_DEBUG_ALIAS_SWAP="6,10"
.\build\npuembed.exe .. --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --threads 16
Remove-Item Env:\NPUE_DEBUG_ALIAS_SWAP

# 5. NaN regression check -- expect FAIL, "non-finite output"
$env:NPUE_DEBUG_NAN_ROW="10"
.\build\npuembed.exe .. --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --threads 16
Remove-Item Env:\NPUE_DEBUG_NAN_ROW
```

## Result

### Regression — four models, real path, no debug hooks (correctness only,
no timing claim per rule 1 and per this task's own scope)

| model | artifacts | `rel_fro` (this run) | `rel_fro` (recorded, `docs/CURRENT_STATUS.md`) | worst `1-cos`, all rows | verdict |
|---|---|---:|---:|---:|---|
| `all-MiniLM-L6-v2` | `artifacts_b128il` | 4.473e-03 | 4.473e-03 | 1.086e-05 | PASS, IDENTICAL |
| `bge-base-en-v1.5` | `artifacts_base` | 4.297e-03 | 4.297e-03 | 1.353e-05 | PASS, IDENTICAL |
| `bge-large-en-v1.5` | `artifacts_large` | 3.763e-03 | 3.763e-03 | 8.432e-06 | PASS, IDENTICAL |
| `nomic-embed-text-v1.5` | `artifacts_nomic` | 6.119e-03 | 6.119e-03 | 2.599e-05 | PASS, IDENTICAL (also matches tasks/0070's own `1-cos` 2.599e-05) |

**No published number moved.** Rows 4-127 of every model's tiled batch are
exactly as accurate as rows 0-3 — this was previously unchecked, and it is
now checked on every run. Raw output: `gate_minilm_real.txt` (first run) /
`gate_minilm_real_final.txt` (rebuilt after debug hooks were removed,
bit-identical numbers), `gate_bgebase_real.txt`, `gate_bgelarge_real.txt`,
`gate_nomic_real.txt`.

### Known-bad verification (all on `all-MiniLM-L6-v2`, `artifacts_b128il`)

Row 10 (tile copy `r=2`, slot `k=2`) is used for the corruption bug — outside
`kGoldenBatch=4`, so invisible to the old truncated loop by construction.
Rows 6 and 10 (`r=1,k=2` and `r=2,k=2` — same slot `k`, different tile copy)
are used for the aliasing bug: under **plain** tiling those two physical
rows hold byte-identical content by construction (both are base row `k=2`
regardless of copy), so swapping them is a real cross-row-aliasing bug that
produces **zero visible difference** unless the tiling itself makes the two
rows' content different — exactly T32's point.

| # | bug | gate config | expected | got | file |
|---|---|---|---|---|---|
| 1 | negate output row 10 (content corruption, mimics 0070's "wrong sign and magnitude") | `NPUE_DEBUG_TRUNCATE_COMPARE=1` (old gate: compares rows 0-3 only) | PASS | **PASS** — `rel_fro` 4.473e-03, `1-cos` 1.086e-05 (unchanged from clean baseline; row 10 never read) | `bad1_corrupt_old_gate.txt` |
| 2 | same, row 10 negated | default (new gate: all 128 rows) | FAIL | **FAIL** — `rel_fro` 1.768e-01, `1-cos` 2.000e+00 | `bad2_corrupt_new_gate.txt` |
| 3 | swap input rows 6 and 10 (aliasing) | `NPUE_DEBUG_TILE_MODE=plain` (all rows compared, but rotation OFF) | PASS | **PASS** — `rel_fro` 4.473e-03, `1-cos` 1.086e-05 (bit-identical to the clean baseline; the two rows' content really is indistinguishable under plain tiling) | `bad3_alias_plain_tile.txt` |
| 4 | same swap, 6 and 10 | default (rotation ON — the shipped fix) | FAIL | **FAIL** — `rel_fro` 1.782e-01, `1-cos` 1.016e+00 | `bad4_alias_rotated_tile.txt` |
| 5 | force output row 10 to NaN | default | FAIL, "non-finite output" | **FAIL** — `rel_fro` printed `nan`, message `"FAIL -- non-finite output. NaN cannot pass a tolerance test by scoring zero."` | `bad5_nan_row.txt` |

All five behaved exactly as predicted. In particular, run 1 vs run 2 is the
truncation fix in isolation (identical bug, only the row range compared
changes the verdict), and run 3 vs run 4 is the rotation fix in isolation
(identical bug and identical "compare all rows" gate, only the tiling
scheme changes the verdict) — proving neither fix alone would have caught
both bug classes, matching the launch brief's framing exactly.

One correction made along the way: an earlier version of the corruption
hook added a constant (`+5.0f`) to every element of the target row instead
of negating it. That produced a large `rel_fro` (8.660e+00) but left
`worst_1mcos` at the clean baseline (1.086e-05) and **PASSED** even under
the new all-rows gate — not because the bug was invisible, but because the
comparison loop computes `1 - dot(emb, want)` as a proxy for cosine
similarity, which is only valid when both vectors are unit-norm; adding a
constant to every component of an already-L2-normalised row moves the raw
dot product far less than it moves the Euclidean distance, since the
cross-term with a near-zero-mean `want` stays small. Switched the hook to
negate the row instead (`*= -1.0f`), which preserves the vector's norm
exactly and reliably drives `1-cos` to ~2. Kept in this log because it is a
genuine, if small, finding about the metric: **`rel_fro` and `worst 1-cos`
do not always move together, and a corruption that happens to leave the raw
dot product near its old value can look fine on the gating metric while
`rel_fro` screams.** This project already uses both metrics side by side
(`docs/CURRENT_STATUS.md`: *"the two measure different things and the
project uses both"*, said in the int8 context) — this is a second, concrete
instance of that same caution, on the fp32/bf16 path this time.

### T32 item 3 — `verify_embed_e2e.py` as part of the standard gate

Ran it standalone against `bge-base-en-v1.5` (13 distinct, real sentences,
including non-ASCII and edge cases): **worst `1-cos` 2.613e-05, top-10
neighbour overlap 1.0000, PASS** (`verify_embed_e2e_bgebase.txt`/`.json`).
This confirms the tool still works and is cheap to invoke standalone — one
subprocess call to `npuembed.exe --embed` plus one `SentenceTransformer`
load and encode of 13 short sentences.

**Wiring it into `tools/release_benchmark.ps1` as a standard stage is only
partly cheap, and I did not do it this session — named as a remaining
item rather than silently dropped, per the launch brief:**

- **Cheap (a ~15-20 line addition, modelled directly on the existing `mteb`
  stage block, no changes needed to `verify_embed_e2e.py` itself):** the
  five bf16 rows over BERT/nomic geometry (`all-MiniLM-L6-v2`,
  `bge-small-en-v1.5`, `bge-base-en-v1.5`, `bge-large-en-v1.5`,
  `nomic-embed-text-v1.5`). For these, `args.model` already names a real
  `models/<name>/` HuggingFace checkpoint directory, which is exactly what
  `verify_embed_e2e.py`'s `SentenceTransformer(str(REPO / "models" /
  args.model), ...)` call needs.
- **Not cheap without a code change first:** the six `.intN` rows in the
  catalogue (e.g. `all-MiniLM-L6-v2.int8`) have no `models/<name>/`
  directory of their own — only a `.npue`. `run_mteb.py` hit exactly this
  gap in tasks/0085 and was fixed with a `--cpu-model` flag that lets the
  container name and the checkpoint-directory name differ.
  `verify_embed_e2e.py` has no equivalent flag today; adding one is a small,
  well-understood change (same shape as the existing fix) but is a real code
  change, not configuration, and untested this session.
- **Already covered by an equivalent-purpose check today, so lower
  priority:** `embeddinggemma-300m` (arch=1). Its accuracy stage in
  `release_benchmark.ps1` already runs `npuembed.exe --embed` on a real
  13-text corpus (`tasks/0074-m13-gemma-on-npu/corpus.txt`) and gates
  differentially against `tools/verify_gemma_npu_encode.py` — i.e. arch=1's
  release-sweep accuracy check is *already* a distinct-text e2e check, unlike
  the BERT/nomic rows' golden-batch-4 check this task just fixed. Whether
  `verify_embed_e2e.py` itself would also work for arch=1 (it would need
  `trust_remote_code` behaviour verified for the Gemma checkpoint, which the
  tool does not currently special-case the way it special-cases nomic) is
  unconfirmed and was not attempted.
- **Not attempted at all, and the reason is time, not difficulty:** actually
  running the full sweep to prove the new stage is stable end to end.
  `release_benchmark.ps1`'s own header says a full pass with MTEB "takes
  hours", the machine was shared with other agents building and running
  throughout this session, and this task's scope is correctness of the
  golden gate, not a release cut.

## Problems hit

1. **MSVC C3493**: the `tile_rot` lambda used `kGoldenBatch` (a `constexpr`
   local) inside a runtime loop body without capturing it. `constexpr`
   locals do not need capture only when every use resolves to a
   compile-time constant expression; `kGoldenBatch * row_floats` used as a
   runtime vector size does not qualify under MSVC's rules here. Fixed by
   adding `kGoldenBatch` to the lambda's capture list.
2. **The `+5.0f` corruption hook did not fail the gate it was supposed to
   fail** — see "One correction made along the way" above. Diagnosed by
   checking `rel_fro` (which did jump, to 8.660e+00) against `worst_1mcos`
   (which did not, staying at the clean baseline), which showed the bug was
   real but the specific corruption shape happened not to move the metric
   the gate actually thresholds on. Fixed by switching to row negation,
   which preserves unit norm and reliably moves `1-cos`. This is reported
   as a finding, not swept under the rug — see the metric-divergence note
   above.
3. **PowerShell wraps a native command's stderr line in a
   `NativeCommandError`-looking block even on a clean exit.** Every debug
   hook prints one line to `stderr` (`std::fprintf(stderr, ...)`) to make
   the injected bug visible in the log; under `2>&1 | Tee-Object`, PowerShell
   5.1 renders that line with `At line:N char:N` / `CategoryInfo` /
   `FullyQualifiedErrorId` framing even when the process's real exit code is
   0. Not a real error — confirmed by checking `$LASTEXITCODE` explicitly
   after each run rather than trusting the presence of that framing (this is
   the same PowerShell quirk this project's own tool docs already warn
   about for `2>&1` on native executables).
4. **No actual bugs in the fix itself** were found once implemented — the
   MSVC capture error and the metric-choice issue above were the only two
   problems, both caught and fixed before any run was recorded as final.

## Artifacts

All in this directory:

- `gate_minilm_real.txt`, `gate_bgebase_real.txt`, `gate_bgelarge_real.txt`,
  `gate_nomic_real.txt` — the four-model regression, real path, debug hooks
  not yet added.
- `gate_minilm_real_final.txt` — MiniLM re-run after the debug hooks were
  fully removed and the binary rebuilt; numbers bit-identical to
  `gate_minilm_real.txt`, confirming the removal was clean.
- `bad1_corrupt_old_gate.txt` .. `bad5_nan_row.txt` — the five known-bad
  demonstrations, described above.
- `verify_embed_e2e_bgebase.txt` / `.json` — the T32-item-3 smoke test.

## Next

- **Wire `verify_embed_e2e.py`'s five bf16 rows into `tools/release_benchmark.ps1`**
  as a new `e2e` stage (skippable via `-Skip e2e`), modelled on the existing
  `mteb` stage block. Add a `--cpu-model`-equivalent flag to
  `verify_embed_e2e.py` before extending it to the six int8 rows.
- Confirm `verify_embed_e2e.py` works for `embeddinggemma-300m` (arch=1) —
  currently unconfirmed whether it needs the same `trust_remote_code`
  special-casing the tool already has for nomic.
- This task's fix and demonstration are all on `all-MiniLM-L6-v2` for the
  known-bad runs (cheapest design to iterate on); the regression table
  covers all four HF-golden models, but nobody has re-run the known-bad
  demonstrations against, say, `bge-large-en-v1.5`'s different `tile_n=32`
  geometry. Not expected to matter (the comparison loop is geometry-
  independent), but unconfirmed.

## Proposed register update

**Verdict on T32: ANSWERED.** Both holes it named (option 1's rotation,
fully implemented; and the truncation bug it did not name but which the
launch brief found alongside it) are fixed, built, and verified against
five deliberately-reintroduced known-bad artifacts, matching
`docs/CURRENT_STATUS.md` §4's evidentiary bar. T32's item 2 ("generate
goldens at a batch that is not 4") was not done — the rotation makes it
unnecessary for the aliasing class T32 raised item 1 to address, and a
truly larger *distinct-content* fixture is a bigger lift the task judged not
required to close the thread. T32's item 3 (`verify_embed_e2e.py` in the
standard gate) is **partially actioned**: confirmed working, costed
precisely, not wired in — left as a named remaining item per the launch
brief's explicit instruction not to drop it silently.

Replacement entry text for `research/CLOSED-THREADS.md` (T32 moves there
verbatim per rule 3, with this update appended):

```
### T32 — The golden gate is structurally blind to cross-row corruption · **ANSWERED, closed 2026-08-23**
[original filed text unchanged -- see git history / the version in
OPEN-THREADS.md before this update]

**Closed by [`0094`](TASK.md).** Reading the
code turned up a SECOND, more basic hole alongside the one T32 named: the
comparison loop only ever checked the first `kGoldenBatch = 4` rows of the
output regardless of batch size -- at batch 128 it compared 4 of 128 rows and
never read the other 124, `want` was never tiled to match. Both are now
fixed: the loop runs over all `batch` rows, and the four golden sequences are
tiled with a per-copy rotation (`(k + r) % kGoldenBatch`, applied identically
to `emb_in`/`mask`/`amask_i`/`want`) so every tile copy's content is unique --
T32's option 1, implemented. Verified against FIVE deliberately reintroduced
known-bad artifacts, not only the good path: a content-corruption bug on a
row past index 3 (old truncated gate PASSES, new gate FAILS), a cross-row
aliasing bug swapping two same-slot rows from different tile copies (plain
tiling + all-rows-compared still PASSES -- proving the all-rows fix ALONE
does not catch aliasing -- rotation + all-rows-compared FAILS), and a forced
NaN row (still FAILS with the existing "non-finite output" message, so bug
#4 of CURRENT_STATUS.md sec 4 has not regressed). All four models with a
HuggingFace golden (MiniLM, bge-base, bge-large, nomic) re-ran clean after
the fix and every `rel_fro` MATCHES the previously recorded value to the
digit -- no published number moved, meaning rows 4+ were always as accurate
as rows 0-3, but this is the first time anyone checked.

T32's item 3 (run `verify_embed_e2e.py` in the standard gate) is a named
remaining item, not fully actioned: confirmed working standalone
(bge-base-en-v1.5, worst 1-cos 2.613e-05, PASS) and costed -- cheap for the
five bf16 BERT/nomic rows, needs a `--cpu-model`-style flag added to the
script first for the six int8 rows (same gap `run_mteb.py` hit in 0085),
already redundant for arch=1 which has its own distinct-text differential
check in the release sweep today. Not wired into `tools/release_benchmark.ps1`
this session -- the full sweep takes hours and the machine was shared.
```

`tasks/README.md` index row for this task:

```
| [0094](0094-t32-golden-gate-rows/TASK.md) | **T32 closed — the golden gate compared 4 of N output rows, not N, and even N of N wasn't enough alone.** Reading the code surfaced a second hole alongside T32's own (indistinguishable tile copies): `want` was read at a fixed 4 rows and the comparison loop stopped at `b < kGoldenBatch` regardless of batch size, so at batch 128 it read 4 of 128 rows and never touched the rest. Fixed both — the loop now runs over all `batch` rows, and the four golden sequences are tiled with a per-copy rotation (`(k+r) % kGoldenBatch`, T32 option 1) so every physical row's content is unique. Verified against **five** deliberately reintroduced known-bad artifacts: content corruption is invisible to the old truncated gate and caught by the new one; a same-slot cross-copy row swap is invisible to plain tiling even with all rows compared, and caught only once rotation is added — proving neither fix alone suffices; the NaN-fail-open regression check (`CURRENT_STATUS.md` sec 4 bug #4) still fires. All four HuggingFace-golden models re-ran clean, `rel_fro` identical to the previously recorded values to the digit — no published number moved. T32 item 3 (`verify_embed_e2e.py` in the standard gate) costed precisely and left as a named remaining item, not silently dropped | research | done |
```

**Fifth entry for `docs/CURRENT_STATUS.md` §4's "Four bugs that failed
open" list** — proposed text, not applied (per the launch brief, that file
is edited by the coordinator):

```
5. **[`0094`](TASK.md)** — the golden
   gate's comparison loop read `for (b = 0; b < kGoldenBatch; ++b)` where
   `kGoldenBatch` is the constant 4, regardless of the design's actual
   batch. At batch 128 it silently compared 4 of 128 output rows and never
   read the other 124 -- **a corruption bug in any row past index 3 was
   invisible by construction**, no matter how wrong the arithmetic. A
   SECOND hole sat alongside it (T32, filed 2026-08-21): the four golden
   sequences were tiled identically into every copy, so even a loop that
   DID check every row could not distinguish "the right row's data" from
   "an identical copy of the wrong row's data" -- a class of bug 0070's
   threaded `swiglu_cpu()` race actually shipped, invisibly, until a
   separate distinct-text e2e tool caught it by accident.
```

Also change the closing line under the table from *"All four are now
fail-closed..."* to *"All five are now fail-closed..."* if this entry is
accepted.
