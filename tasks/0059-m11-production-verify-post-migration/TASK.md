# 0059 — Production verification after the mlir-aie 1.4 migration

- **Date** 2026-08-20
- **Milestone** M11 (follow-on to [`0058`](../0058-m11-iron-1.4-migration/TASK.md))
- **Status** done

## Goal

[`0058`](../0058-m11-iron-1.4-migration/TASK.md) upgraded the reference toolchain
`C:\dev\mlir-aie` from pinned mlir-aie 1.3.4 to `main` (7e00b57955e, past v1.4.1) and
migrated 15 IRON design *scripts*. Its stated architectural claim was that this cannot
touch **production** at all: shipped artifacts (`runtime/artifacts_*/*.xclbin`,
`models/*.npue`) are static binaries XRT loads regardless of what built them, and the
C++ runtime (`runtime/src/`) has zero Python/IRON dependency. That claim was never
actually exercised end to end — this task does that, per the user's explicit request
("La oss være sikre på at oppdateringen til mlir-aie 1.4.1 er ferdig, at alt fungerer
som før. Sammenligne embeddings osv.").

Passing means: the C++ runtime rebuilds clean from source, and all four shipped models
reproduce their **pre-upgrade `1-cos` figures**, not just "under the 2e-03 gate" —
identical numbers are the actual proof nothing drifted, since a real regression could
still clear a generous gate.

## Context

- `tasks/0058-m11-iron-1.4-migration/TASK.md`, `research/notes/0008-iron-1.4-migration.md`
  — what changed and why it was believed not to touch production.
- `docs/CURRENT_STATUS.md` §1 table — the last-recorded per-model `1-cos` figures
  this task compares against.
- `tools/verify_embed_e2e.py` — the existing end-to-end (tokenizer included) gate;
  not modified.
- Historical `1-cos` figures for the golden-vs-`.npue` validation encode:
  `all-MiniLM-L6-v2` 1.086e-05 (`0038`), `bge-small-en-v1.5` 8.348e-06 (`0039`),
  `bge-base-en-v1.5` 1.353e-05 (`0051`), `bge-large-en-v1.5` 8.432e-06 (`0042`).
- Historical `verify_embed_e2e.py` output for `all-MiniLM-L6-v2` / `artifacts_b128il`:
  saved at `tasks/0036-m8-tokenizer/verify_embed_e2e.json`, worst `1-cos`
  2.643574029825846e-05, pairwise-similarity mean 3.493e-04 (matches the "mean 1-cos
  3.493e-04" figure quoted in `0038`'s results table — same script, same statistic).
- Historical `verify_embed_e2e.py` output for `bge-base-en-v1.5` / `artifacts_base`:
  saved at `tasks/0036-m8-tokenizer/verify_embed_e2e_bge-base-en-v1.5_artifacts_base.json`,
  worst `1-cos` 2.612818933023231e-05 (matches CLAUDE.md's "2.613e-05 end to end").
- No prior saved `verify_embed_e2e.py` run exists for `bge-small-en-v1.5` /
  `artifacts_b128il` or `bge-large-en-v1.5` / `artifacts_large` specifically (only their
  golden-validation-encode figures were recorded); those two get a first saved baseline
  here rather than an exact-match comparison.
- Model → artifact-set mapping confirmed by reading `runtime/src/main.cpp`'s
  `pick_artifacts()` and the exact commands in `tasks/0039`, `tasks/0042`, `tasks/0051`:
  `all-MiniLM-L6-v2` and `bge-small-en-v1.5` → `artifacts_b128il` (h=384);
  `bge-base-en-v1.5` → `artifacts_base` (h=768); `bge-large-en-v1.5` → `artifacts_large`
  (h=1024).

## What was done

1. **Contention guard.** `xrt-smi examine -r all` showed nine `WorkloadsSessionHost.exe`
   hw_contexts, all `Idle`, none `Active` — the normal background Windows-ML/DirectML
   contexts, not a foreign workload. `Get-Process python*` found four running Python
   processes; `Get-CimInstance Win32_Process` showed their command lines were
   Claude Code's own `blender-mcp` extension venv — unrelated to the NPU, confirmed not
   to hold any hw_context in the `xrt-smi` table above. Guard clear.

2. **Rebuilt the C++ runtime from a clean state**, not just re-invoked `cmake --build`
   on an up-to-date tree (a no-op build proves nothing). `cmake --build build --config
   Release --target clean` removed all 8 prior object files, then a full rebuild.
   **Problem hit and worked around**: the Bash tool's `cmd /c "... && ..."` invocation of
   `vcvars64.bat` silently produced only the cmd.exe startup banner and no further
   output — looked like a PTY/quoting interaction specific to that tool, not a build
   failure (confirmed by trying an un-vcvars'd build first, which failed cleanly with
   `cstdint: No such file or directory`, proving the *environment*, not the build, was
   the problem). Worked around by writing the chained commands to a `.bat` file and
   invoking it through the **PowerShell** tool instead of Bash, which set `INCLUDE`/`LIB`
   correctly and printed full Ninja output. This is a harness/shell quirk, not a
   toolchain regression — filed here rather than acted on further.

3. Ran the **existing** validation-encode command (`npuembed.exe .. --model X
   --artifacts Y --threads 16`, no `--embed`) for all four shipped models, and the
   **existing** `tools/verify_embed_e2e.py` (tokenizer + `sentence-transformers`
   reference, from `.venv-ref`) for all four, exactly as `tools/verify_embed_e2e.py`
   and the `release` skill already prescribe. No script was modified.

4. Ran `--bench 5 --pipeline 2` on `all-MiniLM-L6-v2` / `artifacts_b128il` as an
   **order-of-magnitude sanity check only** — wall clock, not a hardware trace, so per
   CLAUDE.md rule 1 this is not an NPU performance claim.

## Commands

```powershell
# 0. contention guard
& "C:\Windows\System32\AMD\xrt-smi.exe" examine -r all
Get-Process python*,npuembed*,npuembeddings* -ErrorAction SilentlyContinue |
    Select-Object Id,ProcessName,StartTime
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Select-Object ProcessId,CommandLine

# 1. clean rebuild of the C++ runtime (MSVC env via vcvars64.bat, see Problems hit)
cd runtime
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target clean
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake --build build --config Release

# 2. per-model validation encode (golden vs .npue-driven NPU encode)
.\build\npuembed.exe .. --model all-MiniLM-L6-v2  --artifacts artifacts_b128il --threads 16
.\build\npuembed.exe .. --model bge-small-en-v1.5 --artifacts artifacts_b128il --threads 16
.\build\npuembed.exe .. --model bge-base-en-v1.5  --artifacts artifacts_base   --threads 16
.\build\npuembed.exe .. --model bge-large-en-v1.5 --artifacts artifacts_large  --threads 16

# 3. per-model end-to-end check (tokenizer + NPU encode, vs sentence-transformers)
cd ..
& ".\.venv-ref\Scripts\python.exe" tools\verify_embed_e2e.py `
    --model all-MiniLM-L6-v2  --artifacts artifacts_b128il --threads 24
& ".\.venv-ref\Scripts\python.exe" tools\verify_embed_e2e.py `
    --model bge-small-en-v1.5 --artifacts artifacts_b128il --threads 24
& ".\.venv-ref\Scripts\python.exe" tools\verify_embed_e2e.py `
    --model bge-base-en-v1.5  --artifacts artifacts_base   --threads 24
& ".\.venv-ref\Scripts\python.exe" tools\verify_embed_e2e.py `
    --model bge-large-en-v1.5 --artifacts artifacts_large  --threads 24

# 4. throughput sanity check (NOT a performance claim -- wall clock only)
cd runtime
.\build\npuembed.exe .. --model all-MiniLM-L6-v2 --artifacts artifacts_b128il `
    --threads 24 --pipeline 2 --bench 5
```

## Result

**Build**: clean rebuild succeeded, `BUILD_EXIT=0`, all 8 translation units compiled
and linked with zero warnings-as-errors issues. No Python or IRON involved at any
step — `cmake`, `ninja`, `cl.exe` (MSVC 14.51) and the XRT headers/libs only.

**Validation encode (golden `.npue`-driven NPU encode vs HuggingFace, `--threads 16`,
no `--embed`)** — every figure is **bit-identical** to the last-recorded value:

| model | artifacts | `1-cos` today | `1-cos` historical | source | match |
|---|---|---:|---:|---|:---:|
| `all-MiniLM-L6-v2` | `artifacts_b128il` | 1.086e-05 | 1.086e-05 | `tasks/0038` | **identical** |
| `bge-small-en-v1.5` | `artifacts_b128il` | 8.348e-06 | 8.348e-06 | `tasks/0039` | **identical** |
| `bge-base-en-v1.5` | `artifacts_base` | 1.353e-05 | 1.353e-05 | `tasks/0051` | **identical** |
| `bge-large-en-v1.5` | `artifacts_large` | 8.432e-06 | 8.432e-06 | `tasks/0042` | **identical** |

All four printed `PASS -- tolerance 2e-03 on 1-cos, no Python in this process`.

**End-to-end (`verify_embed_e2e.py`: WordPiece tokenizer + NPU encode + pooling +
normalise, vs `sentence-transformers` on the same texts, `--threads 24`)**:

| model | artifacts | worst `1-cos` today | worst `1-cos` historical | match |
|---|---|---:|---:|:---:|
| `all-MiniLM-L6-v2` | `artifacts_b128il` | 2.644e-05 | 2.6436e-05 (`tasks/0036`) | **match to 4 s.f.** |
| `bge-small-en-v1.5` | `artifacts_b128il` | 3.022e-05 | *(none saved — first baseline)* | PASS, consistent order of magnitude |
| `bge-base-en-v1.5` | `artifacts_base` | 2.613e-05 | 2.612818933023231e-05 (`tasks/0036`) | **byte-identical JSON** (`git diff` empty) |
| `bge-large-en-v1.5` | `artifacts_large` | 3.801e-04 | *(none saved — first baseline)* | PASS, well inside gate |

All four: top-10 neighbour overlap **1.0000**, `PASS`. `bge-large`'s worst case (index 9
and 10: empty string and a whitespace-only string) is a known-noisy degenerate input —
near-zero-signal embeddings amplify relative cosine error — not a real-text regression;
its worst *real-sentence* `1-cos` is 1.256e-05, same order as the golden figure
8.432e-06. `bge-small` and `bge-large` had no prior saved `verify_embed_e2e.py` run at
these exact artifact sets to diff against byte-for-byte (0039 and 0042 record only the
golden-encode figure), so those two rows are the first recorded baseline for that
specific check rather than an exact-match confirmation — flagged, not glossed over.

**Contention guard**: clear before every hardware run (`xrt-smi` showed no `Active`
context; the four running `python*` processes were Claude Code's own `blender-mcp`
extension, unrelated to XRT).

**Throughput sanity check** (wall clock only, NOT an NPU performance claim):
`all-MiniLM-L6-v2` / `artifacts_b128il`, `--pipeline 2 --bench 5`:

```
5 pipelined groups of 2 x 128 sequences at seq 64
    wall   288.62 ms   ->     887.0 seq/s
    npu    exclusive -- 9 hw_context(s), none Active but ours
```

887.0 seq/s is within the historical range for this configuration (877.0 in `0042`,
892.7 / 962.6 in `0052` after the lane-default change to 4 — this run used
`--pipeline 2` explicitly, matching `0042`'s figure most closely). Same order of
magnitude, no catastrophic regression signal. Not re-measured via hardware trace, so
this is exactly what CLAUDE.md rule 1 calls a dispatch/wall-clock sanity signal, not a
performance result.

## Problems hit

- **`cmd /c "vcvars64.bat && cmake --build ..."` through the Bash tool produced only
  the cmd.exe startup banner and nothing else** — no error, no build output, command
  returned quickly. Symptom looked like a hang or silent failure at first.
  **Cause**: a PTY/quoting interaction specific to the Bash tool's handling of nested
  `cmd /c "..."` invocations spawning a fresh interactive-looking `cmd.exe`, not an
  actual build problem — confirmed because (a) a build attempted without `vcvars64.bat`
  through the same Bash tool failed *cleanly and legibly* with
  `fatal error C1083: Cannot open include file: 'cstdint'` (proving `INCLUDE`/`LIB`
  were genuinely unset, and proving the tool *can* surface real cl.exe errors), and (b)
  writing the identical command sequence to a `.bat` file and invoking it through the
  **PowerShell** tool instead worked immediately with full Ninja output.
  **Fix**: use the PowerShell tool for any command that needs the MSVC dev environment
  chained via `vcvars64.bat`; the temp `.bat` helper was deleted afterward (it lived in
  the gitignored `runtime/build/`, so no working-tree pollution). This is a harness
  quirk worth remembering for the next C++ rebuild, not a toolchain or project bug.
- Nothing else. No `1-cos` figure drifted, no model refused to load, no build step
  failed once the environment was set up correctly.

## Artifacts

- `tasks/0036-m8-tokenizer/verify_embed_e2e_all-MiniLM-L6-v2_artifacts_b128il.json` (new)
- `tasks/0036-m8-tokenizer/verify_embed_e2e_bge-small-en-v1.5_artifacts_b128il.json` (new)
- `tasks/0036-m8-tokenizer/verify_embed_e2e_bge-large-en-v1.5_artifacts_large.json` (new)
- `tasks/0036-m8-tokenizer/verify_embed_e2e_bge-base-en-v1.5_artifacts_base.json`
  (re-written by this run; `git diff` against the pre-upgrade version is **empty** —
  byte-identical output)
- All four land in `tasks/0036-m8-tokenizer/` because that is `verify_embed_e2e.py`'s
  own hardcoded default `--out` directory (see the script's `main()`), not because this
  task belongs there.
- `runtime/build/npuembed.exe` — rebuilt binary, gitignored, not checked in.

## Next

**The mlir-aie 1.3.4 → 1.4.x upgrade is verified complete at the production level.**
Every shipped model reproduces its pre-upgrade accuracy figure exactly (or, for the two
without a prior saved e2e baseline, passes comfortably and now has one on record), the
C++ runtime builds clean with zero Python/IRON involvement, and a throughput sanity
check shows no gross regression. Nothing here is blocked or needs follow-up.

The one still-open item from `0058` — `tools/export_gemm_rtp.py` cannot **rebuild**
production xclbins on the new toolchain (a pretty-printer format change defeats its
cache marker) — is unaffected by this task's findings and remains tracked in
`research/OPEN-THREADS.md` (search `export_gemm_rtp.py`), not reopened here since it
was never in scope: this task verifies what already shipped, not the ability to
rebuild it.
