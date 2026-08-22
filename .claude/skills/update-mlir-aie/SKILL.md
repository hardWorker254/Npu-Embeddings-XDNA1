---
name: update-mlir-aie
description: Upgrade the C:\dev\mlir-aie toolchain (mlir_aie / llvm-aie / Peano) to a newer version, and migrate this repo's IRON design scripts if the upgrade breaks the Python API. Use when the user says "update mlir-aie", "upgrade the toolchain", "bleeding edge", or asks whether a newer mlir-aie has some feature.
---

# Upgrading the mlir-aie / IRON toolchain

Written from the 1.3.4 → main (past v1.4.1) upgrade in `tasks/0058`, which
broke every one of this repo's 15 IRON design scripts and took a full
evening: an infrastructure upgrade here is not a version bump, it is a
migration that may need re-verifying on hardware. Do not skip steps because
"it's probably fine."

**Toolchain fact, not obvious from upstream's docs**: this is a
**wheel-based install**, not a source build. `C:\dev\mlir-aie\ironenv` gets
`mlir_aie`/`llvm-aie` from prebuilt Windows wheels on GitHub Releases via
`utils/iron_setup.py`. The `C:\dev\mlir-aie` git checkout itself is mostly
along for the ride — it supplies `utils/iron_setup.py`, the Peano version
pin (`utils/peano-requirements.txt`), and (useful for comparison) upstream's
own `programming_examples/`, already written against whatever API the
checked-out commit ships. No LLVM compilation happens locally.

## 0. Before touching anything

1. **Check for live NPU work.** Another session's background agent may be
   mid-build against the current toolchain — checking out a new branch or
   swapping installed wheels under it will corrupt that work.
   ```powershell
   & "C:\Windows\System32\AMD\xrt-smi.exe" examine -r all   # no Active foreign hw_context
   Get-Process python*,npuembed*,aiecc*,clang* -ErrorAction SilentlyContinue
   ```
2. **Back up the current environment**, so rollback is a `pip install` away,
   not a repair job:
   ```powershell
   & "C:\dev\mlir-aie\ironenv\Scripts\python.exe" -m pip show mlir_aie   # record exact version
   & "C:\dev\mlir-aie\ironenv\Scripts\python.exe" -m pip show llvm-aie   # record exact version
   & "C:\dev\mlir-aie\ironenv\Scripts\python.exe" -m pip freeze > <somewhere>\ironenv_pip_freeze_before.txt
   ```
3. **Check the local `C:\dev\mlir-aie` checkout for uncommitted state**
   before switching branches — it is a *reference* tree per `CLAUDE.md`, but
   past sessions have left scratch edits in it (tracing config added to
   upstream's own `programming_examples/`, tasks 0002/0003). `git status
   --short --branch` first. Cross-check any modified tracked file against
   the task that plausibly produced it (search `tasks/*/TASK.md` for the
   filename) before discarding — don't assume "modified" means "disposable."
   An untracked directory with its own `.git` file (e.g.
   `third_party/hrx-xclbinutil`) is usually an upstream submodule the
   superproject's `.gitmodules` hasn't wired up yet, not local content —
   check `curl -s https://raw.githubusercontent.com/Xilinx/mlir-aie/<target-branch>/.gitmodules`
   before deleting it. Submodule diffs of the form `Submodule X <a>...<b>
   (commits not present)` are pointer drift, not dirty content — safe, `git
   submodule update --init --recursive` (which `iron_setup.py` runs by
   default) fixes them.

## 1. Move the checkout, then run setup

```powershell
cd C:\dev\mlir-aie
git fetch origin main
git checkout main         # requires step 0's tree to be clean
git pull --ff-only origin main
```

Then run `utils\iron_setup.py`. **`conda activate iron` does not reliably
work in a non-interactive PowerShell tool call** (the conda hook is a
`.bat`/function shim that needs an interactive or `conda init`-hooked
shell) — invoke the conda `iron` environment's Python by full path instead,
which is equivalent and does not silently no-op:

```powershell
& "C:\Users\vegar\.conda\envs\iron\python.exe" --version   # sanity check: 3.13.15
cd C:\dev\mlir-aie
& "C:\Users\vegar\.conda\envs\iron\python.exe" utils\iron_setup.py --dev
```

**Check `--help` before assuming last time's flags still work.**
`iron_setup.py`'s own CLI changed shape between 1.3.4 and main — the old
version had `install`/`update`/`env` subcommands; the current version is a
single "reconcile" invocation with `--dev`/`--extras`/`--wheelhouse`/etc.
and no subcommands at all. `--dev` now specifically means "install the
latest rolling **development** wheel via `pip --upgrade --pre`" — this is
bleeding edge, not just "whatever's newest stable." If you want the last
**tagged release** instead, pin `mlir_aie==<version>` explicitly, e.g. via
`--wheelhouse` or by checking out that tag rather than `main` before
running setup (see §4 for how to fetch a specific tagged wheel without
installing it, to compare first).

## 2. A known first-run gap: `llvm-objcopy.exe` not found

The Windows post-install step (patches `crt1.o`, aliases `.lib`→`.a` for the
Peano/llvm-aie toolchain) needs `llvm-objcopy.exe`, but it is *not* shipped
in the `llvm-aie` wheel's own `bin/` — it lives in the `mlir_aie` wheel's
`bin/`, and PATH may not have it yet during the same setup run that just
installed it:

```powershell
$env:PATH = "C:\dev\mlir-aie\ironenv\Lib\site-packages\mlir_aie\bin;" + $env:PATH
& "C:\Users\vegar\.conda\envs\iron\python.exe" utils\iron_setup.py --dev   # re-run
```
Idempotent — packages already at the target version show "Requirement
already satisfied" and only the failed post-install step re-runs. (It is
*not* a Visual Studio component in this setup, even though `llvm-objcopy`
sometimes ships as part of VS/LLVM installs elsewhere — here it comes from
the `mlir_aie` wheel.)

A successful run ends with:
```
IRON environment is ready.
  Python 3.13.15 | mlir_aie <version> | llvm-aie <version>
  XRT: C:\Xilinx\XRT
  NPU2: 1
To activate the IRON environment, run:
  PowerShell   . .\iron_env.ps1
```
`iron_env.ps1`/`iron_env.cmd` are now **generated by `iron_setup.py`
itself** (first line: `# Generated by utils/iron_setup.py. Re-run setup to
refresh this file.`) — no more manual maintenance of these files.

## 3. Purge the JIT cache — mandatory, not optional

Every cached `.xclbin`/`insts.bin` under `~/.npu/cache` was built by the
*old* toolchain. The cache's own purge markers are content-hash-based, not
version-aware, so a stale entry from before the upgrade can silently be
served to a script that thinks it built something fresh:

```bash
rm -rf ~/.npu/cache/*
```

## 4. Check for API breakage before trusting anything else works

**Do not assume a wheel bump is compatible.** Diff the `aie/iron/` Python
package between the old and new wheel versions *before* spending time
debugging individual scripts — this tells you the real blast radius in one
shot instead of discovering it file by file:

```powershell
# Download (not install) both wheels for comparison
& $py -m pip download mlir_aie==<old> --no-deps -d $old_dir -f "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v<old>/"
& $py -m pip download mlir_aie==<new> --no-deps -d $new_dir -f "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v<new>/"
# Extract mlir_aie/python/aie/iron/** from each .whl (it's a zip) into two folders, then:
# a per-file line-count/diff-status comparison across both trees
```
If most files in `aie/iron/` differ (not just the one function you already
know changed), expect this to be a **migration**, not a patch — verifying
one file's build success proves nothing about the other fourteen.

**This project's own precedent**: `Runtime()` went from a zero-arg
constructor + `with rt.sequence(...) as (...):` context manager to a
callback-based `Runtime(seq_fn, fn_args)` mirroring `Worker(core_fn,
fn_args)`. `research/notes/0008-iron-1.4-migration.md` is the concrete,
worked example of what a migration guide for this project should look
like — six transformation rules, each with old/new code side by side, plus
"what did NOT change" (explicitly confirmed, not assumed). Write one like
it for any future breaking upgrade; it is what makes migrating file 2
through N fast instead of re-discovering the pattern per file.

**Upstream's own `programming_examples/` is the best reference for correct
new-API usage** — since the checkout just moved to the new commit, every
example in that tree is already written against it. Diff your own script
against its closest upstream analogue (e.g. our `experiments/m1-hello-npu/
saxpy.py` vs. upstream's `programming_examples/getting_started/01_SAXPY/
saxpy.py`) to see the transformation directly, rather than reverse-engineering
it from error messages alone.

## 5. Verify — against real historical numbers, never a fresh read-back

Per `CLAUDE.md` traps 6b/6c: a migrated script "running without crashing"
is not verification. For each affected file, find its own historical
number (a task's `TASK.md`, `CLAUDE.md`'s own headline tables, or an
adjacent recorded artifact) and reproduce it exactly or within known
run-to-run noise:

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd <repo>\experiments\<...>
python <script>.py <known-good args>
```

Do not trust "the file shows as edited" as "confirmed working" — `git
status` shows what was *touched*, not what was *tested*. If migrating and
verifying were done by two different sessions, the second must
independently re-run every number, not just read the first session's notes
— a claimed result and a confirmed result are different things.

Blocked results are legitimate outcomes too: a design that fails on a
**pre-existing, independently documented** hardware limit (e.g. this
project's 16 KB program-memory wall from task 0032) is "migrated correctly,
blocked by something unrelated" — not a migration regression. State which
one it is; don't leave it ambiguous.

## 6. Production is not at risk, but the tool that rebuilds it might be

Already-built `runtime/artifacts_*/*.xclbin` files are static binaries. XRT
loads them regardless of what Python environment or mlir-aie version built
them — an IRON API change cannot corrupt a file that already exists on
disk. **But if anything needs to *regenerate* those artifacts** (e.g.
`tools/export_gemm_rtp.py`), check it specifically: it can break from
unrelated changes that have nothing to do with the API migration, such as
the MLIR pretty-printer's own textual output format changing (this
project's cache-marker string match broke when `aie.dma_bd`'s printed form
moved from `<size=,stride=>` to `sizes=[...] strides=[...]`, independent
of any `Runtime`/`ObjectFifo` change). Test the actual rebuild path before
assuming the upgrade is fully done.

## Rollback

```powershell
& "C:\dev\mlir-aie\ironenv\Scripts\python.exe" -m pip install --force-reinstall `
    "mlir_aie==<old version>" "llvm-aie==<old version>" `
    -f "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v<old version>/"
rm -rf ~/.npu/cache/*   # old wheel, purge again
```
Then `git checkout <old commit/tag>` in `C:\dev\mlir-aie` if the Peano pin
or examples tree also needs to match.

## Checklist

- [ ] no foreign NPU context / no live agent using the toolchain
- [ ] `pip freeze` + exact `mlir_aie`/`llvm-aie` versions recorded before touching anything
- [ ] `C:\dev\mlir-aie` working tree clean, cross-checked (not blindly discarded)
- [ ] `iron_setup.py --help` checked — flags may have changed shape between versions
- [ ] `~/.npu/cache` fully purged after the upgrade
- [ ] `aie/iron/` diffed old-vs-new before assuming a small blast radius
- [ ] every affected script re-verified against a real historical number, never a fresh read-back
- [ ] `tools/export_gemm_rtp.py` (or whatever rebuilds shipped artifacts) tested specifically, not assumed fine
- [ ] a `tasks/NNNN` log + `research/notes/NNNN-*.md` migration guide written if the API broke
- [ ] `research/OPEN-THREADS.md` updated for anything left unverified/unfixed
