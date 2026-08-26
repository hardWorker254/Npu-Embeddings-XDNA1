# 0106 — Toolchain provenance: record which mlir-aie/Peano built a design

- **Date** 2026-08-25
- **Milestone** research/tooling (T39)
- **Status** done

## Goal

Close T39 (`research/OPEN-THREADS.md`): nothing in the build path records
which mlir-aie/Peano toolchain produced a given design. `tasks/0102`'s audit
(filed as `research/notes/0009-toolchain-provenance.md`) had to fall back on
`.xclbin` mtime to reconstruct which shipped designs predate the 1.3.4 →
1.4.2 upgrade, in a repo whose trap 7c forbids exactly that identification
method, because no better source existed. Note 0009 priced the fix at under
an hour: a `toolchain.json` sidecar next to every exported `design.json`,
plus a runtime startup-banner line to report it.

## Context

- `research/notes/0009-toolchain-provenance.md` — the audit that filed T39,
  read in full before starting. Its "Provenance proposal" section (around
  line 287) specifies the fix.
- `research/OPEN-THREADS.md` T39 — the open thread this closes.
- `tasks/0104-*` — the precedent this task follows for the
  `datapath`/`datapath_recorded` field and its runtime banner line:
  ABSENT-IS-NOT-A-VALUE (a pre-0104 design has no `emulate_bfp16` key and
  must read as UNRECORDED, not silently guessed at), and "report the value
  you read, not the intention" (`runtime/src/main.cpp` reads
  `d_qkv.info().datapath_recorded` off the *loaded design*, never off a flag).
  This task's `toolchain_recorded` field and its two banner lines
  (`runtime/src/main.cpp:4058-4069` and `:5280-5290`, the `GemmaNpuEncoder`
  and BERT `d_qkv` code paths) copy that discipline directly.
- Scope instruction, given directly for this task: **do not** copy a
  design's `toolchain.json` into the `.npue` container, contrary to note
  0009's proposal. See "Departure from note 0009" below.

## What was done

**1. A small shared Python module, `tools/toolchain_provenance.py`.**
`write_toolchain_json(out_dir, mlir_aie_root=MLIR_AIE_ROOT)` reads three
strings and writes them to `out_dir/toolchain.json`:

- `mlir_aie_version` — `importlib.metadata.version("mlir_aie")`. Cheaper than
  note 0009's suggested `pip show mlir_aie` (no subprocess) and returns the
  identical string — confirmed against CLAUDE.md's own quoted value,
  `1.4.2.dev16+g7e00b57`.
- `peano_version` — `importlib.metadata.version("llvm-aie")`, same mechanism.
  `21.0.0.2026080301+c9c5ecb7`, matching CLAUDE.md's Peano row modulo the
  git-suffix digits CLAUDE.md doesn't quote.
- `mlir_aie_git_head` — `git -C C:\dev\mlir-aie rev-parse HEAD`, via
  `subprocess.run`, non-raising.

Every one of the three is independently wrapped so a failure in one does not
take down the others or the export: `_pkg_version()` and `_git_head()` each
catch every exception and return the literal string `"unavailable"`. Nothing
here can turn an export failure into a build failure — provenance is
recorded best-effort, exactly as note 0009's pricing assumed.

**2. Both export tools call it, next to every `design.json` they already
write**, per note 0009's proposal:

- `tools/export_gemm_rtp.py` — one call, after writing the single shared
  `gemm_rtp/design.json` (the unified design serves all batch tiers and all
  four shapes from one xclbin, so there is exactly one `design.json` to sit
  beside).
- `tools/export_xclbin.py` — two call sites, because this tool writes SEVEN
  separate `design.json` files (one per `qkv`/`attn_out`/`ffn_up`/`ffn_down`
  GEMM directory in `build_one()`, one per `gelu`/`layernorm`/`softmax`
  eltwise directory in `export_eltwise()`). Both call
  `write_toolchain_json(dst)` immediately after their `design.json` write,
  so every directory this tool produces gets its own sidecar.

**3. The runtime reads it in `Design::Design()`** (`runtime/src/npu_device.cpp`,
new block right after the `design.json` parse, before `buffer_bytes`). It
opens `dir + "/toolchain.json"` — a **separate** `ifstream`, not a key added
to `design.json` — and if the file exists, reads the three strings with the
same dependency-free `json_str()` helper design.json already uses, and sets
`toolchain_recorded = true`. If the file does not exist, the three fields
keep their struct defaults (`"unavailable"`) and `toolchain_recorded` stays
`false`. Same ABSENT-IS-NOT-A-VALUE split as `datapath_recorded`:
`toolchain_recorded` is `false` only when there is no sidecar at all, never
when the sidecar exists but a value inside it degraded to `"unavailable"` —
those two situations must print differently (UNRECORDED vs. a stated
"unavailable" value), and conflating them would be exactly the kind of lie
`datapath_recorded`'s own comment (`runtime/include/npu_device.hpp:88-99`)
was written to prevent.

New `DesignInfo` fields (`runtime/include/npu_device.hpp`):
`mlir_aie_version`, `peano_version`, `mlir_aie_git_head` (all
`std::string`, default `"unavailable"`), `toolchain_recorded` (`bool`,
default `false`).

**4. The runtime prints it**, one line, at both banner sites that already
print the `datapath` line (there are exactly two — the `GemmaNpuEncoder`
path at `main.cpp:4058` for arch 1/2, and the BERT `d_qkv` path at
`main.cpp:5280` for arch 0's unified-and-legacy-7-xclbin cases). Considered
putting it only in the startup banner as note 0009 suggested, but the
project's status-line discipline (0104's own principle, restated in this
task's brief) is that a status line states a value it read, so it goes right
next to `datapath`, which is the line it is most analogous to:

```
  datapath   bf16 MMAC, C as fp32
  toolchain  mlir_aie 1.4.2.dev16+g7e00b57, peano 21.0.0.2026080301+c9c5ecb7, mlir-aie HEAD 7e00b57955e108fe9d8e9419f5828a0c7e650858
```

or, for a design with no sidecar:

```
  datapath   UNRECORDED (design predates tasks/0104), C as fp32
  toolchain  UNRECORDED (design predates tasks/0106)
```

The 7-resident-xclbin (non-unified) path reports `d_qkv`'s toolchain only,
same reasoning as the existing `datapath` line right above it: one line per
encoder, not one per design, and `qkv` is the design already used as the
representative for `datapath`.

## Departure from note 0009's proposal

**Note 0009 also proposed that `tools/pack_npue.py` copy a design's
`toolchain.json` into the `.npue` container header. This was NOT done, on
purpose.**

A `.npue` is packed from HuggingFace weights by `pack_npue.py`, which never
invokes mlir-aie at all (`python tools/pack_npue.py` runs with no NPU
toolchain in the loop) — so a design's toolchain is not a property of the
container, it is a property of the *design that will later serve it*, and
those are different builds on different schedules. The relationship between
the two is many-to-many in both directions: one container is served by
several designs (`tasks/0104` gave MiniLM and bge-small *different* designs
— `artifacts_minilm_bfp16` and `artifacts_small_bf16` — at the same
geometry), and one design (e.g. `artifacts_b128il`, hidden 384) serves
several containers (MiniLM and bge-small both). Baking one design's build
string into a container would assert a 1:1 relationship that does not
exist.

This is the same mistake as putting the datapath in the container, which was
explicitly rejected when `tasks/0104` was briefed: the `.npue` is
byte-identical whether served by the bf16 design or the bfp16-emulated one,
and only `design.json` (now, also `toolchain.json`) knows which. So:
**`toolchain.json` lives next to `design.json`, in the artifacts directory
only, and `.npue`/`tools/pack_npue.py`/`tools/npue.py` are untouched by this
task.**

## Commands

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# Confirm the two package versions read the same way CLAUDE.md quotes them
python -c "import importlib.metadata as m; print('mlir_aie', m.version('mlir_aie'))"
python -c "import importlib.metadata as m; print('llvm-aie', m.version('llvm-aie'))"
# -> mlir_aie 1.4.2.dev16+g7e00b57
# -> llvm-aie 21.0.0.2026080301+c9c5ecb7

# Confirm C:\dev\mlir-aie is a git checkout and its HEAD
cd C:\dev\mlir-aie
git rev-parse HEAD
git rev-parse --is-inside-work-tree
# -> 7e00b57955e108fe9d8e9419f5828a0c7e650858
# -> true
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# Runtime compiles clean with the new fields/reader/banner lines
cd runtime
cmake --build build --config Release
cd ..

# Re-export the shipping MiniLM/bge-small design set (matches tasks/0037's
# own command exactly -- the same 4-tier, 8-column production build), now
# with a real toolchain.json written alongside each design.json
python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 `
    --out runtime\artifacts_b128il

# Runtime prints the recorded toolchain on the re-exported set, and still
# passes the golden gate
cd runtime
.\build\npuembed.exe .. --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --threads 16

# Runtime prints UNRECORDED on a design this task did NOT re-export
.\build\npuembed.exe .. --model bge-base-en-v1.5 --artifacts artifacts_base --threads 16
cd ..
```

**Graceful-degradation proof** (against the failing case, not only the
working one, per the brief):

```powershell
# 1. Isolated function test, iron env active: real mlir-aie root vs. a
#    directory that is not a git checkout at all
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python -c "
import sys; sys.path.insert(0, 'tools')
from pathlib import Path
from toolchain_provenance import write_toolchain_json
scratch = Path(r'C:\Users\vegar\AppData\Local\Temp\claude\C--Users-vegar-Documents-GitHub-NpuEmbeddings\a0672a5d-77f9-447c-a309-d51b903b0596\scratchpad')
print(write_toolchain_json(scratch / 'degrade_test_out'))
print(write_toolchain_json(scratch / 'degrade_test_out', mlir_aie_root=scratch / 'not_a_git_dir'))
"
# -> {'mlir_aie_version': '1.4.2.dev16+g7e00b57', 'peano_version': '21.0.0.2026080301+c9c5ecb7', 'mlir_aie_git_head': '7e00b57955e108fe9d8e9419f5828a0c7e650858'}
# -> {'mlir_aie_version': '1.4.2.dev16+g7e00b57', 'peano_version': '21.0.0.2026080301+c9c5ecb7', 'mlir_aie_git_head': 'unavailable'}

# 2. End-to-end tool test: temporarily point tools/toolchain_provenance.py's
#    MLIR_AIE_ROOT constant at the same non-git scratch directory, then run
#    a REAL (cheap) export through the actual export tool -- not just the
#    helper function in isolation
#    [edited MLIR_AIE_ROOT in tools/toolchain_provenance.py -> scratch\not_a_git_dir]
python tools\export_gemm_rtp.py --batch 4 --cols 2 --out runtime\artifacts_degrade_test
# -> ...
#    toolchain  mlir_aie 1.4.2.dev16+g7e00b57, peano 21.0.0.2026080301+c9c5ecb7, mlir-aie HEAD unavailable
#    wrote runtime\artifacts_degrade_test\gemm_rtp -- ONE xclbin, 4 streams (4 shapes x 1 batch tiers)
#    (exit code 0 -- the export did not fail over provenance)
#    [reverted MLIR_AIE_ROOT to C:\dev\mlir-aie; deleted runtime\artifacts_degrade_test]
```

## Result

**Runtime, recorded case** (`artifacts_b128il`, re-exported this task):

```
NpuEmbeddings C++ runtime -- full encode
  bo-mode    host_only (data-buffer allocation)
  model      all-MiniLM-L6-v2: 78 tensors, 69.00 MB, checkpoint 53aa51172d142c89
  shape      sentence-transformers/all-MiniLM-L6-v2: 6 layers, hidden 384, 12 heads x 32, ffn 1536, mean pooling
  designs    ONE xclbin, 16 streams (4 batch tiers), one hw_context
  datapath   bf16 MMAC, C as fp32
  toolchain  mlir_aie 1.4.2.dev16+g7e00b57, peano 21.0.0.2026080301+c9c5ecb7, mlir-aie HEAD 7e00b57955e108fe9d8e9419f5828a0c7e650858
  shape      batch 128 x seq 64  (M = 8192)
  ...
  embedding rel_fro vs HF golden           4.473e-03
  worst 1 - cos vs HuggingFace             1.086e-05  (all 128 rows, ...)

PASS -- tolerance 2e-03 on 1-cos, no Python in this process
```

`1-cos 1.086e-05` reproduces note 0009's own audit table figure for this
exact design set exactly, on a genuinely fresh rebuild from the unmodified
source tree — an independent third confirmation of the number 0009 already
cited two of (`0059`'s bit-identical re-verification of the never-rebuilt
binary, and `0060`'s from-scratch 1.4.2 rebuild).

**Runtime, unrecorded case** (`artifacts_base`, not touched this task):

```
  designs    ONE xclbin, 16 streams (4 batch tiers), one hw_context
  datapath   UNRECORDED (design predates tasks/0104), C as fp32
  toolchain  UNRECORDED (design predates tasks/0106)
```

**`runtime/artifacts_b128il/gemm_rtp/toolchain.json`** (new file, gitignored
directory, not checked in — same as every other file under `runtime/artifacts_*`):

```json
{
  "mlir_aie_version": "1.4.2.dev16+g7e00b57",
  "peano_version": "21.0.0.2026080301+c9c5ecb7",
  "mlir_aie_git_head": "7e00b57955e108fe9d8e9419f5828a0c7e650858"
}
```

**Which artifacts were re-exported and which were not.** Only
`runtime/artifacts_b128il` (MiniLM + bge-small's design set) was rebuilt —
one `python tools\export_gemm_rtp.py` invocation, ~a few minutes for all 16
streams (4 shapes x 4 batch tiers) at 8 columns. `artifacts_base`,
`artifacts_large`, `artifacts_nomic`, `artifacts_gemma`, every
`artifacts_*_bfp16`/`artifacts_int8*` directory, and every eltwise design
(`gelu`/`layernorm`/`softmax`, which `export_xclbin.py` also now sidecars)
were **not** rebuilt — re-exporting six model design sets each requiring an
8-column, multi-tier IRON+Peano recompile is well past "under an hour" and
was explicitly out of scope (brief: "do not re-export everything unless it
is cheap"). Every one of those un-re-exported directories will correctly
report `toolchain UNRECORDED (design predates tasks/0106)` until someone
re-exports it — which is the honest state, not a gap to paper over.

## Problems hit

1. **`npuembed.exe .. --artifacts <dir> --threads 16` alone now refuses**
   with `error: several models are installed; say which with --model
   <name>` — a CLI change from when the docs' example commands (`tasks/0031`
   onward) were written; this project now has 17 `.npue` files installed,
   not one. Fix: pass `--model all-MiniLM-L6-v2` (or `bge-base-en-v1.5`
   etc.) explicitly. Not a bug this task caused or needed to fix — recorded
   because the next person hitting this exact error message should not have
   to re-derive it.
2. **Background PowerShell + `Tee-Object` gave an empty log file for
   several minutes** while `export_gemm_rtp.py --batches 4,16,32,128 --cols
   8` was genuinely running (confirmed via `Get-Process python` showing
   ~945 MB RSS and active CPU on the compile process) — stdout is
   block-buffered when piped through `Tee-Object`, so nothing appears until
   the process exits or a buffer fills. Not a bug, just a trap for anyone
   watching a background export "for progress": absence of output is not
   evidence of a hang.

## Artifacts

- `tools/toolchain_provenance.py` — new, the shared sidecar writer.
- `tools/export_gemm_rtp.py`, `tools/export_xclbin.py` — one/two call
  sites added, plus the corresponding import line each.
- `runtime/include/npu_device.hpp` — four new `DesignInfo` fields.
- `runtime/src/npu_device.cpp` — the `toolchain.json` reader block.
- `runtime/src/main.cpp` — two new banner-print sites (arch 1/2 and arch 0
  paths).
- `runtime/artifacts_b128il/gemm_rtp/toolchain.json` — the one real
  artifact this task produced on disk (gitignored, not checked in, exists
  only on this machine).
- Nothing in `tools/pack_npue.py`, `tools/npue.py`, or the `.npue` header
  struct changed — the deliberate departure above.

## Next

- T39 closes. Register text is below, for the coordinator to apply
  (`research/OPEN-THREADS.md`, `research/CLOSED-THREADS.md`,
  `research/notes/0009-toolchain-provenance.md`, `tasks/README.md` are not
  edited by this task per its own brief).
- Whoever next re-exports `artifacts_base`, `artifacts_large`,
  `artifacts_nomic`, `artifacts_gemma`, or any `*_bfp16`/`int8*` set for an
  unrelated reason gets a real `toolchain.json` for free — no follow-up task
  needed to backfill the rest; they simply read UNRECORDED until touched.

## Proposed register update

**T39 closure text**, to move from `research/OPEN-THREADS.md` into
`research/CLOSED-THREADS.md`:

> ### T39 — Nothing records which toolchain built an artifact · **ANSWERED,
> closed by [`0106`](TASK.md), 2026-08-25**
>
> `tools/export_gemm_rtp.py` and `tools/export_xclbin.py` now write a
> `toolchain.json` sidecar next to every `design.json` they emit
> (`mlir_aie_version`, `peano_version`, `mlir_aie_git_head`, each
> independently best-effort — a failure to read one degrades that field to
> `"unavailable"` rather than failing the export). The runtime reads it in
> `Design::Design()` and prints it on both banner paths, right next to the
> `datapath` line, with the same ABSENT-IS-NOT-A-VALUE discipline
> `tasks/0104` established: a design with no sidecar reads `toolchain
> UNRECORDED (design predates tasks/0106)`, never a guess.
>
> **One deliberate departure from note 0009's proposal**: the container
> (`.npue`) does NOT get a copy of its serving design's `toolchain.json`.
> `pack_npue.py` never invokes mlir-aie, and the design<->container
> relationship is many-to-many in both directions (`tasks/0104` gave MiniLM
> and bge-small different designs at the same geometry) — baking one
> design's build string into a container would assert a 1:1 relationship
> that does not exist, the same reasoning that kept the datapath field out
> of the container in the first place.
>
> Re-exported and verified on one real shipping design set
> (`runtime/artifacts_b128il`, MiniLM + bge-small, the same 4-tier
> 8-column production build `tasks/0037` first produced): the runtime now
> prints `toolchain mlir_aie 1.4.2.dev16+g7e00b57, peano
> 21.0.0.2026080301+c9c5ecb7, mlir-aie HEAD 7e00b57955e108fe9d8e9419f5828a0c7e650858`,
> and the golden gate reproduces note 0009's own cited figure for this
> design, `1-cos 1.086e-05`, exactly. Every other shipped design set still
> reads UNRECORDED until someone re-exports it — not backfilled, since doing
> so for six model design sets is well past this task's "under an hour"
> budget.

**Proposed amendment to `research/notes/0009-toolchain-provenance.md`**, to
append after its "Proposal, not implemented" paragraph (which should be
retitled "Proposal, implemented in tasks/0106" or similarly marked done):

> **Implemented in [`tasks/0106`](../../tasks/0106-toolchain-provenance/TASK.md),
> 2026-08-25, with one change from this proposal: the container half was NOT
> taken.** This paragraph proposed `tools/pack_npue.py` copy a design's
> `toolchain.json` into the `.npue` header. 0106 argues that is a category
> error and does not do it: `pack_npue.py` never invokes mlir-aie (a
> container is packed from HuggingFace weights, full stop), so a design's
> toolchain is not a property of the container it happens to be served
> with — and the design<->container relationship is many-to-many in both
> directions, so there is no single design whose provenance a container
> could even correctly claim. `toolchain.json` lives only next to
> `design.json`, exactly where the export tools already write it. The
> sidecar mechanism, the runtime read/print, and the three fields
> (`mlir_aie_version`, `peano_version`, `mlir_aie_git_head`) are otherwise
> exactly as proposed here, and `importlib.metadata.version()` turned out to
> be a cheaper way to the same two version strings than the `pip show`
> invocation this paragraph suggested (no subprocess, identical value —
> confirmed against `mlir_aie` and `llvm-aie`).

**Proposed `tasks/README.md` index row** (M-column follows this section's
own header, "research/tooling"):

```markdown
| [0106](0106-toolchain-provenance/TASK.md) | **T39 closed: `toolchain.json` sidecar records which mlir-aie/Peano built a design** — both export tools write it next to `design.json` (three best-effort strings, never fails the export); the runtime reads and prints it next to `datapath`, UNRECORDED for designs that predate it. Re-exported and verified on `artifacts_b128il` (MiniLM/bge-small): `1-cos 1.086e-05` reproduces note 0009's own figure on a fresh 1.4.2 rebuild. Deliberately does NOT copy a design's toolchain into the `.npue` container — the design<->container relationship is many-to-many, the same reasoning that kept the datapath field out of it | research/tooling (T39) | done |
```

---

## Coordinator review, appended 2026-08-25

**The one design that got provenance was not a shipping design.** The brief
said "re-export at least one shipping design" and this task chose
`artifacts_b128il`, using `tasks/0037`'s original command. That directory
**stopped being a shipping design the day before**: `tasks/0104` replaced it
with `artifacts_small_bf16`, and `tools/make_release.ps1` now names
`artifacts_small_bf16`, `artifacts_minilm_bfp16`, `artifacts_base_bfp16`,
`artifacts_large_bfp16` and `artifacts_nomic_bfp16`. So the feature worked and
covered nothing that ships — all five release directories read `UNRECORDED`.

Re-exported all six (the five in the release list plus `artifacts_gemma_bfp16`,
which `embed` uses even though `serve` does not ship it), with the commands
recorded verbatim in `tasks/0104`. All six now carry
`mlir_aie 1.4.2.dev16+g7e00b57`, `peano 21.0.0.2026080301+c9c5ecb7`,
`mlir-aie HEAD 7e00b57955e108fe9d8e9419f5828a0c7e650858`, and the runtime
prints both `datapath` and `toolchain` per model.

### Two failures on the way, and the first one was mine

**1. My re-export script claimed success while two of six had failed.** It set
`$ErrorActionPreference = 'Stop'` and printed `ALL SIX RE-EXPORTED`. That
preference does **not** apply to a native command's exit code — only to
PowerShell cmdlet errors — so `python ... ; python ... ` sailed past two
failures. The log had the evidence at line 60:

```
[XRT] ERROR: Failed to submit command to hw queue (0xc00002b6): The device has been removed.
```

Caught only because the artifact check afterwards listed five directories, not
six. **Rewritten to test `$LASTEXITCODE` after every call and to refuse to
claim success** — the same fail-open shape `docs/CURRENT_STATUS.md` §4 lists
five of, this time in a throwaway script, which is exactly where it is easiest
to let slide.

**2. A transient NPU device-removal produced a subtly wrong build, and the
exporter's own guard caught it.** The retry of `artifacts_base_bfp16` failed
with:

```
identity qkv@b128 vs qkv@b4    82 differing bytes  DIVERGED
static configurations diverged -- the streams do NOT share an xclbin,
refusing to export a lying artifact
```

82 against a threshold of 80, where `tasks/0104` built the identical
configuration at ≤80 the day before. **Not reproducible**: a third run gave
68–73 differing bytes, all OK, and exported cleanly. So the divergence was a
consequence of the device-removal event disturbing the build, not a real
property of the configuration — and the guard did precisely its job, refusing
to ship an artifact produced during a disturbed build rather than passing it
at a threshold nobody would have re-examined. Worth recording as a positive
result for that check, which is otherwise invisible when everything works.

The NPU was verified healthy afterwards (`xrt-smi examine --report
aie-partitions`: *"No hardware contexts running on device"*).
