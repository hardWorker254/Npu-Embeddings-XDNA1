# 0104 — Adopt bfp16 per model, and close the datapath fail-open that adoption creates

- **Date** 2026-08-24
- **Milestone** M13 (T23)
- **Status** done — the guard is built and verified against a known-bad
  artifact, all five adopted production artifact sets are built and
  hardware-verified (golden gate + objdump), and the sixth (bge-small)
  is confirmed unchanged on its original plain-bf16 set.

## Goal

The user adopted bfp16 (`--emulate-bfp16 --c-bf16`) **per model, wherever
MTEB passes** — five of six models pass
([`0101`](../0101-t23-bfp16-gates-on-1.4.2/TASK.md),
[`0103`](../0103-t23-bfp16-all-models/TASK.md)); `bge-small-en-v1.5` fails at
−0.5010 against the −0.5 line and stays on plain bf16. Before building
anything, close the fail-open this decision creates: nothing in
`design.json` or the selection path records which MMAC datapath a design
was built for, so a model that did NOT clear the bfp16 gate could be handed
a bfp16 design silently — and `bge-small` shares MiniLM's hidden-384
geometry with the model that DID clear it, which is exactly the pairing
this has to prevent.

## Context

Read first, per the brief: `CLAUDE.md`, `docs/CURRENT_STATUS.md` §4 ("Five
bugs that failed open"), `tasks/0101`, `tasks/0103`. The precedent to follow
is int8's: `a_dtype` is written into `design.json`
(`tools/export_gemm_rtp.py`), read by `runtime/src/npu_device.cpp:210`,
unknown values throw, and int8's differing `b_layout_hash` makes a
mismatched pair refuse at stage time.

**Verified before designing anything, as instructed:**

- `runtime/artifacts_b128il/gemm_rtp/design.json` (production) has no field
  recording the datapath — confirmed by reading the file directly.
- The `b_layout_hash` of a bfp16 design and a plain-bf16 one at the same
  geometry are **byte-identical**: built `runtime/artifacts_minilm_bfp16`
  (hidden 384, `--emulate-bfp16 --c-bf16`) and compared its
  `b_layout_hash` to `artifacts_b128il`'s — both
  `94266693ea31aa674279b0cb124eb9a731276064eae1442ac55841dc266dddaf`. This
  makes sense: `gemm_b_layout()` only ever sees `dtype="BF16"` for either
  path (`tools/export_gemm_rtp.py`'s `b_layout` computation reads
  `dtype="I8" if args.int8 else "BF16"` — bfp16 is not int8), so the
  existing int8-vs-bf16 pairing guard (`want_layout` in `design_fits()`)
  cannot tell a bfp16 design from a plain-bf16 one.
- `design_fits()` in `runtime/src/main.cpp` (~line 3433, pre-change) matched
  on geometry (and, since tasks/0080, `b_layout_hash`) only — no datapath
  axis existed.
- Confirmed the exact bite: with `bge-small` and MiniLM sharing hidden 384,
  two directories (`artifacts_b128il`, a hypothetical bfp16 one) would both
  fit `bge-small`, and `pick_artifacts()`'s alphabetical tie-break — not a
  correctness decision — would pick between them.

## Part 1 — the guard

### Mechanism

1. **`tools/export_gemm_rtp.py`** now writes `"emulate_bfp16": bool(...)`
   into `design.json`'s `meta` dict, alongside the pre-existing `c_dtype`
   (already records the C transport width, satisfying that half of
   requirement 1 for free — no separate field was needed for it).
2. **`runtime/include/npu_device.hpp`**: `DesignInfo` gains
   `bool emulate_bfp16 = false;`.
3. **`runtime/src/npu_device.cpp`**: a new `json_bool()` helper (same style
   as the file's existing `json_str`/`json_int`), and `Design::Design()`
   reads `info_.emulate_bfp16 = json_bool(js, "emulate_bfp16", false);`.
   Absent means false — correct for every design exported before this task.
4. **`runtime/src/main.cpp`**: `design_fits()` and `pick_artifacts()` each
   gain a `want_datapath` parameter (`"bf16"` / `"bfp16"` / `""` =
   unspecified, mirroring the existing `want_layout` pattern from 0080).
   `design_fits()` reads `emulate_bfp16` out of the candidate's
   `design.json` and refuses if it disagrees with `want_datapath`.
5. **`runtime/include/hub.hpp`**: `CatalogEntry` gains
   `std::string datapath = "bf16";` — the deployment decision, **not** a
   container field (the `.npue` is byte-identical either way, per the
   brief's explicit instruction not to invent one).
6. **`runtime/src/hub.cpp`**: `table()`'s literal rows are unchanged
   (no field was inserted into their positional initialisers, which would
   have meant restating every trailing field for every row); instead the
   built-in table is now an immediately-invoked lambda that builds the rows
   and then does `for (auto &e : rows) if (e.name != "bge-small-en-v1.5")
   e.datapath = "bfp16";` — the adoption decision lives in one place, keyed
   by name, with the real gate verdicts cited in a comment.
7. **Call sites now pass the model's adopted datapath**, looked up by name
   from `npue::hub::find()`:
   - The BERT/nomic `serve`/`embed` dispatch (`main.cpp`, the automatic
     `pick_artifacts()` call keyed off `want`, the model name typed on the
     command line).
   - `run_gemma_mode()`'s automatic `pick_artifacts()` call, keyed off the
     container path's stem (arch=1 has no `--model` string available at
     that point, only a path).
   - `print_catalog()`'s two `have_design`/`ready` checks — the catalogue
     loop uses `e.datapath`; the "locally packed, not in the catalogue"
     loop (models with no adoption decision at all) uses the explicit safe
     default `"bf16"`.
   - An explicit `--artifacts` always wins over all of this, unchanged —
     same precedent int8 already established (`tasks/0080`'s own comment,
     quoted in the brief).
8. **Status line** (requirement 4): both the BERT unified path and the
   Gemma path now print `datapath   bfp16-emulated MMAC, C as bf16` (or
   `bf16 MMAC, C as fp32`), read from the **loaded `Design`'s own
   `info().emulate_bfp16`**, never from a flag — the same "reports the
   intention, not the value" discipline `docs/CURRENT_STATUS.md` names for
   `tasks/0042`/`0081`.

### Known-bad verification (Part 1 step 5)

Built `runtime/artifacts_minilm_bfp16` (see Part 2) first specifically to
use as the known-bad artifact, since every **pre-existing** bfp16 research
directory (`artifacts_minilm_bfp_cbf16`, `artifacts_bfp_rtp`, etc., from
tasks 0052/0053/0101) predates this field and therefore reads as `"bf16"`
by the same "absent means false" rule that makes old designs safely
default — they would NOT have exercised the guard.

**1. `list` against the real repo, both directories present.** `bge-small`
resolves to `artifacts_b128il` (plain bf16); the new
`artifacts_minilm_bfp16` is correctly **excluded** from its "also fits"
candidates — confirmed, then superseded by the cleaner isolated test below
(same conclusion, no ambiguity about which of ten candidate directories did
the work).

**2. Isolated known-bad test — the one whose output is saved.** Built a scratch root
(`guard_test_root/`, NTFS junctions, deleted after the test — not
committed) containing **only** `artifacts_minilm_bfp16` under `runtime/`,
and the real `models/` directory junctioned in so both containers are
"installed" without a fetch:

```powershell
cmd /c mklink /J guard_test_root\runtime\only_bfp16 runtime\artifacts_minilm_bfp16
cmd /c mklink /J guard_test_root\models models   # real models/, so nothing needs fetching

.\build\npuembeddings.exe list --root guard_test_root
.\build\npuembeddings.exe embed bge-small-en-v1.5 tiny_input.txt --root guard_test_root
.\build\npuembeddings.exe embed all-MiniLM-L6-v2  tiny_input.txt --root guard_test_root
```

Result (`known_bad_bge_small_refuses.txt`, `known_bad_embed_refuses.txt`,
`known_good_minilm_embed.txt`):

| model | design present | `list` state | `embed` result |
|---|---|---|---|
| `bge-small-en-v1.5` | only the bfp16 one | **`no design`** | **refuses**, exit 2: `error: no NPU design for hidden 384 intermediate 1536, datapath bf16 under ... -- this release carries designs for the geometries it was built with; export one with tools\export_gemm_rtp.py --hidden 384 --intermediate 1536` |
| `all-MiniLM-L6-v2` | same directory | `ready` | **succeeds**, exit 0, and prints `datapath   bfp16-emulated MMAC, C as bf16` |

This is the direct proof required by Part 1 step 5: the identical directory
is refused for the model that failed the gate and accepted for the model
that passed it — not by luck of alphabetical sort (bge-small's real
production directory, `artifacts_b128il`, does sort first, which is why the
scratch-root test removes it from the candidate set entirely rather than
relying on that).

## Part 2 — build and adopt

### Which directory belongs to which model

**Chose to keep `artifacts_b128il` as bge-small's** (no rebuild, no rename)
**and give MiniLM a new directory, `artifacts_minilm_bfp16`.** Rationale:
`artifacts_b128il` is the one production directory that must not regress —
it is bge-small's now-exclusive design and was already the well-tested
default name several tools and docs reference. Touching it would be the
higher-risk move for zero benefit; every other adopted model already needed
a brand-new directory anyway (none of them shared one with an unadopted
model), so making MiniLM's the one new directory at hidden 384 keeps the
pattern uniform: **every model that changed datapath gets a new directory
name; the one model whose datapath did not change keeps its old one
untouched.**

### Builds

Recipes matched each model's current shipping geometry exactly (read from
the tasks that originally built the plain-bf16 production sets: `0032`
MiniLM, `0051` bge-base, `0042` bge-large, `0069` nomic, `0074` gemma), with
`--emulate-bfp16 --c-bf16` added:

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 `
    --emulate-bfp16 --c-bf16 --out runtime\artifacts_minilm_bfp16

python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 --hidden 768 `
    --emulate-bfp16 --c-bf16 --out runtime\artifacts_base_bfp16

python tools\export_gemm_rtp.py --hidden 1024 -n 32 --batch 128 `
    --emulate-bfp16 --c-bf16 --out runtime\artifacts_large_bfp16

python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 `
    --hidden 768 --intermediate 3072 --gated-ffn -n 48 `
    --emulate-bfp16 --c-bf16 --out runtime\artifacts_nomic_bfp16

python tools\export_gemm_rtp.py --batch 128 --batches 4,16,32,128 --cols 8 `
    --hidden 768 --intermediate 1152 --gated-ffn --qkv-n 1536 `
    --emulate-bfp16 --c-bf16 --out runtime\artifacts_gemma_bfp16
```

All five: **every identity check (≤80 differing bytes) OK**, one xclbin per
model. Geometry cross-checked from each `design.json`:

| set | hidden | intermediate | gated | qkv_n | emulate_bfp16 | c_dtype | tiers |
|---|---:|---:|---|---:|---|---|---|
| `artifacts_minilm_bfp16` | 384 | 1536 | no | 1152 | true | bf16 | 4,16,32,128 |
| `artifacts_base_bfp16` | 768 | 3072 | no | 2304 | true | bf16 | 4,16,32,128 |
| `artifacts_large_bfp16` | 1024 | 4096 | no | 3072 | true | bf16 | 128 |
| `artifacts_nomic_bfp16` | 768 | 3072 | yes | 2304 | true | bf16 | 4,16,32,128 |
| `artifacts_gemma_bfp16` | 768 | 1152 | yes | 1536 | true | bf16 | 4,16,32,128 |

### objdump verification (AIE objdump, not the MSVC one on PATH — trap 1)

```powershell
C:\dev\mlir-aie\ironenv\Lib\site-packages\llvm-aie\bin\llvm-objdump.exe -d <cache-dir>\matmul_bf16_f32_<hash>.o
```

| model | matmul config hash | `vconv.bfp16ebs8.fp32` | `crrnd` | narrow (bf16-C epilogue) `crrnd` |
|---|---|---:|---:|---:|
| MiniLM | `333c4d33` | 20 | 3 | 3 |
| bge-base | `333c4d33` | 20 | 3 | 3 |
| bge-large | `0c5c718d` (different tile shape, `n=32`) | 20 | 3 | 3 |
| nomic | `333c4d33` | 20 | 3 | 3 |
| gemma | `333c4d33` | 20 | 3 | 3 |

Every one matches the pattern `0101` confirmed as genuinely-on (20/3, where
0098's stale pre-migration artifact had 0). Emulation is real in all five
builds, not a silent no-op from trap 1.

### Golden gate — every model, on hardware

```powershell
.\build\npuembed.exe .. --model <model> --artifacts <set> --threads 16
```

(gemma uses the differential harness instead — see below.)

| model | artifacts | `rel_fro` | worst `1-cos` | verdict | vs 0101/0103 |
|---|---|---:|---:|---|---|
| `all-MiniLM-L6-v2` | `artifacts_minilm_bfp16` | 2.430e-02 | 3.406e-04 | PASS | **exact match** (0101: 2.430e-02 / 3.406e-04) |
| `bge-small-en-v1.5` | `artifacts_b128il` (unchanged) | 3.789e-03 | 8.348e-06 | PASS | **exact match** to `docs/CURRENT_STATUS.md`'s recorded plain-bf16 baseline |
| `bge-base-en-v1.5` | `artifacts_base_bfp16` | 1.941e-02 | 2.284e-04 | PASS | **exact match** (0101: 1.941e-02 / 2.284e-04) |
| `bge-large-en-v1.5` | `artifacts_large_bfp16` | 1.973e-02 | 2.626e-04 | PASS | **exact match** (0103: 1.973e-02 / 2.626e-04) |
| `nomic-embed-text-v1.5` | `artifacts_nomic_bfp16` | 4.845e-02 | 1.402e-03 | PASS | **new measurement** — 0101/0103 never ran nomic's golden gate, only its MTEB gate |
| `embeddinggemma-300m` | `artifacts_gemma_bfp16` | — (differential harness) | 1-cos worst 2.315e-04 | PASS | **new measurement** — same reason |

**Nothing that had a prior number moved — all four are bit-exact
reproductions.** Two numbers are genuinely new (nomic, gemma), consistent
with 0103's MTEB PASS for both but not previously checked against the
golden gate at all.

**Finding, reported rather than smoothed over: nomic's margin is much
thinner than the other four.** MiniLM/bge-base/bge-large/gemma all land
7–10x inside the 2e-03 tolerance (2.3–3.4e-04). Nomic lands at **1.402e-03
— only ~1.4x inside**, the closest any bfp16 golden-gate result has come to
the line while still passing (compare: bge-base's plain-bf16+fp32-C bfp16
FAIL in 0101 was at 2.395e-03, just *outside*). MTEB already passed cleanly
for nomic (0103: mean +0.01, worst −0.25), and MTEB is this project's
stated accuracy authority (0035), so this does not change the adoption
verdict — but it is the kind of number that should be watched if nomic's
FFN geometry (K=6144, the widest single dispatch in the whole catalogue)
is touched again.

Gemma's golden check is the differential harness (`0074`'s protocol: NPU
path vs the host-only `.cpp_test` control, 13 distinct sentences):

```powershell
.\runtime\build\npuembed.exe . --model embeddinggemma-300m.cpp_test --embed corpus.txt out_cpu.f32
.\runtime\build\npuembed.exe . --model embeddinggemma-300m --embed corpus.txt out_npu.f32 --threads 24 --pipeline 4
.venv-ref\Scripts\python.exe tools\verify_gemma_npu_encode.py --npu out_npu.f32 --cpu out_cpu.f32
```

Result: `1-cos` mean 1.749e-04, worst 2.315e-04 (row 0), gate 2e-03,
**PASS**, nearest-is-self 13/13 — comfortably inside, in family with
MiniLM/base/large rather than with nomic's thinner margin. The design
correctly auto-selected (`designs .\runtime\artifacts_gemma_bfp16`,
`datapath bfp16-emulated MMAC, C as bf16`) with no `--artifacts` given,
confirming the guard's Gemma-path call site works end to end.

### End-to-end confirmation of the automatic selection path

Beyond the golden-gate runs (which used explicit `--artifacts`), ran
`embed` with **no `--artifacts`**, so the full catalogue → `design_fits()`
→ selection chain executes exactly as `serve` would use it:

```powershell
.\build\npuembeddings.exe embed bge-base-en-v1.5      tiny_input.txt --root ..
.\build\npuembeddings.exe embed nomic-embed-text-v1.5 tiny_input.txt --root .. --prefix search_document
```

Both auto-selected their `*_bfp16` set and printed
`datapath   bfp16-emulated MMAC, C as bf16` with no operator input beyond
the model name. (`e2e_embed_bge_base_auto.txt`, `e2e_embed_nomic_auto.txt`.)

## `tools/make_release.ps1`

Default `-Artifacts` list updated: `artifacts_base`, `artifacts_large`,
`artifacts_nomic` replaced by their `_bfp16` equivalents; `artifacts_b128il`
kept (still bge-small's, unchanged) and `artifacts_minilm_bfp16` added
(MiniLM's new set) — five directories now, covering the same five models as
before plus the geometry split MiniLM/bge-small now needs.
`embeddinggemma-300m` remains **outside** this list — it was never in it
(arch=1 has no `--serve`/HTTP path yet, only `embed`; the default list is
what a `serve`-oriented release ships), a pre-existing gap this task did
not create and did not attempt to close. `artifacts_gemma_bfp16` exists,
is objdump- and golden-gate-verified, and works via `embed`; it is simply
not part of the packaged release zip, same as `artifacts_gemma` (plain
bf16) never was.

Also added `emulate_bfp16` to the per-set manifest entry (`$sets` /
`designs` in the built manifest JSON) — the durable, at-a-glance record of
which datapath each shipped directory carries, read the same way `hidden`/
`gated_ffn` already are (from `design.json`, never inferred from the
directory name).

Syntax-checked with `[System.Management.Automation.Language.Parser]::ParseFile`
(no interpreter available to run the full pipeline without a fresh sweep
and exe; not exercised end to end).

## Build verification

`cmake --build runtime\build --config Release` — clean build, both
`npuembed.exe` and `npuembeddings.exe` link (pre-existing `getenv`
deprecation warnings only, unrelated to this change).

## Problems hit

1. **The obvious "known-bad" artifact wasn't one.** Every *pre-existing*
   bfp16 research directory under `runtime/` (`artifacts_minilm_bfp_cbf16`,
   `artifacts_bfp_rtp`, `artifacts_cbf16`, …, from 0052/0053/0101) predates
   `emulate_bfp16` ever being written, so under the "absent means false"
   rule they silently read as `"bf16"` — safe by construction for *shipped*
   designs, but it meant they could NOT exercise the guard as a known-bad
   case. Solved by building the real `artifacts_minilm_bfp16` (Part 2's own
   production artifact) first and using *that* as Part 1's known-bad probe.
   Recorded because it is exactly the kind of thing that would have let a
   weaker verification pass silently — pointing the test at a directory that
   *looked* bfp16 by name but was not, datapath-wise, by the new field.
2. **`print_catalog`'s "also fits" note initially looked wrong** — with
   both `artifacts_b128il` and the new `artifacts_minilm_bfp16` present, the
   real repo's `list` output printed the same "10 design sets serve hidden
   384" note **five times**, which looked like a bug. Traced to the second
   ("locally packed, not in the catalogue") loop, which this task also
   wired to require `"bf16"` explicitly: several uncatalogued locally-packed
   models (`all-MiniLM-L6-v2.int8`, `all-MiniLM-L6-v2.int8n64`,
   `bge-small-en-v1.5.int8`) are hidden-384 and now correctly also exclude
   the bfp16 directory from their candidate list — each producing an
   identical note. Not a bug; a side benefit of closing the same hole for
   uncatalogued models too.
3. **`root "."` is relative to the invoking shell's cwd, not the exe's
   location.** The Gemma differential-gate commands from `0074` are written
   to run from the repo root (`.\runtime\build\npuembed.exe . --model ...`);
   running the equivalent from inside `runtime\` with root `..` instead of
   `.` first failed with "no models/*.npue under .". Fixed by matching
   0074's exact cwd convention rather than adapting it.
4. **`Remove-Item` on an NTFS junction refuses non-interactively** ("Windows
   PowerShell is in NonInteractive mode") when targeted directly; removing
   the **parent** directory with `-Recurse -Force` deleted the whole scratch
   tree (junctions included) without hitting the same prompt. Used for
   cleanup; no scratch files or junctions were left in the repo.

## What was not done

- **No repeat runs of the golden gate or objdump** beyond the one pass
  recorded here — each cell is a single, deterministic measurement (the
  NPU path has already been established as bit-reproducible run to run,
  0101 Problems hit #3), and every number that had a prior reference
  reproduced it exactly, which is itself the strongest evidence available
  that these are not one-off readings.
- **No fresh MTEB run.** The task brief's Part 2 step 3 asks only for the
  golden `1-cos` gate; MTEB was already run for all six models in
  `0101`/`0103` on this exact toolchain and datapath and is not re-run
  here. The golden-gate numbers reproducing 0101/0103's own golden-gate
  numbers bit-exactly is corroborating evidence the artifact sets built
  today are the same computation as the ones MTEB-gated in those tasks.
- **`tools/make_release.ps1` was not run end to end** (needs a fresh
  benchmark sweep and a full release cut, out of this task's scope per the
  brief — "the code changes" and "the rebuilt artifact sets" are the
  deliverables, not a cut release). Syntax-checked only.
- **`research/OPEN-THREADS.md` / `CLOSED-THREADS.md` / `tasks/README.md`
  were not edited**, per the brief — proposed text is below.

## Artifacts

All in this directory:

- `export_{minilm,base,large,nomic,gemma}_bfp16.log` — the five build logs
  (identity checks, cache purges).
- `gate_{minilm,base,large,nomic}_bfp16.txt`, `gate_bge_small_bf16.txt` —
  golden-gate output for five of the six models (flag-form, explicit
  `--artifacts`).
- `gate_gemma_cpu_control.txt`, `gate_gemma_npu_bfp16.txt`,
  `gate_gemma_diff_bfp16.txt`, `out_gemma_{cpu,npu}.f32` — Gemma's
  differential golden check.
- `objdump_{minilm,base,large,nomic,gemma}_bfp16_matmul.txt`,
  `objdump_{minilm,base,large,gemma}_bfp16_narrow.txt` — AIE objdump,
  every one of the five builds.
- `known_bad_bge_small_refuses.txt`, `known_bad_embed_refuses.txt`,
  `known_good_minilm_embed.txt` — Part 1's known-bad proof (scratch-root
  `list` and `embed` runs).
- `e2e_embed_bge_base_auto.txt`, `e2e_embed_nomic_auto.txt` — end-to-end
  automatic-selection confirmation (no `--artifacts`).
- `tiny_input.txt` — the one-line input text used throughout.

Artifact sets built this session (in `runtime/`, not checked in):
`artifacts_minilm_bfp16`, `artifacts_base_bfp16`, `artifacts_large_bfp16`,
`artifacts_nomic_bfp16`, `artifacts_gemma_bfp16`.

## Proposed register update

**T23 can close.** Restated verdict, now shipped rather than only
measured: bfp16 + bf16-C is adopted on five of six models
(`all-MiniLM-L6-v2`, `bge-base-en-v1.5`, `bge-large-en-v1.5`,
`nomic-embed-text-v1.5`, `embeddinggemma-300m`), each with its own new
production artifact set (`artifacts_{minilm,base,large,nomic,gemma}_bfp16`),
every one objdump-verified (`vconv.bfp16ebs8.fp32`×20, `crrnd`×3 in both
the matmul and the narrowing epilogue) and golden-gate-verified on
hardware, reproducing 0101/0103's own numbers bit-exactly where a prior
number existed (MiniLM, bge-base, bge-large) and passing cleanly where it
did not (nomic 1.402e-03 — the thinnest margin of the five, still ~1.4x
inside tolerance; gemma 2.315e-04 via the differential harness).
`bge-small-en-v1.5` stays on its original plain-bf16 `artifacts_b128il`,
unchanged and reproducing its recorded baseline exactly (3.789e-03).

Because adopting per model creates a real fail-open — geometry and even
`b_layout_hash` cannot tell a bfp16 design from a plain-bf16 one at the
same width, and `bge-small` shares MiniLM's hidden-384 geometry with the
one model that DID clear the gate — `design.json` now records
`emulate_bfp16`, the runtime's catalogue records each model's adopted
`datapath`, and `design_fits()`/`pick_artifacts()` refuse a mismatch,
verified against a known-bad artifact (a scratch root holding *only* a
real bfp16 design at bge-small's geometry: bge-small refuses with a named
error, MiniLM succeeds against the identical directory). An explicit
`--artifacts` still overrides, same precedent as int8's mismatched-pairing
case. The runtime's status line now states which datapath it selected,
read from the loaded design, not from a flag.

`tasks/README.md` index row:

```
| [0104](0104-adopt-bfp16-per-model/TASK.md) | **bfp16 adopted per model, and the fail-open adoption creates is closed first.** Geometry (and even `b_layout_hash`, tasks/0080's guard) cannot tell a bfp16 design from a plain-bf16 one at the same width -- confirmed by building one of each at hidden 384 and diffing `design.json`, byte-identical `b_layout_hash`. `design.json` now records `emulate_bfp16`, the runtime catalogue records each model's adopted datapath (`CatalogEntry::datapath`), and `design_fits()`/`pick_artifacts()` refuse a mismatch -- verified against a KNOWN-BAD artifact (a scratch root holding only a real bfp16 design at bge-small's geometry: bge-small refuses, exit 2, named error; MiniLM succeeds against the identical directory and its status line now prints which datapath it selected). Built and hardware-verified all five adopted production sets (`artifacts_{minilm,base,large,nomic,gemma}_bfp16`) -- objdump-confirmed genuinely emulated (20x `vconv.bfp16ebs8.fp32`, 3x `crrnd`, every build) and golden-gate PASS on all six models, reproducing 0101/0103's own numbers BIT-EXACTLY wherever a prior number existed (MiniLM 2.430e-02/3.406e-04, bge-base 1.941e-02/2.284e-04, bge-large 1.973e-02/2.626e-04, bge-small unchanged 3.789e-03/8.348e-06) and passing cleanly on the two never previously golden-gated (nomic 1.402e-03 -- the thinnest margin of the five, still ~1.4x inside; gemma differential 2.315e-04). MiniLM gets a new directory (`artifacts_minilm_bfp16`); bge-small keeps its original `artifacts_b128il` untouched, since it is the one model that must not regress. `tools/make_release.ps1`'s default artifact list updated to match; EmbeddingGemma stays outside it, a pre-existing gap (no `--serve` path yet) this task did not create. T23 closes | M13 (T23) | done |
```

Suggested `CURRENT_STATUS.md` change (§ describing the production
architecture / the datapath): the bf16-in/fp32-out datapath contract
sentence should note that five of six shipped models now run
bfp16-emulated MMAC + bf16-C by default (`bge-small-en-v1.5` is the
exception, on plain bf16), each design set self-declaring its datapath in
`design.json` and the runtime refusing a mismatch against the catalogue's
per-model record — mirroring how the int8 paragraph already reads, so a
reader sees both quantised-datapath decisions described the same way.

---

## Coordinator review, appended 2026-08-23

Three things the review changed. The guard, the rebuilds and the known-bad
proof all stood; these are additions on top.

**1. The catalogue opt-in was inverted, and that is a fail-open.** `hub.cpp`
set the datapath by *exclusion* — `if (e.name != "bge-small-en-v1.5")
e.datapath = "bfp16";` — so a **seventh built-in row added later would inherit
bfp16 without anyone having gated it**. The reasoning for setting it by name
rather than in the positional initialiser is sound and is kept; the polarity is
not. Replaced with an explicit allowlist of the five adopted names, each
carrying its measured MTEB delta as a comment, so a new model defaults to plain
bf16 until someone measures it. (User-added models via `npuembeddings add` were
already safe — they take `CatalogEntry::datapath`'s default.)

**2. Absent is not false.** `DesignInfo::emulate_bfp16` defaulted to false when
the key was missing, so a design that predates the field was *reported* as
plain bf16 rather than as unknown. Added `datapath_recorded`, and the status
line now prints `UNRECORDED (design predates tasks/0104)` instead of making a
claim it cannot support. Selection still treats absent as bf16, deliberately —
refusing would break every set exported before today.

This was not hypothetical: `bge-small` selected `artifacts_b128il`, whose
`gemm_rtp/design.json` has no such key, and reported `UNRECORDED`. Note the
older per-op exporter `tools/export_xclbin.py` has always written the field
(line 441), so only the unified `gemm_rtp` sets were ever affected.

**3. The tie-break was alphabetical; now it is evidential.** Two sets that fit
equally were separated by `std::sort` order. `pick_artifacts()` now
`stable_partition`s sets that record `emulate_bfp16` ahead of those that do
not — still stable, so ordering stays reproducible and never depends on mtime
(trap 7c). And `runtime/artifacts_small_bf16` was exported so bge-small has a
plain-bf16 design that can state so: same geometry as `artifacts_b128il`, same
numbers (`rel_fro` **3.789e-03**, worst `1-cos` **8.348e-06**, matching
`docs/CURRENT_STATUS.md`), and it is what `make_release.ps1` now ships.

### Verified after all three changes

```
all-MiniLM-L6-v2         bfp16-emulated MMAC, C as bf16
bge-small-en-v1.5        bf16 MMAC, C as fp32
bge-base-en-v1.5         bfp16-emulated MMAC, C as bf16
bge-large-en-v1.5        bfp16-emulated MMAC, C as bf16
nomic-embed-text-v1.5    bfp16-emulated MMAC, C as bf16
```

### One false alarm, recorded so nobody re-runs it

The review first read the default `--model X` full-encode path reporting
`bf16` at `1-cos` 3.397e-04 as the status line lying. It is not. That path
defaults to `runtime/artifacts` (main.cpp's `art_name = "artifacts"`), a legacy
M5/M7 set that runs **eltwise on the array** — and 3.4e-04 is exactly that
architecture's historical accuracy, the number `docs/CURRENT_STATUS.md` records
improving to 1.086e-05 when eltwise moved to the host. The design is plain
bf16, it says so, and it is right. Chasing this cost several rounds; the
signature to remember is that **3.4e-04 on MiniLM means NPU eltwise, not
bfp16.**
