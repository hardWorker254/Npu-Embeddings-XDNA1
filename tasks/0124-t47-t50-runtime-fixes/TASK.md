# 0124 — One rebuild, two refusals: `--probe-streams` reads the design it loaded (T47), and `--cpu`/no-`--artifacts` stop failing open (T50)

**Date**: 2026-08-27
**Goal**: Phase 1 of the 0.5.0 plan. T47's byte-accounting fix and T50's two
CLI fixes share one runtime rebuild, so they are one task. The rule applied
three times here is the same one: **report the value you read, never the
intention — and refuse rather than guess.**

## What was done

### 1. T47 — `--probe-streams` reads A/B element size and the N-tiling from the loaded design

The probe hardcoded `2` as the A/B element size and `48.0 * 8.0` as
`tile_n * cols`. Wrong under int8 (1 byte), wrong on bge-large at either
datapath (tile_n 32 at bf16, 64 at int8 — per `0081`), and the inflation was
**differential** because C was counted correctly: `0097` t21's MiniLM int8
table printed 56.5 / 35.9 / 57.4 / 58.5 GB/s where the correct accounting
gives 35.9 / 22.8 / 36.8 / 31.7 (1.57–1.85×).

* `runtime/include/npu_device.hpp` — `DesignInfo` gains `tile_n` and `cols`,
  0 meaning "export predates the fields".
* `runtime/src/npu_device.cpp` — parsed from design.json. `"tile_n"` is
  `b_layout`'s key and `"cols"` is top-level; both are unique exact-key
  matches in every design.json this project's exporters write, which is what
  the naive string reader requires (checked against the shipped sets before
  relying on it).
* `runtime/src/main.cpp` (probe block) — `ab = info().a_elem_bytes`,
  `tile_n`/`cols` from the design, and a **refusal** when they read 0:
  a pre-field design.json gets *"re-export the set; refusing to substitute a
  guess (T47)"*, not 48×8. The banner now states A/B dtype and the tiling it
  used, so the table carries its own provenance.

### 2. T50, fail-open half — `--cpu` is refused where it was ignored

`force_cpu` was parsed in `run_gemma_mode()` only; on the whole BERT family
(arch 0 and 2, both CLI forms) the flag was accepted and the NPU ran anyway,
with correct vectors — which is what made it invisible. There is no host BERT
encoder in this build to route it to (`npue::GemmaEncoder` is arch-1's
host-only control and reads raw `q_proj`-style F32 tensors that only a
host-only container carries), so the honest fix is the refusal:

* the flag path throws
  `--cpu: no host encoder exists for this architecture in this build ...`
  **after** the early arch-1 dispatch (so gemma still honours it),
* and the `embed`/`serve` subcommand rewrite now **forwards** `--cpu` into
  the flag form — it used to drop the flag entirely, so the refusal would
  never have fired on the subcommand form.

### 3. T50, loud half — the flag form's no-`--artifacts` default resolves like the subcommand does

`art_name = "artifacts"` pointed the legacy form at the per-op design set
predating the unified xclbin. Width 384, so MiniLM and bge-small ran; every
wider model died on a staged-buffer size check or a `b_layout_hash` refusal —
the guards *working*, but naming the wrong problem. Now: with no
`--artifacts`, resolution is **deferred until the container is loaded** (the
first use of `art` is well after model load — checked, line ~5699 vs ~5614)
and goes through the same `pick_artifacts(root, hidden, intermediate,
gated_ffn, qkv_n, layout, want_datapath)` call the subcommand path makes,
catalogue datapath included. No match is a refusal that says what to do;
the picked set is printed so the choice is never silent. An explicit
`--artifacts` behaves exactly as before.

## Commands run

```powershell
cd runtime
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL && cmake --build build --config Release'
# [3/3] Linking CXX executable npuembed.exe   (npuembed.exe + npuembeddings.exe, both restamped)
```

Verification, in order:

```powershell
# T47 mechanism -- int8 (ab=1), tile_n 48: MB now 42.5/14.2/56.6/37.7 (was 66.1/... under ab=2)
.\build\npuembed.exe .. --model all-MiniLM-L6-v2.int8 --artifacts artifacts_int8c_mini --probe-streams
# T47 tile_n=64 read from the design (banner: "tile_n 64 x cols 8")
.\build\npuembed.exe .. --model bge-large-en-v1.5.int8n64 --artifacts artifacts_int8c_large_n64 --probe-streams
# control: bfp16 at 48x8 -- byte accounting unchanged where the constants happened to be right
.\build\npuembed.exe .. --model bge-base-en-v1.5 --artifacts artifacts_base_bfp16 --probe-streams

# T50: both forms refuse --cpu on a BERT model (exit 2, the T50 message)
.\build\npuembed.exe .. --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --embed in.txt out.f32 --cpu
.\build\npuembeddings.exe embed all-MiniLM-L6-v2 in.txt out.f32 --root .. --cpu

# T50: legacy form, no --artifacts, on a WIDE model -- used to die on a
# 3.5 MB-vs-884 KB staged-buffer error; now:
.\build\npuembed.exe .. --model bge-base-en-v1.5 --embed in.txt out.f32
#   artifacts  ..\runtime\artifacts_base_bfp16 (picked from the container's geometry; no --artifacts given)
#   datapath   bfp16-emulated MMAC, C as bf16      <- adopted datapath honoured
# exit 0

# no regression across all three architectures: the semantic gate (0121)
python tools\verify_semantics.py
# PASS -- every model puts each paraphrase pair together (6/6 models, 24/24 each)
```

MB-column hand-checks (static arithmetic, machine-independent):
MiniLM int8 qkv = A 8192·384·1·3 + B 384·1152·1·32 + C 8192·1152·2
= 9.4 + 14.2 + 18.9 = **42.5 MB** ✓; bge-large int8n64 qkv (tile_n 64) =
50.3 + 100.7 + 50.3 = **201.3 MB** ✓; bge-base bfp16 qkv = 75.5 + 113.2 +
37.7 = **226.5 MB** ✓. All three match the fixed binary's printed column.

## Problems hit

* **`--cpu` on the shipped pretiled Gemma container fails with `no tensor
  named layer.0.q_proj` — pre-existing, not from this change.**
  `npue::GemmaEncoder` reads raw F32 projection tensors that only a
  host-only container (`--gemma-host-only`) carries; the shipped container
  is pretiled. The failure is loud and truthful, so it is left alone, but it
  means arch-1's `--cpu` control needs a host-only container to actually
  run — worth knowing before anyone builds a CPU-control harness on it.
* The build shell must be a VS developer environment; a bare shell fails
  with `Cannot open include file: 'algorithm'`. The `cmd /c "vcvars64.bat
  && cmake --build ..."` form above is what worked from this session.

## What this closes

**T50 — closed** (both halves shipped and verified above). **T47 — the fix
half**; the audit of published `GB/s` figures is
[`0125`](../0125-t47-gbs-audit/TASK.md), run immediately after on a quiet
machine, and T47 closes there.
