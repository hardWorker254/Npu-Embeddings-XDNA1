# 0060 — Fix `export_gemm_rtp.py`'s `markers_for()` for the mlir-aie 1.4.x `aie.dma_bd` textual form

- **Date** 2026-08-20
- **Milestone** M11 (toolchain migration cleanup)
- **Status** done

## Goal

Close the one item [`0058`](../0058-m11-iron-1.4-migration/TASK.md) left open under
[T22](../../research/OPEN-THREADS.md): `tools/export_gemm_rtp.py`'s `markers_for()`
cache-marker string no longer matches anything in a freshly built `aie.mlir` on the
upgraded (main, past 1.4.1) mlir-aie toolchain, so the tool cannot rebuild any shipped
production xclbin. Fix the one string, verify the fix on a fully fresh cache, and prove
the fixed tool actually produces a *correct* artifact set — not just one that doesn't
crash — by reproducing a shipped model's known `1-cos` figures on hardware.

## Context

- The diagnosis was already done by `0058`/`0059` and `research/notes/0008-iron-1.4-migration.md`
  before this task started: the mlir-aie 1.4.x MLIR pretty-printer changed the
  sequence-body `aie.dma_bd` op's textual form from bracket-tuples
  (`<size = k, stride = n>`) to a flat `sizes = [...] strides = [...]` array. The old
  bracket-tuple form survives only in `aie.objectfifo`'s `dimensionsToStream`
  attribute — a different op — so `markers_for()`'s `f"<size = {k}, stride = {n}>"`
  substring stopped appearing anywhere in a fresh `aie.mlir`, and every export failed
  with `N cache candidates after purge -- expected exactly 1` (N=0).
  This task does not re-diagnose that; it fixes the one function.
- `research/notes/0008-iron-1.4-migration.md` documents **three separate marker-
  specificity fail-open instances found during the 1.4.x migration alone** (the
  `export_gemm_rtp.py` one being fixed here, plus a `@iron.jit`-internal cache-lookup
  instance found re-verifying `cross_column_join_probe.py`, plus the general lesson
  that a full `rm -rf ~/.npu/cache/*` beats trusting any script's scoped marker match
  when switching configurations within a session) — on top of three earlier instances
  from tasks 0030/0053/0054. The mandate from all six: any marker rewrite must stay
  SPECIFIC (able to distinguish genuinely different builds), not just permissive
  enough to stop erroring.
- Scope, carried over unchanged from `0058`/`0059`: do not touch `runtime/src/`,
  `hub.cpp`, `pack_npue.py`, `npue_pack.cpp`, or any shipped `.npue`/`.xclbin`/
  `runtime/artifacts_*` directory currently used by production.

## What was done

1. **Reproduced the bug** on a fully purged cache, confirming the exact same failure
   `0058`/`0059` recorded.
2. **Found the real new marker substring by inspection, not guessing.** Built the four
   production GEMM shapes (qkv, attn_out, ffn_up, ffn_down) at `m=64 k=64 n=48`,
   `--cols 2`, and grepped each fresh `aie.mlir` for `aie.dma_bd`. In every one of the
   four shapes, B's (`%arg1`) `aie.dma_bd` ends its access pattern with the tile
   dimensions as the **last two entries of `sizes`**, immediately followed by
   `strides` — e.g. `sizes = [12, 6, 64, 48] strides = [6144, 73728, 48, 1]` — always
   exactly twice per build (the ping/pong pair), and this exact substring never
   collides with A's or C's `aie.dma_bd` (whose `sizes` end in different values) or
   with `aie.objectfifo`'s surviving bracket-tuple attributes. This is the direct
   translation of the old `<size=k, stride=n>` marker into the new textual form —
   same two numbers (`k`, `n`), same adjacency requirement.
3. **Rewrote `markers_for()`'s second marker** from `f"<size = {k}, stride = {n}>"`
   to `f"{k}, {n}] strides = ["`, with an extended docstring recording the format
   change, the confirmation method, and why the new substring is still specific
   (ties `k` and `n` together, adjacent, and only ever appears in B's `aie.dma_bd`).
4. **Verified on a fully fresh cache** (`rm -rf ~/.npu/cache/*` first): a small
   batch-4/cols-2 export now finds exactly 1 cache candidate per shape (not 0, not
   many) and completes.
5. **Verified the artifact is actually correct**, not just non-crashing, per
   CLAUDE.md rule 6 ("a number without a traceable artifact is not a result"):
   rebuilt the real MiniLM production GEMM design (`--batch 128 --cols 8`, the exact
   `0032` recipe) on a fully fresh cache into a **separate** directory
   (`runtime/artifacts_verify_t22fix/`, gitignored — `runtime/artifacts*/` is in
   `.gitignore`), copied the shipped `runtime/artifacts_b128il/`'s non-`gemm_rtp`
   parts (weights, manifest, host eltwise designs, validation set — none of which
   `export_gemm_rtp.py` produces or this task touches) alongside the freshly built
   `gemm_rtp/`, and ran both the C++ runtime's own validation encode and
   `tools/verify_embed_e2e.py` (tokenizer included) against it. Both reproduced the
   historical MiniLM figures from `0038`/`0059` exactly. `runtime/artifacts_b128il/`
   itself was never written to.
6. Checked the NPU contention guard (`xrt-smi examine -r all`) before and around the
   hardware runs — every hw context showed `Idle`, no foreign `Active` context.

## Commands

```powershell
# env, once per PowerShell call (state does not persist across tool calls)
cd C:\dev\mlir-aie; . .\iron_env.ps1

# contention guard
& "C:\Windows\System32\AMD\xrt-smi.exe" examine -r all
```
```bash
# full purge before reproducing the bug
rm -rf ~/.npu/cache/*
```
```powershell
# reproduce (fails before the fix)
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\tools
python export_gemm_rtp.py --batch 4 --cols 2 --out "$env:TEMP\rtp_test_scratch2"
# -> qkv@b4: 0 cache candidates after purge -- expected exactly 1
```
```bash
# inspect the fresh cache dir's real aie.mlir to find the new dma_bd form
grep -n "aie.dma_bd" ~/.npu/cache/<hash>/aie.mlir
grep -n "<size = " ~/.npu/cache/<hash>/aie.mlir   # confirms old form survives ONLY in objectfifo attrs
grep -c "64, 48]" ~/.npu/cache/<hash>/aie.mlir    # confirms new marker candidate is exactly 2 hits, only in B's dma_bd
```
```powershell
# fix applied to tools/export_gemm_rtp.py (markers_for), then verify on a
# FULLY fresh cache
```
```bash
rm -rf ~/.npu/cache/*
```
```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\tools
python export_gemm_rtp.py --batch 4 --cols 2 --out "$env:TEMP\rtp_test_scratch3"
# -> succeeds, exactly 1 candidate per shape, identity checks all "OK"
```
```bash
# full purge again before the production-shape verification build
rm -rf ~/.npu/cache/*
```
```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python tools\export_gemm_rtp.py --batch 128 --cols 8 --out runtime\artifacts_verify_t22fix
```
```powershell
# assemble a runnable artifact set: shipped weights/manifest/eltwise +
# freshly built gemm_rtp, in the SEPARATE verify directory only
$src = "C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime\artifacts_b128il"
$dst = "C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime\artifacts_verify_t22fix"
Get-ChildItem $src | Where-Object { $_.Name -ne "gemm_rtp" } | ForEach-Object {
    if ($_.PSIsContainer) { Copy-Item $_.FullName -Destination (Join-Path $dst $_.Name) -Recurse -Force }
    else { Copy-Item $_.FullName -Destination (Join-Path $dst $_.Name) -Force }
}
```
```powershell
# contention guard, then the two correctness checks
& "C:\Windows\System32\AMD\xrt-smi.exe" examine -r all

cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime
.\build\npuembed.exe .. --model all-MiniLM-L6-v2 --artifacts artifacts_verify_t22fix --threads 16

cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
& ".\.venv-ref\Scripts\python.exe" tools\verify_embed_e2e.py --model all-MiniLM-L6-v2 --artifacts artifacts_verify_t22fix --threads 16
```

## Result

**The fix.** `tools/export_gemm_rtp.py`, `markers_for()`:

```python
# before (matched nothing after the toolchain upgrade):
f"<size = {k}, stride = {n}>",
# after:
f"{k}, {n}] strides = [",
```

Confirmed the new substring's specificity directly: across all four production
shapes (qkv, attn_out, ffn_up, ffn_down) at `m=64 k=64 n=48`, `sizes = [.., .., 64,
48] strides = [..]` appears **exactly twice** per `aie.mlir` (the ping/pong pair),
always inside B's (`%arg1`) `aie.dma_bd`, never in A's or C's.

**Fresh-export verification, small case** (`--batch 4 --cols 2`, fully purged
cache first):

```
  b4    qkv           [256, 384, 1152] -> 1e9d31869456d0e9ed9862b4
  b4    attn_out       [256, 384, 384] -> c7b935ad0a78b6f78342b29f
  b4    ffn_up        [256, 384, 1536] -> 181116c7b9893c81031768fc
  b4    ffn_down      [256, 1536, 384] -> 1be81801d0b40188d0d1e47f
  identity qkv@b4 vs attn_out@b4    73 differing bytes  OK
  identity qkv@b4 vs ffn_up@b4    72 differing bytes  OK
  identity qkv@b4 vs ffn_down@b4    73 differing bytes  OK
  wrote ...\gemm_rtp -- ONE xclbin, 4 streams (4 shapes x 1 batch tiers)
```

Exactly 1 cache candidate found per shape (was 0 before the fix); all three
identity checks pass under the ≤80-differing-byte (UUID-only) threshold from `0029`.

**Fresh-export verification, real production shape** (`--batch 128 --cols 8`, the
exact `0032` recipe, fully purged cache first, separate output dir
`runtime/artifacts_verify_t22fix/`):

```
  b128  qkv          [8192, 384, 1152] -> db596aac85d99799e0e46274
  b128  attn_out      [8192, 384, 384] -> e5d9c53e25333562a7dc98cd
  b128  ffn_up       [8192, 384, 1536] -> 1393cbd3f55daf2727d946cb
  b128  ffn_down     [8192, 1536, 384] -> d8b5038296f86e12c70c9504
  identity qkv@b128 vs attn_out@b128  66 differing bytes  OK
  identity qkv@b128 vs ffn_up@b128  72 differing bytes  OK
  identity qkv@b128 vs ffn_down@b128  72 differing bytes  OK
  wrote runtime\artifacts_verify_t22fix\gemm_rtp -- ONE xclbin, 4 streams (4 shapes x 1 batch tiers)
```

`final.xclbin` is 122,334 bytes here against the shipped `artifacts_b128il`'s
125,791 — a real size difference, but expected and already-documented: `0058`
recorded the same thing for `build_passthrough.py`'s xclbin ("the total xclbin size
differs only because the two measurements are from different toolchain states").
Not a correctness signal by itself — the numbers below are.

**Correctness on hardware, against the exact historical figures** (`0038`, `0059`):

| check | fresh (`artifacts_verify_t22fix`) | historical | source | match |
|---|---:|---:|---|---|
| golden-vs-`.npue` validation encode, `1-cos` | **1.086e-05** | 1.086e-05 | `tasks/0038`, `tasks/0059` | **identical** |
| `verify_embed_e2e.py` worst `1-cos` (tokenizer + NPU + pooling) | **2.644e-05** | 2.6436e-05 (`0036`) / 2.644e-05 (`0059`) | `tasks/0036`, `tasks/0059` | **match to 4 s.f.** |
| `verify_embed_e2e.py` top-10 neighbour overlap | 1.0000 | 1.0000 | `tasks/0059` | identical |

Both runs printed `PASS`. Full runtime output:

```
NpuEmbeddings C++ runtime -- full encode
  bo-mode    host_only (data-buffer allocation)
  model      all-MiniLM-L6-v2: 78 tensors, 69.00 MB, checkpoint 53aa51172d142c89
  shape      sentence-transformers/all-MiniLM-L6-v2: 6 layers, hidden 384, 12 heads x 32, ffn 1536, mean pooling
  designs    ONE xclbin, 4 streams (1 batch tiers), one hw_context
  shape      batch 128 x seq 64  (M = 8192)
  tiers      128  (requests are right-sized, not padded)
  gelu       on the HOST (fp32) -- 6 fewer NPU dispatches
  softmax    on the HOST (fp32) -- 6 fewer NPU dispatches
  layernorm  on the HOST (fp32) -- 13 fewer NPU dispatches
  bo-align   last data buffer aligned to 131072 B
  weights    21.23 MB staged on the device once, not per call

  embedding rel_fro vs HF golden           4.473e-03
  worst 1 - cos vs HuggingFace             1.086e-05

PASS -- tolerance 2e-03 on 1-cos, no Python in this process
```

```
end-to-end: 13 texts, seq 64, artifacts artifacts_verify_t22fix
    #        1-cos   text
   10    2.644e-05   '   '
    9    2.644e-05   ''
   11    1.682e-05   'Short.'
    8    1.328e-05   'pneumonoultramicroscopicsilicovolcanoconiosis is a long '
    0    1.324e-05   'A man is playing a guitar on stage.'
   12    1.298e-05   'The quick brown fox jumps over the lazy dog while the ca'
    6    1.289e-05   '机器学习模型可以生成句子向量。'
    1    1.229e-05   'Someone plays a guitar at a concert.'

  worst 1-cos 2.644e-05   (tolerance 2e-03)
  pairwise-similarity error over 78 pairs:
    mean 3.493e-04   p99 1.037e-03   max 1.137e-03
    bound implied by the worst 1-cos: 1.454e-02
  top-10 neighbour overlap: 1.0000

PASS -- text in, vector out, matches the reference
```

**Contention guard**, checked before the hardware runs: `xrt-smi examine -r all`
showed every hw context `Status: Idle` (the resident `WorkloadsSessionHost.exe`
contexts, not our process) — no foreign `Active` context, clear to proceed.

**Scope held.** `runtime/artifacts_b128il/` was read from (copied out of, never
into) and never modified. Nothing under `runtime/src/`, `hub.cpp`, `pack_npue.py`,
`npue_pack.cpp`, or any other shipped `.npue`/`.xclbin` was touched. The only
production-tree file changed is `tools/export_gemm_rtp.py` itself.

## Problems hit

1. **Confirming the new marker's uniqueness needed building all four shapes, not
   just the one that failed first.** `export_gemm_rtp.py` raises on the first
   `find_cache()` miss (qkv), so only qkv's `aie.mlir` was available from a single
   run of the broken tool. Built `attn_out`/`ffn_up`/`ffn_down` directly via a
   throwaway script that called `pretiled_array()` the same way the exporter does,
   to get all four `aie.mlir` files to compare — confirming the candidate marker's
   last-two-`sizes` pattern holds across shapes with different `M`/`K`/`N`, not just
   the one that happened to be built first. Deleted after use, not part of the fix.
2. **None on the fix itself** — the reproduction, fix, small-case verification, and
   production-shape verification all worked on the first attempt once the marker
   substring was confirmed by direct inspection rather than guessed.

## Artifacts

- `tools/export_gemm_rtp.py` — the fix (`markers_for()`), plus an extended
  docstring recording the format change and the confirmation method, for the next
  person who hits this class of bug.
- `runtime/artifacts_verify_t22fix/` — the fresh production-shape verification
  build (gitignored, `runtime/artifacts*/` in `.gitignore`; not shipped, not used
  by any script other than this task's verification commands).
- `tasks/0036-m8-tokenizer/verify_embed_e2e_all-MiniLM-L6-v2_artifacts_verify_t22fix.json`
  — `verify_embed_e2e.py`'s standard output location (unchanged tool behaviour),
  the traceable artifact behind the e2e numbers above.
- `research/OPEN-THREADS.md` T22 — updated: the tool-side regression is now fixed
  and verified, closing the one remaining open item.

## Next

- `runtime/artifacts_b128il/` (and the other shipped `runtime/artifacts_*`
  directories) were **not** regenerated on the new toolchain — this task proved the
  tool *can* now rebuild a correct equivalent, deliberately without overwriting the
  validated production set. Regenerating production artifacts on the upgraded
  toolchain, if ever wanted, is a separate deliberate decision, not a side effect of
  this fix.
- No further action item under T22; it closes here.
