# 0090 — T5: does DMA compression pay on real weights? No — the shim has no compression hardware at all

- **Date** 2026-08-23
- **Milestone** none (open-thread closure, per `research/OPEN-THREADS.md` rule 3)
- **Status** done

## Goal

Close [T5](../../research/OPEN-THREADS.md#t5) — *"Does DMA compression pay on real
weights?"* — with a quantitative answer, not an argument. [note
0007](../../research/notes/0007-unused-iron-surface.md) §2 measured the mlir-aie
`dma_compression` silicon probe at **1.39x on `arange`** and suspected (not
measured) that dense weight bytes would not compress the same way. Since
[`0080`](../0080-m13-int8-traffic-bound/TASK.md) put int8 in the traffic-bound
regime (R²=0.987) with B at 23–43% of dispatch traffic, the byte lever is worth
more than when T5 was filed, and the brief asked for: the control ratio, the
measured ratio on real weight bytes (or an honest statement of why it could not
be measured), the end-to-end arithmetic against 0080's own numbers, and a check
of whether compression is even on a DMA path our design uses.

## Context

- `C:\dev\mlir-aie\programming_examples\basic\dma_compression\` — the upstream
  probe. `README.md`, `dma_compression.py`, `kernel.cc`,
  `test_dma_compression.py` read in full before anything else. Key facts it
  documents: compression needs two register writes (per-BD `Enable_Compression`
  bit + per-channel `(De)compression_Enable` bit), the **consumer shim BD length
  must match the compressed byte count or the DMA stalls**, and the only ratio
  it ships is 1.39x on `arange` (an int32 sequence with mostly-zero high bytes).
- `models/bge-base-en-v1.5.npue` (bf16) and `models/bge-base-en-v1.5.int8.npue`
  (int8) — real production weight containers, read via `tools/npue.py`.
- `tasks/0080-m13-int8-traffic-bound/TASK.md` §2 (traffic table) and §1c (the
  fitted cost model `t ≈ 627 µs + traffic/28 GB/s`, R²=0.987) — the numbers
  the end-to-end arithmetic below is built from.
- **Read-only tree rule (CLAUDE.md):** nothing under `C:\dev\mlir-aie` was
  modified. The real-data probe imports the upstream `dma_compression` module
  unchanged and monkeypatches its own copy's module attribute from outside.

## What was done

### 1. Control: reproduce the published ratio, unmodified

Ran the upstream example's `cmp_only` config exactly as shipped, arange input,
on this machine's actual npu2 (Strix, AIE2P).

**Result: PASS, matches=1024 mismatches=1920 untouched=1152,
`compressed_to=71.9%(=1.391x)`, `sha-ok`** against the example's own per-arch
golden sha256. This reproduces the 1.39x note 0007 cited, on our hardware, byte
for byte. Control established.

### 2. Real weight bytes, and a first cheap signal before touching hardware

Extracted the first 16,384 raw on-disk bytes (`Reader.raw()` — the exact byte
order the DMA reads, pre-tiled `block_panel` for bf16, plain for int8, **not**
the de-tiled logical tensor) of `layer.0.{qkv,attn_out,ffn_up,ffn_down}` from
both containers — 8 tiles, 4 shapes x 2 dtypes.

Before running anything on the array, the **zero-byte fraction** of each tile
was measured, since AIE-ML's DMA compression is understood (note 0007) to be a
zero/sparsity scheme and this is the cheapest possible proxy for "is there
anything in this data for that class of compressor to find":

| input | zero-byte fraction |
|---|---:|
| `arange(4096)` (the control's own input) | **51.7%** |
| bf16 `qkv` / `attn_out` / `ffn_up` / `ffn_down` | 0.28% / 0.31% / 0.23% / 0.28% |
| int8 `qkv` / `attn_out` / `ffn_up` / `ffn_down` | 0.99% / 2.31% / 1.21% / 3.13% |

Real weight bytes carry **17x to 220x less** of the raw material (zero bytes)
that this compressor class exploits than the sequence the published 1.39x was
measured on. This does not by itself prove the achieved ratio is worse — it is
a necessary-condition check, not the measurement — but it is a strong prior in
the direction note 0007 guessed.

### 3. The structural blocker — checked directly in the aie-rt source, not assumed

The brief asked to confirm or refute, from source, whether compression sits on
a DMA path our design does not use. It does, and the answer is sharper than
"unused by mlir-aie's own passes" (note 0007's phrasing, which sounds like a
compiler gap): it is **absent from the silicon itself, on our exact chip.**

`third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c` (AIE2P — Strix /
Strix Halo / Krackan / npu2, confirmed by filename and by the README's own
device list) defines one `XAie_DmaMod` struct per DMA module type, each with a
`.Compression` field:

```
1658: static const  XAie_DmaMod Aie2PMemTileDmaMod =
1666:   .Compression = XAIE_FEATURE_AVAILABLE,
1896: static const  XAie_DmaMod Aie2PTileDmaMod =      // compute-tile core DMA
1904:   .Compression = XAIE_FEATURE_AVAILABLE,
2149: static const  XAie_DmaMod Aie2PShimDmaMod =
2157:   .Compression = XAIE_FEATURE_UNAVAILABLE,
```
(full grep saved at `artifacts/aie2p_shim_no_compression.txt`)

**Mem-tile and compute-tile DMA have the compression datapath. Shim DMA does
not — at all, on this chip.** This is a hardware capability flag read by
`XAie_DmaEnableCompression()` (`xaie_dma.c:545`, `if (DmaMod->Compression ==
XAIE_FEATURE_UNAVAILABLE) return XAIE_FEATURE_NOT_SUPPORTED;`), not a
mlir-aie/IRON omission — no compiler change could add it.

**Why this closes the question regardless of the real-data ratio.** Every byte
this project's cost model prices as "traffic" ([`0080`](../0080-m13-int8-traffic-bound/TASK.md)'s
`t ≈ 627 µs + traffic/28 GB/s`, R²=0.987) crosses the shim — that is the only
door from DRAM into the NPU fabric, and it is where B, A and C all cross on
every dispatch of the production design (DDR -> shim -> mem tile -> core
tile). DMA compression can only ever touch the mem-tile-to-core-tile leg
(`Aie2PMemTileDmaMod`/`Aie2PTileDmaMod`), which is on-die, not DRAM-bound, and
not the leg 0080's traffic-bound model is pricing. A perfect compression ratio
on B applied only inside the array would not move a single byte off the leg
that is actually the bottleneck.

### 4. Attempting a real-data hardware ratio anyway — and a genuine caching bug found along the way

Given #3, an exact real-weight ratio would be research trivia rather than a
decision input, but the brief asked for it, so it was attempted with the
constraint the README documents: **the consumer shim BD length must be set
to the actual compressed byte count in advance, or the DMA stalls.** For
`arange` that constant (`RATIOED_N = 2944`) was hand-derived once by the
example's authors; for arbitrary real data it is not known ahead of time, and
guessing wrong in the oversized direction hangs the dispatch
(`ERT_CMD_STATE_TIMEOUT` — a previously-observed-safe failure mode in this
project, `tasks/0054` and `tasks/0087` both record "no stuck hw_context
afterward, `xrt-smi examine -r all` clean").

The one value that is safe to try without knowing the answer is `RATIOED_N =
N` (4096, "assume zero compression, size the destination for the raw input"):
compression cannot emit more bytes than it was given, so this is requesting an
upper bound, not a guess that can overshoot. `probe_real_data.py` does exactly
this: imports the upstream `dma_compression` module unmodified (added to
`sys.path`, never edited on disk) and monkeypatches `dc.RATIOED_N = N` before
calling `dc.dma_compression(in_tensor, out_tensor, config="cmp_only")` with a
real weight tile as input.

**Result was NOT what the monkeypatch should have produced.** Every run —
including a diagnostic with `RATIOED_N = 100` and `use_cache=False` fed
`arange` input, deliberately extreme so a real effect would be unmissable —
returned the *exact* bucket signature as the arange control:
`matches=1024 mismatches=1920 untouched=1152`. If `RATIOED_N` had actually
reached the compiled shim BD, `untouched` would be `N - RATIOED_N` (0 for
`RATIOED_N=4096`, 3996 for `RATIOED_N=100`); it was 1152 = `4096 - 2944` every
time, i.e. **every dispatch actually ran the design compiled for the
control's `RATIOED_N=2944`, never the requested value.**

Isolated before concluding it was a hardware fact rather than a harness bug:
`dc.dma_compression.as_mlir(config="cmp_only")` (bypasses the JIT compile/cache
path entirely per the example's own README) **does** show the requested value
in the generated MLIR text (`contains 2944: False`, `contains 100: True` for
`RATIOED_N=100`) — so the generator itself is correctly parameterised by the
module global. The staleness is downstream of generation, in `iron.jit`'s own
kernel cache: `aie/utils/compile/cache/utils.py::_create_function_cache_key()`
hashes only the call's positional/keyword arguments (tensor shapes/dtypes,
`config="cmp_only"`) plus `CompilableDesign._generation_cache_key()`, which is
`(device, full_elf)` — **neither hashes anything the generator function reads
from its own module's global namespace.** `RATIOED_N` is exactly that: a plain
module-level name the generator's `_linear_tap(RATIOED_N)` call reads via
`LOAD_GLOBAL`, invisible to both the in-process kernel cache and (per the
`use_cache=False` control, which still reproduced the stale result) the
on-disk artifact cache keyed the same way.

**This is a sixth instance of the "silent stale cache" class this project has
hit before** (0053, 0054, 0058's JIT-lookup-layer instance, 0058's
`export_gemm_rtp.py` marker, 0080's `design_fits()`), and a new mechanism
within it: not a marker that is too coarse to notice a *textual* MLIR change,
but a cache key that structurally cannot see a Python **global** the generator
depends on, because nothing in the key derivation inspects the generator's
`__globals__`. Recorded here, not fixed — `dma_compression.py` is upstream
and read-only, and the fix (whatever form it takes upstream) is out of scope
for a thread-closure task.

**Consequence for the measurement:** the real-weight dispatch that did run
(`artifacts/probe_bf16_qkv.log`) completed in 33 ms with no hang, at the
*control's* compiled BD size (2944 ints), not the requested one. That is
weak-but-real information — real bf16 weight data pushed at production speed
through this design does not desynchronize the DMA state machine or hang at
that particular sizing — but it is **not** a ratio measurement, and per #3 it
would not have been a production-relevant one even if the cache bug had not
intervened (compute-tile path, not shim). Given that, further effort to defeat
the cache (e.g. by finding and clearing the on-disk artifact directory between
runs) was judged not worth the additional hardware-timeout risk on shared,
concurrently-used equipment for a number that cannot change the answer to T5.
**The exact compression ratio on real weight bytes remains unmeasured.**

## Commands

```powershell
# environment, every shell
cd C:\dev\mlir-aie
. .\iron_env.ps1

# 1. control -- unmodified upstream example, arange input
cd programming_examples\basic\dma_compression
python test_dma_compression.py cmp_only -v
#  -> PASS matches=1024 mismatches=1920 untouched=1152 compressed_to=71.9%(=1.391x) sha-ok (252 ms)

# 2. extract real weight tiles (no NPU needed)
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
C:\Users\vegar\.conda\envs\iron\python.exe tasks\0090-t5-dma-compression\extract_tiles.py

# 3. real-data probe (needs iron_env active, run from step 1's shell)
python "C:\Users\vegar\Documents\GitHub\NpuEmbeddings\tasks\0090-t5-dma-compression\probe_real_data.py" `
    "C:\Users\vegar\Documents\GitHub\NpuEmbeddings\tasks\0090-t5-dma-compression\artifacts\tiles\bf16_qkv.bin"

# 4. cache-invisibility diagnostic (arange input, RATIOED_N=100, use_cache=False)
python -c "<inline, see artifacts/cache_busting_diagnostic.log for the exact script run>"

# 5. structural check -- grep the vendor driver source directly, no NPU
cd C:\dev\mlir-aie
grep -n "Compression = XAIE\|/\* .* Dma Module \*/\|Aie2P.*DmaMod =" `
    third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c
```

## Result

| check | result | artifact |
|---|---|---|
| control (`cmp_only`, arange) | **PASS, 1.391x, sha-ok** | `artifacts/control_cmp_only.log` |
| real weight zero-byte fraction | 0.23–3.13% vs arange's 51.7% (17–220x less exploitable) | `artifacts/tiles_manifest.json` |
| shim DMA `.Compression` (AIE2P) | **`XAIE_FEATURE_UNAVAILABLE`** — mem-tile and compute-tile are `AVAILABLE` | `artifacts/aie2p_shim_no_compression.txt` |
| real-data hardware ratio | **not obtained** — every attempt silently reran the control's compiled config (cache-invisibility bug, §4) | `artifacts/probe_bf16_qkv.log`, `artifacts/cache_busting_diagnostic.log` |

### End-to-end arithmetic (best case that was ever on the table, before the blocker)

Using [`0080`](../0080-m13-int8-traffic-bound/TASK.md)'s own measured int8
(i32-C) dispatch times and traffic split (§2, §6; M=8192, 8 columns), applying
the *control's* 1.391x ratio (28.1% byte reduction) to **B only**, as if it
could somehow reach the traffic-bound leg:

| shape | dispatch µs (measured) | B share | B saved at 1.391x | new µs (linear) | change |
|---|---:|---:|---:|---:|---:|
| qkv | 2611 | 23.2% | 3.99 MB | 2468.5 | −5.5% |
| attn_out | 1738 | 23.0% | 1.32 MB | 1690.9 | −2.7% |
| ffn_down | 2044 | 43.0% | 5.31 MB | 1854.4 | −9.3% |
| ffn_up | 3445 | 23.1% | 5.31 MB | 3255.4 | −5.5% |
| **sum** | **9838** | | | **9269.2** | **−5.8%** |

(Slope from 0080's fit, 1/28 GB/s = 35.71 µs/MB; fixed 627 µs term per
dispatch unaffected since it does not scale with traffic.)

**So even granting the control's ratio to real weights for free — which §2's
zero-byte evidence argues against — the entire lever was worth at most ~5.8%
of the four production GEMM dispatches, before subtracting the decompression
overhead and register-write cost the mechanism itself needs.** And per §3,
none of it is reachable: the shim leg this arithmetic is computed against has
no compression hardware on AIE2P, full stop.

## Problems hit

- **The stall-risk in the README made a naive ratio sweep unsafe.** "Consumer
  shim BD length must match the compressed byte count, or the DMA stalls" —
  worked around by only ever requesting an *upper bound* (`RATIOED_N = N`),
  never a guess that could exceed the true value in the dangerous direction.
- **`iron.jit`'s cache is structurally blind to a Python global the generator
  reads.** Not a fix attempted here — see §4. This blocked getting an exact
  real-data ratio via the safe path above; recorded rather than chased,
  because #3 already answers the decision question the ratio would have fed.
- Did **not** attempt to bypass the cache by manually locating and deleting
  the on-disk artifact directory between runs (unclear which directory,
  keyed by a hash this task did not reverse), to avoid burning further
  shared-hardware time on a number that cannot change the T5 verdict.

## Artifacts

All under `tasks/0090-t5-dma-compression/`:
- `extract_tiles.py`, `probe_real_data.py` — scripts written for this task.
- `artifacts/control_cmp_only.log` — unmodified upstream control run.
- `artifacts/tiles_manifest.json` + `artifacts/tiles/*.bin` — the 8 real
  weight tile extracts (4 shapes x {bf16, int8}), 16,384 bytes each, sha256'd.
- `artifacts/probe_bf16_qkv.log` — the real-data dispatch that did complete
  (cache-stale config, see §4).
- `artifacts/cache_busting_diagnostic.log` — the `RATIOED_N=100,
  use_cache=False` control showing the cache-invisibility bug directly.
- `artifacts/aie2p_shim_no_compression.txt` — the aie-rt register-table grep
  backing the structural blocker.

Not checked in as a design change: nothing under `C:\dev\mlir-aie` was
modified (read-only reference tree, CLAUDE.md).

## Next

T5 is closed (see the proposed register update below). Two loose ends,
neither worth their own thread given #3 makes the ratio moot for production:

1. The `iron.jit` global-invisible-cache bug (§4) is real and would bite the
   next person who parameterises a design generator through a module
   constant rather than a `CompileTime[T]` kwarg. Not filed as a new open
   thread — it is a toolchain observation about `C:\dev\mlir-aie`, which this
   project reads but does not own or patch.
2. If a *future* chip revision or a different DMA leg ever exposes shim
   compression, this task's arithmetic (§ end-to-end) is the template to
   rerun — 5.8% was the ceiling on the leg that mattered, from the control's
   own ratio; real weight bytes would need measuring properly before trusting
   that number for anything real.

## Proposed register update

**1. Verdict:** T5 is **ANSWERED** — "no, and here is why": AIE2P's shim DMA
module has `.Compression = XAIE_FEATURE_UNAVAILABLE` in the vendor's own
register tables (only mem-tile and compute-tile DMA have it), and 100% of the
weight/activation/output traffic `tasks/0080`'s int8 traffic-bound cost model
prices crosses the shim. Compression cannot reach the leg it would need to
reach, regardless of what ratio it achieves on real weight bytes — which
remains genuinely unmeasured (a real-data hardware ratio attempt was blocked
by an `iron.jit` caching bug, not by the stall risk), and a zero-byte-fraction
proxy (0.23–3.13% for real weights vs 51.7% for the control's `arange`) argues
that ratio would have been poor even if it could have been applied.

**2. Replacement text — append verbatim to `research/CLOSED-THREADS.md`**
(matching the file's existing `<a id="tN"></a>` + `### TN — ... · **ANSWERED
YYYY-MM-DD**` convention, keeping the original OPEN text unmodified above the
new material per rule 3b):

```markdown
<a id="t5"></a>
### T5 — Does DMA compression pay on real weights? · **ANSWERED 2026-08-23**

> This is a **byte** lever, and bytes were free on bf16 (T1) and are the whole
> cost on int8 ([`0080`](../0080-m13-int8-traffic-bound/TASK.md)). B is
> 23–43% of an int8 dispatch's traffic. Worth more than when it was filed, and
> it is still one run.
[note 0007](../../research/notes/0007-unused-iron-surface.md) §2. Lossless, unused by
mlir-aie's own passes for the compute-tile path, and the only published ratio is
**1.39× on `arange`**. One run of `basic/dma_compression`'s `cmp_only` with a
real `.npue` tile answers it permanently.

**No — structurally, not just empirically.** [`0090`](TASK.md)
reproduced the control (1.391x on `arange`, sha-matched against the example's
own golden) and then read the vendor's own AIE2P register tables
(`third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c`): mem-tile and
compute-tile DMA modules carry `.Compression = XAIE_FEATURE_AVAILABLE`; the
**shim DMA module carries `XAIE_FEATURE_UNAVAILABLE`** — on this exact chip,
not as a mlir-aie/IRON omission but as a hardware capability flag with no
software workaround. Every byte `tasks/0080`'s int8 traffic-bound cost model
prices (`t ≈ 627 µs + traffic/28 GB/s`, R²=0.987) crosses the shim — the only
door from DRAM into the NPU fabric — so DMA compression can only ever touch
the mem-tile-to-core-tile leg, which is on-die and not the modelled
bottleneck. A perfect ratio on B applied only inside the array would not move
one byte off the leg that is actually the cost.

Getting an exact ratio on real weight bytes (rather than `arange`) was
attempted and **blocked by a toolchain bug, not by data or hardware risk**:
`iron.jit`'s cache key (`_create_function_cache_key` + `_generation_cache_key`)
hashes call arguments and `(device, full_elf)` but never inspects a
generator's own module globals, so every attempt to resize the destination BD
via `dma_compression.py`'s module-level `RATIOED_N` constant — including with
`use_cache=False` — silently re-ran the control's compiled config regardless
of the requested value (confirmed via `.as_mlir()`, which *does* see the
change, isolating the staleness to the compile/cache layer). Chasing that bug
further was judged not worth the hardware-timeout risk on shared equipment
once the structural blocker already closed the decision question; a
zero-byte-fraction proxy check (real weights: 0.23–3.13% vs `arange`'s 51.7%,
17–220x less of the raw material this compression scheme exploits) is the
closest thing to a ratio signal on record, and it argues the real number would
have been worse than 1.39x even on the leg where it does not matter anyway.

**End-to-end arithmetic, granting the control's ratio to real weights for
free** (which the zero-byte evidence argues against) **and pretending the
shim gap does not exist** (which it does): applying 1.391x to B only, across
`tasks/0080`'s four measured int8 (i32-C) production dispatches at M=8192/8
columns, the four-GEMM sum falls from 9838 µs to 9269 µs — **5.8%**, the
ceiling this lever was ever worth, before subtracting the decompression
overhead the mechanism itself needs. **Verdict: does not pay. Retired as a
production lever; the mem-tile/compute-tile-only compression datapath exists
in the silicon but has no traffic to compress that this design's cost model
prices.**
```

**3. `tasks/README.md` index row** (insert after the existing 0089 row / in
numeric order — this session did not check what else landed at 0088/0089):

```markdown
| [0090](0090-t5-dma-compression/TASK.md) | T5 closed — DMA compression cannot pay: AIE2P's shim DMA has **no compression hardware at all** (`XAIE_FEATURE_UNAVAILABLE` in the vendor's own register table, mem-tile/compute-tile only), and every byte the int8 traffic-bound cost model (0080) prices crosses the shim. Control reproduced (1.391x on `arange`, sha-matched); a real-weight ratio was attempted but blocked by a newly-found `iron.jit` caching bug (cache key blind to a generator's own module globals) rather than by data or hardware risk — real weights carry 17–220x fewer zero bytes than `arange`, the weakest available signal that the true ratio would have been poor even if reachable. Best-case arithmetic (granting the control's ratio to B, ignoring the shim gap): **5.8%** ceiling on the four production GEMM dispatches — not worth building even before the hardware wall | — | done |
```
