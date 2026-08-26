# 0097 — T18 / T21 / T4: closing three measurement gaps in one session

- **Date** 2026-08-23
- **Milestone** M13 (research, not build)
- **Status** done

## Goal

Three open threads had a stated way to close them and nobody had run the commands yet:

1. **T18** — `--probe-streams` and `--bench` disagree by up to 10% on array time
   ([`0048`](../0048-m9-what-is-the-gemm-time/TASK.md)). Explain the gap, or show it
   does not reproduce, and say which number to quote.
2. **T21** — one tile geometry serves all four production GEMM shapes; would a
   per-shape geometry be worth anything, on the int8 datapath where the traffic model
   governs? Price it.
3. **T4** — finish [`tasks/0043`](../0043-m9-attention-geometry/TASK.md): its Results
   section was a literal placeholder, `(filled in below)`, even though CLAUDE.md
   already cites it for a "not worth the fight" verdict.

All three ran on hardware, on a machine with the NPU exclusively ours the whole
session — checked with `xrt-smi examine --report aie-partitions` before starting and
spot-checked throughout. Every row shown was `Idle` (a small set of long-lived
`WorkloadsSessionHost.exe` background contexts, present the whole session, never
`Active`) — the same signal the runtime's own `npu::require_exclusive_npu` gate
treats as non-contending, and `--bench` printed `npu exclusive` on every run that
checks it.

## Context

- [`0048`](../0048-m9-what-is-the-gemm-time/TASK.md) established the iteration-bound
  cost model for bf16 and left T18 standing as an unexplained discrepancy.
- [`0080`](../0080-m13-int8-traffic-bound/TASK.md) / [`0081`](../0081-m13-int8-everywhere/TASK.md)
  put int8 back in the traffic-bound regime and used ONE uniform `tile_n` per model
  (48 for hidden 384/768, 64 for bge-large hidden 1024). T21 asked whether a *per-shape*
  geometry inside one model would do better.
- [`0043`](../0043-m9-attention-geometry/TASK.md) derived, structurally, that any design
  expressing attention is capped at `cols <= 4`, exported four artifact sets to price it,
  and never ran them.

## What was done

### T18

Reran `--probe-streams` and `--bench` on `runtime/artifacts_b128il` (MiniLM, fp32 C,
the exact set 0048 used), three repeats each, same session, idle array both times.
`--bench`'s printed breakdown separates **submit** (build + start the `xrt::run`) from
**wait** (`r.wait()`, the actual hardware completion) — `Design::dispatch_only()`
(`runtime/src/npu_device.cpp:371-405`) is the SAME function both `--probe-streams` and
the production `Design::run()` path call, timed the same way in both: `t_submit` around
building/starting the command, `t_wait` around `r.wait()`. `--probe-streams`'s own
`us/disp` column times the *whole* `dispatch_only()` call (submit+wait together, no
syncs, no host work between calls) — so the naive comparison is `--probe-streams`'s
mean against `--bench`'s **`submit + wait`**, not against `wait` alone; 0048's own
table happened to compare probe against `wait` alone.

Ran both three times each to get a same-session baseline with real spread numbers
(never a single run, per the measurement rules), then also probed the bf16-C set
(`artifacts_cbf16`) once, since 0048 reported that pairing too — but `--bench` could
not be re-run on it this session because that artifact directory only ships
`gemm_rtp/` (no `manifest.json`, no eltwise pieces), so it predates the full-encode
wiring `--bench` needs. Not rebuilt (out of scope: rebuilding it risks silently
changing exactly the code path 0048 measured).

### T21

The four production shapes cannot each get their own tile geometry in the shipped
architecture (one xclbin, one static tile shared by all four instruction streams) —
`tools/export_gemm_rtp.py` enforces this by construction, building all four shapes in
one loop against one `-n`. To measure what a per-shape geometry would be *worth*
without that constraint, wrote a small standalone exporter
(`export_solo_shape.py`, copied into this task's directory) that builds ONE GEMM shape
as its own single-stream "unified-shaped" design (reusing `export_gemm_rtp.py`'s cache
identity/purge machinery so the runtime's existing `--probe-streams` path accepts it
unmodified).

Worked out, by hand, the legal tile-`n` ceiling for each of MiniLM's four shapes at
`cols=8` (constraints: AIE2P int8 microkernel `n % 16 == 0`; design legality
`N % (n*cols) == 0`; and the c-bf16 narrowing kernel's fixed set of entry points,
`m*n` in `{1024,2048,3072,4096}` for `m=64`, so `n <= 64` regardless of what L1 would
allow). Three of MiniLM's four shapes (`qkv`, `attn_out`, `ffn_down`) are already AT
their legal ceiling at the shipped `n=48`; `ffn_up` alone can legally reach `n=64`
(`N=1536`, `1536/(64*8)=3` exactly) — a genuine, buildable, per-shape difference. Built
that design (`artifacts_t21_ffnup_n64`), plus a `n=48` solo control to validate the
single-op methodology reproduces the shipped unified number, plus a matching
`all-MiniLM-L6-v2.int8n64.npue` container (packing at `tile-n=64` succeeds for all four
weight matrices — the container-level `tile_b()` requires only `N % tile_n == 0`, a
*weaker* constraint than the design's `N % (n*cols) == 0`, so the container built even
though the corresponding whole-model n=64 design would not).

### T4

All four of 0043's artifact sets, and the `bge-large-n16.npue` container, already
existed on disk from that task's own export commands — nothing needed rebuilding, only
running. Confirmed each `design.json`'s `tile`/`cols` matched 0043's stated A/B/C/D
geometries exactly before trusting any number. Ran `--probe-streams` on all four (two
repeats for A and D, which bracket both cost axes; one each for B and C, lower priority
since the A→C→D chain is what the task's own commentary is built around). Added a
correctness spot-check (plain validate run, no flags) on A and D — not part of 0043's
original plan, but a fast wrong design is not a result, and CLAUDE.md rule 6 asks for a
traceable number, not just a traceable speed.

Also chased down a claim in this task's own brief — that `tasks/0090` found the shim
DMA's `.Padding` field unavailable, closing note 0007 §1.1's reopening of 0043's
`cols<=4` result. **That claim does not hold up**: `0090` (T5, DMA compression) checked
`.Compression`, not `.Padding`, and neither its `TASK.md` nor its saved register-grep
artifact mentions a `.Padding` field at all. Corrected in the write-up below and in
`0043`'s own Results section — §1.1's mem-tile-scoped padding idea is not closed by
anything measured this session, and the C-vs-D split this task already measured is
exactly the number needed to price it once it is built.

## Commands

See `t18_raw.txt`, `t21_raw.txt`, `t4_raw.txt` in this directory for the full
command-by-command transcript with every number. Representative commands:

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# T18
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --probe-streams
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --threads 24 --bench 5

# T21
python "<scratchpad>\export_solo_shape.py" --out runtime\artifacts_t21_ffnup_n64 `
    --op ffn_up --M 8192 --K 384 --N 1536 -n 64 --cols 8
.\.venv-ref\Scripts\python.exe tools\pack_npue.py --model-dir models\all-MiniLM-L6-v2 `
    --out models\all-MiniLM-L6-v2.int8n64.npue --int8 --tile-n 64
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2.int8n64 --artifacts artifacts_t21_ffnup_n64 --probe-streams
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2.int8 --artifacts artifacts_int8c_mini --probe-streams

# T4
.\runtime\build\npuembed.exe . --model bge-large-en-v1.5 --artifacts artifacts_large --probe-streams
.\runtime\build\npuembed.exe . --model bge-large-n16    --artifacts artifacts_large_m32   --probe-streams
.\runtime\build\npuembed.exe . --model bge-large-n16    --artifacts artifacts_large_m16c8 --probe-streams
.\runtime\build\npuembed.exe . --model bge-large-n16    --artifacts artifacts_large_m16c4 --probe-streams
.\runtime\build\npuembed.exe . --model bge-large-en-v1.5 --artifacts artifacts_large        # correctness
.\runtime\build\npuembed.exe . --model bge-large-n16    --artifacts artifacts_large_m16c4  # correctness
```

## Result

### T18 — ANSWERED: the gap did not reproduce; quote `--bench`'s "wait (hardware)"

| | run1 | run2 | run3 | mean | spread |
|---|---:|---:|---:|---:|---:|
| `--probe-streams` mean of 4 streams (submit+wait, no syncs) | 3029 µs | 3037 µs | 3037 µs | **3034.4 µs** | 0.26% |
| `--bench` "wait (hardware)" only | 3026 µs | 3024 µs | 3035 µs | **3028.3 µs** | 0.19% |
| `--bench` "submit" only | 49 µs | 51 µs | 54 µs | 51.3 µs | — |
| `--bench` "submit + wait" (the number actually comparable to probe) | 3078 µs | 3077 µs | 3092 µs | **3082.3 µs** | — |

Today, on an idle array, in the same session: probe's mean (3034.4 µs) sits **0.2%**
from bench's `wait`-only figure and **1.6%** from bench's `submit+wait` (the
apples-to-apples comparison). Neither gap is anywhere close to the **8.7–10%** 0048
reported. `submit` itself (49–54 µs) is far too small to explain a 300 µs gap even in
the worst case, which rules out "probe times submit+wait, bench's headline number is
wait-only" as more than a ~1.6% partial explanation.

**Conclusion: the two harnesses measure the same underlying quantity — the same
`Design::dispatch_only()` / `xrt::run::wait()` call, timed the same way — and agree to
within run-to-run noise today.** 0048's reported 8–10% gap does not reproduce under
identical commands, identical artifact set, identical idle-array conditions. The most
likely explanation is session-to-session drift (thermal state, driver warm-up, or a
transient background load neither session's contention gate would catch, since the
gate only refuses on a foreign *Active* context, not on driver/thermal variance)
rather than a physical or definitional difference between "back-to-back" and "synced"
dispatch. This was not independently re-verified for the bf16-C pairing (`--bench`
could not run on `artifacts_cbf16`, which lacks the full manifest) — that half of
0048's table stands unconfirmed either way.

**Recommendation: quote `--bench`'s "wait (hardware)" line for array-only time.** It
comes from the actual production dispatch path (`Encoder::run()`, not a synthetic
back-to-back loop) and cleanly separates hardware completion time from command-build
overhead. `--probe-streams` remains the right tool for the *shape-discriminating*
comparisons (0048's `ffn_up` vs `ffn_down`, this task's T21 work) because it needs no
full artifact set — but for "what is the array's per-dispatch time," today's evidence
says either number will do, and previously calling this "unexplained" overstated the
disagreement.

### T21 — ANSWERED: per-shape geometry is worth ~4% of GEMM time on hidden-384/768
models, and zero on bge-large (already captured)

Per-shape legal `n` ceiling at `cols=8` (int8, c-bf16, `m=64,k=64`), MiniLM
(hidden=384):

| shape | N | legal `n` set | max legal `n` | shipped `n` |
|---|---:|---|---:|---:|
| qkv | 1152 | {16, 48} | 48 | 48 |
| attn_out | 384 | {16, 48} | 48 | 48 |
| ffn_down | 384 | {16, 48} | 48 | 48 |
| **ffn_up** | 1536 | {16, 32, 48, **64**} | **64** | 48 |

Three of four shapes are already at their ceiling. `ffn_up` alone has room. Measured
(mean of runs, int8 + c-bf16, M=8192):

| | shipped uniform `n=48` | shape's own legal max `n=64` | ratio |
|---|---:|---:|---:|
| `ffn_up` alone | 1539 µs | 1351 µs | **1.139×** |
| four-shape sum | 4533 µs | 4345 µs (only `ffn_up` swapped) | **1.043×** |

A solo `n=48` control build reproduced the shipped unified design's `ffn_up` number to
0.2% (1542 µs vs 1539 µs), validating the single-op measurement methodology before
trusting the `n=64` result.

**A per-shape geometry would be worth 4.3% of GEMM-array time on MiniLM** (and by the
same divisibility argument, on bge-small/bge-base/nomic — every hidden-384/768 model
this project ships has `intermediate = 4*hidden`, so `ffn_up`'s `N` is always divisible
by `64*8=512` while `qkv`'s `3*hidden` and `attn_out`/`ffn_down`'s `hidden` generally
are not), concentrated entirely in `ffn_up`. **On bge-large (hidden=1024) it is worth
exactly zero**: all four of its shapes divide 512 (`0081` already found this and
shipped `tile_n=64` uniformly across the whole model), so the uniform geometry already
*is* every shape's individual optimum there. **The one-geometry-per-model constraint
costs something only on the narrower models, and only ~4%** — real, structurally
explained, and small. `n=128` (which would remove `qkv`'s and `attn_out`/`ffn_down`'s
ceiling too, and is legal by the L1 budget) is blocked by a missing c-bf16 kernel entry
point (`gemm_pretiled.py`'s `_ok = (1024,2048,3072,4096)` for `m=64`), not by geometry —
a real further lever, but it needs new kernel work, out of scope here.

### T4 — 0043's Results filled in; substantiates CLAUDE.md, with a correction along the way

See the Results section appended directly to
[`tasks/0043-m9-attention-geometry/TASK.md`](../0043-m9-attention-geometry/TASK.md) —
per the task instructions, append there rather than duplicate the numbers here.
Headline: **A → D (the only geometry that can also express attention) costs the four
projection GEMMs 2.229× their production array time** (A→C tile-size penalty alone
1.289×, C→D column-count penalty alone 1.729×, compounding to 2.228× — matches the
measured 2.229× to three figures). Against attention's ~2–5% cost on the host
(CLAUDE.md's own F3 figure), this is unambiguously a bad trade at any plausible array
share of the encode — **substantiates** the "not worth the fight" sentence CLAUDE.md
has been carrying since before 0043 ever measured anything.

**Correction made along the way**: this task's own brief asserted `tasks/0090` found
the shim DMA's `.Padding` field unavailable, closing note 0007 §1.1's reopening of
0043's `cols<=4` conclusion via mem-tile `pad_dimensions`. Checked `0090` directly —
it verified `.Compression`, not `.Padding`; neither field is mentioned together in its
saved artifact. **§1.1 is not closed.** Its mechanism is mem-tile-scoped by its own
design (already cited from `AIEDialect.cpp`'s verifier inside note 0007 itself), which
is consistent with, not refuted by, what 0090 actually found (mem-tile has extra DMA
features, shim does not). **If** padding lets attention run at `cols=8` instead of 4,
the geometry the projections would share is **C**, not **D** — a 1.289× tax, not
2.229× — which changes the arithmetic from "clearly negative" to "probably still
negative at 30–40% array share, close enough to be worth an actual build." That build
does not exist; this session priced the two brackets (C and D) that make it possible to
price §1.1 quickly once it does.

## Problems hit

- **`--bench` would not run on `artifacts_cbf16`** (T18's bf16-C set) — it only ships
  `gemm_rtp/`, no `manifest.json` or eltwise pieces, because it predates the
  full-encode wiring. Worked around by not rebuilding it (rebuilding risks changing the
  exact thing 0048 measured); the bf16-C half of T18's original discrepancy is
  therefore unconfirmed either way this session, stated as such rather than assumed
  resolved.
- **T21's naive plan (build the whole unified design at a bigger `n`) does not work**:
  `export_gemm_rtp.py` builds all four shapes against ONE shared `-n` and asserts
  `N % (n*cols) == 0` per shape, which three of MiniLM's four shapes fail above `n=48`.
  Worked around by writing a standalone single-shape exporter
  (`export_solo_shape.py`, kept in this directory) that reuses the existing cache/purge
  machinery so `--probe-streams` accepts the result unmodified, and validated it against
  a solo `n=48` control before trusting the `n=64` number.
- **The solo `n=64` design initially refused to run**: `layout mismatch` against the
  production MiniLM int8 container (packed at `tile-n=48`). Fixed by packing a second
  container at `--tile-n 64` — legal at the container level even though the
  corresponding whole-model design is not, because `tile_b()`'s constraint
  (`N % tile_n == 0`) is strictly weaker than the design's (`N % (tile_n*cols) == 0`).
  This is itself a small, useful fact: a container can be tiled finer than any design
  that could legally consume it whole.
- **This task's own brief mischaracterised `tasks/0090`'s finding** (see T4 above and
  the correction in `t4_raw.txt`) — caught by reading `0090` directly rather than
  taking the paraphrase at face value. Left as a correction rather than silently fixed,
  per rule 3b's spirit: the wrong claim and the check that caught it are both worth
  keeping.
- **`--probe-streams`'s solo-design correctness was not independently re-verified**
  for the T21 `n=64` build (no full-encode/golden wiring exists for a one-stream
  artifact set). Relies on inference from `0081`, which measured the identical kernel
  entry point (`narrow_4096_i32_bf16`, `m*n=4096`) bit-exact for bge-large's `n=64`
  design — only the tiled dimension's *host* (`N=1536` vs bge-large's `N` values)
  differs here, not the arithmetic. Stated as inference, not measurement.

## Artifacts

- `t18_raw.txt`, `t21_raw.txt`, `t4_raw.txt` — full command transcripts with every
  number, in this directory.
- `export_solo_shape.py` — the standalone single-shape exporter written for T21, kept
  here since it is reusable for any future single-shape geometry probe.
- New runtime artifact directories (gitignored, not checked in, rebuildable from the
  commands above / in the raw transcripts): `runtime/artifacts_t21_ffnup_n64`,
  `runtime/artifacts_t21_ffnup_n48`.
- New model container (gitignored): `models/all-MiniLM-L6-v2.int8n64.npue`.
- Results appended directly to
  [`tasks/0043-m9-attention-geometry/TASK.md`](../0043-m9-attention-geometry/TASK.md).

## Next

- **T18**: retire to `CLOSED-THREADS.md`. If a future session sees the gap again,
  0048/0097's evidence says look for session-level drift (thermal, driver warm-up)
  before assuming a definitional difference.
- **T21**: retire to `CLOSED-THREADS.md` with the 4.3%/0% split as the answer. The real
  remaining lever is the missing `n=128` c-bf16 kernel entry point, which is kernel
  work, not a measurement question — worth its own thread if anyone wants to chase it.
- **T4**: retire (0043 is finished). The note 0007 §1.1 padding idea is a live,
  unbuilt, unpriced-until-built question — worth flagging in the register as its own
  small thread rather than folded silently into T4's closure, since T4 was specifically
  about finishing 0043 and §1.1 is a separate idea 0043 only gestures at.

## Proposed register update

### T18 → ANSWERED

Replace the T18 entry in `OPEN-THREADS.md` with a `CLOSED-THREADS.md` entry:

> ### T18 — `--probe-streams` and `--bench` disagree by up to 10% on array time ·
> **ANSWERED 2026-08-23 ([`0097`](TASK.md))**
>
> [`0048`](../0048-m9-what-is-the-gemm-time/TASK.md) reported probe 3,328 µs vs
> bench 3,028 µs (fp32 C) and asked which to quote. Re-run in `0097`, same artifact
> set, same commands, idle array, three repeats each: probe's mean (3034.4 µs) sits
> **0.2%** from bench's `wait (hardware)` figure (3028.3 µs) and 1.6% from bench's
> `submit + wait` (the number actually comparable to probe's, since probe times the
> whole `dispatch_only()` call). **The 8–10% gap did not reproduce.** Both harnesses
> call the same `Design::dispatch_only()` / `xrt::run::wait()`, timed the same way; the
> likely cause of 0048's gap was session-to-session drift, not a physical or
> definitional difference between back-to-back and synced dispatch. **Quote `--bench`'s
> `wait (hardware)` line** for array-only time — it comes from the production dispatch
> path and cleanly separates hardware time from submit overhead — but `0097` found no
> evidence either number was ever wrong. Not independently re-confirmed for the
> bf16-C pairing (`--bench` could not run on the artifact set that predates full-encode
> wiring).

### T21 → ANSWERED

Replace the T21 entry with:

> ### T21 — One tile geometry for four shapes · **ANSWERED 2026-08-23
> ([`0097`](TASK.md))** · worth ~4%, and only on
> narrow models
>
> Priced directly on int8 (where the traffic model governs, `0080`): three of MiniLM's
> four shapes (`qkv`, `attn_out`, `ffn_down`) are already at their `cols=8` legal `n`
> ceiling at the shipped `n=48`. Only `ffn_up` has room, to `n=64` — measured 1.139×
> faster alone, 1.043× (4.3%) on the four-shape sum. The same divisibility argument
> (`intermediate = 4*hidden` always divides `64*8=512`; `qkv`'s `3*hidden` and
> `attn_out`/`ffn_down`'s `hidden` usually do not) applies to every hidden-384/768 model
> this project ships. **On bge-large (hidden=1024) it is worth exactly zero** — all four
> shapes already divide 512, which is exactly why `0081` could move the whole model to
> `tile_n=64` uniformly and lose nothing. **The one-geometry-per-model constraint costs
> ~4% of GEMM-array time, only on the narrow models, concentrated in one shape.** A
> further lever exists (`n=128`, legal by L1, illegal only because no c-bf16 kernel
> entry point exists for `m*n=8192`) but needs new kernel work — worth its own thread if
> pursued, not covered here.

### T4 → ANSWERED, with a spinoff

`tasks/0043`'s Results section is filled in; retire T4. Recommend opening a small new
thread for note 0007 §1.1 specifically (not previously separately tracked — T4's text
only mentioned it in passing), since `0097` corrected a mischaracterisation of what
`tasks/0090` established and left the mem-tile-padding idea genuinely open with a
priced-but-unbuilt cost bracket:

> ### T-new — Does mem-tile `pad_dimensions` make attention worth folding onto the
> array? · **OPEN**
>
> [note 0007](../../research/notes/0007-unused-iron-surface.md) §1.1 proposes padding attention's
> real N=64 per-column slice from 8 to 16 wide in the mem tile, which would make
> `n=16, cols=8` legal for attention instead of `0043`'s derived `cols<=4` — doubling
> attention's FLOPs (padded half multiplied by zero) to buy back 4 of the array's 8
> columns. Unbuilt. `0043`/`0097` priced the two brackets: if it works, the unified
> geometry the *projections* would share is `0043`'s set **C** (1.289× their production
> array time) rather than set **D** (2.229×) — a materially smaller tax than the naive
> unified design. Against attention's ~2–5% host cost (F3), that is "probably still
> negative, close enough to be worth an actual build" rather than D's "clearly
> negative." Nothing this session built the padded design, measured attention's own
> achievable efficiency at `n=16`, or implemented the strided-destride note 0007 flags
> on the output side.

### `tasks/README.md` index row

```
| [0097](0097-t18-t21-t4-measurements/TASK.md) | T18/T21/T4: probe-vs-bench gap did not reproduce (quote bench's `wait`); per-shape tile geometry worth ~4% on narrow models, 0% on bge-large; 0043's Results filled in — substantiates CLAUDE.md's attention verdict, but note 0007 §1.1's padding idea is not closed | M13 | done |
```
