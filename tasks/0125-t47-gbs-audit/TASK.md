# 0125 — The GB/s audit: nine design sets re-probed with the fixed accounting, and the corrected table confirms both the fix and the ~44 GB/s roof

**Date**: 2026-08-27
**Goal**: the second half of T47 — audit every published `GB/s` figure now
that [`0124`](../0124-t47-t50-runtime-fixes/TASK.md) fixed `--probe-streams`'
byte accounting, and produce the corrected reference table from the shipped
design sets.

## What the audit found in the docs

Less than feared. Grepping `docs/`, `research/` and `CLAUDE.md`:

* **No current doc quotes a probe-streams `GB/s` figure.** The inflated
  numbers live in the task logs (`0097` t21 — a diary, kept verbatim per
  rule 3b) and in T47's own thread text documenting the bug.
* `docs/01-hardware/README.md`'s 33 GB/s is `0010`'s regression fit (its own
  accounting, unaffected); `docs/03-kernels/README.md`'s ~24 GB/s is an
  M2-era M-sweep fit (own accounting, dated but not this bug); the 40–60
  GB/s band is external (AMD / Gemma3 team).
* `tools/roofline.py` computes traffic from its **own** `traffic_bytes()`,
  which was already correct — `0120` §4a *diffed the two implementations*
  to find this bug, so the roofline artifacts (`measured.json`,
  `roofline.json`) need no correction.

So the audit's deliverable is the corrected table below, plus the
confirmation that the fixed binary reproduces `0097`'s hand-corrected
numbers.

## The corrected table

One command per row, `runtime/build/npuembed.exe <root> --model <m>
--artifacts <set> --probe-streams`, 30 repeats each, full transcript in
[`raw.txt`](raw.txt). Machine state: CPU load 4% before / 1% after; `xrt-smi
examine -r aie-partitions` shows only the resident `WorkloadsSessionHost.exe`
contexts, all **Idle**, 0 in-flight submissions (the same background state
the 0119 release sweep ran under). These are host-observed dispatch times —
the y-axis carries T45's caveat exactly as 0120's did; what changed here is
the **byte** column, which is static arithmetic.

Implied GB/s per stream (qkv / attn_out / ffn_up / ffn_down):

| model / set | datapath | tile_n×cols | GB/s |
|---|---|---|---|
| MiniLM / `minilm_bfp16` | bfp16, C bf16 | 48×8 | 43.8 / 30.7 / 45.5 / 31.1 |
| bge-small / `small_bf16` | bf16, C fp32 | 48×8 | 25.2 / 15.0 / 26.4 / 17.2 |
| bge-base / `base_bfp16` | bfp16, C bf16 | 48×8 | 45.1 / 32.9 / 44.5 / 39.6 |
| bge-large / `large_bfp16` | bfp16, C bf16 | **32×8** | 43.9 / 36.3 / 44.3 / 42.6 |
| nomic / `nomic_bfp16` | bfp16, C bf16 | 48×8 | 44.4 / 32.2 / 46.7 / 41.0 |
| MiniLM int8 / `int8c_mini` | i8, C bf16 | 48×8 | 35.5 / 22.1 / 36.3 / 31.6 |
| bge-base int8 / `int8c_base` | i8, C bf16 | 48×8 | 35.9 / 25.9 / 35.7 / 32.2 |
| bge-large int8n64 / `int8c_large_n64` | i8, C bf16 | **64×8** | 39.7 / 28.2 / 40.8 / 37.3 |
| nomic int8 / `int8c_nomic` | i8, C bf16 | 48×8 | 35.6 / 36.1 / 37.6 / 33.8 |

(embeddinggemma has no probe path — `run_gemma_mode` predates
`--probe-streams` — the same gap that left it without an aggregate point in
0120 §6.)

## What the corrected numbers say

1. **The fix reproduces 0097's hand-correction.** MiniLM int8 reads
   35.5 / 22.1 / 36.3 / 31.6 against 0097 t21's corrected
   35.9 / 22.8 / 36.8 / 31.7 — within ~2% on every stream, in a different
   session, on different dispatch times. The correction was arithmetic, and
   the arithmetic now agrees from two directions.
2. **The bfp16 large shapes cluster at 43.8–46.7 GB/s** — an independent
   per-dispatch corroboration of 0120 §3c's inferred ~44 GB/s roof, from a
   different instrument (single-stream probe vs `--bench` aggregate). Still
   an *inference* on a host-observed y-axis; T45's traced measurement is
   what converts it.
3. **`attn_out` reads low on every model** (30.7–36.3 vs its siblings'
   43.8–46.7) — T45's fixed-cost fingerprint, visible per-dispatch: the
   smallest dispatch carries the largest share of any fixed cost.
4. **int8 sits lower than bfp16 on the same shapes** (35–41 vs 44–47) —
   consistent with the same fixed cost weighing more on int8's shorter
   dispatches, not with a datapath-dependent bandwidth. T45's M-sweep
   separates these cleanly.
5. **bge-small's plain-bf16 numbers (15–26 GB/s) are not a bandwidth
   statement at all** — that datapath is compute-bound on the fp32 vector
   unit (0049), so its implied GB/s is just arithmetic-time divided into
   bytes.

## Commands run

```powershell
# full loop in raw.txt header; per model:
runtime\build\npuembed.exe . --model <model> --artifacts <set> --probe-streams
# machine state before/after:
(Get-CimInstance Win32_Processor).LoadPercentage    # 4% -> 1%
C:\Windows\System32\AMD\xrt-smi.exe examine -r aie-partitions   # all contexts Idle
```

## Closure

**T47 is closed by 0124 (fix) + this task (audit).** Closure condition, per
the preamble rule 0123 added: this closure holds while `--probe-streams` is
the only consumer of `DesignInfo::tile_n/cols` — a future consumer that
*defaults* instead of refusing on 0 reintroduces the class. The corrected
table above supersedes any GB/s figure printed by the pre-0124 binary; task
logs keep their printed numbers as diary entries, with 0097 t21 already
carrying its own correction inline.
