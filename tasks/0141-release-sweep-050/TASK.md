# 0141 — The 0.5.0 sweep: seven models, one protocol, the tail gate's first release outing — and the machine's loudest contender was this session itself

**Date**: 2026-08-27
**Goal**: the whole-catalogue re-benchmark the release rule requires, now
seven models, with the new `tail` stage in the protocol. MTEB deliberately
skipped (`-Skip mteb`) and carried forward bit-identically — nomic verified
by hash after the one change touching its path
([`0136`](../0136-gte-runtime/TASK.md)), gte from
[`0137`](../0137-gte-gates/TASK.md)'s own symmetric session.

## Harness changes for this sweep

* `$CATALOG` gains the gte row — with **per-row opt-outs** `interleaved =
  $false; energy = $false`, argued in the adjacent comment: both stages
  need a CPU reference arm, and gte's `trust_remote_code` model is
  unusable without the 0134/0136 buffer repairs (the harnesses would crash
  or measure a silently position-scrambled model). NPU figures stand alone
  and labelled, per docs/05-measurement.
* The stage guards honour the opt-outs (`-and ($m.interleaved -ne
  $false)`, same for energy).

## The runs it took to get a clean gate (kept, per rule 3b)

The sweep's CPU-quiet gate refused **three times**, each naming `bash (pid
15036) ~0.25 core(s)` — which turned out to be **this agent session's own
shell**, kept permanently busy by a leftover "wait for server" poller from
an earlier subagent (4,400 CPU-seconds accumulated). The gate was right
every time: that process was a genuine CPU-side contender that would have
depressed every interleaved ratio. Killing it (not excluding `bash` from
the check — an exclusion would have tainted the ratios silently) let
attempt 4 pass at CPU 2.7% mean with an empty contender list. The recorded
lesson: **an agent-driven sweep must audit its own session for contenders
before blaming the machine.**

```powershell
# attempt 4, the run of record (45 s pre-delay so the launcher's own
# activity clears the 3 s sampling window):
powershell -NoProfile -Command "Start-Sleep 45; cd <repo>; .\tools\release_benchmark.ps1 -Skip mteb -OutDir 'tasks\0141-release-sweep-050'"
```

## Results (full logs and JSONs in this directory; `sweep.json` is the index)

| model | datapath (reported) | seq/s ×3 | NPU/CPU | J/1k better | worst `1-cos` | p99 tail |
|---|---|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | bfp16, C bf16 | 1463.9 / 1462.7 / 1455.8 | 1.855× | 3.42× | 3.406e-04 | 9.652e-04 |
| bge-small-en-v1.5 | bf16, C fp32 | 629.4 / 631.5 / 630.1 | 1.589× | 2.64× | 8.348e-06 | 1.162e-05 |
| bge-base-en-v1.5 | bfp16, C bf16 | 323.2 / 324.9 / 323.9 | 2.713× | 4.52× | 2.284e-04 | 3.711e-04 |
| bge-large-en-v1.5 | bfp16, C bf16 | 95.1 / 95.4 / 95.2 | 2.607× | 4.89× | 2.626e-04 | 6.626e-03 (waiver, T51) |
| nomic-embed-text-v1.5 | bfp16, C bf16 | 261.8 / 262.4 / 262.3 | 3.494× | 6.55× | 1.402e-03 | 1.579e-03 |
| embeddinggemma-300m | bfp16, C bf16 | 180.8 / 182.0 / 182.2 | 1.271× | carried (see below) | PASS (differential) | 4.227e-04 |
| **gte-multilingual-base** | bfp16, C bf16 | 253.0 / 256.2 / 255.4 | — (opt-out) | — (opt-out) | 4.608e-04 | 5.160e-04 |

int8 rows also ran clean this sweep (MiniLM 1713, small 868, base 414,
large-n64 146.7, nomic 324 seq/s; `interleaved` no longer failed on them) —
still not in the bundle, unchanged policy.

**Tail gate: PASS on every model against its recorded ceiling** — the
instrument's first release outing, and bge-large's 100× tail sits inside
its register-linked waiver exactly as designed.

**The regression check is the `vs 0.4.0` column being flat.** Against
0.4.0's run-of-record (`tasks/0119-release-sweep-040/sweep-run1-int8-included.json`,
same harness, same stage): all six carried models within **±1%**
(MiniLM 1455→1461, small 630→630, base 324.6→324, large 94.8→95.2, nomic
261.9→262.2, gemma 179.9→181.7). 0.5.0 changed no array code for them, and
the column says so.

## Problems hit

1. The three quiet-gate refusals above — the sweep's guard catching the
   sweep's operator.
2. **gemma's NPU energy arm produced a degenerate differential**: Δt between
   the low (20-encode) and high (60-encode) runs was −0.003 s and 0.014 s —
   the encode count did not scale the work, so ΔE ≈ 0 and the "ratio" came
   out 14,554×, the exact "cannot tell used-little-energy from
   died-instantly" shape 0085 documented. The number is **discarded**;
   gemma's energy column carries 0.4.0's 3.72× with this provenance note.
   The harness fix (make the gemma energy arm scale encodes, or refuse) is
   release-note-visible and left for 0.6.0's harness pass.
3. Energy ratios moved with the CPU side between sweeps (nomic 4.01× →
   6.55×) — the known ±20%-class caveat, now with a 60% example. Recorded;
   the ratio column stays labelled indicative.

## What was updated from this sweep

README performance section, `docs/CURRENT_STATUS.md` (0.5.0 table + §9
rewrite), `dist/RELEASE-NOTES-0.5.0.md`. All three quote only figures with
artifacts in this directory or the task logs they cite.
