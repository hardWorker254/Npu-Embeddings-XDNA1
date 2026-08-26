# 0009 — Toolchain provenance: the 1.3.4 → 1.4.2 upgrade changed kernel codegen, and nothing records which toolchain built an artifact

- **Date** 2026-08-23
- **Context** [`0099`](../../tasks/0099-t26-rounding-ablation/TASK.md) (T26)
  found that the mlir-aie 1.3.4 → 1.4.2 upgrade silently changed the object
  Peano emits for the emulated bfp16 matmul kernel (`mm.cc`), and that this
  was the actual mechanism behind a 6.6× accuracy anomaly that three separate
  research sessions ([`0052`](../../tasks/0052-m10-research-night/TASK.md),
  [`0053`](../../tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md),
  [`0056`](../../tasks/0056-m10-t26-rounding-and-chain-probe/TASK.md)) had
  chased under other hypotheses. This note is the audit that follows from
  that: which published claims rest on a pre-migration build, and of those,
  which are actually at risk. See
  [`tasks/0102-toolchain-provenance-audit`](../../tasks/0102-toolchain-provenance-audit/TASK.md)
  for the exact commands.

## The mechanism, and its evidence

`experiments/m5-pretiled-gemm/gemm_pretiled.py`'s emulated-bfp16 matmul
kernel (`mm.cc`) wraps its k-loop in
`aie::swap_rounding(conv_even)` / `set_rounding(saved)` — a save/set/restore
triple that should make the kernel's bfp16 quantisation independent of
whatever rounding mode a prior kernel left on that core. The **compiled
object** for one specific, checked-in build lacked it entirely:

```
$ grep -c crrnd experiments/m5-pretiled-gemm/artifacts/objdump_fp32C_rtp_matmul_bf16_f32_333c4d33.txt
0
$ grep -c crrnd experiments/m5-pretiled-gemm/artifacts/objdump_bf16C_rtp_matmul_bf16_f32_333c4d33.txt
0
```

— both re-confirmed directly in this session, not taken on 0098's word. A
fresh rebuild of the **identical config hash** `333c4d33`, from the
unmodified source tree, today:

```
$ grep -c crrnd experiments/m5-pretiled-gemm/artifacts/t26_ablation/objdump_poison_matmul_bf16_f32_333c4d33.txt
3
$ grep -c crrnd experiments/m5-pretiled-gemm/artifacts/t26_ablation/objdump_bf16c_matmul_bf16_f32_333c4d33.txt
3
```

Not a disassembler artifact: the old dump has `mov r29, p0` where the new
one has `mov r22, crrnd` / `mov crrnd, #0xc` / `mov crrnd, r22` (save / set
conv_even / restore), and the two dumps' headers name different cache
directories. [`0098`](../../tasks/0098-t26-kernel-source/TASK.md) traced a
specific *reason* the 1.3.4-era object could drop the triple:
`mmul_bf16_bf16.hpp` calls the **ambient-mode** intrinsic
`::to_v64bfp16ebs8` (`aie2p_srs.h:1308`) rather than the `_conf` form
(`:1331`), so nothing in the IR connects the save/restore pair to the
quantisation between them, and Peano is free to treat the whole triple as
dead. [`0099`](../../tasks/0099-t26-rounding-ablation/TASK.md) then found
the actual object built today **does** contain the triple, from the same
source, same `mlir-aie 1.4.2.dev16+g7e00b57` — so 0098's *mechanism* (DCE via
the ambient call) is not what is currently happening, but the *asymmetry it
predicted* (some builds have the fix-up, some don't) is real and
toolchain-correlated. **The literal root cause of the flip is still open**
(0099's own "Next" section) — the leading candidate is Peano/LLVM codegen
non-determinism, not a version bump alone, since three same-session rebuilds
in 0099 were internally stable. What is not open is that a build from
**before** `42da31d` (1.3.4) is confirmed missing the fix-up, and a build
from **after** it (1.4.2) is confirmed to have it, on the one kernel object
this project has actually disassembled both ways.

## The boundary

`42da31d`, 2026-08-20 16:05:34 +0200 — `feat: upgrade mlir-aie 1.3.4 -> main
(past v1.4.1), migrate 15 IRON scripts`
([`tasks/0058`](../../tasks/0058-m11-iron-1.4-migration/TASK.md)). Everything
in `tasks/0001`–`0058` is pre-migration; `0059` onward is post (0058 *is* the
migration commit's task).

Two follow-on tasks matter to how wide the boundary's effect actually
reaches:

- [`0059`](../../tasks/0059-m11-production-verify-post-migration/TASK.md)
  (same day) rebuilt the **C++ runtime** from clean and re-ran the
  validation encode + `verify_embed_e2e.py` for all four then-shipped models
  against the **existing, unrebuilt** production `.xclbin`s — every figure
  bit-identical to its pre-migration value. This proves the runtime and XRT
  loading path are unaffected, and that a static binary behaves identically
  regardless of which toolchain later runs alongside it. It does **not**
  prove a *fresh* 1.4.2 build reproduces those numbers, because the binaries
  under test were never rebuilt (confirmed independently below).
- [`0060`](../../tasks/0060-m11-export-gemm-rtp-marker-fix/TASK.md) (same
  day, 17:42) fixed the one tool (`tools/export_gemm_rtp.py`) that the
  upgrade broke, then used the fixed tool to rebuild MiniLM's real
  production GEMM shape (`--batch 128 --cols 8`) **from scratch on 1.4.2**
  into a separate, gitignored directory, and reproduced the exact same
  `1-cos` figures (`1.086e-05` golden, `2.644e-05` e2e) as the never-rebuilt
  shipped binary. This is the one place in the record where a production
  shape was actually rebuilt post-migration and compared — and it matches.
  0060 explicitly did **not** overwrite the shipped `runtime/artifacts_*`
  directories, "a separate deliberate decision, not a side effect of this
  fix."

Checked directly (not inferred) which shipped production `.xclbin`s have
ever actually been rebuilt since the migration, by `.xclbin` mtime (these
directories are gitignored, so git history can't answer this):

| artifact set | model(s) | `gemm_rtp/final.xclbin` mtime | vs `42da31d` (Aug 20, 16:05) |
|---|---|---|---|
| `runtime/artifacts_b128il` | MiniLM, bge-small | 2026-08-18 11:23 | **pre**, never rebuilt |
| `runtime/artifacts_base` | bge-base | 2026-08-19 20:50 | **pre**, never rebuilt |
| `runtime/artifacts_large` | bge-large | 2026-08-18 22:55 | **pre**, never rebuilt |
| `runtime/artifacts_nomic` | nomic-embed-text-v1.5 | 2026-08-21 20:50 | **post** |
| `runtime/artifacts_gemma` | embeddinggemma-300m | 2026-08-22 14:07 | **post** |
| `runtime/artifacts_int8*` (all 6 models) | — | 2026-08-22 19:27–23:16 | **post** |

> **On using mtime at all, given trap 7c.** CLAUDE.md says *"never identify a
> build artifact by mtime"* — and that rule is not being broken here, because
> it forbids a different question. Trap 7c is about identifying **which design**
> an artifact is: a JIT *cache hit* does not restamp the directory, so
> "newest wins" can hand you a design you did not ask for. The question here is
> only **"was this file rewritten after date X"**, and its failure mode points
> the safe way: a cache hit that skips a copy leaves an *older* stamp, so mtime
> can only under-report rebuilds — and an artifact that was re-derived without
> being rewritten has identical contents anyway, which is what the argument
> needs. It is still weaker evidence than a content check, and it is not what
> the conclusion rests on: §"the mechanism cannot reach production" below is
> structural and holds whatever these timestamps say.

So the four original BERT-family models still ship on binaries compiled
under 1.3.4, unchanged since Aug 18–19; nomic, EmbeddingGemma and every int8
container were built after the upgrade. This matters for the audit below
because the crrnd mechanism is specific to the **emulated bfp16** matmul
(`--emulate-bfp16` / T26's bf16-C comparison) — a datapath the shipped
production models **do not use**. Production ships plain bf16 in / fp32 out
(CLAUDE.md: `--emulate-bfp16` "RETIRED — fails accuracy; closed by the MTEB
gate," [`0035`](../../tasks/0035-m8-mteb-gate/TASK.md)), whose matmul kernel
never narrows inside the k-loop and has no rounding-mode dependency for the
crrnd bug to act on. int8 uses a different, native `(8,8,8)` MMAC path with
an exact integer accumulator, also outside this mechanism's reach.

## What I checked vs. inferred

**Checked directly, this session:** the boundary commit and its timestamp;
the `.xclbin` mtimes above; the crrnd grep counts on both the pre- and
post-migration disassembly, independently re-run rather than trusted from
0098/0099's prose; that `export_gemm_rtp.py`'s fix (`0c9fc5a`) landed the
same day as the migration and that 0060's rebuild-and-compare is a real,
independent hardware measurement; the source of every `aie::set_rounding`
call site in `experiments/m5-eltwise/kernels/*.cc`; that
`tasks/0044-m9-optimisation-sweep/` contains **only** `TASK.md` (no
disassembly) and that no pre-migration objdump of any eltwise kernel exists
anywhere in the repo; the row-sum numbers quoted below, directly from
0044's own log; every task-date-vs-boundary comparison in the table below
(`git log --format="%ci" -1 -- <path>`).

**Inferred, not directly checked:** that the Aug 18–19 production `.xclbin`
builds (never disassembled) share the same crrnd-absent characteristic as
the Aug 20 12:15 build that was disassembled — reasonable, since both are
1.3.4 builds from an unchanged source tree, but not itself verified.
Immaterial to the audit's verdicts below, since production's default
datapath doesn't route through this kernel's rounding-sensitive path
regardless.

## The trap 2b table: pre-migration, but not vulnerable — and the reason generalises

CLAUDE.md trap 2b's whole table (GELU 4.312e-03 → 2.494e-03, softmax
4.278e-03 → 3.325e-03, LayerNorm 3.326e-03 → 2.059e-03 with the 92×
implementation-error figure) comes from
[`0044`](../../tasks/0044-m9-optimisation-sweep/TASK.md) Part 3, committed
2026-08-19 — pre-migration, and the exact rounding-mode mechanism T26
eventually found elsewhere. It is the obvious place to worry the whole table
expired. It didn't, on two independent grounds:

**1. It was already re-measured on 1.4.2, bit-exact.**
[`0058`](../../tasks/0058-m11-iron-1.4-migration/TASK.md) rebuilt and re-ran
`gelu_kernel.py` (both `poly` and `polyrne`), `layernorm_kernel.py` (base and
`--variant rne`) and `softmax_kernel.py` (`poly` and `poly_rne`) **on
hardware, post-migration**, and got every one of these figures back exactly.
This is not an inference from an old log — it is a same-toolchain-as-today
reproduction already on record.

**2. The elimination mechanism T26 found cannot apply to these kernels, and
the 1.3.4-era measurement itself proves the rounding control took effect.**
0098's account of *why* `mm.cc`'s crrnd triple got eliminated is specific: it
wraps a call into an `aie_api` **header** (`mmul_bf16_bf16.hpp`) that reaches
hardware through an **ambient-mode** intrinsic, so nothing in the IR ties the
save/restore to the quantisation between them. The eltwise kernels are
structurally different — `aie::set_rounding` is called directly in **our
own** `.cc` (`experiments/m5-eltwise/kernels/gelu_poly.cc:533`,
`layernorm_rne.cc:44/50`, `softmax_rne.cc:43/50`), immediately adjacent to
the ordinary `aie::` vector ops that use it, not through a header call with
an ambient fallback. And on the *same 1.3.4-era build* that lost `mm.cc`'s
triple, 0098 independently found `narrow_f32_bf16.o` — same call pattern as
the eltwise kernels, `set_rounding` adjacent to direct vector ops in our own
source — **did** carry its three `mov crrnd, #0xc` writes. Same toolchain,
same session: the direct-call-site kernel kept its rounding control, the
header-mediated one lost it. That is corroborating, not merely analogous,
evidence for the structural distinction.

Independent of any disassembly, 0044's own hardware measurement is
self-validating: elimination means "the rounding mode changes nothing," and
0044 measured the opposite —

```
floor      row sums: min 0.994581  max 1.000000  worst |1-sum| 5.419e-03
conv_even  row sums: min 0.997272  max 1.002485  worst |1-sum| 2.728e-03
```

No row can exceed 1.0 under `floor` (every element rounds down) — the
measured `max` is *exactly* 1.000000. Under `conv_even` the rows straddle
1.0. If `set_rounding` had been compiled to a no-op in these kernels, the two
builds would be bit-identical and this difference could not have been
observed at all. Three paired GELU/softmax/LayerNorm error figures moved the
same way. **The measurement is proof the control took effect, independent of
what any disassembly would show** — and no pre-migration disassembly of
these kernels exists to check anyway (searched the whole repo; the only
`crrnd`-containing text files are the `mm.cc`/`narrow_f32_bf16.o` dumps
already discussed, `t26_ablation`'s and `0091`'s post-migration `gelu_poly`
dumps, and 0098's own evidence file).

**Verdict: trap 2b's table is pre-migration, its mechanism is the same class
T26 found, and it is still correct.** No action.

## Audit table

Dates are the `TASK.md`'s own commit time
(`git log --format="%ci" -1 -- <path>`), used as a proxy for measurement
time — accurate for same-day-committed research tasks (this project's norm),
flagged where I'm not confident.

| claim | task(s) | date | vs `42da31d` | vulnerability class | verdict |
|---|---|---|---|---|---|
| SAXPY 1,617× scalar-vs-vector | [`0002`](../../tasks/0002-m1-hello-npu/TASK.md) | 2026-08-16 | pre | possibly (static instr. count) | **re-verified bit-exact on 1.4.2 in `0058`** — not vulnerable |
| bf16/bfp16 GEMM 25.0→137.3 MACs/cyc, 9.8%→53.6% | [`0003`](../../tasks/0003-m2-bf16-gemm/TASK.md)/[`0004`](../../tasks/0004-m2-multicore-gemm/TASK.md) | 2026-08-17 | pre | possibly (device accuracy + cycles) | **re-verified exact in `0058`** — not vulnerable |
| trap 2b table: GELU/softmax/LayerNorm floor-vs-conv_even | [`0044`](../../tasks/0044-m9-optimisation-sweep/TASK.md) | 2026-08-19 | pre | specifically (rounding mode) | **re-verified exact in `0058`; elimination mechanism structurally inapplicable (direct source call, not ambient header intrinsic) — see above.** No action |
| B-reuse via `consumer_obj_type` refuted; cascade channel counts | [`0046`](../../tasks/0046-m9-b-reuse-asymmetric/TASK.md)/[`0047`](../../tasks/0047-m9-cascade-channel-probe/TASK.md) | 2026-08-19 | pre | possibly (compiler routing) | structural/hardware port-count facts, not rounding- or DCE-dependent; no mechanism named connecting the upgrade to channel topology. No action |
| Production `1-cos`: MiniLM 1.086e-05, bge-small 8.348e-06, bge-base 1.353e-05, bge-large 8.432e-06 | [`0038`](../../tasks/0038-m9-model-driven-runtime/TASK.md)/[`0039`](../../tasks/0039-m9-bge-small/TASK.md)/[`0051`](../../tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md)/[`0042`](../../tasks/0042-m9-bge-large/TASK.md) | 2026-08-18–19 | pre | possibly (device accuracy) | shipped `.xclbin`s **never rebuilt** (mtimes above) — the published number IS the currently-running binary's behaviour, re-verified bit-identical post-migration in `0059`. Also independently reproduced by a **fresh** 1.4.2 rebuild in `0060` (MiniLM, `1.086e-05` again). Not vulnerable — doubly confirmed |
| MTEB gates for the four BERT models (+0.04 MiniLM etc.) | [`0035`](../../tasks/0035-m8-mteb-gate/TASK.md) and successors | 2026-08-18 onward | pre | possibly (downstream of device accuracy) | same reasoning as the row above — measured against the same never-rebuilt binaries | 
| Energy, 1.94× per sequence | [`0034`](../../tasks/0034-m8-energy/TASK.md) | 2026-08-18 | pre | not vulnerable | host-side RAPL measurement, no AIE codegen involved |
| CPU baseline, tokenizer round-trip, `.npue` packing | various (0009, 0018, 0036, 0040...) | pre | pre | not vulnerable | host-side / file-format claims |
| **bfp16+bf16-C 6.6× more accurate than bfp16+fp32-C** | [`0052`](../../tasks/0052-m10-research-night/TASK.md)/[`0053`](../../tasks/0053-m10-t26-probe-bge-base-mteb/TASK.md)/[`0056`](../../tasks/0056-m10-t26-rounding-and-chain-probe/TASK.md) | 2026-08-20 (before 16:05) | pre | **specifically (rounding, the mechanism itself)** | **genuinely expired.** T26 already ANSWERED and the register already amended (see below); re-measurement in progress as `tasks/0101` |
| `--emulate-bfp16` retired, fails MTEB at `1-cos` 3.470e-03 | [`0035`](../../tasks/0035-m8-mteb-gate/TASK.md) | 2026-08-18 | pre | **specifically (same mechanism — plain bfp16+fp32-C is exactly the biased-low path)** | **re-open candidate**, already flagged by T26's own amendment; `tasks/0101` is re-measuring this |
| int8 SmoothQuant accuracy (all 6 models) | [`0077`](../../tasks/0077-m13-int8-gate/TASK.md)/[`0078`](../../tasks/0078-m13-int8-accuracy/TASK.md)/[`0081`](../../tasks/0081-m13-int8-everywhere/TASK.md) | 2026-08-22–23 | post | n/a | measured after the upgrade |
| 0.4.0 release sweep (throughput, `1-cos`, MTEB, energy, all 6 models) | [`0085`](../../tasks/0085-m13-release-sweep/TASK.md) | 2026-08-23 | post | n/a | measured after the upgrade |
| Attention geometry costs 2.229× | [`0097`](../../tasks/0097-t18-t21-t4-measurements/TASK.md) (T4) | 2026-08-23 | post | n/a | measured after the upgrade |
| T5/T6/T7/T8/T10/T11/T12/T15/T18/T21/T32 closures | various | 2026-08-23 | post | n/a | measured after the upgrade |

## Re-open candidates

Exactly one class, and it is already in motion:

**T26's own accuracy case, and `--emulate-bfp16`'s 2026-08-18 retirement,
both rest on the exact mechanism this thread found.** The T26 register entry
in `research/OPEN-THREADS.md` has already been amended (2026-08-23) to say
"this thread's entire accuracy case is toolchain-stale, and must be
re-measured before any decision" — the fp32-C path's bias should be *gone*
on 1.4.2, so plain bfp16+fp32-C (retired at 3.470e-03 in `0035`) may now
pass on its own, and the specific bf16-C combination that passed every gate
on 1.3.4 may no longer be the best choice. **`tasks/0101-t23-bfp16-gates-on-1.4.2`
already exists for this** — its directory is present but its `TASK.md` is
not yet written, i.e. it is an active, in-progress re-measurement (matching
this session's instruction that another agent has exclusive use of the NPU
right now). No new thread is needed; 0101's own title (T23 — "the
bfp16-emulation datapath decision") already covers both halves ([`0099`](../../tasks/0099-t26-rounding-ablation/TASK.md)'s
"T23 consequent note" flagged exactly this). This note does not duplicate
that work — it only confirms the audit doesn't turn up anything 0101 isn't
already positioned to answer.

I looked for other RETIRED claims in `research/CLOSED-THREADS.md` resting on
a pre-migration device measurement with a plausible rounding/DCE mechanism,
and found none beyond the two above. The nearby "centred polynomial basis
worth nothing" and `AIE_LOOP_UNROLL_FULL` retirals (note 0007) are compile-
strategy, not rounding-mode, claims — `AIE_LOOP_UNROLL_FULL`'s retirement in
particular concerns Peano's inlining/scheduling decisions, which *could*
plausibly move with a toolchain upgrade, but I found no evidence it did (no
task re-touches it post-migration) and no specific mechanism to name, so per
this task's discipline it is **not flagged** — a stale-but-plausibly-stable
claim is not the same as an evidenced one.

## What would make this checkable without an archaeology session

Nothing in the JIT cache path, the kernel-object filename
(`matmul_bf16_f32_333c4d33.o` — a config hash, not a toolchain hash), the
exported `design.json`, or the `.npue` container header records **which
mlir-aie/Peano version built a given artifact.** This audit took reading
five task logs, diffing objdump text, and correlating `.xclbin` mtimes
against a commit timestamp — exactly the kind of thing this project's own
habit (rule 3/3b, the fail-open list in CLAUDE.md §"Five bugs that failed
open") says should fail closed instead of relying on memory.

**Proposal, not implemented:** have `tools/export_gemm_rtp.py` and
`tools/export_xclbin.py` write a small `toolchain.json` next to each
exported `design.json` (and every ad-hoc `experiments/*/artifacts/` build
that already records one), capturing three cheap, already-available strings:
`pip show mlir_aie` (the version string CLAUDE.md's environment table
already quotes by hand), the Peano package version, and `git rev-parse HEAD`
of `C:\dev\mlir-aie` if that checkout is a git repo (it is, per the update
workflow). `tools/pack_npue.py` copies the `gemm_rtp/toolchain.json` it
finds into the `.npue`'s own header as a provenance field (a string, not
interpreted — the runtime never needs to act on it, only report it). Total
surface: one new file write in each of the two export tools (a handful of
lines each, all data already resident in the process — no new subprocess
calls), one field added to the `.npue` header struct and `npue.py`'s
reader/writer, and one line in `npuembed.exe`'s startup banner to print it.
No behaviour changes. **Cost: under an hour** — the expensive part of this
task was reconstructing provenance after the fact from timestamps and git
log; recording it at build time is three string reads and a JSON write.
This would not have prevented the T26 anomaly (a rebuild is still a
rebuild), but it would have turned this whole audit from a research task
into `jq .toolchain runtime/artifacts_b128il/gemm_rtp/design.json`.
