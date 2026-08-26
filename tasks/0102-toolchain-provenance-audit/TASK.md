# 0102 — Toolchain provenance audit: which claims rest on 1.3.4 builds, and which are actually at risk

- **Date** 2026-08-23
- **Milestone** research (T26 follow-on)
- **Status** done

## Goal

[`0099`](../0099-t26-rounding-ablation/TASK.md) established that the mlir-aie
1.3.4 → 1.4.2 upgrade (`42da31d`) silently changed the code Peano emits for
at least one kernel (the emulated bfp16 matmul), and that nothing in the
build path records which toolchain produced a given object. This task is a
**read-only archaeology audit**: no NPU access, no rebuilding — walk the
project's published claims, work out which ones were measured on 1.3.4, and
of those, which actually depend on what the AIE compiler emitted (as opposed
to host-side work, file formats, or structural/hardware facts) closely
enough that the toolchain change plausibly moved them.

## Context

- `tasks/0099-t26-rounding-ablation/TASK.md` and T26 in
  `research/CLOSED-THREADS.md` — the mechanism and its first evidence.
- `tasks/0058-m11-iron-1.4-migration/TASK.md` — the migration commit
  (`42da31d`, 2026-08-20 16:05:34 +0200) and its own 15-file re-verification
  table, which turned out to be the single most useful input to this audit:
  most of the numbers this task would otherwise have had to treat as
  "old, presumed fine" were **already re-measured on 1.4.2, bit-exact**, in
  that same task.
- `tasks/0059-m11-production-verify-post-migration/TASK.md` — re-verified the
  **runtime and shipped binaries**, not a rebuild.
- `tasks/0060-m11-export-gemm-rtp-marker-fix/TASK.md` — the one place a
  production GEMM shape was actually rebuilt from scratch on 1.4.2 and
  compared against history.
- Scope: **do not run anything on the NPU** (another agent has exclusive use
  of the array this session) and do not rebuild anything. Do not edit
  `research/OPEN-THREADS.md`, `research/CLOSED-THREADS.md`, `tasks/README.md`,
  `CLAUDE.md` or `docs/CURRENT_STATUS.md` — those are the coordinator's.

## What was done

1. Read `0099` and T26's closed-thread entry in full, and confirmed the
   crrnd evidence independently (fresh `grep -c crrnd` against both the
   pre-migration objdump artifacts and today's post-migration rebuild
   artifacts, not taken on the task logs' word).
2. Read `0058` and `0059` in full, specifically for **what each did and did
   not re-verify** — 0058's own 15-file table turned out to already contain
   direct, on-hardware, post-migration reproductions of several headline
   numbers this audit would otherwise have had to flag as merely "old."
3. Established which shipped production `.xclbin`s have actually been
   rebuilt since the migration, by file mtime (these directories are
   gitignored, so `git log` can't answer this) — found the four original
   BERT-family models' production binaries were **never rebuilt** (still
   dated 2026-08-18/19), while nomic, EmbeddingGemma and every int8
   container **were** built post-migration (2026-08-21/22).
4. Read `tasks/0060` in full — the one task that rebuilt a real production
   GEMM shape from scratch on 1.4.2 and independently reproduced its
   pre-migration `1-cos` figure.
5. For each substantive published number (`CLAUDE.md`, `docs/CURRENT_STATUS.md`,
   `research/CLOSED-THREADS.md`), found the task that produced it and its
   commit date (`git log --format="%ci" -1 -- <path>`), and classified it by
   vulnerability class per the task brief (not vulnerable / possibly
   vulnerable / specifically vulnerable — rounding).
6. Specifically re-examined `tasks/0044-m9-optimisation-sweep` Part 3 (the
   source of CLAUDE.md trap 2b's whole rounding table), since it is the
   pre-migration claim that most resembles T26's mechanism. Confirmed the
   task directory holds **only `TASK.md`**, no disassembly, and that no
   pre-migration objdump of any eltwise kernel exists anywhere in the repo
   (checked with a repo-wide search for `crrnd`-containing text files).
   The coordinator flagged mid-task that a rebuild-and-grep here would
   answer the wrong question (today's toolchain is 1.4.2, not 1.3.4) and
   supplied a self-validation argument instead: 0044's own measured
   `floor`-vs-`conv_even` row-sum difference (`floor` max exactly 1.000000
   under any input; `conv_even` straddles it) is only observable if the
   rounding control took effect, so the pre-migration measurement is its
   own proof against elimination, independent of any disassembly. Verified
   the quoted row-sum lines exist in `0044`'s own log at the stated values,
   and verified the structural distinction the argument rests on by reading
   every `aie::set_rounding` call site in `experiments/m5-eltwise/kernels/*.cc`
   — all called directly in our own source adjacent to the vector ops that
   use them, unlike `mm.cc`'s wrap around an `aie_api` header call. Also
   found corroborating evidence already on record: on the *same* 1.3.4-era
   build that lost `mm.cc`'s crrnd triple, `0098` independently found
   `narrow_f32_bf16.o` — same call pattern as the eltwise kernels — **did**
   keep its three `crrnd` writes.
7. Checked the reverse direction (retired claims that might now reopen):
   scanned `research/CLOSED-THREADS.md`'s RETIRED entries for a plausible
   rounding/DCE mechanism connecting them to the upgrade. Found exactly one
   class (T26 itself, plus `--emulate-bfp16`'s 2026-08-18 retirement, which
   rests on the same mechanism) — already flagged in T26's own amendment and
   already being re-measured by an in-progress task, `tasks/0101-t23-bfp16-gates-on-1.4.2`
   (directory exists, `TASK.md` not yet written — confirmed by directly
   listing the directory, not inferred).
8. Wrote the audit up as
   [`research/notes/0009-toolchain-provenance.md`](../../research/notes/0009-toolchain-provenance.md).

## Commands

```powershell
# boundary commit and timestamp
git log --format="%h %ci %s" -1 42da31d

# crrnd evidence, re-run independently rather than trusted from 0098/0099's prose
grep -c crrnd experiments/m5-pretiled-gemm/artifacts/objdump_fp32C_rtp_matmul_bf16_f32_333c4d33.txt
grep -c crrnd experiments/m5-pretiled-gemm/artifacts/objdump_bf16C_rtp_matmul_bf16_f32_333c4d33.txt
grep -c crrnd experiments/m5-pretiled-gemm/artifacts/t26_ablation/objdump_poison_matmul_bf16_f32_333c4d33.txt
grep -c crrnd experiments/m5-pretiled-gemm/artifacts/t26_ablation/objdump_bf16c_matmul_bf16_f32_333c4d33.txt

# repo-wide search for any pre-migration disassembly to grep (found none for eltwise)
find . -iname "objdump*" -not -path "*/.git/*"
find tasks/0044-m9-optimisation-sweep experiments/m5-eltwise -iname "*objdump*"

# whether the eltwise trap-2b table's row-sum evidence is really in 0044's own log
grep -n "row sums\|floor\|conv_even" tasks/0044-m9-optimisation-sweep/TASK.md

# where aie::set_rounding is actually called, to check the structural argument
grep -n "aie::set_rounding\|swap_rounding" experiments/m5-eltwise/kernels/*.cc

# which shipped production xclbins were ever rebuilt post-migration (mtime,
# since these dirs are gitignored)
stat -c '%y %n' runtime/artifacts_b128il/gemm_rtp/final.xclbin
stat -c '%y %n' runtime/artifacts_base/gemm_rtp/final.xclbin
stat -c '%y %n' runtime/artifacts_large/gemm_rtp/final.xclbin
stat -c '%y %n' runtime/artifacts_nomic/gemm_rtp/final.xclbin
stat -c '%y %n' runtime/artifacts_gemma/gemm_rtp/final.xclbin
stat -c '%y %n' runtime/artifacts_int8_mini/gemm_rtp/final.xclbin
# (repeated for artifacts_int8_large, artifacts_int8c_{base,gemma,large,mini,nomic,large_n64})

# export_gemm_rtp.py fix commit, same day as the migration
git show --stat 0c9fc5a

# task-date-vs-boundary comparisons, one per row of the audit table
git log --format="%h %ci %s" -1 -- tasks/0002-m1-hello-npu/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0003-m2-bf16-gemm/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0004-m2-multicore-gemm/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0034-m8-energy/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0035-m8-mteb-gate/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0038-m9-model-driven-runtime/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0039-m9-bge-small/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0040-m9-honest-cpu-baseline/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0042-m9-bge-large/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0044-m9-optimisation-sweep/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0046-m9-b-reuse-asymmetric/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0047-m9-cascade-channel-probe/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0052-m10-research-night/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0077-m13-int8-gate/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0078-m13-int8-accuracy/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0081-m13-int8-everywhere/TASK.md
git log --format="%h %ci %s" -1 -- tasks/0085-m13-release-sweep/TASK.md

# whether the re-open candidate is already claimed by another in-progress task
ls tasks/0101-t23-bfp16-gates-on-1.4.2
```

## Result

Examined roughly **20 substantive published claims/tables** across
`CLAUDE.md`, `docs/CURRENT_STATUS.md` and `research/CLOSED-THREADS.md`. Full
per-claim table, evidence and reasoning in
[`research/notes/0009-toolchain-provenance.md`](../../research/notes/0009-toolchain-provenance.md).
Summary by class:

- **Not vulnerable (host-side, file-format, or already re-measured
  post-migration)**: the large majority — energy, tokenizer, CPU baselines,
  packing round-trips, every task committed on or after 2026-08-20
  (post-migration: int8 gate/accuracy/everywhere, the 0.4.0 release sweep,
  attention geometry, and every T-thread closed 2026-08-23), plus SAXPY's
  1,617× and the M2 GEMM headline (25.0→137.3 MACs/cyc) — both pre-migration
  but **directly re-verified bit-exact on 1.4.2 in `0058`**.
- **Possibly vulnerable, checked and cleared**: production `1-cos`/MTEB
  figures for the four original BERT models. Their shipped `.xclbin`s were
  **never rebuilt** (confirmed by mtime, still 2026-08-18/19) so the
  published numbers describe the binary actually running today, re-verified
  bit-identical to history in `0059`; separately, `0060` rebuilt MiniLM's
  real production shape from scratch on 1.4.2 and reproduced the identical
  figure — doubly confirmed, not merely un-rebuilt.
- **Specifically vulnerable, genuinely expired**: exactly the two claims
  that share T26's actual mechanism — the 6.6× bfp16+bf16-C accuracy
  advantage (`0052`/`0053`/`0056`) and `--emulate-bfp16`'s 2026-08-18
  retirement at `1-cos` 3.470e-03 (`0035`). Both already flagged by T26's
  own register amendment; both being re-measured right now by
  `tasks/0101-t23-bfp16-gates-on-1.4.2` (directory exists, in progress —
  confirmed, not duplicated here).
- **The one row that most resembled a hidden landmine and turned out not to
  be one**: CLAUDE.md trap 2b's GELU/softmax/LayerNorm rounding table
  (`0044`, pre-migration, the same rounding-mode mechanism class as T26).
  Cleared on two independent grounds: `0058` already reproduced every figure
  bit-exact on 1.4.2 hardware, and the elimination mechanism T26 found (an
  `aie_api` header call reaching hardware through an ambient intrinsic) is
  structurally absent from these kernels, whose `aie::set_rounding` calls
  sit directly in our own source next to the vector ops they gate —
  corroborated by `0098`'s own finding that `narrow_f32_bf16.o` (same call
  pattern) kept its `crrnd` writes on the very build that lost `mm.cc`'s.
  0044's own row-sum measurement (`floor` max exactly 1.000000, `conv_even`
  straddling it) is further, self-validating proof the control took effect
  on that build.

No new register thread is needed. Provenance proposal (priced, not
implemented): a `toolchain.json` written by `export_gemm_rtp.py`/
`export_xclbin.py` at build time (`mlir_aie` version, Peano version,
`C:\dev\mlir-aie`'s git HEAD), copied into the `.npue` header by
`pack_npue.py` and printed in the runtime's startup banner — three string
reads and a JSON write, no new subprocess calls, **under an hour of work**.
Full reasoning and cost breakdown in the note.

## Problems hit

None — this was a read-only session and every check either confirmed or
refuted cleanly. The one course correction: the coordinator flagged mid-task
that I was about to hunt for a pre-migration eltwise-kernel disassembly that
does not exist in the repo (and that rebuilding today would answer the wrong
question, since today's toolchain is 1.4.2), and supplied the self-validation
argument used in the "trap 2b" section above. Verified rather than taken on
trust: confirmed `tasks/0044-m9-optimisation-sweep/` really does hold only
`TASK.md`, confirmed the quoted row-sum lines exist in `0044`'s own log at
the stated values, and confirmed the structural distinction (direct
source-level `set_rounding` calls vs. `mm.cc`'s ambient-intrinsic header
call) by reading the actual kernel source rather than accepting the
characterisation.

## Artifacts

- `research/notes/0009-toolchain-provenance.md` — the audit itself: the
  mechanism, the boundary, the full claim table, the re-open candidates and
  the provenance proposal.
- This file.
- Nothing else was created or modified; no NPU access; no rebuilds.

## Next

- `tasks/0101-t23-bfp16-gates-on-1.4.2` is already re-measuring the one
  genuinely expired claim class (T26 / `--emulate-bfp16`) — not this task's
  job to duplicate or pre-empt.
- The provenance proposal (a `toolchain.json` sidecar) is priced but not
  built — a natural small follow-up whenever someone next touches
  `export_gemm_rtp.py`/`export_xclbin.py`/`pack_npue.py`.
- `docs/CURRENT_STATUS.md` §7's "How to build" prerequisites table still
  says `mlir-aie 1.3.4` (line ~560) — a stale-documentation bug unrelated to
  any accuracy claim, noticed in passing while cross-referencing the
  environment table against `CLAUDE.md`'s (which correctly says 1.4.2). Not
  fixed here — `docs/CURRENT_STATUS.md` is out of this task's edit scope —
  flagged for whoever next touches that file.

## Proposed register update

No thread needs amending beyond what T26's own 2026-08-23 amendment already
says, and no new thread should be filed — `tasks/0101-t23-bfp16-gates-on-1.4.2`
already exists and already covers both halves of the one genuinely expired
claim class this audit found (T26's bf16-C/fp32-C comparison and
`--emulate-bfp16`'s retirement, which share the same mechanism).

`tasks/README.md` index row:

```
| [0102](0102-toolchain-provenance-audit/TASK.md) | Read-only audit of every published claim against the mlir-aie 1.3.4 -> 1.4.2 boundary (`42da31d`) T26 exposed: ~20 claims examined, classified by whether the toolchain change could plausibly have moved them (host-side / file-format claims can't; rounding-mode-dependent AIE codegen can). Most are cleared by direct evidence already on record -- `0058` re-verified SAXPY, the M2 GEMM headline and CLAUDE.md trap 2b's whole rounding table bit-exact on 1.4.2 hardware, and `0060` independently rebuilt a real production GEMM shape from scratch post-migration and reproduced its historical `1-cos`. The four original BERT models' shipped `.xclbin`s were never rebuilt at all (confirmed by mtime); nomic/Gemma/int8 were all built post-migration. Exactly two claims are genuinely expired -- T26's 6.6x accuracy anomaly and `--emulate-bfp16`'s 2026-08-18 retirement, both sharing the mechanism T26 found -- and both are already being re-measured by the in-progress `0101`. Wrote `research/notes/0009-toolchain-provenance.md` and priced a `toolchain.json` build-provenance sidecar at under an hour, not implemented | research | done |
```
