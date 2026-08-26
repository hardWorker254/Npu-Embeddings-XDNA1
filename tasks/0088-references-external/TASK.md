# 0088 — References: external XDNA/NPU projects to spy on

- **Date** 2026-08-23
- **Milestone** M13 (research)
- **Status** done

## Goal

Build `research/references-external.md`: a references document, in the style
of `research/prior-art.md`, surveying public XDNA/NPU repositories for
architecture, tricks and traps — read-only, never vendored. Original scope
was five named repos (`open-xdna`, `hawkpoint-npu-llm`, `whisper-xdna`,
`phoenix-sdr-dsp`, `phoenix-npu-pqc`); grew mid-session to eight (`npu-linux-kit`
found alongside `open-xdna`; `npuhalo` and `Meeting-Ops-UC1-OSS` added by the
user). Must be consistent with and cross-link
[note 0007 §3](../../research/notes/0007-unused-iron-surface.md), not
duplicate or contradict it.

## Context

`research/prior-art.md` §7 already surveyed four of these eight repos in
2026-08-19 ([`0044`](../0044-m9-optimisation-sweep/TASK.md)), at a shallower
read (no pinned commit recorded for at least one clone). This task re-reads
all eight at commits pinned by the coordinator, from local clones under
`externalrepos/` (gitignored), using `Read`/`Grep`/`Glob` on the working tree
rather than `WebFetch`/`WebSearch` on GitHub — the coordinator redirected the
task mid-flight from web-fetch to local-clone reading once the clones existed,
specifically because a full local read answers "XDNA1 vs XDNA2", "hand-written
kernels vs vendor overlay", "actual dispatch path", and "verified license" far
more reliably than README prose.

## What was done

1. Loaded `WebFetch`/`WebSearch` via `ToolSearch` and did an initial pass with
   `WebFetch` on all five originally-named repos' GitHub pages, `BENCHMARKS.md`,
   `README.md`, and the GitHub REST API (`api.github.com/repos/...`) for
   metadata (license, pushed_at, stars). This pass is superseded by the local
   read below wherever the two disagree (see Problems hit).
2. Read `CLAUDE.md`, `research/README.md`, `research/prior-art.md`,
   `research/OPEN-THREADS.md`, `research/notes/0007-unused-iron-surface.md`,
   and `tasks/README.md`'s template/format before writing anything.
3. Coordinator redirected: all repos are cloned locally under
   `externalrepos/<name>` at pinned commits (table below). Re-did the survey
   by reading the actual trees: `git log -1` to confirm each HEAD, `Glob` for
   directory structure, targeted `Grep` across all clones for
   `aie2p|npu2|strix`, `aie2|npu1|phoenix|hawk`, `runlist`, `compress`,
   `set_rounding`, `pad_dimensions`, `cascade`, `xrt::|aiebu|onnxruntime|vitis`,
   `set_current_device`, and reading `LICENSE` files directly (never trusting
   GitHub's license-detection API, which was wrong for two of eight —
   `hawkpoint-npu-llm` and `whisper-xdna` both report `NOASSERTION` via the
   API despite unambiguous LICENSE files).
4. Two more repos added mid-session (`npuhalo`, `Meeting-Ops-UC1-OSS`); read
   the same way. `Meeting-Ops-UC1-OSS` (886 files) was scoped deliberately to
   the embedding/NPU-serving path per the coordinator's steer, not read in
   full — `backend/services/embedding_service.py`,
   `backend/npu_optimization/aie2_kernel_driver.py`,
   `backend/docs/NPU_HARDWARE_EVIDENCE.md`, `backend/npu_final_findings.md`,
   and the top-level/`backend/` `CLAUDE.md` status docs.
5. Wrote `research/references-external.md` (8 sections + intro + bottom
   line), added a pointer row to `research/README.md`'s file-tree listing,
   and added a "See also" cross-link at the top of `prior-art.md` §7.
6. Did **not** edit `tasks/README.md` or `research/OPEN-THREADS.md` /
   `CLOSED-THREADS.md` — per the coordinator's concurrent-agents notice,
   proposed changes are recorded below instead.

## Commands

```powershell
# Confirm each clone's pinned commit (repeated per repo, one shown):
git -C externalrepos/open-xdna log -1 --format='%H %ai'
git -C externalrepos/npu-linux-kit log -1 --format='%H %ai'
git -C externalrepos/hawkpoint-npu-llm log -1 --format='%H %ai'
git -C externalrepos/whisper-xdna log -1 --format='%H %ai'
git -C externalrepos/phoenix-sdr-dsp log -1 --format='%H %ai'
git -C externalrepos/phoenix-npu-pqc log -1 --format='%H %ai'
git -C externalrepos/npuhalo log -1 --format='%H %ai %s'
git -C externalrepos/Meeting-Ops-UC1-OSS log -1 --format='%H %ai %s'
```

Reading was done with `Read`/`Grep`/`Glob` (no shell greps beyond the `git
log` calls above), against the paths listed under Artifacts.

## Result

`research/references-external.md` written, covering, per repo: what it is,
target hardware (verified by grep for `aie2p`/`npu2`/`strix` vs
`aie2`/`npu1`/`phoenix`/`hawk` across the whole tree, not assumed from the
name), how it talks to the NPU, license (verified from the `LICENSE` file
byte content, not GitHub's detector), what's worth stealing, what it warns
us about, and relevance to `OPEN-THREADS.md`.

**Headline findings, in priority order:**

1. **XDNA2 is almost entirely absent from this ecosystem.** Six of eight
   repos are XDNA1 outright. `phoenix-sdr-dsp` has exactly one `npu2`/`aie2p`
   design (`docs/M17_V3_DESIGN.md`, a 64-point FFT portability probe, not
   shipped). `npuhalo` is the one repo genuinely on our silicon generation
   (Strix Halo, 48 AIE2P tiles vs our Strix Point's 32) but drives it
   entirely through FastFlowLM as a product, never IRON/XRT.
2. **`npuhalo` gives a directly measured NPU/iGPU DRAM-contention number on
   XDNA2 shared memory**: concurrent NPU decode work costs a co-resident GPU
   generator **−16.5% decode throughput**, reproduced by both a synthetic
   DRAM-traffic proxy (−17.2%) and a real FastFlowLM load (−16.5%, two
   methods agreeing to one point). Not directly reusable as a number (our
   runtime never shares a bus with a GPU generator) but the first
   same-generation evidence in this survey of what NPU DMA traffic costs a
   co-resident consumer.
3. **`Meeting-Ops-UC1-OSS`'s "NPU-accelerated" embedding service does not
   appear to reach the NPU.** `NPUEmbeddingService` wraps
   `all-MiniLM-L6-v2-int8.onnx` (our own model) via
   `onnxruntime.InferenceSession(providers=['DmlExecutionProvider',
   'CPUExecutionProvider'])` — DirectML, a Windows path, on a repo whose own
   hardware docs describe a Linux/`amdxdna` deployment; on that platform the
   session would fall through to `CPUExecutionProvider`. Separately, the
   repo's Whisper "220x speedup, confirmed" NPU claims trace to
   `aie2_kernel_driver.py`'s own comment *"Generate a mock XCLBIN"* and a
   benchmark harness seeded with `np.random.randint(...)` rather than real
   audio — a concrete, code-level instance of the failure mode `CLAUDE.md`
   rules 1 and 6 exist to prevent (self-reported "100% Complete" status docs
   that contradict the code and each other; `npu_final_findings.md`, written
   later, admits the ONNX/VitisAI path never got x86_64 Linux wheels and the
   project fell back to CPU).
4. **`whisper-xdna` (re-read at a pinned commit) yields facts note 0007
   §3 did not have**: `rawxrt.py` bypasses IRON's Python dispatch wrapper for
   ~1.4×, corroborating this project's own C++-runtime architecture choice;
   `pyxrt.runlist` is documented by MLIR-AIE's own examples as NPU2-only but
   they measured it working on NPU1 too (mild positive signal for
   [T9](../../research/OPEN-THREADS.md#t9), since we are the documented
   target generation); a second, independent design-switch-cost measurement
   (2.81 ms per pair of switches, different hardware and methodology from
   our own [note 0004](../../research/notes/0004-context-switch-cost.md));
   naive per-channel int8 passing only 21/27 tests at cosine 0.78–0.93 —
   corroborating our own closed [T20](../../research/CLOSED-THREADS.md#t20)
   (naive W8A8 failed our gate too, fixed only by SmoothQuant calibration);
   and a wording discrepancy between their README ("truncates toward zero")
   and our own `CLAUDE.md` rule 2b ("floor"/"round towards negative
   infinity") for the default AIE rounding mode — flagged, not resolved
   (the two descriptions agree for positive values, which is all our own
   confirming test exercised).
5. **`hawkpoint-npu-llm`'s scalar-reduction anti-pattern confirmed at the
   exact lines** (`qwen_decoder_layer_bf16.cc`, a `switch((col/32)&3)`
   vectorised partial-accumulate followed by a scalar `for` loop over 32
   lanes) — note 0007 already flagged this repo but had paraphrased rather
   than quoted the code.
6. **`phoenix-sdr-dsp`'s M17 FFT design independently confirms two `CLAUDE.md`
   facts** about our own toolchain (bf16 `mac_dims = (4,8,8)` on aie2p; Peano
   defines `__AIECC__` for the aie2p target) from unrelated kernel code.
7. **`phoenix-npu-pqc`'s failure-preservation discipline** (byte-preserved
   retry logs for a 0/25 physical failure, an explicit "canonical silicon
   validation" vs. "diagnostic" boundary) is the closest philosophical match
   in this survey to this project's own rule 3b.

**License audit (verified from `LICENSE` file bytes, not GitHub's API):**
`open-xdna` and `npu-linux-kit` (Scottcjn) and `Meeting-Ops-UC1-OSS`
(Unicorn-Commander) are **AGPLv3** — do not vendor. `hawkpoint-npu-llm` is
Apache-2.0 WITH LLVM-exception (GitHub API wrongly reports `NOASSERTION`).
`whisper-xdna` is MIT (GitHub API also wrongly reports `NOASSERTION`).
`phoenix-sdr-dsp`, `phoenix-npu-pqc`, and `npuhalo` are Apache-2.0.

## Problems hit

1. **The web-fetched GitHub API license field was wrong for two repos** —
   `NOASSERTION` for both `hawkpoint-npu-llm` and `whisper-xdna`, whose actual
   `LICENSE` files are unambiguous (Apache-2.0-with-LLVM-exception and MIT
   respectively). Cause: GitHub's license detector apparently chokes on
   `hawkpoint-npu-llm`'s LLVM-derived preamble text and, for reasons not
   determined, on `whisper-xdna`'s otherwise-standard MIT file. Fixed by
   reading the file bytes directly once local clones were available — this
   is the reason the coordinator's mid-task redirect from `WebFetch` to local
   reading mattered, not just a style preference.
2. **Initial `WebFetch` summaries (before the local-clone redirect) were
   noticeably shallower and in at least one place imprecise** — e.g. the
   first-pass `hawkpoint-npu-llm` fetch described its license only via the
   API's wrong `NOASSERTION`/a generic "Apache License 2.0 with LLVM
   Exceptions" paraphrase pulled from README prose, and none of the
   first-pass fetches found the `qwen_decoder_layer_bf16.cc` scalar-reduction
   code, the `whisper-xdna` `rawxrt.py`/`runlist` internals, or any of the
   `phoenix-sdr-dsp` `M17_V3_DESIGN.md` `aie2p` confirmation — all of which
   only surfaced from reading the actual tree. Superseded, not merged; the
   final document is written entirely from the local-clone read.
3. **`Meeting-Ops-UC1-OSS` is 886 tracked files with extensive, sometimes
   self-contradictory status documentation** (top-level and `backend/`
   `CLAUDE.md` claim "NPU Integration: 100% Complete" and "220x speedup
   confirmed"; `backend/npu_final_findings.md`, apparently written after,
   documents the ONNX/VitisAI path never working on Linux x86_64 and falling
   back to CPU). Resolved by treating the repo's own status prose as
   unverified and reading the code the claims point at instead
   (`aie2_kernel_driver.py`'s "mock XCLBIN" comment and
   `np.random`-seeded benchmark; `embedding_service.py`'s
   `DmlExecutionProvider` config) — the finding written up is about what the
   code does, not what the docs assert. Did not read the full repo; scoped to
   the embedding/NPU-serving path per the coordinator's steer, so a
   correctness claim about `WhisperX`/diarization elsewhere in the repo is
   not made either way.
4. **No repo in the survey directly answers a numbered `OPEN-THREADS.md`
   thread outright.** Several corroborate already-**closed** findings
   (T20's int8-needs-calibration, T22's mlir-aie 1.4 migration pain,
   established `aie2p` `mac_dims`/`__AIECC__` facts) or give mild,
   non-decisive signal on live ones (T9). This is reported plainly rather
   than stretched into a false "answers T-whatever" claim.

## Artifacts

- `research/references-external.md` — new, the deliverable.
- `research/README.md` — one-line pointer added to the file-tree listing.
- `research/prior-art.md` — "See also" cross-link added at the top of §7.
- `externalrepos/{open-xdna,npu-linux-kit,hawkpoint-npu-llm,whisper-xdna,
  phoenix-sdr-dsp,phoenix-npu-pqc,npuhalo,Meeting-Ops-UC1-OSS}/` — pre-existing
  local clones this task read from; gitignored, not created by this task, not
  modified.
- This file.

All checked in except `externalrepos/` (gitignored by design, per the
coordinator's instructions and `CLAUDE.md`'s read-only-reference-tree
convention).

## Next

Nothing this task unblocks directly — it is a reference document, not a
build. The clearest forward pointers it leaves: `npuhalo`'s "Block FP16"
mention as an unverified lead for whoever next touches
[T23](../../research/OPEN-THREADS.md#t23)/[T26](../../research/OPEN-THREADS.md#t26);
and `phoenix-sdr-dsp`'s `THIRD_PARTY_PROVENANCE.md` hash-ledger pattern as a
possible (not urgent) improvement to how this project tracks which of its own
files derive from mlir-aie's example kernels, per `CLAUDE.md` rule 4.

## Proposed register update

`tasks/README.md` index row (do not apply — another agent owns that file
this session):

```markdown
| [0088](0088-references-external/TASK.md) | **External XDNA/NPU repo survey — `references-external.md`**, eight repos read at pinned commits: XDNA2 is nearly absent from this whole ecosystem (one FFT probe, one FastFlowLM-layer sibling on our own generation); `npuhalo` measures **−16.5%** iGPU decode cost from concurrent NPU DMA traffic on shared XDNA2 memory; `Meeting-Ops-UC1-OSS`'s "NPU-accelerated" MiniLM embedding service configures `DmlExecutionProvider` on a Linux target and its Whisper "220x speedup" traces to a self-described "mock XCLBIN" and `np.random`-seeded benchmark — a concrete outside instance of the failure mode `CLAUDE.md` rules 1/6 guard against; two repos' GitHub-reported licenses (`NOASSERTION`) were wrong, confirmed Apache-2.0/MIT from the LICENSE bytes; three repos (incl. one newly found, `npu-linux-kit`) are AGPLv3 — do not vendor | M13 | done |
```

No `research/OPEN-THREADS.md` thread is newly answered or reopened by this
survey, so no thread-status amendment is proposed. Two threads gain
corroborating (not decisive) evidence, worth a footnote if/when either is
next touched by hand rather than as a standalone edit:

- **[T9](../../research/OPEN-THREADS.md#t9)** (`xrt::runlist`) — `whisper-xdna`
  measured `pyxrt.runlist` working on NPU1 despite MLIR-AIE's own examples
  documenting it NPU2-only; mild positive signal, not new evidence, since we
  are already the documented target generation.
- **[T23](../../research/OPEN-THREADS.md#t23)/[T26](../../research/OPEN-THREADS.md#t26)**
  (bfp16 emulation / accuracy) — `npuhalo`'s `docs/final_verdict.md` §2 names
  *"Block FP16 — XDNA 2 hardware 8+1-bit float format"* as an AMD research
  path it declined to chase for an unrelated (decode-side) reason. Possibly
  the same feature this project calls bfp16, described from a different
  angle — **unverified, no source named by `npuhalo` itself**, flagged as a
  lead only.
