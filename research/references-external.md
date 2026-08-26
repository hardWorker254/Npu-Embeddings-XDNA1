# References — external XDNA/NPU projects to spy on

Eight public repositories building AIE/XDNA software with the same open stack
this project uses (XRT + MLIR-AIE/IRON + Peano) or product-level stacks built
on top of it (FastFlowLM, ONNX Runtime), read for architecture, tricks and
traps — the way `prior-art.md` already reads FastFlowLM. None
is vendored; all are read-only reference material per `CLAUDE.md` rule 4's
spirit (that rule names FastFlowLM specifically, but the same discipline
applies here, sharpened by three of these eight being **AGPL-3.0** — see
§1/§2/§8).

**Method.** All eight were cloned to `externalrepos/` (gitignored, never
committed) and read from the working tree with `Read`/`Grep`/`Glob` — not from
README prose or a web fetch. Every claim below is pinned to the exact commit
read; re-clone and `git log -1` to check it still says what is quoted here.

| repo | path | HEAD | pinned |
|---|---|---|---|
| [`Scottcjn/open-xdna`](https://github.com/Scottcjn/open-xdna) | `externalrepos/open-xdna` | `6c8c680` | 2026-08-05 |
| [`Scottcjn/npu-linux-kit`](https://github.com/Scottcjn/npu-linux-kit) | `externalrepos/npu-linux-kit` | `edee964` | 2026-08-05 |
| [`c8dhjp4tyv-bit/hawkpoint-npu-llm`](https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm) | `externalrepos/hawkpoint-npu-llm` | `12a8448` | 2026-08-02 |
| [`drakosha/whisper-xdna`](https://github.com/drakosha/whisper-xdna) | `externalrepos/whisper-xdna` | `d43cb82` | 2026-07-27 |
| [`midhatn/phoenix-sdr-dsp`](https://github.com/midhatn/phoenix-sdr-dsp) | `externalrepos/phoenix-sdr-dsp` | `67f0673` | 2026-08-18 |
| [`midhatn/phoenix-npu-pqc`](https://github.com/midhatn/phoenix-npu-pqc) | `externalrepos/phoenix-npu-pqc` | `c4acbc5` | 2026-08-18 |
| [`julianmb/npuhalo`](https://github.com/julianmb/npuhalo) | `externalrepos/npuhalo` | `a34fd36` | 2026-08-23 |
| [`Unicorn-Commander/Meeting-Ops-UC1-OSS`](https://github.com/Unicorn-Commander/Meeting-Ops-UC1-OSS) | `externalrepos/Meeting-Ops-UC1-OSS` | `cf1aec3` | 2026-07-30 |

`npu-linux-kit` was not in the original request — it was sitting alongside
`open-xdna` in `externalrepos/`, is by the same author, shares a commit
message pattern, and is explicitly built on `open-xdna`'s bring-up (its own
README: *"Built on the open-xdna gen-1 (XDNA1 / Phoenix) bring-up"*).
`npuhalo` and `Meeting-Ops-UC1-OSS` were added to the request mid-session.
All three are covered as full sections below, not footnotes.

**The one fact that matters more than any other in this document: XDNA2 is
almost entirely absent from this whole ecosystem.** Six of the eight target
XDNA1 (Phoenix/Hawk Point, `npu1`, AIE2, 4 columns), not XDNA2 (Strix,
`npu2`, AIE2P, 8 columns), the silicon this project runs on.
`hawkpoint-npu-llm`'s own `SUPPORT.md` states it plainly: *"XDNA2 / Strix
Point | No | Unsupported."* `open-xdna`'s `BENCHMARKS.md`: *"this is XDNA1
(Phoenix/Hawk Point) only. XDNA2 (Strix Point) users won't directly benefit
without porting — the kernels target the aie2 array; tile/ISA details
differ."* `Meeting-Ops-UC1-OSS` is also XDNA1 (Phoenix, §8). `phoenix-sdr-dsp`
has exactly one `npu2`/`aie2p`-targeted design (§5, a 64-point FFT
portability probe, not a GEMM, not shipped). **`npuhalo` (§7) is the one
genuine exception** — it runs on AMD Ryzen AI Max+ 395 "Strix Halo", XDNA2,
48 AIE2P tiles, the same silicon generation as this project (a larger sibling
of our Ryzen AI 9 HX 370 "Strix Point", 32 tiles) — but it drives the NPU
entirely through **FastFlowLM** (this project's own reference architecture,
`prior-art.md` §1) as a product, never through IRON or a
hand-written kernel. **Read every number in §1–§6 and §8 as "on XDNA1, mostly
transfers as architecture and warnings, not as absolute numbers"** —
consistent with this project's own framing (`CLAUDE.md`'s task brief: *"NPU1
results often do not transfer"*). §7's numbers are on our own generation but
at the FastFlowLM product layer, not the IRON/XRT layer this project builds
at.

Three of six (`whisper-xdna`, `hawkpoint-npu-llm`, `phoenix-sdr-dsp`) were
already surveyed in
[note 0007 §3](notes/0007-unused-iron-surface.md#part-3--from-the-outside-repos),
written 2026-08-19 from an earlier, shallower read (in `whisper-xdna`'s case,
possibly a different clone — that note does not record a pinned commit).
This document does not restate note 0007's findings in full; it points to
them and adds what a full local read of the pinned commits above turned up
that note 0007 did not have.

---

## 1. `Scottcjn/open-xdna` — the XDNA1 Linux bring-up nobody else did

**What it is.** An open-source Linux driver/software bring-up for AMD's
first-generation XDNA1 NPU (Phoenix/Hawk Point, Ryzen 7040/8040), which AMD
does not officially support outside XDNA2. Its own `llms.txt`: *"This is the
NPU generation that FastFlowLM and AMD Lemonade do NOT support (they are
XDNA2 / Strix only)."* 27 stars, actively maintained (`.github/workflows/ci.yml`
present), created 2026-06-24, HEAD `6c8c680` (2026-08-05).

**License — read from the file, not the README.** `LICENSE` is the actual
**GNU AGPLv3** text (confirmed: *"GNU AFFERO GENERAL PUBLIC LICENSE, Version
3"*), with a separate `COMMERCIAL.md` offering a paid non-copyleft
alternative. AGPL is materially stronger copyleft than this project's own
Apache-2.0 — it triggers on network use, not just distribution. **Do not
vendor or adapt any code from this repo or from `npu-linux-kit` (§2, same
license) into this project.** Read-only, same as the FastFlowLM rule but for
a different reason (AGPL vs. their closed-binary terms).

**How it talks to the NPU.** XRT direct (`libxrt_driver_xdna.so` + a matched
staging `amdxdna.ko` kernel module), MLIR-AIE/IRON for kernel compilation,
Peano, and hand-written AIE2 vector kernels (`examples/kernels/*.cc`:
`collapse.cc`, `compact.cc`, `pse_collapse.cc`, `shuffle_demo.cc`). Four
undocumented bring-up requirements its `docs/BRINGUP.md` and
`docs/UPSTREAM_amdxdna_ioctls.md` record: the full XRT driver library (not
just core components), the matched staging kernel module, `llvm-objcopy`
availability for AIE2 ELF parsing, and firmware-version matching (stale
firmware aborts with `ERT_CMD_STATE_ABORT` / mailbox `-22`). None of this
applies to us directly (native Windows, XDNA2, XRT already working per
`CLAUDE.md`'s environment table) but it is the same *class* of "traps that
cost an hour" this project keeps its own list of.

**What is worth stealing.**
- **The honesty, as a model to imitate.** `BENCHMARKS.md`, read directly:
  *"the NPU is not a fast matmul engine — measured, it's ~6x slower than the
  integrated Radeon 780M at dense GEMM and loses on energy-per-GFLOP for
  dense work."* This is a project publishing a negative headline result about
  its own target hardware, in the same spirit as this project's rule 3b.
- **`docs/POWER_INSTRUMENTATION.md`'s differential power methodology** pairs
  DPM power-state (not watts) with real package-power RAPL deltas, and is
  explicit about what it refuses to claim: *"There is no NPU-specific RAPL
  domain, so NPU power is not directly isolable. We report ... delta, never a
  fabricated per-NPU wattage."* Measured: running the NPU raises package
  power **~2.9 W**, of which **~98% lands in `rest` (+2.81 W), not the x86
  cores (+0.07 W)**. This is the same differential-measurement discipline
  this project's own energy task
  ([`0034`](../tasks/0034-m8-energy/TASK.md)) used independently (two encode
  counts, subtract, immune to idle drift) — an independent arrival at the
  same ethic `CLAUDE.md` rule 1 states for us.
- **512×512 int16 matmul at ~68 GFLOP/s, bit-exact**, and NPU matmul measured
  at **+6.6 W** against the iGPU's **+25.9 W** for the same work
  (`BENCHMARKS.md`) — a sanity-check pair of numbers for "the NPU trades
  raw throughput for power," on different silicon and a different dtype than
  ours, not directly comparable to any number in this project.

**What it warns us about.** The pruning-based value proposition (FFN-column
and attention-KV pruning to cut matmul work, `examples/npu_ffn_prune.py`,
`npu_layer_prune.py`) nets **1.27–1.60×** at accuracy costs (cosine
0.958–0.999) that would not clear this project's 2e-3 gate without
verification — not applicable to us (we do not prune), but a reminder that a
"×" headline number needs its accuracy column read next to it, which is
exactly this project's own MTEB-gate discipline.

**Relevance to our open threads.** None directly. It corroborates the ethic
behind `CLAUDE.md` rule 1 (already established, not open) and nothing else.

---

## 2. `Scottcjn/npu-linux-kit` — found alongside `open-xdna`, not requested

**What it is.** A catalogue of practical NPU-suited Linux modules (camera
effects, embeddings, audio, a ggml/llama.cpp backend), explicitly built on
`open-xdna`'s bring-up: *"Built on the open-xdna gen-1 (XDNA1 / Phoenix)
bring-up."* Same AGPLv3 + commercial dual license, **verified from the
`LICENSE` file** (identical AGPLv3 header to §1) — same do-not-vendor rule
applies. HEAD `edee964` (2026-08-05).

**Its `embeddings/` module is this project's exact problem statement, and it
remains unbuilt.** Read directly from `embeddings/README.md`:

> "Batch sentence/RAG embedding generation on the NPU. **Status: candidate,
> not built.** Rationale: embeddings are matmul-heavy and *batchable*, so
> they amortize the NPU's dispatch overhead toward its ~64 GFLOP/s peak
> (measured in open-xdna) and offload the CPU/iGPU at ~6.6 W. Needs: an
> embedding model's matmuls expressed via IRON, + a server interface."

This is `CLAUDE.md`'s F1 (dispatch overhead dominates) and F2 (batching is
mandatory) stated independently by a third party, and it is still a stub —
one paragraph, no code. The only *built and measured* module is
`camera/` (edge-stylize effects, 220 FPS @1080p / 451 FPS @720p including
host DMA round-trip, ~6.6 W, camera-capture-bound not NPU-bound); `audio/`
and `ggml-backend/` are "candidate (untested)" and "planned (biggest lift)"
respectively — its own README table is explicit about which is which,
matching this project's own "say what actually works" discipline.

**Relevance to our open threads.** None directly — reconfirms (does not
newly establish) F1/F2, and reconfirms nobody has published a working NPU
embedding server on this whole family of open-stack projects, matching
`prior-art.md`'s existing bottom line.

---

## 3. `c8dhjp4tyv-bit/hawkpoint-npu-llm` — the fused-decoder-layer precedent

**What it is.** Qwen2.5-0.5B / SmolLM2-135M decoder inference on Hawk Point
XDNA1, "padlocked" to that generation per its own `README.md`: *"Padlocked to
XDNA1 (`npu1`, AIE2, 4 columns). XDNA2/NPU4 silicon (Strix Point, Strix
Halo — 8 columns, larger local memory, shared caches) is not targeted."*
`SUPPORT.md`'s matrix states the same thing as a table row: *"XDNA2 / Strix
Point | No | Unsupported."* Its own `ROADMAP.md` describes XDNA2 as *"AIE4
tiles with 8 columns"* — worth flagging as a probable naming mismatch with
this project's own terminology (`aie2p`/AIE-ML v2, per `CLAUDE.md`'s
toolchain table), not independently verified either way, and not something
this document resolves. License **Apache-2.0 WITH LLVM-exception**, verified
directly from `LICENSE` (GitHub's own API reports `NOASSERTION` for this
repo — a detection artefact of the file's LLVM-derived preamble, not a real
ambiguity; the file itself is unambiguous). HEAD `12a8448` (2026-08-02).

**How it talks to the NPU.** XRT + MLIR-AIE/IRON + Peano, hand-written AIE2
kernels (`npu_llm/kernels/*.cc`) driven by IRON graph definitions
(`npu_llm/designs/*.py`, using `set_current_device(from_name("npu",
n_cols=4))` — the same IRON device-selection call this project's own
`CLAUDE.md` trap 1 warns about, here always called explicitly).

**What is worth stealing.**
- **The fused whole-decoder-layer kernel** (`decoder_layer_bf16.cc`,
  `qwen_decoder_layer_bf16.cc`) is architecturally the shape `CLAUDE.md`'s
  **F1** points at (fuse whole layers into one dispatch) — already noted in
  [note 0007 §3](notes/0007-unused-iron-surface.md), read here for
  architecture, not technique (see next item for why).
- **`docs/PERFORMANCE-ANALYSIS.md`'s refusal to attribute cost to a phase**
  is a close match to this project's own rule 1. Read directly: *"This
  document distinguishes measured repository evidence from static model
  arithmetic. It does not present uninstrumented phase timings ... as
  measurements."* It reports **274 ms/token** (3.65 tok/s at 32-token
  streaming) and explicitly declines to split that into dispatch/AIE/host —
  the same discipline this project applies to wall clock (`CLAUDE.md` rule
  1), independently arrived at.
- **A rigorous, hardware-gated release CI** (`SUPPORT.md`): the tag-triggered
  release workflow requires a labeled physical Hawk-Point runner to pass a
  pinned-model conversion, a 1,000-completion soak, and a digest-matched
  Ollama round trip before the SBOM/signing/GitHub-release job runs at all
  (`needs: hawk-point`); every third-party Action is pinned to a full commit
  SHA. Not urgent for us, but a genuinely stronger fail-closed release gate
  than anything in this project's own `tools/release_benchmark.ps1`-driven
  process — worth a look next time the `release` skill is revisited.

**What it warns us about — confirmed at the exact lines.** `note 0007` already
flagged a "scalar reduction + hot-loop switch" anti-pattern in
`qwen_decoder_layer_bf16.cc`; read directly at the pinned commit, it is:

```cpp
switch ((col / 32) & 3) {
case 0: total0 = aie::add(total0, product); break;
case 1: total1 = aie::add(total1, product); break;
...
}
...
float sum = 0.0f;
for (unsigned lane = 0; lane < 32; ++lane)
  sum += p0[lane] + p1[lane] + p2[lane] + p3[lane];
```

A vectorised partial-sum accumulate gated by an index `switch`, immediately
followed by a **scalar** float loop to finish the reduction — this project's
own [note 0001](notes/0001-aie-kernel-pitfalls.md) trap 5 (never scalar float
math in a kernel — measured 1,617× slower there) and AMD's own documented
anti-pattern (`switch`/branch in the hot loop), in the same function. Read
for what not to do, not copied.

**Design-residency limit, parallel to our own.** Its README states **two
decoder layers per persistent NPU invocation** as a firmware-verified limit —
a program-residency ceiling of the same *shape* as this project's own 16 KB
program-memory wall
([`0032`](../tasks/0032-m7-one-xclbin-production/TASK.md)), though not the
same mechanism or number; a loose architectural parallel, not a transferable
figure.

**Relevance to our open threads.** No numbered thread is answered by this
repo. It corroborates F1 (already an established finding, not an open
thread) and offers the loose T28-adjacent parallel above. Its own headline
negative result — *"Qwen2.5 0.5B NPU path is not faster than an eight-thread
CPU baseline"* — is consistent-in-kind with this project's own dispatch-cost
findings (F1) but is a decoder workload on different silicon, not evidence
about any of our own numbered threads.

---

## 4. `drakosha/whisper-xdna` — the closest sibling, re-read at a pinned commit

**What it is.** The Whisper **encoder** on XDNA1 (Phoenix), the same problem
shape as this project (bf16 GEMMs, fp32 accumulate, a polynomial GELU, a
host/device split, a CPU baseline to beat) and already this project's single
most-cited external source
([note 0007 §3.1–3.5](notes/0007-unused-iron-surface.md#31-the-default-aie-rounding-mode-is-floor----verified-and-we-never-set-it)).
License **MIT**, verified directly (`LICENSE`: *"MIT License, Copyright (c)
2026 Mikhail Kostryukov"* — again GitHub's API reports `NOASSERTION`, again a
detection artefact, not a real ambiguity). HEAD `d43cb82` (2026-07-27), a
young repo (created the same day per `git log`).

**How it talks to the NPU — more precisely than note 0007 recorded.** Two
paths coexist: `src/npu_whisper_encoder.py` compiles designs via IRON/
`@iron.jit` in the ordinary way; `src/rawxrt.py` is a **direct XRT backend
that bypasses the IRON runtime wrapper at dispatch time**, for a measured
**~1.4× speedup** — this project's own C++ runtime already avoids the IRON
Python layer entirely at dispatch (`CLAUDE.md` rule 5, "no Python at
runtime"), so this is independent confirmation the same lever exists and
matters, not a new lever for us.

**New, concrete numbers not in note 0007.**
- **Design-switch cost, independently measured on different hardware.**
  README: *"Switching between overlays costs 2.81 ms per pair of switches"*
  — measured by running one `runlist` alone (6.708 ms) against the same list
  with one foreign launch interleaved (10.516 ms, against 7.709 ms run
  separately). Different generation, different methodology, same phenomenon
  class as this project's own switch-cost model
  ([note 0004](notes/0004-context-switch-cost.md), `~25 µs + 7.2 µs/lock`) —
  the absolute number does not transfer, the shape of the finding
  (context-switching is the expensive part, not dispatch) does.
- **`pyxrt.runlist` works on NPU1 even though the MLIR-AIE examples document
  it as NPU2-only.** README, verbatim: *"`pyxrt.runlist` works on NPU1
  (~24% better per run) even though the MLIR-AIE examples mark it
  NPU2-only."* Mild positive signal for [T9](OPEN-THREADS.md#t9)
  (`xrt::runlist`, unused by our runtime) — we are already the documented
  target generation for this API, so this mainly says the upstream docs'
  own generation restriction is conservative, not that anything changes for
  us.
- **Buffer-access technique**: *"`bo.map()` instead of `read`/`write`"* saved
  **208 ms**, because `bo.write(a.tobytes())` is two copies stacked on the
  dtype conversion and `read()` a third. Same *shape* of lesson as this
  project's own `.numpy()` traps (`CLAUDE.md` 6b/6c), different specific
  bug.
- **Naive int8 without calibration is a real accuracy failure, corroborating
  our own closed thread.** README/HISTORY: per-channel int8 passed only
  **21 of 27** transcription tests against the torch fp32 reference, cosine
  similarity **0.78–0.93**. This project's own naive W8A8 attempt failed the
  2e-3 gate at **2.864e-03**, fixed only by SmoothQuant-style static
  calibration to **1.417e-03**
  ([`0078`](../tasks/0078-m13-int8-accuracy/TASK.md), closing
  [T20](CLOSED-THREADS.md#t20)) — a second, independent project hitting the
  same wall with naive int8 on AIE.
- **Vendor kernels do not port between `aie2` and `aie2p` even at the API
  level, not just performance.** `bench/layernorm_aie2.py`'s own comment:
  *"The stock ml/norm design is aie2p-only; on aie2 it fails to link with
  `ld.lld: error: undefined symbol: sqrtf`."* Their replacement
  `kernels/layer_norm_gb.cc` is explicitly derived from the Apache-licensed
  `aie_kernels/aie2p/layer_norm.cc`, ported *down* to aie2 (the reverse
  direction from anything we'd need, since we are aie2p already, but useful
  confirmation that the aie2/aie2p split is a real API boundary, not only a
  `mac_dims`/burst-length tuning difference — consistent with `CLAUDE.md`
  trap 1's broader point about the silent-arch-fallback risk).
- **A wording difference worth flagging, not resolving here.** Their README
  describes the default AIE rounding behaviour as *"the default fp32→bf16
  conversion **truncates toward zero**"*, where this project's own
  `CLAUDE.md` rule 2b (sourced directly from `aie_api/aie.hpp`/
  `aie_types.hpp`) says the default is `floor`, defined as *"always round
  towards negative infinity"*. These agree for positive values and disagree
  for negative ones. This project's own confirming test (softmax row sums
  never exceeding 1.0 under the default) only exercises positive values, so
  it does not discriminate between the two descriptions either. Not chased
  further here — flagged so a future session does not assume the two
  sources agree on negative-value behaviour without checking.
- **Exactly 6 concurrent `hw_context`s**, confirmed directly in
  `src/rawxrt.py`'s own comment (*"the ceiling of 6 is reached anyway"*) —
  an XRT-level limit, not IRON- or model-specific.

**Relevance to our open threads.** [T9](OPEN-THREADS.md#t9) (mild positive
signal, not decisive), [T20](CLOSED-THREADS.md#t20) (CLOSED — this is
corroboration after the fact, not new evidence toward reopening it). Note
0007 §3.1 (rounding mode), §3.4 (fused attention correct but slower — fp32
accumulators mandatory), §3.5 (per-shape tile geometry swings 2.5×/4.6×,
[T21](OPEN-THREADS.md#t21)) remain the primary record for this repo; this
section adds to it rather than replacing it.

---

## 5. `midhatn/phoenix-sdr-dsp` — the one Windows-native sibling, and the one aie2p design

**What it is.** Research-grade SDR/DSP (FIR, mixer, FFT, channelizers,
demodulators) plus a retained post-quantum-crypto milestone track (M32/M33,
now split out — see §6), on **Phoenix XDNA1**, but **the only one of these
six repos built natively for Windows** — the same platform this project
runs on. Apache-2.0, verified directly from `LICENSE`. 34/34 milestone tests
passing at HEAD `67f0673` (2026-08-18), which is roughly contemporaneous
with this project's own mlir-aie 1.3→1.4 migration
([`0058`](../tasks/0058-m11-iron-1.4-migration/TASK.md)).

**How it talks to the NPU.** XRT Windows SDK + MLIR-AIE/IRON (pinned
`v1.4.1`) + Peano, hand-written AIE2/AIE2P kernels
(`kernels/fft_stockham_f32.cc`, `include/sdr_dsp/*.hpp`). No ONNX Runtime,
no vendor overlay.

**The one `npu2`/`aie2p`-targeted design across all six repos.**
`docs/M17_V3_DESIGN.md`, read directly:

> "| Device | `npu2` (aie2p / XDNA2) | `--dev npu2` |
> | MMUL shape | `<4, 8, 8>` on npu2 | `single_core.py` mac_dim_map, kernel
> lines 137, 306 |"

This independently confirms `CLAUDE.md`'s own bf16 `mac_dims = (4,8,8)` for
aie2p — a completely different kernel domain (a 64-point radix-4 Stockham
FFT) reaching the same hardware fact this project's own toolchain notes
record. It is a portability probe, not a shipped SDR path — `M1_ARCHITECTURE_
DECISION.md` records that the aie2p/`npu2` overlays it found "already present
under `C:\Windows\System32\AMD`" were deliberately **not** loaded for their
Phoenix-targeted work.

**What is worth stealing.**
- **Independent confirmation that Peano defines `__AIECC__` for the aie2p
  target itself.** `M17_V3_DESIGN.md`: *"Peano defines `__AIECC__ = 1`
  (verified via `clang++ --target=aie2p-none-unknown-elf -E -dM`, 2026-08-15)
  ... llvm-aie/Peano also identifies itself as an AIE compiler by setting
  it."* Matches this project's own finding in
  [note 0006](notes/0006-peano-loop-hints.md) (cited via
  [note 0007 §1.9](notes/0007-unused-iron-surface.md)), reached
  independently on unrelated kernel code.
- **A hash-ledgered vendor-provenance file** (`THIRD_PARTY_PROVENANCE.md`):
  every adapted-from-upstream source file gets a row with its own sha256 and
  a note on whether it is an exact copy or an adaptation (e.g. `saxpy.cc`
  "Exact copy; upstream hash is identical" vs. `fft_stockham_f32.cc`
  "Adapted AIE2p kernel; not byte-identical"). More rigorous than this
  project's own `CLAUDE.md` rule 4, which states the *rule* but keeps no
  file-level hash ledger of what in `runtime/` or `experiments/` derives from
  mlir-aie's own example kernels. Worth considering, not built here.
- **The same fail-closed measurement discipline as this project's rule 1,**
  independently: its 34-entry test matrix is explicitly typed — *"29
  direct-hardware entries, four host/NPU composer entries, and one
  intentional CPU reference entry"* — so a reader always knows which class of
  evidence a given number is.

**Relevance to our open threads.** Corroborates [T22](CLOSED-THREADS.md#t22)
(CLOSED — independent evidence that the mlir-aie 1.3→1.4.1 migration broke
things elsewhere too, matching this project's own 0058 findings) and the
already-established `aie2p` `mac_dims`/`__AIECC__` facts. No open thread is
newly answered.

---

## 6. `midhatn/phoenix-npu-pqc` — a genuinely new repo, and the closest philosophical match on rigor

**What it is.** Post-quantum cryptography (FIPS 202/203/204 — SHA-3/SHAKE,
ML-KEM, ML-DSA) targeting **100% NPU device residency**, on **Phoenix NPU1
(XDNA1/AIE2)**. Its own README states plainly: *"Phoenix NPU PQC is a focused
continuation of the PQC work separated from the historical `phoenix-sdr-dsp`
repository."* Created 2026-08-18, same day as `phoenix-sdr-dsp`'s HEAD —
consistent with a same-day split. Apache-2.0, verified from `LICENSE`. Not
GEMM- or embedding-adjacent technically (no matmul kernel content applies to
our datapath); read here for its documentation discipline, which is the
strongest of any of the six.

**How it talks to the NPU.** XRT Windows SDK + MLIR-AIE/IRON + Peano, custom
`pyxrt` bindings, hand-written AIE2 kernels for NTT butterflies, Keccak/SHAKE,
and modular arithmetic (`phoenix_sdr_dsp/pqc/kernels/*.cc`).

**What is worth stealing — a documentation practice, not code.** This is the
closest match among all six repos to this project's own rule 3b ("failures
are the valuable part, never delete or rewrite them"). Its README draws an
explicit, load-bearing line between *"canonical silicon validation"* — one
blessed runner, `run_all_silicon_tests.py`, whose output is the **only**
thing allowed to be described as validation — and everything else
(diagnostic scripts, compile-only probes, retry logs), which cannot claim a
pass no matter what it shows. Its own DR2d milestone (the integrated
ML-KEM-512 key-generation candidate) **failed physically, 0/25, exit 1**, and
rather than hide or delete the six failed retry attempts, they are preserved
byte-for-byte in `docs/pqc_dr2_evidence_20260818/sigma_prf_retry_chain/` —
`retry0` through `retry6`, each its own diagnostic artifact, with the
directory's own scope note: *"read-only research evidence, not an
authorization to execute hardware."* This is this project's own tasks-log
rule ("never delete or rewrite a failed attempt") applied with unusual
rigor, arrived at independently, on unrelated hardware and an unrelated
problem domain.

**What it warns us about.** Nothing technical transfers (cryptography, not
GEMM). The one general lesson: **a hardware hang or a 0/N failure is worth
recording in exactly this much detail** — compile logs, token captures,
relocation records, retry-by-retry — which is the same standard this
project's own `tasks/0087` entry (the relay-at-production-width hang, cited
in [T28](OPEN-THREADS.md#t28)) already meets, independently.

**Relevance to our open threads.** None — different problem domain
entirely. No claim of technical relevance is made here beyond the
documentation-discipline parallel above.

---

## 7. `julianmb/npuhalo` — our own silicon generation, but at the FastFlowLM layer

**What it is.** *"What is the XDNA2 NPU actually good for when a big LLM owns
the iGPU?"* — a research record of eight measured verdicts on heterogeneous
LLM inference on **AMD Ryzen AI Max+ 395 ("Strix Halo")**: 16 Zen 5 cores,
Radeon 8060S iGPU (RDNA3.5, 40 CU, ~72 tok/s on a 35B-A3B FP4 model via
`llama.cpp`), and a **48-tile XDNA2 NPU** running small models through
**FastFlowLM v0.9.46**. Confirmed directly from `README.md`'s own hardware
badge and body text: *"AMD Ryzen AI Max+ 395 (Strix Halo)"*, *"XDNA2 NPU (48
AIE2p tiles, `/dev/accel/accel0`)"*. License **Apache-2.0**, verified from
`LICENSE`. HEAD `a34fd36` (2026-08-23) — a commit made *today*, sanitising
machine-specific paths out of 47+7 tracked result files so its own validator
"now writes repo-relative paths"; one commit tag is `v1.0.0-findings`.

Strix Halo and this project's Strix Point (Ryzen AI 9 HX 370) are both XDNA2/
AIE2P — the same tile ISA, `mac_dims`, and DMA-BD rules this project's
`CLAUDE.md` documents — but Strix Halo carries **48 tiles against our 32**
(a different column/row count, not a different generation) and 128 GB of
unified LPDDR5X-8000 (~273 GB/s) against our machine's much smaller pool.
Numbers below are the closest same-generation data this survey found; the
tile-count and memory-bandwidth differences mean they corroborate direction,
not magnitude.

**How it talks to the NPU.** Not IRON, not hand-written kernels, not XRT
directly — **FastFlowLM's REST API** (this project's own reference
architecture, `prior-art.md` §1), running small chat models
(`LFM2.5-1.2B`, `qwen3.5-0.8b-FLM`) behind an OpenAI-compatible endpoint on
port 8001, orchestrated by a CPU agent loop that also drives the iGPU's
`llama.cpp` generation on port 8012. `npuhalo` treats FastFlowLM as a black
box and measures the *system*, not the kernel.

**What is worth stealing.**
- **A directly measured number for NPU/iGPU DRAM contention on a shared-memory
  XDNA2 chip**, absent everywhere else in this survey: *"concurrent NPU work
  costs the GPU −16.5% decode (directly measured)"* — reproduced twice, once
  via a synthetic DRAM-traffic proxy (`scripts/phase0_contention.py` +
  `bw_stress.py`, −17.2% at ~16 GB/s of added traffic) and once with a real
  FastFlowLM NPU load running concurrently (`scripts/phase0_real_npu.py`,
  38.70 → 32.30 tok/s = **−16.5%**, the two methods agreeing to within a point).
  Their own reading: *"the NPU's DMA wins contention and the GPU pays the
  price."* This project's `CLAUDE.md` already frames itself as bandwidth-bound
  (F2/F3, ~40–60 GB/s reaching the NPU); this is the first evidence in this
  survey of what NPU DMA traffic costs a *co-resident* consumer on the same
  memory controller, on our own silicon generation. Not directly transferable
  as a number (our runtime never shares a decode loop with a GPU generator),
  but a concrete precedent for "the NPU's bus use is not free to everything
  else on the SoC" if this project ever runs concurrently with other
  NPU/GPU work.
- **The same 4 MiB-class L2 capacity figure this project already has,
  reached independently.** §6.1: *"a ~160-384 MB head that cannot fit the
  4 MiB XDNA2 L2 SRAM (per kernel docs; the 32 MB 'MALL' is GPU-side Infinity
  Cache, not NPU SRAM)."* This project's own mem-tile capacity
  (`getMemTileSize()` = 512 KB × 8 columns = 4 MB, [note 0007 §1.4](notes/0007-unused-iron-surface.md))
  lands on the same total — independent corroboration from a completely
  different toolchain layer (FastFlowLM's own docs, not mlir-aie's target
  model), though `npuhalo`'s 48-tile chip may partition that differently than
  our 32-tile one; not reconciled here.
- **The research discipline** — pre-registered decision gates with
  pass/fail thresholds fixed *before* looking at results, immutable
  `manifest.json` per run with dataset SHA-256, paired arms with interleaved
  order, and an explicit `INCONCLUSIVE` verdict class rather than "pick the
  nearest flattering outcome" (`docs/METHODOLOGY.md`) — is the same ethic as
  this project's rule 1 and rule 3b, applied to agentic-eval rather than
  hardware traces, and worth reading as a second, more elaborate model of
  the same practice.
- **A concrete "Block FP16" lead, not chased here.** `docs/final_verdict.md`
  §2 names *"Block FP16 — XDNA 2 hardware 8+1-bit float format ... HW format,
  needs compiler support"* among AMD research paths it declined to adopt.
  This may be the same feature this project calls bfp16
  ([T23](OPEN-THREADS.md#t23), [T26](OPEN-THREADS.md#t26)) described from a
  different angle — **not verified**; `npuhalo` cites it only as a
  compiler-support gap for an unrelated (decode-side) use case and does not
  name a source, so this is flagged as a lead for whoever next touches T23,
  not a claim.

**What it warns us about.** Every NPU-in-the-decode-loop idea it tried was
killed, empirically, including ones that looked promising on paper —
speculative decoding NPU→GPU (two independent implementations, both slower
than GPU-only), a TTFT burst handoff (a previously-measured 1.8× win that
*failed to reproduce* on newer firmware/server code, now losing 1,430 ms vs
730 ms), and a 0.8B query router (25% routing accuracy). The project's own
one-line summary doubles as a warning to any project tempted to add NPU work
into a latency-sensitive shared-memory pipeline: *"the NPU helps most when it
stays out of the generation loop."* Not directly about GEMM or embeddings,
but a relevant caution given this project's own host/device split
architecture and its live thread about device-resident intermediates
([T3](OPEN-THREADS.md#t3)) — the contention finding above says that
architecture's safety margin (no GPU sharing the bus) is doing real work,
not merely simplifying the design.

**Relevance to our open threads.** No numbered thread is answered. The L2
capacity figure corroborates an established fact (not open). The "Block
FP16" mention is a lead for [T23](OPEN-THREADS.md#t23)/[T26](OPEN-THREADS.md#t26),
explicitly unverified, not a finding.

---

## 8. `Unicorn-Commander/Meeting-Ops-UC1-OSS` — an application, and a caution about self-reported NPU claims

**What it is.** A meeting-recording appliance (FastAPI + PostgreSQL + Redis +
Qdrant backend, React/TypeScript frontend) — live transcription, speaker
diarization, LLM-generated meeting notes, semantic search — targeting **AMD
Phoenix NPU (XDNA1, 16 TOPS INT8)**, confirmed from its own
`backend/docs/NPU_HARDWARE_EVIDENCE.md` (*"AMD Phoenix NPU (AI Engine 2) ...
16 TOPS INT8"*) and `backend/npu_final_findings.md` (*"AMD Ryzen 9 8945HS
with NPU"* — Hawk Point, same `npu1`/AIE2 family). License **AGPLv3**,
verified directly from `LICENSE` — same do-not-vendor rule as §1/§2. HEAD
`cf1aec3` (2026-07-30), 886 tracked files, a large application rather than a
kernel project; not actively read in full — this section is scoped to the
embedding/NPU serving path the task brief asked about, per its own file
layout (`backend/services/`, `backend/npu_optimization/`).

**How it talks to the NPU — and here the claims and the code diverge.**
The top-level `CLAUDE.md` and several `backend/NPU_*.md` status files assert
NPU acceleration is *"100% Complete"*, citing **"220x speedup (confirmed)
with custom MLIR-AIE2"**, **0.0045 RTF**, and **4,789 tokens/second** for
Whisper Large-v3 transcription. Reading the code these numbers point at
tells a different story:

- `backend/npu_optimization/aie2_kernel_driver.py` line 116: the driver's
  own comment reads *"Generate a mock XCLBIN that contains our kernel
  metadata"* — the artifact it loads onto (or hands to) the NPU is
  self-described as a mock, not a design compiled by IRON/Peano from a real
  kernel. Its benchmark harness (lines 373, 375–376, 420–422) generates its
  own inputs with `np.random.randint(...)` rather than benchmarking a real
  workload end to end.
- `backend/npu_final_findings.md`, dated after the "100% Complete" claims
  elsewhere, is the project's own contradicting record: *"No public x86_64
  Linux wheels for onnxruntime-vitisai ... All available wheels are ARM64 or
  Windows-only"*, concluding *"Continue using CPU emulation - It's working
  and stable"* for the ONNX/VitisAI path specifically.
- `backend/services/embedding_service.py` — the file most relevant to this
  project's own problem statement — defines `NPUEmbeddingService`, docstring
  *"NPU-Accelerated Embedding Service"*, wrapping `all-MiniLM-L6-v2-int8.onnx`
  (the same MiniLM this project ships) via `onnxruntime.InferenceSession`
  with providers `['DmlExecutionProvider', 'CPUExecutionProvider']`.
  `DmlExecutionProvider` is Microsoft **DirectML** — a Windows/DirectX
  compute path — on a project whose own hardware docs describe Linux
  (`arecord`, `amdxdna` kernel driver, `/dev/accel/accel0`) as the target;
  there is no Vitis AI EP, no XRT, no IRON anywhere in this file. **Read
  plainly, on this project's own stated Linux deployment `DmlExecutionProvider`
  would not be available and the session falls through to
  `CPUExecutionProvider`** — the embedding path is very likely CPU-only
  despite its class name and docstring. This document does not claim to have
  run the code and confirmed the fallback fires; it reports what the two
  lines of provider configuration say on the platform the rest of the repo
  documents itself as running on.

**What is worth stealing.** Architecturally, little: no design here is
IRON/XRT-based, and the one file closest to this project's own problem
(`embedding_service.py`) does not appear to reach the NPU at all. The
concrete useful artifact is the **caching/normalisation shape** around the
embedding call (SHA-based cache keys under `storage/embeddings/`,
`AutoTokenizer` for WordPiece) — generic ONNX-serving hygiene, not anything
specific to AIE.

**What it warns us about — the sharpest finding in this whole survey.** A
repository's own status documentation ("100% Complete", "220x speedup
confirmed") is not evidence, and can directly contradict what the code in
the same repository does, and even contradict *itself* between documents
written weeks apart. This is `CLAUDE.md` rule 1 ("wall-clock time is never
an NPU performance claim") and rule 6 ("a number without a traceable
artifact is not a result") illustrated from the outside: every number this
repo's front-facing docs report for its NPU path traces back to either a
mock XCLBIN with synthetic random inputs, or a provider configuration that
would not reach the NPU on the platform the repo itself documents. Nothing
here is a specific technique to copy; it is a concrete instance of the
failure mode this project's own measurement rules exist to prevent, found in
someone else's shipped "complete" feature.

**Relevance to our open threads.** None. Domain and stack do not overlap with
any numbered thread; recorded for the discipline lesson above.

---

## What none of them do

XDNA2/`aie2p` — the silicon this project actually runs on — is close to
absent across all eight. `phoenix-sdr-dsp` has one non-shipped FFT
portability probe (§5); `npuhalo` (§7) is on the same generation but only at
the FastFlowLM product layer, never IRON/XRT; the other six are XDNA1. None
runs a BERT-family embedding model end to end on an NPU with a result this
survey can trust: `npu-linux-kit`'s own `embeddings/` module, which names
this project's exact problem statement, is an explicit, unbuilt stub two
years into the sibling ecosystem's existence, and `Meeting-Ops-UC1-OSS`'s
`NPUEmbeddingService` (§8) — the one repo surveyed that actually ships an
"NPU embedding" class — appears from its own code not to reach the NPU at
all. None pairs a fully native-Windows toolchain with a C++-only,
zero-Python-at-dispatch runtime the way this project does — the two
Windows-native repos here (`phoenix-sdr-dsp`, `phoenix-npu-pqc`) both
dispatch from Python via IRON/`pyxrt`, `whisper-xdna`'s fastest path
(`rawxrt.py`) is Python bypassing IRON rather than C++ bypassing Python
entirely, and `npuhalo`/`Meeting-Ops-UC1-OSS` both sit a layer above the NPU
runtime rather than driving it directly. This does not narrow
`prior-art.md`'s existing bottom line — it reconfirms it from
a different angle, now including a chip-generation match with no
architecture match (`npuhalo`) and a claimed embedding-service match with no
verified NPU execution (`Meeting-Ops-UC1-OSS`): **there is still no open,
from-scratch, hand-tiled C++ BERT/MiniLM engine on XDNA2.**
