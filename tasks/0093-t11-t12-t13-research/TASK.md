# 0093 — T11, T12, T13: three cheap open threads, settled by reading

- **Date** 2026-08-23
- **Milestone** research (register maintenance, no hardware)
- **Status** done

## Goal

Settle three open threads in `research/OPEN-THREADS.md` — T11 (hw_context partition
width), T12 (larger L2 megatiles), T13 (pre-tiled instability) — by reading only. No
NPU design was built or run. `research/OPEN-THREADS.md`, `research/CLOSED-THREADS.md`
and `tasks/README.md` were read but **not edited** (other agents are editing them
concurrently) — this task's own "Proposed register update" section is the handoff.

## Context

All three threads were read verbatim from `research/OPEN-THREADS.md` before starting:

- T11, lines 497–501: *"Decided in the driver's create-hwctx call, not at XRT level.
  Low value now that production is one xclbin at a fixed width."*
- T12, lines 504–516: re-pointed by T1, then **UNDONE 2026-08-23** — *"This thread's
  ORIGINAL bandwidth framing is the live one there... Re-read the paper with the
  original question, not the re-pointed one."*
- T13, lines 519–523: *"best-case pre-tiled runs match row-major exactly, so it is an
  intermittent stall, not a ceiling... this only matters if it is a symptom of
  something else."*

---

## T11 — Is the hw_context partition width settable at creation?

### What was read

1. `research/notes/0004-context-switch-cost.md` lines 91–125 (full "What that does not
   claim" section) — the note already narrows the open question to one concrete
   sub-question: *"Is the partition width settable at context creation?"* and names
   the Linux UAPI struct `amdxdna_drm_create_hwctx` in `amd/xdna-driver` as the place
   to look, while noting **"no such header exists locally."**
2. `C:\Xilinx\XRT\include\xrt\xrt_hw_context.h` lines 17–53 — the public C++
   `xrt::hw_context` class. `cfg_param_type = std::map<std::string, uint32_t>`
   (line ~46) with its documented key list (lines 38–52): `gops`, `fps`,
   `dma_bandwidth`, `latency`, `frame_execution_time`, `priority`,
   `enable_isp_channel`, `enable_acp_channel`. **No column-count, tile-count, or
   partition-width key anywhere in this list.**
3. `runtime/src/npu_device.cpp:231` — `impl_->ctx = xrt::hw_context(d.device, uuid);`
   — this project's own context creation uses the **no-QoS constructor**
   (`xrt_hw_context.h` lines ~175–183, the "Undocumented construction w/o specifying
   qos"). No `cfg_param_type` is ever populated.
4. `C:\dev\mlir-aie\ironenv\Lib\site-packages\mlir_aie\python\aie\utils\hostruntime\xrtruntime\hostruntime.py`
   — IRON's own hostruntime, grepped for `hw_context`: lines 216, 729, 846 all call
   `pyxrt.hw_context(self._device, xclbin_uuid)` or `pyxrt.hw_context(self._device,
   elf)` — again no `cfg_param`. Lines 404–405 show where column count actually enters
   IRON: `from_name("npu1", n_cols=None)` / `from_name("npu2", n_cols=None)` — a
   **device descriptor chosen before compilation**, baked into the xclbin, not a
   parameter of context creation.
5. `externalrepos/` grepped across all six repos for `create_hwctx|CREATE_HWCTX|
   num_tiles|qos_info|hwctx` (`.py/.md/.c/.h/.txt`). No `.h` files exist in any of the
   six repos (they are Python-only projects) except the XRT tree itself, so the search
   moved to prose/scripts. `externalrepos/Meeting-Ops-UC1-OSS/backend/
   NPU_COMPREHENSIVE_DOCUMENTATION.md` lines 46–81 gives the **actual DRM ioctl
   struct**, reverse-engineered from `/usr/include/drm/amdxdna_accel.h` on a Linux
   Phoenix (XDNA1) box:

   ```c
   struct amdxdna_drm_create_hwctx {
       __u64 ext;
       __u64 ext_flags;
       __u64 qos_p;           // Pointer to QoS info
       __u32 umq_bo;          // User mode queue buffer
       __u32 log_buf_bo;      // Log buffer
       __u32 max_opc;         // Max operations per cycle
       __u32 num_tiles;       // Number of AIE tiles
       __u32 mem_size;        // Size of AIE tile memory
       __u32 umq_doorbell;    // Output: doorbell offset
       __u32 handle;          // Output: context handle
       __u32 syncobj_handle;  // Output: sync object handle
   };
   #define DRM_IOCTL_AMDXDNA_CREATE_HWCTX  0xC0386440
   ```

   `num_tiles` is set by userspace (input field, not output) and is corroborated
   independently across **five** separate scripts in the same repo that all hard-code
   it as a parameter the caller chooses: `debug_hwctx.py:63`, `direct_npu_poc.py:66`,
   `npu_inference.py:19,62`, `npu_runtime_real.py:145`, `test_npu_complete.py:78` — all
   pass `num_tiles=4` (the whole Phoenix array) when creating the context. `open-xdna`
   and `npu-linux-kit` were also grepped; neither carries the struct definition
   (`open-xdna/docs/UPSTREAM_amdxdna_ioctls.md` is about a *different* ioctl gap —
   `GET_ARRAY`/telemetry, not `CREATE_HWCTX` — and was read and ruled irrelevant).
6. `research/prior-art.md` §1 (FastFlowLM) — re-read for context creation. FastFlowLM
   also goes through `xrt::hw_context` / `xrt::ext::kernel` via `aiebu`-assembled
   ELFs; nothing in that section documents a partition-width parameter either. No new
   evidence there, ruled consistent with (2)–(4).

### Verdict — ANSWERED, yes-but-not-here

**Yes at the driver-ioctl level, on Linux, unconfirmed for Windows; no at the XRT
public C++ API level this project (and IRON itself) actually calls.**

- The Linux DRM ioctl that creates a hardware context, `DRM_IOCTL_AMDXDNA_CREATE_HWCTX`,
  takes a `struct amdxdna_drm_create_hwctx` with an explicit `__u32 num_tiles` input
  field — partition width **is** a parameter at context-creation time at that layer,
  confirmed by a struct definition independently corroborated by five call sites in
  one third-party repo (`externalrepos/Meeting-Ops-UC1-OSS`, reverse-engineered from
  `/usr/include/drm/amdxdna_accel.h` on Phoenix/XDNA1 — not our SKU, not confirmed
  against the Windows driver, which is closed).
- At the layer this project (and mlir-aie's own hostruntime) actually programs
  against — `xrt::hw_context`'s public constructors — there is **no such key**. The
  `cfg_param_type` map's eight documented keys are QoS/scheduling hints, not topology.
  Both our runtime and IRON's own Python runtime call the no-QoS constructor and let
  the xclbin's own baked-in partition (chosen via `from_name(..., n_cols=...)` **at
  compile time**, not at context creation) decide the width.
- So the honest statement is: **the mechanism note 0004 hypothesized exists, at the
  driver layer, on Linux** — but it is not reachable from where this project's C++
  runtime sits, and nothing in six external repos, the XRT headers, or IRON's own
  source shows a Windows equivalent being exercised. Given T11's own text already
  flags this as low value now that production is one xclbin at a fixed width, this
  is close enough to settle: RETIRE with the answer recorded, not left open forever
  chasing a Windows-specific confirmation nobody needs.

---

## T12 — Larger L2 megatiles

### What was read

1. `research/papers/2602.06063.md`, in full (122 lines) — the FastFlowLM paper
   summary, re-read for the *original* bandwidth question per the thread's own
   instruction.
2. `research/CLOSED-THREADS.md` T1 entry, lines 184–221 — what "bytes are free" meant,
   scoped, and un-scoped for int8.
3. `experiments/m5-pretiled-gemm/gemm_pretiled.py` lines 152–314 — how B's L2 staging
   works today (`b_reuse` modes, `B_l2_mega_ty`), and CLAUDE.md's own tiling
   constraints.
4. `tasks/0080-m13-int8-traffic-bound/TASK.md` in full — the traffic model that now
   governs int8 (R² 0.987).
5. `tasks/0081-m13-int8-everywhere/TASK.md` §3, lines 71–105 — where the array sits in
   the encode today (30.4% of wall clock on bge-large) and what the next lever is.

### 1. What exactly does the paper vary, and is it "megatile size alone"?

Quoted directly from the summary (`research/papers/2602.06063.md` lines 40–46):

> **Measured megatile sweep — rare, concrete calibration.** Krackan, 2048³ bf16 GEMM:
>
> | megatile | throughput |
> |---|---|
> | 128×512×512 | **5.9 TOPS** |
> | 256×256×512 | **12.0 TOPS** |
> | 512×512×512 | **13.7 TOPS** |

**Not quite "size alone."** `128×512×512` and `256×256×512` have the *same* element
volume (128·512·512 = 256·256·512 = 33,554,432) — that step is a **reshape at
constant L2 footprint** (2.0× throughput from aspect ratio alone: fewer, larger
re-fetches of one operand traded against more of the other). Only the second step,
`256×256×512 → 512×512×512`, is an actual **4× volume increase**, for a further
**1.14×**. So the headline "5.9 → 13.7" conflates two different levers — reshape and
size — and the size-only contribution (last row vs middle row) is much smaller than
the "megatile size alone" framing implies.

### 2. What datapath, dtype, silicon?

**bf16**, on "Krackan Point" (Ryzen AI 7 350) — a different XDNA2 SKU from our Strix
Point HX 370, but the paper's own hardware description (lines 9–13) matches ours
architecturally: **32 CTs, 8 columns × 4 rows, up to 1.8 GHz, 64 KB L1 per CT, 8×
512 KB L2** — i.e. the same `aie2p`/npu2 family, just a different die. So the
silicon match is close enough to trust the *mechanism*, but **the dtype does not
match the question T12 is now asking**: this measurement is bf16, and the live
question (per T1's UNDONE note) is specifically about **int8**, where bytes are the
binding constraint. On bf16, T1 already measured (0048) that bytes are not the
constraint on our design (R² 0.709 for the traffic model, worst 20.1%, against the
iteration model's R² 0.994). The paper's bf16 megatile result is not in tension with
that — 0048's shapes are much smaller than a 2048³ GEMM, and a big-enough tile can
still be bandwidth-bound even when the traffic model doesn't win a regression fit at
our scale. But it means the paper's 5.9→13.7 TOPS number **cannot be quoted as a
directly-transferable int8 prediction** — it is a same-family, wrong-dtype
calibration.

### 3. Does the mechanism apply to our design?

The paper's mechanism, quoted (`2602.06063.md` line 38): reducing data-movement cost
**"from `(N/n)·|A| + (M/m)·|B|` to `(N/4e_x·n)·|A| + (M/8e_y·m)·|B|`"** — i.e.
amortising each operand's L3→L2 fetch over a wider L2-resident block so it is
re-streamed fewer times. That is exactly the mechanism `gemm_pretiled.py` already
has a name for: **`b_reuse`**. Lines 199–276 show three modes already built —
plain per-tile B fifo, a depth-N fifo holding the whole per-column slice
(`b_reuse=True`), and **`b_reuse="mega"`: one L2 object holding the ENTIRE
column slice in a single descriptor** (lines 217–226, 253, 264–276) — which *is*
a megatile for B, already implemented, at the granularity the paper describes.
0046 (cited from 0080 §8) found this blocked not by L2 capacity but by mem-tile
**DMA channel** budget (5 of 8 mem tiles already at 6/6). So the paper's lever and
our own "B-reuse" lever are the same mechanism under two names, and the blocker
identified locally (channels, not capacity) is a fact about our floorplan the paper
does not speak to at all.

The other half of the paper's formula — amortising **A** over a wider N per gather —
maps onto **`tile_n`**, which is exactly what `tasks/0080` §8 point 3 already names
as reopened: *"A re-streaming goes as `N/(n·cols)`... Larger `tile_n` cuts A traffic
directly, and int8's L1 headroom is exactly what makes a bigger tile legal."*

**So yes, the mechanism applies — it already has two names in this project's own
work (`b_reuse="mega"` for B, `tile_n` for A), and 0080 had already independently
arrived at the same lever from the traffic model alone, before this re-read.** T12
is not adding a new idea; it is confirming that the paper and 0080 point at the same
thing from two different directions.

### 4. Predicted value, with arithmetic, and one experiment

Using 0080's fitted int8 traffic model (`t ≈ 627 µs + traffic / ~28 GB/s`, §1c) and
its per-shape traffic breakdown (§2):

| shape | K | N | A (MB) | B (MB) | C i32 (MB) | total i32 (MB) |
|---|---:|---:|---:|---:|---:|---:|
| `ffn_up` | 384 | 1536 | 12.6 | 18.9 | 50.3 | 81.8 |

`ffn_up`'s current `tile_n = 48` legally divides `N/cols = 1536/8 = 192` four times
(`192/48 = 4`); CLAUDE.md's trap 3 budget at int8 byte widths (`in=1`, and `c_bf16`
already narrows `out` to 2 bytes) gives headroom up to `n ≈ 146` before the 63 KB L1
limit is hit (`2·(64·64·1 + 64·n·1 + 64·n·2) < 64512 ⇒ n < 147`), and `n=96` divides
`192` cleanly (`192/96 = 2`) — a **legal, buildable geometry that halves the A
re-streaming factor** (`N/(n·cols)`: 4 → 2). Naively halving the A component
(12.6 → 6.3 MB) shaves 6.3 MB off `ffn_up`'s post-`c-bf16` total (56.6 MB per §6 of
0080, since C is already narrowed there) — an **~11% traffic reduction**, which
through the fitted model (`627 µs + traffic·0.0357 ms/MB`) predicts roughly
**1.03–1.05×** on `ffn_up` alone, and less end to end once diluted by the 61–70%
of wall clock that 0081 §3 shows is already host-side, not array-side. `ffn_down`
(`N=384`, `N/cols=48`) cannot take `n=96` at all — `96 ∤ 48` — so this lever is
shape-specific, not a blanket win, exactly mirroring what 0080 §6 already observed
about bge-large's smaller `tile_n=32` gaining less from C-narrowing.

**The one experiment that tests it:** rebuild the int8 `ffn_up` design at `tile_n=96`
(`export_gemm_rtp.py --int8 --c-bf16 -n 96` restricted to the N=1536 shape) and run
`--probe-streams` against the shipped `n=48` int8+bf16-C build — the same isolated,
per-shape measurement 0080 §1c already used to separate the two cost models, so the
result is directly comparable to the numbers in this task without re-deriving a
baseline.

### Verdict — ANSWERED (mechanism confirmed, predicted value small and shape-specific)

The paper's original bandwidth framing is real and does map onto our design — but
the honest reading is that this project had **already found the same lever from its
own traffic model** (0080 §8) before this re-read, the paper's own headline number
is bf16 (the wrong dtype for the live question) and conflates a reshape with a size
increase, and the predicted local payoff (~3–5% on one of four shapes, before host
dilution) is modest next to 0081 §3's finding that the array is only 30.4% of wall
clock and host-side fusion is "worth far more than anything left on the array."
ANSWERED, not a new discovery — but worth recording precisely because the paper
confirms the *mechanism* independently of 0080's own derivation, which is exactly
the kind of external corroboration this project's research index is for.

---

## T13 — Explain the pre-tiled instability · "or stop caring"

### What was read

1. `tasks/0007-m5-pretiled-gemm-on-npu/TASK.md` §4 (lines 84–96) and §8 (lines
   137–155) — the original finding: rowmajor spread 0.2%, pretiled spread 9.6–26.4%
   (varies by shape/tile-order), pretiled's *best* runs match row-major exactly on
   three of four shapes.
2. `tasks/0008-m5-bfp16-real-data/TASK.md` lines 150–175 and 242 — wall-clock
   confirmation under process isolation: rowmajor spread 1.8–3.3%, pretiled
   9.4–16.5%. Line 242: *"The pre-tiled instability now has two independent
   confirmations (per-core traced, and wall clock) and still no mechanism."*
3. `tools/export_gemm_rtp.py:319` — confirms **production designs are built with
   `pretiled=True`** (`pretiled_array(A, B, C, ..., pretiled=True, ..., rtp=True)`),
   i.e. the access pattern 0007 measured as unstable is the one that ships, not a
   discarded alternative. (`gemm_pretiled.py` line 35 comment and lines 199–314
   corroborate this is the same L3→L2 access-pattern change 0007 §7–8 isolated as
   the actual cause, independent of the offline sub-tile reorder which 0007 §6
   showed was free.)
4. `tasks/0058-m11-iron-1.4-migration/TASK.md` line 139 (the regression-verification
   table row for `gemm_pretiled.py`): *"rowmajor ffn_down M=512 4 cols n=48: mean
   140.9 MACs/cyc, spread 0.0% over 3 runs; **pretiled spread ~10%**"* — matched
   against task 0007's original "140.9 (0.1%)" row and recorded **exact**. This is
   dated **M11**, after the mlir-aie 1.3.4 → 1.4.x migration, i.e. a *current-toolchain*
   re-measurement, not a stale M5 number being carried forward.
5. `tasks/0085-m13-release-sweep/TASK.md` lines 26–39 — the whole-catalogue sweep at
   **production scale** (8 columns, batch 8192, 4 pipeline lanes, end-to-end
   throughput): spread **under 0.6% on five of six rows**, and an explicit correction
   of an earlier "~4%" figure (`tasks/0082` lines 126–130, `tasks/0084` lines 130–132)
   that 0084 traced to **one contended reading** rather than a real property of the
   design.

### Cross-check: does the instability still matter, or was it contention?

This is the one place the two later measurements point in different directions, and
both are real:

- **0058 (M11) re-confirms the instability exists, unmasked, at the exact isolated
  scale 0007 used** (M=512, 4 columns, per-core traced MACs/cycle) — same magnitude,
  same shape (`ffn_down`), on the current toolchain. This directly contradicts the
  hypothesis that the 9–22% spread was a stale artifact of an old mlir-aie version.
- **0085 (M13) shows production-scale end-to-end spread under 0.6%** on five of six
  models, at 8 columns / batch 8192 / 4 pipeline lanes — far tighter than what an
  unmasked 9–22% GEMM-level instability would produce even after dilution by the
  ~30–70% of wall clock that is host-side (0081 §3). If the instability were present
  at production scale at anything like 0007's magnitude, some trace of it should
  survive dilution; 0085 shows essentially none.

These are not the same measurement and are not directly comparable: 0058 is an
isolated per-core trace at **M=512, 4 columns**, matching 0007's original small-scale
config; 0085 is end-to-end wall clock at **M=8192, 8 columns, pipelined**. Nothing
in the record tests the instability's per-core signature *at production geometry*
— that specific cell is empty. So the honest conclusion is not "retired" (0058
falsifies "the modern toolchain fixed it") and not "the same size problem it always
was" (0085 falsifies "it's still costing 9–22% in production").

### Verdict — still OPEN, but re-scoped by what was found

**Stays OPEN.** Per the task brief's own criterion — *"If some later task DOES still
report it, the thread stays open and you should name that task"* — `tasks/0058`
(M11, current toolchain) is exactly that task: it reproduces the pretiled
instability at the original scale, unprompted, as a byproduct of a migration
regression check, with no mechanism proposed then or since. `research/
OPEN-THREADS.md`'s own text ("or stop caring") already flags this as low-priority;
this reading adds a sharper reason to keep it filed rather than a reason to close
it: the thread is not "explained," and the one clean piece of evidence that *might*
have closed it as contention (0085's tight production spread) tests a different
geometry than the one that's unstable, so it cannot be read as a refutation.
**Re-scoped**, not answered: the open question is now narrower — *"does the
`pretiled=True` access pattern's per-core instability reproduce at production scale
(8 columns, batch ≥ 1024), or is it specific to the small isolated config every
existing measurement (0007, 0008, 0058) has used?"* — a single traced run at
production geometry with `--repeat` would settle it, and nothing in the record has
done that yet.

---

## Problems hit

- `externalrepos/` contains no `.h`/`.c` files matching the Linux ioctl surface T11
  asked about — all six repos are Python-first. The DRM ioctl struct had to be
  recovered from a reverse-engineering doc (`NPU_COMPREHENSIVE_DOCUMENTATION.md`) and
  cross-checked against five independent script call sites in the same repo rather
  than from a canonical kernel header, because none is vendored locally. Flagged as
  weaker evidence than a real header would be — it is third-party, Phoenix/XDNA1, and
  Linux-only — and stated as such in the T11 verdict rather than smoothed over.
- T12's "predicted value" arithmetic mixes numbers from two different tables in 0080
  (§2's i32-traffic breakdown and §6's post-`c-bf16` totals) because no single table
  in 0080 gives a post-narrowing per-component (A/B/C) split. The estimate is
  therefore an order-of-magnitude prediction, explicitly labelled as such, not a
  number to cite as a result (rule 6) — the one experiment named is what would turn
  it into one.

## Artifacts

None produced beyond this TASK.md — a reading task per the brief. No files under
`research/` were modified.

## Next

- Build the one experiment named under T12 (`ffn_up` at `tile_n=96`, int8+bf16-C,
  `--probe-streams` against the shipped `n=48` build) if the lever is worth chasing
  ahead of the host-fusion work 0081/0082 already prioritise higher.
- Run the production-geometry repeat-trace named under T13's re-scoped question —
  the cheapest way to finally close it either direction.
- T11 needs no further work; RETIRE per the register update below.

---

## Proposed register update

### T11 — RETIRED

**Verdict: RETIRED**, answered as far as the available sources allow, and T11's own
filed text already said this was low value once production settled on one xclbin.

Proposed `CLOSED-THREADS.md` entry (verbatim original + dated status line):

```markdown
### T11 — Is the hw_context partition width settable at creation? · **RETIRED 2026-08-23**
[note 0004](../../research/notes/0004-context-switch-cost.md) §1 and
[`0025`](../0025-m7-batching-and-crossover/TASK.md). Decided in the
driver's create-hwctx call, not at XRT level. Low value now that production is
one xclbin at a fixed width.

> **RETIRED 2026-08-23 ([`0093`](TASK.md)):**
> yes at the driver-ioctl level, on Linux — `struct amdxdna_drm_create_hwctx` has
> an explicit `__u32 num_tiles` input field passed to
> `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` (reverse-engineered struct, corroborated by
> five independent call sites in `externalrepos/Meeting-Ops-UC1-OSS`; Phoenix/
> XDNA1, not our SKU, Windows driver unconfirmed). **No** at the layer this
> project and IRON's own hostruntime actually call: `xrt::hw_context`'s public
> `cfg_param_type` (`C:\Xilinx\XRT\include\xrt\xrt_hw_context.h`) documents only
> QoS/scheduling keys, no topology key, and both `runtime/src/npu_device.cpp:231`
> and mlir-aie's `xrtruntime/hostruntime.py` call the no-QoS constructor —
> partition width is baked into the xclbin at compile time
> (`from_name("npu2", n_cols=...)`), not chosen at context creation. Retired as
> answered-as-far-as-sources-allow, matching the thread's own stated low value.
```

### T12 — ANSWERED

**Verdict: ANSWERED** (the mechanism is real and already known to this project under
another name; the paper's own headline number does not transfer cleanly to int8).

Proposed `CLOSED-THREADS.md` entry (verbatim original text, including the 2026-08-23
UNDONE block, plus an appended dated status line):

```markdown
### T12 — Larger L2 megatiles · **ANSWERED 2026-08-23**
[`0007`](../0007-m5-pretiled-gemm-on-npu/TASK.md) §3, citing
[2602.06063](https://arxiv.org/abs/2602.06063)'s 5.9 → 13.7 TOPS from megatile size alone.
Was filed as a *bandwidth* lever, which T1 has now retired. If the megatile
result is real it must be acting through **iteration count** (bigger tiles) —
i.e. it is [T17](#t17)
in L2 clothing, and the paper is worth re-reading with that question.

> **UNDONE 2026-08-23**: T1's retirement is now datapath-scoped, and on int8
> the traffic model governs (R² 0.987). **This thread's ORIGINAL bandwidth
> framing is the live one there**, and 2602.06063's 5.9 → 13.7 TOPS may mean
> exactly what it said. Re-read the paper with the original question, not the
> re-pointed one.

> **ANSWERED 2026-08-23 ([`0093`](TASK.md)):**
> the paper's 5.9 → 13.7 TOPS conflates two levers, not "size alone" — the first
> step (128×512×512 → 256×256×512) is a **reshape at constant volume** (2.0×);
> only the second (→ 512×512×512, 4× volume) is a real size increase, for a
> further 1.14×. It is measured on **bf16**, the wrong dtype for the now-live
> question (int8 traffic-boundedness), though on the same `aie2p`/npu2 silicon
> family (Krackan Point, not our Strix Point HX 370). The mechanism — amortise
> each operand's L3→L2 fetch over a wider L2-resident block — **already exists
> in this project under two names**: `gemm_pretiled.py`'s `b_reuse="mega"`
> (B's megatile, blocked by mem-tile DMA channels per 0046, not L2 capacity)
> and `tile_n` (A's megatile axis, reopened independently by
> [`0080`](../0080-m13-int8-traffic-bound/TASK.md) §8 from the traffic
> model alone, before this paper was re-read). Predicted payoff on `ffn_up` at
> a legal `tile_n=96` (vs shipped 48): ~11% traffic reduction, ~1.03–1.05× on
> that shape alone via 0080's fitted model, shape-specific (`ffn_down`'s N=384
> cannot take n=96) and small next to
> [`0081`](../0081-m13-int8-everywhere/TASK.md) §3's finding that the
> array is only 30.4% of wall clock. Not a new discovery — confirms a lever
> this project had already found from its own data.
```

### T13 — still OPEN, re-scoped

**Verdict: still OPEN.** `tasks/0058` (M11, current toolchain) reproduces the
instability at 0007's exact isolated scale; `tasks/0085` (M13, production scale)
shows no trace of it end-to-end — different geometries, not a contradiction, and
the production-scale cell has never been directly tested.

Proposed replacement `OPEN-THREADS.md` entry (revised, stays in the open file):

```markdown
### T13 — Explain the pre-tiled instability · **OPEN** · re-scoped by 0093
[`0007`](../0007-m5-pretiled-gemm-on-npu/TASK.md) §4: best-case pre-tiled
runs match row-major exactly, so it is an intermittent stall, not a ceiling.
Pre-tiling was refuted as a lever, so this only matters if it is a symptom of
something else.

> **RE-SCOPED 2026-08-23 ([`0093`](TASK.md)):**
> confirmed to be neither an old-toolchain artifact nor still costing production
> anything, on the two pieces of evidence available. `tasks/0058` (M11, current
> mlir-aie) re-measured 0007's exact isolated config (M=512, 4 columns,
> `ffn_down`) and reproduced **spread ~10%**, unprompted, as a migration
> regression check — the instability is real on today's toolchain, not fixed by
> the 1.3.4→1.4.x upgrade. `tasks/0085` (M13, production scale: 8 columns,
> batch 8192, 4 lanes) measured end-to-end throughput spread **under 0.6% on
> five of six models** — far tighter than an unmasked 9–22% GEMM-level
> instability would leave after dilution by host work. **Neither result tests
> the other's geometry**, so this is not a contradiction and not a close: no
> measurement anywhere traces the pretiled access pattern's per-core stability
> AT production scale (8 columns, batch ≥ 1024). That is now the precise open
> question — a single traced `--repeat` run at production geometry would
> settle it either direction, and it is the cheapest experiment left on this
> thread.
```

### Proposed `tasks/README.md` index row

```markdown
| [0093](0093-t11-t12-t13-research/TASK.md) | Three cheap open threads settled by reading — T11 RETIRED (partition width is a driver-ioctl parameter on Linux, `num_tiles`, but unreachable from the XRT/IRON layer this project calls), T12 ANSWERED (2602.06063's megatile lever already exists here as `b_reuse="mega"` + `tile_n`, bf16-measured, ~1.03–1.05× predicted on one int8 shape), T13 re-scoped not closed (0058 reproduces the instability on the current toolchain at 0007's scale; 0085's production-scale spread is clean but never tested the same geometry) | research | done |
```
