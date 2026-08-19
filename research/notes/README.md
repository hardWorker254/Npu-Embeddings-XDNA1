# research/notes/

Our own findings — as opposed to `papers/` (other people's published work) and
`prior-art.md` (other people's implementations).

**What belongs here:** things learned by doing that are *durable* but not big enough
for a `docs/` page, and not tied to a single day's work.

- Hardware behaviour that contradicts or refines the documentation
- Toolchain quirks discovered the hard way
- Micro-benchmark results that inform a decision but aren't a kernel design
- Dead ends worth remembering, with the reason they were dead

**What does not belong here:**

| Content | Goes to |
|---|---|
| What happened on a particular day, with commands | [`../../tasks/`](../../tasks/README.md) |
| Durable truth about hardware, toolchain, model, measurement | [`../../docs/`](../../docs/00-overview.md) |
| A specific kernel's design and results | [`../../docs/03-kernels/`](../../docs/03-kernels/README.md) |
| Summary of a paper | `../papers/` |

The distinction that matters: **`tasks/` is a diary, `docs/` is a manual, `notes/` is a
lab notebook.** If a note stabilises into something we rely on, promote it into `docs/`
and leave a pointer.

Naming: `NNNN-short-slug.md`, sharing the numbering space with nothing else — just
sequential within this folder.

## Index

| # | Note | From |
|---|---|---|
| [0001](0001-aie-kernel-pitfalls.md) | AIE kernel pitfalls found by disassembly — scalar float becomes a library call (1,617× measured), `__divsi3`, nop-padded loops, empty traces | M1 |
| [0002](0002-iron-silent-arch-fallback.md) | **IRON silently compiles for NPU1 on an NPU2 machine** unless the device is set explicitly — halves MACs and makes the bfp16 flag (worth 5.5×) a no-op, with no error | M2 |
| [0003](0003-two-designs-per-process.md) | **`Tensor.numpy()` writes never reach the device** — only the first dispatch per process hides it, and a read-back "confirms" the write landed. Includes the retracted "two designs" misdiagnosis and why it survived every confirming test | M5 |
- [0004 — Switching design costs more than computing, and it scales with descriptors](0004-context-switch-cost.md) — the most expensive operation in the stack is changing which design the array is configured for: ~25 µs + 7.2 µs per lock, 89 µs to 2.4 ms. Not reconfiguration-by-difference, not eviction, not residency.
- [0005 — Testing the expert review, claim by claim](0005-expert-review-tests.md) — living scoreboard: 7 of 10 claims confirmed on hardware, 1 refuted by an existing measurement, 2 deferred with a measured pricing argument. **§6b (device-resident intermediates) is still open** and was unblocked by 0032 — see 0007 Part 4.
- [0006 — which loop hints survive Peano, and which are decoration](0006-peano-loop-hints.md) — `AIE_PREPARE_FOR_PIPELINING` compiles to **nothing** under Peano (8 uses in our kernels); the min-iteration hint does bind; the unroll hints were declined by inference and are now **closed by someone else's measurement** (see 0007 §1.9).
- [0007 — the IRON surface we never typed, and what four outside repos already measured](0007-unused-iron-surface.md) — twelve IRON features that came back **zero** when grepped against our own tree (`pad_dimensions`, cross-tile `Buffer`, `CascadeFlow`, `consumer_obj_type`, `disable_synchronization`, hand-wired `TileDma`/`Bd`, …), two levers closed by inspection (`burst_length` is already maximal; the arch fallback halves it), `xrt::runlist` sitting unused in our own XRT, the AIE default rounding mode being **`floor`** — one line (`set_rounding(conv_even)`) worth 1.73×/1.29×/1.62× on GELU, softmax and LayerNorm, and 92× on LayerNorm's implementation error, closing a mystery 0015 misattributed and 0016 left open for 28 tasks — and the correction that matters most: **host eltwise costs 7.5% of an encode while the host↔device transport it forces costs 33%** — and contention had been flattering that ratio, not inflating it.
