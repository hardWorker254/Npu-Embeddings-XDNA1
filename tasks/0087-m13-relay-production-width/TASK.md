# 0087 — T28's streaming relay at production width: it fits, and then it hangs

- **Date** 2026-08-23
- **Status** partial — the L1 question is answered (and my own arithmetic
  corrected); the design compiles at production width and hangs, narrowed to
  "a bf16 output ObjectFifo on this relay" with four suspects refuted

## Goal

**User, item 3 of three:** *"Så T28s strømmende relay — arrayet er 40% igjen,
portbudsjettet tåler 5, og produksjonsgeometrien passer på papiret."*

"På papiret" was [`0083`](../0083-m13-join-port-budget/TASK.md) §4, which I
wrote. This task tests it, and the paper was wrong by a specific, findable
amount.

---

## 1. The starting point

[`0062`](../0062-m11-t28-hierarchical-merge/TASK.md) built the hierarchical
2-hop merge and passed on hardware at `N_DOWN = 16` — a width chosen so one
output row is exactly one 16-lane fp32 vector. Its own §"What was not
attempted" names production scale and says *"not designed or estimated"*.

Reproduced first, unchanged: **rel_fro 1.726e-03**, digit-for-digit 0062's
number. Everything below is measured against that control.

---

## 2. 0083's L1 budget was optimistic by exactly 8,192 bytes

0083 §4 derived a streaming relay at `(k, N_DOWN) = (64, 48)` costing **47,104
of 64,512**, and separately I priced the two-hop form at 61,440. Built at
`N_DOWN = 48` with an fp32 output, the compiler answers with its own map:

```
	(stack)              : 0x0-0x7FF      (2048 bytes)
	Y_out_buff_0         : 0x800-0x37FF   (12288 bytes)
	C_mem_g1_cons_buff_0 : 0x3800-0x67FF  (12288 bytes)
	C_mem_g0_cons_buff_0 : 0x6800-0x97FF  (12288 bytes)
	ffn_down_acc         : 0x9800-0xC7FF  (12288 bytes)
	w_hop0               : 0xC800-0xEBFF  ( 9216 bytes)
	w_hop1               : 0xEC00-0x10FFF ( 9216 bytes)
error: 'aie.tile' op Basic sequential allocation failed.
```

**69,632 of 65,536.** Two errors in my arithmetic, and they sum to the gap
exactly:

| | I assumed | actually |
|---|---:|---:|
| output buffer | 6,144 (bf16) | **12,288 (fp32)** |
| stack | 0 | **2,048** |

6,144 + 2,048 = 8,192, and 61,440 + 8,192 = 69,632. The stack one is
inexcusable — CLAUDE.md trap 3 says in as many words that *"the limit is 63 KB,
not 64 — 1 KB of the 64 KB DMEM is reserved for the program stack"*, and this
design's worker asks for 2 KB.

**The fix is the one [`0080`](../0080-m13-int8-traffic-bound/TASK.md) already
found for the GEMM: narrow the output to bf16.** 63,488 of 65,536, 2,048 spare.
So the relay needs C-narrowing for the same reason the GEMM did, arrived at
from the opposite direction — there it bought bandwidth, here it buys the
address space to exist at all.

---

## 3. It compiles at production width, and hangs

With the output narrowed, `N_DOWN = 48` allocates and compiles. It then returns
`ert_cmd_state.ERT_CMD_STATE_TIMEOUT` — the compile-clean-hang-silently class
0054 hit three times.

**Isolated to one variable.** Two changes went in together (width *and*
narrowing), so both were tested apart:

| `N_DOWN` | output | result |
|---:|---|---|
| 16 | fp32 | **PASS**, rel_fro 1.726e-03 (0062's control) |
| 16 | **bf16** | **HANG** |
| 48 | fp32 | L1 allocation failure (§2) |
| 48 | bf16 | **HANG** |

**The WIDTH is not the cause.** N_DOWN=16 with a narrowed output hangs at a
geometry that passes with fp32, which rules out the port budget, the weight
size and the k-block streaming in one comparison. (§4 goes one further and
rules out the narrowing kernel too — it is the bf16 output *fifo*.)

**The kernel's own documented first suspect is refuted.**
`narrow_f32_bf16.cc`'s header says gemm_pretiled runs it at `stack_size=0xD00`
and that *"if this ever hangs or corrupts, the stack is the first suspect
(traps 5b/0031)"*. Raised from 0x800 to 0xD00: **still hangs.** Recorded so the
next session does not spend the same hour.

---

## 4. What is left, and what it is worth

**SPLIT, and it is the dtype.** The discriminator this section first listed as
future work was built: `NPUE_BF16_COPY=1` makes the relay's accumulator bf16 and
its output kernel a **plain bf16 → bf16 copy** — no narrowing anywhere, no fp32
in the output path at all.

**It still hangs.** So the failing thing is **a bf16 output ObjectFifo on this
relay**, not `narrow_f32_bf16`. Four hypotheses are now dead:

| suspect | verdict |
|---|---|
| the narrowing kernel | **refuted** — a plain bf16 copy hangs identically |
| the relay's stack | refuted — 0x800 → 0xD00, still hangs |
| the producing fifo's depth | refuted — 1 → 2, still hangs |
| the output width | refuted — N_DOWN=16 hangs, and passes at fp32 |

And one comparison narrows it further without being conclusive: **this repo
already drains bf16 from a core to the shim successfully** —
`gemm_pretiled.py --c-bf16` does exactly that, through a join into a mem tile
and out. Its drain tap is a `step_tiler` over the full `(M, N)`; the probe's is
a `simple_tiler((TM, N_DOWN))[0]`. Both count *elements*, so the tap is not
obviously the difference, but the two designs' C paths differ in shape
(join-into-mem-tile vs `forward`) and that is where a next session should look
first.

Remaining suspects, cheapest first:

1. **`forward()` vs `join()` as the core→shim hop.** The working bf16 drain uses
   a join; this probe uses a bare `forward` with no `tile=`, so its placement is
   compiler-chosen. 0054's Problem #2 was that *"a bare point-to-point
   ObjectFifo with no `.forward()` compiles but hangs the hardware with no
   diagnostic"* — a neighbouring shape, and worth re-reading before guessing.
2. **`aie::set_rounding` as core-wide state** — now much less likely, since the
   bf16-copy discriminator sets no rounding mode and hangs anyway.
3. The `--aie-objectfifo-liveness` pass T28 records as landed upstream and
   unavailable here, which turns one class of exactly this failure into a
   compile error.

**And the honest scoping:** even resolved, this is a probe. It runs one k-block
pair through a 4-column merge, not `ffn_down`'s K=1536 across 8 columns, and
its relay matmul is hand-written rather than MMAC-accelerated — 0062's open
question about whether a second `dims_from_stream` composes with a join's own
`dims_to_stream` is still what stands between this and a design that could
*pay*. This task moved the wall from "N_DOWN=16 toy" to "production width, with
the L1 arithmetic corrected and the failure isolated", which is a step and not
the thing.

---

## 5. Also: the fifth marker-specificity fail-open

Making `N_DOWN` a parameter meant the cache marker included it — so sweeping
16 → 48 never matched the *previous* build, `@iron.jit` called it a cache hit,
and the N=48 run silently executed the N=16 binary. It announced itself only
because the host-side element count disagreed (`Tensor argument 'Y' has 3072
elements but the kernel was compiled for 1024`).

Fifth instance in this repo (0030, 0053, 0054, 0083 are the others), and the
second in two days. **The rule that keeps being relearned:** a cache marker must
name the kernel *family*, never the instance under test — keying it on the
variable you are sweeping guarantees it never matches the build you need to
purge.

---

## 6. Commands

```powershell
. C:\dev\mlir-aie\iron_env.ps1
cd experiments\m5-pretiled-gemm

$env:NPUE_N_DOWN='16'; $env:NPUE_NARROW_OUT='0'; python -u hierarchical_merge_ffn_probe.py  # PASS
$env:NPUE_N_DOWN='16'; $env:NPUE_NARROW_OUT='1'; python -u hierarchical_merge_ffn_probe.py  # HANG
$env:NPUE_N_DOWN='48'; $env:NPUE_NARROW_OUT='0'; python -u hierarchical_merge_ffn_probe.py  # L1
$env:NPUE_N_DOWN='48'; $env:NPUE_NARROW_OUT='1'; python -u hierarchical_merge_ffn_probe.py  # HANG
```

---

## Correction (2026-08-23, appended by tasks/0092 -- per CLAUDE.md rule 3b, not editing section 4 in place)

**Section 4's `NPUE_BF16_COPY=1` discriminator was invalid, and its
conclusion should not have been trusted on that test alone.** The
discriminator declared the relay's accumulator Buffer bf16 (2048 B) but kept
calling `ffn_down_hop_matmul_g2_64x48x16`, which is compiled from C++ whose
actual signature is `float *restrict acc` (4096 B of stores). IRON's
`ExternalFunction.arg_types` shapes only the MLIR call-site declaration
(`kernel.py`: `external_func(..., inputs=self._arg_types)`), never the
linked object -- so this call was silently writing 4096 B into a 2048 B
buffer. **Confirmed by reading the toolchain source, the compiler's own
`input_with_addresses.mlir` memory map (`ffn_down_acc` allocated exactly
2048 B, the last buffer on the tile), and `llvm-objdump -d` on the cached
`.o` (a 64-byte-stride vector store, 64 rows = 4096 B written)** -- see
[`tasks/0092`](../0092-t28-relay-bf16-output/TASK.md) Part 1 for the full
evidence and `artifacts/bf16_copy_overflow_evidence.txt` there for the raw
excerpts.

**"It still hangs" was therefore not valid evidence that the fifo, and not
`narrow_f32_bf16`, was the cause** -- the discriminator's OWN new bug was
by itself sufficient to explain a hang, so the test proved nothing about
which suspect was right.

**The conclusion turns out to be correct anyway, but only by a sound
re-test.** 0092 built a genuinely bf16-typed twin kernel (no overflow,
verified consistently bf16 end-to-end in its own
`input_with_addresses.mlir`) and it hangs identically. So "the failing
thing is a bf16 output ObjectFifo, not `narrow_f32_bf16`" **does hold** --
just not for the reason section 4 originally gave. 0092 went further:
a construction that bypasses ALL upstream compute (writes a hardcoded
constant pattern straight to the bf16 output, touching neither the
accumulator nor the weight buffers) **also hangs**, which section 4's own
four "dead" suspects did not test and which narrows the failure to the
output path itself, independent of everything upstream of it. See 0092 for
the full, corrected suspect table.

---

## Second correction (2026-08-23, same day, appended by tasks/0092 Part 4 -- the hang is RESOLVED)

**The hang described throughout this task, and the "bf16 output ObjectFifo"
framing above, was never actually inside the ObjectFifo/ForwardFIFO
mechanism at all.** A coordinator-directed diff of the emitted
`aiex.dma_configure_task_for`/`aie.dma_bd` ops between this task's own
passing fp32 build and hanging bf16 build (same design, only the output
dtype differs) found it directly: `hierarchical_merge_ffn_probe.py`
declared

```python
Y_ty = np.ndarray[(Y_SIZE,), np.dtype[np.float32]]
```

**unconditionally**, never reading `NARROW_OUT` -- so the compiled design's
OUTER host-facing signature stayed fp32 even when every internal
ObjectFifo buffer was correctly bf16. The shim DMA BD this produces reads
`aie.dma_bd(%arg2 : memref<1024xf32> offset = 0 len = 1024 ...)` against a
host buffer `main()` actually allocates at 1024 elements of bf16 (2048 B,
not the 4096 B the BD is configured to move) -- the mem-tile side signals
completion after 2048 B, the shim side waits for 4096 B that will never
come. Fixing the one line (`Y_ty` now reads `NARROW_OUT` like every other
dtype-conditional in this design already did) removed the hang entirely,
with no ObjectFifo, fifo depth, mem-tile placement, or kernel change
required.

**So section 4's "the failing thing is a bf16 output ObjectFifo" was itself
imprecise, not just imprecisely evidenced**: the ObjectFifo mechanism was
never broken; the bug was one line outside it, in the compiled design's
own declared I/O contract with the host. This is why the extensive Part
1-2 bisection in [`tasks/0092`](../0092-t28-relay-bf16-output/TASK.md)
(five suspects, all correctly refuted with real evidence) never found it --
every suspect it tested lived inside the chip, and the bug lived in what
the chip was told the host would send it.

**Production width now passes.** `N_DOWN=48`, real hierarchical 2-hop
merge, real GELU, real second-stage matmul, bf16 output: rel_fro 2.510e-03
against the 3e-2 tolerance (a second, independent L1-budget bug --
`Y_out` needlessly double-buffered when narrowed -- was found and fixed
immediately after, by the very next production-width attempt). See
[`tasks/0092`](../0092-t28-relay-bf16-output/TASK.md) Part 4 for the full
diff, the fix, and every command run.
