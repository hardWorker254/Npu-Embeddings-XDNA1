# 0046 — M9: is B-reuse expressible? A DMA-channel census says no, and says why

**Question.** [`0044`](../0044-m9-optimisation-sweep/TASK.md) identified
`consumer_obj_type=` as the primitive that might unblock B reuse, which
[`0010`](../0010-m5-b-reuse-and-cost-model/TASK.md) priced at **1.26–1.68×** and
failed to build twice. This is the feasibility probe: **does it compile**, and
what does it cost in descriptors?

**Answer: no, and not for the reason anyone had written down.** The production
GEMM at 8 columns is already at **100% of the DMA channel budget on both tile
types**. There is no channel to give B, in any form.

Deliberately no timing here. A stopwatch on a design that does not build
measures nothing.

---

## What was built

`gemm_pretiled.py` gained `b_reuse="asym"`: one `ObjectFifo` carrying the whole
column slice on the producer side and `consumer_obj_type=B_l1_ty` on the
consumer side, with `repeat_count=n_row_blocks` to replay the staged bytes —
the shape upstream's `ml/mobilenet/bottleneck/post_l1.py` uses to hold a weight
buffer on a mem tile and feed compute tiles chunks of it.

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python <scratchpad>\probe_asym.py --mode off     # baseline, builds
python <scratchpad>\probe_asym.py --mode asym    # new
python <scratchpad>\probe_asym.py --mode mega    # 0010's second attempt
python <scratchpad>\count_dma.py <cache-dir>     # the census
```

Shape is production `ffn_up` at batch 128: `8192 × 384 × 1536`, tile
`(64, 64, 48)`, 8 columns. Column B slice = 24 tiles = **144 KB of a 512 KB mem
tile**, and B is re-streamed **32×** today (`M/m/rows`).

---

## Result 1 — baseline, for calibration

| | |
|---|---:|
| `aie.dma_bd` | 880 |
| `aie.lock` | 352 |
| implied switch cost, [`0024`](../0024-m7-dispatch-cost-anatomy/TASK.md)'s model (~25 µs + 7.2 µs/lock) | **2.56 ms** |

`CLAUDE.md` records "2.4 ms for an 8-column GEMM" from 0024, measured on
hardware. The static count reproduces it to 7%, which is the probe validating
itself before it is trusted on anything new.

## Result 2 — `asym` does not build, and the reason is a hard rule

```
error: `repeat_count` unavailable for shim tiles
 note: "aie.objectfifo"(...) {consumerElemType = !aie.objectfifo<memref<64x48xbf16>>,
        elemType = !aie.objectfifo<memref<73728xbf16>>, repeat_count = 32 : i32,
        sym_name = "B_L3L2_0"}
```

The asymmetric fifo is filled from L3, so its producer sits on a **shim** tile,
and **`repeat_count` requires a mem-tile producer** — which matches the API doc
exactly ("causes the *MemTile* DMA to replay the buffer descriptor"). The
single-fifo form is therefore inexpressible **by construction**, not by
resource pressure.

Nor can the two-fifo form carry it: `ObjectFifoHandle.forward()` is
`split([0], ...)` and **neither exposes `consumer_obj_type`**. Reaching it needs
a hand-built `ObjectFifoLink`, i.e. dropping to §2g's primitives.

## Result 3 — `mega` reproduces 0010's failure verbatim, at a different tile

```
error: 'aie.tile' op number of input DMA channel exceeded!
 note: see current operation: "aie.tile"() <{col = 3 : i32, row = 1 : i32}>
```

Same message 0010 recorded, on today's toolchain. **But `row = 1` is a MEM
TILE**, not a core: `BaseNPU2TargetModel` lays out "1 Shim row, 1 memtile row,
and 4 Core rows", with `getTileType` returning `MemTile` for row 1.

`gemm_pretiled.py`'s own comment says the mega workaround *"hits number of input
DMA channel exceeded on a **core** tile"*, and
[`0010`](../0010-m5-b-reuse-and-cost-model/TASK.md) says it "needs a core-side
redesign, not a fifo change". **That attribution is wrong.** The pressure is on
the mem tile, and a core-side redesign would not have helped.

---

## Result 4 — the census, which is the actual finding

`count_dma.py` counts `aie.dma_start(S2MM|MM2S)` per tile region in
`input_with_addresses.mlir` — the **post-placement** module, because `aie.mlir`
is pre-placement and its counts are meaningless (trap 7c). Budgets from
`CLAUDE.md` trap 3b: core 2 in / 2 out, mem tile 6 in / 6 out.

**Baseline production design, 8 columns:**

| mem tile | in | out | | core tiles | in | out |
|---|---:|---:|---|---|---:|---:|
| (0,1) | 5/6 | 2/6 | | **all 32** | **2/2** | 1/2 |
| (1,1) | 5/6 | 2/6 | | | | |
| **(2,1)–(6,1)** | **6/6** | 3/6 | | | | |
| (7,1) | 4/6 | 1/6 | | | | |

**Every one of the 32 core tiles is at 2/2 input channels. Five of eight mem
tiles are at 6/6.** The design has *zero* input-channel headroom where it
matters.

The mem-tile arithmetic is exact and explains the uneven distribution:
`A (1 from shim) + B (1 from shim) + C (4 from the column's four cores) = 6`.
Columns 0, 1 and 7 sit lower only because `n_shim_mem_A = 4`, so just four
columns fetch A at all.

### What this means

**B-reuse is not blocked by capacity, by descriptors, or by the fifo API. It is
blocked by there being no spare input channel on the tiles it would have to
pass through.** The C join alone consumes 4 of the mem tile's 6 inputs, and
that is structural: four core rows each returning a C tile.

This retires the lever in its current form, and it retires it with a number
rather than a compile error. It also corrects two things the repo believed:

1. **The mega failure is at the mem tile, not the core** (0010, and the comment
   in `gemm_pretiled.py`).
2. **"It needs a core-side redesign"** — no. The core tiles are full too (2/2),
   but the op that failed is on row 1.

---

## What would actually free a channel

Not attempted here; recorded so the next attempt starts from the constraint
rather than from the primitive.

- **Fewer C inputs per mem tile.** The join is 4 in because there are 4 core
  rows. Cascade (`CascadeFlow`, [note 0007](../../research/notes/0007-unused-iron-surface.md) §1.3)
  moves partial sums **core to core with no DMA at all**, so a column could
  return C through **one** core instead of four — freeing 3 of the 6 mem-tile
  inputs. That is the single biggest structural change available, and it is the
  one primitive in note 0007 that adds no descriptors.
- **Narrower designs.** Tiles (0,1), (1,1) and (7,1) have slack at 8 columns;
  the pressure is uneven. Whether any width has slack in *every* column is
  untested.
- **`aie_stream=`** (note 0007 §1.6) makes a producer wire-only with no L1
  buffer. Whether it also frees a channel is unknown and worth one probe.

## Status

**Closed as infeasible in this form.** `b_reuse="asym"` stays in the tree,
guarded and documented, because the next person will otherwise reach for the
same primitive; the code now carries the census result next to it.

The 1.26–1.68× that [`0010`](../0010-m5-b-reuse-and-cost-model/TASK.md) priced
is still real and still unclaimed — it just costs a dataflow redesign
(cascade for the C return), not a fifo option.
