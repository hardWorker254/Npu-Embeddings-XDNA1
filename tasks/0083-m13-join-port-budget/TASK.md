# 0083 — T28: separating the join's three walls, and a register that had gone stale

- **Date** 2026-08-23
- **Milestone** M13 (T28 follow-on)
- **Status** done — port budget measured in isolation for the first time; one
  correction to the register; one new instance of a known fail-open class

## Goal

**User:** *"For npu'en støtter streaming mellom tiles, så vi burde ha noen
muligheter der?"* → *"Enig i alt du sier. Men la oss teste litt! Kjør research
på det."*

Tile-to-tile streaming is the one lever that removes bytes from the shim rather
than moving them faster, which [`0080`](../0080-m13-int8-traffic-bound/TASK.md)
made the binding cost on int8. The question is what the array can actually
express.

---

## 1. The register was stale, and it cost this session an hour

**Before any measurement:** on 2026-08-22 I filed a derivation under
[T28](../../research/OPEN-THREADS.md) showing a hierarchical 8→2→1 join fits
the port budget where a flat 8-way does not, presented as a new insight.

**It was already built.** [`0062`](../0062-m11-t28-hierarchical-merge/TASK.md),
2026-08-20: the hierarchical 2-hop merge, real GELU at every producer, two merge
mem tiles each also serving its own column's A/B feed, a relay core reading both
hops via direct `.cons()`, and a real second-stage matmul. **PASS on hardware**,
rel_fro 1.726e-03 against a 3e-2 bound, bit-identical on repeat.

T28 stops at 0057 and never mentions 0062. So the register — the thing CLAUDE.md
rule 3 names as *the authority on what is still open* — said a question was open
that had been answered three days earlier, and its reader re-derived on paper
what was sitting in the tree. **Rule 3 exists because this happened twice
before; this is the third, and the first where the register itself was the
thing that misled.** Corrected in the same commit as this task.

The probe file's own header says `tasks/0061`, which is the **tokenizer** task —
0061's log records that "a concurrent session was doing NPU-heavy phase-fusion
work tonight". The phase-fusion session was 0062 and its probe mislabelled
itself, which is why a grep for the probe's name found the wrong log first.

---

## 2. What had never actually been tested

0057 concluded, with the compiler's own error text, that *"a single JOIN cannot
express the full 8-column `ffn_down` regather, full stop, on this hardware"*, and
0062 recorded GROUP=4 as ruled out by L1 arithmetic — 49,152 B for the one-shot
gather alone.

Re-running 0057's own `cross_column_join_probe.py` at `SRC_COLS=0,1,2,3` today
reproduces the GROUP=4 failure, and the failure is:

```
Y_out_buff_0      : 49152 bytes
C_mem_cons_buff_0 : 49152 bytes
error: 'aie.tile' op Basic sequential allocation failed.
```

**That is L1, not ports.** The port budget is never reached, so *"4 sources do
not fit"* has never been tested — only *"4 sources at production tile size do not
fit"*, which is a different claim and the one 0062 already knew. **Three walls
were being read off one data point:** L1 exhaustion, DMA port count, and stream
routing all fail the same build and only the error text distinguishes them.

---

## 3. The probe: make L1 irrelevant so only topology can fail

`experiments/m5-pretiled-gemm/join_port_budget_probe.py` (new). Tiles are
**512 floats (2 KB)** rather than production's 6,144, so the gathered buffer at
8 sources is 16 KB and the consumer's pair 32 KB — half the L1 budget. Each
producer lives in its own physical column and is fed from its own shim, so the
join is cross-column and the mem tiles carry traffic of their own, as in
production. The classifier reads the error text and names which wall was hit.

New kernel file `experiments/m5-eltwise/kernels/probe_copy.cc` — identity copies
at probe sizes, kept out of `gelu_poly.cc` because that file's two copies are at
*production* tile sizes, which is exactly what confounds this experiment.

### Result — the sweep

| sources | verdict |
|---:|---|
| 2 | **PASS**, rel_fro 0.000e+00 |
| 3 | **PASS**, bit-exact |
| 4 | **PASS**, bit-exact |
| **5** | **PASS**, bit-exact |
| 6 | FAIL — `tile (7,1) requires 1 input/1 output DMA channels, but only 0 input/5 output available` |
| 7 | FAIL — `requires 7 input/1 output ... only 6 input/6 output available` |
| 8 | FAIL — `requires 8 input/1 output ... only 6 input/6 output available` |

Read the 6-source message carefully: it is **not** the join that fails. The
join's six sources consume all six input channels, and then the *outbound relay*
on the same tile has none left — "0 input available" for a hop needing 1. At 7
and 8 the join alone exceeds.

**So the number is 5, not 4**: a mem tile serving both a join and an out-relay
takes **5 sources**; a bare join takes 6. That is the same three-failure
structure 0057 described, now measured at a tile size where L1 cannot be
confounding it, and with the middle case pinned exactly.

**Consequence for the regather.** 8 → 5+3 → 1 fits in two tiers, and 8 → 4+4 → 1
has a spare port on each. The hierarchical form 0062 built at GROUP=2 has more
headroom than it used.

---

## 4. Production scale fits — 0062's own open item, now estimated

0062's §"What was not attempted" says scaling to production *"would need EITHER
more than 2 hops (impossible — a core has exactly 2 input channels) or a
fundamentally different relay strategy (e.g. streaming smaller N_DOWN chunks per
hop) … Not designed or estimated this session."*

Estimated here. The premise of the 49,152 B figure is a relay that **gathers the
whole K chunk at once**. It does not have to: `ffn_down`'s K-reduction *is a
reduction*, so the relay can acquire one k-block, MAC into its accumulator,
release, and acquire the next — which is exactly what `gemm_pretiled.py`'s own
K-loop already does. Then nothing but one k-block is ever resident:

| k | N_DOWN | A (2×) | B (2×) | acc | out | total | 63 KB? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 32 | 16,384 | 8,192 | 8,192 | 4,096 | 36,864 | fits |
| **64** | **48** | **16,384** | **12,288** | **12,288** | **6,144** | **47,104** | **fits, 17,408 B spare** |
| 64 | 64 | 16,384 | 16,384 | 16,384 | 8,192 | 57,344 | fits |
| 64 | 96 | 16,384 | 24,576 | 24,576 | 12,288 | 77,824 | over |
| 128 | 32 | 32,768 | 16,384 | 8,192 | 4,096 | 61,440 | fits |

`(k, N_DOWN) = (64, 48)` is **production's own tile**, and it fits with 27%
spare. The relay's two input channels go to A (from the merge) and B (weights) —
2 of 2, exactly full, no spare.

**This is an arithmetic result, not a measurement.** What it establishes is that
the L1 wall 0062 hit was a property of the one-shot gather it chose, not of
production geometry. Whether a streaming relay of that shape actually routes,
and what it costs in latency against the four dispatches it replaces, is the
build nobody has done.

---

## 5. Problems hit

**The fourth instance of the marker-specificity fail-open** (0030, 0053, 0054
have the other three). The probe's cache-purge marker keyed on
`memref<{GATHER}xf32>`, which is *different for every N_SRC* — so sweeping
N = 2,3,4… never matched the previous build, `@iron.jit` decided it was a cache
hit (it keys on the decorated function, and `N_SRC` is read inside `build()`
where the decorator cannot see it), and **every N silently ran the N=2 binary**.

It announced itself only by luck:

```
Tensor argument 'X' has 1536 elements but the kernel was compiled for 1024 elements
```

A sweep whose host-side shapes happened to agree would have printed N=2's result
under N=8's label, and the conclusion would have been "8 sources work". Fixed by
matching on `probe_copy_512_f32`, which identifies the **family** rather than the
instance, and purging every member.

**The rule this keeps re-teaching:** a cache marker must be *coarse enough to
catch every build of the same probe* and *specific enough to catch nothing else*.
Keying on a value that varies across the sweep gets it exactly backwards.

Two smaller ones: `Tile` imports from `aie.iron.device`, not `aie.iron`; and
`iron.tensor()` defaults to `uint32`, so the input is built with
`iron.zeros(..., dtype=np.float32)` and written through `__setitem__` (trap 6b —
`.numpy()` is a host-only view the device never sees).

---

## 6. Commands

```powershell
. C:\dev\mlir-aie\iron_env.ps1
cd experiments\m5-pretiled-gemm

# the sweep
foreach ($n in 2,3,4,5,6,7,8) {
    $env:NPUE_N_SRC = "$n"; $env:NPUE_DEST_COL = '7'
    Remove-Item Env:NPUE_SRC_COLS -ErrorAction SilentlyContinue
    python -u join_port_budget_probe.py
}

# the L1-confounded control this replaces
$env:NPUE_SRC_COLS='0,1,2,3'; $env:NPUE_DEST_COL='5'
python -u cross_column_join_probe.py     # -> "Basic sequential allocation failed"
```

---

## 7. What this leaves open

* **Build the streaming relay at production geometry.** §4 says it fits; nothing
  says it routes or that it pays. Two extra hops add latency to a chain whose
  purpose is removing it.
* **Does a second `dims_from_stream` on a join's `.cons()` compose with the base
  object's own `dims_to_stream`?** 0062 flagged this and it is still untested
  anywhere. It decides whether the relay can use the MMAC-accelerated
  `kernels.mm()` or is stuck with a hand-written matmul — which decides whether
  the fusion is worth having at all.
* **A bare 6-source join** (no out-relay on the same tile) is expressible per the
  error text but was not built; it would need the consumer on a different tile.
