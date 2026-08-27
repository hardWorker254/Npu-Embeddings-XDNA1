# 0126 — T48 gate 1: the channel re-census, and the hope it was built on is refuted

**Date**: 2026-08-27
**Goal**: T48's first pre-build check — re-census the DMA channels on the
*shipped* design sets, because 0046's census predates C narrowing to bf16 and
T48 conjectured the C join might now be cheaper. One python invocation, no
build, exactly as the thread priced it.

## Matching shipped sets to cache dirs (trap 7c: by contents, never mtime)

`tools/count_dma_channels.py` reads `input_with_addresses.mlir`, which lives
in the JIT cache, not in the artifacts directories. The cache dir for each
shipped set was found by byte-comparing `final.xclbin` against every cache
entry of the same size:

| shipped set | cache dir | diff |
|---|---|---|
| `artifacts_minilm_bfp16` | `5ecbb3a4d92a4174c6985dba` | **0 bytes** |
| `artifacts_large_bfp16` | `231987bc39a1d44bd534395f` | **0 bytes** |
| `artifacts_small_bf16` | `1c680019f65684a61a3775b6` | **0 bytes** |

(bge-base and nomic's xclbins differ from dozens of cache entries by only
~70 bytes — the UUID — because the xclbin encodes tile geometry, columns and
datapath, not the GEMM shapes, which live in the instruction streams. Their
channel structure is the `(64,64,48)×8` class the MiniLM match covers; the
three exact matches cover all three shipped geometry classes: 48×8 bfp16,
32×8 bfp16, 48×8 plain-bf16.)

```
python tools/count_dma_channels.py C:\Users\vegar\.npu\cache\<dir>
```

## The census — identical across all three classes

```
MEM TILES   budget 6 in / 6 out
  (0,1) in 5/6 out 2/6    (1,1) in 5/6 out 2/6
  (2,1)..(6,1) in 6/6 out 3/6   <-- FULL (five of eight)
  (7,1) in 4/6 out 1/6
CORE TILES  budget 2 in / 2 out
  all 32: in 2/2 FULL, out 1/2
```

## What it means for T48

**The conjecture is refuted.** T48 hoped the C join "may be cheaper since C
narrowed to bf16". It is not, and the reason is structural: **a DMA channel
is an integer allocation, and narrowing an element from 4 to 2 bytes changes
the bytes per transfer, not the number of channels the join occupies.** The
census is bit-for-bit the one 0046 took on the pre-narrowing design: every
core tile 2/2 in, five of eight mem tiles 6/6 in, the C join spending 4 of
each full mem tile's 6 inputs.

So T48's gate (b) — "the census shows an expressible wiring" — **fails on
the current dataflow topology.** A B-reuse build cannot be a fifth fifo into
the existing structure; it requires re-plumbing the C join itself (the
cascade collapse 0047 probed — four C returns become one, freeing 3 mem-tile
inputs at the price of 3 outputs, with `kernels.cascade_mm`'s scalar kernel
still the unfunded part), or a hierarchical re-join. That does not kill the
thread — the *pricing* (1.80–2.02× array on paper) still awaits T45's
measured roof and fixed cost — but it hardens the build's floor cost: **the
cheap version of B-reuse does not exist**, exactly as 0046 concluded, and
now confirmed on the datapath that actually ships.

Two spare inputs exist on `(0,1)`/`(1,1)` (the A-carrying columns' mem
tiles) and two on `(7,1)` — an asymmetry a hierarchical B-staging design
could in principle exploit, but three tiles' spares cannot broadcast to
eight columns without transiting the five full ones.

## Commands run

```
python - <<EOF   # content-match shipped xclbins against the cache (script in session log)
EOF
python tools/count_dma_channels.py C:\Users\vegar\.npu\cache\5ecbb3a4d92a4174c6985dba
python tools/count_dma_channels.py C:\Users\vegar\.npu\cache\231987bc39a1d44bd534395f
python tools/count_dma_channels.py C:\Users\vegar\.npu\cache\1c680019f65684a61a3775b6
```

## Status

T48 stays **OPEN**: gate 1 is done (this task), gate 2 — the re-price on
T45's measured roof and intercept — is not. The thread's annotation now
carries this census so the gate decision reads one entry, not three tasks.
