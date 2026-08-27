# 0131 — T48 gate 2: B-reuse re-priced on measured numbers at 1.72× array / 1.31× e2e — and the build is not funded, because the census says the cheap version doesn't exist

**Date**: 2026-08-27
**Goal**: the 0.5.0 plan's B-reuse decision gate, now that its inputs exist:
the measured roof and validated y-axis
([`0130`](../0130-t45-traced-roof/TASK.md): **45.5 GB/s**, wall agreeing to
0.4%), the fixed-cost fits ([`0128`](../0128-t45-m-sweep/TASK.md)), and the
channel re-census ([`0126`](../0126-t48-channel-recensus/TASK.md)).

## The re-price (gate condition a)

Conservative, model-light: subtract **only the saved B bytes** at the
measured roof from the **measured** 8-column dispatch times (bge-large
bfp16, 0128's raw probe, 30 repeats each). B-reuse keeps B staged, so 31 of
32 row-block re-streams vanish; nothing else is credited.

| op | measured now (µs) | B now (MB) | saved (µs) | with reuse (µs) |
|---|---:|---:|---:|---:|
| qkv | 10,346 | 201.3 | 4,286 | 6,060 |
| attn_out | 4,247 | 67.1 | 1,429 | 2,818 |
| ffn_up | 13,349 | 268.4 | 5,715 | 7,634 |
| ffn_down | 13,183 | 268.4 | 5,715 | 7,468 |
| **Σ** | **41,125** | | | **23,979** |

**Array gain 1.715×**; at 0109's measured array share of wall (bge-large,
fused build, 56.7%): **end-to-end 1.310×**.

The register's own prediction — *"expect the real number to be below
1.80×"* — **was right**: 1.72 against the paper price's 1.80–2.02. The gap
is the fixed cost and the sub-roof legs the byte-proportional model ignores.
This estimate is itself optimistic in one way (it assumes the freed
bandwidth is fully absorbed by the remaining A/C traffic and compute at the
same rate) and pessimistic in none we can name; treat 1.72×/1.31× as the
ceiling a build would chase.

## The wiring (gate condition b) — FAILS

0126's census, taken on the shipped sets: every core tile 2/2 in, five of
eight mem tiles 6/6 in, the C join spending 4 of 6 — unchanged by C's
narrowing, because channels are integer allocations. There is **no spare
input path for a staged-B leg on the current topology**. A build is
therefore not "add a fifo": it is the cascade C-collapse (0047 — dataflow
proven on hardware, kernel scalar at 3.19 GFLOPS, vectorisation unfunded)
or a hierarchical C re-join, plus re-validating the `.npue` B layout if
`k` changes. 0047's own words: *"that is not a session."*

## The decision

Per the approved 0.5.0 plan (build iff array gain ≥ ~1.5× **and** the
census shows expressible wiring): **(a) holds at 1.72×, (b) fails — the
build is not funded in 0.5.0.** The gain is real and above the retire line
(1.2×), so T48 **stays OPEN as a priced design task** with the T42/T43
shape: a measured prize and a named trigger, no open question about the
hardware left inside it.

**The trigger, named**: fund the build when someone funds the
vectorised cascade microkernel (or another re-plumb freeing ≥1 mem-tile
input channel per column) — the measured prize is **1.72× array / 1.31×
e2e on bge-large**, and the first target is bge-large at k=64 (its `.npue`
B layout unchanged; k=32 at h=384 would change the container format).
Re-run this arithmetic if the roof, the array share, or the dispatch times
move.

## Commands run

```
python - <<EOF   # the re-price table above (script inline in session log; inputs:
                 # 0128 raw probe times, 0130's 45.5 GB/s, 0109's 56.7% share)
EOF
```

All inputs are stored artifacts: `tasks/0128-t45-m-sweep/raw.txt`,
`tasks/0130-t45-traced-roof/trace.json`, 0109's share table.
