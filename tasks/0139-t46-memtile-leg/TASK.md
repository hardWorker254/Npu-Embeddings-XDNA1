# 0139 — T46 closed: the memtile↔L1 leg has measured headroom, the DRAM leg binds, and the roofline now knows both

**Date**: 2026-08-27
**Goal**: T46 — the roofline models only the DRAM leg, of the three
(DRAM↔shim, memtile↔L1, L1↔register). Timeboxed per the plan; the thread
itself said "no unexplained gap today" is the honest reason it was not
first. What closed it cheaply: 0130's stored trace already contains the
core-side **port events**, so the memtile↔L1 question is answerable from
data on disk plus one accounting argument.

## 1. Measured: the traced core's input ports are mostly idle

Exact B/E interval-union coverage over 0130's steady-state span
(`tasks/0130-t45-traced-roof/trace.json`, production tile, bfp16, 4 cols;
script + output in [`port_analysis.txt`](port_analysis.txt)):

```
captured span: 1,037,680 cycles
  PORT_RUNNING_0    30.9%   (425 intervals)
  PORT_RUNNING_1     3.7%   (26 intervals)
  INSTR_VECTOR      17.6%
  MEMORY_STALL       2.3%
  LOCK_STALL        52.8%   (1,266 intervals)
```

The core's L2→L1 input streams are covered **~31% + ~4%** of the time while
the core spends **52.8% in LOCK_STALL** — waiting for operands that have
not arrived, with the link to fetch them over sitting idle. That is the
signature of an **upstream** (DRAM→L2) constraint, not a memtile↔L1 one:
the leg T46 asked about has roughly 3× headroom at the shipped operating
point. (Caveat carried: one traced core of 16, one shape, and
`PORT_RUNNING_1`'s 26 intervals suggest the second channel's event capture
is sparse — the conclusion leans on port 0 + the lock anatomy, which
agree.)

## 2. Modelled: the second accounting is one argument, as T46 predicted

`tools/roofline.py::traffic_bytes` gains `leg="dram"|"memtile"`. The legs
differ in exactly one term: **B is broadcast from each column's mem tile to
its 4 rows** (A is a split — aggregate equals DRAM; C is a join — same), so
the memtile leg carries `B × N_AIE_ROWS`:

```
bge-base ffn_up (the 0130-traced dispatch):
  cols=4: DRAM 453.0 MB | memtile aggregate  906.0 MB = 2.00x | per-tile @9,970 us: 22.7 GB/s
  cols=8: DRAM 302.0 MB | memtile aggregate  755.0 MB = 2.50x | per-tile @6,782 us: 13.9 GB/s
```

14–23 GB/s per mem tile against a link that the measurement above shows
two-thirds idle — the model and the trace agree the leg is unbound.
Default stays `leg="dram"`, because that is the leg that binds; the smoke
test confirms the default output is byte-for-byte the old one.

## 3. The L1↔register leg

Already owned by 0049's in-core anatomy (77.8% INSTR_VECTOR at the plain-
bf16 operating point) and by 0130's window rate (118.5 of 256 MACs/cyc
within the compute window under streaming) — a *kernel* roofline, not a DMA
one. Nothing new measured here; the pointer is recorded so the three-leg
question has three answers in one place.

## Consequence worth flagging for T48's eventual build

B-reuse removes B's **DRAM** re-streams but not its **L2→L1** broadcasts —
the cores consume B every iteration regardless of where it is staged. At
the re-priced 1.72× array speedup, the per-mem-tile demand rises toward
~24 GB/s (8 cols). Probably still fine (the ports are 69% idle today), but
the build's verification must re-run this port-coverage analysis, because
B-reuse is exactly the change that promotes the memtile leg from slack to
candidate constraint.

## Residual inherited from T45, honestly restated

The fixed cost's shape-dependence (0128: 180 vs 278 µs on one geometry
class) is **not explained** by this task either. The port analysis narrows
it — with input links idle and locks binding, the residual lives in the
schedule (instruction-stream length, tier ramp-in), not in link bandwidth —
but no measurement here pins it. It stays a recorded residual, not a
thread: no current decision depends on it.

## Commands run

```
python - <<EOF   # port coverage (stored as port_analysis.txt, reads 0130's trace.json)
EOF
python - <<EOF   # memtile-leg accounting (stored as memtile_accounting.txt)
EOF
python tools\roofline.py --designs runtime/artifacts_base_bfp16/gemm_rtp/design.json --out <scratch>  # smoke: default output unchanged
```
