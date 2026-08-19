# 0048 — M9: the GEMM is bound by the NUMBER of tile iterations, not by bytes

**Question** ([`OPEN-THREADS`](../../research/OPEN-THREADS.md) T1). Per-dispatch
hardware time is 3,028 µs and neither account explains it: compute says 620 µs
(20%), [`0010`](../0010-m5-b-reuse-and-cost-model/TASK.md)'s traffic model says
3,570 µs (118%). The traffic model underpins the entire business case for
B-reuse, which [`0047`](../0047-m9-cascade-channel-probe/TASK.md) had just
priced as a milestone. **Test the premise before funding the project.**

**Answer: time tracks the count of tile iterations. Bytes are not the
constraint, and B-reuse is therefore dead.**

---

## The discriminating experiment

`--bench` reports **one** `wait` figure averaged over all four shapes, which
cannot separate the two accounts. `--probe-streams` (new) binds each
instruction stream in the unified design and dispatches it 30× back to back,
with no host work in the loop and no result checking — deliberately, because any
host term is exactly what we are trying to see past.

The four production shapes were not chosen for this, but they contain a perfect
control:

> **`ffn_up` and `ffn_down` have IDENTICAL MACs (4.83 GMAC) and differ 1.50× in
> bytes moved.** One is `[8192,384,1536]`, the other `[8192,1536,384]`.

If the design is traffic-bound they must differ by 1.5×. If it is bound by
anything proportional to MACs they must be equal.

```powershell
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 `
    --artifacts artifacts_b128il --probe-streams
```

| stream | M | K | N | GMAC | MB | **µs** | GMAC/ms | GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv | 8192 | 384 | 1152 | 3.62 | 84.9 | 3366 | 1.08 | 25.2 |
| attn_out | 8192 | 384 | 384 | 1.21 | 28.3 | 1479 | 0.82 | 19.1 |
| **ffn_up** | 8192 | 384 | 1536 | **4.83** | **113.2** | **4196** | 1.15 | 27.0 |
| **ffn_down** | 8192 | 1536 | 384 | **4.83** | **75.5** | **4273** | 1.13 | 17.7 |

**4,196 against 4,273 µs — a 1.8% difference, and in the wrong direction.**
50% more bytes cost nothing.

`GMAC/ms` is flat at **1.08–1.15** across the three large shapes. `GB/s` spreads
**17.7 to 27.0**, a factor of 1.5. Flat rate is the bound; the varying one is
not.

### Reproduced on a second artifact set

The `--c-bf16` set from [`0045`](../0045-m9-bf16-gemm-epilogue/TASK.md) moves
22% fewer bytes, so it is an independent instance of the same test:

| stream | MB | µs | GMAC/ms |
|---|---:|---:|---:|
| qkv | 66.1 | 2984 | 1.21 |
| attn_out | 22.0 | 1297 | 0.93 |
| **ffn_up** | **88.1** | **3931** | 1.23 |
| **ffn_down** | **69.2** | **3942** | 1.23 |

Same verdict: 27% more bytes on `ffn_up`, **0.3% less time**.

---

## What it *is* bound by

MACs are not the only thing proportional to MACs. The number of **k-block tile
iterations per core** is

```
(M/m) x (N/n) x (K/k) / n_cores
```

which is `M·N·K / (m·n·k·cores)` — exactly proportional to MACs, and **identical
for `ffn_up` and `ffn_down`** (768 each) while their byte counts differ by 1.5×.
So the experiment cannot separate "compute" from "per-iteration overhead", but it
*does* separate both from "bytes".

Fitting `t = a + b · iterations` on the four points:

| set | fixed | per iteration | worst residual |
|---|---:|---:|---:|
| fp32 C | **573 µs** | **4.72 µs** | 2.3% |
| bf16 C | **419 µs** | **4.57 µs** | 2.3% |

Four points and two parameters, so this is a description rather than a
validated model — but it beats 0010's traffic model, which is out by 18% on the
average and by 50% on the discriminating pair.

**And the per-iteration cost is 4.3× the arithmetic in it.** One k-block is
`64·64·48 = 196,608` MACs; at the **145 MACs/cycle/core** that M2 and M5 traced
in isolation that is 1,356 cycles. We spend **4.72 µs ≈ 5,900 cycles**. The
array runs at about **28–33 MACs/cycle/core in production against 145 traced**.

So: **compute-shaped, but not compute.** Something costs ~4,500 cycles per
k-block iteration and scales with the iteration count.

### What this retires, and what it promotes

- **B-reuse is dead** ([T2](../../research/OPEN-THREADS.md)). It removes *bytes*,
  and bytes are free. The 1.26–1.68× that
  [`0010`](../0010-m5-b-reuse-and-cost-model/TASK.md) priced was priced with the
  model this task refutes. That also retires the cascade milestone
  [`0047`](../0047-m9-cascade-channel-probe/TASK.md) scoped — cascade was only
  ever wanted to free channels *for B-reuse*.
- **Bigger tiles are the lever**, because iterations go as `1/(m·k·n)`. And
  that is L1-bound: the current `(64,64,48)` costs **53,248 B of the 63 KB
  budget**, and every legal larger geometry overflows —
  `(64,64,64)` needs 65,536 B, `(96,64,48)` needs 73,728 B. **The tile geometry
  is not a tuning choice, it is the binding constraint**, which reframes
  note 0007 §1.2 (a core can read its neighbours' L1) from a curiosity into the
  one identified way to make tiles bigger.
- **It explains [`0045`](../0045-m9-bf16-gemm-epilogue/TASK.md)** — narrowing C
  bought +4.9% end to end and ~0% of array time, because it removed bytes.

---

## A discrepancy left standing, not smoothed over

The per-stream probe and `--bench` disagree on the *magnitude* of the bf16-C
effect on array time:

| | probe (mean of 4 streams) | `--bench` (`wait`, 24 dispatches) |
|---|---:|---:|
| fp32 C | 3,328 µs | 3,028 µs |
| bf16 C | 3,038 µs | 2,999 µs |
| change | −8.7% | **−1.0%** |

Both are on an idle array with the contention gate green. The probe dispatches
back to back with no buffer syncs; `--bench` syncs around every dispatch and
runs host work between them, so the two are not measuring quite the same thing.

**The discriminating result does not depend on this.** `ffn_up` vs `ffn_down` is
a within-run comparison and it lands the same way in both artifact sets. But the
absolute per-dispatch figure differs by up to 10% between the two harnesses, and
until that is explained neither should be quoted as *the* array time. Added to
the register.

---

## Commands and files

```powershell
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_b128il --probe-streams
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_cbf16  --probe-streams
```

`runtime/src/main.cpp` — `--probe-streams`. It refuses on a non-unified artifact
set and reads `c_elem_bytes` from the design rather than assuming C's width.
