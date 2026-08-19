# 0041 — does XRT buffer-object flavour change DMA throughput?

**Question.** FastFlowLM allocates every buffer with `xrt::ext::bo(device, size)`
padded up to a **1 MB multiple** (`src/include/buffer.hpp`); we allocate
`xrt::bo(..., XRT_BO_FLAGS_HOST_ONLY, group_id)` at the **exact size**. The
proposed mechanism was IOMMU page granularity — a 1 MB-multiple allocation might
be backed by larger pages, cutting TLB pressure on the DMA path.

Raised while chasing a different question (why FastFlowLM appears to use NPU
memory and we do not — [`0040`](../0040-m9-honest-cpu-baseline/TASK.md)'s
addendum: it is magnitude, not mechanism). The allocation difference is real, so
it got its own measurement rather than a paragraph of speculation.

## Answer: no. Not by allocation class, not by padding, not at all.

Device time is **identical to three digits** across all four modes, and the
alignment the mechanism depends on **is not controlled by the mode**.

---

## Design

FastFlowLM changes two things at once, so their code cannot say which matters.
Four modes separate them (`--bo-mode`, `runtime/src/npu_device.cpp`):

| mode | allocation | size |
|---|---|---|
| `host_only` | `XRT_BO_FLAGS_HOST_ONLY` | exact — **what we ship** |
| `host_only_1m` | `XRT_BO_FLAGS_HOST_ONLY` | rounded to 1 MB — *rounding alone* |
| `ext` | `xrt::ext::bo` | exact — *class alone* |
| `ext_1m` | `xrt::ext::bo` | rounded to 1 MB — *exactly FastFlowLM* |

**Correctness first.** All four produce `1-cos` **1.086e-05** and PASS — so
`ext` mode works despite taking no `group_id`, which was the one thing that
might have failed on npu2.

Two numbers per mode, because they answer different questions: `seq/s` is
end-to-end wall clock, and `NPU dispatch+wait (serialized)` is the runtime's own
device time — **the only part a DMA change could move**. A mode that improves
seq/s without moving device time improved the host, not the DMA.

---

## Commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
& "C:\Users\vegar\.conda\envs\iron\python.exe" `
    experiments\m9-bo-mode\bench_bo_mode.py --rounds 6

# correctness, per mode
runtime\build\npuembed.exe .. --model all-MiniLM-L6-v2 `
    --artifacts artifacts_b128il --threads 24 --pipeline 2 --bo-mode ext_1m
```

Interleaved and steady-state per the protocol as amended in
[`0040`](../0040-m9-honest-cpu-baseline/TASK.md): the four modes run round-robin
inside each round, and the report is the mean of the second half. Machine:
mains, Balanced, `artifacts_b128il`, batch 128, 2 lanes, 24 host threads.

---

## Results

| mode | seq/s (steady) | range | **NPU dispatch+wait** | vs `host_only` |
|---|---:|---:|---:|---:|
| `host_only` | 897.3 | 865.7–906.3 | **154.5 ms** | 1.000× |
| `host_only_1m` | 893.1 | 844.1–901.0 | **154.8 ms** | 0.995× |
| `ext` | 887.2 | 870.3–902.3 | **154.5 ms** | 0.989× |
| `ext_1m` | 884.3 | 873.8–888.2 | **154.7 ms** | 0.985× |

**Device time does not move.** 154.5 / 154.8 / 154.5 / 154.7 ms — a 0.2% band,
while a single mode's own run-to-run seq/s spread is 844–906 (**7%**). The 1.5%
ordering in the seq/s column is smaller than the noise it sits in and points the
*wrong* way for the hypothesis anyway: our current mode is nominally the
fastest.

---

## The mechanism does not exist to be measured

The first alignment reading looked like it supported the story —
`host_only` 64 KB, `ext_1m` 256 KB — so the padding appeared to be buying
something. **It was one sample.** Sampled four times per mode:

| mode | alignment of the mapped pointer, four runs |
|---|---|
| `host_only` | 256 KB, 64 KB, 64 KB, 128 KB |
| `host_only_1m` | 64 KB, 64 KB, 256 KB, 64 KB |
| `ext` | 128 KB, 64 KB, **1 MB**, **2 MB** |
| `ext_1m` | 512 KB, 64 KB, 128 KB, 64 KB |

**Alignment is not a property of the mode.** `ext` returned 2 MB once and 64 KB
another time; `ext_1m`, the mode whose whole purpose is rounding, returned 64 KB
in half its runs. Padding the *size* does not control the *address* — the
allocator returns whatever it returns, with a floor of 64 KB everywhere, which
is already past any DMA burst granularity that would matter.

So the hypothesis fails twice over: the effect is absent, and the mechanism it
would have worked through is not under our control.

**Caveat, stated because it weakens my own evidence:** `alignment_of()` measures
the **host mapped pointer**, not the device-side IOVA, and the DMA engine sees
the IOVA. The alignment table is a proxy. The load-bearing evidence is the
identical device time, which is measured on the right side of the mapping.

---

## Decision

**Keep `XRT_BO_FLAGS_HOST_ONLY`.** It is what we ship, it measures at least as
fast as every alternative, and it is the allocation that asks the driver for the
least. `--bo-mode` stays in the tree as an instrument — it costs nothing, the
default is unchanged, and the next person to notice the FastFlowLM difference
can re-run this in one command instead of re-deriving it.

## What this does not rule out

- **Larger working sets.** MiniLM stages 21 MB of weights and 153 MB of I/O.
  bge-large would stage ~604 MB, and TLB pressure scales with footprint. Re-run
  `bench_bo_mode.py` in Phase D before assuming the answer carries.
- **`xrt::ext::bo`'s other properties.** This measured throughput only. Access
  modes, sharing between processes and export/import are untouched here.
