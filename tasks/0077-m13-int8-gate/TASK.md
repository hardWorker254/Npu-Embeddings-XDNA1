# 0077 — T20's gate, opened: int8 builds, is BIT-EXACT, and traces at 7.7–9.6× the bf16 datapath

**Goal (user, 2026-08-22):** *"int8 er interessant, men vi kan også kjøre int8 på npu — det er
tross alt bedre enn fp. så ikke bare se på CPU."*

That correction was right and it changed the session. I had framed the August paper batch's
int8 material as a *threat* (a faster CPU baseline). The larger half is that **int8 runs on
the array, on a datapath that 0049 measured sitting idle.**

Status: **gate passed on hardware, at production geometry.** int8 designs build, run, and
are **bit-exact** on all four MiniLM GEMM shapes with pre-tiled B, tracing at
**5.5-7.7x the bf16 datapath** (mean ~7.0x, production tile (64,64,48)). End to end that is
worth **at most 1.5-2.2x** (§4c) — the value is that the array stops being the bottleneck,
not that everything gets 7x faster.

---

## 1. Why this was worth one afternoon

[T20](../../research/OPEN-THREADS.md#t20) has been open since
[T16](../../research/OPEN-THREADS.md#t16) promoted it with a one-line case: `aie2p` has
native int8 `mac_dims` **(8,8,8)** against bf16's **(4,8,8)**, and after
[`0049`](../0049-m9-t16-iteration-anatomy/TASK.md) *"changing datapath is the only identified
multi-x lever left on array time."* It stayed open because three things were unresolved —
calibration, the kernel, and whether the geometry translated. The August 2026 paper batch
answered the first and third:

* **Accuracy.** 2209.13325 (long indexed) says the ±50–100 outlier dimensions in post-LN BERT
  **rule out per-tensor int8 activations** — which is exactly what SmoothQuant fixes, and
  [2608.18182](https://arxiv.org/abs/2608.18182) (Intel) reports it working on
  BERT/DistilBERT/XLM-RoBERTa with negligible loss, **upstream in PyTorch/TorchAO**.
* **Where the bug will be.** [2608.13756](https://arxiv.org/abs/2608.13756) proves the
  int32 accumulator is *exact and order-independent*, localising any divergence to **scale
  application and output rounding**, with power-of-two scales as the mitigation.

That left one question that no paper could answer: **does our own design build with int8 at
all?** Every design this project has ever compiled is bf16.

---

## 2. What was run

`experiments/m5-pretiled-gemm/gemm_pretiled.py`, shape [512, 384, 384], **4 columns**
(traceable per CLAUDE.md trap 7), `dtype_in_str="i8"`, `dtype_out_str="i32"`, row-major B.

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd $repo
python -u <scratch>\int8_exact.py     # correctness
python -u <scratch>\int8_trace.py     # traced cycles, bf16 vs int8, one session
```

---

## 3. Result 1 — it builds, and it is BIT-EXACT

int8 x int8 -> int32 has **no rounding anywhere in the reduction**, so the honest gate is
*exact equality against an integer reference*, not a tolerance. That is the integer alibi
used as a test: if the accumulator cannot round, any mismatch is real. Overflow is what
would break the argument, so it is asserted (`K · 127² < 2³¹`) rather than assumed.

| tile (m,k,n) | cols | result | `rel_fro` |
|---|---:|---|---:|
| (64, 64, 48) | 4 | **BIT-EXACT** | 0.000e+00 |
| (64, 96, 48) | 4 | **BIT-EXACT** | 0.000e+00 |
| (64, 64, 64) | 4 | **BIT-EXACT** | 0.000e+00 |

Operands were written through `Tensor.__setitem__` and then read back and compared against
the values *intended* (CLAUDE.md traps 6b and 6c), so this is not a device read-back
validating itself.

**Both newly-legal tiles work.** int8 halves the operand bytes, so the L1 budget
`2·(m·k·in + k·n·in + m·n·out) < 64512` opens up:

| tile | bf16 (in=2) | int8 (in=1) |
|---|---:|---:|
| (64, 64, 48) — production | 53,248 ✓ | 38,912 ✓ |
| (64, 96, 48) double-buffered | **67,584 ✗** | **46,080 ✓** |
| (64, 64, 64) | **65,536 ✗** (over by 1,024) | **49,152 ✓** |

`(64,96,48)` is the tile [T19](../../research/OPEN-THREADS.md#t19) closed **negative** on —
and it closed negative for a *capacity* reason: k=96's vector cycles scaled perfectly (1.5x,
still 8 cyc/MMAC) but B had to be **single-buffered**, and the exposed fill grew the
inter-window gap 84 -> 1,193 cycles. At in=1 it fits double-buffered.

---

## 4. Result 2 — the traced number, and it is not what anyone predicted

Same shape, same 4 columns, same session, hardware trace (rule 1):

| datapath | tile | avg cycles/window | **MACs/cycle/core** | % of peak | vs bf16 |
|---|---|---:|---:|---:|---:|
| bf16 -> f32 | (64,64,48) | 7,467 | **26.3** | 10.3% | 1.00x |
| **int8 -> i32** | (64,64,48) | 971 | **202.5** | 39.5% | **7.69x** |
| **int8 -> i32** | (64,96,48) | 1,169 | **252.3** | 49.3% | **9.59x** |

**Every prior estimate was low, by a lot:**

| source | predicted | |
|---|---|---|
| whisper-xdna, measured on their GEMMs | 1.33x | note 0007 §3.5 |
| attainable TOPS ratio, 38 vs 14.71 | 2.58x | 2512.13282, INDEX |
| `mac_dims` geometry, twice the `r` | 2x | note 0007 §3.5 |
| **measured here** | **7.7–9.6x** | this task |

**Why the gap, and why it is consistent with our own record.**
[`0049`](../0049-m9-t16-iteration-anatomy/TASK.md) established that the production bf16 path
runs at ~100% of the **fp32 vector datapath's** 32 MACs/cycle/core **while the MMAC unit sits
idle**, and that bfp16 *emulation* reaches 137–146 by moving work onto that unit. int8 is
native on the same unit. So this is not a surprise so much as 0049's finding cashed in: the
comparison is not "int8 vs bf16 arithmetic", it is **"the MMAC unit vs the fp32 vector unit"**,
and those differ by an order of magnitude.

note 0007 §3.5 explained whisper-xdna's 1.33x as *"the same movement bound we have measured
everywhere else."* **That explanation predates 0048 and 0049**, which showed our plain-bf16
path is *not* movement-bound. Their bound may be real for them; it is not ours, and the note
should not have been read as a prediction for this machine.

`(64,96,48)` adds a further **1.25x** over `(64,64,48)` on top — consistent with T19's own
finding that k=96's vector cycles scale perfectly and only the exposed B fill hurt.

---

## 4b. Result 3 — production geometry, pre-tiled, all four real shapes

The §4 probe used row-major B at [512,384,384]. This is the layout and the shapes we would
actually ship: **pre-tiled B through the same `tile_b()` that packs the `.npue`**, the four
MiniLM GEMMs, traced at 4 columns, one session.

| shape | [M,K,N] | bf16 | **int8 (64,64,48)** | int8 (64,96,48) |
|---|---|---:|---:|---:|
| qkv | [256,384,1152] | 27.8 | **203.3  (7.30x)** | 158.8  (5.70x) |
| proj | [256,384,384] | 26.3 | **203.1  (7.72x)** | 249.8  (9.49x) |
| ffn_up | [256,384,1536] | 27.7 | **153.2  (5.53x)** | 183.3  (6.62x) |
| ffn_down | [256,1536,384] | 25.6 | **191.1  (7.46x)** | 191.5  (7.47x) |

MACs/cycle/core. **Every int8 run is bit-exact** (`rel_fro` 0.00e+00); every bf16 run passes
at ~1.9e-07. The `tile_b` round-trip assertion ran on all of them, so the pre-tiled layout is
exercised rather than a lookalike.

**Two things this fixes about §5's caveats.** The bf16 column is now four independent
measurements at **25.6–27.8**, tightly clustered and agreeing with
[`0003`](../0003-m2-bf16-gemm/TASK.md)'s 25.0 and 0049's 28.9 — so the "the bf16 side is this
run's weakest number" caveat is largely retired. And the pre-tiled path works, which §7 had
listed as not done.

**k=96 is NOT a uniform win.** It beats k=64 on `proj` (+23%) and `ffn_up` (+20%), ties on
`ffn_down`, and is **22% worse** on `qkv`. That is shape-dependence of exactly the kind
[note 0007](../../research/notes/0007-unused-iron-surface.md) §3.5 warns about (tile size is
per-shape and the swing is large) — and it is a real problem for the one-xclbin architecture,
which requires **one** tile geometry for every shape ([T21](../../research/OPEN-THREADS.md#t21)).
On this evidence the production choice would stay **(64,64,48)**, where the four shapes give
5.53-7.72x with no geometry change at all.

**Headline: int8 at the production tile and the production layout is 5.5-7.7x the bf16
datapath, mean ~7.0x, bit-exact on every shape.**

## 4c. What that is worth END TO END, which is much less

Per-core MACs/cycle is not seq/s, and the difference matters more here than usual.
[`0048`](../0048-m9-what-is-the-gemm-time/TASK.md)'s **573 us fixed cost per dispatch does not
shrink** because the arithmetic got faster, and neither does host eltwise, the C readback, or
the bf16 conversions. Amdahl, using this project's own measured splits:

| model | array share (measured) | upper bound at 7x array |
|---|---:|---:|
| MiniLM-L6 | 40.3% (0044, idle array) | **1.53x** |
| EmbeddingGemma | 48.9% (0074) | **1.72x** |
| bge-large | 60.8% (0045) | **2.20x** |

Those are **upper bounds** and they are generous: part of the "array share" is the fixed
dispatch term, which 7x does not touch. And with int32 C at 4 bytes,
[`0045`](../0045-m9-bf16-gemm-epilogue/TASK.md)'s bf16-C transport saving does not apply
either.

**So the honest conclusion is not "int8 is worth 7x". It is that int8 stops the array being
the bottleneck at all** — after which [T34](../../research/OPEN-THREADS.md#t34) §3 (the host
side) and [T28](../../research/OPEN-THREADS.md#t28) (fusion, which removes dispatches) become
the whole game. That is precisely the Amdahl curve
[2608.18182](https://arxiv.org/abs/2608.18182) measured from the other direction:
accelerating GEMM grew LayerNorm's share **7.5x**.

## 4d. Stored artifacts — every number above names its file

CLAUDE.md rule 6. All under `experiments/m5-pretiled-gemm/artifacts/`, committed:

| §4b row | trace |
|---|---|
| qkv int8 k=64 | `trace_pretiled_kn_st_4c_i8_i32_256x384x1152_t64x64x48.txt` |
| qkv int8 k=96 | `trace_pretiled_kn_st_4c_i8_i32_256x384x1152_t64x96x48.txt` |
| attn_out int8 k=64 | `trace_pretiled_kn_st_4c_i8_i32_256x384x384_t64x64x48.txt` |
| attn_out int8 k=96 | `trace_pretiled_kn_st_4c_i8_i32_256x384x384_t64x96x48.txt` |
| ffn_up int8 k=64 | `trace_pretiled_kn_st_4c_i8_i32_256x384x1536_t64x64x48.txt` |
| ffn_up int8 k=96 | `trace_pretiled_kn_st_4c_i8_i32_256x384x1536_t64x96x48.txt` |
| ffn_down int8 k=64 | `trace_pretiled_kn_st_4c_i8_i32_256x1536x384_t64x64x48.txt` |
| ffn_down int8 k=96 | `trace_pretiled_kn_st_4c_i8_i32_256x1536x384_t64x96x48.txt` |
| §4 row-major pair | `trace_rowmajor_4c_i8_i32_512x384x384_t64x{64,96}x48.txt` |

The bf16 comparison rows have `..._bf16_f32_...` files of the same shape, from the same
session. The post-placement MLIR for each is beside them as `mlir_*.mlir`.

The Perfetto `trace_*.json` are **gitignored by design** (`.gitignore:13`) — they are the raw
event stream, and the `.txt` is what the numbers were parsed from.

## 5. Caveats — read these before quoting 7.7x

1. **This is not production geometry.** Row-major B at [512,384,384], not the pre-tiled
   `.npue` path at production shapes. `tools/npue.py`'s `tile_b` is hardcoded to 2-byte
   elements (`view(np.uint16)`), which is the first thing that broke when int8 was tried —
   a packer limitation, not an array one, and unfixed.
2. **The bf16 window distribution is exactly the artifact 0049 warned about.** Its windows
   ran 583–25,308 cycles (n=23) against int8's 583–1,111 (n=28) and 583–1,419 (n=20). 0049
   states plainly that `run_one`'s printed average mixes 583-cycle zero-kernel windows into
   the matmul mean and that **window-level histograms** are the right instrument. So the
   *bf16 side of this ratio is the least trustworthy number in the table.*
   **What rescues it:** bf16 at 26–29 MACs/cycle/core is independently on record from
   [`0003`](../0003-m2-bf16-gemm/TASK.md) (25.0) and 0049 (28.9, reproduced deliberately as a
   pair). The bf16 level is established; only this run's precision is not. The ratio survives
   at roughly 7–8x rather than being an artifact.
3. **One shape, one run, no repeats**, and no end-to-end number at all. Per-core cycles are
   not seq/s: the encode also pays dispatch, host eltwise and transport, and 0048's fixed
   573 µs/dispatch term does not shrink because the arithmetic got faster.
4. **No accuracy claim whatsoever.** Bit-exactness here is about *integer arithmetic*, not
   about whether an int8-**quantized** model produces good embeddings. That is a
   SmoothQuant + MTEB question and none of it has been done.

---

## 6. Problems hit

- **`iron.rand` returns an all-zero operand for `i8`.** It draws floats in [0,1) and casts,
  so every value truncates to 0; the reference norm is then 0 and `rel_fro` comes out `nan`.
  The harness reported a FAIL that was entirely its own. Fixed in `run_one` with an
  integer-aware generator writing through `Tensor.__setitem__`.
- **`tile_b` is 2-byte-hardcoded**, so `pretiled=True` fails with
  `cannot reshape array of size 73728 into shape (384,384)` — 73,728 being exactly half of
  384×384. Not fixed; §7.
- **The tolerance gate was wrong in kind for integers.** `rel_fro <= 5e-3` would pass a
  subtly wrong int8 kernel. Replaced with exact equality plus an asserted overflow bound.

---

## 7. Not done

- **Pre-tiled int8**: `tile_b`/`untile_b` need to be itemsize-general before an int8 `.npue`
  can exist.
- **Production shapes and 8 columns**, and a proper repeated measurement with window-level
  histograms rather than `run_one`'s printed mean.
- **Quantisation**: SmoothQuant through TorchAO, an int8 container, scale handling on the
  core (power-of-two scales per 2608.13756), and MTEB. That is the whole accuracy half and
  it is where the real risk lives.
- **The C readback grows**: int32 C is 4 bytes like fp32, so none of
  [`0045`](../0045-m9-bf16-gemm-epilogue/TASK.md)'s transport saving applies. With the array
  7x faster, the host share gets proportionally worse — the Amdahl point
  [2608.18182](https://arxiv.org/abs/2608.18182) makes about LayerNorm's share growing
  7.5x once GEMM is accelerated.
