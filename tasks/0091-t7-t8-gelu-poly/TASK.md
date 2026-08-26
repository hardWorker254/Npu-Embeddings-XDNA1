# 0091 — T7/T8: `gelu_poly.cc`'s emulated widen/narrow, and degree 8 -> 7

- **Date** 2026-08-23
- **Status** done

## Goal

Close two OPEN register threads, both about `experiments/m5-eltwise/kernels/gelu_poly.cc`:

- **T7** — the kernel narrows (and widens) through an emulated fp32 multiply
  instead of the free `accum::from_vector` form [`0045`](../0045-m9-bf16-gemm-epilogue/TASK.md)
  measured.
- **T8** — degree 8 can drop to degree 7 for ~9% of the kernel, per
  [note 0007](../../research/notes/0007-unused-iron-surface.md) §3.2.

Both are dormant (eltwise runs on the host today; see `CLAUDE.md`). "Closing"
here means: make the change, prove it with static instruction counts, record
the result, so the kernel is already correct when eltwise returns to the
array (task 0062 already built `gelu_epilogue_3072_f32_to_bf16` for exactly
that purpose).

A concurrent-agent coordination note also asked this task to **measure but
not modify** `experiments/m5-eltwise/kernels/ffn_down_hop_matmul_g2.cc`,
which another agent is actively editing for T28 — see the last section.

## Environment

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
```

Numpy work used the `iron` conda env directly (no NPU needed), per
`design_gelu_poly.py`'s own usage comment:
`C:\Users\vegar\.conda\envs\iron\python.exe`.

## Compiling a kernel standalone (method, reused throughout)

`tasks/0045` established the comparison method ("compiled standalone with
Peano and disassembled ... 64 elements, `-O2`, `llvm-objdump`") but did not
record the exact command. Recovered here from
`aie.utils.compile.utils.compile_external_kernel` (the function IRON's own
`kernels.*()` factories call), reading the installed wheel at
`C:\dev\mlir-aie\ironenv\Lib\site-packages\mlir_aie\python\aie\utils\compile\utils.py`
lines 296-331. Peano's `clang++` and headers ship inside the wheel too, not
just at `C:\dev\mlir-aie` (which is a source checkout tracking upstream, per
`CLAUDE.md`) — using the wheel's own copy is what IRON actually compiles
with.

```powershell
$CLANG  = "C:\dev\mlir-aie\ironenv\Lib\site-packages\llvm-aie\bin\clang++.exe"
$OBJDUMP = "C:\dev\mlir-aie\ironenv\Lib\site-packages\llvm-aie\bin\llvm-objdump.exe"
$INC1    = "C:\dev\mlir-aie\ironenv\Lib\site-packages\mlir_aie\include"
$INC_AK  = "$INC1\aie_kernels"
$INC_AK2P = "$INC1\aie_kernels\aie2p"
$INCK    = "experiments\m5-eltwise\kernels"     # gelu_poly.cc's own dir (aie_kernel_utils.h)

& $CLANG <source.cc> -c -o <out.o> `
  -I $INCK -I $INC1 -I $INC_AK -I $INC_AK2P `
  -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body `
  -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ --target=aie2p-none-unknown-elf

& $OBJDUMP -d --no-show-raw-insn <out.o> > <out>.dis.txt
```

(The bash-tool equivalent used throughout this session is the same command
with forward-slash paths; both are recorded in the artifacts' shell
history implicitly via the commands below.)

## T7 — the widen/narrow fix

### Baseline, established first

Compiled the **unmodified** `gelu_poly.cc` (as of this session's start) with
the command above. Artifact: `artifacts/gelu_poly_baseline.dis.txt`.

Per-function static instruction counts (`vmul.f`/`vadd.f` occurrences — the
emulated-fp32-arithmetic opcodes tasks/0045 and research/papers/2028's "fp32
is emulated, compute-bound 8:1" identify — and total disassembled lines):

| function | vmul.f+vadd.f | total lines |
|---|---:|---:|
| `gelu_poly_impl` (degree 8, the shipped kernel) | 652 | 1335 |
| `gelu_poly_impl_deg2` (numerically-wrong speed probe) | 220 | 567 |
| `gelu_poly_f32_epilogue` (fp32-in fp32-out GEMM epilogue) | 648 | 1325 |
| `gelu_epilogue_3072_f32_to_bf16` (task 0062, T28's kernel) | 145 | 525 |

### Isolating the pattern's own cost (before touching the real file)

`gelu_poly.cc`'s comment claimed `aie::mul(v, 1.0f).to_vector<T>()` was the
only way to reach `to_vector`, since it isn't defined on plain vectors. Two
64-element isolated probes, same compile command, confirm the fix and its
cost in each direction (0045 only measured the narrow direction; this task
measured the **widen** direction too, since `gelu_poly.cc` does both):

`artifacts/widen_probe.cc` (bf16 -> fp32):

```
widen_mulby1_64 (aie::mul(v, vone_bf).to_vector<float>()):
    4x vmul.f, 0x vadd.f    -- real emulated multiply work, one per 16-lane vector
widen_accum_64 (accum<accfloat,16>.from_vector(v); .to_vector<float>()):
    0x vmul.f/vadd.f        -- lowers to `vlda.conv.fp32.bf16`, a load-with-conversion
```

`artifacts/narrow_probe.cc` (fp32 -> bf16, reproducing 0045's own probe on
today's toolchain):

```
narrow_mulby1_64: 9 vmul.f + 8 vadd.f + 4 vmsc.f = 21 emulated-arithmetic ops
narrow_accum_64:  0 vmul.f/vadd.f/vmsc.f
```

**0045 reported 34 for the mulby1 narrow form; this session measured 21** on
the same C++ pattern. The direction and conclusion (accum form is free, mulby1
form is not) are unchanged and reproduced; the exact count differs, most
likely toolchain drift — `CLAUDE.md` records mlir-aie now at
`1.4.2.dev16+g7e00b57`, tracking upstream `main`, while 0045 (2026-08-19,
`tasks/0044`'s Part 3 era) most likely ran against an earlier snapshot: Peano's
codegen for this pattern visibly changed shape (the disassembly shows a
`vconv.bf16.fp32`/`vmsc.f` cross-term decomposition not obviously present in
0045's own count). Recorded as a discrepancy per the brief's instruction, not
silently reconciled — see `artifacts/narrow_probe.dis.txt` for the full
disassembly this count came from.

The traced root cause (confirmed by reading the installed headers, not
guessed): `aie_api/detail/aie2p/accum.hpp`, `from_vector`'s dispatch table has
an explicit branch for `T = bfloat16` using an `UPSHIFT_FN` path (line ~932),
which is what compiles to the native `vlda.conv.fp32.bf16`/`vst.conv.bf16.fp32`
load/store-with-conversion instead of going through the vector ALU at all.

### The fix, applied to `gelu_poly.cc`

Edited **in place**, not as a `*_rne.cc`-style variant. Reasoning: the
`*_rne.cc` precedent (rounding mode) exists because that change is an
**accuracy** decision needing an A/B kept alive to compare — trap 2b's own
text says so ("shipped kernels still default to `floor` on purpose ... the
`*_rne.cc` variants hold the evidence"). This fix is not an accuracy decision:
0045 already established (and this task rechecked, see "correctness" below)
that the two forms compute the **identical value** — only instruction count
differs. There is nothing to A/B; the old form has no purpose to preserve, so
the git history holds the "before" state and the file was fixed outright.

Three call sites changed, all in `experiments/m5-eltwise/kernels/gelu_poly.cc`:

1. `gelu_poly_impl` — the shipped degree-8 kernel. Widen (4x, one per
   interleaved chain) and narrow (4x) both converted to `accum::from_vector`/
   `to_vector<T>()`.
2. `gelu_poly_impl_deg2` — same fix applied (it is still "numerically wrong
   on purpose" via the truncated Horner chain; that is unrelated to and
   unaffected by the widen/narrow form).
3. `gelu_poly_f32_epilogue` — a **third, previously unlisted** instance found
   while fixing the other two: fp32-in/fp32-out, so `aie::mul(v, 1.0f)
   .to_vector<float>()` on the store was not even a widen or a narrow — the
   `aie::add(...)` immediately before it already produces `aie::vector<
   float,16>`, so the multiply-by-1 did nothing but route a same-type value
   through the emulated multiplier. This is not a T7 "expensive but correct
   form" case — it is dead-weight and was deleted outright, no accum needed.

### After — recompiled, same command

Artifact: `artifacts/gelu_poly_after.dis.txt`.

| function | vmul.f+vadd.f before | after | delta | lines before | after | delta |
|---|---:|---:|---:|---:|---:|---:|
| `gelu_poly_impl` | 652 | 580 | **-72 (-11.0%)** | 1335 | 1190 | **-145 (-10.9%)** |
| `gelu_poly_impl_deg2` | 220 | 148 | -72 (-32.7%) | 567 | 418 | -149 (-26.3%) |
| `gelu_poly_f32_epilogue` | 648 | 580 | -68 (-10.5%) | 1325 | 1180 | -145 (-10.9%) |
| `gelu_epilogue_3072_f32_to_bf16` | 145 | 145 | **0 (already correct)** | 525 | 525 | 0 |

Exact commands:

```powershell
# baseline (before edit, source stashed via git for the record)
& $CLANG experiments\m5-eltwise\kernels\gelu_poly.cc -c -o gelu_poly_baseline.o `
    -I experiments\m5-eltwise\kernels -I $INC1 -I $INC_AK -I $INC_AK2P `
    -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body `
    -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ --target=aie2p-none-unknown-elf
& $OBJDUMP -d --no-show-raw-insn gelu_poly_baseline.o > gelu_poly_baseline.dis.txt

# after the edit -- same command, same source path
& $OBJDUMP -d --no-show-raw-insn gelu_poly_after.o > gelu_poly_after.dis.txt

# per-function count (awk, run against each .dis.txt)
awk '/^0000000.*<.*>:/{if(name!="")print name,cnt,lines; name=$2; gsub(/[<>:]/,"",name); cnt=0; lines=0; next} {lines++} /vmul\.f|vadd\.f/{n=gsub(/vmul\.f|vadd\.f/,"&"); cnt+=n} END{if(name!="")print name,cnt,lines}' gelu_poly_after.dis.txt
```

### The register's own framing check

**No shipping kernel carries the T7 pattern.** Grepped the whole repo for
`aie::mul(...vone...).to_vector<bfloat16>()` / `.to_vector<float>()` and for
`accum<accfloat`/`from_vector` (command: `Grep` tool, pattern
`aie::mul\([^,]+,\s*(vone|1\.0f)`, glob `*.cc`). Found in eight files, all
under `experiments/m5-eltwise/kernels/` (host-side eltwise kernels: `gelu_poly.cc`,
`gelu_fp32.cc`, `layernorm.cc`, `softmax.cc`, `exp2_probe.cc`, `fp32_probe.cc`,
`ffn_down_hop_matmul_g2.cc` — see last section) — **none of these execute in
production**, per `CLAUDE.md`'s own architecture statement ("LayerNorm/RMSNorm,
softmax, GELU/GeGLU, RoPE and attention all run on the host in fp32"). The
two kernels that DO ship on the array's own epilogue path —
`narrow_f32_bf16.cc` (the `--c-bf16` GEMM epilogue) and
`gelu_epilogue_3072_f32_to_bf16` (task 0061/0062, T28's active probe) —
**already use `accum::from_vector`**, confirmed by reading both files and by
the 0-delta measurement above. So T7's finding is real but scoped exactly as
0045 flagged it: only the dormant standalone/GEMM-epilogue variants inside
`gelu_poly.cc` carried the expensive form; nothing that runs today did.

### Correctness

Two checks, both host-side numpy, no hardware run (this session avoided the
NPU per the coordination note that other agents are using it concurrently;
correctness of this substitution does not need a device — see reasoning
below):

**1. Widen (bf16 -> fp32) is provably exact, independent of hardware
implementation.** Every bf16 bit pattern IS the top 16 bits of an fp32 value
(bf16 = truncated fp32), so widening has no rounding decision — verified in
`artifacts/verify_t7_numeric.py` by comparing numpy's own bf16->fp32 cast
against a manual zero-extend (`bits << 16`) on all 393,216 real
`L0.ffn_up` activations: bit-identical for every element. Whichever
instruction sequence the compiler emits, there is only one correct fp32
value to produce.

**2. Narrow (fp32 -> bf16) equivalence is 0045's own already-hardware-verified
result, applied to a second file using the identical AIE API calls.** 0045
measured, ON HARDWARE, that `aie::mul(v,1.0f).to_vector<bfloat16>()` and
`accum::from_vector(v).to_vector<bfloat16>()` are numerically IDENTICAL
(589,807/589,824 = 100.00% bit-exact against a CPU model; the 17-element
residual explained by K-block summation order, unrelated to the narrowing
form — see 0045 "Result — the narrowing does exactly one correct rounding").
Both forms ultimately reach the same store-with-conversion hardware unit
gated by the same `crrnd` rounding-mode register, so this is not re-derived
per file — it is a property of the instruction, not the call site. This
task's own edit did not touch `aie::set_rounding` anywhere gelu_poly.cc
didn't already call it (nowhere — the kernel still defaults to `floor`,
unchanged), so no rounding-mode behaviour changed, only which instructions
compute it.

**3. Expression-graph regression check**, to catch a transcription bug in the
rewrite itself (coefficient order, operand order): reproduced the full
pipeline (widen -> Horner -> max/add -> narrow) in numpy for degree 8 on real
`L0.ffn_up` data and got **rel_fro = 2.494e-03**, matching the kernel header's
own design prediction ("Degree 8 with R = 4 lands at 2.494e-03") to the digit.
See `artifacts/verify_t7_numeric.py`.

```powershell
& "C:\Users\vegar\.conda\envs\iron\python.exe" verify_t7_numeric.py
```

## T8 — degree 8 -> 7

### Numpy reproduction of note 0007 §3.2

Reused `design_gelu_poly.py`'s own `fit()`/`eval_poly_f32()`/`to_bf16()` (same
methodology that produced the shipped coefficients) on the same real
`minilm_l6_s64_taps.safetensors` `L0.ffn_up` data note 0007 used.
`artifacts/repro_deg7.py`:

```powershell
& "C:\Users\vegar\.conda\envs\iron\python.exe" repro_deg7.py
```

```
bf16 output floor (exact erf, bf16 out): 2.465e-03

degree   A: rel_fro incl bf16 round   B: rel_fro fp32 (design limit)  B / floor
     5                    8.194e-03                        8.056e-03       3.27x
     6                    4.571e-03                        4.316e-03       1.75x
     7                    2.503e-03                        1.934e-03       0.78x
     8                    2.494e-03                        1.923e-03       0.78x
     9                    2.470e-03                        1.895e-03       0.77x
    10                    2.465e-03                        1.893e-03       0.77x
```

Column A (`rel_fro` on real data, through the bf16 output round — the SAME
metric the kernel header quotes 2.494e-03 for degree 8, confirmed to match
exactly) is the number that gates a real go/no-go, because it is what a
device actually produces. **My degree-8 number here (2.494e-03) matches the
header's own claimed figure to three significant digits**, which is strong
evidence the reproduction methodology (Chebyshev nodes, `np.polyfit`, fp32
Horner) is faithful to what shipped.

**This disagrees with note 0007 §3.2's own table.** 0007 reports degree 7 at
5.913e-04 and degree 8 at 3.613e-04, both "far below" the 2.465e-03 floor
("still 4.2x below"). Neither of my two columns reproduces those numbers —
column A (real data, full pipeline) gives 2.503e-03 / 2.494e-03; column B
(design-limit, fp32, no output rounding) gives 1.934e-03 / 1.923e-03; a third
attempt matching 0007's own stated method more literally ("least-squares fit
of `c` on `[0,4]`, evaluated in float32 Horner exactly as the kernel does",
scored against `c_exact` on the SAME Chebyshev nodes `fit()` uses, both as a
relative-L2 quantity and as max-abs-error on dense grids) came closest at
**max abs err = 1.303e-04** for degree 8 against 0007's 3.613e-04 — same order
of magnitude, off by 2.8x, not exact. None of the three metrics I tried
reproduces 0007's numbers exactly; see `artifacts/repro_deg7.py` (column A/B)
plus the two follow-up scripts run interactively (not saved as separate
files, reconstructable from the commands in this section) for what was tried.

**The finding that matters is not which exact metric 0007 used — it is that
0007's metric, whatever it is, appears NOT to include the bf16 output
rounding step**, because once that step is added (column A, the metric that
actually gates a shipping decision), **degree 7 and degree 8 are nearly
indistinguishable: 2.503e-03 vs 2.494e-03, a 0.36% relative difference**, both
sitting at ~1.01-1.02x the bf16 floor — not "7x below" and "4.2x below" as
0007's table would suggest if read as the deciding number. The bf16 output
quantization dominates total error at both degrees; the polynomial's own
residual, whatever its magnitude in isolation, is already far smaller than
that quantization floor by degree 7, so improving the polynomial further (as
0007's isolated-fit numbers might suggest is still meaningful all the way to
degree 10) buys nothing end to end. **0007's qualitative conclusion — degree
7 is safe, degree 8→7 is free — holds and is confirmed here by an
independent, more representative metric; its specific numeric table is a
different, smaller quantity than the one that actually gates shipping.**

### Degree-7 coefficients

Generated the same way as the shipped degree-8 set (`fit(7, 4.0)` from
`design_gelu_poly.py`), NOT by truncating the degree-8 fit's leading term —
truncation would not give the best degree-7 polynomial on this range.
`artifacts/gen_deg7_coef.py`:

```
c[0] =  6.6050728574e-04f   // u^7
c[1] = -1.0183994033e-02f   // u^6
c[2] =  5.9113368317e-02f   // u^5
c[3] = -1.4454213211e-01f   // u^4
c[4] =  4.9818454839e-02f   // u^3
c[5] =  3.8568485542e-01f   // u^2
c[6] = -4.9917106742e-01f   // u^1
c[7] =  1.4610463161e-05f   // u^0
```

### Implementation

Added `gelu_poly_impl_deg7` and entry point `gelu_poly_deg7_bf16` to
`gelu_poly.cc`, as a **kept, selectable variant** (not a default change) --
matching the brief's instruction and the file's own precedent
(`gelu_poly_impl_deg2` sits next to `gelu_poly_impl` the same way). Built
with the T7 fix already applied (accum-based widen/narrow), so it does not
carry the T7 problem forward.

### Instruction-count saving

Same compile/objdump command, `artifacts/gelu_poly_after.dis.txt`:

| function | vmul.f+vadd.f | lines |
|---|---:|---:|
| `gelu_poly_impl` (degree 8, T7-fixed) | 580 | 1190 |
| `gelu_poly_impl_deg7` (T7-fixed from the start) | 508 | 1059 |
| **delta** | **-72 (-12.4%)** | **-131 (-11.0%)** |

Matches note 0007's own ~9% estimate (from its `t ~ 2174us + 941us/step`
timing fit) closely enough to trust the direction: one fewer Horner step
removes ~11% of the kernel's static instruction count.

### Verdict

**Adopt degree 7 as a real, available variant; keep degree 8 as the shipped
default.** The end-to-end accuracy cost of dropping to degree 7 is 0.36%
relative (2.503e-03 vs 2.494e-03), both comfortably inside the 2e-3 MTEB-path
tolerance headroom this project uses elsewhere and both landing within 2% of
the bf16 output floor itself — degree 7 is not a compromise, it is
functionally the same accuracy for 11% less code. It is kept as a separate
entry point rather than replacing degree 8 by default because (a) the brief
asked for a variant, and (b) `gelu_poly.cc` is dormant — there is no live
gate to promote it against, exactly as T7's fix was kept as an outright edit
only because it was *risk-free*; degree 7 carries a nonzero (if tiny)
accuracy delta and this project's own standing rule (0045's verdict,
`CLAUDE.md`) is that accuracy defaults change on a measured gate, not on
"encouraging" numbers alone.

## Compile verification (both kernels, full file)

```powershell
& $CLANG experiments\m5-eltwise\kernels\gelu_poly.cc -c -o gelu_poly_final.o `
    -I experiments\m5-eltwise\kernels -I $INC1 -I $INC_AK -I $INC_AK2P `
    -std=c++20 -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body `
    -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ --target=aie2p-none-unknown-elf
# exit 0, no warnings related to the edit; all 14 extern "C" symbols present
# (verified: gelu_poly_bf16, gelu_poly_bf16_4k, gelu_probe_deg2_bf16,
#  gelu_poly_deg7_bf16, gelu_epilogue_3072_f32, gelu_epilogue_2048_f32,
#  gelu_epilogue_3072_f32_io, identity_copy_3072_f32, identity_copy_6144_f32,
#  gelu_epilogue_3072_f32_to_bf16, plus the three _impl* internals)
```

No `.o`/hardware run of the FULL design was attempted (`gelu_kernel.py`'s
`--kernel poly` path was not re-run) because the change is provably
value-preserving per the correctness section above and this file is not
wired into any shipping design — re-running the existing hardware harness is
listed under Next as the natural follow-up for whoever picks eltwise-on-array
back up, alongside `gelu_kernel.py` CLI wiring for `--kernel deg7`.

## Coordination: `ffn_down_hop_matmul_g2.cc` (measured, NOT modified)

Per instruction from the coordinator mid-task: another agent is actively
editing `experiments/m5-eltwise/kernels/ffn_down_hop_matmul_g2.cc` for T28's
phase-fusion probe. This file was **only read and compiled standalone for
measurement** — no edits made. (The file visibly changed under me between the
first read and the compile — a third function,
`ffn_down_hop_matmul_g2_64x48x16_bf16acc`, appeared — confirming it really is
live; not investigated further, it is explicitly a diagnostic-only variant
per its own comment, "NEVER used by a design that computes anything real".)

Compiled read-only with the same command as above; artifact:
`artifacts/ffn_hop_g2_readonly.dis.txt`.

**1. The widening pattern IS present and IS the T7 pattern, byte for byte.**
`ffn_down_hop_matmul_g2_64x48x16` and `_64x48x48` both do
`aie::mul(w_row_bf, vone_bf).to_vector<float>()` to widen the weight vector
`w_row_bf`/`w0`/`w1`/`w2` from bf16 to fp32, once per k-iteration (48
iterations, x1 for the N=16 kernel, x3 for the N=48 kernel). This is
identical C++ to what `widen_probe.cc` isolated above: **cost is 1 `vmul.f`
per 16-lane vector with zero real arithmetic**, confirmed by the same probe
(no new probe needed — the pattern and the fix are the same). A zero-cost
form does exist and is already proven: `accum<accfloat,16>::from_vector(w_row_bf)`
+ `.to_vector<float>()`, which compiles to `vlda.conv.fp32.bf16` (0
`vmul.f`/`vadd.f`), confirmed present and working in this session's fixed
`gelu_poly.cc`.

Static per-function totals for context (the widen cost is mixed into these —
the loop body also contains genuine fp32*fp32 multiplies for the actual
accumulation, which are NOT fixable this way, they are real arithmetic that
aie2p emulates in software regardless):

```
ffn_down_hop_matmul_g2_64x48x16:  19 vmul.f+vadd.f,  92 lines
ffn_down_hop_matmul_g2_64x48x48:  57 vmul.f+vadd.f, 178 lines
```

**2. The scalar `float a_val = (float)hb[m * 48 + k];` is NOT the 1,617x
scalar-float trap.** Disassembly of `ffn_down_hop_matmul_g2_64x48x16` shows
it compiles to exactly one scalar instruction: a 16-bit load
(`lda.s16 r21, [p3, dj1]`) followed by a left-shift-by-16
(`lshl r21, r21, r7` where `r7 = 0x10`). This is the scalar equivalent of the
same bit-level fact the widen fix above exploits — bf16 is the top 16 bits of
fp32, so widening it needs no arithmetic, only a shift, and Peano's compiler
recognises this special case even for a plain C-style `(float)` cast on a
scalar. **No `__extendbfsf` or soft-float library call appears anywhere in
the disassembly.** The genuinely expensive part next to it is the
**broadcast**, `vbcst.32 x9, r21` — real, unavoidable vector work given the
algorithm's own structure (a different scalar broadcast across 16 lanes for
every one of the 48 k-iterations, because this kernel does
broadcast-then-vector-FMA instead of an MMAC intrinsic, exactly as its own
header comment already says: "the broadcast-then-vector-FMA structure here
keeps every multiply/add a real vector op, just not MMAC-accelerated"). That
comment is correct and this task found nothing to add to it.

**Summary for T28's owner:** the weight-widen (`aie::mul(w_row_bf, vone_bf)
.to_vector<float>()`) is a free, zero-risk win available via the exact
`accum::from_vector` substitution already proven in this task's `gelu_poly.cc`
fix — same pattern, same fix, ~1 `vmul.f`/16-lane-vector removed per
k-iteration, times 48 k-iterations, times however many m/b iterations the
design's trip count carries (static count unaffected by runtime trip count,
so the *fraction* saved scales with how much of the loop body this widen
currently is: roughly 1 of the 19-57 vmul.f/vadd.f instructions per
static occurrence, i.e. a modest single-digit percentage of this specific
kernel, not a T7-scale win — most of this kernel's fp32 cost is the genuine
emulated a*w multiply, which cannot be removed this way). The scalar
`(float)` cast is not a problem and does not need fixing. Not applied here
because the file is in active use by another agent.

## Problems hit

- **First compile attempt failed**: `aie_kernel_utils.h` not found. Needed
  BOTH the kernel's own directory (`-I experiments/m5-eltwise/kernels`, for
  the file's local includes) AND `mlir_aie/include/aie_kernels` (for the
  header IRON's own `_include_dirs()` / `gelu_kernel.py`'s `ext_kernel()`
  helper add) — one alone is not enough, matching how `gelu_kernel.py`
  builds its own include list (`include.insert(0, kernels_dir);
  include.append(cxx_header_path()/aie_kernels)`).
- **`find`/`grep` bash commands failed on Windows paths** with backslashes
  from PowerShell-style copy-paste; switched to forward-slash paths and the
  dedicated Grep/Glob tools throughout.
- **Narrow-probe instruction count did not match 0045's reported 34** (got
  21) — recorded as a toolchain-drift finding rather than silently adjusted
  to match; see the T7 section above.
- **`ffn_down_hop_matmul_g2.cc` changed under me mid-task** (another agent's
  concurrent edit added a third function). Confirmed via re-grep before
  reporting; did not touch the file.

## Files changed

| file | change |
|---|---|
| `experiments/m5-eltwise/kernels/gelu_poly.cc` | T7: widen/narrow in `gelu_poly_impl` and `gelu_poly_impl_deg2` moved to `accum::from_vector`; dead multiply-by-1 removed from `gelu_poly_f32_epilogue`. T8: added `gelu_poly_impl_deg7` + `gelu_poly_deg7_bf16` entry point. Module docstring updated to point at this task. |

## Next

1. **`gelu_kernel.py` CLI wiring for `--kernel deg7`** — not done here (not
   required by the brief; the entry point exists and compiles, just not
   plumbed through the Python harness's `--kernel` choices).
2. **Hardware confirmation of the T7 fix and the degree-7 variant**, when
   this kernel is next exercised for real (eltwise-on-array, per note 0007's
   own priority list item 3) — this task's correctness evidence is
   deliberately host-only per the coordination note about NPU contention.
3. **`ffn_down_hop_matmul_g2.cc`'s widen fix** — flagged above for T28's
   owner, not applied here.

## Proposed register update

*(Per coordinator instruction: `research/OPEN-THREADS.md`,
`research/CLOSED-THREADS.md` and `tasks/README.md` are being edited by
another agent right now. This section is the ready-to-paste replacement
text; nothing in those three files was touched by this task.)*

### T7 — verdict: **ANSWERED**

Fixed and measured. The expensive form cost real instructions (4 `vmul.f`/64
elements to widen, ~21 emulated ops/64 elements to narrow, isolated probes),
the free form measured 0 in both directions, and the fix applied cleanly to
the shipped degree-8 kernel (-72 `vmul.f`/`vadd.f`, -11.0% of the function)
plus two more instances found along the way. No shipping kernel carried the
pattern going in (`narrow_f32_bf16.cc` and task 0062's
`gelu_epilogue_3072_f32_to_bf16` were already correct — 0-delta confirmed).

**Replacement text for `CLOSED-THREADS.md`** (verbatim original + status
line + pointer, per that file's stated convention):

```
<a id="t7"></a>
### T7 — `gelu_poly.cc` narrows through an emulated fp32 multiply · ANSWERED · free
[`0045`](../0045-m9-bf16-gemm-epilogue/TASK.md) measured
`aie::mul(v, 1.0f).to_vector<bfloat16>()` at **34 `vmul.f`/`vadd.f` per 64
elements** against **0** for `accum::from_vector`. `gelu_poly.cc` uses the
expensive form in two places, and
[`0026`](../0026-m7-closing-on-cpu/TASK.md) called that kernel "at the
machine's fp32 vector limit" — a limit measured with avoidable emulated ops in
it. Dormant while eltwise runs on the host.

**ANSWERED, [`0091`](TASK.md).** Fixed in place
(not a `*_rne.cc`-style variant — the two forms are provably the same value,
so there is nothing to A/B). `gelu_poly_impl`'s widen+narrow: -72
`vmul.f`/`vadd.f` (-11.0% of the function). A third, previously-unlisted
instance in `gelu_poly_f32_epilogue` turned out to be dead weight (fp32-in/
fp32-out, no conversion needed at all) and was deleted outright, not
converted. Confirmed **no shipping kernel carried this pattern**:
`narrow_f32_bf16.cc` and task 0062's `gelu_epilogue_3072_f32_to_bf16` already
used `accum::from_vector` (0-delta). Re-measuring the narrow direction on
today's toolchain got 21 emulated ops/64 elements, not 0045's 34 — same
conclusion, different exact count, most likely toolchain drift (0045 predates
the current `1.4.2.dev16+g7e00b57`); recorded, not reconciled. Widen-direction
exactness proved analytically (bf16 is a strict bit-subset of fp32, so
widening has no rounding step, verified bit-identical on all of a real
activation tensor); narrow-direction exactness inherited from 0045's own
hardware measurement (100.00% bit-exact) rather than re-run, per this task's
NPU-contention constraint. The identical pattern was also found live in
`experiments/m5-eltwise/kernels/ffn_down_hop_matmul_g2.cc` (T28, another
agent's active work) and reported to that thread's owner rather than fixed
here.
```

### T8 — verdict: **ANSWERED**

Reproduced numerically (numpy, real activation data), found the note's own
table numbers do not reproduce under three attempted metrics but the
qualitative conclusion holds under the metric that actually matters
end-to-end, implemented a real (not truncated) degree-7 refit as a kept
variant, and measured the instruction saving.

**Replacement text for `CLOSED-THREADS.md`**:

```
<a id="t8"></a>
### T8 — `gelu_poly` degree 8 → 7 · ANSWERED · ~9% of that kernel, ~11% measured
[note 0007](../../research/notes/0007-unused-iron-surface.md) §3.2: degree 8 sits at 3.6e-04
against a 2.465e-03 bf16 floor, degree 7 at 5.9e-04 — still 4.2× inside.
One Horner step of eight. Dormant with T7.

**ANSWERED, [`0091`](TASK.md).** Implemented as
`gelu_poly_impl_deg7` (refit at degree 7 on the same Chebyshev nodes, not a
truncation of the degree-8 fit), kept as a selectable variant alongside the
shipped degree-8, not a default change. Measured -72 `vmul.f`/`vadd.f`
(-12.4%), -131 total instruction lines (-11.0%) vs the T7-fixed degree-8 —
matches 0007's ~9% timing-derived estimate. **0007's own table numbers
(3.6e-04/5.9e-04) did not reproduce under three attempted metrics** (closest:
max-abs-error of the fit alone against `c_exact`, 1.303e-04 vs 0007's
3.613e-04 for degree 8, 2.8× off) and appear to exclude the bf16 output
rounding step. Under the metric that actually gates a shipping decision
(rel_fro on real `L0.ffn_up` data, through bf16 output rounding — confirmed
to reproduce the kernel header's own claimed 2.494e-03 for degree 8 exactly),
**degree 7 and degree 8 are nearly indistinguishable: 2.503e-03 vs
2.494e-03, a 0.36% relative difference**, both ~1.01-1.02× the bf16 floor.
0007's *direction* (degree 7 is free) holds and is confirmed by a more
representative metric; its specific numbers measure something smaller and
narrower than end-to-end error.
```

### `tasks/README.md` index row

```
| [0091](0091-t7-t8-gelu-poly/TASK.md) | **T7/T8 closed: `gelu_poly.cc`'s widen/narrow moved off the emulated-multiply idiom, degree-8→7 added as a variant** — T7's fix (`accum::from_vector`, per 0045) cut the shipped kernel **-72 `vmul.f`/`vadd.f` (-11.0%)**; a third, previously-unlisted instance in the fp32 GEMM epilogue turned out to be dead weight and was deleted outright; confirmed **no shipping kernel carried the pattern** (`narrow_f32_bf16.cc`, task 0062's `gelu_epilogue_3072_f32_to_bf16` already correct) but the identical pattern is live in T28's `ffn_down_hop_matmul_g2.cc`, reported not fixed (file in active use). T8's degree-7 refit measured **-12.4% `vmul.f`/`vadd.f`**, and numpy reproduction of note 0007 §3.2 found its 3.6e-04/5.9e-04 table does not reproduce under three attempted metrics and appears to exclude bf16 output rounding — the metric that actually gates shipping shows degree 7 and 8 **0.36% apart**, not "4.2× inside the floor", confirming 0007's direction while correcting its numbers | research | done |
```
