# 0108 — T37 ported to bf16/bfp16: the fused epilogue, off the int8-only list

- **Date** 2026-08-25
- **Milestone** M13 (T37, T3, T28)
- **Status** done

## Goal

[`0082`](../0082-m13-fused-ffn-epilogue/TASK.md) fused the host's two multi-pass
GEMM-epilogue chains (the FFN chain and the LayerNorm chain) into one L1-resident pass
per row, gated on `a_elem_bytes == 1` — **int8 only**. Since
[`0104`](../0104-adopt-bfp16-per-model/TASK.md), five of the six shipped models run the
bfp16-emulated datapath, so the measured win (1.39–1.48×, confirmed at three runs/arm in
[`0084`](../0084-m13-host-isa-and-repeats/TASK.md)) is unavailable on the datapath that
actually ships. [`0107`](../0107-t3-t28-pricing/TASK.md) priced the alternative (T28's
hardware relay) at a best case of 1.25×–1.35× and recommended porting this fusion first —
same class of win, far less engineering risk. This task is that port.

Per the brief: measure the addressable share on the bf16 path **before** writing any
fusion code, and only build if it is not small (rule: a measured "this isn't worth it" is
a complete answer, per T2/T9/T12 in `research/CLOSED-THREADS.md`).

## Context

Read in full: [`0082`](../0082-m13-fused-ffn-epilogue/TASK.md) (what was built for int8,
and its hardest-won lesson — bit-identity requires matching intrinsics, not just algebra),
[`0084`](../0084-m13-host-isa-and-repeats/TASK.md) (re-measured 0082 at three runs/arm,
corrected two of its claims), [`0107`](../0107-t3-t28-pricing/TASK.md) (priced T28's relay
on the datapath that ships at 1.25×–1.35× best case and recommended this port instead).

## Step 1 — measure before building

**Instrumentation.** Added two temporary probe timers (`t_probe_ffnup_bias`,
`t_probe_ffndown_conv`) that wrap exactly the `gemm(ffn_up, ...)` and
`gemm(ffn_down, ...)` calls, isolating their contribution to the aggregate `t_bias`
("read out + bias") and `t_conv` ("bf16 convert") buckets `--bench` already reports
summed over all four GEMM shapes. This is a direct measurement, not an allocation
assumption (0107's N-share split was the best available substitute when no per-shape
breakdown existed; this task built one instead). The probes were removed once the
numbers below were recorded — see `git log` / diff for the exact insertion and removal,
not kept in the shipped code.

**Structure confirmed by reading the code first.** The bf16/bfp16 FFN chain is, in fact,
the **same three passes** as int8's unfused chain, not fundamentally shorter as the
brief's framing suggested it might be:

```
gemm(ffn_up)   C-readback + bias, write `up` (fp32)      -- inside "read out + bias"
gelu_cpu(up)   read up, write up                          -- "host gelu", 100% attributable
gemm(ffn_down) read up, write bf16 to device A slot        -- inside "bf16 convert"
```

There is no quantisation step to absorb (the brief's framing was right about that), but
the *pass count* going into the fusion is identical to int8's — three streaming passes
over `up`, the widest tensor in the model — which is why the measured win below turned
out to be the same order of magnitude as int8's, not categorically smaller.

**Measured, idle machine, single-lane `--bench 5` (`--bench 3` for bge-large),
`--threads 24`, adopted datapath per model:**

| model | wall | ffn_up read+bias | ffn_down A-convert | host gelu | **addressable** | % of wall |
|---|---:|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 147.43 ms | 12.93 ms | 12.32 ms | 7.59 ms | 32.84 ms | **22.3%** |
| `bge-small-en-v1.5` (plain bf16) | 370.72 ms | 30.17 ms | 23.26 ms | 14.56 ms | 67.99 ms | **18.3%** |
| `bge-base-en-v1.5` | 669.79 ms | 57.19 ms | 44.05 ms | 41.45 ms | 142.69 ms | **21.3%** |
| `bge-large-en-v1.5` | 2095.72 ms | 154.32 ms | 111.85 ms | 105.06 ms | 371.23 ms | **17.7%** |
| `nomic-embed-text-v1.5` (gated) | 841.34 ms | 116.34 ms | 39.93 ms | 81.58 ms | 237.85 ms | **28.3%** |

`embeddinggemma-300m` (arch=1) uses a separate encoder with its own `geglu()`; its
addressable chain was confirmed qualitatively (the `geglu` bucket, 550 ms unfused in the
bit-identity run below, goes to 0 when fused) rather than probed with the same
instrumentation, since `GemmaNpuEncoder` has no `--bench` mode at all (0105's own
precedent — it is measured by encoding a real corpus).

**Verdict: 17.7%–28.3% of wall clock, comfortably not small — justified building.**
The ordering (nomic highest, bge-large lowest) matches 0082's own traffic-model
prediction exactly: the gated FFN's `ffn_up` output is `2×intermediate`, so the tensor
that stops being materialised is proportionally wider; the biggest model has the most
array time diluting its host share, so the *fraction* addressable shrinks even though
the absolute milliseconds are largest.

## Step 2 — implementation

Followed 0082's structure exactly: extended `gemm()`'s existing `FusedNext`/`fuse`
mechanism rather than adding a parallel one, and reused `--no-fuse-ffn` as the single
flag gating both datapaths' fusion (a container is either int8 or bf16-family, never
both, so the two mechanisms never contend for the same flag inside one process).

**New pieces in `runtime/src/main.cpp`, mirroring the int8 originals one for one:**

| int8 (0082) | bf16 (this task) | what it does |
|---|---|---|
| `struct FusedNext` | `struct FusedNextBf16` | no `asmooth`/`scale` fields — bf16 has no per-row scale |
| `dequant_act_quant()` (free fn) | `Encoder::dequant_act_bf16()` (member) | C-readback+bias, activation, narrow -- but to bf16, not int8; no absmax/quantise tail |
| `add_norm_quant()` | `Encoder::add_norm_bf16()` | add+LN+residual, narrow-to-bf16 tail instead of quantise |
| `gemm(...)`'s `i8 && a_ready` skip | `gemm(...)`'s new `else if (a_ready)` skip (bf16 arm) | the A-conversion bypass, extended to the datapath that never had it |

Wired at all three sites 0082 used: LN site 1 (post `attn_out`, feeding `ffn_up`'s A),
the FFN chain (`ffn_up`→activation→`ffn_down`), and LN site 2 (post `ffn_down`, feeding
the *next layer's* `qkv`, `dst = nullptr` on the last layer since it feeds pooling, not
a GEMM). Gated FFN (arch=2, nomic's SwiGLU) reuses the same `FusedNextBf16` struct with
`gated = true`, exactly as int8 does.

**arch=1 (`embeddinggemma`) is a separate encoder (`GemmaNpuEncoder`) with its own
`gemm()`/`FusedNext`/RMSNorm chain**, per 0082's own precedent of handling it
separately. Ported the FFN/GeGLU-chain fusion only (`FusedNextBf16{uint16_t *dst}`,
`dequant_act_bf16()` using Gemma's `tanh`-approximation GeGLU rather than BERT's
GELU/SwiGLU — copied verbatim from the int8 fused lambda's own GeGLU arm, including its
double-precision scalar tail). **Did not** fuse Gemma's RMSNorm chain (four norms per
layer, no beta, structurally different from post-LN BERT) — this matches 0082 §6's own
scoping decision ("would need its own pass") and is called out rather than half-built.

**Every activation/narrowing arm is copied verbatim from an already-verified source**,
per 0082's hardest lesson (a scalar rewrite of the same algebra is a different number
because `_mm256_fmadd_ps` rounds once where `a*b+c` rounds twice under MSVC
`/fp:precise`): the bf16 `dequant_act_bf16` GELU/SwiGLU arms are byte-for-byte copies of
the int8 fused lambda's arms (which 0082 already proved match `gelu_cpu`/`swiglu_cpu`);
the bf16 C-readback+bias arms are byte-for-byte copies of `gemm()`'s own pre-existing
unfused bf16-C/fp32-C branches; and the narrow-to-bf16 tail calls the exact same
`bf16_fill()` the unfused A-conversion calls (verified to be safe per-row: RNE narrowing
is purely elementwise, so grouping rows differently from one whole-buffer call changes
nothing — each output bit depends only on its own input float).

## Bit-identity

Verified two ways, on an idle NPU, `--threads 24`:

1. **`--embed` on the 520-sentence corpus** (`tasks/0074-m13-gemma-on-npu/corpus_520.txt`),
   fused vs `--no-fuse-ffn`, compared by SHA-256 of the raw `.f32` output (not `1-cos`,
   which agreeing to four digits would not rule out a per-row bug — 0082's own point):

   | model | result |
   |---|---|
   | `all-MiniLM-L6-v2` | **IDENTICAL** |
   | `bge-small-en-v1.5` (plain bf16) | **IDENTICAL** |
   | `bge-base-en-v1.5` | **IDENTICAL** |
   | `bge-large-en-v1.5` | **IDENTICAL** |
   | `nomic-embed-text-v1.5` (gated, `--prefix search_document`) | **IDENTICAL** |
   | `embeddinggemma-300m` (arch=1) | **IDENTICAL** |

2. **The golden gate (`worst 1-cos` vs HuggingFace), fused (default) vs 0105's recorded
   values on the same datapath** — every model reproduced its exact figure, not just
   passed the tolerance:

   | model | this task (fused) | `0105` recorded | match |
   |---|---:|---:|---|
   | `all-MiniLM-L6-v2` | 3.406e-04 | 3.406e-04 | exact |
   | `bge-small-en-v1.5` | 8.348e-06 | 8.348e-06 | exact |
   | `bge-base-en-v1.5` | 2.284e-04 | 2.284e-04 | exact |
   | `bge-large-en-v1.5` | 2.626e-04 | 2.626e-04 | exact |
   | `nomic-embed-text-v1.5` | 1.402e-03 | 1.402e-03 | exact |
   | `embeddinggemma-300m` | 2.315e-04 | 2.315e-04 | exact |

   All PASS at the 2e-03 gate. Logs: `tasks/0108-fuse-epilogue-bfp16/accuracy_fused/`.

## A/B — three runs per arm, one session, idle machine

Single-lane `--bench 5` (`--bench 3` for bge-large; arch=1 has no `--bench`, measured by
encoding the 520-sentence corpus, `--pipeline 4`, matching 0105/0085's precedent).
`xrt-smi examine --report aie-partitions` showed **no active hardware contexts** before
each stage.

| model | unfused mean (ms) | fused mean (ms) | ratio | unfused spread | fused spread |
|---|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 138.52 | 109.52 | **1.265×** | 1.9% | 0.3% |
| `bge-small-en-v1.5` (plain bf16) | 367.16 | 304.00 | **1.208×** | 1.9% | 0.2% |
| `bge-base-en-v1.5` | 618.25 | 494.02 | **1.251×** | 0.7% | 0.2% |
| `bge-large-en-v1.5` | 2075.50 | 1697.35 | **1.223×** | 1.0% | 0.6% |
| `nomic-embed-text-v1.5` (gated) | 815.95 | 608.88 | **1.340×** | 1.0% | 1.0% |
| `embeddinggemma-300m` (arch=1, corpus embed) | 144.93 seq/s | 167.13 seq/s | **1.153×** | 3.7% | 5.2% |

Raw logs: `tasks/0108-fuse-epilogue-bfp16/ab/*.txt` (three files per model per arm).
Within-arm spread is tight (<2% for every BERT-family model on both arms — matching
0084's "under 0.5% on a quiet machine" for the tightest cases and staying well inside
its noise-floor caveat for the rest), so the ratios above are not a single-run artefact.
`embeddinggemma`'s spread is wider on both arms, consistent with 0105's own note that
its differential corpus-encode harness has a higher noise floor (2.9%) than `--bench`.

**Ordering matches the traffic model exactly**: nomic (gated, widest addressable chain)
gains most at 1.340×; bge-large (biggest model, array time dilutes the host's share)
gains least at 1.223×. Same ranking Step 1's addressable-fraction table predicted.

Every number above is wall clock, end to end — not an NPU kernel performance claim
(CLAUDE.md rule 1).

## Result

**Step 1 justified building**: 17.7%–28.3% of wall clock, not small by this project's own
retirement bar (T2/T9/T12). **Bit-identical** on every model, verified by SHA-256 of raw
embedding output (strictest available check) and by exact reproduction of 0105's golden-gate
`1-cos` figures (not merely "within tolerance"). **Measured 1.21×–1.34× on the five
BERT-family models and 1.15× on arch=1**, three runs per arm, idle machine, same session.

**What this means for [`0107`](../0107-t3-t28-pricing/TASK.md)'s park recommendation:**
0107 priced T28's hardware relay, on this exact datapath, at a **best case of
1.25×–1.35×** (MiniLM/bge-base/bge-large), computed against the *unfused* host chain —
zero rebuild overhead assumed, every open item (MMAC composition, hop latency, the
real remaining port-topology scope) only pushing that number down further. This task's
**actually-measured, hardware-verified** fusion gain — 1.265×/1.251×/1.223× on exactly
those three models — lands almost exactly on top of 0107's *ceiling* for the relay, using
a pure C++ rewrite with none of the relay's open questions. Two readings follow:

1. **0107's park recommendation is reinforced, not just reconfirmed.** The relay's
   estimated prize assumed today's unfused chain still runs; it no longer does. Recomputing
   T28's addressable fraction on top of *this* fusion (rather than the unfused baseline
   0107 priced against) would price a materially smaller residual gap than 1.25×–1.35%,
   because the host-side multi-pass overhead the relay would also remove has, in large
   part, already been removed by the cheaper mechanism.
2. **The register's T3/T28 "PRICED" note (0107, appended 2026-08-25) is now itself stale**
   in one respect: it describes the fusion as *"not yet"* ported to bf16 as the reason the
   comparison favoured the cheap lever. That reason has now been resolved by building the
   cheap lever, which strengthens rather than weakens the park case — but the exact
   wording should be updated to say "ported, port 0108" rather than "not yet ported". See
   proposed register text below.

## Commands

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime
cmake --build build --config Release

# NPU health check before any timed run
& "C:\Windows\System32\AMD\xrt-smi.exe" examine --report aie-partitions

# STEP 1 measurement (temporary probe instrumentation, since removed --
# see the diff history for the exact insertion, not present in the shipped code)
.\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts artifacts_minilm_bfp16 --threads 24 --bench 5
.\build\npuembed.exe . --model bge-base-en-v1.5  --artifacts artifacts_base_bfp16  --threads 24 --bench 5
.\build\npuembed.exe . --model bge-large-en-v1.5 --artifacts artifacts_large_bfp16 --threads 24 --bench 3
.\build\npuembed.exe . --model nomic-embed-text-v1.5 --artifacts artifacts_nomic_bfp16 --prefix search_document --threads 24 --bench 5
.\build\npuembed.exe . --model bge-small-en-v1.5 --artifacts artifacts_small_bf16  --threads 24 --bench 5

# BIT-IDENTITY (per model; example shown for MiniLM)
$corpus = "tasks\0074-m13-gemma-on-npu\corpus_520.txt"
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts runtime\artifacts_minilm_bfp16 `
    --embed $corpus a.f32 --threads 24 --no-fuse-ffn
.\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts runtime\artifacts_minilm_bfp16 `
    --embed $corpus b.f32 --threads 24
Get-FileHash a.f32 -Algorithm SHA256; Get-FileHash b.f32 -Algorithm SHA256   # must match
# nomic needs --prefix search_document; embeddinggemma needs no --prefix

# GOLDEN GATE, all six, fused (default)
.\tools\release_benchmark.ps1 -Models "all-MiniLM-L6-v2,bge-small-en-v1.5,bge-base-en-v1.5,bge-large-en-v1.5,nomic-embed-text-v1.5,embeddinggemma-300m" `
    -Skip "throughput,interleaved,energy,mteb" -OutDir "tasks\0108-fuse-epilogue-bfp16\accuracy_fused"

# A/B, 3 runs per arm (example shown for MiniLM; repeated per model, unfused THEN fused)
for ($i=1; $i -le 3; $i++) {
  .\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts runtime\artifacts_minilm_bfp16 `
      --threads 24 --bench 5 --no-fuse-ffn 2>&1 | Out-File "minilm_unfused_run$i.txt"
}
for ($i=1; $i -le 3; $i++) {
  .\runtime\build\npuembed.exe . --model all-MiniLM-L6-v2 --artifacts runtime\artifacts_minilm_bfp16 `
      --threads 24 --bench 5 2>&1 | Out-File "minilm_fused_run$i.txt"
}

# embeddinggemma A/B (no --bench; corpus embed, 3 runs per arm)
for ($i=1; $i -le 3; $i++) {
  .\runtime\build\npuembed.exe . --model embeddinggemma-300m --artifacts runtime\artifacts_gemma_bfp16 `
      --embed $corpus "gemma_unfused_run$i.f32" --threads 24 --pipeline 4 --no-fuse-ffn 2>&1 | Out-File "gemma_unfused_run$i.txt"
}
for ($i=1; $i -le 3; $i++) {
  .\runtime\build\npuembed.exe . --model embeddinggemma-300m --artifacts runtime\artifacts_gemma_bfp16 `
      --embed $corpus "gemma_fused_run$i.f32" --threads 24 --pipeline 4 2>&1 | Out-File "gemma_fused_run$i.txt"
}
```

## Problems hit

1. **`--bench`'s "read out + bias" and "bf16 convert" buckets are aggregated across all
   four GEMM shapes per encode**, not broken out per call site — the same limitation
   0107 flagged for its N-share allocation. Worked around by adding temporary probe
   timers that wrap exactly the two calls of interest (`gemm(ffn_up,...)`,
   `gemm(ffn_down,...)`) and take the delta of the aggregate counter across each call —
   a direct measurement rather than an allocation assumption, and cheaper than it
   sounds because the aggregate counters already exist and only needed a before/after
   snapshot around two call sites. Removed after the numbers were recorded, so the
   shipped code carries no permanent per-shape instrumentation (a genuine one would need
   `--bench` itself extended, which is future work if this granularity is needed again).
2. **`nomic-embed-text-v1.5` refuses an unrecognised `--prefix`.** `"search_document: "`
   (with the trailing colon-space, copied from 0082's own commands) is not one of the
   model's task prefixes; the runtime wants the bare prefix name (`search_document`) and
   applies its own formatting. Fixed by dropping the suffix; worth noting since 0082's own
   commands section has the same (apparently long-stale) string.
3. **PowerShell's `fc` is aliased to `Format-Custom`, not the `fc.exe` file-compare
   utility.** Used `Get-FileHash -Algorithm SHA256` instead, which is a strictly stronger
   check anyway (whole-file, not a diff summary).
4. **`Encoder::exp2_avx2(...)`'s explicit qualification, even from inside code that is
   already an `Encoder` member, is not decoration** — `GemmaNpuEncoder` has no
   `exp2_avx2` of its own and calls `Encoder::exp2_avx2` by name across the two structs.
   Missed on the first read of the int8 fused lambda; caught before writing the bf16
   GeGLU arm by checking whether `GemmaNpuEncoder` defined its own copy (it does not).

## Artifacts

- `runtime/src/main.cpp` — the fusion (see diff; `FusedNextBf16`, `dequant_act_bf16`,
  `add_norm_bf16` in `Encoder`; `FusedNextBf16`, `dequant_act_bf16` in `GemmaNpuEncoder`;
  wiring at both LN sites and the FFN chain in both encoders' `run()`).
- `tasks/0108-fuse-epilogue-bfp16/accuracy_fused/` — golden-gate sweep output, all six
  models, fused (default) path.
- `tasks/0108-fuse-epilogue-bfp16/ab/` — 36 raw log files, three runs × two arms × six
  models, the A/B in the table above.

## Next

- **`--bench` could grow real per-shape instrumentation** if this granularity is needed
  again — the probe-and-delete approach here worked but is not reusable.
- **Re-price T28's relay against the now-fused baseline**, per the "what this means for
  0107" reasoning above — the addressable gap the relay would still close is smaller
  than 0107's 1.25×–1.35× estimate, which assumed the unfused chain.
- Gemma's RMSNorm chain (arch=1) remains unfused, scoped out per 0082's own precedent —
  a future task could evaluate it the same way (measure first) if arch=1's host share
  is later found to dominate.

## Proposed register update

*(`research/OPEN-THREADS.md`, `research/CLOSED-THREADS.md`, `tasks/README.md` and
`docs/CURRENT_STATUS.md` are not edited by this task per the brief; text below is
prepared for whoever integrates it.)*

### T37 (register: search for 0082's entry) — append

> **PORTED TO bf16/bfp16, 2026-08-25 ([`0108`](TASK.md)).**
> The fusion 0082 built for int8 only now also covers the bfp16-emulated + bf16-C
> datapath five of six shipped models actually run (0104), plus bge-small's plain bf16.
> Step-1 measurement (before building) found the addressable chain at 17.7%–28.3% of wall
> clock — the same three-pass structure int8 had, just without a quantisation step to
> absorb. Bit-identical on every model (SHA-256 of raw `--embed` output; exact
> reproduction of 0105's golden-gate `1-cos` figures, not just within-tolerance). Measured
> gain, three runs/arm, idle machine: **1.265×/1.208×/1.251×/1.223×/1.340× on
> MiniLM/bge-small/bge-base/bge-large/nomic, 1.153× on embeddinggemma (arch=1, FFN/GeGLU
> chain only — its RMSNorm chain is out of scope, same as 0082's own precedent)**.

### T3 / T28 (register: 0107's "PRICED" append) — correction

> **0107's park recommendation is reinforced by 0108, and one clause of 0107's own text
> needs updating.** 0107 said *"0082 ... is explicitly not wired for bf16/bfp16 yet"* as
> part of the reasoning for porting it before pricing the relay further. That fusion now
> exists (0108), measured at 1.22×–1.34× on exactly the three models (MiniLM/bge-base/
> bge-large) 0107 priced the relay's *ceiling* against (1.25×–1.35×, computed against the
> UNFUSED chain). The measured fusion gain lands almost exactly at 0107's ceiling for the
> relay — using a pure C++ rewrite with none of the relay's open questions (MMAC
> composition, hop latency, 8-column port topology). **Recommend rewording 0107's
> paragraph 4** ("The single strongest reason...") to reflect that the port is now done,
> not pending, and that re-pricing T28's relay should be against *this fused baseline*,
> not the pre-0108 unfused one — which will price a materially smaller residual gap than
> 1.25×–1.35×, strengthening rather than weakening the park case.

### `tasks/README.md` index row

```
| [0108](0108-fuse-epilogue-bfp16/TASK.md) | **0082's fused epilogue ported from int8-only to the bfp16-emulated + bf16-C datapath five of six shipped models actually run (0104).** STEP 1 measured the addressable chain first (17.7%-28.3% of wall clock across MiniLM/bge-small/bge-base/bge-large/nomic, via temporary probe timers isolating ffn_up's C-readback and ffn_down's A-convert from `--bench`'s aggregated buckets) and found it not small, justifying the build. Extended 0082's own `FusedNext`/`gemm()` mechanism rather than adding a parallel one: new `FusedNextBf16`/`dequant_act_bf16`/`add_norm_bf16` mirroring the int8 originals field-for-field, minus the quantisation scale. Ported to arch=1 (embeddinggemma)'s separate `GemmaNpuEncoder` too, matching 0082's own precedent of fusing only the FFN/GeGLU chain and leaving its four-RMSNorm chain unfused. Bit-identical on all six models by SHA-256 of raw `--embed` output AND exact reproduction of 0105's golden-gate `1-cos` figures (not merely within tolerance). Measured 1.265x/1.208x/1.251x/1.223x/1.340x (MiniLM/bge-small/bge-base/bge-large/nomic) and 1.153x (embeddinggemma), three runs per arm, idle machine, one session. The measured gain on the three models 0107 priced lands almost exactly at 0107's estimated RELAY CEILING (1.25x-1.35x) -- reinforces 0107's park recommendation and flags one clause of 0107's own text ("not wired for bf16/bfp16 yet") as now stale | M13 (T37, T3, T28) | done |
```
