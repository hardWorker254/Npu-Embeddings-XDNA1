# 0051 — M9: bge-base-en-v1.5, and model fetching moved into the executable

**Two goals.** (1) Add the model whose geometry actually fits XDNA2. (2) Get
the download out of PowerShell before 0.2.0, because the pinned-checksum batch
script reads as malware to SmartScreen and AV heuristics.

---

## Part 1 — bge-base-en-v1.5: the model this NPU wanted

### Why this one

The requirements are all measured here, not borrowed:

| requirement | source | why |
|---|---|---|
| `N % (tile_n·cols) == 0` for all four GEMMs | [`0007`](../0007-m5-pretiled-gemm-on-npu/TASK.md), [`0042`](../0042-m9-bge-large/TASK.md) | keeps `tile_n = 48` at 8 columns; bge-large's N=1024 forces 32 |
| `head_dim = 64` | [`0043`](../0043-m9-attention-geometry/TASK.md) | 32 does not tile; MiniLM and bge-small both have 32 |
| wide, not deep | [`0027`](../0027-m7-width-hypothesis/TASK.md), [`0042`](../0042-m9-bge-large/TASK.md) | host share falls as 1/h; our cost is per dispatch, the CPU's is not |
| WordPiece + post-LN + absolute positions | [`0036`](../0036-m8-tokenizer/TASK.md) | anything else is runtime work (RoPE, SwiGLU, ALiBi, BPE) |

bge-base is the only widely-used embedder that satisfies all four:
h=768 (so qkv 2304, attn_out 768, ffn_up 3072, ffn_down 768 — every one a
multiple of 384), head_dim 64, 12 layers, BertModel with exact-erf GELU.

Rejected for named reasons: all-mpnet (relative attention bias), nomic
(RoPE + SwiGLU), jina-v2 (ALiBi + GLU), anything RoBERTa-based (BPE, not
WordPiece). e5-base / gte-base / arctic-embed-m share bge-base's geometry
exactly and are interchangeable architecturally — that choice is quality and
licence, not fit.

### Commands, in order

```powershell
& .\.venv-ref\Scripts\python.exe reference\fetch_model.py --model BAAI/bge-base-en-v1.5 --layers 12
& .\.venv-ref\Scripts\python.exe reference\make_goldens.py --model-dir models\bge-base-en-v1.5 --taps
& .\.venv-ref\Scripts\python.exe tools\pack_npue.py --model-dir models\bge-base-en-v1.5 --out models\bge-base-en-v1.5.npue
& .\.venv-ref\Scripts\python.exe tools\verify_npue.py --model bge-base-en-v1.5 --seq 64
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd <repo>
python tools\export_gemm_rtp.py --batches 4,16,32,128 --batch 128 --cols 8 --hidden 768 --out runtime\artifacts_base
& .\.venv-ref\Scripts\python.exe tools\export_validation.py --model bge-base-en-v1.5
.\runtime\build\npuembed.exe .. --model bge-base-en-v1.5 --artifacts artifacts_base --threads 16
& .\.venv-ref\Scripts\python.exe tools\verify_embed_e2e.py --model bge-base-en-v1.5 --artifacts artifacts_base --threads 24
```

### Results — first attempt, nothing to fix

- checkpoint `c7c1988aae201f80…`, 438.0 MB, 12 layers, hidden 768, 12×64,
  ffn 3072, vocab 30522. `1/sqrt(64) = 0.125` folded into Q.
- `.npue` **265.29 MB**, 150 tensors, layout_hash `94266693ea31aa67…` — the
  **same hash as MiniLM and bge-small**, because tile_n is 48 for all three.
  That is the architectural claim, visible as a constant.
- `verify_npue`: spec conformant, **round-trip bit-exact** (0 of 84,934,656
  bf16 elements differ), layout guard refuses tile_n 64, goldens reproduced.
  `1-cos` fp32 activations 4.656e-06, bf16 **1.873e-05**. Scale fold free
  (1.000×).
- design: **ONE xclbin, 16 streams** (4 shapes × 4 tiers) at 8 columns,
  identity checks **64–69 differing bytes** — the UUID footprint, unchanged
  from the h=384 export.
- hardware validation: **`1-cos` 1.353e-05**, rel_fro 4.297e-03, PASS.
- end to end (text in, vector out): **worst `1-cos` 2.613e-05**, pairwise
  similarity max 1.180e-03 against a 1.446e-02 implied bound, **top-10
  neighbour overlap 1.0000**, PASS.

### Throughput, and a prediction that was 27% high

```
--threads 24 --pipeline 2 --bench 5:  181.2 seq/s, 5.50 cores, NPU 74.1% of wall
```

**I predicted ~230 seq/s from [`0048`](../0048-m9-what-is-the-gemm-time/TASK.md)'s
iteration fit and measured 181.2. Recording that as a miss.** The fit was made
on h=384 dispatches and extrapolated across a width doubling it was never
tested at, and it says nothing about the host side, which also changes shape.

The useful signal is the one the fit did not supply: **the NPU is 74.1% of
wall clock here against MiniLM's ~40%**, so bge-base is far more array-bound
than anything we have run. That is what "fits the architecture" should mean,
and it re-weights the ledger: host levers ([T3](../../research/OPEN-THREADS.md))
buy less on this model, datapath levers (T23/T20) buy more.

Scaling sanity: bge-base has 4× bge-small's MACs per layer at the same depth,
so naive scaling predicts 445/4 ≈ 111 seq/s. **We measure 181.2, i.e. 1.63×
better than the width penalty** — [`0027`](../0027-m7-width-hypothesis/TASK.md)'s
width thesis, on a fourth real model.

**No CPU ratio is claimed here.** [`0040`](../0040-m9-honest-cpu-baseline/TASK.md)'s
rule stands: the defensible quantity is an interleaved ratio measured in one
session, and the CPU side was not re-measured today.

---

## Part 2 — the download moved into the executable

### The problem, stated plainly

`tools/make_release.ps1` generated `get-model.cmd`: a batch file that ran
`curl` to fetch a binary and then compared a `certutil -hashfile` digest
against a hardcoded constant. **That is the exact behavioural signature of a
dropper**, so SmartScreen and AV heuristics treat it as one. The logic was
always right; shipping it as a script was indefensible, and a security warning
was the first thing a new user saw.

### What replaced it

New CLI, translated into the existing flag form so the ~2,700 lines below it
are untouched — a second dispatch path would be a second place for the batch
tiers, the contention gate and the fixture check to drift out of agreement:

```
npuembeddings list
npuembeddings serve <model> [--port N] [--bind ADDR]
npuembeddings embed <model> <in.txt> [out.f32]
npuembeddings help
```

New files: `runtime/include/hub.hpp`, `runtime/src/hub.cpp` — a catalogue of
four models (repo, **pinned sha256**, geometry, `tile_n`, download size, one
note each) and a **WinHTTP** fetcher. WinHTTP rather than curl or libcurl
because it ships with Windows: the release stays one executable with no DLL to
carry, which is the whole point of having removed the script.

`sha256_file()` was **exposed** from `npue_pack.cpp` rather than
reimplemented — the Python side already has four copies of that function and a
fifth in C++ is how they would drift.

Also added: `--root` (both layouts auto-detected), `pick_artifacts()` (the
design set is chosen by **reading which K a design serves**, never by
directory name), and the build now emits `npuembeddings.exe` beside
`npuembed.exe` so every existing script and task log keeps working.

### Fail-closed, tested rather than asserted

| test | result |
|---|---|
| cold fetch in an extracted release | 86.7 MB over HTTPS, hashed, packed, ran |
| **container byte-identical to the repo's** | ✅ MiniLM and bge-small both |
| wrong weights under the right name | **refuses**, exit 2, prints both digests |
| right weights + wrong `config.json` | **refuses**: "catalogue says 12, checkpoint says 6" |
| interrupted download | `.part` never promoted; next run fetched only what was missing |
| `serve` over the real endpoint | `/health` ok, `/v1/embeddings` 768-dim, L2 norm 1.000000 |

The config check is new and deliberate: the catalogue's geometry is a *claim*,
and a claim that is never checked is the fail-open shape this repo has closed
nine times. It runs against the downloaded `config.json` before the packer
does, and pooling is still **read** from `1_Pooling/config.json` with the
catalogue only having to agree.

### The root-resolution bug, found by the user, and the one I made fixing it

**Reported:** running the staged release
(`dist\npuembeddings-0.2.0\npuembeddings.exe list`) printed
`Models (root C:\…\NpuEmbeddings)` — the repository — and listed the repo's
four installed containers as `ready`. The release was serving somebody else's
models while claiming to be self-contained. Everything worked and everything
was wrong.

**Cause:** `default_root()` searched *upwards* before looking at the
executable's own directory. From `dist\npuembeddings-0.2.0` it climbed past
`dist\` to the repo root, which has both `models/` and `runtime/`, and matched
there. A release unzipped anywhere outside the tree was unaffected, which is
why the earlier test passed — **the test proved the layout worked, not that
the search was right.**

**Fix:** the executable's own directory wins whenever it holds a design or a
`models/`. A self-contained directory is self-contained; the upward search
only starts when there is nothing there at all (i.e. `runtime\build\`).

**And a second bug, mine, while fixing the first.** I folded the design check
into the same upward loop as the source-tree check, one level at a time. From
`runtime\build\` the parent is `runtime\`, which carries `artifacts_*/gemm_rtp`
but no `models/` — so the design test matched first and the root became
`runtime\` instead of the repository. This is precisely the failure the
function's own comment already warned about, reintroduced three lines below
the warning. The two searches now each run to completion before the other
starts.

Verified on all four layouts rather than the one that was reported:

| layout | root |
|---|---|
| source tree, `runtime\build\` | repository root ✅ |
| release staged **inside** the repo | the release directory ✅ |
| release unzipped outside the repo | the unzip directory ✅ |
| `--root` given explicitly | honoured ✅ |

**The mitigation that made this findable is worth keeping**: `list` prints the
root it resolved. Had it merely printed the table, the release would have
looked correct.

### A bug this found in my own first version

`default_root()` probed for `models/` **and** a design. That is right for the
old single-width release and wrong for the new one: `models/` is created by
the first fetch, and an empty directory does not survive a zip — so a fresh
release silently fell back to root `..` and would have written the model
outside the release directory. It now recognises the source tree by `models/`
+ `runtime/`, and a release by carrying a design at the top level or one
directory down. Caught by unzipping the release and running it as a new user
would, which is the only way it was ever going to show up.

### A bare invocation now answers the question it is asking

`npuembeddings.exe` with no arguments used to take `root = ".."` and start the
golden-vector validation encode -- a developer default from when the only
caller was a task log. Double-clicking the executable, which is exactly what a
release invites, would then either dispatch to the NPU or fail with a path
error about a directory the user never named. It now prints the usage summary
and the model table. The flag form is untouched: `npuembed.exe ..` alone still
runs the validation encode, because that is `argc == 2`.

### Release layout (0.2.0)

`get-model.cmd` is gone. The bundle carries `npuembeddings.exe`, one design
set **per width** (`artifacts_b128il` h=384, `artifacts_base` h=768) and three
thin launchers that each call one subcommand. 0.40 MB zipped.
`make_release.ps1` no longer hardcodes MiniLM's name, sequence length or
embedding dim, and it **reads each design's hidden size out of its
`design.json`** instead of trusting the directory name.

---

## Shipped

`tools\make_release.ps1 -Version 0.2.0` -> `dist
puembeddings-0.2.0-win-x64.zip`,
**0.40 MB**. Verified by unzipping outside the repository and running it as a
new user would:

- bare invocation prints help + the catalogue
- `embed bge-base-en-v1.5` fetched 438 MB, verified the pin, packed, and ran
- the container it produced is **byte-identical** to the one this task
  validated at `1-cos` 1.353e-05

The public repository (`repo/`) was regenerated with
`tools\sync_public_repo.py`, which **refused the first attempt**: a
`[Rösti](../../research/papers/)` link in
[`0040`](../0040-m9-honest-cpu-baseline/TASK.md) pointed at the excluded
directory and would have shipped a 404. Fixed to point at the summary, which
the sync then rewrites into the arXiv citation it should always have been. A
second dead link surfaced with it -- note 0006 referenced
`tasks/0031-m7-eltwise-il4`, and the directory is `-ilp`.

## Files

New: `runtime/include/hub.hpp`, `runtime/src/hub.cpp`, this task.
Changed: `runtime/src/main.cpp` (subcommands, `default_root`,
`pick_artifacts`, `print_catalog`), `runtime/src/npue_pack.cpp` and
`runtime/include/npue_pack.hpp` (`sha256_file`), `runtime/CMakeLists.txt`
(winhttp, hub.cpp, the `npuembeddings.exe` copy), `tools/make_release.ps1`.
Artifacts: `models/bge-base-en-v1.5.npue`, `runtime/artifacts_base/`,
`reference/goldens/bge-base-en-v1.5_l12_s64_*.safetensors`,
`runtime/artifacts/validation/bge-base-en-v1.5/`.
