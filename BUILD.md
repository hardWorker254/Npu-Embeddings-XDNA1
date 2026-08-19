# Building NpuEmbeddings from source

This builds everything a [release](../../releases) contains: the NPU designs,
the model container, and the runtime executable. If you only want to *use* the
project, download a release instead — see [README.md](README.md).

Expect the toolchain setup to be the hard part. Once IRON works, the project
itself builds in a couple of commands.

---

## 1. What you need, and why

| | why |
|---|---|
| **A Ryzen AI machine (XDNA2)** | The designs are compiled for `aie2p` and are run on hardware as part of the build — the exporter dispatches each design once to force compilation. Developed on a Ryzen AI 9 HX 370, 8 columns × 4 rows. |
| **AMD Ryzen AI Software** (driver + XRT) | The runtime links `xrt_coreutil.dll`; the build needs XRT's headers and import library. |
| **MLIR-AIE / IRON** | Compiles the AIE kernels and the dataflow graph into `final.xclbin` + `insts.bin`. This is the dependency that makes the build heavy — everything NPU-side goes through it. |
| **Peano (LLVM-AIE)** | The kernel compiler IRON drives. Ships with the IRON install. |
| **MSVC** (VS 2022 or newer) + CMake + Ninja | For the C++ runtime. |
| **Python 3.13** with numpy, torch (CPU), transformers, sentence-transformers | Build-time only: fetching the checkpoint, packing the model, generating tables, and verification. **No Python is used at runtime.** |

### Installing MLIR-AIE

Follow the upstream instructions — they are maintained, and duplicating them
here would only rot:

**→ <https://github.com/Xilinx/mlir-aie>** (see *Getting Started* for Windows)

What this project assumes afterwards:

- IRON is installed and activated by dot-sourcing its environment script, e.g.
  `. C:\dev\mlir-aie\iron_env.ps1` — **dot-sourced**, not executed, or the
  environment variables do not survive.
- The target is `aie2p` with `NPU2=1` (the environment script sets this).
- Peano is the kernel compiler. `xchesscc` is *not* required and is not used
  — it needs Vitis, which has no Windows build.

> **`XILINX_XRT` must stay unset.** IRON's own setup calls it out as poisoning
> Windows builds, and the Ryzen AI installer can leak it into a shell. Use
> `XRT_ROOT` instead. If a build fails in confusing ways, check this first.

> **The upstream example Makefiles do not work natively** on Windows — they
> assume POSIX and have WSL hooks. This project does not use them; its designs
> are driven directly from Python.

---

## 2. Build, in order

Everything below runs from the repository root, in a shell where IRON's
environment has been dot-sourced.

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1
cd <path-to>\NpuEmbeddings
```

### 2.1 Python environments

Two are used, deliberately kept apart so that a `pip install` accident cannot
break the toolchain that was hardest to get working:

| environment | role |
|---|---|
| IRON's own env | **Build only.** Do not install anything into it. |
| `.venv-ref` | The reference and verification env: transformers, sentence-transformers, mteb, openai. |

```powershell
# a venv off your Python 3.13, inheriting numpy/torch rather than duplicating them
python -m venv --system-site-packages .venv-ref
& ".\.venv-ref\Scripts\python.exe" -m pip install transformers sentence-transformers safetensors
# only needed for the accuracy gate and the endpoint test
& ".\.venv-ref\Scripts\python.exe" -m pip install mteb openai
```

### 2.2 Fetch and pack the model

```powershell
python reference\fetch_model.py        # sentence-transformers/all-MiniLM-L6-v2 -> models/
python reference\make_goldens.py       # per-layer reference tensors from HuggingFace
python tools\gen_tokenizer_tables.py   # Unicode tables -> runtime/include/
python tools\pack_npue.py              # -> models/all-MiniLM-L6-v2.npue
python tools\verify_npue.py            # bit-exact round trip, layout guard, goldens
python tools\export_validation.py      # golden check vectors for the runtime
```

`pack_npue.py` produces the `.npue` container: weights converted to bf16 and
**pre-tiled into the exact order the NPU's DMA will read them**, biases and
LayerNorm parameters kept in fp32, the `1/√head_dim` scale folded into Q, and
the tokenizer vocabulary carried along as bytes. `verify_npue.py` checks the
round trip is bit-exact and that the layout hash matches what the designs
expect — a mismatched layout would otherwise produce confidently wrong
embeddings.

### 2.3 Compile the NPU designs

```powershell
python tools\export_gemm_rtp.py --batch 128 --batches 4,16,32,128 `
                                --cols 8 --out runtime\artifacts_b128il
```

This builds each (shape × batch tier) design, verifies that **all of them share
one static configuration** — they must differ only in UUID metadata, ~64–70
bytes — and emits one `final.xclbin` plus sixteen instruction streams. If any
pair diverges the export refuses, because an artifact claiming one hardware
context while needing several would be a lie that only shows up as a
performance mystery later.

Expect this to take a while: it is sixteen full IRON compilations.

<details>
<summary>Older / alternative design sets</summary>

`tools\export_xclbin.py` builds the seven-design set used before the
one-xclbin architecture (separate GEMM, GELU, LayerNorm and softmax designs).
It is still useful for measuring the NPU eltwise kernels in isolation:

```powershell
python tools\export_xclbin.py --cols 8 --elt-cols 8 --batch 128 `
                              --out runtime\artifacts_b128e8
```

The runtime auto-detects which kind of set it was given.
</details>

### 2.4 Build the runtime

```powershell
cd runtime
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd ..
```

If XRT is not at `C:\Xilinx\XRT`, pass `-DXRT_ROOT=<path>`.

Two things in `CMakeLists.txt` are load-bearing and are commented there:
`project()` must precede `find_package(XRT)` (otherwise linking silently
downgrades to static), and `/Zc:__cplusplus` is required or XRT's headers
demand Boost.

### 2.5 Check it

```powershell
# the golden check: does the C++ path reproduce the HuggingFace reference?
runtime\build\npuembed.exe . --artifacts artifacts_b128il --threads 24 --pipeline 2

# tokenizer, against HuggingFace, token for token
& ".\.venv-ref\Scripts\python.exe" tools\verify_tokenizer.py

# the two model packers must agree BYTE FOR BYTE (Python reference vs the
# C++ one a release uses) -- a disagreement would be right-sized weights in
# the wrong order, which no tolerance check catches
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_pack_parity.py

# end to end: text in, vectors out, vs sentence-transformers
& ".\.venv-ref\Scripts\python.exe" tools\verify_embed_e2e.py

# the endpoint, driven by the official OpenAI client
runtime\build\npuembed.exe . --artifacts artifacts_b128il --pipeline 2 --serve 8420
& ".\.venv-ref\Scripts\python.exe" tools\verify_endpoint.py --port 8420
```

Expected: `1-cos` ≈ 1.086e-05 against the reference, and an exact match on
every tokenized sequence.

### 2.6 Package a release

```powershell
.\tools\make_release.ps1 -Version v0.2.0
.\tools\make_release.ps1 -Version v0.2.0 -Artifacts artifacts_b128il,artifacts_base
```

Stages `npuembeddings.exe` and one design set **per width** into `dist\`,
writes a manifest with the sha256 of every file, and zips it (~400 KB). Each
design's hidden size is read out of its own `design.json`, not taken from the
directory name.

**The model is deliberately not in the release.** The user runs
`npuembeddings serve <model>`, and the executable downloads the checkpoint
from HuggingFace, verifies its sha256 against the catalogue compiled into the
binary (`runtime/src/hub.cpp`), cross-checks the checkpoint's own `config.json`
against that catalogue, and builds the container — the same layout as
`tools/pack_npue.py`, byte for byte, with no Python on the user's machine.

> **Why not a script.** Until 0.1.x this was `get-model.cmd`: `curl` to fetch
> a binary, then `certutil -hashfile` compared against a hardcoded digest.
> That is the behavioural signature of a dropper, so SmartScreen and AV
> heuristics flagged it. The checks are unchanged; they just live in the
> executable now ([`0051`](tasks/0051-m9-bge-base-and-in-exe-fetch/TASK.md)).

---

## 3. Measuring

The project's measurement rules are documented in [`docs/`](docs/) and are
worth reading before quoting any number. The short version:

- **Wall clock is never a claim about kernel quality.** The NPU is shared, so
  wall clock measures how busy the machine is as much as how good the code is.
  Kernel figures come from hardware traces or static instruction counts; wall
  clock is valid only for end-to-end throughput and host cost, and is labelled
  as such.
- **Never a single run.** Some paths show double-digit run-to-run spread.
- **Every sweep needs a control with a known correct value**, or a silent
  corruption looks like a result.

```powershell
# end-to-end throughput, five encodes
runtime\build\npuembed.exe . --artifacts artifacts_b128il --threads 24 --pipeline 2 --bench 5

# where the time goes, per design and per stage
runtime\build\npuembed.exe . --artifacts artifacts_b128il --bench 5      # prints the split
runtime\build\npuembed.exe . --probe-design artifacts_b128il/gemm_rtp    # dispatch vs switch

# energy, via the Windows RAPL energy counters
.\tools\energy_compare.ps1 -Low 20 -High 60 -Threads 24 -Repeats 3

# the accuracy gate
& ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py
```

---

## 4. Troubleshooting

| symptom | cause |
|---|---|
| `no design set found for --artifacts ...` | The path is resolved against the source tree (`runtime/<name>`) and a release layout (`<root>/<name>`). Check the directory contains `gemm_rtp/design.json`. |
| Linking produces a binary that fails at load | `project()` after `find_package(XRT)` in CMake — see 2.4. |
| Confusing XRT errors during the IRON build | `XILINX_XRT` is set. Unset it. |
| `'aie.tile' op Basic sequential allocation also failed` | The design exceeds a core's 64 KB local memory (63 KB usable — 1 KB is program stack). |
| `Overflow of program memory` | The core program exceeds 16 KB. Two of this project's kernels fit; three do not. |
| A design times out or produces garbage after a kernel edit | Worker stack too small. It does not fault, it corrupts. This has bitten four times; the sizes in the designs are deliberate. |
| A rebuilt design behaves like the old one | The JIT cache served a stale entry. The exporters purge matching entries before building for exactly this reason; if you drive IRON directly, clear `~/.npu/cache`. |
| `worst 1 - cos` suddenly ~1.0 | A design/stream mismatch — the right buffer sizes with the wrong contents. Re-export and check the identity output. |

More detail, and the reasoning behind each of these, is in
[`docs/`](docs/) and the task logs in [`tasks/`](tasks/README.md).
