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

### 2.2 Pack the model

The checkpoint is already fetched into `models/`; this fork has no separate
fetch step. `pack_npue.py` reads it directly.

```sh
python tools/gen_tokenizer_tables.py   # Unicode tables -> runtime/include/
python tools/pack_npue.py --device npu1   # -> models/all-MiniLM-L6-v2.npue
python tools/verify_npue.py            # bit-exact round trip, layout guard, goldens
python tools/export_validation.py      # golden check vectors for the runtime
```

`pack_npue.py` produces the `.npue` container: weights converted to bf16 and
**pre-tiled into the exact order the NPU's DMA will read them**, biases and
LayerNorm parameters kept in fp32, the `1/√head_dim` scale folded into Q, and
the tokenizer vocabulary carried along as bytes. `verify_npue.py` checks the
round trip is bit-exact and that the layout hash matches what the designs
expect — a mismatched layout would otherwise produce confidently wrong
embeddings.

**`--device` is not optional in practice.** The B panel's order inside a tile is
the MMAC sub-tile, and the sub-tile is **not** the same on both boards: npu1
(aie2) consumes `(s=8, t=4)`, npu2 (aie2p) consumes `(8, 8)`. So `--device`
selects the pair, the exporter picks the matching one from `--arch`, and the
two are compared by the layout hash that already existed. A container packed
for the other generation is not rejected by that check — it was derived from
the same wrong constant on both sides — it is merely wrong, everywhere, with
plausible numbers. It now hashes differently and *is* rejected. Run the
exporter and the packer for the same generation, and re-pack when you switch
`--dev`.

```sh
# both generations, both implementations, byte for byte
python tools/verify_pack_parity.py --device npu1
python tools/verify_pack_parity.py --device npu2
```

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

# the two model packers must agree BYTE FOR BYTE (Python reference vs the
# C++ one a release uses) -- a disagreement would be right-sized weights in
# the wrong order, which no tolerance check catches
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_pack_parity.py

# the Whisper audio front end against transformers: samples, log-mel, the mel
# filter bank, both convolutions, and the four refusals (stereo, 8-bit, wrong
# rate, not-audio). Needs ffmpeg for the conversion case
python tools\verify_whisper_features.py

# the Whisper tokenizer three ways: C++, the reference, HuggingFace
python tools\verify_whisper_tokenizer.py

# the Whisper NPU stacks against transformers: the encoder's hidden states, one
# teacher-forced decoder step at a time (state AND logits AND top-1), the greedy
# argmax chain against transformers' OWN generate(), the text those ids decode
# to, that 16 workers give the same bytes as 1, and two refusals. Needs the NPU,
# the container, the design set and the checkpoint. This is the gate that says
# the two stacks COMPUTE the model rather than merely dispatching
#   NOTE: for a whisper checkpoint do NOT pass --max-seq. The default is per
#   family (256 for embedders, the checkpoint's own max_source_positions for
#   whisper), and a whisper value shorter than one 30 s window writes a
#   container that refuses every request. The packer refuses it by name.
#
# It also holds the ARRAY front end: the NPU conv output against torch at its
# own max-abs tolerance (that path is bf16, not fp32) and the whole encoder run
# a second time on it, which is the run that says `--npu-extra-ops conv` is a default
# rather than a curiosity.
python tools\verify_whisper_model.py

# the same thing one level up, where a request sees it: the `transcribe` CLI and
# POST /v1/audio/transcriptions against transformers' generate() per window and
# its own merge -- the long-form window schedule, the per-window text, the
# merged text, and ten refusals. Needs a BUILT runtime (runtime\build\npuembed
# by default; --exe to point at another)
python tools\verify_whisper_cli.py

# the ANSWER, not another program's opinion: word error rate against a HUMAN
# transcript, with transformers' WER on the same audio printed beside ours so a
# regression is distinguishable from a bad reference. No audio is committed --
# the corpus lives outside the tree:
#   python tools\verify_whisper.py --audio C:\clips\a.wav --ref "the words"
#   python tools\verify_whisper.py --corpus C:\speech-corpus --max-wer 0.20
# --npu-ops conv passes conv through to the runtime's --npu-extra-ops, so the same
# audio says whether the device moved the transcript
#   python tools\verify_whisper.py --corpus C:\speech-corpus --npu-ops conv

# does the design set COMPUTE what it claims to? random matrices through the
# exported instruction streams, compared with numpy. Needs the NPU, and needs
# no model: this is the only check that sees a B operand packed for the wrong
# generation, because it does not involve a container at all. Add
# `--npue models/<m>.npue --tensor qkv=<tensor>` to also stage real weights
python tools\verify_design_numerics.py

# semantics without a reference: same-topic texts must rank closer than
# different-topic ones; stdlib only, so it runs on a cold release too
python tools\verify_semantics.py

# the tail: no single input may come back badly wrong (per-model p99 ratchet)
python tools\verify_tail.py

# the endpoint, driven by the official OpenAI client
runtime\build\npuembed.exe . --artifacts artifacts_b128il --pipeline 2 --serve 8420
& ".\.venv-ref\Scripts\python.exe" tools\verify_endpoint.py --port 8420
```

Expected: `1-cos` ≈ 1.086e-05 against the reference.

### 2.6 Package a release

Stage the runtime executable and one design set **per width** into `dist\`, and
write a manifest with the sha256 of every file. Each design's hidden size is
read out of its own `design.json`, not taken from the directory name. This fork
ships no packaging script; the staging is done by hand.

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
