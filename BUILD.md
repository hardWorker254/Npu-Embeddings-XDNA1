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
| **Python 3.13** with numpy, torch (CPU), transformers, sentence-transformers | Build-time only: fetching the checkpoint, packing the model, generating tables, and verification. **No Python is used at runtime.** The one exception is the NPU backend itself, which links `PYTHONPATH=/opt/xilinx/xrt/python` so that `xrt._NPU_RUNTIME` resolves; with that variable unset the driver reports the runtime as `cpu` and every export fails at load rather than at build. |

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

Not sure which of the Python steps comes next? `python tools/pipeline.py` prints
the map — task to command — and `python tools/pipeline.py check` reports what
this machine is still missing. The steps below are the same ones, in order, with
the reasoning.

### 2.1 Python environments

Three are used, deliberately kept apart so that a `pip install` accident cannot
break the toolchain that was hardest to get working:

| environment | role |
|---|---|
| **`.venv`** | **The build environment.** Everything the exporters, the packer and the gates need, including the four AMD/Xilinx wheels. Made by `./bootstrap.sh`. |
| IRON's own env | **Build only.** Do not install anything into it. |
| `.venv-ref` | The reference and verification env: sentence-transformers, mteb, for the `reference/` golden generators. Only those need it. |

**`.venv` — the one command (Linux / macOS):**

```sh
./bootstrap.sh                      # makes .venv, installs the pins, fetches + verifies the toolchain wheels
source .venv/bin/activate
python tools/pipeline.py check      # 24 of 27 tools
```

`bootstrap.sh` is idempotent and offline-friendly: the four wheels are pinned
by **sha256** in `requirements.amd.txt` and cached in `.wheel-cache/`, so a
second run is seconds, and a wheel whose digest does not match is refused
rather than installed. Those tags (`latest-wheels`, `nightly`, …) are rolling
upstream — the digest, not the tag, is what says what this build uses.

`requirements.txt` is the human-readable list of *why* each package is here;
`requirements.lock.txt` is what actually gets installed.

Two things the venv deliberately does **not** own, because they are not Python:

- **XRT.** `xclbinutil` and `pyxrt` come from `/opt/xilinx/xrt` — the exporter
  shells out to `xclbinutil` in its last stage and dies without it:
  `aiecc: tool 'xclbinutil' not found`. `source /opt/xilinx/xrt/setup.sh`
  before exporting. `verify_design_numerics` refuses by name for the same
  reason.
- **`mlir-aie` vs `mlir-aie-no-rtti`.** They install the *same* `aie.pth` and
  the same `mlir_aie/` tree, so both cannot coexist: the payload silently
  belongs to whichever was installed last while *both* dist-infos remain.
  `.venv` has no-rtti only, which is the build that was live. Installing
  `mlir-aie` on top of it changes what compiles and leaves
  `toolchain.json` reporting the wrong version.

**`.venv-ref` (Windows, as before):**

```powershell
# a venv off your Python 3.13, inheriting numpy/torch rather than duplicating them
python -m venv --system-site-packages .venv-ref
& ".\.venv-ref\Scripts\python.exe" -m pip install transformers sentence-transformers
# only needed for the accuracy gate and the endpoint test
& ".\.venv-ref\Scripts\python.exe" -m pip install mteb openai
```

### 2.2 Pack the model

**Weights are not fetched — you place them.** Every `models/<name>/` directory
in this repository carries the config, the tokenizer and `CHECKPOINT.json`, but
no weights: a checkout should never have to trust a binary it did not choose.
Put the ONNX export there yourself, under the basenames the graph itself uses:

```sh
models/all-MiniLM-L6-v2/onnx/model.onnx          # the graph
models/all-MiniLM-L6-v2/onnx/model.onnx_data      # only if the graph names one
```

An export that keeps its weights in a side file names it in `external_data`;
place that file beside the graph under exactly that basename, and never rename
or patch the graph to suit a layout. `CHECKPOINT.json` says which bytes are
acceptable: `file` is a list of exactly the files `sha256` covers, graph first
and then each side file, and both are empty (`file: []`, `sha256: null`) until
a *complete* export is there. A graph without its weights is not a
checkpoint, so a half-placed one is reported as absent rather than half-trusted.

```sh
python tools/verify/verify_onnx_reader.py   # the reader against onnx's own descriptors
```

Then pack. The checkpoint is read from `models/` directly; there is no
separate fetch step.

```sh
python tools/gen/gen_tokenizer_tables.py   # Unicode tables -> runtime/include/
python tools/pack/pack_npue.py --device npu1   # -> models/all-MiniLM-L6-v2.npue
python tools/verify/verify_npue.py            # bit-exact round trip, layout guard, goldens
python tools/export/export_validation.py      # golden check vectors for the runtime
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
(aie2) consumes `(s=8, t=4)` for bf16, npu2 (aie2p) consumes `(8, 8)`. It is
also **not** the same for both operand dtypes on the same board — npu1's int8
MMAC consumes `(8, 8)`, where its bf16 one consumes `(8, 4)` — so the pair is
selected by device *and* by dtype, and an int4 container uses the int8 pair
because it widens to int8 before staging. So `--device` selects the pair, the
exporter picks the matching one from `--arch` and the operand dtype, and the two
are compared by the layout hash that already existed. A container packed for the
other generation is not rejected by that check — it was derived from the same
wrong constant on both sides — it is merely wrong, everywhere, with plausible
numbers. It now hashes differently and *is* rejected. Run the exporter and the
packer for the same generation, and re-pack when you switch `--dev`.

Getting the dtype wrong does the same thing one level in, and is harder to
notice because the container and the design still agree with each other: they
agreed on bf16's `(8, 4)` while the int8 array consumed `(8, 8)`, so
`bge-small` on the int8 datapath came back at `1 - cos` 8.6e-01 with nothing
failing. `verify_i4_scheme.py` section 8 re-measures the sub-tile table against
`aie.iron.kernels.mm` on every cheap-gate run, so it is a measurement and not a
constant.

```sh
# both generations, both implementations, byte for byte
python tools/verify/verify_pack_parity.py --device npu1
python tools/verify/verify_pack_parity.py --device npu2
```

### 2.3 Compile the NPU designs

```powershell
python tools/export/export_gemm_rtp.py --batch 128 --batches 4,16,32,128 `
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
runtime\build\npuembeddings.exe . --artifacts artifacts_b128il --threads 24 --pipeline 2

# the two model packers must agree BYTE FOR BYTE (Python reference vs the
# C++ one a release uses) -- a disagreement would be right-sized weights in
# the wrong order, which no tolerance check catches
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools/verify/verify_pack_parity.py

# the Whisper audio front end against transformers: samples, log-mel, the mel
# filter bank, both convolutions, and the four refusals (stereo, 8-bit, wrong
# rate, not-audio). Needs ffmpeg for the conversion case
python tools/verify/verify_whisper_features.py

# the Whisper tokenizer three ways: C++, the reference, HuggingFace
python tools/verify/verify_whisper_tokenizer.py

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
python tools/verify/verify_whisper_model.py

# the same thing one level up, where a request sees it: the `transcribe` CLI and
# POST /v1/audio/transcriptions against transformers' generate() per window and
# its own merge -- the long-form window schedule, the per-window text, the
# merged text, and ten refusals. Needs a BUILT runtime (runtime\build\npuembeddings
# by default; --exe to point at another)
python tools/verify/verify_whisper_cli.py

# the ANSWER, not another program's opinion: word error rate against a HUMAN
# transcript, with transformers' WER on the same audio printed beside ours so a
# regression is distinguishable from a bad reference. No audio is committed --
# the corpus lives outside the tree:
#   python tools/verify/verify_whisper.py --audio C:\clips\a.wav --ref "the words"
#   python tools/verify/verify_whisper.py --corpus C:\speech-corpus --max-wer 0.20
# --npu-ops conv passes conv through to the runtime's --npu-extra-ops, so the same
# audio says whether the device moved the transcript
#   python tools/verify/verify_whisper.py --corpus C:\speech-corpus --npu-ops conv

# does the design set COMPUTE what it claims to? random matrices through the
# exported instruction streams, compared with numpy. Needs the NPU, and needs
# no model: this is the only check that sees a B operand packed for the wrong
# generation, because it does not involve a container at all. Add
# `--npue models/<m>.npue --tensor qkv=<tensor>` to also stage real weights
python tools/verify/verify_design_numerics.py

# the image classifier's PACKER and forward pass against transformers, on both
# weight schemes. The bf16 half is the pool + attention + pre-LN stack and the
# host head; the int8 half is the same with every I8 panel's four scheme
# invariants and the top-1 agreement with fp32. im2col is composed against HF's
# OWN Conv2d, which is the only place in the tree that knows the pixel half of
# the patch layout and the weight half at the same time. Needs the checkpoint
# and a container, and no NPU -- the host kernels are what this checks
python tools/verify/verify_vit.py --container models/vit-base-patch16-224.npue \
    --model-dir models/vit-base-patch16-224

# the image front end (decode, resize, normalise, im2col) against PIL, which is
# the library transformers' own image processor calls. The decode is EXACT on
# every format both sides read; the resize is PIL-equivalent within one 8-bit
# LSB for all four codes this build implements (LANCZOS, BILINEAR, BICUBIC,
# BOX) and exactly zero on a constant raster; normalise and im2col are
# bit-identical to numpy and to vit_int8.im2col_patches. NEAREST, HAMMING, a
# 16-bit PNG, a fake PNG, a truncated PNG and a text file are all refused BY
# NAME -- this gate is why HAMMING is a refusal and not a kernel. Needs g++ with
# libpng and libjpeg headers; no NPU
python tools/verify/verify_vit_image.py --npue models/vit-base-patch16-224.npue

# the HOST-ONLY half of the image path, and the one that catches a silent bug:
# the container contract read_geometry enforces, the classifier head's stride,
# and the int8 host kernels on a real 197x768 operand. Built twice from one
# source (AVX2 and scalar). The head case is the point: classifier.weight is
# [d_model, num_labels] and the checkpoint holds it the other way round, so a
# transpose there is a finite wrong answer at full confidence -- the probe
# prints both and this gate REQUIRES them to differ, so the check is known to be
# able to fail. Needs g++, numpy and a container; no NPU, no design set
#   pass the --int8 container, or the SmoothQuant divisor fold is not exercised
#   and the gate says so rather than passing quietly
python tools/verify/verify_vit_model.py --npue models/vit-base-patch16-224.npue

# semantics without a reference: same-topic texts must rank closer than
# different-topic ones; stdlib only, so it runs on a cold release too
python tools/verify/verify_semantics.py

# the tail: no single input may come back badly wrong (per-model p99 ratchet)
python tools/verify/verify_tail.py

# the endpoint, driven by the official OpenAI client
runtime\build\npuembeddings.exe . --artifacts artifacts_b128il --pipeline 2 --serve 8420
& ".\.venv-ref\Scripts\python.exe" tools/verify/verify_endpoint.py --port 8420
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
`tools/pack/pack_npue.py`, byte for byte, with no Python on the user's machine.

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
runtime\build\npuembeddings.exe . --artifacts artifacts_b128il --threads 24 --pipeline 2 --bench 5

# where the time goes, per design and per stage
runtime\build\npuembeddings.exe . --artifacts artifacts_b128il --bench 5      # prints the split
runtime\build\npuembeddings.exe . --probe-design artifacts_b128il/gemm_rtp    # dispatch vs switch
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
