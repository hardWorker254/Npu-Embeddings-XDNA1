# NpuEmbeddings
# SPDX-License-Identifier: Apache-2.0

# 3.24 for $<LINK_LIBRARY:WHOLE_ARCHIVE,...> -- see the note above the three
# binaries; without it the five self-registering ModelLoaders are not in the
# link and every flag-form command dies with "no model loader for arch".
cmake_minimum_required(VERSION 3.24)
project(npuembeddings_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED XRT_ROOT)
  if(DEFINED ENV{XILINX_XRT})
    set(XRT_ROOT $ENV{XILINX_XRT})
  elseif(DEFINED ENV{XRT_ROOT})
    set(XRT_ROOT $ENV{XRT_ROOT})
  elseif(WIN32)
    set(XRT_ROOT "C:/Xilinx/XRT")
  else()
    set(XRT_ROOT "/opt/xilinx/xrt")
  endif()
endif()
message(STATUS "XRT_ROOT: ${XRT_ROOT}")

find_path(XRT_INCLUDE_DIR xrt/xrt_device.h
          HINTS "${XRT_ROOT}/include" ENV XRT_INCLUDE_DIR
          REQUIRED)

find_library(XRT_COREUTIL_LIB NAMES xrt_coreutil
             HINTS "${XRT_ROOT}/lib" "${XRT_ROOT}/lib/Release"
             REQUIRED)

message(STATUS "XRT headers: ${XRT_INCLUDE_DIR}")
message(STATUS "XRT lib:     ${XRT_COREUTIL_LIB}")

if(WIN32 AND NOT DEFINED BOOST_ROOT AND EXISTS "C:/dev/boost_1_88_0")
  set(BOOST_ROOT "C:/dev/boost_1_88_0")
endif()

if(DEFINED BOOST_ROOT AND EXISTS "${BOOST_ROOT}")
  message(STATUS "Boost (fallback only): ${BOOST_ROOT}")
endif()

# ONE name, not two. This target was called `npuembed`, with a POST_BUILD copy
# beside it called `npuembeddings` so that the documented command line worked.
# The two files were byte-identical by construction, every document and every
# error message named the copy, and the copy was the one people actually ran --
# so the name that was BUILT and the name that was USED were different, and
# BUILD.md had to document the older one for its own commands to work. One
# target, one name: npuembeddings.
#
# THREE BINARIES, ONE LIBRARY. The sources below build a static library, and
# three three-line mains under src/bin/ turn it into npuembeddings, npuaudio
# and npuimage -- each setting app::g_family and calling app::entry() (see
# cli/family.hpp for what the split means). The library is compiled ONCE
# because the split is a statement about WHO RUNS WHAT and not about the code:
# three copies of the runtime would be three copies of every measurement's
# subject, and a fix applied to one binary and not the other two is exactly how
# two binaries end up disagreeing about a refusal.
set(NPUE_SOURCES
  # The body of main(), shared by all three. Each binary's own main() is in
  # src/bin/ and is compiled into the executable below, not into this library.
  src/entry.cpp
  src/runtime.cpp
  # Subtask 1: core infrastructure (device/design/model/pool).
  src/device.cpp
  src/design.cpp
  src/model.cpp
  src/pool.cpp
  # Subtask 10: shared utilities under common/.
  src/common/npue_pack.cpp
  src/common/onnx_read.cpp
  src/npu_contention.cpp
  src/common/hub.cpp
  # The four resamplers arch=7 and arch=8 share: cv2.INTER_LINEAR, cv2.INTER_AREA,
  # cv2.warpAffine with border 0, and a rectangle copy, plus the normalisation and
  # a float bilinear for the one front end that resizes AFTER normalising. They
  # were arch=7's own definitions until arch=8 needed the same four, and a second
  # copy of a box-average is a copy whose output is the model's accuracy.
  src/common/raster.cpp
  # Subtask 2: tokenizer system (base interface + WordPiece + XLM-R + Gemma).
  # The old src/tokenizer*.cpp were dropped with their headers; WordPiece is
  # now npue::WordPiece in tokenizers/wordpiece.cpp.
  src/tokenizers/tokenizer.cpp
  src/tokenizers/wordpiece.cpp
  # arch=1 (EmbeddingGemma family, tasks/0061/0063/00XX): all three are
  # standalone and XRT-free (see their own file headers) -- added here only
  # so main.cpp's run_gemma_mode() can call them. Nothing in them touches
  # runtime/device.hpp or XRT.
  src/tokenizers/gemma.cpp
  src/encoders/gemma_kernels.cpp
  # C++ port of tools/gen_gemma_tokenizer_table.py (tasks/0067): a
  # dependency-free JSON DOM parser and the table generator built on it.
  # Both are XRT-free like the arch=1 files above; json_min is generic and
  # not itself EmbeddingGemma-specific.
  src/common/json_min.cpp
  src/tokenizers/gemma_tokenizer_gen.cpp
  # arch=3 (XLM-R family, T52, tasks/0127/0133): SentencePiece Unigram
  # tokenizer + the C++ port of tools/gen_xlmr_tokenizer_table.py. Both
  # XRT-free like the gemma pair above; not yet wired into main.cpp (the
  # arch-3 integration is a later task) -- compiled in so the packer-side
  # generator ships with the binary, same rationale as gemma's.
  src/tokenizers/xlmr.cpp
  src/tokenizers/xlmr_tokenizer_gen.cpp
  # Subtask 3: Embedding model system
  src/embed_models/model_loader.cpp
  src/embed_models/bert_model.cpp
  src/embed_models/gemma_model.cpp
  # arch=5 (image classification, arch 5 vit_patch16_prenorm_gelu): the model
  # wrapper, the image front end (PNG/JPEG via libpng/libjpeg, PIL-equivalent
  # resize, normalise, im2col), the classifier stack, and the session that owns
  # the one design set. The GEMM/LayerNorm/GELU/attention it dispatches are
  # npue::whisper::npu_ops, already listed above -- the arithmetic is shared and
  # only the schedule is arch=5's, which is why nothing from whisper/ is
  # duplicated here.
  src/embed_models/vit_model.cpp
  src/vit/image.cpp
  src/vit/head.cpp
  src/vit/encoder.cpp
  src/vit/classify.cpp
  # arch=6 (body pose). src/pose/net.cpp is compiled at -O3 on its own: its
  # host GEMM is a nested axpy whose inner loop is what -O3 vectorises, and the
  # file-wide default here is -O2, which does not. The rest of the pose module
  # is I/O and graph walking and is happy at -O2.
  src/pose/geometry.cpp
  src/pose/image.cpp
  src/pose/net.cpp
  src/pose/decode.cpp
  src/pose/session.cpp
  src/pose/result_json.cpp
  src/embed_models/pose_model.cpp
  # arch=7 (MediaPipe hands: the palm detector and the hand-landmark network in
  # ONE container). Added in the same order the file is read: the geometry, then
  # the image front end, then the graph walk, the heads, the decode and the
  # session. src/hands/net.cpp gets -O3 on its own for the reason
  # src/pose/net.cpp does -- it is the same host GEMM, now in
  # include/common/conv_host.hpp, plus a depthwise convolution.
  src/hands/geometry.cpp
  src/hands/image.cpp
  src/hands/decode.cpp
  src/hands/net.cpp
  src/hands/session.cpp
  # arch=8 (MediaPipe body pose: the person detector and the pose-landmark
  # network in ONE container). Same file order as arch=7's block above, and for
  # the same reason each file says what it is for. src/mppose/net.cpp gets -O3 on
  # its own: it is the same host GEMM and the same depthwise kernel as arch=7's,
  # and its d2s is a bandwidth-bound plane copy that -O3 turns into memcpys.
  src/mppose/geometry.cpp
  src/mppose/image.cpp
  src/mppose/decode.cpp
  src/mppose/net.cpp
  src/mppose/session.cpp
  # The array side of a CONVOLUTION, shared by arch=6 and arch=8. Listed once and
  # used twice rather than kept per architecture, because the dangerous parts are
  # the ones with a plausible wrong answer -- the padded-K repack, the widened
  # bias, and "one panel per convolution, never per stream name" -- and two copies
  # of those are free to drift on exactly the details that fail silently.
  # conv/array.cpp is XRT-dependent (it names npu::Device and NpuGemm) while the
  # two sessions above are not, which is why the split is at this line and not
  # inside the backend.
  src/conv/array.cpp
  # Subtask 7: STT extension (Whisper). The tokenizer stub lives with the
  # other tokenizers but is grouped here with the rest of the Whisper
  # placeholders.
  src/tokenizers/whisper.cpp
  src/whisper/whisper_model.cpp
  src/whisper/stt_engine.cpp
  src/whisper/whisper.cpp
  # P3: the audio front end. XRT-free like the tokenizer above -- the log-mel
  # spectrogram is pure host code, and tools/verify/verify_whisper_features.py holds
  # it against transformers. The two convolutions live here too, as the fp32
  # REFERENCE; src/whisper/conv1d.cpp below is the NPU path over the same
  # weights, and it is this one that dispatches.
  src/whisper/features.cpp
  src/whisper/audio.cpp
  # P4: the Whisper NPU stacks -- the shared GEMM/layernorm/GELU/attention layer
  # and the two stacks on top of it. Not reached from Runtime::run yet (the STT
  # branch lands with P5), but compiled in so arch=4 is built and type-checked
  # by the same build that serves every other architecture, rather than only by
  # the gate's own g++ line.
  src/whisper/npu_ops.cpp
  src/whisper/conv1d.cpp
  # The elementwise designs Whisper can be given instead of the host passes:
  # LayerNorm, softmax and GELU, each one xclbin the runtime opens only when
  # --npu-ops names it.
  src/whisper/eltwise.cpp
  src/whisper/attention_npu.cpp
  src/whisper/mel_proj.cpp
  src/whisper/fft_npu.cpp
  src/whisper/logits_npu.cpp
  src/whisper/encoder.cpp
  src/whisper/decoder.cpp
  # P5: the transcription session (device, both design sets, long-form, the
  # merge) and the OpenAI-shaped /v1/audio/transcriptions endpoint.
  src/whisper/transcribe.cpp
  # Subtask 4: Encoder system
  src/encoders/bert_encoder.cpp
  src/encoders/gemma_npu_encoder.cpp
  src/encoders/gemma_host_encoder.cpp
  src/encoder.cpp
  # Subtask 5: server system (server.hpp is header-only; these are the two
  # translation units that were previously inline in main.cpp/run_execute.hpp).
  src/server/embed_backend.cpp
  src/server/embed_service.cpp
  src/server/stt_backend.cpp
  src/server/pose_backend.cpp
  src/server/vit_backend.cpp
  # Subtask 6: CLI system
  src/cli/cli.cpp
  src/cli/subcommand.cpp
)

add_library(npue_core STATIC ${NPUE_SOURCES})

target_include_directories(npue_core PUBLIC include "${XRT_INCLUDE_DIR}")

if(DEFINED BOOST_ROOT AND EXISTS "${BOOST_ROOT}")
  target_include_directories(npue_core PUBLIC "${BOOST_ROOT}")
endif()

# The arch=5 image front end's decoders, on BOTH platforms and outside the
# platform conditional on purpose: the choice is about the FORMATS, not about
# the OS. libpng and libjpeg are the two formats this runtime reads, and both
# are decoders whose output is well defined (8-bit RGB), so the alternative --
# shipping a hand-written PNG inflate -- would be a second decoder to keep
# correct rather than a dependency avoided. Every variant (palette, 16-bit,
# gray, RGBA, CMYK JPEG, progressive) is normalised to RGB by the libraries
# themselves; vit/image.cpp composites RGBA onto white and refuses everything it
# was not asked for.
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

if(WIN32)
  target_link_libraries(npue_core PUBLIC winhttp ws2_32)
  target_link_libraries(npue_core PUBLIC PNG::PNG JPEG::JPEG)
else()
  target_link_libraries(npue_core PUBLIC ${XRT_COREUTIL_LIB})

  find_package(CURL REQUIRED)
  target_link_libraries(npue_core PUBLIC CURL::libcurl)
  target_link_libraries(npue_core PUBLIC PNG::PNG JPEG::JPEG)
endif()

if(MSVC)
  set(NPUE_HOST_ARCH "AVX2" CACHE STRING
      "MSVC /arch: level for the host kernels (AVX2 or AVX512)")
  target_compile_options(npue_core PRIVATE /W3 /EHsc /Zc:__cplusplus /O2
                         /arch:${NPUE_HOST_ARCH})
  target_compile_definitions(npue_core PRIVATE WIN32_LEAN_AND_MEAN)
else()
  # THE HOST ISA, AS A CACHE VARIABLE -- and the same shape as the MSVC branch
  # above, which already had one and which is why the two disagreed about
  # whether this binary is portable.
  #
  # WHAT WAS WRONG, MEASURED ON THIS MACHINE (AMD Ryzen 7 PRO 8845HS, Zen 4):
  # `-march=native` resolves to `znver4`, which turns on AVX-512, and the binary
  # carried 1948 `zmm` references. So the executable was not merely tuned for
  # this CPU -- it REQUIRED this CPU's ISA, and on any AMD without AVX-512 it
  # dies with SIGILL before main(). That is Zen 1, Zen 2 and Zen 3: EPYC Naples,
  # Rome, Milan and Genoa up to Zen 3, and every desktop Ryzen before the 7000
  # series. "It runs on AMD" was true of one generation of AMD and not of the
  # rest.
  #
  # IT ALSO COULD NOT BE OVERRIDDEN, which is the part that made this a trap
  # rather than an inconvenience:
  #
  #   cmake -DCMAKE_CXX_FLAGS="-march=x86-64-v3" ...
  #   FLAGS = -march=x86-64-v3 -O3 -DNDEBUG -std=gnu++17 -Wall -O2 -march=native
  #
  # `target_compile_options` is APPENDED after CMAKE_CXX_FLAGS, so the last
  # -march wins and the build still emitted AVX-512. (The same append is why the
  # -O3 a few lines below does not apply to this target: `-O2` lands after it.)
  # A reader who read the flags, saw their own -march, and assumed it applied had
  # a binary with AVX-512 in it and no way to find out from the build log.
  #
  # THE DEFAULT IS `x86-64-v3`, NOT `native`, and that is a decision with a
  # measurement behind it rather than a preference:
  #
  #   * v3 = AVX2 + FMA + BMI, no AVX-512. Every CPU since 2013 and every AMD
  #     since Excavator.
  #   * A v3 build of this tree contains ZERO `zmm` references -- the AVX-512
  #     branch in src/encoders/bert_encoder.cpp is `#if defined(__AVX512F__)`, so
  #     it compiles out and the AVX2 path takes over.
  #   * Its output matches the znver4+AVX-512 build to 9.5e-08 max abs diff,
  #     cosine 1.000000000 -- float32 summation order, the same order the tree
  #     already calls "bf16 rounding" elsewhere.
  #   * It costs nothing measurable here: bge-large over 256 texts 8419 ms against
  #     8398 ms (0.25%, inside the run-to-run spread), and MediaPipe Hands with
  #     every convolution on the host 113 ms against 114 ms.
  #
  # So the portable build is not a fallback here, it is the same answer for less
  # money. `-DNPUE_HOST_ARCH=native` is still there for anyone who has measured a
  # win on their own silicon and wants it -- and it is a CACHE variable like the
  # MSVC one, so it is visible in cmake-gui and in CMakeCache.txt rather than
  # being a fact buried in a target property.
  set(NPUE_HOST_ARCH "x86-64-v3" CACHE STRING
      "ISA the host kernels are compiled for: a portable baseline (x86-64-v2, \
x86-64-v3), a specific -march=..., or `native` to tune for the build machine. \
`native` makes the binary UNRUNNABLE on CPUs without that ISA -- see the note \
above; measured free on this tree for x86-64-v3.")
  target_compile_options(npue_core PRIVATE -Wall -O2 -march=${NPUE_HOST_ARCH})
  target_compile_definitions(npue_core PRIVATE _FILE_OFFSET_BITS=64)
endif()

# The pose host GEMM at -O3. One source, one reason: its inner loop is an axpy
# over the channel axis and -O3's vectoriser is what turns it into loads and
# FMAs, while -O2 (the file-wide default above) leaves it as a scalar loop. The
# same arithmetic without vectorisation is not 3% slower, because the kernel is
# memory-bound on B and on the accumulators -- it is a different kernel.
#
# This is stated as a property of THIS source and not as a global -O3, because
# the rest of the runtime is I/O, dispatch bookkeeping and BF16 conversion loops,
# where -O3 buys nothing measurable and costs compile time and binary size.
if(NOT MSVC)
  set_source_files_properties(src/pose/net.cpp PROPERTIES
    COMPILE_OPTIONS "-O3")
  # arch=7 for the same reason and one more: its depthwise kernel is a plain
  # nested loop whose inner two axes the compiler vectorises at -O3 and does not
  # at -O2, and it is 39 of this container's 100 convolutions.
  set_source_files_properties(src/hands/net.cpp PROPERTIES
    COMPILE_OPTIONS "-O3")
  # arch=8 for the same reason plus its own op: 62 of its 161 convolutions are
  # depthwise, and its three DepthToSpace steps are pure plane copies whose
  # memcpy the compiler emits at -O3 and does not at -O2.
  set_source_files_properties(src/mppose/net.cpp PROPERTIES
    COMPILE_OPTIONS "-O3")
endif()

# THE THREE BINARIES. Each is one translation unit that sets its family and
# calls app::entry(); everything else it runs comes from npue_core, linked
# once per executable. They are built by default -- all three, because a tree
# that built only npuembeddings would leave the other two as names in a
# refusal message with no file behind them.
#
# They land in the same directory (runtime/build/) on purpose: default_root()
# walks up from argv[0], and one directory of three binaries keeps every
# documented command line a `./runtime/build/<verb>` away.
#
# WHOLE ARCHIVE, and the reason is a rule about static libraries rather than a
# preference: ld extracts an archive member ONLY when it resolves a symbol
# nothing else has. The five ModelLoaders (bert, gemma, vit, pose, whisper)
# contain no symbol anybody calls -- each holds a namespace-scope lambda that
# registers itself at static init -- so none of the five members was ever
# extracted, `model_loaders()` stayed EMPTY, and every path that reaches
# npue::load_model() died with
#
#     error: no model loader for arch 'bert_abs_gelu_postln'
#
# i.e. it named the arch it had just successfully read out of the container.
# Those paths are the ones no verb exercises: the flag form (`npuembeddings ..
# --model NAME --embed ...`) and `tokenize <name> <file>`. Measured on this
# machine after the split, and caught by verify_whisper_cli's "flag form"
# refusal case and verify_endpoint's --embed cross-check -- two gates out of
# 28, because 26 of them go through subcommands that build their encoder
# themselves.
#
# It was a REGRESSION of "one library, three mains": before the split every
# source was listed on the executable itself, so all of them were in the link
# unconditionally and the question never arose. `nm -C runtime/build/npuembeddings
# | grep -c BertModelLoader` printed 0 against 10 in libnpue_core.a.
#
# WHOLE_ARCHIVE puts every member in the link the way the pre-split
# add_executable did, on every platform CMake knows (GNU ld's --whole-archive,
# MSVC's /WHOLEARCHIVE), and it needs CMake 3.24 -- hence the version below.
foreach(npue_bin npuembeddings npuaudio npuimage)
  add_executable(${npue_bin} src/bin/${npue_bin}_main.cpp)
  target_link_libraries(${npue_bin} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,npue_core>")
  if(MSVC)
    target_compile_options(${npue_bin} PRIVATE /W3 /EHsc /Zc:__cplusplus /O2
                           /arch:${NPUE_HOST_ARCH})
    target_compile_definitions(${npue_bin} PRIVATE WIN32_LEAN_AND_MEAN)
  else()
    target_compile_options(${npue_bin} PRIVATE -Wall -O2 -march=${NPUE_HOST_ARCH})
    target_compile_definitions(${npue_bin} PRIVATE _FILE_OFFSET_BITS=64)
  endif()
endforeach()
