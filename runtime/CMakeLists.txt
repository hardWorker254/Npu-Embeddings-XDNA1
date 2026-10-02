# NpuEmbeddings
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
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
add_executable(npuembeddings
  src/main.cpp
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
  # --npu-extra-ops names it.
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
  # Subtask 6: CLI system
  src/cli/cli.cpp
  src/cli/subcommand.cpp
)

target_include_directories(npuembeddings PRIVATE include "${XRT_INCLUDE_DIR}")

if(DEFINED BOOST_ROOT AND EXISTS "${BOOST_ROOT}")
  target_include_directories(npuembeddings PRIVATE "${BOOST_ROOT}")
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
  target_link_libraries(npuembeddings PRIVATE winhttp ws2_32)
  target_link_libraries(npuembeddings PRIVATE PNG::PNG JPEG::JPEG)
else()
  target_link_libraries(npuembeddings PRIVATE ${XRT_COREUTIL_LIB})

  find_package(CURL REQUIRED)
  target_link_libraries(npuembeddings PRIVATE CURL::libcurl)
  target_link_libraries(npuembeddings PRIVATE PNG::PNG JPEG::JPEG)
endif()

if(MSVC)
  set(NPUE_HOST_ARCH "AVX2" CACHE STRING
      "MSVC /arch: level for the host kernels (AVX2 or AVX512)")
  target_compile_options(npuembeddings PRIVATE /W3 /EHsc /Zc:__cplusplus /O2
                         /arch:${NPUE_HOST_ARCH})
  target_compile_definitions(npuembeddings PRIVATE WIN32_LEAN_AND_MEAN)
else()
  target_compile_options(npuembeddings PRIVATE -Wall -O2 -march=native)
  target_compile_definitions(npuembeddings PRIVATE _FILE_OFFSET_BITS=64)
endif()
