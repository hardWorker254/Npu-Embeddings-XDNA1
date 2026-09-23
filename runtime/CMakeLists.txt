# NpuEmbeddings
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(npuembed_runtime LANGUAGES CXX)

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

add_executable(npuembed
  src/main.cpp
  src/runtime.cpp
  # Subtask 1: core infrastructure (device/design/model/pool).
  src/device.cpp
  src/design.cpp
  src/model.cpp
  src/pool.cpp
  # Subtask 10: shared utilities under common/.
  src/common/npue_pack.cpp
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
  # Subtask 7: STT extension (Whisper). The tokenizer stub lives with the
  # other tokenizers but is grouped here with the rest of the Whisper
  # placeholders.
  src/tokenizers/whisper.cpp
  src/whisper/whisper_model.cpp
  src/whisper/stt_engine.cpp
  src/whisper/whisper.cpp
  # Subtask 4: Encoder system
  src/encoders/bert_encoder.cpp
  src/encoders/gemma_npu_encoder.cpp
  src/encoders/gemma_host_encoder.cpp
  src/encoder.cpp
  # Subtask 5: server system (server.hpp is header-only; these are the two
  # translation units that were previously inline in main.cpp/run_execute.hpp).
  src/server/embed_backend.cpp
  src/server/embed_service.cpp
  # Subtask 6: CLI system
  src/cli/cli.cpp
  src/cli/subcommand.cpp
)

target_include_directories(npuembed PRIVATE include "${XRT_INCLUDE_DIR}")

if(DEFINED BOOST_ROOT AND EXISTS "${BOOST_ROOT}")
  target_include_directories(npuembed PRIVATE "${BOOST_ROOT}")
endif()

if(WIN32)
  target_link_libraries(npuembed PRIVATE winhttp ws2_32)
else()
  target_link_libraries(npuembed PRIVATE ${XRT_COREUTIL_LIB})

  find_package(CURL REQUIRED)
  target_link_libraries(npuembed PRIVATE CURL::libcurl)
endif()

if(WIN32)
  set(NPUE_COPY_EXT ".exe")
else()
  set(NPUE_COPY_EXT "")
endif()

add_custom_command(TARGET npuembed POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:npuembed>"
          "$<TARGET_FILE_DIR:npuembed>/npuembeddings${NPUE_COPY_EXT}")

if(MSVC)
  set(NPUE_HOST_ARCH "AVX2" CACHE STRING
      "MSVC /arch: level for the host kernels (AVX2 or AVX512)")
  target_compile_options(npuembed PRIVATE /W3 /EHsc /Zc:__cplusplus /O2
                         /arch:${NPUE_HOST_ARCH})
  target_compile_definitions(npuembed PRIVATE WIN32_LEAN_AND_MEAN)
else()
  target_compile_options(npuembed PRIVATE -Wall -O2 -march=native)
  target_compile_definitions(npuembed PRIVATE _FILE_OFFSET_BITS=64)
endif()
