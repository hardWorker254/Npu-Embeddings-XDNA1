# NpuEmbeddings -- M7 C++ runtime.
# SPDX-License-Identifier: Apache-2.0
#
# TRAP, from CLAUDE.md and mlir-aie #3048:
#   project() MUST come before find_package(XRT), or linking silently
#   downgrades to static. Silently -- the build succeeds and the binary is
#   wrong. Do not reorder these two lines.
cmake_minimum_required(VERSION 3.20)
project(npuembed_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# XRT lives at C:\Xilinx\XRT on this machine, NOT under Program Files, and
# XILINX_XRT must stay unset because it poisons Windows builds (iron_setup.py).
# XRT_ROOT is the variable to use.
if(NOT DEFINED XRT_ROOT)
  if(DEFINED ENV{XRT_ROOT})
    set(XRT_ROOT $ENV{XRT_ROOT})
  else()
    set(XRT_ROOT "C:/Xilinx/XRT")
  endif()
endif()
message(STATUS "XRT_ROOT: ${XRT_ROOT}")

find_path(XRT_INCLUDE_DIR xrt/xrt_device.h
          HINTS "${XRT_ROOT}/include" REQUIRED)
find_library(XRT_COREUTIL_LIB NAMES xrt_coreutil
             HINTS "${XRT_ROOT}/lib" "${XRT_ROOT}/lib/Release" REQUIRED)
message(STATUS "XRT headers: ${XRT_INCLUDE_DIR}")
message(STATUS "XRT lib:     ${XRT_COREUTIL_LIB}")

# Boost is only needed if the compiler reports pre-C++17, which on MSVC means
# /Zc:__cplusplus was not passed (see below). With that flag XRT uses std::any
# and this is unused -- kept as a fallback so the build works either way.
if(NOT DEFINED BOOST_ROOT AND EXISTS "C:/dev/boost_1_88_0")
  set(BOOST_ROOT "C:/dev/boost_1_88_0")
endif()
if(DEFINED BOOST_ROOT AND EXISTS "${BOOST_ROOT}")
  message(STATUS "Boost (fallback only): ${BOOST_ROOT}")
endif()

add_executable(npuembed
  src/main.cpp
  src/npue.cpp
    src/tokenizer.cpp
  src/npue_pack.cpp
  src/npu_device.cpp
  src/npu_contention.cpp
  src/hub.cpp
  # arch=1 (EmbeddingGemma family, tasks/0061/0063/00XX): all three are
  # standalone and XRT-free (see their own file headers) -- added here only
  # so main.cpp's run_gemma_mode() can call them. Nothing in them touches
  # npu_device.hpp/XRT.
  src/tokenizer_gemma.cpp
  src/gemma_kernels.cpp
  src/gemma_encode.cpp
  # C++ port of tools/gen_gemma_tokenizer_table.py (tasks/0067): a
  # dependency-free JSON DOM parser and the table generator built on it.
  # Both are XRT-free like the arch=1 files above; json_min is generic and
  # not itself EmbeddingGemma-specific.
  src/json_min.cpp
  src/gemma_tokenizer_gen.cpp
)
target_include_directories(npuembed PRIVATE include "${XRT_INCLUDE_DIR}")
if(DEFINED BOOST_ROOT AND EXISTS "${BOOST_ROOT}")
  target_include_directories(npuembed PRIVATE "${BOOST_ROOT}")
endif()
# ws2_32: the /v1/embeddings endpoint (include/http.hpp) talks Winsock.
# winhttp: fetching a model (src/hub.cpp). Both ship with Windows, so the
# release stays one executable with no DLL to carry -- which is the point of
# having replaced the curl+certutil batch script with this.
target_link_libraries(npuembed PRIVATE "${XRT_COREUTIL_LIB}" ws2_32 winhttp)

# The product is `npuembeddings`; `npuembed.exe` is kept as the name every
# task log, script and doc in this repo already invokes. One binary, two
# names, so neither the new CLI nor the existing measurement scripts break.
add_custom_command(TARGET npuembed POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:npuembed>"
          "$<TARGET_FILE_DIR:npuembed>/npuembeddings.exe")

if(MSVC)
  # /Zc:__cplusplus is REQUIRED, not cosmetic.
  #
  # xrt/detail/any.h picks std::any when __cplusplus >= 201703L and falls back
  # to boost/any.hpp otherwise. MSVC reports __cplusplus as 199711L unless this
  # flag is given -- even under -std:c++17 -- so without it the build demands
  # Boost and fails with:
  #     fatal error C1083: Cannot open include file: 'boost/any.hpp'
  # which reads as a missing dependency and is really a missing flag.
  # /O2 explicitly: a Release-less single-config generator would otherwise build
  # this unoptimised, and the host-side bf16 conversion is 13.8 M elements per
  # encode. /arch:AVX2 enables the vectorised conversion in main.cpp -- Strix
  # Point is Zen 5, so AVX2 is a floor, not an assumption.
  target_compile_options(npuembed PRIVATE /W3 /EHsc /Zc:__cplusplus /O2 /arch:AVX2)
  # XRT headers use std::min/max; the Windows macros break them. XRT's own
  # headers also define NOMINMAX, so let theirs win rather than warning.
  target_compile_definitions(npuembed PRIVATE WIN32_LEAN_AND_MEAN)
endif()
