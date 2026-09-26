//===- test_design_selection.cpp ---------------------------------*- C++ -*-===//
//
// Does design_fits() agree with the running device?
//
// The rule it has to get right: a design set is NOT portable between
// generations. design.json records arch/device, and a set exported for npu2
// must be refused on npu1 even when the geometry fits on paper -- a set that
// loads and computes plausible wrong numbers is worse than one that refuses.
//
// A set with no recorded device predates the field and is accepted, which is
// the third case here.
//
// Build and run (no NPU, no artifacts of its own -- the fixtures are written
// to a temp dir):
//   g++ -std=c++17 -O1 -I runtime/include runtime/tests/test_design_selection.cpp \
//       -o /tmp/test_design_selection && /tmp/test_design_selection
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "common/design_selection.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace app;

int main() {
  const std::string dir = std::filesystem::temp_directory_path().string() +
                          "/nds_artifacts_npu1";
  const std::string tmp = std::filesystem::temp_directory_path().string() +
                          "/nds_legacy_set";
  namespace fs = std::filesystem;
  fs::remove_all(dir);
  fs::remove_all(tmp);

  // A design set for npu1 only: the arch/device pair is the whole point.
  fs::create_directories(dir + "/gemm_rtp");
  std::ofstream(dir + "/gemm_rtp/design.json")
      << R"({"arch": 1, "device": "npu1", "streams": [
           {"op": "qkv", "K": 384, "N": 1152},
           {"op": "attn_out", "K": 384, "N": 384},
           {"op": "ffn_up", "K": 384, "N": 1536},
           {"op": "ffn_down", "K": 1536, "N": 384}]})";

  set_running_device("npu1");
  const bool fits1 = design_fits(dir, 384, 1536, false, 1152);
  set_running_device("npu2");
  const bool fits2 = design_fits(dir, 384, 1536, false, 1152);

  std::printf("running npu1 -> fits: %d\n", (int)fits1);
  std::printf("running npu2 -> fits: %d\n", (int)fits2);

  // No device/arch recorded: must still fit (legacy sets).
  fs::create_directories(tmp + "/gemm_rtp");
  std::ofstream(tmp + "/gemm_rtp/design.json")
      << R"({"streams": [
           {"op": "qkv", "K": 384, "N": 1152},
           {"op": "attn_out", "K": 384, "N": 384},
           {"op": "ffn_up", "K": 384, "N": 1536},
           {"op": "ffn_down", "K": 1536, "N": 384}]})";
  set_running_device("npu2");
  const bool fits_legacy = design_fits(tmp, 384, 1536, false, 1152);
  std::printf("legacy (no device) on npu2 -> fits: %d\n", (int)fits_legacy);

  const bool ok = fits1 && !fits2 && fits_legacy;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  fs::remove_all(dir);
  fs::remove_all(tmp);
  return ok ? 0 : 1;
}
