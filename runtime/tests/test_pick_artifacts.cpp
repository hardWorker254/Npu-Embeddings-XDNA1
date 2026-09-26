//===- test_pick_artifacts.cpp ---------------------------------*- C++ -*-===//
//
// Does pick_artifacts() find the right directory?
//
// The layout it has to resolve: <root>/<model>/artifacts_npu<N>/..., with the
// model name optional. Getting this wrong is silent in the worst way -- the
// runtime starts, finds *some* design set, and computes with a geometry that
// was not built for the model. So the picked path is compared exactly.
//
// Build and run (no NPU; the fixture is written to a temp dir):
//   g++ -std=c++17 -O1 -I runtime/include runtime/tests/test_pick_artifacts.cpp \
//       -o /tmp/test_pick_artifacts && /tmp/test_pick_artifacts
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "common/design_selection.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace app;

static void mk(const std::string &dir) {
  std::filesystem::create_directories(dir + "/gemm_rtp");
  std::ofstream f(dir + "/gemm_rtp/design.json");
  f << R"({
  "streams": [
    {"op":"qkv","file":"a.bin","batch":1,"slot":0,"M":64,"K":384,"N":1152},
    {"op":"attn_out","file":"b.bin","batch":1,"slot":1,"M":64,"K":384,"N":384},
    {"op":"ffn_up","file":"c.bin","batch":1,"slot":2,"M":64,"K":384,"N":1536},
    {"op":"ffn_down","file":"d.bin","batch":1,"slot":3,"M":64,"K":1536,"N":384}
  ]
})";
}

int main() {
  const std::string root =
      std::filesystem::temp_directory_path().string() + "/nds_pick_root";
  std::filesystem::remove_all(root);
  mk(root + "/all-MiniLM-L6-v2/artifacts_npu1");

  const std::string got =
      pick_artifacts(root, 384, 1536, false, 0, "", "", "all-MiniLM-L6-v2");
  std::printf("nested by name  -> '%s'\n", got.c_str());

  const std::string no_name =
      pick_artifacts(root, 384, 1536, false, 0, "", "", "");
  std::printf("without name    -> '%s'\n", no_name.c_str());

  const auto cands = artifacts_candidates(root, "all-MiniLM-L6-v2");
  std::printf("candidates:\n");
  for (const auto &c : cands) std::printf("  %s\n", c.c_str());

  const bool ok = got == root + "/all-MiniLM-L6-v2/artifacts_npu1";
  std::printf("RESULT %s\n", ok ? "PASS" : "FAIL");
  std::filesystem::remove_all(root);
  return ok ? 0 : 1;
}
