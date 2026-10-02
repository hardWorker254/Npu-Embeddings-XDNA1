//===- test_pack_mac.cpp --------------------------------------------*- C++ -*-//
//
// Do the C++ packer and tools/pack/pack_npue.py agree, for BOTH generations?
//
// The B panel's byte order is the MMAC sub-tile, and it differs per board
// (npu1 s8/t4, npu2 s8/t8). Two implementations of one layout is already a
// risk; two implementations of a layout that is itself per-generation is two
// risks, and the failure is invisible: the byte count, the shapes and the
// layout_hash all agree, and only the products are wrong. So this packs the
// same checkpoint twice -- once per device -- and prints each file's sha256
// for tools/verify/verify_pack_parity.py to compare against the Python packer.
//
// NOT a self-contained gate on purpose: the C++ side must be byte-identical to
// the reference, and the reference is the Python packer, so the comparison
// belongs where both are visible. tools/verify/verify_pack_parity.py grew a --device
// flag for exactly this, and this binary is what it drives on the platforms
// where the pack path is reachable from the CLI.
//
// Build:
//   g++ -std=c++17 -O1 -I runtime/include runtime/tests/test_pack_mac.cpp \
//       runtime/src/common/npue_pack.cpp runtime/src/common/onnx_read.cpp \
//       -o /tmp/test_pack_mac
// Run:
//   /tmp/test_pack_mac <checkpoint-dir> <out-prefix>
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "common/npue_pack.hpp"

#include <cstdio>
#include <string>

int main(int argc, char **argv) {
  using namespace npue;
  if (argc < 3) {
    std::printf("usage: %s <checkpoint-dir> <out-prefix>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const std::string prefix = argv[2];

  for (const char *dev : {"npu1", "npu2"}) {
    const MacGeom mac = mac_for_device(dev);
    const Layout lay = gemm_b_layout(64, 48, mac.s, mac.t);
    const std::string out = prefix + "." + dev + ".npue";
    auto log = [](const std::string &m) { std::printf("  %s\n", m.c_str()); };
    try {
      prepare_model(dir, dir + "/vocab.txt",
                    dir + "/config.json", "mean", "test/repo", out,
                    lay.json, lay.hash, 64, 48, 256, log, mac);
    } catch (const std::exception &e) {
      std::printf("  %s: pack failed: %s\n", dev, e.what());
      return 1;
    }
    std::printf("%s %s\n", sha256_file(out).c_str(), out.c_str());
  }
  return 0;
}
