//===- int4_panel_probe.cpp -------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the int4 DECODE, compiled and compared to Python's.
// SPDX-License-Identifier: Apache-2.0
//
// tools/verify/verify_i4_scheme.py builds a container with the packer, asks
// Python for the panel the array consumes, and asks this program for the same
// panel. The two must agree BYTE FOR BYTE -- not "closely", because a nibble
// read at the wrong index, a group folded at the wrong row, or a sign extended
// from the wrong bit all still produce a right-sized, right-hashed panel of
// plausible-looking numbers, and the only thing that distinguishes them is
// equality with the other implementation of the same arithmetic.
//
// This is the check `verify_i8_scheme.py` cannot make: an int8 container
// already agrees with itself by construction, but int4 is the FIRST format
// where the runtime does arithmetic the writer also did. npue.fold_i4() and
// common/int4_panel.hpp are two definitions of one order, and two definitions
// of one order is exactly what byte equality exists to hold apart.
//
// Usage:  int4_panel_probe <file.npue> <out-prefix> [--bf16-design] <name>...
//        writes <out-prefix>.0, <out-prefix>.1, ... in order, and prints
//        `ok <name> <bytes> staged <bytes> stored <bytes>` per operand.
//        Exit 1 on the first refusal, with `FAIL <name>` then the message.
//
// Build (verify_i4_scheme.py does this itself):
//   g++ -std=c++17 -O2 -Iruntime/include \
//       tools/verify/int4_panel_probe.cpp runtime/src/model.cpp -o probe

#include <cstdio>
#include <fstream>
#include <string>

#include "common/int4_panel.hpp"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: int4_panel_probe <file.npue> <out-prefix> "
                 "[--bf16-design] <name>...\n");
    return 2;
  }
  const std::string prefix = argv[2];
  size_t a_bytes = 1;
  int first = 3;
  if (std::string(argv[3]) == "--bf16-design") {
    a_bytes = 2;
    first = 4;
  }
  if (first >= argc) {
    std::fprintf(stderr, "no tensor names given\n");
    return 2;
  }

  try {
    npue::File f(argv[1]);
    int n = 0;
    for (int i = first; i < argc; ++i) {
      const std::string name = argv[i];
      try {
        npue::Panel p = npue::gemm_b_panel(f, name, a_bytes);
        // staged_bytes() is the same accessor's opinion of what stage() holds.
        // It is a separate code path from the decode and they must not drift:
        // vit and whisper keep a running total from it, and a total that is
        // half the truth for an I4 operand is a printout nobody would chase.
        const size_t staged = npue::staged_bytes(f, name);
        if (staged != p.bytes.bytes) {
          std::printf("FAIL %s\n     staged_bytes says %zu, the panel is %zu\n",
                      name.c_str(), staged, p.bytes.bytes);
          return 1;
        }
        std::string err;
        std::ofstream out(prefix + "." + std::to_string(n++),
                          std::ios::binary | std::ios::trunc);
        if (!out) {
          std::printf("FAIL %s\n     cannot write %s\n", name.c_str(),
                      (prefix + "." + std::to_string(n)).c_str());
          return 1;
        }
        out.write(static_cast<const char *>(p.bytes.data),
                  static_cast<std::streamsize>(p.bytes.bytes));
        out.close();
        std::printf("ok %-26s panel %9zu  staged %9zu  stored %9llu  %s\n",
                    name.c_str(), p.bytes.bytes, staged,
                    static_cast<unsigned long long>(f.info(name).nbytes),
                    f.info(name).dtype.c_str());
      } catch (const std::exception &e) {
        std::printf("FAIL %s\n     %s\n", name.c_str(), e.what());
        return 1;
      }
    }
    std::printf("count=%d\n", n);
  } catch (const std::exception &e) {
    std::printf("FAIL (file)\n     %s\n", e.what());
    return 1;
  }
  return 0;
}
