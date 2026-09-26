//===- npu_ops_flag.hpp -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one vocabulary for "which elementwise ops run on the array".
//
// THE FLAG
// --------
//   --npu-ops <codes>       a comma-separated subset of the codes below; the
//                          default (and an empty list) is NONE of them, which
//                          is the measured-faster host path
//
// The codes are short because this is typed on a command line and the long
// names run to nine characters: `layn`, `softm`, `gelu`. `--npu-ops layn,softm`
// is the old `--npu-eltwise --host-gelu`, and `--npu-ops` with nothing after it
// is the old `--npu-eltwise --host-ln --host-sm --host-gelu`. There is no
// inverse flag: an op is on the host exactly when it is not in the list, and
// two spellings for one setting is how a flag and its inverse drift apart.
//
// THE TABLE IS ALSO THE DESIGN-DIRECTORY NAME
// -------------------------------------------
// `design` is the sibling directory the runtime opens and the exporter writes,
// so the code, the directory and the kernel family cannot be three unrelated
// strings. `long_name` is only ever shown to a human.
//
// WHY THERE IS NO CENTRAL DEFINITION
// ---------------------------------
// The exporter that builds these sets is Python and this parser is C++, and
// neither can include the other. So the table is written twice -- here and in
// tools/npu_ops.py -- with each pointing at the other. The failure mode of that
// duplication is a refusal by name, not a wrong number: an unknown code throws
// and lists the valid ones on both sides, and a design directory that does not
// exist is refused with the command that builds it.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace app {

struct NpuOp {
  const char *code;      // what --npu-ops takes
  const char *design;    // the sibling design directory, and the kernel family
  const char *long_name; // for messages only
};

inline const std::vector<NpuOp> &npu_op_table() {
  // Keep in sync with tools/npu_ops.py -- see the header.
  static const std::vector<NpuOp> table = {
      {"gelu", "gelu", "GELU"},
      {"layn", "layernorm", "LayerNorm"},
      {"softm", "softmax", "softmax"},
  };
  return table;
}

inline std::string npu_op_codes() {
  std::string s;
  for (const auto &op : npu_op_table()) {
    if (!s.empty()) s += ", ";
    s += op.code;
  }
  return s;
}

inline const NpuOp *find_npu_op(const std::string &code) {
  for (const auto &op : npu_op_table())
    if (code == op.code) return &op;
  return nullptr;
}

// "gelu, layn" and " gelu ,layn " both mean {gelu, layn}; an empty component is
// skipped, so a trailing comma and `--npu-ops ""` are the same thing.
inline std::set<std::string> parse_npu_ops(const std::string &list) {
  std::set<std::string> out;
  std::string item;
  auto commit = [&]() {
    std::string code;
    for (char c : item)
      if (!std::isspace(static_cast<unsigned char>(c))) code.push_back(c);
    item.clear();
    if (code.empty()) return;
    if (!find_npu_op(code))
      throw std::runtime_error(
          "--npu-ops: '" + code +
          "' is not an op this build knows. Valid codes: [" + npu_op_codes() +
          "] (layn = LayerNorm, softm = softmax, gelu = GELU). Nothing listed "
          "means all three run on the host, which is the measured-faster path.");
    out.insert(code);
  };
  for (char c : list) {
    if (c == ',') {
      commit();
      continue;
    }
    item.push_back(c);
  }
  commit();
  return out;
}

// The flags this replaced. They are REFUSED, not ignored: a script that still
// says --host-ln has asked for a specific thing, and dropping it on the floor is
// how "the flag was there and nothing happened" happens -- the one failure the
// subcommand whitelist is already written to prevent (see subcommand.cpp).
inline const std::vector<std::pair<const char *, const char *>> &
removed_op_flags() {
  static const std::vector<std::pair<const char *, const char *>> v = {
      {"--npu-eltwise", "--npu-ops gelu,layn,softm"},
      {"--host-gelu", "--npu-ops without gelu"},
      {"--host-ln", "--npu-ops without layn"},
      {"--host-sm", "--npu-ops without softm"},
  };
  return v;
}

// Takes the arguments as strings because the two callers hold different types
// (Runtime::run's char**, the subcommand whitelist's const char* const*), and a
// const cast to make one signature fit both is how a check ends up reading the
// wrong thing.
inline void refuse_removed_op_flags(const std::vector<std::string> &args) {
  for (const auto &a : args)
    for (const auto &r : removed_op_flags())
      if (a == r.first)
        throw std::runtime_error(
            std::string(r.first) +
            " is gone: one flag now says which ops go on the array, and an op "
            "is on the host when it is NOT listed. Use " + r.second +
            ". (The old flags are refused rather than ignored so a stale command "
            "line cannot look like it worked.)");
}

inline void refuse_removed_op_flags(int argc, char **argv) {
  std::vector<std::string> args;
  args.reserve(argc > 0 ? argc : 0);
  for (int i = 0; i < argc; ++i) args.emplace_back(argv[i] ? argv[i] : "");
  refuse_removed_op_flags(args);
}

}  // namespace app
