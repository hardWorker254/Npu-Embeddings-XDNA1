//===- npu_ops_flag.hpp -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one vocabulary for "which ops go on the array".
//
// THE FLAG
// --------
//   --npu-ops <codes>   a comma-separated subset of the codes below; the
//                            default (and an empty list) is NONE of them, which
//                            is the measured-faster host path
//
// ONE SPELLING, AND ONLY ONE SIDE HAS A FLAG ANY MORE
// ----------------------------------------------------
// This flag SELECTS; it does not build. One export command
// (`export_gemm_rtp.py --target X`) compiles every design that X can actually
// use -- the registry that decides is tools/lib/npu_ops.py, read by
// exporters/gemm_rtp/build.py -- and this flag is the only thing that says
// which of the built ones to RUN. So there is no second name to keep in step
// with this one, and a user cannot hold two spellings of the same selection.
//
// The history is worth one paragraph because the bug it caused was silent. The
// runtime used to say `--npu-ops` and the exporter `--npu-extra-ops`: two names
// one suffix apart, taking the same codes, so `serve ... --npu-extra-ops
// gelu,softm,layn,conv` selected nothing, was dropped by the subcommand
// whitelist without a word, and printed a status block claiming the host. The
// runtime's spelling is now the surviving one, and the exporter's old name is
// refused BY NAME (see removed_op_flags) rather than aliased -- a flag that
// works under two names is two flags, and the undocumented one is the one
// people keep typing.
//
// The codes are short because this is typed on a command line and the long
// names run to nine characters: `layn`, `softm`, `gelu`. `--npu-ops
// layn,softm` is the old `--npu-eltwise --host-gelu`, and the flag with nothing
// after it is the old `--npu-eltwise --host-ln --host-sm --host-gelu`. There is
// no inverse flag: an op is on the host exactly when it is not in the list, and
// two spellings for one setting is how a flag and its inverse drift apart.
//
// THE TABLE IS ALSO THE DESIGN-DIRECTORY NAME
// -------------------------------------------
// `design` is the sibling directory the runtime opens and the exporter writes,
// so the code, the directory and the kernel family cannot be three unrelated
// strings. `long_name` is only ever shown to a human.
//
// ONE ROW IS NOT A DIRECTORY: `conv`
// ----------------------------------
// Whisper's conv1/conv2 are a GEMM-shaped op, not an eltwise one, and they run
// on the encoder set's OWN [rows, d, d] stream (attn_out's shape) rather than
// on a sibling xclbin -- so conv's design field is empty, and asking an exporter
// to build it would compile a directory nothing opens. The code is here because
// the question it answers is the same one this flag exists to answer: an op is
// on the array when it is listed. A container that is not a speech-to-text model
// refuses the code by name rather than ignoring it; see run_setup.hpp.
//
// WHY THERE IS NO CENTRAL DEFINITION
// ---------------------------------
// The exporter that builds these sets is Python and this parser is C++, and
// neither can include the other. So the table is written twice -- here and in
// tools/lib/npu_ops.py -- with each pointing at the other. The failure mode of that
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
  // Keep in sync with tools/lib/npu_ops.py -- see the header.
  static const std::vector<NpuOp> table = {
      {"gelu", "gelu", "GELU"},
      {"layn", "layernorm", "LayerNorm"},
      {"softm", "softmax", "softmax"},
      // No design directory, and deliberately so: see the header. A design of
      // "" is what tells the loading code there is nothing to open.
      {"conv", "", "conv1d (Whisper's audio front end)"},
      // Also no directory, and for a different reason: this one ADDS two
      // instruction streams (attn_qk, attn_av) to the two GEMM sets, so the
      // export has work to do and it happens inside the gemm_rtp /
      // gemm_rtp_dec directories rather than beside them. At run time the
      // streams are either in design.json or they are not, and asking for this
      // code when they are not is refused by name.
      {"attn", "", "Whisper's attention, as GEMMs"},
      {"mproj", "", "Whisper's mel filter bank, as a GEMM"},
      {"fft", "", "Whisper's 400-point transform, as a GEMM"},
      {"logit", "", "the vocabulary projection, as a GEMM"},
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

// Comma-joined codes, in TABLE order rather than the order the user typed them.
// A message that lists what was asked for should read the same every time for
// the same set, and "layn,softm" is easier to match against a help block than a
// permutation of itself. The flag value itself is echoed unchanged elsewhere, so
// what the user typed is never lost -- this is for repetition, not for evidence.
inline std::string npu_op_list(const std::set<std::string> &codes) {
  std::string s;
  for (const auto &op : npu_op_table())
    if (codes.count(op.code)) {
      if (!s.empty()) s += ",";
      s += op.code;
    }
  return s;
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
          "] (layn = LayerNorm, softm = softmax, gelu = GELU, conv = Whisper's "
          "conv1/conv2, attn = Whisper's attention as GEMMs, mproj = the mel "
          "filter bank as a GEMM, fft = the 400-point transform as a GEMM, "
          "logit = the vocabulary projection as a GEMM). "
          "Nothing listed means every op runs on the host, which is the "
          "measured-faster path.");
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
      // The name this flag used to carry while the exporter had a flag of its
      // own. Refused rather than aliased: a flag that still works under two
      // names is two flags, and the second one is the one nobody documents.
      // Now the exporter needs none, so the shorter name is the one left.
      {"--npu-extra-ops", "--npu-ops"},
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

// THE STT-ONLY ELTWISE FLAG, REFUSED AT RUN TIME
// ----------------------------------------------
// `--extra-ops` is tools/export/export_eltwise.py's own spelling and stays that
// tool's: it is the low-level per-kernel builder (its default builds all three
// eltwise designs, and the flag only narrows that for a direct call), and the
// gate that compares exporters against history requires the flag to exist (see
// tools/verify/parity_exporters.py REQUIRED_FLAGS_ELT). The GEMM exporter has no
// flag at all any more, so nobody reaches this one by accident from a top-level
// export -- but a script that does reach for it, having learned it from a README
// that predates the split, is told what to type instead.
inline const std::vector<std::pair<const char *, const char *>> &
exporter_only_flags() {
  static const std::vector<std::pair<const char *, const char *>> v = {
      {"--extra-ops", "--npu-ops"},
  };
  return v;
}

inline void refuse_exporter_only_flags(const std::vector<std::string> &args) {
  for (const auto &a : args)
    for (const auto &r : exporter_only_flags())
      if (a == r.first)
        throw std::runtime_error(
            std::string(r.first) +
            " is tools/export/export_eltwise.py's BUILD flag: at run time it selected "
            "nothing and was dropped without a word, so a command line that "
            "asked for an op quietly ran without it. To send ops to the array, "
            "use " + r.second + ".");
}

inline void refuse_exporter_only_flags(int argc, char **argv) {
  std::vector<std::string> args;
  args.reserve(argc > 0 ? static_cast<size_t>(argc) : 0);
  for (int i = 0; i < argc; ++i) args.emplace_back(argv[i] ? argv[i] : "");
  refuse_exporter_only_flags(args);
}

inline void refuse_removed_op_flags(int argc, char **argv) {
  std::vector<std::string> args;
  args.reserve(argc > 0 ? argc : 0);
  for (int i = 0; i < argc; ++i) args.emplace_back(argv[i] ? argv[i] : "");
  refuse_removed_op_flags(args);
}

}  // namespace app
