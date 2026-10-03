//===- flags.hpp --------------------------------------------------*- C++ -*-===//
//
// Every flag this binary accepts, in one table, with each one's arity.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#ifndef NPU_CLI_FLAGS_HPP
#define NPU_CLI_FLAGS_HPP

#include <cstddef>
#include <string>

namespace app {

// WHY THIS TABLE EXISTS
// --------------------
// The parser in cli.cpp used to end with a silent fall-through: an option it did
// not recognise was not stored, not skipped and not reported -- the command ran
// as if the flag had not been typed. That is how
// `--prepare-model models/all-MiniLM-L6-v2 --out /tmp/x.npue` wrote the
// container to models/all-MiniLM-L6-v2/all-MiniLM-L6-v2.npue and reported
// success: `--out` is unknown, and because the output path is POSITIONAL the
// value after it was eaten as nothing at all.
//
// So the parser now refuses an unrecognised option. That is only correct if the
// parser knows the whole flag set, and it does not: most of the runtime is
// dispatched by scanning argv directly (Runtime::run reads --dev and --bench off
// it; run_execute.hpp reads --embed, --encode-file, --probe and
// --probe-streams; run_probes.hpp reads the other six probes and the two soaks;
// vit_mode.hpp, stt_mode.hpp and gemma_mode.hpp each read their own). The first
// version of that refusal was written against the parser's own list alone and
// consequently rejected 21 real flags -- `--dev npu1`, `serve --json`,
// `--top-k`, `--classify <png>`, every `--probe-*` -- all of which had worked
// for years by virtue of falling through. verify_pack_parity caught it, because
// it is the one caller that passes --dev.
//
// The table below is the flag set. `flag_is_known()` is what the refusal asks,
// and it also recognises the two REFUSED-BY-NAME families so that their specific
// message still wins: this table is consulted inside CLI::parse(), which main.cpp
// runs BEFORE the subcommand dispatcher, so without them `--extra-ops` and
// `--npu-eltwise` would stop saying what is wrong with them and start saying only
// that they are unrecognised -- strictly less useful, on the flags most likely to
// appear in a stale command line.
//
// ARITY, and why the parser does not consume it
// --------------------------------------------
// `arity` is the number of values the flag swallows. Nothing in the parse loop
// uses it to skip arguments, and that is deliberate: every consumer reads argv
// itself, so consuming the value here would only mean the value stopped being
// visible to them, and the positional flags that guard themselves with
// `argv[i + 1][0] != '-'` (--embed, --tokenize, --add, --prepare-model) already
// handle a value that follows a flag. The arity is recorded because the next
// thing anyone will want from this table is to consume values, and a table that
// does not say how many is a table that cannot answer that.
//
// It is also checked by tools/verify/verify_cli_flags.py, which greps the sources
// for `argv[..] == "--..."` and fails if a flag the code reads is not in the
// table, or if the table names a flag nothing reads. The grep that regenerates
// this list by hand:
//
//   grep -rhoE '"--[a-z0-9-]+"' runtime/src runtime/include \
//     | sort -u
//
// The two flags with an arity above one are the probes that name two design
// directories or a directory plus two numbers.
inline int flag_arity(const std::string &a) {
  struct Entry { const char *name; int arity; };
  static const Entry kFlags[] = {
      // -- the ones cli.cpp parses into CLIArgs.
      {"--bench", 1},        {"--threads", 1},     {"--pipeline", 1},
      {"--prefix", 1},       {"--artifacts", 1},   {"--model", 1},
      {"--serve", 0},        {"--port", 1},        {"--bind", 1},
      {"--embed", 1},        {"--add", 2},         {"--tokenize", 1},
      {"--prepare-model", 2},{"--list-models", 0}, {"--help", 0},
      {"--allow-truncation", 0}, {"--allow-contention", 0},
      {"--max-len", 1},      {"--source-repo", 1}, {"--gemma-host-only", 0},
      {"--tile-k", 1},       {"--tile-n", 1},      {"--npu-extra-ops", 1},
      {"--root", 1},
      // -- read straight off argv by Runtime::run and the setup headers.
{"--dev", 1},          {"--bo-mode", 1},
      {"--cpu", 0},          {"--no-fuse-ffn", 0}, {"--sim-c-bf16", 0},
      // -- gemma_mode.hpp. `--guard-contention` is not in --help because it is
      // not typed by a person: tools/release_benchmark.ps1 passes it, and
      // without this row the unrecognised-option refusal would break the
      // release benchmark with an error about a flag it never mentions itself.
      {"--guard-contention", 0},
      // -- run_execute.hpp / run_probes.hpp.
      {"--probe", 0},        {"--probe-streams", 0}, {"--probe-pair", 0},
      {"--probe-ctx", 0},    {"--probe-design", 1},  {"--probe-insts", 2},
      {"--probe-rtp", 2},    {"--probe-bo", 3},      {"--soak-npu", 1},
      {"--soak-cpu", 1},     {"--encode-file", 1},   {"--transcribe", 1},
      {"--audio", 1},
      // -- vit_mode.hpp (arch=5) and stt_mode.hpp (the whisper modes).
      {"--classify", 1},     {"--json", 0},
      {"--top-k", 0},        {"--convert", 0},    {"--language", 1},
      {"--task", 1},         {"--max-new", 1},    {"--chunk-seconds", 1},
      {"--stride-seconds", 1},
      // -- pose_mode.hpp (arch=6). --text is a no-argument policy flag that
      // switches the mode from its default JSON to a human summary; the four
      // others are the decode thresholds, and each takes a value so that
      // `flag_takes_value()` in cli/subcommand.cpp swallows the number rather
      // than opening it as a PNG.
      {"--pose", 1},         {"--text", 0},        {"--conf", 1},
      {"--iou", 1},          {"--kpt", 1},         {"--max-det", 1},
      // Diagnostic: dump every graph node's output for a node-by-node diff
      // against tools/verify/verify_pose.py. See pose_mode.hpp.
      {"--pose-dump", 1},
      // The pose endpoint's port. A SEPARATE flag from --serve because that one
      // is the embedding endpoint's: same arity, same reader, different mode,
      // and a container's arch picks which of the two a given model reaches.
      {"--pose-server", 1},
      {"--token", 1},
      // -- REFUSED BY NAME, not accepted. Listed so that CLI::parse() lets them
      // through to refuse_removed_op_flags / refuse_exporter_only_flags, whose
      // messages name the replacement; run_setup.hpp calls both at line 314 and
      // stt/vit_mode.hpp at theirs, all of which are downstream of this parse.
      {"--npu-eltwise", 0},  {"--host-gelu", 0}, {"--host-ln", 0},
      {"--host-sm", 0},      {"--npu-ops", 0},   {"--extra-ops", 0},
  };
  for (const Entry &e : kFlags)
    if (a == e.name) return e.arity;
  return -1;  // not a flag this binary knows
}

inline bool flag_is_known(const std::string &a) { return flag_arity(a) >= 0; }

// `-h` is the one short option. It is spelled out rather than folded into the
// table because the table is keyed on the long spelling everywhere else, and a
// one-off two-character special case reads more clearly here than it does as a
// row whose every sibling starts with "--".
inline bool flag_is_known_short(const std::string &a) { return a == "-h"; }

}  // namespace app

#endif  // NPU_CLI_FLAGS_HPP
