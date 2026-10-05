//===- flags.hpp --------------------------------------------------*- C++ -*-===//
//
// Every flag this binary accepts, in one table, with each one's arity.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#ifndef NPU_CLI_FLAGS_HPP
#define NPU_CLI_FLAGS_HPP

#include <cctype>
#include <cstddef>
#include <cstdlib>
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
//   grep -rhoE '"--[a-z0-9-]+"' runtime/src runtime/include | sort -u
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
      {"--tile-k", 1},       {"--tile-n", 1},      {"--npu-ops", 1},
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
      // -- hands_mode.hpp (arch=7). --hands is the image list, arity 1, and it
      // REPEATS -- run_hands() rewrites positionals into it and hands_mode
      // accumulates every occurrence, the same treatment --classify and --pose
      // get. --max-hands caps how many detections reach the landmark network;
      // it is a COUNT and not a threshold, because every hand is an independent
      // crop and the only lever on the second stage is how many of them to run.
      // This mode's score, NMS and crop thresholds are NOT here on purpose:
      // they come from the container, and hands_mode REFUSES --conf/--iou/--kpt/
      // --max-det by name rather than accepting and ignoring them.
      {"--hands", 1},       {"--max-hands", 1},
      // Diagnostic: dump every graph node's output of BOTH networks, for the
      // node-by-node diff against tools/verify/verify_hands.py. See
      // hands_mode.hpp.
      {"--hands-dump", 1},
      // -- mppose_mode.hpp (arch=8). Same treatment as the two above and the same
      // reasons: --mppose is the image list, arity 1, REPEATED, and --max-people
      // caps how many detections reach the landmark network -- a COUNT, because
      // every person is an independent ROTATED crop and the only lever on the
      // second stage is how many of them to run. No threshold flags again: the
      // detector's score and NMS and the landmark net's confidence all come from
      // the container, and mppose_mode REFUSES --conf/--iou/--kpt/--max-det AND
      // arch=6's and arch=7's by name rather than accepting and ignoring them --
      // because two different containers called `pose` and `hands` answer to those
      // flags with different semantics, and this mode is the third.
      {"--mppose", 1},       {"--max-people", 1},
      // Diagnostic: dump every graph node's output of BOTH networks, for the
      // node-by-node diff against tools/verify/verify_mppose.py. See
      // mppose_mode.hpp.
      {"--mppose-dump", 1},
      {"--token", 1},
      // -- REFUSED BY NAME, not accepted. Listed so that CLI::parse() lets them
      // through to refuse_removed_op_flags / refuse_exporter_only_flags, whose
      // messages name the replacement; run_setup.hpp calls both at line 314 and
      // stt/vit_mode.hpp at theirs, all of which are downstream of this parse.
      //
      // --npu-extra-ops is here for the same reason and used to be absent: it was
      // this build's LIVE flag when that table was written, so nothing noticed it
      // being refused rather than accepted. verify_cli_flags.py is what caught
      // it, and it caught it because it checks for a second row with the same
      // name rather than only checking that every accepted flag parses.
      {"--npu-eltwise", 0},  {"--host-gelu", 0}, {"--host-ln", 0},
      {"--host-sm", 0},      {"--npu-extra-ops", 0}, {"--extra-ops", 0},
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

// --serve, read ONCE, for every architecture that has an endpoint.
//
// ONE reader, three callers: the embedding endpoint (run_execute.hpp), the speech
// endpoint (stt_mode.hpp) and the image endpoints (vit_mode.hpp, pose_mode.hpp).
// Three hand-written scans of the same two flags is how `--serve` with no port
// came to mean port 0 in the speech mode: std::atoi("") is 0, and 0 is a legal
// request to bind(), so `npuembeddings <root> --model m.npue --serve` started,
// printed its usual banner and listened on an ephemeral port it never named.
// The default below is 8080, which is what `serve` has always documented.
//
// The grammar is deliberately the loose one `serve` has always accepted:
// `--serve [PORT]`, where PORT is the next argument and only if it starts with a
// digit. `--serve=8080` is not accepted, because nothing has ever documented it.
//
// `from` is the first argv index to read, because the callers do not all pass the
// same slice: the arch modes read from 1 (argv[0] is the executable, argv[1] the
// root) and the embedding path has always started at 2. Neither can be the index
// of `--serve` in either form, so the argument is bookkeeping, not policy.
//
// It lives here rather than in cli.hpp because cli.hpp includes the catalogue, the
// hub and the tokenizer facade, and a header read by three mode headers should not
// drag all of that in behind it.
inline bool read_serve(int argc, char *const *argv, int &port,
                       std::string &bind_addr, int from = 1) {
  port = 8080;
  bind_addr = "127.0.0.1";
  bool found = false;
  for (int i = from; i < argc; ++i) {
    if (std::string(argv[i]) != "--serve") continue;
    found = true;
    if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0])))
      port = std::atoi(argv[++i]);
  }
  for (int i = from; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--bind") bind_addr = argv[i + 1];
  return found;
}

}  // namespace app

#endif  // NPU_CLI_FLAGS_HPP
