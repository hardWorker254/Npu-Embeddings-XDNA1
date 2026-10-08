//===- family.hpp -----------------------------------------------------------*- C++ -*-===//
//
// WHICH BINARY THIS PROCESS IS, and which subcommand belongs to which one.
//
// One source tree builds three executables that share every line of runtime
// code and differ in exactly one thing: who they refuse. The split is what a
// reader meets by typing -- `npuaudio transcribe ...` and `npuembeddings
// transcribe ...` cannot both be the answer to "how do I transcribe?", and a
// binary that quietly ran a mode it is not named for would leave the naming to
// be learned from the source.
//
// THE FAMILY IS DECIDED TWICE, and both answers are needed:
//
//   * from the SPELLING of argv[1] -- `transcribe` is Audio whatever the
//     arguments say. Checked in SubcommandDispatcher::dispatch, before any
//     handler runs, so refusing costs nothing and no container is opened.
//   * from the ARCH of the container -- arch `whisper_encdec_gelu` is Audio
//     whatever the verb was. Checked where a path has just been resolved
//     (refuse_other_family), because the flag form has NO verb to check:
//     `npuembeddings --model whisper-base.npue --transcribe a.wav` would
//     otherwise reach a mode this binary is not named for.
//
// The second check exists for `serve` too, which is the one verb all three
// binaries have: the endpoint is picked by the container's arch, so a binary
// serving a container of another family is the same mismatch under a
// different spelling.
//
// Both refusals quote the ONE roster below, so a reader refused by either
// route gets the same three lines and the same program name to type next.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <stdexcept>
#include <string>

#include "common/app_state.hpp"  // is_stt_arch / is_vit_arch / is_pose_arch
#include "runtime/model.hpp"     // npue::File: the container's own arch

namespace app {

enum class Family { Embed, Audio, Image };

// THE TWO FACTS A MAIN() SETS, and nothing else in the tree writes them.
// `inline` (C++17) so the three binaries get exactly one object each across
// every translation unit without a .cpp whose only job is to define them.
//
// The default is npuembeddings because that is the binary this tree built
// before there were three, and because an unset global should name the
// behaviour a reader already trusts rather than an empty string.
inline Family g_family = Family::Embed;
inline const char *g_bin = "npuembeddings";

inline const char *bin_of(Family who) {
  switch (who) {
    case Family::Audio: return "npuaudio";
    case Family::Image: return "npuimage";
    case Family::Embed: break;
  }
  return "npuembeddings";
}

// THE VERBS, one place. `serve` answers in all three -- see the header. A verb
// added to register_default_subcommands must be added here too, and the two
// tables disagreeing is visible at a glance rather than at run time: dispatch
// refuses what this does not list, so an unlisted verb is simply nobody's.
inline bool owns(Family who, const std::string &sub) {
  switch (who) {
    case Family::Audio:
      return sub == "transcribe" || sub == "serve";
    case Family::Image:
      return sub == "classify" || sub == "pose" || sub == "hands" ||
             sub == "mppose" || sub == "serve";
    case Family::Embed:
      break;
  }
  return sub == "embed" || sub == "list" || sub == "add" ||
         sub == "tokenize" || sub == "serve";
}

// Which binary a verb belongs to, for the refusal to name. `serve` never
// reaches this: it is owned by every family, so dispatch never refuses it.
inline Family owner_of(const std::string &sub) {
  if (sub == "transcribe") return Family::Audio;
  if (sub == "classify" || sub == "pose" || sub == "hands" ||
      sub == "mppose")
    return Family::Image;
  return Family::Embed;
}

// The verbs as one line of prose -- for `--help`'s "this is X: ..." and for
// any place that has to say what a binary takes.
inline const char *verbs_of(Family who) {
  switch (who) {
    case Family::Audio: return "transcribe, serve";
    case Family::Image: return "classify, pose, hands, mppose, serve";
    case Family::Embed: break;
  }
  return "embed, list, add, tokenize, serve";
}

// The roster, as a fixed string so the help text and BOTH refusals print the
// same three lines. A refusal that names a program the help text does not
// list is the failure this exists to make impossible.
inline const char *roster() {
  return "    npuembeddings   embed, list, add, tokenize, serve\n"
         "    npuaudio        transcribe, serve\n"
         "    npuimage        classify, pose, hands, mppose, serve";
}

// A container's ARCH names its family. Embed is the default and covers every
// BERT/nomic/Gemma arch, which is also why an arch this build has no mode for
// stays in npuembeddings: an unknown arch is at worst text, and the loading
// path below the check still reports what it found.
inline Family family_of_arch(const std::string &arch) {
  if (is_stt_arch(arch)) return Family::Audio;
  if (is_vit_arch(arch) || is_pose_arch(arch)) return Family::Image;
  return Family::Embed;
}

// THE CONTAINER REFUSAL, thrown rather than printed, so it takes the one
// error path every other refusal takes (`error: ...` on stderr, exit 2).
inline void refuse_other_family(const std::string &path) {
  npue::File probe(path);
  std::string arch;
  try {
    arch = probe.config_string("arch");
  } catch (const std::exception &) {
    return;  // no arch: a pre-arch container; the loader names it below
  }
  const Family owner = family_of_arch(arch);
  if (owner == g_family) return;
  throw std::runtime_error(
      path + " is `" + arch + "`, which " + bin_of(owner) +
      " runs, and this is " + std::string(g_bin) + ".\n" + roster());
}

// THE SUBCOMMAND REFUSAL, as a message rather than a throw, because dispatch
// answers with an exit code instead of an unwind.
inline std::string refuse_subcommand(const std::string &sub) {
  return "`" + sub + "` runs in " + std::string(bin_of(owner_of(sub))) +
         ", and this is " + g_bin + ".\n" + roster();
}

// The one worked example in `--help`, per family: it has to be a command the
// reader of THIS binary can type, or the example teaches the wrong verb.
inline const char *usage_example() {
  switch (g_family) {
    case Family::Audio:
      return "npuaudio transcribe models/whisper-base.npue audio.wav";
    case Family::Image:
      return "npuimage classify models/vit-base-patch16-224.npue image.png";
    case Family::Embed:
      break;
  }
  return "npuembeddings embed models/all-MiniLM-L6-v2.npue in.txt";
}

// The body of main(), shared by all three binaries. Each main() sets the two
// globals above and calls this; from there down it is one codebase.
int entry(int argc, char **argv);

}  // namespace app
