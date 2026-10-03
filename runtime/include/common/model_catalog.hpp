//===- model_catalog.hpp --------------------------------------------------*- C++ -*-===//
//
// The installed-model catalogue: everything shown about a model is read
// from its .npue container -- there is no list of models in this binary.
//
// Split out of main.cpp verbatim.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "common/app_state.hpp"
#include "runtime/model.hpp"

namespace app {

// ---------------------------------------------------------------------------
// Which model to run.
//
// Four sites used to name all-MiniLM-L6-v2 as a literal. The set of installed
// models is now whatever is in models/*.npue, and everything shown about them
// is read from the containers -- there is no list of models in this binary.

struct ModelEntry {
  std::string path, name, repo, pooling, arch, error, gemm_layout;
  int64_t layers = 0, hidden = 0, heads = 0, head_dim = 0, ffn = 0, seq = 0;
  // 0 = the container did not say, i.e. every BERT and nomic container, for
  // which the fused qkv really is 3*hidden. An MQA/GQA container states it
  // (tasks/0074) and it is NOT derivable from anything else here.
  int64_t qkv_n = 0;
  bool gated_ffn = false;
  double mb = 0;
  // "embed" | "stt" | "cls" | "pose" -- which SUBCOMMAND runs this container, or
  // "" for a container that predates the key and is an embedder (every arch=0/2
  // file ever written).
  //
  // It is a property of the CONTAINER, not of a list in this binary: the pose
  // container says `"kind": "pose"`. Reading it is what stops `list` from
  // printing a pose container as UNREADABLE -- a file that opens perfectly and
  // is listed as broken, which is what happened to whisper containers before
  // `pooling` was made optional and would happen again to every architecture
  // that is not an embedder.
  std::string kind;
};


inline std::vector<ModelEntry> discover_models(const std::string &root) {
  namespace fs = std::filesystem;
  std::vector<ModelEntry> v;
  std::error_code ec;
  const fs::path dir = fs::path(root) / "models";
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    if (it->path().extension() != ".npue") continue;
    ModelEntry m;
    m.path = it->path().string();
    m.name = it->path().stem().string();
    // A container that will not open is LISTED with its error rather than
    // skipped: a model silently missing from the table is a worse failure
    // than one that is visibly broken.
    try {
      npue::File f(m.path);
      // `arch` and `kind` FIRST, and both optional. Everything below them is an
      // EMBEDDER's field: a pose container has no source_repo (it is packed from
      // a local ONNX, not fetched from HuggingFace) and no num_layers, hidden or
      // max_seq_len, and asking for the first one is what printed pose
      // containers as UNREADABLE -- a container that opens fine, listed as
      // broken, and reachable only by spelling out its path. Same shape of bug
      // as `pooling` on arch=4, one architecture further on.
      try {
        m.arch = f.config_string("arch");
      } catch (const std::exception &) {
      }
      try {
        m.kind = f.config_string("kind");
      } catch (const std::exception &) {
      }
      if (m.kind.empty())
        m.kind = (m.arch == "yolov8_pose_c2f_silu_dfl") ? "pose"
                : (m.arch == "whisper_encoder_decoder" ||
                   m.arch == "whisper_decoder")            ? "stt"
                                                          : "embed";
      try {
        m.repo = f.config_string("source_repo");
      } catch (const std::exception &) {
        m.repo = "n/a";   // packed from a local file, not fetched
      }
      // `pooling` is an embedder's field. An arch=4 container answers "text
      // for audio" and has no pooling mode at all, so asking for the key was
      // what printed whisper containers as UNREADABLE -- a model that opens
      // fine, listed as broken, and unreachable by name.
      m.pooling = "n/a";
      try {
        m.pooling = f.config_string("pooling");
      } catch (const std::exception &) {
      }
      // And so are the geometry keys. A pose container states its own -- the
      // input size, the keypoint count, the strides -- under names this table
      // has no column for, and asking for `num_layers` is the same mistake one
      // line further down. They stay 0, and 0 is already how the table prints
      // "not stated".
      if (m.kind == "embed") {
        m.layers = f.config_int("num_layers");
        m.hidden = f.config_int("hidden");
        m.heads = f.config_int("num_heads");
        m.head_dim = f.config_int("head_dim");
        m.ffn = f.config_int("intermediate");
        m.gated_ffn = config_flag(f, "gated_ffn", false);
        try {
          m.qkv_n = f.config_int("qkv_n");
        } catch (const std::exception &) {
          m.qkv_n = 0;   // predates the key; 3*hidden is right for those
        }
        // Whether this container's GEMM operands are tiled bf16 for the array or
        // plain row-major F32 for a host forward pass (tasks/0074). A container
        // that predates the key is arch=1 host-only if it is Gemma, and tiled
        // otherwise -- every arch=0/2 container ever written is tiled.
        try {
          m.gemm_layout = f.config_string("gemm_layout");
        } catch (const std::exception &) {
          m.gemm_layout = (m.arch == "gemma3_mqa_rope_geglu") ? "host"
                                                                : "pretiled_bf16";
        }
        m.seq = f.config_int("max_seq_len");
      }
      m.mb = f.data_length() / 1e6;
    } catch (const std::exception &e) {
      m.error = e.what();
    }
    v.push_back(std::move(m));
  }
  std::sort(v.begin(), v.end(),
            [](const ModelEntry &a, const ModelEntry &b) {
              return a.name < b.name;
            });
  return v;
}

inline void print_model_table(const std::vector<ModelEntry> &v) {
  std::printf("\nInstalled models (from %s):\n\n",
              "models/*.npue");
  std::printf("  %-24s %6s %7s %7s %6s %8s  %s\n", "--model", "layers",
              "hidden", "pooling", "MB", "max seq", "source");
  for (const auto &m : v) {
    if (!m.error.empty()) {
      std::printf("  %-24s  UNREADABLE: %s\n", m.name.c_str(),
                  m.error.c_str());
      continue;
    }
    // A container that is not an embedder gets its KIND in the pooling column
    // and dashes where the geometry would be, and then a line saying which
    // subcommand runs it. The columns are empty rather than zero-filled because
    // a `0 layers` row reads as a claim -- a BERT-family container with no
    // layers is a broken container -- and here it is a number the architecture
    // does not have.
    if (m.kind != "embed") {
      std::printf("  %-24s %6s %7s %7s %6.0f %8s  %s\n", m.name.c_str(), "-",
                  "-", m.kind.c_str(), m.mb, "-", m.repo.c_str());
      continue;
    }
    std::printf("  %-24s %6lld %7lld %7s %6.0f %8lld  %s\n", m.name.c_str(),
                (long long)m.layers, (long long)m.hidden, m.pooling.c_str(),
                m.mb, (long long)m.seq, m.repo.c_str());
  }
  // And the command, for the same reason the catalogue table carries a notes
  // column: a row labelled `pose` with no subcommand next to it invites
  // `embed yolov8n-pose`, which fails with a message about pooling rather than
  // about the wrong mode.
  for (const auto &m : v) {
    if (!m.error.empty() || m.kind == "embed") continue;
    const char *how = m.kind == "pose"  ? "npuembeddings pose"
                      : m.kind == "cls"  ? "npuembeddings classify"
                      : m.kind == "stt"  ? "npuembeddings transcribe"
                                         : nullptr;
    if (how)
      std::printf("  %-24s   a %s container: `%s <name> <input>` is how it "
                  "runs; `embed` is not.\n",
                  m.name.c_str(), m.kind.c_str(), how);
  }
  std::printf("\n  Wider and deeper models score better and run slower; the\n"
              "  measured throughput and MTEB for each are in docs/.\n\n");
}

// Does this argument NAME A FILE rather than a model? A ".npue" suffix, or any
// path separator: model names in this tree are flat (models/<name>.npue, one
// level, no dots in the stem), so anything with a separator or a container
// suffix is a path or a mistake -- and mistaking it for a name is what made
// `embed ./builds/mine.npue in.txt` print the model table instead of saying the
// file is not there.
inline bool looks_like_container_path(const std::string &arg) {
  if (arg.empty()) return false;
  if (arg.size() > 5 && arg.compare(arg.size() - 5, 5, ".npue") == 0)
    return true;
  return arg.find('/') != std::string::npos ||
         arg.find('\\') != std::string::npos;
}

// ...and is that file actually there?
inline bool is_container_path(const std::string &arg) {
  if (!looks_like_container_path(arg)) return false;
  std::error_code ec;
  return std::filesystem::is_regular_file(arg, ec) && !ec;
}

// ONE message for "you pointed at a file that is not there", from the flag form
// and from the subcommands alike: two spellings of the same refusal is how they
// drift apart.
[[noreturn]] inline void throw_missing_container(const std::string &want,
                                                 const std::string &root) {
  throw std::runtime_error(
      "no such container: " + want + "\n"
      "  A path is used exactly as given, so the file has to exist where you\n"
      "  said it does. A model NAME has no separator and no .npue suffix, and\n"
      "  is looked up under " + root + "/models/<name>.npue");
}

// Resolve a NAME (or a path to a .npue) to a container under `root`. A path
// wins over a name of the same spelling, because a path names a FILE and a
// name names a row in <root>/models/ -- and they need not be the same file:
// `--model ./builds/mine.npue` must not be re-resolved against models/.
inline std::string resolve_model_path(const std::string &root,
                                      const std::string &want) {
  if (looks_like_container_path(want)) {
    if (is_container_path(want)) return want;
    throw_missing_container(want, root);
  }

  const auto models = discover_models(root);
  if (models.empty())
    throw std::runtime_error(
        "no models/*.npue under " + root + " -- build one with "
        "`npuembeddings --prepare-model <checkpoint-dir>`; see BUILD.md");

  if (want.empty()) {
    // AMBIGUITY is what makes --model required. One installed model is not
    // ambiguous, and demanding the flag would only make the user type the
    // single possible answer. Two are, and choosing for them is how this
    // project's fail-open bugs have always looked.
    if (models.size() == 1) return models[0].path;
    print_model_table(models);
    throw std::runtime_error(
        "several models are installed; say which with --model <name>");
  }

  for (const auto &m : models)
    if (m.name == want) {
      if (!m.error.empty())
        throw std::runtime_error("model " + want + " will not open: " +
                                 m.error);
      return m.path;
    }
  print_model_table(models);
  throw std::runtime_error("no model named '" + want + "' is installed");
}

// Resolve --model to a container. Accepts a name as printed in the table or a
// path to a .npue directly.
inline std::string resolve_model_path(const std::string &root, int argc,
                               char **argv) {
  std::string want;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--model") want = argv[i + 1];
  return resolve_model_path(root, want);
}

}  // namespace app
