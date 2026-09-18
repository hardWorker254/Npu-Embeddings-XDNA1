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
#include <string>
#include <vector>

#include "app_state.hpp"
#include "npue.hpp"

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
      m.repo = f.config_string("source_repo");
      m.pooling = f.config_string("pooling");
      m.arch = f.config_string("arch");
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
    std::printf("  %-24s %6lld %7lld %7s %6.0f %8lld  %s\n", m.name.c_str(),
                (long long)m.layers, (long long)m.hidden, m.pooling.c_str(),
                m.mb, (long long)m.seq, m.repo.c_str());
  }
  std::printf("\n  Wider and deeper models score better and run slower; the\n"
              "  measured throughput and MTEB for each are in docs/.\n\n");
}

// Resolve --model to a container. Accepts a name as printed in the table or a
// path to a .npue directly.
inline std::string resolve_model_path(const std::string &root, int argc,
                               char **argv) {
  std::string want;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--model") want = argv[i + 1];

  if (!want.empty() && want.size() > 5 &&
      want.compare(want.size() - 5, 5, ".npue") == 0 &&
      std::ifstream(want).good())
    return want;

  const auto models = discover_models(root);
  if (models.empty())
    throw std::runtime_error(
        "no models/*.npue under " + root + " -- build one with "
        "`npuembed --prepare-model <checkpoint-dir>`; see BUILD.md");

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

}  // namespace app
