//===- cli_add.hpp -------------------------------------------*- C++ -*-===//
//
// `--prepare-model <dir> [out.npue]` (and the `add` subcommand's flag form):
// build a model container from an upstream checkpoint. The checkpoint's OWN
// config.json picks the packer (BERT / gemma3 / nomic / gte), never the
// directory name.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "http.hpp"
#include "npue.hpp"
#include "npue_pack.hpp"

namespace app {

  // --prepare-model <dir> [out.npue]: build the model container from an
  // upstream checkpoint. `dir` holds model.safetensors, vocab.txt and
  // config.json as downloaded from HuggingFace.
  //
  // The release ships no weights: they belong to
  // sentence-transformers/all-MiniLM-L6-v2, and fetching them from the
  // canonical source with a checksum beats trusting a blob in a zip. This is
  // what keeps that a two-step setup rather than a Python install.

// Returns true when the flag was present (the container has been written and
// the caller must exit 0); false to fall through to the encode paths.
inline bool maybe_prepare_model(int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--prepare-model") {
    const std::string dir = argv[i + 1];
    // The container is named after the checkpoint directory, not after
    // MiniLM. This was a literal until a second model made it visible.
    std::string out =
        dir + "/" + std::filesystem::path(dir).filename().string() + ".npue";
    if (i + 2 < argc && argv[i + 2][0] != '-') out = argv[i + 2];

    // arch=1 (EmbeddingGemma / Gemma3 family): a completely different tensor
    // shape and container, routed to its own packer rather than threaded
    // through the BERT logic below -- tools/pack_npue.py's main() makes the
    // same decision the same way. Detected from the checkpoint's OWN
    // config.json (model_type), never assumed from --prepare-model's
    // directory name. tasks/0065-m12-embeddinggemma-cpp-packer.
    {
      std::ifstream cfg_probe(dir + "/config.json");
      if (cfg_probe) {
        std::stringstream cs;
        cs << cfg_probe.rdbuf();
        const std::string model_type =
            npue::http::json_field_string(cs.str(), "model_type", "");
        if (model_type == "gemma3_text") {
          std::string source_repo;
          for (int k = 1; k < argc - 1; ++k)
            if (std::string(argv[k]) == "--source-repo")
              source_repo = argv[k + 1];
          if (source_repo.empty()) {
            std::ifstream cf(dir + "/CHECKPOINT.json");
            if (!cf)
              throw std::runtime_error(
                  "no CHECKPOINT.json under " + dir + " and no --source-repo "
                  "given -- refusing to guess which repository these "
                  "weights came from");
            std::stringstream ckcs;
            ckcs << cf.rdbuf();
            source_repo =
                npue::http::json_field_string(ckcs.str(), "repo_id", "");
            if (source_repo.empty())
              throw std::runtime_error(dir + "/CHECKPOINT.json has no "
                                       "repo_id");
          }
          std::printf("  source     %s\n", source_repo.c_str());
          std::printf("NpuEmbeddings -- preparing %s\n", out.c_str());
          // Same two knobs the BERT path takes, plus the host-only escape
          // (tasks/0074). Defaults are the production geometry, so a cold
          // clone with only --token self-produces a container the ARRAY can
          // run -- which is the point: before this it self-produced a
          // host-only container and quietly ran at 0.2 seq/s.
          int64_t gtk = 64, gtn = 48;
          bool ghost = false;
          for (int k = 1; k < argc; ++k) {
            const std::string a = argv[k];
            if (a == "--gemma-host-only") ghost = true;
            if (k + 1 < argc && a == "--tile-k") gtk = std::atoll(argv[k + 1]);
            if (k + 1 < argc && a == "--tile-n") gtn = std::atoll(argv[k + 1]);
          }
          npue::prepare_model_gemma(dir, out, source_repo,
                                    [](const std::string &s) {
                                      std::printf("%s\n", s.c_str());
                                    },
                                    gtk, gtn, ghost);
          std::printf("  wrote %s\n", out.c_str());
          return true;
        }
      }
    }

    // Tile size is a PROPERTY OF THE MODEL, not a constant. The design
    // asserts N % (tile_n * n_cols) == 0, and bge-large's N in
    // {1024, 3072, 4096} makes 48 illegal -- the legal set there is
    // {8, 16, 32, 64} and 64 does not fit L1 (65,536 B against the 63 KB
    // budget), so it must be 32. Both packers now take it and neither
    // freezes the resulting hash.
    int64_t tile_k = 64, tile_n = 48;
    for (int k = 1; k < argc - 1; ++k) {
      if (std::string(argv[k]) == "--tile-n") tile_n = std::atoi(argv[k + 1]);
      if (std::string(argv[k]) == "--tile-k") tile_k = std::atoi(argv[k + 1]);
    }
    const npue::Layout lay = npue::gemm_b_layout(tile_k, tile_n);
    const std::string layout = lay.json;
    const std::string layout_hash = lay.hash;
    std::printf("  layout     tile (%lld, %lld), hash %s...\n",
                (long long)tile_k, (long long)tile_n,
                layout_hash.substr(0, 16).c_str());

    // Pooling comes from the checkpoint's own 1_Pooling/config.json, the
    // same source tools/pack_npue.py reads. Both packers must agree or
    // verify_pack_parity fails, which is the point of having the gate.
    std::string pooling;
    {
      std::ifstream pf(dir + "/1_Pooling/config.json");
      if (!pf)
        throw std::runtime_error(
            "no 1_Pooling/config.json under " + dir + " -- cannot tell "
            "whether this checkpoint pools by mean or by CLS");
      std::stringstream ps;
      ps << pf.rdbuf();
      const std::string pj = ps.str();
      auto flag = [&](const char *k) {
        const size_t i = pj.find(k);
        if (i == std::string::npos) return false;
        const size_t c = pj.find(':', i);
        return pj.compare(pj.find_first_not_of(" \t", c + 1), 4, "true") == 0;
      };
      const bool cls = flag("pooling_mode_cls_token");
      const bool mean = flag("pooling_mode_mean_tokens");
      if (cls == mean)
        throw std::runtime_error(
            "1_Pooling/config.json asks for neither or both of cls and mean; "
            "this runtime implements exactly those two");
      pooling = cls ? "cls" : "mean";
      std::printf("  pooling    %s (from 1_Pooling/config.json)\n",
                  pooling.c_str());
    }

    // Which repository these weights came from. tools/pack_npue.py reads
    // CHECKPOINT.json for this and so must we, or the two packers disagree.
    // A container that misattributes its own weights is a licensing
    // statement, so an unknown repo REFUSES rather than guessing.
    std::string source_repo;
    for (int k = 1; k < argc - 1; ++k)
      if (std::string(argv[k]) == "--source-repo") source_repo = argv[k + 1];
    if (source_repo.empty()) {
      std::ifstream cf(dir + "/CHECKPOINT.json");
      if (!cf)
        throw std::runtime_error(
            "no CHECKPOINT.json under " + dir + " and no --source-repo given "
            "-- refusing to guess which repository these weights came from");
      std::stringstream cs;
      cs << cf.rdbuf();
      source_repo = npue::http::json_field_string(cs.str(), "repo_id", "");
      if (source_repo.empty())
        throw std::runtime_error(dir + "/CHECKPOINT.json has no repo_id");
    }
    std::printf("  source     %s\n", source_repo.c_str());

    // arch=2 (nomic-embed-text-v1.5): RoPE + gated SwiGLU rather than
    // BERT's absolute-position + GELU -- routed to its own packer, mirroring
    // tools/pack_npue.py's `model_type == "nomic_bert"` branch in main().
    // Detected the same way the gemma3_text branch above is (config.json's
    // OWN model_type), never assumed from --prepare-model's directory name.
    // Placed HERE, after tile_k/tile_n/layout/pooling/source_repo are
    // already resolved above, because nomic shares every one of those
    // resolutions with the BERT path unchanged -- only the packer differs.
    // tasks/0071.
    {
      std::ifstream cfg_probe2(dir + "/config.json");
      if (cfg_probe2) {
        std::stringstream cs2;
        cs2 << cfg_probe2.rdbuf();
        const std::string model_type =
            npue::http::json_field_string(cs2.str(), "model_type", "");
        if (model_type == "nomic_bert") {
          std::printf("NpuEmbeddings -- preparing %s (arch=nomic_bert_rope_swiglu)\n",
                      out.c_str());
          npue::prepare_model_nomic(dir, pooling, source_repo, out, layout,
                                    layout_hash, tile_k, tile_n, 256,
                                    [](const std::string &s) {
                                      std::printf("%s\n", s.c_str());
                                    });
          std::printf("  wrote %s\n", out.c_str());
          return true;
        }
        // arch=3 (gte-multilingual-base): model_type "new", same dispatch
        // rule as the nomic branch above (the checkpoint's OWN config.json,
        // never the directory name). max_seq 64 matches the Python-packed
        // container this mirror is held byte-identical to (tasks/0135
        // packed --max-seq 64; under RoPE the position table is zeros, so
        // max_seq only caps request length). tasks/0138.
        if (model_type == "new") {
          std::printf("NpuEmbeddings -- preparing %s (arch=gte_new_rope_geglu)\n",
                      out.c_str());
          npue::prepare_model_gte(dir, pooling, source_repo, out, layout,
                                  layout_hash, tile_k, tile_n, 64,
                                  [](const std::string &s) {
                                    std::printf("%s\n", s.c_str());
                                  });
          std::printf("  wrote %s\n", out.c_str());
          return true;
        }
      }
    }

    std::printf("NpuEmbeddings -- preparing %s\n", out.c_str());
    npue::prepare_model(dir + "/model.safetensors", dir + "/vocab.txt",
                        dir + "/config.json", pooling, source_repo, out,
                        "", layout, layout_hash,
                        tile_k, tile_n, 256,
                        [](const std::string &s) {
                          std::printf("%s\n", s.c_str());
                        });
    std::printf("  wrote %s\n", out.c_str());
    return true;
  }
  return false;
}

}  // namespace app
