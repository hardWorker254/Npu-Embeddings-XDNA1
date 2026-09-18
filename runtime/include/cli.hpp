//===- cli.hpp ------------------------------------------------------------*- C++ -*-===//
//
// CLI-facing helpers: the --prefix resolution policy, usage text, the
// model catalogue printer and the unpinned-model warning.
//
// Split out of main.cpp verbatim.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <string>
#include <vector>

#include "app_state.hpp"
#include "design_selection.hpp"
#include "hub.hpp"
#include "model_catalog.hpp"
#include "tokenizer_facade.hpp"
#include "npue.hpp"

namespace app {

// Resolves --prefix against the container's own task-prefix table
// (tasks/0071). Returns the literal text to prepend to every input text
// before tokenization -- "" for a container with no "prompts" table
// (g_prompts.empty()), which is BYTE-IDENTICAL to this runtime's behaviour
// before this task for MiniLM/bge-small/bge-base/bge-large: no prefix
// concept, no banner line, nothing prepended.
//
// A model that DOES have a prompts table always prints which prefix it is
// about to apply, on stderr. An unknown --prefix name lists the container's
// real options rather than guessing.
//
// THE CONTAINER DEFAULT NO LONGER APPLIES (tasks/0118). It used to: an omitted
// --prefix fell through to g_prompt_default, so nomic quietly embedded
// everything as `search_document`. That is a silently-applied default, which
// is how the wrong prefix ships (docs/04-model/README.md:24) -- and unlike a
// wrong number it cannot be seen downstream, because a wrongly-prefixed vector
// is correctly shaped, correctly normed and deterministic. `prompt_default`
// survives in the container as ADVISORY metadata, for harnesses choosing which
// prompt to exercise; nothing in this runtime applies it.
//
// `--prefix ""` is still legal and still means no prefix at all, matching
// check_gemma_prefix() and the endpoint's `"prompt_name": ""`. The distinction
// that now matters is OMITTED vs EMPTY, which is what `from_cli` carries.
inline std::string resolve_prefix(int argc, char **argv) {
  std::string name;
  bool from_cli = false;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--prefix") { name = argv[i + 1]; from_cli = true; }

  if (g_prompts.empty()) {
    // REFUSE rather than ignore. This model has no task-prefix concept, so a
    // --prefix the caller asked for cannot be applied -- and silently
    // proceeding would hand back vectors the caller believes are prefixed.
    // That is the same shape as the status lines this project has had to fix
    // repeatedly: reporting the intention rather than the value. A script
    // sweeping one --prefix across the whole catalogue SHOULD break here,
    // because its BERT results would otherwise differ from what it intended.
    if (from_cli)
      throw std::runtime_error(
          "--prefix '" + name + "' was given, but this model has no task "
          "prefixes -- its container carries no 'prompts' table, so nothing "
          "would be prepended. Refusing rather than returning vectors that "
          "are not what was asked for.");
    return std::string();
  }

  const std::string list = join_names(prompt_names_sorted());

  // REQUIRED, not defaulted. See the note above the function.
  if (!from_cli)
    throw std::runtime_error(
        "this model has task prefixes and one must be named: pass --prefix "
        "with one of [" + list + "], or --prefix \"\" for no prefix at "
        "all. Refusing to pick one for you -- a wrongly-prefixed embedding is "
        "correctly shaped and correctly normed, so nothing downstream can tell "
        "that the answer is wrong.");

  // Named explicitly and empty: no prefix at all. Same convention as
  // check_gemma_prefix() and the endpoint's `"prompt_name": ""`.
  if (name.empty()) {
    std::fprintf(stderr, "  prefix     (none -- --prefix \"\" given)\n");
    return std::string();
  }

  const auto it = g_prompts.find(name);
  if (it == g_prompts.end())
    throw std::runtime_error(
        "--prefix '" + name + "' is not one of this model's task prefixes: "
        "[" + list + "]");

  std::fprintf(stderr, "  prefix     '%s' -> \"%s\"\n", name.c_str(),
              it->second.c_str());
  return it->second;
}

inline void print_usage() {
  std::printf(
      "NpuEmbeddings -- BERT embeddings on the AMD Ryzen AI NPU (XDNA2)\n"
      "\n"
      "  npuembeddings list\n"
      "        every model this build can run, and which are installed\n"
      "\n"
      "  npuembeddings serve <model> [--port N] [--bind ADDR]\n"
      "        OpenAI-shaped /v1/embeddings endpoint. Downloads and verifies\n"
      "        the model first if it is not installed yet.\n"
      "\n"
      "  npuembeddings embed <model> <in.txt> [out.f32]\n"
      "        embed a text file, one text per line\n"
      "\n"
      "  npuembeddings add <org/model> [<sha256>]\n"
      "        teach this installation about a model that is not built in --\n"
      "        typically a finetune of one that is. Reads the repository's\n"
      "        config.json, derives the geometry, checks a design serves it,\n"
      "        and writes models/catalog.json. No weights are downloaded until\n"
      "        the first serve/embed.\n"
      "        WITHOUT a sha256 the weights are NOT verified. That is allowed,\n"
      "        and it is warned about on every single run.\n"
      "\n"
      "  Options for serve/embed:\n"
      "    --port N          listen port (default 8080)\n"
      "    --bind ADDR       interface (default 127.0.0.1, localhost only)\n"
      "    --threads N       host thread budget (default 24 for these)\n"
      "    --pipeline N      concurrent encode lanes (default 2)\n"
      "    --artifacts DIR   override the design set\n"
      "    --root DIR        override where models/ and the design live\n"
      "    --token VALUE     HuggingFace access token for a GATED model\n"
      "                      (falls back to the HF_TOKEN env var if omitted)\n"
      "    --prefix NAME     task prefix to prepend to every input text, for\n"
      "                      `embed` and `--bench` only. REQUIRED for a model\n"
      "                      whose container carries a prompt table (nomic\n"
      "                      wants 'search_document' or 'search_query';\n"
      "                      EmbeddingGemma has 14) -- there is no default,\n"
      "                      because a wrongly-prefixed embedding is\n"
      "                      correctly shaped and correctly normed, so\n"
      "                      nothing downstream can tell it is wrong. Pass\n"
      "                      --prefix \"\" for no prefix at all. An unknown\n"
      "                      NAME refuses and lists the real options, and so\n"
      "                      does passing one to a model with no table (the\n"
      "                      four BERT models). Always printed on stderr.\n"
      "                      `serve` REJECTS this flag: its prompt is chosen\n"
      "                      per request -- POST \"prompt_name\", and GET\n"
      "                      /health lists what the model accepts.\n"
      "    --allow-truncation\n"
      "                      embed the first `seq` tokens of an input that\n"
      "                      is too long, instead of refusing it. OFF by\n"
      "                      default: a truncated text still returns a\n"
      "                      correctly shaped, correctly normed vector, so\n"
      "                      nothing downstream can tell that the answer is\n"
      "                      wrong -- and inputs sharing a preamble truncate\n"
      "                      to IDENTICAL vectors. Without this flag such an\n"
      "                      input is an error naming its real token count;\n"
      "                      with it, a one-line warning on stderr.\n"
      "                      This build runs at the sequence length its\n"
      "                      design was exported for -- see\n"
      "                      tools/export_gemm_rtp.py --seq to build for a\n"
      "                      longer one.\n"
      "\n"
      "  The flag form is unchanged and still works:\n"
      "    npuembeddings <root> --model NAME --artifacts DIR --serve [port]\n"
      "  and carries the probes and benchmarks; see docs/CURRENT_STATUS.md.\n"
      "\n");
}

inline void print_catalog(const std::string &root) {
  const auto installed = discover_models(root);
  auto is_installed = [&](const std::string &n) -> const ModelEntry * {
    for (const auto &m : installed)
      if (m.name == n) return &m;
    return nullptr;
  };

  std::printf("\nModels (root %s)\n\n", root.c_str());
  std::printf("  %-20s %-9s %6s %6s %8s %9s  %s\n", "model", "state",
              "layers", "hidden", "pooling", "size", "notes");

  for (const auto &e : npue::hub::catalog()) {
    const ModelEntry *m = is_installed(e.name);
    // "installed" is not the same as "runnable": the design set for this
    // width has to be present too, and a release ships one width. Saying so
    // here beats a confusing failure at dispatch.
    const bool have_design =
        !pick_artifacts(root, e.hidden, e.ffn, e.gated_ffn, e.qkv_n, "",
                        e.datapath).empty();
    // "cpu" is now a property of the CONTAINER, not of the architecture
    // (tasks/0074). arch=1 ran entirely on the host until this release and the
    // row said so unconditionally; it now has an NPU design and a pre-tiled
    // container, and an unconditional "cpu" would be the same kind of lie the
    // unconditional "ready" was before tasks/0069 -- just pointing the other
    // way. A host-only Gemma container (packed with --gemma-host-only, which
    // is still the correctness control) genuinely is "cpu", and says so.
    // "no encoder" is NOT the same as "no design": nomic-embed-text-v1.5 has a
    // matching design set and a packed container, and would still return
    // embeddings for the wrong model if run through the BERT encoder. Saying
    // "ready" because the design matches would be exactly the fail-open
    // set_model_shape() now refuses at dispatch. tasks/0069.
    const char *state = !m                              ? "available"
                        : !encoder_implemented(m->arch) ? "no encoder"
                        : m->gemm_layout == "host"      ? "cpu"
                        : have_design                   ? "ready"
                                                        : "no design";
    char size[32];
    if (m)
      std::snprintf(size, sizeof size, "%.0f MB", m->mb);
    else
      std::snprintf(size, sizeof size, "%.0f MB dl", e.download_mb);
    std::printf("  %-20s %-9s %6lld %6lld %8s %9s  %s\n", e.name.c_str(),
                state, (long long)e.layers, (long long)e.hidden,
                e.pooling.c_str(), size, e.note.c_str());
  }

  // Anything packed locally that the catalogue does not know about. It is
  // perfectly valid -- `--prepare-model` builds one from any BERT checkpoint
  // -- and hiding it would make the table a lie about what `serve` accepts.
  bool header = false;
  for (const auto &m : installed) {
    if (npue::hub::find(m.name)) continue;
    if (!header) {
      std::printf("\n  Locally packed (not in the catalogue):\n");
      header = true;
    }
    if (!m.error.empty()) {
      std::printf("  %-20s UNREADABLE: %s\n", m.name.c_str(),
                  m.error.c_str());
      continue;
    }
    std::printf("  %-20s %-9s %6lld %6lld %8s %6.0f MB  %s\n", m.name.c_str(),
                // Same three-way split as the catalogue table above: a
                // host-only arch is neither "ready" nor "no design".
                !encoder_implemented(m.arch)              ? "no encoder"
                : m.gemm_layout == "host"                 ? "cpu"
                // Not in the catalogue, so no adoption decision was ever
                // made for it -- require plain bf16, the safe default
                // (CatalogEntry::datapath's own default, tasks/0104).
                : pick_artifacts(root, m.hidden, m.ffn, m.gated_ffn,
                                 m.qkv_n, "", "bf16").empty()
                    ? "no design" : "ready",
                (long long)m.layers, (long long)m.hidden, m.pooling.c_str(),
                m.mb, m.repo.c_str());
  }

  std::printf(
      "\n  ready      installed, with a matching NPU design -- `serve` runs it\n"
      "  available  not downloaded yet -- `serve` fetches and verifies it\n"
      "  cpu        installed, but this CONTAINER holds row-major host-side\n"
      "             GEMM operands, so it runs entirely on the CPU. Repack it\n"
      "             (the pre-tiled NPU layout is the default) to use the array\n"
      "  no design  installed, but no design set for this geometry is present\n"
      "  no encoder installed, and a design may match, but this build has no\n"
      "             forward pass for the architecture -- it will refuse rather\n"
      "             than return embeddings for the wrong model\n"
      "\n  npuembeddings serve <model>\n\n");
}

// arch=1 (EmbeddingGemma family). TWO paths, chosen from the CONTAINER, never
// from a model name or a flag default:
//
//   * `gemm_layout == "pretiled_bf16"` + a matching design set (tasks/0074)
//     runs GemmaNpuEncoder -- four GEMMs per layer on the array, 97.7% of the
//     model's MACs, with RMSNorm/RoPE/GeGLU/attention on the host.
//   * anything else runs npue::GemmaEncoder, the host-only reference
//     (tasks/0064, 1-cos 5.496e-13 against reference/encoder_gemma.py). It is
//     kept, not retired: it is the control this file is checked against, and
//     `--cpu` selects it deliberately so the two can be compared on one input.
//
// This is a SEPARATE code path from Encoder::run(), not a branch inside it --
// see GemmaNpuEncoder's own header comment for why. The BERT path is untouched
// and this function is reachable only when the .npue's own config["arch"] says
// gemma3_mqa_rope_geglu.
// EVERY TIME, not once at install time (user decision, 2026-08-22). A warning
// you saw last month is not a warning you see today, and the whole point of
// allowing an unpinned `add` is that the person running it knows the weights
// were never verified. Printed for any model whose catalogue row carries no
// sha256 -- which only `add` can produce.
inline void warn_if_unpinned(const std::string &name) {
  const npue::hub::CatalogEntry *e = npue::hub::find(name);
  if (!e || !npue::hub::unpinned(*e)) return;
  std::printf("\n  !! '%s' was added WITHOUT a sha256 pin (%s).\n",
              e->name.c_str(), e->repo.c_str());
  std::printf("  !! Its weights are NOT verified against anything.\n");
  std::printf("  !! To pin it:  npuembeddings add %s <sha256>\n\n",
              e->repo.c_str());
}


// --tokenize <file> [max_len]: one text per line in, one line of token ids
// out. No NPU, no model encode -- the mode tools/verify_tokenizer.py drives
// to compare against HuggingFace token for token. Returns true when handled.
inline bool maybe_tokenize(const std::string &root, int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--tokenize") {
    const std::string in_path = argv[i + 1];
    int max_len = 64;
    if (i + 2 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 2][0])))
      max_len = std::atoi(argv[i + 2]);
    const std::string vpath = resolve_model_path(root, argc, argv);
    npue::File vm(vpath);
    auto tok = load_tokenizer(vm, vpath);
    std::ifstream in(in_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + in_path);
    std::string line;
    // Deliberately NOT an error here, and deliberately not on stdout either.
    // This mode exists for tools/verify_tokenizer.py to diff against
    // HuggingFace token for token, and HuggingFace truncates too -- refusing
    // would make the two incomparable at exactly the lengths worth comparing,
    // and an extra stdout line would desynchronise the diff. So: the same ids
    // as before on stdout, and a count on stderr, so a truncating corpus
    // cannot be mistaken for a clean one.
    size_t n_lines = 0, n_cut = 0;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      const auto e = tok.encode(line, max_len);
      ++n_lines;
      if (e.truncated) ++n_cut;
      for (size_t k = 0; k < e.input_ids.size(); ++k)
        std::printf("%s%d", k ? " " : "", e.input_ids[k]);
      std::printf("\n");
    }
    if (n_cut)
      std::fprintf(stderr,
                   "  NOTE  %zu of %zu lines exceeded max_len=%d and were "
                   "truncated. Token-for-token agreement below that cut says "
                   "nothing about the text past it.\n",
                   n_cut, n_lines, max_len);
    return true;
  }
  return false;
}

}  // namespace app
