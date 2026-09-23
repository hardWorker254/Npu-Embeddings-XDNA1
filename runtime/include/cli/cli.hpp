//===- cli/cli.hpp -----------------------------------------------------*- C++ -*-===//
//
// CLI argument parser: CLI class parses argc/argv into CLIArgs.
// Helper functions (print_usage, print_catalog, warn_if_unpinned,
// maybe_tokenize, resolve_prefix) moved here from cli.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <string>
#include <vector>

#include "common/app_state.hpp"
#include "common/design_selection.hpp"
#include "common/hub.hpp"
#include "common/model_catalog.hpp"
#include "runtime/model.hpp"
#include "tokenizers/tokenizer_facade.hpp"

namespace app {

struct CLIArgs {
    std::string root;
    int bench = 0;
    int threads = 1;
    int pipeline = 0;
    std::string prefix;
    bool allow_truncation = false;
    bool cpu = false;
    std::string artifacts;
    std::string model;
    bool serve = false;
    int port = 8080;
    std::string bind = "127.0.0.1";
    std::string embed_file;
    std::string embed_out;
    std::string add_repo;
    std::string add_sha;
    std::string tokenize_file;
    int tokenize_max_len = 64;
    std::string prepare_model_dir;
    std::string prepare_model_out;
    bool list_models = false;
    bool help = false;
    bool allow_contention = false;
    int max_len = 64;
    bool gemma_host_only = false;
    int64_t tile_k = 64;
    int64_t tile_n = 48;
    std::string source_repo;
    std::string subcommand;
    std::vector<std::string> subcommand_args;
};

class CLI {
public:
    static CLIArgs parse(int argc, char **argv);
};

// Resolves --prefix against the container's task-prefix table.
// Returns the literal text to prepend to every input text before
// tokenization -- "" for a container with no "prompts" table.
// See the full rationale in the original cli.hpp.
inline std::string resolve_prefix(int argc, char **argv) {
    std::string name;
    bool from_cli = false;
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--prefix") { name = argv[i + 1]; from_cli = true; }

    if (g_prompts.empty()) {
        if (from_cli)
            throw std::runtime_error(
                "--prefix '" + name + "' was given, but this model has no task "
                "prefixes -- its container carries no 'prompts' table, so nothing "
                "would be prepended. Refusing rather than returning vectors that "
                "are not what was asked for.");
        return std::string();
    }

    const std::string list = join_names(prompt_names_sorted());

    if (!from_cli)
        throw std::runtime_error(
            "this model has task prefixes and one must be named: pass --prefix "
            "with one of [" + list + "], or --prefix \"\" for no prefix at "
            "all. Refusing to pick one for you -- a wrongly-prefixed embedding is "
            "correctly shaped and correctly normed, so nothing downstream can tell "
            "that the answer is wrong.");

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
        "    --npu-eltwise     run GELU, LayerNorm and softmax on the array\n"
        "                      instead of the host. Requires the sibling\n"
        "                      gelu/, layernorm/ and softmax/ design sets next\n"
        "                      to gemm_rtp (tools/export_gemm_rtp.py\n"
        "                      --npu-eltwise, or tools/export_eltwise.py);\n"
        "                      a missing one is refused by name, never silently\n"
        "                      run on the host. OFF by default: the host path is\n"
        "                      the measured-faster one, so this flag can LOWER\n"
        "                      throughput. --host-ln/--host-sm/--host-gelu\n"
        "                      still force an individual op back onto the host.\n"
        "                      Because it needs gemm_rtp plus the three eltwise\n"
        "                      designs (four hw_contexts), it uses much of the\n"
        "                      device's context budget; the run is refused by\n"
        "                      name if another process owns enough contexts\n"
        "                      (see --allow-contention).\n"
        "    --allow-contention\n"
        "                      proceed even though another process holds NPU\n"
        "                      hw_contexts. Without it, a run that would exceed\n"
        "                      the device's context budget is refused before any\n"
        "                      context is built, and a timed run refuses to\n"
        "                      report a throughput. ANY NUMBER FROM A CONTENDED\n"
        "                      RUN IS NOT AN NPU PERFORMANCE CLAIM.\n"
        "    --dev npu1|npu2   the NPU generation this process runs on. A\n"
        "                      design set built for the other generation is\n"
        "                      refused, not loaded. Defaults to NPU_DEVICE, or\n"
        "                      NPU2=1, or npu1\n"
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
        const bool have_design =
            !pick_artifacts(root, e.hidden, e.ffn, e.gated_ffn, e.qkv_n, "",
                            e.datapath, e.name).empty();
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
                    !encoder_implemented(m.arch)              ? "no encoder"
                    : m.gemm_layout == "host"                 ? "cpu"
                    : pick_artifacts(root, m.hidden, m.ffn, m.gated_ffn,
                                     m.qkv_n, "", "bf16", m.name).empty()
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

inline void warn_if_unpinned(const std::string &name) {
    const npue::hub::CatalogEntry *e = npue::hub::find(name);
    if (!e || !npue::hub::unpinned(*e)) return;
    std::printf("\n  !! '%s' was added WITHOUT a sha256 pin (%s).\n",
                e->name.c_str(), e->repo.c_str());
    std::printf("  !! Its weights are NOT verified against anything.\n");
    std::printf("  !! To pin it:  npuembeddings add %s <sha256>\n\n",
                e->repo.c_str());
}

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
