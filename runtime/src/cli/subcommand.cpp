//===- subcommand.cpp ------------------------------------------------*- C++ -*-===//
//
// Subcommand dispatch implementation. Routes `list`, `serve`,
// `embed`, `add`, `tokenize` to their registered handlers.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "cli/subcommand.hpp"
#include "cli/cli.hpp"
#include "common/design_selection.hpp"
#include "common/hub.hpp"
#include "common/model_catalog.hpp"
#include "embed_models/model_loader.hpp"
#include "runtime/runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace app {

namespace {

int run_list(int, char **argv) {
    print_catalog(default_root(argv[0]));
    return 0;
}

// Flags the subcommands forward verbatim to the flag-form path, so a
// subcommand and the equivalent `--model ...` invocation cannot drift. The
// download-on-demand behavior comes from hub::ensure_model(), which is the
// whole point of `serve`/`embed`: the checksum comparison that used to live in
// a batch file now happens inside the executable.
namespace {

std::string ensure(const std::string &root, const std::string &name,
                   const std::string &token) {
    return npue::hub::ensure_model(
        root, name,
        [](const std::string &s) {
            std::printf("%s\n", s.c_str());
            std::fflush(stdout);
        },
        token);
}

void forward_common(const char *const *argv, int argc,
                    std::vector<std::string> &store) {
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        // Every no-argument policy flag the runtime reads off raw argv must be
        // listed here. A flag missing from this whitelist is not rejected -- it
        // is silently dropped before Runtime::run sees it, so
        // `serve ... --npu-eltwise` would report the array while running the
        // host path.
        // That fail-open is what the flag exists to prevent (SUBTASKS.md
        // subtask 4), so the list is the runtime's flag set, not an arbitrary
        // subset.
        if (a == "--cpu" || a == "--allow-truncation" || a == "--npu-eltwise" ||
            a == "--host-ln" || a == "--host-sm" || a == "--host-gelu" ||
            a == "--sim-c-bf16" || a == "--no-fuse-ffn" ||
            a == "--allow-contention")
            store.push_back(a);
        else if ((a == "--threads" || a == "--pipeline" || a == "--prefix" ||
                  a == "--artifacts" || a == "--dev" || a == "--bo-mode") &&
                 i + 1 < argc) {
            store.push_back(a);
            store.push_back(argv[++i]);
        }
    }
}

int launch(const std::string &exe, const std::string &root,
           const std::vector<std::string> &store) {
    std::vector<std::string> args = {exe, root};
    args.insert(args.end(), store.begin(), store.end());
    std::vector<char *> ptrs;
    for (auto &s : args) ptrs.push_back(s.data());
    npue::Runtime runtime(ptrs[3], root, "");
    return runtime.run((int)ptrs.size(), ptrs.data());
}

}  // namespace

int run_serve(int argc, char **argv) {
    std::string root = default_root(argv[0]);
    int port = 8080;
    std::string bind = "127.0.0.1";
    std::string cli_token;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--root") root = argv[i + 1];
        if (std::string(argv[i]) == "--port") port = std::atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--bind") bind = argv[i + 1];
        if (std::string(argv[i]) == "--token") cli_token = argv[i + 1];
    }
    if (argc < 3 || argv[2][0] == '-')
        throw std::runtime_error("`serve` needs a model name");
    const std::string model_name = argv[2];
    warn_if_unpinned(model_name);
    const std::string container = ensure(root, model_name, cli_token);
    std::vector<std::string> store = {"--model", container,
        "--threads", "24", "--pipeline", "4", "--bind", bind,
        "--serve", std::to_string(port)};
    forward_common(argv, argc, store);
    return launch(argv[0], root, store);
}

int run_embed(int argc, char **argv) {
    std::string root = default_root(argv[0]);
    std::string cli_token;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--root") root = argv[i + 1];
        if (std::string(argv[i]) == "--token") cli_token = argv[i + 1];
    }
    if (argc < 3 || argv[2][0] == '-')
        throw std::runtime_error("`embed` needs a model name");
    const std::string model_name = argv[2];
    if (argc < 4 || argv[3][0] == '-')
        throw std::runtime_error(
            "`embed` needs a file: npuembeddings embed <model> <in.txt> "
            "[out.f32]");
    warn_if_unpinned(model_name);
    const std::string container = ensure(root, model_name, cli_token);
    std::vector<std::string> store = {"--model", container,
        "--threads", "24", "--pipeline", "4", "--embed", argv[3]};
    if (argc > 4 && argv[4][0] != '-') store.push_back(argv[4]);
    forward_common(argv, argc, store);
    return launch(argv[0], root, store);
}

int run_add(int argc, char **argv) {
    std::string root = default_root(argv[0]);
    std::string cli_token;
    for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--root") root = argv[i + 1];
        if (std::string(argv[i]) == "--token") cli_token = argv[i + 1];
    }
    if (argc < 3 || argv[2][0] == '-')
        throw std::runtime_error(
            "`add` needs a HuggingFace repository:\n"
            "    npuembeddings add <org/model> [<sha256>]");
    const std::string repo = argv[2];
    std::string sha;
    if (argc > 3 && argv[3][0] != '-') sha = argv[3];
    if (!sha.empty() && sha.size() != 64)
        throw std::runtime_error(
            "'" + sha + "' is not a sha256 (expected 64 hex chars). "
            "Refusing rather than storing a pin that can never match.");
    npue::hub::CatalogEntry e = npue::hub::probe_repo(
        repo, [](const std::string &m) { std::printf("%s\n", m.c_str()); },
        cli_token);
    e.sha256 = sha;
    const std::string pad_note =
        e.qkv_n ? "  (qkv fused and padded to " + std::to_string(e.qkv_n) + ")"
                : std::string();
    std::printf("\n  %-10s %s\n", "repo", e.repo.c_str());
    std::printf("  %-10s %s\n", "name", e.name.c_str());
    std::printf("  %-10s hidden %lld, %lld layers, %lld heads, ffn %lld%s\n",
                "geometry", (long long)e.hidden, (long long)e.layers,
                (long long)e.heads, (long long)e.ffn,
                e.gated_ffn ? " (gated FFN)" : "");
    std::printf("  %-10s %s\n", "pooling", e.pooling.c_str());
    std::printf("  %-10s tile_n %lld%s\n", "tiling", (long long)e.tile_n,
                pad_note.c_str());
    const std::string art =
        pick_artifacts(root, e.hidden, e.ffn, e.gated_ffn, e.qkv_n, "", "",
                       e.name);
    std::printf("  %-10s %s\n", "design",
                art.empty()
                    ? "NONE INSTALLED for this geometry -- `serve` will "
                      "refuse until one is exported"
                    : art.c_str());
    if (npue::hub::unpinned(e)) {
        std::printf("\n  !! WARNING: no sha256 given, so these weights "
                    "will NOT be verified.\n");
        std::printf("  !! Nothing checks that what %s serves is what you\n",
                    e.repo.c_str());
        std::printf("  !! expect -- not now, and not on any later "
                    "re-fetch.\n");
        std::printf("  !! This warning repeats every time the model runs.\n");
        std::printf("  !! To pin it:  npuembeddings add %s <sha256>\n",
                    e.repo.c_str());
    }
    npue::hub::add_to_user_catalog(root, e);
    std::printf("\n  added to %s\n",
                (std::filesystem::path(root) / "models" /
                 "catalog.json").string().c_str());
    std::printf("  run:  npuembeddings embed %s <in.txt>\n", e.name.c_str());
    return 0;
}

int run_tokenize(int argc, char **argv) {
    std::string root = default_root(argv[0]);
    std::string model_name;
    int max_len = 64;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--root") root = argv[i + 1];
        if (std::string(argv[i]) == "--max-len")
            max_len = std::atoi(argv[i + 1]);
    }
    if (argc < 3 || argv[2][0] == '-')
        throw std::runtime_error("`tokenize` needs a model name");
    model_name = argv[2];
    if (argc < 4 || argv[3][0] == '-')
        throw std::runtime_error(
            "`tokenize` needs a file: npuembeddings tokenize <model> "
            "<in.txt> [--max-len N]");
    std::string model_path = resolve_model_path(root, argc, argv);
    auto model = npue::load_model(model_path);
    auto tok = model->make_tokenizer();
    std::ifstream in(argv[3], std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + std::string(argv[3]));
    std::string line;
    size_t n_lines = 0, n_cut = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto e = tok->encode(line, max_len);
        ++n_lines;
        if (e.truncated) ++n_cut;
        for (size_t k = 0; k < e.input_ids.size(); ++k)
            std::printf("%s%d", k ? " " : "", e.input_ids[k]);
        std::printf("\n");
    }
    if (n_cut)
        std::fprintf(stderr,
                     "  NOTE  %zu of %zu lines exceeded max_len=%d and were "
                     "truncated.\n",
                     n_cut, n_lines, max_len);
    return 0;
}

}  // namespace

void SubcommandDispatcher::register_handler(const std::string &name,
                                           Handler handler) {
    handlers_[name] = std::move(handler);
}

bool SubcommandDispatcher::dispatch(int argc, char **argv,
                                   int &exit_code) const {
    if (argc < 2) return false;
    const std::string sub = argv[1];
    auto it = handlers_.find(sub);
    if (it == handlers_.end()) return false;
    exit_code = it->second(argc, argv);
    return true;
}

void register_default_subcommands(SubcommandDispatcher &dispatcher) {
    dispatcher.register_handler("list", run_list);
    dispatcher.register_handler("serve", run_serve);
    dispatcher.register_handler("embed", run_embed);
    dispatcher.register_handler("add", run_add);
    dispatcher.register_handler("tokenize", run_tokenize);
}

}  // namespace app