//===- main.cpp -----------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- thin entry point.
// Parses CLI args, dispatches subcommands, or loads a model and runs
// the encode/benchmark pipeline.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "cli/cli.hpp"
#include "cli/subcommand.hpp"
#include "runtime/runtime.hpp"
#include "embed_models/model_loader.hpp"

#include <exception>

int main(int argc, char **argv) try {
    auto args = app::CLI::parse(argc, argv);

    if (args.help || argc == 1) {
        app::print_usage();
        return 0;
    }

    // Process-global policy, read before any path can run. See the old main()'s
    // rationale: every encode path has to see the same value, so it is set once.
    app::g_allow_truncation = args.allow_truncation;

    const std::string root = args.root.empty()
        ? app::default_root(argv[0]) : args.root;
    npue::hub::load_user_catalog(root);

    if (args.list_models) {
        app::print_model_table(app::discover_models(root));
        return 0;
    }

    app::SubcommandDispatcher dispatcher;
    app::register_default_subcommands(dispatcher);

    int exit_code = 0;
    if (dispatcher.dispatch(argc, argv, exit_code))
        return exit_code;

    if (app::maybe_tokenize(root, argc, argv))
        return 0;

    const std::string model_path =
        app::resolve_model_path(root, argc, argv);
    (void)npue::load_model(model_path);
    npue::Runtime runtime(model_path, root, args.artifacts);
    return runtime.run(argc, argv);
} catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
