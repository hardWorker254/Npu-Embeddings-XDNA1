//===- entry.cpp ------------------------------------------------------------*- C++ -*-===//
//
// The body of main(), shared by all three binaries.
//
// npuembeddings, npuaudio and npuimage each have a three-line main() under
// src/bin/ that sets app::g_family / app::g_bin and calls app::entry(). The
// family is therefore decided ONCE, at process start, by the binary that was
// exec'd -- never inferred from argv, because a renamed copy of the binary
// would then change which modes it runs, and a reader who renamed a binary to
// fit a script would get a program that disagrees with its own name in a way
// nothing on the command line could explain.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "cli/family.hpp"
#include "cli/cli.hpp"
#include "cli/subcommand.hpp"
#include "runtime/runtime.hpp"
#include "embed_models/model_loader.hpp"

#include <exception>

namespace app {

// The container's family is checked inside resolve_model_path(), which every
// path here goes through -- flag form, `--tokenize`, and the single-installed-
// model shortcut alike. Subcommands resolve through resolve_container(), which
// checks the same way, so nothing below has to remember to.
int entry(int argc, char **argv) try {
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

    // --prepare-model builds a container from the ONNX export placed under a
    // checkpoint directory. It runs after the utility flags above and BEFORE
    // resolve_model_path(), because it does not consume an installed model --
    // it produces one, and on a tree holding several containers
    // resolve_model_path() would refuse to guess which one the caller meant.
    if (app::maybe_prepare_model(args, argc, argv))
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

}  // namespace app
