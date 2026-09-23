//===- cli.cpp -------------------------------------------------------*- C++ -*-===//
//
// CLI argument parser: CLI::parse() reads argc/argv into CLIArgs.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "cli/cli.hpp"

namespace app {

CLIArgs CLI::parse(int argc, char **argv) {
    CLIArgs args;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        if (a == "--bench" && i + 1 < argc)
            args.bench = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc)
            args.threads = std::atoi(argv[++i]);
        else if (a == "--pipeline" && i + 1 < argc)
            args.pipeline = std::atoi(argv[++i]);
        else if (a == "--prefix" && i + 1 < argc)
            args.prefix = argv[++i];
        else if (a == "--allow-truncation")
            args.allow_truncation = true;
        else if (a == "--cpu")
            args.cpu = true;
        else if (a == "--artifacts" && i + 1 < argc)
            args.artifacts = argv[++i];
        else if (a == "--model" && i + 1 < argc)
            args.model = argv[++i];
        else if (a == "--serve")
            args.serve = true;
        else if (a == "--port" && i + 1 < argc)
            args.port = std::atoi(argv[++i]);
        else if (a == "--bind" && i + 1 < argc)
            args.bind = argv[++i];
        else if (a == "--embed" && i + 1 < argc) {
            args.embed_file = argv[++i];
            if (i + 1 < argc && argv[i + 1][0] != '-')
                args.embed_out = argv[++i];
        }
        else if (a == "--add" && i + 1 < argc) {
            args.add_repo = argv[++i];
            if (i + 1 < argc && argv[i + 1][0] != '-')
                args.add_sha = argv[++i];
        }
        else if (a == "--tokenize" && i + 1 < argc) {
            args.tokenize_file = argv[++i];
            if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0])))
                args.tokenize_max_len = std::atoi(argv[++i]);
        }
        else if (a == "--prepare-model" && i + 1 < argc) {
            args.prepare_model_dir = argv[++i];
            if (i + 1 < argc && argv[i + 1][0] != '-')
                args.prepare_model_out = argv[++i];
        }
        else if (a == "--list-models")
            args.list_models = true;
        else if (a == "--help" || a == "-h")
            args.help = true;
        else if (a == "--allow-contention")
            args.allow_contention = true;
        else if (a == "--max-len" && i + 1 < argc)
            args.max_len = std::atoi(argv[++i]);
        else if (a == "--source-repo" && i + 1 < argc)
            args.source_repo = argv[++i];
        else if (a == "--gemma-host-only")
            args.gemma_host_only = true;
        else if (a == "--tile-k" && i + 1 < argc)
            args.tile_k = std::atoll(argv[++i]);
        else if (a == "--tile-n" && i + 1 < argc)
            args.tile_n = std::atoll(argv[++i]);
        else if (a == "--root" && i + 1 < argc)
            args.root = argv[++i];
    }

    return args;
}

}  // namespace app