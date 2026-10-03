//===- cli.cpp -------------------------------------------------------*- C++ -*-===//
//
// CLI argument parser: CLI::parse() reads argc/argv into CLIArgs.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "cli/cli.hpp"

#include "cli/flags.hpp"

#include <stdexcept>

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
        else if (a == "--npu-extra-ops" && i + 1 < argc)
            args.npu_ops = argv[++i];
        else if (a == "--root" && i + 1 < argc)
            args.root = argv[++i];
        // AN UNRECOGNISED FLAG IS REFUSED, not ignored. This parser has no
        // catch-all, so before this every unknown option fell off the end and
        // the command ran as if it had not been given -- which is how
        // `--prepare-model models/all-MiniLM-L6-v2 --out /tmp/x.npue` wrote
        // the container to models/all-MiniLM-L6-v2/all-MiniLM-L6-v2.npue and
        // reported success. The output path there is POSITIONAL, so `--out`
        // was both unknown and the flag after it was eaten as nothing.
        //
        // A mistyped flag on a command that PACKS OR RUNS is not a cosmetic
        // mistake: the two flags in this file that silently changed the
        // meaning of a container are exactly the ones worth catching.
        //
        // WHAT COUNTS AS RECOGNISED is flag_is_known(), NOT this file's own list
        // above. The runtime is dispatched largely by scanning argv directly, so
        // most flags never reach this parser's branches at all -- --dev,
        // --bo-mode, --classify, --json, --top-k, every --probe-* and both
        // soaks are read by run_setup.hpp, run_execute.hpp, run_probes.hpp,
        // vit_mode.hpp and stt_mode.hpp. Asking this loop what it happens to
        // handle rejected 21 of them, all of which had worked for years by
        // falling through; verify_pack_parity found it, because it is the one
        // caller that passes --dev. The table in flags.hpp is the flag set, and
        // tools/verify/verify_cli_flags.py is what keeps it from going stale.
        else if (a.rfind("--", 0) == 0 && !flag_is_known(a)) {
            // `--` alone, and a flag the caller deliberately passed through,
            // are not ours to judge; anything else starting with -- that the
            // table does not name is.
            throw std::runtime_error(
                "unrecognised option " + a + ". Run `npuembeddings --help` for "
                "the options this build accepts; a flag that is not recognised "
                "is refused rather than ignored, because an ignored flag "
                "changes what gets built without saying so.");
        }
    }

    return args;
}

}  // namespace app