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

// A NAME and a PATH are different things, and this is where the difference is
// decided for every subcommand that takes a model argument.
//
//   <name>  resolved under <root>/models/<name>.npue by hub::ensure_model(),
//           which FETCHES whatever the catalogue knows and the tree lacks.
//   <path>  used as given, because it already names a file that exists.
//           Routing it through the name resolver is what made
//               embed models/all-MiniLM-L6-v2.npue in.txt
//           fail with "no model named ... is installed" plus the whole model
//           table -- a path's stem is not a model name, and the answer offered
//           a list of models to a user who was already holding the file. A path
//           that does not exist is refused by path (throw_missing_container)
//           rather than silently re-read as a name: `embed ./builds/mine.npue`
//           with a typo must not embed a DIFFERENT model that happens to be
//           installed.
//
// Silence about pins is deliberate and is the same silence the flag form has:
// hub::find() says nothing about a file the user points at ("not an error,
// because a user may have packed their own container"). The runtime names a
// path-held model after the FILE (runtime.cpp: g_model_name = path stem),
// which is also how it finds runtime/<stem>/artifacts_npu<N>.
std::string resolve_container(const std::string &root, const std::string &arg,
                              const std::string &token) {
    if (is_container_path(arg)) return arg;
    if (looks_like_container_path(arg)) throw_missing_container(arg, root);
    return ensure(root, arg, token);
}

// THE FLAGS THAT CARRY A VALUE, one list, named. Two call sites need it -- the
// forwarder below, and run_classify()'s positional scan -- and a second
// hand-written copy is exactly the drift this file's comment about whitelists
// warns about: run_classify would take `--threads 8`'s VALUE as an image path
// and then refuse to open the integer 8 as a PNG.
//
// Written as a function rather than a table of pairs because the caller wants
// two different things from it -- "forward this and its value" and "does this
// one swallow the next argument" -- and both are the same question.
bool flag_takes_value(const std::string &a) {
    static const char *const kWithValue[] = {
        "--threads", "--pipeline", "--prefix", "--artifacts", "--dev",
        "--bo-mode", "--npu-extra-ops", "--language", "--task", "--max-new",
        "--chunk-seconds", "--stride-seconds", "--classify", "--image",
        "--root", "--port", "--bind", "--token", "--max-len",
    };
    for (const char *f : kWithValue)
        if (a == f) return true;
    return false;
}

void forward_common(const char *const *argv, int argc,
                    std::vector<std::string> &store) {
    // A REMOVED flag is refused here rather than dropped by the whitelist below.
    // This function is the gate: whatever it does not forward, Runtime::run
    // never sees, so a flag the user typed that silently disappears is exactly
    // the failure the whitelist exists to prevent -- and a removed flag is the
    // case where it is most likely, because the command line is old.
    {
      std::vector<std::string> args;
      for (int i = 0; i < argc; ++i) args.emplace_back(argv[i]);
      refuse_removed_op_flags(args);
      refuse_exporter_only_flags(args);
    }
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
        if (a == "--cpu" || a == "--allow-truncation" ||
            a == "--sim-c-bf16" || a == "--no-fuse-ffn" ||
            a == "--allow-contention" ||
            // the STT mode's own no-argument flags. They are listed here for
            // the same reason as the rest: a flag the whitelist drops is a flag
            // the runtime never sees, and `serve <whisper> --convert` would
            // then transcribe with a WAV reader that refuses every mp3.
            a == "--convert" || a == "--json" ||
            // arch=5's own no-argument flags. Same reason, and --top-k is the
            // one that would hurt most: dropping it would print one label and
            // leave the user with no way to see how close the runner-up was.
            a == "--top-k")
            store.push_back(a);
        else if (flag_takes_value(a) && i + 1 < argc) {
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
        throw std::runtime_error(
            "`serve` needs a model name or a path to a .npue container");
    const std::string model_name = argv[2];
    if (!is_container_path(model_name)) warn_if_unpinned(model_name);
    const std::string container = resolve_container(root, model_name, cli_token);
    std::vector<std::string> store = {"--model", container,
        "--threads", "24", "--pipeline", "4", "--bind", bind,
        "--serve", std::to_string(port)};
    forward_common(argv, argc, store);
    return launch(argv[0], root, store);
    // A Whisper container is served by the STT mode off the same flags: it
    // reads --serve and --bind and ignores --threads/--pipeline in favour of
    // its own pool, and it answers POST /v1/audio/transcriptions instead of
    // /v1/embeddings. The two modes never meet: the mode is picked by the
    // container's arch, not by a flag.
}

int run_embed(int argc, char **argv) {
    std::string root = default_root(argv[0]);
    std::string cli_token;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--root") root = argv[i + 1];
        if (std::string(argv[i]) == "--token") cli_token = argv[i + 1];
    }
    if (argc < 3 || argv[2][0] == '-')
        throw std::runtime_error(
            "`embed` needs a model name or a path to a .npue container");
    const std::string model_name = argv[2];
    if (argc < 4 || argv[3][0] == '-')
        throw std::runtime_error(
            "`embed` needs a file: npuembeddings embed <model> <in.txt> "
            "[out.f32]");
    if (!is_container_path(model_name)) warn_if_unpinned(model_name);
    const std::string container = resolve_container(root, model_name, cli_token);
    std::vector<std::string> store = {"--model", container,
        "--threads", "24", "--pipeline", "4", "--embed", argv[3]};
    if (argc > 4 && argv[4][0] != '-') store.push_back(argv[4]);
    forward_common(argv, argc, store);
    return launch(argv[0], root, store);
}

// `transcribe` for a speech-to-text model. The audio file is written to the
// flag form, which Runtime::run then dispatches by the container's arch -- one
// code path for `npuembeddings transcribe ...` and
// `npuembeddings <root> --model ... --transcribe ...`.
int run_transcribe(int argc, char **argv) {
    std::string root = default_root(argv[0]);
    std::string cli_token;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--root") root = argv[i + 1];
        if (std::string(argv[i]) == "--token") cli_token = argv[i + 1];
    }
    if (argc < 3 || argv[2][0] == '-')
        throw std::runtime_error(
            "`transcribe` needs a model name or a path to a .npue container");
    const std::string model_name = argv[2];
    if (argc < 4 || argv[3][0] == '-')
        throw std::runtime_error(
            "`transcribe` needs an audio file:\n"
            "    npuembeddings transcribe <model> <audio.wav> [--language en]\n"
            "  (--convert ingests through ffmpeg, for mp3/m4a/webm and for WAVs "
            "the reader refuses)");
    if (!is_container_path(model_name)) warn_if_unpinned(model_name);
    const std::string container = resolve_container(root, model_name, cli_token);
    // No --threads/--pipeline defaults here: the STT mode splits host work over
    // one pool and knows its own shape, and `serve`'s "24 threads, 4 lanes" is
    // an embedding-lane statement that means nothing to an autoregressive
    // decoder.
    std::vector<std::string> store = {"--model", container, "--transcribe",
                                      argv[3]};
    forward_common(argv, argc, store);
    return launch(argv[0], root, store);
}

// `classify` for an image classifier. The image paths are written to the flag
// form as repeated --classify, which Runtime::run then dispatches by the
// container's arch -- one code path for
// `npuembeddings classify <model> <a.png> <b.png>` and
// `npuembeddings <root> --model ... --classify <a.png>`.
//
// It is deliberately run_transcribe's shape rather than run_embed's: no
// --threads/--pipeline defaults, because "24 threads, 4 lanes" is a statement
// about the embedding pipeline's per-request GEMM batching and this mode walks
// one image through one pool. Its own --threads default lives in vit_mode.hpp.
int run_classify(int argc, char **argv) {
    std::string root = default_root(argv[0]);
    std::string cli_token;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--root") root = argv[i + 1];
        if (std::string(argv[i]) == "--token") cli_token = argv[i + 1];
    }
    if (argc < 3 || argv[2][0] == '-')
        throw std::runtime_error(
            "`classify` needs a model name or a path to a .npue container");
    const std::string model_name = argv[2];
    if (argc < 4 || argv[3][0] == '-')
        throw std::runtime_error(
            "`classify` needs an image:\n"
            "    npuembeddings classify <model> <image.png> [more.png ...]\n"
            "  (PNG and JPEG; anything else is refused rather than guessed at. "
            "--top-k prints the runners-up to stderr, --json to stdout)");
    if (!is_container_path(model_name)) warn_if_unpinned(model_name);
    const std::string container = resolve_container(root, model_name, cli_token);
    std::vector<std::string> store = {"--model", container};
    // Every leading non-flag argument is an image, so
    // `classify <model> a.png b.png --top-k` classifies both. A flag's VALUE is
    // never mistaken for a path: flag_takes_value() is the same list
    // forward_common() uses, so `--threads 8 a.png` classifies a.png and not
    // the integer 8.
    bool swallow = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (swallow) {
            swallow = false;
            continue;
        }
        if (!a.empty() && a[0] == '-') {
            swallow = flag_takes_value(a);
            continue;
        }
        store.push_back("--classify");
        store.push_back(a);
    }
    forward_common(argv, argc, store);
    return launch(argv[0], root, store);
}

int run_add(int argc, char **argv) {    std::string root = default_root(argv[0]);
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
        throw std::runtime_error(
            "`tokenize` needs a model name or a path to a .npue container");
    model_name = argv[2];
    if (argc < 4 || argv[3][0] == '-')
        throw std::runtime_error(
            "`tokenize` needs a file: npuembeddings tokenize <model> "
            "<in.txt> [--max-len N]");
    // The POSITIONAL is the model, so resolve that -- not argv's "--model",
    // which this form does not have. Reading --model made `tokenize <name>
    // <file>` print the whole model table and refuse ("several models are
    // installed; say which with --model") on any machine with two models, and
    // made a path to a container unusable here when the flag form already
    // took one. Same resolution as `--model`, one source: resolve_model_path()
    // accepts a name or a .npue path, and does not fetch, which is what
    // `--tokenize` on the flag form does today.
    const std::string model_path = resolve_model_path(root, model_name);
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
    dispatcher.register_handler("transcribe", run_transcribe);
    dispatcher.register_handler("classify", run_classify);
}

}  // namespace app