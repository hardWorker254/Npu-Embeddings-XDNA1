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
        "--bo-mode", "--npu-ops", "--language", "--task", "--max-new",
        "--chunk-seconds", "--stride-seconds", "--classify", "--pose",
        // --audio is the generic spelling of --transcribe, for the
        // `npuembeddings <root> --model m.npue --audio a.wav` form rather than
        // the `transcribe` subcommand, and stt_mode.hpp reads it with the same
        // flag("--audio") either way. It was missing here, which is not a
        // parse error: forward_common() simply forwarded neither the flag nor
        // its value, and stt_mode.hpp then reported that neither --transcribe
        // nor --serve was given. Found by tools/verify/verify_cli_flags.py's
        // two-tables-agree check.
        "--audio",
        "--conf", "--iou", "--kpt", "--max-det", "--pose-dump",
        // arch=7. --hands is the image list (run_hands() rewrites positionals
        // into it, and it REPEATS, so it is accumulated by the loop in
        // hands_mode.hpp rather than read last-wins -- the same treatment
        // --classify and --pose get). --max-hands and --hands-dump are this
        // mode's own and were added to this table in the same commit that added
        // the mode, because a flag missing here is not a parse error: forward
        // _common drops it silently and hands_mode then reports that it was
        // never given. That is what verify_cli_flags.py's two-tables-agree check
        // exists to catch, and it caught --audio.
        "--hands", "--max-hands", "--hands-dump",
        // --serve is deliberately NOT here: it is arity 0 in flags.hpp (the port
        // is the next argv entry, and only if it starts with a digit), and it is
        // injected by run_serve rather than forwarded from the user's line. A
        // user who types `serve m --serve 9000` gets 9000 from run_serve's own
        // parse of --port, which is the documented spelling.
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
            a == "--top-k" ||
            // arch=6's own no-argument flag. --text switches the pose mode from
            // its default JSON to a human summary, so dropping it here would
            // leave `pose ... --text` printing JSON and looking like the flag
            // had done nothing.
            a == "--text")
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
    // ONE verb, four endpoints, and the container's arch picks which:
    //
    //   arch=1,2,3  BERT-family text   POST /v1/embeddings
    //   arch=4      Whisper            POST /v1/audio/transcriptions
    //   arch=5      ViT classifier     POST /v1/classify
    //   arch=6      YOLOv8-pose        POST /v1/pose
    //
    // arch=7 (MediaPipe hands) is the ONE arch with no endpoint, and `serve` on
    // a hands container is refused by name in hands_mode.hpp rather than being
    // routed to a mode that has nothing to serve. That makes the list above
    // four and not five, and it is a real gap rather than a design choice: see
    // the refusal there for what a server/hands_backend.cpp would need.
    //
    // The modes never meet and none of them is chosen by a flag: runtime.cpp
    // dispatches on arch before it ever looks at --serve. That is why there is no
    // `pose-server` or `classify-server` verb -- a second verb per architecture
    // would be a second thing to learn for the same model, and the URL space is
    // already unambiguous because each arch owns a different path.
    //
    // --threads 24 / --pipeline 4 are the embedding pipeline's per-request GEMM
    // batching, and the three other modes read --threads for their own pool and
    // ignore --pipeline. So the image and speech endpoints inherit a pool of 24
    // rather than their own default of 16; the pose section's measured timings
    // are quoted at the 24 `serve` hands them, and the CLI numbers at 16.
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
            "    npuembeddings serve <model>            (POST /v1/classify)\n"
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

// `pose` for an arch=6 body-pose model. Same shape as run_classify and for the
// same reasons: the images are written to the flag form as repeated --pose,
// which Runtime::run dispatches by the container's arch, so
// `npuembeddings pose <model> <a.png> <b.png>` and
// `npuembeddings <root> --model ... --pose <a.png>` are one code path.
//
// It adds four value flags over run_classify -- --conf, --iou, --kpt, --max-det
// -- which are also added to flag_takes_value() above, for the same reason
// run_classify lists --classify there: a positional scan that did not know they
// took a value would try to open "0.25" as a PNG.
int run_pose(int argc, char **argv) {
    std::string root = default_root(argv[0]);
    std::string cli_token;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--root") root = argv[i + 1];
        if (std::string(argv[i]) == "--token") cli_token = argv[i + 1];
    }
    if (argc < 3 || argv[2][0] == '-')
        throw std::runtime_error(
            "`pose` needs a model name or a path to a .npue container");
    const std::string model_name = argv[2];
    if (argc < 4 || argv[3][0] == '-')
        throw std::runtime_error(
            "`pose` needs an image, or you meant `serve`:\n"
            "    npuembeddings pose <model> <image.png> [more.png ...]\n"
            "    npuembeddings serve <model>            (POST /v1/pose)\n"
            "  (PNG and JPEG; anything else is refused rather than guessed at)\n"
            "  --conf 0.25 --iou 0.70 --kpt 0.50 --max-det 300\n"
            "  --text                 a human summary instead of JSON\n"
            "  --npu-ops conv    run the convolutions on the array\n"
            "                          (slower here -- measured, see\n"
            "                          runtime/include/pose/net.hpp)");
    if (!is_container_path(model_name)) warn_if_unpinned(model_name);
    const std::string container = resolve_container(root, model_name, cli_token);
    std::vector<std::string> store = {"--model", container};
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
        store.push_back("--pose");
        store.push_back(a);
    }
    forward_common(argv, argc, store);
    return launch(argv[0], root, store);
}

// arch=7. Mirrors run_pose: positionals become --hands, the model name and
// --root/--token are picked off the front, and everything else is forwarded for
// forward_common's whitelist to judge.
int run_hands(int argc, char **argv) {
    std::string root = default_root(argv[0]);
    std::string cli_token;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--root") root = argv[i + 1];
        if (std::string(argv[i]) == "--token") cli_token = argv[i + 1];
    }
    if (argc < 3 || argv[2][0] == '-')
        throw std::runtime_error(
            "`hands` needs a model name or a path to a .npue container");
    const std::string model_name = argv[2];
    if (argc < 4 || argv[3][0] == '-')
        throw std::runtime_error(
            "`hands` needs an image:\n"
            "    npuembeddings hands <model> <image.png> [more.png ...]\n"
            "  (PNG and JPEG; anything else is refused rather than guessed at)\n"
            "  --max-hands 1     run the landmark network on at most N\n"
            "  --text            a human summary instead of JSON\n"
            "  --hands-dump FILE every graph node's output, for the gate\n"
            "  (the score, NMS and crop thresholds come from the container --\n"
            "   they are part of the checkpoint, so there is no flag for them,\n"
            "   and typing one is refused rather than ignored)\n"
            "  `serve` is not an alternative here: arch 7 has no HTTP endpoint,\n"
            "  which hands_mode.hpp refuses by name and says why.");
    if (!is_container_path(model_name)) warn_if_unpinned(model_name);
    const std::string container = resolve_container(root, model_name, cli_token);
    std::vector<std::string> store = {"--model", container};
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
        store.push_back("--hands");
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
    dispatcher.register_handler("pose", run_pose);
    dispatcher.register_handler("hands", run_hands);
}

}  // namespace app