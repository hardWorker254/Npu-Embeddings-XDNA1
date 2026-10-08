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
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cli/family.hpp"
#include "common/app_state.hpp"
#include "common/design_selection.hpp"
#include "common/hub.hpp"
#include "common/npue_pack.hpp"
#include "common/npu_ops_flag.hpp"
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
    // Which elementwise ops go on the array; empty = none of them. Parsed and
    // validated where it is used (setup_flags_pools), because the RunContext
    // owns the derived host_* booleans and the encoder's contract.
    std::string npu_ops;
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
    // fputs, NOT printf, and this is not a style preference: the argument below
    // IS the format string if this is a printf, so every `%` in 200 lines of help
    // text is a conversion specification. "2.6%." in the pose block printed a
    // literal `0` into the middle of a sentence -- the text read "2.6%.0 The
    // status block prints both sides" -- which is exactly the kind of help output
    // that teaches a reader to skim past the numbers.
    const Family who = g_family;

    std::fputs(
        "NpuEmbeddings -- BERT-family embeddings on the AMD NPU (XDNA1/npu1,\n"
        "XDNA2/npu2; --dev selects the generation)\n"
        "\n", stdout);

    // WHICH OF THE THREE THIS IS, before any verb: a reader who typed the
    // wrong binary learns it here rather than from a refusal later. This is
    // the SAME roster string the refusals print (cli/family.hpp), so the help
    // text and the refusal cannot drift about which program to run next.
    std::printf("  This is %s.\n\n", g_bin);
    std::fputs(roster(), stdout);
    std::fputs("\n\n", stdout);

    // EVERY VERB BLOCK IS GUARDED BY owns(), so nothing here documents a
    // command this binary would refuse. Help that lists a verb and then sends
    // the reader to another program for it is the failure the split exists to
    // end -- and a block that printed unconditionally would be a second roster
    // to keep in step with the first.
    //
    // The program name is BUILT from g_bin rather than typed: `npuaudio` and
    // `npuimage` are the same text with one word changed, and the word that
    // differs is the one a reader is here to learn.
    if (owns(who, "list"))
        std::fputs((std::string("  ") + g_bin +
                    " list\n"
                    "        every model this build can run, and which are installed\n"
                    "\n").c_str(), stdout);

    std::fputs((std::string("  ") + g_bin +
                " serve <model> [--port N] [--bind ADDR]\n"
                "        the HTTP endpoint for WHATEVER the container is. One verb,\n"
                "        four endpoints, and the model's arch picks which:\n"
                "          /v1/embeddings           a BERT-family text model\n"
                "          /v1/audio/transcriptions Whisper\n"
                "          /v1/classify             a ViT classifier\n"
                "          /v1/pose                 YOLOv8-pose\n"
                "        Downloads and verifies the model first if it is not\n"
                "        installed yet. The mode is chosen by the container's arch, never\n"
                "        by a flag, and each endpoint answers only for its own path -- a\n"
                "        wrong path is a 404 naming the right one, never another\n"
                "        endpoint's answer.\n"
                "        Only this binary's own family is served here (" + g_bin +
                "): another\n"
                "        binary's container is refused by name before the listener opens,\n"
                "        rather than answered by an endpoint this process is not named for.\n"
                "\n").c_str(), stdout);

    if (owns(who, "embed"))
        std::fputs((std::string("  ") + g_bin +
                    " embed <model> <in.txt> [out.f32]\n"
                    "        embed a text file, one text per line\n"
                    "\n").c_str(), stdout);

    if (owns(who, "transcribe"))
        std::fputs((std::string("  ") + g_bin +
                    " transcribe <model> <audio.wav> [--language en]\n"
                    "        transcribe 16 kHz audio with a Whisper model. The transcript\n"
                    "        goes to stdout on its own and the status block to stderr, so\n"
                    "        it pipes. --json prints the OpenAI-shaped object instead,\n"
                    "        with one segment per 30 s window. --convert ingests through\n"
                    "        ffmpeg (mp3, m4a, webm, or a WAV the reader refuses).\n"
                    "\n").c_str(), stdout);

    if (owns(who, "classify"))
        std::fputs((std::string("  ") + g_bin +
                    " classify <model> <image.png> [more.png ...]\n"
                    "        classify images with a ViT model (vit-base-patch16-224), one\n"
                    "        per argument. The label goes to stdout on its own and the\n"
                    "        status block to stderr, so it pipes. PNG and JPEG only --\n"
                    "        anything else is refused by name rather than guessed at, and\n"
                    "        so is a non-square resize or a shortest-edge crop: the\n"
                    "        container says which geometry it was trained with and this\n"
                    "        reads it. --top-k prints the runners-up to stderr, --json\n"
                    "        the objects to stdout. `embed` is not how this runs.\n"
                    "        Over HTTP: `" + g_bin +
                    " serve <model>`, which answers POST\n"
                    "        /v1/classify with the same object `classify --json` prints (both\n"
                    "        call npue::vit::prediction_json), one `image` part and an\n"
                    "        optional `top_k`. GET /v1/labels returns the whole vocabulary, so\n"
                    "        a client can turn an id into a name without the container.\n"
                    "\n").c_str(), stdout);

    if (owns(who, "pose"))
        std::fputs((std::string("  ") + g_bin +
                    " pose <model> <image.png> [more.png ...]\n"
                    "        detect people and their 17 COCO keypoints with a YOLOv8-pose\n"
                    "        model. Same shape as classify and for the same reasons: one\n"
                    "        image per argument, the result on stdout and the status block\n"
                    "        on stderr. --json prints one object per image with every box,\n"
                    "        every joint, the 12-edge skeleton and the letterbox transform,\n"
                    "        --text a human summary. --conf/--iou/--kpt/--max-det move the\n"
                    "        thresholds the container was not given. --pose-dump FILE\n"
                    "        writes every graph node's fp32 output, for diffing against an\n"
                    "        independent interpreter.\n"
                    "        CPU BY DEFAULT, and that is a measurement rather than a\n"
                    "        preference. --npu-ops conv moves the 72 convolutions to\n"
                    "        the array and nothing else, needs a container packed with --npu\n"
                    "        and a design set, and is SLOWER here: 258 ms of network on 16\n"
                    "        threads against 384 ms dispatched as GEMMs, because N must be a\n"
                    "        multiple of 128 (2.6x arithmetic padding waste before anything\n"
                    "        runs), the stem alone is 100 of the 436 dispatches, and a\n"
                    "        dispatch costs 660 us of which 140 us is the device's GEMM. Per\n"
                    "        layer 19 of the 72 are faster on the array and an oracle\n"
                    "        splitting each onto its faster side measured 146 ms -- 2.6%.\n"
                    "        The status block prints both sides, per image.\n"
                    "        MEASURED on the CPU path, 640x640 input, i8: 29 ms front end,\n"
                    "        258 ms network of which 88 ms is the GEMM, 57 ms im2col, 10 ms\n"
                    "        the weight transpose and 7 ms the output transpose, 1 ms decode.\n"
                    "\n"
                    "        The same model over HTTP: `" + g_bin +
                    " serve <model>`, which\n"
                    "        answers POST /v1/pose. Multipart with one `image` part (PNG or\n"
                    "        JPEG, by magic bytes) and optional conf, iou, kpt, max_det. The\n"
                    "        answer is byte for byte what `pose --json` prints -- both call\n"
                    "        npue::pose::result_json -- so the endpoint and the CLI cannot\n"
                    "        drift. Also GET /health and GET /v1/models. Per-request\n"
                    "        thresholds apply to that request only.\n"
                    "\n").c_str(), stdout);

    if (owns(who, "add"))
        std::fputs((std::string("  ") + g_bin +
                    " add <org/model> [<sha256>]\n"
                    "        teach this installation about a model that is not built in --\n"
                    "        typically a finetune of one that is. Reads the repository's\n"
                    "        config.json, derives the geometry, checks a design serves it,\n"
                    "        and writes models/catalog.json. No weights are downloaded until\n"
                    "        the first serve/embed.\n"
                    "        WITHOUT a sha256 the weights are NOT verified. That is allowed,\n"
                    "        and it is warned about on every single run.\n"
                    "\n").c_str(), stdout);

    // THE MODEL ARGUMENT, once, for whichever verbs printed above. The example
    // is per-family on purpose -- a worked command the reader of THIS binary
    // cannot run teaches the wrong verb, which is what the guards above exist
    // to avoid, and an example that names a file which does not exist would
    // teach it twice.
    std::fputs((std::string(
        "  <model> in any of this binary's verbs is either a NAME -- looked up\n"
        "        in models/, fetched and verified against the catalogue's sha256\n"
        "        when the catalogue knows it -- or a PATH to a .npue, used as\n"
        "        given and never re-resolved by name:\n"
        "            ") + usage_example() +
        "\n"
        "        The runtime names a path-held model after its file, which is\n"
        "        also how it finds runtime/<stem>/artifacts_npu<N>.\n"
        "\n").c_str(), stdout);

    // THE OPTIONS ARE NOT FILTERED, unlike the verbs above. They are one list
    // annotated with the verb each belongs to (`--language CODE   transcribe:`),
    // so a reader can tell in the line itself whether it applies -- and the
    // annotations are a property of the flag, not of the binary, whereas a verb
    // block is a command to type.
    std::fputs(
        "  Options for this binary's verbs:\n"
        "    --port N          listen port (default 8080)\n"
        "    --bind ADDR       interface (default 127.0.0.1, localhost only)\n"
        "    --threads N       host thread budget (default 24 for these)\n"
        "    --pipeline N      concurrent encode lanes (serve/embed pass 4;\n"
        "                      1 lane = no pipelining. A later --pipeline on the\n"
        "                      command line wins, so --pipeline 1 disables it)\n"
        "    --artifacts DIR   override the design set\n"
        "    --npu-ops CODES\n"
        "                      send the listed ops to the ARRAY instead of the\n"
        "                      host, comma-separated. An op not listed runs on\n"
        "                      the host, so there is no inverse flag. The\n"
        "                      default -- nothing listed -- runs every\n"
        "                      op in this process; for most codes that\n"
        "                      is the measured-faster side, and for\n"
        "                      gemm it is not.\n"
        "                        gemm   the four per-layer GEMMs\n"
        "                        gelu   GELU (exact erf; a Whisper\n"
        "                               container declares this one)\n"
        "                        layn   LayerNorm\n"
        "                        softm  softmax\n"
        "                        conv   Whisper's conv1/conv2\n"
        "                        attn   attention, as QK^T and softmax.V GEMMs\n"
        "                        mproj  the slaney mel bank, as a GEMM\n"
        "                        fft    the 400-point transform, as a GEMM\n"
        "                        logit  the tied-embedding projection, in\n"
        "                               eight chunks of the vocabulary\n"
        "                      The same nine codes in every architecture;\n"
        "                      which of them yours can honour is in\n"
        "                      NPU_OPS.md, and a code that cannot is refused\n"
        "                      by name with the reason, never ignored.\n"
        "                      Worth asking for? Measured, per 3 s of\n"
        "                      whisper-tiny on 16 threads: conv 0.94 -> 0.66 s\n"
        "                      wall and 5.5 -> 3.8 s CPU; gemm 1.06 -> 0.17 s\n"
        "                      on an embedder and 0.435 -> 0.241 s on a ViT.\n"
        "                      Every other code is either neutral or slower.\n"
        "                      README, 'What to move to the NPU'.\n"
        "                      The exporter's flag is the SAME STRING and it\n"
        "                      BUILDS what this one selects. gelu/layn/softm\n"
        "                      write a sibling design set each (gelu/,\n"
        "                      layernorm/, softmax/) and cost one extra\n"
        "                      hw_context EACH; gemm/conv/attn/mproj/fft/\n"
        "                      logit are streams inside the GEMM sets and\n"
        "                      cost none.\n"
        "                      A missing design is refused by name, never\n"
        "                      silently run on the host -- and a run that\n"
        "                      would exceed the context budget is refused\n"
        "                      before any context is built unless\n"
        "                      --allow-contention says the contention is\n"
        "                      intended (see below).\n"
        "                      The speech-to-text codes are refused on an\n"
        "                      embedder. Replaces --npu-ops, --npu-eltwise and\n"
        "                      --host-ln/--host-sm/--host-gelu, which are now\n"
        "                      refused by name so a stale command line cannot\n"
        "                      look like it worked.\n"
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
        "    --language CODE   transcribe: the language to prime the decoder\n"
        "                      with (en, de, ru, ...). There is no detection\n"
        "                      and no default worth having: an assumed language\n"
        "                      is a fluent transcript of the wrong language, so\n"
        "                      it defaults to en and says so on stderr. A code\n"
        "                      the model's tokenizer has no token for is\n"
        "                      refused, never approximated.\n"
        "    --task NAME      transcribe: 'transcribe' (default) or\n"
        "                      'translate' -- English output either way\n"
        "    --convert        transcribe: ingest through ffmpeg instead of the\n"
        "                      WAV reader. The reader refuses what it cannot\n"
        "                      represent (a second channel, 8-bit, 44.1 kHz)\n"
        "                      by name; this is the way in for those\n"
        "    --max-new N      transcribe: cap on generated tokens per window\n"
        "                      (default: the model's own position bound, 448)\n"
        "    --chunk-seconds N  transcribe: window length, default 30 -- the\n"
        "                      model's own. Longer is refused: the encoder has\n"
        "                      weights for a fixed number of positions and a\n"
        "                      cut window is a transcript of the first 30 s\n"
        "                      presented as the whole recording\n"
        "    --stride-seconds N transcribe: overlap on EACH side of a window,\n"
        "                      default 5 (transformers' own), so windows start\n"
        "                      chunk - 2*stride apart and the merge\n"
        "                      de-duplicates the overlap\n"
        "    --json           transcribe: print {\"text\", \"segments\", ...}\n"
        "    --top-k          classify: print the runners-up to stderr, under\n"
        "                      the image. The whole row is ranked and the top\n"
        "                      ten shown -- a truncated list reads as \"nothing\n"
        "                      else is close\", which is a different claim\n"
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
        "                      tools/export/export_gemm_rtp.py --seq to build for a\n"
        "                      longer one.\n"
        "\n", stdout);

    std::fputs((std::string(
        "  The flag form is unchanged and still works:\n"
        "    ") + g_bin +
        " <root> --model NAME --artifacts DIR --serve [port]\n"
        "  and carries the probes and benchmarks; see docs/CURRENT_STATUS.md.\n"
        "\n").c_str(), stdout);
}

// A speech-to-text model needs BOTH design sets, and `pick_artifacts` only knows
// how to find an embedder-shaped one. Reporting an STT row as installed on the
// strength of its container alone would say "ready" for a model whose decoder
// design does not exist -- and the refusal then arrives 0.2 s into a request
// instead of in the table a reader checks first.
//
// THE PATHS ARE model_set_candidates()'s, all four of them per generation. The
// first version of this looked only in runtime/<name>/artifacts_npu<N>, which is
// where the exporter USED to write -- so a set exported to the current
// runtime/artifacts/<name>/artifacts_npu<N> (and carried inside the container
// besides) was reported as "no design" for whisper-tiny and whisper-base, the
// two rows this function exists to answer. A stale path here is not a wrong
// directory, it is a wrong status in the one table a reader checks first.
inline int stt_design_sets(const std::string &root, const std::string &name) {
    namespace fs = std::filesystem;
    for (const char *sub : {"artifacts_npu1", "artifacts_npu2"})
        for (const std::string &n : {name, name + "-i8"})
            for (const fs::path &base :
                 {fs::path(root) / "runtime" / "artifacts",
                  fs::path(root) / "artifacts",
                  fs::path(root) / "runtime", fs::path(root)}) {
                const fs::path d = base / n / sub;
                if (std::ifstream(d / "gemm_rtp" / "design.json").good() &&
                    std::ifstream(d / "gemm_rtp_dec" / "design.json").good())
                    return 1;
            }
    return 0;
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
        // A CARRIED SET COUNTS, and THIS is the line that matters: the catalogue
        // loop is the one that prints every model the catalogue knows, including
        // the ones present on disk -- the installed loop further down skips
        // exactly those (`if (npue::hub::find(m.name)) continue;`). So a fix
        // applied only to the installed loop changes nothing for all-MiniLM,
        // which is both installed AND catalogued, and the state column kept
        // saying "no design" for a file that carries the set inside it.
        //
        // The two sources are OR-ed, not chosen between: pick_artifacts is still
        // what answers for a container that does not carry one, and
        // `m->carries_design` is the container's own statement, read from its
        // manifest rather than inferred from whatever happens to sit on disk
        // beside it.
        const bool have_design =
            (m && m->carries_design) ||
            !pick_artifacts(root, e.hidden, e.ffn, e.gated_ffn, e.qkv_n, "",
                            e.datapath, e.name).empty();
        const char *state = !m                              ? "available"
                            : !encoder_implemented(m->arch) ? "no encoder"
                            // An STT row is ready only with BOTH of its design
                            // sets; one of them cannot transcribe anything.
                            //
                            // A CARRIED SET COUNTS -- here too, as it does in
                            // `have_design` above: the container states its own
                            // two sets under design/gemm_rtp/ and
                            // design/gemm_rtp_dec/, and requiring both KEYS
                            // rather than one is what keeps this from calling
                            // ready a container that carries the encoder alone.
                            // The disk scan is the other half of the OR, for a
                            // container packed before designs travelled inside
                            // it.
                            : is_stt_arch(m->arch) && m->arch == "whisper_encdec_gelu"
                                ? ((m->carries_design && m->carries_design_dec) ||
                                           stt_design_sets(root, e.name)
                                       ? "ready"
                                       : "no design")
                            // arch=5 needs ONE set -- gemm_rtp -- and says so
                            // with `ready`/`no design` like an embedder, because
                            // that is what it is as far as "can this run" goes.
                            // What it is NOT is an embedder, and the two
                            // differences that matter (no pooling column value
                            // the user chose, and `classify` rather than
                            // `embed`) are stated in the notes and in the
                            // `cls` line below the table.
                            : is_vit_arch(m->arch)
                                ? (have_design ? "ready" : "no design")
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
        // A pose container's state says "pose" and nothing about design sets,
        // because there is nothing to say: every convolution runs on the host,
        // so the file is runnable as it stands. A "no design" here would tell
        // the user to go and export something, which is both false and the only
        // instruction in this table with no command behind it.
        // The layers/hidden columns are DASHES, not zeros, for a detector: a
        // `0 layers` row reads as a claim about a malformed container, and
        // 0 here means the architecture does not have the concept. Same reason
        // `pooling` says `n/a` rather than `none`.
        const char *lay = is_pose_arch(m.arch) ? "-" : nullptr;
        char layers[24], hidden[24];
        if (lay) {
            std::snprintf(layers, sizeof layers, "%s", lay);
            std::snprintf(hidden, sizeof hidden, "%s", lay);
        } else {
            std::snprintf(layers, sizeof layers, "%lld", (long long)m.layers);
            std::snprintf(hidden, sizeof hidden, "%lld", (long long)m.hidden);
        }
        // The last column, assembled ONCE and read here. `pixel_command_note`
        // answers empty for anything that is not one of the three pixel kinds, so
        // the two branches below are a pixel detector's command and everyone
        // else's repo -- and there is no fourth case to remember, because the
        // function is what decides what a pixel kind is.
        const std::string note = pixel_command_note(m.kind);

        std::printf("  %-20s %-9s %6s %6s %8s %6.0f MB  %s\n", m.name.c_str(),
                    !encoder_implemented(m.arch)              ? "no encoder"
                    // These three columns say WHICH COMMAND RUNS THE FILE, and
                    // they read it from the container's own `kind` rather than
                    // from the architecture. The two used to be the same thing
                    // because there was one detector; there are THREE now
                    // (arch=6 pose, arch=7 hands, arch=8 mppose) and they are
                    // three verbs, so a state column derived from the arch could
                    // only ever name one of them -- it would have labelled a
                    // MediaPipe Hands container `pose` and sent the user to a
                    // command that dispatches by arch and answers "not a pose
                    // container" on a file that finds hands perfectly well.
                    //
                    // `kind` is the container's own claim and the packer writes
                    // it, so this cannot drift from the file the way a list in
                    // this header would.
                    : subcommand_for_kind(m.kind)           ? m.kind.c_str()
                    : is_stt_arch(m.arch)                    ? "stt"
                    // A locally packed arch=5 container: `classify`, not
                    // `embed`. The column says which command runs it, because
                    // "ready" here would otherwise be read as "ready to embed"
                    // and the first thing a user tries is the wrong one.
                    : is_vit_arch(m.arch)                    ? "cls"
                    : is_pose_arch(m.arch)                   ? "pose"
                    : m.gemm_layout == "host"                 ? "cpu"
                    // A CONTAINER THAT CARRIES ITS OWN SET IS READY, whatever
                    // is on disk. The `||` is the whole fix: pick_artifacts()
                    // scans the filesystem, so a downloaded .npue that needs
                    // nothing beside it was listed as "no design" -- the one
                    // status this project has the most evidence about being
                    // wrong.
                    : !m.carries_design &&
                          pick_artifacts(root, m.hidden, m.ffn, m.gated_ffn,
                                         m.qkv_n, "", "bf16", m.name).empty()
                        ? "no design" : "ready",
                    layers, hidden, m.pooling.c_str(),
                    m.mb,
                    // The last column is the command for a non-embedder and the
                    // source repo for the rest, because for an embedder the repo
                    // is the thing worth checking and for a locally packed
                    // detector the repo is "n/a" -- the file came from a local
                    // ONNX and there is nothing to name.
                    //
                    // Built from `kind` rather than from the arch, for the reason
                    // the state column above is: the command is a property of the
                    // file. `note` was assembled before this printf, so the state
                    // column and this one cannot name different subcommands.
                    !note.empty()            ? note.c_str()
                        : is_vit_arch(m.arch) ? "npuimage classify <name> "
                                               "<image>"
                        : is_stt_arch(m.arch) ? "npuaudio transcribe <name> "
                                               "<audio.wav>"
                                              : m.repo.c_str());
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
        "  stt        installed, and it is a SPEECH-TO-TEXT model: no pooling,\n"
        "             and it needs TWO design sets -- gemm_rtp for the encoder\n"
        "             stack and gemm_rtp_dec for the decoder. `transcribe` and\n"
        "             `serve` are how it runs; `embed` is not. A whisper row says\n"
        "             `no design` until BOTH are exported, which is one command:\n"
        "               python tools/export/export_gemm_rtp.py --target <name> \\\n"
        "                   --arch 1 --out runtime\n"
        "  cls         installed, and it is an IMAGE CLASSIFIER: no pooling and\n"
        "             no text, and it needs ONE design set -- gemm_rtp, the same\n"
        "             four streams bge-base-en-v1.5 uses, because a ViT's GEMM\n"
        "             shapes ARE bge-base's and its patch embedding is [n,768]x\n"
        "             [768,768], which is attn_out's shape. `classify` is how it\n"
        "             runs; `embed` is not:\n"
        "               npuimage classify <name> <image.png>\n\n");
    // The last line of the legend names THIS binary, because the legend is
    // printed by `list` (npuembeddings) and by the ambiguity refusal in
    // resolve_model_path() -- which npuaudio and npuimage reach too. One
    // literal there would have been a legend offering a program the reader
    // did not run.
    std::printf("  %s serve <model>\n\n", g_bin);
}

inline void warn_if_unpinned(const std::string &name) {
    const npue::hub::CatalogEntry *e = npue::hub::find(name);
    if (!e || !npue::hub::unpinned(*e)) return;
    std::printf("\n  !! '%s' has NO sha256 pin in this build (%s).\n",
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

// --prepare-model <dir> [out.npue]: build a model container from the ONNX
// export placed under `dir` (BUILD.md §2.2). The checkpoint's OWN
// config.json picks the packer (BERT / gemma3 / nomic / gte), never the
// directory name.
//
// The release ships no weights: they belong to their authors, and taking an
// export from the canonical source with a checksum beats trusting a blob in
// someone's zip. This flag is what keeps that a two-step setup rather than a
// Python install -- and it is exactly what tools/verify/verify_pack_parity.py
// drives, which is why every resolution below mirrors pack_npue.py's main()
// step for step: same pooling source, same source_repo source, same mac, same
// per-family max_seq. A difference in ANY of them is a byte difference, which
// is the whole content of that gate.
//
// Returns true when the flag was present (the container has been written and
// the caller must exit 0); false to fall through to the encode paths.
inline bool maybe_prepare_model(const CLIArgs &args, int argc, char **argv) {
    if (args.prepare_model_dir.empty()) return false;
    const std::string dir = args.prepare_model_dir;

    // THE DESIGN SET TO STORE INSIDE THE CONTAINER, resolved the same way
    // tools/lib/design_embed.py resolves it: <root>/runtime/artifacts/<model>
    // [/-i8]/artifacts_npu1, accepted only when it really holds a gemm_rtp set.
    //
    // Two packers have to agree on this byte for byte -- verify_pack_parity.py
    // compares them -- so the rule is written once per language and the gate
    // compares what they PRODUCED rather than what they intended. A resolver that
    // disagreed about the directory would show up as a length difference in the
    // container and nothing subtler than that, which is the good case.
    //
    // The model name comes from the DIRECTORY being packed (models/<name>), which
    // is also the container's basename and therefore what the runtime will look the
    // embedded set up by. Deriving it from anything else would produce a container
    // that carries a set the reader cannot find.
    //
    // THE ROOT IS FOUND BY WALKING UP FROM THE CHECKPOINT, not by counting
    // levels and not from argv[0]. `dir` is models/<name> in this tree, so
    // "parent_path().parent_path()" happens to be the repository root -- and it
    // happened to be that twice, which is not a rule. The question is "where is
    // runtime/artifacts from here", and the only honest way to answer it is to ask
    // the filesystem going up until something answers.
    //
    // The earlier argv[0] version found nothing, so the C++ packer stored no set
    // and parity complained about a 26-byte JSON difference with the four
    // artifacts keys simply absent from that side. Under --prepare-model the thing
    // that matters is next to the CHECKPOINT, not next to the binary.
    const std::filesystem::path dirp(dir);
    const std::string model_name = dirp.filename().string();
    // A HELPER WITH EARLY RETURNS rather than nested loops with breaks. The loop
    // version was traced and observed doing the wrong thing: after finding
    // <model>/artifacts_npu1 it went on to try <model>-i8 and stored THAT, so the
    // float container carried the int8 set -- a container whose layout_hash, stream
    // table and geometry are all for a datapath it does not use. verify_pack_parity
    // caught it as a size difference in two design entries (72046 vs 71533 bytes),
    // which is the only reason it was caught at all rather than shipping.
    //
    // The rewrite is not cosmetic: three nested scopes each needing to stop the
    // search is a place to get the exit wrong, and an early return cannot be
    // reached by the wrong break.
    const auto find_design_dir = [&]() -> std::string {
      std::error_code ec;
      std::filesystem::path up =
          std::filesystem::absolute(dirp, ec).parent_path();
      for (int depth = 0; depth < 4 && up != up.root_path(); ++depth) {
        for (const std::string &n : {model_name, model_name + "-i8"})
          for (const char *gen : {"artifacts_npu1", "artifacts_npu2"}) {
            const std::string c =
                (up / "runtime" / "artifacts" / n / gen).string();
            if (std::ifstream(c + "/gemm_rtp/final.xclbin").good())
              return c;
          }
        up = up.parent_path();
      }
      return {};
    };
    const std::string design_dir = find_design_dir();

    // --dev is read HERE as well as in Runtime::run(), because that has not
    // been entered yet and the B panel's sub-tile is the MMAC geometry: npu1
    // is (s=8, t=4), npu2 is (8, 8). A container packed for one and read by a
    // design built for the other is NOT rejected -- the byte count, the shapes
    // and the layout_hash all agree, because both sides used the same wrong
    // constant -- it just computes wrong numbers. So the target is resolved
    // before anything is laid out, not after.
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--dev")
            app::set_running_device(argv[i + 1]);
    const npue::MacGeom mac = npue::mac_for_device(app::running_device());

    // The container is named after the checkpoint directory, not after MiniLM.
// This was a literal until a second model made it visible.
    //
    // AND IT GOES ALONGSIDE THE MODEL DIRECTORY, not inside it. It used to be
    // `models/<name>/<name>.npue`, which is a place no reader looks:
    // `model_catalog` globs `models/*.npue` and `ensure_model` looks for
    // `<root>/models/<name>.npue`, so the output of `--prepare-model` -- the
    // documented one-command way to build a container -- was invisible to both.
    // Running it produced a container, printed `wrote ...`, and left `embed
    // <name>` still saying the model was not installed. The sibling location is
    // where every other path that writes a container already puts it
    // (pack_npue.py, and `embed`'s own auto-fetch), so this is the third
    // spelling of the same answer being reduced to one.
    std::string out = args.prepare_model_out;
    if (out.empty())
        out = std::filesystem::path(dir).parent_path().string() + "/" +
              std::filesystem::path(dir).filename().string() + ".npue";

    auto read_all = [](const std::string &p) {
        std::ifstream f(p, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + p);
        std::ostringstream s;
        s << f.rdbuf();
        return s.str();
    };
    auto note = [](const std::string &s) {
        std::printf("%s\n", s.c_str());
    };

    // Detected from the checkpoint's OWN config.json, as pack_npue.py does.
    const std::string model_type =
        app::json_field_string(read_all(dir + "/config.json"), "model_type");

    // Two families have no C++ packer at all. Saying so here, before anything
    // is written, beats failing three stages into a half-built container.
    if (model_type == "whisper")
        throw std::runtime_error(
            "there is no C++ packer for arch=4. Run:\n"
            "    python tools/pack/pack_npue.py --model-dir " + dir +
            "\n        --out " + out + " --device " +
            app::running_device() + " --max-seq 1500\n"
            "  (--max-seq 1500 is the model's own max_source_positions; the "
            "default of 256 produces a container the encoder refuses.)");
    if (model_type == "vit")
        throw std::runtime_error(
            "there is no C++ packer for arch=5, and the --int8 form cannot be "
            "written in C++ at all (its calibration needs torch forward hooks "
            "over an image corpus). Run:\n"
            "    python tools/pack/pack_npue.py --model-dir " + dir +
            "\n        --out " + out + " --device " + app::running_device() +
            "\n"
            "  There is deliberately NO --max-seq: a ViT's position count is "
            "fixed at (224/16)^2 + 1 = 197 by the image size.");

    // arch=1 (EmbeddingGemma / Gemma3 MQA+RoPE+GeGLU): a completely different
    // tensor shape and container, routed to its own packer rather than
    // threaded through the BERT logic below -- pack_npue.py's main() makes
    // the same decision the same way.
    if (model_type == "gemma3_text") {
        std::string source_repo = args.source_repo;
        if (source_repo.empty()) {
            const std::string ck = read_all(dir + "/CHECKPOINT.json");
            source_repo = app::json_field_string(ck, "repo_id");
            if (source_repo.empty())
                throw std::runtime_error(
                    "no CHECKPOINT.json under " + dir + " and no --source-repo "
                    "given -- refusing to guess which repository these "
                    "weights came from");
        }
        std::printf("  source     %s\n", source_repo.c_str());
        std::printf("NpuEmbeddings -- preparing %s\n", out.c_str());
        // Defaults are the production geometry, so a cold clone with no flags
        // self-produces a container the ARRAY can run -- before this flag it
        // self-produced a host-only container and quietly ran at 0.2 seq/s.
        npue::prepare_model_gemma(dir, out, source_repo, note, args.tile_k,
                                  args.tile_n, args.gemma_host_only, mac);
        std::printf("  wrote %s\n", out.c_str());
        return true;
    }

    // Tile size is a PROPERTY OF THE MODEL, not a constant. The design
    // asserts N % (tile_n * n_cols) == 0, and bge-large's N in
    // {1024, 3072, 4096} makes 48 illegal -- the legal set there is
    // {8, 16, 32, 64} and 64 does not fit L1 (65,536 B against the 63 KB
    // budget), so it must be 32. Both packers take it and neither freezes
    // the resulting hash.
    const int64_t tile_k = args.tile_k, tile_n = args.tile_n;
    const npue::Layout lay =
        npue::gemm_b_layout(tile_k, tile_n, mac.s, mac.t);
    std::printf("  layout     tile (%lld, %lld), mac (s=%lld, t=%lld), hash "
                "%s...\n",
                (long long)tile_k, (long long)tile_n, (long long)mac.s,
                (long long)mac.t, lay.hash.substr(0, 16).c_str());

    // Pooling comes from the checkpoint's own 1_Pooling/config.json, the same
    // source pack_npue.py reads. Both packers must agree or
    // verify_pack_parity fails, which is the point of having the gate.
    std::string pooling;
    {
        const std::string pj = read_all(dir + "/1_Pooling/config.json");
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
                "1_Pooling/config.json asks for neither or both of cls and "
                "mean; this runtime implements exactly those two");
        pooling = cls ? "cls" : "mean";
        std::printf("  pooling    %s (from 1_Pooling/config.json)\n",
                    pooling.c_str());
    }

    // Which repository these weights came from. pack_npue.py reads
    // CHECKPOINT.json for this and so must we, or the two packers disagree. A
    // container that misattributes its own weights is a licensing statement,
    // so an unknown repo REFUSES rather than guessing.
    std::string source_repo = args.source_repo;
    if (source_repo.empty()) {
        const std::string ck = read_all(dir + "/CHECKPOINT.json");
        source_repo = app::json_field_string(ck, "repo_id");
        if (source_repo.empty())
            throw std::runtime_error(dir +
                "/CHECKPOINT.json has no repo_id and no --source-repo given "
                "-- refusing to guess which repository these weights came "
                "from");
    }
    std::printf("  source     %s\n", source_repo.c_str());

    // arch=2 (nomic-embed-text-v1.5): RoPE + gated SwiGLU rather than BERT's
    // absolute-position + GELU -- routed to its own packer, mirroring
    // pack_npue.py's `model_type == "nomic_bert"` branch. Placed here because
    // nomic shares every resolution above with the BERT path unchanged; only
    // the packer differs.
    if (model_type == "nomic_bert") {
        std::printf("NpuEmbeddings -- preparing %s "
                    "(arch=nomic_bert_rope_swiglu)\n", out.c_str());
        npue::prepare_model_nomic(dir, pooling, source_repo, out, lay.json,
                                  lay.hash, tile_k, tile_n, 256, note, mac);
        std::printf("  wrote %s\n", out.c_str());
        return true;
    }
    // arch=3 (gte-multilingual-base): model_type "new", same dispatch rule.
    // max_seq 64 matches the Python-packed container this mirror is held
    // byte-identical to (pack_npue.py's default for this family is the same
    // number): under RoPE the position table is zeros, so max_seq only caps
    // request length, and the shipped designs and goldens are seq-64.
    if (model_type == "new") {
        std::printf("NpuEmbeddings -- preparing %s (arch=gte_new_rope_geglu)\n",
                    out.c_str());
        npue::prepare_model_gte(dir, pooling, source_repo, out, lay.json,
                                lay.hash, tile_k, tile_n, 64, note, mac);
        std::printf("  wrote %s\n", out.c_str());
        return true;
    }

    std::printf("NpuEmbeddings -- preparing %s\n", out.c_str());
    npue::prepare_model(dir, dir + "/vocab.txt", dir + "/config.json", pooling,
                        source_repo, out, lay.json, lay.hash, tile_k, tile_n,
                        256, note, mac, design_dir);
    std::printf("  wrote %s\n", out.c_str());
    return true;
}

}  // namespace app
