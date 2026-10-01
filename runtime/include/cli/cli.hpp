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
    std::printf(
        "NpuEmbeddings -- BERT-family embeddings on the AMD NPU (XDNA1/npu1,\n"
        "XDNA2/npu2; --dev selects the generation)\n"
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
        "  npuembeddings transcribe <model> <audio.wav> [--language en]\n"
        "        transcribe 16 kHz audio with a Whisper model. The transcript\n"
        "        goes to stdout on its own and the status block to stderr, so\n"
        "        it pipes. --json prints the OpenAI-shaped object instead,\n"
        "        with one segment per 30 s window. --convert ingests through\n"
        "        ffmpeg (mp3, m4a, webm, or a WAV the reader refuses).\n"
        "\n"
        "  npuembeddings classify <model> <image.png> [more.png ...]\n"
        "        classify images with a ViT model (vit-base-patch16-224), one\n"
        "        per argument. The label goes to stdout on its own and the\n"
        "        status block to stderr, so it pipes. PNG and JPEG only --\n"
        "        anything else is refused by name rather than guessed at, and\n"
        "        so is a non-square resize or a shortest-edge crop: the\n"
        "        container says which geometry it was trained with and this\n"
        "        reads it. --top-k prints the runners-up to stderr, --json\n"
        "        the objects to stdout. `embed` is not how this runs.\n"
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
        "  Options for serve/embed/transcribe/classify:\n"
        "    --port N          listen port (default 8080)\n"
        "    --bind ADDR       interface (default 127.0.0.1, localhost only)\n"
        "    --threads N       host thread budget (default 24 for these)\n"
        "    --pipeline N      concurrent encode lanes (serve/embed pass 4;\n"
        "                      1 lane = no pipelining. A later --pipeline on the\n"
        "                      command line wins, so --pipeline 1 disables it)\n"
        "    --artifacts DIR   override the design set\n"
         "    --npu-extra-ops CODES\n"
         "                      send the listed ops to the ARRAY instead of the\n"
         "                      host, comma-separated. An op not listed runs on\n"
         "                      the host, so there is no inverse flag. The\n"
         "                      default -- nothing listed -- is the\n"
         "                      measured-faster host path for all of them.\n"
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
         "                      Worth asking for? Measured, per 3 s of\n"
         "                      whisper-tiny on 16 threads: conv 0.94 -> 0.66 s\n"
         "                      wall and 5.5 -> 3.8 s CPU, and every other code\n"
         "                      is either neutral or slower. README,\n"
         "                      'What to move to the NPU'.\n"
         "                      The exporter's flag is the SAME STRING and it\n"
         "                      BUILDS what this one selects. gelu/layn/softm\n"
         "                      write a sibling design set each (gelu/,\n"
         "                      layernorm/, softmax/) and cost one extra\n"
         "                      hw_context EACH; conv/attn/mproj/fft/logit are\n"
         "                      streams inside the GEMM sets and cost none.\n"
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
        "\n"
        "  The flag form is unchanged and still works:\n"
        "    npuembeddings <root> --model NAME --artifacts DIR --serve [port]\n"
        "  and carries the probes and benchmarks; see docs/CURRENT_STATUS.md.\n"
        "\n");
}

// A speech-to-text model needs BOTH design sets, and `pick_artifacts` only knows
// how to find an embedder-shaped one. Reporting an STT row as installed on the
// strength of its container alone would say "ready" for a model whose decoder
// design does not exist -- and the refusal then arrives 0.2 s into a request
// instead of in the table a reader checks first.
inline int stt_design_sets(const std::string &root, const std::string &name) {
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const char *sub : {"artifacts_npu1", "artifacts_npu2"}) {
        const fs::path d = fs::path(root) / "runtime" / name / sub;
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
        const bool have_design =
            !pick_artifacts(root, e.hidden, e.ffn, e.gated_ffn, e.qkv_n, "",
                            e.datapath, e.name).empty();
        const char *state = !m                              ? "available"
                            : !encoder_implemented(m->arch) ? "no encoder"
                            // An STT row is ready only with BOTH of its design
                            // sets; one of them cannot transcribe anything, and
                            // the container being present says nothing about
                            // either.
                            : is_stt_arch(m->arch) && m->arch == "whisper_encdec_gelu"
                                ? (stt_design_sets(root, e.name) ? "ready"
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
        std::printf("  %-20s %-9s %6lld %6lld %8s %6.0f MB  %s\n", m.name.c_str(),
                    !encoder_implemented(m.arch)              ? "no encoder"
                    : is_stt_arch(m.arch)                    ? "stt"
                    // A locally packed arch=5 container: `classify`, not
                    // `embed`. The column says which command runs it, because
                    // "ready" here would otherwise be read as "ready to embed"
                    // and the first thing a user tries is the wrong one.
                    : is_vit_arch(m.arch)                    ? "cls"
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
        "               npuembeddings classify <name> <image.png>\n"
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
