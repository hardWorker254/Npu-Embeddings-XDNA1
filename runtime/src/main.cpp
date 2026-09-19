//===- main.cpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- M7: the full MiniLM encode in C++, no Python in the process.
// SPDX-License-Identifier: Apache-2.0
//
//   token ids -> embeddings -> 6 layers -> mean pool -> L2 normalize
//
// On the NPU: the four projection/FFN GEMMs per layer, GELU, LayerNorm,
// softmax -- seven resident designs, each holding its own xclbin.
//
// On the host: the embedding gather (a gather, never a multiply), attention's
// per-head GEMMs (their [64,32]x[32,64] shapes fail the whole-array design's
// M % (m*4) == 0), bias adds, pooling. Exactly the split tasks/0021 measured in
// Python, so the two runtimes are comparable.
//
// Weights come out of the .npue by mmap and are handed to DMA untouched: the
// designs are built pretiled, matching how the file stores them.
//
//   npuembed <repo-root> [--bench N]
//
// Without --bench it validates against the HuggingFace-derived golden. With it,
// it runs N encodes and reports wall clock and CPU time -- the numbers
// tasks/0018 could only get for Python.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <functional>
#include <mutex>
#include <string>
#include <cctype>
#include <thread>
#include <vector>
#include <ctime>

#include "gemma_encode.hpp"
#include "gemma_kernels.hpp"
#include "hub.hpp"
#include "json_min.hpp"
#include "npu_contention.hpp"
#include "npu_device.hpp"
#include "npue.hpp"
#include "npue_pack.hpp"
#include "http.hpp"
#include "tokenizer.hpp"
#include "tokenizer_xlmr.hpp"
#include "app_state.hpp"
#include "bert_encoder.hpp"
#include "cli.hpp"
#include "cli_add.hpp"
#include "run_context.hpp"
#include "run_probes.hpp"
#include "run_setup.hpp"
#include "run_execute.hpp"
#include "design_selection.hpp"
#include "embed_service.hpp"
#include "gemma_mode.hpp"
#include "gemma_npu_encoder.hpp"
#include "host_kernels.hpp"
#include "model_catalog.hpp"
#include "pool.hpp"
#include "tokenizer_facade.hpp"
#include "app_state.hpp"
#include "bert_encoder.hpp"

// NOMINMAX is defined here rather than on the command line: XRT's own headers
// define it too, and defining it globally makes every XRT translation unit warn
// about the redefinition. Locally, before windows.h, it just works -- and
// without it `std::max` becomes `std::(...)` and the errors point at the wrong
// line entirely.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

using namespace app;
}  // namespace

// --- subcommands (0.2.0) --------------------------------------------------
//
// `npuembeddings list` and `npuembeddings serve <model>` exist because the
// flag form below (`npuembed <root> --model X --artifacts Y --serve`) asks a
// first-time user for three things they have no way to know: where the root
// is, which artifact set matches their model, and that `--model` is spelled
// like the container stem. All three are derivable, so they are derived.
//
// The subcommands are TRANSLATED into the flag form and then fall through to
// the same code path. That is deliberate: a second dispatch path would be a
// second place for the batch tiers, the contention gate and the fixture check
// to drift out of agreement, and this project has had five bugs of exactly
// that shape.

int main(int argc, char **argv) try {
  // No arguments at all: say what this is and what it can run.
  //
  // The old behaviour was to take root = ".." and start the golden-vector
  // validation encode -- a developer default that made sense when the only
  // caller was a task log. Double-clicking the executable, which is what a
  // release invites, would then either dispatch to the NPU or fail with a
  // path error about a directory the user never named. Neither answers the
  // question a bare invocation is actually asking.
  if (argc == 1) {
    print_usage();
    print_catalog(default_root(argv[0]));
    return 0;
  }

  // Read once, here, rather than in each of the three encode paths: the
  // policy is global to the process and every path has to honour it, so a
  // per-path lookup is three chances to miss one. Scanned from the raw argv
  // before subcommand rewriting, so `serve --allow-truncation` and
  // `--serve --allow-truncation` behave identically.
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--allow-truncation") g_allow_truncation = true;

  // Subcommands, translated into the flag form the rest of main() reads.
  std::vector<std::string> store;
  if (argc > 1) {
    const std::string sub = argv[1];
    const bool is_sub = (sub == "list" || sub == "serve" || sub == "embed" ||
                         sub == "add" ||
                         sub == "help" || sub == "--help" || sub == "-h");
    if (is_sub) {
      // --root is read before anything else, since it decides where we look
      // for the model we are about to talk about. --token is read here too
      // (rather than only where ensure_model() needs it) so its scan sits
      // next to --root's, matching this function's existing per-flag-loop
      // style; never printed anywhere below.
      std::string sub_root;
      std::string cli_token;
      for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--root") sub_root = argv[i + 1];
        if (std::string(argv[i]) == "--token") cli_token = argv[i + 1];
      }
      if (sub_root.empty()) sub_root = default_root(argv[0]);

      if (sub == "help" || sub == "--help" || sub == "-h") {
        print_usage();
        return 0;
      }
      // Every path below this point may look a model up, so the user
      // catalogue has to be merged in first. One call, one writer.
      npue::hub::load_user_catalog(sub_root, [](const std::string &m) {
        std::printf("%s\n", m.c_str());
      });

      if (sub == "list") {
        print_catalog(sub_root);
        return 0;
      }

      // `add <repo> [<sha256>]` -- teach this installation about a model that
      // is not built in, typically a finetune of one that is.
      //
      // It does NOT download the weights: it reads the repository's own
      // config.json, DERIVES the geometry from that, checks the geometry can
      // actually be tiled, and writes a catalogue row. The first `serve` or
      // `embed` then fetches and packs through exactly the same path every
      // built-in model uses -- there is no second fetch path to drift.
      if (sub == "add") {
        if (argc < 3 || argv[2][0] == '-')
          throw std::runtime_error(
              "`add` needs a HuggingFace repository:\n"
              "    npuembeddings add <org/model> [<sha256>]\n"
              "  The sha256 is the digest of that repository's "
              "model.safetensors. Omit it and the weights are NOT verified -- "
              "which is allowed, and warned about every time.");
        const std::string repo = argv[2];
        std::string sha;
        if (argc > 3 && argv[3][0] != '-') sha = argv[3];
        if (!sha.empty() && sha.size() != 64)
          throw std::runtime_error(
              "'" + sha + "' is not a sha256 (expected 64 hex characters). "
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
            pick_artifacts(sub_root, e.hidden, e.ffn, e.gated_ffn, e.qkv_n);
        std::printf("  %-10s %s\n", "design",
                    art.empty()
                        ? "NONE INSTALLED for this geometry -- `serve` will "
                          "refuse until one is exported"
                        : art.c_str());

        if (npue::hub::unpinned(e)) {
          std::printf("\n  !! WARNING: no sha256 given, so these weights will "
                      "NOT be verified.\n");
          std::printf("  !! Nothing checks that what %s serves is what you\n",
                      e.repo.c_str());
          std::printf("  !! expect -- not now, and not on any later "
                      "re-fetch.\n");
          std::printf("  !! This warning repeats every time the model runs.\n");
          std::printf("  !! To pin it:  npuembeddings add %s <sha256>\n",
                      e.repo.c_str());
        }
        npue::hub::add_to_user_catalog(sub_root, e);
        std::printf("\n  added to %s\n",
                    (std::filesystem::path(sub_root) / "models" /
                     "catalog.json").string().c_str());
        std::printf("  run:  npuembeddings embed %s <in.txt>\n",
                    e.name.c_str());
        return 0;
      }

      // serve / embed both need a model named as the next positional.
      if (argc < 3 || argv[2][0] == '-') {
        print_catalog(sub_root);
        throw std::runtime_error("`" + sub + "` needs a model name");
      }
      const std::string want = argv[2];
      warn_if_unpinned(want);
      if (sub == "embed" && argc < 4)
        throw std::runtime_error(
            "`embed` needs a file: npuembeddings embed <model> <in.txt> "
            "[out.f32]");

      // Fetch it if we do not have it. This is the whole point of the
      // subcommand: the checksum comparison that used to live in a batch
      // file now happens here, inside the executable.
      const std::string container = npue::hub::ensure_model(
          sub_root, want, [](const std::string &s) {
            std::printf("%s\n", s.c_str());
            std::fflush(stdout);
          }, cli_token);

      // arch=1 (EmbeddingGemma) has no NPU design at all -- the artifacts
      // lookup below is a BERT-only question that must never be asked for
      // it (tasks/0066: it threw "no NPU design for hidden 768" on a fresh
      // root with no BERT designs installed, even though the fetch+pack
      // above had already succeeded). Route straight to run_gemma_mode with
      // the subcommand's ORIGINAL argv (in.txt/out.f32 at argv[3]/argv[4] --
      // the flag-form rewrite a few lines down hasn't happened yet), the
      // same dispatch the early `--prepare-model`-adjacent check uses for a
      // container reached directly by path.
      {
        bool is_gemma = false;
        try {
          npue::File gpeek(container);
          is_gemma = (gpeek.config_string("arch") == "gemma3_mqa_rope_geglu");
        } catch (const std::exception &) {
          // Not a container config_string() can read the way we expect --
          // fall through to the BERT path below, which will fail with its
          // own, more specific error if this really is not a BERT
          // container. Deliberately NOT wrapping run_gemma_mode() itself in
          // this catch: once we know it IS Gemma, any error it throws is a
          // real, informative error that must propagate, not get silently
          // swallowed into the wrong (BERT) failure path.
        }
        if (is_gemma) {
          // `serve <model> [port]` reaches the same run_gemma_mode as
          // `embed`, now that arch=1 has an endpoint (tasks/0115). The
          // subcommand's port sits where embed's in.txt does, at argv[3].
          std::vector<std::string> gstore = {argv[0]};
          if (sub == "serve") {
            gstore.push_back("--serve");
            if (argc > 3 && std::isdigit(static_cast<unsigned char>(argv[3][0])))
              gstore.push_back(argv[3]);
          } else {
            gstore.push_back("--embed");
            gstore.push_back(argv[3]);
            if (argc > 4 && argv[4][0] != '-') gstore.push_back(argv[4]);
          }
          // Forward the flags the NPU path actually reads (--threads,
          // --prefix, --cpu, --artifacts, --max-len). They used to be dropped
          // here, which was harmless while this arch had exactly one code
          // path and is not now: `embed <model> x.txt --cpu` would silently
          // run the NPU path and report it as the control.
          for (int i = 3; i < argc; ++i) {
            if (argv[i][0] != '-') continue;
            gstore.push_back(argv[i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') gstore.push_back(argv[++i]);
          }
          std::vector<char *> gptrs;
          for (auto &s : gstore) gptrs.push_back(s.data());
          npue::File gmodel(container);
          return run_gemma_mode(gmodel, container, sub_root,
                                (int)gptrs.size(), gptrs.data());
        }
      }

      // Which design serves this model. --artifacts still wins if given.
      std::string art;
      for (int i = 2; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--artifacts") art = argv[i + 1];
      if (art.empty()) {
        int64_t hidden = 0, inter = 0, qkv_n = 0;
        bool gated = false;
        // The container's own B layout, so an int8 model cannot be handed a
        // bf16 design (tasks/0080) -- their (op, K, N) are identical and only
        // the layout tells them apart. Any tiled GEMM tensor carries it;
        // layer.0.qkv exists in every architecture this runtime encodes.
        std::string layout;
        try {
          npue::File f(container);
          hidden = f.config_int("hidden");
          inter = f.config_int("intermediate");
          gated = config_flag(f, "gated_ffn", false);
          try { qkv_n = f.config_int("qkv_n"); } catch (const std::exception &) {}
          layout = f.info("layer.0.qkv").layout_hash;
        } catch (const std::exception &) {
        }
        // Which MMAC datapath THIS model was adopted for (tasks/0104, T23) --
        // read from the catalogue, keyed by the name the user typed, never
        // guessed from geometry: bge-small shares MiniLM's hidden-384
        // geometry and did NOT clear the bfp16 MTEB gate (tasks/0103). A
        // model not in the catalogue (locally packed, or `add`ed without a
        // datapath ever being decided for it) gets the default CatalogEntry's
        // "bf16" -- the safe choice, and what every design set here was until
        // this task.
        const auto *ce = npue::hub::find(want);
        const std::string want_datapath = ce ? ce->datapath : "bf16";
        art = pick_artifacts(sub_root, hidden, inter, gated, qkv_n, layout,
                             want_datapath);
        if (art.empty())
          throw std::runtime_error(
              "no NPU design for hidden " + std::to_string(hidden) +
              " intermediate " + std::to_string(inter) +
              (gated ? " (gated FFN)" : "") + ", datapath " + want_datapath +
              " under " + sub_root +
              " -- this release carries designs for the geometries it was "
              "built with; export one with tools/export_gemm_rtp.py --hidden " +
              std::to_string(hidden) + " --intermediate " +
              std::to_string(inter) + (gated ? " --gated-ffn" : "") +
              (want_datapath == "bfp16" ? " --emulate-bfp16 --c-bf16" : ""));
      }

      // Defaults that suit a server rather than a measurement. The flag form
      // keeps its conservative --threads 1, because a benchmark that quietly
      // used 24 cores would misreport the per-core claim.
      // pipeline 4, not 2: measured 2026-08-20 (tasks/0052) -- bge-base
      // 175.4 -> 209.1 seq/s and MiniLM 892.7 -> 962.6 going 2 -> 4 lanes,
      // saturating at 5. The cost is group latency (a 4-lane group is ~2.4 s
      // of work on bge-base), which a throughput server accepts.
      std::string threads = "24", pipeline = "4";
      for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--threads") threads = argv[i + 1];
        if (std::string(argv[i]) == "--pipeline") pipeline = argv[i + 1];
      }
      // --prefix (tasks/0071): passed through unchanged when given. Absent
      // by default -- resolve_prefix() falls back to the container's own
      // prompt_default, or is a no-op for a model with no prompts table.
      std::string prefix;
      bool have_prefix = false;
      for (int i = 2; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--prefix") {
          prefix = argv[i + 1];
          have_prefix = true;
        }

      store = {argv[0], sub_root,       "--model",    want,
               "--artifacts", art,      "--threads",  threads,
               "--pipeline",  pipeline};
      // Forward --cpu so the flag path can REFUSE it (tasks/0124, T50). It
      // used to be dropped here, which combined with the flag path ignoring
      // it into `embed <model> x.txt --cpu` silently running the NPU -- the
      // exact fail-open shape tasks/0118 removed from --prefix.
      for (int i = 3; i < argc; ++i)
        if (std::string(argv[i]) == "--cpu") store.push_back("--cpu");
      if (have_prefix) {
        store.push_back("--prefix");
        store.push_back(prefix);
      }
      if (sub == "serve") {
        std::string port = "8080", bind = "127.0.0.1";
        for (int i = 2; i < argc - 1; ++i) {
          if (std::string(argv[i]) == "--port") port = argv[i + 1];
          if (std::string(argv[i]) == "--bind") bind = argv[i + 1];
        }
        store.push_back("--bind");
        store.push_back(bind);
        store.push_back("--serve");
        store.push_back(port);
      } else {
        store.push_back("--embed");
        store.push_back(argv[3]);
        if (argc > 4 && argv[4][0] != '-') store.push_back(argv[4]);
      }

      static std::vector<char *> ptrs;
      ptrs.clear();
      for (auto &s : store) ptrs.push_back(s.data());
      argc = (int)ptrs.size();
      argv = ptrs.data();
    }
  }

  const std::string root = (argc > 1) ? argv[1] : "..";


  // RunContext replaces the tower of locals that the decomposed command
  // functions read and write. Early fields are filled here; the probe
  // functions below just consume them.
  RunContext ctx;
  ctx.argc = argc;
  ctx.argv = argv;
  ctx.root = root;

  if (maybe_prepare_model(argc, argv)) return 0;
  if (int gemma_r = maybe_gemma_mode(root, argc, argv); gemma_r >= 0)
    return gemma_r;

  ctx.bench = 0;
  for (int i = 2; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--bench") ctx.bench = std::atoi(argv[i + 1]);

  // --cpu selects the host-only control encoder, which exists for arch=1
  // only (run_gemma_mode honours it above). The BERT family has no host
  // encoder in this build, and until tasks/0124 the flag was parsed there
  // and silently ignored -- a caller asking for the CPU control got the NPU,
  // with correct vectors, which is what made it invisible (T50, found by
  // 0121's semantic gate reading the status line). Refusing beats ignoring
  // (tasks/0118).
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == "--cpu")
      throw std::runtime_error(
          "--cpu: no host encoder exists for this architecture in this "
          "build -- the flag would be ignored and the NPU would run anyway "
          "(T50). It is honoured for embeddinggemma-300m (arch=1) only; "
          "drop the flag, or use a host reference implementation "
          "(reference/encoder_*.py) as the CPU control.");

  // --artifacts selects which export to load, so two builds of the same
  // designs can be compared in the same session rather than across a rebuild.
  // --artifacts names a design set. It is resolved against both layouts this
  // ships in: the source tree (<root>/runtime/<name>) and an extracted
  // release, where the design sits beside the executable (<root>/<name>, or
  // <root> itself when the name is "."). An absolute path is taken as given.
  // Chosen by which candidate actually CONTAINS a design, so a typo is an
  // error about the design rather than a confusing one about a missing file.
  std::string art_name;
  for (int i = 2; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--artifacts") art_name = argv[i + 1];

  // With no --artifacts, the flag form used to default to the literal
  // "artifacts" -- a per-op design set predating the unified xclbin, whose
  // width happens to be 384. MiniLM and bge-small ran; every wider model
  // died on a staged-buffer size check or a b_layout_hash refusal (the
  // guards working -- no wrong answer was ever returned, but the error named
  // the wrong problem). Since tasks/0124 (T50) the no-flag form resolves
  // through pick_artifacts() from the loaded container's own geometry --
  // the same call the `embed`/`serve` subcommands make -- which happens
  // AFTER the model is loaded, below.
  std::string art;
  if (!art_name.empty()) {
    auto has_design = [](const std::string &d) {
      return std::ifstream(d + "/gemm_rtp/design.json").good() ||
             std::ifstream(d + "/qkv/design.json").good();
    };
    const std::vector<std::string> candidates = {
        art_name,                          // absolute, or relative to cwd
        root + "/" + art_name,             // an extracted release
        root + "/runtime/" + art_name,     // the source tree
    };
    for (const auto &c : candidates)
      if (has_design(c)) { art = c; break; }
    if (art.empty())
      throw std::runtime_error(
          "no design set found for --artifacts '" + art_name +
          "'; looked for gemm_rtp/design.json or qkv/design.json under " +
          candidates[0] + ", " + candidates[1] + " and " + candidates[2]);
  }
  // Golden check vectors. Development-only: a release ships the model and
  // the design, not the test fixtures, so their absence is normal and is only
  // an error if the golden check is actually the mode being run.

  if (maybe_tokenize(root, argc, argv)) return 0;

  // --list-models and exit: the same table --model prints on ambiguity.
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--list-models") {
      print_model_table(discover_models(root));
      return 0;
    }

  ctx.model_path = resolve_model_path(root, argc, argv);
  ctx.model = std::make_unique<npue::File>(ctx.model_path);
  set_model_shape(*ctx.model);
  g_model_name = std::filesystem::path(ctx.model_path).stem().string();

  // No --artifacts given: pick the design set from the container's own
  // geometry and adopted datapath, exactly as the `embed`/`serve`
  // subcommands do (tasks/0124, T50 -- see the comment at art_name above).
  if (art.empty()) {
    int64_t qkv_n = 0;
    try { qkv_n = ctx.model->config_int("qkv_n"); } catch (const std::exception &) {}
    std::string layout;
    try { layout = ctx.model->info("layer.0.qkv").layout_hash;
    } catch (const std::exception &) {}
    const auto *ce = npue::hub::find(g_model_name);
    const std::string want_datapath = ce ? ce->datapath : "bf16";
    art = pick_artifacts(root, g_hidden,
                         ctx.model->config_int("intermediate"),
                         config_flag(*ctx.model, "gated_ffn", false), qkv_n,
                         layout, want_datapath);
    if (art.empty())
      throw std::runtime_error(
          "no NPU design set matches " + g_model_name + " (hidden " +
          std::to_string(g_hidden) + ", datapath " + want_datapath +
          ") under " + root + " -- name one with --artifacts, or export one "
          "with tools/export_gemm_rtp.py. The old fallback to the literal "
          "'artifacts' directory is gone: it served only hidden-384 models "
          "and failed everything wider with a misleading error (T50).");
    std::printf("  artifacts  %s (picked from the container's geometry; "
                "no --artifacts given)\n", art.c_str());
  }
  ctx.art = art;

  // Fixtures live per model. The flat directory is the pre-multi-model layout
  // and is still honoured so an existing checkout keeps working; the
  // source_sha256 guard below is what makes either location safe.
  std::string val =
      root + "/runtime/artifacts/validation/" + g_model_name;
  // Whether these fixtures are THIS model's by name, or something we fell
  // back to. The distinction decides what a sha mismatch MEANS, below.
  bool val_is_own = std::ifstream(val + "/emb_sum.f32").good();
  if (!val_is_own) {
    // FIXTURES BELONG TO A CHECKPOINT, NOT TO A CONTAINER NAME (tasks/0079).
    // `bge-large-en-v1.5.int8` is the same weights as `bge-large-en-v1.5`
    // quantised, so its goldens are the same goldens -- but the directory is
    // named for the container and the lookup missed. Search the per-model
    // directories for one whose validation.json records THIS container's
    // source_sha256, which is the identity the guard below already uses.
    // Falls through to the flat pre-multi-model directory if none matches.
    namespace fs = std::filesystem;
    std::error_code vec;
    const std::string want = ctx.model->config_string("source_sha256");
    const fs::path base = fs::path(root) / "runtime" / "artifacts" / "validation";
    for (fs::directory_iterator it(base, vec), end; !vec && it != end;
         it.increment(vec)) {
      if (!it->is_directory(vec)) continue;
      std::ifstream vf(it->path() / "validation.json");
      if (!vf) continue;
      std::stringstream vb;
      vb << vf.rdbuf();
      if (npue::http::json_field_string(vb.str(), "source_sha256", "") != want)
        continue;
      if (!std::ifstream((it->path() / "emb_sum.f32").string()).good()) continue;
      val = it->path().string();
      val_is_own = true;
      std::printf("  fixtures   %s (matched by checkpoint sha, not by name)\n",
                  it->path().filename().string().c_str());
      break;
    }
  }
  if (!val_is_own) val = root + "/runtime/artifacts/validation";
  bool have_val = std::ifstream(val + "/emb_sum.f32").good();
  // THE FIXTURE MUST BELONG TO THIS MODEL.
  //
  // MiniLM-L6 and bge-small have identical hidden, heads, head_dim, ffn,
  // vocab and golden batch, so every fixture file is the same SIZE. Feeding
  // one model's fixtures to the other would compare plausible numbers against
  // the wrong target and report a pass. validation.json has carried the
  // checkpoint's sha256 since it was written and nothing ever read it --
  // which is the same shape as the six fail-open bugs before it.
  ctx.val = val;
  ctx.val_is_own = val_is_own;
  ctx.have_val = have_val;
  if (ctx.have_val) {
    std::ifstream vf(ctx.val + "/validation.json");
    if (!vf)
      throw std::runtime_error(
          "found fixtures under " + ctx.val + " but no validation.json to say "
          "which checkpoint they belong to -- re-run "
          "tools/export_validation.py");
    std::stringstream vs;
    vs << vf.rdbuf();
    const std::string want_sha =
        npue::http::json_field_string(vs.str(), "source_sha256", "");
    const std::string got_sha = ctx.model->config_string("source_sha256");
    // A MISMATCH ON THE FLAT FALLBACK IS NOT AN ERROR -- it is the answer to
    // "does this model have fixtures?", and the answer is no (0076). The flat
    // directory predates the per-model layout and holds ONE model's vectors;
    // a model added with `npuembeddings add` will never match it, and
    // refusing to `embed` because someone else's fixtures are on disk would
    // block a correct run over an unrelated file. The guard's purpose --
    // never compare against another model's answers -- is served by treating
    // them as absent. A mismatch in the model's OWN directory still throws:
    // there the fixtures claim to be this model's and are stale.
    if (!ctx.val_is_own && want_sha != got_sha) {
      ctx.have_val = false;
    } else if (want_sha.empty() || want_sha != got_sha)
      throw std::runtime_error(
          "the golden fixtures were made from checkpoint " +
          want_sha.substr(0, 16) + "... but this model is " +
          got_sha.substr(0, 16) + "... -- they would compare the right shapes "
          "against the wrong answers. Re-run tools/export_validation.py.");
  }
  std::printf("NpuEmbeddings C++ runtime -- full encode\n");
  std::printf("  bo-mode    %s (data-buffer allocation)\n", npu::bo_mode_name());
  std::printf("  model      %s: %zu tensors, %.2f MB, checkpoint %s\n",
              g_model_name.c_str(), ctx.model->tensor_count(),
              ctx.model->data_length() / 1e6,
              ctx.model->config_string("source_sha256").substr(0, 16).c_str());
  std::printf("  shape      %s: %lld layers, hidden %lld, %lld heads x %lld, "
              "ffn %lld, %s pooling\n",
              g_source_repo.c_str(), (long long)g_layers, (long long)g_hidden,
              (long long)g_heads, (long long)g_head_dim, (long long)g_ffn,
              g_cls_pool ? "CLS" : "mean");

  // --bo-mode: how data buffers are allocated. MUST be set before any Design
  // exists, because Design's constructor allocates. See npu_device.cpp for
  // what the four modes separate.
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--bo-mode") {
      const std::string m = argv[i + 1];
      if (m == "host_only") npu::set_bo_mode(npu::BoMode::host_only);
      else if (m == "host_only_1m") npu::set_bo_mode(npu::BoMode::host_only_1m);
      else if (m == "ext") npu::set_bo_mode(npu::BoMode::ext);
      else if (m == "ext_1m") npu::set_bo_mode(npu::BoMode::ext_1m);
      else throw std::runtime_error(
          "--bo-mode " + m + ": expected host_only, host_only_1m, ext or ext_1m");
    }

  ctx.dev = std::make_unique<npu::Device>();

  // The measurement probes need only the device + design set, so they
  // run before the encode path builds the Encoder. Each returns -1 when
  // its flag was not present, so the first match owns the process.
  if (int _r = maybe_probe_pair(ctx); _r >= 0) return _r;
  if (int _r = maybe_probe_bo(ctx); _r >= 0) return _r;
  if (int _r = maybe_probe_design(ctx); _r >= 0) return _r;
  if (int _r = maybe_probe_insts(ctx); _r >= 0) return _r;
  if (int _r = maybe_probe_rtp(ctx); _r >= 0) return _r;
  if (int _r = maybe_probe_ctx(ctx); _r >= 0) return _r;
  if (int _r = maybe_soak_npu(ctx); _r >= 0) return _r;
  if (int _r = maybe_soak_cpu(ctx); _r >= 0) return _r;

  // The old monolithic body is decomposed onto RunContext: design set + golden
  // vectors, host flags + pools + the Encoder, then the late command functions
  // dispatch their flag and return a process exit code (-1 == fall through).
  load_designs(ctx);
  load_goldens(ctx);
  setup_flags_pools(ctx);
  if (int _r = setup_encoder(ctx); _r != 0) return _r;
  if (int _r = maybe_probe(ctx); _r >= 0) return _r;
  if (int _r = maybe_probe_streams(ctx); _r >= 0) return _r;
  if (int _r = maybe_embed(ctx); _r >= 0) return _r;
  if (int _r = maybe_serve(ctx); _r >= 0) return _r;
  if (int _r = maybe_encode_file(ctx); _r >= 0) return _r;
  return run_bench_or_check(ctx);
} catch (const std::exception &e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 2;
}