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
#include "embed_service.hpp"
#include "gemma_npu_encoder.hpp"
#include "host_kernels.hpp"
#include "pool.hpp"

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

  int bench = 0;
  for (int i = 2; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--bench") bench = std::atoi(argv[i + 1]);

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

  const std::string model_path = resolve_model_path(root, argc, argv);
  npue::File model(model_path);
  set_model_shape(model);
  g_model_name = std::filesystem::path(model_path).stem().string();

  // No --artifacts given: pick the design set from the container's own
  // geometry and adopted datapath, exactly as the `embed`/`serve`
  // subcommands do (tasks/0124, T50 -- see the comment at art_name above).
  if (art.empty()) {
    int64_t qkv_n = 0;
    try { qkv_n = model.config_int("qkv_n"); } catch (const std::exception &) {}
    std::string layout;
    try { layout = model.info("layer.0.qkv").layout_hash;
    } catch (const std::exception &) {}
    const auto *ce = npue::hub::find(g_model_name);
    const std::string want_datapath = ce ? ce->datapath : "bf16";
    art = pick_artifacts(root, g_hidden,
                         model.config_int("intermediate"),
                         config_flag(model, "gated_ffn", false), qkv_n,
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
    const std::string want = model.config_string("source_sha256");
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
  if (have_val) {
    std::ifstream vf(val + "/validation.json");
    if (!vf)
      throw std::runtime_error(
          "found fixtures under " + val + " but no validation.json to say "
          "which checkpoint they belong to -- re-run "
          "tools/export_validation.py");
    std::stringstream vs;
    vs << vf.rdbuf();
    const std::string want_sha =
        npue::http::json_field_string(vs.str(), "source_sha256", "");
    const std::string got_sha = model.config_string("source_sha256");
    // A MISMATCH ON THE FLAT FALLBACK IS NOT AN ERROR -- it is the answer to
    // "does this model have fixtures?", and the answer is no (0076). The flat
    // directory predates the per-model layout and holds ONE model's vectors;
    // a model added with `npuembeddings add` will never match it, and
    // refusing to `embed` because someone else's fixtures are on disk would
    // block a correct run over an unrelated file. The guard's purpose --
    // never compare against another model's answers -- is served by treating
    // them as absent. A mismatch in the model's OWN directory still throws:
    // there the fixtures claim to be this model's and are stale.
    if (!val_is_own && want_sha != got_sha) {
      have_val = false;
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
              g_model_name.c_str(), model.tensor_count(),
              model.data_length() / 1e6,
              model.config_string("source_sha256").substr(0, 16).c_str());
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
  // `dev` aliases keep the not-yet-extracted tail compiling; the probe layer
  // already uses ctx.dev-> and later steps drop the alias entirely.
  npu::Device &dev = *ctx.dev;

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

  // Unified mode: art/gemm_rtp holds ONE xclbin whose four instruction
  // streams are the four GEMM shapes (tools/export_gemm_rtp.py). Every design
  // reference below binds to that one Design; the eltwise ops are forced onto
  // the host, and the encode runs in a single hw_context -- zero switches.
  const bool unified =
      std::ifstream(art + "/gemm_rtp/design.json").good();
  std::unique_ptr<npu::Design> ud;
  std::unique_ptr<npu::Design> ld_qkv, ld_ao, ld_fu, ld_fd, ld_gelu, ld_ln,
      ld_sm;
  std::vector<StreamEntry> streams;
  if (unified) {
    ud = std::make_unique<npu::Design>(dev, art + "/gemm_rtp");
    std::ifstream sj(art + "/gemm_rtp/design.json");
    std::stringstream sbuf;
    sbuf << sj.rdbuf();
    streams = parse_streams(sbuf.str());
    if (streams.empty()) {
      // A pre-0037 export: four streams, no tiers, the old flat names.
      ud->load_instr(art + "/gemm_rtp/insts_attn_out.bin");   // 1
      ud->load_instr(art + "/gemm_rtp/insts_ffn_up.bin");     // 2
      ud->load_instr(art + "/gemm_rtp/insts_ffn_down.bin");   // 3
      std::printf("  designs    ONE xclbin, 4 instruction streams, one "
                  "hw_context\n");
    } else {
      // Load in slot order and CHECK it -- a stream bound to the wrong slot
      // would compute a different shape with the right buffer sizes, which
      // is exactly the failure mode this project has hit five times.
      std::sort(streams.begin(), streams.end(),
                [](const StreamEntry &a, const StreamEntry &b) {
                  return a.slot < b.slot;
                });
      for (const auto &s : streams) {
        const size_t got = ud->load_instr(art + "/gemm_rtp/" + s.file);
        if (static_cast<int64_t>(got) != s.slot)
          throw std::runtime_error("stream " + s.file + " landed in slot " +
                                   std::to_string(got) + ", design.json says " +
                                   std::to_string(s.slot));
      }
      std::set<int64_t> tset;
      for (const auto &s : streams) tset.insert(s.batch);
      std::printf("  designs    ONE xclbin, %zu streams (%zu batch tiers), "
                  "one hw_context\n", streams.size(), tset.size());
    }
  } else {
    ld_qkv = std::make_unique<npu::Design>(dev, art + "/qkv");
    ld_ao = std::make_unique<npu::Design>(dev, art + "/attn_out");
    ld_fu = std::make_unique<npu::Design>(dev, art + "/ffn_up");
    ld_fd = std::make_unique<npu::Design>(dev, art + "/ffn_down");
    ld_gelu = std::make_unique<npu::Design>(dev, art + "/gelu");
    ld_ln = std::make_unique<npu::Design>(dev, art + "/layernorm");
    ld_sm = std::make_unique<npu::Design>(dev, art + "/softmax");
    std::printf("  designs    7 resident xclbins\n");
  }
  npu::Design &d_qkv = unified ? *ud : *ld_qkv;
  npu::Design &d_ao = unified ? *ud : *ld_ao;
  npu::Design &d_fu = unified ? *ud : *ld_fu;
  npu::Design &d_fd = unified ? *ud : *ld_fd;
  npu::Design &d_gelu = unified ? *ud : *ld_gelu;
  npu::Design &d_ln = unified ? *ud : *ld_ln;
  npu::Design &d_sm = unified ? *ud : *ld_sm;

  // WHICH DATAPATH WAS ACTUALLY SELECTED (tasks/0104), read off the loaded
  // design, never off a flag or the directory name that happened to be
  // picked -- "reports the intention, not the value" is a cost this project
  // has already paid twice (tasks/0042, 0081) for a_dtype/c_dtype; bfp16 gets
  // the same discipline from day one.
  if (!d_qkv.info().datapath_recorded)
    std::printf("  datapath   UNRECORDED (design predates tasks/0104), "
                "C as %s\n",
                d_qkv.info().c_elem_bytes == 2 ? "bf16" : "fp32");
  else
    std::printf("  datapath   %s MMAC, C as %s\n",
                d_qkv.info().emulate_bfp16 ? "bfp16-emulated" : "bf16",
                d_qkv.info().c_elem_bytes == 2 ? "bf16" : "fp32");

  // WHICH TOOLCHAIN BUILT THIS DESIGN (T39, tasks/0106) -- read off d_qkv,
  // same reasoning as the datapath line above (7-design and unified sets
  // both report their qkv design's provenance).
  if (!d_qkv.info().toolchain_recorded)
    std::printf("  toolchain  UNRECORDED (design predates tasks/0106)\n");
  else
    std::printf("  toolchain  mlir_aie %s, peano %s, mlir-aie HEAD %s\n",
                d_qkv.info().mlir_aie_version.c_str(),
                d_qkv.info().peano_version.c_str(),
                d_qkv.info().mlir_aie_git_head.c_str());

  // Batch comes from the design, not from a constant here, so a mismatch is
  // impossible rather than merely unlikely.
  // The design says what sequence length it was built for; the container
  // says how many positions it can feed. set_design_seq checks the second
  // against the first rather than trusting either alone.
  if (d_qkv.info().seq <= 0)
    throw std::runtime_error(
        "this design set records no sequence length -- re-export it with "
        "tools/export_gemm_rtp.py, or add \"seq\": 64 to its design.json if "
        "you know it was built for seq 64");
  set_design_seq(d_qkv.info().seq);

  const int64_t rows = d_qkv.info().M, batch = rows / g_seq;
  if (rows % g_seq || batch < 1)
    throw std::runtime_error("design M=" + std::to_string(rows) +
                             " is not a whole number of seq-" +
                             std::to_string(g_seq) + " sequences");
  std::printf("  shape      batch %lld x seq %lld  (M = %lld)\n",
              (long long)batch, (long long)g_seq, (long long)rows);

  // The goldens are batch 4 -- that is what M3 generated and what the accuracy
  // claim rests on. For larger batches the four sequences are TILED to fill
  // the design. That measures throughput honestly (the array does the full
  // work) -- but plain tiling (copy r's row k == the same base row k, for
  // every r) makes every physical copy of the golden batch BYTE-IDENTICAL,
  // and a row-indexing or cross-row-aliasing bug that reads the wrong copy
  // still reads identical data, so it is invisible to a comparison that only
  // checks content. This is exactly the bug class T32
  // (research/OPEN-THREADS.md) filed after tasks/0070's threaded
  // `swiglu_cpu()`: a genuine cross-row read/write race whose corruption
  // this golden gate could not see (it PASSED), caught only by a separate
  // distinct-content e2e run (`tools/verify_embed_e2e.py`) that failed at
  // worst `1-cos` 0.44.
  //
  // Fix (T32 option 1): rotate which of the 4 base sequences lands in row k
  // of copy r by `(k + r) % kGoldenBatch`, and apply the SAME rotation to
  // the expected output. This is still an exact golden -- it is a
  // relabelling of which known-good row goes where, not new data -- and it
  // is trivially invertible (row b's expected content is base row
  // `(b % kGoldenBatch + b / kGoldenBatch) % kGoldenBatch`). It costs
  // nothing extra to tile this way instead of plainly, and it turns "identical
  // content wherever it lands" into "content that must match its own row",
  // so a row-indexing or cross-row-aliasing bug now changes the answer.
  //
  // This does NOT, by itself, extend the accuracy claim past 4 distinct
  // sequences of *content* -- there are still only 4 distinct sentences in
  // the batch. What it buys is that every row of the OUTPUT is now checked
  // (see the comparison loop below) against the specific golden row it is
  // supposed to reproduce, so corruption or misrouting in any row, not just
  // rows 0-3, is visible.
  constexpr int64_t kGoldenBatch = 4;
  auto tile_rot = [kGoldenBatch](const std::vector<float> &v, int64_t reps,
                                 int64_t row_floats) {
    std::vector<float> out(static_cast<size_t>(reps * kGoldenBatch * row_floats));
    for (int64_t r = 0; r < reps; ++r)
      for (int64_t k = 0; k < kGoldenBatch; ++k) {
        const int64_t src = (k + r) % kGoldenBatch;
        std::memcpy(out.data() + static_cast<size_t>((r * kGoldenBatch + k) * row_floats),
                    v.data() + static_cast<size_t>(src * row_floats),
                    static_cast<size_t>(row_floats) * sizeof(float));
      }
    return out;
  };
  if (batch % kGoldenBatch)
    throw std::runtime_error("batch " + std::to_string(batch) +
                             " is not a multiple of the golden batch 4");
  const int64_t reps4 = batch / kGoldenBatch;

  //
  // A RELEASE ships the model and the design, not these fixtures, so when they
  // are absent the buffers are sized-but-empty and only the modes that
  // actually consume them complain. `need_goldens` is that complaint, raised
  // at the point of use so the message names the mode.
  auto need_goldens = [&]() {
    if (!have_val)
      throw std::runtime_error(
          "no golden check vectors under " + val + " -- this build can run "
          "--embed, --serve, --tokenize and --encode-file, but not the golden "
          "check or --bench. Generate them with tools/export_validation.py.");
  };
  std::vector<float> emb_in, mask, want, amask_i;
  if (have_val) {
    emb_in = tile_rot(read_f32(val + "/emb_sum.f32",
                               static_cast<size_t>(kGoldenBatch * g_seq * g_hidden)),
                      reps4, g_seq * g_hidden);
    mask = tile_rot(read_f32(val + "/add_mask.f32",
                             static_cast<size_t>(kGoldenBatch * g_seq)),
                    reps4, g_seq);
    // `want` is rotated with the IDENTICAL permutation as emb_in/mask/
    // amask_i, so row b of the output must match base golden row
    // `(b % kGoldenBatch + b / kGoldenBatch) % kGoldenBatch` -- the same
    // sequence that was actually fed into row b.
    want = tile_rot(read_f32(val + "/embedding_expected.f32",
                             static_cast<size_t>(kGoldenBatch * g_hidden)),
                    reps4, g_hidden);
    amask_i = tile_rot(read_f32(val + "/attention_mask.f32",
                                static_cast<size_t>(kGoldenBatch * g_seq)),
                       reps4, g_seq);
  } else {
    // The Encoder needs a mask of the right shape at construction; every
    // other mode overwrites it per chunk before dispatching.
    mask.assign(static_cast<size_t>(rows), 0.f);
  }

  // --threads controls the attention pool only; everything else is one thread.
  // Default 1 keeps the "0.2 cores busy" claim of tasks/0023 intact by default,
  // so turning it up is an explicit trade of cores for wall clock -- which is
  // the trade the CPU baseline already makes with 12-17 of them.
  int nthreads = 1;
  for (int i = 2; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--threads") nthreads = std::atoi(argv[i + 1]);
  bool host_ln = false;
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == "--host-ln") host_ln = true;
  bool host_sm = false;
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == "--host-sm") host_sm = true;
  bool host_gelu = false;
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == "--host-gelu") host_gelu = true;
  bool sim_c_bf16 = false;
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == "--sim-c-bf16") sim_c_bf16 = true;
  bool no_fuse_ffn = false;
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == "--no-fuse-ffn") no_fuse_ffn = true;
  int pipeline = 0;                 // 0 = off; N = N concurrent lanes
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == "--pipeline") {
      pipeline = 2;
      if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(
                              argv[i + 1][0])))
        pipeline = std::atoi(argv[i + 1]);
    }
  // Pipelining splits the thread budget: each lane gets its own pool, so no
  // lane can stall another's host work.
  std::vector<std::unique_ptr<Pool>> pools;
  if (pipeline > 1) {
    for (int l = 0; l < pipeline; ++l)
      pools.push_back(std::make_unique<Pool>(std::max(1, nthreads / pipeline)));
  } else {
    pools.push_back(std::make_unique<Pool>(nthreads));
  }
  Pool &pool = *pools[0];

  Encoder enc{model,  d_qkv, d_ao, d_fu, d_fd, d_gelu, d_ln, d_sm, mask};
  enc.batch = batch;
  enc.rows = rows;
  enc.pool = &pool;
  if (unified) {
    // The unified artifact has no eltwise designs by construction.
    host_ln = host_sm = host_gelu = true;
    enc.unified = true;
    enc.is_qkv = 0;
    enc.is_ao = 1;
    enc.is_fu = 2;
    enc.is_fd = 3;
    if (!streams.empty()) {
      std::set<int64_t> tset;
      for (const auto &s : streams) tset.insert(s.batch);
      for (int64_t b : tset) {
        std::array<size_t, 4> slots{};
        bool complete = true;
        const char *ops[4] = {"qkv", "attn_out", "ffn_up", "ffn_down"};
        for (int k = 0; k < 4; ++k) {
          auto it = std::find_if(streams.begin(), streams.end(),
                                 [&](const StreamEntry &s) {
                                   return s.batch == b && s.op == ops[k];
                                 });
          if (it == streams.end()) { complete = false; break; }
          slots[k] = static_cast<size_t>(it->slot);
        }
        if (!complete) continue;         // a tier missing an op is not a tier
        enc.tiers.push_back(b);
        enc.tier_slots.push_back(slots);
      }
      enc.use_tier(batch);
      std::printf("  tiers      ");
      for (size_t i = 0; i < enc.tiers.size(); ++i)
        std::printf("%s%lld", i ? ", " : "", (long long)enc.tiers[i]);
      std::printf("  (requests are right-sized, not padded)\n");
    }
  }
  enc.host_ln = host_ln;
  enc.host_sm = host_sm;
  enc.host_gelu = host_gelu;
  enc.sim_c_bf16 = sim_c_bf16;
  enc.fuse_ffn_epilogue = !no_fuse_ffn;
  if (sim_c_bf16) {
    // Say so loudly, and refuse where it would mean nothing -- a status line
    // that reports the intention rather than the value is this project's
    // recurring fail-open (tasks/0042's `tile (64, 32)`).
    if (enc.qkv.info().a_elem_bytes != 1) {
      std::fprintf(stderr,
                   "--sim-c-bf16 is only meaningful on an int8 design "
                   "(this one carries %d-byte operands)\n",
                   (int)enc.qkv.info().a_elem_bytes);
      return 2;
    }
    if (enc.qkv.info().c_elem_bytes == 2) {
      std::fprintf(stderr,
                   "--sim-c-bf16 simulates a narrowed-C design; this design "
                   "already narrows C on the core, so the flag would only "
                   "round a second time\n");
      return 2;
    }
    std::printf("  SIMULATION int32 C rounded to bf16 before dequantisation --\n"
                "             prices a narrowed-C design; NOT a shipped path\n");
  }
  if (host_gelu)
    std::printf("  gelu       on the HOST (fp32) -- %lld fewer NPU dispatches\n",
                (long long)g_layers);
  if (host_sm)
    std::printf("  softmax    on the HOST (fp32) -- %lld fewer NPU dispatches\n",
                (long long)g_layers);
  if (host_ln)
    // One before the layer stack plus two per layer.
    std::printf("  layernorm  on the HOST (fp32) -- %lld fewer NPU dispatches\n",
                (long long)(1 + 2 * g_layers));
  const size_t staged = enc.stage_all();
  // What the allocation mode actually bought, in addresses. Printed
  // because "1 MB padding gives large-page backing" is a mechanism
  // claim, and the alignment is the only visible part of it.
  std::printf("  bo-align   last data buffer aligned to %zu B%s\n",
              npu::last_bo_alignment(),
              npu::last_bo_alignment() >= (1u << 21) ? " (>= 2 MB)" : "");
  std::printf("  weights    %.2f MB staged on the device once, not per call\n",
              staged / 1e6);

  static std::mutex npu_mutex;
  std::vector<std::unique_ptr<Encoder>> lanes;   // lanes[0] aliases enc below
  if (pipeline > 1) {
    if (!unified)
      throw std::runtime_error(
          "--pipeline requires the unified gemm_rtp artifact");
    enc.npu_mu = &npu_mutex;
    for (int l = 1; l < pipeline; ++l) {
      lanes.push_back(std::make_unique<Encoder>(
          Encoder{model, d_qkv, d_ao, d_fu, d_fd, d_gelu, d_ln, d_sm, mask}));
      Encoder &e2 = *lanes.back();
      e2.batch = batch;
      e2.rows = rows;
      e2.pool = pools[l].get();
      e2.unified = true;
      e2.is_qkv = 0; e2.is_ao = 1; e2.is_fu = 2; e2.is_fd = 3;
      e2.host_ln = e2.host_sm = e2.host_gelu = true;
      e2.sim_c_bf16 = enc.sim_c_bf16;
      e2.fuse_ffn_epilogue = enc.fuse_ffn_epilogue;
      // The staged weights and parameters are the design's, not a lane's.
      e2.s_qkv = enc.s_qkv; e2.s_ao = enc.s_ao;
      e2.s_fu = enc.s_fu; e2.s_fd = enc.s_fd;
      e2.b_qkv = enc.b_qkv; e2.b_ao = enc.b_ao;
      e2.b_fu = enc.b_fu; e2.b_fd = enc.b_fd;
      // int8 scales are the DESIGN's and the CONTAINER's, not a lane's --
      // same reasoning as the staged weights above, and the same failure if
      // forgotten: lane 0 worked, lanes 1+ dereferenced a null wscale and the
      // process segfaulted only under --pipeline (tasks/0078).
      e2.ws_qkv = enc.ws_qkv; e2.ws_ao = enc.ws_ao;
      e2.ws_fu = enc.ws_fu;   e2.ws_fd = enc.ws_fd;
      e2.as_qkv = enc.as_qkv; e2.as_ao = enc.as_ao;
      e2.as_fu = enc.as_fu;   e2.as_fd = enc.as_fd;
      e2.s_ln = enc.s_ln; e2.h_gamma = enc.h_gamma; e2.h_beta = enc.h_beta;
      // The tier table is POLICY, and every lane needs it. A lane without it
      // silently falls back to the pre-0037 flat slot contract (0,1,2,3),
      // which under the 16-stream export selects the wrong shapes entirely --
      // measured as 1-cos 1.0 on whichever chunk that lane happened to take.
      e2.tiers = enc.tiers;
      e2.tier_slots = enc.tier_slots;
      e2.use_tier(batch);
      // Each extra lane gets its own A and C buffers on the shared design;
      // lane 0 keeps the base slots.
      e2.slot_a = d_qkv.stage_alloc(0, d_qkv.info().buffer_bytes[0]);
      e2.slot_c = d_qkv.stage_alloc(2, d_qkv.info().buffer_bytes[2]);
      e2.npu_mu = &npu_mutex;
    }
    for (const auto &lp : lanes) {
      if (lp->tiers != enc.tiers || lp->tier_slots.size() != enc.tier_slots.size())
        throw std::runtime_error(
            "lane stream policy differs from lane 0 -- refusing to run, "
            "because the lanes would compute different things");
    }
    std::printf("  pipeline   %d concurrent encodes of %lld, one NPU mutex, "
                "%d host threads per lane\n", pipeline, (long long)batch,
                pools[0]->size());
  }

  auto pool_and_normalise = [&](const std::vector<float> &h) {
    std::vector<float> out(batch * g_hidden, 0.f);
    pool_rows(h.data(), amask_i.data(), batch, out.data());
    return out;
  };

  // --probe answers one question: is the ~1300 us per dispatch the array doing
  // work, or the driver swapping designs in and out?
  //
  // Seven designs are resident in seven hw_contexts on an eight-column NPU.
  // They cannot all be configured at once, so if the driver reconfigures on
  // every dispatch, repeating ONE design should be fast and alternating between
  // two should be slow. If both are the same, the hypothesis is dead and the
  // time really is the array.
  //
  // Nothing here checks results -- it dispatches on whatever is in the buffers.
  // That is deliberate: it isolates dispatch cost from everything else.
  for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == "--probe") {
    const int reps = 100;
    std::printf("\n  dispatch probe -- %d repeats, no host work in the loop\n",
                reps);
    std::printf("    same design repeatedly:\n");
    for (npu::Design *d : {&d_qkv, &d_ao, &d_fu, &d_fd, &d_gelu, &d_ln,
                           &d_sm}) {
      d->dispatch_only();                       // warm this context in
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) d->dispatch_only();
      double us = (now_s() - t0) / reps * 1e6;
      std::printf("      %-12s %8.0f us\n", d->info().name.c_str(), us);
    }
    std::printf("    alternating between two designs:\n");
    struct Pair { npu::Design *a, *b; const char *label; };
    for (Pair p : {Pair{&d_qkv, &d_fu, "qkv <-> ffn_up"},
                   Pair{&d_qkv, &d_gelu, "qkv <-> gelu"},
                   Pair{&d_ln, &d_sm, "layernorm <-> softmax"}}) {
      p.a->dispatch_only();
      p.b->dispatch_only();
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) { p.a->dispatch_only();
                                       p.b->dispatch_only(); }
      double us = (now_s() - t0) / (2 * reps) * 1e6;
      std::printf("      %-22s %8.0f us\n", p.label, us);
    }
    return 0;
  }

  // --probe-streams: what IS the GEMM's per-dispatch time made of?
  //
  // tasks/0048 / OPEN-THREADS T1. `--bench` reports ONE wait figure averaged
  // over all four shapes, which cannot distinguish the two candidate accounts:
  //
  //   compute-bound  -> time tracks MACs
  //   traffic-bound  -> time tracks bytes moved (tasks/0010's model)
  //
  // The four shapes have deliberately different ratios -- ffn_up and ffn_down
  // have IDENTICAL MACs and differ 1.5x in traffic, which is the discriminating
  // pair -- so timing them separately decides it.
  //
  // No host work in the loop and no result checking: it dispatches whatever is
  // in the buffers. That is the point. Any host term would be the thing we are
  // trying to see past.
  for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == "--probe-streams") {
    if (!unified || streams.empty())
      throw std::runtime_error("--probe-streams needs a unified gemm_rtp set");
    const int reps = 30;
    const size_t cb = d_qkv.info().c_elem_bytes;
    // A/B element size and the N-tiling READ from the loaded design, exactly
    // as cb above is (T47, tasks/0124). This block used to hardcode 2 and
    // 48.0*8.0, which inflated every published int8 GB/s by 1.57-1.85x --
    // differentially, because C was counted correctly. A design.json that
    // predates the tile_n/cols fields is a refusal, not a guess.
    const size_t ab = d_qkv.info().a_elem_bytes;
    const int64_t tile_n = d_qkv.info().tile_n, cols = d_qkv.info().cols;
    if (tile_n <= 0 || cols <= 0)
      throw std::runtime_error(
          "--probe-streams: this design.json records no tile_n/cols -- it "
          "predates the fields. Re-export the set (tools/export_gemm_rtp.py); "
          "refusing to substitute a guess (T47).");
    const int64_t mrows = 4, tm = 64;      // design rows, tile m
    std::printf("\n  probe-streams -- %d repeats, no host work, A/B %s, "
                "C %s, tile_n %lld x cols %lld\n",
                reps, ab == 1 ? "i8" : "bf16", cb == 2 ? "bf16" : "fp32",
                (long long)tile_n, (long long)cols);
    std::printf("    %-10s %6s %6s %6s  %8s  %8s  %9s  %8s  %8s\n",
                "stream", "M", "K", "N", "GMAC", "MB", "us/disp",
                "GMAC/ms", "GB/s");
    // ALL tiers, not just the top one (T45, tasks/0128): the four batch
    // tiers give an M-sweep 256 -> 8192 on identical geometry, which is
    // exactly the intercept measurement the fixed-cost fits (0010: 150 us,
    // 0048: 573 us, 0080: 627 us) disagreed about. Timing-only -- the
    // buffers hold whatever is staged; a dispatch reads the same bytes
    // regardless of their values.
    for (const auto &st : streams) {
      d_qkv.bind_instr(static_cast<size_t>(st.slot));
      d_qkv.dispatch_only();                       // warm
      const double t0 = now_s();
      for (int r = 0; r < reps; ++r) d_qkv.dispatch_only();
      const double us = (now_s() - t0) / reps * 1e6;
      // tasks/0010's traffic accounting: A re-streamed once per n-block group,
      // B once per row block, C once.
      const double nb_groups =
          std::max(1.0, double(st.N) / double(tile_n * cols));
      const double row_blocks = double(st.M) / double(tm) / double(mrows);
      const double mb = (double(st.M) * st.K * double(ab) * nb_groups
                         + double(st.K) * st.N * double(ab) * row_blocks
                         + double(st.M) * st.N * cb) / 1e6;
      const double gmac = double(st.M) * st.K * st.N / 1e9;
      std::printf("    %-10s %6lld %6lld %6lld  %8.2f  %8.1f  %9.0f  %8.2f  %8.1f\n",
                  st.op.c_str(), (long long)st.M, (long long)st.K,
                  (long long)st.N, gmac, mb, us, gmac / (us / 1000.0),
                  mb / 1e3 / (us / 1e6));
    }
    std::printf("\n\n    Read the LAST TWO COLUMNS. If GMAC/ms is flat across shapes the"
                " design is compute-bound; if GB/s is flat it is traffic-bound;"
                " if neither, it is something we have not modelled"
                " (OPEN-THREADS T1).\n");
    return 0;
  }

  // TEXT IN, VECTORS OUT. One service, used by --embed (batch, from a file)
  // and --serve (an OpenAI-shaped HTTP endpoint). Sharing it is the point:
  // the endpoint cannot drift from the thing the tests measure.
  struct EmbedService {
    AnyTokenizer tok;
    const float *w_word, *w_pos, *w_typ;
    Encoder *lead;
    std::vector<Encoder *> all;
    int64_t fallback_batch;

    // Greedy against the tier ladder: 64 texts with tiers {4,16,32,128}
    // becomes 32+32, both exact, instead of one half-padded 128.
    std::vector<std::pair<int64_t, int64_t>> plan(int64_t n) const {
      std::vector<std::pair<int64_t, int64_t>> jobs;
      int64_t base = 0;
      while (base < n) {
        const int64_t left = n - base;
        int64_t take = lead->tiers.empty() ? std::min(fallback_batch, left) : 0;
        for (int64_t tr : lead->tiers)
          if (tr <= left && tr > take) take = tr;
        if (take == 0)
          take = lead->tiers.empty() ? left
                                     : std::min(left, lead->tiers.front());
        jobs.emplace_back(base, take);
        base += take;
      }
      return jobs;
    }

    // `prefix_text` is the literal text to prepend, "" for none. It is an
    // ARGUMENT rather than a member (tasks/0118) because --serve now takes the
    // prompt per request: holding it as state is what made one server able to
    // answer only one kind of query. Prepended to the RAW text before
    // tokenization, the same place tools/verify_embed_e2e.py does it, so the
    // two agree on what "applying a prefix" means.
    void chunk(Encoder &e, const std::vector<std::string> &texts,
               int64_t base, int64_t take, const std::string &prefix_text,
               std::vector<float> &out, int64_t *tokens) const {
      const size_t row_floats = static_cast<size_t>(g_seq) * g_hidden;
      const int64_t bt = e.use_tier(take);
      std::vector<float> buf(static_cast<size_t>(bt) * row_floats, 0.f);
      std::vector<float> cmask(static_cast<size_t>(bt) * g_seq, -1.0e30f);
      std::vector<float> cam(static_cast<size_t>(bt) * g_seq, 0.f);
      int64_t ntok = 0;
      for (int64_t b = 0; b < take; ++b) {
        const auto en = prefix_text.empty()
            ? tok.encode(texts[base + b], static_cast<int>(g_seq))
            : tok.encode(prefix_text + texts[base + b],
                        static_cast<int>(g_seq));
        // `base + b` is the caller's own index. Tiers are an implementation
        // detail of how this runtime batches, and naming a tier-local row
        // would send someone looking at the wrong text.
        check_truncation(en.truncated, en.n_tokens_full,
                         static_cast<size_t>(base + b), g_seq);
        ntok += en.n_tokens;
        for (int64_t s = 0; s < g_seq; ++s) {
          const int32_t id = en.input_ids[s];
          const float m = static_cast<float>(en.attention_mask[s]);
          cam[b * g_seq + s] = m;
          cmask[b * g_seq + s] = m > 0 ? 0.f : -1.0e30f;
          float *dst = buf.data() + (b * g_seq + s) * g_hidden;
          const float *wv = w_word + static_cast<size_t>(id) * g_hidden;
          const float *pv = w_pos + static_cast<size_t>(s) * g_hidden;
          for (int64_t c = 0; c < g_hidden; ++c)
            dst[c] = wv[c] + pv[c] + w_typ[c];
        }
      }
      e.add_mask = cmask;
      auto h = e.run(buf);
      pool_rows(h.data(), cam.data(), take, out.data() + base * g_hidden);
      if (tokens) *tokens += ntok;
    }

    std::vector<float> embed(const std::vector<std::string> &texts,
                             const std::string &prefix_text,
                             int64_t *tokens = nullptr) {
      std::vector<float> out(texts.size() * g_hidden, 0.f);
      const auto jobs = plan(static_cast<int64_t>(texts.size()));
      std::atomic<int64_t> tok_total{0};
      if (all.size() > 1 && jobs.size() > 1) {
        std::atomic<size_t> next{0};
        std::vector<std::thread> ts;
        // chunk() can throw -- npue::InputTooLong on a caller's bad input, or
        // anything e.run() raises on a device error -- and an exception that
        // escapes a std::thread's entry point calls std::terminate. This
        // branch had no handler, which was survivable only for as long as
        // nothing on the path threw. Capture the first, stop handing out work,
        // rethrow on the joining thread.
        std::mutex emu;
        std::exception_ptr first_err;
        std::atomic<bool> stop{false};
        auto worker = [&](Encoder *e) {
          for (size_t j = next++; j < jobs.size(); j = next++) {
            if (stop.load(std::memory_order_relaxed)) return;
            try {
              int64_t nt = 0;
              chunk(*e, texts, jobs[j].first, jobs[j].second, prefix_text,
                    out, &nt);
              tok_total += nt;
            } catch (...) {
              // First one wins. Which job reports first is a thread race, so
              // for a request with several oversized inputs the index named is
              // whichever lane got there -- deliberately not "the lowest",
              // because pretending to a determinism the scheduler does not
              // provide would be the worse lie. The caller has to fix all of
              // them regardless.
              std::lock_guard<std::mutex> lk(emu);
              if (!first_err) first_err = std::current_exception();
              stop.store(true, std::memory_order_relaxed);
              return;
            }
          }
        };
        for (size_t l = 1; l < all.size(); ++l)
          ts.emplace_back([&, l] { worker(all[l]); });
        worker(lead);
        for (auto &th : ts) th.join();
        if (first_err) std::rethrow_exception(first_err);
      } else {
        for (const auto &j : jobs) {
          int64_t nt = 0;
          chunk(*lead, texts, j.first, j.second, prefix_text, out, &nt);
          tok_total += nt;
        }
      }
      if (tokens) *tokens = tok_total.load();
      return out;
    }
  };

  auto make_service = [&]() {
    // No prefix is resolved here any more (tasks/0118). --embed resolves one
    // from --prefix; --serve takes the name per request. Resolving it at
    // construction is what coupled a whole server process to one prompt.
    EmbedService svc{load_tokenizer(model, model_path),
                     model.raw("embeddings.word").as<float>(),
                     model.raw("embeddings.position").as<float>(),
                     model.raw("embeddings.token_type").as<float>(),
                     &enc, {}, batch};
    svc.all.push_back(&enc);
    for (auto &lp : lanes) svc.all.push_back(lp.get());
    return svc;
  };

  // --embed <textfile> [outfile]
  for (int i = 2; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--embed") {
    const std::string in_path = argv[i + 1];
    std::string out_path;
    if (i + 2 < argc && argv[i + 2][0] != '-') out_path = argv[i + 2];

    auto svc = make_service();
    // --prefix is REQUIRED here for a model that has a prompts table, and
    // there is no container default any more -- see resolve_prefix().
    const std::string prefix_text = resolve_prefix(argc, argv);
    std::printf("  tokenizer  %zu tokens, from the .npue\n",
                svc.tok.vocab_size());

    std::vector<std::string> texts;
    {
      std::ifstream in(in_path, std::ios::binary);
      if (!in) throw std::runtime_error("cannot open " + in_path);
      std::string line;
      while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        texts.push_back(line);
      }
    }
    std::printf("  input      %zu texts\n", texts.size());

    const double t0 = now_s();
    auto out = svc.embed(texts, prefix_text);
    const double el = now_s() - t0;
    std::printf("  embedded   %zu texts in %.2f s  ->  %.1f seq/s\n",
                texts.size(), el, texts.size() / el);

    if (!out_path.empty()) {
      std::ofstream of(out_path, std::ios::binary);
      of.write(reinterpret_cast<const char *>(out.data()),
               out.size() * sizeof(float));
      if (!of) throw std::runtime_error("failed writing " + out_path);
      std::printf("  wrote      %s  [%zu, %lld] fp32\n", out_path.c_str(),
                  texts.size(), (long long)g_hidden);
    } else {
      for (size_t b = 0; b < std::min<size_t>(texts.size(), 4); ++b) {
        std::printf("  [%zu]", b);
        for (int64_t c = 0; c < 6; ++c)
          std::printf(" %+.4f", out[b * g_hidden + c]);
        std::printf(" ...\n");
      }
    }
    return 0;
  }

  // --serve [port]: an OpenAI-shaped POST /v1/embeddings endpoint.
  //
  // Requests are handled ONE AT A TIME on purpose. The NPU serializes
  // dispatches anyway (research/notes/0004), and the lanes already
  // parallelise inside a single request -- so concurrent request handling
  // would add contention and lock complexity to buy nothing. Throughput comes
  // from batching within a request, which is what an embeddings client does.
  for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == "--serve") {
    int port = 8080;
    if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0])))
      port = std::atoi(argv[i + 1]);
    std::string bind_addr = "127.0.0.1";
    for (int k = 2; k < argc - 1; ++k)
      if (std::string(argv[k]) == "--bind") bind_addr = argv[k + 1];

    // --prefix is a PROCESS-WIDE setting and `serve` no longer has one
    // (tasks/0118). Refusing beats ignoring: a script that used to pin a
    // prefix here would otherwise keep running and quietly serve unprefixed
    // vectors, which is the same fail-open shape this task exists to remove.
    for (int k = 2; k < argc; ++k)
      if (std::string(argv[k]) == "--prefix")
        throw std::runtime_error(
            "--prefix does not apply to `serve`: the task prompt is chosen per "
            "request now. Send \"prompt_name\" in the POST body instead, and GET "
            "/health lists the names this model accepts.");

    auto svc = make_service();
    EmbedBackend be;
    be.vocab_size = svc.tok.vocab_size();
    be.prompt_names = prompt_names_sorted();
    be.hidden = g_hidden;
    be.seq = g_seq;
    // By reference: `svc` outlives serve_http(), which runs the accept loop
    // and only returns when the server stops. The name has already been
    // checked against be.prompt_names, so a miss here can only be the \"\" that
    // means no prefix at all.
    be.embed = [&svc](const std::vector<std::string> &t, const std::string &pn,
                      int64_t *n) {
      const auto it = g_prompts.find(pn);
      return svc.embed(t, it == g_prompts.end() ? std::string() : it->second,
                       n);
    };
    return serve_http(be, g_model_name + "-npu", port, bind_addr);
  }

  // --encode-file <dir>: encode arbitrary prepared inputs and write the
  // pooled, L2-normalised embeddings back. This is the bridge that lets MTEB
  // (Python, .venv-ref) drive the C++ NPU runtime (tasks/0035).
  //
  //   <dir>/emb_sum.f32         [n_rows, g_seq, g_hidden]  fp32
  //   <dir>/add_mask.f32        [n_rows, g_seq]           fp32, 0 or -1e30
  //   <dir>/attention_mask.f32  [n_rows, g_seq]           fp32, 1 or 0
  //   <dir>/out.f32             [n_rows, g_hidden]        fp32   (written)
  //
  // n_rows need not be a multiple of the design's batch: the last chunk is
  // PADDED with zero rows, which are then discarded. A padded row is masked
  // out of its own pooling and cannot influence any other row -- every op in
  // the encoder is row-independent except attention, which is per (batch,
  // head) and therefore also row-independent across sequences.
  for (int i = 2; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--encode-file") {
    const std::string dir = argv[i + 1];
    std::ifstream f(dir + "/emb_sum.f32", std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + dir + "/emb_sum.f32");
    const size_t bytes = static_cast<size_t>(f.tellg());
    const size_t row_floats = static_cast<size_t>(g_seq) * g_hidden;
    if (bytes % (row_floats * sizeof(float)))
      throw std::runtime_error("emb_sum.f32 is not a whole number of "
                               "seq-by-hidden rows");
    const int64_t n_rows = static_cast<int64_t>(bytes /
                                                (row_floats * sizeof(float)));
    auto all_emb = read_f32(dir + "/emb_sum.f32", n_rows * row_floats);
    auto all_add = read_f32(dir + "/add_mask.f32", n_rows * g_seq);
    auto all_am = read_f32(dir + "/attention_mask.f32", n_rows * g_seq);

    std::printf("  encode-file %lld sequences in chunks of %lld\n",
                (long long)n_rows, (long long)batch);
    std::vector<float> out(static_cast<size_t>(n_rows) * g_hidden, 0.f);

    const double t0 = now_s();
    for (int64_t base = 0; base < n_rows; base += batch) {
      const int64_t take = std::min<int64_t>(batch, n_rows - base);
      // Pad the tail chunk with zero rows; they are masked and discarded.
      std::vector<float> chunk(static_cast<size_t>(batch) * row_floats, 0.f);
      std::memcpy(chunk.data(), all_emb.data() + base * row_floats,
                  take * row_floats * sizeof(float));
      std::vector<float> cmask(static_cast<size_t>(batch) * g_seq, -1.0e30f);
      std::memcpy(cmask.data(), all_add.data() + base * g_seq,
                  take * g_seq * sizeof(float));
      std::vector<float> cam(static_cast<size_t>(batch) * g_seq, 0.f);
      std::memcpy(cam.data(), all_am.data() + base * g_seq,
                  take * g_seq * sizeof(float));

      enc.add_mask = cmask;
      auto h = enc.run(chunk);

      // Pool and normalise -- the SAME function the production path uses,
      // which the comment here used to claim while calling different code.
      pool_rows(h.data(), cam.data(), take, out.data() + base * g_hidden);
    }
    const double el = now_s() - t0;

    std::ofstream of(dir + "/out.f32", std::ios::binary);
    of.write(reinterpret_cast<const char *>(out.data()),
             out.size() * sizeof(float));
    if (!of) throw std::runtime_error("failed writing " + dir + "/out.f32");
    std::printf("  encode-file done: %.2f s  ->  %.1f seq/s  (wrote out.f32)\n",
                el, n_rows / el);
    return 0;
  }

  if (bench > 0) need_goldens();

  // A timed run REFUSES to start when the array is not ours. tasks/0044 read
  // 221.4 seq/s against a true 694.0 because a leftover npuembed.exe from an
  // earlier session still held an Active hw_context, and nothing in this
  // banner said so. See include/npu_contention.hpp.
  if (bench > 0) {
    bool allow_contention = false;
    for (int i = 2; i < argc; ++i)
      if (std::string(argv[i]) == "--allow-contention") allow_contention = true;
    if (!npu::require_exclusive_npu(npu::survey_contexts(), allow_contention))
      return 2;
  }

  if (bench > 0 && pipeline > 1) {
    auto run_all = [&] {
      std::vector<std::thread> ts;
      for (auto &lp : lanes)
        ts.emplace_back([&, e = lp.get()] { e->run(emb_in); });
      enc.run(emb_in);
      for (auto &th : ts) th.join();
    };
    run_all();                                    // warm every lane
    enc.reset_timers();
    for (auto &lp : lanes) lp->reset_timers();
    d_qkv.t_submit = d_qkv.t_wait = 0.0;
    d_qkv.n_dispatch = 0;
    double w0 = now_s(), c0 = cpu_seconds();
    for (int i = 0; i < bench; ++i) run_all();
    double w1 = now_s(), c1 = cpu_seconds();
    const double wall = (w1 - w0) / bench, cpu = (c1 - c0) / bench;
    const int64_t seqs = pipeline * batch;
    std::printf("\n  %d pipelined groups of %d x %lld sequences at seq "
                "%lld\n", bench, pipeline, (long long)batch, (long long)g_seq);
    std::printf("    wall %8.2f ms   ->  %8.1f seq/s\n", wall * 1e3,
                seqs / wall);
    std::printf("    cpu  %8.2f ms   ->  %8.2f cores busy\n", cpu * 1e3,
                cpu / wall);
    const double npu_locked =
        (d_qkv.t_submit + d_qkv.t_wait) / bench;
    std::printf("    NPU dispatch+wait (serialized) %8.2f ms  %5.1f%%   "
                "%d dispatches/group\n",
                npu_locked * 1e3, npu_locked / wall * 100,
                d_qkv.n_dispatch / bench);
    auto lane = [&](int idx, const Encoder &e) {
      const double host = (e.t_conv + e.t_bias + e.t_attn + e.t_hostln +
                           e.t_hostsm + e.t_hostgelu) / bench;
      std::printf("    p%d host work                %8.2f ms  %5.1f%%   "
                  "(conv %.1f  bias %.1f  attn %.1f  elt %.1f)\n",
                  idx, host * 1e3, host / wall * 100, e.t_conv / bench * 1e3,
                  e.t_bias / bench * 1e3, e.t_attn / bench * 1e3,
                  (e.t_hostln + e.t_hostsm + e.t_hostgelu) / bench * 1e3);
    };
    lane(1, enc);
    for (size_t l = 0; l < lanes.size(); ++l)
      lane(static_cast<int>(l) + 2, *lanes[l]);
    return 0;
  }

  if (bench > 0) {
    // Unique designs only: in unified mode all seven references alias ONE
    // Design, and summing it seven times reported 241% of wall.
    std::vector<npu::Design *> uniq;
    for (npu::Design *d : {&d_qkv, &d_ao, &d_fu, &d_fd, &d_gelu, &d_ln, &d_sm})
      if (std::find(uniq.begin(), uniq.end(), d) == uniq.end())
        uniq.push_back(d);
    enc.run(emb_in);                               // warm
    enc.reset_timers();
    for (npu::Design *d : uniq)
      { d->t_submit = d->t_wait = 0.0; d->n_dispatch = 0; }
    double w0 = now_s();
    double c0 = cpu_seconds();
    for (int i = 0; i < bench; ++i) enc.run(emb_in);
    double w1 = now_s();
    double c1 = cpu_seconds();
    double wall = (w1 - w0) / bench, cpu = (c1 - c0) / bench;
    std::printf("\n  %d encodes of %lld sequences at seq %lld\n", bench,
                (long long)batch, (long long)g_seq);
    std::printf("    wall %8.2f ms   ->  %8.1f seq/s\n", wall * 1e3,
                batch / wall);
    std::printf("    cpu  %8.2f ms   ->  %8.2f cores busy\n", cpu * 1e3,
                cpu / wall);

    // A single number says "slow". This says which half to fix.
    const double conv = enc.t_conv / bench, in = enc.t_in / bench;
    const double disp = enc.t_disp / bench, out = enc.t_out / bench;
    const double bias = enc.t_bias / bench;
    const double npu = conv + in + disp + out + bias;
    const double attn = enc.t_attn / bench;
    const int nd = enc.n_dispatch / bench;
    std::printf("\n    NPU path (copy+sync+dispatch) %8.2f ms  %5.1f%%   "
                "%d dispatches\n",
                npu * 1e3, npu / wall * 100, nd);
    std::printf("      bf16 convert (both ways)    %8.2f ms  %5.1f%%\n",
                conv * 1e3, conv / wall * 100);
    std::printf("      sync to device              %8.2f ms  %5.1f%%\n",
                in * 1e3, in / wall * 100);
    std::printf("      dispatch + wait             %8.2f ms  %5.1f%%   "
                "%6.0f us each\n",
                disp * 1e3, disp / wall * 100, disp / nd * 1e6);
    {
      double sub = 0, wt = 0;
      for (npu::Design *d : uniq) {
        sub += d->t_submit;
        wt += d->t_wait;
      }
      sub /= bench;
      wt /= bench;
      std::printf("        submit (build + start)    %8.2f ms  %5.1f%%   "
                  "%6.0f us each\n", sub * 1e3, sub / wall * 100,
                  sub / nd * 1e6);
      std::printf("        wait (hardware)           %8.2f ms  %5.1f%%   "
                  "%6.0f us each\n", wt * 1e3, wt / wall * 100, wt / nd * 1e6);
    }
    std::printf("      sync from device            %8.2f ms  %5.1f%%\n",
                out * 1e3, out / wall * 100);
    std::printf("      read out + bias             %8.2f ms  %5.1f%%\n",
                bias * 1e3, bias / wall * 100);
    std::printf("    host attention (QK^T, A.V)   %8.2f ms  %5.1f%%"
                "   (qk %.1f  av %.1f)\n",
                attn * 1e3, attn / wall * 100,
                enc.t_qk / bench * 1e3, enc.t_av / bench * 1e3);
    if (enc.t_hostgelu > 0.0)
      std::printf("    host gelu                    %8.2f ms  %5.1f%%\n",
                  enc.t_hostgelu / bench * 1e3,
                  enc.t_hostgelu / bench / wall * 100);
    if (enc.t_hostsm > 0.0)
      std::printf("    host softmax                 %8.2f ms  %5.1f%%\n",
                  enc.t_hostsm / bench * 1e3,
                  enc.t_hostsm / bench / wall * 100);
    if (enc.t_hostln > 0.0)
      std::printf("    host layernorm               %8.2f ms  %5.1f%%\n",
                  enc.t_hostln / bench * 1e3,
                  enc.t_hostln / bench / wall * 100);
    // The three host eltwise kernels have their own lines above, so they must
    // come OUT of the residual bucket -- without this they were counted twice
    // and "everything else" read 24.4% on bge-large where the truth is 12.8%
    // (tasks/0081). An over-stated unexplained bucket is the worst kind of
    // wrong number: it points optimisation at a phantom.
    const double named = enc.t_hostgelu / bench + enc.t_hostsm / bench
                       + enc.t_hostln / bench;
    const double rest = wall - npu - attn - named;
    std::printf("    everything else              %8.2f ms  %5.1f%%"
                "   (residual adds, pooling, embedding lookup)\n",
                rest * 1e3, rest / wall * 100);

    // Per design: if wait() is real hardware time it must scale with the work,
    // and these seven differ by 24x in MACs. If it does not scale, the number
    // is the wait path, not the array.
    std::printf("\n    per design      calls   MACs/call    wait us/call\n");
    for (npu::Design *d : uniq) {
      const auto &in = d->info();
      const double macs = (in.kind == "gemm")
                              ? double(in.M) * in.K * in.N : 0.0;
      std::printf("    %-14s %6d  %10.3g    %10.0f\n", in.name.c_str(),
                  d->n_dispatch / bench, macs,
                  d->t_wait / d->n_dispatch * 1e6);
    }
    return 0;
  }

  need_goldens();
  std::vector<float> hidden1;
  if (pipeline > 1) {
    std::vector<std::vector<float>> hs(lanes.size());
    std::vector<std::thread> ts;
    for (size_t l = 0; l < lanes.size(); ++l)
      ts.emplace_back([&, l] { hs[l] = lanes[l]->run(emb_in); });
    hidden1 = enc.run(emb_in);
    for (auto &th : ts) th.join();
    // Same input, deterministic math on every lane: the outputs must be
    // BIT-IDENTICAL, or the lanes are corrupting each other's buffers.
    for (size_t l = 0; l < hs.size(); ++l)
      if (hs[l].size() != hidden1.size() ||
          std::memcmp(hidden1.data(), hs[l].data(),
                      hidden1.size() * sizeof(float)) != 0) {
        std::printf("\nFAIL -- lane %zu disagrees bitwise; cross-lane "
                    "corruption\n", l + 2);
        return 1;
      }
    std::printf("  pipeline   %zu lanes agree bitwise on %zu floats\n",
                lanes.size() + 1, hidden1.size());
  } else {
    hidden1 = enc.run(emb_in);
  }
  auto emb = pool_and_normalise(hidden1);

  // Compare EVERY row, not just the first 4. `want` was tiled with the same
  // per-copy rotation as the inputs (see above), so this checks each of the
  // `batch` output rows against the specific golden row it is supposed to
  // reproduce -- previously this loop stopped at `kGoldenBatch` (4), so at
  // batch 128 it compared 4 of 128 rows and never read the other 124 at all.
  // That is a bigger hole than "the copies are indistinguishable" (T32): it
  // is truncation, and it means a corruption bug anywhere past row 3 was
  // never observed, let alone made indistinguishable by identical tiling.
  double num = 0.0, den = 0.0, worst_1mcos = 0.0;
  for (int64_t b = 0; b < batch; ++b) {
    double dot = 0.0;
    for (int64_t c = 0; c < g_hidden; ++c) {
      double diff = emb[b * g_hidden + c] - want[b * g_hidden + c];
      num += diff * diff;
      den += static_cast<double>(want[b * g_hidden + c]) * want[b * g_hidden + c];
      dot += static_cast<double>(emb[b * g_hidden + c]) * want[b * g_hidden + c];
    }
    worst_1mcos = std::max(worst_1mcos, 1.0 - dot);
  }
  const double rel_fro = std::sqrt(num) / std::sqrt(den);
  const double tol = 2e-3;

  std::printf("\n  %-38s %11.3e\n", "embedding rel_fro vs HF golden", rel_fro);
  std::printf("  %-38s %11.3e  (all %lld rows, %lld distinct sentences "
              "rotated across tile copies)\n",
              "worst 1 - cos vs HuggingFace", worst_1mcos, (long long)batch,
              (long long)kGoldenBatch);

  // NaN must FAIL, and it took an explicit check to make it.
  //
  // std::max(0.0, NaN) returns 0.0: every comparison with NaN is false, so max
  // returns its first argument. A GELU kernel that produced NaN therefore
  // reported `worst 1 - cos = 0.000e+00` and PASSED -- a perfect score -- while
  // rel_fro printed `nan` on the line above. A tolerance test whose failure
  // mode is a perfect score is not a test.
  //
  // Fourth instance of a check failing open in this project (tasks/0022, 0024,
  // 0025, here) and the first one inside the validation itself.
  if (!std::isfinite(rel_fro) || !std::isfinite(worst_1mcos)) {
    std::printf("\nFAIL -- non-finite output. NaN cannot pass a tolerance "
                "test by scoring zero.\n");
    return 1;
  }
  std::printf("\n%s -- tolerance %.0e on 1-cos, no Python in this process\n",
              worst_1mcos <= tol ? "PASS" : "FAIL", tol);
  return worst_1mcos <= tol ? 0 : 1;
} catch (const std::exception &e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 2;
}