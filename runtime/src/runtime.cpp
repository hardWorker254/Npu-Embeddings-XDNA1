//===- runtime.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Runtime class implementation.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "runtime/runtime.hpp"
#include "runtime/run_context.hpp"
#include "runtime/run_setup.hpp"
#include "runtime/run_execute.hpp"
#include "runtime/run_probes.hpp"
#include "runtime/gemma_mode.hpp"
#include "runtime/stt_mode.hpp"
#include "runtime/vit_mode.hpp"
#include "runtime/pose_mode.hpp"
#include "runtime/hands_mode.hpp"
#include "runtime/mppose_mode.hpp"
#include "runtime/model.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>

using namespace app;

namespace npue {

Runtime::Runtime(const std::string &model_path, const std::string &root,
                 const std::string &art)
    : model_path_(model_path), root_(root), art_(art) {}

int Runtime::run(int argc, char **argv) {
    // --dev names the NPU generation this process runs on -- the same name
    // tools/export/export_gemm_rtp.py writes into design.json. It must be known before
    // any design is selected, because design_fits() refuses a set built for the
    // other generation. NPU_DEVICE / NPU2 are the environment defaults.
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--dev")
            app::set_running_device(argv[i + 1]);

    // --artifacts may arrive either from the constructor (the flag-form path
    // parses it in main.cpp) or, for the `serve`/`embed` subcommands, only in
    // argv: launch() constructs this Runtime with an empty art and forwards
    // the flag verbatim, so without this the subcommand silently ignored it
    // and always auto-selected. Reading it here fixes that and stays
    // consistent with --dev/--bench/--bo-mode, which are also read off argv.
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--artifacts")
            art_ = argv[i + 1];

    // arch=1 (EmbeddingGemma) containers own a completely different pipeline
    // (no seven-design selection, a host-only --cpu control, MQA) and are
    // dispatched before the BERT design resolution, exactly as the old main()
    // did. Returns -1 when the container is not Gemma.
    if (int gemma_r = app::maybe_gemma_mode(root_, argc, argv); gemma_r >= 0)
        return gemma_r;

    // arch=4 (Whisper) is the other architecture with a pipeline of its own:
    // two design sets, an audio front end, an autoregressive decoder, and none
    // of the embedding machinery. It is dispatched here, beside the Gemma mode
    // and before the BERT design resolution, and returns -1 for every other
    // container. --artifacts is read above, so the same flag works for both.
    if (int stt_r = app::maybe_stt_mode(root_, argc, argv, model_path_); stt_r >= 0)
        return stt_r;

    // arch=5 (a ViT) is the third architecture with a pipeline of its own: an
    // image front end, four GEMM streams on ONE design set, a host head, and
    // none of the embedding machinery -- no pooling, no golden fixtures, no
    // text. Dispatched here, beside the other two, and returns -1 for every
    // other container. --artifacts is read above, so the same flag works.
    if (int cls_r = app::maybe_vit_mode(root_, argc, argv, model_path_); cls_r >= 0)
        return cls_r;

    // arch=6 (a YOLOv8-pose) is the fourth architecture with a pipeline of its
    // own: an image front end, 72 convolutions as an explicit op graph, a
    // detection head, and none of the embedding machinery. Dispatched here,
    // beside the other three, and returns -1 for every other container.
    //
    // It is also the only one whose DEFAULT path opens no device: the
    // convolutions run on the host unless --npu-ops conv moves them onto
    // the array, and that default is a measured result rather than an omission
    // (see runtime/include/runtime/pose_mode.hpp for the arithmetic).
    if (int pose_r = app::maybe_pose_mode(root_, argc, argv, model_path_); pose_r >= 0)
        return pose_r;

    // arch=7 (MediaPipe hands: a palm detector and a hand-landmark network in
    // ONE container, run in that order with the whole decode between them) is
    // the fifth architecture with a pipeline of its own, dispatched here beside
    // the other four. Like arch=6 its default path opens no device -- and unlike
    // arch=6 it has no --npu-ops path at all, refused with the reason: there is
    // no design set carrying these graphs' 31 distinct dense (K, N) pairs, so
    // there is no array number to compare the host's against, and accepting the
    // flag would print host timings under a flag that says otherwise.
    if (int hands_r = app::maybe_hands_mode(root_, argc, argv, model_path_);
        hands_r >= 0)
        return hands_r;

    // arch=8 (MediaPipe Pose: a person detector and a pose-landmark network in ONE
    // container, run in that order with the whole decode between them) is the
    // sixth architecture with a pipeline of its own. It is dispatched AFTER arch=7
    // and beside it rather than beside arch=6, because the two are not variants:
    // arch=6 is one network over one frame, and this runs a second, separate
    // network once per detected person on a crop ROTATED by an angle the first one
    // predicted. Like both of them its default path opens no device, and it has no
    // --npu-ops path at all -- refused with the reason: no design set here carries
    // these two graphs' 99 distinct dense (K, N) pairs, so there is no array number
    // to compare the host's against.
    if (int mppose_r = app::maybe_mppose_mode(root_, argc, argv, model_path_);
        mppose_r >= 0)
        return mppose_r;

    RunContext ctx;
    ctx.argc = argc;
    ctx.argv = argv;
    ctx.root = root_;

    // --bench selects the timed path; 0 means the golden check.
    for (int i = 2; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--bench")
            ctx.bench = std::atoi(argv[i + 1]);

    // --cpu selects the host-only control encoder, which exists for arch=1
    // only -- and arch=1 was already dispatched above. Refuse it here rather
    // than silently running the NPU, which is what made this flag invisible
    // (T50, tasks/0124).
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--cpu")
            throw std::runtime_error(
                "--cpu: no host encoder exists for this architecture in this "
                "build -- the flag would be ignored and the NPU would run "
                "anyway (T50). It is honoured for embeddinggemma-300m (arch=1) "
                "only; drop the flag, or use a host reference implementation "
                "(reference/encoder_*.py) as the CPU control.");

    ctx.model = std::make_unique<npue::File>(model_path_);
    set_model_shape(*ctx.model);
    g_model_name = std::filesystem::path(model_path_).stem().string();

    // --artifacts names a design set. It is resolved against both layouts this
    // ships in: the source tree (<root>/runtime/<name>) and an extracted
    // release (<root>/<name>, or <root> itself). An absolute path is taken as
    // given. Chosen by which candidate actually CONTAINS a design, so a typo
    // is an error about the design rather than a confusing one about a file.
    const bool art_from_cli = !art_.empty();
    if (!art_.empty()) {
        auto has_design = [](const std::string &d) {
            return std::ifstream(d + "/gemm_rtp/design.json").good() ||
                   std::ifstream(d + "/qkv/design.json").good();
        };
        // The candidate list lives in design_selection.hpp so the BERT and
        // arch=1 (EmbeddingGemma) paths cannot drift. It now includes the
        // per-model layout's <name>/artifacts_npu<arch>, which is what makes
        // `--artifacts <model>` work one level deeper than before.
        const std::vector<std::string> candidates =
            artifacts_candidates(root_, art_);
        std::string found;
        for (const auto &c : candidates)
            if (has_design(c)) { found = c; break; }
        if (found.empty()) {
            std::string looked;
            for (size_t i = 0; i < candidates.size(); ++i)
                looked += (i ? ", " : "") + candidates[i];
            throw std::runtime_error(
                "no design set found for --artifacts '" + art_ +
                "'; looked for gemm_rtp/design.json or qkv/design.json under " +
                looked);
        }

        // NAMING A SET IS NOT THE SAME AS IT FITTING. `has_design` above asks
        // whether the directory holds a design, and the implicit path -- where
        // the runtime chooses -- then asks design_fits whether it serves THIS
        // container. Handing --artifacts used to stop at the first question, so
        // `--artifacts <model>` on an int8 container named the bf16 set and the
        // encode died at layer 0 on a layout hash, which is a crash with a
        // cause three steps from where the mistake was.
        //
        // The container's own B layout is the one thing that distinguishes two
        // sets of otherwise identical geometry, and both sides of it are
        // recorded, so it is checked here rather than inferred. An empty answer
        // on either side is not a mismatch: a design exported before the field
        // existed cannot be excluded by it, and refusing those would break every
        // older set in the tree.
        //
        // select_set_for_layout is the same filter the Whisper resolver uses.
        // It was written there first and then reused here rather than the other
        // way round, because Whisper's version has to look for two files and
        // this one's error message is the richer of the two.
        {
            std::string want_layout;
            try {
                want_layout = ctx.model->info("layer.0.qkv").layout_hash;
            } catch (const std::exception &) {}
            const std::string by_layout = select_set_for_layout(
                candidates, has_design, want_layout);
            if (by_layout.empty() && !want_layout.empty()) {
                // Nothing usable, or nothing with this container's layout. The
                // first is "your --artifacts is not a design set"; the second
                // is the one that looks like a working directory and is not.
                bool saw_usable = false;
                for (const auto &c : candidates)
                    saw_usable = saw_usable || has_design(c);
                if (saw_usable) {
                    const std::string got = design_b_layout_hash(found);
                    throw std::runtime_error(
                        "--artifacts '" + art_ + "' resolved to " + found +
                        ", whose B-operand layout is " + got.substr(0, 12) +
                        "..., but this container's is " +
                        want_layout.substr(0, 12) +
                        "..., and no other candidate under " + root_ +
                        " declares it. They are the same shapes in different "
                        "element types, so nothing in that directory can "
                        "execute it. Export one for this datapath, or run a "
                        "container built for that one.");
                }
            }
            if (!by_layout.empty()) {
                if (by_layout != found && !want_layout.empty())
                    std::fprintf(stderr,
                                 "note: --artifacts %s resolves to %s, whose B "
                                 "layout is %s; this container's is %s. Using %s "
                                 "instead.\n",
                                 art_.c_str(), found.c_str(),
                                 design_b_layout_hash(found).substr(0, 12).c_str(),
                                 want_layout.substr(0, 12).c_str(),
                                 by_layout.c_str());
                found = by_layout;
            }
        }
        art_ = found;
    }

    // Declared OUTSIDE the branch below because the status line needs it too, and a
    // self-sufficient container never enters that branch -- so an inner declaration
    // would be in scope exactly when it is not the answer.
    bool self_sufficient =
        npu::prefer_embedded(ctx.model.get(), "", "gemm_rtp").has("design.json");

    if (art_.empty() && !self_sufficient) {
        int64_t qkv_n = 0;
        try { qkv_n = ctx.model->config_int("qkv_n"); } catch (const std::exception &) {}
        std::string layout;
        try { layout = ctx.model->info("layer.0.qkv").layout_hash;
        } catch (const std::exception &) {}
        const auto *ce = npue::hub::find(g_model_name);
        const std::string want_datapath = ce ? ce->datapath : "bf16";
        // A CONTAINER THAT CARRIES ITS OWN SET NEEDS NO DIRECTORY, and this check
        // has to come BEFORE pick_artifacts rather than after it: pick_artifacts
        // answers by scanning the filesystem, so a reader with nothing on disk was
        // refused here with "no NPU design set matches ... under <root>" while
        // holding a container that carried the set. Asking the question after
        // would mean keeping a message that is false whenever the file answers it.
        //
        // So `art_` is left EMPTY on purpose in that case. Everything downstream
        // already goes through prefer_embedded(), which falls back to
        // `artifacts + "/" + <set>` -- and an empty `art` makes that path a
        // relative "/gemm_rtp", which is never consulted because the container
        // answers first. Nothing reads `art_` as a path except pick_artifacts
        // itself and the status line, and the status line is told which case it is.
        art_ = pick_artifacts(root_, g_hidden,
                                    ctx.model->config_int("intermediate"),
                                    config_flag(*ctx.model, "gated_ffn", false),
                        qkv_n, layout, want_datapath, g_model_name);
        if (art_.empty())
            throw std::runtime_error(
                "no NPU design set matches " + g_model_name + " (hidden " +
                std::to_string(g_hidden) + ", datapath " + want_datapath +
                ", device " + running_device() + ") under " + root_ +
                " -- name one with --artifacts, or export one for this "
                "generation with tools/export/export_gemm_rtp.py");
    }
    // THE STATUS LINE MUST SAY WHICH OF THE TWO IT IS. "artifacts" followed by an
    // empty string is what this printed for a self-sufficient container before, and
    // an empty path in a status block reads as a bug in the thing printing it.
    if (self_sufficient && !art_from_cli)
        std::printf("  artifacts  the container's own design set "
                    "(design/gemm_rtp; no directory needed, no --artifacts given)\n");
    else if (!art_from_cli)
        std::printf("  artifacts  %s (picked from the container's geometry; "
                    "no --artifacts given)\n", art_.c_str());
    ctx.art = art_;

    std::string val =
        root_ + "/runtime/artifacts/validation/" + g_model_name;
    bool val_is_own = std::ifstream(val + "/emb_sum.f32").good();
    if (!val_is_own) {
        namespace fs = std::filesystem;
        std::error_code vec;
        const std::string want = ctx.model->config_string("source_sha256");
        const fs::path base = fs::path(root_) / "runtime" / "artifacts" / "validation";
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
    if (!val_is_own) val = root_ + "/runtime/artifacts/validation";
    bool have_val = std::ifstream(val + "/emb_sum.f32").good();
    ctx.val = val;
    ctx.val_is_own = val_is_own;
    ctx.have_val = have_val;
    if (ctx.have_val) {
        std::ifstream vf(ctx.val + "/validation.json");
        if (!vf)
            throw std::runtime_error(
                "found fixtures under " + ctx.val + " but no validation.json");
        std::stringstream vs;
        vs << vf.rdbuf();
        const std::string want_sha =
            npue::http::json_field_string(vs.str(), "source_sha256", "");
        const std::string got_sha = ctx.model->config_string("source_sha256");
        if (!ctx.val_is_own && want_sha != got_sha) {
            ctx.have_val = false;
        } else if (want_sha.empty() || want_sha != got_sha)
            throw std::runtime_error(
                "the golden fixtures were made from checkpoint " +
                want_sha.substr(0, 16) + "... but this model is " +
                got_sha.substr(0, 16) + "... -- re-run tools/export/export_validation.py");
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
    // exists, because Design's constructor allocates.
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

    if (int _r = maybe_probe_pair(ctx); _r >= 0) return _r;
    if (int _r = maybe_probe_bo(ctx); _r >= 0) return _r;
    if (int _r = maybe_probe_design(ctx); _r >= 0) return _r;
    if (int _r = maybe_probe_insts(ctx); _r >= 0) return _r;
    if (int _r = maybe_probe_rtp(ctx); _r >= 0) return _r;
    if (int _r = maybe_probe_ctx(ctx); _r >= 0) return _r;
    if (int _r = maybe_soak_npu(ctx); _r >= 0) return _r;
    if (int _r = maybe_soak_cpu(ctx); _r >= 0) return _r;

    // Flags first: --npu-eltwise and the --host-* overrides decide which
    // designs load_designs must resolve (and which missing ones it refuses),
    // and --pipeline decides how many pools exist. None of them depend on the
    // designs, so parsing before loading costs nothing.
    setup_flags_pools(ctx);
    load_designs(ctx);
    load_goldens(ctx);
    if (int _r = setup_encoder(ctx); _r != 0) return _r;
    if (int _r = maybe_probe(ctx); _r >= 0) return _r;
    if (int _r = maybe_probe_streams(ctx); _r >= 0) return _r;
    if (int _r = maybe_embed(ctx); _r >= 0) return _r;
    if (int _r = maybe_serve(ctx); _r >= 0) return _r;
    if (int _r = maybe_encode_file(ctx); _r >= 0) return _r;
    return run_bench_or_check(ctx);
}

}  // namespace npue