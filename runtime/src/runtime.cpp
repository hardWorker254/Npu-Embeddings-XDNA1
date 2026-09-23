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
    // tools/export_gemm_rtp.py writes into design.json. It must be known before
    // any design is selected, because design_fits() refuses a set built for the
    // other generation. NPU_DEVICE / NPU2 are the environment defaults.
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--dev")
            app::set_running_device(argv[i + 1]);

    // arch=1 (EmbeddingGemma) containers own a completely different pipeline
    // (no seven-design selection, a host-only --cpu control, MQA) and are
    // dispatched before the BERT design resolution, exactly as the old main()
    // did. Returns -1 when the container is not Gemma.
    if (int gemma_r = app::maybe_gemma_mode(root_, argc, argv); gemma_r >= 0)
        return gemma_r;

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
        const std::vector<std::string> candidates = {
            art_, root_ + "/" + art_, root_ + "/runtime/" + art_};
        std::string found;
        for (const auto &c : candidates)
            if (has_design(c)) { found = c; break; }
        if (found.empty())
            throw std::runtime_error(
                "no design set found for --artifacts '" + art_ +
                "'; looked for gemm_rtp/design.json or qkv/design.json under " +
                candidates[0] + ", " + candidates[1] + " and " + candidates[2]);
        art_ = found;
    }

    if (art_.empty()) {
        int64_t qkv_n = 0;
        try { qkv_n = ctx.model->config_int("qkv_n"); } catch (const std::exception &) {}
        std::string layout;
        try { layout = ctx.model->info("layer.0.qkv").layout_hash;
        } catch (const std::exception &) {}
        const auto *ce = npue::hub::find(g_model_name);
        const std::string want_datapath = ce ? ce->datapath : "bf16";
        art_ = pick_artifacts(root_, g_hidden,
                                 ctx.model->config_int("intermediate"),
                                 config_flag(*ctx.model, "gated_ffn", false), qkv_n,
                                 layout, want_datapath);
        if (art_.empty())
            throw std::runtime_error(
                "no NPU design set matches " + g_model_name + " (hidden " +
                std::to_string(g_hidden) + ", datapath " + want_datapath +
                ", device " + running_device() + ") under " + root_ +
                " -- name one with --artifacts, or export one for this "
                "generation with tools/export_gemm_rtp.py");
    }
    if (!art_from_cli)
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
                got_sha.substr(0, 16) + "... -- re-run tools/export_validation.py");
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