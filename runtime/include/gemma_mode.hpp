//===- gemma_mode.hpp -----------------------------------------------------*- C++ -*-===//
//
// The arch=1 (EmbeddingGemma) CLI wiring: two paths chosen from the
// CONTAINER's gemm_layout -- GemmaNpuEncoder on the array, or the host-only
// npue::GemmaEncoder as the discriminating control (--cpu). --serve goes
// through EmbedBackend like every other architecture.
//
// Split out of main.cpp verbatim.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "app_state.hpp"
#include "design_selection.hpp"
#include "embed_service.hpp"
#include "gemma_encode.hpp"
#include "gemma_npu_encoder.hpp"
#include "host_kernels.hpp"
#include "model_catalog.hpp"
#include "hub.hpp"
#include "npu_device.hpp"
#include "npue.hpp"
#include "pool.hpp"
#include "tokenizer.hpp"
#include "tokenizer_gemma.hpp"

namespace app {

// Refuse an unknown --prefix by LISTING the real ones, rather than throwing
// tokenizer_gemma's bare "no task prefix named X". Same standard tasks/0071
// set for nomic. An empty name is legal and means no prefix at all, which is
// sentence-transformers' own default for this checkpoint
// (`default_prompt_name: null`).
//
// OMITTING it is no longer legal when `required` (tasks/0118). This path used
// to default to "document" -- the last silent default in this runtime, and the
// worst-placed one, because EmbeddingGemma's table holds 14 prompts and the
// checkpoint itself names none of them as a default. `from_cli` is what
// separates "not given" from `--prefix ""`, which still means no prefix.
inline void check_gemma_prefix(const npue::GemmaTokenizer &tok,
                        const std::string &name, bool from_cli,
                        bool required) {
  if (required && !from_cli) {
    std::string all;
    const auto names = tok.prefix_names();
    for (size_t i = 0; i < names.size(); ++i) all += (i ? ", " : "") + names[i];
    throw std::runtime_error(
        "this model has task prefixes and one must be named: pass --prefix "
        "with one of [" + all + "], or --prefix \"\" for no prefix at all. "
        "Refusing to pick one for you -- a wrongly-prefixed embedding is "
        "correctly shaped and correctly normed, so nothing downstream can tell "
        "that the answer is wrong.");
  }
  if (name.empty()) return;
  const auto names = tok.prefix_names();
  if (std::find(names.begin(), names.end(), name) != names.end()) return;
  std::string all;
  for (size_t i = 0; i < names.size(); ++i)
    all += (i ? ", " : "") + names[i];
  throw std::runtime_error("--prefix '" + name + "' is not a task prefix this "
                           "model defines. It has: " + all +
                           ". Pass --prefix \"\" for no prefix at all.");
}

inline int run_gemma_mode(npue::File &model, const std::string &model_path,
                   const std::string &root, int argc, char **argv) {
  auto has_flag = [&](const char *f) {
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == f) return true;
    return false;
  };
  auto flag_val = [&](const char *f, std::string dflt) {
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == f) return std::string(argv[i + 1]);
    return dflt;
  };

  // --serve [port] [--bind addr] (tasks/0115, T34's last unbuilt item). This
  // used to refuse. Nothing about arch=1 was ever incompatible with an HTTP
  // endpoint -- the endpoint was simply written against the BERT encoder's
  // type. serve_http() above now takes the three things it actually needs, so
  // this path supplies them like any other.
  int serve_port = -1;
  std::string serve_bind = "127.0.0.1";
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--serve") {
      serve_port = 8080;
      if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0])))
        serve_port = std::atoi(argv[i + 1]);
    }
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--bind") serve_bind = argv[i + 1];

  const int max_len = std::atoi(flag_val("--max-len", "64").c_str());
  // NO DEFAULT any more (tasks/0118). This was `flag_val("--prefix",
  // "document")` -- a task prompt nobody asked for, silently applied, on a
  // model with 14 of them. check_gemma_prefix() enforces it below; `serve`
  // rejects the flag outright because its prompt is per request.
  const bool has_prefix_flag = has_flag("--prefix");
  const std::string prefix = flag_val("--prefix", "");
  const bool force_cpu = has_flag("--cpu");
  if (serve_port > 0 && has_prefix_flag)
    throw std::runtime_error(
        "--prefix does not apply to `serve`: the task prompt is chosen per "
        "request now. Send \"prompt_name\" in the POST body instead, and GET "
        "/health lists the names this model accepts.");

  std::string layout;
  try {
    layout = model.config_string("gemm_layout");
  } catch (const std::exception &) {
    layout = "host";        // a container packed before tasks/0074
  }

  // `--artifacts` is a NAME as often as a path -- every script in this repo
  // passes `artifacts_gemma`, not a path -- so resolve it against the same
  // three candidates the BERT path uses (cwd, an extracted release, the source
  // tree). Taking it verbatim made the sweep fail with "cannot open
  // artifacts_gemma/gemm_rtp/design.json", and the sweep then MISREPORTED that
  // exit code as a contention refusal.
  std::string art = flag_val("--artifacts", "");
  if (!art.empty()) {
    const std::vector<std::string> cands = {art, root + "/" + art,
                                            root + "/runtime/" + art};
    std::string found;
    for (const auto &c : cands)
      if (std::ifstream(c + "/gemm_rtp/design.json").good()) { found = c; break; }
    if (found.empty())
      throw std::runtime_error(
          "no design set found for --artifacts '" + art + "'; looked for "
          "gemm_rtp/design.json under " + cands[0] + ", " + cands[1] +
          " and " + cands[2]);
    art = found;
  }
  if (art.empty() && layout == "pretiled_bf16") {
    int64_t qn = 0;
    try { qn = model.config_int("qkv_n"); } catch (const std::exception &) {}
    // Look this model up by its catalogue name -- the container stem -- so
    // an adopted bfp16 datapath (tasks/0104) is required here exactly as it
    // is on the BERT serve/embed path. A container reached directly by path
    // (not through the catalogue) has no entry, so npue::hub::find() returns
    // nullptr and the default CatalogEntry::datapath, "bf16", applies.
    const std::string mname =
        std::filesystem::path(model_path).stem().string();
    const auto *ce = npue::hub::find(mname);
    art = pick_artifacts(root, model.config_int("hidden"),
                         model.config_int("intermediate"), true, qn, "",
                         ce ? ce->datapath : "bf16");
  }
  const bool use_npu = !force_cpu && layout == "pretiled_bf16" && !art.empty();

  std::printf("NpuEmbeddings C++ runtime -- EmbeddingGemma (arch=1)\n");
  std::printf("  model      %s\n", model_path.c_str());

  // Read the input up front: both paths want the same texts, and a missing
  // file should fail before an xclbin is loaded.
  std::vector<std::string> texts;
  std::string in_path, out_path;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--embed") {
      in_path = argv[i + 1];
      if (i + 2 < argc && argv[i + 2][0] != '-') out_path = argv[i + 2];
    }
  if (!in_path.empty()) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + in_path);
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      texts.push_back(line);
    }
  } else {
    texts.push_back(
        "The AMD Ryzen AI NPU accelerates transformer encoder models.");
  }

  auto write_out = [&](const std::vector<float> &out, int64_t hidden) {
    if (!out_path.empty()) {
      std::ofstream of(out_path, std::ios::binary);
      of.write(reinterpret_cast<const char *>(out.data()),
               static_cast<std::streamsize>(out.size() * sizeof(float)));
      if (!of) throw std::runtime_error("failed writing " + out_path);
      std::printf("  wrote      %s  [%zu, %lld] fp32\n", out_path.c_str(),
                  texts.size(), (long long)hidden);
    } else {
      for (size_t b = 0; b < std::min<size_t>(texts.size(), 4); ++b) {
        std::printf("  [%zu]", b);
        for (int64_t c = 0; c < 6; ++c)
          std::printf(" %+.4f", out[b * static_cast<size_t>(hidden) + c]);
        std::printf(" ...\n");
      }
    }
  };

  if (!use_npu) {
    npue::GemmaEncoder enc(model);
    std::printf("  hidden     %lld, tokenizer %zu tokens\n",
                (long long)enc.hidden(), enc.tok.vocab_size());
    // Say WHY the host path was taken. "cpu" with no reason is the kind of
    // silent downgrade that gets measured and reported as if it were the
    // fast path.
    std::printf("  path       HOST-only (%s)\n",
                force_cpu             ? "--cpu given"
                : layout != "pretiled_bf16"
                    ? "container holds row-major F32 operands"
                    : "no matching NPU design set found");
    std::printf("  NOTE: every op runs on the CPU. Any seq/s below is "
                "host-only and is NOT an NPU performance claim "
                "(CLAUDE.md rule 1).\n");
    check_gemma_prefix(enc.tok, prefix, has_prefix_flag, serve_port <= 0);
    if (serve_port <= 0)
      std::printf("  input      %zu texts, max_len=%d, prefix='%s' -> %s\n",
                  texts.size(), max_len, prefix.c_str(),
                  prefix.empty() ? "(none)"
                                 : enc.tok.prefix_text(prefix).c_str());
    auto host_embed = [&](const std::vector<std::string> &txts,
                          const std::string &pname, int64_t *tokens) {
      std::vector<float> o(txts.size() * static_cast<size_t>(enc.hidden()));
      if (tokens) *tokens = 0;
      for (size_t t = 0; t < txts.size(); ++t) {
        // Tokenized twice on this path -- once to report the count, once
        // inside encode_one. On a path that is already host-only and orders
        // of magnitude slower than the array, that is the cheap way to give
        // `usage.prompt_tokens` a real number instead of a zero.
        if (tokens)
          *tokens += enc.tok.encode(txts[t], max_len, pname).n_tokens;
        const auto v = enc.encode_one(txts[t], max_len, pname, t,
                                      g_allow_truncation);
        std::memcpy(o.data() + t * v.size(), v.data(),
                    v.size() * sizeof(float));
      }
      return o;
    };

    // --serve works here too (tasks/0115). It must: `serve_port` is parsed
    // before this branch, so leaving it unhandled would have made
    // `--serve --cpu` embed the placeholder sentence once and exit -- a
    // silent no-op, the failure shape this project keeps finding.
    if (serve_port > 0) {
      EmbedBackend be;
      be.vocab_size = enc.tok.vocab_size();
      // Straight from the GEMATOK1 blob -- arch 1's prompts live there, not in
      // the container config the BERT path reads. serve_http() only ever sees
      // the names.
      be.prompt_names = enc.tok.prefix_names();
      std::sort(be.prompt_names.begin(), be.prompt_names.end());
      be.hidden = enc.hidden();
      be.seq = max_len;
      be.embed = host_embed;
      // "-cpu", not "-npu": the model id is what a client sees, and naming a
      // host-only server after the array is exactly the silent mislabel the
      // "path HOST-only" line above exists to prevent.
      return serve_http(be,
                        std::filesystem::path(model_path).stem().string() + "-cpu",
                        serve_port, serve_bind);
    }

    const double t0 = now_s();
    std::vector<float> out = host_embed(texts, prefix, nullptr);
    const double el = now_s() - t0;
    std::printf("  embedded   %zu texts in %.2f s  ->  %.2f seq/s (host-only)\n",
                texts.size(), el, texts.size() / std::max(el, 1e-9));
    write_out(out, enc.hidden());
    return 0;
  }

  // ---- NPU path -----------------------------------------------------------
  //
  // The contention gate, when this run is a MEASUREMENT (tasks/0044's ninth
  // fail-open). A foreign Active hw_context -- most often a stale npuembed.exe
  // from an earlier command in the same session -- read MiniLM at 221 seq/s
  // against a true 691. It is opt-in here rather than always-on because an
  // ordinary `embed` is not a performance claim and should not refuse to run
  // because something else is using the array; `--guard-contention` is what
  // tools/release_benchmark.ps1 passes.
  if (has_flag("--guard-contention") &&
      !npu::require_exclusive_npu(npu::survey_contexts(),
                                  has_flag("--allow-contention")))
    return 2;

  npu::Device dev;
  npu::Design d(dev, art + "/gemm_rtp");
  std::vector<StreamEntry> streams;
  {
    std::ifstream sj(art + "/gemm_rtp/design.json");
    std::stringstream sbuf;
    sbuf << sj.rdbuf();
    streams = parse_streams(sbuf.str());
  }
  if (streams.empty())
    throw std::runtime_error(art + "/gemm_rtp/design.json lists no streams -- "
                             "re-export with tools/export_gemm_rtp.py");
  std::sort(streams.begin(), streams.end(),
            [](const StreamEntry &a, const StreamEntry &b) {
              return a.slot < b.slot;
            });
  for (const auto &s : streams) {
    const size_t got = d.load_instr(art + "/gemm_rtp/" + s.file);
    if (static_cast<int64_t>(got) != s.slot)
      throw std::runtime_error("stream " + s.file + " landed in slot " +
                               std::to_string(got) + ", design.json says " +
                               std::to_string(s.slot));
  }

  if (d.info().seq <= 0)
    throw std::runtime_error("this design set records no sequence length");
  const int64_t seq = d.info().seq;
  if (max_len != seq)
    throw std::runtime_error(
        "--max-len " + std::to_string(max_len) + " but the design was compiled "
        "for seq " + std::to_string(seq) + ". The NPU path encodes at the "
        "design's own sequence length; use --cpu for another length, or "
        "re-export the design.");

  int nthreads = std::atoi(flag_val("--threads", "16").c_str());
  if (nthreads < 1) nthreads = 1;
  int nlanes = std::atoi(flag_val("--pipeline", "2").c_str());
  if (nlanes < 1) nlanes = 1;
  std::vector<std::unique_ptr<Pool>> pools;
  for (int l = 0; l < nlanes; ++l)
    pools.push_back(std::make_unique<Pool>(std::max(1, nthreads / nlanes)));

  GemmaNpuEncoder enc(model, d, *pools[0]);
  // T37 (tasks/0082). Same switch as the BERT path, for the same reason: the
  // fused and unfused epilogues must both stay runnable so they can be A/B'd
  // byte-for-byte. Default on -- it is bit-identical and strictly less traffic.
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == "--no-fuse-ffn") enc.fuse_ffn_epilogue = false;
  enc.seq = seq;
  enc.batch = d.info().M / seq;
  enc.rows = enc.batch * seq;
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
    if (!complete) continue;
    enc.tiers.push_back(b);
    enc.tier_slots.push_back(slots);
  }
  enc.slot_a = d.stage_alloc(0, d.info().buffer_bytes[0]);
  enc.slot_c = d.stage_alloc(2, d.info().buffer_bytes[2]);
  const size_t staged = enc.stage_all();

  // Extra lanes. The staged weights and the tier table belong to the DESIGN,
  // not to a lane, and are copied rather than re-staged; only the A and C
  // buffers are per-lane. A lane missing the tier table would silently fall
  // back to the flat (0,1,2,3) slot contract, which under a 16-stream export
  // selects entirely the wrong shapes -- measured as 1-cos 1.0 on the BERT
  // path when exactly that happened (tasks/0037).
  static std::mutex npu_mutex;
  std::vector<std::unique_ptr<GemmaNpuEncoder>> extra;
  if (nlanes > 1) {
    enc.npu_mu = &npu_mutex;
    for (int l = 1; l < nlanes; ++l) {
      extra.push_back(std::make_unique<GemmaNpuEncoder>(model, d, *pools[l]));
      GemmaNpuEncoder &e2 = *extra.back();
      e2.seq = enc.seq;
      e2.batch = enc.batch;
      e2.rows = enc.rows;
      e2.tiers = enc.tiers;
      e2.tier_slots = enc.tier_slots;
      e2.s_qkv = enc.s_qkv; e2.s_ao = enc.s_ao;
      e2.s_fu = enc.s_fu;   e2.s_fd = enc.s_fd;
      e2.b_qkv = enc.b_qkv; e2.b_ao = enc.b_ao;
      e2.b_fu = enc.b_fu;   e2.b_fd = enc.b_fd;
      // The int8 scale vectors are the CONTAINER's, not a lane's -- same
      // reasoning as the staged weights above, and the same failure if
      // forgotten. tasks/0078 hit exactly this on the BERT encoder (lane 0
      // worked, lanes 1+ had no scales) and the guard it added is what caught
      // it here: `--pipeline 4` on an int8 Gemma threw "this encoder has no
      // quantisation scales" instead of returning wrong embeddings.
      e2.ws_qkv = enc.ws_qkv; e2.ws_ao = enc.ws_ao;
      e2.ws_fu = enc.ws_fu;   e2.ws_fd = enc.ws_fd;
      e2.as_qkv = enc.as_qkv; e2.as_ao = enc.as_ao;
      e2.as_fu = enc.as_fu;   e2.as_fd = enc.as_fd;
      e2.fuse_ffn_epilogue = enc.fuse_ffn_epilogue;
      e2.use_tier(enc.batch);
      e2.slot_a = d.stage_alloc(0, d.info().buffer_bytes[0]);
      e2.slot_c = d.stage_alloc(2, d.info().buffer_bytes[2]);
      e2.npu_mu = &npu_mutex;
      if (e2.tiers != enc.tiers)
        throw std::runtime_error("lane stream policy differs from lane 0");
    }
  }
  std::vector<GemmaNpuEncoder *> all_lanes{&enc};
  for (auto &e : extra) all_lanes.push_back(e.get());

  std::printf("  hidden     %lld, tokenizer %zu tokens\n",
              (long long)enc.hidden, enc.tok.vocab_size());
  std::printf("  path       NPU -- 4 GEMMs/layer x %lld layers = %lld "
              "dispatches, ONE xclbin, one hw_context\n",
              (long long)enc.layers, (long long)(4 * enc.layers));
  std::printf("  designs    %s  (%zu streams, %zu batch tiers)\n", art.c_str(),
              streams.size(), tset.size());
  // WHICH DATAPATH WAS ACTUALLY SELECTED (tasks/0104), read off the loaded
  // design, not off a flag -- the intention-not-value slip named at the
  // "staged" line just below has cost this project time before (tasks/0042,
  // 0081), and now applies to bfp16 too.
  if (!enc.d.info().datapath_recorded)
    std::printf("  datapath   UNRECORDED (design predates tasks/0104), "
                "C as %s\n",
                enc.d.info().c_elem_bytes == 2 ? "bf16" : "fp32");
  else
    std::printf("  datapath   %s MMAC, C as %s\n",
                enc.d.info().emulate_bfp16 ? "bfp16-emulated" : "bf16",
                enc.d.info().c_elem_bytes == 2 ? "bf16" : "fp32");
  // WHICH TOOLCHAIN BUILT THIS DESIGN (T39, tasks/0106) -- read off the
  // loaded design's toolchain.json, same UNRECORDED-not-guessed discipline
  // as the datapath line just above.
  if (!enc.d.info().toolchain_recorded)
    std::printf("  toolchain  UNRECORDED (design predates tasks/0106)\n");
  else
    std::printf("  toolchain  mlir_aie %s, peano %s, mlir-aie HEAD %s\n",
                enc.d.info().mlir_aie_version.c_str(),
                enc.d.info().peano_version.c_str(),
                enc.d.info().mlir_aie_git_head.c_str());
  std::printf("  qkv        N=%lld  (q[0,%lld) k[%lld,%lld) v[%lld,%lld), "
              "%lld zero-padded cols)\n",
              (long long)enc.qkv_n, (long long)enc.hidden,
              (long long)enc.k_off, (long long)(enc.k_off + enc.kv_w),
              (long long)enc.v_off, (long long)(enc.v_off + enc.kv_w),
              (long long)(enc.qkv_n - enc.v_off - enc.kv_w));
  // Read the dtype off the design rather than asserting it: this line said
  // "bf16" over int8 weights the moment arch=1 got an int8 path, which is the
  // same intention-not-value slip as tasks/0042's tile banner (tasks/0081).
  std::printf("  staged     %.1f MB of tiled %s weights on the device\n",
              staged / 1e6,
              enc.d.info().a_elem_bytes == 1 ? "int8" : "bf16");
  std::printf("  host       RMSNorm x%lld, RoPE, GeGLU, MQA attention "
              "(2.3%% of MACs)\n", (long long)(4 * enc.layers + 1));
  // ALWAYS SAY WHICH PREFIX WAS APPLIED, and refuse an unknown name by
  // listing the real ones -- the standard tasks/0071 set for nomic, which this
  // arch had not been held to. It matters most for MTEB: the harness applies
  // the same prefix to the CPU side, and a silent mismatch would show up as a
  // datapath difference rather than as the harness bug it is.
  check_gemma_prefix(enc.tok, prefix, has_prefix_flag, serve_port <= 0);
  if (serve_port <= 0)
    std::printf("  input      %zu texts, seq=%lld, prefix='%s' -> %s\n",
                texts.size(), (long long)seq, prefix.c_str(),
                prefix.empty() ? "(none)"
                               : enc.tok.prefix_text(prefix).c_str());

  if (nlanes > 1)
    std::printf("  pipeline   %d concurrent lanes, one NPU mutex, %d host "
                "threads per lane\n", nlanes, pools[0]->size());

  // Right-size each group to the smallest tier that holds it, then pad the
  // last group by REPEATING its own last text rather than with empty strings:
  // a padded lane costs full array time either way, and repeating a real text
  // keeps the tokenizer on the same code path.
  struct Job {
    GemmaNpuEncoder *e;
    size_t start = 0, take = 0;
    std::vector<std::string> group;
  };
  const int64_t tier_max = enc.tiers.empty() ? enc.batch : enc.tiers.back();

  // The lane loop, as a callable over an arbitrary batch of texts (tasks/0115).
  // It used to be written straight against the file-loaded `texts`, which is
  // the only reason --serve could not reuse it; the body below is unchanged
  // apart from taking its input as a parameter and accumulating token counts.
  auto embed_texts = [&](const std::vector<std::string> &texts,
                         const std::string &pname,
                         int64_t *tokens) -> std::vector<float> {
  std::vector<float> out(texts.size() * static_cast<size_t>(enc.hidden));
  std::atomic<int64_t> tok_total{0};
  size_t done = 0;
  while (done < texts.size()) {
    std::vector<Job> jobs;
    for (size_t l = 0; l < all_lanes.size() && done < texts.size(); ++l) {
      const size_t remaining = texts.size() - done;
      const int64_t want =
          static_cast<int64_t>(std::min<size_t>(remaining,
                                                static_cast<size_t>(tier_max)));
      Job j;
      j.e = all_lanes[l];
      const int64_t b = j.e->use_tier(want);
      j.start = done;
      j.take = std::min<size_t>(static_cast<size_t>(b), remaining);
      j.group.reserve(static_cast<size_t>(b));
      for (int64_t i = 0; i < b; ++i)
        j.group.push_back(texts[std::min(done + static_cast<size_t>(i),
                                         texts.size() - 1)]);
      done += j.take;
      jobs.push_back(std::move(j));
    }
    // An exception thrown on a worker thread would otherwise call
    // std::terminate and lose the message. Captured, then rethrown on this
    // thread after every lane has been joined.
    //
    // An exception_ptr rather than a string, because the TYPE carries meaning
    // now: npue::InputTooLong must stay distinguishable from a runtime_error
    // all the way out, or a caller that maps it to a 4xx sees a 5xx instead.
    // Flattening it to `what()` here would be a fail-open one rethrow wide.
    std::vector<std::exception_ptr> errs(jobs.size());
    auto run_job = [&](size_t k) {
      try {
        int64_t nt = 0;
        const std::vector<float> v = jobs[k].e->encode_batch(
            jobs[k].group, pname, jobs[k].start, jobs[k].take, &nt);
        tok_total += nt;
        std::memcpy(out.data() + jobs[k].start * static_cast<size_t>(enc.hidden),
                    v.data(),
                    jobs[k].take * static_cast<size_t>(enc.hidden) *
                        sizeof(float));
      } catch (...) {
        errs[k] = std::current_exception();
      }
    };
    std::vector<std::thread> th;
    for (size_t k = 1; k < jobs.size(); ++k)
      th.emplace_back([&, k] { run_job(k); });
    run_job(0);
    for (auto &t : th) t.join();
    for (const auto &e : errs)
      if (e) std::rethrow_exception(e);
  }
  if (tokens) *tokens = tok_total.load();
  return out;
  };

  // --serve on arch=1 (T34's last unbuilt item, tasks/0115). Everything the
  // endpoint needs is now in hand, so it is the same server the BERT path
  // runs -- not a second implementation that could drift from it.
  if (serve_port > 0) {
    EmbedBackend be;
    be.vocab_size = enc.tok.vocab_size();
    // From the GEMATOK1 blob, like the host path above -- arch 1's prompts
    // live in the tokenizer, not in the container config the BERT path reads.
    be.prompt_names = enc.tok.prefix_names();
    std::sort(be.prompt_names.begin(), be.prompt_names.end());
    be.hidden = enc.hidden;
    be.seq = seq;
    be.embed = [&](const std::vector<std::string> &t, const std::string &pn,
                   int64_t *n) {
      return embed_texts(t, pn, n);
    };
    return serve_http(be,
                      std::filesystem::path(model_path).stem().string() + "-npu",
                      serve_port, serve_bind);
  }

  const double t0 = now_s();
  std::vector<float> out = embed_texts(texts, prefix, nullptr);
  const double el = now_s() - t0;
  // Fold every lane's counters into lane 0 so the breakdown describes the run
  // rather than whichever lane happened to be reported.
  for (auto &e : extra) {
    enc.t_conv += e->t_conv; enc.t_in += e->t_in; enc.t_disp += e->t_disp;
    enc.t_out += e->t_out;   enc.t_bias += e->t_bias; enc.t_norm += e->t_norm;
    enc.t_attn += e->t_attn; enc.t_rope += e->t_rope; enc.t_geglu += e->t_geglu;
    enc.t_tok += e->t_tok;   enc.n_dispatch += e->n_dispatch;
  }
  std::printf("  embedded   %zu texts in %.2f s  ->  %.1f seq/s (wall clock, "
              "end-to-end -- NOT an NPU kernel claim, CLAUDE.md rule 1)\n",
              texts.size(), el, texts.size() / std::max(el, 1e-9));
  std::printf("  breakdown  npu %.0f ms (in %.0f, dispatch %.0f, out %.0f) | "
              "conv %.0f | bias %.0f | norm %.0f | attn %.0f | rope %.0f | "
              "geglu %.0f | tok %.0f  [%d dispatches]\n",
              (enc.t_in + enc.t_disp + enc.t_out) * 1e3, enc.t_in * 1e3,
              enc.t_disp * 1e3, enc.t_out * 1e3, enc.t_conv * 1e3,
              enc.t_bias * 1e3, enc.t_norm * 1e3, enc.t_attn * 1e3,
              enc.t_rope * 1e3, enc.t_geglu * 1e3, enc.t_tok * 1e3,
              enc.n_dispatch);
  write_out(out, enc.hidden);
  return 0;
}


// Early dispatch for arch=1 containers, resolved BEFORE --artifacts
// resolution -- which THROWS if no BERT NPU design is found under `root`, a
// precondition this arch does not share (it has no NPU design at all, by
// design -- see run_gemma_mode's own comment). Exceptions here are swallowed
// and reported as -1, so the caller falls through to the original resolution
// path, which reports the real error (missing model, ambiguous --model, ...)
// exactly as if this dispatch did not exist. The model is opened via mmap
// twice in the Gemma case, which is cheap and never touches the BERT path.
// Returns the process exit code, or -1 when the container is not Gemma.
inline int maybe_gemma_mode(const std::string &root, int argc, char **argv) {
  {
    std::string peek_path;
    try {
      peek_path = resolve_model_path(root, argc, argv);
    } catch (const std::exception &) {
    }
    if (!peek_path.empty()) {
      bool is_gemma = false;
      try {
        npue::File peek(peek_path);
        is_gemma = (peek.config_string("arch") == "gemma3_mqa_rope_geglu");
      } catch (const std::exception &) {
      }
      if (is_gemma) {
        npue::File gmodel(peek_path);
        return run_gemma_mode(gmodel, peek_path, root, argc, argv);
      }
    }
  }
  return -1;
}

}  // namespace app
