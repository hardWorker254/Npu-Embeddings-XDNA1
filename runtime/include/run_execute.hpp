#ifndef NPUEMBEDDINGS_RUN_EXECUTE_HPP
#define NPUEMBEDDINGS_RUN_EXECUTE_HPP

#include "run_setup.hpp"
#include "host_kernels.hpp"
#include "embed_service.hpp"
#include "cli.hpp"
#include "tokenizer_facade.hpp"
#include "npu_contention.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace app {

// ---------------------------------------------------------------------------
// EmbedService: the shared text-in / vectors-out service used by --embed
// (batch, from a file) and --serve (an OpenAI-shaped HTTP endpoint). Sharing
// it is the point: the endpoint cannot drift from the thing the tests measure.
// Split out of main.cpp verbatim.
// ---------------------------------------------------------------------------

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

// Build an EmbedService bound to the RunContext's encoder + model.
inline EmbedService make_service(RunContext &ctx) {
  // No prefix is resolved here any more (tasks/0118). --embed resolves one
  // from --prefix; --serve takes the name per request. Resolving it at
  // construction is what coupled a whole server process to one prompt.
  EmbedService svc{load_tokenizer(*ctx.model, ctx.model_path),
                   ctx.model->raw("embeddings.word").as<float>(),
                   ctx.model->raw("embeddings.position").as<float>(),
                   ctx.model->raw("embeddings.token_type").as<float>(),
                   &*ctx.enc, {}, ctx.batch};
  svc.all.push_back(&*ctx.enc);
  for (auto &lp : ctx.lanes) svc.all.push_back(lp.get());
  return svc;
}

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
inline int maybe_probe(RunContext &ctx) {
  for (int i = 2; i < ctx.argc; ++i) if (std::string(ctx.argv[i]) == "--probe") {
    const int reps = 100;
    std::printf("\n  dispatch probe -- %d repeats, no host work in the loop\n",
                reps);
    std::printf("    same design repeatedly:\n");
    for (npu::Design *d : {&ctx.d_qkv(), &ctx.d_ao(), &ctx.d_fu(), &ctx.d_fd(), &ctx.d_gelu(), &ctx.d_ln(),
                           &ctx.d_sm()}) {
      d->dispatch_only();                       // warm this context in
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) d->dispatch_only();
      double us = (now_s() - t0) / reps * 1e6;
      std::printf("      %-12s %8.0f us\n", d->info().name.c_str(), us);
    }
    std::printf("    alternating between two designs:\n");
    struct Pair { npu::Design *a, *b; const char *label; };
    for (Pair p : {Pair{&ctx.d_qkv(), &ctx.d_fu(), "qkv <-> ffn_up"},
                   Pair{&ctx.d_qkv(), &ctx.d_gelu(), "qkv <-> gelu"},
                   Pair{&ctx.d_ln(), &ctx.d_sm(), "layernorm <-> softmax"}}) {
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
  return -1;
}

inline int maybe_probe_streams(RunContext &ctx) {
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
  for (int i = 2; i < ctx.argc; ++i) if (std::string(ctx.argv[i]) == "--probe-streams") {
    if (!ctx.unified || ctx.streams.empty())
      throw std::runtime_error("--probe-streams needs a unified gemm_rtp set");
    const int reps = 30;
    const size_t cb = ctx.d_qkv().info().c_elem_bytes;
    // A/B element size and the N-tiling READ from the loaded design, exactly
    // as cb above is (T47, tasks/0124). This block used to hardcode 2 and
    // 48.0*8.0, which inflated every published int8 GB/s by 1.57-1.85x --
    // differentially, because C was counted correctly. A design.json that
    // predates the tile_n/cols fields is a refusal, not a guess.
    const size_t ab = ctx.d_qkv().info().a_elem_bytes;
    const int64_t tile_n = ctx.d_qkv().info().tile_n, cols = ctx.d_qkv().info().cols;
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
    for (const auto &st : ctx.streams) {
      ctx.d_qkv().bind_instr(static_cast<size_t>(st.slot));
      ctx.d_qkv().dispatch_only();                       // warm
      const double t0 = now_s();
      for (int r = 0; r < reps; ++r) ctx.d_qkv().dispatch_only();
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
  return -1;
}

// --- command dispatches (late) ---------------------------------------------
//
// These live here because they need the full Encoder + designs + goldens.
// Each returns -1 when its flag is absent (fall through) or >= 0 (exit code).

inline int maybe_embed(RunContext &ctx) {
  // --embed <textfile> [outfile]
  for (int i = 2; i < ctx.argc - 1; ++i)
      if (std::string(ctx.argv[i]) == "--embed") {
    const std::string in_path = ctx.argv[i + 1];
    std::string out_path;
    if (i + 2 < ctx.argc && ctx.argv[i + 2][0] != '-') out_path = ctx.argv[i + 2];

    auto svc = make_service(ctx);
    // --prefix is REQUIRED here for a model that has a prompts table, and
    // there is no container default any more -- see resolve_prefix().
    const std::string prefix_text = resolve_prefix(ctx.argc, ctx.argv);
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
  return -1;
}

inline int maybe_serve(RunContext &ctx) {
  // --serve [port]: an OpenAI-shaped POST /v1/embeddings endpoint.
  //
  // Requests are handled ONE AT A TIME on purpose. The NPU serializes
  // dispatches anyway (research/notes/0004), and the lanes already
  // parallelise inside a single request -- so concurrent request handling
  // would add contention and lock complexity to buy nothing. Throughput comes
  // from batching within a request, which is what an embeddings client does.
  for (int i = 2; i < ctx.argc; ++i) if (std::string(ctx.argv[i]) == "--serve") {
    int port = 8080;
    if (i + 1 < ctx.argc && std::isdigit(static_cast<unsigned char>(ctx.argv[i + 1][0])))
      port = std::atoi(ctx.argv[i + 1]);
    std::string bind_addr = "127.0.0.1";
    for (int k = 2; k < ctx.argc - 1; ++k)
      if (std::string(ctx.argv[k]) == "--bind") bind_addr = ctx.argv[k + 1];

    // --prefix is a PROCESS-WIDE setting and `serve` no longer has one
    // (tasks/0118). Refusing beats ignoring: a script that used to pin a
    // prefix here would otherwise keep running and quietly serve unprefixed
    // vectors, which is the same fail-open shape this task exists to remove.
    for (int k = 2; k < ctx.argc; ++k)
      if (std::string(ctx.argv[k]) == "--prefix")
        throw std::runtime_error(
            "--prefix does not apply to `serve`: the task prompt is chosen per "
            "request now. Send \"prompt_name\" in the POST body instead, and GET "
            "/health lists the names this model accepts.");

    auto svc = make_service(ctx);
    EmbedBackend be;
    be.vocab_size = svc.tok.vocab_size();
    be.prompt_names = prompt_names_sorted();
    be.hidden = g_hidden;
    be.seq = g_seq;
    // By reference: `svc` outlives serve_http(), which runs the accept loop
    // and only returns when the server stops. The name has already been
    // checked against be.prompt_names, so a miss here can only be the "" that
    // means no prefix at all.
    be.embed = [&svc](const std::vector<std::string> &t, const std::string &pn,
                      int64_t *n) {
      const auto it = g_prompts.find(pn);
      return svc.embed(t, it == g_prompts.end() ? std::string() : it->second,
                       n);
    };
    return serve_http(be, g_model_name + "-npu", port, bind_addr);
  }
  return -1;
}

inline int maybe_encode_file(RunContext &ctx) {
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
  for (int i = 2; i < ctx.argc - 1; ++i)
      if (std::string(ctx.argv[i]) == "--encode-file") {
    const std::string dir = ctx.argv[i + 1];
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
                (long long)n_rows, (long long)ctx.batch);
    std::vector<float> out(static_cast<size_t>(n_rows) * g_hidden, 0.f);

    const double t0 = now_s();
    for (int64_t base = 0; base < n_rows; base += ctx.batch) {
      const int64_t take = std::min<int64_t>(ctx.batch, n_rows - base);
      // Pad the tail chunk with zero rows; they are masked and discarded.
      std::vector<float> chunk(static_cast<size_t>(ctx.batch) * row_floats, 0.f);
      std::memcpy(chunk.data(), all_emb.data() + base * row_floats,
                  take * row_floats * sizeof(float));
      std::vector<float> cmask(static_cast<size_t>(ctx.batch) * g_seq, -1.0e30f);
      std::memcpy(cmask.data(), all_add.data() + base * g_seq,
                  take * g_seq * sizeof(float));
      std::vector<float> cam(static_cast<size_t>(ctx.batch) * g_seq, 0.f);
      std::memcpy(cam.data(), all_am.data() + base * g_seq,
                  take * g_seq * sizeof(float));

      ctx.enc->add_mask = cmask;
      auto h = ctx.enc->run(chunk);

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
  return -1;
}

inline int run_bench_or_check(RunContext &ctx) {
  if (ctx.bench > 0) ctx.need_goldens();

  // A timed run REFUSES to start when the array is not ours. tasks/0044 read
  // 221.4 seq/s against a true 694.0 because a leftover npuembed.exe from an
  // earlier session still held an Active hw_context, and nothing in this
  // banner said so. See include/npu_contention.hpp.
  if (ctx.bench > 0) {
    bool allow_contention = false;
    for (int i = 2; i < ctx.argc; ++i)
      if (std::string(ctx.argv[i]) == "--allow-contention") allow_contention = true;
    if (!npu::require_exclusive_npu(npu::survey_contexts(), allow_contention))
      return 2;
  }

  if (ctx.bench > 0 && ctx.pipeline > 1) {
    auto run_all = [&] {
      std::vector<std::thread> ts;
      for (auto &lp : ctx.lanes)
        ts.emplace_back([&, e = lp.get()] { e->run(ctx.emb_in); });
      ctx.enc->run(ctx.emb_in);
      for (auto &th : ts) th.join();
    };
    run_all();                                    // warm every lane
    ctx.enc->reset_timers();
    for (auto &lp : ctx.lanes) lp->reset_timers();
    npu::Design &d_qkv = ctx.d_qkv();
    d_qkv.t_submit = d_qkv.t_wait = 0.0;
    d_qkv.n_dispatch = 0;
    double w0 = now_s(), c0 = cpu_seconds();
    for (int i = 0; i < ctx.bench; ++i) run_all();
    double w1 = now_s(), c1 = cpu_seconds();
    const double wall = (w1 - w0) / ctx.bench, cpu = (c1 - c0) / ctx.bench;
    const int64_t seqs = ctx.pipeline * ctx.batch;
    std::printf("\n  %d pipelined groups of %d x %lld sequences at seq "
                "%lld\n", ctx.bench, ctx.pipeline, (long long)ctx.batch, (long long)g_seq);
    std::printf("    wall %8.2f ms   ->  %8.1f seq/s\n", wall * 1e3,
                seqs / wall);
    std::printf("    cpu  %8.2f ms   ->  %8.2f cores busy\n", cpu * 1e3,
                cpu / wall);
    const double npu_locked =
        (d_qkv.t_submit + d_qkv.t_wait) / ctx.bench;
    std::printf("    NPU dispatch+wait (serialized) %8.2f ms  %5.1f%%   "
                "%d dispatches/group\n",
                npu_locked * 1e3, npu_locked / wall * 100,
                d_qkv.n_dispatch / ctx.bench);
    auto lane = [&](int idx, const Encoder &e) {
      const double host = (e.t_conv + e.t_bias + e.t_attn + e.t_hostln +
                           e.t_hostsm + e.t_hostgelu) / ctx.bench;
      std::printf("    p%d host work                %8.2f ms  %5.1f%%   "
                  "(conv %.1f  bias %.1f  attn %.1f  elt %.1f)\n",
                  idx, host * 1e3, host / wall * 100, e.t_conv / ctx.bench * 1e3,
                  e.t_bias / ctx.bench * 1e3, e.t_attn / ctx.bench * 1e3,
                  (e.t_hostln + e.t_hostsm + e.t_hostgelu) / ctx.bench * 1e3);
    };
    lane(1, *ctx.enc);
    for (size_t l = 0; l < ctx.lanes.size(); ++l)
      lane(static_cast<int>(l) + 2, *ctx.lanes[l]);
    return 0;
  }

  if (ctx.bench > 0) {
    // Unique designs only: in unified mode all seven references alias ONE
    // Design, and summing it seven times reported 241% of wall.
    std::vector<npu::Design *> uniq;
    npu::Design &d_qkv = ctx.d_qkv();
    npu::Design &d_ao = ctx.d_ao();
    npu::Design &d_fu = ctx.d_fu();
    npu::Design &d_fd = ctx.d_fd();
    npu::Design &d_gelu = ctx.d_gelu();
    npu::Design &d_ln = ctx.d_ln();
    npu::Design &d_sm = ctx.d_sm();
    for (npu::Design *d : {&d_qkv, &d_ao, &d_fu, &d_fd, &d_gelu, &d_ln, &d_sm})
      if (std::find(uniq.begin(), uniq.end(), d) == uniq.end())
        uniq.push_back(d);
    ctx.enc->run(ctx.emb_in);                               // warm
    ctx.enc->reset_timers();
    for (npu::Design *d : uniq)
      { d->t_submit = d->t_wait = 0.0; d->n_dispatch = 0; }
    double w0 = now_s();
    double c0 = cpu_seconds();
    for (int i = 0; i < ctx.bench; ++i) ctx.enc->run(ctx.emb_in);
    double w1 = now_s();
    double c1 = cpu_seconds();
    double wall = (w1 - w0) / ctx.bench, cpu = (c1 - c0) / ctx.bench;
    std::printf("\n  %d encodes of %lld sequences at seq %lld\n", ctx.bench,
                (long long)ctx.batch, (long long)g_seq);
    std::printf("    wall %8.2f ms   ->  %8.1f seq/s\n", wall * 1e3,
                ctx.batch / wall);
    std::printf("    cpu  %8.2f ms   ->  %8.2f cores busy\n", cpu * 1e3,
                cpu / wall);

    // A single number says "slow". This says which half to fix.
    const double conv = ctx.enc->t_conv / ctx.bench, in = ctx.enc->t_in / ctx.bench;
    const double disp = ctx.enc->t_disp / ctx.bench, out = ctx.enc->t_out / ctx.bench;
    const double bias = ctx.enc->t_bias / ctx.bench;
    const double npu = conv + in + disp + out + bias;
    const double attn = ctx.enc->t_attn / ctx.bench;
    const int nd = ctx.enc->n_dispatch / ctx.bench;
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
      sub /= ctx.bench;
      wt /= ctx.bench;
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
                ctx.enc->t_qk / ctx.bench * 1e3, ctx.enc->t_av / ctx.bench * 1e3);
    if (ctx.enc->t_hostgelu > 0.0)
      std::printf("    host gelu                    %8.2f ms  %5.1f%%\n",
                  ctx.enc->t_hostgelu / ctx.bench * 1e3,
                  ctx.enc->t_hostgelu / ctx.bench / wall * 100);
    if (ctx.enc->t_hostsm > 0.0)
      std::printf("    host softmax                 %8.2f ms  %5.1f%%\n",
                  ctx.enc->t_hostsm / ctx.bench * 1e3,
                  ctx.enc->t_hostsm / ctx.bench / wall * 100);
    if (ctx.enc->t_hostln > 0.0)
      std::printf("    host layernorm               %8.2f ms  %5.1f%%\n",
                  ctx.enc->t_hostln / ctx.bench * 1e3,
                  ctx.enc->t_hostln / ctx.bench / wall * 100);
    // The three host eltwise kernels have their own lines above, so they must
    // come OUT of the residual bucket -- without this they were counted twice
    // and "everything else" read 24.4% on bge-large where the truth is 12.8%
    // (tasks/0081). An over-stated unexplained bucket is the worst kind of
    // wrong number: it points optimisation at a phantom.
    const double named = ctx.enc->t_hostgelu / ctx.bench + ctx.enc->t_hostsm / ctx.bench
                       + ctx.enc->t_hostln / ctx.bench;
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
                  d->n_dispatch / ctx.bench, macs,
                  d->t_wait / d->n_dispatch * 1e6);
    }
    return 0;
  }

  ctx.need_goldens();
  std::vector<float> hidden1;
  if (ctx.pipeline > 1) {
    std::vector<std::vector<float>> hs(ctx.lanes.size());
    std::vector<std::thread> ts;
    for (size_t l = 0; l < ctx.lanes.size(); ++l)
      ts.emplace_back([&, l] { hs[l] = ctx.lanes[l]->run(ctx.emb_in); });
    hidden1 = ctx.enc->run(ctx.emb_in);
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
                ctx.lanes.size() + 1, hidden1.size());
  } else {
    hidden1 = ctx.enc->run(ctx.emb_in);
  }
  auto emb = pool_normalise(ctx, hidden1);

  // Compare EVERY row, not just the first 4. `want` was tiled with the same
  // per-copy rotation as the inputs (see above), so this checks each of the
  // `batch` output rows against the specific golden row it is supposed to
  // reproduce -- previously this loop stopped at `kGoldenBatch` (4), so at
  // batch 128 it compared 4 of 128 rows and never read the other 124 at all.
  // That is a bigger hole than "the copies are indistinguishable" (T32): it
  // is truncation, and it means a corruption bug anywhere past row 3 was
  // never observed, let alone made indistinguishable by identical tiling.
  double num = 0.0, den = 0.0, worst_1mcos = 0.0;
  for (int64_t b = 0; b < ctx.batch; ++b) {
    double dot = 0.0;
    for (int64_t c = 0; c < g_hidden; ++c) {
      double diff = emb[b * g_hidden + c] - ctx.want[b * g_hidden + c];
      num += diff * diff;
      den += static_cast<double>(ctx.want[b * g_hidden + c]) * ctx.want[b * g_hidden + c];
      dot += static_cast<double>(emb[b * g_hidden + c]) * ctx.want[b * g_hidden + c];
    }
    worst_1mcos = std::max(worst_1mcos, 1.0 - dot);
  }
  const double rel_fro = std::sqrt(num) / std::sqrt(den);
  const double tol = 2e-3;

  std::printf("\n  %-38s %11.3e\n", "embedding rel_fro vs HF golden", rel_fro);
  std::printf("  %-38s %11.3e  (all %lld rows, %lld distinct sentences "
              "rotated across tile copies)\n",
              "worst 1 - cos vs HuggingFace", worst_1mcos, (long long)ctx.batch,
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

}

}  // namespace app

#endif  // NPUEMBEDDINGS_RUN_EXECUTE_HPP
