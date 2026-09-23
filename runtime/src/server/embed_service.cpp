#include "server/embed_service.hpp"
#include "runtime/model.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

namespace app {

std::vector<std::pair<int64_t, int64_t>> EmbedService::plan(int64_t n) const {
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

void EmbedService::chunk(npue::BertEncoder &e,
                         const std::vector<std::string> &texts,
                         int64_t base, int64_t take,
                         const std::string &prefix_text,
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

std::vector<float> EmbedService::embed(const std::vector<std::string> &texts,
                                        const std::string &prefix_text,
                                        int64_t *tokens) {
  std::vector<float> out(texts.size() * g_hidden, 0.f);
  const auto jobs = plan(static_cast<int64_t>(texts.size()));
  std::atomic<int64_t> tok_total{0};
  if (all.size() > 1 && jobs.size() > 1) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> ts;
    std::mutex emu;
    std::exception_ptr first_err;
    std::atomic<bool> stop{false};
    auto worker = [&](npue::BertEncoder *e) {
      for (size_t j = next++; j < jobs.size(); j = next++) {
        if (stop.load(std::memory_order_relaxed)) return;
        try {
          int64_t nt = 0;
          chunk(*e, texts, jobs[j].first, jobs[j].second, prefix_text,
                out, &nt);
          tok_total += nt;
        } catch (...) {
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

EmbedService make_service(RunContext &ctx) {
  EmbedService svc{load_tokenizer(*ctx.model, ctx.model_path),
                    ctx.model->raw("embeddings.word").as<float>(),
                    ctx.model->raw("embeddings.position").as<float>(),
                    ctx.model->raw("embeddings.token_type").as<float>(),
                    &*ctx.enc, {}, ctx.batch};
  svc.all.push_back(&*ctx.enc);
  for (auto &lp : ctx.lanes) svc.all.push_back(lp.get());
  return svc;
}

}  // namespace app