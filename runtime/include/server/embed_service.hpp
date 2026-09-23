#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/app_state.hpp"
#include "common/host_kernels.hpp"
#include "runtime/types.hpp"
#include "runtime/run_context.hpp"
#include "encoders/bert_encoder.hpp"
#include "server/embed_backend.hpp"
#include "tokenizers/tokenizer_facade.hpp"

namespace app {

// The BERT-family text-in / vectors-out service. The raw-forward methods it
// needs (`run`, `use_tier`, `tiers`, `add_mask`) are concrete on
// npue::BertEncoder, not on the npue::Encoder text-in interface (Task 0), so
// this service names that concrete type directly.
struct EmbedService {
  AnyTokenizer tok;
  const float *w_word, *w_pos, *w_typ;
  npue::BertEncoder *lead;
  std::vector<npue::BertEncoder *> all;
  int64_t fallback_batch;

  std::vector<std::pair<int64_t, int64_t>> plan(int64_t n) const;
  void chunk(npue::BertEncoder &e, const std::vector<std::string> &texts,
             int64_t base, int64_t take, const std::string &prefix_text,
             std::vector<float> &out, int64_t *tokens) const;
  std::vector<float> embed(const std::vector<std::string> &texts,
                           const std::string &prefix_text,
                           int64_t *tokens = nullptr);
};

EmbedService make_service(RunContext &ctx);

}  // namespace app