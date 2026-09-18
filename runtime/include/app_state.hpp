//===- app_state.hpp ----------------------------------------*- C++ -*-===//
//
// Process-wide model state, split out of main.cpp (code verbatim): the model
// geometry read from the .npue container, the task-prefix table, the
// truncation and pooling policy. set_model_shape() is the only writer of the
// geometry; it runs once before any encoder exists.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "json_min.hpp"
#include "npue.hpp"

namespace app {

// Model geometry, READ FROM THE CONTAINER at startup rather than compiled in.
//
// This runtime serves several models -- MiniLM-L6 (6 layers, mean pooling),
// bge-small (12 layers, CLS pooling), bge-large (24 layers, hidden 1024,
// head_dim 64) -- and their depth, width and pooling all differ. As
// constexpr, `g_layers = 6` would have run a 12-layer model for six layers and
// returned a plausible wrong vector with exit code 0.
//
// They are file scope and mutable because they are read at ~150 sites;
// threading a struct through all of them would be a large diff for no
// behavioural gain. set_model_shape() is the ONLY writer, it runs once before
// any Encoder exists, and every value starts at 0 so that a missed
// initialisation divides by zero or allocates nothing rather than quietly
// using a stale MiniLM number.
// head_dim / 8, bounded so the attention kernels can hold the vectors on the
// stack. 16 covers head_dim up to 128; every BERT-family encoder we target is
// 32 or 64.
constexpr int kMaxHeadVecs = 16;

inline int64_t g_seq = 0, g_hidden = 0, g_heads = 0, g_head_dim = 0;
int64_t g_ffn = 0, g_layers = 0, g_max_positions = 0;
inline bool g_cls_pool = false, g_l2_normalize = true;

// TRUNCATION POLICY. Default: refuse. See npue::InputTooLong for why this is
// worth a flag rather than a constant.
//
// The default is a BEHAVIOUR CHANGE and deliberately so -- it is the whole
// point of the change. It cannot move any measured number, though: an input
// that fitted before still fits and still produces the bit-identical vector,
// and the only inputs whose behaviour changes are the ones that were being
// silently answered wrong. --allow-truncation restores the old behaviour
// exactly, and says so on stderr each time it fires.
inline bool g_allow_truncation = false;
// Set once truncation has actually been permitted and taken, so the warning
// is printed once per run rather than once per row (a 2,048-input request
// would otherwise emit 2,048 identical lines).
inline std::atomic<bool> g_truncation_warned{false};

// One place, so the message is the same wherever the cut is detected.
// `index` is the caller's own input index -- a global row number, not an
// offset into whatever tier the runtime happened to batch it into, because
// the caller cannot see tiers.
inline void check_truncation(bool truncated, int32_t n_tokens_full, size_t index,
                      int64_t limit) {
  if (!truncated) return;
  if (!g_allow_truncation) throw npue::InputTooLong(index, n_tokens_full, limit);
  if (!g_truncation_warned.exchange(true))
    std::fprintf(stderr,
                 "  WARNING  --allow-truncation: input %zu is %d tokens and "
                 "was CUT to %lld. Its vector is a vector of the first %lld "
                 "tokens, not of the text. Further cuts this run are not "
                 "reported.\n",
                 index, static_cast<int>(n_tokens_full), (long long)limit,
                 (long long)limit);
}
inline std::string g_model_name, g_source_repo;

// arch=2 (nomic-embed-text-v1.5, tasks/0069-0070): RoPE on Q/K inside the
// fused qkv buffer, and a gated SwiGLU FFN in place of plain GELU. Both are
// false/0 for every arch=0 (BERT) container -- set_model_shape() is the only
// writer, same discipline as every other g_* geometry field above.
inline bool g_rope = false, g_gated_ffn = false;
inline double g_rope_theta = 0.0;
// arch=3 (gte-multilingual-base, tasks/0134-0136): the RoPE frequency set IS
// the model -- inv_freq_i = 160000^(-i/32) / 8^(1/32), which is NOT
// expressible as any single theta (the NTK correction is a constant factor,
// not a power law; deriving from rope_theta alone is wrong by 1.9e-02 relfro
// at layer 0, measured in tasks/0134). Read from the container's
// "rope_inv_freq" config array; EMPTY for every other arch, in which case
// g_rope_theta is the source. set_model_shape() is the only writer.
inline std::vector<float> g_rope_inv_freq;
// Which activation the gated FFN applies to its gate half. SiLU used to be
// hardcoded while the container's "activation" key was write-only (T33's
// latent key, made load-bearing by tasks/0135): arch=2 says "silu", arch=3
// says "gelu" -- torch's default EXACT erf GELU, NOT gelu8's polynomial and
// NOT Gemma's tanh approximation. An unknown value REFUSES at load.
// set_model_shape() is the only writer; irrelevant when g_gated_ffn is false.
enum class GatedAct { Silu, GeluErf };
inline GatedAct g_gated_act = GatedAct::Silu;

// Exact erf GELU: 0.5*x*(1+erf(x/sqrt(2))), computed in double like the
// numpy oracle (reference/encoder_gte.py's gelu_exact* both erf in float64)
// and rounded once at the end. Scalar on purpose: correctness first, and the
// bfp16 datapath noise (~2e-04) is three decades above the double-vs-float
// difference this choice removes from the comparison.
inline float gelu_erf_exact(float x) {
  const double xd = static_cast<double>(x);
  return static_cast<float>(
      0.5 * xd * (1.0 + std::erf(xd * 0.70710678118654752440)));
}

// nomic's task-prefix table (tasks/0071): name -> literal prefix text, e.g.
// "search_document" -> "search_document: ". Empty for every container that
// carries no "prompts" key -- the four BERT models have no prefix concept at
// all, and `g_prompts.empty()` is the single source of truth for that,
// rather than a second bool that could drift from it. set_model_shape() is
// the only writer, cleared unconditionally on every call for the same
// "no container can leak state into the next" reason as g_rope/g_gated_ffn.
inline std::map<std::string, std::string> g_prompts;
inline std::string g_prompt_default;

// The container's prompt names, sorted, and a formatter for them. ONE source,
// used by every refusal that has to list them -- the CLI's, the endpoint's and
// /health's -- so the three cannot drift into disagreeing about what this
// model offers.
inline std::vector<std::string> prompt_names_sorted() {
  std::vector<std::string> names;
  names.reserve(g_prompts.size());
  for (const auto &kv : g_prompts) names.push_back(kv.first);
  std::sort(names.begin(), names.end());
  return names;
}

inline std::string join_names(const std::vector<std::string> &names) {
  std::string list;
  for (size_t i = 0; i < names.size(); ++i) list += (i ? ", " : "") + names[i];
  return list;
}

// Batch is NOT a constant: it is read back from the loaded design's M, so the
// runtime cannot disagree with the xclbin it was handed. Every GEMM in the
// encoder is over all tokens of all sequences at once, so batching is purely a
// larger M -- and it is the lever that survives tasks/0024, because the 49
// design switches per encode cost the same no matter how many sequences that
// encode carries.

// A config key added after the container format already shipped has to be
// readable from containers that predate it. This repo's rule is that a missing
// key THROWS rather than defaulting -- right for geometry, which must never be
// guessed -- so the back-compat default is stated explicitly, here, at the one
// place it applies, rather than by weakening config_string(). Same shape as the
// pre-0036 `tokenizer.vocab` fallback in load_tokenizer().
//
// `config_string` returns the raw JSON scalar text, so a JSON `true` arrives as
// the four characters "true".
// Which container architectures this build can actually EXECUTE, as opposed to
// merely load. Kept in one place so the `list` table and the dispatch-time
// refusal in set_model_shape() cannot drift apart: a table that says "ready"
// while dispatch throws is its own kind of lie, just a politer one.
//
// A packed container and a matching design are NOT sufficient. arch=2 (nomic)
// deliberately reuses BERT's tensor names and shapes so the packer and the NPU
// dispatch path work unchanged -- which means the BERT encoder will read it
// happily and compute the wrong model. tasks/0069.
inline bool encoder_implemented(const std::string &arch) {
  return arch == "bert_abs_gelu_postln" ||      // Encoder, NPU GEMM path
         arch == "nomic_bert_rope_swiglu" ||    // Encoder, NPU GEMM path (0070)
         // GemmaNpuEncoder on the array, or the host-only npue::GemmaEncoder
         // when the container is not pre-tiled. Which one runs is decided from
         // the container's `gemm_layout`, in run_gemma_mode() (tasks/0074).
         arch == "gemma3_mqa_rope_geglu" ||
         // gte-multilingual-base (0.5.0): BERT tensor names ON PURPOSE, so
         // the packer and the NPU dispatch path serve it unchanged; the
         // encoder deltas -- RoPE from rope_inv_freq, exact-erf GELU on the
         // gate half, real biases, XLM-R Unigram tokenizer -- are all
         // data-driven off the container (tasks/0134-0136).
         arch == "gte_new_rope_geglu";
}

inline bool config_flag(const npue::File &f, const char *key, bool fallback) {
  try {
    return f.config_string(key) == "true";
  } catch (const std::exception &) {
    return fallback;   // container predates the key: arch 0 and 1 are ungated
  }
}
// Pool [take, seq, hidden] hidden states into [take, hidden], then optionally
// L2 normalise. ONE implementation: there were three, and they disagreed --
// the golden path accumulated in float while the other two used double, so a
// comment claiming they matched was wrong by a rounding.
//
// `am` is the 1/0 attention mask, [rows, seq].
inline void pool_rows(const float *h, const float *am, int64_t take, float *out) {
  std::vector<double> acc(static_cast<size_t>(g_hidden));
  for (int64_t b = 0; b < take; ++b) {
    const float *amb = am + b * g_seq;
    const float *hb = h + b * g_seq * g_hidden;

    if (g_cls_pool) {
      // The [CLS] token is position 0 by construction (tokenizer.cpp emits it
      // first). If it is masked the sequence is empty, and returning zeros
      // would be a silently plausible answer.
      if (amb[0] == 0.f)
        throw std::runtime_error("CLS pooling on a sequence whose first "
                                 "token is masked");
      for (int64_t c = 0; c < g_hidden; ++c) acc[c] = hb[c];
    } else {
      float denom = 0.f;
      for (int64_t s = 0; s < g_seq; ++s) denom += amb[s];
      denom = std::max(denom, 1e-9f);
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int64_t s = 0; s < g_seq; ++s) {
        const float m = amb[s];
        if (m == 0.f) continue;
        const float *hr = hb + s * g_hidden;
        for (int64_t c = 0; c < g_hidden; ++c) acc[c] += hr[c] * m;
      }
      for (int64_t c = 0; c < g_hidden; ++c) acc[c] /= denom;
    }

    float *o = out + b * g_hidden;
    if (g_l2_normalize) {
      double nrm = 0.0;
      for (int64_t c = 0; c < g_hidden; ++c) nrm += acc[c] * acc[c];
      nrm = std::sqrt(std::max(nrm, 1e-24));
      for (int64_t c = 0; c < g_hidden; ++c)
        o[c] = static_cast<float>(acc[c] / nrm);
    } else {
      for (int64_t c = 0; c < g_hidden; ++c) o[c] = static_cast<float>(acc[c]);
    }
  }
}

// Populate the geometry from the container. Every value is REQUIRED: a
// missing key throws from npue::File rather than defaulting, because a
// default here is indistinguishable from a correct value and this project has
// shipped six bugs of exactly that shape.
inline void set_model_shape(npue::File &m) {
  // FAIL CLOSED ON AN ARCHITECTURE THIS BINARY DOES NOT IMPLEMENT.
  //
  // Encoder::run() started as a pure BERT forward pass: absolute positions, a
  // plain GELU FFN, no rotary anything. arch=2 (nomic) deliberately reuses
  // BERT's tensor names and shapes -- that is what makes the packer and NPU
  // dispatch path free -- so a container this build does NOT implement would
  // otherwise be read happily and run through the wrong math, returning
  // embeddings that look entirely reasonable. Nothing downstream could tell.
  //
  // The arch=1 (Gemma) containers are dispatched away before they reach here,
  // but that check lives at three call sites and works by naming the ONE arch
  // it diverts; anything it does not recognise falls through to this path. So
  // the guard has to be here, stated as a whitelist of what is IMPLEMENTED
  // rather than a blacklist of what is not. tasks/0069, extended for arch=2
  // in tasks/0070 -- encoder_implemented() is the single source both this
  // refusal and the `list`/`serve` tables read, so they cannot drift.
  const std::string arch = m.config_string("arch");
  // SINGLE-SOURCED as of tasks/0136: this refusal used to restate the
  // whitelist as inline string literals while the comment above claimed
  // encoder_implemented() was "the single source" -- so a new arch could
  // land in one list and still be refused by the other. It calls the real
  // list now. The gemma diversion stays as-is: arch=1 IS implemented, but by
  // run_gemma_mode(), not by this BERT-family path -- a gemma container
  // reaching here means that diversion was bypassed, and running it through
  // the setup below would be the exact fail-open this guard exists to stop.
  if (!encoder_implemented(arch) || arch == "gemma3_mqa_rope_geglu")
    throw std::runtime_error(
        "container architecture '" + arch + "' has no encoder in this build. "
        "The NPU GEMM designs for it may well be present -- the tensor names "
        "and shapes are shared with BERT on purpose -- but running it through "
        "the BERT encoder would silently return embeddings for the wrong "
        "model. Refusing.");

  g_layers = m.config_int("num_layers");
  g_hidden = m.config_int("hidden");
  g_heads = m.config_int("num_heads");
  g_head_dim = m.config_int("head_dim");
  g_ffn = m.config_int("intermediate");
  g_source_repo = m.config_string("source_repo");
  // NOT g_seq: `max_seq_len` is how many position embeddings were packed,
  // which is 256 while the designs are compiled for 64. The sequence length
  // belongs to the design and is set by set_design_seq().
  g_max_positions = m.config_int("max_seq_len");

  // arch=2: RoPE on Q/K (never V) inside the fused qkv buffer, and a gated
  // SwiGLU FFN. Both default false/0 -- deterministic every call, so a BERT
  // container after a nomic one in the same process (there is none today,
  // but nothing enforces that) cannot inherit stale state.
  g_gated_ffn = config_flag(m, "gated_ffn", false);
  g_rope = false;
  g_rope_theta = 0.0;
  g_rope_inv_freq.clear();
  g_gated_act = GatedAct::Silu;
  if (arch == "nomic_bert_rope_swiglu") {
    const std::string pet = m.config_string("position_embedding_type");
    if (pet != "rope")
      throw std::runtime_error(
          "container arch is nomic_bert_rope_swiglu but "
          "position_embedding_type is '" + pet + "', expected 'rope' -- "
          "refusing rather than guessing how position is encoded");
    if (!g_gated_ffn)
      throw std::runtime_error(
          "container arch is nomic_bert_rope_swiglu but gated_ffn is not "
          "true -- refusing rather than running a plain (ungated) FFN over "
          "a fused fc11|fc12 weight");
    // swiglu_halves pins which half of the fused ffn_up gets SiLU. READ IT,
    // do not trust the constant -- tools/pack_npue.py writes this exact
    // string today, but a packer that silently changed the fusion order
    // would otherwise compute out = silu(fc11(x)) * fc12(x), the wrong
    // candidate tasks/0068 Q2 measured at rel_fro 4.022e+00 (2.5e7x worse).
    const std::string halves = m.config_string("swiglu_halves");
    if (halves != "fc11_up|fc12_gate")
      throw std::runtime_error(
          "unrecognised swiglu_halves ordering '" + halves + "' -- expected "
          "'fc11_up|fc12_gate'; refusing rather than guessing which half of "
          "the fused ffn_up gets SiLU");
    g_rope_theta = m.config_double("rope_theta");
    if (g_rope_theta <= 0.0)
      throw std::runtime_error(
          "nomic_bert_rope_swiglu container has a non-positive rope_theta");
    g_rope = true;
  }

  // arch=3 (tasks/0136): nomic's shape with three deltas, every one read
  // from the container rather than assumed -- exact-erf GELU on the gate
  // half (the "activation" key, below), real biases (the bias slots are
  // added unconditionally, so nothing here changes), and a RoPE frequency
  // set that is DATA, because no single theta can express it (tasks/0134).
  if (arch == "gte_new_rope_geglu") {
    const std::string pet = m.config_string("position_embedding_type");
    if (pet != "rope")
      throw std::runtime_error(
          "container arch is gte_new_rope_geglu but position_embedding_type "
          "is '" + pet + "', expected 'rope' -- refusing rather than "
          "guessing how position is encoded");
    if (!g_gated_ffn)
      throw std::runtime_error(
          "container arch is gte_new_rope_geglu but gated_ffn is not true -- "
          "refusing rather than running a plain (ungated) FFN over a fused "
          "up|gate weight");
    // Same job as arch=2's swiglu_halves assert: pin which half of the fused
    // ffn_up is the gate. The key is descriptive prose after the marker, so
    // match the marker prefix, not the whole string.
    const std::string halves = m.config_string("glu_halves");
    if (halves.rfind("up_first|gate_second", 0) != 0)
      throw std::runtime_error(
          "unrecognised glu_halves ordering '" + halves + "' -- expected it "
          "to begin 'up_first|gate_second'; refusing rather than guessing "
          "which half of the fused ffn_up gets the activation");
    // rope_inv_freq IS the model (tasks/0134): inv_freq_i =
    // 160000^(-i/32) / 8^(1/32). A container without it REFUSES -- falling
    // back to deriving from rope_theta is measured wrong by 1.9e-02 relfro
    // at layer 0, and silently so.
    std::string raw_freq;
    try {
      raw_freq = m.config_string("rope_inv_freq");
    } catch (const std::exception &) {
      throw std::runtime_error(
          "gte_new_rope_geglu container carries no 'rope_inv_freq' -- the "
          "frequency set is not derivable from rope_theta (wrong by 1.9e-02 "
          "relfro at layer 0, tasks/0134), so refusing rather than falling "
          "back. Repack with tools/pack_npue.py");
    }
    const npue::json::Value v = npue::json::parse(raw_freq);
    for (const auto &e : v.as_array())
      g_rope_inv_freq.push_back(static_cast<float>(e.as_number()));
    if (static_cast<int64_t>(g_rope_inv_freq.size()) != g_head_dim / 2)
      throw std::runtime_error(
          "rope_inv_freq has " + std::to_string(g_rope_inv_freq.size()) +
          " entries, expected head_dim/2 = " +
          std::to_string(g_head_dim / 2));
    g_rope = true;
  }

  // The gated activation is DATA (tasks/0135 made the write-only key
  // load-bearing). Missing on an arch=2 container means "silu" -- packed
  // nomic containers may predate the read -- but an arch=3 container
  // without it is malformed, and an unknown value refuses on either arch.
  if (g_gated_ffn && arch != "gemma3_mqa_rope_geglu") {
    std::string act;
    try {
      act = m.config_string("activation");
    } catch (const std::exception &) {
      if (arch == "gte_new_rope_geglu")
        throw std::runtime_error(
            "gte_new_rope_geglu container carries no 'activation' key -- "
            "refusing rather than guessing which activation the gate half "
            "gets");
      act = "silu";
    }
    if (act == "silu")
      g_gated_act = GatedAct::Silu;
    else if (act == "gelu")
      g_gated_act = GatedAct::GeluErf;
    else
      throw std::runtime_error(
          "unknown gated-FFN activation '" + act + "' -- this build "
          "implements 'silu' (SiLU, arch=2) and 'gelu' (exact erf GELU, "
          "arch=3); refusing rather than substituting one");
  }

  // The task-prefix table (tasks/0071). Optional: the four BERT models'
  // containers carry no "prompts" key at all, and that has to leave
  // g_prompts genuinely empty -- not throw -- so resolve_prefix() below can
  // use emptiness as "this model has no prefix concept" without a second
  // flag that could drift from it. config_string() throws on a missing key,
  // so the absence check is a try/catch, same shape as config_flag() above.
  g_prompts.clear();
  g_prompt_default.clear();
  {
    std::string raw;
    try {
      raw = m.config_string("prompts");
    } catch (const std::exception &) {
      // No "prompts" key -- this container has no task-prefix concept.
      // g_prompts stays empty, which IS the "no prefix" signal.
    }
    if (!raw.empty()) {
      const npue::json::Value v = npue::json::parse(raw);
      for (const auto &kv : v.as_object())
        g_prompts[kv.first] = kv.second.as_string();
      if (g_prompts.empty())
        throw std::runtime_error(
            "container has a 'prompts' key but it parsed to zero entries -- "
            "refusing rather than silently running with no prefix");
      // prompt_default is REQUIRED once prompts exists: a container that
      // advertises a prefix table but names no default is malformed, not
      // merely prefix-less.
      g_prompt_default = m.config_string("prompt_default");
      if (g_prompts.find(g_prompt_default) == g_prompts.end())
        throw std::runtime_error(
            "prompt_default '" + g_prompt_default + "' is not a key in "
            "this container's own prompts table");
    }
  }

  // Pooling is data. sentence-transformers ships the answer in
  // 1_Pooling/config.json and the packer copies it here; a container that
  // predates that carries "mean", which is what MiniLM wants anyway.
  const std::string pool = m.config_string("pooling");
  if (pool == "cls") g_cls_pool = true;
  else if (pool == "mean") g_cls_pool = false;
  else throw std::runtime_error("unknown pooling mode '" + pool +
                                "' in the .npue -- expected mean or cls");

  // SO IS NORMALISATION, and it was a literal that should have been data
  // (T33). `pack_npue.py` has always written `l2_normalize` into every
  // container and this runtime always ignored it, which stayed harmless only
  // because every model so far wanted `true`.
  //
  // Reading it changes NOTHING today -- every shipped container says true, and
  // the default here is still true for a container that predates the key. What
  // it changes is that the assumption is now stated by the data rather than
  // asserted by the binary, so deciding nomic's case becomes a repack instead
  // of a code change. The decision itself is still open: nomic's
  // sentence-transformers pipeline has no `Normalize` module and returns
  // vectors of norm ~20.9, and 0073 measured Banking77 at 83.77 unnormalised
  // against 79.23 normalised -- 4.5 points on a logistic-regression task,
  // invisible to cosine. That is a question about which geometry downstream
  // code wants, not a precision question, and it is the user's to answer.
  g_l2_normalize = config_flag(m, "l2_normalize", true);

  if (g_layers <= 0 || g_hidden <= 0 || g_heads <= 0 || g_max_positions <= 0)
    throw std::runtime_error("the .npue reports a non-positive shape");
  if (g_head_dim * g_heads != g_hidden)
    throw std::runtime_error("head_dim * heads != hidden in the .npue");
  if (g_head_dim % 8 || g_head_dim / 8 > kMaxHeadVecs)
    throw std::runtime_error(
        "head_dim " + std::to_string(g_head_dim) + " must be a multiple of 8 "
        "and at most " + std::to_string(kMaxHeadVecs * 8) +
        " for the host attention kernels");
  // The host AVX2 paths step 8 floats with no scalar tail.
  if (g_hidden % 8)
    throw std::runtime_error("this runtime requires hidden to be a multiple "
                             "of 8");
}

// The sequence length comes from the design, and the container has to be able
// to feed it. Two independent sources that must agree in one direction: a
// design asking for more positions than were packed would index past the
// position table.
inline void set_design_seq(int64_t seq) {
  if (seq <= 0 || seq % 8)
    throw std::runtime_error("design seq " + std::to_string(seq) +
                             " must be positive and a multiple of 8");
  if (seq > g_max_positions)
    throw std::runtime_error(
        "design seq " + std::to_string(seq) + " exceeds the " +
        std::to_string(g_max_positions) + " position embeddings in the .npue");
  g_seq = seq;
}

}  // namespace app
