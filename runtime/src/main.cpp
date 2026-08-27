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
#include <windows.h>

namespace {

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

int64_t g_seq = 0, g_hidden = 0, g_heads = 0, g_head_dim = 0;
int64_t g_ffn = 0, g_layers = 0, g_max_positions = 0;
bool g_cls_pool = false, g_l2_normalize = true;

// TRUNCATION POLICY. Default: refuse. See npue::InputTooLong for why this is
// worth a flag rather than a constant.
//
// The default is a BEHAVIOUR CHANGE and deliberately so -- it is the whole
// point of the change. It cannot move any measured number, though: an input
// that fitted before still fits and still produces the bit-identical vector,
// and the only inputs whose behaviour changes are the ones that were being
// silently answered wrong. --allow-truncation restores the old behaviour
// exactly, and says so on stderr each time it fires.
bool g_allow_truncation = false;
// Set once truncation has actually been permitted and taken, so the warning
// is printed once per run rather than once per row (a 2,048-input request
// would otherwise emit 2,048 identical lines).
std::atomic<bool> g_truncation_warned{false};

// One place, so the message is the same wherever the cut is detected.
// `index` is the caller's own input index -- a global row number, not an
// offset into whatever tier the runtime happened to batch it into, because
// the caller cannot see tiers.
void check_truncation(bool truncated, int32_t n_tokens_full, size_t index,
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
std::string g_model_name, g_source_repo;

// arch=2 (nomic-embed-text-v1.5, tasks/0069-0070): RoPE on Q/K inside the
// fused qkv buffer, and a gated SwiGLU FFN in place of plain GELU. Both are
// false/0 for every arch=0 (BERT) container -- set_model_shape() is the only
// writer, same discipline as every other g_* geometry field above.
bool g_rope = false, g_gated_ffn = false;
double g_rope_theta = 0.0;
// arch=3 (gte-multilingual-base, tasks/0134-0136): the RoPE frequency set IS
// the model -- inv_freq_i = 160000^(-i/32) / 8^(1/32), which is NOT
// expressible as any single theta (the NTK correction is a constant factor,
// not a power law; deriving from rope_theta alone is wrong by 1.9e-02 relfro
// at layer 0, measured in tasks/0134). Read from the container's
// "rope_inv_freq" config array; EMPTY for every other arch, in which case
// g_rope_theta is the source. set_model_shape() is the only writer.
std::vector<float> g_rope_inv_freq;
// Which activation the gated FFN applies to its gate half. SiLU used to be
// hardcoded while the container's "activation" key was write-only (T33's
// latent key, made load-bearing by tasks/0135): arch=2 says "silu", arch=3
// says "gelu" -- torch's default EXACT erf GELU, NOT gelu8's polynomial and
// NOT Gemma's tanh approximation. An unknown value REFUSES at load.
// set_model_shape() is the only writer; irrelevant when g_gated_ffn is false.
enum class GatedAct { Silu, GeluErf };
GatedAct g_gated_act = GatedAct::Silu;

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
std::map<std::string, std::string> g_prompts;
std::string g_prompt_default;

// The container's prompt names, sorted, and a formatter for them. ONE source,
// used by every refusal that has to list them -- the CLI's, the endpoint's and
// /health's -- so the three cannot drift into disagreeing about what this
// model offers.
std::vector<std::string> prompt_names_sorted() {
  std::vector<std::string> names;
  names.reserve(g_prompts.size());
  for (const auto &kv : g_prompts) names.push_back(kv.first);
  std::sort(names.begin(), names.end());
  return names;
}

std::string join_names(const std::vector<std::string> &names) {
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

// ---------------------------------------------------------------------------
// Which model to run.
//
// Four sites used to name all-MiniLM-L6-v2 as a literal. The set of installed
// models is now whatever is in models/*.npue, and everything shown about them
// is read from the containers -- there is no list of models in this binary.

struct ModelEntry {
  std::string path, name, repo, pooling, arch, error, gemm_layout;
  int64_t layers = 0, hidden = 0, heads = 0, head_dim = 0, ffn = 0, seq = 0;
  // 0 = the container did not say, i.e. every BERT and nomic container, for
  // which the fused qkv really is 3*hidden. An MQA/GQA container states it
  // (tasks/0074) and it is NOT derivable from anything else here.
  int64_t qkv_n = 0;
  bool gated_ffn = false;
  double mb = 0;
};

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
bool encoder_implemented(const std::string &arch) {
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

bool config_flag(const npue::File &f, const char *key, bool fallback) {
  try {
    return f.config_string(key) == "true";
  } catch (const std::exception &) {
    return fallback;   // container predates the key: arch 0 and 1 are ungated
  }
}

std::vector<ModelEntry> discover_models(const std::string &root) {
  namespace fs = std::filesystem;
  std::vector<ModelEntry> v;
  std::error_code ec;
  const fs::path dir = fs::path(root) / "models";
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    if (it->path().extension() != ".npue") continue;
    ModelEntry m;
    m.path = it->path().string();
    m.name = it->path().stem().string();
    // A container that will not open is LISTED with its error rather than
    // skipped: a model silently missing from the table is a worse failure
    // than one that is visibly broken.
    try {
      npue::File f(m.path);
      m.repo = f.config_string("source_repo");
      m.pooling = f.config_string("pooling");
      m.arch = f.config_string("arch");
      m.layers = f.config_int("num_layers");
      m.hidden = f.config_int("hidden");
      m.heads = f.config_int("num_heads");
      m.head_dim = f.config_int("head_dim");
      m.ffn = f.config_int("intermediate");
      m.gated_ffn = config_flag(f, "gated_ffn", false);
      try {
        m.qkv_n = f.config_int("qkv_n");
      } catch (const std::exception &) {
        m.qkv_n = 0;   // predates the key; 3*hidden is right for those
      }
      // Whether this container's GEMM operands are tiled bf16 for the array or
      // plain row-major F32 for a host forward pass (tasks/0074). A container
      // that predates the key is arch=1 host-only if it is Gemma, and tiled
      // otherwise -- every arch=0/2 container ever written is tiled.
      try {
        m.gemm_layout = f.config_string("gemm_layout");
      } catch (const std::exception &) {
        m.gemm_layout = (m.arch == "gemma3_mqa_rope_geglu") ? "host"
                                                            : "pretiled_bf16";
      }
      m.seq = f.config_int("max_seq_len");
      m.mb = f.data_length() / 1e6;
    } catch (const std::exception &e) {
      m.error = e.what();
    }
    v.push_back(std::move(m));
  }
  std::sort(v.begin(), v.end(),
            [](const ModelEntry &a, const ModelEntry &b) {
              return a.name < b.name;
            });
  return v;
}

void print_model_table(const std::vector<ModelEntry> &v) {
  std::printf("\nInstalled models (from %s):\n\n",
              "models/*.npue");
  std::printf("  %-24s %6s %7s %7s %6s %8s  %s\n", "--model", "layers",
              "hidden", "pooling", "MB", "max seq", "source");
  for (const auto &m : v) {
    if (!m.error.empty()) {
      std::printf("  %-24s  UNREADABLE: %s\n", m.name.c_str(),
                  m.error.c_str());
      continue;
    }
    std::printf("  %-24s %6lld %7lld %7s %6.0f %8lld  %s\n", m.name.c_str(),
                (long long)m.layers, (long long)m.hidden, m.pooling.c_str(),
                m.mb, (long long)m.seq, m.repo.c_str());
  }
  std::printf("\n  Wider and deeper models score better and run slower; the\n"
              "  measured throughput and MTEB for each are in docs/.\n\n");
}

// Resolve --model to a container. Accepts a name as printed in the table or a
// path to a .npue directly.
std::string resolve_model_path(const std::string &root, int argc,
                               char **argv) {
  std::string want;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--model") want = argv[i + 1];

  if (!want.empty() && want.size() > 5 &&
      want.compare(want.size() - 5, 5, ".npue") == 0 &&
      std::ifstream(want).good())
    return want;

  const auto models = discover_models(root);
  if (models.empty())
    throw std::runtime_error(
        "no models/*.npue under " + root + " -- build one with "
        "`npuembed --prepare-model <checkpoint-dir>`; see BUILD.md");

  if (want.empty()) {
    // AMBIGUITY is what makes --model required. One installed model is not
    // ambiguous, and demanding the flag would only make the user type the
    // single possible answer. Two are, and choosing for them is how this
    // project's fail-open bugs have always looked.
    if (models.size() == 1) return models[0].path;
    print_model_table(models);
    throw std::runtime_error(
        "several models are installed; say which with --model <name>");
  }

  for (const auto &m : models)
    if (m.name == want) {
      if (!m.error.empty())
        throw std::runtime_error("model " + want + " will not open: " +
                                 m.error);
      return m.path;
    }
  print_model_table(models);
  throw std::runtime_error("no model named '" + want + "' is installed");
}

// The vocabulary lives inside the .npue as of 0036, so a deployed model is
// ONE file. A model packed before that still works: fall back to the loose
// vocab.txt and say so, rather than failing on a file that is merely older.
// One tokenizer interface for the BERT-family encode path, two tokenizer
// families behind it (tasks/0136). The facade lives HERE rather than giving
// XlmrTokenizer a WordPiece-shaped encode(), because the max_len /
// padding / truncation semantics are this runtime's policy (0110's
// refuse-on-overflow contract runs on the Encoded fields), not a property
// of the Unigram algorithm -- tokenizer_xlmr.cpp stays the line-for-line
// port of its Python reference, diffable function by function. Chosen over
// branching at the call sites because the encode() calls sit inside
// EmbedService::chunk() and the --tokenize loop, and a branch at each
// would be the drift-prone shape encoder_implemented() exists to prevent.
struct AnyTokenizer {
  std::unique_ptr<npue::Tokenizer> wordpiece;
  std::unique_ptr<npue::XlmrTokenizer> xlmr;

  size_t vocab_size() const {
    return wordpiece ? wordpiece->vocab_size() : xlmr->vocab_size();
  }

  npue::Encoded encode(const std::string &text, int max_len) const {
    if (wordpiece) return wordpiece->encode(text, max_len);
    // XLM-R Unigram. XlmrTokenizer::encode() returns the FULL <s>...</s>
    // sequence, unpadded and untruncated (its header: an input that does
    // not fit is the caller's error to raise, not the tokenizer's to
    // hide). This adds the WordPiece path's exact max_len semantics on
    // top: truncation keeps <s> + the first (max_len - 2) pieces + </s>,
    // which is HuggingFace's longest_first truncation under the
    // "<s> A </s>" post-processor, so --tokenize stays diffable against
    // AutoTokenizer. n_tokens_full/truncated feed check_truncation()
    // unchanged -- 0110's refuse-on-overflow applies to arch=3 exactly as
    // to arch=0/2.
    std::vector<int32_t> full = xlmr->encode(text);
    npue::Encoded e;
    e.n_tokens_full = static_cast<int32_t>(full.size());
    e.truncated = e.n_tokens_full > max_len;
    if (e.truncated) {
      full.resize(static_cast<size_t>(max_len));
      full.back() = xlmr->eos_id;
    }
    e.n_tokens = static_cast<int32_t>(full.size());
    e.input_ids = std::move(full);
    e.input_ids.resize(static_cast<size_t>(max_len), xlmr->pad_id);
    e.attention_mask.assign(static_cast<size_t>(max_len), 0);
    for (int32_t s = 0; s < e.n_tokens; ++s) e.attention_mask[s] = 1;
    e.token_type_ids.assign(static_cast<size_t>(max_len), 0);
    return e;
  }
};

AnyTokenizer load_tokenizer(npue::File &model,
                            const std::string &model_path) {
  // arch=3: the XLMRTOK1 Unigram blob, stored whole in the container
  // (tasks/0135) and consumed in place. The ARCH decides, not the absence
  // of a tokenizer.vocab key -- absence already means something else below.
  std::string arch;
  try {
    arch = model.config_string("arch");
  } catch (const std::exception &) {
    // Pre-arch container: WordPiece, like everything else that old.
  }
  if (arch == "gte_new_rope_geglu") {
    auto v = model.raw("tokenizer.xlmr_table");
    AnyTokenizer t;
    t.xlmr = std::make_unique<npue::XlmrTokenizer>(
        npue::XlmrTokenizer::from_table_bytes(
            reinterpret_cast<const char *>(v.data), v.bytes));
    // The facade's padding and the sequence template lean on XLM-R's
    // specials -- check the blob rather than assume it.
    if (t.xlmr->bos_id != 0 || t.xlmr->eos_id != 2 || t.xlmr->pad_id != 1)
      throw std::runtime_error(
          "tokenizer.xlmr_table specials are not XLM-R's <s>=0, </s>=2, "
          "<pad>=1 -- refusing rather than padding with the wrong id");
    return t;
  }
  try {
    auto v = model.raw("tokenizer.vocab");
    AnyTokenizer t;
    t.wordpiece = std::make_unique<npue::Tokenizer>(
        npue::Tokenizer::from_vocab_bytes(
            reinterpret_cast<const char *>(v.data), v.bytes));
    return t;
  } catch (const std::exception &) {
    // Pre-0036 container: the loose checkpoint directory beside it, derived
    // from the container's own name rather than assumed to be MiniLM's.
    const std::string p =
        std::filesystem::path(model_path).replace_extension().string() +
        "/vocab.txt";
    std::printf("  tokenizer  .npue has no vocabulary; using %s\n", p.c_str());
    AnyTokenizer t;
    t.wordpiece = std::make_unique<npue::Tokenizer>(
        npue::Tokenizer::from_vocab_file(p));
    return t;
  }
}

// One entry of gemm_rtp's `streams` array: which instruction-stream slot
// runs which op at which batch tier. Parsed here rather than in npu_device
// because it is encoder policy, not device mechanics.
struct StreamEntry {
  std::string op, file;
  int64_t batch = 0, slot = 0, M = 0, K = 0, N = 0;
};

std::vector<StreamEntry> parse_streams(const std::string &json) {
  std::vector<StreamEntry> out;
  size_t i = json.find("\"streams\"");
  if (i == std::string::npos) return out;
  i = json.find('[', i);
  if (i == std::string::npos) return out;
  const size_t end = json.find(']', i);
  auto str_field = [&](size_t from, size_t to, const char *key) {
    const std::string k = std::string("\"") + key + "\"";
    size_t a = json.find(k, from);
    if (a == std::string::npos || a > to) return std::string();
    a = json.find('"', json.find(':', a) + 1) + 1;
    return json.substr(a, json.find('"', a) - a);
  };
  auto int_field = [&](size_t from, size_t to, const char *key) -> int64_t {
    const std::string k = std::string("\"") + key + "\"";
    size_t a = json.find(k, from);
    if (a == std::string::npos || a > to) return 0;
    return std::stoll(json.substr(json.find(':', a) + 1));
  };
  size_t p = i;
  while (true) {
    const size_t ob = json.find('{', p);
    if (ob == std::string::npos || ob > end) break;
    const size_t cb = json.find('}', ob);
    StreamEntry e;
    e.op = str_field(ob, cb, "op");
    e.file = str_field(ob, cb, "file");
    e.batch = int_field(ob, cb, "batch");
    e.slot = int_field(ob, cb, "slot");
    e.M = int_field(ob, cb, "M");
    e.K = int_field(ob, cb, "K");
    e.N = int_field(ob, cb, "N");
    if (!e.op.empty()) out.push_back(e);
    p = cb + 1;
  }
  return out;
}

// Pool [take, seq, hidden] hidden states into [take, hidden], then optionally
// L2 normalise. ONE implementation: there were three, and they disagreed --
// the golden path accumulated in float while the other two used double, so a
// comment claiming they matched was wrong by a rounding.
//
// `am` is the 1/0 attention mask, [rows, seq].
void pool_rows(const float *h, const float *am, int64_t take, float *out) {
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
void set_model_shape(npue::File &m) {
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
void set_design_seq(int64_t seq) {
  if (seq <= 0 || seq % 8)
    throw std::runtime_error("design seq " + std::to_string(seq) +
                             " must be positive and a multiple of 8");
  if (seq > g_max_positions)
    throw std::runtime_error(
        "design seq " + std::to_string(seq) + " exceeds the " +
        std::to_string(g_max_positions) + " position embeddings in the .npue");
  g_seq = seq;
}

// Resolves --prefix against the container's own task-prefix table
// (tasks/0071). Returns the literal text to prepend to every input text
// before tokenization -- "" for a container with no "prompts" table
// (g_prompts.empty()), which is BYTE-IDENTICAL to this runtime's behaviour
// before this task for MiniLM/bge-small/bge-base/bge-large: no prefix
// concept, no banner line, nothing prepended.
//
// A model that DOES have a prompts table always prints which prefix it is
// about to apply, on stderr. An unknown --prefix name lists the container's
// real options rather than guessing.
//
// THE CONTAINER DEFAULT NO LONGER APPLIES (tasks/0118). It used to: an omitted
// --prefix fell through to g_prompt_default, so nomic quietly embedded
// everything as `search_document`. That is a silently-applied default, which
// is how the wrong prefix ships (docs/04-model/README.md:24) -- and unlike a
// wrong number it cannot be seen downstream, because a wrongly-prefixed vector
// is correctly shaped, correctly normed and deterministic. `prompt_default`
// survives in the container as ADVISORY metadata, for harnesses choosing which
// prompt to exercise; nothing in this runtime applies it.
//
// `--prefix ""` is still legal and still means no prefix at all, matching
// check_gemma_prefix() and the endpoint's `"prompt_name": ""`. The distinction
// that now matters is OMITTED vs EMPTY, which is what `from_cli` carries.
std::string resolve_prefix(int argc, char **argv) {
  std::string name;
  bool from_cli = false;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--prefix") { name = argv[i + 1]; from_cli = true; }

  if (g_prompts.empty()) {
    // REFUSE rather than ignore. This model has no task-prefix concept, so a
    // --prefix the caller asked for cannot be applied -- and silently
    // proceeding would hand back vectors the caller believes are prefixed.
    // That is the same shape as the status lines this project has had to fix
    // repeatedly: reporting the intention rather than the value. A script
    // sweeping one --prefix across the whole catalogue SHOULD break here,
    // because its BERT results would otherwise differ from what it intended.
    if (from_cli)
      throw std::runtime_error(
          "--prefix '" + name + "' was given, but this model has no task "
          "prefixes -- its container carries no 'prompts' table, so nothing "
          "would be prepended. Refusing rather than returning vectors that "
          "are not what was asked for.");
    return std::string();
  }

  const std::string list = join_names(prompt_names_sorted());

  // REQUIRED, not defaulted. See the note above the function.
  if (!from_cli)
    throw std::runtime_error(
        "this model has task prefixes and one must be named: pass --prefix "
        "with one of [" + list + "], or --prefix \"\" for no prefix at "
        "all. Refusing to pick one for you -- a wrongly-prefixed embedding is "
        "correctly shaped and correctly normed, so nothing downstream can tell "
        "that the answer is wrong.");

  // Named explicitly and empty: no prefix at all. Same convention as
  // check_gemma_prefix() and the endpoint's `"prompt_name": ""`.
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

std::vector<float> read_f32(const std::string &path, size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  size_t bytes = static_cast<size_t>(f.tellg());
  if (bytes != count * sizeof(float))
    throw std::runtime_error(path + ": expected " +
                             std::to_string(count * sizeof(float)) +
                             " bytes, found " + std::to_string(bytes));
  f.seekg(0);
  std::vector<float> v(count);
  f.read(reinterpret_cast<char *>(v.data()), bytes);
  return v;
}

std::vector<int32_t> read_i32(const std::string &path, size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0);
  std::vector<int32_t> v(count);
  f.read(reinterpret_cast<char *>(v.data()), count * sizeof(int32_t));
  return v;
}

// fp32 -> bf16, round-to-nearest-even. The rounding tools/npue.py uses when
// packing; truncation would bias every value toward zero.
inline uint16_t to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
inline float from_bf16(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// The vectorised forms below are BIT-IDENTICAL to the scalar ones above, which
// is the only reason they are safe to swap in: every integer op used has the
// same semantics on uint32 as on __m256i lanes, and after the >> 16 the values
// are in [0, 65535] so packus never actually saturates. The scalar tail keeps
// the two paths agreeing on any n.
//
// 13.8 M elements per encode go through these (tasks/0024), which is why they
// are worth writing out.
#if defined(__AVX2__)
#include <immintrin.h>

void bf16_fill(void *dst, const float *src, size_t n) {
  auto *d = static_cast<uint16_t *>(dst);
  const __m256i k7fff = _mm256_set1_epi32(0x7FFF);
  const __m256i kone = _mm256_set1_epi32(1);
  auto rne = [&](__m256i u) {
    __m256i odd = _mm256_and_si256(_mm256_srli_epi32(u, 16), kone);
    return _mm256_srli_epi32(
        _mm256_add_epi32(u, _mm256_add_epi32(k7fff, odd)), 16);
  };
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256i a = rne(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(src + i)));
    __m256i b = rne(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(src + i + 8)));
    // packus interleaves the two 128-bit lanes; 0xD8 puts them back in order.
    __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + i), p);
  }
  for (; i < n; ++i) d[i] = to_bf16(src[i]);
}

void bf16_read(float *dst, const void *src, size_t n) {
  const auto *s = static_cast<const uint16_t *>(src);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s + i));
    __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(u));
  }
  for (; i < n; ++i) dst[i] = from_bf16(s[i]);
}
#else
void bf16_fill(void *dst, const float *src, size_t n) {
  auto *d = static_cast<uint16_t *>(dst);
  for (size_t i = 0; i < n; ++i) d[i] = to_bf16(src[i]);
}
void bf16_read(float *dst, const void *src, size_t n) {
  const auto *s = static_cast<const uint16_t *>(src);
  for (size_t i = 0; i < n; ++i) dst[i] = from_bf16(s[i]);
}
#endif

double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// int8 quantisation, shared by BOTH encoders.
//
// These were inline in Encoder::gemm() until tasks/0081 gave arch=1 an int8
// path too. GemmaNpuEncoder is a separate encoder (RMSNorm x4, MQA, per-layer
// RoPE, GeGLU), and copying ~120 lines of hand-vectorised quantisation into it
// would have created exactly the kind of duplicate that drifts: the reciprocal
// hoist below was a 63 ms fix found once (tasks/0080), and a second copy would
// not have it.
// ---------------------------------------------------------------------------

// A -> int8, per row, with the SmoothQuant divisor folded into the same pass
// (tasks/0078). `ias` is 1/asmooth, reciprocated ONCE by the caller: dividing
// by asmooth[j] in both the max pass and the quantise pass was two divisions
// per element and cost 74 ms of the 126 ms the array had saved.
//
// The divisor is NOT folded into the preceding norm. BERT is post-LN, so the
// norm's output feeds the residual as well as this GEMM (tasks/0078 4a); on
// Gemma the same holds for a different reason -- `pre_feedforward_layernorm`'s
// output is consumed by the GeGLU pair only, but `input_layernorm`'s feeds the
// residual, and one code path is worth more than one folded multiply.
template <typename ParRows>
void quantise_a_int8(const float *a, int64_t rows, int64_t K, const float *ias,
                     int8_t *q_base, float *a_scale, ParRows par_rows) {
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float *x = a + r * K;
      int8_t *q = q_base + r * K;
      float mx = 0.f;
      int64_t j = 0;
#if defined(__AVX2__)
      const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
      __m256 acc = _mm256_setzero_ps();
      for (; j + 8 <= K; j += 8)
        acc = _mm256_max_ps(acc, _mm256_and_ps(
            _mm256_mul_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(ias + j)),
            absmask));
      __m128 h = _mm_max_ps(_mm256_castps256_ps128(acc),
                            _mm256_extractf128_ps(acc, 1));
      h = _mm_max_ps(h, _mm_movehl_ps(h, h));
      h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
      mx = _mm_cvtss_f32(h);
#endif
      for (; j < K; ++j) {
        const float v = std::fabs(x[j] * ias[j]);
        if (v > mx) mx = v;
      }
      const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
      a_scale[static_cast<size_t>(r)] = sc;
      const float inv = 1.0f / sc;
      j = 0;
#if defined(__AVX2__)
      const __m256 invv = _mm256_set1_ps(inv);
      const __m256 hi = _mm256_set1_ps(127.0f);
      const __m256 lo = _mm256_set1_ps(-127.0f);
      for (; j + 8 <= K; j += 8) {
        __m256 v = _mm256_mul_ps(
            _mm256_mul_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(ias + j)),
            invv);
        v = _mm256_min_ps(_mm256_max_ps(v, lo), hi);
        // cvtps_epi32 rounds per MXCSR, i.e. nearest-even by default -- the
        // same rule tools/pack_npue.py's np.rint uses on the weights, so the
        // two halves of the product round the same way.
        __m256i i32 = _mm256_cvtps_epi32(v);
        __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                      _mm256_extracti128_si256(i32, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(q + j),
                         _mm_packs_epi16(p16, p16));
      }
#endif
      for (; j < K; ++j) {
        float v = std::nearbyintf(x[j] * ias[j] * inv);
        if (v > 127.f) v = 127.f;
        if (v < -127.f) v = -127.f;
        q[j] = static_cast<int8_t>(v);
      }
    }
  });
}

// GELU, the same degree-8 minimax polynomial gelu_cpu() uses, so the fused and
// unfused paths are bit-identical rather than merely close.
#if defined(__AVX2__)
inline __m256 gelu8(__m256 v) {
  const __m256 u = _mm256_min_ps(
      _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v), _mm256_set1_ps(4.0f));
  __m256 pl = _mm256_fmadd_ps(_mm256_set1_ps(-7.2340282171e-05f), u,
                              _mm256_set1_ps(1.8179518005e-03f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-1.7707383379e-02f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(8.4577147641e-02f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-1.9228671834e-01f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(9.8431124458e-02f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(3.6137852062e-01f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-4.9454128936e-01f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-1.3007010117e-04f));
  return _mm256_add_ps(_mm256_max_ps(v, _mm256_setzero_ps()), pl);
}
#endif
inline float gelu8(float v) {
  const float u = std::min(std::fabs(v), 4.0f);
  float pl = -7.2340282171e-05f;
  pl = pl * u + 1.8179518005e-03f;
  pl = pl * u + -1.7707383379e-02f;
  pl = pl * u + 8.4577147641e-02f;
  pl = pl * u + -1.9228671834e-01f;
  pl = pl * u + 9.8431124458e-02f;
  pl = pl * u + 3.6137852062e-01f;
  pl = pl * u + -4.9454128936e-01f;
  pl = pl * u + -1.3007010117e-04f;
  return std::max(v, 0.0f) + pl;
}

// THE ffn_up EPILOGUE IN ONE PASS (tasks/0081 T37).
//
// The unfused chain walks the widest tensor in the model six times: the
// dequantiser writes fp32 `up`, GELU reads and writes it, and the next GEMM's
// quantiser reads it again and writes int8. At bge-large's batch 128 that
// tensor is 134 MB, so the chain is ~636 MB per layer and ~15 GB per encode --
// and tasks/0081 section 3 measured the host at 69.6% of the encode, nearly
// all of it memory traffic rather than arithmetic.
//
// Fused it is ~100 MB per layer: read C, and write ffn_down's int8 operand.
// Nothing else is materialised. The row (16 KB at most) stays in L1 across the
// three sub-passes, so the absmax GELU's output needs costs a cache hit rather
// than a DRAM sweep -- which is the whole reason a per-row activation scale is
// affordable at all (tasks/0079 measured the static alternative at 57x worse).
//
// Bit-identical to the unfused path by construction: same polynomial, same
// rounding, same order.
// `act` transforms the dequantised row IN PLACE and leaves `out_n` values at
// row[0, out_n) -- N for a plain GELU, N/2 for a gated FFN that combines two
// halves into one. It is a parameter rather than a branch so the caller keeps
// ownership of the exact intrinsics, which is what makes the fused and
// unfused paths bit-identical rather than merely close.
template <typename ParRows, typename Act>
void dequant_act_quant(const void *c, size_t c_bytes, int64_t rows, int64_t N,
                       int64_t out_n, Act act,
                       const float *sa_up, const float *wscale,
                       const float *bias, const float *ias_next,
                       int8_t *dst, float *sa_next, ParRows par_rows) {
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    std::vector<float> row(static_cast<size_t>(N));
    for (int64_t r = r0; r < r1; ++r) {
      const float sa = sa_up[static_cast<size_t>(r)];
      float *v = row.data();
      int64_t j = 0;
      // 1. dequantise + bias + GELU, into L1. THE INTRINSICS MUST MATCH the
      // unfused path's exactly -- `_mm256_fmadd_ps` rounds once where
      // `a*b + c` rounds twice, so a scalar rewrite of the same formula is
      // NOT the same number. Measured: 1.161e-03 unfused against 1.180e-03
      // for a scalar fused pass. Both pass the gate, but a fused path that
      // silently changes the result is a fused path nobody can A/B.
#if defined(__AVX2__)
      const __m256 sav = _mm256_set1_ps(sa);
      if (c_bytes == 2) {
        const uint16_t *cr = static_cast<const uint16_t *>(c) + r * N;
        for (; j + 16 <= N; j += 16) {
          __m256i raw = _mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          __m256 lo = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16));
          __m256 hi = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16));
          _mm256_storeu_ps(v + j, _mm256_fmadd_ps(
              _mm256_mul_ps(lo, sav), _mm256_loadu_ps(wscale + j),
              _mm256_loadu_ps(bias + j)));
          _mm256_storeu_ps(v + j + 8, _mm256_fmadd_ps(
              _mm256_mul_ps(hi, sav), _mm256_loadu_ps(wscale + j + 8),
              _mm256_loadu_ps(bias + j + 8)));
        }
      } else {
        const int32_t *cr = static_cast<const int32_t *>(c) + r * N;
        for (; j + 8 <= N; j += 8) {
          __m256 cf = _mm256_cvtepi32_ps(_mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j)));
          _mm256_storeu_ps(v + j, _mm256_fmadd_ps(
              _mm256_mul_ps(cf, sav), _mm256_loadu_ps(wscale + j),
              _mm256_loadu_ps(bias + j)));
        }
      }
#endif
      for (; j < N; ++j) {
        const float cf = c_bytes == 2
            ? from_bf16(static_cast<const uint16_t *>(c)[r * N + j])
            : static_cast<float>(static_cast<const int32_t *>(c)[r * N + j]);
        v[j] = cf * sa * wscale[j] + bias[j];
      }
      // 2. the activation, in place, narrowing to out_n.
      act(v, N);
      // 3. row absmax of the smoothed value, and 4. quantise. Both over a row
      // that is now hot in L1, which is what makes this worth doing at all.
      float mx = 0.f;
      j = 0;
#if defined(__AVX2__)
      {
        const __m256 absmask =
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 acc = _mm256_setzero_ps();
        for (; j + 8 <= out_n; j += 8)
          acc = _mm256_max_ps(acc, _mm256_and_ps(
              _mm256_mul_ps(_mm256_loadu_ps(v + j),
                            _mm256_loadu_ps(ias_next + j)), absmask));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(acc),
                              _mm256_extractf128_ps(acc, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
        mx = _mm_cvtss_f32(h);
      }
#endif
      for (; j < out_n; ++j) {
        const float a = std::fabs(v[j] * ias_next[j]);
        if (a > mx) mx = a;
      }
      const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
      sa_next[static_cast<size_t>(r)] = sc;
      const float inv = 1.0f / sc;
      int8_t *q = dst + r * out_n;
      j = 0;
#if defined(__AVX2__)
      {
        const __m256 invv = _mm256_set1_ps(inv);
        const __m256 vhi = _mm256_set1_ps(127.0f);
        const __m256 vlo = _mm256_set1_ps(-127.0f);
        for (; j + 8 <= out_n; j += 8) {
          __m256 t = _mm256_mul_ps(
              _mm256_mul_ps(_mm256_loadu_ps(v + j),
                            _mm256_loadu_ps(ias_next + j)), invv);
          t = _mm256_min_ps(_mm256_max_ps(t, vlo), vhi);
          __m256i i32 = _mm256_cvtps_epi32(t);
          __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                        _mm256_extracti128_si256(i32, 1));
          _mm_storel_epi64(reinterpret_cast<__m128i *>(q + j),
                           _mm_packs_epi16(p16, p16));
        }
      }
#endif
      for (; j < out_n; ++j) {
        float t = std::nearbyintf(v[j] * ias_next[j] * inv);
        if (t > 127.f) t = 127.f;
        if (t < -127.f) t = -127.f;
        q[j] = static_cast<int8_t>(t);
      }
    }
  });
}

// C -> fp32: y = acc * sa[row] * wscale[col] + bias[col]. A rank-1
// outer-product scaling folded into the pass that already reads C and adds the
// bias. `c_bytes` selects the transport width the design chose: 4 = int32
// accumulator straight out, 2 = narrowed to bf16 on the core (tasks/0080).
template <typename ParRows>
void dequantise_c(const void *c, size_t c_bytes, int64_t rows, int64_t N,
                  const float *a_scale, const float *wscale, const float *bias,
                  float *out, ParRows par_rows, bool sim_bf16 = false) {
  if (c_bytes == 2) {
    const uint16_t *cb = static_cast<const uint16_t *>(c);
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const uint16_t *cr = cb + r * N;
        float *o = out + r * N;
        const float sa = a_scale[static_cast<size_t>(r)];
        int64_t j = 0;
#if defined(__AVX2__)
        const __m256 sav = _mm256_set1_ps(sa);
        for (; j + 16 <= N; j += 16) {
          // One 32-byte streaming load carries 16 bf16 against 8 int32 -- the
          // same instruction count for twice the elements, which is the whole
          // point of narrowing C.
          __m256i raw = _mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          __m256 lo = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16));
          __m256 hi = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16));
          _mm256_storeu_ps(o + j,
              _mm256_fmadd_ps(_mm256_mul_ps(lo, sav),
                              _mm256_loadu_ps(wscale + j),
                              _mm256_loadu_ps(bias + j)));
          _mm256_storeu_ps(o + j + 8,
              _mm256_fmadd_ps(_mm256_mul_ps(hi, sav),
                              _mm256_loadu_ps(wscale + j + 8),
                              _mm256_loadu_ps(bias + j + 8)));
        }
#endif
        for (; j < N; ++j)
          o[j] = from_bf16(cr[j]) * sa * wscale[j] + bias[j];
      }
    });
    return;
  }
  const int32_t *ci = static_cast<const int32_t *>(c);
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int32_t *cr = ci + r * N;
      float *o = out + r * N;
      const float sa = a_scale[static_cast<size_t>(r)];
      int64_t j = 0;
#if defined(__AVX2__)
      // Streaming loads: C is a write-combined XRT host bo and ordinary loads
      // from it stall per line (tasks/0024).
      const __m256 sav = _mm256_set1_ps(sa);
      for (; j + 8 <= N; j += 8) {
        __m256 cf = _mm256_cvtepi32_ps(_mm256_stream_load_si256(
            reinterpret_cast<const __m256i *>(cr + j)));
        if (sim_bf16) {
          // --sim-c-bf16: round exactly as a narrowed-C design would, to price
          // one before building it (tasks/0080). RNE to bf16 = add half an ulp
          // plus the tie-break bit, then truncate.
          __m256i u = _mm256_castps_si256(cf);
          u = _mm256_add_epi32(
              u, _mm256_add_epi32(
                     _mm256_set1_epi32(0x7FFF),
                     _mm256_and_si256(_mm256_srli_epi32(u, 16),
                                      _mm256_set1_epi32(1))));
          cf = _mm256_castsi256_ps(
              _mm256_and_si256(u, _mm256_set1_epi32(int(0xFFFF0000u))));
        }
        _mm256_storeu_ps(o + j,
            _mm256_fmadd_ps(_mm256_mul_ps(cf, sav),
                            _mm256_loadu_ps(wscale + j),
                            _mm256_loadu_ps(bias + j)));
      }
#endif
      for (; j < N; ++j) {
        float cf = static_cast<float>(cr[j]);
        if (sim_bf16) {
          uint32_t u;
          std::memcpy(&u, &cf, 4);
          u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
          std::memcpy(&cf, &u, 4);
        }
        o[j] = cf * sa * wscale[j] + bias[j];
      }
    }
  });
}

// A persistent pool, because attention is called 12 times per encode and
// spawning threads each time would cost more than it saves.
//
// The calling thread takes chunk 0 and participates, so `n` threads means
// n-1 spawned. Work is partitioned by (batch, head) pair, and every pair writes
// a disjoint slice of `scores` and `ctx`, so there is no sharing to guard.
class Pool {
public:
  explicit Pool(int n) : n_(n < 1 ? 1 : n) {
    for (int i = 1; i < n_; ++i)
      workers_.emplace_back([this, i] {
        int seen = 0;
        for (;;) {
          std::function<void(int, int)> f;
          {
            std::unique_lock<std::mutex> lk(m_);
            cv_work_.wait(lk, [&] { return quit_ || gen_ != seen; });
            if (quit_) return;
            seen = gen_;
            f = fn_;
          }
          f(i, n_);
          {
            std::lock_guard<std::mutex> lk(m_);
            if (--remaining_ == 0) cv_done_.notify_one();
          }
        }
      });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      quit_ = true;
    }
    cv_work_.notify_all();
    for (auto &t : workers_) t.join();
  }
  Pool(const Pool &) = delete;
  Pool &operator=(const Pool &) = delete;

  int size() const { return n_; }

  void run(const std::function<void(int, int)> &f) {
    if (n_ == 1) { f(0, 1); return; }
    {
      std::lock_guard<std::mutex> lk(m_);
      fn_ = f;
      remaining_ = n_ - 1;
      ++gen_;
    }
    cv_work_.notify_all();
    f(0, n_);
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] { return remaining_ == 0; });
  }

private:
  int n_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_work_, cv_done_;
  std::function<void(int, int)> fn_;
  int gen_ = 0, remaining_ = 0;
  bool quit_ = false;
};

#if defined(__AVX2__)
inline float hsum256(__m256 v) {
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v),
                         _mm256_extractf128_ps(v, 1));
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
#endif

double cpu_seconds() {
  FILETIME c, e, k, u;
  GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
  auto to_s = [](FILETIME f) {
    return ((static_cast<uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime) *
           1e-7;
  };
  return to_s(k) + to_s(u);
}

// The whole encoder. Designs are constructed once by the caller and reused --
// that is the point of the exercise.
struct Encoder {
  npue::File &model;
  npu::Design &qkv, &attn_out, &ffn_up, &ffn_down, &gelu, &layernorm, &softmax;


  // Staged once: the mask, in the form softmax consumes.
  std::vector<float> add_mask;   // [batch, g_seq]

  // Unified gemm_rtp mode (tasks/0032): all four GEMM refs above point at ONE
  // design; each op is an instruction-stream slot bound before dispatch. The
  // slot order is the export contract of tools/export_gemm_rtp.py:
  // qkv=0 (insts.bin), attn_out=1, ffn_up=2, ffn_down=3 (load_instr order).
  bool unified = false;
  size_t is_qkv = 0, is_ao = 0, is_fu = 0, is_fd = 0;

  // Batch tiers (0037): one xclbin carries a stream per (op, batch), so the
  // encoder can size a request instead of padding it to the largest design.
  // `tiers` is ascending; `use_tier` picks the smallest that fits and points
  // is_* at its slots.
  std::vector<int64_t> tiers;
  std::vector<std::array<size_t, 4>> tier_slots;   // qkv, attn_out, ffn_up, ffn_down

  int64_t use_tier(int64_t want) {
    if (tiers.empty()) return batch;
    size_t pick = tiers.size() - 1;
    for (size_t i = 0; i < tiers.size(); ++i)
      if (tiers[i] >= want) { pick = i; break; }
    batch = tiers[pick];
    rows = batch * g_seq;
    is_qkv = tier_slots[pick][0];
    is_ao = tier_slots[pick][1];
    is_fu = tier_slots[pick][2];
    is_fd = tier_slots[pick][3];
    return batch;
  }

  // Two-encode pipelining (tasks/0033): two Encoder instances share the ONE
  // unified design; each owns its A and C slots, and every NPU interaction
  // (bind + sync + dispatch) happens under this mutex. The array serializes
  // dispatches anyway (note 0004) -- the lock only makes explicit what the
  // hardware enforces -- while each pipeline's HOST work overlaps the other
  // pipeline's NPU work.
  std::mutex *npu_mu = nullptr;
  size_t slot_a = 0, slot_c = 0;

  int64_t batch = 0, rows = 0;   // rows = batch * g_seq, from the design's M
  Pool *pool = nullptr;

  // Chunk a flat range over the pool. Chunks are 64-element aligned so the AVX2
  // conversions never see a split vector and every worker takes the fast path.
  template <typename F> void par(size_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1 || n < 65536) { f(size_t(0), n); return; }
    pool->run([&](int w, int nw) {
      const size_t chunk = ((n / nw) + 63) & ~size_t(63);
      const size_t lo = std::min(n, chunk * size_t(w));
      const size_t hi = std::min(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }

  // Row-parallel, for passes where a flat byte range would split a row --
  // int8 quantisation takes a per-row maximum, so a chunk boundary inside a
  // row would give two different scales to one row's halves (tasks/0078).
  template <typename F> void par_rows(int64_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1) { f(int64_t(0), n); return; }
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n, chunk * w);
      const int64_t hi = std::min<int64_t>(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }

  // Scratch reused across layers. `residual = x` used to allocate and copy
  // 12.6 MB per layer at batch 128 -- 75 MB per encode of pure copying.
  std::vector<float> residual;
  // Scratch, sized once. These used to be six fresh vectors per run() -- ~90
  // MB of allocate-and-touch per encode at batch 128, and ~280 MB at
  // bge-large's width. resize() after the first call is a no-op.
  std::vector<float> qkvbuf, ctx, proj, up, down, scores;
  // arch=2 only: swiglu_cpu()'s output, [rows][g_ffn] -- a separate buffer
  // from `up` because an in-place version of this compaction is NOT safe
  // under threading (see swiglu_cpu's own comment). Empty/unused for every
  // arch=0 container.
  std::vector<float> gated;

  // arch=2 RoPE tables, [g_seq, g_head_dim] each, built ONCE per Encoder (not
  // per layer, not per call) the first time apply_rope_qkv() runs. Every
  // layer shares the same table -- unlike Gemma, nomic has a single rope_theta
  // for the whole model, not a per-layer local/global split (gemma_kernels.hpp
  // trap 2b). Using npue::gemma_rope_tables() here even though nothing about
  // it is Gemma-specific: it is plain NeoX RoPE table construction, and
  // duplicating it for a second arch would be the actual mistake.
  std::vector<float> rope_cos, rope_sin;
  bool rope_ready = false;

  // Device-resident weights, one slot per layer per design, plus the bias
  // pointers straight into the mapped file. Filled by stage_all().
  std::vector<size_t> s_qkv, s_ao, s_fu, s_fd;
  std::vector<const float *> b_qkv, b_ao, b_fu, b_fd;
  // int8 only (tasks/0078): per-output-channel weight scales and the
  // per-input-channel SmoothQuant divisor, straight out of the mapped .npue.
  // Empty for every bf16 container, and the int8 path is selected by the
  // DESIGN's a_dtype, so a mismatch is caught by stage_all() rather than by
  // dereferencing an empty vector.
  std::vector<const float *> ws_qkv, ws_ao, ws_fu, ws_fd;
  std::vector<const float *> as_qkv, as_ao, as_fu, as_fd;
  std::vector<size_t> s_ln;      // 0 = embeddings, then ln1/ln2 per layer
  // Host-side views of the same parameters, straight into the mapped .npue.
  // tasks/0031 measured a LayerNorm call at 725 us of kernel inside ~3 ms of
  // switch+conversion; a threaded fp32 AVX2 LayerNorm on the host costs
  // ~0.5 ms and removes 13 design switches outright. --host-ln selects it.
  std::vector<const float *> h_gamma, h_beta;
  bool host_ln = false;
  bool host_sm = false;
  bool host_gelu = false;
  // Measure-before-build (tasks/0080). Under int8 the GEMM is bandwidth-bound
  // again (0010's model, which 0048 superseded for bf16), and C is 61% of the
  // traffic on three of four shapes -- so narrowing C is the top lever. Doing
  // it needs NO extra core input, because int32 -> bf16 is a pure format
  // conversion: the host still applies sa[i]*wscale[j]. What it costs is
  // accuracy, and that is knowable without writing a kernel. This flag rounds
  // the accumulator exactly as such a design would (int32 -> fp32 -> bf16,
  // round-to-nearest-even, i.e. conv_even per trap 2b) and changes nothing
  // else, so the 1-cos gate prices the design before it exists.
  bool sim_c_bf16 = false;
  // T37 (tasks/0081): fuse the ffn_up epilogue -- dequantise, GELU and the
  // next GEMM's quantisation in ONE pass, so the widest tensor in the model is
  // never materialised in fp32. On by default where it applies; --no-fuse-ffn
  // turns it off, which is how the two paths are compared.
  // T37-BF16 (tasks/0108): the same flag also gates the bf16/bfp16 analogue
  // (narrow-to-bf16 instead of quantise, no scale) -- one flag, two datapaths,
  // so --no-fuse-ffn A/Bs whichever one the loaded container actually uses.
  bool fuse_ffn_epilogue = true;
  double t_hostln = 0.0, t_hostsm = 0.0, t_hostgelu = 0.0;


  // Where the time goes. A single number for the whole encode says "slow";
  // this says which half to fix.
  double t_npu = 0.0;      // memcpy + sync + dispatch, i.e. everything a
                           // fused design would subsume
  double t_attn = 0.0;     // the per-head QK^T and A.V loops on the host
  // Split, because they are different problems: QK^T ends in a horizontal
  // reduction and holds Q in registers (compute), A.V accumulates per output
  // element and streams scores + V (memory). tasks/0086 widened A.V to 512
  // bits for exactly zero, which only makes sense if it is the memory half --
  // and that is a claim this counter can check instead of infer.
  double t_qk = 0.0, t_av = 0.0;
  int n_dispatch = 0;

  // Splitting t_npu further, because removing 21 MB of memcpy and vectorising
  // 13.8 M conversions bought only 9%: the cost is not where it was assumed to
  // be, and one aggregate number cannot say where it is instead (tasks/0024).
  double t_conv = 0.0;     // fp32 <-> bf16 both directions
  double t_in = 0.0;       // sync_to_device
  double t_disp = 0.0;     // kernel(...) + wait
  double t_out = 0.0;      // sync_from_device
  double t_bias = 0.0;     // reading the result buffer, adding bias

  void reset_timers() {
    // t_qk/t_av belong here too. Left out, they accumulated the warm-up encode
    // as well as the benched ones and summed to 1.5x their own parent t_attn --
    // a new counter that is not reset is a counter measuring a different window
    // from everything printed beside it (tasks/0086).
    t_qk = t_av = 0.0;
    t_npu = t_attn = 0.0;
    t_hostln = t_hostsm = t_hostgelu = 0.0;
    t_conv = t_in = t_disp = t_out = t_bias = 0.0;
    n_dispatch = 0;
  }

  // Move every weight onto the device once. Returns the bytes staged, so the
  // banner can state the cost of the trade rather than hiding it.
  size_t stage_all() {
    size_t bytes = 0;
    const bool i8 = qkv.info().a_elem_bytes == 1;
    auto one = [&](npu::Design &d, const std::string &name,
                   std::vector<size_t> &slots,
                   std::vector<const float *> &bias,
                   std::vector<const float *> *wsc = nullptr,
                   std::vector<const float *> *asm_ = nullptr) {
      // The design says what layout it needs; the .npue says what it holds.
      // Refuse unless both spoke AND they agree.
      //
      // tasks/0022 shipped pre-tiled weights into a row-major design: right
      // sizes, wrong order, rel_fro 1.186 -- "a buffer-size check catches a
      // wrong size, never a wrong layout". The layout hash that catches it has
      // been in the file since M4 and was never read on this side.
      const std::string &want = d.info().b_layout_hash;
      const std::string &got = model.info(name).layout_hash;
      if (want.empty())
        throw std::runtime_error(d.info().name + "/design.json has no "
                                 "b_layout_hash -- re-export with "
                                 "tools/export_xclbin.py");
      if (got.empty())
        throw std::runtime_error(name + ": .npue tensor carries no "
                                 "layout_hash -- repack with "
                                 "tools/pack_npue.py");
      if (want != got)
        throw std::runtime_error(
            name + ": layout mismatch -- design " + d.info().name +
            " wants " + want.substr(0, 16) + "..., file has " +
            got.substr(0, 16) + "... The bytes would be the right size and the "
            "wrong order.");
      auto w = model.raw(name);
      slots.push_back(d.stage(1, w.data, w.bytes));
      bias.push_back(model.raw(name + ".bias").as<float>());
      bytes += w.bytes;
      if (i8 && wsc) {
        // Refuse rather than dereference: an int8 DESIGN with a bf16
        // CONTAINER would otherwise fail here with a bare "no such tensor",
        // and the layout-hash check above has already told the user which
        // half is wrong -- so this only fires if a container carries tiled
        // int8 bytes without the scales that give them meaning.
        wsc->push_back(model.raw(name + ".wscale").as<float>());
        asm_->push_back(model.raw(name + ".asmooth").as<float>());
      }
    };
    for (int64_t L = 0; L < g_layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      one(qkv, p + "qkv", s_qkv, b_qkv, &ws_qkv, &as_qkv);
      one(attn_out, p + "attn_out", s_ao, b_ao, &ws_ao, &as_ao);
      one(ffn_up, p + "ffn_up", s_fu, b_fu, &ws_fu, &as_fu);
      one(ffn_down, p + "ffn_down", s_fd, b_fd, &ws_fd, &as_fd);
    }

    // gamma and beta share one buffer: a core tile has two input DMA channels
    // and the activations need one of them (tasks/0020).
    std::vector<float> gb(2 * g_hidden);
    auto ln_one = [&](const std::string &g, const std::string &b) {
      std::memcpy(gb.data(), model.raw(g).data, g_hidden * sizeof(float));
      std::memcpy(gb.data() + g_hidden, model.raw(b).data,
                  g_hidden * sizeof(float));
      s_ln.push_back(layernorm.stage(1, gb.data(), gb.size() * sizeof(float)));
      bytes += gb.size() * sizeof(float);
      h_gamma.push_back(model.raw(g).as<float>());
      h_beta.push_back(model.raw(b).as<float>());
    };
    if (host_ln) {
      // No device staging: only the host pointers and the site numbering.
      auto ln_host = [&](const std::string &g, const std::string &b) {
        s_ln.push_back(s_ln.size() + 1);
        h_gamma.push_back(model.raw(g).as<float>());
        h_beta.push_back(model.raw(b).as<float>());
      };
      ln_host("embeddings.ln.weight", "embeddings.ln.bias");
      for (int64_t L = 0; L < g_layers; ++L) {
        const std::string p = "layer." + std::to_string(L) + ".";
        ln_host(p + "ln1.weight", p + "ln1.bias");
        ln_host(p + "ln2.weight", p + "ln2.bias");
      }
      return bytes;
    }
    ln_one("embeddings.ln.weight", "embeddings.ln.bias");
    for (int64_t L = 0; L < g_layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      ln_one(p + "ln1.weight", p + "ln1.bias");
      ln_one(p + "ln2.weight", p + "ln2.bias");
    }
    return bytes;
  }

  // `lap` charges the elapsed time to a bucket and returns the new mark, so
  // each stage is attributed without a timer call being able to drift.
  double lap(double t0, double &bucket) {
    double t = now_s();
    bucket += t - t0;
    return t;
  }

  // bf16 in, bf16 out, one input buffer -- GELU and softmax.
  void eltwise(npu::Design &d, float *x, size_t n) {
    double t0 = now_s();
    par(n, [&](size_t lo, size_t hi) {
      bf16_fill(static_cast<uint16_t *>(d.host_ptr(0)) + lo, x + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    d.sync_to_device(0);
    t0 = lap(t0, t_in);
    d.dispatch_only();
    t0 = lap(t0, t_disp);
    d.sync_from_device(1);
    t0 = lap(t0, t_out);
    par(n, [&](size_t lo, size_t hi) {
      bf16_read(x + lo, static_cast<const uint16_t *>(d.host_ptr(1)) + lo,
                hi - lo);
    });
    lap(t0, t_conv);
    ++n_dispatch;
  }

  // fp32 two-pass LayerNorm on the host, parallelized over rows: the same
  // two-pass mean/variance formula as the NPU kernel and the M3 oracle, in
  // fp32 throughout -- if anything MORE accurate than the bf16 round trip it
  // replaces. The golden check decides.
  void layer_norm_cpu(std::vector<float> &x, size_t site) {
    double t0 = now_s();
    const float *g = h_gamma[site], *b = h_beta[site];
    const int64_t n_rows = static_cast<int64_t>(x.size()) / g_hidden;
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n_rows + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n_rows, chunk * w);
      const int64_t hi = std::min<int64_t>(n_rows, lo + chunk);
      for (int64_t r = lo; r < hi; ++r) {
        float *row = x.data() + r * g_hidden;
#if defined(__AVX2__)
        __m256 s = _mm256_setzero_ps();
        for (int64_t j = 0; j < g_hidden; j += 8)
          s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
        const float mean = hsum256(s) / g_hidden;
        const __m256 mv = _mm256_set1_ps(mean);
        __m256 v = _mm256_setzero_ps();
        for (int64_t j = 0; j < g_hidden; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          v = _mm256_fmadd_ps(d, d, v);
        }
        const float var = hsum256(v) / g_hidden;
        const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
        for (int64_t j = 0; j < g_hidden; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          __m256 y = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                     _mm256_loadu_ps(g + j),
                                     _mm256_loadu_ps(b + j));
          _mm256_storeu_ps(row + j, y);
        }
#else
        double sm = 0.0;
        for (int64_t j = 0; j < g_hidden; ++j) sm += row[j];
        const float mean = static_cast<float>(sm / g_hidden);
        double sv = 0.0;
        for (int64_t j = 0; j < g_hidden; ++j) {
          const float d = row[j] - mean;
          sv += static_cast<double>(d) * d;
        }
        const float var = static_cast<float>(sv / g_hidden);
        const float is = 1.0f / std::sqrt(var + 1e-12f);
        for (int64_t j = 0; j < g_hidden; ++j)
          row[j] = (row[j] - mean) * is * g[j] + b[j];
#endif
      }
    });
    t_hostln += now_s() - t0;
  }

#if defined(__AVX2__)
  static inline __m256 exp2_avx2(__m256 x) {
    const __m256 c0 = _mm256_set1_ps(1.5483275463e-05f);
    const __m256 c1 = _mm256_set1_ps(1.5669833174e-04f);
    const __m256 c2 = _mm256_set1_ps(1.3331825236e-03f);
    const __m256 c3 = _mm256_set1_ps(9.6164605538e-03f);
    const __m256 c4 = _mm256_set1_ps(5.5504156855e-02f);
    const __m256 c5 = _mm256_set1_ps(2.4022684109e-01f);
    const __m256 c6 = _mm256_set1_ps(6.9314717694e-01f);
    const __m256 c7 = _mm256_set1_ps(9.9999998955e-01f);
    __m256i k = _mm256_cvttps_epi32(x);
    __m256 f = _mm256_sub_ps(x, _mm256_cvtepi32_ps(k));
    __m256 pl = _mm256_fmadd_ps(c0, f, c1);
    pl = _mm256_fmadd_ps(pl, f, c2);
    pl = _mm256_fmadd_ps(pl, f, c3);
    pl = _mm256_fmadd_ps(pl, f, c4);
    pl = _mm256_fmadd_ps(pl, f, c5);
    pl = _mm256_fmadd_ps(pl, f, c6);
    pl = _mm256_fmadd_ps(pl, f, c7);
    __m256i bits = _mm256_slli_epi32(
        _mm256_add_epi32(k, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(pl, _mm256_castsi256_ps(bits));
  }
#endif

  // fp32 softmax over rows of g_seq on the host. Same structure as the NPU
  // kernel (max-subtract, exp2 with the -120 argument floor, one reciprocal),
  // but fp32 end to end -- like host LayerNorm, it removes dispatches AND
  // beats the bf16 path on accuracy.
  // The padding mask, as an explicit pass. Only the NPU-softmax branch needs
  // this: softmax_cpu folds the same addition into its per-row prologue, where
  // the row is already in L1 and it costs nothing, while an aie softmax kernel
  // has no second operand to take it from.
  void add_additive_mask(std::vector<float> &scores) {
    const int64_t rows_per_seq = g_heads * g_seq;
    const int64_t n_rows = static_cast<int64_t>(scores.size()) / g_seq;
    pool->run([&](int w, int nw) {
      for (int64_t r = w; r < n_rows; r += nw) {
        float *row = scores.data() + r * g_seq;
        const float *mk = add_mask.data() + (r / rows_per_seq) * g_seq;
        for (int64_t j = 0; j < g_seq; ++j) row[j] += mk[j];
      }
    });
  }

  void softmax_cpu(std::vector<float> &scores) {
    double t0 = now_s();
    const int64_t n_rows = static_cast<int64_t>(scores.size()) / g_seq;
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n_rows + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n_rows, chunk * w);
      const int64_t hi = std::min<int64_t>(n_rows, lo + chunk);
      const int64_t rows_per_seq = g_heads * g_seq;
      for (int64_t r = lo; r < hi; ++r) {
        float *row = scores.data() + r * g_seq;
        // The additive padding mask, folded in here rather than in qk(): the
        // row is already resident, so this is free, and it leaves qk() as the
        // pure matmul an array kernel could run. Same single float addition
        // qk() used to do, so the result is unchanged to the bit.
        const float *mk = add_mask.data() + (r / rows_per_seq) * g_seq;
        for (int64_t j = 0; j < g_seq; ++j) row[j] += mk[j];
#if defined(__AVX2__)
        __m256 mx = _mm256_loadu_ps(row);
        for (int64_t j = 8; j < g_seq; j += 8)
          mx = _mm256_max_ps(mx, _mm256_loadu_ps(row + j));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx),
                               _mm256_extractf128_ps(mx, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const __m256 mv = _mm256_set1_ps(_mm_cvtss_f32(m4));
        const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
        const __m256 argfloor = _mm256_set1_ps(-120.0f);
        __m256 sum = _mm256_setzero_ps();
        for (int64_t j = 0; j < g_seq; j += 8) {
          __m256 a = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j), mv),
                                   log2e);
          __m256 e = exp2_avx2(_mm256_max_ps(a, argfloor));
          _mm256_storeu_ps(row + j, e);
          sum = _mm256_add_ps(sum, e);
        }
        const __m256 inv = _mm256_set1_ps(1.0f / hsum256(sum));
        for (int64_t j = 0; j < g_seq; j += 8)
          _mm256_storeu_ps(row + j,
                           _mm256_mul_ps(_mm256_loadu_ps(row + j), inv));
#else
        float m = row[0];
        for (int64_t j = 1; j < g_seq; ++j) m = std::max(m, row[j]);
        float sum = 0.f;
        for (int64_t j = 0; j < g_seq; ++j) {
          row[j] = std::exp(row[j] - m);
          sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (int64_t j = 0; j < g_seq; ++j) row[j] *= inv;
#endif
      }
    });
    t_hostsm += now_s() - t0;
  }

  void gelu_cpu(std::vector<float> &x) {
    double t0 = now_s();
    par(x.size(), [&](size_t lo, size_t hi) {
      size_t i = lo;
#if defined(__AVX2__)
      const __m256 vR = _mm256_set1_ps(4.0f);
      const __m256 vz = _mm256_setzero_ps();
      const __m256 sign = _mm256_set1_ps(-0.0f);
      const __m256 c0 = _mm256_set1_ps(-7.2340282171e-05f);
      const __m256 c1 = _mm256_set1_ps(1.8179518005e-03f);
      const __m256 c2 = _mm256_set1_ps(-1.7707383379e-02f);
      const __m256 c3 = _mm256_set1_ps(8.4577147641e-02f);
      const __m256 c4 = _mm256_set1_ps(-1.9228671834e-01f);
      const __m256 c5 = _mm256_set1_ps(9.8431124458e-02f);
      const __m256 c6 = _mm256_set1_ps(3.6137852062e-01f);
      const __m256 c7 = _mm256_set1_ps(-4.9454128936e-01f);
      const __m256 c8 = _mm256_set1_ps(-1.3007010117e-04f);
      for (; i + 8 <= hi; i += 8) {
        __m256 v = _mm256_loadu_ps(x.data() + i);
        __m256 u = _mm256_min_ps(_mm256_andnot_ps(sign, v), vR);
        __m256 pl = _mm256_fmadd_ps(c0, u, c1);
        pl = _mm256_fmadd_ps(pl, u, c2);
        pl = _mm256_fmadd_ps(pl, u, c3);
        pl = _mm256_fmadd_ps(pl, u, c4);
        pl = _mm256_fmadd_ps(pl, u, c5);
        pl = _mm256_fmadd_ps(pl, u, c6);
        pl = _mm256_fmadd_ps(pl, u, c7);
        pl = _mm256_fmadd_ps(pl, u, c8);
        _mm256_storeu_ps(x.data() + i,
                         _mm256_add_ps(_mm256_max_ps(v, vz), pl));
      }
#endif
      for (; i < hi; ++i) {
        const float v = x[i];
        const float u = std::min(std::fabs(v), 4.0f);
        float pl = -7.2340282171e-05f;
        pl = pl * u + 1.8179518005e-03f;
        pl = pl * u + -1.7707383379e-02f;
        pl = pl * u + 8.4577147641e-02f;
        pl = pl * u + -1.9228671834e-01f;
        pl = pl * u + 9.8431124458e-02f;
        pl = pl * u + 3.6137852062e-01f;
        pl = pl * u + -4.9454128936e-01f;
        pl = pl * u + -1.3007010117e-04f;
        x[i] = std::max(v, 0.0f) + pl;
      }
    });
    t_hostgelu += now_s() - t0;
  }

  // arch=2 gated FFN activation: [rows][2*inter] -> [rows][inter].
  //   out[r][j] = lo[r][j] * silu(hi[r][j])
  // lo = cols [0, inter) (fc11, the untouched up-path), hi = cols
  // [inter, 2*inter) (fc12, the SiLU gate) -- pinned by the container's
  // swiglu_halves == "fc11_up|fc12_gate", asserted once in set_model_shape()
  // rather than trusted here.
  //
  // silu(x) = x / (1 + exp(-x)); exp(-x) = exp2(-x*log2e), reusing
  // exp2_avx2 instead of adding a second exponential. The argument floor
  // mirrors softmax_cpu's -120: when x (the gate, hi[j]) is large positive,
  // -x*log2e is large negative, and unfloored that corrupts exp2_avx2's
  // internal int32 conversion (cvttps2dq's "indefinite integer" case)
  // instead of cleanly underflowing toward 0 -- the same failure mode
  // softmax's own masked (very negative) rows hit without that floor.
  //
  // OUT OF PLACE, into a caller-supplied buffer -- NOT the in-place scheme
  // this function originally shipped with. The "safe in-place, compacting
  // forward" proof this comment used to carry was WRONG, and it was wrong in
  // exactly the way the task that wrote it demanded be checked for
  // ("VERIFY this claim numerically... rather than trusting the algebra") --
  // caught by that numerical check, on real hardware, tasks/0070:
  //   Row r WRITES [r*inter, (r+1)*inter) and READS [r*2*inter,(r+1)*2*inter).
  //   The original proof showed no row r' > r can have its READ range
  //   clobbered by row r's WRITE -- true, but it never checked r' < r. Row
  //   r's write range and row r' = floor(r/2)'s READ range overlap for
  //   EVERY r >= 1 (write=[r*inter,(r+1)*inter), read=[2r'*inter,(2r'+2)*inter),
  //   and r=2r' or r=2r'+1 both fall inside that read range by construction).
  //   Sequentially this is harmless (r' < r is always processed first in an
  //   ascending loop, so its read completes before r's write). Threaded, it
  //   is not: pool->run() hands CONTIGUOUS chunks to independent threads with
  //   no ordering between them, so whenever r and floor(r/2) land in
  //   different chunks (e.g. r'=63 in one thread's chunk, r=127 in another's,
  //   with no happens-before edge), row 127's write can race row 63's read.
  //   Measured effect: rows immediately after such a boundary came back
  //   catastrophically wrong (rel err up to 1.23, i.e. wrong sign / wrong
  //   magnitude, not bf16 noise) while every other row matched the oracle to
  //   ~1e-3 -- found via reference/encoder_nomic.py's own L0.fc11/fc12/gated
  //   taps, which isolated the corruption to exactly this function on a
  //   4-sentence batch after `--embed` (the golden gate never caught it
  //   because it tiles ONE 4-sentence batch 32x, so every "different" row is
  //   actually identical content and a wrong-row read is indistinguishable
  //   from a right-row read).
  void swiglu_cpu(const std::vector<float> &x, std::vector<float> &out) {
    double t0 = now_s();
    const int64_t inter = g_ffn;
    const int64_t n_rows = rows;
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n_rows + nw - 1) / nw;
      const int64_t lo_r = std::min<int64_t>(n_rows, chunk * w);
      const int64_t hi_r = std::min<int64_t>(n_rows, lo_r + chunk);
      for (int64_t r = lo_r; r < hi_r; ++r) {
        const float *lo = x.data() + r * 2 * inter;
        const float *hi = lo + inter;
        float *dst = out.data() + r * inter;
        // arch=3 (tasks/0136): exact-erf GELU on the gate half, same halves
        // order. The SiLU arm below is byte-for-byte what arch=2 always ran.
        if (g_gated_act == GatedAct::GeluErf) {
          for (int64_t j = 0; j < inter; ++j)
            dst[j] = lo[j] * gelu_erf_exact(hi[j]);
          continue;
        }
        int64_t j = 0;
#if defined(__AVX2__)
        const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
        const __m256 argfloor = _mm256_set1_ps(-120.0f);
        const __m256 one = _mm256_set1_ps(1.0f);
        for (; j + 8 <= inter; j += 8) {
          __m256 xv = _mm256_loadu_ps(hi + j);
          __m256 a = _mm256_max_ps(
              _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
              argfloor);
          __m256 e = exp2_avx2(a);
          __m256 s = _mm256_div_ps(xv, _mm256_add_ps(one, e));
          __m256 loV = _mm256_loadu_ps(lo + j);
          _mm256_storeu_ps(dst + j, _mm256_mul_ps(loV, s));
        }
#endif
        for (; j < inter; ++j) {
          const float xv = hi[j];
          float a = -xv * 1.4426950408889634f;
          if (a < -120.0f) a = -120.0f;
          const float e = std::exp2(a);
          const float s = xv / (1.0f + e);
          dst[j] = lo[j] * s;
        }
      }
    });
    // Reuses gelu_cpu's bucket -- it is the same "FFN activation, on the
    // host, in place of an NPU dispatch" cost this timer already names.
    t_hostgelu += now_s() - t0;
  }

  void layer_norm(std::vector<float> &x, size_t slot) {
    if (host_ln) { layer_norm_cpu(x, slot - 1); return; }
    double t0 = now_s();
    layernorm.bind(1, slot);
    par(x.size(), [&](size_t lo, size_t hi) {
      bf16_fill(static_cast<uint16_t *>(layernorm.host_ptr(0)) + lo,
                x.data() + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    layernorm.sync_to_device(0);
    t0 = lap(t0, t_in);
    layernorm.dispatch_only();
    t0 = lap(t0, t_disp);
    layernorm.sync_from_device(2);
    t0 = lap(t0, t_out);
    par(x.size(), [&](size_t lo, size_t hi) {
      bf16_read(x.data() + lo,
                static_cast<const uint16_t *>(layernorm.host_ptr(2)) + lo,
                hi - lo);
    });
    lap(t0, t_conv);
    ++n_dispatch;
  }

  // Per-row activation scales for the int8 path, [rows]. Filled by gemm()'s
  // quantisation pass and consumed by its dequantisation pass in the same
  // call, so it never has to be threaded anywhere.
  std::vector<float> a_scale;
  // ...except when the ffn_up epilogue is fused (T37), where one pass produces
  // BOTH this GEMM's dequantised result and the next one's quantised operand,
  // so the two sets of row scales are live at once.
  std::vector<float> a_scale_next;
  std::vector<float> inv_smooth_next;
  const float *inv_smooth_next_src = nullptr;
  // 1/asmooth for the GEMM currently being run, rebuilt only when the pointer
  // changes (i.e. per op, not per layer-iteration, and never per row).
  std::vector<float> inv_smooth;
  const float *inv_smooth_src = nullptr;

  // Set when gemm() should not write `out` at all, but instead run the
  // activation and quantise straight into the NEXT GEMM's operand.
  struct FusedNext {
    const float *asmooth;   // the next GEMM's per-input-channel divisor
    int8_t *dst;            // the next GEMM's device A slot
    float *scale;           // filled with the next GEMM's per-row scales
    bool gated;             // SwiGLU narrows 2*inter -> inter; GELU does not
  };

  // T37-BF16 (tasks/0108): as FusedNext, for the bf16/bfp16 datapath. No
  // quantisation on this path, so no smoothing divisor and no per-row scale
  // -- the only thing that survives is the narrow-to-bf16 write, straight
  // into the next GEMM's device A slot.
  struct FusedNextBf16 {
    uint16_t *dst;           // the next GEMM's device A slot (bf16)
    bool gated;              // SwiGLU/GeGLU narrows 2*inter -> inter
  };

  // nullptr for a bf16 container, where these vectors are empty and the int8
  // arms of gemm() never run.
  static const float *i8w(const std::vector<const float *> &v, int64_t L) {
    return v.empty() ? nullptr : v[static_cast<size_t>(L)];
  }

  // T37-BF16 (tasks/0108): as dequant_act_quant (T37, tasks/0082), for the
  // bf16/bfp16 datapath. STEP 1 of this task measured the three passes this
  // collapses -- ffn_up's C-readback+bias (into `up`), the activation in
  // place, and ffn_down's A-conversion (out of `up`) -- at 17.7-28.3% of
  // wall clock across the shipped catalogue; `up` is never materialised.
  // No quantisation step exists on this path, so unlike dequant_act_quant
  // there is no per-row scale and no smoothing divisor to carry.
  //
  // BIT-IDENTICAL to the unfused path BY CONSTRUCTION, not by review: the
  // C-readback+bias arm below is copied verbatim from gemm()'s own unfused
  // bf16-C and fp32-C branches (the ones just below this function), the
  // activation arms are copied verbatim from the int8 fused epilogue's own
  // lambda (itself verified bit-identical to gelu_cpu/swiglu_cpu, tasks/0082
  // sec 2), and the final narrowing calls the SAME bf16_fill() the unfused
  // A-conversion calls. Calling bf16_fill per row rather than once over the
  // whole buffer changes nothing: it is round-to-nearest-even, purely
  // elementwise -- every output bit depends only on its own input float, not
  // on its neighbours or its position in the array.
  void dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
                        int64_t out_n, bool gated, const float *bias,
                        uint16_t *dst) {
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      std::vector<float> row(static_cast<size_t>(N));
      for (int64_t r = r0; r < r1; ++r) {
        float *v = row.data();
        int64_t j = 0;
#if defined(__AVX2__)
        if (c_bytes == 2) {
          const uint16_t *cr = static_cast<const uint16_t *>(c) + r * N;
          for (; j + 16 <= N; j += 16) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            __m256i lo = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
            __m256i hi = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
            _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(lo),
                                                  _mm256_loadu_ps(bias + j)));
            _mm256_storeu_ps(v + j + 8, _mm256_add_ps(_mm256_castsi256_ps(hi),
                                                  _mm256_loadu_ps(bias + j + 8)));
          }
        } else {
          const float *cr = static_cast<const float *>(c) + r * N;
          for (; j + 8 <= N; j += 8) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(raw),
                                                  _mm256_loadu_ps(bias + j)));
          }
        }
#endif
        for (; j < N; ++j) {
          const float cf = c_bytes == 2
              ? from_bf16(static_cast<const uint16_t *>(c)[r * N + j])
              : static_cast<const float *>(c)[r * N + j];
          v[j] = cf + bias[j];
        }
        // Activation, in place -- COPIED VERBATIM from the int8 fused
        // epilogue's own lambda just below, not re-derived.
        if (!gated) {
          int64_t k = 0;
#if defined(__AVX2__)
          for (; k + 8 <= N; k += 8)
            _mm256_storeu_ps(v + k, gelu8(_mm256_loadu_ps(v + k)));
#endif
          for (; k < N; ++k) v[k] = gelu8(v[k]);
        } else if (g_gated_act == GatedAct::GeluErf) {
          // arch=3 (tasks/0136): exact-erf GELU on the gate half. The SiLU
          // arm below is byte-for-byte what arch=2 always ran.
          const int64_t inter = N / 2;
          const float *hi = v + inter;
          for (int64_t k = 0; k < inter; ++k)
            v[k] = v[k] * gelu_erf_exact(hi[k]);
        } else {
          const int64_t inter = N / 2;
          const float *hi = v + inter;
          int64_t k = 0;
#if defined(__AVX2__)
          const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
          const __m256 argfloor = _mm256_set1_ps(-120.0f);
          const __m256 one = _mm256_set1_ps(1.0f);
          for (; k + 8 <= inter; k += 8) {
            __m256 xv = _mm256_loadu_ps(hi + k);
            __m256 a = _mm256_max_ps(
                _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
                argfloor);
            __m256 e = exp2_avx2(a);
            __m256 sg = _mm256_div_ps(xv, _mm256_add_ps(one, e));
            _mm256_storeu_ps(v + k, _mm256_mul_ps(_mm256_loadu_ps(v + k), sg));
          }
#endif
          for (; k < inter; ++k) {
            const float xv = hi[k];
            float a = -xv * 1.4426950408889634f;
            if (a < -120.0f) a = -120.0f;
            v[k] = v[k] * (xv / (1.0f + std::exp2f(a)));
          }
        }
        bf16_fill(dst + r * out_n, v, static_cast<size_t>(out_n));
      }
    });
  }

  void gemm(npu::Design &d, size_t islot, const std::vector<float> &a,
            size_t wslot, const float *bias, std::vector<float> &out,
            int64_t N, const float *wscale = nullptr,
            const float *asmooth = nullptr,
            FusedNext *fuse = nullptr, bool a_ready = false,
            FusedNextBf16 *fuse_bf16 = nullptr) {
    const bool i8 = d.info().a_elem_bytes == 1;
    if (i8 && (wscale == nullptr || asmooth == nullptr))
      throw std::runtime_error(
          "int8 design but this encoder has no quantisation scales -- the "
          "container is bf16, or a pipeline lane was constructed without "
          "copying ws_*/as_* from lane 0");
    double t0 = now_s();
    if (i8 && a_ready) {
      // A was written straight into the device slot by the PREVIOUS gemm's
      // fused epilogue (T37), and a_scale already holds its row scales. There
      // is nothing to convert: the whole point is that this tensor is never
      // materialised in fp32 at all.
    } else if (i8) {
      // QUANTISE A, per row, with the SmoothQuant divisor folded into the same
      // pass (tasks/0078). Two reads of each row would cost a second sweep of
      // 12.6 MB at batch 128; one pass computes max|x/s| and the second writes
      // the rounded quotient.
      //
      // The smoothing divisor is NOT folded into LayerNorm: BERT is post-LN,
      // so each LayerNorm's output feeds the residual as well as this GEMM,
      // and scaling gamma/beta would scale the residual too (tasks/0078 4a).
      const int64_t K = static_cast<int64_t>(a.size()) / rows;
      a_scale.resize(static_cast<size_t>(rows));
      // RECIPROCATE THE SMOOTHING VECTOR ONCE PER GEMM, not twice per element.
      // The first version divided by asmooth[j] in both the max pass and the
      // quantise pass -- two divisions per element, and `conv` went 36.2 ms
      // (bf16) to 109.9 ms, eating 74 ms of the 126 ms the array had saved.
      // K floats of setup replaces 2*rows*K divisions.
      if (inv_smooth.size() != static_cast<size_t>(K) ||
          inv_smooth_src != asmooth) {
        inv_smooth.resize(static_cast<size_t>(K));
        for (int64_t j = 0; j < K; ++j) inv_smooth[j] = 1.0f / asmooth[j];
        inv_smooth_src = asmooth;
      }
      const float *ias = inv_smooth.data();
      auto *abuf = static_cast<int8_t *>(d.slot_ptr(0, slot_a));
      quantise_a_int8(a.data(), rows, K, ias, abuf, a_scale.data(),
                      [&](int64_t n, auto f) { par_rows(n, f); });
      t0 = lap(t0, t_conv);
    } else if (a_ready) {
      // T37-BF16 (tasks/0108): A was written straight into the device slot,
      // already narrowed to bf16, by the PREVIOUS gemm's fused epilogue --
      // the bf16 analogue of the i8-and-a_ready arm above. Nothing to do.
    } else {
    auto *abuf = static_cast<uint16_t *>(d.slot_ptr(0, slot_a));
    par(a.size(), [&](size_t lo, size_t hi) {
      bf16_fill(abuf + lo, a.data() + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    }
    const float *c;
    {
      std::unique_lock<std::mutex> lk;
      if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
      if (unified) d.bind_instr(islot);
      d.bind(0, slot_a);
      d.bind(1, wslot);        // weights are already on the device
      d.bind(2, slot_c);
      d.sync_to_device(0, a.size() * d.info().a_elem_bytes);
      t0 = lap(t0, t_in);
      d.dispatch_only();
      t0 = lap(t0, t_disp);
      // NPUE-M9 (tasks/0045): with --c-bf16 the design narrows C on the core
      // after the fp32 K reduction, so this moves half the bytes. The size
      // comes from the design, never from an assumption about the datatype.
      const size_t cb = d.info().c_elem_bytes;
      d.sync_from_device(2, static_cast<size_t>(rows) * N * cb);
      t0 = lap(t0, t_out);
      // The pointer survives the unlock -- it is THIS pipeline's own bo; the
      // other pipeline binds its own slots and never touches this memory.
      c = static_cast<const float *>(d.slot_ptr(2, slot_c));
    }
    // The bias add reads the result buffer directly. It used to be a memcpy
    // out followed by a second pass over the same 21 MB; this is one pass.
    //
    // STREAMING loads in both arms: the C buffer is an XRT host bo, and
    // ordinary loads from it measured ~80 ms per encode (~2 GB/s) -- the
    // signature of uncached/write-combined memory, where each load stalls the
    // core. movntdqa reads a whole WC line per transaction. Alignment holds:
    // the bo map is page-aligned and N is a multiple of 16.
    if (i8 && fuse) {
      // FUSED EPILOGUE (T37): dequantise, apply GELU, and quantise into the
      // next GEMM's operand in ONE pass over the widest tensor in the model.
      // `out` is deliberately never written -- see dequant_gelu_quant.
      const int64_t Kn = fuse->gated ? N / 2 : N;
      if (inv_smooth_next.size() != static_cast<size_t>(Kn) ||
          inv_smooth_next_src != fuse->asmooth) {
        inv_smooth_next.resize(static_cast<size_t>(Kn));
        for (int64_t j = 0; j < Kn; ++j)
          inv_smooth_next[j] = 1.0f / fuse->asmooth[j];
        inv_smooth_next_src = fuse->asmooth;
      }
      const int64_t out_n = fuse->gated ? N / 2 : N;
      dequant_act_quant(
          c, d.info().c_elem_bytes, rows, N, out_n,
          [gated = fuse->gated](float *v, int64_t n) {
            if (!gated) {
              int64_t j = 0;
#if defined(__AVX2__)
              for (; j + 8 <= n; j += 8)
                _mm256_storeu_ps(v + j, gelu8(_mm256_loadu_ps(v + j)));
#endif
              for (; j < n; ++j) v[j] = gelu8(v[j]);
              return;
            }
            // arch=3 (tasks/0136): no int8 gte container exists yet
            // (pack_npue refuses --int8 for arch=3), but if one arrives this
            // arm must not silently run SiLU over a GELU model.
            if (g_gated_act == GatedAct::GeluErf) {
              const int64_t inter = n / 2;
              const float *hi = v + inter;
              for (int64_t j = 0; j < inter; ++j)
                v[j] = v[j] * gelu_erf_exact(hi[j]);
              return;
            }
            // SwiGLU, narrowing 2*inter -> inter in place. Identical
            // intrinsics to swiglu_cpu, including its -120 argument floor.
            const int64_t inter = n / 2;
            const float *hi = v + inter;
            int64_t j = 0;
#if defined(__AVX2__)
            const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
            const __m256 argfloor = _mm256_set1_ps(-120.0f);
            const __m256 one = _mm256_set1_ps(1.0f);
            for (; j + 8 <= inter; j += 8) {
              __m256 xv = _mm256_loadu_ps(hi + j);
              __m256 a = _mm256_max_ps(
                  _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
                  argfloor);
              __m256 e = Encoder::exp2_avx2(a);
              __m256 sg = _mm256_div_ps(xv, _mm256_add_ps(one, e));
              _mm256_storeu_ps(v + j, _mm256_mul_ps(_mm256_loadu_ps(v + j), sg));
            }
#endif
            for (; j < inter; ++j) {
              const float xv = hi[j];
              float a = -xv * 1.4426950408889634f;
              if (a < -120.0f) a = -120.0f;
              v[j] = v[j] * (xv / (1.0f + std::exp2f(a)));
            }
          },
          a_scale.data(), wscale, bias, inv_smooth_next.data(), fuse->dst,
          fuse->scale, [&](int64_t n, auto f) { par_rows(n, f); });
    } else if (i8) {
      // DEQUANTISE: y = int32_acc * sa[row] * wscale[col] + bias[col], folded
      // into the pass that already reads C and adds the bias -- one extra
      // multiply per output element, no extra sweep of the 679 MB tasks/0044
      // measured this readback at. The helper picks the transport width from
      // the design, so a narrowed-C set (tasks/0080) needs nothing here.
      dequantise_c(c, d.info().c_elem_bytes, rows, N, a_scale.data(), wscale,
                   bias, out.data(),
                   [&](int64_t n, auto f) { par_rows(n, f); }, sim_c_bf16);
    } else if (fuse_bf16) {
      // T37-BF16 (tasks/0108): `out` is deliberately never written -- see
      // dequant_act_bf16 above.
      const int64_t out_n = fuse_bf16->gated ? N / 2 : N;
      dequant_act_bf16(c, d.info().c_elem_bytes, N, out_n, fuse_bf16->gated,
                       bias, fuse_bf16->dst);
    } else if (d.info().c_elem_bytes == 2) {
      const uint16_t *cb16 = reinterpret_cast<const uint16_t *>(c);
      par(size_t(rows), [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
          const uint16_t *cr = cb16 + r * N;
          float *o = out.data() + r * N;
          int64_t j = 0;
#if defined(__AVX2__)
          // One 32-byte streaming load carries 16 bf16, against 8 fp32 --
          // which is the whole point: same instruction count, half the traffic.
          for (; j + 16 <= N; j += 16) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            __m256i lo = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
            __m256i hi = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
            _mm256_storeu_ps(o + j,
                             _mm256_add_ps(_mm256_castsi256_ps(lo),
                                           _mm256_loadu_ps(bias + j)));
            _mm256_storeu_ps(o + j + 8,
                             _mm256_add_ps(_mm256_castsi256_ps(hi),
                                           _mm256_loadu_ps(bias + j + 8)));
          }
#endif
          for (; j < N; ++j) o[j] = from_bf16(cr[j]) + bias[j];
        }
      });
    } else {
      par(size_t(rows), [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
          const float *cr = c + r * N;
          float *o = out.data() + r * N;
          int64_t j = 0;
#if defined(__AVX2__)
          for (; j + 8 <= N; j += 8) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            _mm256_storeu_ps(o + j, _mm256_add_ps(_mm256_castsi256_ps(raw),
                                                  _mm256_loadu_ps(bias + j)));
          }
#endif
          for (; j < N; ++j) o[j] = cr[j] + bias[j];
        }
      });
    }
    lap(t0, t_bias);
    ++n_dispatch;
  }

  // THE OTHER MULTI-PASS CHAIN (T37, tasks/0082 section 5).
  //
  //   add_into(x, y)      read y, read residual, write x
  //   layer_norm(x)       read x, write x
  //   memcpy(residual, x) read x, write residual
  //   quantise for next   read x, write int8
  //
  // Eight streaming passes over a rows x hidden tensor, TWICE per layer --
  // 33.5 MB at bge-large's batch 128. Every one of them touches the same row,
  // and a row is 4 KB, so all four kernels can share one L1-resident copy:
  // read y and residual once, write x, residual and the int8 operand once.
  //
  // `dst` may be null (the last layer's LN feeds pooling, not a GEMM), in
  // which case this is add + norm + residual with no quantisation.
  //
  // BIT-IDENTICAL to the four kernels it replaces, and that is not automatic:
  // it holds only because the intrinsics and the accumulation ORDER match
  // theirs exactly. tasks/0082 measured a scalar rewrite of the same algebra
  // landing on a different number, because fmadd rounds once where a*b+c
  // rounds twice.
  void add_norm_quant(std::vector<float> &x, const std::vector<float> &y,
                      size_t site, const float *ias_next, int8_t *dst,
                      float *scale_next) {
    double t0 = now_s();
    const float *g = h_gamma[site], *b = h_beta[site];
    const int64_t H = g_hidden;
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        float *row = x.data() + r * H;
        const float *yr = y.data() + r * H;
        float *res = residual.data() + r * H;
        int64_t j = 0;
#if defined(__AVX2__)
        for (; j + 8 <= H; j += 8)                       // == add_into
          _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_loadu_ps(yr + j),
                                                  _mm256_loadu_ps(res + j)));
#endif
        for (; j < H; ++j) row[j] = yr[j] + res[j];
#if defined(__AVX2__)
        __m256 s = _mm256_setzero_ps();                  // == layer_norm_cpu
        for (j = 0; j + 8 <= H; j += 8)
          s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
        const float mean = hsum256(s) / H;
        const __m256 mv = _mm256_set1_ps(mean);
        __m256 v = _mm256_setzero_ps();
        for (j = 0; j + 8 <= H; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          v = _mm256_fmadd_ps(d, d, v);
        }
        const float var = hsum256(v) / H;
        const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
        for (j = 0; j + 8 <= H; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          __m256 yv = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                      _mm256_loadu_ps(g + j),
                                      _mm256_loadu_ps(b + j));
          _mm256_storeu_ps(row + j, yv);
          _mm256_storeu_ps(res + j, yv);                 // == memcpy residual
        }
#else
        double sm = 0.0;
        for (j = 0; j < H; ++j) sm += row[j];
        const float mean = static_cast<float>(sm / H);
        double vs = 0.0;
        for (j = 0; j < H; ++j) vs += double(row[j] - mean) * (row[j] - mean);
        const float is = 1.0f / std::sqrt(static_cast<float>(vs / H) + 1e-12f);
        for (j = 0; j < H; ++j) {
          row[j] = (row[j] - mean) * is * g[j] + b[j];
          res[j] = row[j];
        }
#endif
        if (!dst) continue;
        float mx = 0.f;                                  // == quantise_a_int8
        j = 0;
#if defined(__AVX2__)
        {
          const __m256 absmask =
              _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
          __m256 acc = _mm256_setzero_ps();
          for (; j + 8 <= H; j += 8)
            acc = _mm256_max_ps(acc, _mm256_and_ps(
                _mm256_mul_ps(_mm256_loadu_ps(row + j),
                              _mm256_loadu_ps(ias_next + j)), absmask));
          __m128 h = _mm_max_ps(_mm256_castps256_ps128(acc),
                                _mm256_extractf128_ps(acc, 1));
          h = _mm_max_ps(h, _mm_movehl_ps(h, h));
          h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
          mx = _mm_cvtss_f32(h);
        }
#endif
        for (; j < H; ++j) {
          const float a = std::fabs(row[j] * ias_next[j]);
          if (a > mx) mx = a;
        }
        const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
        scale_next[static_cast<size_t>(r)] = sc;
        const float inv = 1.0f / sc;
        int8_t *q = dst + r * H;
        j = 0;
#if defined(__AVX2__)
        {
          const __m256 invv = _mm256_set1_ps(inv);
          const __m256 vhi = _mm256_set1_ps(127.0f);
          const __m256 vlo = _mm256_set1_ps(-127.0f);
          for (; j + 8 <= H; j += 8) {
            __m256 t = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_loadu_ps(row + j),
                              _mm256_loadu_ps(ias_next + j)), invv);
            t = _mm256_min_ps(_mm256_max_ps(t, vlo), vhi);
            __m256i i32 = _mm256_cvtps_epi32(t);
            __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                          _mm256_extracti128_si256(i32, 1));
            _mm_storel_epi64(reinterpret_cast<__m128i *>(q + j),
                             _mm_packs_epi16(p16, p16));
          }
        }
#endif
        for (; j < H; ++j) {
          float t = std::nearbyintf(row[j] * ias_next[j] * inv);
          if (t > 127.f) t = 127.f;
          if (t < -127.f) t = -127.f;
          q[j] = static_cast<int8_t>(t);
        }
      }
    });
    t_hostln += now_s() - t0;
  }

  // T37-BF16 (tasks/0108): as add_norm_quant, for the bf16/bfp16 datapath --
  // add + LayerNorm + residual copy + the NEXT gemm's A-conversion, one
  // L1-resident pass instead of the unfused chain's four streaming ones
  // (add_into, layer_norm_cpu, memcpy(residual), and the next gemm()'s own
  // bf16_fill). No quantisation on this path, so this is a strict subset of
  // add_norm_quant's work -- the add/LN/residual arithmetic below is copied
  // verbatim from it, with the int8 absmax+quantise tail replaced by a
  // single bf16_fill call. `dst` may be null (the last layer's LN feeds
  // pooling, not a GEMM), in which case this is add + norm + residual only,
  // exactly like the unfused path's own unconditional residual write.
  void add_norm_bf16(std::vector<float> &x, const std::vector<float> &y,
                     size_t site, uint16_t *dst) {
    double t0 = now_s();
    const float *g = h_gamma[site], *b = h_beta[site];
    const int64_t H = g_hidden;
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        float *row = x.data() + r * H;
        const float *yr = y.data() + r * H;
        float *res = residual.data() + r * H;
        int64_t j = 0;
#if defined(__AVX2__)
        for (; j + 8 <= H; j += 8)                       // == add_into
          _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_loadu_ps(yr + j),
                                                  _mm256_loadu_ps(res + j)));
#endif
        for (; j < H; ++j) row[j] = yr[j] + res[j];
#if defined(__AVX2__)
        __m256 s = _mm256_setzero_ps();                  // == layer_norm_cpu
        for (j = 0; j + 8 <= H; j += 8)
          s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
        const float mean = hsum256(s) / H;
        const __m256 mv = _mm256_set1_ps(mean);
        __m256 v = _mm256_setzero_ps();
        for (j = 0; j + 8 <= H; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          v = _mm256_fmadd_ps(d, d, v);
        }
        const float var = hsum256(v) / H;
        const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
        for (j = 0; j + 8 <= H; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          __m256 yv = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                      _mm256_loadu_ps(g + j),
                                      _mm256_loadu_ps(b + j));
          _mm256_storeu_ps(row + j, yv);
          _mm256_storeu_ps(res + j, yv);                 // == memcpy residual
        }
#else
        double sm = 0.0;
        for (j = 0; j < H; ++j) sm += row[j];
        const float mean = static_cast<float>(sm / H);
        double vs = 0.0;
        for (j = 0; j < H; ++j) vs += double(row[j] - mean) * (row[j] - mean);
        const float is = 1.0f / std::sqrt(static_cast<float>(vs / H) + 1e-12f);
        for (j = 0; j < H; ++j) {
          row[j] = (row[j] - mean) * is * g[j] + b[j];
          res[j] = row[j];
        }
#endif
        if (dst) bf16_fill(dst + r * H, row, static_cast<size_t>(H));
      }
    });
    t_hostln += now_s() - t0;
  }

  // x += y, elementwise. The residual adds move 12.6 MB per layer at batch 128.
  void add_into(std::vector<float> &x, const std::vector<float> &y) {
    par(x.size(), [&](size_t lo, size_t hi) {
      size_t i = lo;
#if defined(__AVX2__)
      for (; i + 8 <= hi; i += 8)
        _mm256_storeu_ps(x.data() + i,
                         _mm256_add_ps(_mm256_loadu_ps(y.data() + i),
                                       _mm256_loadu_ps(residual.data() + i)));
#endif
      for (; i < hi; ++i) x[i] = y[i] + residual[i];
    });
  }

  // scores[b,h,i,j] = dot(Q[b,i,h], K[b,j,h]) + mask[b,j]
  // scores[b,h,i,j] = Q[b,i,h] . K[b,j,h]. NO mask: this is the operation an
  // array kernel would perform, and the mask is a property of the batch rather
  // than of the matmul. add_additive_mask() below applies it.
  // NV is head_dim/8 as a COMPILE-TIME constant where we have one, so the
  // inner loop unrolls and qv[]/acc[] stay in registers. NV == 0 keeps the
  // fully generic path for a width we have not met yet.
  template <int NV>
  void qk_impl(const std::vector<float> &qkvbuf, std::vector<float> &scores) {
    const int64_t pairs = batch * g_heads;
    // __restrict, because these are members now: the compiler could prove two
    // fresh local allocations did not overlap and cannot prove it for two
    // fields of the same object, and without the proof every store to dst[j]
    // re-issues the loads. Measured at 2x on this loop.
    const float *__restrict qkv_p = qkvbuf.data();
    float *__restrict sc_p = scores.data();
    pool->run([&](int w, int nw) {
      for (int64_t p = w; p < pairs; p += nw) {
        const int64_t b = p / g_heads, h = p % g_heads;
        for (int64_t i = 0; i < g_seq; ++i) {
          const float *q = &qkv_p[(b * g_seq + i) * 3 * g_hidden + h * g_head_dim];
          float *dst = &sc_p[(p * g_seq + i) * g_seq];
          // head_dim / 8 vectors, held across the j loop. head_dim is 32 for
          // MiniLM and bge-small and 64 for bge-large; kMaxHeadVecs bounds the
          // stack array and set_model_shape() refuses anything larger.
#if defined(__AVX512F__)
          // QK^T AT 512 BITS. Unlike A.V (which gained nothing, tasks/0086),
          // this one pays -- and not because the arithmetic is wider. The win
          // is that `_mm512_reduce_add_ps` replaces `hsum256`'s four-
          // instruction shuffle chain, and QK^T does one reduction per (i, j)
          // pair. Microbenchmarked at bge-large's geometry: 1.33x on the inner
          // loop, against 1.00x for the same widening applied to A.V.
          //
          // NOT BIT-IDENTICAL, and it cannot be: summing head_dim floats as
          // lanes of 16 associates differently from lanes of 8. That is a
          // legitimate reassociation, not an error, but it means the byte
          // comparison the fusions used does not apply here -- this is gated on
          // 1-cos instead.
          //
          // head_dim is a multiple of 8 but not necessarily of 16, so the
          // 512-bit part takes what it can and a 256-bit tail finishes.
          const int64_t nv = NV ? NV : g_head_dim / 8;
          const int64_t nz = nv / 2;
          const bool zt = (nv & 1) != 0;
          __m512 zq[(NV ? NV : kMaxHeadVecs) / 2 + 1];
          __m256 yq;
          for (int64_t v = 0; v < nz; ++v) zq[v] = _mm512_loadu_ps(q + v * 16);
          if (zt) yq = _mm256_loadu_ps(q + nz * 16);
#else
          __m256 qv[NV ? NV : kMaxHeadVecs];
          const int64_t nv = NV ? NV : g_head_dim / 8;
          for (int64_t v = 0; v < nv; ++v) qv[v] = _mm256_loadu_ps(q + v * 8);
#endif
          for (int64_t j = 0; j < g_seq; ++j) {
            const float *k = &qkv_p[(b * g_seq + j) * 3 * g_hidden + g_hidden +
                                    h * g_head_dim];
#if defined(__AVX512F__)
            __m512 zs = _mm512_mul_ps(zq[0], _mm512_loadu_ps(k));
            for (int64_t v = 1; v < nz; ++v)
              zs = _mm512_fmadd_ps(zq[v], _mm512_loadu_ps(k + v * 16), zs);
            float acc = _mm512_reduce_add_ps(zs);
            if (zt) acc += hsum256(_mm256_mul_ps(yq,
                                                 _mm256_loadu_ps(k + nz * 16)));
            dst[j] = acc;
#elif defined(__AVX2__)
            // Accumulate in the same order the unrolled version did, so the
            // floating-point result is unchanged for head_dim 32.
            __m256 s = _mm256_mul_ps(qv[0], _mm256_loadu_ps(k));
            for (int64_t v = 1; v < nv; ++v)
              s = _mm256_fmadd_ps(qv[v], _mm256_loadu_ps(k + v * 8), s);
            dst[j] = hsum256(s);
#else
            float s = 0.f;
            for (int64_t d = 0; d < g_head_dim; ++d) s += q[d] * k[d];
            dst[j] = s;
#endif
          }
        }
      }
    });
  }

  // Dispatch on the width the container reported. head_dim 32 is MiniLM and
  // bge-small, 64 is bge-large; anything else still works, just generically.
  void qk(const std::vector<float> &qkvbuf, std::vector<float> &scores) {
    switch (g_head_dim) {
      case 32: qk_impl<4>(qkvbuf, scores); break;
      case 64: qk_impl<8>(qkvbuf, scores); break;
      default: qk_impl<0>(qkvbuf, scores); break;
    }
  }

  // ctx[b,i,h] = sum_j scores[b,h,i,j] * V[b,j,h]
  template <int NV>
  void av_impl(const std::vector<float> &scores,
               const std::vector<float> &qkvbuf, std::vector<float> &ctx) {
    const int64_t pairs = batch * g_heads;
    const float *__restrict sc_p = scores.data();
    const float *__restrict qkv_p = qkvbuf.data();
    float *__restrict ctx_p = ctx.data();
    pool->run([&](int w, int nw) {
      for (int64_t p = w; p < pairs; p += nw) {
        const int64_t b = p / g_heads, h = p % g_heads;
        for (int64_t i = 0; i < g_seq; ++i) {
          const float *a = &sc_p[(p * g_seq + i) * g_seq];
          float *o = &ctx_p[(b * g_seq + i) * g_hidden + h * g_head_dim];
#if defined(__AVX2__)
#if defined(__AVX512F__)
          // A.V AT 512 BITS IS BIT-IDENTICAL TO THE 256-BIT FORM, and that is
          // not luck -- it is why this half was done first (tasks/0086).
          //
          // Each accumulator lane owns ONE output element and sums over j in
          // the same order either way; widening only changes how many lanes
          // ride in a register, never which numbers are added or when. So the
          // byte-comparison harness applies here exactly as it did to the
          // fusions.
          //
          // QK^T is the opposite case and is NOT converted: it ends in a
          // horizontal reduction (`hsum256`), and reducing head_dim floats as
          // 2 lanes of 16 sums them in a different ORDER than 4 lanes of 8.
          // That is a reassociation, so it would change the result -- small,
          // legitimate, and no longer checkable by byte comparison. Left for a
          // measurement that uses the 1-cos gate instead.
          //
          // head_dim is a multiple of 8 (set_model_shape refuses otherwise) but
          // not necessarily of 16, so the 512-bit path takes the multiple-of-16
          // part and a 256-bit tail finishes it.
          const int64_t nv = NV ? NV : g_head_dim / 8;
          const int64_t nz = nv / 2;                  // 512-bit accumulators
          __m512 zacc[(NV ? NV : kMaxHeadVecs) / 2 + 1];
          __m256 yacc;
          for (int64_t v = 0; v < nz; ++v) zacc[v] = _mm512_setzero_ps();
          const bool tail = (nv & 1) != 0;
          if (tail) yacc = _mm256_setzero_ps();
          for (int64_t j = 0; j < g_seq; ++j) {
            const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                    2 * g_hidden + h * g_head_dim];
            const __m512 zaj = _mm512_set1_ps(a[j]);
            for (int64_t k = 0; k < nz; ++k)
              zacc[k] = _mm512_fmadd_ps(zaj, _mm512_loadu_ps(v + k * 16),
                                        zacc[k]);
            if (tail)
              yacc = _mm256_fmadd_ps(_mm256_set1_ps(a[j]),
                                     _mm256_loadu_ps(v + nz * 16), yacc);
          }
          for (int64_t v = 0; v < nz; ++v)
            _mm512_storeu_ps(o + v * 16, zacc[v]);
          if (tail) _mm256_storeu_ps(o + nz * 16, yacc);
#else
          __m256 acc[NV ? NV : kMaxHeadVecs];
          const int64_t nv = NV ? NV : g_head_dim / 8;
          for (int64_t v = 0; v < nv; ++v) acc[v] = _mm256_setzero_ps();
          for (int64_t j = 0; j < g_seq; ++j) {
            const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                    2 * g_hidden + h * g_head_dim];
            const __m256 aj = _mm256_set1_ps(a[j]);
            for (int64_t k = 0; k < nv; ++k)
              acc[k] = _mm256_fmadd_ps(aj, _mm256_loadu_ps(v + k * 8), acc[k]);
          }
          for (int64_t v = 0; v < nv; ++v)
            _mm256_storeu_ps(o + v * 8, acc[v]);
#endif
#else
          for (int64_t d = 0; d < g_head_dim; ++d) o[d] = 0.f;
          for (int64_t j = 0; j < g_seq; ++j) {
            const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                    2 * g_hidden + h * g_head_dim];
            for (int64_t d = 0; d < g_head_dim; ++d) o[d] += a[j] * v[d];
          }
#endif
        }
      }
    });
  }

  void av(const std::vector<float> &scores, const std::vector<float> &qkvbuf,
          std::vector<float> &ctx) {
    switch (g_head_dim) {
      case 32: av_impl<4>(scores, qkvbuf, ctx); break;
      case 64: av_impl<8>(scores, qkvbuf, ctx); break;
      default: av_impl<0>(scores, qkvbuf, ctx); break;
    }
  }

  // arch=2: rotate Q and K IN PLACE inside the fused qkv buffer -- Q at
  // column offset 0, K at `hidden`, V at `2*hidden` -- rather than repacking
  // into [B,H,S,D] the way gemma_encode.cpp does. Repacking would be a
  // ~200 MB shuffle per layer at production batch; rotating the 64 floats of
  // each head in place needs no extra buffer at all.
  //
  // For each (b, s) row and each head h, at `row_base + q_off + h*head_dim`
  // (`half = head_dim/2`):
  //   for d in [0, half):
  //     x1 = v[d]; x2 = v[d + half]
  //     v[d]        = x1*cos[s][d] - x2*sin[s][d]
  //     v[d + half] = x2*cos[s][d] + x1*sin[s][d]
  // Both halves read cos[s][d]/sin[s][d] for the SAME d -- that is what
  // NeoX's concat(freqs, freqs) means (gemma_kernels.hpp's own
  // gemma_rope_tables() already duplicates the table this way), so only the
  // first `half` columns of the table are ever read here. Applied to Q
  // (offset 0) and K (offset g_hidden). NEVER V.
  void apply_rope_qkv(std::vector<float> &qkv) {
    if (!rope_ready) {
      // Built ONCE per Encoder: g_seq/g_head_dim/g_rope_theta are fixed for
      // the whole design (set_design_seq() runs once, before any Encoder
      // exists), so every layer of every call shares this table.
      rope_cos.resize(static_cast<size_t>(g_seq * g_head_dim));
      rope_sin.resize(static_cast<size_t>(g_seq * g_head_dim));
      if (!g_rope_inv_freq.empty()) {
        // arch=3: the frequencies come from the container (tasks/0134 --
        // not derivable from any single theta). Same NeoX
        // concat(freqs, freqs) table layout gemma_rope_tables() emits, and
        // the same double-angle, round-once-at-the-end arithmetic; the
        // rotation below only ever reads the first half of each row.
        const int64_t half = g_head_dim / 2;
        for (int64_t s = 0; s < g_seq; ++s) {
          float *cs = rope_cos.data() + s * g_head_dim;
          float *sn = rope_sin.data() + s * g_head_dim;
          for (int64_t j = 0; j < half; ++j) {
            const double ang =
                static_cast<double>(s) *
                static_cast<double>(g_rope_inv_freq[static_cast<size_t>(j)]);
            const float c = static_cast<float>(std::cos(ang));
            const float si = static_cast<float>(std::sin(ang));
            cs[j] = c;
            cs[half + j] = c;
            sn[j] = si;
            sn[half + j] = si;
          }
        }
      } else {
        npue::gemma_rope_tables(g_seq, g_head_dim, g_rope_theta,
                                rope_cos.data(), rope_sin.data());
      }
      rope_ready = true;
    }
    const int64_t half = g_head_dim / 2;
    const int64_t row_stride = 3 * g_hidden;
    const int64_t n_rows = batch * g_seq;
    float *__restrict p = qkv.data();
    const float *__restrict cos_p = rope_cos.data();
    const float *__restrict sin_p = rope_sin.data();

    auto rotate_pair = [half](float *v, const float *cs, const float *sn) {
      int64_t d = 0;
#if defined(__AVX2__)
      for (; d + 8 <= half; d += 8) {
        __m256 x1 = _mm256_loadu_ps(v + d);
        __m256 x2 = _mm256_loadu_ps(v + d + half);
        __m256 c = _mm256_loadu_ps(cs + d);
        __m256 s = _mm256_loadu_ps(sn + d);
        __m256 o1 = _mm256_sub_ps(_mm256_mul_ps(x1, c), _mm256_mul_ps(x2, s));
        __m256 o2 = _mm256_add_ps(_mm256_mul_ps(x2, c), _mm256_mul_ps(x1, s));
        _mm256_storeu_ps(v + d, o1);
        _mm256_storeu_ps(v + d + half, o2);
      }
#endif
      for (; d < half; ++d) {
        const float x1 = v[d], x2 = v[d + half];
        v[d] = x1 * cs[d] - x2 * sn[d];
        v[d + half] = x2 * cs[d] + x1 * sn[d];
      }
    };

    pool->run([&](int w, int nw) {
      for (int64_t row = w; row < n_rows; row += nw) {
        const int64_t s = row % g_seq;    // [b][s] row order -> s = row % seq
        const float *cs = cos_p + s * g_head_dim;
        const float *sn = sin_p + s * g_head_dim;
        float *row_base = p + row * row_stride;
        for (int64_t h = 0; h < g_heads; ++h) {
          rotate_pair(row_base + h * g_head_dim, cs, sn);              // Q
          rotate_pair(row_base + g_hidden + h * g_head_dim, cs, sn);   // K
        }
      }
    });
  }

  std::vector<float> run(const std::vector<float> &emb_in) {
    std::vector<float> x = emb_in;
    layer_norm(x, s_ln[0]);

    qkvbuf.resize(rows * 3 * g_hidden);
    ctx.resize(rows * g_hidden);
    proj.resize(rows * g_hidden);
    up.resize(rows * (g_gated_ffn ? 2 : 1) * g_ffn);
    if (g_gated_ffn) gated.resize(rows * g_ffn);
    down.resize(rows * g_hidden);
    scores.resize(batch * g_heads * g_seq * g_seq);

    residual.resize(x.size());
    // Set by the PREVIOUS iteration's fused LayerNorm, which already wrote
    // this layer's qkv operand and its row scales into the device slot.
    bool qkv_a_ready = false;
    for (int64_t L = 0; L < g_layers; ++L) {
      if (!qkv_a_ready)
        std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));

      gemm(qkv, is_qkv, x, s_qkv[L], b_qkv[L], qkvbuf, 3 * g_hidden,
           i8w(ws_qkv, L), i8w(as_qkv, L), nullptr, /*a_ready=*/qkv_a_ready);
      qkv_a_ready = false;
      // arch=2: rotate Q and K in place, strictly after the GEMM (RoPE is a
      // per-position rotation of the projected q/k, not of the input) and
      // strictly before qk() reads them. No-op (false) for every arch=0
      // container.
      if (g_rope) apply_rope_qkv(qkvbuf);

      double ta = now_s();
      // QK^T per head, on the host: [64,32]x[32,64] does not tile (head_dim 32
      // fails the whole-array design's M % (m*4) == 0).
      // 1/sqrt(head_dim) is already folded into Q by the .npue.
      //
      // head_dim is 32 = four AVX2 vectors, and the 32 floats of one head ARE
      // contiguous even though consecutive rows are 3*hidden apart. So the dot
      // product vectorises without any repacking.
      qk(qkvbuf, scores);
      t_attn += now_s() - ta; t_qk += now_s() - ta;

      if (host_sm) {
        softmax_cpu(scores);  // applies add_mask itself
      } else {
        add_additive_mask(scores);
        eltwise(softmax, scores.data(), scores.size());
      }

      ta = now_s();
      // A.V. Each (b, h, i) owns its own 32 output floats, so this accumulates
      // in registers and stores once -- no zero-fill of ctx needed, and no
      // sharing between threads.
      av(scores, qkvbuf, ctx);
      t_attn += now_s() - ta; t_av += now_s() - ta;

      gemm(attn_out, is_ao, ctx, s_ao[L], b_ao[L], proj, g_hidden,
           i8w(ws_ao, L), i8w(as_ao, L));
      // T37 site 1: add + LayerNorm + residual copy + ffn_up's quantisation,
      // one L1-resident pass instead of four streaming ones.
      const bool fuse_ln = fuse_ffn_epilogue && host_ln &&
                           ffn_up.info().a_elem_bytes == 1;
      // T37-BF16 (tasks/0108): the bf16/bfp16 analogue -- same fused pass,
      // narrowing straight to bf16 instead of quantising (no scale, no
      // smoothing divisor).
      const bool fuse_ln_bf16 = fuse_ffn_epilogue && host_ln &&
                                ffn_up.info().a_elem_bytes == 2;
      if (fuse_ln) {
        const float *asf = i8w(as_fu, L);
        if (inv_smooth.size() != static_cast<size_t>(g_hidden) ||
            inv_smooth_src != asf) {
          inv_smooth.resize(static_cast<size_t>(g_hidden));
          for (int64_t j = 0; j < g_hidden; ++j) inv_smooth[j] = 1.0f / asf[j];
          inv_smooth_src = asf;
        }
        a_scale.resize(static_cast<size_t>(rows));
        // `s_ln[...]` is a design SLOT; h_gamma/h_beta are indexed by
        // site, and layer_norm() converts with `slot - 1`. Passing the
        // slot straight through segfaulted on the last layer.
        add_norm_quant(x, proj, s_ln[1 + 2 * L] - 1, inv_smooth.data(),
                       static_cast<int8_t *>(ffn_up.slot_ptr(0, slot_a)),
                       a_scale.data());
      } else if (fuse_ln_bf16) {
        add_norm_bf16(x, proj, s_ln[1 + 2 * L] - 1,
                     static_cast<uint16_t *>(ffn_up.slot_ptr(0, slot_a)));
      } else {
        add_into(x, proj);
        layer_norm(x, s_ln[1 + 2 * L]);
        std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));
      }
      // T37: when the whole ffn_up -> GELU -> ffn_down chain is int8 and GELU
      // is on the host, the epilogue is fused and `up` is never materialised.
      // Gated FFNs fuse too: SwiGLU narrows 2*inter -> inter inside the same
      // L1-resident row, which is if anything a bigger saving because their
      // ffn_up output is twice as wide.
      FusedNext fn{i8w(as_fd, L),
                   static_cast<int8_t *>(ffn_down.slot_ptr(0, slot_a)),
                   nullptr, g_gated_ffn};
      const bool fuse_ffn = fuse_ffn_epilogue && host_gelu &&
                            ffn_up.info().a_elem_bytes == 1 &&
                            ffn_down.info().a_elem_bytes == 1;
      if (fuse_ffn) {
        a_scale_next.resize(static_cast<size_t>(rows));
        fn.scale = a_scale_next.data();
      }
      // T37-BF16 (tasks/0108): the bf16/bfp16 analogue of `fn`/`fuse_ffn`
      // above -- STEP 1 measured this chain (ffn_up's C-readback+bias, the
      // activation, ffn_down's A-convert) at 17.7-28.3% of wall clock across
      // the shipped catalogue before this fusion existed.
      FusedNextBf16 fn_bf16{
          static_cast<uint16_t *>(ffn_down.slot_ptr(0, slot_a)), g_gated_ffn};
      const bool fuse_ffn_bf16 = fuse_ffn_epilogue && host_gelu &&
                                 ffn_up.info().a_elem_bytes == 2 &&
                                 ffn_down.info().a_elem_bytes == 2;
      gemm(ffn_up, is_fu, x, s_fu[L], b_fu[L], up,
           g_gated_ffn ? 2 * g_ffn : g_ffn, i8w(ws_fu, L), i8w(as_fu, L),
           fuse_ffn ? &fn : nullptr, /*a_ready=*/fuse_ln || fuse_ln_bf16,
           fuse_ffn_bf16 ? &fn_bf16 : nullptr);

      // arch=2: SwiGLU (fc11 * silu(fc12), fused as one [hidden, 2*inter]
      // ffn_up) in place of plain GELU over a [hidden, inter] ffn_up --
      // writes into the separate `gated` buffer (see swiglu_cpu's own
      // comment for why NOT in place). Every arch=0 container has
      // g_gated_ffn == false and takes the untouched branch below.
      if (fuse_ffn) {
        // T37: the activation's output was never written. The ffn_up epilogue
        // already produced ffn_down's int8 operand in the device slot and its
        // row scales, so this dispatch has nothing to convert.
        a_scale.swap(a_scale_next);
        gemm(ffn_down, is_fd, g_gated_ffn ? gated : up, s_fd[L], b_fd[L], down,
             g_hidden, i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
      } else if (fuse_ffn_bf16) {
        // T37-BF16: `up`/`gated` was never written -- the ffn_up epilogue
        // already produced ffn_down's bf16 operand in the device slot.
        gemm(ffn_down, is_fd, g_gated_ffn ? gated : up, s_fd[L], b_fd[L], down,
             g_hidden, i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
      } else if (g_gated_ffn) {
        swiglu_cpu(up, gated);
        gemm(ffn_down, is_fd, gated, s_fd[L], b_fd[L], down, g_hidden,
                 i8w(ws_fd, L), i8w(as_fd, L));
      } else if (false) {
        // T37: `up` was never written. The ffn_up epilogue already produced
        // ffn_down's int8 operand in the device slot and its row scales, so
        // this dispatch has nothing to convert.
        a_scale.swap(a_scale_next);
        gemm(ffn_down, is_fd, up, s_fd[L], b_fd[L], down, g_hidden,
             i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
      } else {
        if (host_gelu)
          gelu_cpu(up);
        else
          eltwise(gelu, up.data(), up.size());
        gemm(ffn_down, is_fd, up, s_fd[L], b_fd[L], down, g_hidden,
                 i8w(ws_fd, L), i8w(as_fd, L));
      }
      // T37 site 2: the same fusion at the layer's second LayerNorm. Its
      // consumer is the NEXT layer's qkv, so the loop's own
      // `memcpy(residual, x)` is what this replaces -- and on the last layer
      // there is no next GEMM, only pooling, so `dst` is null there.
      const bool last = (L + 1 == g_layers);
      if (fuse_ffn_epilogue && host_ln && qkv.info().a_elem_bytes == 1) {
        const float *asq = last ? nullptr : i8w(as_qkv, L + 1);
        if (asq && (inv_smooth.size() != static_cast<size_t>(g_hidden) ||
                    inv_smooth_src != asq)) {
          inv_smooth.resize(static_cast<size_t>(g_hidden));
          for (int64_t j = 0; j < g_hidden; ++j) inv_smooth[j] = 1.0f / asq[j];
          inv_smooth_src = asq;
        }
        if (asq) a_scale.resize(static_cast<size_t>(rows));
        add_norm_quant(x, down, s_ln[2 + 2 * L] - 1,
                       asq ? inv_smooth.data() : nullptr,
                       asq ? static_cast<int8_t *>(qkv.slot_ptr(0, slot_a))
                           : nullptr,
                       asq ? a_scale.data() : nullptr);
        qkv_a_ready = !last;
      } else if (fuse_ffn_epilogue && host_ln && qkv.info().a_elem_bytes == 2) {
        // T37-BF16 (tasks/0108): as the int8 branch above, narrowing to bf16
        // instead of quantising -- no scale to compute.
        add_norm_bf16(x, down, s_ln[2 + 2 * L] - 1,
                      last ? nullptr
                           : static_cast<uint16_t *>(qkv.slot_ptr(0, slot_a)));
        qkv_a_ready = !last;
      } else {
        add_into(x, down);
        layer_norm(x, s_ln[2 + 2 * L]);
      }
    }
    return x;
  }
};

// ===========================================================================
// EmbeddingGemma-300M on the array (arch=1) -- tasks/0074.
// ===========================================================================
//
// WHY A SECOND ENCODER AND NOT A BRANCH IN Encoder::run().
//
// arch=2 (nomic) could be a branch because it IS a BERT block with two things
// swapped: RoPE instead of an absolute position table, and a gated FFN instead
// of a plain one. Gemma is not. Its block is a four-RMSNorm sandwich
// (pre-norm AND post-norm around both sub-layers), its attention carries
// q_norm/k_norm between the projection and RoPE, its RoPE base changes per
// layer, it scales the embedding by sqrt(hidden), and it ends in two post-pool
// Dense heads. Threading all of that through a function that serves five
// shipped models as `if (arch == ...)` would put the shipped models one typo
// away from a silent wrong answer, which is the failure this project keeps
// finding. So this is separate code that happens to reuse the same MACHINERY:
// Pool, the bf16 conversions, npu::Design staging, and the same
// bind/sync/dispatch/bias sequence Encoder::gemm() uses.
//
// The host-only npue::GemmaEncoder (tasks/0064, verified to 1-cos 5.496e-13
// against reference/encoder_gemma.py) is NOT replaced by this. It is the
// discriminating control: the same container geometry, the same tokenizer,
// every GEMM in double precision on the CPU. Any disagreement between the two
// beyond the bf16 floor is a bug in THIS file.
//
// WHAT RUNS WHERE. Four GEMMs per layer go to the array -- 97.7% of the
// model's MACs (tasks/0074 sec 4). Attention's QK^T and A.V stay on the host:
// at head_dim 256 and seq 64 their N is 64, which fails the design's
// `N % (n * n_aie_cols) == 0` outright, and F3 prices the whole of attention
// at 2.3% of MACs here. RMSNorm, RoPE and GeGLU stay on the host on
// tasks/0032's measured precedent that a host eltwise pass beats an NPU
// dispatch at these widths.
struct GemmaNpuEncoder {
  npue::File &model;
  npu::Design &d;
  Pool *pool = nullptr;
  npue::GemmaTokenizer tok;

  // Geometry, read from the container. Nothing here is a literal: this file
  // has no idea that hidden is 768 or that there are 24 layers.
  int64_t hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, inter = 0,
          layers = 0, dense_hidden = 0, qkv_n = 0, swp = 6;
  int64_t q_off = 0, k_off = 0, v_off = 0, kv_w = 0;
  double eps = 1e-6, rope_theta = 0.0, rope_theta_local = 0.0,
         attn_scale = 1.0;

  int64_t seq = 0, batch = 0, rows = 0;
  std::vector<int64_t> tiers;
  std::vector<std::array<size_t, 4>> tier_slots;
  size_t is_qkv = 0, is_ao = 0, is_fu = 0, is_fd = 0;
  size_t slot_a = 0, slot_c = 0;
  // Lanes (tasks/0033's mechanism, applied to this arch): several encoders
  // share the ONE design, each owning its A and C slots, with every NPU
  // interaction under one mutex. The array serialises dispatches anyway, so
  // the lock only makes explicit what the hardware enforces -- what overlaps
  // is one lane's HOST work with another's array work. It matters more here
  // than on any BERT model: the first measurement of this path put the array
  // at 48% of wall clock and the host at 47%, which is as close to the ideal
  // case for overlap as this project has met.
  std::mutex *npu_mu = nullptr;

  std::vector<size_t> s_qkv, s_ao, s_fu, s_fd;
  std::vector<const float *> b_qkv, b_ao, b_fu, b_fd;
  // int8 (tasks/0081). Per-output-channel weight scales and per-input-channel
  // SmoothQuant divisors, one pointer per layer per op, filled by stage_all()
  // and empty on a bf16 container.
  std::vector<const float *> ws_qkv, ws_ao, ws_fu, ws_fd;
  std::vector<const float *> as_qkv, as_ao, as_fu, as_fd;
  // Per-row activation scales for the current GEMM, [rows].
  std::vector<float> a_scale;
  // 1/asmooth for the GEMM being run, rebuilt only when the pointer changes --
  // K floats of setup against 2*rows*K divisions (tasks/0080).
  std::vector<float> inv_smooth;
  const float *inv_smooth_src = nullptr;
  std::vector<float> a_scale_next, inv_smooth_next;
  const float *inv_smooth_next_src = nullptr;
  bool fuse_ffn_epilogue = true;

  struct LayerHost {
    const float *q_norm, *k_norm, *ln_in, *ln_pa, *ln_pf, *ln_pof;
  };
  std::vector<LayerHost> lh;
  const float *w_embed = nullptr, *w_norm = nullptr;
  const float *w_dense2 = nullptr, *w_dense3 = nullptr;

  // Scratch, sized once per batch and reused across layers and calls.
  std::vector<float> x, hbuf, qkvbuf, ctx, proj, upbuf, gatedbuf, down,
      scores, add_mask, cos_g, sin_g, cos_l, sin_l;
  std::vector<int32_t> ids;
  std::vector<uint8_t> mask;

  double t_conv = 0, t_in = 0, t_disp = 0, t_out = 0, t_bias = 0,
         t_norm = 0, t_attn = 0, t_rope = 0, t_geglu = 0, t_tok = 0;
  int n_dispatch = 0;
  void reset_timers() {
    t_conv = t_in = t_disp = t_out = t_bias = 0;
    t_norm = t_attn = t_rope = t_geglu = t_tok = 0;
    n_dispatch = 0;
  }

  GemmaNpuEncoder(npue::File &m, npu::Design &design, Pool &p)
      : model(m), d(design), pool(&p) {
    const std::string arch = m.config_string("arch");
    if (arch != "gemma3_mqa_rope_geglu")
      throw std::runtime_error("GemmaNpuEncoder given arch '" + arch + "'");
    // The container must say it holds PRE-TILED operands. A host-only
    // container carries the same tensor VALUES in the same file under
    // different names and a different layout; reading one as the other is
    // tasks/0022's rel_fro 1.186 all over again.
    const std::string layout = m.config_string("gemm_layout");
    if (layout != "pretiled_bf16")
      throw std::runtime_error(
          "this container's gemm_layout is '" + layout + "', not "
          "'pretiled_bf16' -- it holds host-side row-major F32 operands and "
          "has no tiled weights for the array. Repack it with "
          "tools/pack_npue.py (the NPU layout is now the default).");

    hidden = m.config_int("hidden");
    heads = m.config_int("num_heads");
    kv_heads = m.config_int("num_key_value_heads");
    head_dim = m.config_int("head_dim");
    inter = m.config_int("intermediate");
    layers = m.config_int("num_layers");
    dense_hidden = m.config_int("dense_hidden");
    swp = m.config_int("sliding_window_pattern");
    eps = m.config_double("rms_norm_eps");
    rope_theta = m.config_double("rope_theta");
    rope_theta_local = m.config_double("rope_local_base_freq");
    attn_scale = std::pow(m.config_double("query_pre_attn_scalar"), -0.5);
    qkv_n = m.config_int("qkv_n");
    kv_w = kv_heads * head_dim;

    if (kv_heads != 1)
      throw std::runtime_error(
          "this encoder's attention loop assumes num_key_value_heads == 1 "
          "(EmbeddingGemma-300M); a GQA checkpoint needs the K/V reuse "
          "generalised first");
    if (head_dim * heads != hidden)
      throw std::runtime_error("head_dim * num_heads != hidden");
    if (head_dim % 2)
      throw std::runtime_error("odd head_dim -- RoPE cannot half-split it");
    if (m.config_string("geglu_halves") != "gate|up")
      throw std::runtime_error(
          "unrecognised geglu_halves '" + m.config_string("geglu_halves") +
          "' -- expected 'gate|up'; refusing rather than guessing which half "
          "of the fused ffn_up gets the GELU. tasks/0068 Q2 measured the "
          "swapped variant on the sibling architecture at rel_fro 4.022e+00.");

    // Q/K/V offsets are READ, never derived. `3*hidden` is the BERT answer and
    // it is wrong here twice over: MQA makes K and V narrower than Q, and the
    // operand is zero-padded past them (tasks/0074).
    q_off = 0;
    k_off = hidden;
    v_off = hidden + kv_w;
    if (qkv_n < v_off + kv_w)
      throw std::runtime_error("qkv_n is too small to hold Q|K|V");

    w_embed = m.raw("embed_tokens.weight").as<float>();
    w_norm = m.raw("norm.weight").as<float>();
    w_dense2 = m.raw("dense2.weight").as<float>();
    w_dense3 = m.raw("dense3.weight").as<float>();
    lh.resize(static_cast<size_t>(layers));
    for (int64_t L = 0; L < layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      LayerHost &l = lh[static_cast<size_t>(L)];
      l.q_norm = m.raw(p + "q_norm.weight").as<float>();
      l.k_norm = m.raw(p + "k_norm.weight").as<float>();
      l.ln_in = m.raw(p + "input_layernorm.weight").as<float>();
      l.ln_pa = m.raw(p + "post_attention_layernorm.weight").as<float>();
      l.ln_pf = m.raw(p + "pre_feedforward_layernorm.weight").as<float>();
      l.ln_pof = m.raw(p + "post_feedforward_layernorm.weight").as<float>();
    }
    auto tv = m.raw("tokenizer.gemma_table");
    tok = npue::GemmaTokenizer::from_table_bytes(
        reinterpret_cast<const char *>(tv.data), tv.bytes);
  }

  template <typename F> void par(size_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1 || n < 65536) {
      f(size_t(0), n);
      return;
    }
    pool->run([&](int w, int nw) {
      const size_t chunk = ((n / nw) + 63) & ~size_t(63);
      const size_t lo = std::min(n, chunk * size_t(w));
      const size_t hi = std::min(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }
  // Row-parallel, for the strided passes where a flat byte range would split
  // a row.
  template <typename F> void par_rows(int64_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1) { f(int64_t(0), n); return; }
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n, chunk * w);
      const int64_t hi = std::min<int64_t>(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }

  double lap(double t0, double &bucket) {
    const double t = now_s();
    bucket += t - t0;
    return t;
  }

  size_t stage_all() {
    size_t bytes = 0;
    const bool i8 = d.info().a_elem_bytes == 1;
    auto one = [&](const std::string &name, std::vector<size_t> &slots,
                   std::vector<const float *> &bias,
                   std::vector<const float *> *wsc,
                   std::vector<const float *> *asm_) {
      const std::string &want = d.info().b_layout_hash;
      const std::string &got = model.info(name).layout_hash;
      if (want.empty() || got.empty() || want != got)
        throw std::runtime_error(
            name + ": B layout mismatch -- design wants " +
            (want.empty() ? std::string("(nothing stated)") : want.substr(0, 16)) +
            ", container has " +
            (got.empty() ? std::string("(nothing stated)") : got.substr(0, 16)) +
            ". The bytes would be the right size and the wrong order.");
      auto w = model.raw(name);
      slots.push_back(d.stage(1, w.data, w.bytes));
      // Gemma has no biases anywhere; the packer zero-fills them so this
      // dispatch path stays byte-for-byte the BERT one (tasks/0074 sec 5).
      bias.push_back(model.raw(name + ".bias").as<float>());
      if (i8) {
        // Refuse rather than dereference: an int8 DESIGN with a bf16
        // CONTAINER passes the layout check only if the container is also
        // int8, but a container packed by an older packer would carry i8
        // bytes without the scales that give them meaning.
        wsc->push_back(model.raw(name + ".wscale").as<float>());
        asm_->push_back(model.raw(name + ".asmooth").as<float>());
      }
      bytes += w.bytes;
    };
    for (int64_t L = 0; L < layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      one(p + "qkv", s_qkv, b_qkv, &ws_qkv, &as_qkv);
      one(p + "attn_out", s_ao, b_ao, &ws_ao, &as_ao);
      one(p + "ffn_up", s_fu, b_fu, &ws_fu, &as_fu);
      one(p + "ffn_down", s_fd, b_fd, &ws_fd, &as_fd);
    }
    return bytes;
  }

  int64_t use_tier(int64_t want) {
    if (tiers.empty()) return batch;
    size_t pick = tiers.size() - 1;
    for (size_t i = 0; i < tiers.size(); ++i)
      if (tiers[i] >= want) { pick = i; break; }
    batch = tiers[pick];
    rows = batch * seq;
    is_qkv = tier_slots[pick][0];
    is_ao = tier_slots[pick][1];
    is_fu = tier_slots[pick][2];
    is_fd = tier_slots[pick][3];
    return batch;
  }

  // Identical in shape to Encoder::gemm() -- convert A to bf16 into this
  // pipeline's own slot, bind, sync, dispatch, then read C back with a
  // streaming load and add the (zero) bias.
  // The scale vectors are empty on a bf16 container, so this yields nullptr
  // and gemm()'s own check decides whether that is legal for the design.
  static const float *at(const std::vector<const float *> &v, int64_t L) {
    return L < static_cast<int64_t>(v.size()) ? v[static_cast<size_t>(L)]
                                              : nullptr;
  }

  // T37 (tasks/0082): as Encoder::FusedNext, for arch=1's GeGLU. `gated` is
  // always true here -- Gemma has no un-gated FFN.
  struct FusedNext {
    const float *asmooth;
    int8_t *dst;
    float *scale;
  };

  // T37-BF16 (tasks/0108): as Encoder::FusedNextBf16, for arch=1's GeGLU.
  struct FusedNextBf16 {
    uint16_t *dst;
  };

  // T37-BF16 (tasks/0108): as Encoder::dequant_act_bf16, for arch=1's GeGLU
  // (gelu_pytorch_tanh(gate) * up, NOT the same function as BERT's GELU --
  // see geglu()'s own comment). Bit-identical to the unfused
  // gemm(...)+geglu() pair by construction: the C-readback+bias arm is
  // copied verbatim from this gemm()'s own unfused branches below, and the
  // activation arm is copied verbatim from the int8 fused epilogue's lambda
  // above (itself bit-identical to geglu(), including the scalar tail's
  // double-precision rounding).
  void dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
                        int64_t out_n, const float *bias, uint16_t *dst) {
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      std::vector<float> row(static_cast<size_t>(N));
      for (int64_t r = r0; r < r1; ++r) {
        float *v = row.data();
        int64_t j = 0;
#if defined(__AVX2__)
        if (c_bytes == 2) {
          const uint16_t *cr = static_cast<const uint16_t *>(c) + r * N;
          for (; j + 16 <= N; j += 16) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            __m256i lo = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
            __m256i hi = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
            _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(lo),
                                                  _mm256_loadu_ps(bias + j)));
            _mm256_storeu_ps(v + j + 8, _mm256_add_ps(_mm256_castsi256_ps(hi),
                                                  _mm256_loadu_ps(bias + j + 8)));
          }
        } else {
          const float *cr = static_cast<const float *>(c) + r * N;
          for (; j + 8 <= N; j += 8) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(raw),
                                                  _mm256_loadu_ps(bias + j)));
          }
        }
#endif
        for (; j < N; ++j) {
          const float cf = c_bytes == 2
              ? from_bf16(static_cast<const uint16_t *>(c)[r * N + j])
              : static_cast<const float *>(c)[r * N + j];
          v[j] = cf + bias[j];
        }
        // GeGLU, in place, narrowing to out_n -- COPIED VERBATIM from the
        // int8 fused epilogue's lambda / geglu() above.
        const int64_t inter = N / 2;
        const float *u = v + inter;
        int64_t k = 0;
#if defined(__AVX2__)
        const __m256 c_half = _mm256_set1_ps(0.5f);
        const __m256 c_one = _mm256_set1_ps(1.0f);
        const __m256 c_sq = _mm256_set1_ps(0.7978845608028654f);
        const __m256 c_k = _mm256_set1_ps(0.044715f);
        const __m256 c_2log2e = _mm256_set1_ps(2.885390081777927f);
        const __m256 c_lim = _mm256_set1_ps(15.0f);
        for (; k + 8 <= inter; k += 8) {
          __m256 xv = _mm256_loadu_ps(v + k);
          __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
          __m256 y = _mm256_mul_ps(c_sq, _mm256_fmadd_ps(c_k, x3, xv));
          y = _mm256_min_ps(
              _mm256_max_ps(y, _mm256_sub_ps(_mm256_setzero_ps(), c_lim)),
              c_lim);
          __m256 e = Encoder::exp2_avx2(_mm256_mul_ps(y, c_2log2e));
          __m256 th = _mm256_div_ps(_mm256_sub_ps(e, c_one),
                                    _mm256_add_ps(e, c_one));
          __m256 act = _mm256_mul_ps(_mm256_mul_ps(c_half, xv),
                                     _mm256_add_ps(c_one, th));
          _mm256_storeu_ps(v + k, _mm256_mul_ps(act, _mm256_loadu_ps(u + k)));
        }
#endif
        for (; k < inter; ++k) {
          const double xv = v[k];
          const double y =
              0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
          const float act =
              static_cast<float>(0.5 * xv * (1.0 + std::tanh(y)));
          v[k] = static_cast<float>(static_cast<double>(act) *
                                    static_cast<double>(u[k]));
        }
        bf16_fill(dst + r * out_n, v, static_cast<size_t>(out_n));
      }
    });
  }

  void gemm(size_t islot, const float *a, size_t a_len, size_t wslot,
            const float *bias, std::vector<float> &out, int64_t N,
            const float *wscale = nullptr, const float *asmooth = nullptr,
            FusedNext *fuse = nullptr, bool a_ready = false,
            FusedNextBf16 *fuse_bf16 = nullptr) {
    const bool i8 = d.info().a_elem_bytes == 1;
    if (i8 && (wscale == nullptr || asmooth == nullptr))
      throw std::runtime_error(
          "int8 design but this encoder has no quantisation scales -- the "
          "container is bf16, or it was packed before tools/pack_npue.py "
          "--int8 supported arch=1");
    double t0 = now_s();
    if (i8 && a_ready) {
      // A is already in the device slot, written by the previous GEMM's fused
      // epilogue, and a_scale already holds its row scales.
    } else if (i8) {
      const int64_t K = static_cast<int64_t>(a_len) / rows;
      a_scale.resize(static_cast<size_t>(rows));
      if (inv_smooth.size() != static_cast<size_t>(K) ||
          inv_smooth_src != asmooth) {
        inv_smooth.resize(static_cast<size_t>(K));
        for (int64_t j = 0; j < K; ++j) inv_smooth[j] = 1.0f / asmooth[j];
        inv_smooth_src = asmooth;
      }
      quantise_a_int8(a, rows, K, inv_smooth.data(),
                      static_cast<int8_t *>(d.slot_ptr(0, slot_a)),
                      a_scale.data(),
                      [&](int64_t n, auto f) { par_rows(n, f); });
      t0 = lap(t0, t_conv);
    } else if (a_ready) {
      // T37-BF16 (tasks/0108): A was written straight into the device slot,
      // already narrowed to bf16, by the PREVIOUS gemm's fused epilogue.
    } else {
    auto *abuf = static_cast<uint16_t *>(d.slot_ptr(0, slot_a));
    par(a_len, [&](size_t lo, size_t hi) {
      bf16_fill(abuf + lo, a + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    }
    const float *c;
    {
      std::unique_lock<std::mutex> lk;
      if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
      d.bind_instr(islot);
      d.bind(0, slot_a);
      d.bind(1, wslot);
      d.bind(2, slot_c);
      d.sync_to_device(0, a_len * d.info().a_elem_bytes);
      t0 = lap(t0, t_in);
      d.dispatch_only();
      t0 = lap(t0, t_disp);
      const size_t cb = d.info().c_elem_bytes;
      d.sync_from_device(2, static_cast<size_t>(rows) * N * cb);
      t0 = lap(t0, t_out);
      c = static_cast<const float *>(d.slot_ptr(2, slot_c));
    }
    if (i8 && fuse) {
      // FUSED GeGLU EPILOGUE (T37): dequantise, gate, and quantise ffn_down's
      // operand in one L1-resident pass. `out` is never written.
      const int64_t out_n = N / 2;
      if (inv_smooth_next.size() != static_cast<size_t>(out_n) ||
          inv_smooth_next_src != fuse->asmooth) {
        inv_smooth_next.resize(static_cast<size_t>(out_n));
        for (int64_t j = 0; j < out_n; ++j)
          inv_smooth_next[j] = 1.0f / fuse->asmooth[j];
        inv_smooth_next_src = fuse->asmooth;
      }
      dequant_act_quant(
          c, d.info().c_elem_bytes, rows, N, out_n,
          [](float *v, int64_t n) {
            // Identical intrinsics to geglu(), including its +-15 clamp before
            // exp2 -- a scalar rewrite of the same algebra is a different
            // number (tasks/0082 section 2).
            const int64_t inter = n / 2;
            const float *u = v + inter;
            int64_t j = 0;
#if defined(__AVX2__)
            const __m256 c_half = _mm256_set1_ps(0.5f);
            const __m256 c_one = _mm256_set1_ps(1.0f);
            const __m256 c_sq = _mm256_set1_ps(0.7978845608028654f);
            const __m256 c_k = _mm256_set1_ps(0.044715f);
            const __m256 c_2log2e = _mm256_set1_ps(2.885390081777927f);
            const __m256 c_lim = _mm256_set1_ps(15.0f);
            for (; j + 8 <= inter; j += 8) {
              __m256 xv = _mm256_loadu_ps(v + j);
              __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
              __m256 y = _mm256_mul_ps(c_sq, _mm256_fmadd_ps(c_k, x3, xv));
              y = _mm256_min_ps(
                  _mm256_max_ps(y, _mm256_sub_ps(_mm256_setzero_ps(), c_lim)),
                  c_lim);
              __m256 e = Encoder::exp2_avx2(_mm256_mul_ps(y, c_2log2e));
              __m256 th = _mm256_div_ps(_mm256_sub_ps(e, c_one),
                                        _mm256_add_ps(e, c_one));
              __m256 act = _mm256_mul_ps(_mm256_mul_ps(c_half, xv),
                                         _mm256_add_ps(c_one, th));
              _mm256_storeu_ps(v + j,
                               _mm256_mul_ps(act, _mm256_loadu_ps(u + j)));
            }
#endif
            for (; j < inter; ++j) {
              const double xv = v[j];
              const double y =
                  0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
              const float act =
                  static_cast<float>(0.5 * xv * (1.0 + std::tanh(y)));
              v[j] = static_cast<float>(static_cast<double>(act) *
                                        static_cast<double>(u[j]));
            }
          },
          a_scale.data(), wscale, bias, inv_smooth_next.data(), fuse->dst,
          fuse->scale, [&](int64_t n, auto f) { par_rows(n, f); });
    } else if (i8) {
      // Same rank-1 dequantisation the BERT encoder uses, from the same
      // helper -- the transport width comes from the design, so a narrowed-C
      // int8 set (tasks/0080) needs nothing extra here.
      dequantise_c(c, d.info().c_elem_bytes, rows, N, a_scale.data(), wscale,
                   bias, out.data(),
                   [&](int64_t n, auto f) { par_rows(n, f); });
    } else if (fuse_bf16) {
      // T37-BF16 (tasks/0108): `out` is deliberately never written -- see
      // dequant_act_bf16 above.
      dequant_act_bf16(c, d.info().c_elem_bytes, N, N / 2, bias,
                       fuse_bf16->dst);
    } else if (d.info().c_elem_bytes == 2) {
      const uint16_t *cb16 = reinterpret_cast<const uint16_t *>(c);
      par_rows(rows, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
          const uint16_t *cr = cb16 + r * N;
          float *o = out.data() + r * N;
          for (int64_t j = 0; j < N; ++j) o[j] = from_bf16(cr[j]) + bias[j];
        }
      });
    } else {
      par_rows(rows, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
          const float *cr = c + r * N;
          float *o = out.data() + r * N;
          int64_t j = 0;
#if defined(__AVX2__)
          // Streaming loads: C is an XRT write-combined host bo and ordinary
          // loads from it stall per line (tasks/0024). N is a multiple of 48
          // here, so the 8-wide tail is handled by the scalar loop.
          for (; j + 8 <= N; j += 8)
            _mm256_storeu_ps(o + j,
                             _mm256_add_ps(_mm256_castsi256_ps(
                                               _mm256_stream_load_si256(
                                                   reinterpret_cast<const __m256i *>(cr + j))),
                                           _mm256_loadu_ps(bias + j)));
#endif
          for (; j < N; ++j) o[j] = cr[j] + bias[j];
        }
      });
    }
    lap(t0, t_bias);
    ++n_dispatch;
  }

  // Gemma3RMSNorm over `dim` contiguous elements per row, with independent
  // input and output row strides so it can work on a slice of the fused qkv
  // buffer in place. `out = x * rsqrt(mean(x^2) + eps) * (1 + w)` -- the
  // `1 +` is Gemma's and omitting it is the single easiest way to produce a
  // plausible wrong answer here (gemma_kernels.hpp's own warning).
  //
  // fp32 accumulation, not the double reduction npue::rms_norm_cpu uses. That
  // function exists to match reference/encoder_gemma.py bit-for-bit and is
  // still what the host-only control runs; this path already rounds every
  // GEMM through bf16, so a double-precision reduction here would buy nothing
  // measurable and costs a factor on 96 norm sites per encode. The end-to-end
  // check against the control is what decides whether that is true.
  void rms_norm(const float *xin, int64_t in_stride, float *xout,
                int64_t out_stride, int64_t n_rows, int64_t dim,
                const float *w) {
    const double t0 = now_s();
    const float e = static_cast<float>(eps);
    par_rows(n_rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const float *in = xin + r * in_stride;
        float *o = xout + r * out_stride;
        int64_t j = 0;
        float ss;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; j + 8 <= dim; j += 8) {
          __m256 v = _mm256_loadu_ps(in + j);
          acc = _mm256_fmadd_ps(v, v, acc);
        }
        ss = hsum256(acc);
#else
        ss = 0.f;
#endif
        for (; j < dim; ++j) ss += in[j] * in[j];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(dim) + e);
        j = 0;
#if defined(__AVX2__)
        const __m256 iv = _mm256_set1_ps(inv);
        const __m256 one = _mm256_set1_ps(1.0f);
        for (; j + 8 <= dim; j += 8)
          _mm256_storeu_ps(o + j,
                           _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(in + j), iv),
                                         _mm256_add_ps(one, _mm256_loadu_ps(w + j))));
#endif
        for (; j < dim; ++j) o[j] = in[j] * inv * (1.0f + w[j]);
      }
    });
    t_norm += now_s() - t0;
  }

  // RoPE on Q (every head) and K (the single KV head) in place, inside the
  // fused qkv buffer. Position is `row % seq`, which holds because rows are
  // laid out [batch][seq]. NeoX convention (concat(freqs,freqs), rotate-half),
  // matching gemma_rope_tables().
  void apply_rope(std::vector<float> &qkv, const float *cs_t, const float *sn_t) {
    const double t0 = now_s();
    const int64_t half = head_dim / 2;
    float *__restrict p = qkv.data();
    auto rot = [half](float *v, const float *cs, const float *sn) {
      int64_t dd = 0;
#if defined(__AVX2__)
      for (; dd + 8 <= half; dd += 8) {
        __m256 x1 = _mm256_loadu_ps(v + dd);
        __m256 x2 = _mm256_loadu_ps(v + dd + half);
        __m256 c = _mm256_loadu_ps(cs + dd);
        __m256 s = _mm256_loadu_ps(sn + dd);
        _mm256_storeu_ps(v + dd,
                         _mm256_sub_ps(_mm256_mul_ps(x1, c), _mm256_mul_ps(x2, s)));
        _mm256_storeu_ps(v + dd + half,
                         _mm256_add_ps(_mm256_mul_ps(x2, c), _mm256_mul_ps(x1, s)));
      }
#endif
      for (; dd < half; ++dd) {
        const float a = v[dd], b = v[dd + half];
        v[dd] = a * cs[dd] - b * sn[dd];
        v[dd + half] = b * cs[dd] + a * sn[dd];
      }
    };
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t s = r % seq;
        const float *cs = cs_t + s * head_dim;
        const float *sn = sn_t + s * head_dim;
        float *base = p + r * qkv_n;
        for (int64_t hh = 0; hh < heads; ++hh)
          rot(base + q_off + hh * head_dim, cs, sn);
        rot(base + k_off, cs, sn);           // kv_heads == 1
      }
    });
    t_rope += now_s() - t0;
  }

  // out = gelu_pytorch_tanh(gate) * up, over the fused [gate | up] ffn_up
  // buffer. Gemma's activation is the tanh approximation, NOT the exact-erf
  // GELU the BERT path uses -- a different function, kept deliberately
  // separate (gemma_kernels.hpp).
  void geglu(const std::vector<float> &fused, std::vector<float> &out) {
    const double t0 = now_s();
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const float *g = fused.data() + r * 2 * inter;
        const float *u = g + inter;
        float *o = out.data() + r * inter;
        int64_t j = 0;
#if defined(__AVX2__)
        const __m256 c_half = _mm256_set1_ps(0.5f);
        const __m256 c_one = _mm256_set1_ps(1.0f);
        const __m256 c_sq = _mm256_set1_ps(0.7978845608028654f);  // sqrt(2/pi)
        const __m256 c_k = _mm256_set1_ps(0.044715f);
        // 2*log2(e): tanh(y) = (2^(2y*log2e) - 1) / (2^(2y*log2e) + 1)
        const __m256 c_2log2e = _mm256_set1_ps(2.885390081777927f);
        // Clamp before exp2 so a large activation saturates tanh instead of
        // overflowing to inf and producing (inf-1)/(inf+1) = NaN. |y| >= 15
        // is tanh = +-1 to well inside fp32 already.
        const __m256 c_lim = _mm256_set1_ps(15.0f);
        for (; j + 8 <= inter; j += 8) {
          __m256 xv = _mm256_loadu_ps(g + j);
          __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
          __m256 y = _mm256_mul_ps(c_sq, _mm256_fmadd_ps(c_k, x3, xv));
          y = _mm256_min_ps(_mm256_max_ps(y, _mm256_sub_ps(_mm256_setzero_ps(), c_lim)),
                            c_lim);
          __m256 e = Encoder::exp2_avx2(_mm256_mul_ps(y, c_2log2e));
          __m256 th = _mm256_div_ps(_mm256_sub_ps(e, c_one),
                                    _mm256_add_ps(e, c_one));
          __m256 act = _mm256_mul_ps(_mm256_mul_ps(c_half, xv),
                                     _mm256_add_ps(c_one, th));
          _mm256_storeu_ps(o + j, _mm256_mul_ps(act, _mm256_loadu_ps(u + j)));
        }
#endif
        // Scalar tail, in the reference's own two-stage rounding (round `act`
        // to fp32, THEN promote and multiply by `up`). It never runs at this
        // model's intermediate width -- 1152 is a multiple of 8 -- and is kept
        // matching the reference rather than matching the vector body above,
        // so a future width with a tail lands on the more accurate form.
        for (; j < inter; ++j) {
          const double xv = g[j];
          const double y = 0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
          const float act = static_cast<float>(0.5 * xv * (1.0 + std::tanh(y)));
          o[j] = static_cast<float>(static_cast<double>(act) *
                                    static_cast<double>(u[j]));
        }
      }
    });
    t_geglu += now_s() - t0;
  }

  // MQA attention on the host. Every one of the `heads` query heads attends to
  // the SAME single K/V head -- mathematically identical to repeat_kv() but
  // without materialising the repeat.
  void attention(const std::vector<float> &qkv, std::vector<float> &out) {
    const double t0 = now_s();
    const int64_t pairs = batch * heads;
    par_rows(pairs, [&](int64_t p0, int64_t p1) {
      std::vector<float> row(static_cast<size_t>(seq));
      for (int64_t pi = p0; pi < p1; ++pi) {
        const int64_t b = pi / heads, hh = pi % heads;
        const float *base = qkv.data() + b * seq * qkv_n;
        const float *mk = add_mask.data() + b * seq;
        for (int64_t i = 0; i < seq; ++i) {
          const float *qi = base + i * qkv_n + q_off + hh * head_dim;
          float mx = -3.4e38f;
          for (int64_t j = 0; j < seq; ++j) {
            const float *kj = base + j * qkv_n + k_off;
            int64_t dd = 0;
            float acc;
#if defined(__AVX2__)
            __m256 a = _mm256_setzero_ps();
            for (; dd + 8 <= head_dim; dd += 8)
              a = _mm256_fmadd_ps(_mm256_loadu_ps(qi + dd),
                                  _mm256_loadu_ps(kj + dd), a);
            acc = hsum256(a);
#else
            acc = 0.f;
#endif
            for (; dd < head_dim; ++dd) acc += qi[dd] * kj[dd];
            const float sv =
                acc * static_cast<float>(attn_scale) + mk[j];
            row[static_cast<size_t>(j)] = sv;
            mx = std::max(mx, sv);
          }
          float sum = 0.f;
          for (int64_t j = 0; j < seq; ++j) {
            const float e = std::exp(row[static_cast<size_t>(j)] - mx);
            row[static_cast<size_t>(j)] = e;
            sum += e;
          }
          const float inv = 1.0f / sum;
          float *o = out.data() + (b * seq + i) * hidden + hh * head_dim;
          std::memset(o, 0, sizeof(float) * static_cast<size_t>(head_dim));
          for (int64_t j = 0; j < seq; ++j) {
            const float w = row[static_cast<size_t>(j)] * inv;
            const float *vj = base + j * qkv_n + v_off;
            int64_t dd = 0;
#if defined(__AVX2__)
            const __m256 wv = _mm256_set1_ps(w);
            for (; dd + 8 <= head_dim; dd += 8)
              _mm256_storeu_ps(o + dd,
                               _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj + dd),
                                               _mm256_loadu_ps(o + dd)));
#endif
            for (; dd < head_dim; ++dd) o[dd] += w * vj[dd];
          }
        }
      }
    });
    t_attn += now_s() - t0;
  }

  // A plain threaded fp32 host GEMM, for the two post-pool Dense heads only.
  // They run once per SEQUENCE, not once per token (tasks/0074 sec 4).
  void gemm_host(const float *a, int64_t M, int64_t K, const float *b,
                 int64_t N, float *c) const {
    par_rows(M, [&](int64_t r0, int64_t r1) {
      for (int64_t i = r0; i < r1; ++i) {
        const float *ar = a + i * K;
        float *cr = c + i * N;
        std::memset(cr, 0, sizeof(float) * static_cast<size_t>(N));
        for (int64_t k = 0; k < K; ++k) {
          const float av = ar[k];
          if (av == 0.f) continue;
          const float *br = b + k * N;
          int64_t j = 0;
#if defined(__AVX2__)
          const __m256 avv = _mm256_set1_ps(av);
          for (; j + 8 <= N; j += 8)
            _mm256_storeu_ps(cr + j, _mm256_fmadd_ps(avv, _mm256_loadu_ps(br + j),
                                                     _mm256_loadu_ps(cr + j)));
#endif
          for (; j < N; ++j) cr[j] += av * br[j];
        }
      }
    });
  }

  void ensure_tables() {
    if (!cos_g.empty()) return;
    cos_g.resize(static_cast<size_t>(seq * head_dim));
    sin_g.resize(cos_g.size());
    cos_l.resize(cos_g.size());
    sin_l.resize(cos_g.size());
    npue::gemma_rope_tables(seq, head_dim, rope_theta, cos_g.data(), sin_g.data());
    npue::gemma_rope_tables(seq, head_dim, rope_theta_local, cos_l.data(),
                            sin_l.data());
  }

  // Encode `texts` (padded/tiled by the caller to exactly `batch` entries).
  // Returns [batch][hidden], L2-normalized.
  // `index_base` and `n_real` exist only so a truncation error can name the
  // CALLER's input. This function is handed a group that the caller has
  // padded up to the tier by repeating its last real text, so rows at
  // `b >= n_real` are duplicates whose index does not exist upstream --
  // checking them would report a row number the caller cannot look up. A
  // duplicate that truncates is a copy of a real row that also truncates, and
  // the real one is checked first, so nothing escapes by being skipped here.
  // `tokens`, when given, accumulates the REAL texts' token counts -- the same
  // thing the BERT path's chunk() reports and the same field `usage.
  // prompt_tokens` needs (tasks/0115). Padding repeats are excluded, which is
  // what `n_real` already distinguishes for the truncation check.
  std::vector<float> encode_batch(const std::vector<std::string> &texts,
                                  const std::string &prefix,
                                  size_t index_base = 0,
                                  size_t n_real = static_cast<size_t>(-1),
                                  int64_t *tokens = nullptr) {
    if (static_cast<int64_t>(texts.size()) != batch)
      throw std::runtime_error("encode_batch given " +
                               std::to_string(texts.size()) +
                               " texts, tier is " + std::to_string(batch));
    ensure_tables();
    double t0 = now_s();
    ids.assign(static_cast<size_t>(rows), 0);
    mask.assign(static_cast<size_t>(rows), 0);
    for (int64_t b = 0; b < batch; ++b) {
      const npue::GemmaEncoded en =
          tok.encode(texts[static_cast<size_t>(b)], static_cast<int>(seq), prefix);
      if (static_cast<size_t>(b) < n_real) {
        check_truncation(en.truncated, en.n_tokens_full,
                         index_base + static_cast<size_t>(b), seq);
        if (tokens) *tokens += en.n_tokens;
      }
      for (int64_t s = 0; s < seq; ++s) {
        ids[static_cast<size_t>(b * seq + s)] = en.input_ids[static_cast<size_t>(s)];
        mask[static_cast<size_t>(b * seq + s)] =
            static_cast<uint8_t>(en.attention_mask[static_cast<size_t>(s)]);
      }
    }
    t_tok += now_s() - t0;

    x.assign(static_cast<size_t>(rows * hidden), 0.f);
    hbuf.resize(x.size());
    qkvbuf.resize(static_cast<size_t>(rows * qkv_n));
    ctx.resize(x.size());
    proj.resize(x.size());
    upbuf.resize(static_cast<size_t>(rows * 2 * inter));
    gatedbuf.resize(static_cast<size_t>(rows * inter));
    down.resize(x.size());
    add_mask.resize(static_cast<size_t>(rows));

    const float MASK_FILL = -3.4028235e38f;
    for (int64_t r = 0; r < rows; ++r)
      add_mask[static_cast<size_t>(r)] = mask[static_cast<size_t>(r)] ? 0.f : MASK_FILL;

    // embed: x = W[id] * sqrt(hidden)
    const float escale = static_cast<float>(std::sqrt(static_cast<double>(hidden)));
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const float *wv = w_embed + static_cast<size_t>(ids[static_cast<size_t>(r)]) * hidden;
        float *dst = x.data() + r * hidden;
        for (int64_t c = 0; c < hidden; ++c) dst[c] = wv[c] * escale;
      }
    });

    for (int64_t L = 0; L < layers; ++L) {
      const LayerHost &l = lh[static_cast<size_t>(L)];
      const bool full = npue::gemma_is_full_attention_layer(L, swp);
      const float *cs_t = full ? cos_g.data() : cos_l.data();
      const float *sn_t = full ? sin_g.data() : sin_l.data();

      rms_norm(x.data(), hidden, hbuf.data(), hidden, rows, hidden, l.ln_in);
      gemm(is_qkv, hbuf.data(), hbuf.size(), s_qkv[L], b_qkv[L], qkvbuf,
           qkv_n, at(ws_qkv, L), at(as_qkv, L));

      // q_norm / k_norm: RMSNorm over head_dim, PER HEAD, strictly between the
      // projection and RoPE. Each (row, head) slice of head_dim floats is
      // contiguous inside the fused buffer even though consecutive rows are
      // qkv_n apart, so a strided call needs no repacking.
      for (int64_t hh = 0; hh < heads; ++hh)
        rms_norm(qkvbuf.data() + q_off + hh * head_dim, qkv_n,
                 qkvbuf.data() + q_off + hh * head_dim, qkv_n, rows, head_dim,
                 l.q_norm);
      rms_norm(qkvbuf.data() + k_off, qkv_n, qkvbuf.data() + k_off, qkv_n,
               rows, head_dim, l.k_norm);

      apply_rope(qkvbuf, cs_t, sn_t);
      attention(qkvbuf, ctx);

      gemm(is_ao, ctx.data(), ctx.size(), s_ao[L], b_ao[L], proj,
           hidden, at(ws_ao, L), at(as_ao, L));
      rms_norm(proj.data(), hidden, proj.data(), hidden, rows, hidden, l.ln_pa);
      par(x.size(), [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) x[i] += proj[i];
      });

      rms_norm(x.data(), hidden, hbuf.data(), hidden, rows, hidden, l.ln_pf);
      const bool fuse_ffn = fuse_ffn_epilogue &&
                            d.info().a_elem_bytes == 1 && at(as_fd, L);
      FusedNext fn{at(as_fd, L), static_cast<int8_t *>(d.slot_ptr(0, slot_a)),
                   nullptr};
      if (fuse_ffn) {
        a_scale_next.resize(static_cast<size_t>(rows));
        fn.scale = a_scale_next.data();
      }
      // T37-BF16 (tasks/0108): the bf16/bfp16 analogue of `fn`/`fuse_ffn`
      // above -- no quantisation scale on this path, so no `at(as_fd, L)`
      // gate is needed.
      const bool fuse_ffn_bf16 = fuse_ffn_epilogue && d.info().a_elem_bytes == 2;
      FusedNextBf16 fn_bf16{static_cast<uint16_t *>(d.slot_ptr(0, slot_a))};
      gemm(is_fu, hbuf.data(), hbuf.size(), s_fu[L], b_fu[L], upbuf,
           2 * inter, at(ws_fu, L), at(as_fu, L), fuse_ffn ? &fn : nullptr,
           /*a_ready=*/false, fuse_ffn_bf16 ? &fn_bf16 : nullptr);
      if (fuse_ffn || fuse_ffn_bf16) {
        if (fuse_ffn) a_scale.swap(a_scale_next);
      } else {
        geglu(upbuf, gatedbuf);
      }
      gemm(is_fd, gatedbuf.data(), gatedbuf.size(), s_fd[L], b_fd[L], down,
           hidden, at(ws_fd, L), at(as_fd, L), nullptr,
           /*a_ready=*/fuse_ffn || fuse_ffn_bf16);
      rms_norm(down.data(), hidden, down.data(), hidden, rows, hidden, l.ln_pof);
      par(x.size(), [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) x[i] += down[i];
      });
    }

    rms_norm(x.data(), hidden, x.data(), hidden, rows, hidden, w_norm);

    // masked mean pool, include_prompt=true
    std::vector<float> pooled(static_cast<size_t>(batch * hidden), 0.f);
    for (int64_t b = 0; b < batch; ++b) {
      double denom = 0.0;
      float *o = pooled.data() + b * hidden;
      for (int64_t s = 0; s < seq; ++s) {
        if (!mask[static_cast<size_t>(b * seq + s)]) continue;
        denom += 1.0;
        const float *row = x.data() + (b * seq + s) * hidden;
        for (int64_t c = 0; c < hidden; ++c) o[c] += row[c];
      }
      const float inv = static_cast<float>(1.0 / std::max(denom, 1e-9));
      for (int64_t c = 0; c < hidden; ++c) o[c] *= inv;
    }

    std::vector<float> d2(static_cast<size_t>(batch * dense_hidden));
    gemm_host(pooled.data(), batch, hidden, w_dense2, dense_hidden, d2.data());
    std::vector<float> out(static_cast<size_t>(batch * hidden));
    gemm_host(d2.data(), batch, dense_hidden, w_dense3, hidden, out.data());

    for (int64_t b = 0; b < batch; ++b) {
      float *o = out.data() + b * hidden;
      double nrm = 0.0;
      for (int64_t c = 0; c < hidden; ++c) nrm += static_cast<double>(o[c]) * o[c];
      const float inv = static_cast<float>(1.0 / std::max(std::sqrt(nrm), 1e-12));
      for (int64_t c = 0; c < hidden; ++c) o[c] *= inv;
    }
    return out;
  }
};

}  // namespace

namespace {

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

// Where the model and design directories live, when nobody says.
//
// Two layouts must both work: an extracted release (exe beside models/ and
// gemm_rtp/) and the source tree (exe in runtime/build/). Probing for the
// directories rather than assuming a depth means neither is privileged, and a
// wrong guess reports what it looked for instead of failing later on a
// confusing missing-file error.
std::string default_root(const char *argv0) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path start = fs::absolute(fs::path(argv0), ec).parent_path();

  // THE EXECUTABLE'S OWN DIRECTORY WINS, whenever it holds anything of ours.
  //
  // Walking up before checking it was a bug, and a quiet one: a release
  // staged or unzipped INSIDE the source tree (dist\npuembeddings-0.2.0\)
  // climbed past its own directory, found the repository's models/ and
  // runtime/, and served the repo's four containers while claiming to be the
  // release. Everything worked and everything was wrong -- which is the
  // failure shape this project keeps meeting. A self-contained directory is
  // self-contained; the search only starts when there is nothing here.
  auto has_design = [&](const fs::path &d) {
    if (fs::exists(d / "gemm_rtp", ec)) return true;
    // Several widths: one design set per subdirectory.
    for (fs::directory_iterator it(d, ec), end; !ec && it != end;
         it.increment(ec))
      if (it->is_directory(ec) && fs::exists(it->path() / "gemm_rtp", ec))
        return true;
    return false;
  };
  if (has_design(start) || fs::exists(start / "models", ec))
    return start.string();

  // Nothing here, so this is a build directory (runtime\build\). Now search
  // upwards -- and the two searches must each run to completion before the
  // other starts, never interleaved a level at a time. The source tree is
  // recognised by BOTH models/ and runtime/, because a design alone would
  // stop at runtime/, which carries the design sets but not the models. I
  // wrote exactly that bug while fixing this function: checking both
  // conditions at each level made runtime/ win over the repository root.
  auto walk = [&](auto &&match) -> std::string {
    fs::path dir = start.parent_path();
    for (int up = 0; up < 5 && !dir.empty(); ++up) {
      if (match(dir)) return dir.string();
      const fs::path next = dir.parent_path();
      if (next == dir) break;               // hit the drive root
      dir = next;
    }
    return "";
  };

  const std::string src = walk([&](const fs::path &d) {
    return fs::exists(d / "models", ec) && fs::exists(d / "runtime", ec);
  });
  if (!src.empty()) return src;

  const std::string rel = walk(has_design);
  if (!rel.empty()) return rel;
  return "..";
}

// Does this design set serve THIS model? Every op's (K, N) must match what the
// model's geometry implies -- not merely `hidden` appearing as some "K".
//
// The old predicate asked only the latter, and its own comment named precisely
// the danger it was failing to catch: "a design built for another width has the
// same filenames and loads fine -- it would simply compute the wrong thing."
// That was sound only because every model shipped so far has ffn == 4*hidden,
// which makes `hidden` determine all four shapes. It stops being sound the
// moment two models share a K set and differ in an N.
//
// nomic-embed-text-v1.5 is the first: its K set {768, 3072} is IDENTICAL to
// bge-base's, while its gated ffn_up is N=6144 against bge-base's N=3072. The
// old check accepts bge-base's design for nomic, and the runtime then
// dispatches a stream built for HALF the output width -- no error, no warning,
// the gate half silently lost. tasks/0069, thread T31.
//
// Matched against the `streams` array, which every design.json has carried
// since 0032, so this works unchanged on design sets exported long before the
// geometry keys existed -- no re-export needed to close the hole.
bool design_fits(const std::string &design_dir, int64_t hidden,
                 int64_t intermediate, bool gated_ffn, int64_t qkv_n = 0,
                 const std::string &want_layout = "",
                 const std::string &want_datapath = "") {
  if (hidden <= 0 || intermediate <= 0) return false;
  std::ifstream f(design_dir + "/gemm_rtp/design.json");
  if (!f) return false;
  std::stringstream b;
  b << f.rdbuf();
  const std::string js = b.str();
  const std::vector<StreamEntry> streams = parse_streams(js);
  if (streams.empty()) return false;

  // GEOMETRY IS NOT ENOUGH ONCE THERE IS MORE THAN ONE DATAPATH.
  // tasks/0080: an int8 container and a bf16 container of the same model have
  // identical (op, K, N) on every stream, so this function accepted a bf16
  // design for an int8 model and the encode died at stage time on the layout
  // hash. Failing closed, but selecting wrongly. The container knows its own
  // B layout -- pass it, and the choice becomes a fact about the DATA rather
  // than about which directory sorts first. Empty means "caller did not say",
  // which keeps every pre-0080 call site behaving exactly as before.
  if (!want_layout.empty()) {
    const size_t k = js.find("\"b_layout_hash\"");
    if (k == std::string::npos) return false;
    const size_t q1 = js.find('"', js.find(':', k) + 1);
    if (q1 == std::string::npos) return false;
    const size_t q2 = js.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    if (js.substr(q1 + 1, q2 - q1 - 1) != want_layout) return false;
  }

  // THE MMAC DATAPATH (tasks/0104, T23). bfp16 is a SECOND thing want_layout
  // above cannot catch: it changes MMAC precision, not B's tiling, so a
  // bfp16 design and a plain-bf16 design at the SAME geometry carry the
  // SAME b_layout_hash (gemm_b_layout() only ever sees dtype="BF16"). Without
  // this check, a model adopted for one datapath and NOT the other -- exactly
  // bge-small, which shares MiniLM's hidden-384 geometry and FAILED the
  // bfp16 MTEB gate at -0.5010 (tasks/0103) -- would have two directories
  // fit it, and pick_artifacts()'s alphabetical tie-break would decide which
  // datapath it runs, silently. want_datapath is "bf16" or "bfp16"; empty
  // means "caller did not say" (every call site before this field existed,
  // and any explicit --artifacts override, which still wins over this
  // function entirely).
  if (!want_datapath.empty()) {
    const size_t k = js.find("\"emulate_bfp16\"");
    bool is_bfp16 = false;
    if (k != std::string::npos) {
      size_t p = js.find(':', k) + 1;
      while (p < js.size() && (js[p] == ' ' || js[p] == '\n' || js[p] == '\t'))
        ++p;
      is_bfp16 = js.compare(p, 4, "true") == 0;
    }
    if ((is_bfp16 ? "bfp16" : "bf16") != want_datapath) return false;
  }

  // tasks/0074: qkv's width was `3 * hidden`, which is true exactly when
  // num_key_value_heads == num_attention_heads. EmbeddingGemma-300M has ONE
  // KV head at head_dim 256, so its fused (and tile-padded) qkv is 1536 wide
  // against 3*768 = 2304 -- the check would have rejected its own correct
  // design, and in the other direction it is the T31 fail-open one field to
  // the left. 0 means "this container did not say", which is every BERT and
  // nomic container ever packed, and for those 3*hidden IS the answer.
  struct Want { const char *op; int64_t K, N; };
  const Want want[] = {
      {"qkv",      hidden,       qkv_n > 0 ? qkv_n : 3 * hidden},
      {"attn_out", hidden,       hidden},
      {"ffn_up",   hidden,       gated_ffn ? 2 * intermediate : intermediate},
      {"ffn_down", intermediate, hidden},
  };
  // Every op must be PRESENT, and EVERY occurrence of it must match -- a design
  // carrying one right batch tier and one wrong one does not fit.
  for (const Want &w : want) {
    bool seen = false;
    for (const StreamEntry &s : streams) {
      if (s.op != w.op) continue;
      seen = true;
      if (s.K != w.K || s.N != w.N) return false;
    }
    if (!seen) return false;
  }
  return true;
}

// The design set for a model, when --artifacts is not given.
//
// Three layouts have to work and none is privileged: a single-width release
// (<root>/gemm_rtp), a multi-width release (<root>/<set>/gemm_rtp) and the
// source tree (<root>/runtime/artifacts*/gemm_rtp). Each candidate is tested
// by whether its design actually serves this width, so the answer is a fact
// about the design rather than a naming convention.
std::string pick_artifacts(const std::string &root, int64_t hidden,
                           int64_t intermediate, bool gated_ffn,
                           int64_t qkv_n = 0,
                           const std::string &want_layout = "",
                           const std::string &want_datapath = "") {
  namespace fs = std::filesystem;
  if (hidden <= 0 || intermediate <= 0) return "";
  std::error_code ec;
  if (design_fits(root, hidden, intermediate, gated_ffn, qkv_n, want_layout,
                  want_datapath))
    return root;

  // Sorted, so the choice is reproducible rather than filesystem-order
  // dependent -- and never by mtime, which a JIT cache hit does not restamp
  // (CLAUDE.md trap 7c).
  std::vector<std::string> cands;
  for (const fs::path base : {fs::path(root), fs::path(root) / "runtime"})
    for (fs::directory_iterator it(base, ec), end; !ec && it != end;
         it.increment(ec))
      if (it->is_directory(ec)) cands.push_back(it->path().string());
  std::sort(cands.begin(), cands.end());

  // SEVERAL sets can serve one width and NOT be interchangeable in speed.
  // tasks/0080 added an int8 design set that narrows C to bf16; it carries the
  // same b_layout_hash as the int32-C set (C's width is not part of B's
  // layout), so both pass design_fits and the sort silently prefers whichever
  // sorts first -- which happened to be the slower one. A wrong pairing still
  // refuses at stage time on the layout hash, so this is not a correctness
  // hole; it is the "status line reports the intention, not the value" shape
  // that has cost this project time repeatedly (traps 7c, tasks/0042). Name
  // what was chosen and what else fitted, and let the caller pass --artifacts.
  std::vector<std::string> fits;
  for (const auto &c : cands)
    if (design_fits(c, hidden, intermediate, gated_ffn, qkv_n, want_layout,
                    want_datapath))
      fits.push_back(c);
  if (fits.empty()) return "";
  // TIE-BREAK ON EVIDENCE, NOT ON SPELLING. Among sets that fit equally,
  // prefer one whose design.json actually records `emulate_bfp16`, so the
  // status line can state the datapath instead of declining to. Both kinds
  // are genuinely correct here -- absent is treated as bf16 for selection --
  // but picking the self-describing one turns "UNRECORDED" into a real
  // answer for free. Stable, so alphabetical order still decides within each
  // group and the choice stays reproducible (never mtime -- trap 7c).
  std::stable_partition(fits.begin(), fits.end(), [](const std::string &p) {
    std::ifstream f(p + "/gemm_rtp/design.json");
    if (!f) return false;
    std::string js((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
    return js.find("\"emulate_bfp16\"") != std::string::npos;
  });
  if (fits.size() > 1) {
    std::fprintf(stderr,
                 "note: %zu design sets serve hidden %lld; using %s\n",
                 fits.size(), (long long)hidden,
                 fs::path(fits[0]).filename().string().c_str());
    for (size_t i = 1; i < fits.size(); ++i)
      std::fprintf(stderr, "      also fits: %s  (--artifacts to choose)\n",
                   fs::path(fits[i]).filename().string().c_str());
  }
  return fits[0];
}

void print_usage() {
  std::printf(
      "NpuEmbeddings -- BERT embeddings on the AMD Ryzen AI NPU (XDNA2)\n"
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
      "  npuembeddings add <org/model> [<sha256>]\n"
      "        teach this installation about a model that is not built in --\n"
      "        typically a finetune of one that is. Reads the repository's\n"
      "        config.json, derives the geometry, checks a design serves it,\n"
      "        and writes models/catalog.json. No weights are downloaded until\n"
      "        the first serve/embed.\n"
      "        WITHOUT a sha256 the weights are NOT verified. That is allowed,\n"
      "        and it is warned about on every single run.\n"
      "\n"
      "  Options for serve/embed:\n"
      "    --port N          listen port (default 8080)\n"
      "    --bind ADDR       interface (default 127.0.0.1, localhost only)\n"
      "    --threads N       host thread budget (default 24 for these)\n"
      "    --pipeline N      concurrent encode lanes (default 2)\n"
      "    --artifacts DIR   override the design set\n"
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
      "                      tools/export_gemm_rtp.py --seq to build for a\n"
      "                      longer one.\n"
      "\n"
      "  The flag form is unchanged and still works:\n"
      "    npuembeddings <root> --model NAME --artifacts DIR --serve [port]\n"
      "  and carries the probes and benchmarks; see docs/CURRENT_STATUS.md.\n"
      "\n");
}

void print_catalog(const std::string &root) {
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
    // "installed" is not the same as "runnable": the design set for this
    // width has to be present too, and a release ships one width. Saying so
    // here beats a confusing failure at dispatch.
    const bool have_design =
        !pick_artifacts(root, e.hidden, e.ffn, e.gated_ffn, e.qkv_n, "",
                        e.datapath).empty();
    // "cpu" is now a property of the CONTAINER, not of the architecture
    // (tasks/0074). arch=1 ran entirely on the host until this release and the
    // row said so unconditionally; it now has an NPU design and a pre-tiled
    // container, and an unconditional "cpu" would be the same kind of lie the
    // unconditional "ready" was before tasks/0069 -- just pointing the other
    // way. A host-only Gemma container (packed with --gemma-host-only, which
    // is still the correctness control) genuinely is "cpu", and says so.
    // "no encoder" is NOT the same as "no design": nomic-embed-text-v1.5 has a
    // matching design set and a packed container, and would still return
    // embeddings for the wrong model if run through the BERT encoder. Saying
    // "ready" because the design matches would be exactly the fail-open
    // set_model_shape() now refuses at dispatch. tasks/0069.
    const char *state = !m                              ? "available"
                        : !encoder_implemented(m->arch) ? "no encoder"
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

  // Anything packed locally that the catalogue does not know about. It is
  // perfectly valid -- `--prepare-model` builds one from any BERT checkpoint
  // -- and hiding it would make the table a lie about what `serve` accepts.
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
                // Same three-way split as the catalogue table above: a
                // host-only arch is neither "ready" nor "no design".
                !encoder_implemented(m.arch)              ? "no encoder"
                : m.gemm_layout == "host"                 ? "cpu"
                // Not in the catalogue, so no adoption decision was ever
                // made for it -- require plain bf16, the safe default
                // (CatalogEntry::datapath's own default, tasks/0104).
                : pick_artifacts(root, m.hidden, m.ffn, m.gated_ffn,
                                 m.qkv_n, "", "bf16").empty()
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
      "\n  npuembeddings serve <model>\n\n");
}

// arch=1 (EmbeddingGemma family). TWO paths, chosen from the CONTAINER, never
// from a model name or a flag default:
//
//   * `gemm_layout == "pretiled_bf16"` + a matching design set (tasks/0074)
//     runs GemmaNpuEncoder -- four GEMMs per layer on the array, 97.7% of the
//     model's MACs, with RMSNorm/RoPE/GeGLU/attention on the host.
//   * anything else runs npue::GemmaEncoder, the host-only reference
//     (tasks/0064, 1-cos 5.496e-13 against reference/encoder_gemma.py). It is
//     kept, not retired: it is the control this file is checked against, and
//     `--cpu` selects it deliberately so the two can be compared on one input.
//
// This is a SEPARATE code path from Encoder::run(), not a branch inside it --
// see GemmaNpuEncoder's own header comment for why. The BERT path is untouched
// and this function is reachable only when the .npue's own config["arch"] says
// gemma3_mqa_rope_geglu.
// EVERY TIME, not once at install time (user decision, 2026-08-22). A warning
// you saw last month is not a warning you see today, and the whole point of
// allowing an unpinned `add` is that the person running it knows the weights
// were never verified. Printed for any model whose catalogue row carries no
// sha256 -- which only `add` can produce.
void warn_if_unpinned(const std::string &name) {
  const npue::hub::CatalogEntry *e = npue::hub::find(name);
  if (!e || !npue::hub::unpinned(*e)) return;
  std::printf("\n  !! '%s' was added WITHOUT a sha256 pin (%s).\n",
              e->name.c_str(), e->repo.c_str());
  std::printf("  !! Its weights are NOT verified against anything.\n");
  std::printf("  !! To pin it:  npuembeddings add %s <sha256>\n\n",
              e->repo.c_str());
}

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
void check_gemma_prefix(const npue::GemmaTokenizer &tok,
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

// ---------------------------------------------------------------------------
// The OpenAI-shaped HTTP endpoint, shared by every architecture (tasks/0115).
//
// This used to live inline in the BERT path and read `EmbedService` directly,
// which is the whole reason arch=1 refused `--serve`: not that anything about
// EmbeddingGemma is incompatible with an HTTP endpoint, but that the endpoint
// was written against one encoder's TYPE. The coupling turned out to be three
// members wide -- vocabulary size, the prompt whitelist, and "embed these
// texts" -- so it is expressed as exactly those three here, and both paths
// hand them over.
//
// Requests are handled ONE AT A TIME on purpose. The NPU serializes dispatches
// anyway (research/notes/0004), and the lanes already parallelise inside a
// single request -- so concurrent request handling would add contention and
// lock complexity to buy nothing. Throughput comes from batching within a
// request, which is what an embeddings client does.
struct EmbedBackend {
  size_t vocab_size = 0;
  // This model's task-prompt names, sorted. EMPTY means the model has no
  // prefix concept at all, and that emptiness is the single source of truth
  // for it -- same discipline as g_prompts.empty() on the CLI side.
  //
  // Deliberately the NAMES and not the table: arch 1 keeps its prompts in the
  // GEMATOK1 tokenizer blob and arch 0/2 keep theirs in the container config,
  // and unifying those two stores is not this task's job. Each wiring site
  // fills this from whichever store it already reads and resolves name -> text
  // inside its own lambda; the handler below validates against this vector and
  // never learns which store answered.
  std::vector<std::string> prompt_names;
  int64_t hidden = 0;
  int64_t seq = 0;
  // Must throw npue::InputTooLong for an input that does not fit (tasks/0110)
  // rather than truncating: the handler below maps that type to 400, and a
  // backend that flattened it to a runtime_error would return 500 for what is
  // the caller's error.
  //
  // `prompt_name` is per REQUEST (tasks/0118), and "" means no prefix at all.
  // A name reaching here has already been checked against prompt_names.
  std::function<std::vector<float>(const std::vector<std::string> &,
                                   const std::string &,
                                   int64_t *)> embed;
};

int serve_http(const EmbedBackend &be, const std::string &model_id, int port,
               const std::string &bind_addr) {
  std::printf("  tokenizer  %zu tokens, from the .npue\n", be.vocab_size);
  std::printf("\n  serving http://%s:%d/v1/embeddings   "
              "(model %s, seq %lld)\n",
              bind_addr.c_str(), port, model_id.c_str(), (long long)be.seq);
  std::printf("  POST {\"input\": \"text\" | [\"a\",\"b\"], "
              "\"encoding_format\": \"float\"|\"base64\"}\n\n");
  // PER REQUEST, not per process (tasks/0118). This used to be a startup
  // choice applied uniformly for the life of the process, with a NOTE here
  // saying so -- which forced a deployment wanting both search_query and
  // search_document to run two servers, each holding an hw_context on a shared
  // NPU. The name is a request field now, required for a model that has a
  // table, and discoverable from GET /health without having to provoke a 400.
  if (!be.prompt_names.empty())
    std::printf("  REQUIRED per request: \"prompt_name\", one of [%s] "
                "(or \"\" for no prefix at all). A request without it is "
                "400.\n\n", join_names(be.prompt_names).c_str());

    const size_t kMaxTexts = 2048;
    npue::http::Server server(static_cast<uint16_t>(port), bind_addr);
    server.run([&](const npue::http::Request &req, int &status,
                   std::string &ctype, std::string &body) {
      auto fail = [&](int code, const char *type, const std::string &msg) {
        status = code;
        body = "{\"error\":{\"message\":\"" + npue::http::json_escape(msg) +
               "\",\"type\":\"" + type + "\"}}";
      };

      if (req.method == "GET" && (req.path == "/health" || req.path == "/")) {
        body = "{\"status\":\"ok\",\"model\":\"" + model_id +
               "\",\"backend\":\"amd-xdna2-npu\"";
        // DISCOVERY, so a client never has to provoke a 400 to learn what
        // this model offers (tasks/0118). This replaces the old "prefix" key,
        // which named the one server-wide prefix and is meaningless now that
        // the choice is per request. Still emitted only for a model that has
        // a prompt table at all, so a BERT model's /health is byte-identical
        // to what it was.
        if (!be.prompt_names.empty()) {
          body += ",\"prompt_names\":[";
          for (size_t i = 0; i < be.prompt_names.size(); ++i)
            body += (i ? ",\"" : "\"") +
                    npue::http::json_escape(be.prompt_names[i]) + "\"";
          body += "],\"prompt_required\":true";
        }
        body += "}";
        return;
      }
      if (req.method == "GET" && req.path == "/v1/models") {
        body = "{\"object\":\"list\",\"data\":[{\"id\":\"" + model_id +
               "\",\"object\":\"model\",\"owned_by\":\"npuembeddings\"}]}";
        return;
      }
      if (req.path != "/v1/embeddings") {
        fail(404, "not_found", "unknown path " + req.path);
        return;
      }
      if (req.method != "POST") {
        fail(400, "invalid_request_error", "use POST for /v1/embeddings");
        return;
      }

      std::vector<std::string> texts;
      std::string err;
      if (!npue::http::json_string_or_array(req.body, "input", texts, err)) {
        fail(400, "invalid_request_error", err);
        return;
      }
      if (texts.empty()) {
        fail(400, "invalid_request_error", "'input' is empty");
        return;
      }
      if (texts.size() > kMaxTexts) {
        fail(413, "invalid_request_error",
             "at most " + std::to_string(kMaxTexts) + " inputs per request, "
             "got " + std::to_string(texts.size()));
        return;
      }
      const std::string fmt =
          npue::http::json_field_string(req.body, "encoding_format", "float");
      if (fmt != "float" && fmt != "base64") {
        fail(400, "invalid_request_error",
             "encoding_format must be 'float' or 'base64', got '" + fmt + "'");
        return;
      }

      // THE TASK PROMPT, PER REQUEST (tasks/0118). Three refusals, each with a
      // counterpart in resolve_prefix() on the CLI side, so the two cannot
      // disagree about what this model accepts: a model with a table that was
      // not told which prompt, a model without one that was told anyway, and a
      // name that is not in the table. Every one of them LISTS what this model
      // really offers instead of guessing.
      //
      // OpenAI's own /v1/embeddings has no such field -- its embedding models
      // are symmetric, so the problem does not arise there. Among servers that
      // host asymmetric models the field is not standardised: Cohere requires
      // `input_type` (and nomic's four names are Cohere's values verbatim),
      // vLLM puts `input_type` on its OpenAI-compatible route, and TEI put
      // `prompt_name` on its native /embed and deliberately kept it off the
      // OpenAI one. `prompt_name` is the name here because EmbeddingGemma's
      // table holds 14 sentence-transformers keys -- STS, BitextMining,
      // Summarization -- and calling those an "input type" would be a lie.
      std::string prompt_name;
      const auto pk =
          npue::http::json_field_kind(req.body, "prompt_name", prompt_name);
      if (pk == npue::http::FieldKind::Other) {
        fail(400, "invalid_request_error",
             "'prompt_name' must be a string (\"\" means no prefix at all)");
        return;
      }
      const bool have_prompt = pk == npue::http::FieldKind::String;
      if (!be.prompt_names.empty() && !have_prompt) {
        fail(400, "invalid_request_error",
             "this model requires 'prompt_name'; valid names: [" +
             join_names(be.prompt_names) + "], or \"\" for no prefix at all. "
             "Refusing to pick one for you -- a wrongly-prefixed embedding is "
             "correctly shaped and correctly normed, so nothing downstream can "
             "tell that the answer is wrong.");
        return;
      }
      if (be.prompt_names.empty() && have_prompt) {
        // Mirrors resolve_prefix()'s first refusal. A client sweeping one
        // prompt across the whole catalogue SHOULD break here, because its
        // BERT results would otherwise differ from what it intended.
        fail(400, "invalid_request_error",
             "'prompt_name' was given, but this model has no task prompts -- "
             "nothing would be prepended. Refusing rather than returning "
             "vectors that are not what was asked for.");
        return;
      }
      if (have_prompt && !prompt_name.empty() &&
          std::find(be.prompt_names.begin(), be.prompt_names.end(),
                    prompt_name) == be.prompt_names.end()) {
        fail(400, "invalid_request_error",
             "'" + prompt_name + "' is not one of this model's prompt names: ["
             + join_names(be.prompt_names) + "]");
        return;
      }

      int64_t n_tokens = 0;
      std::vector<float> emb;
      try {
        emb = be.embed(texts, prompt_name, &n_tokens);
      } catch (const npue::InputTooLong &e) {
        // 400, not 500: the request is the thing that is wrong, and an
        // operator told 500 goes looking at the NPU. This catch must come
        // FIRST -- InputTooLong derives from std::runtime_error, so the
        // generic handler below would otherwise swallow it.
        fail(400, "invalid_request_error", e.what());
        return;
      } catch (const std::exception &e) {
        fail(500, "internal_error", e.what());
        return;
      }

      // 384 floats per row: reserve rather than grow, or a 2048-input
      // response reallocates its way through tens of MB.
      std::string out;
      out.reserve(texts.size() * (fmt == "base64" ? 2200 : 4600) + 256);
      out += "{\"object\":\"list\",\"data\":[";
      char num[40];
      for (size_t r = 0; r < texts.size(); ++r) {
        if (r) out += ',';
        out += "{\"object\":\"embedding\",\"index\":" + std::to_string(r) +
               ",\"embedding\":";
        const float *v = emb.data() + r * be.hidden;
        if (fmt == "base64") {
          out += '"';
          out += npue::http::base64(reinterpret_cast<const uint8_t *>(v),
                                    static_cast<size_t>(be.hidden) * sizeof(float));
          out += '"';
        } else {
          out += '[';
          for (int64_t c = 0; c < be.hidden; ++c) {
            if (c) out += ',';
            std::snprintf(num, sizeof num, "%.7g", v[c]);
            out += num;
          }
          out += ']';
        }
        out += '}';
      }
      out += "],\"model\":\"" + model_id +
             "\",\"usage\":{\"prompt_tokens\":" + std::to_string(n_tokens) +
             ",\"total_tokens\":" + std::to_string(n_tokens) + "}}";
      body.swap(out);
    });
  return 0;
}

int run_gemma_mode(npue::File &model, const std::string &model_path,
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

}  // namespace

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

  // --prepare-model <dir> [out.npue]: build the model container from an
  // upstream checkpoint. `dir` holds model.safetensors, vocab.txt and
  // config.json as downloaded from HuggingFace.
  //
  // The release ships no weights: they belong to
  // sentence-transformers/all-MiniLM-L6-v2, and fetching them from the
  // canonical source with a checksum beats trusting a blob in a zip. This is
  // what keeps that a two-step setup rather than a Python install.
  for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--prepare-model") {
    const std::string dir = argv[i + 1];
    // The container is named after the checkpoint directory, not after
    // MiniLM. This was a literal until a second model made it visible.
    std::string out =
        dir + "/" + std::filesystem::path(dir).filename().string() + ".npue";
    if (i + 2 < argc && argv[i + 2][0] != '-') out = argv[i + 2];

    // arch=1 (EmbeddingGemma / Gemma3 family): a completely different tensor
    // shape and container, routed to its own packer rather than threaded
    // through the BERT logic below -- tools/pack_npue.py's main() makes the
    // same decision the same way. Detected from the checkpoint's OWN
    // config.json (model_type), never assumed from --prepare-model's
    // directory name. tasks/0065-m12-embeddinggemma-cpp-packer.
    {
      std::ifstream cfg_probe(dir + "/config.json");
      if (cfg_probe) {
        std::stringstream cs;
        cs << cfg_probe.rdbuf();
        const std::string model_type =
            npue::http::json_field_string(cs.str(), "model_type", "");
        if (model_type == "gemma3_text") {
          std::string source_repo;
          for (int k = 1; k < argc - 1; ++k)
            if (std::string(argv[k]) == "--source-repo")
              source_repo = argv[k + 1];
          if (source_repo.empty()) {
            std::ifstream cf(dir + "/CHECKPOINT.json");
            if (!cf)
              throw std::runtime_error(
                  "no CHECKPOINT.json under " + dir + " and no --source-repo "
                  "given -- refusing to guess which repository these "
                  "weights came from");
            std::stringstream ckcs;
            ckcs << cf.rdbuf();
            source_repo =
                npue::http::json_field_string(ckcs.str(), "repo_id", "");
            if (source_repo.empty())
              throw std::runtime_error(dir + "/CHECKPOINT.json has no "
                                       "repo_id");
          }
          std::printf("  source     %s\n", source_repo.c_str());
          std::printf("NpuEmbeddings -- preparing %s\n", out.c_str());
          // Same two knobs the BERT path takes, plus the host-only escape
          // (tasks/0074). Defaults are the production geometry, so a cold
          // clone with only --token self-produces a container the ARRAY can
          // run -- which is the point: before this it self-produced a
          // host-only container and quietly ran at 0.2 seq/s.
          int64_t gtk = 64, gtn = 48;
          bool ghost = false;
          for (int k = 1; k < argc; ++k) {
            const std::string a = argv[k];
            if (a == "--gemma-host-only") ghost = true;
            if (k + 1 < argc && a == "--tile-k") gtk = std::atoll(argv[k + 1]);
            if (k + 1 < argc && a == "--tile-n") gtn = std::atoll(argv[k + 1]);
          }
          npue::prepare_model_gemma(dir, out, source_repo,
                                    [](const std::string &s) {
                                      std::printf("%s\n", s.c_str());
                                    },
                                    gtk, gtn, ghost);
          std::printf("  wrote %s\n", out.c_str());
          return 0;
        }
      }
    }

    // Tile size is a PROPERTY OF THE MODEL, not a constant. The design
    // asserts N % (tile_n * n_cols) == 0, and bge-large's N in
    // {1024, 3072, 4096} makes 48 illegal -- the legal set there is
    // {8, 16, 32, 64} and 64 does not fit L1 (65,536 B against the 63 KB
    // budget), so it must be 32. Both packers now take it and neither
    // freezes the resulting hash.
    int64_t tile_k = 64, tile_n = 48;
    for (int k = 1; k < argc - 1; ++k) {
      if (std::string(argv[k]) == "--tile-n") tile_n = std::atoi(argv[k + 1]);
      if (std::string(argv[k]) == "--tile-k") tile_k = std::atoi(argv[k + 1]);
    }
    const npue::Layout lay = npue::gemm_b_layout(tile_k, tile_n);
    const std::string layout = lay.json;
    const std::string layout_hash = lay.hash;
    std::printf("  layout     tile (%lld, %lld), hash %s...\n",
                (long long)tile_k, (long long)tile_n,
                layout_hash.substr(0, 16).c_str());

    // Pooling comes from the checkpoint's own 1_Pooling/config.json, the
    // same source tools/pack_npue.py reads. Both packers must agree or
    // verify_pack_parity fails, which is the point of having the gate.
    std::string pooling;
    {
      std::ifstream pf(dir + "/1_Pooling/config.json");
      if (!pf)
        throw std::runtime_error(
            "no 1_Pooling/config.json under " + dir + " -- cannot tell "
            "whether this checkpoint pools by mean or by CLS");
      std::stringstream ps;
      ps << pf.rdbuf();
      const std::string pj = ps.str();
      auto flag = [&](const char *k) {
        const size_t i = pj.find(k);
        if (i == std::string::npos) return false;
        const size_t c = pj.find(':', i);
        return pj.compare(pj.find_first_not_of(" \t", c + 1), 4, "true") == 0;
      };
      const bool cls = flag("pooling_mode_cls_token");
      const bool mean = flag("pooling_mode_mean_tokens");
      if (cls == mean)
        throw std::runtime_error(
            "1_Pooling/config.json asks for neither or both of cls and mean; "
            "this runtime implements exactly those two");
      pooling = cls ? "cls" : "mean";
      std::printf("  pooling    %s (from 1_Pooling/config.json)\n",
                  pooling.c_str());
    }

    // Which repository these weights came from. tools/pack_npue.py reads
    // CHECKPOINT.json for this and so must we, or the two packers disagree.
    // A container that misattributes its own weights is a licensing
    // statement, so an unknown repo REFUSES rather than guessing.
    std::string source_repo;
    for (int k = 1; k < argc - 1; ++k)
      if (std::string(argv[k]) == "--source-repo") source_repo = argv[k + 1];
    if (source_repo.empty()) {
      std::ifstream cf(dir + "/CHECKPOINT.json");
      if (!cf)
        throw std::runtime_error(
            "no CHECKPOINT.json under " + dir + " and no --source-repo given "
            "-- refusing to guess which repository these weights came from");
      std::stringstream cs;
      cs << cf.rdbuf();
      source_repo = npue::http::json_field_string(cs.str(), "repo_id", "");
      if (source_repo.empty())
        throw std::runtime_error(dir + "/CHECKPOINT.json has no repo_id");
    }
    std::printf("  source     %s\n", source_repo.c_str());

    // arch=2 (nomic-embed-text-v1.5): RoPE + gated SwiGLU rather than
    // BERT's absolute-position + GELU -- routed to its own packer, mirroring
    // tools/pack_npue.py's `model_type == "nomic_bert"` branch in main().
    // Detected the same way the gemma3_text branch above is (config.json's
    // OWN model_type), never assumed from --prepare-model's directory name.
    // Placed HERE, after tile_k/tile_n/layout/pooling/source_repo are
    // already resolved above, because nomic shares every one of those
    // resolutions with the BERT path unchanged -- only the packer differs.
    // tasks/0071.
    {
      std::ifstream cfg_probe2(dir + "/config.json");
      if (cfg_probe2) {
        std::stringstream cs2;
        cs2 << cfg_probe2.rdbuf();
        const std::string model_type =
            npue::http::json_field_string(cs2.str(), "model_type", "");
        if (model_type == "nomic_bert") {
          std::printf("NpuEmbeddings -- preparing %s (arch=nomic_bert_rope_swiglu)\n",
                      out.c_str());
          npue::prepare_model_nomic(dir, pooling, source_repo, out, layout,
                                    layout_hash, tile_k, tile_n, 256,
                                    [](const std::string &s) {
                                      std::printf("%s\n", s.c_str());
                                    });
          std::printf("  wrote %s\n", out.c_str());
          return 0;
        }
        // arch=3 (gte-multilingual-base): model_type "new", same dispatch
        // rule as the nomic branch above (the checkpoint's OWN config.json,
        // never the directory name). max_seq 64 matches the Python-packed
        // container this mirror is held byte-identical to (tasks/0135
        // packed --max-seq 64; under RoPE the position table is zeros, so
        // max_seq only caps request length). tasks/0138.
        if (model_type == "new") {
          std::printf("NpuEmbeddings -- preparing %s (arch=gte_new_rope_geglu)\n",
                      out.c_str());
          npue::prepare_model_gte(dir, pooling, source_repo, out, layout,
                                  layout_hash, tile_k, tile_n, 64,
                                  [](const std::string &s) {
                                    std::printf("%s\n", s.c_str());
                                  });
          std::printf("  wrote %s\n", out.c_str());
          return 0;
        }
      }
    }

    std::printf("NpuEmbeddings -- preparing %s\n", out.c_str());
    npue::prepare_model(dir + "/model.safetensors", dir + "/vocab.txt",
                        dir + "/config.json", pooling, source_repo, out,
                        "", layout, layout_hash,
                        tile_k, tile_n, 256,
                        [](const std::string &s) {
                          std::printf("%s\n", s.c_str());
                        });
    std::printf("  wrote %s\n", out.c_str());
    return 0;
  }

  // arch=1 (EmbeddingGemma family): resolved and dispatched EARLY, before
  // the --artifacts resolution below -- which THROWS if no BERT NPU design
  // is found under `root`, a precondition this arch does not share (it has
  // no NPU design at all, by design -- see run_gemma_mode's own comment).
  // Without this early exit, a Gemma-only invocation on a machine with no
  // BERT design installed would fail before ever reaching the model it
  // actually asked for. Exceptions here are swallowed and fall through to
  // the ORIGINAL resolution path below unchanged, which reports the real
  // error (missing model, ambiguous --model, ...) exactly as if this block
  // did not exist -- the model is opened via mmap twice in the Gemma case
  // (once here, once implicitly inside run_gemma_mode via the File this
  // block constructs), which is cheap and never touches the BERT path.
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


  // --tokenize <file> [max_len]: one text per line in, one line of token
  // ids out. No NPU, no model -- this is the mode tools/verify_tokenizer.py
  // drives to compare against HuggingFace token for token.
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
    // Deliberately NOT an error here, and deliberately not on stdout either.
    // This mode exists for tools/verify_tokenizer.py to diff against
    // HuggingFace token for token, and HuggingFace truncates too -- refusing
    // would make the two incomparable at exactly the lengths worth comparing,
    // and an extra stdout line would desynchronise the diff. So: the same ids
    // as before on stdout, and a count on stderr, so a truncating corpus
    // cannot be mistaken for a clean one.
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
    return 0;
  }

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

  npu::Device dev;

  // --probe-pair runs BEFORE the other five designs exist. If seven resident
  // contexts on an eight-column NPU are what forces the reconfiguration, then
  // alternating between two designs with only two loaded should be cheap. If it
  // costs the same as it does with seven loaded, the penalty is inherent to
  // switching and no amount of trimming the resident set will help.
  for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == "--probe-pair") {
    npu::Design a(dev, art + "/qkv"), b(dev, art + "/ffn_up");
    const int reps = 100;
    std::printf("\n  probe-pair -- only 2 designs loaded, %d repeats\n", reps);
    auto run = [&](const char *label, npu::Design &x, npu::Design *y) {
      x.dispatch_only();
      if (y) y->dispatch_only();
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) { x.dispatch_only();
                                       if (y) y->dispatch_only(); }
      double us = (now_s() - t0) / (reps * (y ? 2 : 1)) * 1e6;
      std::printf("      %-22s %8.0f us\n", label, us);
    };
    run("qkv alone", a, nullptr);
    run("ffn_up alone", b, nullptr);
    run("qkv <-> ffn_up", a, &b);
    return 0;
  }

  // --probe-design <dir> measures ONE design's switch cost in isolation:
  // dispatch it alone, then alternate two contexts holding the same xclbin.
  // The difference is the switch, with compute subtracted out.
  //
  // The point is to compare designs of EQUAL WIDTH and very unequal
  // configuration complexity. A→A' already showed the cost does not depend on
  // how DIFFERENT the two configurations are; it does not follow that it is
  // independent of how MUCH configuration there is. If a trivial 1-column
  // passthrough switches as slowly as a 1-column GEMM, the per-column cost is
  // fixed and unreachable. If it is cheaper, dataflow complexity is a knob.
  // --probe-bo <design-dir> <chunk_mb> <count>: can this machine hold the XRT
  // buffers a wider model needs? bge-large wants ~1023 MB across two lanes
  // (604 MB of staged weights plus 209 MB of A/B/C per lane) against MiniLM's
  // 175 MB. Answering that with one command beats discovering it after a
  // four-xclbin build at h=1024.
  for (int i = 2; i < argc - 3; ++i)
      if (std::string(argv[i]) == "--probe-bo") {
    const std::string dir = root + "/runtime/" + argv[i + 1];
    const size_t chunk = static_cast<size_t>(std::atof(argv[i + 2]) * 1e6);
    const size_t count = static_cast<size_t>(std::atoi(argv[i + 3]));
    npu::Design d0(dev, dir);
    std::printf("  probe      %zu x %.1f MB = %.1f MB, mode %s\n",
                count, chunk / 1e6, count * chunk / 1e6, npu::bo_mode_name());
    const double t0 = now_s();
    const size_t ok = d0.probe_alloc(chunk, count, true);
    std::printf("  allocated  %zu of %zu (%.1f MB) in %.2f s\n",
                ok, count, ok * chunk / 1e6, now_s() - t0);
    return ok == count ? 0 : 3;
  }

  for (int i = 2; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--probe-design") {
    const std::string dir = root + "/runtime/" + argv[i + 1];
    npu::Design a(dev, dir), a2(dev, dir);
    const int reps = 100;
    a.dispatch_only();
    double t0 = now_s();
    for (int r = 0; r < reps; ++r) a.dispatch_only();
    const double alone = (now_s() - t0) / reps * 1e6;
    a.dispatch_only();
    a2.dispatch_only();
    t0 = now_s();
    for (int r = 0; r < reps; ++r) { a.dispatch_only(); a2.dispatch_only(); }
    const double pair = (now_s() - t0) / (2 * reps) * 1e6;
    std::printf("  %-22s alone %7.0f us   A<->A' %7.0f us   switch %7.0f us\n",
                argv[i + 1], alone, pair, pair - alone);
    return 0;
  }

  // --probe-insts <dirA> <dirB>: THE step-0 measurement for the one-xclbin
  // architecture (Roesti et al., FCCM 2025: keep one static design and
  // vary only the runtime sequence).
  //
  // dirA and dirB hold the SAME static design with two different runtime
  // sequences: their xclbins differ only in UUID metadata (verified byte by
  // byte before this probe existed), their insts.bin differ. Load dirA's
  // xclbin ONCE, its context ONCE, both instruction streams -- and alternate.
  //
  //   alternation ~= alone      -> the design switch is gone. Every operation
  //                                whose static design can be shared becomes an
  //                                instruction stream, and the 49 switches per
  //                                encode (~60 ms at batch 128) simply vanish.
  //   alternation ~= two-context cost -> the switch is tied to the instruction
  //                                stream itself, and the one-xclbin road ends.
  for (int i = 2; i < argc - 2; ++i)
      if (std::string(argv[i]) == "--probe-insts") {
    const std::string da = root + "/runtime/" + argv[i + 1];
    const std::string db = root + "/runtime/" + argv[i + 2];
    npu::Design a(dev, da);
    const size_t sB = a.load_instr(db + "/insts.bin");
    npu::Design a2(dev, da);              // control: second context, same bytes
    const int reps = 100;
    std::printf("\n  probe-insts -- %d repeats\n", reps);
    auto once = [&](const char *label, auto &&body) {
      body();                             // warm
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) body();
      std::printf("      %-38s %8.0f us\n", label,
                  (now_s() - t0) / reps * 1e6);
    };
    // Correctness first: completion status is not data. Both sequences copy
    // input to output (in different task order), so with a ramp staged in,
    // each stream must reproduce it exactly.
    {
      const size_t n = a.info().buffer_bytes[0] / 2;
      auto *in = static_cast<uint16_t *>(a.host_ptr(0));
      for (size_t j = 0; j < n; ++j) in[j] = static_cast<uint16_t>(j * 2654435761u >> 16);
      a.sync_to_device(0);
      for (size_t slot : {size_t(0), sB}) {
        auto *out = static_cast<uint16_t *>(a.host_ptr(1));
        std::memset(out, 0, a.info().buffer_bytes[1]);
        a.sync_to_device(1);
        a.bind_instr(slot);
        a.dispatch_only();
        a.sync_from_device(1);
        size_t bad = 0;
        for (size_t j = 0; j < n; ++j) bad += (out[j] != in[j]);
        std::printf("      stream %zu output: %s (%zu of %zu wrong)\n", slot,
                    bad ? "WRONG" : "exact", bad, n);
        if (bad) return 1;
      }
      a.bind_instr(0);
    }

    once("A alone (stream 0)", [&] { a.dispatch_only(); });
    a.bind_instr(sB);
    once("B alone (stream 1, same context)", [&] { a.dispatch_only(); });
    once("A <-> B, ONE context, two streams", [&] {
      a.bind_instr(0);
      a.dispatch_only();
      a.bind_instr(sB);
      a.dispatch_only();
    });
    once("A <-> A', TWO contexts (control)", [&] {
      a.bind_instr(0);
      a.dispatch_only();
      a2.dispatch_only();
    });
    // the paired loops dispatch twice per iteration
    std::printf("      (paired rows are per two dispatches; halve to"
                " compare)\n");
    return 0;
  }

  // --probe-rtp <dirA> <dirB>: the FUNCTIONAL half of one-xclbin step 1.
  //
  // dirA and dirB are two RTP-ified GEMM shapes whose static configurations
  // are byte-identical modulo UUIDs (gemm_rtp_probe.py verified that). Load
  // dirA's xclbin ONCE, both instruction streams, and run BOTH shapes through
  // the one context -- each stream carries its own shim BDs and its own RTP
  // writes (loop bounds), so the same ELF computes different shapes.
  //
  // Correctness by the constant-B trick: with every element of B equal to c,
  // C[i,j] = c * sum_k A[i,k] regardless of B's tiled layout -- so the host
  // reference needs no de-tiling and any wrong loop bound, routing or RTP
  // value shows up as a wrong sum.
  for (int i = 2; i < argc - 2; ++i)
      if (std::string(argv[i]) == "--probe-rtp") {
    const std::string da = root + "/runtime/" + argv[i + 1];
    const std::string db = root + "/runtime/" + argv[i + 2];
    npu::Design a(dev, da);
    const size_t sB = a.load_instr(db + "/insts.bin");
    // read dirB's shape
    npu::Design binfo(dev, db);
    const int64_t M0 = a.info().M, K0 = a.info().K, N0 = a.info().N;
    const int64_t M1 = binfo.info().M, K1 = binfo.info().K,
                  N1 = binfo.info().N;
    std::printf("\n  probe-rtp -- one xclbin (%s), two shapes\n",
                argv[i + 1]);
    // This probe reads C as fp32 directly. A --c-bf16 artifact would still
    // "work" and produce plausible-looking wrong sums, which is the exact
    // failure mode tasks/0009 and CLAUDE.md trap 6c are about. Refuse.
    if (a.info().c_elem_bytes != 4)
      throw std::runtime_error(
          "--probe-rtp reads C as fp32; this design emits bf16 C "
          "(tasks/0045). Use an artifact set exported without --c-bf16.");

    const float cB = 0.5f;
    auto run_shape = [&](size_t slot, int64_t M_, int64_t K_, int64_t N_,
                         const char *label) {
      auto *pa = static_cast<uint16_t *>(a.host_ptr(0));
      std::vector<float> arow(static_cast<size_t>(M_ * K_));
      for (size_t j = 0; j < arow.size(); ++j)
        arow[j] = 0.001f * static_cast<float>((j * 37) % 200) - 0.1f;
      bf16_fill(pa, arow.data(), arow.size());
      a.sync_to_device(0);
      auto *pb = static_cast<uint16_t *>(a.host_ptr(1));
      const size_t nb = static_cast<size_t>(K_ * N_);
      const uint16_t cbits = to_bf16(cB);
      for (size_t j = 0; j < nb; ++j) pb[j] = cbits;
      a.sync_to_device(1);
      std::memset(a.host_ptr(2), 0,
                  static_cast<size_t>(M_ * N_) * sizeof(float));
      a.sync_to_device(2);
      a.bind_instr(slot);
      a.dispatch_only();
      a.sync_from_device(2);
      const float *c = static_cast<const float *>(a.host_ptr(2));
      double worst = 0.0;
      for (int64_t r = 0; r < M_; ++r) {
        float sum = 0.f;
        for (int64_t kk = 0; kk < K_; ++kk)
          sum += from_bf16(to_bf16(arow[static_cast<size_t>(r * K_ + kk)]));
        const float want = from_bf16(cbits) * sum;
        for (int64_t j = 0; j < N_; ++j) {
          const double rel = std::abs(c[r * N_ + j] - want) /
                             std::max(1e-6, std::abs(double(want)));
          worst = std::max(worst, rel);
        }
      }
      std::printf("      %-22s worst rel err %.3e  %s\n", label, worst,
                  worst < 2e-2 ? "OK" : "WRONG");
      return worst < 2e-2;
    };
    bool ok = run_shape(0, M0, K0, N0, "stream 0 (own shape)");
    ok &= run_shape(sB, M1, K1, N1, "stream 1 (other shape)");
    ok &= run_shape(0, M0, K0, N0, "stream 0 again");
    if (!ok) return 1;

    const int reps = 100;
    auto once = [&](const char *label, auto &&body) {
      body();
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) body();
      std::printf("      %-34s %8.0f us\n", label,
                  (now_s() - t0) / reps * 1e6);
    };
    a.bind_instr(0);
    once("shape A alone", [&] { a.dispatch_only(); });
    a.bind_instr(sB);
    once("shape B alone", [&] { a.dispatch_only(); });
    once("A <-> B (per two dispatches)", [&] {
      a.bind_instr(0);
      a.dispatch_only();
      a.bind_instr(sB);
      a.dispatch_only();
    });
    return 0;
  }

  // --probe-ctx separates two explanations that --probe cannot tell apart.
  //
  // Alternating designs costs ~1200 us more than repeating one. That could be
  // the array being RECONFIGURED (different configuration data must be loaded)
  // or the driver SWITCHING HARDWARE CONTEXTS (a fixed cost that does not care
  // what is in them). The test: load the SAME xclbin into two contexts and
  // alternate. Identical configuration, two contexts.
  //
  //   A <-> A' as expensive as A <-> B  ->  context switch; design width is
  //                                        irrelevant and only the number of
  //                                        switches can be reduced.
  //   A <-> A' cheap                    ->  reconfiguration; configuration
  //                                        volume is the lever.
  for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == "--probe-ctx") {
    npu::Design a(dev, art + "/qkv");
    npu::Design a2(dev, art + "/qkv");        // same bytes, second context
    npu::Design b(dev, art + "/ffn_up");
    const int reps = 100;
    std::printf("\n  probe-ctx -- %d repeats, %s\n", reps, art.c_str());
    auto pair = [&](const char *label, npu::Design &x, npu::Design &y) {
      x.dispatch_only();
      y.dispatch_only();
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) { x.dispatch_only(); y.dispatch_only(); }
      std::printf("      %-34s %8.0f us\n", label,
                  (now_s() - t0) / (2 * reps) * 1e6);
    };
    a.dispatch_only();
    double t0 = now_s();
    for (int r = 0; r < reps; ++r) a.dispatch_only();
    std::printf("      %-34s %8.0f us\n", "qkv alone (one context)",
                (now_s() - t0) / reps * 1e6);
    pair("qkv <-> qkv, TWO contexts", a, a2);
    pair("qkv <-> ffn_up, two contexts", a, b);
    return 0;
  }

  // --soak-npu <seconds>: dispatch in a tight loop with NO host work at all,
  // for the energy control experiment (tasks/0034). The question it answers is
  // whether the RAPL package meter SEES the NPU: if package power does not
  // move above idle while the array is saturated, the meter does not cover the
  // NPU block and every NPU energy figure is a lower bound.
  //
  // One thread, one buffer set, no conversion, no sync -- the same
  // dispatch_only() loop --probe uses, held for a measurable duration.
  for (int i = 2; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--soak-npu") {
    const double secs = std::atof(argv[i + 1]);
    const std::string dir =
        std::ifstream(art + "/gemm_rtp/design.json").good()
            ? art + "/gemm_rtp" : art + "/qkv";
    npu::Design d(dev, dir);
    std::printf("  soak-npu   %s, %.1f s, dispatch only, zero host work\n",
                dir.c_str(), secs);
    d.dispatch_only();                              // warm
    const double t0 = now_s();
    int64_t n = 0;
    while (now_s() - t0 < secs) { d.dispatch_only(); ++n; }
    const double el = now_s() - t0;
    std::printf("  soak-npu   %lld dispatches in %.2f s  (%.0f us each)\n",
                (long long)n, el, el / n * 1e6);
    return 0;
  }

  // --soak-cpu <seconds> [threads]: the mirror control. Busy fp32 AVX2 work,
  // no NPU at all, so the same meter can be shown to move for CPU load.
  for (int i = 2; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--soak-cpu") {
    const double secs = std::atof(argv[i + 1]);
    int nt = 12;
    if (i + 2 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 2][0])))
      nt = std::atoi(argv[i + 2]);
    std::printf("  soak-cpu   %.1f s on %d threads, no NPU\n", secs, nt);
    std::vector<std::thread> ts;
    std::vector<double> sink(nt, 0.0);
    const double t0 = now_s();
    for (int w = 0; w < nt; ++w)
      ts.emplace_back([&, w] {
        float acc = 1.0f;
        std::vector<float> buf(4096, 1.000001f);
        while (now_s() - t0 < secs)
          for (int r = 0; r < 64; ++r)
            for (size_t j = 0; j < buf.size(); ++j) acc = acc * 0.9999f + buf[j];
        sink[w] = acc;
      });
    for (auto &th : ts) th.join();
    std::printf("  soak-cpu   done in %.2f s (sink %.3f)\n", now_s() - t0,
                sink[0]);
    return 0;
  }

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
