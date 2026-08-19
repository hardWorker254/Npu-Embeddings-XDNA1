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
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <functional>
#include <mutex>
#include <string>
#include <cctype>
#include <thread>
#include <vector>

#include "hub.hpp"
#include "npu_contention.hpp"
#include "npu_device.hpp"
#include "npue.hpp"
#include "npue_pack.hpp"
#include "http.hpp"
#include "tokenizer.hpp"

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
std::string g_model_name, g_source_repo;

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
  std::string path, name, repo, pooling, error;
  int64_t layers = 0, hidden = 0, heads = 0, head_dim = 0, ffn = 0, seq = 0;
  double mb = 0;
};

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
      m.layers = f.config_int("num_layers");
      m.hidden = f.config_int("hidden");
      m.heads = f.config_int("num_heads");
      m.head_dim = f.config_int("head_dim");
      m.ffn = f.config_int("intermediate");
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
npue::Tokenizer load_tokenizer(npue::File &model,
                               const std::string &model_path) {
  try {
    auto v = model.raw("tokenizer.vocab");
    return npue::Tokenizer::from_vocab_bytes(
        reinterpret_cast<const char *>(v.data), v.bytes);
  } catch (const std::exception &) {
    // Pre-0036 container: the loose checkpoint directory beside it, derived
    // from the container's own name rather than assumed to be MiniLM's.
    const std::string p =
        std::filesystem::path(model_path).replace_extension().string() +
        "/vocab.txt";
    std::printf("  tokenizer  .npue has no vocabulary; using %s\n", p.c_str());
    return npue::Tokenizer::from_vocab_file(p);
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

  // Pooling is data. sentence-transformers ships the answer in
  // 1_Pooling/config.json and the packer copies it here; a container that
  // predates that carries "mean", which is what MiniLM wants anyway.
  const std::string pool = m.config_string("pooling");
  if (pool == "cls") g_cls_pool = true;
  else if (pool == "mean") g_cls_pool = false;
  else throw std::runtime_error("unknown pooling mode '" + pool +
                                "' in the .npue -- expected mean or cls");

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

  // Scratch reused across layers. `residual = x` used to allocate and copy
  // 12.6 MB per layer at batch 128 -- 75 MB per encode of pure copying.
  std::vector<float> residual;
  // Scratch, sized once. These used to be six fresh vectors per run() -- ~90
  // MB of allocate-and-touch per encode at batch 128, and ~280 MB at
  // bge-large's width. resize() after the first call is a no-op.
  std::vector<float> qkvbuf, ctx, proj, up, down, scores;

  // Device-resident weights, one slot per layer per design, plus the bias
  // pointers straight into the mapped file. Filled by stage_all().
  std::vector<size_t> s_qkv, s_ao, s_fu, s_fd;
  std::vector<const float *> b_qkv, b_ao, b_fu, b_fd;
  std::vector<size_t> s_ln;      // 0 = embeddings, then ln1/ln2 per layer
  // Host-side views of the same parameters, straight into the mapped .npue.
  // tasks/0031 measured a LayerNorm call at 725 us of kernel inside ~3 ms of
  // switch+conversion; a threaded fp32 AVX2 LayerNorm on the host costs
  // ~0.5 ms and removes 13 design switches outright. --host-ln selects it.
  std::vector<const float *> h_gamma, h_beta;
  bool host_ln = false;
  bool host_sm = false;
  bool host_gelu = false;
  double t_hostln = 0.0, t_hostsm = 0.0, t_hostgelu = 0.0;


  // Where the time goes. A single number for the whole encode says "slow";
  // this says which half to fix.
  double t_npu = 0.0;      // memcpy + sync + dispatch, i.e. everything a
                           // fused design would subsume
  double t_attn = 0.0;     // the per-head QK^T and A.V loops on the host
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
    t_npu = t_attn = 0.0;
    t_hostln = t_hostsm = t_hostgelu = 0.0;
    t_conv = t_in = t_disp = t_out = t_bias = 0.0;
    n_dispatch = 0;
  }

  // Move every weight onto the device once. Returns the bytes staged, so the
  // banner can state the cost of the trade rather than hiding it.
  size_t stage_all() {
    size_t bytes = 0;
    auto one = [&](npu::Design &d, const std::string &name,
                   std::vector<size_t> &slots,
                   std::vector<const float *> &bias) {
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
    };
    for (int64_t L = 0; L < g_layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      one(qkv, p + "qkv", s_qkv, b_qkv);
      one(attn_out, p + "attn_out", s_ao, b_ao);
      one(ffn_up, p + "ffn_up", s_fu, b_fu);
      one(ffn_down, p + "ffn_down", s_fd, b_fd);
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

  void gemm(npu::Design &d, size_t islot, const std::vector<float> &a,
            size_t wslot, const float *bias, std::vector<float> &out,
            int64_t N) {
    double t0 = now_s();
    auto *abuf = static_cast<uint16_t *>(d.slot_ptr(0, slot_a));
    par(a.size(), [&](size_t lo, size_t hi) {
      bf16_fill(abuf + lo, a.data() + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    const float *c;
    {
      std::unique_lock<std::mutex> lk;
      if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
      if (unified) d.bind_instr(islot);
      d.bind(0, slot_a);
      d.bind(1, wslot);        // weights are already on the device
      d.bind(2, slot_c);
      d.sync_to_device(0, a.size() * sizeof(uint16_t));
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
    if (d.info().c_elem_bytes == 2) {
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
#if defined(__AVX2__)
          // head_dim / 8 vectors, held across the j loop. head_dim is 32 for
          // MiniLM and bge-small and 64 for bge-large; kMaxHeadVecs bounds the
          // stack array and set_model_shape() refuses anything larger.
          __m256 qv[NV ? NV : kMaxHeadVecs];
          const int64_t nv = NV ? NV : g_head_dim / 8;
          for (int64_t v = 0; v < nv; ++v) qv[v] = _mm256_loadu_ps(q + v * 8);
#endif
          for (int64_t j = 0; j < g_seq; ++j) {
            const float *k = &qkv_p[(b * g_seq + j) * 3 * g_hidden + g_hidden +
                                    h * g_head_dim];
#if defined(__AVX2__)
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

  std::vector<float> run(const std::vector<float> &emb_in) {
    std::vector<float> x = emb_in;
    layer_norm(x, s_ln[0]);

    qkvbuf.resize(rows * 3 * g_hidden);
    ctx.resize(rows * g_hidden);
    proj.resize(rows * g_hidden);
    up.resize(rows * g_ffn);
    down.resize(rows * g_hidden);
    scores.resize(batch * g_heads * g_seq * g_seq);

    residual.resize(x.size());
    for (int64_t L = 0; L < g_layers; ++L) {
      std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));

      gemm(qkv, is_qkv, x, s_qkv[L], b_qkv[L], qkvbuf, 3 * g_hidden);

      double ta = now_s();
      // QK^T per head, on the host: [64,32]x[32,64] does not tile (head_dim 32
      // fails the whole-array design's M % (m*4) == 0).
      // 1/sqrt(head_dim) is already folded into Q by the .npue.
      //
      // head_dim is 32 = four AVX2 vectors, and the 32 floats of one head ARE
      // contiguous even though consecutive rows are 3*hidden apart. So the dot
      // product vectorises without any repacking.
      qk(qkvbuf, scores);
      t_attn += now_s() - ta;

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
      t_attn += now_s() - ta;

      gemm(attn_out, is_ao, ctx, s_ao[L], b_ao[L], proj, g_hidden);
      add_into(x, proj);
      layer_norm(x, s_ln[1 + 2 * L]);

      std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));
      gemm(ffn_up, is_fu, x, s_fu[L], b_fu[L], up, g_ffn);

      if (host_gelu)
        gelu_cpu(up);
      else
        eltwise(gelu, up.data(), up.size());

      gemm(ffn_down, is_fd, up, s_fd[L], b_fd[L], down, g_hidden);
      add_into(x, down);
      layer_norm(x, s_ln[2 + 2 * L]);
    }
    return x;
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

// Does this design set serve a model of this width? qkv, attn_out and ffn_up
// all have K = hidden, so `hidden` must appear as a K in design.json. Checked
// rather than assumed because a design built for another width has the same
// filenames and loads fine -- it would simply compute the wrong thing.
bool design_fits(const std::string &design_dir, int64_t hidden) {
  std::ifstream f(design_dir + "/gemm_rtp/design.json");
  if (!f) return false;
  std::stringstream b;
  b << f.rdbuf();
  const std::string s = b.str();
  const std::string want = "\"K\": " + std::to_string(hidden);
  const std::string want2 = "\"K\":" + std::to_string(hidden);
  auto exact = [&](const std::string &pat) {
    for (size_t p = s.find(pat); p != std::string::npos;
         p = s.find(pat, p + 1)) {
      const size_t e = p + pat.size();
      if (e >= s.size() || !isdigit((unsigned char)s[e])) return true;
    }
    return false;
  };
  return exact(want) || exact(want2);
}

// The design set for a model, when --artifacts is not given.
//
// Three layouts have to work and none is privileged: a single-width release
// (<root>/gemm_rtp), a multi-width release (<root>/<set>/gemm_rtp) and the
// source tree (<root>/runtime/artifacts*/gemm_rtp). Each candidate is tested
// by whether its design actually serves this width, so the answer is a fact
// about the design rather than a naming convention.
std::string pick_artifacts(const std::string &root, int64_t hidden) {
  namespace fs = std::filesystem;
  if (hidden <= 0) return "";
  std::error_code ec;
  if (design_fits(root, hidden)) return root;

  // Sorted, so the choice is reproducible rather than filesystem-order
  // dependent -- and never by mtime, which a JIT cache hit does not restamp
  // (CLAUDE.md trap 7c).
  std::vector<std::string> cands;
  for (const fs::path base : {fs::path(root), fs::path(root) / "runtime"})
    for (fs::directory_iterator it(base, ec), end; !ec && it != end;
         it.increment(ec))
      if (it->is_directory(ec)) cands.push_back(it->path().string());
  std::sort(cands.begin(), cands.end());
  for (const auto &c : cands)
    if (design_fits(c, hidden)) return c;
  return "";
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
      "  Options for serve/embed:\n"
      "    --port N          listen port (default 8080)\n"
      "    --bind ADDR       interface (default 127.0.0.1, localhost only)\n"
      "    --threads N       host thread budget (default 24 for these)\n"
      "    --pipeline N      concurrent encode lanes (default 2)\n"
      "    --artifacts DIR   override the design set\n"
      "    --root DIR        override where models/ and the design live\n"
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
    const bool have_design = !pick_artifacts(root, e.hidden).empty();
    const char *state = !m ? "available"
                           : (have_design ? "ready" : "no design");
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
                pick_artifacts(root, m.hidden).empty() ? "no design" : "ready",
                (long long)m.layers, (long long)m.hidden, m.pooling.c_str(),
                m.mb, m.repo.c_str());
  }

  std::printf(
      "\n  ready      installed, with a matching NPU design -- `serve` runs it\n"
      "  available  not downloaded yet -- `serve` fetches and verifies it\n"
      "  no design  installed, but no design set for this width is present\n"
      "\n  npuembeddings serve <model>\n\n");
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

  // Subcommands, translated into the flag form the rest of main() reads.
  std::vector<std::string> store;
  if (argc > 1) {
    const std::string sub = argv[1];
    const bool is_sub = (sub == "list" || sub == "serve" || sub == "embed" ||
                         sub == "help" || sub == "--help" || sub == "-h");
    if (is_sub) {
      // --root is read before anything else, since it decides where we look
      // for the model we are about to talk about.
      std::string sub_root;
      for (int i = 2; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--root") sub_root = argv[i + 1];
      if (sub_root.empty()) sub_root = default_root(argv[0]);

      if (sub == "help" || sub == "--help" || sub == "-h") {
        print_usage();
        return 0;
      }
      if (sub == "list") {
        print_catalog(sub_root);
        return 0;
      }

      // serve / embed both need a model named as the next positional.
      if (argc < 3 || argv[2][0] == '-') {
        print_catalog(sub_root);
        throw std::runtime_error("`" + sub + "` needs a model name");
      }
      const std::string want = argv[2];
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
          });

      // Which design serves this model. --artifacts still wins if given.
      std::string art;
      for (int i = 2; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--artifacts") art = argv[i + 1];
      if (art.empty()) {
        int64_t hidden = 0;
        try {
          npue::File f(container);
          hidden = f.config_int("hidden");
        } catch (const std::exception &) {
        }
        art = pick_artifacts(sub_root, hidden);
        if (art.empty())
          throw std::runtime_error(
              "no NPU design for hidden " + std::to_string(hidden) +
              " under " + sub_root +
              " -- this release carries designs for the widths it was built "
              "with; export one with tools/export_gemm_rtp.py --hidden " +
              std::to_string(hidden));
      }

      // Defaults that suit a server rather than a measurement. The flag form
      // keeps its conservative --threads 1, because a benchmark that quietly
      // used 24 cores would misreport the per-core claim.
      std::string threads = "24", pipeline = "2";
      for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--threads") threads = argv[i + 1];
        if (std::string(argv[i]) == "--pipeline") pipeline = argv[i + 1];
      }

      store = {argv[0], sub_root,       "--model",    want,
               "--artifacts", art,      "--threads",  threads,
               "--pipeline",  pipeline};
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

  int bench = 0;
  for (int i = 2; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--bench") bench = std::atoi(argv[i + 1]);

  // --artifacts selects which export to load, so two builds of the same
  // designs can be compared in the same session rather than across a rebuild.
  // --artifacts names a design set. It is resolved against both layouts this
  // ships in: the source tree (<root>/runtime/<name>) and an extracted
  // release, where the design sits beside the executable (<root>/<name>, or
  // <root> itself when the name is "."). An absolute path is taken as given.
  // Chosen by which candidate actually CONTAINS a design, so a typo is an
  // error about the design rather than a confusing one about a missing file.
  std::string art_name = "artifacts";
  for (int i = 2; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--artifacts") art_name = argv[i + 1];

  std::string art;
  {
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
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      const auto e = tok.encode(line, max_len);
      for (size_t k = 0; k < e.input_ids.size(); ++k)
        std::printf("%s%d", k ? " " : "", e.input_ids[k]);
      std::printf("\n");
    }
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

  // Fixtures live per model. The flat directory is the pre-multi-model layout
  // and is still honoured so an existing checkout keeps working; the
  // source_sha256 guard below is what makes either location safe.
  std::string val =
      root + "/runtime/artifacts/validation/" + g_model_name;
  if (!std::ifstream(val + "/emb_sum.f32").good())
    val = root + "/runtime/artifacts/validation";
  const bool have_val = std::ifstream(val + "/emb_sum.f32").good();
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
    if (want_sha.empty() || want_sha != got_sha)
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
  // claim rests on. For larger batches the four sequences are TILED to fill the
  // design. That measures throughput honestly (the array does the full work)
  // while making no accuracy claim beyond batch 4, which is where --validate
  // still runs.
  constexpr int64_t kGoldenBatch = 4;
  auto tile = [](const std::vector<float> &v, int64_t reps) {
    std::vector<float> out;
    out.reserve(v.size() * reps);
    for (int64_t r = 0; r < reps; ++r) out.insert(out.end(), v.begin(), v.end());
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
    emb_in = tile(read_f32(val + "/emb_sum.f32",
                           static_cast<size_t>(kGoldenBatch * g_seq * g_hidden)),
                  reps4);
    mask = tile(read_f32(val + "/add_mask.f32",
                         static_cast<size_t>(kGoldenBatch * g_seq)), reps4);
    want = read_f32(val + "/embedding_expected.f32",
                    static_cast<size_t>(kGoldenBatch * g_hidden));
    amask_i = tile(read_f32(val + "/attention_mask.f32",
                            static_cast<size_t>(kGoldenBatch * g_seq)), reps4);
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
      // The staged weights and parameters are the design's, not a lane's.
      e2.s_qkv = enc.s_qkv; e2.s_ao = enc.s_ao;
      e2.s_fu = enc.s_fu; e2.s_fd = enc.s_fd;
      e2.b_qkv = enc.b_qkv; e2.b_ao = enc.b_ao;
      e2.b_fu = enc.b_fu; e2.b_fd = enc.b_fd;
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
    const int64_t mrows = 4, tm = 64;      // design rows, tile m
    std::printf("\n  probe-streams -- %d repeats, no host work, C is %s\n",
                reps, cb == 2 ? "bf16" : "fp32");
    std::printf("    %-10s %6s %6s %6s  %8s  %8s  %9s  %8s  %8s\n",
                "stream", "M", "K", "N", "GMAC", "MB", "us/disp",
                "GMAC/ms", "GB/s");
    for (const auto &st : streams) {
      if (st.batch != batch) continue;
      d_qkv.bind_instr(static_cast<size_t>(st.slot));
      d_qkv.dispatch_only();                       // warm
      const double t0 = now_s();
      for (int r = 0; r < reps; ++r) d_qkv.dispatch_only();
      const double us = (now_s() - t0) / reps * 1e6;
      // tasks/0010's traffic accounting: A re-streamed once per n-block group,
      // B once per row block, C once.
      const double nb_groups = double(st.N) / (48.0 * 8.0) > 0
          ? std::max(1.0, double(st.N) / (48.0 * 8.0)) : 1.0;
      const double row_blocks = double(st.M) / double(tm) / double(mrows);
      const double mb = (double(st.M) * st.K * 2 * nb_groups
                         + double(st.K) * st.N * 2 * row_blocks
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
    npue::Tokenizer tok;
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

    void chunk(Encoder &e, const std::vector<std::string> &texts,
               int64_t base, int64_t take, std::vector<float> &out,
               int64_t *tokens) const {
      const size_t row_floats = static_cast<size_t>(g_seq) * g_hidden;
      const int64_t bt = e.use_tier(take);
      std::vector<float> buf(static_cast<size_t>(bt) * row_floats, 0.f);
      std::vector<float> cmask(static_cast<size_t>(bt) * g_seq, -1.0e30f);
      std::vector<float> cam(static_cast<size_t>(bt) * g_seq, 0.f);
      int64_t ntok = 0;
      for (int64_t b = 0; b < take; ++b) {
        const auto en = tok.encode(texts[base + b], static_cast<int>(g_seq));
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
                             int64_t *tokens = nullptr) {
      std::vector<float> out(texts.size() * g_hidden, 0.f);
      const auto jobs = plan(static_cast<int64_t>(texts.size()));
      std::atomic<int64_t> tok_total{0};
      if (all.size() > 1 && jobs.size() > 1) {
        std::atomic<size_t> next{0};
        std::vector<std::thread> ts;
        auto worker = [&](Encoder *e) {
          for (size_t j = next++; j < jobs.size(); j = next++) {
            int64_t nt = 0;
            chunk(*e, texts, jobs[j].first, jobs[j].second, out, &nt);
            tok_total += nt;
          }
        };
        for (size_t l = 1; l < all.size(); ++l)
          ts.emplace_back([&, l] { worker(all[l]); });
        worker(lead);
        for (auto &th : ts) th.join();
      } else {
        for (const auto &j : jobs) {
          int64_t nt = 0;
          chunk(*lead, texts, j.first, j.second, out, &nt);
          tok_total += nt;
        }
      }
      if (tokens) *tokens = tok_total.load();
      return out;
    }
  };

  auto make_service = [&]() {
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
    auto out = svc.embed(texts);
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

    auto svc = make_service();
    const std::string model_id = g_model_name + "-npu";
    std::printf("  tokenizer  %zu tokens, from the .npue\n",
                svc.tok.vocab_size());
    std::printf("\n  serving http://%s:%d/v1/embeddings   (model %s, seq %lld)\n",
                bind_addr.c_str(), port, model_id.c_str(), (long long)g_seq);
    std::printf("  POST {\"input\": \"text\" | [\"a\",\"b\"], "
                "\"encoding_format\": \"float\"|\"base64\"}\n\n");

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
               "\",\"backend\":\"amd-xdna2-npu\"}";
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

      int64_t n_tokens = 0;
      std::vector<float> emb;
      try {
        emb = svc.embed(texts, &n_tokens);
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
        const float *v = emb.data() + r * g_hidden;
        if (fmt == "base64") {
          out += '"';
          out += npue::http::base64(reinterpret_cast<const uint8_t *>(v),
                                    static_cast<size_t>(g_hidden) * sizeof(float));
          out += '"';
        } else {
          out += '[';
          for (int64_t c = 0; c < g_hidden; ++c) {
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
    std::printf("    host attention (QK^T, A.V)   %8.2f ms  %5.1f%%\n",
                attn * 1e3, attn / wall * 100);
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
    std::printf("    everything else              %8.2f ms  %5.1f%%\n",
                (wall - npu - attn) * 1e3, (wall - npu - attn) / wall * 100);

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

  double num = 0.0, den = 0.0, worst_1mcos = 0.0;
  for (int64_t b = 0; b < kGoldenBatch; ++b) {
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
  std::printf("  %-38s %11.3e\n", "worst 1 - cos vs HuggingFace", worst_1mcos);

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
