//===- npue_pack.cpp ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- build the .npue container from an upstream checkpoint.
// SPDX-License-Identifier: Apache-2.0
//
// WHY THIS EXISTS IN C++ AT ALL
// -----------------------------
// tools/pack_npue.py already does this, and Python at BUILD time is fine by
// this project's rules. But the release does not ship the model: the weights
// belong to sentence-transformers/all-MiniLM-L6-v2, and a 66 MB binary blob
// that a user cannot easily check against the original is a worse deal than
// a two-line download from the canonical source with a sha256 to verify.
//
// That trade only works if preparing the model is easy. A Python step would
// have made "download, unzip, run" false. So the packing is here too, and the
// release ships a script that curls two files and calls this.
//
// TWO IMPLEMENTATIONS OF THE SAME LAYOUT IS A RISK, SO IT IS TESTED
// -----------------------------------------------------------------
// A silent disagreement between this and pack_npue.py would produce
// correctly-sized weights in the wrong order -- the exact failure tasks/0022
// hit, and one that a size check cannot catch. So the gate is not "it looks
// right", it is `--prepare-model` and `pack_npue.py` producing a
// BYTE-IDENTICAL file. tools/verify_pack_parity.py runs that comparison.
//
// The container format is documented in docs/04-model/npue-format.md and
// implemented for reference in tools/npue.py.

#include "npue_pack.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace npue {
namespace {

constexpr uint32_t kAlign = 4096;
constexpr int kMacS = 8, kMacT = 8;      // bf16 MMAC sub-tile on aie2p

// --- safetensors ----------------------------------------------------------
// The format is deliberately simple: 8 bytes of little-endian header length,
// that many bytes of JSON, then the tensor bytes. The JSON maps a name to
// {dtype, shape, data_offsets}. Only F32 appears in this checkpoint, and an
// unexpected dtype is refused rather than reinterpreted.
struct Tensor {
  std::string dtype;
  std::vector<int64_t> shape;
  const uint8_t *data = nullptr;
  size_t bytes = 0;
  const float *f32() const { return reinterpret_cast<const float *>(data); }
  int64_t rows() const { return shape.size() > 1 ? shape[0] : 1; }
  int64_t cols() const { return shape.empty() ? 0 : shape.back(); }
  int64_t count() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

std::string json_str_field(const std::string &s, size_t from, size_t to,
                           const char *key) {
  const std::string k = std::string("\"") + key + "\"";
  size_t i = s.find(k, from);
  if (i == std::string::npos || i > to) return {};
  i = s.find('"', s.find(':', i) + 1) + 1;
  return s.substr(i, s.find('"', i) - i);
}

std::vector<int64_t> json_int_array(const std::string &s, size_t from,
                                    size_t to, const char *key) {
  std::vector<int64_t> out;
  const std::string k = std::string("\"") + key + "\"";
  size_t i = s.find(k, from);
  if (i == std::string::npos || i > to) return out;
  i = s.find('[', i);
  const size_t end = s.find(']', i);
  size_t p = i + 1;
  while (p < end) {
    while (p < end && !(std::isdigit(static_cast<unsigned char>(s[p])) ||
                        s[p] == '-')) ++p;
    if (p >= end) break;
    out.push_back(std::stoll(s.substr(p)));
    while (p < end && (std::isdigit(static_cast<unsigned char>(s[p])) ||
                       s[p] == '-')) ++p;
  }
  return out;
}

std::map<std::string, Tensor> read_safetensors(const std::vector<uint8_t> &buf) {
  if (buf.size() < 8) throw std::runtime_error("safetensors: file too short");
  uint64_t hlen = 0;
  std::memcpy(&hlen, buf.data(), 8);
  if (8 + hlen > buf.size())
    throw std::runtime_error("safetensors: header length exceeds file");
  const std::string js(reinterpret_cast<const char *>(buf.data() + 8),
                       static_cast<size_t>(hlen));
  const uint8_t *base = buf.data() + 8 + hlen;

  std::map<std::string, Tensor> out;
  size_t p = 0;
  while (true) {
    // Each entry is  "name":{...}. Find the next name at brace depth 1.
    const size_t q1 = js.find('"', p);
    if (q1 == std::string::npos) break;
    const size_t q2 = js.find('"', q1 + 1);
    if (q2 == std::string::npos) break;
    const std::string name = js.substr(q1 + 1, q2 - q1 - 1);
    const size_t ob = js.find('{', q2);
    if (ob == std::string::npos) break;
    const size_t cb = js.find('}', ob);
    if (cb == std::string::npos) break;
    p = cb + 1;
    if (name == "__metadata__") continue;

    Tensor t;
    t.dtype = json_str_field(js, ob, cb, "dtype");
    // A non-F32 tensor is not an error by itself -- this checkpoint carries
    // embeddings.position_ids as I64 and nothing reads it. It becomes an
    // error only if something asks for it, which get() enforces. Rejecting
    // the whole file here would refuse a checkpoint that is perfectly usable;
    // ignoring the dtype at read time would reinterpret integers as floats.
    t.shape = json_int_array(js, ob, cb, "shape");
    const auto off = json_int_array(js, ob, cb, "data_offsets");
    if (off.size() != 2)
      throw std::runtime_error("safetensors: " + name + " has no data_offsets");
    t.data = base + off[0];
    t.bytes = static_cast<size_t>(off[1] - off[0]);
    if (t.dtype == "F32" && t.bytes != static_cast<size_t>(t.count()) * 4)
      throw std::runtime_error("safetensors: " + name + " size disagrees with "
                               "its shape");
    out.emplace(name, t);
  }
  return out;
}

// --- sha256 ---------------------------------------------------------------
// The container records the checksum of the checkpoint it was built from, and
// the goldens assert against it. Without it a .npue cannot say which weights
// it came from, which is the whole point of pinning a source.
struct Sha256 {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint64_t len = 0;
  uint8_t buf[64] = {};
  size_t have = 0;

  static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void block(const uint8_t *p) {
    static const uint32_t K[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
      0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
      0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
      0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
      0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
      0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
      0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
      0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
      0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
             (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = ror(w[i-15],7) ^ ror(w[i-15],18) ^ (w[i-15] >> 3);
      const uint32_t s1 = ror(w[i-2],17) ^ ror(w[i-2],19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      const uint32_t S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22);
      const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + mj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }

  void update(const uint8_t *p, size_t n) {
    len += n;
    while (n) {
      const size_t take = std::min(n, size_t(64) - have);
      std::memcpy(buf + have, p, take);
      have += take; p += take; n -= take;
      if (have == 64) { block(buf); have = 0; }
    }
  }

  std::string hex() {
    const uint64_t bits = len * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t zero = 0;
    while (have != 56) update(&zero, 1);
    uint8_t be[8];
    for (int i = 0; i < 8; ++i) be[i] = uint8_t(bits >> (56 - 8 * i));
    len -= 8;                       // the length field is not part of the data
    update(be, 8);
    std::string s;
    static const char *H = "0123456789abcdef";
    for (uint32_t v : h)
      for (int i = 28; i >= 0; i -= 4) s.push_back(H[(v >> i) & 0xF]);
    return s;
  }
};

// --- the container --------------------------------------------------------

// fp32 -> bf16 bits, round-to-nearest-even. Must match tools/npue.py exactly:
// truncation would bias every one of 10.6 M weights toward zero.
inline uint16_t bf16_rne(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return static_cast<uint16_t>(((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF);
}

// The pre-tiling, matching tools/npue.py's tile_b(order="k,n"):
//   [K,N] -> [kb][nb][tk/s][tn/t][s][t]
// Both re-layouts the runtime design would otherwise do at load time are
// absorbed here, which is the whole point of the format.
std::vector<uint16_t> tile_b(const float *mat, int64_t K, int64_t N,
                             int64_t tk, int64_t tn) {
  if (K % tk || N % tn)
    throw std::runtime_error(
        "operand [" + std::to_string(K) + "," + std::to_string(N) +
        "] does not tile evenly by (" + std::to_string(tk) + "," +
        std::to_string(tn) + "): K%tk=" + std::to_string(K % tk) +
        ", N%tn=" + std::to_string(N % tn));
  const int64_t kb_n = K / tk, nb_n = N / tn;
  std::vector<uint16_t> out(static_cast<size_t>(K) * N);
  size_t w = 0;
  for (int64_t kb = 0; kb < kb_n; ++kb)
    for (int64_t nb = 0; nb < nb_n; ++nb)
      for (int64_t si = 0; si < tk / kMacS; ++si)
        for (int64_t ti = 0; ti < tn / kMacT; ++ti)
          for (int64_t s = 0; s < kMacS; ++s)
            for (int64_t t = 0; t < kMacT; ++t) {
              const int64_t r = kb * tk + si * kMacS + s;
              const int64_t c = nb * tn + ti * kMacT + t;
              out[w++] = bf16_rne(mat[r * N + c]);
            }
  return out;
}

struct Entry {
  std::string name, role, dtype, layout_json, layout_hash;
  std::vector<int64_t> shape;
  uint64_t offset = 0, nbytes = 0;
};

class Writer {
public:
  void add(const std::string &name, const void *data, size_t bytes,
           const char *dtype, const char *role,
           const std::vector<int64_t> &shape,
           const std::string &layout_json = {},
           const std::string &layout_hash = {}) {
    Entry e;
    e.name = name;
    e.role = role;
    e.dtype = dtype;
    e.shape = shape;
    e.offset = offset_;
    e.nbytes = bytes;
    e.layout_json = layout_json;
    e.layout_hash = layout_hash;
    entries_.push_back(e);
    const uint8_t *p = static_cast<const uint8_t *>(data);
    blob_.insert(blob_.end(), p, p + bytes);
    offset_ += bytes;
    // Pad AFTER each tensor so the next one starts aligned.
    const uint64_t pad = (kAlign - (offset_ % kAlign)) % kAlign;
    blob_.insert(blob_.end(), static_cast<size_t>(pad), 0);
    offset_ += pad;
  }

  void write(const std::string &path, const std::string &config_json) const {
    std::string js = "{\"config\":" + config_json + ",\"tensors\":[";
    for (size_t i = 0; i < entries_.size(); ++i) {
      const Entry &e = entries_[i];
      if (i) js += ',';
      js += "{\"name\":\"" + e.name + "\",\"role\":\"" + e.role +
            "\",\"dtype\":\"" + e.dtype + "\",\"logical_shape\":[";
      for (size_t k = 0; k < e.shape.size(); ++k)
        js += (k ? "," : "") + std::to_string(e.shape[k]);
      js += "],\"padded_shape\":[";
      for (size_t k = 0; k < e.shape.size(); ++k)
        js += (k ? "," : "") + std::to_string(e.shape[k]);
      js += "],\"offset\":" + std::to_string(e.offset) +
            ",\"nbytes\":" + std::to_string(e.nbytes);
      if (!e.layout_json.empty())
        js += ",\"layout\":" + e.layout_json +
              ",\"layout_hash\":\"" + e.layout_hash + "\"";
      js += "}";
    }
    js += "]}";

    const uint64_t json_offset = 64;
    uint64_t data_offset = json_offset + js.size();
    data_offset += (kAlign - (data_offset % kAlign)) % kAlign;

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    uint8_t head[64] = {};
    std::memcpy(head, "NPUE", 4);
    const uint32_t version = 1, arch = 0, flags = 1;   // FLAG_PRETILED
    std::memcpy(head + 4, &version, 4);
    std::memcpy(head + 8, &arch, 4);
    std::memcpy(head + 12, &flags, 4);
    const uint64_t jlen = js.size();
    std::memcpy(head + 16, &json_offset, 8);
    std::memcpy(head + 24, &jlen, 8);
    std::memcpy(head + 32, &data_offset, 8);
    std::memcpy(head + 40, &offset_, 8);
    f.write(reinterpret_cast<char *>(head), 64);
    f.write(js.data(), static_cast<std::streamsize>(js.size()));
    const std::string pad(static_cast<size_t>(data_offset - json_offset -
                                              js.size()), '\0');
    f.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    f.write(reinterpret_cast<const char *>(blob_.data()),
            static_cast<std::streamsize>(blob_.size()));
    if (!f) throw std::runtime_error("write failed: " + path);
  }

  size_t count() const { return entries_.size(); }
  uint64_t data_bytes() const { return offset_; }

private:
  std::vector<Entry> entries_;
  std::vector<uint8_t> blob_;
  uint64_t offset_ = 0;
};

std::vector<uint8_t> slurp(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  f.read(reinterpret_cast<char *>(buf.data()), n);
  return buf;
}

}  // namespace

// The one C++ SHA-256, exposed. The downloader must verify a checkpoint with
// EXACTLY the implementation that later records `source_sha256` into the
// container -- a second copy is how the Python side ended up with four.
// Streamed rather than slurped: `model.safetensors` is 438 MB for bge-base
// and there is no reason to hold it twice.
std::string sha256_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  Sha256 s;
  std::vector<char> buf(1 << 20);
  while (f) {
    f.read(buf.data(), (std::streamsize)buf.size());
    const std::streamsize got = f.gcount();
    if (got > 0)
      s.update(reinterpret_cast<const uint8_t *>(buf.data()), (size_t)got);
  }
  return s.hex();
}

// The [K,N] operand for a GEMM: transpose the checkpoint's [N,K] weight,
// optionally fold a scale, convert and pre-tile.
static void add_gemm_b(Writer &w, const std::string &name, const Tensor &t,
                       int64_t tk, int64_t tn, const std::string &layout_json,
                       const std::string &layout_hash, float fold = 1.0f,
                       int64_t fold_cols = 0) {
  const int64_t N = t.rows(), K = t.cols();     // checkpoint stores [out, in]
  std::vector<float> m(static_cast<size_t>(K) * N);
  const float *src = t.f32();
  for (int64_t r = 0; r < K; ++r)
    for (int64_t c = 0; c < N; ++c) {
      float v = src[c * K + r];                 // transpose to [K, N]
      if (fold != 1.0f && c < fold_cols) v = v * fold;
      m[r * N + c] = v;
    }
  const auto tiled = tile_b(m.data(), K, N, tk, tn);
  w.add(name, tiled.data(), tiled.size() * 2, "BF16", "gemm_b", {K, N},
        layout_json, layout_hash);
}

Layout gemm_b_layout(int64_t tile_k, int64_t tile_n, int64_t mac_s,
                     int64_t mac_t) {
  const std::string k = std::to_string(tile_k), n = std::to_string(tile_n);
  const std::string s = std::to_string(mac_s), tt = std::to_string(mac_t);
  Layout L;
  // Insertion order, matching tools/npue.py's dict literal.
  L.json = "{\"kind\":\"block_panel\",\"tile_k\":" + k + ",\"tile_n\":" + n +
           ",\"order\":\"k,n,kt,nt\",\"inner\":\"s,t\"" +
           ",\"mac_s\":" + s + ",\"mac_t\":" + tt + ",\"dtype\":\"BF16\"}";
  // json.dumps(..., sort_keys=True, separators=(",", ":")) -- keys sorted
  // alphabetically: dtype, inner, kind, mac_s, mac_t, order, tile_k, tile_n.
  const std::string canonical =
      "{\"dtype\":\"BF16\",\"inner\":\"s,t\",\"kind\":\"block_panel\",\"mac_s\":" + s +
      ",\"mac_t\":" + tt + ",\"order\":\"k,n,kt,nt\",\"tile_k\":" + k +
      ",\"tile_n\":" + n + "}";
  Sha256 h;
  h.update(reinterpret_cast<const uint8_t *>(canonical.data()),
           canonical.size());
  L.hash = h.hex();
  return L;
}

void prepare_model(const std::string &safetensors, const std::string &vocab,
                   const std::string &config_json_path,
                   const std::string &pooling,
                   const std::string &source_repo,
                   const std::string &out, const std::string &source_sha,
                   const std::string &layout_json,
                   const std::string &layout_hash,
                   int64_t tile_k, int64_t tile_n, int64_t max_seq,
                   void (*log)(const std::string &)) {
  const auto st_buf = slurp(safetensors);
  const auto src = read_safetensors(st_buf);
  std::string sha = source_sha;
  if (sha.empty()) {
    Sha256 s;
    s.update(st_buf.data(), st_buf.size());
    sha = s.hex();
  }
  auto get = [&](const std::string &n) -> const Tensor & {
    auto it = src.find(n);
    if (it == src.end())
      throw std::runtime_error("checkpoint has no tensor '" + n + "'");
    if (it->second.dtype != "F32")
      throw std::runtime_error("checkpoint tensor '" + n + "' is " +
                               it->second.dtype + "; this packer reads F32");
    return it->second;
  };

  // The architecture is read from config.json rather than assumed, so a
  // different BERT-family checkpoint fails loudly instead of silently
  // packing the wrong shapes.
  const std::string cfg(reinterpret_cast<const char *>(
      slurp(config_json_path).data()), slurp(config_json_path).size());
  auto cfg_int = [&](const char *key) -> int64_t {
    const std::string k = std::string("\"") + key + "\"";
    const size_t i = cfg.find(k);
    if (i == std::string::npos)
      throw std::runtime_error(std::string("config.json has no ") + key);
    return std::stoll(cfg.substr(cfg.find(':', i) + 1));
  };
  const int64_t L = cfg_int("num_hidden_layers");
  const int64_t H = cfg_int("num_attention_heads");
  const int64_t hidden = cfg_int("hidden_size");
  const int64_t inter = cfg_int("intermediate_size");
  const int64_t vocab_size = cfg_int("vocab_size");
  const int64_t type_vocab = cfg_int("type_vocab_size");
  const int64_t head_dim = hidden / H;
  // FLOAT, not double. numpy multiplies a float32 array by a Python float in
  // float32; computing in double and rounding once at the end is a DIFFERENT
  // rounding, and it moved 86 of 2304 bias values by 1 ULP -- found only
  // because the parity test compares bytes rather than tolerances.
  const float scale = static_cast<float>(1.0 / std::sqrt(
      static_cast<double>(head_dim)));

  std::ostringstream cj;
  cj.precision(17);
  cj << "{\"arch\":\"bert_abs_gelu_postln\""
     << ",\"source_repo\":\"" << source_repo << "\""
     << ",\"source_sha256\":\"" << sha << "\""
     << ",\"num_layers\":" << L << ",\"num_heads\":" << H
     << ",\"hidden\":" << hidden << ",\"head_dim\":" << head_dim
     << ",\"intermediate\":" << inter
     << ",\"layer_norm_eps\":1e-12"
     << ",\"vocab_size\":" << vocab_size
     << ",\"max_seq_len\":" << max_seq
     << ",\"pooling\":\"" << pooling << "\",\"l2_normalize\":true"
     << ",\"activation\":\"gelu_erf_exact\""
     << ",\"tile_k\":" << tile_k << ",\"tile_n\":" << tile_n
     << ",\"mac_s\":" << kMacS << ",\"mac_t\":" << kMacT
     << ",\"fusions\":{\"qkv_fused\":true,\"transposed_to_kn\":true,"
        "\"qk_scale_folded_into_q\":true,\"gemm_operands_bf16\":true,"
        "\"biases_and_layernorm_fp32\":true,"
        "\"position_embeddings_presliced_to\":" << max_seq << "}"
     << ",\"not_implemented\":[\"pooler.dense (unused by "
        "sentence-transformers)\"]}";

  Writer w;
  auto add_f32 = [&](const std::string &name, const Tensor &t,
                     const char *role, const std::vector<int64_t> &shape,
                     int64_t limit_rows = 0) {
    const size_t n = static_cast<size_t>(
        limit_rows ? limit_rows * t.cols() : t.count());
    w.add(name, t.data, n * 4, "F32", role, shape);
  };

  add_f32("embeddings.word", get("embeddings.word_embeddings.weight"),
          "embedding", {vocab_size, hidden});
  add_f32("embeddings.position", get("embeddings.position_embeddings.weight"),
          "embedding", {max_seq, hidden}, max_seq);
  add_f32("embeddings.token_type",
          get("embeddings.token_type_embeddings.weight"), "embedding",
          {type_vocab, hidden});
  add_f32("embeddings.ln.weight", get("embeddings.LayerNorm.weight"),
          "layernorm", {hidden});

  const auto vb = slurp(vocab);
  w.add("tokenizer.vocab", vb.data(), vb.size(), "U8", "tokenizer",
        {static_cast<int64_t>(vb.size())});

  add_f32("embeddings.ln.bias", get("embeddings.LayerNorm.bias"), "layernorm",
          {hidden});

  for (int64_t i = 0; i < L; ++i) {
    const std::string p = "encoder.layer." + std::to_string(i) + ".";
    const std::string sa = p + "attention.self.";
    const std::string ao = p + "attention.output.";
    const std::string tag = "layer." + std::to_string(i) + ".";

    // QKV fused along `out`, transposed to [K, N], and 1/sqrt(head_dim)
    // folded into the Q block only -- the first `hidden` columns.
    {
      const Tensor &q = get(sa + "query.weight");
      const Tensor &k = get(sa + "key.weight");
      const Tensor &v = get(sa + "value.weight");
      const int64_t N = 3 * hidden;
      std::vector<float> m(static_cast<size_t>(hidden) * N);
      const Tensor *parts[3] = {&q, &k, &v};
      for (int b = 0; b < 3; ++b)
        for (int64_t o = 0; o < hidden; ++o)
          for (int64_t in = 0; in < hidden; ++in) {
            float val = parts[b]->f32()[o * hidden + in];
            if (b == 0) val = val * scale;             // float32 throughout
            m[in * N + b * hidden + o] = val;
          }
      const auto tiled = tile_b(m.data(), hidden, N, tile_k, tile_n);
      w.add(tag + "qkv", tiled.data(), tiled.size() * 2, "BF16", "gemm_b",
            {hidden, N}, layout_json, layout_hash);

      std::vector<float> bias(static_cast<size_t>(N));
      const char *bn[3] = {"query.bias", "key.bias", "value.bias"};
      for (int b = 0; b < 3; ++b) {
        const float *s = get(sa + bn[b]).f32();
        for (int64_t o = 0; o < hidden; ++o)
          bias[b * hidden + o] = b == 0 ? s[o] * scale : s[o];
      }
      w.add(tag + "qkv.bias", bias.data(), bias.size() * 4, "F32", "bias", {N});
    }

    add_gemm_b(w, tag + "attn_out", get(ao + "dense.weight"), tile_k, tile_n,
               layout_json, layout_hash);
    add_f32(tag + "attn_out.bias", get(ao + "dense.bias"), "bias", {hidden});
    add_f32(tag + "ln1.weight", get(ao + "LayerNorm.weight"), "layernorm",
            {hidden});
    add_f32(tag + "ln1.bias", get(ao + "LayerNorm.bias"), "layernorm",
            {hidden});

    add_gemm_b(w, tag + "ffn_up", get(p + "intermediate.dense.weight"),
               tile_k, tile_n, layout_json, layout_hash);
    add_f32(tag + "ffn_up.bias", get(p + "intermediate.dense.bias"), "bias",
            {inter});
    add_gemm_b(w, tag + "ffn_down", get(p + "output.dense.weight"),
               tile_k, tile_n, layout_json, layout_hash);
    add_f32(tag + "ffn_down.bias", get(p + "output.dense.bias"), "bias",
            {hidden});
    add_f32(tag + "ln2.weight", get(p + "output.LayerNorm.weight"),
            "layernorm", {hidden});
    add_f32(tag + "ln2.bias", get(p + "output.LayerNorm.bias"), "layernorm",
            {hidden});
  }

  w.write(out, cj.str());
  if (log) {
    std::ostringstream s;
    s << "  packed " << w.count() << " tensors, "
      << (w.data_bytes() / 1e6) << " MB of tensor data";
    log(s.str());
  }
}

}  // namespace npue
