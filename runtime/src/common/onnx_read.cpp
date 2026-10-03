//===- onnx_read.cpp ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- read an ONNX checkpoint under the CHECKPOINT's own names.
// SPDX-License-Identifier: Apache-2.0
//
// The C++ half of tools/lib/onnx_weights.py. Every decision recorded there --
// the three name-recovery tiers, the orientation rules, the refusal to guess
// when they disagree, the digest over graph AND side files -- is implemented
// here again rather than expressed differently, because
// tools/verify/verify_pack_parity.py's gate is BYTE EQUALITY and two
// independent readings of one file would eventually produce two correct
//-looking containers with different contents.
//
// See onnx_read.hpp for the field-number table and why it is checked rather
// than trusted.
//===----------------------------------------------------------------------===//

#include "common/onnx_read.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <set>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace npue {
namespace {

// ---------------------------------------------------------------------------
// Protobuf wire format.
// ---------------------------------------------------------------------------

// LEB128 -> value, advancing `i` past it. An unterminated varint is refused
// rather than folded into a plausible number: protobuf itself would keep
// reading, and the answer to a wrong varint is a wrong field LENGTH, which
// silently desynchronises everything after it.
uint64_t read_varint(const uint8_t *b, size_t n, size_t &i) {
  uint64_t v = 0;
  int shift = 0;
  while (true) {
    if (i >= n) throw std::runtime_error("onnx: truncated varint");
    const uint8_t c = b[i++];
    v |= static_cast<uint64_t>(c & 0x7F) << shift;
    if (c < 0x80) return v;
    shift += 7;
    if (shift > 63)
      throw std::runtime_error("onnx: unterminated varint at byte " +
                               std::to_string(i));
  }
}

struct Field {
  int no = 0;
  int wt = 0;
  size_t ps = 0;
  size_t pe = 0;
};

// Walk the fields of one message.
//
// wire 2 yields the payload with the length already applied, which is what
// makes skipping a 500 MB raw_data block free: read a length, jump. Nothing
// is ever copied until a tensor is actually asked for.
//
// The end bound is the enclosing message's, but a payload is only checked
// against the FILE's end -- a sub-message may legitimately extend past where
// its parent's iteration started, and rejecting that would refuse files
// protobuf reads happily. Malformed input is refused rather than clamped:
// reading a few bytes of the next field as a name produces a container full
// of tensors that are all the wrong size.
class FieldIter {
 public:
  FieldIter(const uint8_t *b, size_t n, size_t start, size_t end)
      : b_(b), n_(n), i_(start), end_(end) {}

  bool next(Field &f) {
    if (i_ >= end_) return false;
    const uint64_t key = read_varint(b_, n_, i_);
    f.no = static_cast<int>(key >> 3);
    f.wt = static_cast<int>(key & 7);
    switch (f.wt) {
      case 0:
        f.ps = i_;
        read_varint(b_, n_, i_);  // the value itself; `pe` is its byte end
        f.pe = i_;
        break;
      case 1:
        f.ps = i_;
        if (n_ - i_ < 8) throw std::runtime_error("onnx: truncated 64-bit field");
        f.pe = f.ps + 8;
        i_ = f.pe;
        break;
      case 2: {
        const uint64_t len = read_varint(b_, n_, i_);
        f.ps = i_;
        if (len > n_ - i_)
          throw std::runtime_error(
              "onnx: length-delimited field runs past the end of the file");
        f.pe = f.ps + static_cast<size_t>(len);
        i_ = f.pe;
        break;
      }
      case 5:
        f.ps = i_;
        if (n_ - i_ < 4) throw std::runtime_error("onnx: truncated 32-bit field");
        f.pe = f.ps + 4;
        i_ = f.pe;
        break;
      default:
        throw std::runtime_error("onnx: wire type " + std::to_string(f.wt) +
                                 " at byte " + std::to_string(i_) +
                                 " is not protobuf");
    }
    return true;
  }

 private:
  const uint8_t *b_;
  size_t n_, i_, end_;
};

std::string text(const uint8_t *b, const Field &f) {
  // Bytes, not a decoded-and-re-encoded string: a name is compared, stripped
  // and replaced as bytes, and copying them verbatim cannot lose an
  // impression a UTF-8 round trip could.
  return std::string(reinterpret_cast<const char *>(b + f.ps), f.pe - f.ps);
}

std::string replace_all(std::string s, const std::string &from,
                        const std::string &to) {
  if (from.empty()) return s;
  size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
  return s;
}

bool starts_with(const std::string &s, const std::string &p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

bool ends_with(const std::string &s, const std::string &p) {
  return s.size() >= p.size() &&
         s.compare(s.size() - p.size(), p.size(), p) == 0;
}

std::string abspath(const std::string &p) {
  std::error_code ec;
  const std::filesystem::path a = std::filesystem::absolute(p, ec);
  return (ec ? std::filesystem::path(p) : a).lexically_normal().string();
}

// ---------------------------------------------------------------------------
// The graph.
// ---------------------------------------------------------------------------

struct Node {
  std::vector<std::string> inputs, outputs;
  std::string op;
  std::map<std::string, int64_t> attrs;
  std::vector<size_t> out_consumers;
};

// Where a payload is, in one of the four shapes a TensorProto can carry.
enum class LocKind { kNone, kRaw, kExt, kInline };

struct Init {
  std::string name, orig, how, ext_location;
  std::string ext_path;              // resolved, for source_files()
  std::vector<int64_t> dims;
  int dtype = 0;
  LocKind kind = LocKind::kNone;
  size_t raw_off = 0;
  int64_t ext_off = 0;
  int inl_no = 0, inl_wt = 0;
  size_t inl_ps = 0, inl_pe = 0;
  std::string tag;                   // .npue tag, once known
  int item = 0;                      // bytes per element
  bool transpose = false;
  std::vector<size_t> uses;
};

// AttributeProto -> (name, value, is_int). transA/transB on Gemm are the
// entire reason attributes are decoded: they say whether the weight sitting in
// the node is already [out, in]. Graphs, tensors, floats and strings are
// skipped -- none of them can change an answer this reader gives.
void parse_attr(const uint8_t *b, size_t n, const Field &f, std::string *name,
                int64_t *ival, bool *is_int) {
  name->clear();
  *ival = 0;
  *is_int = false;
  Field g;
  FieldIter it(b, n, f.ps, f.pe);
  while (it.next(g)) {
    if (g.no == 1 && g.wt == 2) {
      *name = text(b, g);
    } else if (g.no == 3 && g.wt == 0) {
      size_t p = g.ps;
      const uint64_t v = read_varint(b, n, p);
      // int64 arrives as a 64-bit two's complement varint, so a negative
      // value comes across as its unsigned spelling. The modular narrowing
      // below is what undoes it; C++20 makes it defined, and every compiler
      // this project builds with has always done exactly this.
      *ival = static_cast<int64_t>(v);
      *is_int = true;
    } else if (g.no == 20 && g.wt == 0) {
      size_t p = g.ps;
      *is_int = read_varint(b, n, p) == 2;  // AttributeType.INT == 2
    }
  }
}

void parse_node(const uint8_t *b, size_t n, const Field &f, Node *out) {
  Field g;
  FieldIter it(b, n, f.ps, f.pe);
  while (it.next(g)) {
    if (g.no == 1 && g.wt == 2) {
      out->inputs.push_back(text(b, g));
    } else if (g.no == 2 && g.wt == 2) {
      out->outputs.push_back(text(b, g));
    } else if (g.no == 4 && g.wt == 2) {
      out->op = text(b, g);
    } else if (g.no == 5 && g.wt == 2) {
      std::string k;
      int64_t v = 0;
      bool is_int = false;
      parse_attr(b, n, g, &k, &v, &is_int);
      if (!k.empty() && is_int) out->attrs[k] = v;
    }
  }
}

// The five inline payload fields -> (.npue tag, bytes per element).
// Reached only for tensors small enough that the exporter wrote typed fields
// instead of raw_data, which is how torch emits scalar constants.
bool inline_tag(int no, std::string *tag, int *item) {
  switch (no) {
    case 4:  *tag = "F32";  *item = 4; return true;  // float_data
    case 5:  *tag = "I32";  *item = 4; return true;  // int32_data
    case 7:  *tag = "I64";  *item = 8; return true;  // int64_data
    case 10: *tag = "F64";  *item = 8; return true;  // double_data
    case 11: *tag = "I64";  *item = 8; return true;  // uint64_data, never a weight
    default: return false;
  }
}

// Whether a PACKED payload in field `no` is LEB128 rather than native bytes.
// True for exactly the integer fields. A packed repeated integer is one varint
// per element; a packed repeated float/double is fixed-width values laid end to
// end. Both arrive as protobuf wire type 2, so the wire type alone cannot say
// how to read the bytes -- and reading a packed int32_data as native
// little-endian returns a DIFFERENT NUMBER for every value >= 256, and reads
// past the end of the field for a short one. A uint8 zero_point of 128 is
// encoded `80 01`, which is that exact case. Mirrors _INLINE_VARINT_FIELDS in
// tools/lib/onnx_weights.py; the two must agree or the packer and the runtime
// read the same checkpoint differently.
inline bool inline_is_varint(int no) {
  return no == 5 || no == 7 || no == 11;
}

// ONNX TensorProto.DataType -> the .npue tag it maps to. info() hands
// back the .npue tag on purpose: callers that switch on `dt == "BF16"`
// were written for the container reader and must not have to know which
// container they are standing in. ONNX also has UINT16/UINT32/UINT64/STRING;
// the tag set has no entry for those, so they are refused rather than
// retyped -- inventing a tag would let a mismatched tensor through a shape
// check that exists to catch exactly that.
bool onnx_tag(int code, std::string *tag, int *item) {
  switch (code) {
    case 1:  *tag = "F32";  *item = 4; return true;   // FLOAT
    case 2:  *tag = "U8";   *item = 1; return true;   // UINT8
    case 3:  *tag = "I8";   *item = 1; return true;   // INT8
    case 5:  *tag = "I16";  *item = 2; return true;   // INT16
    case 6:  *tag = "I32";  *item = 4; return true;   // INT32
    case 7:  *tag = "I64";  *item = 8; return true;   // INT64
    case 9:  *tag = "BOOL"; *item = 1; return true;   // BOOL
    case 10: *tag = "F16";  *item = 2; return true;   // FLOAT16
    case 11: *tag = "F64";  *item = 8; return true;   // DOUBLE
    case 16: *tag = "BF16"; *item = 2; return true;   // BFLOAT16
    default: return false;
  }
}

void parse_tensor(const uint8_t *b, size_t n, const Field &f, Init *t) {
  Field g;
  FieldIter it(b, n, f.ps, f.pe);
  bool have_raw = false;
  while (it.next(g)) {
    if (g.no == 1) {                                   // dims: packed or not
      if (g.wt == 0) {
        size_t p = g.ps;
        t->dims.push_back(
            static_cast<int64_t>(read_varint(b, n, p)));
      } else {
        size_t p = g.ps;
        while (p < g.pe)
          t->dims.push_back(
              static_cast<int64_t>(read_varint(b, n, p)));
      }
    } else if (g.no == 2 && g.wt == 0) {
      size_t p = g.ps;
      t->dtype = static_cast<int>(read_varint(b, n, p));
    } else if (g.no == 8 && g.wt == 2) {
      t->name = text(b, g);
    } else if (g.no == 9 && g.wt == 2) {                // raw_data
      t->kind = LocKind::kRaw;
      t->raw_off = g.ps;
      have_raw = true;
    } else if (g.no == 13 && g.wt == 2) {               // external_data
      // One field-13 message PER KEY: `location`, `offset` and `length` are
      // three separate entries, not three fields of one. Keeping only the
      // last would drop `location` half the time and then read the side file
      // from the wrong place.
      std::string k, v;
      Field h;
      FieldIter kt(b, n, g.ps, g.pe);
      while (kt.next(h)) {
        if (h.no == 1 && h.wt == 2) k = text(b, h);
        else if (h.no == 2 && h.wt == 2) v = text(b, h);
      }
      if (k == "location") t->ext_location = v;
      else if (k == "offset") {
        try {
          t->ext_off = std::stoll(v);
        } catch (...) {
          throw std::runtime_error("onnx: external_data offset '" + v +
                                   "' is not a number");
        }
      }
    } else if ((g.no == 4 || g.no == 5 || g.no == 7 || g.no == 10 ||
                g.no == 11) && !have_raw && t->kind == LocKind::kNone) {
      t->kind = LocKind::kInline;
      t->inl_no = g.no;
      t->inl_wt = g.wt;
      t->inl_ps = g.ps;
      t->inl_pe = g.pe;
    }
  }
  // An external-data path in the graph is RELATIVE to the graph file, and the
  // resolution happens in _build() against the real base directory. Doing it
  // here would produce a path relative to the current working directory,
  // which silently reads nothing (or the wrong file) for every model that
  // keeps its weights in a side file.
}

// ---------------------------------------------------------------------------
// Name recovery and orientation. Verbatim twins of _recover / _transpose_for.
// ---------------------------------------------------------------------------

std::string recover(const std::string &name, const std::vector<size_t> &uses,
                    const std::vector<Node> &nodes, std::string *how) {
  how->clear();
  if (!starts_with(name, "onnx::")) {
    // tier 1: dynamo-style exports spell Linear weights `...proj.MatMul.weight`
    const size_t p = name.find(".MatMul.");
    if (p != std::string::npos) {
      *how = "tier1-drop-.MatMul";
      return name.substr(0, p) + "." + name.substr(p + 8);
    }
    return name;
  }

  for (size_t ui : uses) {
    const Node &node = nodes[ui];
    if (node.op != "MatMul" && node.op != "Gemm") continue;
    // tier 2: the bias paired with this weight keeps its real name
    for (size_t ci : node.out_consumers) {
      const Node &other = nodes[ci];
      if (other.op != "Add") continue;
      for (const std::string &cand : other.inputs) {
        if (ends_with(cand, ".bias") && cand != name) {
          *how = "tier2-bias-anchor";
          return cand.substr(0, cand.size() - 5) + ".weight";
        }
      }
    }
    // tier 3: `/a/b/c/MatMul_output_0` -> `a.b.c.weight`
    if (!node.outputs.empty()) {
      const std::string &out = node.outputs[0];
      std::vector<std::string> parts;
      size_t p = 0;
      while (p <= out.size()) {
        const size_t q = out.find('/', p);
        const std::string seg =
            out.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (!seg.empty()) parts.push_back(seg);
        if (q == std::string::npos) break;
        p = q + 1;
      }
      if (!out.empty() && out[0] == '/' && parts.size() >= 2 &&
          !starts_with(parts.back(), "onnx::")) {
        std::string joined;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
          if (i) joined += ".";
          joined += parts[i];
        }
        *how = "tier3-node-path";
        return joined + ".weight";
      }
    }
  }
  return name;
}

// True when the stored tensor is [in, out] and must become [out, in].
//
// Decided by how the graph USES the tensor, because that is the only thing
// that says what the layout is: MatMul's second operand is [in, out] by
// definition of the op, Gemm states it in transA/transB, and a Gather operand
// is an embedding table already laid out [vocab, dim] == [out, in].
//
// When the uses disagree -- what a tied lm_head produces, one initializer
// serving both a lookup and the logits MatMul -- the bias settles it, because
// bias.shape[0] IS out. With no bias and no agreement this REFUSES: a wrong
// answer builds a container that passes its own layout hash and emits
// embeddings that look like noise, which is the worst failure mode this
// project has.
bool transpose_for(const std::vector<size_t> &uses,
                   const std::vector<Node> &nodes, const std::string &orig,
                   const std::vector<int64_t> &dims, const int64_t *bias0) {
  if (dims.size() != 2) return false;

  bool saw_t = false, saw_f = false;
  auto add = [&](bool v) { (v ? saw_t : saw_f) = true; };
  for (size_t ui : uses) {
    const Node &node = nodes[ui];
    if (node.op == "MatMul") {
      const auto it = std::find(node.inputs.begin(), node.inputs.end(), orig);
      if (it != node.inputs.end())
        add(std::distance(node.inputs.begin(), it) == 1);
    } else if (node.op == "Gemm") {
      const auto a = node.attrs.find("transB");
      add(!(a != node.attrs.end() && a->second != 0));
    } else if (node.op == "Gather") {
      add(false);
    }
  }

  const int nflags = (saw_t ? 1 : 0) + (saw_f ? 1 : 0);
  if (nflags == 1) return saw_t;
  if (nflags > 1) {
    if (!bias0)
      throw std::runtime_error(
          "onnx: layout of '" + orig + "' is ambiguous (MatMul, Gemm and "
          "Gather all consume it) and there is no bias to settle it; "
          "refusing to guess");
    return dims[0] != *bias0;
  }
  if (bias0) {
    if (dims[0] == *bias0) return false;
    if (dims[1] == *bias0) return true;
    throw std::runtime_error(
        "onnx: '" + orig + "' has shape [" + std::to_string(dims[0]) + "," +
        std::to_string(dims[1]) + "] but its bias has " +
        std::to_string(*bias0) + " elements; neither orientation matches");
  }
  return false;
}

// ---------------------------------------------------------------------------
// Widening and transposing. Both happen in ONE pass over the payload, so a
// checkpoint that is both F16 and [in, out] is read once rather than widened
// into a temporary and then transposed into a second copy -- at 3 GB for
// whisper-large-v3 that difference is the difference between fitting and not.
// ---------------------------------------------------------------------------

float bf16_to_f32(const uint8_t *p) {
  // bf16 IS the top half of an fp32: shift, done. No rounding, no special
  // cases -- NaN and Inf patterns come across unchanged.
  const uint32_t bits = static_cast<uint32_t>(p[0]) << 24 |
                        static_cast<uint32_t>(p[1]) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

float f16_to_f32(const uint8_t *p) {
  uint16_t h;
  std::memcpy(&h, p, 2);
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;                                   // +-0
    } else {
      // Subnormal half: normalise it into a normal float. Its own arm,
      // because getting the shift wrong is silent -- the value stays finite
      // and merely wrong.
      uint32_t m = man;
      int shift = 0;
      while (!(m & 0x400u)) { m <<= 1; ++shift; }
      m &= 0x3FFu;
      bits = sign | ((127 - 15 - shift + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (man << 13);         // Inf / NaN
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

}  // namespace

// ---------------------------------------------------------------------------
// Mapping.
// ---------------------------------------------------------------------------

struct OnnxWeights::Mapping {
  std::string path;
  size_t size = 0;
#ifdef _WIN32
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE map = nullptr;
  const uint8_t *base = nullptr;
#else
  void *addr = nullptr;
  const uint8_t *base = nullptr;
#endif

  void open(const std::string &p) {
    path = p;
#ifdef _WIN32
    const std::wstring wpath(p.begin(), p.end());
    file = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
      throw std::runtime_error("cannot open " + p);
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(file, &sz))
      throw std::runtime_error("cannot stat " + p);
    size = static_cast<size_t>(sz.QuadPart);
    if (size == 0) return;   // an empty file maps to nothing, not to an error
    map = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map) throw std::runtime_error("cannot map " + p);
    base = static_cast<const uint8_t *>(
        MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0));
    if (!base) throw std::runtime_error("cannot view " + p);
#else
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + p);
    struct stat st {};
    if (::fstat(fd, &st) < 0) {
      ::close(fd);
      throw std::runtime_error("cannot stat " + p);
    }
    size = static_cast<size_t>(st.st_size);
    if (size == 0) {
      ::close(fd);
      return;
    }
    void *m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);   // the mapping outlives the descriptor, as mmap intends
    if (m == MAP_FAILED)
      throw std::runtime_error("cannot mmap " + p);
    addr = m;
    base = static_cast<const uint8_t *>(m);
#endif
  }

  ~Mapping() {
#ifdef _WIN32
    if (base) UnmapViewOfFile(base);
    if (map) CloseHandle(map);
    if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
    if (addr) ::munmap(addr, size);
#endif
  }
};

const uint8_t *OnnxWeights::map_file(const std::string &p, size_t *size) {
  for (const auto &m : maps_) {
    if (m->path == p) {
      if (size) *size = m->size;
      return m->base;
    }
  }
  auto m = std::unique_ptr<Mapping>(new Mapping);
  m->open(p);
  const uint8_t *base = m->base;
  if (size) *size = m->size;
  maps_.push_back(std::move(m));
  return base;
}

// ---------------------------------------------------------------------------
// _build.
// ---------------------------------------------------------------------------

OnnxWeights::OnnxWeights(const std::string &graph_path,
                         const OnnxReadOptions &opts)
    : path_(graph_path) {
  const std::string base = std::filesystem::path(abspath(path_)).parent_path().string();
  size_t n = 0;
  const uint8_t *b = map_file(path_, &n);

  // ModelProto.graph = 7.
  size_t g_start = 0, g_end = 0;
  bool have_graph = false;
  {
    Field f;
    FieldIter it(b, n, 0, n);
    while (it.next(f)) {
      if (f.no == 7 && f.wt == 2) {
        g_start = f.ps;
        g_end = f.pe;
        have_graph = true;
        break;
      }
    }
  }
  if (!have_graph)
    throw std::runtime_error(path_ + ": no graph in this ONNX file");

  std::vector<Node> nodes;
  std::vector<Init> inits;
  {
    Field f;
    FieldIter it(b, n, g_start, g_end);
    while (it.next(f)) {
      if (f.wt != 2) continue;
      if (f.no == 1) {
        nodes.emplace_back();
        parse_node(b, n, f, &nodes.back());
      } else if (f.no == 5) {
        inits.emplace_back();
        parse_tensor(b, n, f, &inits.back());
      }
    }
  }

  // Two walks over the graph:
  //   by_input      initializer name -> nodes taking it. An initializer uses
  //                 this to find the ops that consume it, which is how both
  //                 the name (tier 2/3) and the layout are read.
  //   out_consumers node -> nodes taking ITS OUTPUT. Tier 2 walks one hop
  //                 forward from a MatMul to the Add carrying the bias that
  //                 pairs with the weight; without this direction it reaches
  //                 backwards to whatever produced the input and recovers an
  //                 unrelated name.
  std::map<std::string, std::vector<size_t>> by_input;
  for (size_t i = 0; i < nodes.size(); ++i)
    for (const std::string &inp : nodes[i].inputs)
      by_input[inp].push_back(i);
  for (size_t ni = 0; ni < nodes.size(); ++ni) {
    Node &n2 = nodes[ni];
    std::vector<size_t> seen;
    std::set<size_t> ids;
    for (const std::string &o : n2.outputs) {
      const auto it = by_input.find(o);
      if (it == by_input.end()) continue;
      for (size_t c : it->second)
        if (c != ni && ids.insert(c).second) seen.push_back(c);
    }
    n2.out_consumers = seen;
  }

  // pass 1: name, prefix/strip/rename -- and remember each initializer's uses
  std::map<std::string, std::vector<size_t>> named;
  std::vector<std::string> fixed_order;
  for (size_t i = 0; i < inits.size(); ++i) {
    Init &t = inits[i];
    if (t.name.empty()) continue;
    t.orig = t.name;
    if (t.kind == LocKind::kNone && !t.ext_location.empty()) {
      t.kind = LocKind::kExt;
      std::filesystem::path p(base);
      p /= t.ext_location;
      t.ext_path = p.string();
    }
    const auto u = by_input.find(t.name);
    if (u != by_input.end()) t.uses = u->second;

    std::string fixed = recover(t.name, t.uses, nodes, &t.how);
    if (!opts.strip.empty() && starts_with(fixed, opts.strip))
      fixed = fixed.substr(opts.strip.size());
    fixed = opts.prefix + fixed;
    for (const auto &rn : opts.rename)
      fixed = replace_all(fixed, rn.first, rn.second);

    if (named.find(fixed) == named.end()) fixed_order.push_back(fixed);
    named[fixed].push_back(i);
  }

  // pass 2: orientation, now that every sibling bias is addressable
  for (const std::string &fixed : fixed_order) {
    const std::vector<size_t> &group = named[fixed];
    if (group.size() != 1) {
      std::string list;
      for (size_t i : group) {
        if (!list.empty()) list += ", ";
        list += "'" + inits[i].orig + "'";
      }
      throw std::runtime_error(path_ + ": " + std::to_string(group.size()) +
                               " tensors recover to '" + fixed + "': [" + list +
                               "]");
    }
    Init &t = inits[group[0]];

    const int64_t *bias0 = nullptr;
    int64_t b0 = 0;
    if (ends_with(fixed, ".weight") && t.dims.size() == 2) {
      const auto sib = named.find(fixed.substr(0, fixed.size() - 7) + ".bias");
      if (sib != named.end() && !sib->second.empty() &&
          inits[sib->second[0]].dims.size() == 1) {
        b0 = inits[sib->second[0]].dims[0];
        bias0 = &b0;
      }
    }
    t.transpose = transpose_for(t.uses, nodes, t.orig, t.dims, bias0);

    std::vector<int64_t> shown = t.dims;
    if (t.transpose) std::reverse(shown.begin(), shown.end());
    if (bias0 && shown.size() == 2 && shown[0] != *bias0)
      throw std::runtime_error(
          path_ + ": '" + t.orig + "' reads as [" +
          std::to_string(shown[0]) + "," + std::to_string(shown[1]) +
          "] but its bias has " + std::to_string(*bias0) +
          " elements; orientation and the checkpoint disagree, refusing to "
          "pack");

    // Which bytes, and in which type. An inline payload names its own type
    // (the field it arrived in IS the type); everything else takes its type
    // from data_type.
    if (t.kind == LocKind::kInline) {
      if (!inline_tag(t.inl_no, &t.tag, &t.item))
        throw std::runtime_error(path_ + ": '" + t.orig +
                                 "' has an unsupported inline payload");
    } else if (!onnx_tag(t.dtype, &t.tag, &t.item)) {
      throw std::runtime_error(path_ + ": '" + t.orig + "' has ONNX dtype " +
                               std::to_string(t.dtype) +
                               ", for which this container has no tag; refusing "
                               "rather than retyping it");
    }
    if (t.kind == LocKind::kNone)
      throw std::runtime_error(path_ + ": '" + t.orig + "' carries no data");

    int64_t count = 1;
    for (int64_t d : t.dims) count *= d;
    const size_t need = static_cast<size_t>(count) * t.item;

    // --- locate the payload ------------------------------------------------
    const uint8_t *src = nullptr;
    std::shared_ptr<std::vector<uint8_t>> scalar;
    switch (t.kind) {
      case LocKind::kRaw: {
        if (t.raw_off > n || n - t.raw_off < need)
          throw std::runtime_error(path_ + ": '" + t.orig +
                                   "' size disagrees with its shape");
        src = b + t.raw_off;
        break;
      }
      case LocKind::kExt: {
        size_t en = 0;
        const uint8_t *eb = map_file(t.ext_path, &en);
        const size_t off = static_cast<size_t>(t.ext_off);
        if (!eb || off > en || en - off < need)
          throw std::runtime_error(path_ + ": '" + t.orig +
                                   "' runs past the end of " + t.ext_path);
        src = eb + off;
        break;
      }
      case LocKind::kInline: {
        // HOW THE BYTES ARE ENCODED is decided by the FIELD, not the wire type:
        // a packed integer field is LEB128, a packed float field is fixed-width
        // values end to end, and both are wire type 2. See inline_is_varint().
        if (inline_is_varint(t.inl_no) &&
            (t.inl_wt == 0 || t.inl_wt == 2)) {
          // One varint per element, materialised little-endian at the element's
          // own width. Counting them is also the only size check available: a
          // packed field's byte length says nothing about how many values are in
          // it, so the SHAPE is what has to agree, and a file where it does not
          // is refused here rather than truncated.
          scalar = std::make_shared<std::vector<uint8_t>>(need);
          size_t p = t.inl_ps;
          for (int64_t e = 0; e < count; ++e) {
            if (p > t.inl_pe)
              throw std::runtime_error(path_ + ": '" + t.orig +
                                       "' packed varints run out before its "
                                       "shape's element count; the file "
                                       "disagrees with itself");
            const uint64_t v = read_varint(b, n, p);
            std::memcpy(scalar->data() + static_cast<size_t>(e) *
                                             static_cast<size_t>(t.item),
                        &v, static_cast<size_t>(t.item));
          }
          if (p != t.inl_pe)
            throw std::runtime_error(path_ + ": '" + t.orig +
                                     "' has packed varints left over after "
                                     "its shape's element count; the file "
                                     "disagrees with itself");
          src = scalar->data();
        } else if (t.inl_wt == 2) {
          if (t.inl_pe - t.inl_ps != need)
            throw std::runtime_error(path_ + ": '" + t.orig +
                                     "' size disagrees with its shape");
          src = b + t.inl_ps;
        } else {                                          // 32-/64-bit fixed
          if (t.inl_pe - t.inl_ps < need || need == 0)
            throw std::runtime_error(path_ + ": '" + t.orig +
                                     "' size disagrees with its shape");
          src = b + t.inl_ps;
        }
        break;
      }
      default:
        throw std::runtime_error(path_ + ": '" + t.orig + "' carries no data");
    }

    const bool widen = t.tag == "F16" || t.tag == "BF16";
    const bool floats = t.tag == "F32" || widen;
    // A raw_data payload starts wherever the preceding length varint ended,
    // so it is NOT guaranteed to sit on its own element boundary. Handing an
    // unaligned address back as `float *` makes every consumer's plain load
    // undefined behaviour -- it happens to work on x86 and happens to be a
    // bus error on the platforms that care, which is precisely the class of
    // "correct until it isn't" this codebase keeps meeting. Copy instead.
    const bool aligned = t.item <= 1 ||
        (reinterpret_cast<uintptr_t>(src) % static_cast<uintptr_t>(t.item)) == 0;
    const bool view_ok = !t.transpose && !widen && aligned;

    Tensor out;
    out.shape = shown;
    out.ext_path = t.kind == LocKind::kExt ? t.ext_path : std::string();

    if (scalar) {
      out.owned_raw = scalar;
      out.data = scalar->data();
      out.bytes = scalar->size();
      out.dtype = t.tag;
    } else if (view_ok) {
      // The common case: a tensor the checkpoint already stores in the
      // orientation it is consumed in, at an address that can be loaded
      // directly. Zero copy.
      out.data = src;
      out.bytes = need;
      out.dtype = t.tag;
    } else if (floats) {
      out.owned = std::make_shared<std::vector<float>>(
          static_cast<size_t>(count));
      float *dst = out.owned->data();
      if (!t.transpose) {
        if (t.tag == "F32") {
          std::memcpy(dst, src, need);
        } else {
          const auto conv = t.tag == "BF16" ? &bf16_to_f32 : &f16_to_f32;
          for (int64_t i = 0; i < count; ++i)
            dst[i] = conv(src + static_cast<size_t>(i) * t.item);
        }
      } else {
        // dims.size() == 2 is guaranteed: transpose_for only ever returns true
        // for a 2-D tensor, and reversing more axes would change the product's
        // meaning rather than its orientation.
        const int64_t A = t.dims[0], B = t.dims[1];
        const auto conv = t.tag == "F32" ? nullptr
                         : t.tag == "BF16" ? &bf16_to_f32
                                           : &f16_to_f32;
        for (int64_t r = 0; r < A; ++r)
          for (int64_t c = 0; c < B; ++c) {
            const uint8_t *s = src + static_cast<size_t>(r * B + c) * t.item;
            float *d = dst + c * A + r;
            if (conv)
              *d = conv(s);
            else
              std::memcpy(d, s, 4);   // memcpy, never a cast: `s` may be odd
          }
      }
      out.data = reinterpret_cast<const uint8_t *>(dst);
      out.bytes = static_cast<size_t>(count) * 4;
      out.dtype = "F32";    // widened at read time; every consumer sees F32
    } else {
      out.owned_raw = std::make_shared<std::vector<uint8_t>>(need);
      uint8_t *dst = out.owned_raw->data();
      if (!t.transpose) {
        std::memcpy(dst, src, need);
      } else {
        const int64_t A = t.dims[0], B = t.dims[1];
        for (int64_t r = 0; r < A; ++r)
          for (int64_t c = 0; c < B; ++c)
            std::memcpy(dst + static_cast<size_t>(c * A + r) * t.item,
                        src + static_cast<size_t>(r * B + c) * t.item,
                        static_cast<size_t>(t.item));
      }
      out.data = dst;
      out.bytes = need;
      out.dtype = t.tag;
    }

    tensors_.emplace(fixed, std::move(out));
    order_.push_back(fixed);
  }

  // source_files(): the graph first, then each side file in the order the
  // tensors named it -- the order model_digest() hashes, so a container's
  // source_sha256 and CHECKPOINT.json's sha256 are one fact.
  files_.push_back(path_);
  std::set<std::string> seen{abspath(path_)};
  for (const std::string &fixed : order_) {
    const Tensor &t = tensors_.at(fixed);
    if (t.ext_path.empty()) continue;
    const std::string ap = abspath(t.ext_path);
    if (seen.insert(ap).second) files_.push_back(t.ext_path);
  }
}

OnnxWeights::~OnnxWeights() { maps_.clear(); }

OnnxCheckpoint onnx_checkpoint(const std::string &dir) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path root = fs::weakly_canonical(dir, ec);
  if (ec) root = fs::path(dir).lexically_normal();

  OnnxCheckpoint c;
  std::set<std::string> seen;
  for (const char *rel : {kModelOnnx, kWhisperEncoderOnnx, kWhisperDecoderOnnx}) {
    const fs::path p = root / rel;
    if (!fs::exists(p)) continue;
    // A graph that will not parse is a checkpoint nobody can read, and this
    // is where the reason is precise ("field 13 truncated", "cannot open
    // onnx/model.onnx_data") -- not somewhere three stack frames deep inside
    // a packer after half a gigabyte has already been written.
    const OnnxWeights w(p.string());
    for (const std::string &f : w.source_files()) {
      std::error_code e2;
      const fs::path pf = fs::weakly_canonical(fs::path(f), e2);
      if (e2 || !fs::exists(pf)) return OnnxCheckpoint{};   // all or nothing
      const std::string rel_to_root = fs::relative(pf, root, e2).string();
      if (e2) return OnnxCheckpoint{};
      if (seen.insert(rel_to_root).second) c.files.push_back(rel_to_root);
    }
    c.graphs.push_back(p.string());
  }
  return c;
}

}  // namespace npue
