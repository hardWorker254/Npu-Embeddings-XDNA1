//===- model.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- .npue reader implementation. See model.hpp.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "runtime/model.hpp"

#include <cstring>

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

#pragma pack(push, 1)
struct FileHeader {
  char magic[4];
  uint32_t version;
  uint32_t arch;
  uint32_t flags;
  uint64_t json_offset;
  uint64_t json_length;
  uint64_t data_offset;
  uint64_t data_length;
  uint8_t reserved[16];
};
#pragma pack(pop)
static_assert(sizeof(FileHeader) == 64, "the .npue header is exactly 64 bytes");

size_t skip_ws(const std::string &s, size_t i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' ||
                          s[i] == '\r'))
    ++i;
  return i;
}

// The escapes a JSON string may carry. json.dumps -- the writer's -- emits
// these and nothing else: `\"` for a quote inside a value, `\\` for a
// backslash, `\/` optionally, the four short forms, and `\uXXXX` for anything
// outside ASCII. So the whole set is listed rather than a prefix matched.
//
// A config value that CONTAINS JSON -- arch=6's "graph" and "npu_streams" are
// JSON text held in a JSON string -- needs `\"`, and refusing it made the reader
// unable to open a pose container at all. An unlisted escape is still refused by
// name rather than passed through: a silent pass-through would turn `A` into
// `AXu0041` and store a name nothing can match.
void append_escape(const std::string &s, size_t &i, std::string &out) {
  const char e = s[i++];
  switch (e) {
    case '"': out += '"'; return;
    case '\\': out += '\\'; return;
    case '/': out += '/'; return;
    case 'b': out += '\b'; return;
    case 'f': out += '\f'; return;
    case 'n': out += '\n'; return;
    case 'r': out += '\r'; return;
    case 't': out += '\t'; return;
    case 'u': {
      if (i + 4 > s.size())
        throw std::runtime_error(".npue: \\u escape is truncated in the directory");
      unsigned cp = 0;
      for (int k = 0; k < 4; ++k) {
        const char h = s[i + static_cast<size_t>(k)];
        unsigned d;
        if (h >= '0' && h <= '9') d = static_cast<unsigned>(h - '0');
        else if (h >= 'a' && h <= 'f') d = static_cast<unsigned>(h - 'a' + 10);
        else if (h >= 'A' && h <= 'F') d = static_cast<unsigned>(h - 'A' + 10);
        else
          throw std::runtime_error(std::string(".npue: \\u escape has '") + h +
                                   "' where a hex digit belongs");
        cp = cp * 16 + d;
      }
      i += 4;
      // Encoded as UTF-8, because the directory is UTF-8 and a code point above
      // 0x7f written as a raw byte would be a different string than the writer
      // meant. Surrogate halves are not paired here: json.dumps writes
      // non-BMP characters as a surrogate PAIR of \u escapes, and a lone half
      // is not a character.
      if (cp < 0x80) {
        out += static_cast<char>(cp);
      } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
      } else {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
      }
      return;
    }
    default:
      throw std::runtime_error(
          std::string(".npue: unsupported escape '\\") + e +
          "' in the directory. The writer emits only \\\" \\\\ \\/ \\b \\f \\n "
          "\\r \\t and \\uXXXX; anything else is a directory this reader does not "
          "understand, and passing it through would store a different string "
          "than the one that was written.");
  }
}

std::string read_string(const std::string &s, size_t &i) {
  if (s[i] != '"') throw std::runtime_error(".npue: expected a JSON string");
  ++i;
  std::string out;
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\') {
      ++i;
      append_escape(s, i, out);
      continue;
    }
    out += s[i++];
  }
  ++i;                       // the closing quote
  return out;
}

std::string read_scalar(const std::string &s, size_t &i) {
  i = skip_ws(s, i);
  if (s[i] == '"') return read_string(s, i);
  size_t start = i;
  while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
  return s.substr(start, i - start);
}

void skip_value(const std::string &s, size_t &i) {
  i = skip_ws(s, i);
  if (s[i] == '"') { read_string(s, i); return; }
  if (s[i] == '{' || s[i] == '[') {
    int depth = 0;
    do {
      if (s[i] == '{' || s[i] == '[') ++depth;
      else if (s[i] == '}' || s[i] == ']') --depth;
      else if (s[i] == '"') { read_string(s, i); continue; }
      ++i;
    } while (i < s.size() && depth > 0);
    return;
  }
  read_scalar(s, i);
}

std::vector<int64_t> read_int_array(const std::string &s, size_t &i) {
  std::vector<int64_t> out;
  i = skip_ws(s, i);
  if (s[i] != '[') throw std::runtime_error(".npue: expected an array");
  ++i;
  while (true) {
    i = skip_ws(s, i);
    if (s[i] == ']') { ++i; break; }
    out.push_back(std::stoll(read_scalar(s, i)));
    i = skip_ws(s, i);
    if (s[i] == ',') ++i;
  }
  return out;
}

// One tensor's "layout" dict, narrowed to what an int4 decode needs:
// TensorInfo::layout_kind plus tile_k/tile_n/mac_s/mac_t. Everything else in
// the dict is deliberately skipped: `order`, `inner` and `dtype` are already
// folded into layout_hash, and design matching compares that hash -- so
// re-reading them here would be a second, weaker copy of a check that already
// exists. (It also means this parser assumes the canonical `order`, exactly as
// Python's Reader.panel() does when it calls untile_b with no order=; the hash
// is what makes the assumption safe, and layout_kind is the cheap half of it
// that can be checked by name.)
//
// Returns false when the dict does not describe a block_panel the decode can
// walk, which the caller treats as "not decodable" rather than as a parse
// error: BF16 and I8 operands never need these numbers, and a container that
// does not use them must keep loading.
bool read_layout(const std::string &s, size_t &i, TensorInfo &t) {
  i = skip_ws(s, i);
  if (s[i] != '{') throw std::runtime_error(".npue: layout is not an object");
  ++i;
  while (true) {
    i = skip_ws(s, i);
    if (s[i] == '}') { ++i; break; }
    std::string k = read_string(s, i);
    i = skip_ws(s, i);
    if (s[i] != ':')
      throw std::runtime_error(".npue: layout entry is not key: value");
    ++i;
    i = skip_ws(s, i);
    if (k == "kind") t.layout_kind = read_scalar(s, i);
    else if (k == "tile_k") t.tile_k = std::stoll(read_scalar(s, i));
    else if (k == "tile_n") t.tile_n = std::stoll(read_scalar(s, i));
    else if (k == "mac_s") t.mac_s = std::stoll(read_scalar(s, i));
    else if (k == "mac_t") t.mac_t = std::stoll(read_scalar(s, i));
    else skip_value(s, i);
    i = skip_ws(s, i);
    if (s[i] == ',') ++i;
  }
  const bool usable = t.layout_kind == "block_panel" && t.tile_k > 0 &&
                      t.tile_n > 0 && t.mac_s > 0 && t.mac_t > 0;
  if (!usable) {
    t.tile_k = t.tile_n = t.mac_s = t.mac_t = 0;
    if (t.layout_kind != "block_panel") t.layout_kind.clear();
  }
  return usable;
}

}  // namespace

File::File(const std::string &path) {
#ifdef _WIN32
  std::wstring wpath(path.begin(), path.end());
  handle_file_ = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                 nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                                 nullptr);
  if (handle_file_ == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot open " + path);

  LARGE_INTEGER sz{};
  GetFileSizeEx(handle_file_, &sz);
  size_ = static_cast<size_t>(sz.QuadPart);

  handle_map_ = CreateFileMappingW(handle_file_, nullptr, PAGE_READONLY, 0, 0,
                                    nullptr);
  if (!handle_map_) throw std::runtime_error("cannot map " + path);
  base_ = static_cast<const uint8_t *>(
      MapViewOfFile(handle_map_, FILE_MAP_READ, 0, 0, 0));
  if (!base_) throw std::runtime_error("cannot view " + path);
#else
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);

  struct stat st{};
  if (::fstat(fd, &st) < 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  size_ = static_cast<size_t>(st.st_size);

  void *m = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (m == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);
  base_ = static_cast<const uint8_t *>(m);
#endif

  if (size_ < sizeof(FileHeader))
    throw std::runtime_error(path + ": truncated");

  if (size_ < sizeof(FileHeader)) throw std::runtime_error(path + ": truncated");
  FileHeader h{};
  std::memcpy(&h, base_, sizeof h);
  if (std::memcmp(h.magic, "NPUE", 4) != 0)
    throw std::runtime_error(path + ": not a .npue file");
  if (h.version != 1)
    throw std::runtime_error(path + ": version " + std::to_string(h.version) +
                             ", expected 1");
  if (h.data_offset % 4096 != 0)
    throw std::runtime_error(path + ": data_offset is not 4096-aligned");
  if (size_ != h.data_offset + h.data_length)
    throw std::runtime_error(path + ": size does not match the header");

  version_ = h.version;
  data_offset_ = h.data_offset;
  data_length_ = h.data_length;

  std::string js(reinterpret_cast<const char *>(base_ + h.json_offset),
                 h.json_length);

  size_t i = skip_ws(js, 0);
  if (js[i] != '{') throw std::runtime_error(".npue: directory is not an object");
  ++i;
  while (true) {
    i = skip_ws(js, i);
    if (js[i] == '}') break;
    std::string key = read_string(js, i);
    i = skip_ws(js, i);
    if (js[i] != ':') throw std::runtime_error(".npue: expected ':'");
    ++i;

    if (key == "config") {
      i = skip_ws(js, i);
      ++i;
      while (true) {
        i = skip_ws(js, i);
        if (js[i] == '}') { ++i; break; }
        std::string ck = read_string(js, i);
        i = skip_ws(js, i);
        ++i;
        i = skip_ws(js, i);
        if (js[i] == '{' || js[i] == '[') {
          size_t start = i;
          skip_value(js, i);
          config_[ck] = js.substr(start, i - start);
        } else {
          config_[ck] = read_scalar(js, i);
        }
        i = skip_ws(js, i);
        if (js[i] == ',') ++i;
      }
    } else if (key == "tensors") {
      i = skip_ws(js, i);
      ++i;
      while (true) {
        i = skip_ws(js, i);
        if (js[i] == ']') { ++i; break; }
        ++i;
        TensorInfo t;
        while (true) {
          i = skip_ws(js, i);
          if (js[i] == '}') { ++i; break; }
          std::string tk = read_string(js, i);
          i = skip_ws(js, i);
          ++i;
          if (tk == "name") t.name = read_scalar(js, i);
          else if (tk == "role") t.role = read_scalar(js, i);
          else if (tk == "dtype") t.dtype = read_scalar(js, i);
          else if (tk == "logical_shape") t.logical_shape = read_int_array(js, i);
          else if (tk == "padded_shape") t.padded_shape = read_int_array(js, i);
          else if (tk == "offset") t.offset = std::stoull(read_scalar(js, i));
          else if (tk == "nbytes") t.nbytes = std::stoull(read_scalar(js, i));
          else if (tk == "layout_hash") t.layout_hash = read_scalar(js, i);
          else if (tk == "layout") read_layout(js, i, t);
          else skip_value(js, i);
          i = skip_ws(js, i);
          if (js[i] == ',') ++i;
        }
        if (t.offset + t.nbytes > data_length_)
          throw std::runtime_error(".npue: tensor " + t.name +
                                   " runs past the data segment");
        tensors_[t.name] = t;
        i = skip_ws(js, i);
        if (js[i] == ',') ++i;
      }
    } else {
      skip_value(js, i);
    }
    i = skip_ws(js, i);
    if (js[i] == ',') ++i;
  }
}

File::~File() {
#ifdef _WIN32
  if (base_) UnmapViewOfFile(base_);
  if (handle_map_) CloseHandle(handle_map_);
  if (handle_file_ && handle_file_ != INVALID_HANDLE_VALUE)
    CloseHandle(handle_file_);
#else
  if (base_) ::munmap(const_cast<uint8_t *>(base_), size_);
#endif
}

const TensorInfo &File::info(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error(".npue: no tensor named " + name);
  return it->second;
}

Span File::raw(const std::string &name) const {
  const TensorInfo &t = info(name);
  return Span{base_ + data_offset_ + t.offset, t.nbytes};
}

int64_t File::config_int(const std::string &key) const {
  return std::stoll(config_string(key));
}

double File::config_double(const std::string &key) const {
  return std::stod(config_string(key));
}

std::string File::config_string(const std::string &key) const {
  auto it = config_.find(key);
  if (it == config_.end())
    throw std::runtime_error(".npue: no config key " + key);
  return it->second;
}

}  // namespace npue