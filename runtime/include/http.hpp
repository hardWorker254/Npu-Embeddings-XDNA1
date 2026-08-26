//===- http.hpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- just enough HTTP and JSON to serve /v1/embeddings.
// SPDX-License-Identifier: Apache-2.0
//
// WHY NOT A LIBRARY
// -----------------
// The whole point of this runtime is that it is one executable with one
// dependency (XRT). An embeddings endpoint needs: accept a POST, find the
// body, pull one array of strings out of it, and write numbers back. That is
// a few hundred lines against Winsock, and it keeps the deployment story --
// copy two files, run -- intact.
//
// WHAT THIS DELIBERATELY IS NOT
// -----------------------------
// Not a general HTTP server and not a general JSON parser. It handles exactly
// the shapes the OpenAI embeddings API uses, and REJECTS anything else rather
// than guessing:
//   * no chunked transfer encoding (Content-Length required)
//   * no TLS -- bind to localhost, or put a reverse proxy in front
//   * the JSON reader understands strings, arrays of strings, numbers and
//     booleans at the top level; it does not build a document tree
// A request that does not fit gets a 4xx with a reason, never a partial guess.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

namespace npue {
namespace http {

struct Request {
  std::string method, path, body;
};

// --- JSON: only what the API needs ----------------------------------------

// Decode a JSON string literal starting at `i` (which must index the opening
// quote). Advances `i` past the closing quote. Handles the escapes the spec
// defines, including \uXXXX with surrogate pairs, because real user text
// arrives that way from JavaScript clients.
inline std::string json_string(const std::string &s, size_t &i) {
  std::string out;
  if (i >= s.size() || s[i] != '"') return out;
  ++i;
  auto put_utf8 = [&out](uint32_t cp) {
    if (cp < 0x80) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  };
  auto hex4 = [&](size_t at) -> uint32_t {
    uint32_t v = 0;
    for (int k = 0; k < 4 && at + k < s.size(); ++k) {
      const char c = s[at + k];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
    }
    return v;
  };
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\' && i + 1 < s.size()) {
      const char e = s[i + 1];
      i += 2;
      switch (e) {
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case 'r': out.push_back('\r'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case '/': out.push_back('/'); break;
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case 'u': {
          uint32_t cp = hex4(i);
          i += 4;
          // A high surrogate must be paired, or the codepoint is wrong.
          if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() &&
              s[i] == '\\' && s[i + 1] == 'u') {
            const uint32_t lo = hex4(i + 2);
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              i += 6;
            }
          }
          put_utf8(cp);
          break;
        }
        default: out.push_back(e); break;
      }
    } else {
      out.push_back(s[i++]);
    }
  }
  if (i < s.size()) ++i;                      // closing quote
  return out;
}

// Find a top-level-ish key and return the offset of its value. Good enough
// because the request objects here are flat.
inline size_t find_value(const std::string &s, const std::string &key) {
  const std::string k = "\"" + key + "\"";
  size_t i = s.find(k);
  if (i == std::string::npos) return std::string::npos;
  i = s.find(':', i + k.size());
  if (i == std::string::npos) return std::string::npos;
  ++i;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' ||
                          s[i] == '\t')) ++i;
  return i;
}

// `input` is either a string or an array of strings, per the API. Arrays of
// token ids are NOT supported and the caller reports that rather than
// silently embedding the digits.
inline bool json_string_or_array(const std::string &s, const std::string &key,
                                 std::vector<std::string> &out,
                                 std::string &err) {
  size_t i = find_value(s, key);
  if (i == std::string::npos) {
    err = "missing field '" + key + "'";
    return false;
  }
  if (s[i] == '"') {
    out.push_back(json_string(s, i));
    return true;
  }
  if (s[i] != '[') {
    err = "field '" + key + "' must be a string or an array of strings";
    return false;
  }
  ++i;
  while (i < s.size()) {
    while (i < s.size() && (s[i] == ' ' || s[i] == ',' || s[i] == '\n' ||
                            s[i] == '\r' || s[i] == '\t')) ++i;
    if (i < s.size() && s[i] == ']') return true;
    if (i >= s.size() || s[i] != '"') {
      err = "'" + key + "' must contain strings (token-id arrays are not "
            "supported)";
      return false;
    }
    out.push_back(json_string(s, i));
  }
  err = "unterminated array in '" + key + "'";
  return false;
}

inline std::string json_field_string(const std::string &s,
                                     const std::string &key,
                                     const std::string &fallback) {
  size_t i = find_value(s, key);
  if (i == std::string::npos || s[i] != '"') return fallback;
  return json_string(s, i);
}

// ABSENT AND EMPTY ARE DIFFERENT for `prompt_name` (tasks/0118), and
// json_field_string() above cannot tell them apart -- it returns the fallback
// for both. A missing `prompt_name` on a model that has task prompts is a
// refusal; `"prompt_name": ""` is an explicit request for no prefix at all.
// Collapsing the two would reintroduce exactly the silent default this task
// removes.
//
// `Other` is its own answer rather than being folded into `Absent`: a `null`
// or numeric prompt_name is a MALFORMED request, and reporting it as "you did
// not send one" would send the caller looking in the wrong place.
enum class FieldKind { Absent, String, Other };

inline FieldKind json_field_kind(const std::string &s, const std::string &key,
                                 std::string &out) {
  size_t i = find_value(s, key);          // json_string() advances it
  if (i == std::string::npos) return FieldKind::Absent;
  if (s[i] != '"') return FieldKind::Other;
  out = json_string(s, i);
  return FieldKind::String;
}

inline std::string json_escape(const std::string &s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          o += buf;
        } else {
          o.push_back(static_cast<char>(c));
        }
    }
  }
  return o;
}

// The OpenAI clients ask for base64 by default, so it is not optional.
inline std::string base64(const uint8_t *data, size_t n) {
  static const char *T =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o;
  o.reserve((n + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 3 <= n; i += 3) {
    const uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) |
                       data[i + 2];
    o.push_back(T[(v >> 18) & 63]);
    o.push_back(T[(v >> 12) & 63]);
    o.push_back(T[(v >> 6) & 63]);
    o.push_back(T[v & 63]);
  }
  if (i < n) {
    uint32_t v = uint32_t(data[i]) << 16;
    if (i + 1 < n) v |= uint32_t(data[i + 1]) << 8;
    o.push_back(T[(v >> 18) & 63]);
    o.push_back(T[(v >> 12) & 63]);
    o.push_back(i + 1 < n ? T[(v >> 6) & 63] : '=');
    o.push_back('=');
  }
  return o;
}

// --- the server ------------------------------------------------------------

class Server {
public:
  explicit Server(uint16_t port, const std::string &bind_addr = "127.0.0.1") {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
      throw std::runtime_error("WSAStartup failed");
    listen_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_ == INVALID_SOCKET) throw std::runtime_error("socket() failed");
    BOOL yes = TRUE;
    setsockopt(listen_, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char *>(&yes), sizeof yes);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, bind_addr.c_str(), &a.sin_addr);
    if (bind(listen_, reinterpret_cast<sockaddr *>(&a), sizeof a) == SOCKET_ERROR)
      throw std::runtime_error("bind() failed on port " + std::to_string(port));
    if (::listen(listen_, 16) == SOCKET_ERROR)
      throw std::runtime_error("listen() failed");
  }

  ~Server() {
    if (listen_ != INVALID_SOCKET) closesocket(listen_);
    WSACleanup();
  }
  Server(const Server &) = delete;
  Server &operator=(const Server &) = delete;

  // Blocks. `handler` returns (status, content_type, body).
  template <typename H>
  void run(H &&handler) {
    for (;;) {
      SOCKET c = accept(listen_, nullptr, nullptr);
      if (c == INVALID_SOCKET) continue;
      Request req;
      if (read_request(c, req)) {
        int status = 200;
        std::string ctype = "application/json", body;
        handler(req, status, ctype, body);
        send_response(c, status, ctype, body);
      }
      closesocket(c);
    }
  }

private:
  SOCKET listen_ = INVALID_SOCKET;

  static bool read_request(SOCKET c, Request &req) {
    std::string buf;
    char tmp[8192];
    size_t header_end = std::string::npos;
    // Headers first, then exactly Content-Length bytes. Chunked encoding is
    // not supported -- better to fail than to mis-frame a body.
    while (header_end == std::string::npos) {
      const int n = recv(c, tmp, sizeof tmp, 0);
      if (n <= 0) return false;
      buf.append(tmp, static_cast<size_t>(n));
      header_end = buf.find("\r\n\r\n");
      if (buf.size() > (1u << 22) && header_end == std::string::npos)
        return false;
    }
    const std::string head = buf.substr(0, header_end);
    size_t sp1 = head.find(' ');
    size_t sp2 = head.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    req.method = head.substr(0, sp1);
    req.path = head.substr(sp1 + 1, sp2 - sp1 - 1);

    size_t clen = 0;
    std::string lower = head;
    for (auto &ch : lower) ch = static_cast<char>(::tolower(ch));
    const size_t cl = lower.find("content-length:");
    if (cl != std::string::npos)
      clen = static_cast<size_t>(std::stoull(lower.substr(cl + 15)));

    req.body = buf.substr(header_end + 4);
    while (req.body.size() < clen) {
      const int n = recv(c, tmp, sizeof tmp, 0);
      if (n <= 0) break;
      req.body.append(tmp, static_cast<size_t>(n));
    }
    return true;
  }

  static void send_response(SOCKET c, int status, const std::string &ctype,
                            const std::string &body) {
    const char *reason = status == 200 ? "OK"
                       : status == 400 ? "Bad Request"
                       : status == 404 ? "Not Found"
                       : status == 413 ? "Payload Too Large"
                                       : "Internal Server Error";
    std::string head = "HTTP/1.1 " + std::to_string(status) + " " + reason +
                       "\r\nContent-Type: " + ctype +
                       "\r\nContent-Length: " + std::to_string(body.size()) +
                       "\r\nAccess-Control-Allow-Origin: *"
                       "\r\nConnection: close\r\n\r\n";
    send(c, head.data(), static_cast<int>(head.size()), 0);
    size_t sent = 0;
    while (sent < body.size()) {
      const int n = send(c, body.data() + sent,
                         static_cast<int>(std::min<size_t>(
                             body.size() - sent, 1u << 20)), 0);
      if (n <= 0) break;
      sent += static_cast<size_t>(n);
    }
  }
};

}  // namespace http
}  // namespace npue
