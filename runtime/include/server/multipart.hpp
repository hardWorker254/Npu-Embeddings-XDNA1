//===- multipart.hpp ----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- just enough multipart/form-data to read a request body.
//
// WHY A PARSER AND NOT A REGEX
// ---------------------------
// The body is BINARY: a WAV file is inside it, and it can contain the boundary
// string by accident, \r\n--boundary- sequences, and NUL bytes. A regex over it
// is a guess about bytes, and a guess about the bytes of an audio file is how a
// transcription silently reads the wrong audio. So this walks the body the way
// RFC 7578 says a parser must: find the first boundary as a line, then for each
// part read the headers up to the blank line, then take EXACTLY the number of
// bytes the headers say, then look for the CRLF that precedes the next boundary.
//
// WHAT IT REFUSES
// --------------
// A missing boundary in the Content-Type, a part with no blank line after its
// headers, a truncated body, a header block over 8 KB, and a boundary over 70
// characters. Each is an error with a reason, not a best-effort parse: the
// alternatives are a wrong transcript and a wrong transcript.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace npue::http {

struct MultipartPart {
  std::string name;          // the form field's name
  std::string filename;      // empty unless the part carried one
  std::string content_type;  // empty unless the part declared one
  std::string data;          // the bytes, verbatim
};

// The boundary out of a Content-Type header, or "" when there is not exactly
// one. `multipart/form-data; boundary=----WebKitFormBoundaryABC` and the quoted
// spelling are both accepted, because clients send both.
//
// The KEY is matched case-insensitively (HTTP header names are), but the VALUE
// is taken from the original header and NOT lowercased: a boundary is an opaque
// byte string, and a boundary containing an uppercase letter is a boundary the
// body will not contain once the header has been folded.
inline std::string multipart_boundary(const std::string &content_type) {
  std::string lower = content_type;
  for (auto &c : lower) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
  if (lower.rfind("multipart/form-data", 0) != 0) return std::string();
  const size_t at = lower.find("boundary=");
  if (at == std::string::npos) return std::string();
  // Same offset in both strings: lowercasing does not move characters.
  size_t i = at + 9;
  if (i < content_type.size() && content_type[i] == '"') {
    ++i;
    const size_t end = content_type.find('"', i);
    if (end == std::string::npos) return std::string();
    if (end - i > 70) return std::string();
    return content_type.substr(i, end - i);
  }
  const size_t end = content_type.find_first_of("; \r\n", i);
  const std::string b = content_type.substr(i, end - i);
  if (b.empty() || b.size() > 70) return std::string();
  return b;
}

// Header value out of one part's header block, lowercased key.
inline std::string part_header(const std::string &headers, const char *name) {
  std::string want = name;
  for (auto &c : want) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
  size_t pos = 0;
  while (pos < headers.size()) {
    size_t eol = headers.find("\r\n", pos);
    if (eol == std::string::npos) eol = headers.size();
    const std::string line = headers.substr(pos, eol - pos);
    const size_t colon = line.find(':');
    if (colon != std::string::npos) {
      std::string key = line.substr(0, colon);
      for (auto &c : key) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
      size_t v = colon + 1;
      while (v < line.size() && (line[v] == ' ' || line[v] == '\t')) ++v;
      if (key == want) return line.substr(v);
    }
    pos = eol + 2;
  }
  return std::string();
}

inline bool parse_multipart(const std::string &body, const std::string &boundary,
                            std::vector<MultipartPart> &out, std::string &err) {
  const std::string dash = "--" + boundary;
  // A body that does not even open with the first boundary is not this media
  // type; saying so is more useful than returning zero parts.
  size_t pos = body.find(dash);
  if (pos == std::string::npos) {
    err = "the body does not contain the boundary " + dash +
          " -- this does not look like multipart/form-data";
    return false;
  }
  pos += dash.size();
  while (true) {
    // "--" after the boundary closes the body; a trailing "--" may follow.
    if (body.compare(pos, 2, "--") == 0) return true;
    if (body.compare(pos, 2, "\r\n") != 0) {
      err = "malformed boundary at byte " + std::to_string(pos) +
            ": expected CRLF after " + dash;
      return false;
    }
    pos += 2;
    const size_t head_end = body.find("\r\n\r\n", pos);
    if (head_end == std::string::npos) {
      err = "a part's headers are not terminated by a blank line";
      return false;
    }
    const std::string headers = body.substr(pos, head_end - pos);
    if (headers.size() > 8192) {
      err = "a part's header block is over 8 KB";
      return false;
    }
    pos = head_end + 4;

    MultipartPart part;
    // Content-Disposition carries both the field name and the filename, quoted.
    const std::string disp = part_header(headers, "content-disposition");
    auto quoted = [&](const char *key, std::string &out) {
      const std::string k = std::string(key) + "=\"";
      size_t k_at = disp.find(k);
      if (k_at == std::string::npos) return false;
      k_at += k.size();
      const size_t end = disp.find('"', k_at);
      if (end == std::string::npos) return false;
      out = disp.substr(k_at, end - k_at);
      return true;
    };
    if (!quoted("name", part.name)) {
      err = "a part has no name= in its Content-Disposition";
      return false;
    }
    quoted("filename", part.filename);
    part.content_type = part_header(headers, "content-type");

    // The part ends at the CRLF BEFORE the next boundary, or at the closing
    // "--" for the last one. Anything else and a WAV containing the boundary's
    // first bytes would end the part early.
    const size_t next = body.find("\r\n" + dash, pos);
    if (next == std::string::npos) {
      err = "the body ends inside a part (no closing boundary after " +
            part.name + ")";
      return false;
    }
    part.data = body.substr(pos, next - pos);
    out.push_back(std::move(part));
    pos = next + 2 + dash.size();
  }
}

}  // namespace npue::http
