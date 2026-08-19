//===- hub.cpp ------------------------------------------------------------===//
//
// NpuEmbeddings -- the model catalogue and the WinHTTP fetcher.
// SPDX-License-Identifier: Apache-2.0
//
// See include/hub.hpp for why this replaced `get-model.cmd`.
//
// WinHTTP rather than libcurl or WinINet: it ships with Windows (so the
// release stays one exe plus one design directory, with no DLL to carry and
// no new licence in the tree), it is the API supported in services, and it
// does certificate validation by default. WinINet is documented as unsuitable
// for non-interactive use.

#include "hub.hpp"

#include "npue_pack.hpp"

#include <windows.h>
#include <winhttp.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace npue {
namespace hub {

namespace {

// --- the catalogue --------------------------------------------------------
//
// Every geometry here is checked against the downloaded config.json before a
// container is built (`verify_config`), so these numbers are a DESCRIPTION
// for the table, never an assumption the packer relies on. The packer reads
// the checkpoint's own config, as it always has.
//
// The pins come from models/<name>/CHECKPOINT.json, which
// reference/fetch_model.py wrote when each model was first brought up and
// validated against its goldens.
const std::vector<CatalogEntry> &table() {
  static const std::vector<CatalogEntry> v = {
      {"all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2",
       "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
       "mean", 384, 6, 12, 1536, 48, 90.9,
       "smallest and fastest; head_dim 32 keeps attention off the array"},
      {"bge-small-en-v1.5", "BAAI/bge-small-en-v1.5",
       "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
       "cls", 384, 12, 12, 1536, 48, 133.5,
       "MiniLM's width at twice the depth; +2.99 MTEB points"},
      {"bge-base-en-v1.5", "BAAI/bge-base-en-v1.5",
       "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7",
       "cls", 768, 12, 12, 3072, 48, 438.0,
       "best geometric fit for this NPU: head_dim 64 and every N a "
       "multiple of 384"},
      {"bge-large-en-v1.5", "BAAI/bge-large-en-v1.5",
       "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7",
       "cls", 1024, 24, 16, 4096, 32, 1340.0,
       "highest quality; N=1024 forces tile_n 32, and 24 layers cost "
       "dispatches"},
  };
  return v;
}

// The files the packer and tokenizer actually consume. Deliberately NOT the
// whole repository: the .onnx and .openvino exports and the pytorch .bin
// duplicate are several hundred megabytes of nothing. This mirrors
// reference/fetch_model.py's ALLOW list, minus the files only the Python
// reference path uses.
struct Want {
  const char *rel;
  bool required;
};
const Want kFiles[] = {
    {"model.safetensors", true},
    {"vocab.txt", true},
    {"config.json", true},
    {"1_Pooling/config.json", true},
};

std::wstring widen(const std::string &s) {
  if (s.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(),
                                    nullptr, 0);
  std::wstring w((size_t)n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
  return w;
}

std::string human(uint64_t bytes) {
  char b[64];
  if (bytes >= (1ull << 30))
    std::snprintf(b, sizeof b, "%.2f GB", double(bytes) / double(1ull << 30));
  else
    std::snprintf(b, sizeof b, "%.1f MB", double(bytes) / double(1ull << 20));
  return b;
}

// A WinHTTP handle that closes itself. WinHttpCloseHandle on a null handle is
// a no-op, so the empty case needs no branch.
struct Handle {
  HINTERNET h = nullptr;
  Handle() = default;
  explicit Handle(HINTERNET x) : h(x) {}
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  ~Handle() { if (h) WinHttpCloseHandle(h); }
  operator HINTERNET() const { return h; }
};

[[noreturn]] void fail(const std::string &what) {
  throw std::runtime_error(what + " (WinHTTP error " +
                           std::to_string(GetLastError()) + ")");
}

struct Url {
  std::wstring host, path;
  INTERNET_PORT port = INTERNET_DEFAULT_HTTPS_PORT;
  bool https = true;
};

Url parse_url(const std::string &url) {
  const std::wstring w = widen(url);
  URL_COMPONENTS c = {};
  c.dwStructSize = sizeof c;
  c.dwHostNameLength = c.dwUrlPathLength = c.dwExtraInfoLength = (DWORD)-1;
  if (!WinHttpCrackUrl(w.c_str(), (DWORD)w.size(), 0, &c))
    fail("cannot parse URL " + url);
  Url u;
  u.host.assign(c.lpszHostName, c.dwHostNameLength);
  u.path.assign(c.lpszUrlPath, c.dwUrlPathLength);
  if (c.dwExtraInfoLength)
    u.path.append(c.lpszExtraInfo, c.dwExtraInfoLength);
  u.port = c.nPort;
  u.https = (c.nScheme == INTERNET_SCHEME_HTTPS);
  return u;
}

}  // namespace

const std::vector<CatalogEntry> &catalog() { return table(); }

const CatalogEntry *find(const std::string &name) {
  for (const auto &e : table())
    if (e.name == name) return &e;
  return nullptr;
}

void download(const std::string &url, const std::string &dest,
              const Log &log) {
  const Url u = parse_url(url);

  Handle session(WinHttpOpen(L"NpuEmbeddings/0.2",
                             WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                             WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS,
                             0));
  if (!session) fail("WinHttpOpen failed");

  // Generous but finite. A stalled CDN must eventually fail rather than hang
  // a `serve` that the user is watching.
  WinHttpSetTimeouts(session, 15000, 15000, 60000, 60000);

  Handle conn(WinHttpConnect(session, u.host.c_str(), u.port, 0));
  if (!conn) fail("cannot connect to " + url);

  Handle req(WinHttpOpenRequest(
      conn, L"GET", u.path.c_str(), nullptr, WINHTTP_NO_REFERER,
      WINHTTP_DEFAULT_ACCEPT_TYPES, u.https ? WINHTTP_FLAG_SECURE : 0));
  if (!req) fail("cannot open request for " + url);

  // WinHTTP follows redirects by default; HuggingFace always redirects
  // `resolve/main/...` to its CDN, so this is the normal path, not an edge
  // case.
  if (!WinHttpSendRequest(req, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                          WINHTTP_NO_REQUEST_DATA, 0, 0, 0))
    fail("request failed for " + url);
  if (!WinHttpReceiveResponse(req, nullptr))
    fail("no response for " + url);

  DWORD status = 0, len = sizeof status;
  WinHttpQueryHeaders(req,
                      WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                      WINHTTP_HEADER_NAME_BY_INDEX, &status, &len,
                      WINHTTP_NO_HEADER_INDEX);
  if (status != 200)
    throw std::runtime_error("HTTP " + std::to_string(status) + " for " + url);

  uint64_t total = 0;
  {
    wchar_t cl[32] = {};
    DWORD n = sizeof cl;
    if (WinHttpQueryHeaders(req, WINHTTP_QUERY_CONTENT_LENGTH,
                            WINHTTP_HEADER_NAME_BY_INDEX, cl, &n,
                            WINHTTP_NO_HEADER_INDEX))
      total = _wcstoui64(cl, nullptr, 10);
  }

  // Download to <dest>.part and rename only on success. An interrupted
  // fetch must never be left looking like a finished one -- the next run
  // would checksum a truncated file and report a MISMATCH, which is a true
  // statement that points at entirely the wrong problem.
  const std::filesystem::path final_path(dest);
  const std::filesystem::path part = final_path.string() + ".part";
  std::filesystem::create_directories(final_path.parent_path());

  std::ofstream out(part, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + part.string());

  std::vector<char> buf(1 << 20);
  uint64_t got = 0;
  int last_pct = -1;
  for (;;) {
    DWORD avail = 0;
    if (!WinHttpQueryDataAvailable(req, &avail)) fail("read failed on " + url);
    if (avail == 0) break;
    while (avail) {
      const DWORD chunk = avail < (DWORD)buf.size() ? avail : (DWORD)buf.size();
      DWORD read = 0;
      if (!WinHttpReadData(req, buf.data(), chunk, &read))
        fail("read failed on " + url);
      if (read == 0) break;
      out.write(buf.data(), (std::streamsize)read);
      if (!out) throw std::runtime_error("write failed on " + part.string());
      got += read;
      avail -= read;
      if (total) {
        const int pct = int(got * 100 / total);
        if (pct != last_pct && pct % 5 == 0) {
          last_pct = pct;
          if (log)
            log("    " + std::to_string(pct) + "%  " + human(got) + " of " +
                human(total));
        }
      }
    }
  }
  out.close();

  if (total && got != total)
    throw std::runtime_error("short read on " + url + ": got " +
                             std::to_string(got) + " of " +
                             std::to_string(total) + " bytes");

  std::error_code ec;
  std::filesystem::remove(final_path, ec);
  std::filesystem::rename(part, final_path, ec);
  if (ec)
    throw std::runtime_error("cannot rename " + part.string() + ": " +
                             ec.message());
}

namespace {

std::string slurp_text(const std::filesystem::path &p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

// A minimal number-field reader. The runtime already has a string-field one
// (npue::http::json_field_string); these config files are flat and machine
// generated, so a scan for `"key"` followed by a number is enough and does
// not justify a JSON parser in the link line.
int64_t json_int(const std::string &s, const std::string &key, int64_t dflt) {
  const std::string k = "\"" + key + "\"";
  size_t p = s.find(k);
  if (p == std::string::npos) return dflt;
  p = s.find(':', p + k.size());
  if (p == std::string::npos) return dflt;
  ++p;
  while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' ||
                          s[p] == '\r'))
    ++p;
  const size_t start = p;
  while (p < s.size() && (isdigit((unsigned char)s[p]) || s[p] == '-')) ++p;
  if (p == start) return dflt;
  return std::stoll(s.substr(start, p - start));
}

// Refuse a checkpoint whose config disagrees with what this build knows about
// it. The catalogue's geometry is a claim, and a claim that is never checked
// is the fail-open shape this project keeps finding (tasks/0038-0045): the
// entry would silently describe one model while the packer built another.
void verify_config(const CatalogEntry &e, const std::filesystem::path &dir) {
  const std::string cfg = slurp_text(dir / "config.json");
  struct Check { const char *key; int64_t want; };
  const Check checks[] = {
      {"hidden_size", e.hidden},
      {"num_hidden_layers", e.layers},
      {"num_attention_heads", e.heads},
      {"intermediate_size", e.ffn},
  };
  std::string bad;
  for (const auto &c : checks) {
    const int64_t got = json_int(cfg, c.key, -1);
    if (got != c.want)
      bad += std::string("\n    ") + c.key + ": catalogue says " +
             std::to_string(c.want) + ", checkpoint says " +
             std::to_string(got);
  }
  if (!bad.empty())
    throw std::runtime_error(
        "the downloaded checkpoint for " + e.name +
        " is not the model this build has catalogued:" + bad +
        "\n  Refusing to pack it. The repository's contents changed, or the "
        "catalogue entry is wrong.");

  // Pooling is read from the checkpoint, never assumed -- 0038 made this a
  // rule after `mean` had been a literal. The catalogue's value only has to
  // AGREE.
  const std::string pool = slurp_text(dir / "1_Pooling" / "config.json");
  const bool cls = pool.find("\"pooling_mode_cls_token\": true") !=
                       std::string::npos ||
                   pool.find("\"pooling_mode_cls_token\":true") !=
                       std::string::npos;
  const bool mean = pool.find("\"pooling_mode_mean_tokens\": true") !=
                        std::string::npos ||
                    pool.find("\"pooling_mode_mean_tokens\":true") !=
                        std::string::npos;
  const std::string got = cls && !mean ? "cls" : (mean && !cls ? "mean" : "");
  if (got != e.pooling)
    throw std::runtime_error(
        "1_Pooling/config.json for " + e.name + " gives pooling '" + got +
        "', the catalogue says '" + e.pooling + "' -- refusing to pack");
}

}  // namespace

std::string ensure_model(const std::string &root, const std::string &name,
                         const Log &log) {
  namespace fs = std::filesystem;
  const fs::path models = fs::path(root) / "models";
  const fs::path container = models / (name + ".npue");

  if (fs::exists(container)) return container.string();

  const CatalogEntry *e = find(name);
  if (!e) {
    std::string known;
    for (const auto &c : table()) known += "\n    " + c.name;
    throw std::runtime_error(
        "'" + name + "' is not installed and is not a model this build knows "
        "how to fetch. Known models:" + known +
        "\n  A container you packed yourself is used by name once it is in "
        "models/.");
  }

  const fs::path dir = models / name;
  if (log) {
    log("");
    log("  " + e->name + " is not installed. Fetching it from " + e->repo +
        ".");
    log("  " + human(uint64_t(e->download_mb * 1024 * 1024)) +
        " of checkpoint, verified against a checksum built into this "
        "executable.");
    log("");
  }

  const std::string base =
      "https://huggingface.co/" + e->repo + "/resolve/main/";
  for (const auto &w : kFiles) {
    const fs::path dest = dir / w.rel;
    if (fs::exists(dest)) {
      if (log) log("  have  " + std::string(w.rel));
      continue;
    }
    if (log) log("  get   " + std::string(w.rel));
    download(base + w.rel, dest.string(), log);
  }

  // The check that used to be `certutil` in a batch file. Same comparison,
  // same pin, no script.
  if (log) log("  hash  model.safetensors");
  const std::string got =
      npue::sha256_file((dir / "model.safetensors").string());
  if (got != e->sha256) {
    throw std::runtime_error(
        "CHECKSUM MISMATCH for " + e->repo + "/model.safetensors\n"
        "    expected " + e->sha256 + "\n"
        "    got      " + got + "\n"
        "  These are not the weights this build was verified against. "
        "Stopping.\n"
        "  Delete " + dir.string() + " and try again; if it persists, the "
        "upstream repository has changed and this build's accuracy numbers "
        "no longer describe it.");
  }
  if (log) log("        ok  " + got.substr(0, 16) + "...");

  verify_config(*e, dir);

  // The pin, written where pack_npue.py and --prepare-model both look for it.
  // Byte-for-byte the layout reference/fetch_model.py writes (json.dumps with
  // indent=2 and no trailing newline), so a checkpoint re-fetched by the
  // executable does not show up as a diff against one fetched by the Python
  // path. Two writers of one file should not disagree about its formatting.
  {
    std::ofstream cf(dir / "CHECKPOINT.json", std::ios::binary);
    cf << "{\n  \"repo_id\": \"" << e->repo
       << "\",\n  \"file\": \"model.safetensors\",\n  \"sha256\": \""
       << e->sha256 << "\"\n}";
  }

  if (log) log("  pack  " + container.filename().string());
  const Layout layout = gemm_b_layout(64, e->tile_n);
  prepare_model((dir / "model.safetensors").string(),
                (dir / "vocab.txt").string(),
                (dir / "config.json").string(), e->pooling, e->repo,
                container.string(), got, layout.json, layout.hash, 64,
                e->tile_n, 256, nullptr);

  if (log) {
    log("  ready " + container.string());
    log("");
  }
  return container.string();
}

}  // namespace hub
}  // namespace npue
