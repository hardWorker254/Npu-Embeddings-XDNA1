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

#include "common/hub.hpp"

#include "common/json_min.hpp"
#include "common/npue_pack.hpp"

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#else
#include <curl/curl.h>
#include <unistd.h>
#include <cstring>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
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
  static const std::vector<CatalogEntry> v = [] {
    std::vector<CatalogEntry> rows = {
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
       "dispatches", /*gated=*/false, /*gemma=*/false},
      // Pin fetched and verified 2026-08-20 (tasks/0066): downloaded
      // model.safetensors from the OFFICIAL gated google/embeddinggemma-300m
      // with a real HF_TOKEN, sha256 cbf5a78393b6a033e0b8a63a57549964
      // f7ed5c6fbeb4ba0694214f36123f2fd2 -- byte-identical to the
      // unsloth/embeddinggemma-300m mirror tasks/0055-0065 verified all
      // night, so every 1-cos/parity figure already on record for that
      // checkpoint is valid for THIS one too, no re-verification needed.
      // NPU path since tasks/0074: `gated_ffn` (GeGLU -- ffn_up emits both
      // halves, N = 2*1152) and `qkv_n` = 1536 (MQA's 1280-wide Q|K|V,
      // zero-padded so `N % (tile_n * n_aie_cols)` holds at tile_n=48). Both
      // are load-bearing for `list`: without them design_fits() asks for
      // N=1152 and N=2304 on the wrong two streams and this row reports "no
      // design" against its own correct design set.
      {"embeddinggemma-300m", "google/embeddinggemma-300m",
       "cbf5a78393b6a033e0b8a63a57549964f7ed5c6fbeb4ba0694214f36123f2fd2",
       "mean", 768, 24, 3, 1152, 48, 1155.0,
       "MQA+RoPE+GeGLU on the array (4 GEMMs/layer); gated, needs HF_TOKEN",
       /*gated=*/true, /*gemma=*/true, /*gated_ffn=*/true, /*qkv_n=*/1536},
      // arch=2 (tasks/0068-0071): RoPE + gated SwiGLU, NOT the BERT-family
      // absolute-position + GELU the other four rows share -- but it packs
      // to the SAME layout_hash and runs on the SAME NPU designs (tasks/
      // 0069). sha256 re-derived locally against the downloaded
      // model.safetensors with npue::sha256_file (tasks/0071) -- matches
      // the pin reference/fetch_model.py recorded when this checkpoint was
      // first brought up (tasks/0068), CLAUDE.md rule 6. `gated_ffn=true`
      // is load-bearing: ffn_up emits 2*ffn (6144, not 3072), and without
      // this bit design_fits() cannot tell this model apart from
      // bge-base-en-v1.5's identical {768,3072} K set (tasks/0069 T31).
      // NEEDS a task prefix (--prefix, tasks/0071) -- omitting one is a
      // measured quality regression, not just a convention
      // (docs/04-model/README.md:24).
      {"nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5",
       "9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c",
       "mean", 768, 12, 12, 3072, 48, 546.9,
       "RoPE + gated SwiGLU (arch=2); same array designs as bge-base; "
       "needs --prefix (search_document / search_query / clustering / "
       "classification)", /*gated=*/false, /*gemma=*/false,
       /*gated_ffn=*/true},
      // arch=3 (tasks/0134-0138): the multilingual encoder -- XLM-R Unigram
      // tokenizer, NTK-corrected RoPE with the frequency set carried as DATA
      // in the container (rope_inv_freq -- a consumer deriving it from
      // rope_theta alone is wrong by 1.9e-02 relfro at layer 0, tasks/0134),
      // gated GeGLU with the halves fused UPSTREAM, and real biases. Packs to
      // the same layout_hash as bge-base/nomic and runs on the same designs
      // (tasks/0135). `gte=true` selects the kFilesGte fetch list and (via
      // model_type "new" in the pack dispatch below) prepare_model_gte().
      // qkv_n is stated explicitly per the container though it equals
      // 3*hidden, so 0 would behave identically in design_fits().
      //
      // The model.safetensors pin below is the one ensure_model() enforces,
      // like every other row (the table's scheme pins exactly that file).
      // The tokenizer/config files were additionally pinned in tasks/0127
      // and re-verified against the local checkout in tasks/0138; recorded
      // here for traceability, NOT enforced by the fetch path:
      //   tokenizer.json           f59925fcb90c92b894cb93e51bb9b4a6105c5c24
      //                            9fe54ce1c704420ac39b81af
      //   tokenizer_config.json    24cebbf2ef20fc317256e03e52ac7b2ca326586f
      //                            946a8427ecac036332bf0933
      //   special_tokens_map.json  8c785abebea9ae3257b61681b4e6fd8365ceafde
      //                            980c21970d001e834cf10835
      //   config.json              711bdc81365fc25d30533cf05b9fdf588e5ba01f
      //                            18540fbbb1307d787597a313
      {"gte-multilingual-base", "Alibaba-NLP/gte-multilingual-base",
       "f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0",
       "cls", 768, 12, 12, 3072, 48, 582.5,
       "multilingual, XLM-R tokenizer; NTK RoPE + gated GeGLU (arch=3); "
       "same array designs as bge-base/nomic",
       /*gated=*/false, /*gemma=*/false, /*gated_ffn=*/true,
       /*qkv_n=*/2304, /*gte=*/true},
    };
    // THE bfp16 ADOPTION (tasks/0104, T23), set by NAME rather than by
    // rewriting every row's positional initialiser above -- the struct's
    // trailing fields (gated/gemma/gated_ffn/qkv_n) are themselves positional
    // and most rows already stop short of them, so reaching `datapath` from
    // the literal would mean restating every field in between for every row,
    // for a decision that has nothing to do with any of them. Real,
    // same-session MTEB gate verdicts (`--sides cpu,npu`, tasks/0101/0103):
    // five of six PASS at bfp16+bf16-C and are adopted; bge-small FAILS at
    // -0.5010 against the -0.5 line (bit-reproducible, not noise) and stays
    // on the plain-bf16 design it always shipped. See docs/CURRENT_STATUS.md
    // for the verdict table.
    //
    // ENUMERATED, not "everything except bge-small". The exclusion form was
    // written first and is a fail-open: a seventh built-in row added later
    // would inherit bfp16 without anyone ever having gated it, which is the
    // failure class docs/CURRENT_STATUS.md sec 4 lists five of. An allowlist
    // makes a new model default to plain bf16 -- CatalogEntry::datapath's own
    // default -- until someone measures it and adds it here.
    static const char *kAdoptedBfp16[] = {
        "all-MiniLM-L6-v2",         // MTEB +0.12 / worst -0.07
        "bge-base-en-v1.5",         // MTEB -0.06 / worst -0.19
        "bge-large-en-v1.5",        // MTEB +0.13 / worst -0.01
        "nomic-embed-text-v1.5",    // MTEB +0.01 / worst -0.25
        "embeddinggemma-300m",      // MTEB +0.16 / worst -0.02
        "gte-multilingual-base",    // MTEB +0.06 / worst -0.06 (tasks/0137)
    };
    for (auto &e : rows)
      for (const char *n : kAdoptedBfp16)
        if (e.name == n) { e.datapath = "bfp16"; break; }
    return rows;
  }();
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

// Gemma's file set: no vocab.txt (its tokenizer lives entirely in
// tokenizer.json + tokenizer_config.json, packed at build time into
// gemma_tokenizer.bin by tools/gen_gemma_tokenizer_table.py -- not fetched
// here, since ensure_model() only fetches the CHECKPOINT, not the generated
// table); needs the sentence-transformers prompt table and both Dense heads
// (tasks/0064's two post-pooling projections).
const Want kFilesGemma[] = {
    {"model.safetensors", true},
    {"config.json", true},
    {"tokenizer.json", true},
    {"tokenizer_config.json", true},
    {"config_sentence_transformers.json", true},
    {"1_Pooling/config.json", true},
    {"2_Dense/config.json", true},
    {"2_Dense/model.safetensors", true},
    {"3_Dense/config.json", true},
    {"3_Dense/model.safetensors", true},
};

// gte's file set (arch=3, model_type "new"): no vocab.txt -- its tokenizer
// is the XLM-R Unigram table, generated at pack time from tokenizer.json by
// prepare_model_gte() (or read from the cached xlmr_tokenizer.bin --
// tasks/0127/0133/0138). tokenizer_config.json and special_tokens_map.json
// are what the tokenizer verifier and any HF cross-check read; modules.json
// carries the 2_Normalize entry that makes l2_normalize the checkpoint's own
// claim rather than this runtime's (tasks/0135). Subdirectory paths are fine
// here for the same reason kFiles' 1_Pooling/config.json already is:
// download() creates the destination's parent directories.
const Want kFilesGte[] = {
    {"model.safetensors", true},
    {"config.json", true},
    {"tokenizer.json", true},
    {"tokenizer_config.json", true},
    {"special_tokens_map.json", true},
    {"1_Pooling/config.json", true},
    {"modules.json", true},
};

#ifdef _WIN32
std::wstring widen(const std::string &s) {
  if (s.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(),
                                    nullptr, 0);
  std::wstring w((size_t)n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
  return w;
}
#endif

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
#ifdef _WIN32
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
#endif

}  // namespace

namespace {
// Built-ins first, then whatever `add` wrote. ONE writer
// (`load_user_catalog`), called once from main() before anything reads this.
std::vector<CatalogEntry> &merged() {
  static std::vector<CatalogEntry> v = table();
  return v;
}
bool g_user_loaded = false;

std::filesystem::path user_catalog_path(const std::string &root) {
  return std::filesystem::path(root) / "models" / "catalog.json";
}
}  // namespace

const std::vector<CatalogEntry> &catalog() { return merged(); }

const CatalogEntry *find(const std::string &name) {
  for (const auto &e : merged())
    if (e.name == name) return &e;
  return nullptr;
}

void load_user_catalog(const std::string &root, const Log &log) {
  if (g_user_loaded) return;          // idempotent: main() may probe twice
  g_user_loaded = true;
  const auto p = user_catalog_path(root);
  std::error_code ec;
  if (!std::filesystem::exists(p, ec)) return;

  std::ifstream f(p, std::ios::binary);
  std::stringstream b;
  b << f.rdbuf();
  const std::string txt = b.str();
  if (txt.empty()) return;

  npue::json::Value doc;
  try {
    doc = npue::json::parse(txt);
  } catch (const std::exception &e) {
    // LOUD, not silent. A malformed user catalogue means `serve <name>` will
    // fail with "unknown model" for a model the user believes they added, and
    // the reason has to be visible at that moment.
    if (log) log("  WARNING: " + p.string() + " is not valid JSON (" +
                 e.what() + ") -- ignoring it; every model it added is now "
                 "unknown to this build");
    return;
  }
  if (!doc.is_object() || !doc.as_object().count("models")) return;

  for (const auto &m : doc.at("models").as_array()) {
    CatalogEntry e;
    auto s = [&](const char *k) -> std::string {
      return m.as_object().count(k) ? m.at(k).as_string() : std::string();
    };
    auto i = [&](const char *k, int64_t d) -> int64_t {
      return m.as_object().count(k) ? (int64_t)m.at(k).as_number() : d;
    };
    auto bl = [&](const char *k) -> bool {
      return m.as_object().count(k) && m.at(k).as_bool();
    };
    e.name = s("name");
    e.repo = s("repo");
    e.sha256 = s("sha256");
    e.pooling = s("pooling");
    e.hidden = i("hidden", 0);
    e.layers = i("layers", 0);
    e.heads = i("heads", 0);
    e.ffn = i("ffn", 0);
    e.tile_n = i("tile_n", 48);
    e.download_mb = (double)i("download_mb", 0);
    e.note = s("note");
    e.gated = bl("gated");
    e.gemma = bl("gemma");
    e.gated_ffn = bl("gated_ffn");
    e.qkv_n = i("qkv_n", 0);
    e.gte = bl("gte");
    if (e.name.empty() || e.repo.empty()) continue;
    // A user row must never shadow a built-in. `add` refuses to write one, so
    // reaching here means the file was hand-edited -- say so and keep the
    // built-in, whose pin this repository actually validated.
    bool shadows = false;
    for (const auto &b0 : table())
      if (b0.name == e.name) shadows = true;
    if (shadows) {
      if (log) log("  WARNING: " + p.string() + " redefines the built-in model '" +
                   e.name + "' -- ignoring the file's version and keeping the "
                   "built-in, whose checksum this build validated");
      continue;
    }
    merged().push_back(std::move(e));
  }
}

void add_to_user_catalog(const std::string &root, const CatalogEntry &e) {
  for (const auto &b0 : table())
    if (b0.name == e.name)
      throw std::runtime_error(
          "'" + e.name + "' is a built-in model. Refusing to shadow it: its "
          "checksum was validated against goldens by this repository, and a "
          "user row that replaced it would leave every table saying the same "
          "name while serving different weights. Pick another name.");
  for (const auto &u : merged())
    if (u.name == e.name)
      throw std::runtime_error("'" + e.name + "' is already in " +
                               user_catalog_path(root).string());

  merged().push_back(e);

  auto esc = [](const std::string &s) {
    std::string o;
    for (char c : s) {
      if (c == '"' || c == '\\') { o += '\\'; o += c; }
      else if (c == '\n') o += "\\n";
      else o += c;
    }
    return o;
  };
  std::string j = "{\n  \"comment\": \"Written by `npuembeddings add`. Rows "
                  "here are MERGED AFTER the built-in catalogue and may not "
                  "shadow it. A row whose sha256 is empty is NOT verified.\",\n"
                  "  \"models\": [\n";
  bool first = true;
  for (const auto &u : merged()) {
    bool builtin = false;
    for (const auto &b0 : table())
      if (b0.name == u.name) builtin = true;
    if (builtin) continue;
    if (!first) j += ",\n";
    first = false;
    j += "    {\"name\": \"" + esc(u.name) + "\", \"repo\": \"" + esc(u.repo) +
         "\", \"sha256\": \"" + esc(u.sha256) + "\", \"pooling\": \"" +
         esc(u.pooling) + "\", \"hidden\": " + std::to_string(u.hidden) +
         ", \"layers\": " + std::to_string(u.layers) + ", \"heads\": " +
         std::to_string(u.heads) + ", \"ffn\": " + std::to_string(u.ffn) +
         ", \"tile_n\": " + std::to_string(u.tile_n) + ", \"qkv_n\": " +
         std::to_string(u.qkv_n) + ", \"gated\": " +
         (u.gated ? "true" : "false") + ", \"gemma\": " +
         (u.gemma ? "true" : "false") + ", \"gated_ffn\": " +
         (u.gated_ffn ? "true" : "false") + ", \"gte\": " +
         (u.gte ? "true" : "false") + ", \"note\": \"" + esc(u.note) +
         "\"}";
  }
  j += "\n  ]\n}\n";

  const auto p = user_catalog_path(root);
  std::filesystem::create_directories(p.parent_path());
  std::ofstream of(p, std::ios::binary);
  of << j;
  if (!of) throw std::runtime_error("failed writing " + p.string());
}

void download(const std::string &url, const std::string &dest,
              const Log &log, const std::string &bearer_token) {
#ifdef _WIN32
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

  // A gated repo's actual bytes need the bearer token on EVERY hop,
  // including the redirect target -- WinHTTP re-sends the header
  // automatically on a same-origin-policy-compatible redirect, and
  // HuggingFace's resolve/main/ -> CDN redirect includes a pre-signed URL
  // that does not need it, so either way this is safe to always attach when
  // a token was given.
  const std::wstring auth_header =
      bearer_token.empty() ? std::wstring()
                           : L"Authorization: Bearer " + widen(bearer_token);
  if (!auth_header.empty() &&
      !WinHttpAddRequestHeaders(
          req, auth_header.c_str(), (DWORD)auth_header.size(),
          WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE))
    fail("cannot set Authorization header for " + url);

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
#else
  CURL *curl = curl_easy_init();
  if (!curl) throw std::runtime_error("cannot initialise libcurl");

  const std::filesystem::path final_path(dest);
  const std::filesystem::path part = final_path.string() + ".part";
  std::filesystem::create_directories(final_path.parent_path());

  std::ofstream out(part, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + part.string());

  // NOTE: no CURLOPT_ACCEPT_ENCODING — negotiating gzip would make the
  // decoded byte count disagree with Content-Length.
  struct Ctx { std::ofstream *out; const Log *log; uint64_t got = 0; int last_pct = -1; } ctx;
  ctx.out = &out; ctx.log = &log;

  auto write_cb = [](char *p, size_t sz, size_t nm, void *ud) -> size_t {
    auto *c = static_cast<Ctx *>(ud);
    const size_t n = sz * nm;
    c->out->write(p, static_cast<std::streamsize>(n));
    if (!*c->out) return 0;              // abort on write failure
    c->got += n;
    return n;
  };
  auto prog_cb = [](void *ud, curl_off_t tot, curl_off_t now,
                    curl_off_t, curl_off_t) -> int {
    auto *c = static_cast<Ctx *>(ud);
    if (!c->log || !*c->log || tot <= 0 || now <= 0) return 0;
    const int pct = static_cast<int>(now * 100 / tot);
    if (pct != c->last_pct && pct % 5 == 0) {
      c->last_pct = pct;
      (*c->log)("    " + std::to_string(pct) + "%  " + human(uint64_t(now)) +
                " of " + human(uint64_t(tot)));
    }
    return 0;
  };

  char errbuf[CURL_ERROR_SIZE] = {};
  curl_slist *headers = nullptr;
  std::string auth;
  if (!bearer_token.empty()) {
    auth = "Authorization: Bearer " + bearer_token;
    headers = curl_slist_append(headers, auth.c_str());
  }

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "NpuEmbeddings/0.2");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);   // HF -> CDN redirect
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 15000L);
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);  // stalled-CDN abort
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
  curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, prog_cb);
  curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &ctx);
  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);

  const CURLcode rc = curl_easy_perform(curl);
  long status = 0;
  curl_off_t clen = -1;
  if (rc == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &clen);
  }
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  out.close();

  if (rc != CURLE_OK)
    throw std::runtime_error("download failed for " + url + ": " +
                             (errbuf[0] ? std::string(errbuf)
                                        : std::string(curl_easy_strerror(rc))));
  if (status != 200)
    throw std::runtime_error("HTTP " + std::to_string(status) + " for " + url);
  if (clen > 0 && ctx.got != static_cast<uint64_t>(clen))
    throw std::runtime_error("short read on " + url + ": got " +
                             std::to_string(ctx.got) + " of " +
                             std::to_string(clen) + " bytes");

  std::error_code ec;
  std::filesystem::remove(final_path, ec);
  std::filesystem::rename(part, final_path, ec);
  if (ec)
    throw std::runtime_error("cannot rename " + part.string() + ": " + ec.message());
#endif
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

// A minimal string-field reader, same "flat, machine-generated" reasoning as
// json_int above. Used only to read config.json's own "model_type" so the
// packer dispatch below decides from the CHECKPOINT's stated architecture,
// never from a catalogue bit -- matching main.cpp's --prepare-model
// dispatch, and the reasoning in hub.hpp's CatalogEntry::gated_ffn comment
// for why arch identity should not be inferred from that flag.
std::string json_str(const std::string &s, const std::string &key) {
  const std::string k = "\"" + key + "\"";
  size_t p = s.find(k);
  if (p == std::string::npos) return {};
  p = s.find(':', p + k.size());
  if (p == std::string::npos) return {};
  const size_t q1 = s.find('"', p + 1);
  if (q1 == std::string::npos) return {};
  const size_t q2 = s.find('"', q1 + 1);
  if (q2 == std::string::npos) return {};
  return s.substr(q1 + 1, q2 - q1 - 1);
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

CatalogEntry probe_repo(const std::string &repo, const Log &log,
                        const std::string &token_override) {
  namespace fs = std::filesystem;
  std::string slug = repo;
  for (char &c : slug)
    if (c == '/' || c == '\\' || c == ':') c = '_';
  const fs::path tmp = fs::temp_directory_path() / ("npue_probe_" + slug);
  fs::create_directories(tmp);

  std::string bearer = token_override;
  if (bearer.empty())
    if (const char *t = std::getenv("HF_TOKEN"); t && *t) bearer = t;

  const std::string base = "https://huggingface.co/" + repo + "/resolve/main/";
  if (log) log("  reading " + repo + "/config.json");
  download(base + "config.json", (tmp / "config.json").string(), nullptr,
           bearer);
  const std::string cfg = slurp_text(tmp / "config.json");

  CatalogEntry e;
  e.repo = repo;
  e.name = repo.substr(repo.find_last_of('/') + 1);
  e.hidden = json_int(cfg, "hidden_size", 0);
  e.layers = json_int(cfg, "num_hidden_layers", 0);
  e.heads = json_int(cfg, "num_attention_heads", 0);
  e.ffn = json_int(cfg, "intermediate_size", 0);
  const std::string mt = json_str(cfg, "model_type");
  if (e.hidden <= 0 || e.layers <= 0 || e.heads <= 0 || e.ffn <= 0)
    throw std::runtime_error(
        repo + "/config.json does not carry the four geometry fields this "
        "runtime needs (hidden_size, num_hidden_layers, "
        "num_attention_heads, intermediate_size). It may not be an "
        "encoder checkpoint at all.");

  // WHICH ARCHITECTURE, from the checkpoint's own model_type -- the same
  // dispatch both packers use. A finetune inherits its base model's
  // model_type, which is exactly why finetunes are the supported case.
  e.gemma = (mt == "gemma3_text");
  e.gte = (mt == "new");  // the NewModel family (gte-multilingual-base):
                          // selects the kFilesGte fetch list and, at pack
                          // time, prepare_model_gte() -- tasks/0138
  e.gated_ffn = e.gemma || e.gte || (mt == "nomic_bert");
  e.gated = e.gemma;      // the Gemma family is licence-gated upstream

  // MQA/GQA narrows the fused qkv and then the packer pads it. Both facts are
  // DERIVED here, never copied from whatever model this was finetuned from --
  // a finetune that changed its head layout would otherwise be handed a
  // design built for the original (tasks/0074's T31-shaped fail-open).
  const int64_t kv = json_int(cfg, "num_key_value_heads", e.heads);
  const int64_t hd = json_int(cfg, "head_dim", e.hidden / std::max<int64_t>(e.heads, 1));

  // Largest legal tile_n: every N must divide `tile_n * 8` (8 columns), and
  // (64, tile_n) must fit the 63 KB L1 budget -- which rules out 64. This is
  // gemm_pretiled.py's own constraint, restated where the container is
  // described rather than where it is compiled.
  auto legal = [&](int64_t t, int64_t qkvn) {
    const int64_t ns[4] = {qkvn, e.hidden, e.gated_ffn ? 2 * e.ffn : e.ffn,
                           e.hidden};
    for (int64_t n : ns)
      if (n % (t * 8)) return false;
    if (e.hidden % 64 || e.ffn % 64) return false;
    return 2 * (64 * 64 * 2 + 64 * t * 2 + 64 * t * 4) < 64512;
  };
  const int64_t qkv_used = e.gemma ? (e.hidden + 2 * kv * hd) : 3 * e.hidden;
  e.tile_n = 0;
  for (int64_t t : {48, 32, 24, 16, 8}) {
    const int64_t gran = t * 8;
    const int64_t padded = ((qkv_used + gran - 1) / gran) * gran;
    if (legal(t, padded)) {
      e.tile_n = t;
      e.qkv_n = e.gemma ? padded : 0;   // only arch=1 states it
      break;
    }
  }
  if (e.tile_n == 0)
    throw std::runtime_error(
        "no legal tile geometry for " + repo + " (hidden " +
        std::to_string(e.hidden) + ", intermediate " + std::to_string(e.ffn) +
        "): its widths do not tile across 8 columns at any tile_n this "
        "runtime supports. The array cannot serve this shape without a new "
        "design, which `add` does not build.");

  if (log) log("  reading " + repo + "/1_Pooling/config.json");
  download(base + "1_Pooling/config.json",
           (tmp / "pooling.json").string(), nullptr, bearer);
  const std::string pj = slurp_text(tmp / "pooling.json");
  const bool cls = pj.find("\"pooling_mode_cls_token\": true") != std::string::npos ||
                   pj.find("\"pooling_mode_cls_token\":true") != std::string::npos;
  const bool mean = pj.find("\"pooling_mode_mean_tokens\": true") != std::string::npos ||
                    pj.find("\"pooling_mode_mean_tokens\":true") != std::string::npos;
  if (cls == mean)
    throw std::runtime_error(
        repo + "/1_Pooling/config.json asks for a pooling mode this runtime "
        "does not implement (it does cls and mean). Refusing rather than "
        "approximating it.");
  e.pooling = cls ? "cls" : "mean";
  e.note = "added locally from " + repo;

  std::error_code ec;
  fs::remove_all(tmp, ec);
  return e;
}

std::string ensure_model(const std::string &root, const std::string &name,
                         const Log &log, const std::string &token_override) {
  namespace fs = std::filesystem;
  const fs::path models = fs::path(root) / "models";
  const fs::path container = models / (name + ".npue");

  if (fs::exists(container)) return container.string();

  const CatalogEntry *e = find(name);
  if (!e) {
    std::string known;
    for (const auto &c : merged()) known += "\n    " + c.name;
    throw std::runtime_error(
        "'" + name + "' is not installed and is not a model this build knows "
        "how to fetch. Known models:" + known +
        "\n  A container you packed yourself is used by name once it is in "
        "models/.");
  }

  // GATED repositories need a token, and this FAILS CLOSED rather than
  // falling back to an ungated mirror -- that fallback is a research-only
  // shortcut (reference/fetch_model_gemma.py does it and says so in its own
  // output); production code does not get to decide on the user's behalf
  // that a third-party mirror is an acceptable substitute for the model the
  // catalogue actually names.
  //
  // Precedence: an explicit `--token` (token_override) wins if given;
  // otherwise fall back to the HF_TOKEN environment variable. Neither value
  // is ever logged -- only whether one was found.
  std::string bearer_token;
  if (e->gated) {
    if (!token_override.empty()) {
      bearer_token = token_override;
    } else if (const char *tok = std::getenv("HF_TOKEN"); tok && *tok) {
      bearer_token = tok;
    }
    if (bearer_token.empty())
      throw std::runtime_error(
          "'" + e->name + "' (" + e->repo + ") is a GATED model. Fetching it "
          "needs two things this run does not have:\n"
          "    1. accept the model's licence once, at "
          "https://huggingface.co/" + e->repo + "\n"
          "    2. a HuggingFace access token for that account, either: pass "
          "--token <value> on the command line, or set the HF_TOKEN "
          "environment variable\n"
          "  Refusing to fall back to an ungated mirror -- that is a "
          "research-only shortcut, not something this build does silently.");
    if (e->sha256.empty())
      throw std::runtime_error(
          "'" + e->name + "' is catalogued as gated but carries no verified "
          "sha256 pin in this build (CLAUDE.md rule 6: a checksum pin is a "
          "result, and needs a session that actually held HF_TOKEN to "
          "produce one). Fetch and verify it once, then record the real "
          "hash in runtime/src/hub.cpp's table() before this can run.");
  }

  const fs::path dir = models / name;
  if (log) {
    log("");
    log("  " + e->name + " is not installed. Fetching it from " + e->repo +
        ".");
    // Say which of the two this actually is. The old line claimed a
    // verification unconditionally, which would have been a lie for a row
    // `add` wrote without a pin -- and `download_mb` is 0 for those, because
    // nothing probed the size.
    if (unpinned(*e))
      log("  NO CHECKSUM PIN for this model -- it was added locally and will "
          "NOT be verified.");
    else
      log("  " + human(uint64_t(e->download_mb * 1024 * 1024)) +
          " of checkpoint, verified against a checksum built into this "
          "executable.");
    log("");
  }

  const std::string base =
      "https://huggingface.co/" + e->repo + "/resolve/main/";
  const auto fetch_list = [&]() -> std::vector<Want> {
    if (e->gemma)
      return std::vector<Want>(std::begin(kFilesGemma), std::end(kFilesGemma));
    if (e->gte)
      return std::vector<Want>(std::begin(kFilesGte), std::end(kFilesGte));
    return std::vector<Want>(std::begin(kFiles), std::end(kFiles));
  }();
  for (const auto &w : fetch_list) {
    const fs::path dest = dir / w.rel;
    if (fs::exists(dest)) {
      if (log) log("  have  " + std::string(w.rel));
      continue;
    }
    if (log) log("  get   " + std::string(w.rel));
    download(base + w.rel, dest.string(), log, bearer_token);
  }

  // The check that used to be `certutil` in a batch file. Same comparison,
  // same pin, no script.
  if (log) log("  hash  model.safetensors");
  const std::string got =
      npue::sha256_file((dir / "model.safetensors").string());
  if (unpinned(*e)) {
    // NO PIN, BY THE USER'S CHOICE (`add <repo>` with no sha256). We cannot
    // verify what we fetched, and pretending otherwise is worse than saying
    // so -- this is the one place in the fetch path that fails OPEN, and it
    // does it loudly. The digest is printed so it can be pinned afterwards.
    if (log) {
      log("");
      log("  !! WARNING: '" + e->name + "' was added WITHOUT a sha256 pin.");
      log("  !! These weights were NOT verified against anything. Whatever");
      log("  !! " + e->repo + " served just now is what will be packed.");
      log("  !! Its digest is:");
      log("  !!   " + got);
      log("  !! To pin it, re-add with that value:");
      log("  !!   npuembeddings add " + e->repo + " " + got);
      log("");
    }
  } else if (got != e->sha256) {
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
  if (log && !unpinned(*e)) log("        ok  " + got.substr(0, 16) + "...");

  // verify_config()'s numeric checks (hidden_size/num_hidden_layers/
  // num_attention_heads/intermediate_size) and its 1_Pooling/config.json
  // pooling check both read key names Gemma's config.json carries too
  // (confirmed tasks/0066) -- no arch branch needed here, only in the fetch
  // list above and the packer dispatch below.
  verify_config(*e, dir);

  // The pin, written where pack_npue.py and --prepare-model both look for it.
  // Byte-for-byte the layout reference/fetch_model.py writes (json.dumps with
  // indent=2 and no trailing newline), so a checkpoint re-fetched by the
  // executable does not show up as a diff against one fetched by the Python
  // path. Two writers of one file should not disagree about its formatting.
  {
    // For an UNPINNED row, record the digest actually received rather than
    // the empty pin: the packer stamps this into the container as
    // `source_sha256`, and a container recording "" would lose the only
    // evidence of which bytes it was built from. The distinction between
    // "verified against a pin" and "this is merely what arrived" lives in the
    // catalogue, which is where a reader can act on it.
    std::ofstream cf(dir / "CHECKPOINT.json", std::ios::binary);
    cf << "{\n  \"repo_id\": \"" << e->repo
       << "\",\n  \"file\": \"model.safetensors\",\n  \"sha256\": \""
       << (unpinned(*e) ? got : e->sha256) << "\"\n}";
  }

  if (log) log("  pack  " + container.filename().string());
  if (e->gemma) {
    prepare_model_gemma(dir.string(), container.string(), e->repo, nullptr);
  } else if (json_str(slurp_text(dir / "config.json"), "model_type") ==
            "nomic_bert") {
    // arch=2 (tasks/0071): read from the CHECKPOINT's own config.json, not
    // from `gated_ffn` -- see the comment on that field in hub.hpp and on
    // this catalogue row above.
    const Layout layout = gemm_b_layout(64, e->tile_n);
    prepare_model_nomic(dir.string(), e->pooling, e->repo, container.string(),
                        layout.json, layout.hash, 64, e->tile_n, 256,
                        nullptr);
  } else if (json_str(slurp_text(dir / "config.json"), "model_type") ==
            "new") {
    // arch=3 (tasks/0135-0138): same dispatch rule as the nomic branch --
    // the CHECKPOINT's own model_type, never a catalogue bit. max_seq is 64,
    // matching the Python-packed container this mirror is held byte-identical
    // to (tasks/0135 packed --max-seq 64): under RoPE the position table is
    // zeros, so max_seq only caps request length, and the shipped designs
    // and goldens for this model are seq-64.
    const Layout layout = gemm_b_layout(64, e->tile_n);
    prepare_model_gte(dir.string(), e->pooling, e->repo, container.string(),
                      layout.json, layout.hash, 64, e->tile_n, 64,
                      nullptr);
  } else {
    const Layout layout = gemm_b_layout(64, e->tile_n);
    prepare_model((dir / "model.safetensors").string(),
                  (dir / "vocab.txt").string(),
                  (dir / "config.json").string(), e->pooling, e->repo,
                  container.string(), got, layout.json, layout.hash, 64,
                  e->tile_n, 256, nullptr);
  }

  if (log) {
    log("  ready " + container.string());
    log("");
  }
  return container.string();
}

}  // namespace hub
}  // namespace npue
