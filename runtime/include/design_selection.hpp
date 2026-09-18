//===- design_selection.hpp -----------------------------------------------*- C++ -*-===//
//
// Where the model and design directories live when nobody says
// (default_root), and which exported design set fits a given model
// geometry (design_fits, pick_artifacts), plus the gemm_rtp `streams`
// table parser.
//
// Split out of main.cpp verbatim.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace app {

// One entry of gemm_rtp's `streams` array: which instruction-stream slot
// runs which op at which batch tier. Parsed here rather than in npu_device
// because it is encoder policy, not device mechanics.
struct StreamEntry {
  std::string op, file;
  int64_t batch = 0, slot = 0, M = 0, K = 0, N = 0;
};

inline std::vector<StreamEntry> parse_streams(const std::string &json) {
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

// Where the model and design directories live, when nobody says.
//
// Two layouts must both work: an extracted release (exe beside models/ and
// gemm_rtp/) and the source tree (exe in runtime/build/). Probing for the
// directories rather than assuming a depth means neither is privileged, and a
// wrong guess reports what it looked for instead of failing later on a
// confusing missing-file error.
inline std::string default_root(const char *argv0) {
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
inline bool design_fits(const std::string &design_dir, int64_t hidden,
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
inline std::string pick_artifacts(const std::string &root, int64_t hidden,
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

}  // namespace app
