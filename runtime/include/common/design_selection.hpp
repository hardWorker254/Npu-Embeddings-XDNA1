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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace app {

// WHICH NPU GENERATION THIS PROCESS IS RUNNING ON (subtask 3).
//
// `design.json` has carried `arch` and `device` since the exporter learned to
// build two generations, but nothing read them at load, so a generation-1 set
// loaded silently on generation 2. Selection now compares the running device
// against the set's own record, so the choice is a fact about the hardware
// rather than about which directory sorted first.
//
// The running device is not discoverable from XRT here, so it comes from the
// environment -- NPU_DEVICE wins, then NPU2=1, then npu1, which is the fork's
// default. runtime.cpp overrides it from `--dev` before any selection runs.
inline std::string &running_device_storage() {
  static std::string dev = []() -> std::string {
    if (const char *d = std::getenv("NPU_DEVICE"); d && *d) return d;
    if (const char *n = std::getenv("NPU2"); n && *n && std::string(n) != "0")
      return "npu2";
    return "npu1";
  }();
  return dev;
}

inline const std::string &running_device() { return running_device_storage(); }

inline void set_running_device(const std::string &d) {
  if (!d.empty()) running_device_storage() = d;
}

inline int64_t device_arch(const std::string &device) {
  if (device == "npu1") return 1;
  if (device == "npu2") return 2;
  return 0;
}

// WHERE A MODEL'S DESIGN SET LIVES IN THE PER-MODEL LAYOUT (subtask 4).
//
// tools/export/export_gemm_rtp.py --target writes
// <root>/artifacts/<model>/artifacts_npu<N>/, so every per-model set in the
// tree shares one parent directory (`runtime/artifacts/`) and is distinguishable
// from runtime/'s hand-maintained source directories by name alone. That is what
// `ARTIFACTS_DIR` in tools/export/exporters/common/paths.py is for; this is the
// reader half of the same convention, and the two have to agree.
//
// The older layouts are still proposed, after it, because a design set is a
// build artifact and a tree can legitimately hold both: <root>/<model>/
// artifacts_npu<N> is where the exporter used to write, and <root>/<model>/
// is where the int8 sets landed before this, because the scan below looks a
// fixed depth down. These are CANDIDATES, not answers -- pick_artifacts tests
// each with design_fits, so a wrong generation is rejected by the design's own
// record rather than by its name. <arch> comes from the running device
// (npu1->1), so only this generation's directory is proposed first.
inline std::vector<std::string> model_set_candidates(
    const std::string &root, const std::string &model_name,
    const std::string &device = "") {
  std::vector<std::string> out;
  if (model_name.empty()) return out;
  namespace fs = std::filesystem;
  const std::string dev = device.empty() ? running_device() : device;
  const int64_t arch = device_arch(dev);
  const fs::path r(root);
  auto add = [&](const fs::path &p) {
    const std::string s = p.string();
    for (const auto &e : out)
      if (e == s) return;
    out.push_back(s);
  };
  if (arch > 0) {
    const std::string sub = "artifacts_npu" + std::to_string(arch);
    // The convention: runtime/artifacts/<model>/artifacts_npu<arch>.
    //
    // <model>-i8 is the SAME model with the int8 datapath, and it needs its own
    // directory because one directory holds one design set and the two do not
    // fit each other: `b_layout_hash` covers the B operand's dtype, so the int8
    // set hashes 177088d6 and the bf16 set 52a4adad. Before this layout the two
    // lived at runtime/<model>/artifacts_npu1/ and runtime/<model>-i8/gemm_rtp/,
    // which is the same idea spelled two ways -- and the second spelling only
    // worked because the fallback scan happened to be exactly two levels deep.
    // Both are proposed so a container is matched against ITS OWN set: an int8
    // container and a bf16 one for the same model both resolve, by name, and
    // design_fits still refuses the mismatched pairing at stage time.
    for (const std::string &name : {model_name, model_name + "-i8"}) {
      add(r / "runtime" / "artifacts" / name / sub);
      add(r / "artifacts" / name / sub);
      // Where the exporter used to write, still proposed so an existing tree
      // keeps working without a re-export.
      add(r / "runtime" / name / sub);
      add(r / name / sub);
    }
  }
  for (const std::string &name : {model_name, model_name + "-i8"}) {
    add(r / "runtime" / "artifacts" / name);
    add(r / "artifacts" / name);
    add(r / "runtime" / name);
    add(r / name);
  }
  return out;
}

// The candidate directory LIST for an explicit --artifacts NAME-or-PATH,
// shared by the BERT and arch=1 (EmbeddingGemma) paths so the two cannot
// drift. `art` is tried verbatim (an absolute path), then under <root>, then
// under <root>/runtime -- the three places this runtime ships. The per-model
// layout adds <name>/artifacts_npu<arch> under both roots AND the current
// <root>/artifacts/<name>/artifacts_npu<arch>, which is what makes
// `--artifacts <model>` work now that export writes one level deeper.
//
// The artifacts/ entries are not optional bookkeeping: several gates pass
// `--artifacts <model>` by NAME (verify_whisper_model.py does), so a list
// without them resolves nothing for exactly the invocation those gates make --
// and it fails as "no such directory" rather than as anything that points at
// the layout being wrong.
inline std::vector<std::string> artifacts_candidates(
    const std::string &root, const std::string &art,
    const std::string &device = "") {
  std::vector<std::string> out;
  if (art.empty()) return out;
  const std::string dev = device.empty() ? running_device() : device;
  const int64_t arch = device_arch(dev);
  // A NAME IS PROPOSED BOTH AS ITSELF AND WITH THE int8 SUFFIX, so that naming
  // a model means "this model's design set" rather than "the directory called
  // exactly this". model_set_candidates has always done both; this list did not,
  // and the two answering differently is how `--artifacts <model>` -- the form
  // several gates pass -- could not reach an int8 container's set even when it
  // was exported and sitting one directory away. The suffixed spellings come
  // AFTER the plain ones, so a model whose two sets would both fit keeps
  // choosing the plain one and nothing that worked before stops working.
  for (const std::string &name : {art, art + "-i8"}) {
    out.push_back(name);
    out.push_back(root + "/" + name);
    out.push_back(root + "/runtime/" + name);
    if (arch > 0) {
      const std::string sub = "artifacts_npu" + std::to_string(arch);
      // The convention first, so a name that exists in both layouts resolves to
      // the one the exporter writes today.
      out.push_back(root + "/artifacts/" + name + "/" + sub);
      out.push_back(root + "/runtime/artifacts/" + name + "/" + sub);
      out.push_back(root + "/" + name + "/" + sub);
      out.push_back(root + "/runtime/" + name + "/" + sub);
    }
  }
  return out;
}

// First string / integer field of a flat top-level JSON object. Enough for the
// two scalar keys this header reads, and tolerant of formatting.
inline std::string json_field_string(const std::string &js, const char *key) {
  const std::string k = std::string("\"") + key + "\"";
  const size_t i = js.find(k);
  if (i == std::string::npos) return std::string();
  const size_t q1 = js.find('"', js.find(':', i) + 1);
  if (q1 == std::string::npos) return std::string();
  const size_t q2 = js.find('"', q1 + 1);
  if (q2 == std::string::npos) return std::string();
  return js.substr(q1 + 1, q2 - q1 - 1);
}

inline int64_t json_field_int(const std::string &js, const char *key,
                              int64_t fallback) {
  const std::string k = std::string("\"") + key + "\"";
  const size_t i = js.find(k);
  if (i == std::string::npos) return fallback;
  try {
    return std::stoll(js.substr(js.find(':', i) + 1));
  } catch (...) {
    return fallback;
  }
}

// One entry of gemm_rtp's `streams` array: which instruction-stream slot
// runs which op at which batch tier. Parsed here rather than in npu_device
// because it is encoder policy, not device mechanics.
struct StreamEntry {
  std::string op, file;
  int64_t batch = 0, slot = 0, M = 0, K = 0, N = 0;
};

// Which --npu-ops code a design set's STREAM NAME belongs to.
//
// It exists for the status lines, which print one row per op group and have to
// decide which of their loaded streams belongs in which row. resolve.py names
// each stream after the operation that consumes it, so the name IS the answer
// -- except for the four per-layer GEMM streams, which are named qkv / attn_out
// / ffn_up / ffn_down (and the self- and cross- attention spellings) and are
// the default.
//
// The DEFAULT is `gemm` on purpose: a stream this function has never heard of
// is one of the four per-layer GEMMs until proven otherwise, so a new base
// stream appears on the GEMM row rather than silently disappearing from it. The
// alternative failure -- an extra name on a row -- was the one that already
// happened: attn_qk, attn_av, mel_proj and dft400 were printed under "encoder
// GEMMs" while that row said `npu`, i.e. five streams claimed for the array
// while they were on the host.
inline const char *code_for_stream(const std::string &op) {
  if (op.rfind("attn_qk", 0) == 0 || op.rfind("attn_av", 0) == 0) return "attn";
  if (op.rfind("mel_proj", 0) == 0) return "mproj";
  if (op.rfind("dft", 0) == 0) return "fft";
  if (op.rfind("conv", 0) == 0) return "conv";
  if (op.rfind("logits", 0) == 0) return "logit";
  return "gemm";
}

// The B-operand layout hash a design set declares, or "" if it declares none.
//
// THIS EXISTS BECAUSE `--artifacts <path>` SELECTED A DESIGN WITHOUT CHECKING
// IT AT ALL, and the hole was invisible until a container that could not run on
// the set got as far as the first GEMM. An explicit --artifacts was resolved by
// "does this directory contain a design.json" -- a test of the DIRECTORY, not
// of the fit -- so `embed <int8 container of bge-base> --artifacts
// bge-base-en-v1.5` resolved to the bf16 set and died at layer 0 with
// "layout mismatch -- design gemm_rtp wants 52a4adad..., file has
// 177088d6...". design_fits has compared b_layout_hash since tasks/0080, but
// only on the path where the RUNTIME picks the set; naming one by hand skipped
// it, which is the fail-open shape this file keeps meeting in a new place.
//
// Narrow on purpose: no geometry, no stream shapes, no datapath. Those are
// design_fits' job and it already does them, on the path that reaches it. This
// answers the one question design_fits cannot be asked here -- "is this
// directory's declared layout the layout in the file I was handed?" -- and an
// empty answer on either side is "cannot tell", which is not a mismatch.
//
// The set name is a parameter because a Whisper directory holds TWO gemm sets
// (gemm_rtp and gemm_rtp_dec) and the decoder's is a different layout.
inline std::string design_b_layout_hash(const std::string &design_dir,
                                        const std::string &set = "gemm_rtp") {
  std::ifstream f(design_dir + "/" + set + "/design.json");
  if (!f) return std::string();
  std::stringstream b;
  b << f.rdbuf();
  const std::string js = b.str();
  const std::string k = "\"b_layout_hash\"";
  const size_t a = js.find(k);
  if (a == std::string::npos) return std::string();
  const size_t q1 = js.find('"', js.find(':', a) + 1);
  if (q1 == std::string::npos) return std::string();
  const size_t q2 = js.find('"', q1 + 1);
  if (q2 == std::string::npos) return std::string();
  return js.substr(q1 + 1, q2 - q1 - 1);
}

// THE CANDIDATE WHOSE DECLARED B LAYOUT IS THE CONTAINER'S.
//
// `usable` is the caller's own test for "this directory is a design set I can
// actually run" -- for an embedder that is a gemm_rtp/design.json, for Whisper
// it is that file AND its gemm_rtp_dec sibling, because neither half
// transcribes anything alone. The two tests genuinely differ, so the helper
// takes the caller's rather than growing one that is right for neither.
//
// WHY IT EXISTS, having now been written three times' worth: every path that
// picks a design set by hand did it by asking only whether the DIRECTORY holds
// the right files, and never whether the DESIGN suits the container. That is
// invisible for bf16, where one set per model is the whole story, and wrong the
// moment a second set exists -- an int8 container resolved to the bf16 set of
// the same name and the run died at the first GEMM on a layout hash. The
// hash is on both sides and is the only thing that separates them, so it is
// checked here, once, for every caller.
//
// THREE RULES, all of which came from being wrong about them:
//   * An empty `want_layout` means the caller could not tell, and the first
//     usable candidate wins -- every pre-int8 call site behaves exactly as it
//     did before.
//   * A design set with NO declared hash is accepted rather than excluded. It
//     was exported before the field existed, so excluding it would break every
//     older set in the tree, and "cannot tell" is not "does not match".
//   * The candidate order is the caller's, untouched. Selection is a filter on
//     that order, never a reordering -- so a model whose two sets would both fit
//     still gets the one the caller proposed first.
inline std::string select_set_for_layout(
    const std::vector<std::string> &cands,
    const std::function<bool(const std::string &)> &usable,
    const std::string &want_layout) {
  for (const std::string &c : cands) {
    if (!usable(c)) continue;
    if (want_layout.empty()) return c;
    const std::string got = design_b_layout_hash(c);
    if (got.empty() || got == want_layout) return c;
  }
  return std::string();
}

// One top-level string field of a design set's design.json, or "" if absent.
// The design's `a_dtype` is what says which operand element type the array was
// built for, and it is not the same question as the container's
// `b_layout_hash`: the hash covers the B panels only, so a container and a
// design can agree perfectly on the hash while the design still expects a bf16
// A operand the runtime cannot supply. Reading the field directly is how that
// stops being invisible.
inline std::string design_field_string(const std::string &path,
                                       const char *key) {
  std::ifstream f(path);
  if (!f) return std::string();
  std::stringstream b;
  b << f.rdbuf();
  return json_field_string(b.str(), key);
}

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
  // HOW DEEP, AND WHY THE CAP. The layouts this has to recognise, shallowest
  // first:
  //
  //   <d>/gemm_rtp                                    a bare design set
  //   <d>/<set>/gemm_rtp                              one level down
  //   <d>/<model>/artifacts_npu<N>/gemm_rtp            the pre-2026 layout
  //   <d>/artifacts/<model>/artifacts_npu<N>/gemm_rtp  the current one
  //
  // so three levels is the deepest, and the walk stops there. The cap is not
  // only tidiness: this runs on whatever directory the process was launched
  // from, and when that is the repository root the alternative is walking
  // .git/objects. DOT-DIRECTORIES ARE SKIPPED for the same reason -- nothing
  // this looks for is ever inside one, and .git has ~256 fan-out at two
  // levels. The previous version was a hand-unrolled two-level loop; this is the
  // same search with the depth written down once instead of in the shape of the
  // code.
  constexpr int kMaxDesignDepth = 3;
  auto has_design = [&](const fs::path &d) {
    std::vector<fs::path> level{d};
    for (int depth = 0; depth <= kMaxDesignDepth && !level.empty(); ++depth) {
      std::vector<fs::path> next;
      for (const fs::path &p : level) {
        std::error_code ec3;
        if (fs::exists(p / "gemm_rtp", ec3)) return true;
        if (depth == kMaxDesignDepth) continue;
        for (fs::directory_iterator it(p, ec3), end;
             !ec3 && it != end; it.increment(ec3)) {
          if (!it->is_directory(ec3)) continue;
          const std::string n = it->path().filename().string();
          if (!n.empty() && n[0] == '.') continue;
          next.push_back(it->path());
        }
      }
      level.swap(next);
    }
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
                 const std::string &want_datapath = "",
                 const std::string &want_device = "") {
  if (hidden <= 0 || intermediate <= 0) return false;
  std::ifstream f(design_dir + "/gemm_rtp/design.json");
  if (!f) return false;
  std::stringstream b;
  b << f.rdbuf();
  const std::string js = b.str();
  const std::vector<StreamEntry> streams = parse_streams(js);
  if (streams.empty()) return false;

  // WHICH GENERATION (subtask 3). The set's own `device` must be the running
  // one; when only `arch` is recorded, its number must match. A set that
  // records neither predates the field and is accepted, which keeps every
  // design exported before the split working exactly as it did -- and the
  // check is fail-closed for every set that says anything at all. The same
  // "empty means caller did not say, use the running value" rule as the two
  // checks below; want_device is the running device, not a caller's intent.
  {
    const std::string want_dev =
        want_device.empty() ? running_device() : want_device;
    if (!want_dev.empty()) {
      const std::string dev = json_field_string(js, "device");
      if (!dev.empty()) {
        if (dev != want_dev) return false;
      } else {
        const int64_t arch = json_field_int(js, "arch", 0);
        const int64_t want_arch = device_arch(want_dev);
        if (arch != 0 && want_arch != 0 && arch != want_arch) return false;
      }
    }
  }

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
    // AN INT8 SET IS NOT A THIRD DATAPATH, and asking it which one it is has
    // no answer. `emulate_bfp16` says "run the MMACs as if the operands were
    // bfp16", and on the int8 datapath there are no bfp16 operands to emulate,
    // so the field is false there by construction -- not because this model
    // chose the plain-bf16 datapath. Comparing it against a target's "bfp16"
    // therefore refused the correct set for every int8 container of every
    // bfp16 target: 13 of the 16 models, i.e. all of them but bge-small,
    // bge-micro and the gemma/nomic pair that were not on bfp16. The symptom
    // was pick_artifacts() returning nothing, which drops use_npu to false and
    // hands an NPU container to the host encoder -- the same crash, from the
    // same place, as passing gemm_layout where the hash belongs.
    //
    // SKIPPED ONLY WHEN THE HASH ALREADY PROVED THE DATATYPE. The check above
    // returns false unless the design's b_layout_hash equals the container's,
    // and those hashes differ between bf16 and int8 by construction, so
    // reaching here with a non-empty want_layout means the two agree on element
    // type and this comparison is redundant. With want_layout empty -- the
    // pre-0080 callers -- the check still runs for int8 sets, because there the
    // hash has NOT been established and "datapath" is the only discriminator
    // left.
    const size_t ki = js.find("\"int8\"");
    bool is_int8 = false;
    if (ki != std::string::npos) {
      size_t p = js.find(':', ki) + 1;
      while (p < js.size() && (js[p] == ' ' || js[p] == '\n' || js[p] == '\t'))
        ++p;
      is_int8 = js.compare(p, 4, "true") == 0;
    }
    if (!is_int8 || want_layout.empty()) {
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
                           const std::string &want_datapath = "",
                           const std::string &model_name = "") {
  namespace fs = std::filesystem;
  if (hidden <= 0 || intermediate <= 0) return "";
  std::error_code ec;
  if (design_fits(root, hidden, intermediate, gated_ffn, qkv_n, want_layout,
                  want_datapath))
    return root;

  // THE PER-MODEL LAYOUT FIRST (subtask 4), by name rather than by scan: the
  // model's own directory is the authoritative place for its design set, and
  // naming it avoids a sibling model's set being chosen when widths coincide.
  // Candidates are tested with design_fits like every other candidate, so a
  // set built for the other generation is still refused.
  if (!model_name.empty())
    for (const auto &c : model_set_candidates(root, model_name))
      if (design_fits(c, hidden, intermediate, gated_ffn, qkv_n, want_layout,
                      want_datapath))
        return c;

  // Sorted, so the choice is reproducible rather than filesystem-order
  // dependent -- and never by mtime, which a JIT cache hit does not restamp
  // (CLAUDE.md trap 7c).
  std::vector<std::string> cands;
  for (const auto &base : {fs::path(root), fs::path(root) / "runtime"})
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
