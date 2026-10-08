//===- gemma_mode.hpp -----------------------------------------------------*- C++ -*-===//
//
// The arch=1 (EmbeddingGemma) CLI wiring: two paths chosen from the
// CONTAINER's gemm_layout -- GemmaNpuEncoder on the array, or the host-only
// npue::GemmaEncoder as the discriminating control (--cpu). --serve goes
// through EmbedBackend like every other architecture.
//
// Split out of main.cpp verbatim.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/app_state.hpp"
#include "common/design_selection.hpp"
#include "runtime/design.hpp"        // prefer_embedded: the container's own set
#include "server/embed_backend.hpp"
#include "encoders/gemma_host_encoder.hpp"
#include "encoders/gemma_npu_encoder.hpp"
#include "common/host_kernels.hpp"
#include "common/model_catalog.hpp"
#include "common/npu_ops_flag.hpp"   // parse_npu_ops, the one --npu-ops parser
#include "runtime/npu_contention.hpp"
#include "common/hub.hpp"
#include "runtime/design.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "tokenizers/gemma.hpp"

namespace app {

// Refuse an unknown --prefix by LISTING the real ones, rather than throwing
// tokenizer_gemma's bare "no task prefix named X". Same standard tasks/0071
// set for nomic. An empty name is legal and means no prefix at all, which is
// sentence-transformers' own default for this checkpoint
// (`default_prompt_name: null`).
//
// OMITTING it is no longer legal when `required` (tasks/0118). This path used
// to default to "document" -- the last silent default in this runtime, and the
// worst-placed one, because EmbeddingGemma's table holds 14 prompts and the
// checkpoint itself names none of them as a default. `from_cli` is what
// separates "not given" from `--prefix ""`, which still means no prefix.
inline void check_gemma_prefix(const npue::GemmaTokenizer &tok,
                        const std::string &name, bool from_cli,
                        bool required) {
  if (required && !from_cli) {
    std::string all;
    const auto names = tok.prefix_names();
    for (size_t i = 0; i < names.size(); ++i) all += (i ? ", " : "") + names[i];
    throw std::runtime_error(
        "this model has task prefixes and one must be named: pass --prefix "
        "with one of [" + all + "], or --prefix \"\" for no prefix at all. "
        "Refusing to pick one for you -- a wrongly-prefixed embedding is "
        "correctly shaped and correctly normed, so nothing downstream can tell "
        "that the answer is wrong.");
  }
  if (name.empty()) return;
  const auto names = tok.prefix_names();
  if (std::find(names.begin(), names.end(), name) != names.end()) return;
  std::string all;
  for (size_t i = 0; i < names.size(); ++i)
    all += (i ? ", " : "") + names[i];
  throw std::runtime_error("--prefix '" + name + "' is not a task prefix this "
                           "model defines. It has: " + all +
                           ". Pass --prefix \"\" for no prefix at all.");
}

inline int run_gemma_mode(npue::File &model, const std::string &model_path,
                   const std::string &root, int argc, char **argv) {
  auto has_flag = [&](const char *f) {
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == f) return true;
    return false;
  };
  auto flag_val = [&](const char *f, std::string dflt) {
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == f) return std::string(argv[i + 1]);
    return dflt;
  };

  // --serve [port] [--bind addr] (tasks/0115, T34's last unbuilt item). This
  // used to refuse. Nothing about arch=1 was ever incompatible with an HTTP
  // endpoint -- the endpoint was simply written against the BERT encoder's
  // type. serve_http() above now takes the three things it actually needs, so
  // this path supplies them like any other.
  int serve_port = -1;
  std::string serve_bind = "127.0.0.1";
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--serve") {
      serve_port = 8080;
      if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0])))
        serve_port = std::atoi(argv[i + 1]);
    }
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--bind") serve_bind = argv[i + 1];

  const int max_len = std::atoi(flag_val("--max-len", "64").c_str());
  // NO DEFAULT any more (tasks/0118). This was `flag_val("--prefix",
  // "document")` -- a task prompt nobody asked for, silently applied, on a
  // model with 14 of them. check_gemma_prefix() enforces it below; `serve`
  // rejects the flag outright because its prompt is per request.
  const bool has_prefix_flag = has_flag("--prefix");
  const std::string prefix = flag_val("--prefix", "");
  const bool force_cpu = has_flag("--cpu");
  if (serve_port > 0 && has_prefix_flag)
    throw std::runtime_error(
        "--prefix does not apply to `serve`: the task prompt is chosen per "
        "request now. Send \"prompt_name\" in the POST body instead, and GET "
        "/health lists the names this model accepts.");

  // --npu-ops is REFUSED here, not honoured -- and not by default
  // either, which is the bug this block fixes. Every code it names is an
  // eltwise design directory (gelu / layernorm / softmax), and an arch=1
  // design set carries gemm_rtp and nothing else: GemmaNpuEncoder computes
  // RMSNorm, softmax and GeGLU on the host and reads no per-op host/array
  // choice at all, so there is no flag to switch and no stream to switch it
  // to. The flag would parse, set nothing and change nothing -- measured as
  // bit-identical output (max|d| 0.000e+00), which is the "the flag was there
  // and nothing happened" failure run_setup.hpp refuses by name for the BERT
  // path and vit_mode.hpp refuses for the classifier.
  //
  // That measurement covered `attn` too, and `attn` is no longer part of it:
  // attention IS a separate pass here, and it now has a branch (see
  // GemmaNpuEncoder::attention). The codes that are still ignored -- `layn`,
  // `gelu` -- are named rather than silently accepted, exactly as before.
  std::set<std::string> npu_codes;   // the parsed --npu-ops, needed later too
  {
    std::string listing;
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--npu-ops") listing = argv[i + 1];
    npu_codes = parse_npu_ops(listing);
    if (!npu_codes.empty()) {
      // THREE CODES ARE HONOURED AND FIVE ARE REFUSED, and the split is by what a
      // refusal has to say about them rather than by whether they parse.
      //
      // `gemm`, `attn` and `softm` are honoured, and the rest are refused.
      //
      // `gemm` is the newest of the three and the one that changed the shape of
      // this block: the four per-layer GEMMs used to run on the array whenever
      // this encoder was built, with no code to select them, which made
      // `--npu-ops` a flag whose own documented rule ("host unless listed") was
      // false for its largest operation. GemmaNpuEncoder::gemm now asks
      // op_on_array("gemm") the way BertEncoder::gemm does, and everything else
      // below this line keeps meaning what it meant.
      //
      // `attn` and `softm` are honoured. Attention is a separate pass --
      // `attention(qkvbuf, ctx)` in GemmaNpuEncoder -- and takes the same
      // NpuAttention Whisper and the BERT encoder use; `softm` is the softmax
      // inside it, dispatched on art + "/softmax", its own xclbin, so the two
      // flags compose without either implying the other.
      //
      // The other six are refused, and they are refused for two different
      // reasons that are kept apart on purpose:
      //
      //   layn, gelu   this encoder runs RMSNorm and its GATED GeGLU on the
      //                host and has no per-op host/array choice to read --
      //                unlike the BERT encoder it takes no flag per op. So
      //                accepting them would print nothing, change nothing and
      //                hand back identical vectors, which is the failure this
      //                file exists to rule out. The registry marks these two
      //                cells blocked for this model, with the same reason.
      //
      //   conv, mproj, fft, logit   four operations of a DIFFERENT model. conv
      //                is Whisper's mel front end, mproj the mel filter bank,
      //                fft the 400-point transform, logit the vocabulary
      //                projection -- an embedder has none of them, so the
      //                registry marks these cells absent rather than blocked,
      //                and a refusal that called them blocked would tell a
      //                reader that exporting harder might work. (`gemm` is NOT
      //                among them: this encoder has four per-layer GEMMs of
      //                its own and moving them is exactly what it honours.)
      //
      // An earlier version of this comment said the exporter built
      // attn_qk/attn_av only under `kind == "stt"` and that n_kv came from
      // `frames`. Both were true when written and both were the export's
      // policy rather than the model's; they are fixed, and a refusal naming a
      // reason that no longer holds sends the reader to repair the wrong thing.
      const std::set<std::string> honoured = {"gemm", "attn", "softm"};
      std::string deadseen, onhost, absent;
      for (const auto &c : npu_codes) {
        if (honoured.count(c)) continue;
        if (!deadseen.empty()) deadseen += ", ";
        deadseen += c;
        if (c == "layn" || c == "gelu") {
          if (!onhost.empty()) onhost += ", ";
          onhost += c;
        } else {
          if (!absent.empty()) absent += ", ";
          absent += c;
        }
      }
      if (!deadseen.empty()) {
        std::string why;
        if (!onhost.empty())
          why += "\n    " + onhost + ": this encoder runs RMSNorm and its "
                "GATED GeGLU on the host and has no per-op host/array choice "
                "to read for them. Accepting them would print nothing, change "
                "nothing and hand back identical vectors -- measured as max|d| "
                "0.000e+00 -- and exporting a design will not help either: the "
                "exporter reads the same registry and marks these cells blocked "
                "for this model.";
        if (!absent.empty())
          why += "\n    " + absent + ": Whisper's audio front end, mel filter "
                "bank, 400-point transform and vocabulary projection. An "
                "embedder has no such operation, so these cells are absent -- "
                "there is nothing to move, and a design directory for them "
                "would be opened by nothing.";
        throw std::runtime_error(
            "--npu-ops " + deadseen + ":" + why +
            "\n  This architecture honours --npu-ops gemm, --npu-ops attn and "
            "--npu-ops softm.");
      }
    }
  }

  // TWO DIFFERENT LAYOUTS, and one name for both of them was a live bug.
  //
  // `gemm_layout` is the container's own config field: "pretiled_bf16" or
  // "host". It answers "were these operands pre-tiled for the array?", i.e.
  // whether an NPU run is possible at all.
  //
  // `want_layout` is the B-operand LAYOUT HASH the container records --
  // "52a4adadbddc..." for bf16, "177088d6bc9f..." for int8. It answers "which
  // of this model's two design sets is mine?".
  //
  // These were one variable named `layout`, and a change made earlier in this
  // file passed it to pick_artifacts() where the hash was wanted. The value it
  // passed was the string "pretiled_bf16", which matches no set's
  // b_layout_hash, so pick_artifacts() returned nothing, `art` stayed empty,
  // use_npu went false, and a perfectly good NPU container was handed to
  // gemma_host_encoder -- which then asked for layer.0.q_proj, a tensor only
  // the host packer emits, and failed with "no tensor named layer.0.q_proj".
  // The message points at the container and the cause is 200 lines earlier.
  //
  // The symptom only appears when --artifacts is NOT given: with it, `art` is
  // already set and pick_artifacts() is never called. Every sweep in this
  // session passed --artifacts (a container in /tmp has no sibling directory
  // to be found through), so the broken path was the one path never taken --
  // until the tail gate, which resolves by itself, hit it.
  std::string gemm_layout;
  try {
    gemm_layout = model.config_string("gemm_layout");
  } catch (const std::exception &) {
    gemm_layout = "host";    // a container packed before tasks/0074
  }

  // `--artifacts` is a NAME as often as a path -- every script in this repo
  // passes `artifacts_gemma`, not a path -- so resolve it against the same
  // three candidates the BERT path uses (cwd, an extracted release, the source
  // tree). Taking it verbatim made the sweep fail with "cannot open
  // artifacts_gemma/gemm_rtp/design.json", and the sweep then MISREPORTED that
  // exit code as a contention refusal.
  std::string art = flag_val("--artifacts", "");
  std::string want_layout;
  try {
    want_layout = model.info("layer.0.qkv").layout_hash;
  } catch (const std::exception &) {
    // A container that does not name it cannot be filtered on. An empty answer
    // means "cannot tell", which is not "does not match" -- see
    // select_set_for_layout's three rules.
  }
  // AN EXPLICIT --artifacts ONLY, for the same reason vit_mode gives: a
  // container that carries gemm_rtp must fall through to the pick_artifacts
  // branch below, which finds nothing on the disk and then loads the set out
  // of the container at line 473. Sending it here would ask "." for one.
  if (!art.empty()) {
    // Same shared candidate list as the BERT path (design_selection.hpp), now
    // including the per-model <name>/artifacts_npu<arch> layout. Only
    // gemm_rtp/design.json counts here: this arch loads gemm_rtp and nothing
    // else, unlike the BERT path which also accepts a qkv/ set.
    const std::vector<std::string> cands = artifacts_candidates(root, art);
    auto usable = [](const std::string &c) {
      return std::ifstream(c + "/gemm_rtp/design.json").good();
    };
    // AND IT IS FILTERED ON THE CONTAINER'S B LAYOUT, not just on holding a
    // design.json. This branch is the third place that had to learn this: it
    // took the first candidate with a gemm_rtp, so an int8 gemma container named
    // --artifacts embeddinggemma-300m, got the bf16 set of the same name, and
    // died at the first GEMM with "B layout mismatch -- design wants
    // 52a4adadbddc, container has 177088d6bc9f". Passing --artifacts is a
    // statement about WHERE, not about WHICH DATAPATH, and this branch had been
    // reading it as both. select_set_for_layout is shared with the BERT and
    // Whisper resolvers so the three cannot drift again.
    //
    // The note below compares against the first USABLE candidate, not against
    // cands.front(). Those are different things: artifacts_candidates proposes
    // several spellings of one name and front() is <root>/<name>/artifacts_npu1,
    // which usually holds nothing at all. Comparing against it prints a
    // correction on every single run -- including the bf16 runs, where nothing
    // was corrected -- and a note that is always true is a note nobody reads.
    std::string first_usable;
    for (const auto &c : cands)
      if (usable(c)) { first_usable = c; break; }
    std::string found = select_set_for_layout(cands, usable, want_layout);
    if (found.empty()) {
      std::string looked;
      for (size_t i = 0; i < cands.size(); ++i)
        looked += (i ? ", " : "") + cands[i];
      bool saw_usable = false;
      for (const auto &c : cands) saw_usable = saw_usable || usable(c);
      if (saw_usable && !want_layout.empty())
        throw std::runtime_error(
            "--artifacts '" + art +
            "' resolves only to design sets whose B-operand layout is not this "
            "container's (" + want_layout.substr(0, 12) +
            "...). They are the same shapes in different element types, so "
            "nothing in them can execute this file; looked under " + looked +
            ". Export a set for this datapath, or run a bf16 container.");
      throw std::runtime_error(
          "no design set found for --artifacts '" + art + "'; looked for "
          "gemm_rtp/design.json under " + looked);
    }
    if (found != first_usable && !want_layout.empty())
      std::fprintf(stderr,
                   "note: --artifacts %s resolves to a design set whose B "
                   "layout is not this container's (%s vs %s); using %s "
                   "instead.\n",
                   art.c_str(),
                   design_b_layout_hash(first_usable).substr(0, 12).c_str(),
                   want_layout.substr(0, 12).c_str(), found.c_str());
    art = found;
  }
  if (art.empty() && gemm_layout == "pretiled_bf16") {
    int64_t qn = 0;
    try { qn = model.config_int("qkv_n"); } catch (const std::exception &) {}
    // Look this model up by its catalogue name -- the container stem -- so
    // an adopted bfp16 datapath (tasks/0104) is required here exactly as it
    // is on the BERT serve/embed path. A container reached directly by path
    // (not through the catalogue) has no entry, so npue::hub::find() returns
    // nullptr and the default CatalogEntry::datapath, "bf16", applies.
    const std::string mname =
        std::filesystem::path(model_path).stem().string();
    const auto *ce = npue::hub::find(mname);
    art = pick_artifacts(root, model.config_int("hidden"),
                         model.config_int("intermediate"), true, qn,
                         // THE CONTAINER'S OWN B LAYOUT, and it is `want_layout` -- the hash from
                         // tensor metadata -- not `gemm_layout`. See the note
                         // where those two are declared: they were one
                         // variable, and passing this one matched nothing.
                         want_layout,
                         ce ? ce->datapath : "bf16", mname);
  }
  // THE SET, NOT THE PATH. `!art.empty()` read "a directory was resolved" as
  // "the array is available", and for a container that carries its own set no
  // directory is resolved -- so a 1.02 GB file with gemm_rtp inside it would
  // have run entirely on the host while line 473 was ready to load that set.
  const bool use_npu =
      !force_cpu && gemm_layout == "pretiled_bf16" &&
      npu::array_requested(&model, art);

  std::printf("NpuEmbeddings C++ runtime -- EmbeddingGemma (arch=1)\n");
  std::printf("  model      %s\n", model_path.c_str());

  // Read the input up front: both paths want the same texts, and a missing
  // file should fail before an xclbin is loaded.
  std::vector<std::string> texts;
  std::string in_path, out_path;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--embed") {
      in_path = argv[i + 1];
      if (i + 2 < argc && argv[i + 2][0] != '-') out_path = argv[i + 2];
    }
  if (!in_path.empty()) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + in_path);
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      texts.push_back(line);
    }
  } else {
    texts.push_back(
        "The AMD Ryzen AI NPU accelerates transformer encoder models.");
  }

  auto write_out = [&](const std::vector<float> &out, int64_t hidden) {
    if (!out_path.empty()) {
      std::ofstream of(out_path, std::ios::binary);
      of.write(reinterpret_cast<const char *>(out.data()),
               static_cast<std::streamsize>(out.size() * sizeof(float)));
      if (!of) throw std::runtime_error("failed writing " + out_path);
      std::printf("  wrote      %s  [%zu, %lld] fp32\n", out_path.c_str(),
                  texts.size(), (long long)hidden);
    } else {
      for (size_t b = 0; b < std::min<size_t>(texts.size(), 4); ++b) {
        std::printf("  [%zu]", b);
        for (int64_t c = 0; c < 6; ++c)
          std::printf(" %+.4f", out[b * static_cast<size_t>(hidden) + c]);
        std::printf(" ...\n");
      }
    }
  };

  if (!use_npu) {
    npue::GemmaHostEncoder enc(model);
    std::printf("  hidden     %lld, tokenizer %zu tokens\n",
                (long long)enc.hidden(), enc.tok.vocab_size());
    // Say WHY the host path was taken. "cpu" with no reason is the kind of
    // silent downgrade that gets measured and reported as if it were the
    // fast path.
    std::printf("  path       HOST-only (%s)\n",
                force_cpu             ? "--cpu given"
                : gemm_layout != "pretiled_bf16"
                    ? "container holds row-major F32 operands"
                    : "no matching NPU design set found");
    std::printf("  NOTE: every op runs on the CPU. Any seq/s below is "
                "host-only and is NOT an NPU performance claim "
                "(CLAUDE.md rule 1).\n");
    check_gemma_prefix(enc.tok, prefix, has_prefix_flag, serve_port <= 0);
    if (serve_port <= 0)
      std::printf("  input      %zu texts, max_len=%d, prefix='%s' -> %s\n",
                  texts.size(), max_len, prefix.c_str(),
                  prefix.empty() ? "(none)"
                                 : enc.tok.prefix_text(prefix).c_str());
    auto host_embed = [&](const std::vector<std::string> &txts,
                          const std::string &pname, int64_t *tokens) {
      std::vector<float> o(txts.size() * static_cast<size_t>(enc.hidden()));
      if (tokens) *tokens = 0;
      for (size_t t = 0; t < txts.size(); ++t) {
        // Tokenized twice on this path -- once to report the count, once
        // inside encode_one. On a path that is already host-only and orders
        // of magnitude slower than the array, that is the cheap way to give
        // `usage.prompt_tokens` a real number instead of a zero.
        if (tokens)
          *tokens += enc.tok.encode(txts[t], max_len, pname).n_tokens;
        const auto v = enc.encode_one(txts[t], max_len, pname, t,
                                      g_allow_truncation);
        std::memcpy(o.data() + t * v.size(), v.data(),
                    v.size() * sizeof(float));
      }
      return o;
    };

    // --serve works here too (tasks/0115). It must: `serve_port` is parsed
    // before this branch, so leaving it unhandled would have made
    // `--serve --cpu` embed the placeholder sentence once and exit -- a
    // silent no-op, the failure shape this project keeps finding.
    if (serve_port > 0) {
      EmbedBackend be;
      be.vocab_size = enc.tok.vocab_size();
      // Straight from the GEMATOK1 blob -- arch 1's prompts live there, not in
      // the container config the BERT path reads. serve_http() only ever sees
      // the names.
      be.prompt_names = enc.tok.prefix_names();
      std::sort(be.prompt_names.begin(), be.prompt_names.end());
      be.hidden = enc.hidden();
      be.seq = max_len;
      be.embed = host_embed;
      // "-cpu", not "-npu": the model id is what a client sees, and naming a
      // host-only server after the array is exactly the silent mislabel the
      // "path HOST-only" line above exists to prevent.
      return serve_http(be,
                        std::filesystem::path(model_path).stem().string() + "-cpu",
                        serve_port, serve_bind);
    }

    const double t0 = now_s();
    std::vector<float> out = host_embed(texts, prefix, nullptr);
    const double el = now_s() - t0;
    std::printf("  embedded   %zu texts in %.2f s  ->  %.2f seq/s (host-only)\n",
                texts.size(), el, texts.size() / std::max(el, 1e-9));
    write_out(out, enc.hidden());
    return 0;
  }

  // ---- NPU path -----------------------------------------------------------
  //
  // The contention gate, when this run is a MEASUREMENT (tasks/0044's ninth
  // fail-open). A foreign Active hw_context -- most often a stale npuembeddings.exe
  // from an earlier command in the same session -- read MiniLM at 221 seq/s
  // against a true 691. It is opt-in here rather than always-on because an
  // ordinary `embed` is not a performance claim and should not refuse to run
  // because something else is using the array; `--guard-contention` is what
  // tools/release_benchmark.ps1 passes.
  if (has_flag("--guard-contention") &&
      !npu::require_exclusive_npu(npu::survey_contexts(),
                                  has_flag("--allow-contention")))
    return 2;

  npu::Device dev;
  // ONE SOURCE FOR THE CORE, THE STREAM TABLE AND THE INSTRUCTION STREAMS.
  // Taking the design.json from disk while the xclbin came out of the
  // container would pair a stream list with a core that does not have those
  // streams, and that reads as a dispatch mismatch rather than as "these two
  // do not belong together".
  const npu::DesignSource gemm_src = npu::prefer_embedded(&model, art, "gemm_rtp");
  npu::Design d(dev, gemm_src);
  std::vector<StreamEntry> streams = parse_streams(gemm_src.text("design.json"));
  if (streams.empty())
    throw std::runtime_error(gemm_src.label() +
                             " has no streams in design.json -- re-export "
                             "with tools/export/export_gemm_rtp.py");
  std::sort(streams.begin(), streams.end(),
            [](const StreamEntry &a, const StreamEntry &b) {
              return a.slot < b.slot;
            });
  for (const auto &s : streams) {
    const size_t got = d.load_instr(gemm_src, s.file);
    if (static_cast<int64_t>(got) != s.slot)
      throw std::runtime_error("stream " + s.file + " landed in slot " +
                               std::to_string(got) + ", design.json says " +
                               std::to_string(s.slot));
  }

  if (d.info().seq <= 0)
    throw std::runtime_error("this design set records no sequence length");
  const int64_t seq = d.info().seq;
  if (max_len != seq)
    throw std::runtime_error(
        "--max-len " + std::to_string(max_len) + " but the design was compiled "
        "for seq " + std::to_string(seq) + ". The NPU path encodes at the "
        "design's own sequence length; use --cpu for another length, or "
        "re-export the design.");

  int nthreads = std::atoi(flag_val("--threads", "16").c_str());
  if (nthreads < 1) nthreads = 1;
  int nlanes = std::atoi(flag_val("--pipeline", "2").c_str());
  if (nlanes < 1) nlanes = 1;
  std::vector<std::unique_ptr<Pool>> pools;
  for (int l = 0; l < nlanes; ++l)
    pools.push_back(std::make_unique<Pool>(std::max(1, nthreads / nlanes)));

  // Per-lane array state, declared BEFORE the encoders so it outlives them:
  // a lane's NpuAttention holds a pointer into its NpuEltwise, and an object
  // destroyed while something still points at it is a bug that only fires on
  // the run after a failure. One softmax design per lane for the same reason
  // the BERT path gives every lane its own: the design has ONE input and ONE
  // output buffer, so two lanes dispatching into it would hand each other's
  // rows back -- thread-scheduling dependent, and worse, self-consistent.
  struct GemmaAttnLane {
    std::unique_ptr<npue::whisper::NpuEltwise> softmax;
    std::vector<std::unique_ptr<npue::whisper::NpuAttention>> attn;
  };
  std::vector<GemmaAttnLane> attn_lanes(1);   // lane 0 now; lanes 1+ below
  // The softmax design itself, open only when `softm` named it. Declared here
  // rather than inside the builder because it outlives every encoder: the
  // NpuEltwise a lane holds points into it, and a design destroyed one scope
  // earlier than the thing dispatching into it is a crash on teardown and
  // nothing between.
  std::unique_ptr<npu::Design> sm_design;
  npue::GemmaNpuEncoder enc(model, d, *pools[0]);
  // T37 (tasks/0082). Same switch as the BERT path, for the same reason: the
  // fused and unfused epilogues must both stay runnable so they can be A/B'd
  // byte-for-byte. Default on -- it is bit-identical and strictly less traffic.
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == "--no-fuse-ffn") enc.fuse_ffn_epilogue = false;
  enc.seq_ = seq;
  enc.batch = d.info().M / seq;
  enc.rows = enc.batch * seq;
  std::set<int64_t> tset;
  for (const auto &s : streams) tset.insert(s.batch);
  for (int64_t b : tset) {
    std::array<size_t, 6> slots{};
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
    if (!complete) continue;
    // The two attention streams, looked up the same way and left at ZERO when
    // absent. Zero rather than absent-from-the-row: a tier row has to be the
    // same shape whether or not this set was exported with `attn`, and a row
    // that silently shrank for one export would be read by code that indexes it
    // four deep. The zero is then refused below, by tier, if and only if the
    // flag asked for `attn` -- because a set without them is a perfectly good
    // set for a run that did not ask.
    for (int k = 4; k < 6; ++k) {
      const char *want = (k == 4) ? "attn_qk" : "attn_av";
      auto it = std::find_if(streams.begin(), streams.end(),
                             [&](const StreamEntry &s) {
                               return s.batch == b && s.op == want;
                             });
      if (it != streams.end()) slots[k] = static_cast<size_t>(it->slot);
    }
    enc.tiers.push_back(b);
    enc.tier_slots.push_back(slots);
  }
  enc.slot_a = d.stage_alloc(0, d.info().buffer_bytes[0]);
  enc.slot_c = d.stage_alloc(2, d.info().buffer_bytes[2]);
  const size_t staged = enc.stage_all();

  // -- the array softmax, and the attention when `attn` was asked for ------
  //
  // The softmax is its OWN xclbin: art + "/softmax", its own design, its own
  // hw_context, its own input and output buffers per lane. That is what makes
  // `softm` a real choice on a model whose softmax has no pass of its own to
  // take -- it lives inside attention(), so the design does not need a pass to
  // ride, it needs a buffer to be handed a score row. This model's rows are
  // seq_ wide and the design is built at this target's n_kv, so the score rows
  // handed to it are padded and the tail is -1e30 (see
  // GemmaNpuEncoder::attention) -- the same value NpuAttention writes, proven
  // on the BERT path.
  //
  // The builder is a LAMBDA per lane, for the reason run_setup.hpp's identical
  // one is: every lane needs its own buffers on the shared design, and one
  // instance handed to several lanes answers with whichever lane finished last.
  // The dispatch mutex, declared HERE rather than forty lines below where the
  // extra lanes are built: build_attn_lane() needs it too, and an attention
  // dispatch that skips it tears another lane's `active` binding. Same object
  // for the softmax design as for the GEMM one, matching what BertEncoder says
  // at eltwise() -- one mutex, because the runtime already treats "the NPU is
  // being driven by another lane" as one condition rather than per-design
  // conditions, and a second mutex would only add a lock-order question.
  static std::mutex npu_mutex;

  auto build_attn_lane = [&](Pool &p, size_t lane_index) {
    GemmaAttnLane lane;
    if (npu_codes.count("softm")) {
      lane.softmax = std::make_unique<npue::whisper::NpuEltwise>(
          *sm_design, p, npue::whisper::EltwiseKind::Softmax);
      lane.softmax->alloc_buffers();
      lane.softmax->npu_mu = &npu_mutex;
    }
    if (!npu_codes.count("attn")) return lane;
    if (enc.tiers.empty())
      throw std::runtime_error(
          art + "/gemm_rtp: --npu-ops attn was given, but this set carries no "
          "complete tier at all (all four of qkv/attn_out/ffn_up/ffn_down), so "
          "there is nothing to attach an attention to. Re-export with "
          "tools/export/export_gemm_rtp.py.");
    for (size_t t = 0; t < enc.tier_slots.size(); ++t) {
      const std::array<size_t, 6> &row = enc.tier_slots[t];
      const StreamEntry *qk = nullptr, *av = nullptr;
      for (const auto &st : streams) {
        if (row[4] && static_cast<size_t>(st.slot) == row[4]) qk = &st;
        if (row[5] && static_cast<size_t>(st.slot) == row[5]) av = &st;
      }
      if (!qk || !av)
        throw std::runtime_error(
            art + "/gemm_rtp: --npu-ops attn was given, but batch tier " +
            std::to_string(enc.tiers[t]) + " carries no attn_qk/attn_av. The "
            "flag says these streams run on the array, and a set without them "
            "would answer from the host under a flag that says otherwise. "
            "Re-export this target with the registry's `attn` cell honoured.");
      if (qk->N != av->K)
        throw std::runtime_error(
            art + ": attn_qk's N is " + std::to_string(qk->N) +
            " and attn_av's K is " + std::to_string(av->K) +
            ". The score chunk travels from one to the other as the A operand, "
            "so the two are the same padded n_kv.");
      if (qk->K < enc.head_dim || qk->K % enc.head_dim)
        throw std::runtime_error(
            art + ": attn_qk's K is " + std::to_string(qk->K) +
            " and this container's head_dim is " + std::to_string(enc.head_dim) +
            ". The Q operand of a score is one head, so K is the head width "
            "padded UP to the design's tile_k -- never down to a head.");
      if (av->N < enc.head_dim)
        throw std::runtime_error(art + ": attn_av's N is " +
                                 std::to_string(av->N) + " and a head is " +
                                 std::to_string(enc.head_dim) + " wide.");
      if (lane.softmax && lane.softmax->cols() != qk->N)
        throw std::runtime_error(
            art + "/softmax has rows " + std::to_string(lane.softmax->cols()) +
            " wide and the attn streams' score row is " + std::to_string(qk->N) +
            ". The softmax design reduces along the whole row, so it has to be "
            "the width of the score row it is handed.");
      lane.attn.push_back(std::make_unique<npue::whisper::NpuAttention>(
          d, p, qk->N, enc.head_dim, av->N));
      auto &a = lane.attn.back();
      a->set_streams(row[4], row[5], qk->M, qk->K);
      // MULTI-QUERY, and both numbers are spelled out rather than derived: this
      // model has ONE key/value head against `heads` query heads, and a K|V
      // half-width of kv_w (256) and not d_model (768). Handing it d_model
      // would walk 768 floats past a 256-wide block on the second query head
      // and return a plausible-looking wrong answer.
      a->set_kv_geometry(enc.kv_w, enc.kv_heads);
      a->set_softmax(lane.softmax.get());
      a->set_npu_mutex(&npu_mutex);
      a->alloc_buffers();
    }
    (void)lane_index;
    return lane;
  };

  sm_design = npu_codes.count("softm")
                  ? std::make_unique<npu::Design>(
                        dev, npu::prefer_embedded(&model, art, "softmax"))
                  : nullptr;
  if (sm_design) {
    attn_lanes[0] = build_attn_lane(*pools[0], 0);
    enc.attns = std::move(attn_lanes[0].attn);
    enc.set_softmax(attn_lanes[0].softmax.get());
  } else if (npu_codes.count("attn")) {
    attn_lanes[0] = build_attn_lane(*pools[0], 0);
    enc.attns = std::move(attn_lanes[0].attn);
    enc.set_softmax(nullptr);
  }

  // Extra lanes. The staged weights and the tier table belong to the DESIGN,
  // not to a lane, and are copied rather than re-staged; only the A and C
  // buffers are per-lane. A lane missing the tier table would silently fall
  // back to the flat (0,1,2,3) slot contract, which under a 16-stream export
  // selects entirely the wrong shapes -- measured as 1-cos 1.0 on the BERT
  // path when exactly that happened (tasks/0037).
  std::vector<std::unique_ptr<npue::GemmaNpuEncoder>> extra;
  if (nlanes > 1) {
    enc.npu_mu = &npu_mutex;
    for (int l = 1; l < nlanes; ++l) {
      extra.push_back(std::make_unique<npue::GemmaNpuEncoder>(model, d, *pools[l]));
      npue::GemmaNpuEncoder &e2 = *extra.back();
      e2.seq_ = enc.seq_;
      e2.batch = enc.batch;
      e2.rows = enc.rows;
      e2.tiers = enc.tiers;
      e2.tier_slots = enc.tier_slots;
      e2.s_qkv = enc.s_qkv; e2.s_ao = enc.s_ao;
      // The host path's untiled operand, shared for the same reason the staged
      // slot vectors are shared below: one cache, mutexed, and WeightCache's
      // copy constructor is deleted so it cannot become four.
      e2.host_w_ = enc.host_w_;
      e2.s_fu = enc.s_fu;   e2.s_fd = enc.s_fd;
      e2.b_qkv = enc.b_qkv; e2.b_ao = enc.b_ao;
      e2.b_fu = enc.b_fu;   e2.b_fd = enc.b_fd;
      // The int8 scale vectors are the CONTAINER's, not a lane's -- same
      // reasoning as the staged weights above, and the same failure if
      // forgotten. tasks/0078 hit exactly this on the BERT encoder (lane 0
      // worked, lanes 1+ had no scales) and the guard it added is what caught
      // it here: `--pipeline 4` on an int8 Gemma threw "this encoder has no
      // quantisation scales" instead of returning wrong embeddings.
      e2.ws_qkv = enc.ws_qkv; e2.ws_ao = enc.ws_ao;
      e2.ws_fu = enc.ws_fu;   e2.ws_fd = enc.ws_fd;
      e2.as_qkv = enc.as_qkv; e2.as_ao = enc.as_ao;
      e2.as_fu = enc.as_fu;   e2.as_fd = enc.as_fd;
      e2.fuse_ffn_epilogue = enc.fuse_ffn_epilogue;
      // Its OWN attention and its OWN softmax, built after the tier table is
      // copied and BEFORE use_tier(), which is what reads attns. A lane whose
      // attns came after the tier pick would run one job with a null attn_ and
      // answer from the host, under a flag that says otherwise -- and only on
      // lanes 1+, which is the shape that hides it from a one-lane test.
      attn_lanes.push_back(build_attn_lane(*pools[l], attn_lanes.size()));
      e2.attns = std::move(attn_lanes.back().attn);
      e2.set_softmax(attn_lanes.back().softmax.get());
      e2.use_tier(enc.batch);
      e2.slot_a = d.stage_alloc(0, d.info().buffer_bytes[0]);
      e2.slot_c = d.stage_alloc(2, d.info().buffer_bytes[2]);
      e2.npu_mu = &npu_mutex;
      if (e2.tiers != enc.tiers)
        throw std::runtime_error("lane stream policy differs from lane 0");
    }
  }
  std::vector<npue::GemmaNpuEncoder *> all_lanes{&enc};
  for (auto &e : extra) all_lanes.push_back(e.get());

  std::printf("  hidden     %lld, tokenizer %zu tokens\n",
              (long long)enc.hidden_, enc.tok.vocab_size());
  // THE DISPATCH COUNT READS THE FLAG, and it was the one number in this block
  // that did not: it always printed `4 * layers` even when the run had asked
  // for nothing and dispatched zero times. `gemm` is the ninth code; with it
  // unnamed the four per-layer GEMMs are this process's, and saying a run did
  // 96 dispatches while the counter beside it read 0 was exactly the
  // self-contradiction this block exists to avoid.
  const bool gemm_on_array = enc.gemm_on_array();
  std::printf("  path       %s -- 4 GEMMs/layer x %lld layers = %lld "
              "%s, ONE xclbin, one hw_context\n",
              gemm_on_array ? "NPU" : "NPU encoder, host GEMMs",
              (long long)enc.layers, (long long)(gemm_on_array ? 4 * enc.layers : 0),
              gemm_on_array ? "dispatches" : "dispatches (--npu-ops gemm moves them)");
  // WHICH OF THE TWO -- same reason as vit_mode: a container carrying gemm_rtp
  // leaves art empty, and a blank where a path belongs reads as a printer bug.
  std::printf("  designs    %s  (%zu streams, %zu batch tiers)\n",
               (art.empty() &&
                npu::prefer_embedded(&model, "", "gemm_rtp").has("design.json"))
                   ? "the container's own design set (design/gemm_rtp; no directory needed)"
                   : art.c_str(),
               streams.size(), tset.size());
  // WHICH DATAPATH WAS ACTUALLY SELECTED (tasks/0104), read off the loaded
  // design, not off a flag -- the intention-not-value slip named at the
  // "staged" line just below has cost this project time before (tasks/0042,
  // 0081), and now applies to bfp16 too.
  if (!enc.d.info().datapath_recorded)
    std::printf("  datapath   UNRECORDED (design predates tasks/0104), "
                "C as %s\n",
                enc.d.info().c_elem_bytes == 2 ? "bf16" : "fp32");
  else
    std::printf("  datapath   %s MMAC, C as %s\n",
                enc.d.info().datapath_name(),
                enc.d.info().c_elem_bytes == 2 ? "bf16" : "fp32");
  // WHICH TOOLCHAIN BUILT THIS DESIGN (T39, tasks/0106) -- read off the
  // loaded design's toolchain.json, same UNRECORDED-not-guessed discipline
  // as the datapath line just above.
  if (!enc.d.info().toolchain_recorded)
    std::printf("  toolchain  UNRECORDED (design predates tasks/0106)\n");
  else
    std::printf("  toolchain  mlir_aie %s, peano %s, mlir-aie HEAD %s\n",
                enc.d.info().mlir_aie_version.c_str(),
                enc.d.info().peano_version.c_str(),
                enc.d.info().mlir_aie_git_head.c_str());
  std::printf("  qkv        N=%lld  (q[0,%lld) k[%lld,%lld) v[%lld,%lld), "
              "%lld zero-padded cols)\n",
              (long long)enc.qkv_n, (long long)enc.hidden_,
              (long long)enc.k_off, (long long)(enc.k_off + enc.kv_w),
              (long long)enc.v_off, (long long)(enc.v_off + enc.kv_w),
              (long long)(enc.qkv_n - enc.v_off - enc.kv_w));
  // Read the dtype off the design rather than asserting it: this line said
  // "bf16" over int8 weights the moment arch=1 got an int8 path, which is the
  // same intention-not-value slip as tasks/0042's tile banner (tasks/0081).
  std::printf("  staged     %.1f MB of tiled %s weights on the device\n",
              staged / 1e6,
              enc.d.info().a_elem_bytes == 1 ? "int8" : "bf16");
  // WHERE EACH PIECE OF THE ATTENTION RUNS, read off the objects built above
  // and never off the flag that led here -- the same rule and the same wording
  // run_setup.hpp's status block follows for the BERT path, because the two
  // blocks have to be readable side by side.
  //
  // This line used to be one unconditional "host ... MQA attention (2.3% of
  // MACs)", so a --npu-ops attn run that had just put 2 x heads x layers GEMMs
  // onto attn_qk/attn_av was described by a status saying the opposite of the
  // time beside it, and a --npu-ops softm run was described as doing nothing at
  // all while its softmax sat on the array for 1.5 s. A status that cannot be
  // reconciled with the numbers next to it is the failure this file is supposed
  // to rule out.
  const bool attn_on_array = !enc.attns.empty();
  const bool softm_on_array = sm_design != nullptr;
  auto where = [](const char *code, bool host, const npu::Design &d,
                  const char *host_note) {
    const NpuOp *op = find_npu_op(code);
    if (host) {
      std::printf("  %-6s %-10s on the HOST (fp32) -- %s\n", code,
                  op ? op->long_name : "?", host_note);
    } else {
      std::printf("  %-6s %-10s on the ARRAY (%s, arch %lld%s%s)\n", code,
                  op ? op->long_name : "?",
                  d.info().name.empty() ? d.info().kind.c_str()
                                       : d.info().name.c_str(),
                  (long long)d.info().arch,
                  d.info().device.empty() ? "" : " ",
                  d.info().device.c_str());
    }
  };
  // WHAT RAN WHERE, one row per op -- and `gemm` leads because it is the
  // operation with the most of them. The line beneath used to list only what
  // had always been host, so a run whose four GEMMs per layer were on the host
  // described itself by everything except them.
  where("gemm", !gemm_on_array, d,
        gemm_on_array ? "" : "4 GEMMs/layer x the qkv/attn_out/ffn_up/ffn_down "
                             "streams, on this process");
  std::printf("  host       RMSNorm x%lld, RoPE, GeGLU%s%s\n",
              (long long)(4 * enc.layers + 1),
              attn_on_array && softm_on_array
                  ? ""
                  : attn_on_array
                        ? ", MQA softmax"
                        : softm_on_array ? ", MQA QK^T and softmax.V"
                                         : ", MQA attention (2.3% of MACs)",
              gemm_on_array ? "" : ", 4 GEMMs per layer");
  where("attn", !attn_on_array, d,
        "MQA attention over the same 2.3% of MACs, on the host");
  where("softm", !softm_on_array,
        softm_on_array ? *sm_design : d,
        "softmax inside attention, on the host");
  // ALWAYS SAY WHICH PREFIX WAS APPLIED, and refuse an unknown name by
  // listing the real ones -- the standard tasks/0071 set for nomic, which this
  // arch had not been held to. It matters most for MTEB: the harness applies
  // the same prefix to the CPU side, and a silent mismatch would show up as a
  // datapath difference rather than as the harness bug it is.
  check_gemma_prefix(enc.tok, prefix, has_prefix_flag, serve_port <= 0);
  if (serve_port <= 0)
    std::printf("  input      %zu texts, seq=%lld, prefix='%s' -> %s\n",
                texts.size(), (long long)seq, prefix.c_str(),
                prefix.empty() ? "(none)"
                               : enc.tok.prefix_text(prefix).c_str());

  if (nlanes > 1)
    std::printf("  pipeline   %d concurrent lanes, one NPU mutex, %d host "
                "threads per lane\n", nlanes, pools[0]->size());

  // Right-size each group to the smallest tier that holds it, then pad the
  // last group by REPEATING its own last text rather than with empty strings:
  // a padded lane costs full array time either way, and repeating a real text
  // keeps the tokenizer on the same code path.
  struct Job {
    npue::GemmaNpuEncoder *e;
    size_t start = 0, take = 0;
    std::vector<std::string> group;
  };
  const int64_t tier_max = enc.tiers.empty() ? enc.batch : enc.tiers.back();

  // The lane loop, as a callable over an arbitrary batch of texts (tasks/0115).
  // It used to be written straight against the file-loaded `texts`, which is
  // the only reason --serve could not reuse it; the body below is unchanged
  // apart from taking its input as a parameter and accumulating token counts.
  auto embed_texts = [&](const std::vector<std::string> &texts,
                         const std::string &pname,
                         int64_t *tokens) -> std::vector<float> {
  std::vector<float> out(texts.size() * static_cast<size_t>(enc.hidden_));
  std::atomic<int64_t> tok_total{0};
  size_t done = 0;
  while (done < texts.size()) {
    std::vector<Job> jobs;
    for (size_t l = 0; l < all_lanes.size() && done < texts.size(); ++l) {
      const size_t remaining = texts.size() - done;
      const int64_t want =
          static_cast<int64_t>(std::min<size_t>(remaining,
                                                static_cast<size_t>(tier_max)));
      Job j;
      j.e = all_lanes[l];
      const int64_t b = j.e->use_tier(want);
      j.start = done;
      j.take = std::min<size_t>(static_cast<size_t>(b), remaining);
      j.group.reserve(static_cast<size_t>(b));
      for (int64_t i = 0; i < b; ++i)
        j.group.push_back(texts[std::min(done + static_cast<size_t>(i),
                                         texts.size() - 1)]);
      done += j.take;
      jobs.push_back(std::move(j));
    }
    // An exception thrown on a worker thread would otherwise call
    // std::terminate and lose the message. Captured, then rethrown on this
    // thread after every lane has been joined.
    //
    // An exception_ptr rather than a string, because the TYPE carries meaning
    // now: npue::InputTooLong must stay distinguishable from a runtime_error
    // all the way out, or a caller that maps it to a 4xx sees a 5xx instead.
    // Flattening it to `what()` here would be a fail-open one rethrow wide.
    std::vector<std::exception_ptr> errs(jobs.size());
    auto run_job = [&](size_t k) {
      try {
        int64_t nt = 0;
        const std::vector<float> v = jobs[k].e->encode_batch(
            jobs[k].group, pname, jobs[k].start, jobs[k].take, &nt);
        tok_total += nt;
        std::memcpy(out.data() + jobs[k].start * static_cast<size_t>(enc.hidden_),
                    v.data(),
                    jobs[k].take * static_cast<size_t>(enc.hidden_) *
                        sizeof(float));
      } catch (...) {
        errs[k] = std::current_exception();
      }
    };
    std::vector<std::thread> th;
    for (size_t k = 1; k < jobs.size(); ++k)
      th.emplace_back([&, k] { run_job(k); });
    run_job(0);
    for (auto &t : th) t.join();
    for (const auto &e : errs)
      if (e) std::rethrow_exception(e);
  }
  if (tokens) *tokens = tok_total.load();
  return out;
  };

  // --serve on arch=1 (T34's last unbuilt item, tasks/0115). Everything the
  // endpoint needs is now in hand, so it is the same server the BERT path
  // runs -- not a second implementation that could drift from it.
  if (serve_port > 0) {
    EmbedBackend be;
    be.vocab_size = enc.tok.vocab_size();
    // From the GEMATOK1 blob, like the host path above -- arch 1's prompts
    // live in the tokenizer, not in the container config the BERT path reads.
    be.prompt_names = enc.tok.prefix_names();
    std::sort(be.prompt_names.begin(), be.prompt_names.end());
    be.hidden = enc.hidden_;
    be.seq = seq;
    be.embed = [&](const std::vector<std::string> &t, const std::string &pn,
                   int64_t *n) {
      return embed_texts(t, pn, n);
    };
    return serve_http(be,
                      std::filesystem::path(model_path).stem().string() + "-npu",
                      serve_port, serve_bind);
  }

  const double t0 = now_s();
  std::vector<float> out = embed_texts(texts, prefix, nullptr);
  const double el = now_s() - t0;
  // Fold every lane's counters into lane 0 so the breakdown describes the run
  // rather than whichever lane happened to be reported.
  for (auto &e : extra) {
    enc.t_conv += e->t_conv; enc.t_in += e->t_in; enc.t_disp += e->t_disp;
    enc.t_out += e->t_out;   enc.t_bias += e->t_bias; enc.t_norm += e->t_norm;
    enc.t_attn += e->t_attn; enc.t_rope += e->t_rope; enc.t_geglu += e->t_geglu;
    enc.t_tok += e->t_tok;   enc.n_dispatch += e->n_dispatch;
    enc.n_elt_dispatch += e->n_elt_dispatch;
  }
  std::printf("  embedded   %zu texts in %.2f s  ->  %.1f seq/s (wall clock, "
              "end-to-end -- NOT an NPU kernel claim, CLAUDE.md rule 1)\n",
              texts.size(), el, texts.size() / std::max(el, 1e-9));
  // The softmax's OWN dispatches, when the attention put them on the array:
  // `n_dispatch` counts what went through the gemm_rtp xclbin's hw_context and
  // a softmax running in softmax/ has a context of its own, so without this
  // suffix `--npu-ops softm` reported the same 96 dispatches as a run that did
  // not ask for it while spending 1.5 s inside those dispatches.
  const std::string elt_note =
      enc.n_elt_dispatch > 0
          ? " + " + std::to_string(static_cast<long long>(enc.n_elt_dispatch)) +
                " softmax dispatches"
          : std::string();
  std::printf("  breakdown  npu %.0f ms (in %.0f, dispatch %.0f, out %.0f) | "
              "conv %.0f | bias %.0f | norm %.0f | attn %.0f | rope %.0f | "
              "geglu %.0f | tok %.0f  [%d dispatches%s]\n",
              (enc.t_in + enc.t_disp + enc.t_out) * 1e3, enc.t_in * 1e3,
              enc.t_disp * 1e3, enc.t_out * 1e3, enc.t_conv * 1e3,
              enc.t_bias * 1e3, enc.t_norm * 1e3, enc.t_attn * 1e3,
              enc.t_rope * 1e3, enc.t_geglu * 1e3, enc.t_tok * 1e3,
              enc.n_dispatch, elt_note.c_str());
  write_out(out, enc.hidden_);
  return 0;
}


// Early dispatch for arch=1 containers, resolved BEFORE --artifacts
// resolution -- which THROWS if no BERT NPU design is found under `root`, a
// precondition this arch does not share (it has no NPU design at all, by
// design -- see run_gemma_mode's own comment). Exceptions here are swallowed
// and reported as -1, so the caller falls through to the original resolution
// path, which reports the real error (missing model, ambiguous --model, ...)
// exactly as if this dispatch did not exist. The model is opened via mmap
// twice in the Gemma case, which is cheap and never touches the BERT path.
// Returns the process exit code, or -1 when the container is not Gemma.
inline int maybe_gemma_mode(const std::string &root, int argc, char **argv) {
  {
    std::string peek_path;
    try {
      peek_path = resolve_model_path(root, argc, argv);
    } catch (const std::exception &) {
    }
    if (!peek_path.empty()) {
      bool is_gemma = false;
      try {
        npue::File peek(peek_path);
        is_gemma = (peek.config_string("arch") == "gemma3_mqa_rope_geglu");
      } catch (const std::exception &) {
      }
      if (is_gemma) {
        npue::File gmodel(peek_path);
        return run_gemma_mode(gmodel, peek_path, root, argc, argv);
      }
    }
  }
  return -1;
}

}  // namespace app
