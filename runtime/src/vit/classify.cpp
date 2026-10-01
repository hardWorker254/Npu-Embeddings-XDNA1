//===- classify.cpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one image-classification session. See classify.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "vit/classify.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/design_selection.hpp"
#include "common/host_kernels.hpp"
#include "vit/image.hpp"

namespace npue::vit {
namespace {

std::string read_text(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Every stream of a design set, loaded into the slot design.json records, and
// CHECKED against it: a set loaded in the wrong order runs the wrong
// instruction stream for an op and returns a plausible number.
std::vector<app::StreamEntry> load_streams(npu::Design &d,
                                           const std::string &dir) {
  std::vector<app::StreamEntry> streams =
      app::parse_streams(read_text(dir + "/design.json"));
  if (streams.empty())
    throw std::runtime_error(dir +
                             "/design.json lists no streams -- re-export with "
                             "tools/export/export_gemm_rtp.py");
  std::sort(streams.begin(), streams.end(),
            [](const app::StreamEntry &a, const app::StreamEntry &b) {
              return a.slot < b.slot;
            });
  for (const auto &s : streams) {
    const size_t got = d.load_instr(dir + "/" + s.file);
    if (static_cast<int64_t>(got) != s.slot)
      throw std::runtime_error("stream " + s.file + " landed in slot " +
                               std::to_string(got) + ", design.json says " +
                               std::to_string(s.slot));
  }
  return streams;
}

const app::StreamEntry &find_op(const std::vector<app::StreamEntry> &streams,
                                const std::string &op, int64_t batch) {
  for (const auto &s : streams)
    if (s.op == op && s.batch == batch) return s;
  throw std::runtime_error(
      "no " + op + " stream at batch tier " + std::to_string(batch) + " -- a "
      "classifier needs the four BERT-family streams and nothing else (see "
      "tools/data/npu_targets.json kinds.cls.streams); re-export with "
      "tools/export/export_gemm_rtp.py --target <model>");
}

}  // namespace

Session::Session(npue::File &model, const std::string &model_name,
                 const std::string &artifacts, int threads)
    : model_(model),
      geom_(read_geometry(model, model_name)),
      art_(artifacts),
      name_(model_name),
      dev_(std::make_unique<npu::Device>()),
      pool_(std::make_unique<app::Pool>(std::max(1, threads))),
      design_(std::make_unique<npu::Design>(*dev_, artifacts + "/gemm_rtp")),
      enc_(model, *design_, *pool_, geom_) {
  const std::vector<app::StreamEntry> streams =
      load_streams(*design_, artifacts + "/gemm_rtp");
  const std::vector<int64_t> tiers = [&] {
    std::vector<int64_t> t;
    for (const auto &s : streams) t.push_back(s.batch);
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    return t;
  }();
  if (tiers.size() != 1)
    throw std::runtime_error(
        artifacts + "/gemm_rtp has " + std::to_string(tiers.size()) +
        " batch tiers. One image is " + std::to_string(geom_.n_pos) +
        " positions, which is ONE dispatch of one tier; a second tier would "
        "mean a second chunking policy and this build refuses to guess which "
        "one it is. Export with a single tier, as kinds.cls does.");

  EncoderStreams es;
  es.qkv = static_cast<size_t>(find_op(streams, "qkv", tiers[0]).slot);
  es.attn_out = static_cast<size_t>(find_op(streams, "attn_out", tiers[0]).slot);
  es.ffn_up = static_cast<size_t>(find_op(streams, "ffn_up", tiers[0]).slot);
  es.ffn_down = static_cast<size_t>(find_op(streams, "ffn_down", tiers[0]).slot);
  es.rows = find_op(streams, "qkv", tiers[0]).M;
  enc_.set_streams(es);
  rows_ = es.rows;
  for (const auto &s : streams) {
    bool seen = false;
    for (const auto &have : ops_) seen = seen || have == s.op;
    if (!seen) ops_.push_back(s.op);
  }

  // The design's row count has to be at least one dispatch's worth of
  // positions, and its A buffer has to hold the widest K any one of the five
  // GEMMs needs. Both are the design's own numbers checked against the
  // container's, so a set exported for bge-base and a set exported for a
  // wider model are distinguished here rather than at the first dispatch.
  const npu::DesignInfo &di = design_->info();
  if (es.rows < geom_.n_pos)
    throw std::runtime_error(
        artifacts + "/gemm_rtp computes " + std::to_string(es.rows) +
        " rows per dispatch and this container has " +
        std::to_string(geom_.n_pos) +
        " positions. NpuGemm::run walks the tensor in chunks of the design's "
        "row count and refuses a chunk larger than one dispatch, so this "
        "container would need two dispatches of different geometry. Re-export "
        "with --seq " + std::to_string(geom_.n_pos) + ".");
  const int64_t widest_k = std::max(geom_.patch_dim, geom_.intermediate);
  const int64_t widest_n = std::max({geom_.d_model, 3 * geom_.d_model,
                                    geom_.intermediate});
  if (static_cast<size_t>(es.rows) * widest_k * di.a_elem_bytes >
          di.buffer_bytes[0] ||
      static_cast<size_t>(es.rows) * widest_n * di.c_elem_bytes >
          di.buffer_bytes[design_->output_index()])
    throw std::runtime_error(
        artifacts + "/gemm_rtp's buffers are too small for this container: " +
        std::to_string(es.rows) + " rows x " + std::to_string(widest_k) +
        " of A and x " + std::to_string(widest_n) +
        " of C do not fit. Re-export with tools/export/export_gemm_rtp.py "
        "--target " + name_ + ".");
  // THE OPERAND DTYPE PAIRING, refused here rather than at the first dispatch.
  // An int8 container against a bf16 design would fail on the layout hash --
  // which is a correct and load-bearing check -- but only after the weights
  // had been read off disk, and the message would be about a layout rather than
  // about the pairing the user actually got wrong.
  {
    const std::string a = model_.info("layer.0.qkv").dtype;
    const bool i8_design = di.a_elem_bytes == 1;
    if ((a == "I8") != i8_design)
      throw std::runtime_error(
          "container/design pairing: " + name_ + "'s GEMM operands are " + a +
          " and " + artifacts + "/gemm_rtp is a " +
          (i8_design ? "int8" : "bf16") +
          " design. These are two different containers and two different "
          "layout hashes; a design set is paired with one of them by name. "
          "Re-export with --int8 for an int8 container, or repack without "
          "--int8, or pass --artifacts for the matching set.");
  }

  staged_ = enc_.stage_all();

  // -- the label vocabulary ------------------------------------------------
  // One newline-separated name per label, from the packer. A table whose line
  // count is not num_labels is a container whose printed names would be offset
  // from its own logits, so it is refused rather than shown with an index
  // fallback for the rows past the end.
  {
    const npue::Span t = model_.raw("labels.table");
    const std::string blob(static_cast<const char *>(t.data), t.bytes);
    std::string cur;
    for (char c : blob) {
      if (c == '\n') {
        labels_.push_back(cur);
        cur.clear();
      } else {
        cur.push_back(c);
      }
    }
    if (!cur.empty()) labels_.push_back(cur);
    if (static_cast<int64_t>(labels_.size()) != geom_.num_labels)
      throw std::runtime_error(
          name_ + ": labels.table has " + std::to_string(labels_.size()) +
          " names and the head has " + std::to_string(geom_.num_labels) +
          " outputs. Printing label " + std::to_string(labels_.size()) +
          " as '" + (labels_.empty() ? std::string() : labels_.back()) +
          "' would name a class that is not the one the argmax chose. Repack "
          "with tools/pack/pack_npue.py.");
  }
}

std::vector<float> Session::preprocess_file(const std::string &path,
                                            const Geometry &g) {
  return npue::vit::preprocess_file(path, g);
}

Prediction Session::classify(const npue::vit::Image &im) {
  Prediction r;
  enc_.reset_timers();
  double t = app::now_s();
  const std::vector<float> patches = npue::vit::preprocess(im, geom_);
  r.front_end_s = app::now_s() - t;

  t = app::now_s();
  const std::vector<float> h = enc_.run(patches);
  r.encoder_s = app::now_s() - t;

  t = app::now_s();
  enc_.classify(h, r.logits);
  r.head_s = app::now_s() - t;
  r.n_dispatch = enc_.gemm().n_dispatch;

  // The argmax, and then the softmax over the FULL logit row -- the max is
  // taken first so the exponential cannot overflow, and the probability is
  // computed from the same row rather than from the top few, because a
  // truncated softmax is not a probability and reporting it as one is how a
  // confident-looking 0.99 comes out of a model that is guessing.
  if (r.logits.empty())
    throw std::runtime_error(name_ + ": the head produced no logits");
  int64_t best = 0;
  for (size_t i = 1; i < r.logits.size(); ++i)
    if (r.logits[i] > r.logits[best]) best = static_cast<int64_t>(i);
  r.label = best;
  float mx = r.logits[static_cast<size_t>(best)];
  double sum = 0.0;
  for (float v : r.logits) sum += std::exp(static_cast<double>(v) - mx);
  r.top1 = sum > 0.0 ? static_cast<float>(std::exp(0.0) / sum) : 0.f;
  return r;
}

Prediction Session::classify_file(const std::string &path) {
  return classify(decode_image(path));
}

}  // namespace npue::vit
