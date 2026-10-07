//===- classify.cpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one image-classification session. See classify.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "vit/classify.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/design_selection.hpp"
#include "common/host_kernels.hpp"
#include "vit/image.hpp"

// The two elementwise designs are Whisper's classes, not ViT's -- the kernels
// and their dispatch are one implementation, shared on purpose. Two aliases
// rather than 2*uses of the full name below, so the reader can see at the top
// that these are borrowed and not local.
using npue::whisper::EltwiseKind;
using npue::whisper::NpuEltwise;

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
                                           const npu::DesignSource &src) {
  std::vector<app::StreamEntry> streams =
      app::parse_streams(src.text("design.json"));
  if (streams.empty())
    throw std::runtime_error(src.label() +
                             " has no streams in design.json -- re-export "
                             "with "
                             "tools/export/export_gemm_rtp.py");
  std::sort(streams.begin(), streams.end(),
            [](const app::StreamEntry &a, const app::StreamEntry &b) {
              return a.slot < b.slot;
            });
  for (const auto &s : streams) {
    const size_t got = d.load_instr(src, s.file);
    if (static_cast<int64_t>(got) != s.slot)
      throw std::runtime_error("stream " + s.file + " landed in slot " +
                               std::to_string(got) + ", design.json says " +
                               std::to_string(s.slot));
  }
  return streams;
}

std::vector<app::StreamEntry> load_streams(npu::Design &d,
                                           const std::string &dir) {
  // The directory form, kept so the tests and any caller holding a plain path
  // keep working. Everything that HAS a container goes through the Source form
  // below, and the two must agree: a design whose streams came out of the
  // container while its stream table was read off disk is a stream list paired
  // with a core that does not have those streams, and that surfaces as a
  // dispatch mismatch several layers down rather than as "these two do not
  // belong together".
  return load_streams(d, npu::DesignSource::from_dir(dir));
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
                 const std::string &artifacts, int threads,
                 const std::set<std::string> &npu_ops)
    : model_(model),
      geom_(read_geometry(model, model_name)),
      art_(artifacts),
      name_(model_name),
      dev_(std::make_unique<npu::Device>()),
      pool_(std::make_unique<app::Pool>(std::max(1, threads))),
      design_(std::make_unique<npu::Design>(
          *dev_, npu::prefer_embedded(&model, artifacts, "gemm_rtp"))),
      enc_(model, *design_, *pool_, geom_) {
  const std::vector<app::StreamEntry> streams =
      load_streams(*design_,
                npu::prefer_embedded(&model, artifacts, "gemm_rtp"));
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
    // I4 counts as an int8 container, because that is what it is at run time:
    // gemm_b_panel widens the nibbles to an int8 panel before stage(), the
    // scales alongside them are int8's, and the design is the int8 one. There
    // is no third pairing to name.
    if ((a == "I8" || a == "I4") != i8_design)
      throw std::runtime_error(
          "container/design pairing: " + name_ + "'s GEMM operands are " + a +
          " and " + artifacts + "/gemm_rtp is a " +
          (i8_design ? "int8" : "bf16") +
          " design. These are two different containers and two different "
          "layout hashes; a design set is paired with one of them by name. "
          "Re-export with --int8 for an int8 container, or repack without "
          "--int8, or pass --artifacts for the matching set.");
  }

  // -- the optional elementwise designs, opened BEFORE stage_all() so the
  // encoder can stage its 25 gamma|beta pairs as part of the same pass.
  //
  // Same shape as Whisper's open_elt(), for the same reason: each design
  // carries the MODEL's own width and epsilon compiled into its kernel, so the
  // checks belong where the design is opened and nowhere later. A layernorm
  // design exported for a different model would normalise the wrong number of
  // channels and produce silently wrong logits -- no shape mismatch, no
  // exception, just a different answer.
  auto open_elt = [&](const std::string &code, const std::string &dir,
                      EltwiseKind kind, std::unique_ptr<npu::Design> &design,
                      std::unique_ptr<NpuEltwise> &op) {
    if (!npu_ops.count(code)) return;
    design = std::make_unique<npu::Design>(
        *dev_, npu::prefer_embedded(&model, artifacts, dir));
    op = std::make_unique<NpuEltwise>(*design, *pool_, kind);
    op->alloc_buffers();
    if (kind == EltwiseKind::LayerNorm && op->cols() != geom_.d_model)
      throw std::runtime_error(
          dir + "/design.json has rows " + std::to_string(op->cols()) +
          " columns wide, and this container's d_model is " +
          std::to_string(geom_.d_model) +
          ". The kernel's row width is compiled in, so this design normalises "
          "the wrong number of channels. Re-export it for this model: "
          "python tools/export/export_gemm_rtp.py --target " + name_ +
          " --arch 1 --npu-ops " + code);
    if (kind == EltwiseKind::LayerNorm) {
      // ViT's layer_norm_eps is 1e-12, an order of magnitude below the 1e-5 the
      // Whisper designs are built with, so this check is the one that fires
      // when somebody points --npu-ops layn at a design set that a
      // previous export happened to leave next to this one.
      const double want = geom_.ln_eps;
      const double got = design->info().ln_eps;
      if (got <= 0.0 || std::abs(got - want) > 1e-12 * std::max(1.0, want))
        throw std::runtime_error(
            dir + "/design.json was built with layer_norm_eps " +
            app::eps_text(got) + " and this container says " +
            app::eps_text(want) +
            ". The epsilon is inside a square root, so the two are not a "
            "rounding difference. Re-export the design for this container.");
      elt_notes_.push_back(dir + ": " + std::to_string(op->rows()) + " rows x " +
                           std::to_string(op->cols()) + ", eps " +
                           app::eps_text(got));
    } else {
      // One flat span of activations, so the pair that reads as "1 rows x N" is
      // spelled as what it is: the elements one dispatch covers.
      elt_notes_.push_back(dir + ": " +
                           std::to_string(op->rows() * op->cols()) +
                           " elements per dispatch");
    }
  };
  open_elt("layn", "layernorm", EltwiseKind::LayerNorm, ln_design_, ln_);
  open_elt("gelu", "gelu", EltwiseKind::Gelu, gelu_design_, gelu_);
  // The softmax, which has a sibling design of its own like the other two.
  //
  // It is a SEPARATE xclbin from gemm_rtp -- its own directory, its own
  // hw_context, which is why the budget check in vit_mode.hpp counts it -- and
  // it needs no GEMM to be on the array to be dispatched. That is what makes
  // `softm` a real per-op choice here instead of a code that only means
  // something as a side effect of `attn`: this architecture's softmax is inside
  // its attention, so `attention()` takes the operator and steps aside between
  // its two GEMMs -- QK^T on the host, the score row to the array, softmax.V
  // back on the host. What does not move is the pair of GEMMs; moving them is
  // the `attn` code, and a `softm` alone must not quietly ship them.
  //
  // The two flags compose: with `attn` too, NpuAttention runs all of it and
  // takes this same operator through set_softmax() below.
  open_elt("softm", "softmax", EltwiseKind::Softmax, sm_design_, sm_);
  if (sm_)
    ops_.push_back("softm at row width " + std::to_string(sm_->cols()) + ", " +
                   std::to_string(sm_->rows()) + " rows per dispatch (one "
                   "hw_context of its own)");
  // -- attention as two GEMMs, on gemm_rtp's OWN attn_qk/attn_av slots.
  //
  // No separate design and no extra hw_context: the streams are inside the set
  // design_ already loaded, which is why this costs a slot and nothing else.
  // A set without them was exported before the registry honoured `attn` for
  // kinds.cls, and asking for the array on it is refused by name rather than
  // answered from the host.
  //
  // n_kv comes from the CONTAINER's max_seq_len, not from geom_.n_pos: this
  // container preslices to its own window, and building a panel wider than a
  // tensor that cannot address past it is the failure resolve.py already
  // documents. geom_.n_pos is what the runtime walks; the padded count is what
  // the design was built at, and NpuAttention takes the former for the real
  // key count and the latter from the stream.
  if (npu_ops.count("attn")) {
    const app::StreamEntry &qk = find_op(streams, "attn_qk", tiers[0]);
    const app::StreamEntry &av = find_op(streams, "attn_av", tiers[0]);
    const int64_t hd = geom_.head_dim;
    if (qk.N != av.K)
      throw std::runtime_error(artifacts + "/gemm_rtp: attn_qk's N is " +
                               std::to_string(qk.N) + " and attn_av's K is " +
                               std::to_string(av.K) +
                               ". The score chunk travels from one to the other "
                               "as the A operand, so the two are the same padded "
                               "n_kv.");
    if (qk.K < hd || qk.K % hd)
      throw std::runtime_error(
          artifacts + "/gemm_rtp: attn_qk's K is " + std::to_string(qk.K) +
          " and this container's head_dim is " + std::to_string(hd) +
          ". The Q operand of a score is one head, so K is the head width padded "
          "UP to the design's tile_k -- never down to a head, and never a value a "
          "head does not divide.");
    if (av.N < hd)
      throw std::runtime_error(artifacts + "/gemm_rtp: attn_av's N is " +
                               std::to_string(av.N) + " and a head is " +
                               std::to_string(hd) +
                               " wide. The design pads this one UP to its own N "
                               "granularity, never down to a head.");
    if (sm_ && sm_->cols() != qk.N)
      throw std::runtime_error(
          artifacts + "/softmax has rows " + std::to_string(sm_->cols()) +
          " wide and the attn streams' score row is " + std::to_string(qk.N) +
          ". The softmax design reduces along the whole row, so it has to be the "
          "width of the score row it is handed.");
    attn_ = std::make_unique<npue::whisper::NpuAttention>(*design_, *pool_,
                                                         qk.N, hd, av.N);
    attn_->set_streams(static_cast<size_t>(qk.slot),
                       static_cast<size_t>(av.slot), qk.M, qk.K);
    attn_->set_softmax(sm_.get());
    // A ViT's encoder attends over every position it was given and there is no
    // padding to mask -- geom_.n_pos is the whole image, CLS included. So no
    // additive mask here, which is why set_additive_mask is never called: null
    // is the class's own default, not an oversight.
    attn_->alloc_buffers();
    ops_.push_back("attn  at slots " + std::to_string(qk.slot) + "/" +
                   std::to_string(av.slot) + ", M " + std::to_string(qk.M) +
                   ", K " + std::to_string(qk.K) + ", score row " +
                   std::to_string(qk.N) + ", context " + std::to_string(av.N));
  }
  enc_.set_layernorm(ln_.get());
  enc_.set_gelu(gelu_.get());
  enc_.set_attention(attn_.get());
  enc_.set_softmax(sm_.get());   // read only when attention is on the host

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
  // GEMMs plus the two attention GEMMs when `attn` asked for them. One number
  // for both because they are the same kind of work on the same xclbin -- the
  // field is not "the four layer GEMMs", it is how many times the array was
  // asked to multiply, and an `attn` run that reported 49 while dispatching
  // 49+2*12*n_pos would be reporting a number no reader can reconcile with the
  // time beside it.
  r.n_dispatch = enc_.gemm().n_dispatch + enc_.attn_dispatch();
  r.n_elt_dispatch = enc_.elt_dispatch();

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

std::string prediction_json(const Prediction &p, const std::string &label_name,
                            const std::string &image_label, int64_t top_k) {
  // A label name comes out of the container's own labels.table and is therefore
  // NOT trusted to be JSON-safe: a fine-tuned classifier's classes are whatever
  // the trainer typed, and a name with a quote in it would otherwise produce a
  // document no parser accepts. Escaped here rather than at the call site, once.
  auto esc = [](const std::string &s) {
    std::string o;
    for (char c : s) {
      if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); continue; }
      if (c == '\n') { o += "\\n"; continue; }
      if (c == '\r') { o += "\\r"; continue; }
      if (c == '\t') { o += "\\t"; continue; }
      if (static_cast<unsigned char>(c) < 0x20) { o += ' '; continue; }
      o.push_back(c);
    }
    return o;
  };
  // %.9g rather than std::to_string: the old emitter printed a float through
  // std::to_string, which is six DECIMAL PLACES, so a top-1 of 0.9999997 was
  // printed as "1.000000" and read as a certainty the model did not have. Nine
  // significant digits is still far more than a softmax carries and is what the
  // pose emitter uses.
  char num[64];
  auto pnum = [&](double v) {
    std::snprintf(num, sizeof num, "%.9g", v);
    return std::string(num);
  };

  std::string o = "{\"image\": \"" + esc(image_label) + "\", \"label\": " +
                  std::to_string(p.label) + ", \"name\": \"" + esc(label_name) +
                  "\", \"p\": " + pnum(p.top1);
  if (top_k > 1 && !p.logits.empty()) {
    // The RUNNERS-UP, by the same argmax the label came from, and out of the
    // SAME softmax -- so their probabilities sum with `p` to 1 rather than being
    // three independently normalised numbers.
    const float mx = *std::max_element(p.logits.begin(), p.logits.end());
    double sum = 0.0;
    for (float v : p.logits) sum += std::exp(static_cast<double>(v) - mx);
    std::vector<size_t> order(p.logits.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    const size_t want = std::min<size_t>(
        static_cast<size_t>(top_k), order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<long>(want),
                      order.end(),
                      [&](size_t a, size_t b) { return p.logits[a] > p.logits[b]; });
    o += ", \"top_k\": [";
    for (size_t i = 0; i < want; ++i) {
      o += (i ? ", " : "");
      o += "{\"label\": " + std::to_string(order[i]) + ", \"p\": " +
           pnum(sum > 0.0 ? std::exp(static_cast<double>(
                                     p.logits[order[i]]) - mx) / sum
                          : 0.0) +
           "}";
    }
    o += "]";
  }
  // Two dispatch counts, because they are two different questions. `dispatches`
  // is the layer GEMMs plus, when `attn` is on, attn_qk and attn_av -- 49 on a
  // host run and 49 + 2*12*n_pos on an array one, because both are GEMMs on the
  // same xclbin. `elt_dispatches` is the sibling elementwise designs, and it is
  // 0 on a run that asked for none -- which is what makes the field worth
  // carrying: a `--npu-ops softm` run that reported 0 there would be reporting
  // that the flag moved nothing, which is exactly the claim it must not make.
  o += ", \"front_end_s\": " + pnum(p.front_end_s) +
       ", \"encoder_s\": " + pnum(p.encoder_s) +
       ", \"head_s\": " + pnum(p.head_s) +
       ", \"dispatches\": " + std::to_string(p.n_dispatch) +
       ", \"elt_dispatches\": " + std::to_string(p.n_elt_dispatch) + "}";
  return o;
}

}  // namespace npue::vit
