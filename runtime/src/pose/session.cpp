//===- session.cpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one pose session. See session.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "pose/session.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "common/design_selection.hpp"
#include "common/host_kernels.hpp"
#include "runtime/design.hpp"
#include "runtime/device.hpp"
#include "whisper/npu_ops.hpp"

namespace npue::pose {

namespace {

std::string read_text(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Every stream of a design set, loaded into the slot design.json records, and
// CHECKED against it. A set loaded in the wrong order runs the wrong
// instruction stream for an op and returns a plausible number -- the same
// discipline vit::classify.cpp applies, copied rather than shared because the
// two sessions have genuinely different lifetime requirements (this one builds
// no device at all on the default path).
std::vector<app::StreamEntry> load_streams(npu::Design &d, const std::string &dir) {
  std::vector<app::StreamEntry> streams =
      app::parse_streams(read_text(dir + "/design.json"));
  if (streams.empty())
    throw std::runtime_error(dir +
                             "/design.json lists no streams -- re-export the "
                             "pose target with tools/export/export_gemm_rtp.py "
                             "--target pose");
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

}  // namespace

// -- the array backend ------------------------------------------------------------

// NpuConvBackend owns the device, the design and the staged weight panels, and
// presents the three methods Network needs. It exists as a class rather than a
// few free functions so that Network's default path can hold a null pointer to
// it and never name npu::Design at all -- which is what lets `pose` link and
// run with no XRT present.
class NpuConvBackend final : public NpuConvs {
public:
  NpuConvBackend(npue::File &model, const Geometry &g, const std::string &name,
                 const std::string &artifacts,
                 const std::vector<Layer> &graph, app::Pool &pool)
      : name_(name),
        dev_(std::make_unique<npu::Device>()),
        design_(std::make_unique<npu::Design>(*dev_, artifacts + "/gemm_rtp")),
        pool_(pool) {
    // panels_ is a SLOT PER CONVOLUTION, not per graph node, so it is sized from
    // Geometry::convs and never from `graph`. This was written as `(void)g;`
    // with the comment "the graph already carries every shape the panel lookup
    // needs", and that comment is what hid the crash: the graph carries the
    // SHAPES, but it does not carry the slots, and the slot index is
    // Geometry::convs' size. Indexing an unsized vector is undefined behaviour,
    // so the first convolution wrote to whatever followed an empty vector on the
    // heap -- a segfault on the array path and nothing at all on the CPU path,
    // which is why the only run that found it was the first run ever made on
    // hardware.
    panels_.resize(g.convs.size());

    const std::vector<app::StreamEntry> streams =
        load_streams(*design_, artifacts + "/gemm_rtp");
    rows_ = design_->info().M;
    if (rows_ <= 0)
      throw std::runtime_error(artifacts +
                               "/gemm_rtp: the design reports M = 0, so a "
                               "dispatch computes no rows");

    gemm_ = std::make_unique<npue::whisper::NpuGemm>(*design_, pool_);
    gemm_->npu_mu = &mu_;
    gemm_->alloc_buffers();

    // One B slot per DISTINCT stream, staged once from the container. The slot
    // is keyed by the layer's stream name, because two convolutions that share a
    // padded (K, N) have different weights and must not share a panel -- only
    // two convolutions with the SAME stream AND the same conv index may.
    for (size_t i = 0; i < graph.size(); ++i) {
      const Layer &L = graph[i];
      if (L.op != Op::Conv) continue;
      if (L.stream.empty())
        throw std::runtime_error(
            name_ + ": graph node " + std::to_string(i) +
            " (conv " + std::to_string(L.conv) +
            ") has no \"stream\" name, so --npu-ops conv cannot run it. "
            "This container was packed without array panels (pack with "
            "tools/pack/packers/pose.py --npu), and the flag is explicit about "
            "moving work to the array, so it refuses rather than running this "
            "one layer on the host and reporting a split it did not ask for.");
      const std::string op = "conv." + std::to_string(L.conv) + ".btile";
      // The two containers of convolution indices -- the graph's `conv` and
      // Geometry::convs' own positions -- are checked here rather than trusted,
      // because a disagreement is an out-of-bounds write and not a wrong number.
      // The `.btile` name is derived from the graph's index while the panel slot
      // is the vector's, so an off-by-one between them would stage one
      // convolution's weights under another's slot and return a plausible,
      // wrong, network.
      if (L.conv < 0 ||
          static_cast<size_t>(L.conv) >= panels_.size())
        throw std::runtime_error(
            name_ + ": graph node " + std::to_string(i) + " calls itself conv " +
            std::to_string(L.conv) + ", but the container holds " +
            std::to_string(panels_.size()) +
            " convolution weights. The graph and the weight table disagree about "
            "how many convolutions this network has, so there is no correct slot "
            "to stage this one into.");
      if (g.convs[static_cast<size_t>(L.conv)].index != L.conv)
        throw std::runtime_error(
            name_ + ": the container's conv " + std::to_string(L.conv) +
            " records itself at position " +
            std::to_string(g.convs[static_cast<size_t>(L.conv)].index) +
            ". Read that back as a slot and it would point at another "
            "convolution's weights.");
      if (!model.has(op))
        throw std::runtime_error(
            name_ + ": the container has no " + op +
            ", which the graph's stream '" + L.stream +
            "' needs. Re-pack with --npu so the pre-tiled panel is present.");
      // ONE PANEL PER CONVOLUTION, staged unconditionally. The map this replaced
      // was keyed by the stream NAME, on the stated rule that two convolutions
      // sharing a padded (K, N) "must not share a panel" -- and then reused the
      // first one's anyway. The stream names the SHAPE, and 72 convolutions here
      // collapse to 14 shapes, so 58 of them were dispatched against another
      // layer's weights. Nothing reports that: the panel is the right size, the
      // layout_hash matches, and the products are arithmetic.
      //
      // There is nothing to deduplicate. The loop runs once per graph node and a
      // convolution index belongs to exactly one node, so every panel is staged
      // exactly once; the map could only ever have suppressed a second staging of
      // the SAME convolution, which cannot happen.
      const app::StreamEntry &e = entry_for(L.stream, streams);
      Panel p;
      p.slot = gemm_->stage_operand(model, op);
      p.instr = static_cast<size_t>(e.slot);
      p.k = e.K;
      p.n = e.N;
      // The real widths come from the weight this panel belongs to, and the
      // padded ones from the design, and the two are cross-checked: a panel
      // staged for conv64x128 against a design stream that says K=64 is fine,
      // one that says K=128 is a design set exported for another model, and
      // without this check the dispatch would read past the panel and return
      // plausible products.
      const ConvW &cw = g.convs[static_cast<size_t>(L.conv)];
      const int64_t real_k = static_cast<int64_t>(cw.cin) * cw.kh * cw.kw;
      if (p.k < real_k || p.n < cw.cout)
        throw std::runtime_error(
            name_ + ": conv " + std::to_string(L.conv) + " needs " +
            std::to_string(real_k) + "x" + std::to_string(cw.cout) +
            " but the design's stream '" + L.stream + "' is " +
            std::to_string(p.k) + "x" + std::to_string(p.n) +
            ". The design set was exported for a different model.");
      p.bias.assign(static_cast<size_t>(p.n), 0.0f);
      if (cw.b)
        for (int64_t j = 0; j < cw.cout; ++j)
          p.bias[static_cast<size_t>(j)] = cw.b[j];
      panels_[static_cast<size_t>(L.conv)] = std::move(p);
      staged_ += model.info(op).nbytes;
      ops_.push_back(L.stream);
    }
  }

  ~NpuConvBackend() override = default;

  // out[m, padded_n] = a[m, padded_k] @ b[padded_k, padded_n] + bias[padded_n].
  // `conv` indexes Geometry::convs, which is what panels_ was keyed by.
  //
  // The widths are the PANEL's, not the `k` and `n` the caller passes, and the
  // caller's own bias argument is ignored. Both because this is the only place
  // that knows what the compiled core runs: NpuGemm::run packs `m * k` A
  // columns as one contiguous run and adds `bias[j]` for j < n, so a narrower k
  // makes row r's tail read row r+1's data and a narrower n reads past the
  // caller's bias. A caller that passed the right widths and a caller that
  // passed the wrong ones would then be indistinguishable here, which is why the
  // check below refuses rather than silently correcting.
  int64_t gemm(int64_t conv, const float *a, int64_t m, int64_t k, int64_t n,
               const float *bias, float *out) override {
    (void)bias;
    if (conv < 0 || static_cast<size_t>(conv) >= panels_.size())
      throw std::runtime_error(name_ + ": convolution " + std::to_string(conv) +
                               " was never staged");
    const Panel &p = panels_[static_cast<size_t>(conv)];
    if (k != p.k || n != p.n)
      throw std::runtime_error(
          name_ + ": convolution " + std::to_string(conv) + " was dispatched " +
          std::to_string(k) + "x" + std::to_string(n) + " but its design is " +
          std::to_string(p.k) + "x" + std::to_string(p.n) +
          ". The A rows handed over must be padded to the design's K and the "
          "result read back at the design's N.");
    gemm_->run(p.instr, a, m, rows_, p.k, p.slot, p.bias.data(), p.n, out);
    return m;
  }

  int64_t padded_k(int64_t conv) const override {
    return panel_at(conv).k;
  }
  int64_t padded_n(int64_t conv) const override {
    return panel_at(conv).n;
  }
  int64_t rows_per_dispatch() const override { return rows_; }
  int64_t dispatches() const override { return gemm_->n_dispatch; }
  double t_array() const override { return gemm_->t_dispatch + gemm_->t_convert; }
  size_t staged_bytes() const { return staged_; }
  // The stream names this backend found and staged, in graph order, for the
  // status line. Distinct names only -- one conv out of 73 sharing another's
  // panel would otherwise print the same stream 19 times.
  std::vector<std::string> stream_ops() const {
    std::vector<std::string> v;
    for (const auto &s : ops_)
      if (std::find(v.begin(), v.end(), s) == v.end()) v.push_back(s);
    return v;
  }

private:
  struct Panel {
    size_t slot = 0;
    size_t instr = 0;
    // The design's padded K and N for this convolution, and a bias widened to
    // padded_n. NpuGemm::run reads `bias[j]` for every j < n, so handing it the
    // convolution's own bias would read past its end on 128 of a 16-channel
    // convolution's bias. The tail is zero, which is exact: those columns are
    // channels the tensor does not have, and the host narrows the result back
    // to the real N on the way out.
    int64_t k = 0;
    int64_t n = 0;
    std::vector<float> bias;
  };

  const app::StreamEntry &entry_for(const std::string &name,
                                    const std::vector<app::StreamEntry> &streams) {
    for (const auto &s : streams)
      if (s.op == name) return s;
    throw std::runtime_error(
        "the design set has no stream named '" + name +
        "'. A pose design set carries one stream per distinct padded (K, N) "
        "the network needs, and the graph names the one each convolution uses; "
        "re-export it for this model, or do not pass --npu-ops conv.");
  }

  const Panel &panel_at(int64_t conv) const {
    if (conv < 0 || static_cast<size_t>(conv) >= panels_.size())
      throw std::runtime_error(name_ + ": convolution " + std::to_string(conv) +
                               " was never staged");
    return panels_[static_cast<size_t>(conv)];
  }

  std::string name_;
  std::unique_ptr<npu::Device> dev_;
  std::unique_ptr<npu::Design> design_;
  app::Pool &pool_;
  std::unique_ptr<npue::whisper::NpuGemm> gemm_;
  std::mutex mu_;
  std::vector<Panel> panels_;
  std::vector<std::string> ops_;
  int64_t rows_ = 0;
  size_t staged_ = 0;
};

// -- the session ------------------------------------------------------------------

Session::Session(npue::File &model, const std::string &model_name,
                 const std::string &artifacts, int threads,
                 const DecodeParams &params)
    : model_(model),
      geom_(read_geometry(model, model_name)),
      art_(artifacts),
      name_(model_name),
      params_(params),
      pool_(std::make_unique<app::Pool>(std::max(1, threads))) {
  Placement place;
  if (!art_.empty()) {
    // A non-empty artifacts string means the caller asked for the array. That
    // request is honoured or refused; there is no fallback to the host, because
    // a run that reported array timings while executing on the CPU would be
    // worse than no run at all.
    if (!std::filesystem::exists(art_ + "/gemm_rtp/design.json"))
      throw std::runtime_error(
          art_ + " has no gemm_rtp/design.json. A pose session on the array "
                 "needs one design set -- the same kind bge-base's is, plus the "
                 "padded (K, N) streams this network's convolutions require. "
                 "Use tools/export/export_gemm_rtp.py --target pose, or drop "
                 "the flag to run the convolutions on the CPU, which is the "
                 "default and the measured faster choice here (see "
                 "runtime/include/pose/net.hpp).");
    backend_ = std::make_unique<NpuConvBackend>(model, geom_, name_, art_,
                                              geom_.graph, *pool_);
    place.conv_on_array = true;
    array_ = true;
    ops_ = backend_->stream_ops();
    rows_ = backend_->rows_per_dispatch();
  }
  net_ = std::make_unique<Network>(geom_, place, *pool_, backend_.get());
}

Session::~Session() = default;

std::vector<float> Session::preprocess_file(const std::string &path,
                                            const Geometry &g, Letterbox &lb) {
  return npue::pose::preprocess_file(path, g, lb);
}

Result Session::detect(const npue::vit::Image &im) {
  Result r;
  const double t_all = app::now_s();

  const double t0 = app::now_s();
  Letterbox lb;
  std::vector<float> px = letterbox_normalise(im, geom_, lb);
  r.width = lb.src_w;
  r.height = lb.src_h;
  r.scale = lb.scale;
  r.pad_x = lb.pad_x;
  r.pad_y = lb.pad_y;
  r.front_end_s = app::now_s() - t0;

  Tensor in(3, geom_.input_size, geom_.input_size);
  std::copy(px.begin(), px.end(), in.d.begin());

  const double t1 = app::now_s();
  Tensor head = net_->run(in);
  r.network_s = app::now_s() - t1;

  const double t2 = app::now_s();
  r.people = decode(head, geom_, lb, params_);
  r.decode_s = app::now_s() - t2;

  const Cost &c = net_->cost();
  r.dispatches = c.dispatches;
  r.convs_host = c.convs_host;
  r.convs_array = c.convs_array;
  r.t_wmat_s = c.t_wmat;
  r.t_im2col_s = c.t_im2col;
  r.t_gemm_s = c.t_gemm;
  r.t_transpose_s = c.t_transpose;
  r.t_array_s = c.t_array;
  r.t_array_repack_s = c.t_array_repack;
  r.t_array_transpose_s = c.t_array_transpose;
  r.array = array_;
  r.total_s = app::now_s() - t_all;
  return r;
}

Result Session::detect_file(const std::string &path) {
  return detect(npue::vit::decode_image(path));
}

std::string decode_defaults_line() {
  const DecodeParams p;
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "conf %.2f, iou %.2f, keypoint %.2f, max_det %lld",
                p.conf, p.iou, p.keypoint, static_cast<long long>(p.max_det));
  return buf;
}

}  // namespace npue::pose