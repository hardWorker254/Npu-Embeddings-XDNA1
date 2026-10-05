//===- array.cpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the array side of a convolution. See array.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "conv/array.hpp"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "common/design_selection.hpp"
#include "common/host_kernels.hpp"
#include "runtime/design.hpp"
#include "runtime/device.hpp"
#include "whisper/npu_ops.hpp"

namespace npue::conv {
namespace {

std::string read_text(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Every stream of a design set, loaded into the slot design.json records, and
// CHECKED against it.
//
// This is arch=6's loader, transcribed rather than shared, because that one lives
// in an anonymous namespace inside pose/session.cpp and names pose in its error
// message. It is 20 lines and the duplication is deliberate: moving arch=6's
// measured path to a shared helper is its own change, and doing it in the same
// commit as arch=8's array path would put two unrelated things in one diff. The
// message below is therefore GENERIC, where arch=6's says "pose" -- a
// conv-only set has three possible owners and naming one of them from the shared
// copy would be a lie for the other two.
std::vector<app::StreamEntry> load_streams(npu::Design &d,
                                           const std::string &dir) {
  std::vector<app::StreamEntry> streams =
      app::parse_streams(read_text(dir + "/design.json"));
  if (streams.empty())
    throw std::runtime_error(
        dir + "/design.json lists no streams -- re-export it with "
        "tools/export/export_gemm_rtp.py --target <model> --arch 1 -n 32");
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

const app::StreamEntry &entry_for(const std::string &name,
                                  const std::vector<app::StreamEntry> &streams,
                                  const std::string &dir) {
  for (const auto &s : streams)
    if (s.op == name) return s;
  throw std::runtime_error(
      dir + " has no stream named '" + name +
      "'. The design set was exported for a different model, or the container "
      "names a stream its design does not carry -- and a convolution against a "
      "missing slot is a wrong answer rather than an error.");
}

// Everything the device side needs, in one place.
class Backend final : public Convs {
public:
  Backend(npue::File &model, const std::vector<ConvSlot> &slots,
          const std::string &name, const std::string &artifacts,
          app::Pool &pool)
      : name_(name),
        dev_(std::make_unique<npu::Device>()),
        design_(std::make_unique<npu::Design>(*dev_, artifacts + "/gemm_rtp")),
        pool_(pool) {
    // ONE PANEL PER CONVOLUTION, NEVER PER STREAM NAME. This is the rule that
    // matters most in this file and it is the one an earlier version got wrong:
    // panels were keyed by stream name, on the stated reasoning that two
    // convolutions sharing a padded (K, N) "must not share a panel" -- and the
    // lookup then returned the first one anyway. The stream names the SHAPE, and
    // arch=6's 72 convolutions collapse to 14 shapes, so 58 of them were
    // dispatched against another layer's weights. Nothing reports that: the
    // panel is the right size, the layout_hash matches, and the products are
    // arithmetic.
    //
    // There is nothing to deduplicate. Each slot number occurs exactly once in
    // `slots`, so a map could only ever have suppressed a second staging of the
    // SAME panel, which cannot happen.
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

    panels_.resize(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
      const ConvSlot &s = slots[i];
      if (s.slot < 0 || static_cast<size_t>(s.slot) >= panels_.size())
        throw std::runtime_error(
            name_ + ": " + s.group + " conv " + std::to_string(s.conv) +
            " claims slot " + std::to_string(s.slot) + " and the list has " +
            std::to_string(panels_.size()) +
            ". Slot numbers are assigned by the caller across every graph, so a "
            "number outside that range means the caller and this backend were "
            "built from different convolution lists.");
      if (!panels_[static_cast<size_t>(s.slot)].bias.empty())
        throw std::runtime_error(
            name_ + ": slot " + std::to_string(s.slot) + " is claimed by " +
            s.group + " conv " + std::to_string(s.conv) +
            " and by an earlier entry. Each convolution is staged once, and a "
            "second staging would overwrite the first panel.");
      if (s.stream.empty())
        throw std::runtime_error(
            name_ + ": " + s.group + " conv " + std::to_string(s.conv) +
            " has no \"stream\" name, so --npu-ops conv cannot run it. This "
            "container was packed without array panels (re-pack it with the "
            "packer's --npu), and the flag is explicit about moving work to the "
            "array, so it refuses rather than running this one layer on the host "
            "and reporting a split it did not ask for.");
      const std::string op = s.panel_name();
      if (!model.has(op))
        throw std::runtime_error(
            name_ + ": the container has no " + op +
            ", which the graph's stream '" + s.stream +
            "' needs. Re-pack with --npu so the pre-tiled panel is present.");
      const app::StreamEntry &e = entry_for(s.stream, streams,
                                             artifacts + "/gemm_rtp");
      Panel p;
      p.slot = gemm_->stage_operand(model, op);
      p.instr = static_cast<size_t>(e.slot);
      p.k = e.K;
      p.n = e.N;
      // The real widths come from the weight this panel belongs to, and the
      // padded ones from the design, and the two are cross-checked: a panel
      // staged for conv64x128 against a design stream saying K=64 is fine, one
      // saying K=128 is a design set exported for another model, and without the
      // check the dispatch would read past the panel and return plausible
      // products.
      const int64_t real_k = s.k();
      if (p.k < real_k || p.n < s.cout)
        throw std::runtime_error(
            name_ + ": " + s.group + " conv " + std::to_string(s.conv) +
            " needs " + std::to_string(real_k) + "x" + std::to_string(s.cout) +
            " but the design's stream '" + s.stream + "' is " +
            std::to_string(p.k) + "x" + std::to_string(p.n) +
            ". The design set was exported for a different model.");
      // NpuGemm::run reads bias[j] for every j < n, so handing it the
      // convolution's own bias would read past the end on any convolution whose
      // cout is not the padded width. The tail is zero, which is exact: those
      // columns are channels the tensor does not have, and the host narrows the
      // result back to the real N on the way out.
      p.bias.assign(static_cast<size_t>(p.n), 0.0f);
      if (s.bias)
        for (int64_t j = 0; j < s.cout; ++j)
          p.bias[static_cast<size_t>(j)] = s.bias[j];
      panels_[static_cast<size_t>(s.slot)] = std::move(p);
      staged_ += model.info(op).nbytes;
      // Distinct names only, in slot order: one convolution out of 99 sharing
      // another's stream would otherwise print the same name dozens of times.
      if (ops_.empty() || ops_.back() != s.stream) ops_.push_back(s.stream);
    }
    if (panels_.empty())
      throw std::runtime_error(
          name_ + ": --npu-ops conv was asked for and no convolution could be "
          "staged. A design set with no slots behind it is not a design set.");
  }

  ~Backend() override = default;

  // out[m, padded_n] = a[m, padded_k] @ b[padded_k, padded_n] + bias[padded_n].
  //
  // The widths are the PANEL's, not the caller's, and the caller's bias argument
  // is ignored -- both because this is the only place that knows what the
  // compiled core runs. A caller passing the right widths and one passing the
  // wrong ones would be indistinguishable here, so the check refuses rather than
  // silently correcting.
  int64_t gemm(int64_t conv, const float *a, int64_t m, int64_t k, int64_t n,
               const float *bias, float *out) override {
    (void)bias;
    const Panel &p = at(conv);
    if (k != p.k || n != p.n)
      throw std::runtime_error(
          name_ + ": convolution slot " + std::to_string(conv) +
          " was dispatched " + std::to_string(k) + "x" + std::to_string(n) +
          " but its design is " + std::to_string(p.k) + "x" + std::to_string(p.n) +
          ". The A rows handed over must be padded to the design's K and the "
          "result read back at the design's N.");
    gemm_->run(p.instr, a, m, rows_, p.k, p.slot, p.bias.data(), p.n, out);
    return m;
  }

  int64_t padded_k(int64_t conv) const override { return at(conv).k; }
  int64_t padded_n(int64_t conv) const override { return at(conv).n; }
  int64_t rows_per_dispatch() const override { return rows_; }
  int64_t dispatches() const override { return gemm_->n_dispatch; }
  double t_array() const override { return gemm_->t_dispatch + gemm_->t_convert; }
  size_t staged_bytes() const override { return staged_; }
  std::vector<std::string> stream_ops() const override { return ops_; }

private:
  struct Panel {
    size_t slot = 0;
    size_t instr = 0;
    int64_t k = 0;
    int64_t n = 0;
    std::vector<float> bias;
  };

  // (group, conv) -> slot is the CALLER's job, and this file deliberately does not
  // redo it: two spellings of one mapping is exactly the class of thing
  // ConvSlot::slot exists to avoid, and a wrong one is a panel of the right
  // shape holding another layer's weights.
  const Panel &at(int64_t conv) const {
    const size_t i = static_cast<size_t>(conv);
    if (conv < 0 || i >= panels_.size() || panels_[i].bias.empty())
      throw std::runtime_error(name_ + ": convolution slot " +
                               std::to_string(conv) + " was never staged");
    return panels_[i];
  }

  std::string name_;
  std::mutex mu_;
  std::unique_ptr<npu::Device> dev_;
  std::unique_ptr<npu::Design> design_;
  std::unique_ptr<npue::whisper::NpuGemm> gemm_;
  app::Pool &pool_;
  int64_t rows_ = 0;
  std::vector<Panel> panels_;
  std::vector<std::string> ops_;
  size_t staged_ = 0;
};

}  // namespace

std::unique_ptr<Convs> make_array_backend(
    npue::File &model, const std::vector<ConvSlot> &slots,
    const std::string &container_name, const std::string &artifacts_dir,
    app::Pool &pool) {
  return std::make_unique<Backend>(model, slots, container_name, artifacts_dir,
                                   pool);
}

}  // namespace npue::conv