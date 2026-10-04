//===- session.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one pose session: front end, network, decode. And, only when
// asked, an array backend for the convolutions.
//
// THE DEFAULT PATH OPENS NO DEVICE
// --------------------------------
// This is the first Session in the tree that can be constructed with no
// npu::Device, no Design and no artifacts directory at all. That is not an
// optimisation, it is the measured shape of the model: 150 ms on 16 CPU threads
// against 290 ms on the array for the same 72 dispatched convolutions (see
// net.hpp for the arithmetic behind both numbers). So `Session(...)` with no
// `artifacts` constructs a host-only session, and `npuembeddings pose` and
// `python/npue_pose.py` both work on a machine with no /dev/accel0 at all --
// which is also what makes the parity gate in tools/verify/verify_pose.py
// runnable anywhere.
//
// --npu-ops conv flips it. The array backend is then REQUIRED rather than
// optional, and if the design set lacks the streams this network needs, the
// refusal is by name: a flag that says "put this on the array" must not quietly
// leave it on the host, because the resulting run looks successful and its
// numbers are not the ones the flag asked for.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pose/decode.hpp"
#include "pose/geometry.hpp"
#include "pose/image.hpp"
#include "pose/net.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "vit/image.hpp"

namespace npue::pose {

// One image's answer, with the timings split so a client can see where the
// frame went. Same shape as vit::Prediction, and for the same reason: the CLI
// prints it to stderr and the server can return it.
struct Result {
  std::vector<Person> people;
  double front_end_s = 0.0;   // decode + letterbox + normalise
  double network_s = 0.0;
  double decode_s = 0.0;
  double total_s = 0.0;
  int64_t width = 0, height = 0;   // the SOURCE image, so a client can scale
  // The front end's transform, reported so a client can map its own
  // annotations (a drawn box, a cursor) through the same pipeline the model saw
  // rather than guessing at the letterbox.
  double scale = 1.0;
  int64_t pad_x = 0, pad_y = 0;
  int64_t dispatches = 0;
  int64_t convs_host = 0, convs_array = 0;
  // The host convolution's split, carried out of Cost so the CLI can print it.
  // Zero on the array path, where there is one GEMM per dispatch and the host
  // does no arithmetic at all.
  double t_wmat_s = 0.0, t_im2col_s = 0.0, t_gemm_s = 0.0, t_transpose_s = 0.0;
  // The non-convolution span, on BOTH backends: SiLU plus every non-conv node.
  double t_elementwise_s = 0.0;
  // The array path's wall time in the convolution loop, which INCLUDES the
  // host-side per-chunk A repack and the transpose of C into NCHW. It is
  // deliberately the same span as t_host on the other side -- both are "the
  // time conv() took", not "the time the device was busy" -- so the two are
  // comparable and neither can look better than it is by measuring a different
  // thing than its counterpart.
  double t_array_s = 0.0;
  double t_array_repack_s = 0.0, t_array_transpose_s = 0.0, t_array_gemm_s = 0.0;
  bool array = false;
};

class Session {
public:
  // `artifacts` may be empty: that is the host-only session and it is the
  // default. Non-empty REQUIRES that the design set carry the convolution
  // streams -- see the header note.
  Session(npue::File &model, const std::string &model_name,
          const std::string &artifacts, int threads,
          const DecodeParams &params = DecodeParams());

  // Declared, not defaulted, so it is DEFINED in session.cpp where
  // NpuConvBackend is a complete type. A defaulted destructor here would
  // instantiate unique_ptr's deleter against an incomplete type at every point
  // that destroys a Session -- which is the whole point of the forward
  // declaration, and it is why the build is told where the definition lives
  // rather than being trusted to find it.
  ~Session();

  const Geometry &geometry() const { return geom_; }
  // The Network itself, so a verification harness can install a per-node hook
  // (Network::on_node). The hook has to go on THIS object, not on one a caller
  // builds: it is a member of the network the runs below already use, and a
  // second Network would mean a second copy of the 72 convolution weights.
  Network &network() { return *net_; }
  const std::string &name() const { return name_; }
  const std::string &artifacts() const { return art_; }
  const DecodeParams &params() const { return params_; }
  void set_params(const DecodeParams &p) { params_ = p; }
  bool array() const { return array_; }
  // The stage names the loaded set carries, in slot order, for the status line.
  const std::vector<std::string> &stream_ops() const { return ops_; }
  int64_t rows_per_dispatch() const { return rows_; }

  Result detect(const npue::vit::Image &im);
  Result detect_file(const std::string &path);

  // The normalised letterbox only, for the host-side gates that want to check
  // the front end without running the 72 convolutions. Same function detect()
  // calls, so a gate that passes here has passed there.
  static std::vector<float> preprocess_file(const std::string &path,
                                            const Geometry &g, Letterbox &lb);

private:
  npue::File &model_;
  Geometry geom_;
  std::string art_, name_;
  DecodeParams params_;

  // Declaration order is construction order and it matters: the network holds
  // REFERENCES to a Pool and to the array backend, so both have to exist first.
  std::unique_ptr<app::Pool> pool_;
  std::unique_ptr<class NpuConvBackend> backend_;
  std::unique_ptr<Network> net_;
  std::vector<std::string> ops_;
  int64_t rows_ = 0;
  bool array_ = false;
};

// The default parameter values, as text, for --help and the refusal messages.
std::string decode_defaults_line();

}  // namespace npue::pose