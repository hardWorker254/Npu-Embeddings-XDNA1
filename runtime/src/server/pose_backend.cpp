//===- pose_backend.cpp -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the image-in, pose-out HTTP endpoint.
// See server/pose_backend.hpp for the refusals and why each one is there.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "server/pose_backend.hpp"

#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "pose/result_json.hpp"
#include "server/multipart.hpp"
#include "server/server.hpp"
#include "vit/image.hpp"

namespace {

// File-local signal handler: sets only the flag, nothing that is not
// async-signal-safe. The accept loop notices it within one select() timeout and
// unwinds, so the pool and the array backend release their hw_contexts.
void on_stop_signal(int) { npue::http::g_server_stop = 1; }

// strtod, with the two checks that matter: the WHOLE field has to be a number,
// and there must be no trailing junk. std::stod throws three different
// exception types for three different mistakes, and strtod's silent
// "0.25abc" -> 0.25 would answer a request with a threshold nobody sent.
bool parse_double(const std::string &s, double &out) {
  if (s.empty()) return false;
  errno = 0;
  char *end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (errno == ERANGE || end != s.c_str() + s.size()) return false;
  if (!std::isfinite(v)) return false;
  out = v;
  return true;
}

bool parse_int(const std::string &s, int64_t &out) {
  if (s.empty()) return false;
  errno = 0;
  char *end = nullptr;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  if (errno == ERANGE || end != s.c_str() + s.size()) return false;
  out = static_cast<int64_t>(v);
  return true;
}

std::string field(const std::vector<npue::http::MultipartPart> &parts,
                  const char *name) {
  for (const auto &p : parts)
    if (p.name == name) return p.data;
  return std::string();
}

// A double, rendered without a trailing ".0" on whole numbers -- "conf=1" reads
// better than "conf=1.000000" in an echo of what the client sent.
std::string num(double v) {
  char buf[64];
  if (v == static_cast<double>(static_cast<int64_t>(v)))
    std::snprintf(buf, sizeof buf, "%lld",
                  static_cast<long long>(static_cast<int64_t>(v)));
  else
    std::snprintf(buf, sizeof buf, "%.6g", v);
  return std::string(buf);
}

}  // namespace

namespace app {

int serve_pose(npue::pose::Session &session, const std::string &model_id,
               int port, const std::string &bind_addr,
               const npue::pose::DecodeParams &defaults) {
  std::signal(SIGINT, on_stop_signal);
  std::signal(SIGTERM, on_stop_signal);
  const auto &g = session.geometry();
  std::printf("\n  serving http://%s:%d/v1/pose   (model %s)\n", bind_addr.c_str(),
              port, model_id.c_str());
  std::printf("  POST multipart/form-data: image (required, PNG or JPEG), conf, "
              "iou, kpt, max_det\n");
  std::printf("  %lldx%lld input, %lld people max, %s backend\n",
              static_cast<long long>(g.input_size),
              static_cast<long long>(g.input_size),
              static_cast<long long>(defaults.max_det),
              session.array() ? "array" : "host");
  // Flushed, not left to the buffer. Every status line above this one goes to
  // stderr and is unbuffered, so this banner is the LAST thing a redirected log
  // receives; without the flush it sits in a block buffer until 4 KiB of output
  // accumulates, which for a server that then prints nothing means the log never
  // shows the port at all. An operator tailing that log would conclude the
  // process had not started. All four endpoints do this.
  std::fflush(stdout);

  npue::http::Server server(static_cast<uint16_t>(port), bind_addr);
  server.run([&](const npue::http::Request &req, int &status, std::string &ctype,
                 std::string &body) {
    auto fail = [&](int code, const char *type, const std::string &msg) {
      status = code;
      body = "{\"error\":{\"message\":\"" + npue::http::json_escape(msg) +
             "\",\"type\":\"" + type + "\"}}";
    };

    if (req.method == "GET" && (req.path == "/health" || req.path == "/")) {
      body = "{\"status\":\"ok\",\"model\":\"" + model_id +
             "\",\"backend\":\"amd-xdna2-npu\",\"kind\":\"pose\",\"engine\":\"" +
             (session.array() ? "array" : "host") +
             "\",\"input_size\":" + std::to_string(g.input_size) +
             ",\"conf\":" + num(defaults.conf) +
             ",\"iou\":" + num(defaults.iou) +
             ",\"keypoint\":" + num(defaults.keypoint) +
             ",\"max_det\":" + std::to_string(defaults.max_det) +
             ",\"skeleton\":\"" +
             npue::http::json_escape(npue::pose::skeleton_names_csv()) + "\"}";
      return;
    }
    if (req.method == "GET" && req.path == "/v1/models") {
      body = "{\"object\":\"list\",\"data\":[{\"id\":\"" + model_id +
             "\",\"object\":\"model\",\"owned_by\":\"npuembeddings\"}]}";
      return;
    }
    if (req.path != "/v1/pose") {
      fail(404, "not_found",
           "unknown path " + req.path +
               " -- this model serves /v1/pose, not /v1/embeddings. "
               "`npuembeddings serve <model>` picks the endpoint from the "
               "container's arch: text models answer /v1/embeddings, Whisper "
               "/v1/audio/transcriptions, a ViT /v1/classify and YOLOv8-pose "
               "/v1/pose.");
      return;
    }
    if (req.method != "POST") {
      fail(400, "invalid_request_error", "use POST for /v1/pose");
      return;
    }

    const std::string boundary =
        npue::http::multipart_boundary(req.content_type);
    if (boundary.empty()) {
      fail(400, "invalid_request_error",
           "expected Content-Type: multipart/form-data with a boundary; got '" +
               req.content_type +
               "'. This endpoint takes the image as a form part -- not as raw "
               "bytes, and not as JSON.");
      return;
    }
    std::vector<npue::http::MultipartPart> parts;
    std::string err;
    if (!npue::http::parse_multipart(req.body, boundary, parts, err)) {
      fail(400, "invalid_request_error", "multipart: " + err);
      return;
    }

    // -- the image -----------------------------------------------------------
    const npue::http::MultipartPart *image = nullptr;
    int n_image = 0;
    for (const auto &p : parts)
      if (p.name == "image") { ++n_image; image = &p; }
    if (n_image > 1) {
      fail(400, "invalid_request_error",
           "the request carries " + std::to_string(n_image) +
               " 'image' parts; which one is the picture is not a question to "
               "answer by taking the first");
      return;
    }
    if (!image || image->data.empty()) {
      fail(400, "invalid_request_error",
           "'image' is required, and the one that arrived was empty");
      return;
    }
    if (image->data.size() > 64u * 1024u * 1024u) {
      fail(413, "invalid_request_error",
           "the upload is " + std::to_string(image->data.size() / (1u << 20)) +
               " MB; this build caps a request at 64 MB");
      return;
    }

    // -- thresholds, per request and only for this request -------------------
    // Every one of these refuses rather than falling back to the default: a
    // request answered at thresholds the client did not ask for looks exactly
    // like a request answered correctly at the client's thresholds, and the
    // detection count is the product of these numbers.
    npue::pose::DecodeParams opts = defaults;
    struct NumField {
      const char *name;
      double *dst;
      double lo, hi;
    } doubles[] = {{"conf", &opts.conf, 0.0, 1.0},
                  {"iou", &opts.iou, 0.0, 1.0},
                  {"kpt", &opts.keypoint, 0.0, 1.0}};
    for (const auto &f : doubles) {
      const std::string s = field(parts, f.name);
      if (s.empty()) continue;
      double v = 0.0;
      if (!parse_double(s, v) || v < f.lo || v > f.hi) {
        fail(400, "invalid_request_error",
             std::string(f.name) + "='" + s +
                 "' is not a number in [" + num(f.lo) + ", " + num(f.hi) +
                 "]. This build would otherwise answer at its own default, and "
                 "the person count you get back is the product of this "
                 "number.");
        return;
      }
      *f.dst = v;
    }
    const std::string max_det = field(parts, "max_det");
    if (!max_det.empty()) {
      int64_t v = 0;
      if (!parse_int(max_det, v) || v < 1 || v > 10000) {
        fail(400, "invalid_request_error",
             "max_det='" + max_det +
                 "' is not an integer in [1, 10000]");
        return;
      }
      opts.max_det = v;
      // max_nms is the pre-NMS candidate cap and is derived from max_det so a
      // client cannot ask for 300 detections out of a 30000-candidate pool
      // without also getting the pool. Lowering it below max_det would make
      // max_det unreachable in a way that depends on the picture.
      if (opts.max_nms < opts.max_det) opts.max_nms = opts.max_det;
    }

    // -- decode straight from the bytes --------------------------------------
    // No temporary file. The decoder dispatches on magic, not on the extension,
    // so the client's filename is used for the `image` label and for error
    // messages and for nothing else.
    const std::string label =
        image->filename.empty() ? std::string("upload") : image->filename;
    try {
      const npue::vit::Image im =
          npue::vit::decode_image_bytes(std::vector<uint8_t>(
                                            image->data.begin(),
                                            image->data.end()),
                                        label);
      // set_params for the duration of the call only, and restored below: a
      // request's --conf must not become the next request's default. detect()
      // reads session.params(), and there is no per-call override on it.
      const npue::pose::DecodeParams saved = session.params();
      session.set_params(opts);
      npue::pose::Result r;
      try {
        r = session.detect(im);
      } catch (...) {
        session.set_params(saved);
        throw;
      }
      session.set_params(saved);

      npue::pose::JsonContext ctx;
      ctx.geom = &g;
      ctx.params = &opts;
      ctx.image_label = label;
      ctx.array = session.array();
      ctype = "application/json";
      body = npue::pose::result_json(r, ctx);
    } catch (const std::exception &e) {
      // 400 for what the CALLER sent (a format the front end refuses) and 500
      // only for what is ours. A wrong request is not a server error, and a
      // client that sees 500 stops trying.
      const std::string what = e.what();
      const bool callers_fault =
          what.find("PNG") != std::string::npos ||
          what.find("JPEG") != std::string::npos ||
          what.find("magic") != std::string::npos ||
          what.find("truncated") != std::string::npos;
      fail(callers_fault ? 400 : 500,
           callers_fault ? "invalid_request_error" : "internal_error", what);
    }
  });
  return 0;
}

}  // namespace app
