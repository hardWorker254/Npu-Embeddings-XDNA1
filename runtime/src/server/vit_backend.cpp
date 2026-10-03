//===- vit_backend.cpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the image-in, label-out HTTP endpoint.
// See server/vit_backend.hpp for the refusals and why each one is there.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "server/vit_backend.hpp"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "server/multipart.hpp"
#include "server/server.hpp"
#include "vit/image.hpp"

namespace {

// File-local signal handler: sets only the flag, nothing that is not
// async-signal-safe. The accept loop notices it within one select() timeout and
// unwinds, so the device and the design release their hw_contexts.
void on_stop_signal(int) { npue::http::g_server_stop = 1; }

// strtoll with the two checks that matter: the WHOLE field has to be a number and
// there must be no trailing junk. strtoll's silent "5abc" -> 5 would answer a
// request with a count the client did not send.
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

}  // namespace

namespace app {

int serve_vit(npue::vit::Session &session, const std::string &model_id,
              int port, const std::string &bind_addr, int64_t default_top_k) {
  std::signal(SIGINT, on_stop_signal);
  std::signal(SIGTERM, on_stop_signal);
  const auto &g = session.geometry();
  const int64_t n_labels = static_cast<int64_t>(session.labels().size());
  std::printf("\n  serving http://%s:%d/v1/classify   (model %s)\n",
              bind_addr.c_str(), port, model_id.c_str());
  std::printf("  POST multipart/form-data: image (required, PNG or JPEG), top_k "
              "(default %lld)\n",
              static_cast<long long>(default_top_k));
  std::printf("  %lld labels, %lldpx input\n",
              static_cast<long long>(n_labels),
              static_cast<long long>(g.image_size));
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
             "\",\"backend\":\"amd-xdna2-npu\",\"kind\":\"classify\",\"int8\":" +
             (session.int8() ? "true" : "false") +
             ",\"input_size\":" + std::to_string(g.image_size) +
             ",\"patch_size\":" + std::to_string(g.patch_size) +
             ",\"labels\":" + std::to_string(n_labels) +
             ",\"top_k\":" + std::to_string(default_top_k) +
             // Named `dispatches_so_far`, not `dispatches_per_image`: it is the
             // encoder's running counter, so it is 0 on a freshly started server
             // and one image's worth afterwards. Calling it "per image" would
             // report 0 for a model that in fact costs 49, and a client reading
             // that number to size a fleet would size it wrong.
             ",\"dispatches_so_far\":" +
             std::to_string(session.n_dispatch()) + "}";
      return;
    }
    if (req.method == "GET" && req.path == "/v1/models") {
      body = "{\"object\":\"list\",\"data\":[{\"id\":\"" + model_id +
             "\",\"object\":\"model\",\"owned_by\":\"npuembeddings\"}]}";
      return;
    }
    // The label vocabulary, so a client can turn an id into a name without
    // shipping the container's labels.table itself. Every label, not a page of
    // them: a partial list would make an id silently unnameable.
    if (req.method == "GET" && req.path == "/v1/labels") {
      body = "{\"object\":\"list\",\"data\":[";
      for (int64_t i = 0; i < n_labels; ++i)
        body += (i ? ", " : "") +
                ("{\"id\": " + std::to_string(i) + ", \"name\": \"" +
                 npue::http::json_escape(session.labels()[static_cast<size_t>(i)]) +
                 "\"}");
      body += "]}";
      return;
    }
    if (req.path != "/v1/classify") {
      fail(404, "not_found",
           "unknown path " + req.path +
               " -- this model serves /v1/classify, not /v1/embeddings. "
               "`npuembeddings serve <model>` picks the endpoint from the "
               "container's arch: text models answer /v1/embeddings, Whisper "
               "/v1/audio/transcriptions, a ViT /v1/classify and YOLOv8-pose "
               "/v1/pose.");
      return;
    }
    if (req.method != "POST") {
      fail(400, "invalid_request_error", "use POST for /v1/classify");
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

    int64_t top_k = default_top_k;
    const std::string want = field(parts, "top_k");
    if (!want.empty()) {
      if (!parse_int(want, top_k) || top_k < 1) {
        fail(400, "invalid_request_error",
             "top_k='" + want +
                 "' is not an integer of at least 1. This build would otherwise "
                 "answer with its own default, and a client asking for the "
                 "runners-up would not know it did not get them.");
        return;
      }
      // Capped at the vocabulary rather than refused: top_k=1000 on a 1000-label
      // model is a request for everything, not a mistake, and the array it
      // returns is simply the whole vocabulary. The cap is silent in the output
      // because `top_k` echoes what was actually returned.
      if (top_k > n_labels) top_k = n_labels;
    }

    const std::string label =
        image->filename.empty() ? std::string("upload") : image->filename;
    try {
      // Decoded straight from the multipart bytes: an upload has no path, and
      // writing one to a temporary file just to read it back would put a
      // filesystem between the request and the decoder, with a name collision
      // and a cleanup path that has to be right on every error path.
      const npue::vit::Image im = npue::vit::decode_image_bytes(
          std::vector<uint8_t>(image->data.begin(), image->data.end()), label);
      const npue::vit::Prediction p = session.classify(im);
      ctype = "application/json";
      body = npue::vit::prediction_json(
          p, session.labels()[static_cast<size_t>(p.label)], label, top_k);
    } catch (const std::exception &e) {
      // 400 for what the CALLER sent (a format the front end refuses, a
      // geometry the container will not take) and 500 only for what is ours.
      const std::string what = e.what();
      const bool callers_fault =
          what.find("PNG") != std::string::npos ||
          what.find("JPEG") != std::string::npos ||
          what.find("magic") != std::string::npos ||
          what.find("truncated") != std::string::npos ||
          what.find("square") != std::string::npos ||
          what.find("shortest-edge") != std::string::npos;
      fail(callers_fault ? 400 : 500,
           callers_fault ? "invalid_request_error" : "internal_error", what);
    }
  });
  return 0;
}

}  // namespace app
