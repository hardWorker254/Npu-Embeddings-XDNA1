//===- stt_backend.cpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the OpenAI-shaped speech-to-text endpoint.
// See server/stt_backend.hpp for the refusals and why each one is there.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "server/stt_backend.hpp"

#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "server/multipart.hpp"
#include "server/server.hpp"
#include "whisper/audio.hpp"

namespace npue {

std::string transcript_json(const whisper::Transcript &t, bool verbose) {
  std::string o = "{\"text\":\"" + http::json_escape(t.text) + "\"";
  if (verbose) {
    char num[64];
    o += ",\"task\":\"" + http::json_escape(t.task) + "\"";
    o += ",\"language\":\"" + http::json_escape(t.language) + "\"";
    std::snprintf(num, sizeof num, "%.3f",
                  static_cast<double>(t.n_samples) / whisper::kSampleRate);
    o += ",\"duration\":" + std::string(num);
    o += ",\"chunks\":" + std::to_string(t.n_chunks);
    o += ",\"segments\":[";
    for (size_t i = 0; i < t.segments.size(); ++i) {
      const auto &s = t.segments[i];
      if (i) o += ',';
      o += "{\"id\":" + std::to_string(i) +
           ",\"seek\":" + std::to_string(s.start_sample);
      std::snprintf(num, sizeof num, "%.2f", s.start_s);
      o += ",\"start\":" + std::string(num);
      std::snprintf(num, sizeof num, "%.2f", s.end_s);
      o += ",\"end\":" + std::string(num);
      o += ",\"text\":\"" + http::json_escape(s.text) + "\"}";
    }
    o += ']';
  }
  o += "}";
  return o;
}

}  // namespace npue

namespace {

// File-local signal handler: sets only the flag, nothing that is not
// async-signal-safe. The accept loop notices it within one select() timeout and
// unwinds, so the Device and both Designs release their hw_contexts.
void on_stop_signal(int) { npue::http::g_server_stop = 1; }

// The upload goes through a real file, because both ingest paths (the WAV reader
// and ffmpeg) work on paths and neither should grow a memory variant just for
// the HTTP case.
//
// mkstemp needs the template to END in six X's, so the suffix is appended after
// the name exists -- a template like "...XXXXXX.wav" is rejected, and a rejected
// mkstemp that nobody checks is a file named with literal X's in the temp
// directory. Unlinked in the destructor; a process killed between the two leaves
// the file behind, which is the price of not adding a second ingest path.
class TempAudio {
public:
  explicit TempAudio(const char *suffix) {
    const char *dir = std::getenv("TMPDIR");
    const std::string tmpl =
        std::string(dir && *dir ? dir : "/tmp") + "/npu-stt-XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const int fd = ::mkstemp(buf.data());
    if (fd < 0)
      throw std::runtime_error("cannot create a temporary file for the upload");
    ::close(fd);
    path_ = std::string(buf.data()) + suffix;
    if (std::rename(buf.data(), path_.c_str()) != 0) {
      ::unlink(buf.data());
      throw std::runtime_error("cannot name the temporary upload " + path_);
    }
  }
  ~TempAudio() { std::remove(path_.c_str()); }
  const std::string &path() const { return path_; }
  void write(const std::string &data) const {
    std::FILE *f = std::fopen(path_.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path_);
    if (!data.empty() &&
        std::fwrite(data.data(), 1, data.size(), f) != data.size()) {
      std::fclose(f);
      throw std::runtime_error("short write to " + path_);
    }
    std::fclose(f);
  }

private:
  std::string path_;
};

std::string field(const std::vector<npue::http::MultipartPart> &parts,
                  const char *name) {
  for (const auto &p : parts)
    if (p.name == name) return p.data;
  return std::string();
}

bool has_field(const std::vector<npue::http::MultipartPart> &parts,
               const char *name) {
  for (const auto &p : parts)
    if (p.name == name) return true;
  return false;
}

bool ends_with(const std::string &s, const char *suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

}  // namespace

namespace app {

int serve_stt(npue::whisper::Session &session, const std::string &model_id,
              int port, const std::string &bind_addr,
              const npue::whisper::TranscribeOptions &defaults) {
  std::signal(SIGINT, on_stop_signal);
  std::signal(SIGTERM, on_stop_signal);
  const auto &g = session.geometry();
  std::printf("  tokenizer  %zu ids, from the .npue\n",
              session.tokenizer().vocab_size());
  std::printf("\n  serving http://%s:%d/v1/audio/transcriptions   (model %s)\n",
              bind_addr.c_str(), port, model_id.c_str());
  std::printf("  POST multipart/form-data: file (required), model, language, "
              "task,\n              response_format (json | verbose_json | "
              "text)\n");
  std::printf("  %lld mel bins, %lld encoder positions, greedy, temperature 0, "
              "no timestamps\n\n",
              static_cast<long long>(g.mel_bins),
              static_cast<long long>(g.max_seq));
  // Flushed, not left to the buffer. Every status line above this one goes to
  // stderr and is unbuffered, so this banner is the LAST thing a redirected log
  // receives; without the flush it sits in a block buffer until 4 KiB of output
  // accumulates, which for a server that then prints nothing means the log never
  // shows the port at all. An operator tailing that log would conclude the
  // process had not started. All four endpoints do this.
  std::fflush(stdout);

  npue::http::Server server(static_cast<uint16_t>(port), bind_addr);
  server.run([&](const npue::http::Request &req, int &status,
                 std::string &ctype, std::string &body) {
    auto fail = [&](int code, const char *type, const std::string &msg) {
      status = code;
      body = "{\"error\":{\"message\":\"" + npue::http::json_escape(msg) +
             "\",\"type\":\"" + type + "\"}}";
    };

    if (req.method == "GET" && (req.path == "/health" || req.path == "/")) {
      body = "{\"status\":\"ok\",\"model\":\"" + model_id +
             "\",\"backend\":\"amd-xdna2-npu\",\"kind\":\"stt\",\"language\":\"" +
             defaults.language + "\",\"task\":\"" + defaults.task +
             "\",\"timestamps\":false,\"chunk_seconds\":" +
             std::to_string(defaults.chunk_seconds) + ",\"stride_seconds\":" +
             std::to_string(defaults.stride_seconds) + "}";
      return;
    }
    if (req.method == "GET" && req.path == "/v1/models") {
      body = "{\"object\":\"list\",\"data\":[{\"id\":\"" + model_id +
             "\",\"object\":\"model\",\"owned_by\":\"npuembeddings\"}]}";
      return;
    }
    if (req.path != "/v1/audio/transcriptions") {
      fail(404, "not_found",
           "unknown path " + req.path +
               " -- this model serves /v1/audio/transcriptions, not /v1/"
               "embeddings. `npuaudio serve <model>` picks the endpoint from the "
               "container's arch: text models answer /v1/embeddings, Whisper "
               "/v1/audio/transcriptions, a ViT /v1/classify and YOLOv8-pose "
               "/v1/pose.");
      return;
    }
    if (req.method != "POST") {
      fail(400, "invalid_request_error",
           "use POST for /v1/audio/transcriptions");
      return;
    }

    const std::string boundary =
        npue::http::multipart_boundary(req.content_type);
    if (boundary.empty()) {
      fail(400, "invalid_request_error",
           "expected Content-Type: multipart/form-data with a boundary; got '" +
               req.content_type +
               "'. This endpoint takes the audio as a form part -- not as raw "
               "bytes, and not as JSON.");
      return;
    }
    std::vector<npue::http::MultipartPart> parts;
    std::string err;
    if (!npue::http::parse_multipart(req.body, boundary, parts, err)) {
      fail(400, "invalid_request_error", "multipart: " + err);
      return;
    }

    // -- refusals that are POLICY, not parsing ------------------------------
    const std::string model = field(parts, "model");
    if (!model.empty() && model != model_id) {
      fail(400, "invalid_request_error",
           "this endpoint serves '" + model_id + "', not '" + model +
               "'. One process serves one model; start another for another.");
      return;
    }
    const std::string temperature = field(parts, "temperature");
    if (!temperature.empty() && temperature != "0" && temperature != "0.0") {
      fail(400, "invalid_request_error",
           "temperature " + temperature +
               " was requested. This build is greedy only, and answering a "
               "sampling request with a deterministic transcript would be a lie "
               "about what happened. Send temperature 0, or leave it out.");
      return;
    }
    if (has_field(parts, "prompt")) {
      fail(400, "invalid_request_error",
           "'prompt' (conditioning the decoder on previous text) is not "
           "implemented here. Accepting it and ignoring it would return a "
           "transcription of the audio alone, which is not what was asked for.");
      return;
    }
    if (field(parts, "timestamp_granularities") == "word" ||
        field(parts, "response_format") == "word" ||
        field(parts, "response_format") == "segment") {
      fail(400, "invalid_request_error",
           "word or segment timings were requested; this build decodes no "
           "timestamps, so it cannot answer with them. Use response_format "
           "json or verbose_json -- verbose_json's 'segments' are THIS build's "
           "30 s windows with their offsets, not the model's own word "
           "boundaries.");
      return;
    }

    // -- the audio ----------------------------------------------------------
    const npue::http::MultipartPart *audio = nullptr;
    int n_audio = 0;
    for (const auto &p : parts)
      if (p.name == "file") { ++n_audio; audio = &p; }
    if (n_audio > 1) {
      fail(400, "invalid_request_error",
           "the request carries " + std::to_string(n_audio) +
               " 'file' parts; which one is the audio is not a question to "
               "answer by taking the first");
      return;
    }
    if (!audio || audio->data.empty()) {
      fail(400, "invalid_request_error",
           "'file' is required, and the one that arrived was empty");
      return;
    }
    if (audio->data.size() > 200u * 1024u * 1024u) {
      fail(413, "invalid_request_error",
           "the upload is " + std::to_string(audio->data.size() / (1u << 20)) +
               " MB; this build caps a request at 200 MB");
      return;
    }

    npue::whisper::TranscribeOptions opts = defaults;
    const std::string language = field(parts, "language");
    const std::string task = field(parts, "task");
    if (!language.empty()) opts.language = language;
    if (!task.empty()) opts.task = task;
    // A non-WAV upload (mp3, m4a, webm, ...) goes through ffmpeg; the WAV
    // reader refuses those by design, and this is the documented way in -- the
    // same --convert the CLI has.
    opts.convert = !(ends_with(audio->filename, ".wav") ||
                     ends_with(audio->filename, ".WAV"));

    try {
      TempAudio tmp(opts.convert ? ".bin" : ".wav");
      tmp.write(audio->data);
      const auto t = session.transcribe_file(tmp.path(), opts);
      const std::string fmt = field(parts, "response_format");
      if (fmt == "text") {
        ctype = "text/plain; charset=utf-8";
        body = t.text + "\n";
      } else {
        body = npue::transcript_json(t, fmt == "verbose_json");
      }
    } catch (const std::exception &e) {
      // 400 for what the CALLER sent (a rate, a language, a file the front end
      // refuses) and 500 only for what is ours. The serve path draws the same
      // line: a wrong request is not a server error, and a client that sees 500
      // stops trying.
      const std::string what = e.what();
      const bool callers_fault =
          what.find("sample rate") != std::string::npos ||
          what.find("channels") != std::string::npos ||
          what.find("bits per sample") != std::string::npos ||
          what.find("--language") != std::string::npos ||
          what.find("--task") != std::string::npos ||
          what.find("ffmpeg") != std::string::npos;
      fail(callers_fault ? 400 : 500,
           callers_fault ? "invalid_request_error" : "internal_error", what);
    }
  });
  return 0;
}

}  // namespace app
