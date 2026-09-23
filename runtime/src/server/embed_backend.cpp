#include "server/embed_backend.hpp"
#include "runtime/model.hpp"

#include <csignal>

namespace npue {
namespace http {
// The stop flag the SIGINT/SIGTERM handler sets; Server::run polls it. See
// server.hpp for why this exists (SUBTASKS.md subtask 7).
volatile std::sig_atomic_t g_server_stop = 0;
}  // namespace http
}  // namespace npue

namespace {
// File-local signal handler: sets only the flag, nothing that is not
// async-signal-safe.
void on_stop_signal(int) { npue::http::g_server_stop = 1; }
}  // namespace

namespace app {

int serve_http(const EmbedBackend &be, const std::string &model_id, int port,
               const std::string &bind_addr) {
  // Unwind on Ctrl-C / SIGTERM instead of dying with the hw_contexts held.
  // A handler that only sets a sig_atomic_t is async-signal-safe; the accept
  // loop notices it within one select() timeout and returns through the stack,
  // where ~RunContext destroys the Device and every Design.
  std::signal(SIGINT, on_stop_signal);
  std::signal(SIGTERM, on_stop_signal);
  std::printf("  tokenizer  %zu tokens, from the .npue\n", be.vocab_size);
  std::printf("\n  serving http://%s:%d/v1/embeddings   "
              "(model %s, seq %lld)\n",
              bind_addr.c_str(), port, model_id.c_str(), (long long)be.seq);
  std::printf("  POST {\"input\": \"text\" | [\"a\",\"b\"], "
              "\"encoding_format\": \"float\"|\"base64\"}\n\n");
  if (!be.prompt_names.empty())
    std::printf("  REQUIRED per request: \"prompt_name\", one of [%s] "
                "(or \"\" for no prefix at all). A request without it is "
                "400.\n\n", join_names(be.prompt_names).c_str());

  const size_t kMaxTexts = 2048;
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
             "\",\"backend\":\"amd-xdna2-npu\"";
      if (!be.prompt_names.empty()) {
        body += ",\"prompt_names\":[";
        for (size_t i = 0; i < be.prompt_names.size(); ++i)
          body += (i ? ",\"" : "\"") +
                  npue::http::json_escape(be.prompt_names[i]) + "\"";
        body += "],\"prompt_required\":true";
      }
      body += "}";
      return;
    }
    if (req.method == "GET" && req.path == "/v1/models") {
      body = "{\"object\":\"list\",\"data\":[{\"id\":\"" + model_id +
             "\",\"object\":\"model\",\"owned_by\":\"npuembeddings\"}]}";
      return;
    }
    if (req.path != "/v1/embeddings") {
      fail(404, "not_found", "unknown path " + req.path);
      return;
    }
    if (req.method != "POST") {
      fail(400, "invalid_request_error", "use POST for /v1/embeddings");
      return;
    }

    std::vector<std::string> texts;
    std::string err;
    if (!npue::http::json_string_or_array(req.body, "input", texts, err)) {
      fail(400, "invalid_request_error", err);
      return;
    }
    if (texts.empty()) {
      fail(400, "invalid_request_error", "'input' is empty");
      return;
    }
    if (texts.size() > kMaxTexts) {
      fail(413, "invalid_request_error",
           "at most " + std::to_string(kMaxTexts) + " inputs per request, "
           "got " + std::to_string(texts.size()));
      return;
    }
    const std::string fmt =
        npue::http::json_field_string(req.body, "encoding_format", "float");
    if (fmt != "float" && fmt != "base64") {
      fail(400, "invalid_request_error",
           "encoding_format must be 'float' or 'base64', got '" + fmt + "'");
      return;
    }

    std::string prompt_name;
    const auto pk =
        npue::http::json_field_kind(req.body, "prompt_name", prompt_name);
    if (pk == npue::http::FieldKind::Other) {
      fail(400, "invalid_request_error",
           "'prompt_name' must be a string (\"\" means no prefix at all)");
      return;
    }
    const bool have_prompt = pk == npue::http::FieldKind::String;
    if (!be.prompt_names.empty() && !have_prompt) {
      fail(400, "invalid_request_error",
           "this model requires 'prompt_name'; valid names: [" +
           join_names(be.prompt_names) + "], or \"\" for no prefix at all. "
           "Refusing to pick one for you -- a wrongly-prefixed embedding is "
           "correctly shaped and correctly normed, so nothing downstream can "
           "tell that the answer is wrong.");
      return;
    }
    if (be.prompt_names.empty() && have_prompt) {
      fail(400, "invalid_request_error",
           "'prompt_name' was given, but this model has no task prompts -- "
           "nothing would be prepended. Refusing rather than returning "
           "vectors that are not what was asked for.");
      return;
    }
    if (have_prompt && !prompt_name.empty() &&
        std::find(be.prompt_names.begin(), be.prompt_names.end(),
                  prompt_name) == be.prompt_names.end()) {
      fail(400, "invalid_request_error",
           "'" + prompt_name + "' is not one of this model's prompt names: ["
           + join_names(be.prompt_names) + "]");
      return;
    }

    int64_t n_tokens = 0;
    std::vector<float> emb;
    try {
      emb = be.embed(texts, prompt_name, &n_tokens);
    } catch (const npue::InputTooLong &e) {
      fail(400, "invalid_request_error", e.what());
      return;
    } catch (const std::exception &e) {
      fail(500, "internal_error", e.what());
      return;
    }

    std::string out;
    out.reserve(texts.size() * (fmt == "base64" ? 2200 : 4600) + 256);
    out += "{\"object\":\"list\",\"data\":[";
    char num[40];
    for (size_t r = 0; r < texts.size(); ++r) {
      if (r) out += ',';
      out += "{\"object\":\"embedding\",\"index\":" + std::to_string(r) +
             ",\"embedding\":";
      const float *v = emb.data() + r * be.hidden;
      if (fmt == "base64") {
        out += '"';
        out += npue::http::base64(reinterpret_cast<const uint8_t *>(v),
                                  static_cast<size_t>(be.hidden) * sizeof(float));
        out += '"';
      } else {
        out += '[';
        for (int64_t c = 0; c < be.hidden; ++c) {
          if (c) out += ',';
          std::snprintf(num, sizeof num, "%.7g", v[c]);
          out += num;
        }
        out += ']';
      }
      out += '}';
    }
    out += "],\"model\":\"" + model_id +
           "\",\"usage\":{\"prompt_tokens\":" + std::to_string(n_tokens) +
           ",\"total_tokens\":" + std::to_string(n_tokens) + "}}";
    body.swap(out);
  });
  return 0;
}

}  // namespace app