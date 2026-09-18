//===- embed_service.hpp -------------------------------------*- C++ -*-===//
//
// The OpenAI-shaped HTTP endpoint, shared by every architecture. It only
// needs the backend's vocabulary size, prompt whitelist and embed callback,
// so it is written against EmbedBackend rather than any encoder type.
// Split out of main.cpp verbatim.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "app_state.hpp"
#include "http.hpp"
#include "npue.hpp"

namespace app {

// ---------------------------------------------------------------------------
// The OpenAI-shaped HTTP endpoint, shared by every architecture (tasks/0115).
//
// This used to live inline in the BERT path and read `EmbedService` directly,
// which is the whole reason arch=1 refused `--serve`: not that anything about
// EmbeddingGemma is incompatible with an HTTP endpoint, but that the endpoint
// was written against one encoder's TYPE. The coupling turned out to be three
// members wide -- vocabulary size, the prompt whitelist, and "embed these
// texts" -- so it is expressed as exactly those three here, and both paths
// hand them over.
//
// Requests are handled ONE AT A TIME on purpose. The NPU serializes dispatches
// anyway (research/notes/0004), and the lanes already parallelise inside a
// single request -- so concurrent request handling would add contention and
// lock complexity to buy nothing. Throughput comes from batching within a
// request, which is what an embeddings client does.
struct EmbedBackend {
  size_t vocab_size = 0;
  // This model's task-prompt names, sorted. EMPTY means the model has no
  // prefix concept at all, and that emptiness is the single source of truth
  // for it -- same discipline as g_prompts.empty() on the CLI side.
  //
  // Deliberately the NAMES and not the table: arch 1 keeps its prompts in the
  // GEMATOK1 tokenizer blob and arch 0/2 keep theirs in the container config,
  // and unifying those two stores is not this task's job. Each wiring site
  // fills this from whichever store it already reads and resolves name -> text
  // inside its own lambda; the handler below validates against this vector and
  // never learns which store answered.
  std::vector<std::string> prompt_names;
  int64_t hidden = 0;
  int64_t seq = 0;
  // Must throw npue::InputTooLong for an input that does not fit (tasks/0110)
  // rather than truncating: the handler below maps that type to 400, and a
  // backend that flattened it to a runtime_error would return 500 for what is
  // the caller's error.
  //
  // `prompt_name` is per REQUEST (tasks/0118), and "" means no prefix at all.
  // A name reaching here has already been checked against prompt_names.
  std::function<std::vector<float>(const std::vector<std::string> &,
                                   const std::string &,
                                   int64_t *)> embed;
};

inline int serve_http(const EmbedBackend &be, const std::string &model_id, int port,
               const std::string &bind_addr) {
  std::printf("  tokenizer  %zu tokens, from the .npue\n", be.vocab_size);
  std::printf("\n  serving http://%s:%d/v1/embeddings   "
              "(model %s, seq %lld)\n",
              bind_addr.c_str(), port, model_id.c_str(), (long long)be.seq);
  std::printf("  POST {\"input\": \"text\" | [\"a\",\"b\"], "
              "\"encoding_format\": \"float\"|\"base64\"}\n\n");
  // PER REQUEST, not per process (tasks/0118). This used to be a startup
  // choice applied uniformly for the life of the process, with a NOTE here
  // saying so -- which forced a deployment wanting both search_query and
  // search_document to run two servers, each holding an hw_context on a shared
  // NPU. The name is a request field now, required for a model that has a
  // table, and discoverable from GET /health without having to provoke a 400.
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
        // DISCOVERY, so a client never has to provoke a 400 to learn what
        // this model offers (tasks/0118). This replaces the old "prefix" key,
        // which named the one server-wide prefix and is meaningless now that
        // the choice is per request. Still emitted only for a model that has
        // a prompt table at all, so a BERT model's /health is byte-identical
        // to what it was.
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

      // THE TASK PROMPT, PER REQUEST (tasks/0118). Three refusals, each with a
      // counterpart in resolve_prefix() on the CLI side, so the two cannot
      // disagree about what this model accepts: a model with a table that was
      // not told which prompt, a model without one that was told anyway, and a
      // name that is not in the table. Every one of them LISTS what this model
      // really offers instead of guessing.
      //
      // OpenAI's own /v1/embeddings has no such field -- its embedding models
      // are symmetric, so the problem does not arise there. Among servers that
      // host asymmetric models the field is not standardised: Cohere requires
      // `input_type` (and nomic's four names are Cohere's values verbatim),
      // vLLM puts `input_type` on its OpenAI-compatible route, and TEI put
      // `prompt_name` on its native /embed and deliberately kept it off the
      // OpenAI one. `prompt_name` is the name here because EmbeddingGemma's
      // table holds 14 sentence-transformers keys -- STS, BitextMining,
      // Summarization -- and calling those an "input type" would be a lie.
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
        // Mirrors resolve_prefix()'s first refusal. A client sweeping one
        // prompt across the whole catalogue SHOULD break here, because its
        // BERT results would otherwise differ from what it intended.
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
        // 400, not 500: the request is the thing that is wrong, and an
        // operator told 500 goes looking at the NPU. This catch must come
        // FIRST -- InputTooLong derives from std::runtime_error, so the
        // generic handler below would otherwise swallow it.
        fail(400, "invalid_request_error", e.what());
        return;
      } catch (const std::exception &e) {
        fail(500, "internal_error", e.what());
        return;
      }

      // 384 floats per row: reserve rather than grow, or a 2048-input
      // response reallocates its way through tens of MB.
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
