//===- stt_backend.hpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the OpenAI-shaped speech-to-text endpoint.
//
// /v1/audio/transcriptions, multipart/form-data, one `file` part and the
// optional `model`, `language`, `task`, `response_format`, `temperature`,
// `prompt` fields. It answers with the transcript and, when asked for
// verbose_json, with the per-window segments and their offsets.
//
// WHAT IT REFUSES, AND WHY EACH ONE MATTERS
// ------------------------------------------
//   * a body that is not multipart          -- it cannot tell what it was given
//   * a missing or empty `file` part        -- there is no audio to transcribe
//   * a second `file` part                  -- ambiguous, and the first one
//                                              would be the one it silently used
//   * `temperature` other than 0            -- this build is greedy ONLY, and a
//                                              non-zero temperature answered with
//                                              a deterministic transcript would
//                                              be a lie about the request
//   * `prompt`                              -- conditioning text is a real
//                                              Whisper feature that this build
//                                              does not implement; accepting it
//                                              and ignoring it is worse than
//                                              refusing it
//   * `timestamp_granularities`             -- no timestamps, see below
//
// TIMESTAMPS ARE NOT IMPLEMENTED AND SAY SO
// ----------------------------------------
// Every response carries "task", "language" and "duration", and a request that
// asks for granularities or for a timestamped response gets a 400 naming that.
// A client that asked for word timings and got a string with no timings in it
// would have to notice.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

#include "whisper/transcribe.hpp"

namespace app {

// Serves until interrupted. Returns the process exit code.
int serve_stt(npue::whisper::Session &session, const std::string &model_id,
              int port, const std::string &bind_addr,
              const npue::whisper::TranscribeOptions &defaults);

}  // namespace app

namespace npue {

// The response object, in the shape the OpenAI endpoint returns. One definition
// for the CLI's --json and for the endpoint, so the two cannot drift.
//
// `verbose` adds the per-window segments with their sample offsets. They are
// THIS build's windows, not the model's own segment boundaries: without
// timestamps there is no way to know where inside a window a word was said, and
// inventing that would be a lie with a number attached.
std::string transcript_json(const whisper::Transcript &t, bool verbose);

}  // namespace npue
