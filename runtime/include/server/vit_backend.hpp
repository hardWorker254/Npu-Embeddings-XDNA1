//===- vit_backend.hpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the image-in, label-out HTTP endpoint.
//
//	POST /v1/classify   multipart/form-data, one `image` part (PNG or JPEG),
//	                   optional `top_k`
//
// WHY IT EXISTS AT ALL, AND WHY IT TOOK A SEPARATE DECISION TO ADD IT
// ------------------------------------------------------------------
// The other three architectures have had an endpoint for a long time: a
// BERT-family model answers /v1/embeddings, Whisper answers
// /v1/audio/transcriptions, and this tree added /v1/pose for YOLOv8-pose. The
// image classifier was the one that refused, with a message saying there was no
// endpoint for it. That refusal was honest and it was also an inconsistency:
// `serve` is the verb, the container's arch picks the endpoint, and an arch that
// refuses the verb is an arch a user has to discover by typing. So it is here.
//
// The answer is byte for byte what `npuembeddings classify ... --json` prints,
// because both call npue::vit::prediction_json. See pose/result_json.hpp for why
// that is stated as a contract rather than left to inspection.
//
// WHAT IT REFUSES, AND WHY EACH ONE MATTERS
// ------------------------------------------
//   * a body that is not multipart    -- an image is binary, and this cannot
//                                        tell a JPEG from a JSON string that
//                                        happens to be long
//   * a missing or empty `image` part -- there is no picture to look at
//   * a second `image` part           -- ambiguous, and taking the first would
//                                        be a silent answer to an ambiguous
//                                        question
//   * `top_k` below 1 or above the number of labels -- "the top 0 labels" is a
//                                        malformed request, and answering it
//                                        with an empty array would look like a
//                                        model that recognised nothing
//   * anything but PNG and JPEG       -- the front end's own refusal, by name,
//                                        rather than a decoder that guesses
//
// WHAT IT DOES NOT DO
// -------------------
// No batch (one image per request), no top-k over the wire by default (the CLI
// prints the runners-up to stderr and the endpoint does not, so that a `p` of
// 0.99 in a response is not competing with two other numbers for attention), no
// per-request model selection, and no concurrent overlap: one request at a time,
// which is what the other three endpoints do and why.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

#include "vit/classify.hpp"

namespace app {

// Serves until interrupted. Returns the process exit code.
//
// `default_top_k` is how many labels a request that names no `top_k` gets; 1 means
// the argmax alone, which is what the CLI's own default JSON prints.
int serve_vit(npue::vit::Session &session, const std::string &model_id,
              int port, const std::string &bind_addr, int64_t default_top_k);

}  // namespace app
