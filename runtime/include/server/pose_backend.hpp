//===- pose_backend.hpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the image-in, pose-out HTTP endpoint.
//
//	POST /v1/pose   multipart/form-data, one `image` part (PNG or JPEG),
//	                optional `conf`, `iou`, `kpt`, `max_det`
//
// The answer is byte for byte what `npuimage pose ... --json` prints: both
// call npue::pose::result_json. That is stated as a contract rather than left to
// inspection because the two are compared against each other -- the CLI's is
// what somebody reads to decide whether the model is right, the endpoint's is
// what a program parses, and "the server and the CLI disagree" is not a bug
// anybody can act on without first diffing two JSON documents by hand.
//
// WHY `serve` AND NOT A VERB OF ITS OWN
// -------------------------------------
// It was a `pose-server` subcommand, and the reasoning was that `serve` is an
// OpenAI-shaped endpoint: JSON in, vectors out. That reasoning was half right. It
// is right that /v1/embeddings must not answer with landmarks -- no client on
// either side would accept that. It was wrong about the fix, because a subcommand
// per architecture solves a problem the URL space never had: the path already says
// what the answer is, and the container's arch says which mode runs. Whisper has
// answered /v1/audio/transcriptions off the same `--serve` for years without
// confusing anything.
//
// So there is ONE verb and FOUR endpoints, and `serve` is the only one to learn:
//
//   /v1/embeddings           arch 1,2,3   a BERT-family text model
//   /v1/audio/transcriptions arch 4       Whisper
//   /v1/classify             arch 5       a ViT classifier
//   /v1/pose                 arch 6       YOLOv8-pose, this one
//
// The modes are dispatched on the container's arch in runtime.cpp, before `--serve`
// is ever read, so no two of them can meet and no flag chooses between them. What
// /v1/embeddings must NOT answer with is enforced by the path, not by a verb.
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
//   * `conf`/`iou`/`kpt` that are not numbers in range -- silently falling back
//                                        to the default would answer a request
//                                        with thresholds the client did not ask
//                                        for, and the detection count would be
//                                        read as the model's
//   * anything but PNG and JPEG       -- the front end's own refusal, by name,
//                                        rather than a decoder that guesses
//
// WHAT IT DOES NOT DO
// -------------------
// No streaming, no batch, no world landmarks, no segmentation masks, and no
// per-request model selection. One request is one image on one process, and a
// client that wants a second model starts a second process -- which is the same
// rule the speech endpoint follows, and for the same reason: one hw_context per
// model is the budget this hardware has.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

#include "pose/session.hpp"

namespace app {

// Serves until interrupted. Returns the process exit code.
//
// `params` is the session's own default decode parameters and is also what a
// request that names no thresholds gets; a request that names them gets its own
// values for THAT request only, because mutating the session's would make one
// request's `--conf` the next request's default.
int serve_pose(npue::pose::Session &session, const std::string &model_id,
               int port, const std::string &bind_addr,
               const npue::pose::DecodeParams &defaults);

}  // namespace app