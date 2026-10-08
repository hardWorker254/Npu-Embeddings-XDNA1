//===- result_json.hpp ----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- ONE JSON spelling of a pose result, for both callers.
//
// `npuimage pose ... --json` and the HTTP endpoint answer the same object,
// byte for byte, because they call the same function. That is not tidiness: the
// CLI's output is what a reader compares against MediaPipe's PoseLandmarker by
// eye, and the endpoint's output is what a program parses. Two emitters would
// drift -- a key renamed on one side, a rounding difference on the other -- and
// the drift would show up as "the server and the CLI disagree", which is a
// question nobody can answer without diffing two JSON documents by hand.
//
// WHAT IS IN THE OBJECT, AND WHY
// ------------------------------
//   landmarks[]     one per person, in the SAME order as the model's 17 COCO
//                   joints. Every joint carries both its index and its name, so
//                   a client that reads the name and a client that indexes
//                   positionally agree by construction -- the one thing that
//                   cannot be true of an unnamed array.
//   letterbox       the front end's own transform, so a client can map ITS
//                   annotations (a drawn box, a cursor) through the same
//                   pipeline the model saw rather than guessing at the resize.
//   thresholds      the decode parameters this answer was produced under. An
//                   answer without them cannot be compared with another answer
//                   at a different `--conf`, and comparing them anyway is how a
//                   detection count gets read as a property of the model.
//   backend         "npu" or "host", and the dispatch count on the array. A pose
//                   answer does not come out of a vacuum and the numbers differ:
//                   the array's datapath is bf16 (net.hpp), so its keypoints are
//                   measurably further from fp32's than the host's.
//   timing_s        the four spans, split, because one wall-clock number hides
//                   which of them moved.
//
// `image` is a LABEL, not a path: the server has no path for an upload, and a
// field called "image" that is sometimes a filename and sometimes a client-
// supplied string is a field nobody can rely on.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

#include "pose/decode.hpp"
#include "pose/geometry.hpp"
#include "pose/session.hpp"

namespace npue::pose {

// Everything about the RUN that Result itself does not carry. Passed in rather
// than read off the session so that this function has no dependency on a
// Session, and so a verification harness can serialise a Result it built by
// hand.
struct JsonContext {
  const Geometry *geom = nullptr;
  const DecodeParams *params = nullptr;
  // Free text naming the input: a filename for the CLI, the upload's filename
  // for the server, whatever the client called it.
  std::string image_label;
  // True when the convolutions ran on the array. It changes two things in the
  // output: the backend string, and the dispatch count.
  bool array = false;
};

// The whole object, pretty-printed with two-space indents. `ctx.geom` and
// `ctx.params` must both be non-null; there is no sensible default for either,
// and a Result without them would be an answer whose units are unknown.
std::string result_json(const Result &r, const JsonContext &ctx);

}  // namespace npue::pose