#include "pose/result_json.hpp"

#include <cstdio>
#include <stdexcept>

namespace npue::pose {

std::string result_json(const Result &r, const JsonContext &ctx) {
  if (!ctx.geom)
    throw std::runtime_error(
        "pose JSON: no geometry, so the input size and the head's channel "
        "counts are unknown; an answer that does not say what model produced it "
        "is not comparable with another answer");
  if (!ctx.params)
    throw std::runtime_error(
        "pose JSON: no decode parameters, so a detection count cannot be "
        "compared with a detection count from a different --conf");
  const Geometry &g = *ctx.geom;
  const DecodeParams &p = *ctx.params;

  std::string out = "{\n  \"image\": \"";
  for (char c : ctx.image_label) {
    if (c == '"' || c == '\\') out.push_back('\\');
    if (c == '\n') { out += "\\n"; continue; }
    out.push_back(c);
  }
  char buf[768];
  std::snprintf(
      buf, sizeof(buf),
      "\",\n  \"width\": %lld,\n  \"height\": %lld,\n"
      "  \"input_size\": %lld,\n"
      "  \"letterbox\": {\"scale\": %.6f, \"pad_x\": %lld, \"pad_y\": %lld},\n"
      "  \"thresholds\": {\"conf\": %.3f, \"iou\": %.3f, \"kpt\": %.3f, "
      "\"max_det\": %lld},\n"
      "  \"backend\": {\"conv\": \"%s\", \"dispatches\": %lld},\n"
      "  \"timing_s\": {\"front_end\": %.4f, \"network\": %.4f, \"decode\": "
      "%.4f, \"total\": %.4f},\n  \"landmarks\": [",
      static_cast<long long>(r.width), static_cast<long long>(r.height),
      static_cast<long long>(g.input_size), r.scale,
      static_cast<long long>(r.pad_x), static_cast<long long>(r.pad_y), p.conf,
      p.iou, p.keypoint, static_cast<long long>(p.max_det),
      ctx.array ? "npu" : "host",
      static_cast<long long>(r.dispatches), r.front_end_s, r.network_s,
      r.decode_s, r.total_s);
  out += buf;
  for (size_t i = 0; i < r.people.size(); ++i)
    out += (i ? ",\n    " : "\n    ") +
           npue::pose::person_json(r.people[i], static_cast<int64_t>(i));
  out += r.people.empty() ? "],\n" : "\n  ],\n";
  // The 17 names as 17 STRINGS, not one comma-joined line. The array form is
  // the one a client can index -- landmarks[k].keypoints[i].name is a lookup --
  // and the CSV exists for humans reading stderr, which is not this. Both come
  // from skeleton_names(), so the two cannot spell the order differently.
  out += "  \"skeleton\": {\"keypoints\": [";
  {
    const char *const *names = npue::pose::skeleton_names();
    for (int64_t i = 0; i < npue::pose::kNumKeypoints; ++i)
      out += (i ? ", " : "") + ("\"" + std::string(names[i]) + "\"");
  }
  out += "], \"edges\": [";
  const std::vector<std::string> edges = skeleton_edges();
  for (size_t i = 0; i < edges.size(); ++i)
    out += (i ? ", " : "") + ("\"" + edges[i] + "\"");
  out += "]}\n}\n";
  return out;
}

}  // namespace npue::pose