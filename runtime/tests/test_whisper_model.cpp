//===- test_whisper_model.cpp -----------------------------------*- C++ -*-===//
//
// Runs the Whisper NPU stacks so a gate can hold them against transformers.
//
// Same arrangement as the audio front end's gate: a test binary rather than a
// CLI, because nothing in the shipped runtime reaches arch=4 yet, and because
// the decoder needs a forced token sequence driven from outside -- a gate that
// fed the decoder its own previous output could not tell a correct first step
// from a correct hundredth.
//
//   argv[1] <model.npue>      the container: geometry and every weight
//   argv[2] <artifacts>       design set root; <artifacts>/gemm_rtp for the
//                             encoder and <artifacts>/gemm_rtp_dec for the
//                             decoder, both read through their own design.json
//   argv[3] <audio.wav>       16 kHz mono; the front end runs here so the gate
//                             sees the same mel the model would
//   --ids a,b,c               teacher-forced decoder steps at positions 0..n-1
//   --greedy a,b,c            greedy continuation of this prompt
//   --max-new N               cap on GENERATED tokens (0: the checkpoint's own
//                             position bound). The gate passes the same cap to
//                             transformers' generate(), so the two chains are
//                             the same string and not just a shared prefix
//   --step-tier N             decoder tier for one step (default: the smallest)
//   --prefill-tier N          decoder tier for the cross-attention K|V prefill
//                             (default: the largest)
//   --threads N               host workers
//   --skip-decoder            encoder only, and do not open the decoder xclbin
//
// Output, all on stdout, all hex, so the gate parses one thing and never
// round-trips a float through a decimal pipe:
//   conv <rows> <cols> <hex>            the front end's output
//   enc <rows> <cols> <hex>             encoder hidden states (final LayerNorm)
//   step <pos> <top1> <cols> <hex>      one decoder step's final state
//   logits <pos> <vocab> <hex>          that step's logits
//   greedy <n> <id> ...                 the generated ids
//   text <hex>                          those ids as text, through the
//                                       container's own tokenizer table
//
// Build:
//   g++ -std=c++17 -O2 -march=native -I runtime/include -I /opt/xilinx/xrt/include \
//       runtime/tests/test_whisper_model.cpp runtime/src/whisper/*.cpp \
//       runtime/src/whisper/features.cpp runtime/src/whisper/audio.cpp \
//       runtime/src/{device,design,model,pool}.cpp runtime/src/npu_contention.cpp \
//       -L /opt/xilinx/xrt/lib -lxrt_coreutil -o /tmp/test_whisper_model
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/design_selection.hpp"  // parse_streams
#include "common/host_kernels.hpp"      // now_s
#include "runtime/design.hpp"
#include "runtime/device.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "tokenizers/whisper.hpp"
#include "whisper/audio.hpp"
#include "whisper/decoder.hpp"
#include "whisper/encoder.hpp"
#include "whisper/geometry.hpp"
#include "whisper/features.hpp"

namespace {

std::string to_hex(const void *data, size_t bytes) {
  static const char *d = "0123456789abcdef";
  const auto *p = static_cast<const unsigned char *>(data);
  std::string out;
  out.reserve(bytes * 2);
  for (size_t i = 0; i < bytes; ++i) {
    out.push_back(d[p[i] >> 4]);
    out.push_back(d[p[i] & 0xF]);
  }
  return out;
}

std::string read_text(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Load every stream of a design set into its own slot, in the order design.json
// records, and CHECK that the slot the file was written with is the slot the
// loader hands back. A set whose streams were loaded in a different order runs
// the wrong instruction stream for an op and returns a plausible number: the
// slot number is the only place that mismatch can be seen.
std::vector<app::StreamEntry> load_streams(npu::Design &d,
                                           const std::string &dir) {
  const std::string js = read_text(dir + "/design.json");
  std::vector<app::StreamEntry> streams = app::parse_streams(js);
  if (streams.empty())
    throw std::runtime_error(dir +
                             "/design.json lists no streams -- re-export with "
                             "tools/export_gemm_rtp.py");
  std::sort(streams.begin(), streams.end(),
            [](const app::StreamEntry &a, const app::StreamEntry &b) {
              return a.slot < b.slot;
            });
  for (const auto &s : streams) {
    const size_t got = d.load_instr(dir + "/" + s.file);
    if (static_cast<int64_t>(got) != s.slot)
      throw std::runtime_error("stream " + s.file + " landed in slot " +
                               std::to_string(got) + ", design.json says " +
                               std::to_string(s.slot));
  }
  return streams;
}

const app::StreamEntry &find_op(const std::vector<app::StreamEntry> &streams,
                                 const std::string &op, int64_t batch) {
  for (const auto &s : streams)
    if (s.op == op && s.batch == batch) return s;
  throw std::runtime_error("no " + op + " stream at batch tier " +
                           std::to_string(batch) +
                           " -- re-export the design set with this tier");
}

std::vector<int64_t> parse_ids(const std::string &s) {
  std::vector<int64_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(std::stoll(item));
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <model.npue> <artifacts> <audio.wav> "
                 "[--ids a,b] [--greedy a,b] [--max-new N] [--step-tier N] "
                 "[--prefill-tier N] [--threads N] [--skip-decoder]\n",
                 argv[0]);
    return 2;
  }
  using namespace npue::whisper;

  const std::string npue_path = argv[1], art = argv[2], audio = argv[3];
  std::vector<int64_t> ids, prompt;
  int threads = 1;
  int64_t max_new = 0;
  int64_t step_tier = -1, prefill_tier = -1;
  bool skip_decoder = false;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--ids") ids = parse_ids(next());
    else if (a == "--max-new") max_new = std::atoll(next().c_str());
    else if (a == "--greedy") prompt = parse_ids(next());
    else if (a == "--threads") threads = std::max(1, std::atoi(next().c_str()));
    else if (a == "--step-tier") step_tier = std::atoll(next().c_str());
    else if (a == "--prefill-tier") prefill_tier = std::atoll(next().c_str());
    else if (a == "--skip-decoder") skip_decoder = true;
    else {
      std::fprintf(stderr, "unknown flag %s\n", a.c_str());
      return 2;
    }
  }

  try {
    npue::File model(npue_path);
    const Geometry geom = read_geometry(model, npue_path);
    app::Pool pool(threads);
    app::Pool *use = threads > 1 ? &pool : nullptr;

    // The front end runs here rather than in the gate, so the C++ side and the
    // Python side start from the same samples and the mel is not a second
    // variable in the comparison.
    const npue::whisper::Audio a = npue::whisper::ingest(audio, false);
    const MelSpec mel =
        log_mel_30s(a.samples, static_cast<int>(geom.mel_bins), nullptr, use);
    const std::vector<float> conv = conv_front_end(
        mel, model.raw("frontend.conv1.weight").as<float>(),
        model.raw("frontend.conv1.bias").as<float>(),
        model.raw("frontend.conv2.weight").as<float>(),
        model.raw("frontend.conv2.bias").as<float>(),
        static_cast<int>(geom.d_model), use);
    const int64_t n_src = kEncoderPositions;
    std::printf("conv %lld %lld %s\n", static_cast<long long>(n_src),
                static_cast<long long>(geom.d_model),
                to_hex(conv.data(), conv.size() * 4).c_str());

    npu::Device dev;
    npu::Design enc_design(dev, art + "/gemm_rtp");
    const std::vector<app::StreamEntry> enc_streams =
        load_streams(enc_design, art + "/gemm_rtp");

    // The encoder design set carries one batch tier, so its rows are whatever
    // that tier's streams say. A set with several tiers is a different
    // architecture (that is the decoder's shape), not a wider encoder.
    std::vector<int64_t> tiers;
    for (const auto &s : enc_streams) tiers.push_back(s.batch);
    std::sort(tiers.begin(), tiers.end());
    tiers.erase(std::unique(tiers.begin(), tiers.end()), tiers.end());
    if (tiers.size() != 1)
      throw std::runtime_error(
          art + "/gemm_rtp has " + std::to_string(tiers.size()) +
          " batch tiers; the encoder is exported for one, because 1500 "
          "positions are walked in chunks of that tier's row count and a "
          "second tier would mean a second chunking policy");
    const int64_t eb = tiers[0];
    EncoderStreams es;
    es.qkv = static_cast<size_t>(find_op(enc_streams, "qkv", eb).slot);
    es.attn_out = static_cast<size_t>(find_op(enc_streams, "attn_out", eb).slot);
    es.ffn_up = static_cast<size_t>(find_op(enc_streams, "ffn_up", eb).slot);
    es.ffn_down = static_cast<size_t>(find_op(enc_streams, "ffn_down", eb).slot);
    es.rows = find_op(enc_streams, "qkv", eb).M;

    WhisperEncoder enc(model, enc_design, pool, geom);
    enc.set_streams(es);
    const double t_stage = app::now_s();
    enc.stage_all();
    const double t_enc0 = app::now_s();
    const std::vector<float> hidden = enc.run(conv, n_src);
    const double t_enc1 = app::now_s();
    std::printf("enc %lld %lld %s\n", static_cast<long long>(n_src),
                static_cast<long long>(geom.d_model),
                to_hex(hidden.data(), hidden.size() * 4).c_str());
    std::fprintf(stderr,
                 "encoder: %lld layers x %lld rows, %lld dispatches in %.3f s "
                 "(stage %.3f s)\n",
                 static_cast<long long>(geom.enc_layers),
                 static_cast<long long>(es.rows),
                 static_cast<long long>(enc.gemm().n_dispatch), t_enc1 - t_enc0,
                 t_enc0 - t_stage);

    if (skip_decoder) return 0;

    npu::Design dec_design(dev, art + "/gemm_rtp_dec");
    const std::vector<app::StreamEntry> dec_streams =
        load_streams(dec_design, art + "/gemm_rtp_dec");
    std::vector<int64_t> dtiers;
    for (const auto &s : dec_streams) dtiers.push_back(s.batch);
    std::sort(dtiers.begin(), dtiers.end());
    dtiers.erase(std::unique(dtiers.begin(), dtiers.end()), dtiers.end());
    if (step_tier < 0) step_tier = dtiers.front();
    if (prefill_tier < 0) prefill_tier = dtiers.back();

    WhisperDecoder dec(model, dec_design, pool, geom);
    for (int64_t b : dtiers) {
      DecoderTier t;
      t.batch = b;
      const app::StreamEntry &q = find_op(dec_streams, "self_qkv", b);
      t.rows = q.M;
      t.streams.self_qkv = static_cast<size_t>(q.slot);
      t.streams.self_attn_out =
          static_cast<size_t>(find_op(dec_streams, "self_attn_out", b).slot);
      t.streams.cross_q = static_cast<size_t>(find_op(dec_streams, "cross_q", b).slot);
      t.streams.cross_kv =
          static_cast<size_t>(find_op(dec_streams, "cross_kv", b).slot);
      t.streams.cross_attn_out =
          static_cast<size_t>(find_op(dec_streams, "cross_attn_out", b).slot);
      t.streams.ffn_up = static_cast<size_t>(find_op(dec_streams, "ffn_up", b).slot);
      t.streams.ffn_down =
          static_cast<size_t>(find_op(dec_streams, "ffn_down", b).slot);
      dec.add_tier(t);
    }
    dec.set_step_tier(step_tier);
    dec.set_prefill_tier(prefill_tier);
    dec.stage_all();
    // The decoding policy, so the greedy ids this prints are the ones the
    // reference implementation would pick. A container without it is refused.
    load_suppression(dec, model, argv[1]);
    const double t_src0 = app::now_s();
    dec.set_source(hidden, n_src);
    const double t_src1 = app::now_s();
    std::fprintf(stderr,
                 "decoder: step tier %lld rows, prefill tier %lld rows, "
                 "cross K|V for %lld positions in %.3f s\n",
                 static_cast<long long>(step_tier),
                 static_cast<long long>(prefill_tier),
                 static_cast<long long>(n_src), t_src1 - t_src0);

    for (size_t i = 0; i < ids.size(); ++i) {
      std::vector<float> h, lg;
      const int32_t top = dec.step(static_cast<int32_t>(ids[i]),
                                   static_cast<int64_t>(i), &h, &lg);
      std::printf("step %zu %d %lld %s\n", i, static_cast<int>(top),
                  static_cast<long long>(h.size()),
                  to_hex(h.data(), h.size() * 4).c_str());
      std::printf("logits %zu %lld %s\n", i,
                  static_cast<long long>(lg.size()),
                  to_hex(lg.data(), lg.size() * 4).c_str());
    }
    if (!prompt.empty()) {
      // The stop token is looked up BY NAME in the container's own table, not
      // hardcoded: 50257 is what openai/whisper-* happens to use, and a
      // container that renumbered it would otherwise be decoded until it hit
      // max_target_positions -- a long, wrong transcript instead of a refusal.
      const auto tbl = model.raw("tokenizer.whisper_table");
      const npue::WhisperTokenizer tok(tbl.data, tbl.bytes);
      const int32_t stop = tok.token_id("<|endoftext|>");
      const std::vector<int32_t> out = dec.greedy(
          std::vector<int32_t>(prompt.begin(), prompt.end()), stop, max_new);
      std::printf("greedy %zu", out.size());
      for (int32_t id : out) std::printf(" %d", id);
      std::printf("\n");
      const std::string text = tok.decode(out);
      std::printf("text %s\n", to_hex(text.data(), text.size()).c_str());
    }
    std::fprintf(stderr, "decoder: %lld dispatches total\n",
                 static_cast<long long>(dec.gemm().n_dispatch));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
