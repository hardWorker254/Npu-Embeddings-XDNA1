//===- transcribe.cpp -----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one Whisper transcription session. See transcribe.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/transcribe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/design_selection.hpp"  // parse_streams, app::StreamEntry
#include "common/host_kernels.hpp"       // now_s
#include "whisper/audio.hpp"
#include "whisper/features.hpp"

namespace npue::whisper {
namespace {

std::string read_text(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Every stream of a design set, loaded into the slot design.json records, and
// CHECKED against it: a set loaded in the wrong order runs the wrong instruction
// stream for an op and returns a plausible number.
std::vector<app::StreamEntry> load_streams(npu::Design &d,
                                           const std::string &dir) {
  std::vector<app::StreamEntry> streams =
      app::parse_streams(read_text(dir + "/design.json"));
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
  throw std::runtime_error(
      "no " + op + " stream at batch tier " + std::to_string(batch) + " -- " +
      op + " is missing at that tier; re-export the design set (see "
      "tools/npu_targets.json kinds.stt.streams)");
}

std::vector<int64_t> tiers_of(const std::vector<app::StreamEntry> &streams) {
  std::vector<int64_t> t;
  for (const auto &s : streams) t.push_back(s.batch);
  std::sort(t.begin(), t.end());
  t.erase(std::unique(t.begin(), t.end()), t.end());
  return t;
}

// transformers' merge, line for line: tokenization_whisper's
// _find_longest_common_sequence with no timestamps.
//
// Windows overlap, so consecutive windows say the same words twice. This slides
// a match of length i across the boundary -- the last i tokens of the left
// against the first i of the right -- scores it as matches/i + i/10000 (the
// epsilon breaks ties towards the LONGER overlap, which is what keeps a repeated
// "the the the" from being swallowed) and keeps the best one with more than a
// single match. The seam is then cut at the MIDPOINT of that overlap: the left
// half is kept, the right half dropped, and the search continues on the tail of
// the right window. With nothing matching, the midpoint lands at the end of the
// left window and the effect is plain concatenation -- which is the right
// behaviour for windows that share no words at all.
std::vector<int32_t> merge_windows(
    const std::vector<std::vector<int32_t>> &seqs) {
  if (seqs.empty()) return {};
  std::vector<int32_t> total;
  std::vector<int32_t> left = seqs[0];
  for (size_t s = 1; s < seqs.size(); ++s) {
    const std::vector<int32_t> &right = seqs[s];
    const int64_t ll = static_cast<int64_t>(left.size());
    const int64_t rl = static_cast<int64_t>(right.size());
    double best = 0.0;
    int64_t ls = ll, lp = ll, rs = 0, rp = 0;
    for (int64_t i = 1; i < ll + rl; ++i) {
      const int64_t a = std::max<int64_t>(0, ll - i);
      const int64_t b = std::min<int64_t>(ll, ll + rl - i);
      const int64_t c = std::max<int64_t>(0, i - ll);
      const int64_t d = std::min<int64_t>(rl, i);
      if (b - a != d - c) continue;  // unreachable by construction; the
                                     // reference raises here, and a request
                                     // should not die over it
      int64_t matches = 0;
      for (int64_t k = 0; k < b - a; ++k)
        if (left[static_cast<size_t>(a + k)] ==
            right[static_cast<size_t>(c + k)])
          ++matches;
      const double score = static_cast<double>(matches) / static_cast<double>(i) +
                           static_cast<double>(i) / 10000.0;
      if (matches > 1 && score > best) {
        best = score;
        ls = a; lp = b; rs = c; rp = d;
      }
    }
    const int64_t left_mid = (lp + ls) / 2;
    const int64_t right_mid = (rp + rs) / 2;
    total.insert(total.end(), left.begin(),
                 left.begin() + static_cast<long>(left_mid));
    left.assign(right.begin() + static_cast<long>(right_mid), right.end());
  }
  total.insert(total.end(), left.begin(), left.end());
  return total;
}

}  // namespace

std::string resolve_stt_artifacts(const std::string &root,
                                  const std::string &artifacts,
                                  const std::string &model_name) {
  // The same candidate list the embeddings path uses, so `--artifacts <model>`
  // and the per-model runtime/<model>/artifacts_npu<arch> layout both work.
  std::vector<std::string> candidates;
  if (!artifacts.empty()) {
    if (std::filesystem::path(artifacts).is_absolute())
      candidates = {artifacts};
    else
      candidates = app::artifacts_candidates(root, artifacts);
  } else {
    candidates = app::artifacts_candidates(root, model_name);
  }
  std::string looked;
  for (const auto &c : candidates) {
    const bool enc = std::ifstream(c + "/gemm_rtp/design.json").good();
    const bool dec = std::ifstream(c + "/gemm_rtp_dec/design.json").good();
    if (enc && dec) return c;
    if (looked.empty()) looked = c;
    else looked += ", " + c;
    if (enc) looked += " (gemm_rtp only, no gemm_rtp_dec)";
    if (dec) looked += " (gemm_rtp_dec only, no gemm_rtp)";
  }
  throw std::runtime_error(
      "no Whisper design set found; looked at " +
      (looked.empty() ? root : looked) +
      ". A speech-to-text model needs TWO sets: gemm_rtp for the encoder stack "
      "and gemm_rtp_dec for the decoder's seven streams. Export both with "
      "tools/export_gemm_rtp.py for this model and generation, or name one "
      "with --artifacts.");
}

Session::Session(npue::File &model, const std::string &model_name,
                 const std::string &artifacts, int threads)
    : model_(model),
      geom_(read_geometry(model, model_name)),
      art_(artifacts),
      name_(model_name),
      dev_(std::make_unique<npu::Device>()),
      pool_(std::make_unique<app::Pool>(std::max(1, threads))),
      enc_design_(std::make_unique<npu::Design>(*dev_, artifacts + "/gemm_rtp")),
      dec_design_(
          std::make_unique<npu::Design>(*dev_, artifacts + "/gemm_rtp_dec")),
      enc_(model, *enc_design_, *pool_, geom_),
      dec_(model, *dec_design_, *pool_, geom_),
      tok_(nullptr) {
  const std::vector<app::StreamEntry> enc_streams =
      load_streams(*enc_design_, artifacts + "/gemm_rtp");
  const std::vector<app::StreamEntry> dec_streams =
      load_streams(*dec_design_, artifacts + "/gemm_rtp_dec");

  // -- the encoder: one batch tier, walked in chunks of that tier's rows
  const std::vector<int64_t> etiers = tiers_of(enc_streams);
  if (etiers.size() != 1)
    throw std::runtime_error(
        artifacts + "/gemm_rtp has " + std::to_string(etiers.size()) +
        " batch tiers. The encoder walks its 1500 positions in chunks of ONE "
        "tier's row count, so a second tier would mean a second chunking "
        "policy and this build refuses to guess which one it is.");
  EncoderStreams es;
  es.qkv = static_cast<size_t>(find_op(enc_streams, "qkv", etiers[0]).slot);
  es.attn_out =
      static_cast<size_t>(find_op(enc_streams, "attn_out", etiers[0]).slot);
  es.ffn_up = static_cast<size_t>(find_op(enc_streams, "ffn_up", etiers[0]).slot);
  es.ffn_down =
      static_cast<size_t>(find_op(enc_streams, "ffn_down", etiers[0]).slot);
  es.rows = find_op(enc_streams, "qkv", etiers[0]).M;
  enc_.set_streams(es);

  // -- the decoder: every tier, plus which one a step and which one the
  //    cross-attention prefill uses
  const std::vector<int64_t> dtiers = tiers_of(dec_streams);
  for (int64_t b : dtiers) {
    DecoderTier t;
    t.batch = b;
    t.rows = find_op(dec_streams, "self_qkv", b).M;
    t.streams.self_qkv = static_cast<size_t>(find_op(dec_streams, "self_qkv", b).slot);
    t.streams.self_attn_out =
        static_cast<size_t>(find_op(dec_streams, "self_attn_out", b).slot);
    t.streams.cross_q = static_cast<size_t>(find_op(dec_streams, "cross_q", b).slot);
    t.streams.cross_kv = static_cast<size_t>(find_op(dec_streams, "cross_kv", b).slot);
    t.streams.cross_attn_out =
        static_cast<size_t>(find_op(dec_streams, "cross_attn_out", b).slot);
    t.streams.ffn_up = static_cast<size_t>(find_op(dec_streams, "ffn_up", b).slot);
    t.streams.ffn_down =
        static_cast<size_t>(find_op(dec_streams, "ffn_down", b).slot);
    dec_.add_tier(t);
  }
  // A step is one real row, so the SMALLEST tier is the cheapest one that can
  // hold it; the prefill is the only place that touches all 1500 positions, so
  // it runs in the widest. Both are the same trade the gate measures.
  dec_.set_step_tier(dtiers.front());
  dec_.set_prefill_tier(dtiers.back());

  enc_.stage_all();
  dec_.stage_all();
  // The checkpoint's decoding policy, from the container. Refused when absent:
  // see decoder.hpp for why a missing policy is not a default.
  load_suppression(dec_, model_, model_name);
  const auto tbl = model_.raw("tokenizer.whisper_table");
  tok_ = std::make_unique<npue::WhisperTokenizer>(tbl.data, tbl.bytes);
}

void Session::reset() { dec_.reset(); }

std::vector<int32_t> Session::prompt(const TranscribeOptions &o) const {
  // transformers' order, which is not the obvious one: language comes SECOND,
  // before the task token. Primed the other way round the model still answers
  // fluently and the whole transcript is in a different language, so the order
  // is a correctness property and not a style choice.
  const int32_t sot = tok_->token_id("<|startoftranscript|>");
  const std::string lang = "<|" + o.language + "|>";
  const std::string task = "<|" + o.task + "|>";
  const int32_t lang_id = tok_->token_id(lang);
  const int32_t task_id = tok_->token_id(task);
  const int32_t nots = tok_->token_id("<|notimestamps|>");
  if (lang_id < 0)
    throw std::runtime_error(
        "--language " + o.language + ": this container's tokenizer has no " +
        lang + ". A Whisper can only be primed with a language it has a token "
        "for; there is no fallback to English, because a wrong language is a "
        "fluent transcript of the wrong thing.");
  if (task_id < 0)
    throw std::runtime_error("--task " + o.task + ": expected 'transcribe' or "
                             "'translate', which this container has tokens for");
  return {sot, lang_id, task_id, nots};
}

Segment Session::run_window(const std::vector<float> &samples, int64_t start,
                            int64_t end, const TranscribeOptions &o,
                            Transcript &timing) {
  Segment seg;
  seg.start_sample = start;
  seg.end_sample = end;
  seg.start_s = static_cast<double>(start) / kSampleRate;
  seg.end_s = static_cast<double>(end) / kSampleRate;

  const std::vector<float> window(samples.begin() + static_cast<long>(start),
                                  samples.begin() + static_cast<long>(end));
  const double t_mel0 = app::now_s();
  const MelSpec mel = log_mel_30s(window, static_cast<int>(geom_.mel_bins),
                                  nullptr, pool_.get());
  const double t_conv0 = app::now_s();
  timing.mel_seconds += t_conv0 - t_mel0;
  const std::vector<float> conv = conv_front_end(
      mel, model_.raw("frontend.conv1.weight").as<float>(),
      model_.raw("frontend.conv1.bias").as<float>(),
      model_.raw("frontend.conv2.weight").as<float>(),
      model_.raw("frontend.conv2.bias").as<float>(),
      static_cast<int>(geom_.d_model), pool_.get());
  const double t_enc0 = app::now_s();
  timing.conv_seconds += t_enc0 - t_conv0;   // t_conv0 was the mel's end
  const std::vector<float> hidden = enc_.run(conv, kEncoderPositions);
  timing.encoder_seconds += app::now_s() - t_enc0;

  // The cross-attention cache is built once per WINDOW, not once per session:
  // it depends on the encoder output, and a second window is a different audio.
  dec_.set_source(hidden, kEncoderPositions);
  const double t_dec0 = app::now_s();
  seg.ids = dec_.greedy(prompt(o), tok_->token_id("<|endoftext|>"), o.max_new);
  timing.decoder_seconds += app::now_s() - t_dec0;
  seg.text = tok_->decode(seg.ids);
  return seg;
}

Transcript Session::transcribe_samples(const std::vector<float> &samples,
                                       const TranscribeOptions &o) {
  if (samples.empty())
    throw std::runtime_error("transcribe: no audio samples");
  // A window of C seconds is C*sample_rate samples, which the front end turns
  // into C*sample_rate/hop frames and conv2's stride 2 halves -- 50 positions a
  // second at Whisper's 16 kHz / hop 160. The encoder's position count is the
  // MODEL's own (1500 for every shipped size), so a longer window would be cut
  // by the encoder's own guard with nothing to say so at this level. Refused
  // here instead, naming the number: a silently cut window is a transcript of
  // the first 30 s presented as the whole recording.
  const int64_t pos_per_second = kSampleRate / 2 / geom_.hop_length;
  if (o.chunk_seconds <= 0 ||
      static_cast<int64_t>(o.chunk_seconds) * pos_per_second > geom_.max_seq)
    throw std::runtime_error(
        "transcribe: a " + std::to_string(o.chunk_seconds) +
        " s window is " +
        std::to_string(static_cast<int64_t>(o.chunk_seconds) * pos_per_second) +
        " encoder positions, and this container's position table holds " +
        std::to_string(geom_.max_seq) + ".\n"
        "  Re-pack it with the full table -- that is the fix:\n"
        "    python tools/pack_npue.py --model-dir <ckpt> --out <this file> "
        "--device <npu1|npu2> --max-seq " +
        std::to_string(geom_.max_seq) + "\n"
        "  Shrinking --chunk-seconds to " +
        std::to_string(geom_.max_seq / pos_per_second) +
        " would NOT be a fix: the window would cover " +
        std::to_string(geom_.max_seq / pos_per_second) +
        " s of every " + std::to_string(o.chunk_seconds) +
        " s of audio, and the rest would be silence the model would "
        "transcribe as something.");
  if (o.stride_seconds < 0 || o.stride_seconds * 2 > o.chunk_seconds)
    throw std::runtime_error(
        "transcribe: the stride must be at most half the chunk length "
        "(transformers' own rule; two strides span the overlap)");

  reset();
  Transcript t;
  t.language = o.language;
  t.task = o.task;
  t.n_samples = static_cast<int64_t>(samples.size());
  const int64_t chunk = static_cast<int64_t>(o.chunk_seconds) * kSampleRate;
  const int64_t stride = static_cast<int64_t>(o.stride_seconds) * kSampleRate;
  // transformers strides BOTH sides, so consecutive windows start one
  // (chunk - 2*stride) apart, not (chunk - stride): 20 s for the default 30/5.
  const int64_t step = chunk - 2 * stride;
  const int64_t n = static_cast<int64_t>(samples.size());

  std::vector<std::vector<int32_t>> window_ids;
  for (int64_t pos = 0;; pos += step) {
    const int64_t end = std::min(n, pos + chunk);
    Segment seg = run_window(samples, pos, end, o, t);
    window_ids.push_back(seg.ids);
    t.segments.push_back(std::move(seg));
    ++t.n_chunks;
    // The last window covers the tail, so no redundant sliver after it -- the
    // reference's loop keeps stepping until pos >= n and then transcribes a
    // one-second window for nothing.
    if (end >= n) break;
  }
  t.text = tok_->decode(merge_windows(window_ids));
  t.n_dispatches = n_dispatches();
  return t;
}

Transcript Session::transcribe_file(const std::string &path,
                                    const TranscribeOptions &o) {
  const double t0 = app::now_s();
  Audio a = ingest(path, o.convert);
  Transcript t = transcribe_samples(a.samples, o);
  t.audio_seconds = app::now_s() - t0;
  return t;
}

}  // namespace npue::whisper
