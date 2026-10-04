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
#include "whisper/attention_npu.hpp"
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
                             "tools/export/export_gemm_rtp.py");
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
      "tools/data/npu_targets.json kinds.stt.streams)");
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
                                  const std::string &model_name,
                                  const std::string &want_layout) {
  // The same candidate list the embeddings path uses, so `--artifacts <model>`
  // and the per-model runtime/artifacts/<model>/artifacts_npu<arch> layout both
  // work.
  std::vector<std::string> candidates;
  if (!artifacts.empty()) {
    if (std::filesystem::path(artifacts).is_absolute())
      candidates = {artifacts};
    else
      candidates = app::artifacts_candidates(root, artifacts);
  } else {
    candidates = app::artifacts_candidates(root, model_name);
  }

  // USABLE MEANS BOTH HALVES, and the layout filter is applied to the ENCODER
  // set because that is the half whose `gemm_rtp` records a hash the container
  // also records: the container's `encoder.layers.0.qkv` layout_hash. The
  // decoder's is not consulted, deliberately -- it is a separate design with its
  // own hash, and the exporter builds both halves in one run from one set of
  // flags, so a set whose encoder matches this container's dtypes was built for
  // the same run. Checking the decoder too would be a second opinion about a
  // file this function already decided exists.
  //
  // Before the filter, this took the first candidate holding both files, which
  // is the bf16 set for every model that has both -- so every int8 Whisper
  // container resolved to the bf16 design of the same name and died at
  // `encoder.layers.0.qkv` on "layout mismatch -- design gemm_rtp wants
  // 4d729172b0d8..., file has f0a4a89650...". The Whisper path had its own
  // resolver and so had none of the checks the embedding path grew.
  auto both_halves = [](const std::string &c) {
    return std::ifstream(c + "/gemm_rtp/design.json").good() &&
           std::ifstream(c + "/gemm_rtp_dec/design.json").good();
  };
  const std::string chosen =
      app::select_set_for_layout(candidates, both_halves, want_layout);
  if (!chosen.empty()) return chosen;

  // Nothing usable, or nothing with this container's layout. The message has to
  // say WHICH, because the two are different problems with different fixes --
  // and the layout case is the one that looks like a working directory and is
  // not.
  std::string looked;
  bool saw_usable = false;
  for (const auto &c : candidates) {
    const bool enc = std::ifstream(c + "/gemm_rtp/design.json").good();
    const bool dec = std::ifstream(c + "/gemm_rtp_dec/design.json").good();
    if (enc && dec) {
      saw_usable = true;
      looked += (looked.empty() ? "" : ", ") + c + " (both halves, but its " +
                "gemm_rtp declares B layout " +
                app::design_b_layout_hash(c).substr(0, 12) + "...)";
      continue;
    }
    if (looked.empty()) looked = c;
    else looked += ", " + c;
    if (enc) looked += " (gemm_rtp only, no gemm_rtp_dec)";
    if (dec) looked += " (gemm_rtp_dec only, no gemm_rtp)";
  }
  if (saw_usable && !want_layout.empty()) {
    throw std::runtime_error(
        "no Whisper design set matches this container's B-operand layout ("
        + want_layout.substr(0, 12) +
        "...). Every complete design set under " + root +
        " was built for a different element type, so the bytes would be the "
        "right size and the wrong order -- looked at " + looked +
        ". Export one for this datapath with "
        "tools/export/export_gemm_rtp.py --target " + model_name +
        " --int8, or run a bf16 container.");
  }
  throw std::runtime_error(
      "no Whisper design set found; looked at " +
      (looked.empty() ? root : looked) +
      ". A speech-to-text model needs TWO sets: gemm_rtp for the encoder stack "
      "and gemm_rtp_dec for the decoder's seven streams. Export both with "
      "tools/export/export_gemm_rtp.py for this model and generation, or name one "
      "with --artifacts.");
}

Session::Session(npue::File &model, const std::string &model_name,
                 const std::string &artifacts, int threads,
                 const std::set<std::string> &npu_ops)
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
  // -- the elementwise designs, BEFORE the stacks are asked to stage anything.
  // Each is one xclbin and one hw_context, and the npu1 driver allows six of
  // those, so this is where a run that asked for too many has to be told.
  // maybe_stt_mode() counts them and checks the budget before constructing this
  // session at all; here each one is opened by name and CHECKED against the
  // container, because an eltwise design carries the model's own width and
  // epsilon compiled into its kernel.
  auto open_elt = [&](const std::string &code, const std::string &dir,
                      EltwiseKind kind, std::unique_ptr<npu::Design> &design,
                      std::unique_ptr<NpuEltwise> &op) {
    if (!npu_ops.count(code)) return;
    design = std::make_unique<npu::Design>(*dev_, artifacts + "/" + dir);
    op = std::make_unique<NpuEltwise>(*design, *pool_, kind);
    op->alloc_buffers();
    // The WIDTH check is LayerNorm's alone, and it is per op because the three
    // designs have three different meanings for a row: a LayerNorm row is
    // d_model wide, a softmax row is an attention score row (n_kv, checked
    // where the scores are handed to it), and a GELU "row" is a flat span of
    // activations the runtime walks in chunks. Checking all three against
    // d_model would refuse two designs that are exactly right.
    if (kind == EltwiseKind::LayerNorm && op->cols() != geom_.d_model)
      throw std::runtime_error(
          dir + "/design.json has rows " + std::to_string(op->cols()) +
          " columns wide, and this container's d_model is " +
          std::to_string(geom_.d_model) +
          ". The kernel's row width is compiled in, so this design normalises "
          "the wrong number of channels. Re-export it for this model: "
          "python tools/export/export_gemm_rtp.py --target " + name_ +
          " --arch 1 --npu-ops " + code);
    if (kind == EltwiseKind::LayerNorm) {
      const double want = geom_.ln_eps;
      const double got = design->info().ln_eps;
      if (got <= 0.0 || std::abs(got - want) > 1e-12 * std::max(1.0, want))
        throw std::runtime_error(
            dir + "/design.json was built with layer_norm_eps " +
            app::eps_text(got) + " and this container says " +
            app::eps_text(want) +
            ". The epsilon is inside a square root, so the two are not a "
            "rounding difference. Re-export the design for this container.");
      elt_notes_.push_back(dir + ": " + std::to_string(op->rows()) + " rows x " +
                           std::to_string(op->cols()) + ", eps " +
                           app::eps_text(got));
    } else if (kind == EltwiseKind::Gelu) {
      // One flat span of activations, so the pair that reads as "1 rows x N" is
      // spelled as what it is: the elements one dispatch covers.
      elt_notes_.push_back(dir + ": " +
                           std::to_string(op->rows() * op->cols()) +
                           " elements per dispatch");
    } else {
      elt_notes_.push_back(dir + ": " + std::to_string(op->rows()) + " rows x " +
                           std::to_string(op->cols()));
    }
  };
  open_elt("layn", "layernorm", EltwiseKind::LayerNorm, ln_design_, ln_);
  open_elt("softm", "softmax", EltwiseKind::Softmax, softm_design_, softm_);
  open_elt("gelu", "gelu", EltwiseKind::Gelu, gelu_design_, gelu_);

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
  // The status line's schedule, read off the loaded set rather than written
  // down: a printed table that lists four encoder streams when the export
  // carries five describes a design that is not the one loaded, which is worse
  // than printing nothing.
  enc_rows_ = es.rows;
  for (const auto &s : enc_streams) enc_ops_.push_back(s.op);
  // The decoder set repeats its seven streams once per batch tier, and the
  // status line prints the tier row counts separately, so the NAMES go in once:
  // printing seven names three times reads as twenty-one GEMMs per layer.
  for (const auto &s : dec_streams) {
    bool seen = false;
    for (const auto &have : dec_ops_) seen = seen || have == s.op;
    if (!seen) dec_ops_.push_back(s.op);
  }
  // The stacks take their LayerNorm (and GELU) from the array only after the
  // designs are open, so a stack can never dispatch into a design that does not
  // exist. Both stacks share the sessions' pool and the designs' single
  // dispatch window: an STT request is serial, so the lock is not taken.
  if (ln_) {
    enc_.set_layernorm(ln_.get());
    dec_.set_layernorm(ln_.get());
  }
  if (gelu_) {
    enc_.set_gelu(gelu_.get());
    dec_.set_gelu(gelu_.get());
  }

  // -- attention as two GEMMs, on the two sets' own attn_qk / attn_av streams.
  // A set that does not carry them was exported without --npu-ops attn,
  // and asking for the array on it is refused by name rather than answered from
  // the host -- the flag was in the command and nothing happened.
  auto open_attn = [&](const std::string &code, const std::string &dir,
                       npu::Design &design,
                       const std::vector<app::StreamEntry> &streams,
                       int64_t batch, int64_t rows_here,
                       std::unique_ptr<NpuAttention> &out) {
    if (!npu_ops.count(code)) return;
    const app::StreamEntry *qk = nullptr;
    const app::StreamEntry *av = nullptr;
    for (const auto &s : streams) {
      if (s.batch != batch) continue;
      if (s.op == "attn_qk" && !qk) qk = &s;
      if (s.op == "attn_av" && !av) av = &s;
    }
    if (!qk || !av)
      throw std::runtime_error(
          dir + " has no attn_qk/attn_av streams at batch tier " +
          std::to_string(batch) +
          ", so --npu-ops attn cannot run attention on the array here. "
          "Re-export this model with the code in the list: python "
          "tools/export/export_gemm_rtp.py --target " + name_ + " --arch 1 "
          "--npu-ops attn");
    if (qk->M != rows_here)
      throw std::runtime_error(
          dir + ": the attn streams compute " + std::to_string(qk->M) +
          " rows per dispatch and the stack walks " + std::to_string(rows_here) +
          ". The two have to be the same number -- the score chunk one dispatch "
          "produces IS the query chunk the next GEMM reads. Re-export the set.");
    if (qk->N != av->K)
      throw std::runtime_error(
          dir + ": attn_qk's N is " + std::to_string(qk->N) + " and attn_av's K "
          "is " + std::to_string(av->K) +
          ". The score chunk travels from one to the other as the A operand, so "
          "the two numbers are the same padded n_kv and a set that says "
          "otherwise is not one of ours.");
    if (qk->K != geom_.head_dim)
      throw std::runtime_error(
          dir + ": attn_qk's K is " + std::to_string(qk->K) +
          " and this container's head_dim is " + std::to_string(geom_.head_dim) +
          ". The Q operand of a score is one head, so that K is the head width "
          "and nothing else rounds it.");
    if (av->N < geom_.head_dim)
      throw std::runtime_error(
          dir + ": attn_av's N is " + std::to_string(av->N) +
          " and a head is " + std::to_string(geom_.head_dim) +
          " wide. The design pads this one UP to its own N granularity, never "
          "down to a head.");
    if (softm_ && softm_->cols() != qk->N)
      throw std::runtime_error(
          dir + "/softmax has rows " + std::to_string(softm_->cols()) +
          " wide and the attn streams' score row is " + std::to_string(qk->N) +
          ". The softmax design has to be the width of the score row it is "
          "handed, because it reduces along the whole row.");
    out = std::make_unique<NpuAttention>(design, *pool_, qk->N, geom_.head_dim,
                                        av->N);
    out->set_streams(static_cast<size_t>(qk->slot),
                     static_cast<size_t>(av->slot), qk->M);
    out->set_softmax(softm_.get());
    out->alloc_buffers();
  };
  open_attn("attn", artifacts + "/gemm_rtp", *enc_design_, enc_streams,
            etiers[0], es.rows, enc_attn_);
  // -- the mel filter bank as a GEMM --------------------------------------
  // A stream of its own in the same set, and the bank is a weight, so this one
  // stages once at session start like any other operand. A set without the
  // stream is a refusal for the same reason as the attention's.
  if (npu_ops.count("mproj")) {
    const app::StreamEntry *mp = nullptr;
    for (const auto &st : enc_streams)
      if (st.op == "mel_proj" && st.batch == etiers[0]) mp = &st;
    if (!mp)
      throw std::runtime_error(
          artifacts + "/gemm_rtp has no mel_proj stream at batch tier " +
          std::to_string(etiers[0]) +
          ", so --npu-ops mproj cannot run the mel bank on the array "
          "here. Re-export this model with the code in the list: python "
          "tools/export/export_gemm_rtp.py --target " + name_ +
          " --arch 1 --npu-ops mproj");
    if (mp->K < kMelBins || mp->N < geom_.mel_bins)
      throw std::runtime_error(
          artifacts + "/gemm_rtp: mel_proj is " + std::to_string(mp->K) + "x" +
          std::to_string(mp->N) + " and this front end's bank is " +
          std::to_string(kMelBins) + "x" + std::to_string(geom_.mel_bins) +
          ". A design narrower than the tensor it multiplies would drop the top "
          "frequency bins and the top mel bands, which is silence, not "
          "rounding.");
    mel_proj_ = std::make_unique<NpuMelProj>(*enc_design_, *pool_, kMelBins,
                                              geom_.mel_bins);
    mel_proj_->set_streams(static_cast<size_t>(mp->slot), mp->M);
    mel_proj_->alloc_buffers(mel_filter_bank(static_cast<int>(geom_.mel_bins)));
    mel_note_ = mel_proj_->note(kMelFrames);
  }

  // -- the 400-point transform as a GEMM ----------------------------------
  // Same shape of argument as the mel bank: a stream in the same set, a weight
  // staged once, and a refusal by name when the set does not carry the stream.
  if (npu_ops.count("fft")) {
    const app::StreamEntry *df = nullptr;
    for (const auto &st : enc_streams)
      if (st.op == "dft400" && st.batch == etiers[0]) df = &st;
    if (!df)
      throw std::runtime_error(
          artifacts + "/gemm_rtp has no dft400 stream at batch tier " +
          std::to_string(etiers[0]) +
          ", so --npu-ops fft cannot run the transform on the array "
          "here. Re-export this model with the code in the list: python "
          "tools/export/export_gemm_rtp.py --target " + name_ +
          " --arch 1 --npu-ops fft");
    if (df->K < kNfft)
      throw std::runtime_error(
          artifacts + "/gemm_rtp: dft400's K is " + std::to_string(df->K) +
          " and a frame is " + std::to_string(kNfft) +
          " samples. A narrower operand would transform a truncated frame, "
          "which is a different spectrum.");
    fft_ = std::make_unique<NpuFft>(*enc_design_, *pool_, kNfft, kMelBins);
    fft_->set_streams(static_cast<size_t>(df->slot), df->M);
    fft_->alloc_buffers();
    fft_note_ = fft_->note(kMelFrames);
  }

  if (enc_attn_) {
    enc_.set_attention(enc_attn_.get());
    attn_note_ = enc_attn_->note(geom_.max_seq, geom_.heads, geom_.enc_layers);
  }
  {
    std::vector<int64_t> rows;
    for (int64_t b : tiers_of(dec_streams))
      rows.push_back(find_op(dec_streams, "self_qkv", b).M);
    dec_tier_rows_ = rows;
  }

  // -- the audio front end, on the encoder set's OWN [rows, d, d] stream
  //
  // attn_out is a context-by-out_proj GEMM of exactly the shape a convolution
  // needs -- M rows, K = d_model, N = d_model -- so conv1 and conv2 run on its
  // instruction stream with no new export. A set that does not have one is not
  // this architecture's: the front end then stays on the host, and a request
  // that asked for the array is refused by name rather than answered from the
  // host (see maybe_stt_mode).
  conv_note_ = "this design set has no [rows, d, d] stream to run them on";
  {
    const app::StreamEntry *gemm_stream = nullptr;
    for (const auto &s : enc_streams)
      if (s.K == geom_.d_model && s.N == geom_.d_model) {
        gemm_stream = &s;
        break;
      }
    // An int8 design has the [rows, d, d] stream -- attn_out is a GEMM of the
    // right shape whatever the operand type -- and still cannot run the front
    // end on it. NpuConv1d::stage() builds this operand's B panel from the
    // container's F32 conv weights through a bf16 tiler, and there are no
    // .wscale/.asmooth sidecars to quantise them against; an int8 panel is I8
    // with its own MAC sub-tile, so that tiler would produce the right byte
    // count of the wrong element type in the wrong order. Rather than let
    // stage() refuse -- which it must, since nothing above it can catch a
    // mis-tiled panel -- the stream is not offered here, and conv_available()
    // answers false so a request that asked for the array is refused by name.
    const bool int8_enc = enc_design_->info().a_elem_bytes != 2;
    if (gemm_stream && int8_enc)
      conv_note_ =
          "an int8 design's conv panel cannot be tiled from fp32 conv weights; "
          "the front end runs on the host";
    if (gemm_stream && !int8_enc) {
      conv_gemm_ = std::make_unique<NpuGemm>(*enc_design_, *pool_);
      conv_gemm_->alloc_buffers();
      conv1_ = std::make_unique<NpuConv1d>(
          *enc_design_, *pool_, *conv_gemm_,
          static_cast<size_t>(gemm_stream->slot), gemm_stream->M,
          gemm_stream->K, gemm_stream->N);
      conv2_ = std::make_unique<NpuConv1d>(
          *enc_design_, *pool_, *conv_gemm_,
          static_cast<size_t>(gemm_stream->slot), gemm_stream->M,
          gemm_stream->K, gemm_stream->N);
      conv1_->stage("frontend.conv1",
                    model_.raw("frontend.conv1.weight").as<float>(),
                    model_.raw("frontend.conv1.bias").as<float>(),
                    geom_.mel_bins, geom_.d_model, 3);
      conv2_->stage("frontend.conv2",
                    model_.raw("frontend.conv2.weight").as<float>(),
                    model_.raw("frontend.conv2.bias").as<float>(),
                    geom_.d_model, geom_.d_model, 3);
      // Per WINDOW, not "so far": the status line prints before the first
      // transcription, so a running counter reads as zero and says nothing
      // about what a request will cost. This one is a fact about the geometry.
      const int64_t rows = gemm_stream->M;
      const int64_t conv1_disp =
          (kMelFrames + rows - 1) / rows * conv1_->k_blocks();
      const int64_t conv2_disp =
          (kEncoderPositions + rows - 1) / rows * conv2_->k_blocks();
      conv_note_ = gemm_stream->op + " stream, K = N = " +
                   std::to_string(gemm_stream->K) + ", M = " +
                   std::to_string(rows) + ", " +
                   std::to_string(conv1_->k_blocks()) + "+" +
                   std::to_string(conv2_->k_blocks()) + " K blocks, " +
                   std::to_string(conv1_disp + conv2_disp) +
                   " dispatches per window";    }
  }

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
  // The decoder attends only at the STEP tier: its cross-attention prefill is a
  // GEMM over the encoder output, and a step is one query row, so the tier the
  // step uses is the only one whose attn streams are ever dispatched.
  {
    const app::StreamEntry *sq = nullptr;
    for (const auto &s : dec_streams)
      if (s.op == "self_qkv" && s.batch == dtiers.front()) sq = &s;
    if (sq)
      open_attn("attn", artifacts + "/gemm_rtp_dec", *dec_design_, dec_streams,
                dtiers.front(), sq->M, dec_attn_);
  }
  if (dec_attn_) dec_.set_attention(dec_attn_.get());

  // -- the vocabulary projection, in chunks, on the decoder set -------------
  // At the STEP tier, which is the only place the projection runs. The chunk
  // count and width come from the set's own logits_i streams, and the container's
  // vocab_size has to FIT them: a container with more ids than the streams cover
  // would silently lose the tail of the vocabulary, which is a plausible-looking
  // transcript of a truncated model.
  if (npu_ops.count("logit")) {
    const app::StreamEntry *first = nullptr;
    for (const auto &st : dec_streams)
      if (st.batch == dtiers.front() && st.op.rfind("logits_", 0) == 0) {
        if (!first || st.slot < first->slot) first = &st;
      }
    if (!first)
      throw std::runtime_error(
          artifacts + "/gemm_rtp_dec has no logits_i streams at batch tier " +
          std::to_string(dtiers.front()) +
          ", so --npu-ops logit cannot run the projection on the array. "
          "Re-export this model with the code in the list: python "
          "tools/export/export_gemm_rtp.py --target " + name_ +
          " --arch 1 --npu-ops logit");
    std::vector<size_t> lslots;
    int64_t chunk_n = 0;
    for (const auto &st : dec_streams) {
      if (st.batch != dtiers.front() || st.op.rfind("logits_", 0) != 0) continue;
      if (st.K != geom_.d_model)
        throw std::runtime_error(
            artifacts + "/gemm_rtp_dec: a logits stream's K is " +
            std::to_string(st.K) + " and d_model is " +
            std::to_string(geom_.d_model));
      if (chunk_n && st.N != chunk_n)
        throw std::runtime_error(
            artifacts + "/gemm_rtp_dec: the logits streams are " +
            std::to_string(chunk_n) + " and " + std::to_string(st.N) +
            " columns wide. The vocabulary is cut into equal chunks, and a set "
            "whose chunks differ is not one of ours.");
      chunk_n = st.N;
      lslots.push_back(static_cast<size_t>(st.slot));
    }
    std::sort(lslots.begin(), lslots.end());
    const int64_t cover = static_cast<int64_t>(lslots.size()) * chunk_n;
    if (cover < geom_.vocab)
      throw std::runtime_error(
          artifacts + "/gemm_rtp_dec: " + std::to_string(lslots.size()) +
          " chunks of " + std::to_string(chunk_n) + " cover " +
          std::to_string(cover) + " ids and this container's vocabulary is " +
          std::to_string(geom_.vocab) +
          ". The last ids would have no operand and the argmax would never see "
          "them. Re-export the set for this container.");
    logit_ = std::make_unique<NpuLogits>(*dec_design_, *pool_, geom_.d_model,
                                         geom_.vocab,
                                         static_cast<int>(lslots.size()));
    logit_->set_design(dec_design_.get());
    logit_->set_streams(lslots, first->M, chunk_n);
    logit_->alloc_buffers(model_.raw("decoder.embed_tokens").as<float>());
    logit_note_ = logit_->note();
    dec_.set_logits(logit_.get());
  }

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
  // The front end in three steps, because the middle one can be a GEMM: the
  // power spectrum, then the mel bank, then the floor. With no mel_proj stream
  // the bank is the host's and the three steps are the same arithmetic in the
  // same order -- the split is a scheduling seam, not a second implementation.
  MelSpec mel;
  if (mel_proj_ || fft_) {
    // Two of the three front-end steps on the array, in the order they depend on
    // each other: the transform produces the power spectrum, the bank projects
    // it. Either one alone is a valid request, and with neither the whole thing
    // is the host's -- the same arithmetic in the same order either way.
    std::vector<float> power;
    if (fft_) {
      const std::vector<float> fr =
          windowed_frames_30s(window, static_cast<int>(geom_.mel_bins), nullptr,
                              pool_.get());
      power.assign(static_cast<size_t>(kMelBins) * kMelFrames, 0.0f);
      fft_->run(fr, kMelFrames, power.data());
    } else {
      power = power_30s(window, static_cast<int>(geom_.mel_bins), nullptr,
                        pool_.get());
    }
    std::vector<float> projected;
    if (mel_proj_) {
      mel.n_mels = static_cast<int>(geom_.mel_bins);
      mel.frames = kMelFrames;
      mel.data.assign(static_cast<size_t>(mel.n_mels) * kMelFrames, 0.0f);
      mel_proj_->run(power, kMelFrames, mel.data.data());
      // The GEMM produces the bank product; the log is a per-element map and
      // stays here. Without this line the tensor reaching mel_floor_scale is a
      // projection (values of 10^1, not 10^0) and the floor is applied to the
      // wrong scale -- a plausible-looking spectrogram off by a factor of ten in
      // the log, which the encoder absorbs as a shifted input.
      mel_log_inplace(mel.data, pool_.get());
    } else {
      project_and_log(power, static_cast<int>(geom_.mel_bins), pool_.get(),
                      projected);
      mel.n_mels = static_cast<int>(geom_.mel_bins);
      mel.frames = kMelFrames;
      mel.data = std::move(projected);
    }
    mel_floor_scale(mel.data);
  } else {
    mel = log_mel_30s(window, static_cast<int>(geom_.mel_bins), nullptr,
                      pool_.get());
  }
  const double t_conv0 = app::now_s();
  timing.mel_seconds += t_conv0 - t_mel0;
  const std::vector<float> conv =
      (o.conv_npu && conv1_)
          ? conv_front_end_npu(mel, *conv1_, *conv2_,
                               static_cast<int>(geom_.d_model))
          : conv_front_end(
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
        "    python tools/pack/pack_npue.py --model-dir <ckpt> --out <this file> "
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
