//===- transcribe.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- one Whisper transcription session: the device, the two design
// sets, the staged weights, the encoder, the decoder, the tokenizer and the
// long-form schedule.
//
// WHY A SESSION AND NOT A FUNCTION
// -------------------------------
// Every one of those is expensive and every one of them outlives a single
// request: two hw_contexts and an xclbin each, ~120 MB of staged weights, a
// 1.5 MB tokenizer table. A per-request design would make a 30 s transcription
// mostly setup. So the session is built once, transcribes many times, and
// `reset()` between them -- the KV cache belongs to the request, not to the
// session.
//
// REQUESTS ARE SERIAL
// -------------------
// The decoder keeps one KV cache and one "how many tokens have I generated"
// counter, so two concurrent transcriptions in one process would interleave
// into each other's answer. The server handles one request at a time and the
// CLI calls it once, so there is no lock here beyond the window NpuGemm already
// takes around bind/dispatch. A future concurrent path needs one session per
// lane, exactly as the BERT path's lanes do.
//
// THE LONG-FORM SCHEDULE IS HUGGINGFACE'S, NOT A GUESS
// ---------------------------------------------------
// 30 s windows with a 5 s stride on EACH side, so consecutive windows start 20 s
// apart, and the merge is the same longest-common-sequence over token ids that
// transformers' own non-timestamp path uses. Both are spelled out in
// transcribe.cpp. The alternative -- concatenating the texts -- duplicates every
// word that falls in the overlap, and "our long-form transcript differs from
// every reference implementation" is not a defensible position for an endpoint
// whose whole job is to return what the model said.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/device.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "tokenizers/whisper.hpp"
#include "whisper/conv1d.hpp"
#include "whisper/decoder.hpp"
#include "whisper/eltwise.hpp"
#include "whisper/fft_npu.hpp"
#include "whisper/encoder.hpp"
#include "whisper/features.hpp"   // kSampleRate, the one audio constant callers need
#include "whisper/geometry.hpp"
#include "whisper/logits_npu.hpp"
#include "whisper/mel_proj.hpp"

namespace npue::whisper {

struct TranscribeOptions {
  // The language is NAMED, never detected: this build has no language detector,
  // and a guessed language produces a fluent transcript of the wrong language.
  // The caller is told which language was used (see Transcript::language), so an
  // assumption is visible rather than silent.
  std::string language = "en";
  std::string task = "transcribe";   // or "translate"
  // Ingest through ffmpeg instead of the WAV reader. Same flag, same reasons as
  // the audio front end: the WAV reader refuses what it cannot represent, and
  // this is how those files get in.
  bool convert = false;
  // Cap on GENERATED tokens per window. 0 = the checkpoint's own position bound.
  int64_t max_new = 0;
  int chunk_seconds = 30;            // the model's own window; see the header
  int stride_seconds = 5;            // HF's default for a 30 s window
  // conv1/conv2 on the NPU or on the host, from --npu-extra-ops conv. Off unless it
  // is asked for, because an op is on the host exactly when the flag does not
  // name it; the host path is also the fp32 reference
  // tools/verify/verify_whisper_features.py holds the NPU path against.
  bool conv_npu = false;
};

// One window's own output. `text` is that window alone, so the overlap is
// visible in the transcript instead of only in the merge; `text` of the whole
// request is the merged one.
struct Segment {
  int64_t start_sample = 0, end_sample = 0;
  double start_s = 0.0, end_s = 0.0;
  std::vector<int32_t> ids;
  std::string text;
};

struct Transcript {
  std::string text;                 // merged, decoded once
  std::vector<Segment> segments;
  std::string language, task;
  int64_t n_samples = 0;
  int64_t n_chunks = 0;
  // Where the time went, per request. Audio and mel are host work; enc/dec are
  // the NPU. A 30 s window is 1.3 s of convolution for whisper-tiny, so the
  // front end is not a rounding error and the split is worth printing.
  double audio_seconds = 0.0, mel_seconds = 0.0, conv_seconds = 0.0;
  double encoder_seconds = 0.0, decoder_seconds = 0.0;
  int64_t n_dispatches = 0;
};

// The design set this architecture needs: <artifacts>/gemm_rtp for the encoder
// and <artifacts>/gemm_rtp_dec for the decoder. Both, or the session refuses:
// a Whisper request that silently transcribes from the encoder stack alone is
// not a thing that can be half-done.
std::string resolve_stt_artifacts(const std::string &root,
                                  const std::string &artifacts,
                                  const std::string &model_name,
                                  const std::string &want_layout = "");

class Session {
public:
  // `model_name` is the container's name, for the status line and the endpoint
  // id; `artifacts` is the design-set root, already resolved by
  // resolve_stt_artifacts.
  //
  // `npu_ops` is the set of op codes from --npu-extra-ops, and it is taken HERE
  // rather than per request because it decides which xclbins exist: each element
  // op is its own design directory and its own hw_context, and a design that is
  // not built cannot be opened later. Every code that names a directory is
  // opened by name, and a directory that is missing is a refusal, not a
  // fallback to the host -- the flag was in the command and quietly doing
  // nothing is the failure this project treats as worst.
  Session(npue::File &model, const std::string &model_name,
          const std::string &artifacts, int threads,
          const std::set<std::string> &npu_ops = std::set<std::string>());

  const Geometry &geometry() const { return geom_; }
  const std::string &name() const { return name_; }
  const std::string &artifacts() const { return art_; }
  const npue::WhisperTokenizer &tokenizer() const { return *tok_; }
  int64_t n_dispatches() const { return enc_.gemm().n_dispatch +
                                       dec_.gemm().n_dispatch +
                                       (conv_gemm_ ? conv_gemm_->n_dispatch : 0) +
                                       elt_dispatch();
  }
  // Dispatches spent on the elementwise designs, so the request total counts
  // every op that ran on the array and not only the GEMM ones.
  int64_t elt_dispatch() const {
    return (ln_ ? ln_->n_dispatch : 0) + (softm_ ? softm_->n_dispatch : 0) +
           (gelu_ ? gelu_->n_dispatch : 0) +
           (enc_attn_ ? enc_attn_->n_dispatch : 0) +
           (dec_attn_ ? dec_attn_->n_dispatch : 0) +
           (mel_proj_ ? mel_proj_->n_dispatch : 0) +
           (fft_ ? fft_->n_dispatch : 0) + (logit_ ? logit_->n_dispatch : 0);
  }
  // What the array attention costs, per window, from the geometry rather than
  // from a counter that reads zero before the first request.
  const std::string &attn_note() const { return attn_note_; }
  const std::string &mel_note() const { return mel_note_; }
  const std::string &fft_note() const { return fft_note_; }
  const std::string &logit_note() const { return logit_note_; }
  // Whether an op is on the array in THIS session: the flag was given and the
  // design is loaded, which is the only question the status line may answer.
  bool on_array(const std::string &code) const {
    if (code == "layn") return ln_ != nullptr;
    if (code == "softm") return softm_ != nullptr;
    if (code == "gelu") return gelu_ != nullptr;
    if (code == "conv") return conv1_ != nullptr;
    if (code == "attn") return enc_attn_ != nullptr || dec_attn_ != nullptr;
    if (code == "mproj") return mel_proj_ != nullptr;
    if (code == "fft") return fft_ != nullptr;
    if (code == "logit") return logit_ != nullptr;
    return false;
  }
  // One line per eltwise design for the status block: the directory, its row
  // capacity and width, and the epsilon compiled into a LayerNorm. Empty when
  // the op is on the host, which is what "host" rows in the table say.
  const std::vector<std::string> &elt_notes() const { return elt_notes_; }
  // Whether this design set has a [rows, d, d] stream to run the convolutions
  // on. False means the front end is the host's, and a request that asked for
  // the array is refused rather than answered from the host.
  bool conv_available() const { return conv1_ != nullptr; }
  // What the array path WOULD be, for the status line and for the refusal when
  // it is asked for and cannot be had: the stream it borrows, its K and N, its
  // K-block split and the dispatches a window costs. "npu" alone would hide
  // that this shares the encoder's attn_out instruction stream rather than
  // owning one, and a running counter would read as zero before the first
  // request. Says why it is unavailable when there is no such stream.
  const std::string &conv_device() const { return conv_note_; }
  const NpuGemm *conv_gemm() const { return conv_gemm_.get(); }
  // The stream names each design set actually loaded, in slot order, for the
  // status line. Read off the loaded design set rather than off a hardcoded
  // list, so the printed schedule cannot describe an export that is not the one
  // loaded: a set with a different stream set would print a table that lies.
  const std::vector<std::string> &enc_stream_ops() const { return enc_ops_; }
  const std::vector<std::string> &dec_stream_ops() const { return dec_ops_; }
  // Rows per dispatch, per set, and the decoder's tiers. The status line prints
  // these because "M=512" is the difference between one dispatch and four.
  int64_t enc_rows() const { return enc_rows_; }
  const std::vector<int64_t> &dec_tier_rows() const { return dec_tier_rows_; }

  // WHICH DATAPATH THIS DESIGN ACTUALLY USES, read off the loaded design and
  // not off npu_targets.json's pin. The two can differ on purpose -- a design
  // exported with --datapath bf16 drops the bfloat16 emulation the targets pin
  // -- and a status line that reported the pin would be reporting an intention.
  // "UNRECORDED" when the design predates the field, never a guess.
  std::string datapath_note() const {
    const auto &i = enc_design_->info();
    const std::string c = i.c_elem_bytes == 2 ? "bf16"
                                              : (i.c_elem_bytes == 4 ? "fp32"
                                                                    : "?");
    if (!i.datapath_recorded)
      return "UNRECORDED (design predates the field), C as " + c;
    return std::string(i.datapath_name()) + " MMAC, C as " + c;
  }

  Transcript transcribe_file(const std::string &path,
                             const TranscribeOptions &o);
  Transcript transcribe_samples(const std::vector<float> &samples,
                                const TranscribeOptions &o);

private:
  // Forget the KV cache and the first-step suppression of the last request.
  void reset();

  npue::File &model_;
  Geometry geom_;
  std::string art_, name_;
  // Declaration order is construction order, and it matters: the encoder and
  // the decoder hold REFERENCES to a Design and a Pool, so the device has to
  // exist before the designs and the designs before the stacks. Constructing
  // them as unique_ptrs and default-constructing the stacks would mean null
  // references for a moment; this way there is no such moment.
  std::unique_ptr<npu::Device> dev_;
  std::unique_ptr<app::Pool> pool_;
  std::unique_ptr<npu::Design> enc_design_, dec_design_;
  WhisperEncoder enc_;
  WhisperDecoder dec_;
  // The audio front end's own GEMM, its A/C buffers and the two convolutions
  // staged on them. Null when the design set has no [rows, d, d] stream to run
  // them on, and then the front end is the host's.
  std::unique_ptr<NpuGemm> conv_gemm_;
  std::unique_ptr<NpuConv1d> conv1_, conv2_;
  std::string conv_note_;
  // The elementwise designs this session opened, one xclbin each, plus what
  // each is for the status line. Null means the op is on the host.
  std::unique_ptr<npu::Design> ln_design_, softm_design_, gelu_design_;
  std::unique_ptr<NpuEltwise> ln_, softm_, gelu_;
  // Attention as two GEMMs, one per design set, built from that set's attn_qk
  // and attn_av streams. Null means the host pass.
  std::unique_ptr<NpuAttention> enc_attn_, dec_attn_;
  // The mel filter bank as a GEMM, on the encoder set's mel_proj stream. Null
  // means the front end's projection is the host's, which is what the empty
  // --npu-extra-ops list means.
  std::unique_ptr<NpuMelProj> mel_proj_;
  std::string mel_note_;
  // The 400-point transform as a GEMM against the twiddle matrix. Null means the
  // front end's transform is the host's mixed-radix one.
  std::unique_ptr<NpuFft> fft_;
  std::string fft_note_;
  // The vocabulary projection as a GEMM, in chunks. Null is the host's fp32
  // matvec, which is the faster one and the default.
  std::unique_ptr<NpuLogits> logit_;
  std::string logit_note_;
  std::vector<std::string> elt_notes_;
  std::string attn_note_;
  // What each loaded design set actually carries, for the status line.
  std::vector<std::string> enc_ops_, dec_ops_;
  int64_t enc_rows_ = 0;
  std::vector<int64_t> dec_tier_rows_;
  std::unique_ptr<npue::WhisperTokenizer> tok_;
  // One window: the front end, the encoder, then a greedy decode. `seg` is
  // filled in; the returned ids are what the merge consumes.
  Segment run_window(const std::vector<float> &samples, int64_t start,
                     int64_t end, const TranscribeOptions &o, Transcript &timing);
  std::vector<int32_t> prompt(const TranscribeOptions &o) const;
};

}  // namespace npue::whisper
