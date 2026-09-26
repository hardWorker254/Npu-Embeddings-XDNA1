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
#include <string>
#include <vector>

#include "runtime/design.hpp"
#include "runtime/device.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "tokenizers/whisper.hpp"
#include "whisper/decoder.hpp"
#include "whisper/encoder.hpp"
#include "whisper/features.hpp"   // kSampleRate, the one audio constant callers need
#include "whisper/geometry.hpp"

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
                                  const std::string &model_name);

class Session {
public:
  // `model_name` is the container's name, for the status line and the endpoint
  // id; `artifacts` is the design-set root, already resolved by
  // resolve_stt_artifacts.
  Session(npue::File &model, const std::string &model_name,
          const std::string &artifacts, int threads);

  const Geometry &geometry() const { return geom_; }
  const std::string &name() const { return name_; }
  const std::string &artifacts() const { return art_; }
  const npue::WhisperTokenizer &tokenizer() const { return *tok_; }
  int64_t n_dispatches() const { return enc_.gemm().n_dispatch +
                                       dec_.gemm().n_dispatch; }

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
    return std::string(i.emulate_bfp16 ? "bfp16-emulated" : "bf16") +
           " MMAC, C as " + c;
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
  std::unique_ptr<npue::WhisperTokenizer> tok_;
  // One window: the front end, the encoder, then a greedy decode. `seg` is
  // filled in; the returned ids are what the merge consumes.
  Segment run_window(const std::vector<float> &samples, int64_t start,
                     int64_t end, const TranscribeOptions &o, Transcript &timing);
  std::vector<int32_t> prompt(const TranscribeOptions &o) const;
};

}  // namespace npue::whisper
