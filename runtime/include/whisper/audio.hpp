//===- audio.hpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- audio ingest for speech-to-text: WAV in, 16 kHz mono float
// samples out.
// SPDX-License-Identifier: Apache-2.0
//
// Whisper's weights were trained on 16 kHz mono. Everything here exists to get
// to that, and to fail loudly when it cannot:
//
//   * A 16-bit or 32-bit-float PCM WAV at 16 kHz, mono, is read directly.
//   * Anything else -- another rate, more than one channel, a compressed
//     codec -- is REFUSED with the command that fixes it, not resampled
//     quietly. A resample that is 0.1% off produces a transcript that is
//     wrong in no visible way, which is this project's favourite failure.
//
// The conversion path shells out to ffmpeg THROUGH execve, with an argv and no
// shell, so a path with a space, a quote or a semicolon in it is a filename
// and not an injection. ffmpeg reads the file and writes s16le on stdout; the
// bytes are read with a hard cap and a wall-clock deadline, because the caller
// is an unauthenticated HTTP endpoint and a subprocess that hangs or a file
// that is 40 GB must both be refusals, not an OOM and a stuck request.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npue::whisper {

struct Audio {
  std::vector<float> samples;   // mono, [-1, 1], 16 kHz
  int sample_rate = 0;
  int64_t duration_samples = 0; // before any 30 s chunking
};

struct IngestLimits {
  // 25 minutes of 16 kHz s16le is 48 MB. Long enough for a lecture, short
  // enough that a whole-request body cap is a number rather than a hope.
  size_t max_bytes = 48u * 1024 * 1024;
  int timeout_s = 60;
  std::string ffmpeg = "ffmpeg";
};

// Strict: what read_wav accepts, and it says what to do about the rest.
Audio read_wav(const std::string &path);

// ffmpeg -> 16 kHz mono s16le on a pipe. `path` is passed as ONE argv entry,
// never through a shell.
Audio convert_with_ffmpeg(const std::string &path, const IngestLimits &lim = {});

// read_wav, falling back to ffmpeg when --convert was asked for. `convert` is
// the CLI's flag: false means "a WAV or nothing", true means "ffmpeg first,
// because the caller said the file is not a WAV".
Audio ingest(const std::string &path, bool convert,
             const IngestLimits &lim = {});

}  // namespace npue::whisper
