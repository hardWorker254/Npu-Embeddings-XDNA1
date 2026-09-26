//===- audio.cpp -------------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- audio ingest. See whisper/audio.hpp.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "whisper/audio.hpp"

#include "whisper/features.hpp"  // kSampleRate: one definition of the rate

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

// execve takes the environment explicitly, and glibc only declares `environ`
// under _GNU_SOURCE. Declared here rather than switching the feature macro for
// the whole file: one symbol, one line, no effect on anything else.
extern char **environ;
#endif

namespace npue::whisper {
namespace {

uint32_t rd_u32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t rd_u16(const uint8_t *p) {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                               (static_cast<uint16_t>(p[1]) << 8));
}

[[noreturn]] void refuse(const std::string &path, const std::string &why,
                         const std::string &fix) {
  throw std::runtime_error(path + ": " + why + ". " + fix);
}

std::string slurp(const std::string &path, size_t cap) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  const auto n = f.tellg();
  if (n < 0) throw std::runtime_error(path + ": not seekable");
  if (static_cast<size_t>(n) > cap)
    throw std::runtime_error(path + ": " + std::to_string(n) +
                             " bytes is over the " + std::to_string(cap) +
                             " byte limit for an audio request");
  f.seekg(0, std::ios::beg);
  std::string out(static_cast<size_t>(n), '\0');
  f.read(out.data(), n);
  return out;
}

}  // namespace

Audio read_wav(const std::string &path) {
  const std::string raw = slurp(path, 64u * 1024 * 1024);
  const auto *p = reinterpret_cast<const uint8_t *>(raw.data());
  const size_t n = raw.size();
  if (n < 12 || std::memcmp(p, "RIFF", 4) != 0 || std::memcmp(p + 8, "WAVE", 4) != 0)
    refuse(path, "not a RIFF/WAVE file",
           "convert it: --convert (ffmpeg -f s16le -ar 16000 -ac 1)");

  // Walk the chunks. A WAV is a list of typed blocks and the two we want are
  // not adjacent, not first, and not necessarily in this order -- LIST and fact
  // chunks sit between them often enough that assuming otherwise is a bug
  // waiting for a real file.
  bool have_fmt = false;
  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  const uint8_t *data = nullptr;
  size_t data_bytes = 0;
  size_t off = 12;
  while (off + 8 <= n) {
    char id[5] = {0};
    std::memcpy(id, p + off, 4);
    const uint32_t sz = rd_u32(p + off + 4);
    const size_t body = off + 8;
    if (body + sz > n) break;  // truncated: use what is there, fmt decides
    if (std::memcmp(id, "fmt ", 4) == 0 && sz >= 16) {
      format = rd_u16(p + body);
      channels = rd_u16(p + body + 2);
      rate = rd_u32(p + body + 4);
      bits = rd_u16(p + body + 14);
      // WAVE_FORMAT_EXTENSIBLE (0xFFFE): the real format is the first two
      // bytes of the SubFormat GUID, which is where every encoder puts it.
      if (format == 0xFFFE && sz >= 26) format = rd_u16(p + body + 24);
      have_fmt = true;
    } else if (std::memcmp(id, "data", 4) == 0) {
      data = p + body;
      data_bytes = sz;
    }
    off = body + sz + (sz & 1);  // chunks are word-aligned
  }

  if (!have_fmt || !data)
    refuse(path, "no fmt /data chunk",
           "it is not a usable PCM WAV; convert it: --convert");
  if (format == 3 && bits == 32) {
    // IEEE float: accepted as-is, it is lossless and 16 kHz mono is the only
    // thing that matters.
  } else if (format == 1 && bits == 16) {
    // 16-bit PCM: the common case.
  } else {
    refuse(path,
           "format " + std::to_string(format) + ", " + std::to_string(bits) +
               " bits per sample (need 16-bit PCM or 32-bit float)",
           "convert it: --convert");
  }
  if (channels != 1)
    refuse(path, std::to_string(channels) + " channels",
           "Whisper is 16 kHz MONO; downmix with --convert");
  if (rate != kSampleRate)
    refuse(path, std::to_string(rate) + " Hz",
           "Whisper is 16 kHz; resample with --convert");

  Audio a;
  a.sample_rate = static_cast<int>(rate);
  if (format == 1) {
    const size_t count = data_bytes / 2;
    a.samples.resize(count);
    for (size_t i = 0; i < count; ++i) {
      const int16_t v = static_cast<int16_t>(rd_u16(data + 2 * i));
      a.samples[i] = static_cast<float>(v) / 32768.0f;
    }
  } else {
    const size_t count = data_bytes / 4;
    a.samples.resize(count);
    std::memcpy(a.samples.data(), data, count * 4);
  }
  a.duration_samples = static_cast<int64_t>(a.samples.size());
  return a;
}

#ifndef _WIN32
namespace {

// execve does NOT search PATH -- that is execvpe/execvp, and the difference
// matters here: execve("ffmpeg") fails with ENOENT on every system, and the
// child then exits 127, which reads like "your file is broken" unless the
// parent knows to look for a missing binary. So the search happens here, once,
// before the fork, and the failure names the binary.
std::string resolve_on_path(const std::string &name) {
  if (name.find('/') != std::string::npos) return name;  // already a path
  const char *path_env = std::getenv("PATH");
  const std::string paths = path_env ? path_env : "/usr/bin:/bin";
  size_t start = 0;
  while (start <= paths.size()) {
    const size_t colon = paths.find(':', start);
    const std::string dir = paths.substr(
        start, colon == std::string::npos ? std::string::npos : colon - start);
    if (!dir.empty()) {
      const std::string full = dir + "/" + name;
      if (::access(full.c_str(), X_OK) == 0) return full;
    }
    if (colon == std::string::npos) break;
    start = colon + 1;
  }
  throw std::runtime_error(
      name + " is not on PATH, so no conversion is possible. Install it, or "
             "supply 16 kHz mono 16-bit PCM WAV.");
}

}  // namespace

Audio convert_with_ffmpeg(const std::string &path, const IngestLimits &lim) {
  // Resolved before the fork: a missing ffmpeg is a configuration error and
  // should not cost a pipe, a child and a waitpid to report.
  const std::string exe = resolve_on_path(lim.ffmpeg);
  int fds[2];
  if (::pipe(fds) != 0)
    throw std::runtime_error(std::string("pipe: ") + std::strerror(errno));

  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    throw std::runtime_error(std::string("fork: ") + std::strerror(errno));
  }
  if (pid == 0) {
    // Child. execve with an argv array and NO shell: the path is an argument,
    // so a file called `a; rm -rf ~` is a file.
    ::close(fds[0]);
    ::dup2(fds[1], STDOUT_FILENO);
    ::close(fds[1]);
    // ffmpeg's diagnostics on stderr, and nothing on stdin: without -nostdin a
    // converter that decides to prompt will block forever on a pipe nobody
    // writes to, and the request hangs until the deadline kills it.
    const std::string src = "-i";
    const std::string format = "-f";
    const std::string s16le = "s16le";
    const std::string ar = "-ar";
    const std::string rate = std::to_string(kSampleRate);
    const std::string ac = "-ac";
    const std::string one = "1";
    const std::string dash = "-";
    const std::string loglevel = "-loglevel";
    const std::string err = "error";
    const std::string hide = "-hide_banner";
    const std::string nostdin = "-nostdin";
    const std::string prog = "ffmpeg";
    char *const argv[] = {
        const_cast<char *>(prog.c_str()),
        const_cast<char *>(hide.c_str()),
        const_cast<char *>(nostdin.c_str()),
        const_cast<char *>(loglevel.c_str()),
        const_cast<char *>(err.c_str()),
        const_cast<char *>(src.c_str()),
        const_cast<char *>(path.c_str()),
        const_cast<char *>(format.c_str()),
        const_cast<char *>(s16le.c_str()),
        const_cast<char *>(ar.c_str()),
        const_cast<char *>(rate.c_str()),
        const_cast<char *>(ac.c_str()),
        const_cast<char *>(one.c_str()),
        const_cast<char *>(dash.c_str()),
        nullptr};
    ::execve(exe.c_str(), argv, environ);
    // execve does not return on success. 127 is the shell's "not found"
    // convention and is what the parent turns into a message naming the
    // binary, because "No such file or directory" from a bare execve says
    // nothing about which path was tried.
    ::_exit(127);
  }

  ::close(fds[1]);
  Audio a;
  a.sample_rate = kSampleRate;
  std::string out;
  out.reserve(1 << 16);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(lim.timeout_s);
  for (;;) {
    const long left_ms = static_cast<long>(std::chrono::duration_cast<
                                           std::chrono::milliseconds>(
                                               deadline - std::chrono::steady_clock::now())
                                               .count());
    if (left_ms <= 0) {
      ::kill(pid, SIGKILL);
      ::close(fds[0]);
      int st = 0;
      ::waitpid(pid, &st, 0);
      throw std::runtime_error(
          path + ": ffmpeg did not finish within " +
          std::to_string(lim.timeout_s) +
          " s and was killed. A conversion that outlives its request is a "
          "refusal, not a slow answer.");
    }
    struct pollfd p{fds[0], POLLIN, 0};
    const int n = ::poll(&p, 1, static_cast<int>(std::min<long>(left_ms, 500)));
    if (n < 0) {
      if (errno == EINTR) continue;
      ::kill(pid, SIGKILL);
      ::close(fds[0]);
      throw std::runtime_error(std::string("poll: ") + std::strerror(errno));
    }
    if (n == 0) continue;  // deadline not reached, nothing to read yet
    char buf[65536];
    const ssize_t got = ::read(fds[0], buf, sizeof buf);
    if (got < 0) {
      if (errno == EINTR) continue;
      ::kill(pid, SIGKILL);
      ::close(fds[0]);
      throw std::runtime_error(std::string("read: ") + std::strerror(errno));
    }
    if (got == 0) break;  // EOF: ffmpeg is done
    if (out.size() + static_cast<size_t>(got) > lim.max_bytes) {
      // Kill BEFORE waiting: the child is blocked writing to a pipe nobody is
      // going to read, so a waitpid here would hang until the deadline and
      // then report a timeout for what is really a size refusal.
      ::kill(pid, SIGKILL);
      ::close(fds[0]);
      int killed = 0;
      ::waitpid(pid, &killed, 0);
      throw std::runtime_error(
          path + ": converted audio is over the " +
          std::to_string(lim.max_bytes / (2 * kSampleRate)) +
          " second limit for one request. Split the recording; long-form "
          "transcription chunks at 30 s, so this is about the request, not "
          "about the model.");
    }
    out.append(buf, static_cast<size_t>(got));
  }
  ::close(fds[0]);

  int st = 0;
  ::waitpid(pid, &st, 0);
  if (WIFSIGNALED(st))
    throw std::runtime_error(path + ": ffmpeg was killed (signal " +
                             std::to_string(WTERMSIG(st)) + ")");
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
    const int code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    throw std::runtime_error(
        path + ": ffmpeg exited " + std::to_string(code) +
        (code == 127
             ? " -- " + exe +
                   " is not on PATH, so no conversion is possible. Install "
                   "it, or supply 16 kHz mono 16-bit PCM."
             : ". The file is probably not audio, or is a codec ffmpeg was not "
               "built with."));
  }
  if (out.size() < 2)
    throw std::runtime_error(path + ": ffmpeg produced no audio samples");

  const size_t count = out.size() / 2;
  a.samples.resize(count);
  const auto *q = reinterpret_cast<const uint8_t *>(out.data());
  for (size_t i = 0; i < count; ++i) {
    const int16_t v = static_cast<int16_t>(rd_u16(q + 2 * i));
    a.samples[i] = static_cast<float>(v) / 32768.0f;
  }
  a.duration_samples = static_cast<int64_t>(count);
  return a;
}
#else
Audio convert_with_ffmpeg(const std::string &path, const IngestLimits &) {
  throw std::runtime_error(
      path + ": --convert needs the POSIX ffmpeg path, which this build does "
             "not have. Supply 16 kHz mono 16-bit PCM WAV instead.");
}
#endif

Audio ingest(const std::string &path, bool convert, const IngestLimits &lim) {
  if (convert) return convert_with_ffmpeg(path, lim);
  return read_wav(path);
}

}  // namespace npue::whisper
