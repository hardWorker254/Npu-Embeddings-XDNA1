//===- hub.hpp ----------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the model catalogue, and fetching a model without Python
// and without a shell script.
// SPDX-License-Identifier: Apache-2.0
//
// WHY THIS EXISTS. The release used to ship `get-model.cmd`: a batch file
// that ran `curl` and then compared a `certutil -hashfile` digest against a
// pinned one. That is exactly the shape of a dropper -- a script that
// downloads a payload and checks it against a hardcoded hash -- so
// SmartScreen and every AV heuristic treat it as one, and the first thing a
// new user saw was a security warning. The behaviour was always fine; the
// packaging was indefensible. Doing it inside the signed executable removes
// the script, the `curl` dependency and the `certutil` call in one move.
//
// The verification itself is UNCHANGED and non-negotiable: the checkpoint is
// fetched from its canonical repository and its SHA-256 is compared against a
// pin compiled into this binary. A mismatch REFUSES. We still ship no
// weights: they belong to their authors, and a checksum against the canonical
// source beats trusting a blob in someone's zip.
//
// Everything here is Windows-only by construction (WinHTTP). That is the only
// platform this runtime targets.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace npue {
namespace hub {

// A model this build knows how to fetch and run.
//
// This is a catalogue of what is FETCHABLE, and it is deliberately not the
// same thing as `discover_models()`'s list of what is INSTALLED. Keeping the
// two separate is what lets `list` show a model the user does not have yet.
//
// `tile_n` is a property of the model, not a constant: the design asserts
// `N % (tile_n * cols) == 0`, and bge-large's N in {1024, 3072, 4096} makes
// 48 illegal (tasks/0042). Carrying it here means the user never has to know
// that, and never has to pass a flag whose correct value is a fact about the
// checkpoint.
struct CatalogEntry {
  std::string name;         // container stem, and what `serve <name>` takes
  std::string repo;         // HuggingFace repo id
  std::string sha256;       // pin for model.safetensors
  std::string pooling;      // "cls" or "mean" -- cross-checked against the
                            // downloaded 1_Pooling/config.json
  int64_t hidden = 0, layers = 0, heads = 0, ffn = 0;
  int64_t tile_n = 48;
  double download_mb = 0;   // the checkpoint, not the container
  std::string note;         // one line for the table
};

const std::vector<CatalogEntry> &catalog();

// Look a name up in the catalogue. Returns nullptr when it is not there --
// which is not an error, because a user may have packed their own container.
const CatalogEntry *find(const std::string &name);

using Log = std::function<void(const std::string &)>;

// Fetch one file over HTTPS into `dest`, following redirects (the HuggingFace
// CDN always redirects). Writes to `dest + ".part"` and renames on success,
// so an interrupted download can never be mistaken for a complete one.
void download(const std::string &url, const std::string &dest,
              const Log &log);

// Make sure `<root>/models/<name>.npue` exists, fetching and packing it if it
// does not. Returns the container path.
//
// Fails closed at every step: an unknown name, a checksum mismatch, a pooling
// mode that disagrees with the catalogue, or a failed download all throw
// rather than proceeding with something plausible.
std::string ensure_model(const std::string &root, const std::string &name,
                         const Log &log);

}  // namespace hub
}  // namespace npue
