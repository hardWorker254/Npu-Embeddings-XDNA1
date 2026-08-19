//===- npue_pack.hpp ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- build a .npue from an upstream checkpoint, without Python.
// SPDX-License-Identifier: Apache-2.0
//
// The release does not ship the model. These weights belong to
// sentence-transformers/all-MiniLM-L6-v2, and a user is better served by
// fetching them from the canonical source with a checksum to verify than by
// trusting a 66 MB blob in someone's zip file. This is what makes that
// practical: two downloads and one command, no Python, no toolchain.
//
// The result must be BYTE-IDENTICAL to tools/pack_npue.py's output. Two
// implementations of one binary layout is a real risk -- a disagreement would
// mean correctly-sized weights in the wrong order, which no size check
// catches -- so it is verified rather than assumed
// (tools/verify_pack_parity.py).

#pragma once

#include <cstdint>
#include <string>

namespace npue {

// `layout_json` and `layout_hash` describe the pre-tiled B layout and must be
// exactly what the compiled designs expect; the runtime compares the hash
// before it will dispatch, so a mismatch fails loudly instead of producing
// plausible garbage.
// The canonical B-layout descriptor and its hash, built in ONE place.
//
// tools/npue.py's docstring records why: every hand-written copy of this dict
// is a chance for two sides to drift, and the drift is invisible -- the hash
// changes, the bytes do not, and the check meant to catch wrong layouts starts
// reporting a mismatch that is not one. main.cpp used to carry both the JSON
// and a FROZEN hash for tile_n = 48, which made a second tile size
// unexpressible without editing the packer.
//
// `json` preserves tools/npue.py's INSERTION order (the bytes that go in the
// file); `hash` is over the key-SORTED form, which is what npue.py hashes.
struct Layout {
  std::string json;
  std::string hash;
};
Layout gemm_b_layout(int64_t tile_k, int64_t tile_n, int64_t mac_s = 8,
                     int64_t mac_t = 8);

// The one C++ SHA-256. Exposed so the downloader (src/hub.cpp) verifies a
// checkpoint with exactly the implementation that records `source_sha256`
// into the container. Streams the file; safe on the 438 MB checkpoints.
std::string sha256_file(const std::string &path);

void prepare_model(const std::string &safetensors, const std::string &vocab,
                   const std::string &config_json_path,
                   const std::string &pooling,
                   const std::string &source_repo,
                   const std::string &out, const std::string &source_sha,
                   const std::string &layout_json,
                   const std::string &layout_hash,
                   int64_t tile_k, int64_t tile_n, int64_t max_seq,
                   void (*log)(const std::string &) = nullptr);

}  // namespace npue
