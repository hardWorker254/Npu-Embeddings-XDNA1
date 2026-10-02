//===- onnx_read.hpp ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- read an ONNX checkpoint under the names and in the
// orientation the CHECKPOINT itself used.
// SPDX-License-Identifier: Apache-2.0
//
// WHY THIS EXISTS
// ---------------
// The packers ask for tensors by the names the checkpoint itself used
// ("embeddings.word_embeddings.weight", "encoder.layer.3.attn.self.query.weight")
// in the orientation a Linear weight is stored in there, [out, in]. ONNX
// disagrees about both: exporters rename, re-root and re-orient, and every
// disagreement is invisible until something packs the wrong numbers.
//
// This file is the C++ half of tools/lib/onnx_weights.py. It walks the
// protobuf wire format itself rather than linking a protobuf runtime, for the
// same reason the Python reader does: the release's whole pitch is one
// binary with one dependency (XRT), and pulling in a generated descriptor set
// to read two nested messages would trade that for a build step.
//
// The contract it keeps with the Python reader is exactly the one that makes
// tools/verify/verify_pack_parity.py a byte comparison:
//
//   * every tensor is served under the CHECKPOINT's name and the CHECKPOINT's
//     orientation, so no packer changed shape when the source did;
//   * `source_files()` lists the same files in the same order
//     model_digest() hashes, so a container records the same source_sha256
//     Python's pack_npue.py records;
//   * a layout that cannot be decided is REFUSED, never guessed -- a wrong
//     orientation builds a container that passes its own layout hash and
//     emits embeddings that look like noise, which is the worst failure mode
//     this project has.
//
// The field numbers below were read off the descriptors of the installed
// onnx package, and tools/verify/verify_onnx_reader.py re-checks that table
// against `onnx` on every run. That matters because protobuf does not
// complain about a wrong field number: it quietly returns a different tensor.
//
//   ModelProto.graph                    = 7   (length-delimited)
//   GraphProto.node / .initializer      = 1 / 5
//   NodeProto.input/output/op_type/attr = 1 / 2 / 4 / 5
//   TensorProto.dims/data_type/name     = 1 / 2 / 8
//   TensorProto.raw_data/external_data  = 9 / 13
//   AttributeProto.name/i/type          = 1 / 3 / 20
//   StringStringEntryProto.key/value    = 1 / 2
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace npue {

// Where a checkpoint's graph lives, relative to its own directory. One place,
// because the packers, the fetch path and reference/onnx_io.py all spell it
// and three spellings of one path is a pin that cannot be matched.
//
// These are the C++ twins of tools/lib/onnx_weights.py's MODEL_ONNX,
// WHISPER_ENCODER_ONNX and WHISPER_DECODER_ONNX. The graph never moves: an
// export's `external_data.location` is a RELATIVE path baked into the graph
// at export time, so a side file living beside the graph under the basename
// the graph names is the only layout that reads without patching the file --
// and patching the file would change the very digest it is pinned by.
constexpr const char *kModelOnnx = "onnx/model.onnx";
constexpr const char *kWhisperEncoderOnnx = "onnx/encoder_model.onnx";
constexpr const char *kWhisperDecoderOnnx = "onnx/decoder_model.onnx";

// One tensor of a checkpoint, under the name and in the orientation the
// CHECKPOINT used.
//
// `data` points either straight into a read-only mapping of the file the
// bytes live in (the common case: F32, already [out, in], so nothing is
// copied) or into one of the two owning buffers below. It is stable for as
// long as the OnnxWeights that produced it lives -- which is why those are
// shared_ptr rather than members: a std::map moves its values, and a moved
// Tensor must not leave a dangling `data` behind it.
struct Tensor {
  std::string dtype;
  std::vector<int64_t> shape;
  const uint8_t *data = nullptr;
  size_t bytes = 0;
  // The side file these bytes came from, or "" for a tensor that lives in the
  // graph itself. Kept on the tensor rather than in a table because
  // source_files() has to walk tensors in DECLARATION order to reproduce the
  // Python reader's list -- and that list is what model_digest() hashes.
  std::string ext_path;

  // Set only when the tensor could not be handed back as a view: widened from
  // F16/BF16, transposed to [out, in], or both. Both conversions are exact in
  // the sense that matters -- fp32 has more exponent range and more mantissa
  // than either half -- so widening cannot be the source of any error
  // measured downstream.
  std::shared_ptr<std::vector<float>> owned;
  // The same, for a tensor that is not a float: a transposed I64 constant has
  // nowhere to live in `owned`, and dropping it would leave `data` pointing at
  // bytes this reader no longer controls.
  std::shared_ptr<std::vector<uint8_t>> owned_raw;

  const float *f32() const { return reinterpret_cast<const float *>(data); }
  int64_t rows() const { return shape.size() > 1 ? shape[0] : 1; }
  int64_t cols() const { return shape.empty() ? 0 : shape.back(); }
  int64_t count() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

// A root the export ADDED (strip), a root the CHECKPOINT has (prefix), and the
// export's mid-name rewrites (rename), applied in that order on top of the
// recovered name -- tools/lib/onnx_weights.py's `_build` order, byte for byte.
//
// `rename` exists because prefix surgery cannot reach a name that was rewritten
// in the middle: onnx-community's embeddinggemma export spells
// `.self_attn.`, `.layernorm.weight` and the post-stack norm as one more layer.
// Only embeddinggemma needs either, and it needs BOTH -- stripping without
// renaming hands the export's spelling straight back to a packer asking for the
// checkpoint's. So they travel together.
struct OnnxReadOptions {
  std::string prefix;
  std::string strip;
  std::vector<std::pair<std::string, std::string>> rename;
};

// embeddinggemma's two rewrites, assembled once: the export re-roots every
// parameter under `model.` and then spells attention, the RMSNorms and the
// post-stack norm differently from the checkpoint the packer reads. The
// `24` is num_hidden_layers (asserted against config.json by
// reference/fetch_model_gemma.py); it is spelled here because a rename rule
// that silently stopped applying would hand the export's spelling back to the
// packer and pack the wrong tensor into the right slot.
//
// The C++ twin of EMBEDDINGGEMMA_STRIP / EMBEDDINGGEMMA_RENAME in
// tools/pack/pack_npue.py -- same three rules, same order, applied after
// prefix and strip, which is _build()'s order in both readers.
inline OnnxReadOptions embeddinggemma_options() {
  OnnxReadOptions o;
  o.strip = "model.";
  o.rename = {{".attn.", ".self_attn."},
              {".layernorm.weight", ".weight"},
              {"layers.24.final_norm_layernorm.weight", "norm.weight"}};
  return o;
}

// The ONNX weights of ONE graph file, plus every side file those weights live
// in, kept mapped for as long as this object is alive.
//
// Construction reads METADATA ONLY: field numbers, names, dims, and where each
// payload starts. No tensor is copied unless its layout forces it, which is
// what keeps a 1.3 GB bge-large graph from costing 1.3 GB of resident memory
// to enumerate.
class OnnxWeights {
 public:
  OnnxWeights(const std::string &graph_path,
              const OnnxReadOptions &opts = OnnxReadOptions());
  OnnxWeights(const OnnxWeights &) = delete;
  OnnxWeights &operator=(const OnnxWeights &) = delete;
  ~OnnxWeights();

  // Every tensor under its CHECKPOINT name, in the order the graph declared
  // them (first declaration wins; a second tensor recovering to a name already
  // taken is refused rather than shadowed).
  const std::map<std::string, Tensor> &tensors() const { return tensors_; }

  // Every file whose BYTES this reader can hand back, graph first.
  //
  // An ONNX model may keep its weights in a side file, and the ones that do
  // are the large ones -- whisper-large-v3's graph is 0.7 MB and the data it
  // points at is 2.5 GB. Hashing only the graph, which is what a plain
  // `sha256(model.onnx)` would do, then pins the tensor names and none of the
  // WEIGHTS: swap the checkpoint out under an unchanged graph and the pin
  // verifies clean. model_digest() walks this list instead.
  const std::vector<std::string> &source_files() const { return files_; }

  const std::string &path() const { return path_; }

 private:
  struct Mapping;

  // The read-only mapping of `p`, created on first use and kept until this
  // object dies: `data` is handed out as a view into it for every tensor that
  // needed no copy, so unmapping early would fault the next read rather than
  // fail politely -- the exact hazard tools/lib/onnx_weights.py documents for
  // its mmap, and the reason array() there returns an owned copy.
  // `size` receives the file's length, which is what every payload bounds
  // check runs against.
  const uint8_t *map_file(const std::string &p, size_t *size);

  std::string path_;
  std::map<std::string, Tensor> tensors_;
  std::vector<std::string> order_;   // fixed names, declaration order
  std::vector<std::string> files_;   // source_files(), graph first
  std::vector<std::unique_ptr<Mapping>> maps_;
};

// What a model directory holds of an ONNX checkpoint -- the fact
// reference/onnx_io.py calls `checkpoint_files()` / `checkpoint_digest()`.
//
// ALL OR NOTHING: a graph whose side file has not arrived carries no weights
// anyone can read, so it contributes nothing and the whole checkpoint reports
// as absent. A half-listed checkpoint is worse than none, because it is then
// possible to pin and verify bytes that were never loaded -- which is the
// reason `file` and `sha256` in CHECKPOINT.json are either both set or both
// empty.
struct OnnxCheckpoint {
  // Every file the graph's bytes live in, relative to `dir`, graph first and
  // then each graph's side files -- exactly what model_digest() hashes.
  std::vector<std::string> files;
  // The graph files themselves, as paths that can be opened.
  std::vector<std::string> graphs;
  bool empty() const { return files.empty(); }
};

// The checkpoint in `dir`, or an empty one when nothing is placed there (or
// what is placed is incomplete). Checked in reference/onnx_io.py's order --
// model.onnx, then whisper's encoder, then its decoder -- because that order
// is part of what the digest covers.
OnnxCheckpoint onnx_checkpoint(const std::string &dir);

}  // namespace npue
