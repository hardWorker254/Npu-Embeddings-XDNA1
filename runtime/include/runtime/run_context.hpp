//===- run_context.hpp ------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the shared state that the decomposed main() pieces pass
// between themselves. Every field that the old monolithic main() held as a
// local through its tail is a member here; the accessor set d_*() resolves the
// unified-vs-per-op design aliasing exactly once, the same aliasing the old code
// spelled out inline as `unified ? *ud : *ld_qkv`.
//
// main() builds a RunContext, fills its early fields (root/argv/bench/art/dev),
// then hands it to a series of command functions that each own one block of the
// old body. Each function fills in what it needs and reads what a previous one
// left behind -- no function takes ten parameters.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NPUEMBEDDINGS_RUN_CONTEXT_HPP
#define NPUEMBEDDINGS_RUN_CONTEXT_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "encoders/bert_encoder.hpp"
#include "common/design_selection.hpp"  // StreamEntry
#include "runtime/model.hpp"
#include "runtime/design.hpp"    // npu::Device, npu::Design
#include "runtime/pool.hpp"      // Pool

namespace app {

// The goldens were generated at this batch; every comparison path is tiled by
// rotation against it (see load_goldens / run_golden_check). Held here so both
// the loader and the checker agree on the constant without re-deriving it.
constexpr int64_t kGoldenBatch = 4;

struct RunContext {
  // --- CLI, fixed up-front -------------------------------------------------
  int argc = 0;
  char **argv = nullptr;
  std::string root;          // repo root the model/artifacts live under
  int bench = 0;            // --bench N, 0 = run the golden check

  // --- flag form (--model/--artifacts/--emb ...) after subcommand rewrite --
  std::string art;          // resolved design set directory
  std::string model_path;

  // --- model + fixtures ----------------------------------------------------
  std::unique_ptr<npue::File> model;
  std::string val;          // fixtures directory
  bool val_is_own = false;  // fixtures matched by checkpoint sha, not name
  bool have_val = false;    // emb_sum.f32 present and sha-matched

  // --- device --------------------------------------------------------------
  std::unique_ptr<npu::Device> dev;

  // --- designs -------------------------------------------------------------
  // In unified mode all seven refs below alias `*ud`; the per-op `ld_*`s are
  // then null. Kept as separate members so the accessor set reads as the old
  // inline `unified ? *ud : *ld_x`.
  bool unified = false;
  std::unique_ptr<npu::Design> ud;
  std::unique_ptr<npu::Design> ld_qkv, ld_ao, ld_fu, ld_fd, ld_gelu, ld_ln,
      ld_sm;
  std::vector<StreamEntry> streams;

  // Which NPU generation the loaded design set was built for, read off the
  // design itself in load_designs. Held here so the status line reports the
  // value rather than the flag or directory name that led to this set
  // (subtask 3).
  std::string design_device;
  int64_t design_arch = 0;
  bool design_generation_recorded = false;

  // Design geometry, derived from the loaded design set.
  int64_t rows = 0, batch = 0;

  // --- goldens -------------------------------------------------------------
  std::vector<float> emb_in, mask, want, amask_i;

  // --- host-side policy flags ---------------------------------------------
  int nthreads = 1;
  bool host_ln = false, host_sm = false, host_gelu = false;
  // --npu-eltwise opts GELU/LayerNorm/softmax OFF the host and onto the array.
  // Only meaningful for the unified gemm_rtp set, which has no eltwise designs
  // of its own: the flag makes load_designs resolve the sibling gelu/,
  // layernorm/ and softmax/ directories, and setup_encoder then stops forcing
  // host execution. Absent the flag the behaviour is the shipped one -- host
  // eltwise, which is the measured-faster path.
  bool npu_eltwise = false;
  bool sim_c_bf16 = false;
  bool no_fuse_ffn = false;
  // --allow-contention: the only override for a run whose hw_context budget is
  // full or unreadable (subtask 7). Set by setup_flags_pools; consumed by the
  // guard in load_designs and by the bench contention check.
  bool allow_contention = false;
  int pipeline = 0;   // 0 = off; N = N concurrent lanes

  // --- pools + encoders ----------------------------------------------------
  // The runtime owns encoder construction (Task 0) and the concrete
  // BERT-family encoder is the one whose raw-forward methods the embedding
  // and benchmark paths use, so the type is named rather than the text-in
  // base interface.
  std::vector<std::unique_ptr<Pool>> pools;
  std::unique_ptr<npue::BertEncoder> enc;
  std::vector<std::unique_ptr<npue::BertEncoder>> lanes;

  // Unified-vs-per-op design aliasing. All return references so the old
  // `Design &d_qkv = ...` call sites become `c.d_qkv()` verbatim.
  npu::Design &d_qkv() const { return unified ? *ud : *ld_qkv; }
  npu::Design &d_ao() const { return unified ? *ud : *ld_ao; }
  npu::Design &d_fu() const { return unified ? *ud : *ld_fu; }
  npu::Design &d_fd() const { return unified ? *ud : *ld_fd; }
  // The three eltwise accessors prefer their own design when --npu-eltwise
  // loaded one and it is actually going to run on the array; otherwise they
  // alias the unified GEMM design so a host-forced op still gets a valid
  // reference. `ld_*` is non-null exactly when load_designs resolved it and
  // the op is not forced back onto the host.
  npu::Design &d_gelu() const {
    return (unified && !(npu_eltwise && ld_gelu)) ? *ud : *ld_gelu;
  }
  npu::Design &d_ln() const {
    return (unified && !(npu_eltwise && ld_ln)) ? *ud : *ld_ln;
  }
  npu::Design &d_sm() const {
    return (unified && !(npu_eltwise && ld_sm)) ? *ud : *ld_sm;
  }

  // The golden check vector -- see need_goldens call sites below.
  void need_goldens() const {
    if (!have_val)
      throw std::runtime_error(
          "no golden check vectors under " + val +
          " -- this build can run --embed, --serve, --tokenize and "
          "--encode-file, but not the golden check or --bench. Generate them "
          "with tools/export_validation.py.");
  }
};

}  // namespace app

#endif  // NPUEMBEDDINGS_RUN_CONTEXT_HPP
