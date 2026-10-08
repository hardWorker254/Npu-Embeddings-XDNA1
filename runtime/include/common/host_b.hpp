//===- host_b.hpp ------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- put a pre-tiled B panel back into fp32 [K, N] for the host.
// SPDX-License-Identifier: Apache-2.0
//
// WHY THIS EXISTS
// ---------------
// The container stores every GEMM operand already tiled for the array
// (block_panel: a flat sequence of (kb, nb) tiles, each in the (r0, c0, s, t)
// order the DMA streams). That is the right format for the array and the wrong
// one for a CPU, which wants row-major [K, N]. Until this file, a run that did
// not dispatch had nowhere to get its weights FROM: the four per-layer GEMMs
// went to the array on the strength of a design set being present, and there
// was no host path to fall back to -- `--cpu` was refused by name for every
// architecture but arch=1.
//
// So this is the other half of `--npu-ops gemm`: a code that can be OFF needs
// somewhere to go, and that somewhere is a CPU multiply.
//
// THE ARITHMETIC IS tile_b()'s RUN BACKWARDS
// -------------------------------------------
// tools/lib/npue.py:482 `tile_b` and :549 `untile_b` are the packing side;
// this is the third and last definition of that order (int4_panel.hpp is the
// fourth, and it walks the same six axes for a different reason). The axes,
// from npue.py's reshape/transpose:
//
//   panel[ (kb, nb, ri, ci, si, ti) ]  with t = ti fastest
//   k = kb*tile_k + ri*mac_s + si
//   j = nb*tile_n + ci*mac_t + ti
//
// which is what the loop below walks, writing `panel[f]` to `out[k*N + j]`.
// A wrong order here still produces a right-sized array of plausible numbers
// -- the same failure mode every other tiling bug in this tree has -- so the
// check is against the model's output, not against a byte count.
//
// WHAT IS AND IS NOT CONVERTED
// ----------------------------
// BF16 is widened (the shift is exact: bf16 -> f32 loses no bits).
// I8/I4 is dequantised per column by `.wscale`, which is the SAME arithmetic
// `dequantise_c` applies to the C side, and I4 arrives widened by
// gemm_b_panel() exactly as it does for the array -- there is no second
// int4 story here.
//
// `.asmooth` is deliberately NOT applied to the weights, because it is already
// in them: the packer pre-multiplies W by asmooth and the ACTIVATION carries
// 1/asmooth (`inv_smooth`, see NpuGemm::run_i8). Applying it twice would
// cancel out and applying it once on the wrong side would not. The caller
// scales its activation row; this file only unwraps the operand.
//
// Env: C++17, no dependencies beyond int4_panel.hpp.

#ifndef NPU_EMBEDDINGS_COMMON_HOST_B_HPP
#define NPU_EMBEDDINGS_COMMON_HOST_B_HPP

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "int4_panel.hpp"

namespace npue::hostb {

// `name`'s B operand as fp32 [K, N], row-major: out[k*N + j] is the multiplier
// for input channel k and output channel j, which is what gemm_nt() reads.
//
// Built ONCE per operand by the caller (it is K*N floats -- as much as the
// whole model in f32) and then reused; this function is pure and keeps no
// state of its own.
inline std::vector<float> host_b_kn(const File &model,
                                    const std::string &name) {
  const TensorInfo &t = model.info(name);
  if (t.padded_shape.size() != 2)
    throw std::runtime_error(
        name + ": host GEMM operand has a " +
        std::to_string(t.padded_shape.size()) +
        "-dimensional padded shape; a dense operand is [K, N]");
  if (!t.logical_shape.empty() &&
      (t.logical_shape[0] != t.padded_shape[0] ||
       t.logical_shape[1] != t.padded_shape[1]))
    // Measured absent across all 498 block_panel tensors in this checkout, so
    // this is a refusal rather than handling: padding K or N at pack time would
    // make out[] wider than the shape the caller indexes it with, and the two
    // disagreeing silently is the wrong-answer-looks-plausible failure.
    throw std::runtime_error(
        name + ": logical [" + std::to_string(t.logical_shape[0]) + "," +
        std::to_string(t.logical_shape[1]) + "] but padded [" +
        std::to_string(t.padded_shape[0]) + "," +
        std::to_string(t.padded_shape[1]) +
        "]. The host builds [K,N] from the padded shape and the caller indexes "
        "it with the logical one, so the two must agree. Repack with "
        "tools/pack/pack_npue.py.");
  const int64_t K = t.padded_shape[0];
  const int64_t N = t.padded_shape[1];
  if (K <= 0 || N <= 0)
    throw std::runtime_error(name + ": GEMM operand with a zero dimension");
  const size_t kn = static_cast<size_t>(K) * static_cast<size_t>(N);

  // No layout at all: row-major fp32 already, as the ViT classifier head is
  // stored (npue.py:168-173). Nothing to untile and nothing to widen.
  if (t.dtype == "F32") {
    const Span raw = model.raw(name);
    if (raw.bytes != kn * sizeof(float))
      throw std::runtime_error(
          name + ": F32 payload is " + std::to_string(raw.bytes) +
          " bytes but [" + std::to_string(K) + "," + std::to_string(N) +
          "] needs " + std::to_string(kn * sizeof(float)));
    const float *p = raw.as<float>();
    return std::vector<float>(p, p + kn);
  }

  if (t.dtype != "BF16" && t.dtype != "I8" && t.dtype != "I4")
    throw std::runtime_error(
        name + ": " + t.dtype +
        " B operand cannot be run on the host. This build widens BF16, "
        "dequantises I8 and I4 (via gemm_b_panel) and copies F32; anything else "
        "would mean inventing a conversion, and a guessed conversion is a wrong "
        "number in a plausible range. Repack with tools/pack/pack_npue.py.");

  if (t.layout_kind != "block_panel" || t.tile_k <= 0 || t.tile_n <= 0 ||
      t.mac_s <= 0 || t.mac_t <= 0)
    throw std::runtime_error(
        name + ": payload carries no usable block_panel layout (kind '" +
        t.layout_kind + "'). The layout is what says where each element sits, "
        "so without it these bytes are a right count in an unknown order. "
        "Repack with tools/pack/pack_npue.py.");
  if (K % t.tile_k || N % t.tile_n || t.tile_k % t.mac_s ||
      t.tile_n % t.mac_t)
    throw std::runtime_error(
        name + ": layout (" + std::to_string(t.tile_k) + "," +
        std::to_string(t.tile_n) + ") mac (" + std::to_string(t.mac_s) + "," +
        std::to_string(t.mac_t) + ") does not tile [" + std::to_string(K) +
        "," + std::to_string(N) + "]");

  // The bytes the array would stage: bf16/I8 straight from the mapping, I4
  // widened to int8 (int4_panel.hpp). `1` for I4 because int4 is weight
  // storage over the int8 datapath -- the widening produces int8 values.
  const Panel p = gemm_b_panel(model, name, t.dtype == "I4" ? 1 : 0);
  const size_t elem = t.dtype == "BF16" ? sizeof(uint16_t) : 1;
  if (p.bytes.bytes != kn * elem)
    throw std::runtime_error(
        name + ": panel is " + std::to_string(p.bytes.bytes) +
        " bytes but [" + std::to_string(K) + "," + std::to_string(N) + "] in " +
        t.dtype + " needs " + std::to_string(kn * elem));

  // Per-column dequantisation scale, for the int8 datapath only. A bf16
  // container has no .wscale and asking would throw on a missing tensor.
  const float *wscale = nullptr;
  if (t.dtype != "BF16") {
    const std::string wn = name + ".wscale";
    if (!model.has(wn))
      throw std::runtime_error(
          name + ": " + t.dtype +
          " payload but no " + wn +
          " in the container. The column scale is the only thing that can put "
          "an integer back in units of the model's weights, so without it these "
          "bytes are not a weight. Repack with tools/pack/pack_npue.py.");
    const TensorInfo &wi = model.info(wn);
    if (wi.dtype != "F32" || wi.padded_shape.size() != 1 ||
        wi.padded_shape[0] != N)
      throw std::runtime_error(
          name + ".wscale is " + wi.dtype + " [" +
          (wi.padded_shape.size() == 1 ? std::to_string(wi.padded_shape[0])
                                       : std::to_string(wi.padded_shape.size()) +
                                             "D") +
          "] but [" + std::to_string(K) + "," + std::to_string(N) +
          "] needs F32 [" + std::to_string(N) +
          "] -- the operand and its scale disagree about the width, and "
          "dequantising at the wrong one would multiply every column by its "
          "neighbour's scale");
    wscale = model.raw(wn).as<float>();
  }

  const int64_t s = t.mac_s, tt = t.mac_t;
  const int64_t tk = t.tile_k, tn = t.tile_n;
  const int64_t r0 = tk / s, c0 = tn / tt;
  const int64_t kb = K / tk, nb = N / tn;
  const int64_t stride_nb = r0 * c0 * s * tt;
  const int64_t stride_r0 = c0 * s * tt;
  const int64_t stride_c0 = s * tt;

  const auto *bf = reinterpret_cast<const uint16_t *>(p.bytes.data);
  const auto *q8 = reinterpret_cast<const int8_t *>(p.bytes.data);

  std::vector<float> out(kn);
  for (int64_t bki = 0; bki < kb; ++bki) {
    const int64_t base_kb = bki * nb * stride_nb;
    const int64_t k0 = bki * tk;
    for (int64_t bni = 0; bni < nb; ++bni) {
      const int64_t base_nb = base_kb + bni * stride_nb;
      const int64_t j0 = bni * tn;
      for (int64_t ri = 0; ri < r0; ++ri) {
        const int64_t base_r = base_nb + ri * stride_r0;
        const int64_t kr = k0 + ri * s;
        for (int64_t ci = 0; ci < c0; ++ci) {
          const int64_t base_c = base_r + ci * stride_c0;
          const int64_t jc = j0 + ci * tt;
          for (int64_t si = 0; si < s; ++si) {
            const int64_t base_s = base_c + si * tt;
            float *const row = out.data() + static_cast<size_t>(kr + si) * N;
            if (t.dtype == "BF16") {
              for (int64_t ti = 0; ti < tt; ++ti) {
                const uint32_t bits =
                    static_cast<uint32_t>(bf[base_s + ti]) << 16;
                float v;
                std::memcpy(&v, &bits, sizeof(v));
                row[jc + ti] = v;
              }
            } else {
              for (int64_t ti = 0; ti < tt; ++ti)
                row[jc + ti] =
                    static_cast<float>(q8[base_s + ti]) * wscale[jc + ti];
            }
          }
        }
      }
    }
  }
  return out;
}

// The SmoothQuant partner of the WEIGHT-side asmooth: the packer pre-multiplied
// W by asmooth at pack time (so the column scales are already in the operand
// host_b_kn hands back), which leaves 1/asmooth to apply to the activation.
// The two cancel exactly and the product is the unsmoothed one -- applying it
// twice, or once on the wrong side, would not cancel. Null is a container with
// no smoothing at all, and the input is then returned untouched.
//
// `scratch` belongs to the caller and is reused across calls, because this is
// one vector-sized temporary per GEMM otherwise.
inline const float *apply_inv_smooth(const float *a, size_t rows, int64_t k,
                                     const float *asmooth,
                                     std::vector<float> &scratch) {
  if (!asmooth) return a;
  scratch.assign(rows * static_cast<size_t>(k), 0.f);
  for (size_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < k; ++c)
      scratch[r * static_cast<size_t>(k) + c] =
          a[r * static_cast<size_t>(k) + c] / asmooth[c];
  return scratch.data();
}

// ONE STAGED OPERAND'S HOST COPY, and where to find it.
//
// The array is handed a DESIGN SLOT when a weight is staged and needs nothing
// else; the host is handed the same slot and needs the CONTAINER TENSOR NAME,
// because host_b_kn reads the container rather than the array's staged bytes.
// `record()` is called from stage_all(), the one place that knows both halves of
// the pair; `get()` is called from the GEMM, which knows only the slot.
//
// A std::map keyed by a void* is deliberate rather than clever: the four designs
// a BERT lane holds are four independent allocators and each numbers its slots
// from zero, so the SLOT alone identifies nothing. The pointer identity is
// exact, and a miss is a refusal naming the pair rather than a guess at which
// of the four was meant.
//
// NON-COPYABLE AND LOCKED, and both are load-bearing. A BERT lane under
// --pipeline is a separate BertEncoder that copies lane 0's STAGED slot
// vectors -- "the staged weights are the design's, not a lane's" -- so it must
// share this rather than copy it, or every lane keeps its own fp32 copy of the
// whole model (bge-base is ~340 MB of it, times four lanes). The deleted copy
// is what turns "forgot to share it" into a compile error instead of a 1.4 GB
// allocation nobody notices. The mutex is what makes the lazy first use legal:
// every lane asks for the same tensor at the same moment.
struct WeightCache {
  std::map<std::pair<const void *, size_t>, std::string> slot_name;
  std::map<std::string, std::vector<float>> kn;  // fp32 [K, N], built on first use
  std::mutex mu;

  WeightCache() = default;
  WeightCache(const WeightCache &) = delete;
  WeightCache &operator=(const WeightCache &) = delete;

  void record(const void *design, size_t slot, const std::string &name) {
    std::lock_guard<std::mutex> lk(mu);
    slot_name[std::make_pair(design, slot)] = name;
  }

  // The fp32 [K, N] for (design, slot), untiled once and then kept. `K` and `N`
  // are what the CALLER multiplies at and are checked against the operand's own
  // -- a GEMM at a different shape from the weights it was staged for is a wrong
  // number in a plausible range, and this is the one place both are in the same
  // expression.
  const std::vector<float> &get(const File &model, const void *design,
                                size_t slot, int64_t K, int64_t N,
                                const char *who) {
    std::lock_guard<std::mutex> lk(mu);
    const auto it = slot_name.find(std::make_pair(design, slot));
    if (it == slot_name.end())
      throw std::runtime_error(
          std::string(who) + ": B slot " + std::to_string(slot) +
          " was not recorded by stage_all(), so a host GEMM has no container "
          "tensor to read its weights from. The array path stages the bytes and "
          "needs only the slot; the host path needs the name, and this is the "
          "one place the two disagree -- so a host run of an unstaged weight is "
          "a gap in staging rather than something to guess at.");
    std::vector<float> &w = kn[it->second];
    if (w.empty()) {
      w = host_b_kn(model, it->second);
      const TensorInfo &t = model.info(it->second);
      if (t.padded_shape[0] != K || t.padded_shape[1] != N)
        throw std::runtime_error(
            it->second + " is [" + std::to_string(t.padded_shape[0]) + "," +
            std::to_string(t.padded_shape[1]) + "] but this GEMM multiplies at "
            "K=" + std::to_string(K) + ", N=" + std::to_string(N) +
            ". The host reads the operand's own shape rather than trusting the "
            "call site's, so the two disagreeing is a shape bug and not "
            "something to pad away.");
    }
    return w;
  }
};

}  // namespace npue::hostb

#endif  // NPU_EMBEDDINGS_COMMON_HOST_B_HPP
