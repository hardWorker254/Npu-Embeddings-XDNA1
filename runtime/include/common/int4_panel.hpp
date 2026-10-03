//===- int4_panel.hpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- widen an int4 B panel into the int8 bytes stage() receives.
// SPDX-License-Identifier: Apache-2.0
//
// int4 is a STORAGE format, not a datapath: the array has no 4-bit MAC, so the
// design, the kernel, the dispatch and the container's own `a_dtype` are all
// the int8 ones and stay that way (tools/lib/gemm_i4.py). The only place the
// two schemes differ in the runtime is here -- the panel has to be widened
// before it is staged, and widening needs the two things the nibbles cannot
// carry on their own: where each one sits, and the per-group factor that puts
// it back in units of `.wscale`.
//
// The arithmetic, and it is the emitter's run backwards:
//
//     t[k,j]  = the stored nibble, sign-extended, in [-8, 7]
//     q[k,j]  = clip(rint(t[k,j] * gscale[k // group, j]), -127, 127)
//
// where `group` is config's `int4_group` (0 == one group over the whole K) and
// `gscale` is the `.gscale` sidecar, F32 [ceil(K/group), N]. `q` is exactly the
// panel an int8 container would carry at the same shape, up to the int4 error
// that is the point of the format -- so everything downstream of it,
// `dequant_int8` included, is unchanged and shares int8's checks.
//
// `rint`/`nearbyint` is round-half-to-EVEN under the default FE_TONEAREST,
// which is what Python's `np.rint` does; the two sides must produce
// byte-identical panels or a gate comparing them measures the tie-break rather
// than the packing. This header and npue.py's `unpack_i4`/`fold_i4` are the
// only two definitions of that order -- a disagreement would still produce a
// right-sized, right-hashed file with wrong numbers in it, which is the failure
// this format spends most of its refusals on.
//
// Env: C++17, no dependencies beyond model.hpp.

#ifndef NPU_EMBEDDINGS_COMMON_INT4_PANEL_HPP
#define NPU_EMBEDDINGS_COMMON_INT4_PANEL_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/model.hpp"

namespace npue {

// The B operand as Design::stage is handed it.
//
// For BF16 and I8 this is a Span straight into the mapped file -- no copy,
// which is the point of the format. For I4 it is `scratch`, a panel this
// struct owns, with `bytes` pointing into it. `scratch` therefore has to
// outlive `bytes`, and the way that is kept true is that stage() copies into a
// staging slot before it returns and no caller has ever kept a Panel past its
// next line: returned by value so the four encoders share one rule instead of
// four that each remember it differently.
struct Panel {
  std::vector<int8_t> scratch;
  Span bytes;
};

// The B panel for `name`, in the order and width the ARRAY consumes.
//
// `a_elem_bytes` is the design's own A-operand width, passed in so the refusal
// for a mismatch lives where the bytes are about to be handed over rather than
// only at container-selection time. The container says `a_dtype "i8"` for an
// int4 pack, so design_selection already refuses it against a bf16 design by
// name; this is the same fact asserted at the last possible moment, because a
// widened panel handed to a design built for something else would be the same
// "right size, wrong meaning" shape as a layout mismatch -- and a layout
// mismatch is already a hard error three lines above every call of this
// function.
inline Panel gemm_b_panel(const File &model, const std::string &name,
                          size_t a_elem_bytes) {
  const TensorInfo &t = model.info(name);

  if (t.dtype != "I4") {
    Panel p;
    p.bytes = model.raw(name);
    return p;
  }

  if (a_elem_bytes != 1)
    throw std::runtime_error(
        name + ": I4 panel about to be staged against a design whose A operand "
        "is " + std::to_string(a_elem_bytes) + " bytes. int4 is weight storage "
        "over the int8 datapath -- there is no 4-bit MAC, so this widening "
        "still produces int8 values, and int8 values into a non-int8 design is "
        "a wrong number rather than a wrong width. Pair the container with an "
        "int8 design (it says a_dtype \"i8\"), or repack with --dtype bf16.");

  if (t.padded_shape.size() != 2)
    throw std::runtime_error(
        name + ": I4 payload has a " + std::to_string(t.padded_shape.size()) +
        "-dimensional padded shape; an int4 GEMM operand is a [K, N] panel");
  const int64_t K = t.padded_shape[0];
  const int64_t N = t.padded_shape[1];
  if (K <= 0 || N <= 0 || (K * N) % 2)
    throw std::runtime_error(
        name + ": I4 payload of [" + std::to_string(K) + "," +
        std::to_string(N) + "] holds an odd number of nibbles; two int4 values "
        "make a byte, so this is a shape bug and half of it would be a "
        "neighbour's bytes");

  if (t.layout_kind != "block_panel" || t.tile_k <= 0 || t.tile_n <= 0 ||
      t.mac_s <= 0 || t.mac_t <= 0)
    throw std::runtime_error(
        name + ": I4 payload carries no usable block_panel layout (kind '" +
        t.layout_kind + "'). The layout is what says where each nibble sits, "
        "so without it the bytes are not a weight -- they are the right count "
        "in an unknown order. Repack with tools/pack/pack_npue.py --dtype i4.");
  if (K % t.tile_k || N % t.tile_n || t.tile_k % t.mac_s ||
      t.tile_n % t.mac_t)
    throw std::runtime_error(
        name + ": I4 layout (" + std::to_string(t.tile_k) + "," +
        std::to_string(t.tile_n) + ") mac (" + std::to_string(t.mac_s) + "," +
        std::to_string(t.mac_t) + ") does not tile [" + std::to_string(K) +
        "," + std::to_string(N) + "]");

  const Span pk = model.raw(name);
  if (pk.bytes != static_cast<uint64_t>((K * N) / 2))
    throw std::runtime_error(
        name + ": I4 payload is " + std::to_string(pk.bytes) + " bytes but [" +
        std::to_string(K) + "," + std::to_string(N) + "] needs " +
        std::to_string((K * N) / 2) + " -- the payload and the panel disagree "
        "about their size, and half of it would be a neighbour's bytes");

  // The group size is config, not a guess and not derived from the sidecar's
  // height: the packer writes both and this is where they are made to agree.
  int64_t declared = 0;
  try {
    declared = model.config_int("int4_group");
  } catch (const std::exception &e) {
    throw std::runtime_error(
        name + ": I4 payload, but the container states no int4_group -- the "
        "group size says which rows of K share a scale, and without it a "
        "nibble has no way back to an int8 value. (" + std::string(e.what()) +
        ") Repack with tools/pack/pack_npue.py --dtype i4.");
  }
  if (declared < 0)
    throw std::runtime_error(
        name + ": config int4_group is " + std::to_string(declared) +
        "; rows per group must be >= 0, where 0 means one group over the "
        "whole K");
  const int64_t group = declared == 0 ? K : declared;
  const int64_t G = (K + group - 1) / group;

  const std::string gsname = name + ".gscale";
  if (!model.has(gsname))
    throw std::runtime_error(
        name + ": I4 payload but no " + gsname + " in the container. The "
        "group scale is the only thing that can put a nibble back in units of "
        + name + ".wscale, so without it these bytes are not a weight. Repack "
        "with tools/pack/pack_npue.py --dtype i4.");
  const TensorInfo &gi = model.info(gsname);
  if (gi.dtype != "F32" || gi.padded_shape.size() != 2 ||
      gi.padded_shape[0] != G || gi.padded_shape[1] != N ||
      gi.nbytes != static_cast<uint64_t>(G) * N * sizeof(float))
    throw std::runtime_error(
        name + ".gscale is " + gi.dtype + " [" +
        (gi.padded_shape.size() == 2
             ? std::to_string(gi.padded_shape[0]) + "," +
                   std::to_string(gi.padded_shape[1])
             : std::to_string(gi.padded_shape.size()) + "D") +
        "] but int4_group=" + std::to_string(declared) + " over K=" +
        std::to_string(K) + " needs F32 [" + std::to_string(G) + "," +
        std::to_string(N) + "] -- the panel and its scales disagree about the "
        "group size, and folding at the wrong one would weight every row by "
        "its neighbour's scale");

  const float *gs = model.raw(gsname).as<float>();
  const uint8_t *in = pk.as<uint8_t>();

  Panel p;
  p.scratch.assign(static_cast<size_t>(K) * static_cast<size_t>(N), 0);
  int8_t *out = p.scratch.data();

  // THE INDEX ARITHMETIC, and it is tile_b()'s flat order read backwards.
  // `tile_b` builds the payload as [kb, nb, r0, c0, s, t] (t fastest) out of
  // logical row k = kb*tile_k + r0*s + s_i and column j = nb*tile_n + c0*t +
  // t_i, so walking those six axes -- rather than dividing f apart per element
  // -- produces both the flat index and the (k,j) that gscale is indexed by.
  // Six divides per element over bge-large's 300M would be seconds spent in
  // divisions; this is a multiply per axis instead, and it is the same
  // arithmetic.
  const int64_t s = t.mac_s, tt = t.mac_t;
  const int64_t kb = K / t.tile_k, nb = N / t.tile_n;
  const int64_t r0 = t.tile_k / s, c0 = t.tile_n / tt;
  const int64_t stride_nb = r0 * c0 * s * tt;
  const int64_t stride_r0 = c0 * s * tt;
  const int64_t stride_c0 = s * tt;

  for (int64_t bki = 0; bki < kb; ++bki) {
    const int64_t base_kb = bki * nb * stride_nb;
    const int64_t k0 = bki * t.tile_k;
    for (int64_t bni = 0; bni < nb; ++bni) {
      const int64_t base_nb = base_kb + bni * stride_nb;
      const int64_t j0 = bni * t.tile_n;
      for (int64_t ri = 0; ri < r0; ++ri) {
        const int64_t base_r = base_nb + ri * stride_r0;
        const int64_t k_row = k0 + ri * s;
        for (int64_t ci = 0; ci < c0; ++ci) {
          const int64_t base_c = base_r + ci * stride_c0;
          const int64_t j_col = j0 + ci * tt;
          for (int64_t si = 0; si < s; ++si) {
            const int64_t base_s = base_c + si * tt;
            const int64_t k = k_row + si;
            // Per ROW, not per ri: a group (32 by default) is not necessarily a
            // multiple of mac_s (4 on npu1), so a stride-s run of rows can
            // straddle a group boundary and each row can belong to a different
            // one. Hoisting this one level higher would weight rows by their
            // neighbour's scale for any --int4-group the mac_s does not divide
            // -- silently, and with the file still the right size and hashed.
            const float *grow = gs + (k / group) * N;
            for (int64_t ti = 0; ti < tt; ++ti) {
              const int64_t f = base_s + ti;
              const uint8_t b = in[f >> 1];
              const int nv = (f & 1) ? int(b >> 4) : int(b & 0x0F);
              const float t_val = (nv > 7) ? float(nv - 16) : float(nv);
              float q = std::nearbyint(t_val * grow[j_col + ti]);
              if (q > 127.0f) q = 127.0f;
              else if (q < -127.0f) q = -127.0f;
              out[f] = static_cast<int8_t>(q);
            }
          }
        }
      }
    }
  }

  p.bytes.data = p.scratch.data();
  p.bytes.bytes = p.scratch.size();
  return p;
}

// Bytes stage() will hold for `name` -- the size of what the array receives,
// not of what the file stores. Equal to the entry's nbytes for BF16 and I8;
// for I4 it is twice nbytes, because the payload is half-width and the panel
// staged is a full int8 one. Three callers keep a running total of staged
// bytes and would each otherwise have to re-derive which of the two a dtype
// means, and getting it wrong is a number in a printout, not a crash -- which
// is exactly how it would stay wrong.
inline size_t staged_bytes(const File &model, const std::string &name) {
  const TensorInfo &t = model.info(name);
  if (t.dtype != "I4") return model.raw(name).bytes;
  if (t.padded_shape.size() != 2)
    throw std::runtime_error(name + ": I4 payload without a [K, N] shape");
  return static_cast<size_t>(t.padded_shape[0]) *
         static_cast<size_t>(t.padded_shape[1]);
}

}  // namespace npue

#endif  // NPU_EMBEDDINGS_COMMON_INT4_PANEL_HPP
