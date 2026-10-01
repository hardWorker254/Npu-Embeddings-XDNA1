//===- test_vit_model.cpp ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the arch=5 parts of the runtime that need NO DEVICE.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//
//
// WHAT THIS IS, AND WHY IT IS SEPARATE FROM test_vit_image.cpp
// -------------------------------------------------------------
// test_vit_image.cpp holds the image front end against PIL: decode, resize,
// normalise, im2col. This one holds the three things behind it that a host-only
// box can still falsify, and that no other gate reaches:
//
//   geometry   read_geometry()'s arch=5 contract. Every tensor name, every
//              shape and every relationship the whole stack assumes is decided
//              there, and every one of them is silent when wrong. A gate that
//              only ever reads a GOOD container has an unknown sensitivity, so
//              the refusals are printed here as output for the gate to name,
//              exactly as the image gate treats them.
//   head       head_matvec(). `classifier.weight` is [d_model, num_labels]
//              row-major, so label j is a stride-num_labels walk. Read it the
//              other way round -- which is how torch's nn.Linear stores it, and
//              so how the CHECKPOINT stores it -- and the result is a finite,
//              plausible, entirely wrong set of logits whose argmax is another
//              label. Nothing else in the stack can catch that: every shape
//              agrees. This is the only place in the tree where a transposition
//              produces a confident answer, so it is the place worth a gate.
//   i8         The int8 host kernels on a REAL ViT operand. quantise_a_int8 is
//              shared with arch=1/2 and verify_i8_kernels.py already holds it
//              bit-for-byte across three builds; what is NOT held anywhere is
//              that it is fed the operand ViT actually feeds it -- 197 rows of
//              768, an odd row count and an odd K that a synthetic matrix with
//              round numbers would never exercise the tail of.
//
// THE WRONG-STRIDE LINE IS DELIBERATE
// -----------------------------------
// `head_wrong_stride` is the transpose of the right answer, computed here on
// purpose. The gate requires it to DIFFER from `head`. Without that the gate
// cannot tell a correct head from a probe that printed zeros twice, which is
// the difference between a gate and a smoke test.
//
// OUTPUT
// ------
//   geometry <key>=<value> ...      one line, or `refused <message>`
//   cls_row <768 float32 as hex>    the CLS row the head was fed
//   head <1000 float32 as hex>      the container's own head, that row
//   head_split <same>               the same labels one at a time
//   head_wrong_stride <same>        the transpose -- MUST differ from `head`
//   i8_q <197*768 int8 as hex>      position_embeddings, quantised
//   i8_scale <197 float32 as hex>   ... and its per-row scales
//   i8_out <197*768 float32 as hex> dequantise_c on a deterministic int32 C
//
// Hex, not decimal, for the same reason as the image gate: 151296 floats do not
// survive a decimal round trip through a pipe without a rounding step the gate
// would then have to reason about separately from what it is measuring.
//
// Build (no NPU, no XRT, no design set; needs a packed container):
//   g++ -std=c++17 -O2 -I runtime/include runtime/tests/test_vit_model.cpp \
//       runtime/src/vit/head.cpp runtime/src/common/json_min.cpp \
//       runtime/src/model.cpp -o /tmp/test_vit_model
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Deliberately NOT including vit/encoder.hpp: it would pull the AIE stack in
// for twelve lines of arithmetic that live in vit/head.cpp precisely so a
// host-only gate does not have to.
#include "common/host_kernels.hpp"
#include "runtime/model.hpp"
#include "vit/geometry.hpp"
#include "vit/head.hpp"

namespace {

std::string to_hex(const void *data, size_t bytes) {
  static const char *d = "0123456789abcdef";
  const auto *p = static_cast<const unsigned char *>(data);
  std::string out;
  out.reserve(bytes * 2);
  for (size_t i = 0; i < bytes; ++i) {
    out.push_back(d[p[i] >> 4]);
    out.push_back(d[p[i] & 0xF]);
  }
  return out;
}

// A CLS row that is not constant and not smooth. A constant row makes every
// label's logit a multiple of the same number, which is a weaker test than it
// looks: a transposed head against a constant row produces constant-per-label
// logits too, and the argmax can survive. A deterministic LCG keeps the gate
// reproducible without the two sides having to agree on a random seed, which is
// one more thing to get wrong.
std::vector<float> make_cls_row(int64_t d) {
  std::vector<float> v(static_cast<size_t>(d));
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int64_t i = 0; i < d; ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    // (s >> 40) is 24 bits, so this is in [-1, 1) with no float surprise.
    v[static_cast<size_t>(i)] =
        static_cast<float>(static_cast<int32_t>(s >> 40)) / 8388608.0f - 1.0f;
  }
  return v;
}

// The same kind of generator for the synthetic int32 C. Fixed values rather
// than anything derived from the container, so the gate can compute the
// reference from the printed hex alone if it wants to.
int32_t lcg_i32(uint64_t &s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return static_cast<int32_t>(static_cast<uint32_t>(s >> 32));
}

// A buffer whose DATA is 64-byte aligned, which is what dequantise_c's
// stream-load precondition needs and what a std::vector<int32_t> does NOT give
// on this glibc (16, so the very first call refused). The runtime never sees
// this problem because its C is a DMA slot the driver page-aligns.
//
// Over-allocate by one line and hand back the rounded-up pointer, so the
// storage is a plain std::vector and there is no paired new/delete to get wrong
// in a probe. The offset is in ELEMENTS, not bytes: the first version computed
// it in bytes, added 48 to a `T*` and got p + 192, which is 16 mod 64 -- the
// same failure with the assertion still in place, which is what the assertion
// is for. The alignment is ASSERTED rather than assumed: a probe that silently
// handed dequantise_c a misaligned buffer and reported the refusal as a result
// would be worse than no probe.
template <typename T>
class AlignedBuf {
public:
  explicit AlignedBuf(size_t n) : n_(n), raw_(n + 16) {
    const auto p = reinterpret_cast<uintptr_t>(raw_.data());
    const uintptr_t aligned = (p + 63u) & ~static_cast<uintptr_t>(63);
    off_ = static_cast<size_t>(aligned - p) / sizeof(T);
  }
  T *data() { return raw_.data() + off_; }
  const T *data() const { return raw_.data() + off_; }
  bool aligned64() const {
    return (reinterpret_cast<uintptr_t>(data()) & 63u) == 0;
  }

private:
  size_t n_, off_ = 0;
  std::vector<T> raw_;
};

void print_geometry(const npue::vit::Geometry &g) {
  std::printf("geometry");
  auto i = [&](const char *k, int64_t v) { std::printf(" %s=%lld", k, (long long)v); };
  auto f = [&](const char *k, double v) { std::printf(" %s=%.17g", k, v); };
  i("d_model", g.d_model);
  i("heads", g.heads);
  i("head_dim", g.head_dim);
  i("layers", g.layers);
  i("intermediate", g.intermediate);
  i("num_labels", g.num_labels);
  i("image_size", g.image_size);
  i("crop_size", g.crop_size);
  i("patch_size", g.patch_size);
  i("num_channels", g.num_channels);
  i("patch_dim", g.patch_dim);
  i("n_patches", g.n_patches);
  i("n_pos", g.n_pos);
  std::printf(" mean=");
  for (size_t k = 0; k < g.mean.size(); ++k)
    std::printf("%s%.17g", k ? "," : "", g.mean[k]);
  std::printf(" std=");
  for (size_t k = 0; k < g.std_dev.size(); ++k)
    std::printf("%s%.17g", k ? "," : "", g.std_dev[k]);
  std::printf(" resample=%d ln_eps=%.17g attn_scale=%.17g qkv_scale_folded=%d\n",
              static_cast<int>(g.resample), g.ln_eps, g.attn_scale,
              g.qkv_scale_folded ? 1 : 0);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.npue> [--section geometry|head|i8]\n",
                 argv[0]);
    return 2;
  }
  std::string only;
  for (int i = 2; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--section") == 0) only = argv[i + 1];
  const auto want = [&](const char *s) { return only.empty() || only == s; };

  npue::File model(argv[1]);
  npue::vit::Geometry g;
  try {
    g = npue::vit::read_geometry(model, argv[1]);
  } catch (const std::exception &e) {
    // A refusal here is OUTPUT, not a crash: "this container is refused with
    // this message" is one of the properties the gate checks, and the gate
    // passes deliberately wrong containers to find out whether the refusal is
    // specific enough to act on.
    std::printf("refused %s\n", e.what());
    return 0;
  }
  if (want("geometry")) print_geometry(g);

  const int64_t d = g.d_model, nlab = g.num_labels;

  // ---- the head, on the container's own weights -------------------------
  if (want("head")) {
    if (!model.has("classifier.weight") || !model.has("classifier.bias")) {
      std::printf("refused %s: the classifier head is not in the container. "
                  "It is the one operand that never reaches the device, so "
                  "there is nothing to hold the matvec against.\n", argv[1]);
      return 0;
    }
    const auto wi = model.info("classifier.weight");
    if (wi.dtype != "F32" || wi.logical_shape != std::vector<int64_t>{d, nlab}) {
      std::printf("refused classifier.weight is [%lld, %lld] %s, not [%lld, "
                  "%lld] F32 -- the matvec's stride is only defined for that "
                  "layout.\n",
                  wi.logical_shape.size() > 0 ? (long long)wi.logical_shape[0] : -1,
                  wi.logical_shape.size() > 1 ? (long long)wi.logical_shape[1] : -1,
                  wi.dtype.c_str(), (long long)d, (long long)nlab);
      return 0;
    }
    const float *head = model.raw("classifier.weight").as<float>();
    const float *bias = model.raw("classifier.bias").as<float>();
    const std::vector<float> row = make_cls_row(d);

    // The row is printed so the gate does not have to re-derive the generator
    // to check the matvec. A gate that reimplemented the LCG would be holding
    // two implementations of an LCG against each other, which is a different
    // and much weaker claim than "the matvec is right".
    std::printf("cls_row %s\n", to_hex(row.data(), row.size() * 4).c_str());

    std::vector<float> got(static_cast<size_t>(nlab), 0.f);
    npue::vit::head_matvec(head, bias, row.data(), d, nlab, 0, nlab, got.data());
    std::printf("head %s\n", to_hex(got.data(), got.size() * 4).c_str());

    // The same labels one at a time. head_matvec's contract says label j's
    // result does not depend on the split; this is how the gate checks it,
    // because VitEncoder::classify splits the labels across the pool and a
    // gate that only ever saw the single-threaded path would not know.
    //
    // The base pointer is the WHOLE logits array even though the block is one
    // label: head_matvec indexes logits, bias and head by the ABSOLUTE label j,
    // because those three all share one index space and bias[j] vs
    // bias[j - j0] is exactly the kind of offset a block API invites someone to
    // get wrong. That is why the block is a range and not a pointer pair.
    std::vector<float> split(static_cast<size_t>(nlab), 0.f);
    for (int64_t j = 0; j < nlab; ++j)
      npue::vit::head_matvec(head, bias, row.data(), d, nlab, j, j + 1,
                             split.data());
    std::printf("head_split %s\n", to_hex(split.data(), split.size() * 4).c_str());

    // The transpose of the right answer, on purpose. See the header.
    std::vector<float> wrong(static_cast<size_t>(nlab), 0.f);
    for (int64_t j = 0; j < nlab; ++j) {
      double acc = bias[j];
      for (int64_t t = 0; t < d; ++t)
        acc += static_cast<double>(row[t]) *
               static_cast<double>(head[j * d + t]);  // <-- the stride, swapped
      wrong[static_cast<size_t>(j)] = static_cast<float>(acc);
    }
    std::printf("head_wrong_stride %s\n",
                to_hex(wrong.data(), wrong.size() * 4).c_str());
  }

  // ---- the int8 host kernels on a real ViT operand ----------------------
  if (want("i8")) {
    // frontend.position_embeddings is the one [n_pos, d_model] F32 table the
    // stack hands straight to quantise_a_int8 as an activation, so it is the
    // honest operand: 197 rows (an odd count, so the row loop's tail runs) of
    // 768 (a multiple of 8, so the AVX2 body runs to the end and the scalar
    // tail never executes on this host -- which is why the gate also builds the
    // scalar-only binary and compares).
    if (!model.has("frontend.position_embeddings")) {
      std::printf("refused %s: frontend.position_embeddings is absent, so "
                  "there is no ViT activation to quantise.\n", argv[1]);
      return 0;
    }
    const auto pi = model.info("frontend.position_embeddings");
    if (pi.dtype != "F32" || pi.logical_shape.size() != 2 ||
        pi.logical_shape[0] != g.n_pos || pi.logical_shape[1] != d) {
      std::printf("refused frontend.position_embeddings is not [%lld, %lld] "
                  "F32.\n", (long long)g.n_pos, (long long)d);
      return 0;
    }
    const int64_t rows = g.n_pos, K = d;
    const float *a = model.raw("frontend.position_embeddings").as<float>();

    // The divisor. ViT's patch_embed carries its own asmooth and it is exactly
    // K wide, so the container's own is used rather than a synthetic one --
    // an all-ones ias would make the quantiser's per-row max trivially the
    // row's own max and would not exercise the fold at all.
    std::vector<float> ias(static_cast<size_t>(K), 1.0f);
    if (model.has("frontend.patch_embed.asmooth")) {
      const auto ai = model.info("frontend.patch_embed.asmooth");
      if (ai.dtype == "F32" && ai.logical_shape.size() == 1 &&
          ai.logical_shape[0] == K) {
        const float *w = model.raw("frontend.patch_embed.asmooth").as<float>();
        for (int64_t j = 0; j < K; ++j) ias[static_cast<size_t>(j)] = 1.0f / w[j];
        std::printf("i8_asmooth from the container\n");
      }
    } else {
      std::printf("i8_asmooth all ones: this container is not int8, so the "
                  "divisor is synthetic and the row maxima are the plain "
                  "per-row maxima\n");
    }

    std::vector<int8_t> q(static_cast<size_t>(rows * K));
    std::vector<float> scale(static_cast<size_t>(rows));
    app::quantise_a_int8(a, rows, K, ias.data(), q.data(), scale.data(),
                          [](int64_t n, auto &&f) { f(0, n); });
    std::printf("i8_q %s\n", to_hex(q.data(), q.size()).c_str());
    std::printf("i8_scale %s\n", to_hex(scale.data(), scale.size() * 4).c_str());

    // dequantise_c on a DETERMINISTIC int32 C of ViT shape, with the
    // container's own wscale and bias for patch_embed where they exist. The C
    // is synthetic because a real one is the AIE's accumulator and there is no
    // AIE here; what this holds is the dequantise FORMULA and the two
    // refusals around it, which is all that is host-side.
    const int64_t N = d;
    AlignedBuf<int32_t> c(static_cast<size_t>(rows * N));
    if (!c.aligned64()) {
      std::printf("i8_refused the probe could not build a 64-byte-aligned C, so "
                  "the result below would be dequantise_c's ALIGNMENT refusal "
                  "and not its arithmetic\n");
      return 0;
    }
    uint64_t s = 0xDEADBEEFCAFEF00Dull;
    for (int64_t i = 0; i < rows * N; ++i)
      c.data()[i] = lcg_i32(s) & 0x00FFFFFF;  // 24 bits: inside one bf16's exponent
    std::vector<float> wscale(static_cast<size_t>(N), 1.0f), cbias(static_cast<size_t>(N), 0.f);
    if (model.has("frontend.patch_embed.wscale") &&
        model.has("frontend.patch_embed.bias")) {
      const float *ws = model.raw("frontend.patch_embed.wscale").as<float>();
      const float *bs = model.raw("frontend.patch_embed.bias").as<float>();
      for (int64_t j = 0; j < N; ++j) {
        wscale[static_cast<size_t>(j)] = ws[j];
        cbias[static_cast<size_t>(j)] = bs[j];
      }
      std::printf("i8_wscale from the container\n");
    }
    std::vector<float> out(static_cast<size_t>(rows * N));
    try {
      app::dequantise_c(c.data(), 4, rows, N, scale.data(), wscale.data(),
                         cbias.data(), out.data(),
                         [](int64_t n, auto &&f) { f(0, n); });
      std::printf("i8_out %s\n", to_hex(out.data(), out.size() * 4).c_str());
    } catch (const std::exception &e) {
      std::printf("i8_refused %s\n", e.what());
    }

    // The alignment precondition, on a ViT N that is NOT a multiple of 8. Every
    // ViT N this tree packs (768, 2304, 3072) satisfies it, so the refusal is
    // only reachable by asking -- which is the point: the gate has to know that
    // a container asking for a different hidden size would be refused here
    // rather than faulting on an AVX2 host and passing on a scalar one. Note
    // this is only a REFUSAL under -mavx2; the scalar build has no such
    // precondition, which is why verify_i8_kernels.py builds both and this gate
    // records which build it saw.
    //
    // The bias is a real buffer, not nullptr: dequantise_c's `bias` is not
    // nullable and faults on one in BOTH branches (see host_kernels.hpp's
    // header). The first version of this probe passed null, and the avx2 build
    // SIGSEGVed on the row -- which is how that note got written.
    {
      const int64_t bad = d + 1;  // 769, stride 3076 = 4 mod 32
      AlignedBuf<int32_t> bc(static_cast<size_t>(bad));
      std::vector<float> bo(static_cast<size_t>(bad), 0.f);
      std::vector<float> bbias(static_cast<size_t>(bad), 0.f);
      try {
        app::dequantise_c(bc.data(), 4, 1, bad, scale.data(), wscale.data(),
                          bbias.data(), bo.data(),
                          [](int64_t n, auto &&f) { f(0, n); });
        std::printf("i8_align accepted N=%lld\n", (long long)bad);
      } catch (const std::exception &e) {
        std::printf("i8_align refused N=%lld %s\n", (long long)bad, e.what());
      }
    }
  }
  return 0;
}
