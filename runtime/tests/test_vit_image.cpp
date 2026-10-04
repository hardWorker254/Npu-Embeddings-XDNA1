//===- test_vit_image.cpp ------------------------------------------*- C++ -*-===//
//
// Runs the C++ image front end (decode -> resize -> normalise -> im2col) so
// tools/verify/verify_vit_image.py can hold it against PIL.
//
// Same arrangement as the Whisper front end's gate: a probe binary rather than
// the CLI, because the front end needs a container for its geometry and nothing
// else, and because the four steps must be observable SEPARATELY. "The label was
// wrong" is not a bug report; "the resized raster differs from PIL's by 1 LSB on
// 3% of channels" is, and it points at resample_axis() and nowhere else.
//
//   argv[1] <model.npue>     for the geometry (read_geometry, refusals and all)
//   argv[2] <image>          a PNG or a JPEG
//   --resample N             override the container's PIL code, 0..5
//   --side N                 override the container's image_size, so the gate
//                            can resize to sizes the container does not use.
//                            The im2col still uses the container's patch, so
//                            --side only makes sense at a whole number of
//                            patches; the gate uses multiples of 16.
//   --pool N                 run the resize a SECOND time on an N-thread pool and
//                            report whether the two rasters are byte-identical.
//                            Prints one extra line:
//                              poolmatch <same|differs> <n_bytes_differing>
//                            This is not a second claim about PIL. The serial
//                            result is the one PIL is compared against; this
//                            says that the threads do not change it, which is a
//                            different property and the one the pose front end
//                            actually depends on, since it is the only caller
//                            that passes a pool.
//   --keep-resized <path>    also write the resized uint8 raster here
//
// Output, all on stdout, one line per stage:
//   decoded <w> <h> <hex>          8-bit RGB straight out of the decoder
//   resize  <w> <h> <hex>          after the square resize, still 8-bit RGB
//   pixels  <hex>                  float32 [3,S,S], (x/255 - mean)/std
//   patches <rows> <cols> <hex>    float32 [n_patches, patch_dim]
//   refused <message>
//
// Hex, not decimal, for the same reason the Whisper gate uses it: these are
// 150528 and 150528 floats, and a decimal round trip through a pipe is a
// rounding step the gate would then have to reason about separately from the
// thing it is measuring.
//
// decoded and resize are 8-BIT on purpose. That is the whole claim in
// vit/image.hpp's header: the resize is claimed to be PIL-EQUIVALENT, not
// bit-exact, and the only way to state that with a number is to measure the
// worst channel difference and the fraction of channels that differ at all
// against PIL's own uint8 output. The float stages below then follow from
// whichever raster each side actually produced.
//
// Build:
//   g++ -std=c++17 -O2 -I runtime/include runtime/tests/test_vit_image.cpp \
//       runtime/src/vit/image.cpp runtime/src/common/json_min.cpp \
//       runtime/src/model.cpp runtime/src/pool.cpp \
//       -lpng -ljpeg -lpthread -o /tmp/test_vit_image
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/pool.hpp"
#include "vit/geometry.hpp"
#include "vit/image.hpp"

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

std::string float_hex(const std::vector<float> &v) {
  return to_hex(v.data(), v.size() * 4);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <model.npue> <image> [--resample N] [--side N] "
                 "[--pool N] [--keep-resized <path>]\n", argv[0]);
    return 2;
  }

  npue::vit::Geometry g;
  int pool_threads = 0;
  try {
    g = npue::vit::read_geometry(npue::File(argv[1]), argv[1]);
  } catch (const std::exception &e) {
    // A refusal from read_geometry is OUTPUT for this gate, not a crash: "this
    // container is refused with this message" is the property being checked.
    std::printf("refused %s\n", e.what());
    return 0;
  }
  // Flags scanned in ANY ORDER rather than by fixed argv slot, because the
  // gate passes all three and a positional scheme makes the order of a test's
  // own options load-bearing.
  for (int i = 3; i < argc - 1; ++i) {
    const std::string a = argv[i];
    if (a == "--resample") {
      const int code = std::atoi(argv[i + 1]);
      if (code < 0 || code > 5) {
        std::fprintf(stderr, "resample %d is not a PIL.Image code 0..5\n", code);
        return 2;
      }
      g.resample = static_cast<npue::vit::Resample>(code);
    } else if (a == "--side") {
      g.image_size = std::atoll(argv[i + 1]);
      if (g.image_size % g.patch_size) {
        std::fprintf(stderr,
                     "--side %lld is not a whole number of %lld-pixel patches\n",
                     (long long)g.image_size, (long long)g.patch_size);
        return 2;
      }
      const int64_t n = g.image_size / g.patch_size;
      g.n_patches = n * n;
    } else if (a == "--pool") {
      pool_threads = std::atoi(argv[i + 1]);
      if (pool_threads < 1) {
        std::fprintf(stderr, "--pool %d is not a thread count\n", pool_threads);
        return 2;
      }
    }
  }

  npue::vit::Image img;
  try {
    img = npue::vit::decode_image(argv[2]);
  } catch (const std::exception &e) {
    std::printf("refused %s\n", e.what());
    return 0;
  }
  std::printf("decoded %lld %lld %s\n", (long long)img.width,
              (long long)img.height,
              to_hex(img.rgb.data(), img.rgb.size()).c_str());

  npue::vit::Image sq;
  try {
    sq = npue::vit::resize_square(img, g.image_size, g.resample);
  } catch (const std::exception &e) {
    // NEAREST lands here. It is refused by name in image.cpp and the gate has
    // to see the NAME, not a nonzero exit code that hides the message.
    std::printf("refused %s\n", e.what());
    return 0;
  }
  std::printf("resize %lld %lld %s\n", (long long)sq.width, (long long)sq.height,
              to_hex(sq.rgb.data(), sq.rgb.size()).c_str());

  if (pool_threads > 0) {
    // resize_square does not take a pool -- only resize_to does, because only the
    // pose letterbox has one to give -- so the comparison calls resize_to with
    // the same target twice. Same arguments, so the serial arm here is the very
    // raster compared against PIL above, not a lookalike.
    app::Pool pool(pool_threads);
    const npue::vit::Image par =
        npue::vit::resize_to(img, g.image_size, g.image_size, g.resample, &pool);
    size_t bad = 0;
    if (par.rgb.size() != sq.rgb.size()) {
      bad = sq.rgb.size() > par.rgb.size() ? sq.rgb.size() : par.rgb.size();
    } else {
      for (size_t i = 0; i < sq.rgb.size(); ++i)
        if (sq.rgb[i] != par.rgb[i]) ++bad;
    }
    std::printf("poolmatch %s %zu\n", bad == 0 ? "same" : "differs", bad);
  }

  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--keep-resized" && i + 1 < argc) {
      std::ofstream f(argv[i + 1], std::ios::binary);
      f.write(reinterpret_cast<const char *>(sq.rgb.data()),
              static_cast<std::streamsize>(sq.rgb.size()));
    }

  const std::vector<float> chw = npue::vit::normalise(sq, g);
  std::printf("pixels %s\n", float_hex(chw).c_str());

  const std::vector<float> rows = npue::vit::im2col_patches(chw.data(), g);
  std::printf("patches %lld %lld %s\n", (long long)(rows.size() / g.patch_dim),
              (long long)g.patch_dim, float_hex(rows).c_str());
  return 0;
}
