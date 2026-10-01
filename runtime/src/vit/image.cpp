//===- image.cpp --------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the image front end. See vit/image.hpp for the four steps
// and why each is its own function.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "vit/image.hpp"

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

extern "C" {
#include <jpeglib.h>
#include <png.h>
}

namespace npue::vit {
namespace {

std::vector<uint8_t> slurp(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open image file " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

// The filter kernels, transcribed from PIL's own Filters.c.
//
// THESE ARE FIVE DIFFERENT FUNCTIONS, and the only thing they share is a name.
// The first version of this file implemented all five as the Keys cubic
// ((a+2)x - (a+3))x^2 + 1 on |x|<1, etc.) with `support` and `a` as parameters,
// which is correct for BICUBIC alone and wrong for the other four by a wide
// margin -- measured, on random noise, a worst channel difference of 112 for
// BILINEAR and 65 for LANCZOS. PIL's actual definitions:
//
//   BOX       1.0 on |x| <= 0.5                       (a RECTANGLE, not a cubic)
//   BILINEAR  1 - |x| on |x| < 1                      (a triangle)
//   BICUBIC   Keys a = -0.5 on |x| < 2                (this is the only cubic)
//   LANCZOS   sinc(x)*sinc(x/3) on |x| < 3             (a windowed sinc, which
//                                                        no Keys parameterisation
//                                                        can express at all)
//
// HAMMING is ABSENT on purpose -- see the refusal in resize_square().
//
// The gate that caught this is tools/verify/verify_vit_image.py section 2, run
// with --codes 1,2,4,5. BICUBIC alone passing is not evidence for the others: it
// is evidence that the INDEX arithmetic is right, which is the part that is
// shared, and it is exactly why the four that were wrong went unnoticed.
//
// `support` is the half-width in SOURCE pixels BEFORE the scale widening.
struct Kernel {
  double support;
  double (*eval)(double);
};

double box_eval(double x) {
  x = std::fabs(x);
  return x <= 0.5 ? 1.0 : 0.0;
}
double bilinear_eval(double x) {
  x = std::fabs(x);
  return x < 1.0 ? 1.0 - x : 0.0;
}
double bicubic_eval(double x) {
  x = std::fabs(x);
  if (x < 1.0) return ((-0.5 + 2.0) * x - (-0.5 + 3.0)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * (-0.5);
  return 0.0;
}
double lanczos_eval(double x) {
  x = std::fabs(x);
  if (x < 3.0) {
    const double pix = M_PI * x;
    return std::sin(pix) * std::sin(pix / 3.0) / (pix * (pix / 3.0));
  }
  return 0.0;
}
Kernel kernel_for(Resample r, int64_t *support_scale) {
  switch (r) {
    // NEAREST is the identity filter at zero width, which the generic path
    // below handles correctly with support = 0.5 and the rectangle kernel --
    // see the refusal in resize(); NEAREST is refused there rather than
    // approximated here.
    case Resample::Nearest: { *support_scale = 1; return {0.5, box_eval}; }
    case Resample::Bilinear: { *support_scale = 1; return {1.0, bilinear_eval}; }
    case Resample::Bicubic: { *support_scale = 1; return {2.0, bicubic_eval}; }
    case Resample::Box: { *support_scale = 1; return {0.5, box_eval}; }
    // HAMMING is HERE ONLY so the switch is total, and unreachable: both
    // refusals in resize_square() fire before this table is consulted. It
    // returns BILINEAR's kernel rather than a invented one, because an
    // unreachable arm has no business carrying a guess.
    case Resample::Hamming: { *support_scale = 1; return {1.0, bilinear_eval}; }
    case Resample::Lanczos: { *support_scale = 1; return {3.0, lanczos_eval}; }
  }
  { *support_scale = 1; return {1.0, bilinear_eval}; }
}

// One separable pass, resampling `in_len` -> `out_len` along one axis of an
// interleaved RGB buffer. This is PIL's ImagingResampleHorizontal/Vertical
// without the fixed-point: float accumulation, and the same index arithmetic.
//
// The index arithmetic IS the load-bearing part and is transcribed exactly:
//
//   scale    = in_len / out_len
//   filterscale = max(1, scale)          <- the antialias widening
//   support  = kernel.support * filterscale
//   centre   = (i + 0.5) * scale
//   xmin     = int(centre - support + 0.5), clamped to [0, in_len)
//   xmax     = int(centre + support + 0.5), clamped to [0, in_len)
//   weight   = kernel.eval((x - centre + 0.5) / filterscale)
//
// and the output is sum(w*v)/sum(w), not a plain mean, so a pixel that straddles
// an edge keeps its level.
void resample_axis(const uint8_t *in, int64_t in_len, uint8_t *out,
                   int64_t out_len, int64_t stride_in, int64_t stride_out,
                   int64_t nchan, const Kernel &kern, double filterscale) {
  const double scale = static_cast<double>(in_len) / static_cast<double>(out_len);
  const double support = kern.support * filterscale;
  const double inv_filter = 1.0 / filterscale;
  // Worst case one source pixel contributes to every output pixel inside a
  // support of `support`. It is computed from kern.support rather than written
  // as "2.0 * filterscale", which was BICUBIC's number and would have silently
  // truncated a LANCZOS window (support 3.0) by a third -- the weights beyond
  // the bound would be dropped, which is not a rounding difference but a
  // different filter.
  const int64_t kmax =
      std::min<int64_t>(in_len, static_cast<int64_t>(std::ceil(support * 2.0)) + 2);
  std::vector<double> w(static_cast<size_t>(kmax));
  for (int64_t i = 0; i < out_len; ++i) {
    const double centre = (static_cast<double>(i) + 0.5) * scale;
    int64_t xmin = static_cast<int64_t>(centre - support + 0.5);
    if (xmin < 0) xmin = 0;
    int64_t xmax = static_cast<int64_t>(centre + support + 0.5);
    if (xmax > in_len) xmax = in_len;
    const int64_t n = xmax - xmin;
    if (n <= 0) {   // cannot happen for support >= 0.5 and scale > 0, but a
                    // zero-weight row would divide by zero below, and a NaN
                    // pixel is the worst thing this function can produce
      for (int64_t c = 0; c < nchan; ++c)
        out[i * stride_out + c] = 0;
      continue;
    }
    if (n > kmax)
      throw std::runtime_error(
          "resample: a filter window of " + std::to_string(n) +
          " source pixels exceeds this build's bound of " +
          std::to_string(kmax) +
          ". Raising kmax is the fix; truncating the window would drop edge "
          "pixels' worth of weight and shift the image by half a pixel.");
    double ww = 0.0;
    for (int64_t x = 0; x < n; ++x) {
      const double v = kern.eval((static_cast<double>(x + xmin) - centre + 0.5) *
                                 inv_filter);
      w[static_cast<size_t>(x)] = v;
      ww += v;
    }
    if (ww == 0.0) ww = 1.0;   // a filter that sums to zero is a bug, not a
                              // division to perform
    for (int64_t c = 0; c < nchan; ++c) {
      double acc = 0.0;
      const uint8_t *base = in + c;
      for (int64_t x = 0; x < n; ++x)
        acc += w[static_cast<size_t>(x)] *
               static_cast<double>(base[(xmin + x) * stride_in]);
      double v = acc / ww;
      // Round half away from zero, then clamp. PIL's 8-bit path does the same
      // clamp with its fixed-point rounding; where the two differ it is in the
      // last bit of the fixed-point sum, which is what the test measures.
      const double r = v < 0.0 ? v - 0.5 : (v > 0.0 ? v + 0.5 : v);
      const long long q = static_cast<long long>(r);
      out[i * stride_out + c] =
          static_cast<uint8_t>(std::min<long long>(255, std::max<long long>(0, q)));
    }
  }
}

}  // namespace

Image decode_png(const std::vector<uint8_t> &bytes, const std::string &name) {
  if (png_sig_cmp(const_cast<png_bytep>(bytes.data()), 0, 8) != 0)
    throw std::runtime_error(name + ": not a PNG (bad signature)");

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                           nullptr, nullptr);
  if (!png) throw std::runtime_error("libpng: out of memory");
  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_read_struct(&png, nullptr, nullptr);
    throw std::runtime_error("libpng: out of memory");
  }

  // The reader's whole state: a cursor into the caller's byte vector. It lives
  // HERE, before the setjmp, because png_set_read_fn keeps the pointer for the
  // rest of the decode and libpng calls back into it after a longjmp too.
  struct MemReader {
    const uint8_t *p;
    size_t n, at;
  } reader{bytes.data(), bytes.size(), 0};

  // EVERY non-trivial local is declared and constructed BEFORE the setjmp
  // below. libpng reports a fatal error by longjmp'ing, and a longjmp over a
  // frame in which a std::vector's constructor has not run leaves that vector
  // "existing" without ever having run -- the destructor then reads a garbage
  // size. Declaring them first is the only ordering that is well defined, and
  // the comment is here because the natural-looking version (declare and fill
  // in one go, setjmp at the top) is the broken one.
  Image out;
  std::vector<std::vector<uint8_t>> buf;
  std::vector<png_bytep> rows;
  std::string failure;

  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, nullptr);
    throw std::runtime_error(name + ": " +
                             (failure.empty() ? "PNG is corrupt or truncated"
                                              : failure));
  }
  // The default handlers print to stderr and longjmp with whatever the
  // png_error_ptr said, which is a bare `const char*` that dies with the
  // frame. This one keeps the message so the caller learns WHICH byte count or
  // WHICH IDAT was wrong instead of "corrupt or truncated".
  png_set_error_fn(png, &failure,
                   [](png_structp p, png_const_charp msg) {
                     auto *keep = static_cast<std::string *>(png_get_error_ptr(p));
                     if (keep) *keep = msg ? msg : "libpng error";
                     png_longjmp(p, 1);
                   },
                   nullptr);

  // The read source. WITHOUT THIS libpng falls back to png_init_io(), which
  // calls fread() on a FILE* it never got -- so the FIRST read dereferences
  // null and the process dies with SIGSEGV, on every PNG, before any of this
  // file's own error handling can run. A memory reader is the whole reason the
  // decoder takes a byte vector rather than a path.
  png_set_read_fn(png, &reader,
                  [](png_structp p, png_bytep out, png_size_t want) {
                    auto *r = static_cast<MemReader *>(png_get_io_ptr(p));
                    const size_t got = r->n - r->at < want ? r->n - r->at : want;
                    std::memcpy(out, r->p + r->at, got);
                    r->at += got;
                    // A short read is png_error's business, not a silent zero:
                    // returning fewer bytes than asked for without complaint is
                    // how a truncated PNG becomes a half-grey image.
                    if (got != want) png_error(p, "unexpected end of PNG data");
                  });

  png_read_info(png, info);

  png_uint_32 w = 0, h = 0;
  int bit_depth = 0, colour = 0;
  png_get_IHDR(png, info, &w, &h, &bit_depth, &colour, nullptr, nullptr, nullptr);

  // A 16-BIT PNG IS REFUSED, and the reason is a measured disagreement rather
  // than a preference.
  //
  // There are two ways to turn 16 bits into 8 and this build cannot pick
  // between them by looking at the file:
  //
  //   libpng's png_set_strip_16 drops the LOW byte -- the high byte, a correct
  //   16->8 reduction. Measured against PIL on the same bytes, this agrees with
  //   it exactly for values below 256 and disagrees above.
  //   PIL's I;16 -> "RGB" CLIP8s: 0->0, 255->255, and everything above 255 ->
  //   255. Measured, in tools/verify/verify_vit_image.py section 1: a 16-bit
  //   gradient comes out as a near-white image, because PIL treats I;16 as a
  //   SIGNED 16-bit mode and clips rather than shifts.
  //
  // transformers' image processor calls .convert("RGB"), so the PIL answer is
  // what the model was trained to see -- and it is a nearly white picture. This
  // build will not hand the model a different image from the one the reference
  // produces, and it will not silently clip a 16-bit scan to white either, which
  // is what matching PIL would mean. Both are guesses about which wrong image
  // is less wrong, so the file is named instead. Re-saving as 8-bit is one
  // command in anything that opens the image.
  if (bit_depth == 16)
    throw std::runtime_error(
        name + ": PNG is 16 bits per channel and this runtime reads 8-bit "
        "images. It will not choose for you: dropping the low byte (what "
        "libpng's strip-16 does, and a correct 16->8 reduction) and what PIL "
        "does -- .convert(\"RGB\") on an I;16 image, which CLIP8s every value "
        "above 255 to white -- give DIFFERENT images, and PIL's is the one "
        "transformers feeds the model, so the two answers are a near-correct "
        "picture and a near-white one. Re-save the image as 8-bit PNG or JPEG "
        "and classify that.");
  if (colour == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
  if (colour == PNG_COLOR_TYPE_GRAY || colour == PNG_COLOR_TYPE_GRAY_ALPHA) {
    if (bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    // Grayscale becomes RGB by REPLICATION, which is what PIL's convert("RGB")
    // does and therefore what transformers' image processor feeds the model.
    //
    // It has to be png_set_gray_to_rgb and NOT png_set_add_alpha: adding an
    // opaque alpha to a GRAY image yields GA, TWO channels, and the refusal
    // below then rejects every grayscale PNG in the world -- which is a
    // photograph, a screenshot and a scan, and the check cannot tell a real
    // two-channel image from a grayscale one libpng was told to mis-handle.
    // GRAY_ALPHA is here for the same reason: an LA PNG is two channels going
    // in, and png_set_gray_to_rgb is what turns it into RGBA rather than
    // leaving it for the refusal to reject.
    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
      // tRNS on a gray image is a transparent-COLOR key. It has to become a
      // real alpha AFTER the gray has become RGB, or the key is expanded
      // against a two-channel image and comes out as a black-and-alpha pair.
      png_set_gray_to_rgb(png);
      png_set_tRNS_to_alpha(png);
    } else {
      png_set_gray_to_rgb(png);
    }
  } else if (png_get_valid(png, info, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(png);
  }
  png_set_interlace_handling(png);
  png_read_update_info(png, info);

  const int channels = png_get_channels(png, info);
  if (channels != 3 && channels != 4)
    throw std::runtime_error(name + ": PNG expands to " +
                             std::to_string(channels) +
                             " channels, which is neither RGB nor RGBA");
  buf.resize(h);
  rows.resize(h);
  const size_t rowbytes = png_get_rowbytes(png, info);
  for (png_uint_32 y = 0; y < h; ++y) {
    buf[y].resize(rowbytes);
    rows[y] = buf[y].data();
  }
  png_read_image(png, rows.data());
  png_read_end(png, nullptr);

  out.width = w;
  out.height = h;
  out.rgb.resize(static_cast<size_t>(w) * h * 3);
  for (png_uint_32 y = 0; y < h; ++y) {
    const uint8_t *s = buf[y].data();
    uint8_t *d = out.rgb.data() + static_cast<size_t>(y) * w * 3;
    if (channels == 3) {
      std::memcpy(d, s, static_cast<size_t>(w) * 3);
    } else {
      // Composite onto WHITE. A viewer shows a transparent PNG over whatever is
      // behind it, and in every document and browser that is white; treating
      // alpha as black instead silently turns a cut-out logo into a dark
      // rectangle and then classifies it as one.
      for (png_uint_32 x = 0; x < w; ++x) {
        const unsigned a = s[static_cast<size_t>(x) * 4 + 3];
        for (int c = 0; c < 3; ++c) {
          const unsigned v = s[static_cast<size_t>(x) * 4 + c];
          d[x * 3 + c] =
              static_cast<uint8_t>((v * a + 255u * (255u - a)) / 255u);
        }
      }
    }
  }
  png_destroy_read_struct(&png, &info, nullptr);
  return out;
}

namespace {

// libjpeg's error handler: format the message into the Jerr we own, then
// longjmp. Written as a free function because <jpeglib.h> is C and an
// error_exit member would have to be a plain function pointer.
struct Jerr {
  struct jpeg_error_mgr pub;
  jmp_buf jump;
  std::string message;
};

void jpeg_error_exit(j_common_ptr cinfo) {
  Jerr *e = reinterpret_cast<Jerr *>(cinfo->err);
  char buf[JMSG_LENGTH_MAX];
  (*cinfo->err->format_message)(cinfo, buf);
  e->message = buf;
  longjmp(e->jump, 1);
}

}  // namespace

Image decode_jpeg(const std::vector<uint8_t> &bytes, const std::string &name) {
  jpeg_decompress_struct cinfo;
  // Same frame discipline as decode_png: cinfo and the error struct and every
  // buffer are constructed before setjmp, because libjpeg longjmps out of its
  // own frames and a local whose constructor has not run must not be
  // destroyed on the way back.
  Jerr jerr;
  // libjpeg's defaults, into a SEPARATE object, then copied. jpeg_std_error
  // writes six-odd function pointers and two counters THROUGH the pointer it is
  // given, so passing null dereferences it immediately -- and passing
  // `&jerr.pub` is a self-assignment, which works and reads as the accident it
  // is. The separate object is what "start from libjpeg's defaults" means.
  {
    struct jpeg_error_mgr defaults;
    jerr.pub = *jpeg_std_error(&defaults);
  }
  jerr.pub.error_exit = jpeg_error_exit;
  jerr.message.clear();
  std::vector<uint8_t> row;

  cinfo.err = &jerr.pub;
  jpeg_create_decompress(&cinfo);
  struct Guard {
    jpeg_decompress_struct *c;
    ~Guard() { jpeg_destroy_decompress(c); }
  } guard{&cinfo};

  if (setjmp(jerr.jump))
    throw std::runtime_error(name + ": " +
                             (jerr.message.empty() ? "JPEG decode failed"
                                                   : jerr.message));
  jpeg_mem_src(&cinfo, bytes.data(), static_cast<unsigned long>(bytes.size()));
  if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK)
    throw std::runtime_error(name + ": JPEG header is not readable");

  // Grayscale and CMYK JPEGs to RGB, so the normalise step's per-channel mean
  // and std both exist. cinfo.out_color_space is libjpeg's documented way and
  // YCbCr -> RGB is the transform it applies.
  cinfo.out_color_space = JCS_RGB;
  jpeg_start_decompress(&cinfo);

  Image out;
  out.width = cinfo.output_width;
  out.height = cinfo.output_height;
  if (cinfo.output_components != 3)
    throw std::runtime_error(name + ": JPEG decodes to " +
                             std::to_string(cinfo.output_components) +
                             " components; this runtime asked libjpeg for RGB");
  out.rgb.resize(static_cast<size_t>(out.width) * out.height * 3);
  row.resize(static_cast<size_t>(out.width) * 3);
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW rp = row.data();
    if (jpeg_read_scanlines(&cinfo, &rp, 1) != 1)
      throw std::runtime_error(name + ": JPEG ended " +
                               std::to_string(cinfo.output_scanline) + " of " +
                               std::to_string(cinfo.output_height) + " lines");
    std::memcpy(out.rgb.data() +
                    static_cast<size_t>(cinfo.output_scanline - 1) * out.width * 3,
                row.data(), static_cast<size_t>(out.width) * 3);
  }
  jpeg_finish_decompress(&cinfo);
  return out;
}

Image decode_image(const std::string &path) {
  const std::vector<uint8_t> bytes = slurp(path);
  if (bytes.size() < 8)
    throw std::runtime_error(path + ": " + std::to_string(bytes.size()) +
                             " bytes, too short to be an image");
  // By MAGIC. A file whose extension disagrees with its content is common; a
  // decoder that keys on the extension returns an error the user cannot act on.
  if (png_sig_cmp(const_cast<png_bytep>(bytes.data()), 0, 8) == 0)
    return decode_png(bytes, path);
  if (bytes[0] == 0xFF && bytes[1] == 0xD8) return decode_jpeg(bytes, path);
  throw std::runtime_error(
      path + ": not a PNG (signature) and not a JPEG (SOI marker). This "
      "runtime reads those two and refuses anything else rather than "
      "guessing -- re-save the image as PNG or JPEG.");
}

Image resize_square(const Image &src, int64_t side, Resample how) {
  if (side <= 0) throw std::runtime_error("resize to " + std::to_string(side) +
                                          " pixels");
  if (src.empty())
    throw std::runtime_error("resize: the source image decoded to nothing");
  // PIL applies no filter when the size already matches, and neither do we: a
  // filter here would be a half-LSB change to a pipeline that did not ask for
  // one, and every downstream gate would be measuring our resize instead of
  // the model's.
  if (src.width == side && src.height == side) return src;

  int64_t support_scale = 1;
  const Kernel kern = kernel_for(how, &support_scale);
  if (how == Resample::Nearest) {
    // NEAREST is NOT implemented here. It is the one PIL code with no
    // meaningful widening -- support 0.5, filterscale 1 -- so it point-samples
    // a downscale and produces an aliasing pattern, and this function's whole
    // reason to exist is that a correct downscale is the difference between a
    // label and noise. Refused by name rather than approximated: a caller
    // asking for NEAREST on a 4-megapixel photograph wants something this
    // build will not silently hand them.
    throw std::runtime_error(
        "resize: NEAREST is not implemented. It point-samples a downscale "
        "(no filter widening) and the aliasing it produces is amplified by "
        "1/std before the classifier sees it. LANCZOS, BILINEAR, BICUBIC and BOX "
        "are implemented and measured against PIL.Image.resize on the same bytes "
        "by tools/verify/verify_vit_image.py; BICUBIC is what this model was "
        "trained with. Re-pack with a preprocessor_config.json that does not ask "
        "for NEAREST.");
  }
  if (how == Resample::Hamming) {
    // HAMMING is refused because it is NOT MEASURED, and the other four are.
    //
    // This function's contract, and the one the header makes, is PIL-
    // equivalence rather than "a smooth interpolation". That contract is
    // MEASURED for LANCZOS, BILINEAR, BICUBIC and BOX in
    // tools/verify/verify_vit_image.py section 2: worst channel difference
    // against PIL at or below 1 8-bit LSB, on a flat field exactly 0, on noise
    // and edges and ramps alike. HAMMING was MEASURED too, against PIL, on the
    // same bytes, and it FAILED: worst channel difference 65 on random noise
    // with a mean of 11.8 over 97% of channels.
    //
    // The cause was chased rather than waved at. PIL's Filters.c definition,
    // 0.54 + 0.46*cos(pi*x) on |x| < 1, was implemented and still failed;
    // so was the Keys cubic a = +0.5 that PIL's older sources used; so was
    // every a + b*cos(pi*x) and a + b*cos(pi*x) + c*cos(2*pi*x) fitted to
    // PIL's measured impulse responses. None of them reproduces it, so this
    // build does not know what PIL's HAMMING does and will not ship a kernel
    // that is close. A wrong-but-smooth filter here is the exact failure this
    // project treats as worst: the label looks confident and means nothing.
    //
    // The container's own resample is what decides. Every ViT checkpoint this
    // tree has packed asks for BICUBIC, so this refusal is about a checkpoint
    // that asks for something this build cannot match -- and it is better that
    // it says so than that it quietly resamples differently from the model.
    throw std::runtime_error(
        "resize: HAMMING (PIL.Image code 5) is refused. LANCZOS, BILINEAR, "
        "BICUBIC and BOX are implemented and measured against PIL.Image.resize "
        "on the same bytes -- within one 8-bit LSB, exactly on a constant "
        "raster -- by tools/verify/verify_vit_image.py. HAMMING was measured "
        "against PIL the same way and does not agree: worst channel difference "
        "65 on a random raster. This build will not ship a resampler that is "
        "close to PIL rather than equal to it, because a smooth filter that is "
        "wrong is a confident label about a picture the model never saw. "
        "Re-pack with a preprocessor_config.json that asks for BICUBIC (which "
        "is what this model was trained with), or resize the image yourself.");
  }

  const int64_t w = src.width, h = src.height;
  // Horizontal first, into a h x side buffer, then vertical. The same order PIL
  // uses, and the same reason: the horizontal pass is the wider of the two and
  // doing it first keeps the intermediate at h*side rather than side*side.
  std::vector<uint8_t> tmp(static_cast<size_t>(h) * side * 3);
  const double fscale_x =
      std::max(1.0, static_cast<double>(w) / static_cast<double>(side));
  for (int64_t y = 0; y < h; ++y) {
    resample_axis(src.rgb.data() + static_cast<size_t>(y) * w * 3, w,
                  tmp.data() + static_cast<size_t>(y) * side * 3, side, 3, 3, 3,
                  kern, fscale_x);
  }
  Image out;
  out.width = side;
  out.height = side;
  out.rgb.resize(static_cast<size_t>(side) * side * 3);
  const double fscale_y =
      std::max(1.0, static_cast<double>(h) / static_cast<double>(side));
  for (int64_t x = 0; x < side; ++x)
    // BOTH strides are side*3 and the output base is the COLUMN, because both
    // buffers are row-major and consecutive OUTPUT indices are consecutive ROWS
    // here, not consecutive pixels. Writing stride_out = 3 -- the value the
    // horizontal pass above uses, and the one the two calls look
    // interchangeable enough to invite -- makes the vertical pass emit the
    // TRANSPOSE of the right answer: correctly shaped, entirely wrong, and
    // invisible on a flat raster. Both strides being equal is not a typo to be
    // tidied up later; it is the fact that makes the vertical pass vertical.
    resample_axis(tmp.data() + x * 3, h, out.rgb.data() + x * 3, side,
                  side * 3, side * 3, 3, kern, fscale_y);
  return out;
}

std::vector<float> normalise(const Image &im, const Geometry &g) {
  if (im.width != g.image_size || im.height != g.image_size)
    throw std::runtime_error("normalise: image is " +
                             std::to_string(im.width) + "x" +
                             std::to_string(im.height) + "px, the container's "
                             "image_size is " + std::to_string(g.image_size) +
                             ". Resize first; the mean/std are per channel of a "
                             "grid this size.");
  if (g.mean.size() != 3 || g.std_dev.size() != 3)
    throw std::runtime_error(
        "normalise: this build's image front end is three channels (RGB) and "
        "the container carries " + std::to_string(g.mean.size()) +
        " means. Re-pack, or extend vit/image.cpp -- silently taking the first "
        "three would be a different normalisation.");
  std::vector<float> out(static_cast<size_t>(g.num_channels) * g.image_size *
                         g.image_size);
  const int64_t n = g.image_size * g.image_size;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t c = 0; c < 3; ++c) {
      // A real DIVIDE by 255, not a multiply by (1/255f).
      //
      // 1/255 is not representable in binary, so the multiply is a different
      // arithmetic from the divide: measured over all 256 pixel values, the two
      // disagree for 176 of them, by exactly 1 ULP, and after (v - mean)/std
      // that is a 1.19e-07 difference in a normalised pixel. Small -- four
      // orders below the int8 gate and six below the bf16 one -- but it is free
      // to be exact, and the reference (numpy, and therefore transformers) does
      // `a / np.float32(255.0)`. "The multiply is the same operation as the
      // divide" is not a claim worth making about the one step whose output is
      // 197x768 numbers per image.
      const float v =
          static_cast<float>(im.rgb[static_cast<size_t>(i) * 3 + c]) / 255.0f;
      out[static_cast<size_t>(c) * n + i] = (v - g.mean[static_cast<size_t>(c)]) /
                                            g.std_dev[static_cast<size_t>(c)];
    }
  return out;
}

std::vector<float> im2col_patches(const float *chw, const Geometry &g) {
  const int64_t n = g.image_size / g.patch_size;   // patches per side
  std::vector<float> out(static_cast<size_t>(g.n_patches) *
                         static_cast<size_t>(g.patch_dim));
  // (c, i, j) -> (c*patch + i)*patch + j, and rows in (patch_row, patch_col)
  // order. See the header and tools/lib/vit_int8.py's im2col_patches, which is
  // the same statement in the other half of this pair.
  for (int64_t p = 0; p < n; ++p)
    for (int64_t q = 0; q < n; ++q) {
      float *dst = out.data() + (p * n + q) * g.patch_dim;
      for (int64_t c = 0; c < g.num_channels; ++c)
        for (int64_t i = 0; i < g.patch_size; ++i)
          for (int64_t j = 0; j < g.patch_size; ++j)
            dst[(c * g.patch_size + i) * g.patch_size + j] =
                chw[(c * g.image_size + p * g.patch_size + i) * g.image_size +
                    q * g.patch_size + j];
    }
  return out;
}

std::vector<float> preprocess(const Image &im, const Geometry &g) {
  const Image sq = resize_square(im, g.image_size, g.resample);
  const std::vector<float> chw = normalise(sq, g);
  return im2col_patches(chw.data(), g);
}

std::vector<float> preprocess_file(const std::string &path, const Geometry &g) {
  return preprocess(decode_image(path), g);
}

}  // namespace npue::vit
