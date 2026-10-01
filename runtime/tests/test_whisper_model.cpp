//===- test_whisper_model.cpp -----------------------------------*- C++ -*-===//
//
// Runs the Whisper NPU stacks so a gate can hold them against transformers.
//
// Same arrangement as the audio front end's gate: a test binary rather than a
// CLI, because nothing in the shipped runtime reaches arch=4 yet, and because
// the decoder needs a forced token sequence driven from outside -- a gate that
// fed the decoder its own previous output could not tell a correct first step
// from a correct hundredth.
//
//   argv[1] <model.npue>      the container: geometry and every weight
//   argv[2] <artifacts>       design set root; <artifacts>/gemm_rtp for the
//                             encoder and <artifacts>/gemm_rtp_dec for the
//                             decoder, both read through their own design.json
//   argv[3] <audio.wav>       16 kHz mono; the front end runs here so the gate
//                             sees the same mel the model would
//   --ids a,b,c               teacher-forced decoder steps at positions 0..n-1
//   --greedy a,b,c            greedy continuation of this prompt
//   --max-new N               cap on GENERATED tokens (0: the checkpoint's own
//                             position bound). The gate passes the same cap to
//                             transformers' generate(), so the two chains are
//                             the same string and not just a shared prefix
//   --step-tier N             decoder tier for one step (default: the smallest)
//   --prefill-tier N          decoder tier for the cross-attention K|V prefill
//                             (default: the largest)
//   --threads N               host workers
//   --conv cpu|npu|both        where conv1/conv2 run. cpu is the fp32 reference,
//                             npu is the array path (bf16 GEMMs on the encoder
//                             set's [rows, d, d] stream), both runs the cpu one
//                             and prints the npu one next to it so the gate can
//                             hold each against transformers in the same run
//   --skip-decoder            encoder only, and do not open the decoder xclbin
//
// Output, all on stdout, all hex, so the gate parses one thing and never
// round-trips a float through a decimal pipe:
//   conv <rows> <cols> <hex>            the front end's output, host path
//   convnpu <rows> <cols> <hex>         the same tensor from the NPU path
//   enc <rows> <cols> <hex>             encoder hidden states (final LayerNorm)
//   step <pos> <top1> <cols> <hex>      one decoder step's final state
//   logits <pos> <vocab> <hex>          that step's logits
//   greedy <n> <id> ...                 the generated ids
//   text <hex>                          those ids as text, through the
//                                       container's own tokenizer table
//
// Build:
//   g++ -std=c++17 -O2 -march=native -I runtime/include -I /opt/xilinx/xrt/include \
//       runtime/tests/test_whisper_model.cpp runtime/src/whisper/*.cpp \
//       runtime/src/whisper/features.cpp runtime/src/whisper/audio.cpp \
//       runtime/src/{device,design,model,pool}.cpp runtime/src/npu_contention.cpp \
//       -L /opt/xilinx/xrt/lib -lxrt_coreutil -o /tmp/test_whisper_model
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/design_selection.hpp"  // parse_streams
#include "common/host_kernels.hpp"      // now_s
#include "runtime/design.hpp"
#include "runtime/device.hpp"
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "tokenizers/whisper.hpp"
#include "whisper/audio.hpp"
#include "whisper/conv1d.hpp"
#include "whisper/decoder.hpp"
#include "whisper/eltwise.hpp"
#include "whisper/encoder.hpp"
#include "whisper/geometry.hpp"
#include "whisper/features.hpp"

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

std::string read_text(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Load every stream of a design set into its own slot, in the order design.json
// records, and CHECK that the slot the file was written with is the slot the
// loader hands back. A set whose streams were loaded in a different order runs
// the wrong instruction stream for an op and returns a plausible number: the
// slot number is the only place that mismatch can be seen.
std::vector<app::StreamEntry> load_streams(npu::Design &d,
                                           const std::string &dir) {
  const std::string js = read_text(dir + "/design.json");
  std::vector<app::StreamEntry> streams = app::parse_streams(js);
  if (streams.empty())
    throw std::runtime_error(dir +
                             "/design.json lists no streams -- re-export with "
                             "tools/export/export_gemm_rtp.py");
  std::sort(streams.begin(), streams.end(),
            [](const app::StreamEntry &a, const app::StreamEntry &b) {
              return a.slot < b.slot;
            });
  for (const auto &s : streams) {
    const size_t got = d.load_instr(dir + "/" + s.file);
    if (static_cast<int64_t>(got) != s.slot)
      throw std::runtime_error("stream " + s.file + " landed in slot " +
                               std::to_string(got) + ", design.json says " +
                               std::to_string(s.slot));
  }
  return streams;
}

const app::StreamEntry &find_op(const std::vector<app::StreamEntry> &streams,
                                 const std::string &op, int64_t batch) {
  for (const auto &s : streams)
    if (s.op == op && s.batch == batch) return s;
  throw std::runtime_error("no " + op + " stream at batch tier " +
                           std::to_string(batch) +
                           " -- re-export the design set with this tier");
}

std::vector<int64_t> parse_ids(const std::string &s) {
  std::vector<int64_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(std::stoll(item));
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <model.npue> <artifacts> <audio.wav> "
                 "[--ids a,b] [--greedy a,b] [--max-new N] [--step-tier N] "
                 "[--prefill-tier N] [--threads N] [--conv cpu|npu|both] "
                 "[--layn host|npu] [--gelu host|npu] [--attn host|npu] "
                 "[--skip-decoder]\n",
                 argv[0]);
    return 2;
  }
  using namespace npue::whisper;

  const std::string npue_path = argv[1], art = argv[2], audio = argv[3];
  std::vector<int64_t> ids, prompt;
  int threads = 1;
  int64_t max_new = 0;
  int64_t step_tier = -1, prefill_tier = -1;
  bool skip_decoder = false;
  std::string conv_mode = "both";
  std::string layn_mode = "host";
  std::string attn_mode = "host";
  std::string gelu_mode = "host";
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--ids") ids = parse_ids(next());
    else if (a == "--max-new") max_new = std::atoll(next().c_str());
    else if (a == "--greedy") prompt = parse_ids(next());
    else if (a == "--threads") threads = std::max(1, std::atoi(next().c_str()));
    else if (a == "--step-tier") step_tier = std::atoll(next().c_str());
    else if (a == "--prefill-tier") prefill_tier = std::atoll(next().c_str());
    else if (a == "--conv") {
      conv_mode = next();
      if (conv_mode != "cpu" && conv_mode != "npu" && conv_mode != "both")
        throw std::runtime_error("--conv " + conv_mode +
                                 ": expected cpu, npu or both");
    } else if (a == "--layn") {
      layn_mode = next();
      if (layn_mode != "host" && layn_mode != "npu")
        throw std::runtime_error("--layn " + layn_mode +
                                 ": expected host or npu");
    } else if (a == "--gelu") {
      gelu_mode = next();
      if (gelu_mode != "host" && gelu_mode != "npu")
        throw std::runtime_error("--gelu " + gelu_mode +
                                 ": expected host or npu");
    } else if (a == "--attn") {
      attn_mode = next();
      if (attn_mode != "host" && attn_mode != "npu")
        throw std::runtime_error("--attn " + attn_mode +
                                 ": expected host or npu");
    } else if (a == "--skip-decoder") skip_decoder = true;
    else {
      std::fprintf(stderr, "unknown flag %s\n", a.c_str());
      return 2;
    }
  }

  try {
    npue::File model(npue_path);
    const Geometry geom = read_geometry(model, npue_path);
    app::Pool pool(threads);
    app::Pool *use = threads > 1 ? &pool : nullptr;

    // The front end runs here rather than in the gate, so the C++ side and the
    // Python side start from the same samples and the mel is not a second
    // variable in the comparison.
    const npue::whisper::Audio a = npue::whisper::ingest(audio, false);
    const MelSpec mel =
        log_mel_30s(a.samples, static_cast<int>(geom.mel_bins), nullptr, use);

    npu::Device dev;
    npu::Design enc_design(dev, art + "/gemm_rtp");
    const std::vector<app::StreamEntry> enc_streams =
        load_streams(enc_design, art + "/gemm_rtp");

    // The encoder design set carries one batch tier, so its rows are whatever
    // that tier's streams say. A set with several tiers is a different
    // architecture (that is the decoder's shape), not a wider encoder.
    std::vector<int64_t> tiers;
    for (const auto &s : enc_streams) tiers.push_back(s.batch);
    std::sort(tiers.begin(), tiers.end());
    tiers.erase(std::unique(tiers.begin(), tiers.end()), tiers.end());
    if (tiers.size() != 1)
      throw std::runtime_error(
          art + "/gemm_rtp has " + std::to_string(tiers.size()) +
          " batch tiers; the encoder is exported for one, because 1500 "
          "positions are walked in chunks of that tier's row count and a "
          "second tier would mean a second chunking policy");
    const int64_t eb = tiers[0];
    EncoderStreams es;
    es.qkv = static_cast<size_t>(find_op(enc_streams, "qkv", eb).slot);
    es.attn_out = static_cast<size_t>(find_op(enc_streams, "attn_out", eb).slot);
    es.ffn_up = static_cast<size_t>(find_op(enc_streams, "ffn_up", eb).slot);
    es.ffn_down = static_cast<size_t>(find_op(enc_streams, "ffn_down", eb).slot);
    es.rows = find_op(enc_streams, "qkv", eb).M;

    const int64_t n_src = kEncoderPositions;
    // --conv: cpu prints the fp32 reference, npu the array path, both prints
    // each next to the other so one run holds both against transformers. The
    // array path borrows the encoder set's [rows, d, d] stream, so a set
    // without one is reported rather than silently skipped.
    std::vector<float> conv;
    double t_conv_cpu = 0.0, t_conv_npu = 0.0;
    if (conv_mode != "npu") {
      const double t0 = app::now_s();
      conv = conv_front_end(
          mel, model.raw("frontend.conv1.weight").as<float>(),
          model.raw("frontend.conv1.bias").as<float>(),
          model.raw("frontend.conv2.weight").as<float>(),
          model.raw("frontend.conv2.bias").as<float>(),
          static_cast<int>(geom.d_model), use);
      t_conv_cpu = app::now_s() - t0;
      std::printf("conv %lld %lld %s\n", static_cast<long long>(n_src),
                  static_cast<long long>(geom.d_model),
                  to_hex(conv.data(), conv.size() * 4).c_str());
    }

    const app::StreamEntry *conv_stream = nullptr;
    for (const auto &s : enc_streams)
      if (s.K == geom.d_model && s.N == geom.d_model) {
        conv_stream = &s;
        break;
      }
    std::vector<float> conv_npu;
    if (conv_mode != "cpu") {
      if (!conv_stream)
        std::fprintf(stderr,
                     "conv: no [rows, d, d] stream in this design set, so "
                     "conv1/conv2 stay on the host\n");
      else {
        NpuGemm conv_gemm(enc_design, pool);
        conv_gemm.alloc_buffers();
        NpuConv1d c1(enc_design, pool, conv_gemm,
                     static_cast<size_t>(conv_stream->slot), conv_stream->M,
                     conv_stream->K, conv_stream->N);
        NpuConv1d c2(enc_design, pool, conv_gemm,
                     static_cast<size_t>(conv_stream->slot), conv_stream->M,
                     conv_stream->K, conv_stream->N);
        c1.stage("frontend.conv1", model.raw("frontend.conv1.weight").as<float>(),
                 model.raw("frontend.conv1.bias").as<float>(), geom.mel_bins,
                 geom.d_model, 3);
        c2.stage("frontend.conv2", model.raw("frontend.conv2.weight").as<float>(),
                 model.raw("frontend.conv2.bias").as<float>(), geom.d_model,
                 geom.d_model, 3);
        const double t0 = app::now_s();
        conv_npu = conv_front_end_npu(mel, c1, c2, static_cast<int>(geom.d_model));
        t_conv_npu = app::now_s() - t0;
        std::printf("convnpu %lld %lld %s\n", static_cast<long long>(n_src),
                    static_cast<long long>(geom.d_model),
                    to_hex(conv_npu.data(), conv_npu.size() * 4).c_str());
        std::fprintf(stderr,
                     "conv: %s stream, %lld+%lld K blocks, %lld dispatches, "
                     "npu %.3f s vs cpu %.3f s\n",
                     conv_stream->op.c_str(),
                     static_cast<long long>(c1.k_blocks()),
                     static_cast<long long>(c2.k_blocks()),
                     static_cast<long long>(conv_gemm.n_dispatch), t_conv_npu,
                     t_conv_cpu);
        if (conv_mode == "npu") conv = conv_npu;
      }
    }
    if (conv.empty())
      throw std::runtime_error("no conv output: both --conv paths produced "
                               "nothing to run the encoder on");

    // The LayerNorm design, when the case asks for it. It is a sibling xclbin of
    // its own, and the two facts that make it a DIFFERENT MODEL rather than a
    // rounding -- its row width and its epsilon -- are compiled into the kernel,
    // so both are checked against the container before anything is staged.
    // Constructed ONLY for this case: the design directory is not there on a
    // host run, and opening it unconditionally would turn a missing optional
    // design into a refusal of the default path.
    std::unique_ptr<npu::Design> ln_design;
    std::unique_ptr<NpuEltwise> ln;
    if (layn_mode == "npu") {
      ln_design = std::make_unique<npu::Design>(dev, art + "/layernorm");
      if (ln_design->info().cols != geom.d_model)
        throw std::runtime_error(
            art + "/layernorm is " + std::to_string(ln_design->info().cols) +
            " columns wide and this container's d_model is " +
            std::to_string(geom.d_model));
      if (std::abs(ln_design->info().ln_eps - geom.ln_eps) > 1e-12)
        throw std::runtime_error(
            art + "/layernorm was built with eps " +
            std::to_string(ln_design->info().ln_eps) +
            ", the container says " + std::to_string(geom.ln_eps));
      ln = std::make_unique<NpuEltwise>(*ln_design, pool,
                                        EltwiseKind::LayerNorm);
      ln->alloc_buffers();
      std::fprintf(stderr,
                   "layernorm: %lld rows x %lld, eps %g\n",
                   static_cast<long long>(ln->rows()),
                   static_cast<long long>(ln->cols()),
                   ln_design->info().ln_eps);
    }

    // Attention as two GEMMs, on the set's own attn_qk / attn_av streams. The
    // same refusals the session makes: a set without those streams cannot run
    // it, and one whose n_kv is not this model's window would tile a B panel
    // that is not the score row.
    std::unique_ptr<npu::Design> sm_design;
    std::unique_ptr<NpuEltwise> sm;
    std::unique_ptr<NpuAttention> attn;
    if (attn_mode == "npu") {
      const app::StreamEntry *qk = nullptr, *av = nullptr;
      for (const auto &st : enc_streams)
        if (st.batch == eb) {
          if (st.op == "attn_qk" && !qk) qk = &st;
          if (st.op == "attn_av" && !av) av = &st;
        }
      if (!qk || !av)
        throw std::runtime_error(
            art + "/gemm_rtp has no attn_qk/attn_av streams at tier " +
            std::to_string(eb) +
            "; re-export with --npu-extra-ops attn to run attention here");
      bool want_softmax = std::ifstream(art + "/softmax/design.json").good();
      if (want_softmax) {
        sm_design = std::make_unique<npu::Design>(dev, art + "/softmax");
        sm = std::make_unique<NpuEltwise>(*sm_design, pool,
                                          EltwiseKind::Softmax);
        sm->alloc_buffers();
        if (sm->cols() != qk->N)
          throw std::runtime_error(
              art + "/softmax is " + std::to_string(sm->cols()) +
              " wide and the score row is " + std::to_string(qk->N));
      }
      attn = std::make_unique<NpuAttention>(enc_design, pool, qk->N,
                                            geom.head_dim, av->N);
      attn->set_streams(static_cast<size_t>(qk->slot),
                        static_cast<size_t>(av->slot), qk->M);
      // Attached AFTER the object exists and BEFORE its buffers: the softmax is
      // a dispatch inside the attention, so it has to be a pointer that outlives
      // this scope, not a temporary.
      if (sm) attn->set_softmax(sm.get());
      attn->alloc_buffers();
      std::fprintf(stderr,
                   "attention: attn_qk %lldx%lldx%lld, attn_av %lldx%lldx%lld, "
                   "%lld rows, softmax %s\n",
                   static_cast<long long>(qk->M), static_cast<long long>(qk->K),
                   static_cast<long long>(qk->N), static_cast<long long>(av->M),
                   static_cast<long long>(av->K), static_cast<long long>(av->N),
                   static_cast<long long>(qk->M),
                   sm ? "on the array" : "on the host");
    }

    // The GELU design, when the case asks for it. The kernel is the model's own
    // exact-erf activation (kernels/gelu_erf.cc), and what has to be checked is
    // that it IS that function and not the tanh polynomial the BERT-family
    // designs use: the two differ by 2.5e-3 relative, which is a different
    // activation rather than a rounding of one.
    std::unique_ptr<npu::Design> gelu_design;
    std::unique_ptr<NpuEltwise> gelu;
    if (gelu_mode == "npu") {
      gelu_design = std::make_unique<npu::Design>(dev, art + "/gelu");
      gelu = std::make_unique<NpuEltwise>(*gelu_design, pool,
                                          EltwiseKind::Gelu);
      gelu->alloc_buffers();
      const int64_t need = es.rows * geom.enc_intermediate;
      if (gelu->rows() * gelu->cols() < need)
        throw std::runtime_error(
            art + "/gelu holds " +
            std::to_string(gelu->rows() * gelu->cols()) +
            " elements and one encoder chunk of GELU is " +
            std::to_string(need) +
            ". It is walked in dispatches of its own, so a smaller design is "
            "slower rather than wrong, but a design exported for another model's "
            "width is not the one to measure.");
      std::fprintf(stderr, "gelu: %lld elements per dispatch\n",
                   static_cast<long long>(gelu->rows() * gelu->cols()));
    }

    WhisperEncoder enc(model, enc_design, pool, geom);
    enc.set_streams(es);
    if (ln) enc.set_layernorm(ln.get());
    if (gelu) enc.set_gelu(gelu.get());
    if (attn) enc.set_attention(attn.get());
    const double t_stage = app::now_s();
    enc.stage_all();
    const double t_enc0 = app::now_s();
    const std::vector<float> hidden = enc.run(conv, n_src);
    const double t_enc1 = app::now_s();
    std::printf("enc %lld %lld %s\n", static_cast<long long>(n_src),
                static_cast<long long>(geom.d_model),
                to_hex(hidden.data(), hidden.size() * 4).c_str());
    std::fprintf(stderr,
                 "encoder: %lld layers x %lld rows, %lld dispatches in %.3f s "
                 "(stage %.3f s)\n",
                 static_cast<long long>(geom.enc_layers),
                 static_cast<long long>(es.rows),
                 static_cast<long long>(enc.gemm().n_dispatch), t_enc1 - t_enc0,
                 t_enc0 - t_stage);

    if (skip_decoder) return 0;

    npu::Design dec_design(dev, art + "/gemm_rtp_dec");
    const std::vector<app::StreamEntry> dec_streams =
        load_streams(dec_design, art + "/gemm_rtp_dec");
    std::vector<int64_t> dtiers;
    for (const auto &s : dec_streams) dtiers.push_back(s.batch);
    std::sort(dtiers.begin(), dtiers.end());
    dtiers.erase(std::unique(dtiers.begin(), dtiers.end()), dtiers.end());
    if (step_tier < 0) step_tier = dtiers.front();
    if (prefill_tier < 0) prefill_tier = dtiers.back();

    WhisperDecoder dec(model, dec_design, pool, geom);
    for (int64_t b : dtiers) {
      DecoderTier t;
      t.batch = b;
      const app::StreamEntry &q = find_op(dec_streams, "self_qkv", b);
      t.rows = q.M;
      t.streams.self_qkv = static_cast<size_t>(q.slot);
      t.streams.self_attn_out =
          static_cast<size_t>(find_op(dec_streams, "self_attn_out", b).slot);
      t.streams.cross_q = static_cast<size_t>(find_op(dec_streams, "cross_q", b).slot);
      t.streams.cross_kv =
          static_cast<size_t>(find_op(dec_streams, "cross_kv", b).slot);
      t.streams.cross_attn_out =
          static_cast<size_t>(find_op(dec_streams, "cross_attn_out", b).slot);
      t.streams.ffn_up = static_cast<size_t>(find_op(dec_streams, "ffn_up", b).slot);
      t.streams.ffn_down =
          static_cast<size_t>(find_op(dec_streams, "ffn_down", b).slot);
      dec.add_tier(t);
    }
    dec.set_step_tier(step_tier);
    dec.set_prefill_tier(prefill_tier);
    dec.stage_all();
    // The decoding policy, so the greedy ids this prints are the ones the
    // reference implementation would pick. A container without it is refused.
    load_suppression(dec, model, argv[1]);
    const double t_src0 = app::now_s();
    dec.set_source(hidden, n_src);
    const double t_src1 = app::now_s();
    std::fprintf(stderr,
                 "decoder: step tier %lld rows, prefill tier %lld rows, "
                 "cross K|V for %lld positions in %.3f s\n",
                 static_cast<long long>(step_tier),
                 static_cast<long long>(prefill_tier),
                 static_cast<long long>(n_src), t_src1 - t_src0);

    for (size_t i = 0; i < ids.size(); ++i) {
      std::vector<float> h, lg;
      const int32_t top = dec.step(static_cast<int32_t>(ids[i]),
                                   static_cast<int64_t>(i), &h, &lg);
      std::printf("step %zu %d %lld %s\n", i, static_cast<int>(top),
                  static_cast<long long>(h.size()),
                  to_hex(h.data(), h.size() * 4).c_str());
      std::printf("logits %zu %lld %s\n", i,
                  static_cast<long long>(lg.size()),
                  to_hex(lg.data(), lg.size() * 4).c_str());
    }
    if (!prompt.empty()) {
      // The stop token is looked up BY NAME in the container's own table, not
      // hardcoded: 50257 is what openai/whisper-* happens to use, and a
      // container that renumbered it would otherwise be decoded until it hit
      // max_target_positions -- a long, wrong transcript instead of a refusal.
      const auto tbl = model.raw("tokenizer.whisper_table");
      const npue::WhisperTokenizer tok(tbl.data, tbl.bytes);
      const int32_t stop = tok.token_id("<|endoftext|>");
      const std::vector<int32_t> out = dec.greedy(
          std::vector<int32_t>(prompt.begin(), prompt.end()), stop, max_new);
      std::printf("greedy %zu", out.size());
      for (int32_t id : out) std::printf(" %d", id);
      std::printf("\n");
      const std::string text = tok.decode(out);
      std::printf("text %s\n", to_hex(text.data(), text.size()).c_str());
    }
    std::fprintf(stderr, "decoder: %lld dispatches total\n",
                 static_cast<long long>(dec.gemm().n_dispatch));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
