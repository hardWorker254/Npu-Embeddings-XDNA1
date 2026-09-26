#ifndef NPUEMBEDDINGS_RUN_SETUP_HPP
#define NPUEMBEDDINGS_RUN_SETUP_HPP

#include "runtime/run_context.hpp"
#include "common/app_state.hpp"
#include "encoders/bert_encoder.hpp"
#include "common/design_selection.hpp"
#include "runtime/npu_contention.hpp"
#include "runtime/pool.hpp"
#include "common/host_kernels.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace app {

inline void load_designs(RunContext &ctx) {
  // Unified mode: art/gemm_rtp holds ONE xclbin whose four instruction
  // streams are the four GEMM shapes (tools/export_gemm_rtp.py). Every design
  // reference below binds to that one Design; the eltwise ops are forced onto
  // the host, and the encode runs in a single hw_context -- zero switches.
  const bool unified =
      std::ifstream(ctx.art + "/gemm_rtp/design.json").good();
  ctx.unified = unified;

  // COUNT BEFORE CONSTRUCTING (subtask 7). The driver allows a fixed number of
  // concurrent hw_contexts (six on the NPU1 driver measured here; see
  // npu::context_budget), and --npu-eltwise requests the unified GEMM plus up
  // to three eltwise designs. A
  // foreign process holding one context, or a leftover run of ours that did
  // not release one, otherwise surfaces as the bare driver string
  // `DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22)`. Ask xrt-smi first
  // and refuse by name while there is still nothing to unwind. A single-context
  // run skips this: the default path must not require a contention tool.
  int want_contexts = 0;
  if (unified) {
    want_contexts = 1 + (ctx.npu_eltwise
                             ? (ctx.host_gelu ? 0 : 1) + (ctx.host_ln ? 0 : 1) +
                                   (ctx.host_sm ? 0 : 1)
                             : 0);
  } else {
    want_contexts = 7;   // legacy per-op set: one xclbin per design
  }
  if (want_contexts > 1 &&
      !npu::require_context_budget(npu::survey_contexts(), want_contexts,
                                   ctx.allow_contention))
    throw std::runtime_error(
        "NPU context budget: refusing to load " +
        std::to_string(want_contexts) +
        " concurrent hw_context(s) -- see the report above (close the other "
        "process, or pass --allow-contention)");

  if (unified) {
    ctx.ud = std::make_unique<npu::Design>(*ctx.dev, ctx.art + "/gemm_rtp");
    std::ifstream sj(ctx.art + "/gemm_rtp/design.json");
    std::stringstream sbuf;
    sbuf << sj.rdbuf();
    ctx.streams = parse_streams(sbuf.str());
    if (ctx.streams.empty()) {
      // A pre-0037 export: four streams, no tiers, the old flat names.
      ctx.ud->load_instr(ctx.art + "/gemm_rtp/insts_attn_out.bin");   // 1
      ctx.ud->load_instr(ctx.art + "/gemm_rtp/insts_ffn_up.bin");     // 2
      ctx.ud->load_instr(ctx.art + "/gemm_rtp/insts_ffn_down.bin");   // 3
      std::printf("  designs    ONE xclbin, 4 instruction streams, one "
                  "hw_context\n");
    } else {
      // Load in slot order and CHECK it -- a stream bound to the wrong slot
      // would compute a different shape with the right buffer sizes, which
      // is exactly the failure mode this project has hit five times.
      std::sort(ctx.streams.begin(), ctx.streams.end(),
                [](const StreamEntry &a, const StreamEntry &b) {
                  return a.slot < b.slot;
                });
      for (const auto &s : ctx.streams) {
        const size_t got = ctx.ud->load_instr(ctx.art + "/gemm_rtp/" + s.file);
        if (static_cast<int64_t>(got) != s.slot)
          throw std::runtime_error("stream " + s.file + " landed in slot " +
                                   std::to_string(got) + ", design.json says " +
                                   std::to_string(s.slot));
      }
      std::set<int64_t> tset;
      for (const auto &s : ctx.streams) tset.insert(s.batch);
      std::printf("  designs    ONE xclbin, %zu streams (%zu batch tiers), "
                  "one hw_context\n", ctx.streams.size(), tset.size());
    }

    // --npu-eltwise: the unified xclbin carries only the four GEMM streams, so
    // the three elementwise ops come from sibling directories, one Design
    // each. Refuse by NAME when one is missing -- falling back to the host
    // after the flag asked for the array is the fail-open this project keeps
    // meeting, and the flag's whole point is to make that impossible.
    if (ctx.npu_eltwise) {
      auto need = [&](const char *op, std::unique_ptr<npu::Design> &dst,
                      bool host_forced) {
        if (host_forced) return;   // --host-<op>: the host path was asked for
        const std::string dir = ctx.art + "/" + op;
        if (!std::ifstream(dir + "/design.json").good())
          throw std::runtime_error(
              "--npu-eltwise asks for " + std::string(op) +
              " on the array, but " + dir + "/design.json does not exist -- "
              "build it with tools/export_eltwise.py (or "
              "tools/export_gemm_rtp.py --npu-eltwise), or drop the flag and "
              "run the host path, which is the measured-faster one");
        dst = std::make_unique<npu::Design>(*ctx.dev, dir);
        const auto &inf = dst->info();
        if (inf.device_recorded && !inf.device.empty() &&
            !running_device().empty() && inf.device != running_device())
          throw std::runtime_error(
              std::string("--npu-eltwise: ") + dir + " was built for device " +
              inf.device + ", but this process runs on " + running_device() +
              " -- rebuild it for this generation or drop the flag");
      };
      need("gelu", ctx.ld_gelu, ctx.host_gelu);
      need("layernorm", ctx.ld_ln, ctx.host_ln);
      need("softmax", ctx.ld_sm, ctx.host_sm);
    }
  } else {
    ctx.ld_qkv = std::make_unique<npu::Design>(*ctx.dev, ctx.art + "/qkv");
    ctx.ld_ao = std::make_unique<npu::Design>(*ctx.dev, ctx.art + "/attn_out");
    ctx.ld_fu = std::make_unique<npu::Design>(*ctx.dev, ctx.art + "/ffn_up");
    ctx.ld_fd = std::make_unique<npu::Design>(*ctx.dev, ctx.art + "/ffn_down");
    ctx.ld_gelu = std::make_unique<npu::Design>(*ctx.dev, ctx.art + "/gelu");
    ctx.ld_ln = std::make_unique<npu::Design>(*ctx.dev, ctx.art + "/layernorm");
    ctx.ld_sm = std::make_unique<npu::Design>(*ctx.dev, ctx.art + "/softmax");
    std::printf("  designs    7 resident xclbins\n");
  }

  // WHICH DATAPATH WAS ACTUALLY SELECTED (tasks/0104), read off the loaded
  // design, never off a flag or the directory name that happened to be
  // picked -- "reports the intention, not the value" is a cost this project
  // has already paid twice (tasks/0042, 0081) for a_dtype/c_dtype; bfp16 gets
  // the same discipline from day one.
  npu::Design &d_qkv = ctx.d_qkv();
  ctx.design_device = d_qkv.info().device;
  ctx.design_arch = d_qkv.info().arch;
  ctx.design_generation_recorded =
      d_qkv.info().device_recorded || d_qkv.info().arch_recorded;
  if (!d_qkv.info().datapath_recorded)
    std::printf("  datapath   UNRECORDED (design predates tasks/0104), "
                "C as %s\n",
                d_qkv.info().c_elem_bytes == 2 ? "bf16" : "fp32");
  else
    std::printf("  datapath   %s MMAC, C as %s\n",
                d_qkv.info().emulate_bfp16 ? "bfp16-emulated" : "bf16",
                d_qkv.info().c_elem_bytes == 2 ? "bf16" : "fp32");

  // WHICH GENERATION WAS ACTUALLY SELECTED (subtask 3), read off the loaded
  // design exactly like the datapath line above -- never off the flag that led
  // here. design_fits() has already refused a set built for the other
  // generation, so this line is the value, not the intention.
  if (!ctx.design_generation_recorded)
    std::printf("  device     UNRECORDED (design predates arch/device metadata)\n");
  else
    std::printf("  device     %s (arch %lld)\n",
                ctx.design_device.empty() ? "?" : ctx.design_device.c_str(),
                (long long)ctx.design_arch);

  // WHICH TOOLCHAIN BUILT THIS DESIGN (T39, tasks/0106) -- read off d_qkv,
  // same reasoning as the datapath line above (7-design and unified sets
  // both report their qkv design's provenance).
  if (!d_qkv.info().toolchain_recorded)
    std::printf("  toolchain  UNRECORDED (design predates tasks/0106)\n");
  else
    std::printf("  toolchain  mlir_aie %s, peano %s, mlir-aie HEAD %s\n",
                d_qkv.info().mlir_aie_version.c_str(),
                d_qkv.info().peano_version.c_str(),
                d_qkv.info().mlir_aie_git_head.c_str());

  // Batch comes from the design, not from a constant here, so a mismatch is
  // impossible rather than merely unlikely.
  // The design says what sequence length it was built for; the container
  // says how many positions it can feed. set_design_seq checks the second
  // against the first rather than trusting either alone.
  if (d_qkv.info().seq <= 0)
    throw std::runtime_error(
        "this design set records no sequence length -- re-export it with "
        "tools/export_gemm_rtp.py, or add \"seq\": 64 to its design.json if "
        "you know it was built for seq 64");
  set_design_seq(d_qkv.info().seq);

  const int64_t rows = d_qkv.info().M, batch = rows / g_seq;
  if (rows % g_seq || batch < 1)
    throw std::runtime_error("design M=" + std::to_string(rows) +
                             " is not a whole number of seq-" +
                             std::to_string(g_seq) + " sequences");
  std::printf("  shape      batch %lld x seq %lld  (M = %lld)\n",
              (long long)batch, (long long)g_seq, (long long)rows);
  ctx.rows = rows; ctx.batch = batch;

}

inline void load_goldens(RunContext &ctx) {
  // The goldens are batch 4 -- that is what M3 generated and what the accuracy
  // claim rests on. For larger batches the four sequences are TILED to fill
  // the design. That measures throughput honestly (the array does the full
  // work) -- but plain tiling (copy r's row k == the same base row k, for
  // every r) makes every physical copy of the golden batch BYTE-IDENTICAL,
  // and a row-indexing or cross-row-aliasing bug that reads the wrong copy
  // still reads identical data, so it is invisible to a comparison that only
  // checks content. This is exactly the bug class T32
  // (research/OPEN-THREADS.md) filed after tasks/0070's threaded
  // `swiglu_cpu()`: a genuine cross-row read/write race whose corruption
  // this golden gate could not see (it PASSED), caught only by a separate
  // distinct-content e2e run (`tools/verify_embed_e2e.py`) that failed at
  // worst `1-cos` 0.44.
  //
  // Fix (T32 option 1): rotate which of the 4 base sequences lands in row k
  // of copy r by `(k + r) % kGoldenBatch`, and apply the SAME rotation to
  // the expected output. This is still an exact golden -- it is a
  // relabelling of which known-good row goes where, not new data -- and it
  // is trivially invertible (row b's expected content is base row
  // `(b % kGoldenBatch + b / kGoldenBatch) % kGoldenBatch`). It costs
  // nothing extra to tile this way instead of plainly, and it turns "identical
  // content wherever it lands" into "content that must match its own row",
  // so a row-indexing or cross-row-aliasing bug now changes the answer.
  //
  // This does NOT, by itself, extend the accuracy claim past 4 distinct
  // sequences of *content* -- there are still only 4 distinct sentences in
  // the batch. What it buys is that every row of the OUTPUT is now checked
  // (see the comparison loop below) against the specific golden row it is
  // supposed to reproduce, so corruption or misrouting in any row, not just
  // rows 0-3, is visible.
  constexpr int64_t kGoldenBatch = 4;
  auto tile_rot = [kGoldenBatch](const std::vector<float> &v, int64_t reps,
                                 int64_t row_floats) {
    std::vector<float> out(static_cast<size_t>(reps * kGoldenBatch * row_floats));
    for (int64_t r = 0; r < reps; ++r)
      for (int64_t k = 0; k < kGoldenBatch; ++k) {
        const int64_t src = (k + r) % kGoldenBatch;
        std::memcpy(out.data() + static_cast<size_t>((r * kGoldenBatch + k) * row_floats),
                    v.data() + static_cast<size_t>(src * row_floats),
                    static_cast<size_t>(row_floats) * sizeof(float));
      }
    return out;
  };
  if (ctx.batch % kGoldenBatch)
    throw std::runtime_error("batch " + std::to_string(ctx.batch) +
                             " is not a multiple of the golden batch 4");
  const int64_t reps4 = ctx.batch / kGoldenBatch;

  //
  // A RELEASE ships the model and the design, not these fixtures, so when they
  // are absent the buffers are sized-but-empty and only the modes that
  // actually consume them complain. `need_goldens` is that complaint, raised
  // at the point of use so the message names the mode.
  if (ctx.have_val) {
    ctx.emb_in = tile_rot(read_f32(ctx.val + "/emb_sum.f32",
                               static_cast<size_t>(kGoldenBatch * g_seq * g_hidden)),
                      reps4, g_seq * g_hidden);
    ctx.mask = tile_rot(read_f32(ctx.val + "/add_mask.f32",
                             static_cast<size_t>(kGoldenBatch * g_seq)),
                    reps4, g_seq);
    // `want` is rotated with the IDENTICAL permutation as emb_in/mask/
    // amask_i, so row b of the output must match base golden row
    // `(b % kGoldenBatch + b / kGoldenBatch) % kGoldenBatch` -- the same
    // sequence that was actually fed into row b.
    ctx.want = tile_rot(read_f32(ctx.val + "/embedding_expected.f32",
                             static_cast<size_t>(kGoldenBatch * g_hidden)),
                    reps4, g_hidden);
    ctx.amask_i = tile_rot(read_f32(ctx.val + "/attention_mask.f32",
                                static_cast<size_t>(kGoldenBatch * g_seq)),
                       reps4, g_seq);
  } else {
    // The Encoder needs a mask of the right shape at construction; every
    // other mode overwrites it per chunk before dispatching.
    ctx.mask.assign(static_cast<size_t>(ctx.rows), 0.f);
  }

}

inline std::vector<float> pool_normalise(const RunContext &ctx,
                                       const std::vector<float> &h) {
  std::vector<float> out(static_cast<size_t>(ctx.batch) * g_hidden, 0.f);
  pool_rows(h.data(), ctx.amask_i.data(), ctx.batch, out.data());
  return out;
}

inline void setup_flags_pools(RunContext &ctx) {
  ctx.nthreads = 1;
  for (int i = 2; i < ctx.argc - 1; ++i)
    if (std::string(ctx.argv[i]) == "--threads") ctx.nthreads = std::atoi(ctx.argv[i + 1]);
  ctx.host_ln = false;
  for (int i = 2; i < ctx.argc; ++i)
    if (std::string(ctx.argv[i]) == "--host-ln") ctx.host_ln = true;
  ctx.host_sm = false;
  for (int i = 2; i < ctx.argc; ++i)
    if (std::string(ctx.argv[i]) == "--host-sm") ctx.host_sm = true;
  ctx.host_gelu = false;
  for (int i = 2; i < ctx.argc; ++i)
    if (std::string(ctx.argv[i]) == "--host-gelu") ctx.host_gelu = true;
  // Positive opt-in for array eltwise. Without it the unified set keeps all
  // three ops on the host, exactly as before. --host-* still wins per op.
  ctx.npu_eltwise = false;
  for (int i = 2; i < ctx.argc; ++i)
    if (std::string(ctx.argv[i]) == "--npu-eltwise") ctx.npu_eltwise = true;
  // The one override for a full or unreadable hw_context budget (subtask 7).
  ctx.allow_contention = false;
  for (int i = 2; i < ctx.argc; ++i)
    if (std::string(ctx.argv[i]) == "--allow-contention") ctx.allow_contention = true;
  ctx.sim_c_bf16 = false;
  for (int i = 2; i < ctx.argc; ++i)
    if (std::string(ctx.argv[i]) == "--sim-c-bf16") ctx.sim_c_bf16 = true;
  ctx.no_fuse_ffn = false;
  for (int i = 2; i < ctx.argc; ++i)
    if (std::string(ctx.argv[i]) == "--no-fuse-ffn") ctx.no_fuse_ffn = true;
  ctx.pipeline = 0;                 // 0 = off; N = N concurrent lanes
  for (int i = 2; i < ctx.argc; ++i)
    if (std::string(ctx.argv[i]) == "--pipeline") {
      ctx.pipeline = 2;
      if (i + 1 < ctx.argc && std::isdigit(static_cast<unsigned char>(
                              ctx.argv[i + 1][0])))
        ctx.pipeline = std::atoi(ctx.argv[i + 1]);
    }
  // Pipelining splits the thread budget: each lane gets its own pool, so no
  // lane can stall another's host work.
  if (ctx.pipeline > 1) {
    for (int l = 0; l < ctx.pipeline; ++l)
      ctx.pools.push_back(std::make_unique<Pool>(std::max(1, ctx.nthreads / ctx.pipeline)));
  } else {
    ctx.pools.push_back(std::make_unique<Pool>(ctx.nthreads));
  }
}

inline int setup_encoder(RunContext &ctx) {
  Pool &pool = *ctx.pools[0];

  // Build the encoder on the host side of the designs. The model, designs and
  // pool are references -- they outlive the encoder via RunContext. The mask
  // is installed after construction: the active tier, not the constructor,
  // fixes its final shape.
  ctx.enc = std::make_unique<npue::BertEncoder>(
      *ctx.model, ctx.d_qkv(), ctx.d_ao(), ctx.d_fu(), ctx.d_fd(),
      ctx.d_gelu(), ctx.d_ln(), ctx.d_sm(), pool);
  ctx.enc->batch = ctx.batch;
  ctx.enc->rows = ctx.rows;
  ctx.enc->add_mask = ctx.mask;
  if (ctx.unified) {
    // The unified artifact has no eltwise designs of its own: by default the
    // three ops run on the host. --npu-eltwise is the positive opt-in that
    // load_designs() already used to require the sibling directories, so with
    // it the host_* flags are left exactly as the CLI set them.
    if (!ctx.npu_eltwise)
      ctx.host_ln = ctx.host_sm = ctx.host_gelu = true;
    ctx.enc->unified = true;
    ctx.enc->is_qkv = 0;
    ctx.enc->is_ao = 1;
    ctx.enc->is_fu = 2;
    ctx.enc->is_fd = 3;
    if (!ctx.streams.empty()) {
      std::set<int64_t> tset;
      for (const auto &s : ctx.streams) tset.insert(s.batch);
      for (int64_t b : tset) {
        std::array<size_t, 4> slots{};
        bool complete = true;
        const char *ops[4] = {"qkv", "attn_out", "ffn_up", "ffn_down"};
        for (int k = 0; k < 4; ++k) {
          auto it = std::find_if(ctx.streams.begin(), ctx.streams.end(),
                                 [&](const StreamEntry &s) {
                                   return s.batch == b && s.op == ops[k];
                                 });
          if (it == ctx.streams.end()) { complete = false; break; }
          slots[k] = static_cast<size_t>(it->slot);
        }
        if (!complete) continue;         // a tier missing an op is not a tier
        ctx.enc->tiers.push_back(b);
        ctx.enc->tier_slots.push_back(slots);
      }
      ctx.enc->use_tier(ctx.batch);
      std::printf("  tiers      ");
      for (size_t i = 0; i < ctx.enc->tiers.size(); ++i)
        std::printf("%s%lld", i ? ", " : "", (long long)ctx.enc->tiers[i]);
      std::printf("  (requests are right-sized, not padded)\n");
    }
  }
  ctx.enc->host_ln = ctx.host_ln;
  ctx.enc->host_sm = ctx.host_sm;
  ctx.enc->host_gelu = ctx.host_gelu;
  ctx.enc->sim_c_bf16 = ctx.sim_c_bf16;
  ctx.enc->fuse_ffn_epilogue = !ctx.no_fuse_ffn;
  if (ctx.sim_c_bf16) {
    // Say so loudly, and refuse where it would mean nothing -- a status line
    // that reports the intention rather than the value is this project's
    // recurring fail-open (tasks/0042's `tile (64, 32)`).
    if (ctx.enc->qkv_.info().a_elem_bytes != 1) {
      std::fprintf(stderr,
                   "--sim-c-bf16 is only meaningful on an int8 design "
                   "(this one carries %d-byte operands)\n",
                   (int)ctx.enc->qkv_.info().a_elem_bytes);
      return 2;
    }
    if (ctx.enc->qkv_.info().c_elem_bytes == 2) {
      std::fprintf(stderr,
                   "--sim-c-bf16 simulates a narrowed-C design; this design "
                   "already narrows C on the core, so the flag would only "
                   "round a second time\n");
      return 2;
    }
    std::printf("  SIMULATION int32 C rounded to bf16 before dequantisation --\n"
                "             prices a narrowed-C design; NOT a shipped path\n");
  }
  // Where each op ACTUALLY runs, read off the design the encoder will call --
  // never off the flag that led here. A host-forced op prints the host line; an
  // array op names the resolved design and the generation it was built for.
  auto where = [](const char *op, bool host, const npu::Design &d,
                  const char *host_note) {
    if (host) {
      std::printf("  %-10s on the HOST (fp32) -- %s\n", op, host_note);
    } else {
      std::printf("  %-10s on the ARRAY (%s, arch %lld%s%s)\n", op,
                  d.info().name.empty() ? op : d.info().name.c_str(),
                  (long long)d.info().arch,
                  d.info().device.empty() ? "" : " ",
                  d.info().device.c_str());
    }
  };
  {
    char gelu_note[64], sm_note[64], ln_note[64];
    std::snprintf(gelu_note, sizeof gelu_note,
                  "%lld fewer NPU dispatches", (long long)g_layers);
    std::snprintf(sm_note, sizeof sm_note,
                  "%lld fewer NPU dispatches", (long long)g_layers);
    std::snprintf(ln_note, sizeof ln_note,
                  "%lld fewer NPU dispatches", (long long)(1 + 2 * g_layers));
    where("gelu", ctx.host_gelu, ctx.d_gelu(), gelu_note);
    where("softmax", ctx.host_sm, ctx.d_sm(), sm_note);
    where("layernorm", ctx.host_ln, ctx.d_ln(), ln_note);
  }
  const size_t staged = ctx.enc->stage_all();
  // What the allocation mode actually bought, in addresses. Printed
  // because "1 MB padding gives large-page backing" is a mechanism
  // claim, and the alignment is the only visible part of it.
  std::printf("  bo-align   last data buffer aligned to %zu B%s\n",
              npu::last_bo_alignment(),
              npu::last_bo_alignment() >= (1u << 21) ? " (>= 2 MB)" : "");
  std::printf("  weights    %.2f MB staged on the device once, not per call\n",
              staged / 1e6);

  static std::mutex npu_mutex;
  if (ctx.pipeline > 1) {
    if (!ctx.unified)
      throw std::runtime_error(
          "--pipeline requires the unified gemm_rtp artifact");
    ctx.enc->npu_mu = &npu_mutex;
    for (int l = 1; l < ctx.pipeline; ++l) {
      ctx.lanes.push_back(std::make_unique<npue::BertEncoder>(
          *ctx.model, ctx.d_qkv(), ctx.d_ao(), ctx.d_fu(), ctx.d_fd(),
          ctx.d_gelu(), ctx.d_ln(), ctx.d_sm(), *ctx.pools[l]));
      npue::BertEncoder &e2 = *ctx.lanes.back();
      e2.batch = ctx.batch;
      e2.rows = ctx.rows;
      e2.add_mask = ctx.mask;
      e2.unified = true;
      e2.is_qkv = 0; e2.is_ao = 1; e2.is_fu = 2; e2.is_fd = 3;
      // Same host/array choice as lane 0, not a hardcoded host path: with
      // --npu-eltwise every lane must dispatch the same ops to the same
      // designs, or the lanes compute different things.
      e2.host_ln = ctx.enc->host_ln;
      e2.host_sm = ctx.enc->host_sm;
      e2.host_gelu = ctx.enc->host_gelu;
      e2.sim_c_bf16 = ctx.enc->sim_c_bf16;
      e2.fuse_ffn_epilogue = ctx.enc->fuse_ffn_epilogue;
      // The staged weights and parameters are the design's, not a lane's.
      e2.s_qkv = ctx.enc->s_qkv; e2.s_ao = ctx.enc->s_ao;
      e2.s_fu = ctx.enc->s_fu; e2.s_fd = ctx.enc->s_fd;
      e2.b_qkv = ctx.enc->b_qkv; e2.b_ao = ctx.enc->b_ao;
      e2.b_fu = ctx.enc->b_fu; e2.b_fd = ctx.enc->b_fd;
      // int8 scales are the DESIGN's and the CONTAINER's, not a lane's --
      // same reasoning as the staged weights above, and the same failure if
      // forgotten: lane 0 worked, lanes 1+ dereferenced a null wscale and the
      // process segfaulted only under --pipeline (tasks/0078).
      e2.ws_qkv = ctx.enc->ws_qkv; e2.ws_ao = ctx.enc->ws_ao;
      e2.ws_fu = ctx.enc->ws_fu;   e2.ws_fd = ctx.enc->ws_fd;
      e2.as_qkv = ctx.enc->as_qkv; e2.as_ao = ctx.enc->as_ao;
      e2.as_fu = ctx.enc->as_fu;   e2.as_fd = ctx.enc->as_fd;
      e2.s_ln = ctx.enc->s_ln; e2.h_gamma = ctx.enc->h_gamma; e2.h_beta = ctx.enc->h_beta;
      // The tier table is POLICY, and every lane needs it. A lane without it
      // silently falls back to the pre-0037 flat slot contract (0,1,2,3),
      // which under the 16-stream export selects the wrong shapes entirely --
      // measured as 1-cos 1.0 on whichever chunk that lane happened to take.
      e2.tiers = ctx.enc->tiers;
      e2.tier_slots = ctx.enc->tier_slots;
      e2.use_tier(ctx.batch);
      // Each extra lane gets its own A and C buffers on the shared design;
      // lane 0 keeps the base slots.
      e2.slot_a = ctx.d_qkv().stage_alloc(0, ctx.d_qkv().info().buffer_bytes[0]);
      e2.slot_c = ctx.d_qkv().stage_alloc(2, ctx.d_qkv().info().buffer_bytes[2]);
      // ...and the same for the eltwise designs, but ONLY for the ops that
      // actually run on the array. Each of them has exactly ONE A and ONE C
      // buffer, so without this every lane overwrote the rows the others were
      // still feeding and read back the others' results -- and which lane won
      // was thread-scheduling dependent, so the same request answered
      // differently from one call to the next. host_ln/host_gelu/host_sm mean
      // the op never touches that design, and in unified mode those accessors
      // alias the GEMM design, whose A and C this lane already owns slots for
      // just above -- allocating again there would only burn device memory.
      auto elt_slots = [](npu::Design &d) {
        npue::BertEncoder::EltSlots s;
        s.a = d.stage_alloc(0, d.info().buffer_bytes[0]);
        s.c = d.stage_alloc(d.output_index(), d.info().buffer_bytes.back());
        return s;
      };
      if (!e2.host_gelu) e2.slots_gelu = elt_slots(ctx.d_gelu());
      if (!e2.host_ln)   e2.slots_ln   = elt_slots(ctx.d_ln());
      if (!e2.host_sm)   e2.slots_sm   = elt_slots(ctx.d_sm());
      e2.npu_mu = &npu_mutex;
    }
    for (const auto &lp : ctx.lanes) {
      if (lp->tiers != ctx.enc->tiers || lp->tier_slots.size() != ctx.enc->tier_slots.size())
        throw std::runtime_error(
            "lane stream policy differs from lane 0 -- refusing to run, "
            "because the lanes would compute different things");
    }
    std::printf("  pipeline   %d concurrent encodes of %lld, one NPU mutex, "
                "%d host threads per lane\n", ctx.pipeline, (long long)ctx.batch,
                ctx.pools[0]->size());
  }

  return 0;
}

}  // namespace app

#endif  // NPUEMBEDDINGS_RUN_SETUP_HPP
