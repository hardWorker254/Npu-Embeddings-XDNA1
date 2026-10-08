#ifndef NPUEMBEDDINGS_RUN_SETUP_HPP
#define NPUEMBEDDINGS_RUN_SETUP_HPP

#include "runtime/run_context.hpp"
#include "common/app_state.hpp"
#include "encoders/bert_encoder.hpp"
#include "common/design_selection.hpp"
#include "common/npu_ops_flag.hpp"
#include "runtime/npu_contention.hpp"
#include "runtime/pool.hpp"
#include "common/host_kernels.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace app {

inline void load_designs(RunContext &ctx) {
  // Unified mode: art/gemm_rtp holds ONE xclbin whose four instruction
  // streams are the four GEMM shapes (tools/export/export_gemm_rtp.py). Every design
  // reference below binds to that one Design; the eltwise ops are forced onto
  // the host, and the encode runs in a single hw_context -- zero switches.
  // UNIFIED MODE IS "there is a gemm_rtp set", NOT "there is a directory".
  //
  // It used to be the second, read as `ifstream(ctx.art + "/gemm_rtp/design.json")
  // .good()`. That made the mode a statement about the FILESYSTEM: a reader who
  // downloaded one self-sufficient .npue and had no runtime/artifacts/ at all was
  // told, by this line alone, that its design set was a pre-0037 export, and sent
  // down the legacy seven-context path looking for ctx.art + "/qkv" and four
  // siblings that a modern set has never had. The container was carrying the right
  // xclbin the whole time.
  //
  // So the question is now asked of the Source, which answers "is it in the
  // container, or on disk, or neither" -- and "neither" still selects the legacy
  // path, exactly as before, because that is the only thing that path is for.
  const npu::DesignSource gemm_src =
      npu::prefer_embedded(ctx.model.get(), ctx.art, "gemm_rtp");
  const bool unified = gemm_src.has("design.json");
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
    // One context for the unified GEMM design plus one per op --npu-ops sends to
    // the array. Counted from the request, not from what got loaded, so the
    // guard refuses BEFORE any Design is built (the point of the whole check).
    //
    // ONLY the codes that carry a design DIRECTORY are counted, because the
    // other six -- gemm, conv, attn, mproj, fft, logit -- are instruction
    // streams INSIDE the unified set and cost a slot of the context that is
    // already being counted by the `1`. Counting all of them made the sum a
    // fiction that grew with every code the reader typed: `--npu-ops gemm` alone
    // asked for two contexts of which one was ever opened. The STT path has
    // always counted this way (stt_mode.hpp); this is the two halves agreeing.
    want_contexts = 1;  // the unified gemm_rtp set itself
    for (const auto &op : app::npu_op_table())
      if (op.design[0] && ctx.npu_ops.count(op.code)) ++want_contexts;
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
    // The SAME source, so the stream table and the Design it belongs to cannot
    // come from two different places. Reading the JSON from disk while the xclbin
    // came out of the container would pair a stream list with a core that does not
    // have those streams, and the failure would surface as a dispatch mismatch
    // several layers down rather than as "these two do not belong together".
    ctx.ud = std::make_unique<npu::Design>(*ctx.dev, gemm_src);
    ctx.streams = parse_streams(gemm_src.text("design.json"));
    if (ctx.streams.empty()) {
      // A pre-0037 export: four streams, no tiers, the old flat names.
      ctx.ud->load_instr(gemm_src, "insts_attn_out.bin");   // 1
      ctx.ud->load_instr(gemm_src, "insts_ffn_up.bin");     // 2
      ctx.ud->load_instr(gemm_src, "insts_ffn_down.bin");   // 3
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
        // Through gemm_src, not ctx.art: an instruction stream is part of the set,
        // and a reader whose set lives in the container has no ctx.art + "/gemm_rtp"
        // to append to. It landed in slot 0 and reported "cannot open /gemm_rtp/
        // insts_qkv_b4.bin" before this, with a leading slash that says exactly
        // how the empty art_ path was being used as a real directory.
        const size_t got = ctx.ud->load_instr(gemm_src, s.file);
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

    // --npu-ops: the unified xclbin carries only the four GEMM streams, so each
    // op sent to the array comes from a sibling directory, one Design each.
    // Refuse by NAME when one is missing -- falling back to the host after the
    // flag asked for the array is the fail-open this project keeps meeting, and
    // the flag's whole point is to make that impossible. An op that is not
    // listed loads nothing, which is why this loop is over the REQUEST and not
    // over the three names.
    for (const char *code : {"gelu", "layn", "softm"}) {
      if (!ctx.on_array(code)) continue;
      const NpuOp *op = find_npu_op(code);
      std::unique_ptr<npu::Design> *dst =
          std::strcmp(code, "gelu") == 0   ? &ctx.ld_gelu
          : std::strcmp(code, "layn") == 0 ? &ctx.ld_ln
                                            : &ctx.ld_sm;
      // prefer_embedded for the elementwise sets too: a container that carries
    // gemm_rtp carries gelu/layernorm/softmax beside it, and --npu-ops gelu on a
    // self-sufficient container is the case where "self-sufficient" would otherwise
    // be a half-truth -- the GEMMs work and the op has nowhere to come from.
    const npu::DesignSource op_src =
        npu::prefer_embedded(ctx.model.get(), ctx.art, op->design);
    const std::string dir = op_src.label();
      // asks the SOURCE, not the filesystem: same reason as `unified` above
      if (!op_src.has("design.json"))
        throw std::runtime_error(
            std::string("--npu-ops ") + code + " (" + op->long_name +
            ") asks for it on the array, but that design set is not there (" +
            dir + " has no design.json). The exporter builds every design the "
            "target can honour -- one command, no flag, printing the list it "
            "chose -- so a set missing this one was built before that, or built "
            "for a different model: re-run `tools/export/export_gemm_rtp.py "
            "--target <model>` against the container at " +
            ctx.model_path +
            " and re-pack. Or drop " + code +
            " from the list and run the host path, which is the "
            "measured-faster one");
      *dst = std::make_unique<npu::Design>(*ctx.dev, op_src);
      const auto &inf = (*dst)->info();
      if (inf.device_recorded && !inf.device.empty() &&
          !running_device().empty() && inf.device != running_device())
        throw std::runtime_error(
            std::string("--npu-ops ") + code + ": " + dir +
            " was built for device " + inf.device + ", but this process runs on " +
            running_device() +
            " -- rebuild it for this generation or drop it from the list");
    }
    // A code this path cannot honour is REFUSED, not skipped. `conv` is the one
    // that matters today: it is a real op with no eltwise directory (Whisper's
    // conv1/conv2, which run on the encoder set's own [rows, d, d] stream), and
    // this container is an embedder, so there is no front end to send it to.
    // The check lives in setup_flags_pools rather than here because this
    // function runs BEFORE the flag is parsed, and a check that reads an empty
    // set is a check that never fires.
  } else {
    // The pre-0037 seven-set layout, read through prefer_embedded for the same
    // reason as everything else. No container in this tree carries it -- the
    // exporter has emitted the unified one for a long time -- so in practice these
    // always resolve on disk, and the point is that the RULE is uniform rather than
    // that this path gains anything.
    ctx.ld_qkv = std::make_unique<npu::Design>(
        *ctx.dev, npu::prefer_embedded(ctx.model.get(), ctx.art, "qkv"));
    ctx.ld_ao = std::make_unique<npu::Design>(
        *ctx.dev, npu::prefer_embedded(ctx.model.get(), ctx.art, "attn_out"));
    ctx.ld_fu = std::make_unique<npu::Design>(
        *ctx.dev, npu::prefer_embedded(ctx.model.get(), ctx.art, "ffn_up"));
    ctx.ld_fd = std::make_unique<npu::Design>(
        *ctx.dev, npu::prefer_embedded(ctx.model.get(), ctx.art, "ffn_down"));
    ctx.ld_gelu = std::make_unique<npu::Design>(
        *ctx.dev, npu::prefer_embedded(ctx.model.get(), ctx.art, "gelu"));
    ctx.ld_ln = std::make_unique<npu::Design>(
        *ctx.dev, npu::prefer_embedded(ctx.model.get(), ctx.art, "layernorm"));
    ctx.ld_sm = std::make_unique<npu::Design>(
        *ctx.dev, npu::prefer_embedded(ctx.model.get(), ctx.art, "softmax"));
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
                d_qkv.info().datapath_name(),
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
        "tools/export/export_gemm_rtp.py, or add \"seq\": 64 to its design.json if "
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
  // A value flag in the LAST position used to be invisible: the loops below
  // stopped at argc-1 because they read argv[i+1], so `--npu-ops gelu` as the
  // final two arguments parsed as nothing and the run quietly did the default.
  // That is the project's worst failure shape -- a flag the user typed that has
  // no effect and no error -- so the bound is argc and a flag with no value
  // after it is refused instead.
  auto value_after = [&](int i, const char *flag) -> const char * {
    if (i + 1 >= ctx.argc)
      throw std::runtime_error(std::string(flag) +
                               " is the last thing on the command line and has "
                               "no value after it");
    return ctx.argv[i + 1];
  };
  ctx.nthreads = 1;
  for (int i = 2; i < ctx.argc; ++i)
    if (std::string(ctx.argv[i]) == "--threads")
      ctx.nthreads = std::atoi(value_after(i, "--threads"));
  refuse_removed_op_flags(ctx.argc, ctx.argv);
  refuse_exporter_only_flags(ctx.argc, ctx.argv);
  // Which ops go on the array. The DEFAULT is the empty list -- all three on the
  // host, which is the measured-faster path -- and a repeated flag replaces the
  // previous one rather than adding to it, the same as --artifacts and --model:
  // the last one on the line is the one that counts, and `--npu-ops ""`
  // clears.
  ctx.npu_ops.clear();
  for (int i = 2; i < ctx.argc; ++i)
    if (std::string(ctx.argv[i]) == "--npu-ops")
      ctx.npu_ops = parse_npu_ops(value_after(i, "--npu-ops"));
  // An op this pipeline has no place for is REFUSED by name here, where the
  // flag has just been parsed. `conv` is the one that matters today: Whisper's
  // conv1/conv2, which the STT mode already dispatched above with (it owns the
  // flag) -- so reaching here means an embedder was asked for a speech-to-text
  // op, and dropping it would be the "the flag was there and nothing happened"
  // failure the subcommand whitelist exists to prevent.
  //
  // `attn` is NOT in this table, and its absence is deliberate: attention has no
  // front end, so an embedder's question is not "does this model have attention"
  // (it always does) but "does the loaded set carry the streams". That is
  // answered by NAME against the loaded stream table further down, where a set
  // without attn_qk/attn_av is refused as an artifact that was never exported
  // for this model -- not as a missing capability.
  for (const auto &code : ctx.npu_ops) {
    // `gemm` joins `attn` in NOT being refused here, and for the same reason in
    // reverse: an embedder's per-layer GEMMs are exactly this pipeline's work,
    // so `--npu-ops gemm` is a request this container can honour. It was the
    // ninth code, and it was the one the old hand-typed allow-list would have
    // rejected as "a speech-to-text op" -- the argument below is about Whisper's
    // conv1/conv2, not about matrix multiplication.
    if (code == "gelu" || code == "layn" || code == "softm" ||
        code == "attn" || code == "gemm")
      continue;
    const NpuOp *op = find_npu_op(code);
    throw std::runtime_error(
        std::string("--npu-ops ") + code + " (" +
        (op ? op->long_name : "unknown op") +
        ") is a speech-to-text op and this container is an embedder, which has "
        "no front end to send it to the array. It belongs to `transcribe <a "
        "whisper model> <audio>`." +
        (code == "logit"
             ? std::string(
                   " `logit` in particular names Whisper's TIED TOKEN EMBEDDING "
                   "used as the logit matrix (decoder.cpp:196), not a generic "
                   "vocabulary layer: an embedder stops at its pooling head and "
                   "has no such tensor at all.")
             : std::string()));
  }
  // A GATED FFN HAS NO PER-OP GELU, and the code is refused HERE, beside the
  // speech-to-text codes above, because this is the same question: which codes
  // can this container honour at all. It has to be asked before load_designs()
  // (runtime.cpp calls setup_flags_pools first), or a gated model that was never
  // given a gelu/ directory fails on the missing design instead -- telling the
  // reader to run an export command that refuses for this very reason.
  //
  // nomic (SwiGLU), gte (GeGLU) and gemma (GeGLU) compute their activation
  // INSIDE the gated path: between ffn_up and ffn_down, in swiglu_cpu or its
  // equivalent. The encoder's only eltwise(gelu_, ...) call sits in the `else`
  // of that branch -- the ungated up -> GELU -> down shape -- so for a gated
  // model host_gelu is read by nothing at all.
  //
  // What this used to do is the thing the status block two hundred lines below
  // says it never does: report the intention rather than the value. host_gelu
  // came straight off the flag, the status printed "gelu GELU on the ARRAY
  // (gelu, arch 1 npu1)" for a design the encoder never opens, and the
  // embeddings came back BIT-IDENTICAL to the host run -- measured, not
  // inferred: relfro 0.000e+00 on both nomic and gte, against 6.1e-03 for
  // bge-base, which is ungated and really does move it. A status line naming a
  // dispatch that does not happen is worse than none, because it is the one a
  // reader trusts.
  //
  // gemma additionally cannot take layn or softm, for a different reason (its
  // encoder takes no per-op flag at all) and in a different place: gemma_mode.hpp
  // refuses those before any of this runs.
  if (app::g_gated_ffn && ctx.on_array("gelu"))
    throw std::runtime_error(
        "--npu-ops gelu: this container has a GATED FFN, whose activation "
        "is part of the gated path between ffn_up and ffn_down rather than a "
        "separate pass over the activations. There is no host-or-array choice to "
        "make here, so asking for it moves nothing -- the run would print the op "
        "as being on the array and hand back bit-identical vectors. The other "
        "codes are unaffected: layn and softm are real per-op choices for this "
        "model and work as they do on any encoder.");
  ctx.host_ln = !ctx.on_array("layn");
  ctx.host_sm = !ctx.on_array("softm");
  ctx.host_gelu = !ctx.on_array("gelu");
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

  // One NpuAttention per tier for ONE lane, in the same order as the tier
  // tables the encoder holds. Declared at function scope because the extra-lane
  // loop at the bottom of this function needs it too.
  //
  // The geometry checks are Whisper's, restated for an embedder: the score
  // chunk one dispatch produces IS the query chunk the next GEMM reads, so
  // attn_qk's M is the row count and attn_av's K is attn_qk's N, and both are
  // the PADDED n_kv. A set that says otherwise is not one of ours, and each
  // message names the two numbers rather than the stream.
  //
  // A LAMBDA, called once per lane, because a lane needs its own buffers on
  // the shared design and its own scratch: sharing one instance across lanes
  // is the failure the eltwise slots below already describe, where the
  // winner is thread-scheduling dependent and the same request answers
  // differently from one call to the next.
  // The runtime's dispatch mutex, declared HERE rather than where the extra
  // lanes are created forty lines below: build_attn_lane() needs it too. This
  // is shared mutable state on the Design (see NpuAttention::set_npu_mutex),
  // and a lane whose attention dispatches unlocked is a lane that can tear
  // another lane's binding mid-sync.
  static std::mutex npu_mutex;

  auto build_attn_lane = [&](app::Pool &p) {
  // Used ONLY in the error messages below, and it is the label rather than a path
  // for the reason prefer_embedded gives one: on a self-sufficient container
  // ctx.art is empty, and every one of those messages would have opened with "/".
  const std::string dir =
      npu::prefer_embedded(ctx.model.get(), ctx.art, "gemm_rtp").label();
  const int64_t hd = app::g_head_dim;
  RunContext::AttnLane lane;
  // The array softmax for THIS lane. Only when softm was asked for; attn
  // alone leaves it on the host, and that is a combination the status
  // block has to be able to report.
  if (ctx.on_array("softm") && ctx.ld_sm) {
    lane.softmax = std::make_unique<npue::whisper::NpuEltwise>(
        *ctx.ld_sm, p, npue::whisper::EltwiseKind::Softmax);
    lane.softmax->alloc_buffers();
    lane.softmax->npu_mu = &npu_mutex;
  }
  for (const auto &row : ctx.enc->tier_slots) {
    const app::StreamEntry *qk = nullptr, *av = nullptr;
    for (const auto &s : ctx.streams) {
      if (static_cast<size_t>(s.slot) == row[4]) qk = &s;
      if (static_cast<size_t>(s.slot) == row[5]) av = &s;
    }
    if (!qk || !av)
      throw std::runtime_error(dir + ": tier slots 4 and 5 do not name "
                               "two streams of this set");
    if (qk->N != av->K)
      throw std::runtime_error(
          dir + ": attn_qk's N is " + std::to_string(qk->N) +
          " and attn_av's K is " + std::to_string(av->K) +
          ". The score chunk travels from one to the other as the A "
          "operand, so the two are the same padded n_kv.");
    if (qk->K < hd || qk->K % hd)
      throw std::runtime_error(
          dir + ": attn_qk's K is " + std::to_string(qk->K) +
          " and this container's head_dim is " + std::to_string(hd) +
          ". The Q operand of a score is one head, so K is the head "
          "width padded UP to the design's tile_k -- never down to a head, "
          "and never a value a head does not divide.");
    if (av->N < hd)
      throw std::runtime_error(
          dir + ": attn_av's N is " + std::to_string(av->N) +
          " and a head is " + std::to_string(hd) +
          " wide. The design pads this one UP to its own N granularity, "
          "never down to a head.");
    if (lane.softmax && lane.softmax->cols() != qk->N)
      throw std::runtime_error(
          dir + "/softmax has rows " +
          std::to_string(lane.softmax->cols()) +
          " wide and the attn streams' score row is " +
          std::to_string(qk->N) +
          ". The softmax design reduces along the whole row, so it has to "
          "be the width of the score row it is handed.");
    lane.attn.push_back(std::make_unique<npue::whisper::NpuAttention>(
        ctx.d_qkv(), p, qk->N, hd, av->N));
    lane.attn.back()->set_streams(static_cast<size_t>(qk->slot),
                                  static_cast<size_t>(av->slot), qk->M,
                                  qk->K);
    // The softmax follows --npu-ops softm, which is a SEPARATE decision:
    // attn on the array does not imply the softmax on it, and reading the
    // flag rather than assuming is what keeps the status line's two
    // entries independent. ctx.ld_sm is null unless softm was asked for,
    // which is exactly the condition for having something to hand over.
    lane.attn.back()->set_softmax(lane.softmax.get());
    lane.attn.back()->set_npu_mutex(&npu_mutex);
    lane.attn.back()->alloc_buffers();
  }
  return lane;
};

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
    // The legacy per-op set has all three resident and its own dispatch path, so
    // the list does not apply to it: ctx.unified is the switch, and on the
    // seven-design path every eltwise design is loaded whether or not it runs.
    if (!ctx.unified)
      ctx.host_ln = ctx.host_sm = ctx.host_gelu = true;
    ctx.enc->unified = true;
    ctx.enc->is_qkv = 0;
    ctx.enc->is_ao = 1;
    ctx.enc->is_fu = 2;
    ctx.enc->is_fd = 3;
    if (!ctx.streams.empty()) {
      std::set<int64_t> tset;
      for (const auto &s : ctx.streams) tset.insert(s.batch);
      // --npu-ops attn asked for the array. The stream slots and the row count
      // are per tier, so this is checked ONCE PER TIER and refused by name
      // rather than answered from the host: a set exported without the code has
      // no attn_qk/attn_av at all, and saying so is the difference between
      // "re-run the exporter" and a model that quietly computes the same
      // numbers it always did.
      const bool want_attn = ctx.on_array("attn");
      for (int64_t b : tset) {
        std::array<size_t, 6> slots{};
        bool complete = true;
        const char *ops[6] = {"qkv", "attn_out", "ffn_up", "ffn_down",
                              "attn_qk", "attn_av"};
        for (int k = 0; k < (want_attn ? 6 : 4); ++k) {
          auto it = std::find_if(ctx.streams.begin(), ctx.streams.end(),
                                 [&](const StreamEntry &s) {
                                   return s.batch == b && s.op == ops[k];
                                 });
          if (it == ctx.streams.end()) { complete = false; break; }
          slots[k] = static_cast<size_t>(it->slot);
        }
        if (!complete) {
          if (!want_attn) continue;   // a tier missing an op is not a tier
          // WHICH stream is missing decides what the message can honestly say.
          // A tier with no attn_qk/attn_av at all is a set exported without the
          // code; a tier whose four GEMMs are incomplete is something else
          // entirely, and conflating the two would send a reader to re-export
          // a set that was never the problem.
          const bool no_gemm = slots[0] == 0 || slots[1] == 0 || slots[2] == 0
                                   || slots[3] == 0;
          const char *missing = no_gemm ? "four GEMM streams (qkv/attn_out/"
                                          "ffn_up/ffn_down)"
                                        : "attn_qk/attn_av";
          // AND the CAUSE follows from the same split. Four GEMMs present but
          // no attention streams means the target could not size them:
          // resolve.py needs the model's window (`max_seq_len`, or `frames`
          // for a stt kind) for attn_qk's N and attn_av's K, and an entry
          // carrying neither is skipped over rather than guessed at, because
          // guessing Whisper's 1500 positions for a model with 256 builds a
          // design that cannot answer for it. So for THIS case a re-export of
          // the same targets file would produce the same set, and saying only
          // "re-export" would send a reader round a loop. Naming the key to
          // add is the difference between an instruction and a loop; the four-
          // GEMM case gets no such hint because no key is missing there.
          const std::string cause =
              no_gemm
                  ? std::string(
                        ". The exporter builds every stream the target's op "
                        "list honours and prints the list it chose, so a set "
                        "without these was built against a container whose "
                        "registry did not honour `attn`, or against a "
                        "different model.")
                  : std::string(
                        ". Either this target's registry row does not honour "
                        "`attn`, or its entry in tools/data/npu_targets.json "
                        "carries no window (`max_seq_len`, or `frames` for a "
                        "stt kind), which is what attn_qk's N and attn_av's K "
                        "are sized from -- the exporter builds the rest of the "
                        "set and leaves these two out rather than inventing a "
                        "width. Add the key, then re-export.");
          throw std::runtime_error(
              std::string("--npu-ops attn was given, but the loaded design set "
                          "carries no ") +
              missing + " at batch tier " + std::to_string(b) + cause +
              " Re-run: python "
              "tools/export/export_gemm_rtp.py --target <model> --arch " +
              std::to_string(ctx.design_arch ? ctx.design_arch : 1) +
              " --out " + ctx.art + " --artifacts " + ctx.art +
              ". Or drop attn from --npu-ops and run the host path, which is "
              "the measured-faster one.");
        }
        ctx.enc->tiers.push_back(b);
        ctx.enc->tier_slots.push_back(slots);
      }
      if (want_attn) {
        ctx.attn_lanes.push_back(build_attn_lane(pool));
        for (auto &a : ctx.attn_lanes[0].attn) ctx.enc->attns.push_back(a.get());
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
  // The op's CODE is printed, not its long name: the code is what --npu-ops
  // takes, and a status line that names something you cannot type is one more
  // thing to translate at the terminal.
  auto where = [](const char *code, bool host, const npu::Design &d,
                  const char *host_note) {
    const NpuOp *op = find_npu_op(code);
    if (host) {
      std::printf("  %-6s %-10s on the HOST (fp32) -- %s\n", code,
                  op ? op->long_name : "?", host_note);
    } else {
      std::printf("  %-6s %-10s on the ARRAY (%s, arch %lld%s%s)\n", code,
                  op ? op->long_name : "?",
                  d.info().name.empty() ? d.info().kind.c_str()
                                       : d.info().name.c_str(),
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
    where("softm", ctx.host_sm, ctx.d_sm(), sm_note);
    where("layn", ctx.host_ln, ctx.d_ln(), ln_note);
    // AND `attn`, which was the one code this block stayed silent about. The
    // silence was not neutral: a `--npu-ops attn` run puts 2 GEMMs x heads x
    // layers x query chunks on attn_qk/attn_av -- 4608 dispatches over the
    // block's 144 on bge-base, 15 texts -- and the block printed the three
    // elementwise lines and said nothing about attention, so the only place
    // the reader could learn that the array had taken it was the QK^T and
    // A*V rows of the time split further down. Every other mode already
    // prints this line (stt_mode, vit_mode, gemma_mode); the BERT path was
    // the one that did not, and it is the path the `attn` measurements in
    // the registry were taken on.
    //
    // Read off `attn_`, which use_tier() resolves from the loaded stream
    // table rather than from the flag -- the same rule this block opens
    // with. The design named is the gemm set's, because attn_qk and attn_av
    // are streams INSIDE it (the slot lookup above) and not a set of their
    // own, so there is no d_attn() to name.
    where("attn", ctx.enc->attn_ == nullptr, ctx.d_qkv(),
          "QK^T and softmax.V as host passes, no attn_qk/attn_av dispatches");
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
      // The host path's untiled operand goes with them, by the same rule as
      // everything two lines above. The shared_ptr is copied, NOT the weights:
      // common/host_b.hpp deletes WeightCache's copy constructor so this line
      // cannot become a per-lane copy of ~340 MB of fp32 without somebody
      // noticing in review -- and the cache is mutexed because every lane asks
      // for the same tensor the first time a GEMM runs on the host.
      e2.host_w_ = ctx.enc->host_w_;
      // The tier table is POLICY, and every lane needs it. A lane without it
      // silently falls back to the pre-0037 flat slot contract (0,1,2,3),
      // which under the 16-stream export selects the wrong shapes entirely --
      // measured as 1-cos 1.0 on whichever chunk that lane happened to take.
      e2.tiers = ctx.enc->tiers;
      e2.tier_slots = ctx.enc->tier_slots;
      // Attention is POLICY like the tier table, and a lane without it would
      // quietly compute on the host while lane 0 computes on the array -- the
      // flag said the array and three quarters of the requests did not use it,
      // which is exactly the intention-versus-value failure this file keeps
      // fixing elsewhere. The instances are BUILT PER LANE, not shared: they
      // allocate their A and C buffers on the shared design, and one instance
      // under four concurrent lanes hands back whichever lane's rows finished
      // last. Same construction, same refusals, this lane's pool.
      if (!ctx.attn_lanes.empty() || ctx.enc->attns.size()) {
        ctx.attn_lanes.push_back(build_attn_lane(*ctx.pools[l]));
        for (auto &a : ctx.attn_lanes.back().attn) e2.attns.push_back(a.get());
      }
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
