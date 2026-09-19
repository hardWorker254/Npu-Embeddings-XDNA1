//===- run_probes.hpp -------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the early-measurement probes soak_npu/soak_cpu
// PROBE-PAIR etc.). Each is one of the old main() loops, verbatim
// except that locals became RunContext members. Returns the process
// exit code when handled, or -1 to fall through to the encode path.
//
// SPDX-License-Identifier: Apache-2.0
#ifndef NPUEMBEDDINGS_RUN_PROBES_HPP
#define NPUEMBEDDINGS_RUN_PROBES_HPP

#include "run_context.hpp"
#include "host_kernels.hpp"  // now_s, cpu_seconds
#include <cmath>        // std::atof/atoi are in <cstdlib>
#include <cstdint>      // int64_t
#include <cstdio>
#include <cstring>
#include <vector>
#include <filesystem>

namespace app {

inline int maybe_probe_pair(RunContext &ctx) {
  // --probe-pair runs BEFORE the other five designs exist. If seven resident
  // contexts on an eight-column NPU are what forces the reconfiguration, then
  // alternating between two designs with only two loaded should be cheap. If it
  // costs the same as it does with seven loaded, the penalty is inherent to
  // switching and no amount of trimming the resident set will help.
  for (int i = 2; i < ctx.argc; ++i) if (std::string(ctx.argv[i]) == "--probe-pair") {
    npu::Design a(*ctx.dev, ctx.art + "/qkv"), b(*ctx.dev, ctx.art + "/ffn_up");
    const int reps = 100;
    std::printf("\n  probe-pair -- only 2 designs loaded, %d repeats\n", reps);
    auto run = [&](const char *label, npu::Design &x, npu::Design *y) {
      x.dispatch_only();
      if (y) y->dispatch_only();
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) { x.dispatch_only();
                                       if (y) y->dispatch_only(); }
      double us = (now_s() - t0) / (reps * (y ? 2 : 1)) * 1e6;
      std::printf("      %-22s %8.0f us\n", label, us);
    };
    run("qkv alone", a, nullptr);
    run("ffn_up alone", b, nullptr);
    run("qkv <-> ffn_up", a, &b);
    return 0;
  }

  return -1;
}

inline int maybe_probe_bo(RunContext &ctx) {
  // --probe-design <dir> measures ONE design's switch cost in isolation:
  // dispatch it alone, then alternate two contexts holding the same xclbin.
  // The difference is the switch, with compute subtracted out.
  //
  // The point is to compare designs of EQUAL WIDTH and very unequal
  // configuration complexity. A→A' already showed the cost does not depend on
  // how DIFFERENT the two configurations are; it does not follow that it is
  // independent of how MUCH configuration there is. If a trivial 1-column
  // passthrough switches as slowly as a 1-column GEMM, the per-column cost is
  // fixed and unreachable. If it is cheaper, dataflow complexity is a knob.
  // --probe-bo <design-dir> <chunk_mb> <count>: can this machine hold the XRT
  // buffers a wider model needs? bge-large wants ~1023 MB across two lanes
  // (604 MB of staged weights plus 209 MB of A/B/C per lane) against MiniLM's
  // 175 MB. Answering that with one command beats discovering it after a
  // four-xclbin build at h=1024.
  for (int i = 2; i < ctx.argc - 3; ++i)
      if (std::string(ctx.argv[i]) == "--probe-bo") {
    const std::string dir = ctx.root + "/runtime/" + ctx.argv[i + 1];
    const size_t chunk = static_cast<size_t>(std::atof(ctx.argv[i + 2]) * 1e6);
    const size_t count = static_cast<size_t>(std::atoi(ctx.argv[i + 3]));
    npu::Design d0(*ctx.dev, dir);
    std::printf("  probe      %zu x %.1f MB = %.1f MB, mode %s\n",
                count, chunk / 1e6, count * chunk / 1e6, npu::bo_mode_name());
    const double t0 = now_s();
    const size_t ok = d0.probe_alloc(chunk, count, true);
    std::printf("  allocated  %zu of %zu (%.1f MB) in %.2f s\n",
                ok, count, ok * chunk / 1e6, now_s() - t0);
    return ok == count ? 0 : 3;
  }

  return -1;
}

inline int maybe_probe_design(RunContext &ctx) {
  for (int i = 2; i < ctx.argc - 1; ++i)
      if (std::string(ctx.argv[i]) == "--probe-design") {
    const std::string dir = ctx.root + "/runtime/" + ctx.argv[i + 1];
    npu::Design a(*ctx.dev, dir), a2(*ctx.dev, dir);
    const int reps = 100;
    a.dispatch_only();
    double t0 = now_s();
    for (int r = 0; r < reps; ++r) a.dispatch_only();
    const double alone = (now_s() - t0) / reps * 1e6;
    a.dispatch_only();
    a2.dispatch_only();
    t0 = now_s();
    for (int r = 0; r < reps; ++r) { a.dispatch_only(); a2.dispatch_only(); }
    const double pair = (now_s() - t0) / (2 * reps) * 1e6;
    std::printf("  %-22s alone %7.0f us   A<->A' %7.0f us   switch %7.0f us\n",
                ctx.argv[i + 1], alone, pair, pair - alone);
    return 0;
  }

  return -1;
}

inline int maybe_probe_insts(RunContext &ctx) {
  // --probe-insts <dirA> <dirB>: THE step-0 measurement for the one-xclbin
  // architecture (Roesti et al., FCCM 2025: keep one static design and
  // vary only the runtime sequence).
  //
  // dirA and dirB hold the SAME static design with two different runtime
  // sequences: their xclbins differ only in UUID metadata (verified byte by
  // byte before this probe existed), their insts.bin differ. Load dirA's
  // xclbin ONCE, its context ONCE, both instruction streams -- and alternate.
  //
  //   alternation ~= alone      -> the design switch is gone. Every operation
  //                                whose static design can be shared becomes an
  //                                instruction stream, and the 49 switches per
  //                                encode (~60 ms at batch 128) simply vanish.
  //   alternation ~= two-context cost -> the switch is tied to the instruction
  //                                stream itself, and the one-xclbin road ends.
  for (int i = 2; i < ctx.argc - 2; ++i)
      if (std::string(ctx.argv[i]) == "--probe-insts") {
    const std::string da = ctx.root + "/runtime/" + ctx.argv[i + 1];
    const std::string db = ctx.root + "/runtime/" + ctx.argv[i + 2];
    npu::Design a(*ctx.dev, da);
    const size_t sB = a.load_instr(db + "/insts.bin");
    npu::Design a2(*ctx.dev, da);              // control: second context, same bytes
    const int reps = 100;
    std::printf("\n  probe-insts -- %d repeats\n", reps);
    auto once = [&](const char *label, auto &&body) {
      body();                             // warm
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) body();
      std::printf("      %-38s %8.0f us\n", label,
                  (now_s() - t0) / reps * 1e6);
    };
    // Correctness first: completion status is not data. Both sequences copy
    // input to output (in different task order), so with a ramp staged in,
    // each stream must reproduce it exactly.
    {
      const size_t n = a.info().buffer_bytes[0] / 2;
      auto *in = static_cast<uint16_t *>(a.host_ptr(0));
      for (size_t j = 0; j < n; ++j) in[j] = static_cast<uint16_t>(j * 2654435761u >> 16);
      a.sync_to_device(0);
      for (size_t slot : {size_t(0), sB}) {
        auto *out = static_cast<uint16_t *>(a.host_ptr(1));
        std::memset(out, 0, a.info().buffer_bytes[1]);
        a.sync_to_device(1);
        a.bind_instr(slot);
        a.dispatch_only();
        a.sync_from_device(1);
        size_t bad = 0;
        for (size_t j = 0; j < n; ++j) bad += (out[j] != in[j]);
        std::printf("      stream %zu output: %s (%zu of %zu wrong)\n", slot,
                    bad ? "WRONG" : "exact", bad, n);
        if (bad) return 1;
      }
      a.bind_instr(0);
    }

    once("A alone (stream 0)", [&] { a.dispatch_only(); });
    a.bind_instr(sB);
    once("B alone (stream 1, same context)", [&] { a.dispatch_only(); });
    once("A <-> B, ONE context, two streams", [&] {
      a.bind_instr(0);
      a.dispatch_only();
      a.bind_instr(sB);
      a.dispatch_only();
    });
    once("A <-> A', TWO contexts (control)", [&] {
      a.bind_instr(0);
      a.dispatch_only();
      a2.dispatch_only();
    });
    // the paired loops dispatch twice per iteration
    std::printf("      (paired rows are per two dispatches; halve to"
                " compare)\n");
    return 0;
  }

  return -1;
}

inline int maybe_probe_rtp(RunContext &ctx) {
  // --probe-rtp <dirA> <dirB>: the FUNCTIONAL half of one-xclbin step 1.
  //
  // dirA and dirB are two RTP-ified GEMM shapes whose static configurations
  // are byte-identical modulo UUIDs (gemm_rtp_probe.py verified that). Load
  // dirA's xclbin ONCE, both instruction streams, and run BOTH shapes through
  // the one context -- each stream carries its own shim BDs and its own RTP
  // writes (loop bounds), so the same ELF computes different shapes.
  //
  // Correctness by the constant-B trick: with every element of B equal to c,
  // C[i,j] = c * sum_k A[i,k] regardless of B's tiled layout -- so the host
  // reference needs no de-tiling and any wrong loop bound, routing or RTP
  // value shows up as a wrong sum.
  for (int i = 2; i < ctx.argc - 2; ++i)
      if (std::string(ctx.argv[i]) == "--probe-rtp") {
    const std::string da = ctx.root + "/runtime/" + ctx.argv[i + 1];
    const std::string db = ctx.root + "/runtime/" + ctx.argv[i + 2];
    npu::Design a(*ctx.dev, da);
    const size_t sB = a.load_instr(db + "/insts.bin");
    // read dirB's shape
    npu::Design binfo(*ctx.dev, db);
    const int64_t M0 = a.info().M, K0 = a.info().K, N0 = a.info().N;
    const int64_t M1 = binfo.info().M, K1 = binfo.info().K,
                  N1 = binfo.info().N;
    std::printf("\n  probe-rtp -- one xclbin (%s), two shapes\n",
                ctx.argv[i + 1]);
    // This probe reads C as fp32 directly. A --c-bf16 artifact would still
    // "work" and produce plausible-looking wrong sums, which is the exact
    // failure mode tasks/0009 and CLAUDE.md trap 6c are about. Refuse.
    if (a.info().c_elem_bytes != 4)
      throw std::runtime_error(
          "--probe-rtp reads C as fp32; this design emits bf16 C "
          "(tasks/0045). Use an artifact set exported without --c-bf16.");

    const float cB = 0.5f;
    auto run_shape = [&](size_t slot, int64_t M_, int64_t K_, int64_t N_,
                         const char *label) {
      auto *pa = static_cast<uint16_t *>(a.host_ptr(0));
      std::vector<float> arow(static_cast<size_t>(M_ * K_));
      for (size_t j = 0; j < arow.size(); ++j)
        arow[j] = 0.001f * static_cast<float>((j * 37) % 200) - 0.1f;
      bf16_fill(pa, arow.data(), arow.size());
      a.sync_to_device(0);
      auto *pb = static_cast<uint16_t *>(a.host_ptr(1));
      const size_t nb = static_cast<size_t>(K_ * N_);
      const uint16_t cbits = to_bf16(cB);
      for (size_t j = 0; j < nb; ++j) pb[j] = cbits;
      a.sync_to_device(1);
      std::memset(a.host_ptr(2), 0,
                  static_cast<size_t>(M_ * N_) * sizeof(float));
      a.sync_to_device(2);
      a.bind_instr(slot);
      a.dispatch_only();
      a.sync_from_device(2);
      const float *c = static_cast<const float *>(a.host_ptr(2));
      double worst = 0.0;
      for (int64_t r = 0; r < M_; ++r) {
        float sum = 0.f;
        for (int64_t kk = 0; kk < K_; ++kk)
          sum += from_bf16(to_bf16(arow[static_cast<size_t>(r * K_ + kk)]));
        const float want = from_bf16(cbits) * sum;
        for (int64_t j = 0; j < N_; ++j) {
          const double rel = std::abs(c[r * N_ + j] - want) /
                             std::max(1e-6, std::abs(double(want)));
          worst = std::max(worst, rel);
        }
      }
      std::printf("      %-22s worst rel err %.3e  %s\n", label, worst,
                  worst < 2e-2 ? "OK" : "WRONG");
      return worst < 2e-2;
    };
    bool ok = run_shape(0, M0, K0, N0, "stream 0 (own shape)");
    ok &= run_shape(sB, M1, K1, N1, "stream 1 (other shape)");
    ok &= run_shape(0, M0, K0, N0, "stream 0 again");
    if (!ok) return 1;

    const int reps = 100;
    auto once = [&](const char *label, auto &&body) {
      body();
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) body();
      std::printf("      %-34s %8.0f us\n", label,
                  (now_s() - t0) / reps * 1e6);
    };
    a.bind_instr(0);
    once("shape A alone", [&] { a.dispatch_only(); });
    a.bind_instr(sB);
    once("shape B alone", [&] { a.dispatch_only(); });
    once("A <-> B (per two dispatches)", [&] {
      a.bind_instr(0);
      a.dispatch_only();
      a.bind_instr(sB);
      a.dispatch_only();
    });
    return 0;
  }

  return -1;
}

inline int maybe_probe_ctx(RunContext &ctx) {
  // --probe-ctx separates two explanations that --probe cannot tell apart.
  //
  // Alternating designs costs ~1200 us more than repeating one. That could be
  // the array being RECONFIGURED (different configuration data must be loaded)
  // or the driver SWITCHING HARDWARE CONTEXTS (a fixed cost that does not care
  // what is in them). The test: load the SAME xclbin into two contexts and
  // alternate. Identical configuration, two contexts.
  //
  //   A <-> A' as expensive as A <-> B  ->  context switch; design width is
  //                                        irrelevant and only the number of
  //                                        switches can be reduced.
  //   A <-> A' cheap                    ->  reconfiguration; configuration
  //                                        volume is the lever.
  for (int i = 2; i < ctx.argc; ++i) if (std::string(ctx.argv[i]) == "--probe-ctx") {
    npu::Design a(*ctx.dev, ctx.art + "/qkv");
    npu::Design a2(*ctx.dev, ctx.art + "/qkv");        // same bytes, second context
    npu::Design b(*ctx.dev, ctx.art + "/ffn_up");
    const int reps = 100;
    std::printf("\n  probe-ctx -- %d repeats, %s\n", reps, ctx.art.c_str());
    auto pair = [&](const char *label, npu::Design &x, npu::Design &y) {
      x.dispatch_only();
      y.dispatch_only();
      double t0 = now_s();
      for (int r = 0; r < reps; ++r) { x.dispatch_only(); y.dispatch_only(); }
      std::printf("      %-34s %8.0f us\n", label,
                  (now_s() - t0) / (2 * reps) * 1e6);
    };
    a.dispatch_only();
    double t0 = now_s();
    for (int r = 0; r < reps; ++r) a.dispatch_only();
    std::printf("      %-34s %8.0f us\n", "qkv alone (one context)",
                (now_s() - t0) / reps * 1e6);
    pair("qkv <-> qkv, TWO contexts", a, a2);
    pair("qkv <-> ffn_up, two contexts", a, b);
    return 0;
  }

  return -1;
}

inline int maybe_soak_npu(RunContext &ctx) {
  // --soak-npu <seconds>: dispatch in a tight loop with NO host work at all,
  // for the energy control experiment (tasks/0034). The question it answers is
  // whether the RAPL package meter SEES the NPU: if package power does not
  // move above idle while the array is saturated, the meter does not cover the
  // NPU block and every NPU energy figure is a lower bound.
  //
  // One thread, one buffer set, no conversion, no sync -- the same
  // dispatch_only() loop --probe uses, held for a measurable duration.
  for (int i = 2; i < ctx.argc - 1; ++i)
      if (std::string(ctx.argv[i]) == "--soak-npu") {
    const double secs = std::atof(ctx.argv[i + 1]);
    const std::string dir =
        std::ifstream(ctx.art + "/gemm_rtp/design.json").good()
            ? ctx.art + "/gemm_rtp" : ctx.art + "/qkv";
    npu::Design d(*ctx.dev, dir);
    std::printf("  soak-npu   %s, %.1f s, dispatch only, zero host work\n",
                dir.c_str(), secs);
    d.dispatch_only();                              // warm
    const double t0 = now_s();
    int64_t n = 0;
    while (now_s() - t0 < secs) { d.dispatch_only(); ++n; }
    const double el = now_s() - t0;
    std::printf("  soak-npu   %lld dispatches in %.2f s  (%.0f us each)\n",
                (long long)n, el, el / n * 1e6);
    return 0;
  }

  return -1;
}

inline int maybe_soak_cpu(RunContext &ctx) {
  // --soak-cpu <seconds> [threads]: the mirror control. Busy fp32 AVX2 work,
  // no NPU at all, so the same meter can be shown to move for CPU load.
  for (int i = 2; i < ctx.argc - 1; ++i)
      if (std::string(ctx.argv[i]) == "--soak-cpu") {
    const double secs = std::atof(ctx.argv[i + 1]);
    int nt = 12;
    if (i + 2 < ctx.argc && std::isdigit(static_cast<unsigned char>(ctx.argv[i + 2][0])))
      nt = std::atoi(ctx.argv[i + 2]);
    std::printf("  soak-cpu   %.1f s on %d threads, no NPU\n", secs, nt);
    std::vector<std::thread> ts;
    std::vector<double> sink(nt, 0.0);
    const double t0 = now_s();
    for (int w = 0; w < nt; ++w)
      ts.emplace_back([&, w] {
        float acc = 1.0f;
        std::vector<float> buf(4096, 1.000001f);
        while (now_s() - t0 < secs)
          for (int r = 0; r < 64; ++r)
            for (size_t j = 0; j < buf.size(); ++j) acc = acc * 0.9999f + buf[j];
        sink[w] = acc;
      });
    for (auto &th : ts) th.join();
    std::printf("  soak-cpu   done in %.2f s (sink %.3f)\n", now_s() - t0,
                sink[0]);
    return 0;
  }

  return -1;
}

}  // namespace app

#endif  // NPUEMBEDDINGS_RUN_PROBES_HPP
