# 0050 — research: the ATB paper (2511.16041), found via a user-supplied repo

**Goal.** The user linked `github.com/tagmanxdna/xdna2-qantizer-projetc` (a
French-language FastFlowLM/Qwen-9B decode analysis) and asked whether it holds
anything worth keeping.

**What was done.** Three targeted WebFetch passes over the repo README and the
arXiv HTML of the paper it cites, with a verbatim-quote verification round on
every load-bearing number (the first extraction got two of them wrong: the
ρ=4 example is 64×224×64 / 111 KB→57.4 KB, and the formula's ρ divides A's
term rather than multiplying it).

**Verdict on the repo itself: nothing to adopt.** It is decode-side (GEMV,
KV-cache throttling, W4A16) on FastFlowLM — the bandwidth-bound regime, the
opposite of our compute-bound encoder GEMM (0048/0049). Two incidental
corroborations: their ~30 GB/s effective GEMV bandwidth is near our measured
33 GB/s marginal (0010), and their "NPU < iGPU for decode" repeats F1. Their
own red-team section flags a cold-start xclbin-load artifact as a
measurement trap — the same class as our trap 7c/first-dispatch effects.

**The keeper is its citation: arXiv 2511.16041, "Can Asymmetric Tile
Buffering Be Beneficial?"** (UCLA Cong group + AMD's Melber; our SKU, our
toolchain). Indexed as [`research/papers/2511.16041.md`](https://arxiv.org/abs/2511.16041)
— the first web-only entry in the index (no PDF in `OthersResarch/`, no
manifest row; provenance noted in the summary).

What it changes here, all conditional on the T23/T20 datapath decision:

- **T23's upside re-priced**: 24.3 TFLOPS BFP16 GEMM demonstrated on our
  silicon (~8× our production array rate; 2.88× of it is microkernel
  hand-optimisation alone, 0.32 → 0.92 TFLOPS/core = ~100% of MMAC peak).
  0049's 2.9× (stock emulated path) is the decision's floor, not ceiling.
- **T19 gets a better mechanism**: shrink A's buffered M (ρ) instead of
  single-buffering B — same L1 relief, no overlap risk. Ceiling stays ≈1.08×
  per 0049.
- **Nothing for today's plain-bf16 path**: ATB feeds the MMAC; ours is idle.
- Independent confirmations: 63 KB L1, 2/2 core streams, 6/6 mem-tile ports,
  1.8 GHz, L2 aggregation ×4 rows / ×8 cols. C single-buffered in their
  scheme, as in our 0045 accumulator.

**Files touched.** `research/papers/2511.16041.md` (new),
`research/papers/INDEX.md` (row + web-only note),
`research/OPEN-THREADS.md` (T19, T23), `tasks/README.md`, `CLAUDE.md`.
