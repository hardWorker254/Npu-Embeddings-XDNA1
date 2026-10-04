//===- pool.cpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- thread pool implementation. See pool.hpp.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "runtime/pool.hpp"

namespace app {

namespace {

// Whether THIS thread is inside some pool's run(). Nested is not supported -- the
// inner run() would wait for a completion count that includes the very worker
// waiting for it -- and the failure mode is a hang, not an error.
//
// It exists because a pool-taking function has become reachable from inside
// par regions. vit::resize_to takes an app::Pool* and the pose front end passes
// one, so anything that resizes from a worker would deadlock the process rather
// than misbehave visibly. Rather than rely on every future caller knowing that,
// the pool answers the question itself: a thread already inside a generation runs
// the work serially and returns. That costs a thread_local read per run() and
// turns a class of hang into a class of slowdown.
thread_local bool t_in_pool = false;

}  // namespace

Pool::Pool(int n) : n_(n < 1 ? 1 : n) {
  for (int i = 1; i < n_; ++i)
    workers_.emplace_back([this, i] {
      int seen = 0;
      for (;;) {
        std::function<void(int, int)> f;
        {
          std::unique_lock<std::mutex> lk(m_);
          cv_work_.wait(lk, [&] { return quit_ || gen_ != seen; });
          if (quit_) return;
          seen = gen_;
          f = fn_;
        }
        f(i, n_);
        {
          std::lock_guard<std::mutex> lk(m_);
          ++completed_;
          cv_done_.notify_one();
        }
      }
    });
}

Pool::~Pool() {
  {
    std::lock_guard<std::mutex> lk(m_);
    quit_ = true;
  }
  cv_work_.notify_all();
  for (auto &t : workers_) t.join();
}

int Pool::size() const { return n_; }

// The completion count is CUMULATIVE, not a countdown, and that is the whole
// point of this function.
//
// The workers' wait predicate is `gen_ != seen`, so a worker that is still
// inside generation g's body when generation g+1 is issued does not miss g+1 --
// it finishes g, then takes g+1. Every generation is therefore executed by
// every worker exactly once, in order, but not necessarily before the NEXT run()
// returns. With a countdown (`remaining_ = n_ - 1`, then `--remaining_` per
// worker) a straggler from generation g decrements generation g+1's counter: the
// count reaches zero one worker early, run() returns, and that worker is still
// executing the CALLER'S LAMBDA. The lambda captures its buffers by reference,
// so the caller's frame is gone -- the writes land wherever the stack now is.
//
// It is silent and it is rare: it needs a worker slow enough to still be running
// when the next run() is issued, which is why an embedder or a pose network on
// the host never hit it. The array path does: it issues three runs per GEMM and
// one GEMM per 1024 output rows, 6675 times for this network. What it corrupted
// was `NpuGemm::run`'s read-back -- par_rows() writing the C rows -- while the
// caller had already started transposing them into the output tensor, so conv 0
// produced 64 correct pixels and 102 336 rows of another chunk's numbers.
void Pool::run(const std::function<void(int, int)> &f) {
  if (n_ == 1) {
    f(0, 1);
    return;
  }
  // Already inside a generation on THIS thread: do the work here and return. See
  // t_in_pool for why. The flag brackets the caller's own share as well, so a
  // third level of nesting is caught the same way.
  if (t_in_pool) {
    f(0, n_);
    return;
  }
  long target = 0;
  {
    std::lock_guard<std::mutex> lk(m_);
    fn_ = f;
    ++gen_;
    // Every worker accounts for exactly one increment per generation, so the
    // count for generation g is (n_ - 1) * g whatever order they finish in. The
    // caller's own share runs on this thread and is not counted.
    target = static_cast<long>(n_ - 1) * gen_;
  }
  cv_work_.notify_all();
  t_in_pool = true;
  f(0, n_);
  t_in_pool = false;
  std::unique_lock<std::mutex> lk(m_);
  cv_done_.wait(lk, [&] { return completed_ >= target; });
}

}  // namespace app