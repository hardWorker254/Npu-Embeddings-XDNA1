//===- pool.hpp -----------------------------------------------*- C++ -*-===//
//
// A persistent worker pool for the host-side passes (attention, conversions).
// Split out of main.cpp verbatim.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace app {

// A persistent pool, because attention is called 12 times per encode and
// spawning threads each time would cost more than it saves.
//
// The calling thread takes chunk 0 and participates, so `n` threads means
// n-1 spawned. Work is partitioned by (batch, head) pair, and every pair writes
// a disjoint slice of `scores` and `ctx`, so there is no sharing to guard.
class Pool {
public:
  explicit Pool(int n) : n_(n < 1 ? 1 : n) {
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
            if (--remaining_ == 0) cv_done_.notify_one();
          }
        }
      });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      quit_ = true;
    }
    cv_work_.notify_all();
    for (auto &t : workers_) t.join();
  }
  Pool(const Pool &) = delete;
  Pool &operator=(const Pool &) = delete;

  int size() const { return n_; }

  void run(const std::function<void(int, int)> &f) {
    if (n_ == 1) { f(0, 1); return; }
    {
      std::lock_guard<std::mutex> lk(m_);
      fn_ = f;
      remaining_ = n_ - 1;
      ++gen_;
    }
    cv_work_.notify_all();
    f(0, n_);
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] { return remaining_ == 0; });
  }

private:
  int n_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_work_, cv_done_;
  std::function<void(int, int)> fn_;
  int gen_ = 0, remaining_ = 0;
  bool quit_ = false;
};

}  // namespace app
