//===- pool.cpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- thread pool implementation. See pool.hpp.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "runtime/pool.hpp"

namespace app {

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
          if (--remaining_ == 0) cv_done_.notify_one();
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

void Pool::run(const std::function<void(int, int)> &f) {
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

}  // namespace app