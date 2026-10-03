//===- pool.hpp ---------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- persistent worker thread pool for host-side passes.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>

namespace app {

class Pool {
public:
  explicit Pool(int n);
  ~Pool();
  Pool(const Pool &) = delete;
  Pool &operator=(const Pool &) = delete;

  int size() const;
  void run(const std::function<void(int, int)> &f);

private:
  int n_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_work_, cv_done_;
  std::function<void(int, int)> fn_;
  // The generation, and how many workers have FINISHED the work of a
  // generation. Both are needed, and the second is cumulative rather than a
  // countdown: see run() for why a countdown releases early.
  int gen_ = 0;
  long completed_ = 0;
  bool quit_ = false;
};

}  // namespace app