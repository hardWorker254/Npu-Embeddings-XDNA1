//===- runtime/runtime.hpp --------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- Main runtime class.
// Loads a .npue container, sets up the NPU device and designs,
// and runs the encode/benchmark pipeline.
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace npue {

class Runtime {
public:
    Runtime(const std::string &model_path, const std::string &root,
            const std::string &art);
    ~Runtime() = default;

    // Run the encode/benchmark pipeline. Returns exit code.
    int run(int argc, char **argv);

private:
    std::string model_path_;
    std::string root_;
    std::string art_;
};

}  // namespace npue