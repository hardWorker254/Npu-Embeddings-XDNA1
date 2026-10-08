//===- npuembeddings_main.cpp ----------------------------------------------*- C++ -*-===//
//
// npuembeddings -- the TEXT binary: embed, list, add, tokenize, serve.
//
// The three binaries are one codebase with three mains; see cli/family.hpp
// for what they do and do not share.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "cli/family.hpp"

// WHO THIS IS is decided here, once, and read everywhere else. Nothing asks
// argv[0] or /proc/self/exe: a binary that inferred its family from its own
// filename would run a different set of modes when it was copied or symlinked
// -- a decision no reader could see and nothing on the command line could
// explain.
int main(int argc, char **argv) {
    app::g_family = app::Family::Embed;
    app::g_bin = "npuembeddings";
    return app::entry(argc, argv);
}
