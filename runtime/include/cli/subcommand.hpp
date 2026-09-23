//===- cli/subcommand.hpp --------------------------------------------*- C++ -*-===//
//
// Subcommand dispatch: routes `list`, `serve`, `embed`, `add`,
// `tokenize` to their handlers. Handlers are registered by the
// application and called with the original argc/argv so they can
// translate subcommand args to flag form and fall through to the
// same code path.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <functional>
#include <map>
#include <string>

namespace app {

class SubcommandDispatcher {
public:
    using Handler = std::function<int(int argc, char **argv)>;

    void register_handler(const std::string &name, Handler handler);

    // Returns true if argv[1] is a registered subcommand and the
    // handler was called. Exit code written to *exit_code.
    // Returns false if argv[1] is not a subcommand (caller should
    // process as the flag form).
    bool dispatch(int argc, char **argv, int &exit_code) const;

private:
    std::map<std::string, Handler> handlers_;
};

// Registers the built-in subcommands: list, serve, embed, add, tokenize.
// Each handler translates its subcommand arguments into the flag form
// and falls through to the same Runtime::run code path the flag form
// uses, so there is no second dispatch path to drift out of agreement.
void register_default_subcommands(SubcommandDispatcher &dispatcher);

}  // namespace app