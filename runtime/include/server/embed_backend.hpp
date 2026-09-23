#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "common/app_state.hpp"
#include "runtime/types.hpp"
#include "server/server.hpp"

namespace app {

struct EmbedBackend {
  size_t vocab_size = 0;
  std::vector<std::string> prompt_names;
  int64_t hidden = 0;
  int64_t seq = 0;
  std::function<std::vector<float>(const std::vector<std::string> &,
                                       const std::string &,
                                       int64_t *)> embed;
};

int serve_http(const EmbedBackend &be, const std::string &model_id, int port,
               const std::string &bind_addr);

}  // namespace app