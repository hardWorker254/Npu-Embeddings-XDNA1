// NpuEmbeddings -- is anything else using the array right now?
// SPDX-License-Identifier: Apache-2.0
//
// See include/npu_contention.hpp for why this exists (tasks/0044).
//
// PARSING NOTE
// ------------
// `xrt-smi examine -r all` prints each hw_context as a FOUR-line record in a
// pipe-delimited table, with the fields for one context spread down the rows
// rather than across them:
//
//   |PID                 |Ctx ID     |Submissions |Migrations  |Err  |Priority |
//   |Process Name        |Status     |Completions |Suspensions |     |GOPS     |
//   |Memory Usage        |Instr BO   |            |            |     |FPS      |
//   ...
//   |63804               |144        |7233        |0           |0    |Normal   |
//   |npuembed.exe        |Active     |7232        |0           |     |N/A      |
//   |1032 MB             |608 KB     |            |            |     |N/A      |
//   |                    |           |            |            |     |N/A      |
//
// So: a row whose first cell is all digits starts a record, the next row's
// first two cells are process name and status, and the row after that carries
// memory. We deliberately do NOT try to be clever about the header or the
// separator rows -- anything that is not digits-then-name-then-memory is
// skipped, and if the whole parse yields nothing we report that as a failure
// rather than as "no contention". A format change must fail closed too.

#include "npu_contention.hpp"

#include <cctype>
#include <cstdio>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#define POPEN _popen
#define PCLOSE _pclose
#else
#include <unistd.h>
#define POPEN popen
#define PCLOSE pclose
#endif

namespace npu {
namespace {

unsigned long this_pid() {
#ifdef _WIN32
  return static_cast<unsigned long>(GetCurrentProcessId());
#else
  return static_cast<unsigned long>(getpid());
#endif
}

std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// Split a "|a |b |c |" row into its cells. Returns empty for a non-row.
// The table is INDENTED six spaces under "HW Contexts:", so leading
// whitespace has to go first -- the first version of this checked
// line.front() == '|' and parsed exactly zero rows. It failed closed and said
// "the output format may have changed", which is how the bug was found.
std::vector<std::string> cells(const std::string &raw) {
  std::vector<std::string> out;
  const std::string line = trim(raw);
  if (line.empty() || line.front() != '|') return out;
  size_t i = 1;
  while (i <= line.size()) {
    size_t j = line.find('|', i);
    if (j == std::string::npos) break;
    out.push_back(trim(line.substr(i, j - i)));
    i = j + 1;
  }
  return out;
}

bool all_digits(const std::string &s) {
  if (s.empty()) return false;
  for (unsigned char c : s)
    if (!std::isdigit(c)) return false;
  return true;
}

// Where xrt-smi lives. The AMD NPU driver puts it in System32\AMD on this
// machine; C:\Xilinx\XRT\bin is the XRT-install location. PATH last, so an
// explicit install wins over whatever a shell happens to have.
const char *kCandidates[] = {
    "C:\\Windows\\System32\\AMD\\xrt-smi.exe",
    "C:\\Xilinx\\XRT\\bin\\xrt-smi.exe",
    "xrt-smi",
};

bool run_capture(const std::string &exe, std::string *out) {
  // 2>&1 so a "not recognised" message lands in `out` instead of the console.
  std::string cmd = "\"" + exe + "\" examine -r all 2>&1";
#ifdef _WIN32
  cmd = "\"" + cmd + "\"";  // cmd.exe eats the outer quotes
#endif
  FILE *p = POPEN(cmd.c_str(), "r");
  if (!p) return false;
  char buf[4096];
  out->clear();
  while (std::fgets(buf, sizeof(buf), p)) out->append(buf);
  int rc = PCLOSE(p);
  return rc == 0 && out->find('|') != std::string::npos;
}

}  // namespace

ContentionReport survey_contexts() {
  ContentionReport r;
  std::string text;
  for (const char *c : kCandidates) {
    if (run_capture(c, &text)) {
      r.tool_ran = true;
      r.tool_path = c;
      break;
    }
  }
  if (!r.tool_ran) {
    r.failure =
        "could not run `xrt-smi examine -r all` (tried System32\\AMD, "
        "C:\\Xilinx\\XRT\\bin, and PATH)";
    return r;
  }

  std::istringstream in(text);
  std::string l1, l2, l3;
  std::vector<std::string> lines;
  while (std::getline(in, l1)) lines.push_back(l1);

  const unsigned long me = this_pid();
  for (size_t i = 0; i + 1 < lines.size(); ++i) {
    auto a = cells(lines[i]);
    if (a.empty() || !all_digits(a[0])) continue;
    auto b = cells(lines[i + 1]);
    if (b.size() < 2 || b[0].empty()) continue;
    // b[0] is a process name, not another number -- guards against a table
    // whose numeric columns happen to stack.
    if (all_digits(b[0])) continue;

    ContextRow row;
    row.pid = std::strtoul(a[0].c_str(), nullptr, 10);
    row.process = b[0];
    row.status = b[1];
    if (i + 2 < lines.size()) {
      auto c = cells(lines[i + 2]);
      if (!c.empty()) row.memory = c[0];
    }
    r.rows.push_back(row);
    if (row.status == "Active" && row.pid != me) r.foreign_active.push_back(row);
    i += 2;  // consumed the record
  }

  if (r.rows.empty()) {
    // The tool ran and printed a table we could not read. That is a format
    // change, and it must not read as "nothing is running".
    r.tool_ran = false;
    r.failure =
        "`xrt-smi examine -r all` ran but no hw_context rows parsed -- the "
        "output format may have changed";
  }
  return r;
}

bool require_exclusive_npu(const ContentionReport &r, bool allow_override) {
  if (r.tool_ran && r.foreign_active.empty()) {
    std::printf("  npu        exclusive -- %zu hw_context(s), none Active but "
                "ours (%s)\n",
                r.rows.size(), r.tool_path.c_str());
    return true;
  }

  std::printf("\n");
  if (!r.tool_ran) {
    std::printf("  !! CANNOT VERIFY THE ARRAY IS IDLE: %s\n", r.failure.c_str());
    std::printf("     An absent data source is not a negative reading "
                "(tasks/0040).\n");
  } else {
    std::printf("  !! ANOTHER PROCESS IS USING THE NPU -- %zu Active "
                "hw_context(s) that are not ours:\n",
                r.foreign_active.size());
    for (const auto &c : r.foreign_active)
      std::printf("       pid %-8lu %-24s %-8s %s\n", c.pid, c.process.c_str(),
                  c.status.c_str(), c.memory.c_str());
    std::printf("     tasks/0044 measured 221.4 seq/s against a true 694.0 in "
                "exactly this state\n"
                "     (per-dispatch hardware wait 15,541 us against 3,024).\n");
  }

  if (allow_override) {
    std::printf("  !! --allow-contention given: continuing anyway. ANY NUMBER "
                "BELOW IS NOT AN NPU\n"
                "     PERFORMANCE CLAIM (CLAUDE.md rule 1).\n\n");
    return true;
  }
  std::printf("\n  Refusing to report a throughput. Close the other process, "
              "or pass --allow-contention\n"
              "  if you actually want a contended number and will label it as "
              "one.\n\n");
  return false;
}

}  // namespace npu
