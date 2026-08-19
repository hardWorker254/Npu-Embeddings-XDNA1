// NpuEmbeddings -- is anything else using the array right now?
// SPDX-License-Identifier: Apache-2.0
//
// WHY THIS EXISTS
// ---------------
// tasks/0044 took a --bench that read 221.4 seq/s. The same command on the
// same binary minutes later read 694.0. The difference was a leftover
// npuembed.exe from an earlier session still holding an Active hw_context
// with 1,032 MB resident -- bge-large's weights. Per-dispatch hardware wait
// was 15,541 us against a true 3,024.
//
// CLAUDE.md rule 1 already says wall clock is never an NPU performance claim
// because the array is shared. What that rule does NOT cover is this case:
//
//   - Interleaving against the CPU (tasks/0040) fixes drift that hits BOTH
//     sides. A resident NPU context hits only the NPU side, so interleaving
//     makes the RATIO wrong, confidently.
//   - The contending process was OURS. Every existing caution is about other
//     people's workloads; a long measurement session produces this one by
//     itself.
//   - Nothing reported it. --bench printed bo-mode, model, tiers, alignment
//     and staging -- every input except the one that moved the number 3.1x.
//
// tasks/0040 ended with "record machine state BY TOOL", having found a
// hand-rolled battery check that reported ON BATTERY for a machine with no
// battery, because Win32_Battery returns nothing there and the else branch
// fired. The tool that knows about hw_contexts is xrt-smi. We were not asking
// it.
//
// FAIL CLOSED, and that includes not finding the tool. An absent data source
// is not a negative reading -- that is the same 0040 lesson -- so "xrt-smi is
// missing" refuses just as loudly as "someone else is Active". The escape
// hatch is explicit (--allow-contention) and it says so in the output, which
// is the difference between a considered override and a fail-open.

#pragma once

#include <string>
#include <vector>

namespace npu {

// One hw_context row as xrt-smi reports it.
struct ContextRow {
  unsigned long pid = 0;
  std::string process;   // e.g. "npuembed.exe"
  std::string status;    // e.g. "Active", "Idle"
  std::string memory;    // e.g. "1032 MB", or "N/A"
};

struct ContentionReport {
  bool tool_ran = false;        // xrt-smi was found AND produced a table
  std::string tool_path;        // which xrt-smi answered, for the record
  std::string failure;          // why tool_ran is false
  std::vector<ContextRow> rows; // every context, ours included
  std::vector<ContextRow> foreign_active;  // Active and not this process
};

// Run `xrt-smi examine -r all` and parse its HW Contexts table.
// Never throws; a failure is reported through ContentionReport::failure.
ContentionReport survey_contexts();

// Print the report, then:
//   - return true  when the array is ours alone;
//   - return false when it is not, or when we could not find out.
// `allow_override` = the --allow-contention flag: it downgrades a refusal to
// a loud warning and returns true. Nothing else does.
bool require_exclusive_npu(const ContentionReport &r, bool allow_override);

}  // namespace npu
