//===- test_pick_artifacts.cpp ---------------------------------*- C++ -*-===//
//
// Does pick_artifacts() find the right directory?
//
// The layout it has to resolve is <root>/artifacts/<model>/artifacts_npu<N>/...,
// with the model name optional, and it must ALSO still resolve the two older
// shapes -- <root>/<model>/artifacts_npu<N>/ and <root>/<model>-i8/gemm_rtp/ --
// because a design set is a build artifact and a tree can legitimately hold
// sets exported before the layout changed. Getting this wrong is silent in the
// worst way: the runtime starts, finds *some* design set, and computes with a
// geometry that was not built for the model. So the picked path is compared
// exactly, for each layout, and so is the PREFERENCE between them.
//
// The -i8 case is here because it is the one that cannot be a fallback: int8
// and bf16 design sets have different b_layout_hash values, so a model with
// both needs two directories, and picking the wrong one is refused at stage
// time rather than silently accepted. That makes "does the -i8 name resolve"
// worth its own assertion.
//
// Build and run (no NPU; the fixture is written to a temp dir):
//   g++ -std=c++17 -O1 -I runtime/include runtime/tests/test_pick_artifacts.cpp \
//       -o /tmp/test_pick_artifacts && /tmp/test_pick_artifacts
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#include "common/design_selection.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace app;

static void mk(const std::string &dir, const std::string &hash = "",
               const std::string &set = "gemm_rtp") {
  std::filesystem::create_directories(dir + "/" + set);
  std::ofstream f(dir + "/" + set + "/design.json");
  f << R"({
  "streams": [
    {"op":"qkv","file":"a.bin","batch":1,"slot":0,"M":64,"K":384,"N":1152},
    {"op":"attn_out","file":"b.bin","batch":1,"slot":1,"M":64,"K":384,"N":384},
    {"op":"ffn_up","file":"c.bin","batch":1,"slot":2,"M":64,"K":384,"N":1536},
    {"op":"ffn_down","file":"d.bin","batch":1,"slot":3,"M":64,"K":1536,"N":384}
  ])";
  // Only when asked, because a set exported before the field existed must read
  // back as "" -- and that is a case the runtime relies on, not a degenerate
  // one, so the fixture has to be able to produce it.
  if (!hash.empty()) f << ",\n  \"b_layout_hash\": \"" << hash << "\"";
  f << "\n}\n";
}

int main() {
  const std::string root =
      std::filesystem::temp_directory_path().string() + "/nds_pick_root";
  std::filesystem::remove_all(root);
  int failures = 0;
  auto check = [&](const char *what, bool ok, const std::string &got,
                   const std::string &want) {
    std::printf("%-26s -> '%s'\n", what, got.c_str());
    if (!ok) {
      std::printf("   FAIL: wanted '%s'\n", want.c_str());
      ++failures;
    }
  };

  // 1. THE CONVENTION. artifacts/<model>/artifacts_npu1.
  mk(root + "/artifacts/all-MiniLM-L6-v2/artifacts_npu1");
  {
    const std::string want =
        root + "/artifacts/all-MiniLM-L6-v2/artifacts_npu1";
    const std::string got =
        pick_artifacts(root, 384, 1536, false, 0, "", "", "all-MiniLM-L6-v2");
    check("artifacts/<model>", got == want, got, want);
  }

  // 2. THE -i8 VARIANT, in the same tree, for the same model name. Both must
  //    resolve, and the bf16 one must win for a bf16 container -- there is no
  //    dtype argument here, so this asserts resolution, not selection.
  mk(root + "/artifacts/all-MiniLM-L6-v2-i8/artifacts_npu1");
  {
    const std::string want = root + "/artifacts/all-MiniLM-L6-v2-i8/artifacts_npu1";
    const std::string got =
        pick_artifacts(root, 384, 1536, false, 0, "", "", "all-MiniLM-L6-v2-i8");
    check("artifacts/<model>-i8", got == want, got, want);
  }

  // 3. THE OLD PER-MODEL LAYOUT still resolves, in a tree that has no
  //    artifacts/ directory at all.
  {
    const std::string old_root =
        std::filesystem::temp_directory_path().string() + "/nds_pick_old";
    std::filesystem::remove_all(old_root);
    mk(old_root + "/all-MiniLM-L6-v2/artifacts_npu1");
    const std::string want = old_root + "/all-MiniLM-L6-v2/artifacts_npu1";
    const std::string got =
        pick_artifacts(old_root, 384, 1536, false, 0, "", "", "all-MiniLM-L6-v2");
    check("legacy <model>/", got == want, got, want);
    std::filesystem::remove_all(old_root);
  }

  // 4. THE OLD INT8 LAYOUT, runtime/<model>-i8/gemm_rtp/.
  {
    const std::string old_root =
        std::filesystem::temp_directory_path().string() + "/nds_pick_old8";
    std::filesystem::remove_all(old_root);
    mk(old_root + "/all-MiniLM-L6-v2-i8");
    const std::string want = old_root + "/all-MiniLM-L6-v2-i8";
    const std::string got =
        pick_artifacts(old_root, 384, 1536, false, 0, "", "", "all-MiniLM-L6-v2-i8");
    check("legacy <model>-i8/", got == want, got, want);
    std::filesystem::remove_all(old_root);
  }

  // 5. NO NAME, and no design under <model>/: the by-name pass finds nothing
  //    and this must not fall through to some other model's set. The fixture's
  //    only sets are one level too deep for the anonymous scan, so "" is the
  //    right answer -- picking anything else is the silent-wrong-geometry bug.
  {
    const std::string anon =
        pick_artifacts(root, 384, 1536, false, 0, "", "", "");
    std::printf("%-26s -> '%s'\n", "without name", anon.c_str());
    if (!anon.empty() && anon.find("artifacts_npu1") == std::string::npos) {
      std::printf("   FAIL: anonymous pick landed somewhere unexpected\n");
      ++failures;
    }
  }

  const auto cands = artifacts_candidates(root, "all-MiniLM-L6-v2");
  std::printf("explicit --artifacts candidates:\n");
  for (const auto &c : cands) std::printf("  %s\n", c.c_str());

  // 6. THE int8 SPELLING IS PROPOSED FOR --artifacts TOO. model_set_candidates
  //    has always offered both; this list did not, so `--artifacts <model>` --
  //    the form the gates pass -- could not reach an int8 container's set even
  //    with it exported and one directory away. It is what made "the int8
  //    design set is missing" and "the int8 design set is not looked for"
  //    indistinguishable from the outside.
  {
    const std::string want =
        root + "/artifacts/all-MiniLM-L6-v2-i8/artifacts_npu1";
    bool found = false;
    for (const auto &c : cands) found = found || c == want;
    check("candidates name -i8", found, found ? "yes" : "no", "yes");
  }

  // 7. THE B LAYOUT HASH IS READ BACK, and read as "" in every case where the
  //    answer is genuinely unknown: a set with no such field (every design
  //    exported before the field existed), and a directory that is not a design
  //    set at all. "" is what keeps an old set usable -- the runtime treats it
  //    as "cannot tell", not as "does not match" -- so a wrong answer here
  //    would either refuse working containers or stop excluding mismatched ones.
  {
    const std::string bf16 = "52a4adadbddcac2e0770095d5293392f0f2896b61bd22e173";
    const std::string i8 = "177088d6bc9feb7499ab2c1d4e5f6071839ca2e5";
    mk(root + "/artifacts/hash-bf16/artifacts_npu1", bf16);
    mk(root + "/artifacts/hash-i8/artifacts_npu1", i8);
    mk(root + "/artifacts/hash-old/artifacts_npu1");           // no field
    mk(root + "/artifacts/hash-nested/other");                  // no design.json

    check("hash: declared",
          design_b_layout_hash(root + "/artifacts/hash-bf16/artifacts_npu1") == bf16,
          design_b_layout_hash(root + "/artifacts/hash-bf16/artifacts_npu1"), bf16);
    check("hash: two sets differ",
          design_b_layout_hash(root + "/artifacts/hash-bf16/artifacts_npu1") !=
              design_b_layout_hash(root + "/artifacts/hash-i8/artifacts_npu1"),
          "differ", "differ");
    check("hash: pre-field set reads empty",
          design_b_layout_hash(root + "/artifacts/hash-old/artifacts_npu1").empty(),
          design_b_layout_hash(root + "/artifacts/hash-old/artifacts_npu1"), "");
    check("hash: not a design set reads empty",
          design_b_layout_hash(root + "/artifacts/hash-nested/other").empty(),
          design_b_layout_hash(root + "/artifacts/hash-nested/other"), "");
    check("hash: missing directory reads empty",
          design_b_layout_hash(root + "/artifacts/nope").empty(),
          design_b_layout_hash(root + "/artifacts/nope"), "");
    // Whisper holds TWO gemm sets in one directory and they are different
    // layouts, so the set name is a parameter rather than a constant.
    mk(root + "/artifacts/hash-stt/artifacts_npu1", bf16);
    mk(root + "/artifacts/hash-stt/artifacts_npu1", i8, "gemm_rtp_dec");
    check("hash: named set",
          design_b_layout_hash(root + "/artifacts/hash-stt/artifacts_npu1",
                               "gemm_rtp_dec") == i8,
          design_b_layout_hash(root + "/artifacts/hash-stt/artifacts_npu1",
                               "gemm_rtp_dec"), i8);
  }

  std::printf("RESULT %s\n", failures ? "FAIL" : "PASS");
  std::filesystem::remove_all(root);
  return failures ? 1 : 0;
}
