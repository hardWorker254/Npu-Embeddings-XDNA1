#!/bin/bash
# Re-export every design set that predates `attn` being buildable.
#
# WHAT THIS CHANGED, measured, on docs/bus.jpg and two short texts:
#
#   before:  --npu-ops attn refused BY NAME -- "the loaded design set carries no
#             attn_qk/attn_av at batch tier 4"
#   after:   every model whose registry row says attn honours can put all of
#             gelu, layn, softm and attn on the array, with host LayerNorm,
#             softmax and GELU all at 0.0 ms
#
#   all-MiniLM-L6-v2    4 codes,  619 dispatches      bge-base-en-v1.5   4 codes, 1237
#   bge-small-en-v1.5   4 codes, 1237                 bge-large-en-v1.5  4 codes, 3241
#   bge-micro-v2        4 codes,  310                 nomic / gte         3 codes, 1225
#
# nomic and gte get three and not four because their FFN is GATED and the runtime
# refuses --npu-ops gelu for them by name, which is correct: a gated FFN has no
# separate activation pass. vit and whisper reach every code including conv.
#
# NONE of this is FASTER -- it is slower, by design and by measurement -- and the
# script exists because "the possibility exists" and "the shipped set carries it"
# were two different things, and the second was false for most of this tree.
#
# WHY THIS EXISTS, measured: the shipped set for bge-micro-v2 carried 12 streams
# (the four GEMM streams x three batch tiers) and NO attn_qk/attn_av, so
# `--npu-ops attn` was refused BY NAME with "the loaded design set carries no
# attn_qk/attn_av at batch tier 4". Re-exporting the same target gives 18 streams
# WITH both, and `--npu-ops gelu,layn,softm,attn` then puts every one of the four
# on the array -- 310 dispatches, host LayerNorm / softmax / GELU all 0.0 ms, and
# the embeddings agree with the host's to cos 0.99982. The sets were built on
# 2026-10-03, before attn was buildable; the registry has said `honours` since.
#
# tile_n IS READ FROM THE SET BEING REPLACED, NOT ASSUMED, and that is the second
# version of this script. It does not tile alike: bge-micro-v2's set is tile_n 48
# and bge-large-en-v1.5's is 32, and the container's B-operand layout_hash is
# compared at load. A re-export at the wrong width produces a set the runtime
# refuses with "resolved to ..., whose B-operand layout is 4d72..., but this
# container's is 52a4..." -- which is exactly what an earlier attempt at `-n 32`
# (POSE's width) produced on an embedder. Taking the number from the file being
# overwritten cannot get it wrong, and a set that is missing gets the exporter's
# default.
#
# pose and mppose are NOT in this list: their registry rows say `attn` is absent,
# which is true -- neither graph has attention -- so their 14- and 22-stream sets
# are complete as they are.
set -u
cd "$(dirname "$0")/.." || exit 1
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
PY=.venv/bin/python
LOG=/tmp/opencode/reexport.log
: > "$LOG"

have_attn() {   # 0 (true) when the set already carries attn_qk and attn_av
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(f"runtime/artifacts/{sys.argv[1]}/artifacts_npu1/gemm_rtp/design.json"))
ops = {s["op"] for s in d["streams"]}
sys.exit(0 if {"attn_qk", "attn_av"} <= ops else 1)
PY
}

tile_n_of() {   # the width the existing set was built at, or empty if there is none
  python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(f"runtime/artifacts/{sys.argv[1]}/artifacts_npu1/gemm_rtp/design.json"))
    print(int(d["b_layout"]["tile_n"]))
except Exception:
    print("")
PY
}

# THE REGISTRY DECIDES WHAT IS WORTH REBUILDING, not this script's memory of it.
# A model whose `attn` cell is `absent` has no attention to add -- pose, mppose and
# hands are conv-only -- so their sets are complete as they are and rebuilding one
# would churn measured artifacts for no gain. yolov8n-pose's 14 streams and
# mediapipe-pose's 22 are the right numbers for those graphs.
attn_buildable() {
  $PY - "$1" <<'PY2'
import json, sys
sys.path.insert(0, "tools/lib")
import npu_ops
t = json.load(open("tools/data/npu_targets.json"))
spec = t["models"][sys.argv[1]]
sys.exit(0 if npu_ops.registry_for(spec.get("kind"), sys.argv[1])["attn"][0]
         in (npu_ops.HONOURS, npu_ops.ON_ARRAY) else 1)
PY2
}

# BOTH SPELLINGS OF A MODEL, and the -i8 one is enumerated FROM THE DISK rather
# than from npu_targets.json, which has no int8 entries: an int8 set is written by
# `--target <base> --int8` and lands in artifacts/<base>-i8, so it is an artifact
# with no registry row behind it. Fifteen of them are on this machine.
#
# NONE OF THEM CAN BE CHECKED HERE, and that is stated rather than glossed: there
# is no int8 CONTAINER on this machine (models/ holds twelve, all float), so a
# rebuilt -i8 set can be read for its streams and cannot be run. The script verifies
# the streams and says the run is out of reach.
built=0; skipped=0; failed=0
# ONE substitution, and that is the fix for a bug this script had twice: a
# backslash continuation INSIDE $( ) collapses the line, so a second $( ) on the
# next line became ARGUMENTS TO THE FIRST COMMAND rather than a second word list.
# The int8 dirs were being passed to python, which ignored them, and the loop ran
# over eighteen names instead of thirty-three and reported "skipped 18" as though
# that were all of them.
model_list() {
  $PY - <<'PY2'
import glob, json, os
t = json.load(open("tools/data/npu_targets.json"))
names = list(t["models"])
# int8 sets live at artifacts/<base>-i8 and have no npu_targets.json entry, so
# they are enumerated from the disk rather than from the registry.
names += [os.path.basename(p) for p in sorted(glob.glob("runtime/artifacts/*-i8"))]
print(" ".join(names))
PY2
}

for spec in $(model_list); do
  base=${spec%-i8}
  if ! attn_buildable "$base"; then
    echo "SKIP  $spec  (its kind has no attn to build -- the set is complete)"; skipped=$((skipped+1)); continue
  fi
  dir="runtime/artifacts/$spec/artifacts_npu1"
  [ -d "$dir" ] || { echo "SKIP  $spec  (no artifacts on this machine)"; skipped=$((skipped+1)); continue; }
  if have_attn "$spec"; then
    echo "SKIP  $spec  (already carries attn_qk/attn_av)"; skipped=$((skipped+1)); continue
  fi
  tn=$(tile_n_of "$spec")
  nflag=""; [ -n "$tn" ] && nflag="-n $tn"
  dflag=""; case "$spec" in *-i8) dflag="--int8";; esac
  echo "BUILD $spec ${nflag:-(default width)} $dflag"
  if timeout 3000 "$PY" tools/export/export_gemm_rtp.py --target "$base" \
       --arch 1 --out runtime/ $nflag $dflag >>"$LOG" 2>&1 \
     && have_attn "$spec"; then
    echo "  ok   $spec"; built=$((built+1))
  else
    echo "  FAIL $spec  (export non-zero, or still no attn -- see $LOG)"; failed=$((failed+1))
  fi
done
echo "=== built $built, skipped $skipped, failed $failed ==="