#!/usr/bin/env bash
# NpuEmbeddings -- one command from a fresh clone to a working build Python.
#
#   git clone <this repo> && cd <repo> && ./bootstrap.sh && source .venv/bin/activate
#   python tools/pipeline.py check
#
# What it does, in order:
#   1. makes .venv/ (if absent) with the Python the AMD wheels are built for
#   2. installs requirements.lock.txt -- the exact PyPI pins
#   3. downloads the four AMD/Xilinx wheels into .wheel-cache/ and installs
#      them, refusing any whose sha256 does not match requirements.amd.txt
#
# Idempotent: re-running adds nothing and breaks nothing. Everything already
# cached is reused, so a second run is a few seconds of pip doing nothing.
#
# Deliberately does NOT touch ~/.local: the whole point of .venv is that a
# global `pip install --user` cannot change what this build compiles with.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

VENV="$REPO/.venv"
CACHE="$REPO/.wheel-cache"
PY="$VENV/bin/python"
WANT_ABI="cp314"          # keep in sync with requirements.amd.txt

# --- 1. the interpreter --------------------------------------------------
# The AIE wheels are per-ABI (cp311..cp314), so the interpreter is not
# negotiable: a venv on 3.13 would find no wheel for mlir_aie_no_rtti.
PYTHON="${PYTHON:-python3}"
if [ ! -x "$PY" ]; then
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        echo "error: $PYTHON not found. Install Python 3.14, or point PYTHON= at it." >&2
        exit 1
    fi
    "$PYTHON" - <<'PY'
import sys
if sys.version_info[:2] != (3, 14):
    raise SystemExit(
        f"error: this tree pins the AMD wheels to Python 3.14 (ABI cp314), "
        f"but {sys.executable} is {sys.version.split()[0]}.\n"
        f"       Re-run with PYTHON=/path/to/python3.14, or edit AIE_ABI in "
        f"requirements.amd.txt to a tag your Python has wheels for.")
PY
    echo "==> creating .venv with $($PYTHON -c 'import sys;print(sys.version.split()[0])')"
    "$PYTHON" -m venv "$VENV"
else
    echo "==> reusing existing .venv"
fi

# --- 2. the PyPI pins ----------------------------------------------------
echo "==> installing requirements.lock.txt"
"$PY" -m pip install --upgrade --quiet pip
"$PY" -m pip install --quiet --no-input -r requirements.lock.txt

# --- 3. the AMD wheels ---------------------------------------------------
# requirements.amd.txt is KEY=VALUE so both bash and Python can read it.
mkdir -p "$CACHE"
. "$REPO/requirements.amd.txt" 2>/dev/null || true

sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# URL-decode a string (%2B -> +). The wheel filename pip gets must be literal,
# while the URL it came from is percent-encoded.
urldecode() {
    local s="$1" out="" i c
    for ((i = 0; i < ${#s}; i++)); do
        c="${s:i:1}"
        if [ "$c" = "%" ] && [ $((i + 2)) -lt ${#s} ]; then
            out+=$(printf "\\x${s:i+1:2}") && i=$((i + 2))
        else
            out+="$c"
        fi
    done
    printf '%s' "$out"
}

fetch() {  # $1=URL  $2=expected sha256
    local url="$1" want="$2" name path
    name="$(basename "$url")"
    name="$(urldecode "$name")"
    path="$CACHE/$name"

    if [ -f "$path" ]; then
        if [ "$(sha_of "$path")" = "$want" ]; then
            echo "    cached   $name"
            return 0
        fi
        echo "    stale    $name -- digest mismatch, re-downloading"
        rm -f "$path"
    fi

    echo "    downloading $name"
    curl -fsSL -o "$path" "$url"

    local got; got="$(sha_of "$path")"
    if [ "$got" != "$want" ]; then
        rm -f "$path"
        echo "error: sha256 mismatch for $name" >&2
        echo "       want $want" >&2
        echo "       got  $got" >&2
        echo "       This tag may have been rewritten upstream. requirements.amd.txt" >&2
        echo "       records what the working build used -- investigate before bumping." >&2
        exit 1
    fi
    echo "    verified $name"
}

echo "==> fetching AMD/Xilinx wheels (sha256-verified)"
fetch "$MLIR_AIE_NO_RTTI_URL" "$MLIR_AIE_NO_RTTI_SHA256"
fetch "$LLVM_AIE_URL"         "$LLVM_AIE_SHA256"
fetch "$MLIR_AIR_URL"         "$MLIR_AIR_SHA256"
# triton-xdna was the fourth, and is gone: nothing in this tree imports triton,
# and every aie.* symbol the exporters use comes from mlir_aie_no_rtti. Note that
# it was what PULLED IN mlir-air, so mlir-air is fetched in its own right above --
# and installing with --no-deps below means a missing wheel cannot be silently
# covered by another one's metadata.

echo "==> installing the toolchain (--no-deps: every requirement is already in the lock)"
"$PY" -m pip install --quiet --no-deps --no-index --no-input \
    "$CACHE"/mlir_aie_no_rtti-*.whl \
    "$CACHE"/llvm_aie-*.whl \
    "$CACHE"/mlir_air-*.whl

# --- verify --------------------------------------------------------------
echo
"$PY" - <<'PY'
import importlib.util as u, importlib.metadata as m, sys
# triton is deliberately NOT in this list. It was, and it asserted a package
# nothing imports -- which is how a 352 MB wheel stayed in the dependency set
# after the code that wanted it was gone. The list below is what
# tools/export/* actually does `import`.
mods = ["numpy", "torch", "transformers", "scipy", "PIL", "openai",
        "aie.iron", "aie.helpers.taplib", "aie.utils", "onnx"]
bad = [n for n in mods if u.find_spec(n) is None]
for n in ("mlir-aie-no-rtti", "mlir-air", "llvm-aie"):
    print(f"    {n:<18} {m.version(n)}")
if bad:
    sys.exit("error: missing " + ", ".join(bad))
print("    imports OK")
PY

echo
echo "==> next:  source .venv/bin/activate"
echo "==> then:  python tools/pipeline.py check"
