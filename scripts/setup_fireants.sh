#!/usr/bin/env bash
# setup_fireants.sh
# Clones the FireANTs repository, installs the package, and builds the fused
# CUDA operations required by segflow4d.
#
# Prerequisites:
#   - conda environment is already activated (e.g. conda activate segflow4d)
#   - git is available
#
# Usage:
#   bash setup_fireants.sh [--dir <clone-directory>]
#
# Options:
#   --dir   Directory into which FireANTs will be cloned.
#           Defaults to ./FireANTs next to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE_DIR="$(dirname "${SCRIPT_DIR}")/FireANTs"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      CLONE_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash setup_fireants.sh [--dir <clone-directory>]" >&2
      exit 1
      ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "==> Checking prerequisites..."

# Verify a conda environment is activated (and not just the base env).
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: No conda environment is active (\$CONDA_PREFIX is unset)." >&2
  echo "       Please run: conda activate <env-name>" >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-base}" == "base" ]]; then
  echo "WARNING: You are running in the 'base' conda environment." >&2
  echo "         This is likely unintended. Consider activating a dedicated env:" >&2
  echo "           conda activate <your-env>" >&2
  echo ""
  read -rp "         Continue anyway? [y/N] " _answer
  if [[ "${_answer}" != [yY] ]]; then
    exit 1
  fi
fi

echo "    Conda env    : ${CONDA_DEFAULT_ENV:-unknown} (${CONDA_PREFIX})"

if ! command -v python &>/dev/null; then
  echo "ERROR: python not found. Please activate your conda environment first." >&2
  exit 1
fi

PYTHON_VER=$(python --version)
echo "    Python       : ${PYTHON_VER}"

# ── Clone FireANTs ────────────────────────────────────────────────────────────
FIREANTS_REPO="https://github.com/rohitrango/FireANTs.git"

if [[ -d "${CLONE_DIR}/.git" ]]; then
  echo ""
  echo "==> FireANTs already cloned at ${CLONE_DIR} — pulling latest changes..."
  git -C "${CLONE_DIR}" pull
else
  echo ""
  echo "==> Cloning FireANTs into ${CLONE_DIR}..."
  git clone "${FIREANTS_REPO}" "${CLONE_DIR}"
fi

# ── Install the main fireants package ────────────────────────────────────────
echo ""
echo "==> Installing fireants (pip install .)..."
pip install "${CLONE_DIR}"

# ── Ensure nvcc version matches PyTorch CUDA build ───────────────────────────
# PyTorch must be compiled with the same CUDA version as nvcc used to build
# the fused ops. If the system nvcc differs, install the matching toolkit into
# the active conda environment so it takes precedence.
echo ""
echo "==> Checking CUDA version compatibility..."

TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || true)
if [[ -z "${TORCH_CUDA_VER}" ]]; then
  echo "WARNING: Could not determine PyTorch CUDA version. Is torch installed?" >&2
else
  echo "    PyTorch CUDA   : ${TORCH_CUDA_VER}"

  NVCC_VER=""
  if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    echo "    nvcc (system)  : ${NVCC_VER}"
  fi

  if [[ "${TORCH_CUDA_VER}" != "${NVCC_VER}" ]]; then
    echo "    CUDA mismatch — installing cuda-nvcc=${TORCH_CUDA_VER} into conda env..."
    if ! command -v conda &>/dev/null; then
      echo "ERROR: conda not found. Please install cuda-toolkit ${TORCH_CUDA_VER} manually." >&2
      exit 1
    fi
    # Install compiler + runtime/library headers — all are needed to build CUDA extensions
    conda install -y -c nvidia \
      "cuda-nvcc=${TORCH_CUDA_VER}.*" \
      "cuda-cudart-dev=${TORCH_CUDA_VER}.*" \
      "cuda-libraries-dev=${TORCH_CUDA_VER}.*"
  fi

  # Even when nvcc version already matched, cuda-cudart-dev (which provides
  # cuda_runtime.h) may still be missing — the cuda-nvcc package alone does
  # NOT include it.  Install it idempotently if the header is absent.
  # Conda places headers under targets/x86_64-linux/include/ rather than
  # directly under include/.
  # Determine the Python site-packages path dynamically
  _PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)

  _find_cuda_runtime_h() {
    local prefix="$1"
    local _candidates=(
      "${prefix}/include"
      "${prefix}/targets/x86_64-linux/include"
    )
    # Also check the pip-installed nvidia package if site-packages is known
    if [[ -n "${_PY_SITE:-}" ]]; then
      _candidates+=("${_PY_SITE}/nvidia/cuda_runtime/include")
    fi
    for d in "${_candidates[@]}"; do
      if [[ -f "${d}/cuda_runtime.h" ]]; then
        echo "${d}"
        return 0
      fi
    done
    return 1
  }

  CUDA_RUNTIME_H_DIR=""
  _find_cuda_runtime_h "${CONDA_PREFIX}" && CUDA_RUNTIME_H_DIR=$(_find_cuda_runtime_h "${CONDA_PREFIX}")
  if [[ -z "${CUDA_RUNTIME_H_DIR}" ]]; then
    echo "    cuda_runtime.h not found — installing cuda-cudart-dev=${TORCH_CUDA_VER}..."
    if ! command -v conda &>/dev/null; then
      echo "ERROR: conda not found. Please install cuda-cudart-dev manually:" >&2
      echo "       conda install -c nvidia cuda-cudart-dev=${TORCH_CUDA_VER}" >&2
      exit 1
    fi
    conda install -y -c nvidia "cuda-cudart-dev=${TORCH_CUDA_VER}.*"
  fi

  # Locate nvcc inside the active conda env (check bin/ directly to avoid
  # picking up nvcc from a different env when CONDA_PREFIX is the base).
  CONDA_NVCC=""
  if [[ -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    CONDA_NVCC="${CONDA_PREFIX}/bin/nvcc"
  else
    # Fallback: search within the env (but NOT the base miniconda tree)
    CONDA_NVCC=$(find "${CONDA_PREFIX}" -maxdepth 4 -name "nvcc" -type f 2>/dev/null | head -1)
  fi
  if [[ -n "${CONDA_NVCC}" ]]; then
    CONDA_CUDA_BIN=$(dirname "${CONDA_NVCC}")
    export PATH="${CONDA_CUDA_BIN}:${PATH}"
    export CUDA_HOME=$(dirname "${CONDA_CUDA_BIN}")
    ACTIVE_NVCC_VER=$("${CONDA_NVCC}" --version | grep -oP "release \K[0-9]+\.[0-9]+")
    echo "    nvcc (active)  : ${ACTIVE_NVCC_VER} → ${CONDA_NVCC}"
    if [[ "${ACTIVE_NVCC_VER}" != "${TORCH_CUDA_VER}" ]]; then
      echo "ERROR: Could not obtain nvcc ${TORCH_CUDA_VER} after conda install." >&2
      echo "       Found ${ACTIVE_NVCC_VER} at ${CONDA_NVCC}." >&2
      exit 1
    fi
  else
    echo "ERROR: nvcc not found in conda env after install. Check conda channel availability." >&2
    exit 1
  fi

  # Re-locate cuda_runtime.h now that CUDA_HOME is resolved, and export
  # CPATH so both nvcc and the host compiler can find the header.
  CUDA_RUNTIME_H_DIR=""
  for _search_root in "${CUDA_HOME}" "${CONDA_PREFIX}"; do
    if [[ -n "${_search_root}" ]]; then
      _find_cuda_runtime_h "${_search_root}" && CUDA_RUNTIME_H_DIR=$(_find_cuda_runtime_h "${_search_root}") && break
    fi
  done
  if [[ -n "${CUDA_RUNTIME_H_DIR}" ]]; then
    export CPATH="${CUDA_RUNTIME_H_DIR}:${CPATH:-}"
    echo "    cuda_runtime.h : ${CUDA_RUNTIME_H_DIR}/cuda_runtime.h"
  fi
fi

# ── Check GPU compute capability is supported by installed PyTorch ────────────
echo ""
echo "==> Checking GPU compute capability..."

python - <<'PYEOF'
import sys
try:
    import torch
except ImportError:
    print("WARNING: torch not importable, skipping arch check", file=sys.stderr)
    sys.exit(0)

if not torch.cuda.is_available():
    print("    No CUDA GPU detected — skipping arch check")
    sys.exit(0)

cc_major, cc_minor = torch.cuda.get_device_capability(0)
gpu_name = torch.cuda.get_device_name(0)
cc = f"{cc_major}.{cc_minor}"
print(f"    GPU            : {gpu_name}")
print(f"    Compute cap.   : sm_{cc_major}{cc_minor}")

# Build the set of supported archs from the installed torch
try:
    supported = torch.cuda.get_arch_list()
except AttributeError:
    # Older PyTorch versions may not have get_arch_list
    supported = []

print(f"    Supported archs: {supported}")

target = f"sm_{cc_major}{cc_minor}"

# CUDA forward compatibility: a binary compiled for sm_XA will run on any GPU
# with compute capability X.B where B >= A (same major version).  We also
# accept PTX ("compute_XX") entries — PTX for compute_XA is JIT-compiled and
# will run on any sm_XB with B >= A.
import re

def _parse_arch(name):
    """Return (major, minor) from 'sm_89', 'compute_86', 'sm_100', etc.

    CUDA arch naming convention:
      - 2 digits: first = major, second = minor  (sm_70 → 7.0, sm_89 → 8.9)
      - 3+ digits: all-but-last = major, last = minor (sm_100 → 10.0, sm_120 → 12.0)
    """
    m = re.match(r'(?:sm|compute)_(\d+)', name)
    if not m:
        return None
    digits = m.group(1)
    if len(digits) <= 2:
        return int(digits[0]), int(digits[1:])
    else:
        return int(digits[:-1]), int(digits[-1])

gpu_cap = (cc_major, cc_minor)
is_supported = False
if supported:
    for arch_name in supported:
        parsed = _parse_arch(arch_name)
        if parsed is None:
            continue
        arch_major, arch_minor = parsed
        # Same major version and the supported arch minor <= GPU minor
        # means forward compatibility covers this GPU.
        if arch_major == cc_major and arch_minor <= cc_minor:
            is_supported = True
            break

if supported and not is_supported:
    print(f"\nERROR: Your GPU ({gpu_name}, compute capability {cc}) is NOT supported", file=sys.stderr)
    print(f"       by the installed PyTorch (torch.version.cuda = {torch.version.cuda}).", file=sys.stderr)
    print(f"       The highest supported arch in this build is: {supported[-1]}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"       To fix: reinstall PyTorch built against a CUDA version that", file=sys.stderr)
    print(f"       supports sm_{cc_major}{cc_minor}. For Blackwell (sm_100) GPUs, CUDA >= 12.8", file=sys.stderr)
    print(f"       is required. Available PyTorch wheel tags: cu118, cu121, cu124, cu126, cu128", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"       Example (CUDA 12.8):", file=sys.stderr)
    print(f"         pip install torch torchvision \\", file=sys.stderr)
    print(f"           --index-url https://download.pytorch.org/whl/cu128 \\", file=sys.stderr)
    print(f"           --trusted-host download.pytorch.org", file=sys.stderr)
    sys.exit(1)
else:
    print(f"    GPU arch supported by installed PyTorch — OK")
PYEOF

# Propagate Python exit code
if [[ $? -ne 0 ]]; then
  exit 1
fi

# ── Build and install fused CUDA ops ─────────────────────────────────────────
FUSED_OPS_DIR="${CLONE_DIR}/fused_ops"

if [[ ! -d "${FUSED_OPS_DIR}" ]]; then
  echo "ERROR: fused_ops directory not found at ${FUSED_OPS_DIR}." >&2
  echo "       The FireANTs repository structure may have changed." >&2
  exit 1
fi

echo ""
echo "==> Building fused CUDA operations in ${FUSED_OPS_DIR}..."
pushd "${FUSED_OPS_DIR}" > /dev/null

# Clean previous build artefacts so that stale object files (possibly
# compiled with a different _GLIBCXX_USE_CXX11_ABI) are never reused.
if [[ -d build ]]; then
  echo "    Cleaning previous build directory..."
  rm -rf build
fi

# Clear stale environment variables that may inject wrong host-compiler paths
# into nvcc (e.g. NVCC_PREPEND_FLAGS=-ccbin ... left over from a previously
# activated conda environment with gcc_linux-64 / gxx_linux-64).
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CUDAHOSTCXX 2>/dev/null || true

# Verify cuda_runtime.h is reachable before attempting the build.
# Search the standard locations including conda's targets/ subdirectory.
CUDA_RT_FOUND=false
for _d in \
  "${CUDA_HOME:-}/include" \
  "${CUDA_HOME:-}/targets/x86_64-linux/include" \
  "${CONDA_PREFIX:-}/include" \
  "${CONDA_PREFIX:-}/targets/x86_64-linux/include"; do
  if [[ -f "${_d}/cuda_runtime.h" ]]; then
    CUDA_RT_FOUND=true
    # Ensure the directory is on CPATH so the compiler can find it.
    export CPATH="${_d}:${CPATH:-}"
    break
  fi
done
if ! ${CUDA_RT_FOUND}; then
  echo "ERROR: cuda_runtime.h not found. Install cuda-cudart-dev:" >&2
  echo "       conda install -c nvidia cuda-cudart-dev" >&2
  exit 1
fi

# ── Ensure host compiler is compatible with CUDA toolkit ─────────────────────
# CUDA 12.x requires GCC ≤ 12, CUDA 11.x requires GCC ≤ 11.
# If the default gcc is too new, look for a compatible version on the system and
# tell nvcc / PyTorch to use it via CC, CXX, and CUDAHOSTCXX.
CUDA_MAJOR=$(echo "${TORCH_CUDA_VER}" | cut -d. -f1)
case "${CUDA_MAJOR}" in
  12) MAX_GCC=12 ;;
  11) MAX_GCC=11 ;;
  *)  MAX_GCC=99 ;;   # unknown — skip the check
esac

GCC_VER=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
if [[ -n "${GCC_VER}" ]] && (( GCC_VER > MAX_GCC )); then
  echo "    System GCC ${GCC_VER} is too new for CUDA ${TORCH_CUDA_VER} (max GCC ${MAX_GCC})."
  # Search for a compatible gcc / g++ on the system
  COMPAT_GCC=""
  for v in $(seq "${MAX_GCC}" -1 9); do
    if command -v "gcc-${v}" &>/dev/null && command -v "g++-${v}" &>/dev/null; then
      COMPAT_GCC="${v}"
      break
    fi
  done

  if [[ -n "${COMPAT_GCC}" ]]; then
    echo "    Using gcc-${COMPAT_GCC} / g++-${COMPAT_GCC} as the host compiler."
    export CC="gcc-${COMPAT_GCC}"
    export CXX="g++-${COMPAT_GCC}"
    export CUDAHOSTCXX="g++-${COMPAT_GCC}"
  else
    echo "ERROR: No GCC ≤ ${MAX_GCC} found. Install one with:" >&2
    echo "       sudo apt install gcc-${MAX_GCC} g++-${MAX_GCC}" >&2
    exit 1
  fi
fi

# Explicitly set the arch list from the actual GPU(s) to avoid any value
# injected by conda activation scripts (e.g. from cuda-nvcc package).
GPU_ARCH_LIST=$(python -c "
import torch, sys
caps = sorted({f'{maj}.{min}' for maj,min in
               [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]})
print(';'.join(caps) if caps else '')
" 2>/dev/null || true)

if [[ -n "${GPU_ARCH_LIST}" ]]; then
  export TORCH_CUDA_ARCH_LIST="${GPU_ARCH_LIST}"
  echo "    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
fi

# ── Ensure CXX11 ABI matches PyTorch ──────────────────────────────────────────
# PyTorch pip wheels are typically built with _GLIBCXX_USE_CXX11_ABI=0 (old ABI).
# Conda-installed compilers default to the new ABI (=1).  A mismatch causes
# "undefined symbol" errors at import time (e.g. c10::Error with std::__cxx11::
# basic_string).  Force the fused ops build to use the same ABI as PyTorch.
TORCH_CXX11_ABI=$(python -c "import torch; print(int(torch.compiled_with_cxx11_abi()))" 2>/dev/null || echo "")
if [[ -n "${TORCH_CXX11_ABI}" ]]; then
  export CXXFLAGS="${CXXFLAGS:-} -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}"
  echo "    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}"
fi

python setup.py build_ext
python setup.py install

popd > /dev/null

# ── Verify installation ───────────────────────────────────────────────────────
echo ""
echo "==> Verifying installation..."
python - <<'EOF'
import importlib, sys

# Import torch first so its __init__.py adds the CUDA / C++ runtime libraries
# (libc10.so, libtorch.so, etc.) to the dynamic linker search path.
import torch

missing = []
for mod in ("fireants", "fireants_fused_ops"):
    try:
        importlib.import_module(mod)
        print(f"    [OK] {mod}")
    except ImportError as e:
        print(f"    [FAIL] {mod}: {e}")
        missing.append(mod)

if missing:
    print("\nERROR: Some modules failed to import. Check build output above.", file=sys.stderr)
    sys.exit(1)
else:
    print("\nFireANTs with fused ops installed successfully!")
EOF

# ── Clean up source directory ─────────────────────────────────────────────────
echo ""
echo "==> Removing FireANTs source directory (${CLONE_DIR})..."
rm -rf "${CLONE_DIR}"
echo "    Done."
