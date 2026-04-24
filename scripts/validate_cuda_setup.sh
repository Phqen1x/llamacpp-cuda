#!/usr/bin/env bash
# validate_cuda_setup.sh — Manual environment validation for llamacpp-cuda builds.
# Run this on a machine with an NVIDIA GPU before releasing or troubleshooting.
#
# Usage:
#   ./scripts/validate_cuda_setup.sh [binary_path]
#
#   binary_path: optional path to llama-server binary
#                (default: ./build/bin/llama-server)

set -euo pipefail

BINARY_PATH="${1:-./build/bin/llama-server}"
PASS=0
FAIL=0
WARN=0

# Color helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL + 1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; WARN=$((WARN + 1)); }

echo "=============================="
echo " llamacpp-cuda Validation"
echo "=============================="
echo ""

# 1. Ubuntu version check
echo "--- Checking Ubuntu version ---"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "${ID}" == "ubuntu" ]] && [[ "${VERSION_ID}" == "22.04" || "${VERSION_ID}" == "24.04" ]]; then
        pass "Ubuntu ${VERSION_ID} (supported)"
    elif [[ "${ID}" == "ubuntu" ]]; then
        warn "Ubuntu ${VERSION_ID} (not officially tested; 22.04/24.04 recommended)"
    else
        warn "Not Ubuntu (${ID} ${VERSION_ID}). Build may still work."
    fi
else
    warn "Cannot read /etc/os-release"
fi

# 2. nvidia-smi
echo ""
echo "--- Checking nvidia-smi ---"
if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [ -n "${gpu_name}" ]; then
        pass "nvidia-smi found GPU: ${gpu_name}"
    else
        fail "nvidia-smi is installed but could not query a GPU"
    fi
else
    fail "nvidia-smi not found (NVIDIA driver not installed)"
fi

# 3. nvcc
echo ""
echo "--- Checking nvcc (CUDA Toolkit) ---"
if command -v nvcc &>/dev/null; then
    cuda_ver=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
    major="${cuda_ver%%.*}"
    if [[ "${major}" -ge 12 ]] 2>/dev/null; then
        pass "nvcc found, CUDA ${cuda_ver} (>= 12.x)"
    else
        warn "nvcc found, CUDA ${cuda_ver} (12.x+ recommended)"
    fi
else
    fail "nvcc not found (install cuda-toolkit-12-x)"
fi

# 4. Binary exists and is executable
echo ""
echo "--- Checking llama-server binary ---"
if [ -f "${BINARY_PATH}" ] && [ -x "${BINARY_PATH}" ]; then
    pass "Binary found and executable: ${BINARY_PATH}"
else
    fail "Binary not found or not executable: ${BINARY_PATH}"
    echo "       Hint: Pass the binary path as an argument: $0 /path/to/llama-server"
fi

# 5. --version flag
echo ""
echo "--- Testing llama-server --version ---"
if [ -f "${BINARY_PATH}" ]; then
    if "${BINARY_PATH}" --version &>/dev/null; then
        ver_output=$("${BINARY_PATH}" --version 2>&1 | head -1)
        pass "--version succeeded: ${ver_output}"
    else
        fail "--version exited non-zero"
    fi
else
    warn "Skipping --version (binary not found)"
fi

# 6. --list-devices mentions CUDA/NVIDIA
echo ""
echo "--- Testing llama-server --list-devices ---"
if [ -f "${BINARY_PATH}" ]; then
    list_output=$("${BINARY_PATH}" --list-devices 2>&1 || true)
    if echo "${list_output}" | grep -qi "cuda\|nvidia"; then
        pass "--list-devices output mentions CUDA/NVIDIA"
    else
        warn "--list-devices did not mention CUDA/NVIDIA (driver may not be loaded)"
        echo "       Output: ${list_output}"
    fi
else
    warn "Skipping --list-devices (binary not found)"
fi

# Summary
echo ""
echo "=============================="
echo " Summary"
echo "=============================="
printf "  %-8s %d\n" "PASS:"  "${PASS}"
printf "  %-8s %d\n" "WARN:"  "${WARN}"
printf "  %-8s %d\n" "FAIL:"  "${FAIL}"
echo ""

if [ "${FAIL}" -gt 0 ]; then
    echo -e "${RED}Validation FAILED — address the failures above before releasing.${NC}"
    exit 1
else
    echo -e "${GREEN}Validation PASSED${NC}"
fi
