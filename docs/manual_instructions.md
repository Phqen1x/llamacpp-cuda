# Manual Build Instructions

> **⚠️ Warning:** These instructions are provided as a reference only.
> The [GitHub Actions workflow](../.github/workflows/build-llamacpp-cuda.yml) is the
> authoritative build process for release artifacts. Builds produced manually may differ
> from official releases due to environment differences, library versions, or flags.
> Use these instructions for local development and debugging only.

---

## 🐧 Ubuntu Build Instructions

### Part 1 — Install Required Software

```bash
# Build tools
sudo apt update
sudo apt install -y cmake ninja-build git wget patchelf

# CUDA Toolkit (choose one):
# Option A: Latest from Ubuntu package manager
sudo apt install -y nvidia-cuda-toolkit

# Option B: Specific version via NVIDIA network repo (recommended for reproducibility)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# Verify installation
nvcc --version
nvidia-smi
```

### Part 2 — Clone llama.cpp

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

### Part 3 — Build llama.cpp with CUDA

```bash
# Set CUDA environment (adjust path if your CUDA is not at /usr/local/cuda)
export CUDA_PATH=/usr/local/cuda
export PATH=${CUDA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}

# Configure — replace "86" with your target sm_ value (75, 80, 86, 89, or 90)
cmake -B build -G Ninja \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86" \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_STATIC=OFF \
    -DGGML_RPC=ON \
    -DLLAMA_BUILD_BORINGSSL=ON \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release -j$(nproc)
```

To build for multiple architectures in a single binary (larger binary, covers all targets):

```bash
-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"
```

### Part 4 — Copy Required CUDA Libraries

Bundle the CUDA runtime libraries alongside the binary for portable distribution:

```bash
build_bin_path="llama.cpp/build/bin"

# Core math libraries (bundled)
cp -v /usr/local/cuda/lib64/libcublas.so*  "${build_bin_path}/" 2>/dev/null || echo "libcublas not found at /usr/local/cuda/lib64"
cp -v /usr/local/cuda/lib64/libcublasLt.so* "${build_bin_path}/" 2>/dev/null || echo "libcublasLt not found at /usr/local/cuda/lib64"
cp -v /usr/local/cuda/lib64/libcurand.so*  "${build_bin_path}/" 2>/dev/null || echo "libcurand not found at /usr/local/cuda/lib64"
```

> **Note on `libcuda.so`:** The CUDA driver library (`libcuda.so`) is part of the NVIDIA
> driver installation and **cannot be legally redistributed**. It is provided by the user's
> NVIDIA driver and must be present on the target system. Do not bundle it.

### Part 5 — Set RPATH for Portable Distribution

Patch the RPATH so the binary finds its bundled libraries regardless of `LD_LIBRARY_PATH`:

```bash
build_bin_path="llama.cpp/build/bin"

for file in "${build_bin_path}"/*.so "${build_bin_path}"/llama-server; do
    if [ -f "${file}" ] && ! [ -L "${file}" ]; then
        patchelf --set-rpath '$ORIGIN' "${file}" 2>/dev/null || true
    fi
done
```

After patching, you can run `llama-server` from the `build/bin` directory directly:

```bash
cd llama.cpp/build/bin
./llama-server --version
./llama-server --list-devices
./llama-server -m /path/to/model.gguf -ngl 99
```

---

## 🎯 GPU Architecture Reference

| GPU Target | Architecture | CMake Value | Representative GPUs |
|---|---|---|---|
| `sm_75` | Turing | `"75"` | RTX 2060/2070/2080 Ti, T4, Quadro RTX, Titan RTX |
| `sm_80` | Ampere (data center) | `"80"` | A100, A30 |
| `sm_86` | Ampere (consumer) | `"86"` | RTX 3060/3070/3080/3090, A10, A40, A5000, A6000, A4000 |
| `sm_89` | Ada Lovelace | `"89"` | RTX 4060/4070/4080/4090, L4, L40S, RTX Ada workstation |
| `sm_90` | Hopper | `"90"` | H100, H200 |

To target a specific GPU, pass its CMake value to `-DCMAKE_CUDA_ARCHITECTURES`. For example,
for an RTX 3090 (sm_86):

```bash
-DCMAKE_CUDA_ARCHITECTURES="86"
```

Forward compatibility: when you compile with `CMAKE_CUDA_ARCHITECTURES=86`, CMake generates
both native sm_86 code and PTX virtual code. The PTX allows the binary to JIT-compile for
newer architectures (sm_89, sm_90) at a small startup cost. For best performance, use the
binary that matches your GPU's native sm_ level.
