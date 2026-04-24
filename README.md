# llamacpp-cuda

[![GitHub release](https://img.shields.io/github/v/release/lemonade-sdk/llamacpp-cuda?label=release)](https://github.com/lemonade-sdk/llamacpp-cuda/releases/latest)
[![Release Date](https://img.shields.io/github/release-date/lemonade-sdk/llamacpp-cuda)](https://github.com/lemonade-sdk/llamacpp-cuda/releases/latest)
[![License](https://img.shields.io/github/license/lemonade-sdk/llamacpp-cuda)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Ubuntu-blue)](https://github.com/lemonade-sdk/llamacpp-cuda/releases/latest)
[![GPU Targets](https://img.shields.io/badge/GPU-sm__75%20sm__80%20sm__86%20sm__89%20sm__90-76b900)](https://github.com/lemonade-sdk/llamacpp-cuda/releases/latest)

Fresh builds of [llama.cpp](https://github.com/ggml-org/llama.cpp) with NVIDIA CUDA acceleration, used as the CUDA backend for [Lemonade Server](https://github.com/lemonade-sdk/lemonade).

> **Contribution & Support Notice**
>
> This repository is maintained by the [lemonade-sdk](https://github.com/lemonade-sdk) team as part of the
> [Lemonade Server](https://github.com/lemonade-sdk/lemonade) project infrastructure. It is primarily
> intended as a build artifact source for Lemonade's CUDA backend. Issues and PRs are welcome, but please
> note that the team's primary support focus is the main Lemonade project.

---

## Supported Devices

| GPU Target | Architecture | Representative GPUs |
|---|---|---|
| `sm_75` | Turing | RTX 2060/2070/2080, T4, Quadro RTX |
| `sm_80` | Ampere (data center) | A100, A30 |
| `sm_86` | Ampere (consumer) | RTX 3060/3070/3080/3090, A10, A40, A5000, A6000 |
| `sm_89` | Ada Lovelace | RTX 4060/4070/4080/4090, L4, L40S |
| `sm_90` | Hopper | H100, H200 |

> **Note on `libcuda.so`:** Unlike the toolkit libraries bundled in each release,
> `libcuda.so` is part of the NVIDIA driver and **cannot be redistributed**. A compatible
> NVIDIA driver must be installed on the target system.

---

## Requirements

| Requirement | Notes |
|---|---|
| Ubuntu 22.04 or 24.04 | Only Linux is supported |
| NVIDIA Driver 525+ | Required at runtime — provides `libcuda.so` |
| CUDA Toolkit 12.x | Required to build from source only |
| CMake 3.20+ | Required to build from source only |

---

## Using a Release

Download the zip for your GPU's `sm_` target from the [latest release](https://github.com/lemonade-sdk/llamacpp-cuda/releases/latest):

```
llama-ubuntu-cuda-sm_75-x64.zip   # Turing
llama-ubuntu-cuda-sm_80-x64.zip   # Ampere (data center)
llama-ubuntu-cuda-sm_86-x64.zip   # Ampere (consumer)
llama-ubuntu-cuda-sm_89-x64.zip   # Ada Lovelace
llama-ubuntu-cuda-sm_90-x64.zip   # Hopper
```

Extract and run:

```bash
unzip llama-ubuntu-cuda-sm_86-x64.zip -d llama-cuda
cd llama-cuda

# Verify
./llama-server --version
./llama-server --list-devices

# Run inference (offload all layers to GPU)
./llama-server -m /path/to/model.gguf -ngl 99
```

No `LD_LIBRARY_PATH` changes needed — RPATH is set to `$ORIGIN` so the bundled
`libcublas`, `libcublasLt`, and `libcurand` are found automatically.

---

## Building from Source

### 1. Install dependencies

```bash
sudo apt update
sudo apt install -y cmake ninja-build git wget patchelf

# Add NVIDIA CUDA network repo and install toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
nvidia-smi
```

### 2. Clone and build llama.cpp

Replace `86` with your GPU's `sm_` number (see table above):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

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

cmake --build build --config Release -j$(nproc)
```

To build a single binary covering all targets (larger binary):

```bash
-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"
```

### 3. Bundle CUDA libraries

```bash
build_bin="llama.cpp/build/bin"
cuda_lib="/usr/local/cuda/lib64"

cp -v "${cuda_lib}"/libcublas.so*   "${build_bin}/"
cp -v "${cuda_lib}"/libcublasLt.so* "${build_bin}/"
cp -v "${cuda_lib}"/libcurand.so*   "${build_bin}/"
```

### 4. Set RPATH for portable distribution

```bash
for file in "${build_bin}"/*.so* "${build_bin}"/llama-*; do
    if [ -f "${file}" ] && ! [ -L "${file}" ]; then
        patchelf --set-rpath '$ORIGIN' "${file}" 2>/dev/null || true
    fi
done
```

### 5. Verify

```bash
cd llama.cpp/build/bin
./llama-server --version
./llama-server --list-devices
```

See [docs/manual_instructions.md](docs/manual_instructions.md) for more detail.

---

## Automated CI Builds

Nightly builds run at 3:00 PM UTC via [GitHub Actions](.github/workflows/build-llamacpp-cuda.yml).
Each run builds one zip per GPU target and creates a tagged release (`b1000`, `b1001`, ...).

The workflow can also be triggered manually from the Actions tab with optional overrides
for `sm_targets`, `cuda_version`, and `llamacpp_version`.

### Triggering a manual build

1. Go to **Actions → Build Llama.cpp + CUDA → Run workflow**
2. Optionally override inputs (defaults build all sm_ targets with CUDA 12.6 from latest llama.cpp)
3. Click **Run workflow**

---

## Testing

### Validate your environment (no GPU required for most checks)

```bash
# Install pytest
pip install pytest

# Run non-GPU smoke tests against a build directory
LLAMACPP_ARTIFACT_DIR=./llama.cpp/build/bin pytest tests/

# Run all tests including GPU tests (requires NVIDIA GPU + driver)
LLAMACPP_ARTIFACT_DIR=./llama.cpp/build/bin pytest tests/ -m gpu
```

The `tests/test_build_smoke.py` file checks:
- Binary exists
- Required CUDA libraries (`libcublas`, `libcublasLt`) are present
- RPATH contains `$ORIGIN`
- `--version` exits 0 (GPU required)
- `--list-devices` mentions CUDA/NVIDIA (GPU required)

### Full environment validation script

```bash
./scripts/validate_cuda_setup.sh ./llama.cpp/build/bin/llama-server
```

This script checks Ubuntu version, `nvidia-smi`, `nvcc` version, binary executability,
`--version`, and `--list-devices`, and prints a PASS/WARN/FAIL summary.

---

## Release Asset Naming

```
llama-ubuntu-cuda-{sm_target}-x64.zip
```

Each zip contains `llama-server`, all llama.cpp binaries, and the bundled CUDA runtime
libraries. Release tags follow the sequential `b####` convention starting at `b1000`.

---

## License

MIT — see [LICENSE](LICENSE).
