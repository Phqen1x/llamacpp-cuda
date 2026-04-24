"""
gather_required_libs.py — Discover and copy CUDA runtime libraries required by a
llama.cpp CUDA binary.

Uses `ldd` to inspect a built llama-server binary, then copies all CUDA/NVIDIA
shared libraries to a destination directory. Libraries are identified by their
resolved path (must contain "cuda" or "nvidia" in the path) or by their name
(libcu* prefix).

This mirrors the purpose of the ROCm equivalent in llamacpp-rocm.

Usage:
    python gather_required_libs.py \
        --binary llama.cpp/build/bin/llama-server \
        --cuda-dir /usr/local/cuda \
        --dest-dir llama.cpp/build/bin

When to update the library list:
    If you add or remove GGML_CUDA features (e.g., enable flash attention, CUBLAS
    alternative), re-run this script against a fresh build to ensure all required
    libraries are captured. The `ldd` approach automatically picks up new dependencies
    without hardcoding library names.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_ldd_output(binary_path: str) -> dict[str, str]:
    """Run ldd on binary and return {lib_name: resolved_path} for each entry."""
    try:
        result = subprocess.run(
            ["ldd", binary_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ldd failed for {binary_path}: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: ldd not found. This script requires Linux.", file=sys.stderr)
        sys.exit(1)

    libs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        # Example lines:
        #   libcublas.so.12 => /usr/local/cuda/lib64/libcublas.so.12 (0x...)
        #   linux-vdso.so.1 (0x...)
        #   /lib/x86_64-linux-gnu/libc.so.6 (0x...)
        if "=>" in line:
            parts = line.split("=>")
            lib_name = parts[0].strip()
            rest = parts[1].strip()
            resolved = rest.split("(")[0].strip()
            if resolved and resolved != "not found":
                libs[lib_name] = resolved
        elif line.startswith("/"):
            # Absolute path without "=>"
            path = line.split("(")[0].strip()
            if path:
                libs[Path(path).name] = path

    return libs


def is_cuda_library(lib_name: str, resolved_path: str) -> bool:
    """Return True if this library is a CUDA/NVIDIA runtime library to bundle."""
    path_lower = resolved_path.lower()
    name_lower = lib_name.lower()

    # Match by resolved path containing cuda or nvidia directories
    if "/cuda/" in path_lower or "/nvidia/" in path_lower:
        return True

    # Match by library name prefix (libcu* covers cublas, curand, cusolver, etc.)
    if name_lower.startswith("libcu"):
        return True

    # nvidia-specific libraries
    if name_lower.startswith("libnv"):
        return True

    return False


def copy_with_symlink_resolution(src: Path, dest_dir: Path) -> bool:
    """Copy src to dest_dir, resolving symlinks to get the real file."""
    try:
        real_src = src.resolve()
        dest_file = dest_dir / src.name

        if not real_src.exists():
            return False

        shutil.copy2(str(real_src), str(dest_file))
        return True
    except (OSError, shutil.Error) as e:
        print(f"  ERROR copying {src}: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and copy CUDA runtime libraries required by llama-server"
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="Path to the llama-server binary",
    )
    parser.add_argument(
        "--cuda-dir",
        default="/usr/local/cuda",
        help="CUDA installation directory (default: /usr/local/cuda)",
    )
    parser.add_argument(
        "--dest-dir",
        required=True,
        help="Destination directory to copy libraries into",
    )
    args = parser.parse_args()

    binary_path = Path(args.binary)
    dest_dir = Path(args.dest_dir)
    cuda_dir = Path(args.cuda_dir)

    if not binary_path.exists():
        print(f"ERROR: Binary not found: {binary_path}", file=sys.stderr)
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting: {binary_path}")
    print(f"CUDA dir:   {cuda_dir}")
    print(f"Dest dir:   {dest_dir}")
    print()

    all_libs = parse_ldd_output(str(binary_path))
    cuda_libs = {
        name: path
        for name, path in all_libs.items()
        if is_cuda_library(name, path)
    }

    # Table header
    col1, col2, col3 = 30, 60, 8
    print(f"{'Library':<{col1}} {'Source Path':<{col2}} {'Copied':<{col3}}")
    print("-" * (col1 + col2 + col3 + 4))

    failed = []
    for lib_name, resolved_path in sorted(cuda_libs.items()):
        src = Path(resolved_path)
        copied = copy_with_symlink_resolution(src, dest_dir)
        status = "yes" if copied else "NO"
        if not copied:
            failed.append(lib_name)
        print(f"{lib_name:<{col1}} {resolved_path:<{col2}} {status:<{col3}}")

    print()
    print(f"Copied {len(cuda_libs) - len(failed)}/{len(cuda_libs)} CUDA libraries.")

    if failed:
        print(f"\nERROR: Failed to copy {len(failed)} required library/libraries:")
        for lib in failed:
            print(f"  - {lib}")
        sys.exit(1)
    else:
        print("All CUDA libraries copied successfully.")


if __name__ == "__main__":
    main()
