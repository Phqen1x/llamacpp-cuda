"""
Smoke tests for llamacpp-cuda build artifacts.

Most tests are runnable without a GPU. Tests marked @pytest.mark.gpu require
an NVIDIA GPU and CUDA runtime to be available.

Run without GPU:
    pytest tests/test_build_smoke.py

Run all (including GPU tests):
    pytest tests/test_build_smoke.py -m gpu
"""

import os
import subprocess
from pathlib import Path

import pytest

# Artifact directory — override via LLAMACPP_ARTIFACT_DIR env var
ARTIFACT_DIR = Path(os.environ.get("LLAMACPP_ARTIFACT_DIR", "./build/bin"))
LLAMA_SERVER = ARTIFACT_DIR / "llama-server"


@pytest.fixture(scope="session")
def artifact_dir() -> Path:
    return ARTIFACT_DIR


@pytest.fixture(scope="session")
def llama_server(artifact_dir: Path) -> Path:
    return artifact_dir / "llama-server"


def test_binary_exists(artifact_dir: Path) -> None:
    """Artifact directory contains llama-server binary."""
    assert (artifact_dir / "llama-server").exists(), (
        f"llama-server not found in {artifact_dir}. "
        "Set LLAMACPP_ARTIFACT_DIR to the extracted artifact directory."
    )


def test_required_libs_present(artifact_dir: Path) -> None:
    """Required CUDA runtime libraries are present in the artifact directory."""
    required = ["libcublas", "libcublasLt"]
    missing = []

    for lib in required:
        found = any(artifact_dir.glob(f"{lib}.so*"))
        if not found:
            missing.append(lib)

    assert not missing, f"Missing required CUDA libraries: {missing}"


def test_rpath_set(llama_server: Path) -> None:
    """llama-server RPATH contains $ORIGIN for portable distribution."""
    if not llama_server.exists():
        pytest.skip(f"llama-server not found at {llama_server}")

    try:
        result = subprocess.run(
            ["patchelf", "--print-rpath", str(llama_server)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("patchelf not installed")

    rpath = result.stdout.strip()
    assert "$ORIGIN" in rpath, (
        f"RPATH does not contain $ORIGIN: '{rpath}'. "
        "Run patchelf --set-rpath '$ORIGIN' on the binary."
    )


@pytest.mark.gpu
def test_version_flag(llama_server: Path) -> None:
    """llama-server --version exits 0."""
    if not llama_server.exists():
        pytest.skip(f"llama-server not found at {llama_server}")

    result = subprocess.run(
        [str(llama_server), "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"--version exited {result.returncode}. stderr: {result.stderr}"
    )


@pytest.mark.gpu
def test_list_devices(llama_server: Path) -> None:
    """llama-server --list-devices output mentions CUDA or NVIDIA."""
    if not llama_server.exists():
        pytest.skip(f"llama-server not found at {llama_server}")

    result = subprocess.run(
        [str(llama_server), "--list-devices"],
        capture_output=True,
        text=True,
    )
    combined = (result.stdout + result.stderr).lower()
    assert "cuda" in combined or "nvidia" in combined, (
        f"--list-devices output does not mention CUDA or NVIDIA. "
        f"Output: {result.stdout + result.stderr}"
    )
