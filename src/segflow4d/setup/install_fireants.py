"""segflow4d-install-fireants
--------------------------
CLI entry point that clones FireANTs, builds the fused CUDA ops, and installs
both into the currently active Python environment.

Usage (after ``pip install segflow4d``):

    segflow4d-install-fireants [--dir <clone-dir>]
"""

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

FIREANTS_REPO = "https://github.com/rohitrango/FireANTs.git"


def _run(cmd: list[str], *, env: dict | None = None, cwd: str | None = None) -> None:
    """Run a command, streaming output to the terminal; raise SystemExit on failure."""
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _check_prerequisites() -> None:
    print("==> Checking prerequisites...")

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        print(
            "ERROR: No conda environment is active ($CONDA_PREFIX is unset).\n"
            "       Please run: conda activate <env-name>",
            file=sys.stderr,
        )
        raise SystemExit(1)

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
    if conda_env == "base":
        print(
            "WARNING: You are running in the 'base' conda environment.\n"
            "         This is likely unintended. Consider activating a dedicated env:\n"
            "           conda activate <your-env>",
            file=sys.stderr,
        )
        answer = input("         Continue anyway? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            raise SystemExit(1)

    print(f"    Conda env    : {conda_env} ({conda_prefix})")
    print(f"    Python       : {sys.version.split()[0]}")


def _clone_or_update(clone_dir: str) -> None:
    if os.path.isdir(os.path.join(clone_dir, ".git")):
        print(f"\n==> FireANTs already cloned at {clone_dir} — pulling latest changes...")
        _run(["git", "-C", clone_dir, "pull"])
    else:
        print(f"\n==> Cloning FireANTs into {clone_dir}...")
        _run(["git", "clone", FIREANTS_REPO, clone_dir])


def _install_fireants_package(clone_dir: str) -> None:
    print("\n==> Installing fireants (pip install .)...")
    _run([sys.executable, "-m", "pip", "install", clone_dir])


def _get_torch_cuda_version() -> str:
    """Return the CUDA version string PyTorch was compiled against (e.g. '12.1'), or ''."""
    try:
        import torch
        return torch.version.cuda or ""
    except ImportError:
        return ""


def _get_nvcc_version(nvcc_path: str = "nvcc") -> str:
    """Return nvcc release version string (e.g. '12.1'), or '' if not found."""
    try:
        out = subprocess.check_output(
            [nvcc_path, "--version"], stderr=subprocess.STDOUT, text=True
        )
        m = re.search(r"release (\d+\.\d+)", out)
        return m.group(1) if m else ""
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def _conda_install(*packages: str) -> None:
    if not shutil.which("conda"):
        print(
            "ERROR: conda not found. Please install the required packages manually.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    _run(["conda", "install", "-y", "-c", "nvidia", *packages])


def _find_cuda_runtime_h(*search_roots: str) -> str:
    """Return the directory containing cuda_runtime.h, or '' if not found."""
    py_site = ""
    try:
        import site
        sites = site.getsitepackages()
        py_site = sites[0] if sites else ""
    except Exception:
        pass

    for root in search_roots:
        if not root:
            continue
        candidates = [
            os.path.join(root, "include"),
            os.path.join(root, "targets", "x86_64-linux", "include"),
        ]
        if py_site:
            candidates.append(os.path.join(py_site, "nvidia", "cuda_runtime", "include"))
        for d in candidates:
            if os.path.isfile(os.path.join(d, "cuda_runtime.h")):
                return d
    return ""


def _ensure_cuda_toolkit(torch_cuda_ver: str) -> dict[str, str]:
    """Ensure nvcc matches the PyTorch CUDA version; install from conda if not.

    Returns a dict of environment variable additions needed for the build step.
    """
    print("\n==> Checking CUDA version compatibility...")

    if not torch_cuda_ver:
        print(
            "WARNING: Could not determine PyTorch CUDA version. Is torch installed?",
            file=sys.stderr,
        )
        return {}

    print(f"    PyTorch CUDA   : {torch_cuda_ver}")

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    nvcc_ver = _get_nvcc_version()
    if nvcc_ver:
        print(f"    nvcc (system)  : {nvcc_ver}")

    if torch_cuda_ver != nvcc_ver:
        print(f"    CUDA mismatch — installing cuda-nvcc={torch_cuda_ver} into conda env...")
        _conda_install(
            f"cuda-nvcc={torch_cuda_ver}.*",
            f"cuda-cudart-dev={torch_cuda_ver}.*",
            f"cuda-libraries-dev={torch_cuda_ver}.*",
        )

    # Locate nvcc inside the conda env
    conda_nvcc = ""
    candidate = os.path.join(conda_prefix, "bin", "nvcc")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        conda_nvcc = candidate
    else:
        for root, dirs, files in os.walk(conda_prefix):
            depth = root.replace(conda_prefix, "").count(os.sep)
            if depth > 4:
                dirs.clear()
                continue
            if "nvcc" in files:
                c = os.path.join(root, "nvcc")
                if os.access(c, os.X_OK):
                    conda_nvcc = c
                    break

    if not conda_nvcc:
        print("ERROR: nvcc not found in conda env after install.", file=sys.stderr)
        raise SystemExit(1)

    conda_cuda_bin = os.path.dirname(conda_nvcc)
    cuda_home = os.path.dirname(conda_cuda_bin)

    active_nvcc_ver = _get_nvcc_version(conda_nvcc)
    print(f"    nvcc (active)  : {active_nvcc_ver} → {conda_nvcc}")

    if active_nvcc_ver != torch_cuda_ver:
        print(
            f"ERROR: Could not obtain nvcc {torch_cuda_ver} after conda install. "
            f"Found {active_nvcc_ver}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Ensure cuda_runtime.h is present
    cuda_runtime_h_dir = _find_cuda_runtime_h(cuda_home, conda_prefix)
    if not cuda_runtime_h_dir:
        print(f"    cuda_runtime.h not found — installing cuda-cudart-dev={torch_cuda_ver}...")
        _conda_install(f"cuda-cudart-dev={torch_cuda_ver}.*")
        cuda_runtime_h_dir = _find_cuda_runtime_h(cuda_home, conda_prefix)

    env_additions: dict[str, str] = {
        "PATH": f"{conda_cuda_bin}:{os.environ.get('PATH', '')}",
        "CUDA_HOME": cuda_home,
    }
    if cuda_runtime_h_dir:
        print(f"    cuda_runtime.h : {cuda_runtime_h_dir}/cuda_runtime.h")
        existing = os.environ.get("CPATH", "")
        env_additions["CPATH"] = f"{cuda_runtime_h_dir}:{existing}" if existing else cuda_runtime_h_dir

    return env_additions


def _check_gpu_compute_capability() -> None:
    print("\n==> Checking GPU compute capability...")

    try:
        import torch
    except ImportError:
        print("WARNING: torch not importable, skipping arch check", file=sys.stderr)
        return

    if not torch.cuda.is_available():
        print("    No CUDA GPU detected — skipping arch check")
        return

    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"    GPU            : {gpu_name}")
    print(f"    Compute cap.   : sm_{cc_major}{cc_minor}")

    try:
        supported = torch.cuda.get_arch_list()
    except AttributeError:
        supported = []
    print(f"    Supported archs: {supported}")

    def _parse_arch(name: str) -> tuple[int, int] | None:
        m = re.match(r"(?:sm|compute)_(\d+)", name)
        if not m:
            return None
        digits = m.group(1)
        if len(digits) <= 2:
            return int(digits[0]), int(digits[1:])
        return int(digits[:-1]), int(digits[-1])

    is_supported = False
    if supported:
        for arch_name in supported:
            parsed = _parse_arch(arch_name)
            if parsed is None:
                continue
            arch_major, arch_minor = parsed
            if arch_major == cc_major and arch_minor <= cc_minor:
                is_supported = True
                break

    if supported and not is_supported:
        print(
            f"\nERROR: Your GPU ({gpu_name}, compute capability {cc_major}.{cc_minor}) "
            f"is NOT supported by the installed PyTorch "
            f"(torch.version.cuda = {torch.version.cuda}).\n"
            f"       The highest supported arch in this build is: {supported[-1]}\n\n"
            f"       To fix: reinstall PyTorch built against a CUDA version that supports "
            f"sm_{cc_major}{cc_minor}. For Blackwell (sm_100) GPUs, CUDA >= 12.8 is required.\n"
            f"       Available PyTorch wheel tags: cu118, cu121, cu124, cu126, cu128",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("    GPU arch supported by installed PyTorch — OK")


def _build_fused_ops(clone_dir: str, torch_cuda_ver: str, env_additions: dict[str, str]) -> None:
    fused_ops_dir = os.path.join(clone_dir, "fused_ops")
    if not os.path.isdir(fused_ops_dir):
        print(
            f"ERROR: fused_ops directory not found at {fused_ops_dir}.\n"
            "       The FireANTs repository structure may have changed.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"\n==> Building fused CUDA operations in {fused_ops_dir}...")

    build_dir = os.path.join(fused_ops_dir, "build")
    if os.path.isdir(build_dir):
        print("    Cleaning previous build directory...")
        shutil.rmtree(build_dir)

    env = os.environ.copy()
    for key in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CUDAHOSTCXX"):
        env.pop(key, None)
    env.update(env_additions)

    # Verify cuda_runtime.h is reachable
    cuda_home = env_additions.get("CUDA_HOME", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    cuda_rt_dir = _find_cuda_runtime_h(cuda_home, conda_prefix)
    if not cuda_rt_dir:
        print(
            "ERROR: cuda_runtime.h not found. Install cuda-cudart-dev:\n"
            "       conda install -c nvidia cuda-cudart-dev",
            file=sys.stderr,
        )
        raise SystemExit(1)
    existing_cpath = env.get("CPATH", "")
    env["CPATH"] = f"{cuda_rt_dir}:{existing_cpath}" if existing_cpath else cuda_rt_dir

    # Check host compiler compatibility with the CUDA version
    cuda_major = int(torch_cuda_ver.split(".")[0]) if torch_cuda_ver else 0
    max_gcc = {12: 12, 11: 11}.get(cuda_major, 99)
    try:
        gcc_ver = int(subprocess.check_output(["gcc", "-dumpversion"], text=True).strip().split(".")[0])
    except (FileNotFoundError, ValueError):
        gcc_ver = 0

    if gcc_ver and gcc_ver > max_gcc:
        print(f"    System GCC {gcc_ver} is too new for CUDA {torch_cuda_ver} (max GCC {max_gcc}).")
        compat_gcc = None
        for v in range(max_gcc, 8, -1):
            if shutil.which(f"gcc-{v}") and shutil.which(f"g++-{v}"):
                compat_gcc = v
                break
        if compat_gcc:
            print(f"    Using gcc-{compat_gcc} / g++-{compat_gcc} as the host compiler.")
            env["CC"] = f"gcc-{compat_gcc}"
            env["CXX"] = f"g++-{compat_gcc}"
            env["CUDAHOSTCXX"] = f"g++-{compat_gcc}"
        else:
            print(
                f"ERROR: No GCC <= {max_gcc} found. Install one with:\n"
                f"       sudo apt install gcc-{max_gcc} g++-{max_gcc}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Set TORCH_CUDA_ARCH_LIST from the actual GPU(s)
    try:
        import torch
        if torch.cuda.is_available():
            caps = sorted({
                f"{maj}.{min_}"
                for maj, min_ in [
                    torch.cuda.get_device_capability(i)
                    for i in range(torch.cuda.device_count())
                ]
            })
            if caps:
                env["TORCH_CUDA_ARCH_LIST"] = ";".join(caps)
                print(f"    TORCH_CUDA_ARCH_LIST={env['TORCH_CUDA_ARCH_LIST']}")
    except Exception:
        pass

    # Match _GLIBCXX_USE_CXX11_ABI to PyTorch's build setting
    try:
        import torch
        abi = int(torch.compiled_with_cxx11_abi())
        existing_cxxflags = env.get("CXXFLAGS", "")
        env["CXXFLAGS"] = f"{existing_cxxflags} -D_GLIBCXX_USE_CXX11_ABI={abi}".strip()
        print(f"    _GLIBCXX_USE_CXX11_ABI={abi}")
    except Exception:
        pass

    _run([sys.executable, "setup.py", "build_ext"], cwd=fused_ops_dir, env=env)
    _run([sys.executable, "setup.py", "install"], cwd=fused_ops_dir, env=env)


def _verify_installation() -> None:
    print("\n==> Verifying installation...")

    # Import torch first so its __init__.py adds CUDA/C++ runtime libs to the
    # dynamic linker search path before we try to load the fused ops extension.
    import torch  # noqa: F401

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
        raise SystemExit(1)

    print("\nFireANTs with fused ops installed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="segflow4d-install-fireants",
        description=(
            "Clone FireANTs, build the fused CUDA operations, and install both "
            "into the current Python environment.\n\n"
            "Prerequisites:\n"
            "  - A conda environment must be active (conda activate <env>)\n"
            "  - PyTorch (GPU build) must already be installed\n"
            "  - git and nvcc (or conda) must be available"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        metavar="PATH",
        default=None,
        help=(
            "Directory to clone FireANTs into. "
            "Defaults to a temporary directory that is removed after installation."
        ),
    )
    args = parser.parse_args()

    use_temp = args.dir is None
    clone_dir = args.dir or tempfile.mkdtemp(prefix="fireants_")

    try:
        _check_prerequisites()
        _clone_or_update(clone_dir)
        _install_fireants_package(clone_dir)
        torch_cuda_ver = _get_torch_cuda_version()
        env_additions = _ensure_cuda_toolkit(torch_cuda_ver)
        _check_gpu_compute_capability()
        _build_fused_ops(clone_dir, torch_cuda_ver, env_additions)
        _verify_installation()
    finally:
        if use_temp and os.path.isdir(clone_dir):
            print(f"\n==> Removing FireANTs source directory ({clone_dir})...")
            shutil.rmtree(clone_dir)
            print("    Done.")


if __name__ == "__main__":
    main()
