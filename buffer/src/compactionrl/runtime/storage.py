"""Keep runtime files on an explicitly selected data filesystem."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def storage_environment(work_dir):
    root = Path(work_dir).resolve()
    cache = root / "cache"
    return {
        "COMPACTIONRL_WORK_DIR": str(root),
        "TMPDIR": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TEMP": str(root / "tmp"),
        "XDG_CACHE_HOME": str(cache),
        "XDG_CONFIG_HOME": str(root / "config"),
        "XDG_DATA_HOME": str(root / "data"),
        "XDG_STATE_HOME": str(root / "state"),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface/hub"),
        "TORCH_HOME": str(cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "UV_CACHE_DIR": str(cache / "uv"),
        "RAY_TMPDIR": str(root / "ray"),
        "COMPACTIONRL_RAY_SPILL_DIR": str(root / "ray_spill"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
        "TMS_INIT_ENABLE_DISK_BACKUP": "0",
    }


def require_data_filesystem(path):
    """Reject Linux root-filesystem destinations before creating anything."""
    target = Path(path).resolve()
    existing = target
    while not existing.exists():
        existing = existing.parent
    if not existing.is_dir():
        raise ValueError(f"Storage parent is not a directory: {existing}")
    if sys.platform == "linux" and os.stat(existing).st_dev == os.stat("/").st_dev:
        raise ValueError(
            f"Refusing system-filesystem writes at {target}. Select a directory "
            "on a mounted data volume for --output and --work-dir."
        )
    return target


def prepare_storage(work_dir, output):
    env = storage_environment(work_dir)
    # Validate every destination first, including pre-existing symlinks.
    flag_keys = {
        "PYTHONDONTWRITEBYTECODE",
        "TMS_INIT_ENABLE_CPU_BACKUP",
        "TMS_INIT_ENABLE_DISK_BACKUP",
    }
    destinations = {Path(output)}
    destinations.update(Path(v) for k, v in env.items() if k not in flag_keys)
    paths = [require_data_filesystem(path) for path in sorted(destinations)]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return env


def validate_ray_socket_path(work_dir):
    # Ray puts Unix sockets beneath a timestamp/PID session directory.
    ray_dir = Path(storage_environment(work_dir)["RAY_TMPDIR"])
    candidate = (
        ray_dir
        / "session_2026-11-16_23-59-59_999999_99999999/sockets/plasma_store999999"
    )
    if sys.platform == "linux" and len(os.fsencode(candidate)) > 107:
        raise ValueError(
            "Ray's Unix socket path would be too long. Select a shorter --work-dir "
            "on the data volume (its ray/ path should be at most 36 bytes)."
        )


def prepare_git_profile(project_root, work_dir):
    """Use a data-volume Git profile when build plugins strip GIT_* variables."""
    git = shutil.which("git")
    if git is None:
        return
    root = Path(project_root).resolve()
    directory = Path(storage_environment(work_dir)["XDG_CONFIG_HOME"]) / "git"
    require_data_filesystem(directory)
    profile = directory / "config"
    if profile.is_symlink():
        raise ValueError(f"Refusing a symbolic Git profile: {profile}")
    directory.mkdir(parents=True, exist_ok=True)
    command = [git, "-c", f"safe.directory={root}", "config", "--file", str(profile)]
    existing = subprocess.run(
        [*command, "--get-all", "safe.directory"], capture_output=True, text=True
    )
    if existing.returncode not in (0, 1):
        raise ValueError(f"Cannot read project Git profile: {profile}")
    values = set(existing.stdout.splitlines())
    for target in (root, root / ".deps/Megatron-LM"):
        if str(target) not in values:
            subprocess.run(
                [*command, "--add", "safe.directory", str(target)], check=True
            )
