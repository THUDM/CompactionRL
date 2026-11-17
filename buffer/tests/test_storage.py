import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from compactionrl.runtime import storage


def test_system_disk_is_rejected_before_directories_are_created(tmp_path, monkeypatch):
    stat = os.stat
    monkeypatch.setattr(storage.sys, "platform", "linux")
    monkeypatch.setattr(
        storage.os,
        "stat",
        lambda *a, **k: SimpleNamespace(st_dev=1, st_mode=stat(*a, **k).st_mode),
    )
    with pytest.raises(ValueError, match="system-filesystem"):
        storage.require_data_filesystem("/")


def test_missing_destination_checks_nearest_existing_mount(tmp_path, monkeypatch):
    stat = os.stat
    monkeypatch.setattr(storage.sys, "platform", "linux")

    def devices(path, *args, **kwargs):
        info = stat(path, *args, **kwargs)
        return SimpleNamespace(
            st_dev=1 if Path(path) == Path("/") else 2,
            st_mode=info.st_mode,
        )

    monkeypatch.setattr(storage.os, "stat", devices)
    target = tmp_path / "new/runtime"
    assert storage.require_data_filesystem(target) == target
    assert not target.exists()


def test_all_cache_paths_are_inside_selected_volume(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", "/tmp")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/root/.cache/torch_extensions")
    env = storage.storage_environment(tmp_path / "runtime")
    flags = {
        "PYTHONDONTWRITEBYTECODE",
        "TMS_INIT_ENABLE_CPU_BACKUP",
        "TMS_INIT_ENABLE_DISK_BACKUP",
    }
    for key, value in env.items():
        if key not in flags:
            assert Path(value).is_relative_to(tmp_path / "runtime")
    assert env["TMS_INIT_ENABLE_DISK_BACKUP"] == "0"


def test_preparation_checks_every_path_before_writing(tmp_path, monkeypatch):
    checked = []

    def reject_cache(path):
        checked.append(Path(path))
        if Path(path).name == "cuda":
            raise ValueError("forbidden mount")
        return Path(path)

    monkeypatch.setattr(storage, "require_data_filesystem", reject_cache)
    with pytest.raises(ValueError, match="forbidden mount"):
        storage.prepare_storage(tmp_path / "runtime", tmp_path / "output")
    assert checked
    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path / "output").exists()


def test_ray_socket_length_fails_before_gpu_launch(monkeypatch):
    monkeypatch.setattr(storage.sys, "platform", "linux")
    storage.validate_ray_socket_path("/data/cr")
    with pytest.raises(ValueError, match="shorter --work-dir"):
        storage.validate_ray_socket_path("/data/" + "very-long-directory/" * 10)


def test_build_git_profile_survives_stripped_git_environment(tmp_path, monkeypatch):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is unavailable")
    monkeypatch.setattr(storage, "require_data_filesystem", lambda p: Path(p).resolve())
    existing_config = tmp_path / "existing-config"
    original = "[user]\n\tname = Existing User\n"
    existing_config.write_text(original)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(existing_config))
    root = tmp_path / "project"
    work = tmp_path / "runtime"
    storage.prepare_git_profile(root, work)
    storage.prepare_git_profile(root, work)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(storage.storage_environment(work))
    result = subprocess.run(
        [git, "-C", str(tmp_path.anchor), "config", "--get-all", "safe.directory"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    values = result.stdout.splitlines()
    assert values.count(str(root)) == 1
    assert values.count(str(root / ".deps/Megatron-LM")) == 1
    assert existing_config.read_text() == original


def test_build_git_profile_rejects_symlink_without_changing_target(
    tmp_path, monkeypatch
):
    if shutil.which("git") is None:
        pytest.skip("Git is unavailable")
    monkeypatch.setattr(storage, "require_data_filesystem", lambda p: Path(p).resolve())
    work = tmp_path / "runtime"
    directory = work / "config/git"
    directory.mkdir(parents=True)
    target = tmp_path / "existing-config"
    target.write_text("preserve me\n")
    (directory / "config").symlink_to(target)
    with pytest.raises(ValueError, match="symbolic Git profile"):
        storage.prepare_git_profile(tmp_path / "project", work)
    assert target.read_text() == "preserve me\n"
