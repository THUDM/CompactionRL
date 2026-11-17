import json
import os
from pathlib import Path
import subprocess
import sys

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]


def test_dataset_download_uses_json_registry_without_cloud_client(
    tmp_path, monkeypatch
):
    from harbor.cli.main import app
    from harbor.tasks.client import TaskClient

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "name": "fixture",
                    "version": "1.0",
                    "description": "local fixture",
                    "tasks": [{"name": "one", "path": "tasks/one"}],
                }
            ]
        )
    )
    calls = []

    def download(self, task_ids, overwrite=False, output_dir=None):
        calls.append((task_ids, overwrite, output_dir))
        return [tmp_path / "one"]

    monkeypatch.setitem(sys.modules, "supabase", None)
    monkeypatch.setattr(TaskClient, "download_tasks", download)
    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "download",
            "fixture@1.0",
            "--registry-path",
            str(registry),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Successfully downloaded 1 task(s)" in result.output
    assert len(calls) == 1
    assert str(calls[0][0][0].path) == "tasks/one"
    assert calls[0][2] == tmp_path / "output"


def test_harbor_host_files_follow_selected_cache_and_temp_directories(tmp_path):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    env = dict(
        os.environ,
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        TMPDIR=str(temporary),
        PYTHONPATH=str(ROOT / "terminus/src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import json; from harbor.constants import CACHE_DIR; "
            "from harbor.agents.terminus_2.tmux_session import TmuxSession; "
            "print(json.dumps([str(CACHE_DIR), str(TmuxSession._HOST_TMUX_BOOTSTRAP_DIR)]))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    cache, bootstrap = json.loads(probe.stdout.splitlines()[-1])
    assert Path(cache).is_relative_to(tmp_path / "cache")
    assert Path(bootstrap).is_relative_to(temporary)
