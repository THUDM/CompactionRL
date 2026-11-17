"""Duck-typed minimal environment used when the terminus-2 loop runs
inside the task container itself.

When terminus-2 is shipped as an installed agent (see
`src/harbor/agents/installed/terminus_2.py`), the agent process executes
inside the container and reaches tmux + the local filesystem directly —
no docker daemon hop. `LocalEnvironment` provides just the slice of the
`BaseEnvironment` surface that `TmuxSession` and `Terminus2` use, with
`exec`/`upload_file`/`download_file` translated to local subprocess
and `shutil` calls.

This is intentionally not a `BaseEnvironment` subclass: the host-side
constructor expects a real environment_dir, task config, etc., none of
which exist for the in-container case. Duck typing keeps the surface
minimal and avoids dragging Docker/Daytona-specific concepts into the
container.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from harbor.environments.base import ExecResult


class LocalEnvironment:
    """Subprocess-backed `environment` for in-container terminus-2 runs."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.is_mounted = True

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        run_env: dict[str, str] | None = None
        if env is not None:
            run_env = dict(os.environ)
            run_env.update(env)

        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            env=run_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.communicate(), timeout=2)
            except asyncio.TimeoutError:
                pass
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")

        return ExecResult(
            stdout=stdout_bytes.decode(errors="replace") if stdout_bytes else None,
            stderr=stderr_bytes.decode(errors="replace") if stderr_bytes else None,
            return_code=process.returncode or 0,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        target = Path(target_path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    async def is_dir(self, path: str) -> bool:
        return Path(path).is_dir()

    async def is_file(self, path: str) -> bool:
        return Path(path).is_file()
