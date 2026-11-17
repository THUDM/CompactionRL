import asyncio
import asyncio.subprocess
import os
import shlex
from pathlib import Path

from pydantic import BaseModel

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.docker import (
    COMPOSE_BASE_PATH,
    COMPOSE_BUILD_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.utils.pager_policy import pager_env_dict


class DockerEnvironmentEnvVars(BaseModel):
    main_image_name: str
    context_dir: str
    host_verifier_logs_path: str
    host_agent_logs_path: str
    host_artifacts_path: str
    env_verifier_logs_path: str
    env_agent_logs_path: str
    t_bench_task_docker_client_container_name: str | None = None
    env_artifacts_path: str
    prebuilt_image_name: str | None = None
    cpus: int = 1
    memory: str = "1G"

    def to_env_dict(self, include_os_env: bool = True) -> dict[str, str]:
        env_dict = {} if not include_os_env else os.environ.copy()

        for field_name, value in self.model_dump(exclude_none=True).items():
            if value is None:
                continue

            env_dict[f"{field_name.upper()}"] = str(value)

        return env_dict


class DockerEnvironment(BaseEnvironment):
    _DOCKER_COMPOSE_BASE_PATH = COMPOSE_BASE_PATH
    _DOCKER_COMPOSE_BUILD_PATH = COMPOSE_BUILD_PATH
    _DOCKER_COMPOSE_PREBUILT_PATH = COMPOSE_PREBUILT_PATH
    _DOCKER_COMPOSE_NO_NETWORK_PATH = COMPOSE_NO_NETWORK_PATH
    _MAIN_SERVICE_NAME = "main"
    _TRANSIENT_MAIN_UNAVAILABLE_PATTERNS = (
        'service "main" is not running',
        "container is restarting",
    )
    _NONRECOVERABLE_MAIN_UNAVAILABLE_PATTERNS = (
        'no container found for service "main"',
    )
    _MAIN_SERVICE_COMMAND_RETRY_ATTEMPTS = 5
    _MAIN_SERVICE_COMMAND_RETRY_DELAY_SEC = 5.0
    _MAIN_SERVICE_RECOVERY_TIMEOUT_SEC = 60
    _MAIN_SERVICE_DIAGNOSTIC_LOG_TAIL_LINES = 200

    # Class-level lock per image name to prevent parallel builds of the same image.
    _image_build_locks: dict[str, asyncio.Lock] = {}

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        keep_containers: bool = False,
        stop_timeout_sec: int | None = 300,
        *args,
        **kwargs,
    ):
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._keep_containers = keep_containers
        self._stop_timeout_sec = stop_timeout_sec

        # TB-hard docker-compose files expect a container_name variable; set it here.
        compose_project_name = self.session_id.lower().replace(".", "-")
        self._env_vars = DockerEnvironmentEnvVars(
            main_image_name=f"hb__{environment_name}",
            context_dir=str(self.environment_dir.resolve().absolute()),
            host_verifier_logs_path=str(trial_paths.verifier_dir.resolve().absolute()),
            host_agent_logs_path=str(trial_paths.agent_dir.resolve().absolute()),
            host_artifacts_path=str(trial_paths.artifacts_dir.resolve().absolute()),
            env_verifier_logs_path=str(EnvironmentPaths.verifier_dir),
            env_agent_logs_path=str(EnvironmentPaths.agent_dir),
            t_bench_task_docker_client_container_name=compose_project_name,
            env_artifacts_path=str(EnvironmentPaths.artifacts_dir),
            prebuilt_image_name=task_env_config.docker_image,
            cpus=task_env_config.cpus,
            memory=f"{task_env_config.memory_mb}M",
        )
        self._use_prebuilt = False
        self._fail_fast_on_main_service_unavailable = False

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return True

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _environment_docker_compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    @property
    def _docker_compose_paths(self) -> list[Path]:
        """
        Returns the docker-compose file(s) to use.

        Two options for task authors:

        Option 1: Simple task (just Dockerfile)
        - No docker-compose needed
        - Uses: base + build/prebuilt

        Option 2: Task with extra services (docker-compose.yaml)
        - Create docker-compose.yaml with additional services or overrides
        - Uses: base + build/prebuilt + docker-compose.yaml
        - Task file is last so it can override scalars from build/prebuilt
        - Relative paths (e.g. build context) resolve relative to the file
          where they are defined, regardless of -f order

        When allow_internet is False, the no-network compose file is appended
        last to set network_mode: none on the main service.
        """
        build_or_prebuilt = (
            self._DOCKER_COMPOSE_PREBUILT_PATH
            if self._use_prebuilt
            else self._DOCKER_COMPOSE_BUILD_PATH
        )

        if self._environment_docker_compose_path.exists():
            paths = [
                self._DOCKER_COMPOSE_BASE_PATH,
                build_or_prebuilt,
                self._environment_docker_compose_path,
            ]
        else:
            paths = [self._DOCKER_COMPOSE_BASE_PATH, build_or_prebuilt]

        if not self.task_env_config.allow_internet:
            paths.append(self._DOCKER_COMPOSE_NO_NETWORK_PATH)

        return paths

    def _validate_definition(self):
        if (
            not self._dockerfile_path.exists()
            and not self._environment_docker_compose_path.exists()
        ):
            raise FileNotFoundError(
                f"{self._dockerfile_path} and {self._environment_docker_compose_path} "
                "not found. Please ensure at least one of these files exist."
            )

    async def _run_docker_compose_command(
        self, command: list[str], check: bool = True, timeout_sec: int | None = None
    ) -> ExecResult:
        """Run a docker compose command and return the result."""
        full_command = [
            "docker-compose",
            "-p",
            self.session_id.lower().replace(".", "-"),
            "--project-directory",
            str(self.environment_dir.resolve().absolute()),
        ]
        for path in self._docker_compose_paths:
            full_command.extend(["-f", str(path.resolve().absolute())])
        full_command.extend(command)

        process = await asyncio.create_subprocess_exec(
            *full_command,
            env=self._env_vars.to_env_dict(include_os_env=True),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            process.terminate()
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except asyncio.TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None

        result = ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
        )

        if check and result.return_code != 0:
            raise RuntimeError(
                f"Docker compose command failed for environment {self.environment_name}. "
                f"Command: {' '.join(full_command)}. "
                f"Return code: {result.return_code}. "
                f"Stdout: {result.stdout}. "
                f"Stderr: {result.stderr}. "
            )

        return result

    async def _run_host_command(
        self,
        command: list[str],
        timeout_sec: int | None = None,
        check: bool = False,
    ) -> ExecResult:
        """Run a host-side command for diagnostics/recovery helpers."""
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            process.terminate()
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except asyncio.TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
            raise RuntimeError(
                f"Command timed out after {timeout_sec} seconds: {' '.join(command)}"
            )

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None
        result = ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"Host command failed: {' '.join(command)}. "
                f"Return code: {result.return_code}. "
                f"Stdout: {result.stdout}. "
                f"Stderr: {result.stderr}. "
            )
        return result

    async def start(self, force_build: bool):
        self._use_prebuilt = not force_build and self.task_env_config.docker_image

        if not self._use_prebuilt:
            # Serialize image builds: if multiple environments with the same image name
            # start concurrently, only one builds while others wait for the cached image.
            image_name = self._env_vars.main_image_name
            lock = self._image_build_locks.setdefault(image_name, asyncio.Lock())
            async with lock:
                await self._run_docker_compose_command(["build"])

        # Remove any stale containers from previous runs with the same session ID.
        try:
            await self._run_docker_compose_command(["down", "--remove-orphans"])
        except RuntimeError:
            pass

        await self._run_docker_compose_command(["up", "--detach", "--wait"])

    async def stop(self, delete: bool):
        timeout_sec = self._stop_timeout_sec
        if timeout_sec is not None and timeout_sec <= 0:
            timeout_sec = None

        # Best-effort: fix ownership of bind-mounted directories so the host
        # user can read/write/delete them after the container is gone.
        await self._chown_to_host_user(str(EnvironmentPaths.logs_dir), recursive=True)
        if self._keep_containers and delete:
            self.logger.warning(
                "Both `keep_containers` and `--delete` option are set. "
                "keep_containers takes precedence."
            )
        if self._keep_containers:
            try:
                await self._run_docker_compose_command(
                    ["stop"], timeout_sec=timeout_sec
                )
            except RuntimeError as e:
                self.logger.warning(f"Docker compose stop failed: {e}")
                if self._is_timeout_error(e):
                    await self._hard_cleanup_containers()
        elif delete:  # delete=True: aggressive cleanup
            try:
                await self._run_docker_compose_command(
                    ["down", "--volumes", "--remove-orphans"],
                    timeout_sec=timeout_sec,
                )
            except RuntimeError as e:
                self.logger.warning(f"Docker compose down failed: {e}")
                if self._is_timeout_error(e):
                    await self._hard_cleanup_containers()

            # await self._cleanup_build_cache()
        else:
            try:
                await self._run_docker_compose_command(
                    ["down"], timeout_sec=timeout_sec
                )
            except RuntimeError as e:
                self.logger.warning(f"Docker compose down failed: {e}")
                if self._is_timeout_error(e):
                    await self._hard_cleanup_containers()

    async def upload_file(self, source_path: Path | str, target_path: str):
        await self._run_main_service_command_with_retry(
            [
                "cp",
                str(source_path),
                f"main:{target_path}",
            ],
            check=True,
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        await self._run_main_service_command_with_retry(
            [
                "cp",
                f"{source_dir}/.",
                f"main:{target_dir}",
            ],
            check=True,
        )

    async def _chown_to_host_user(self, path: str, recursive: bool = False) -> None:
        """Best-effort chown of a container path to the host user's UID:GID.

        No-op on Windows (where os.getuid/os.getgid are unavailable).
        """
        if not hasattr(os, "getuid"):
            return
        flag = "-R " if recursive else ""
        await self.exec(f"chown {flag}{os.getuid()}:{os.getgid()} {shlex.quote(path)}")

    async def download_file(self, source_path: str, target_path: Path | str):
        await self._chown_to_host_user(source_path)
        await self._run_main_service_command_with_retry(
            [
                "cp",
                f"main:{source_path}",
                str(target_path),
            ],
            check=True,
        )

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        await self._chown_to_host_user(source_dir, recursive=True)
        await self._run_main_service_command_with_retry(
            [
                "cp",
                f"main:{source_dir}/.",
                str(target_dir),
            ],
            check=True,
        )

    @property
    def _compose_project_name(self) -> str:
        return self.session_id.lower().replace(".", "-")

    def _is_timeout_error(self, error: Exception) -> bool:
        return "timed out" in str(error).lower()

    def _command_targets_main_service(self, command: list[str]) -> bool:
        if not command:
            return False

        subcommand = command[0]
        if subcommand == "exec":
            return self._MAIN_SERVICE_NAME in command[1:]
        if subcommand == "cp":
            return any(
                part == self._MAIN_SERVICE_NAME
                or part.startswith(f"{self._MAIN_SERVICE_NAME}:")
                for part in command[1:]
            )
        return False

    def _extract_retryable_error_text(
        self,
        result: ExecResult | None = None,
        error: RuntimeError | None = None,
    ) -> str:
        if error is not None:
            return str(error).lower()

        if result is None:
            return ""

        return "\n".join(filter(None, [result.stdout, result.stderr])).lower()

    def _is_transient_main_service_unavailable(
        self,
        command: list[str],
        result: ExecResult | None = None,
        error: RuntimeError | None = None,
    ) -> bool:
        if command[0] not in {"exec", "cp"}:
            return False
        if not self._command_targets_main_service(command):
            return False

        error_text = self._extract_retryable_error_text(result=result, error=error)
        return any(
            pattern in error_text
            for pattern in self._TRANSIENT_MAIN_UNAVAILABLE_PATTERNS
        )

    def _is_nonrecoverable_main_service_unavailable(
        self,
        command: list[str],
        result: ExecResult | None = None,
        error: RuntimeError | None = None,
    ) -> bool:
        if command[0] not in {"exec", "cp"}:
            return False
        if not self._command_targets_main_service(command):
            return False

        error_text = self._extract_retryable_error_text(result=result, error=error)
        return any(
            pattern in error_text
            for pattern in self._NONRECOVERABLE_MAIN_UNAVAILABLE_PATTERNS
        )

    async def _try_recover_main_service(
        self,
        trigger_command: list[str],
        attempt: int,
    ) -> bool:
        if not self._command_targets_main_service(trigger_command):
            return False

        try:
            ps_before = await self._run_docker_compose_command(["ps", "-a"], check=False)
            ps_before_text = (ps_before.stdout or ps_before.stderr or "").strip()
        except Exception as exc:
            ps_before_text = f"<failed to collect compose ps -a before recovery: {exc}>"

        self.logger.warning(
            "Main service unavailable for %s during docker compose %s. "
            "Attempting recovery with `up -d --wait %s` "
            "(retry attempt %s/%s). compose ps -a before recovery:\n%s",
            self.session_id,
            trigger_command[0] if trigger_command else "<unknown>",
            self._MAIN_SERVICE_NAME,
            attempt,
            self._MAIN_SERVICE_COMMAND_RETRY_ATTEMPTS,
            ps_before_text or "<empty>",
        )

        try:
            await self._run_docker_compose_command(
                ["up", "--detach", "--wait", self._MAIN_SERVICE_NAME],
                check=True,
                timeout_sec=self._MAIN_SERVICE_RECOVERY_TIMEOUT_SEC,
            )
        except Exception as exc:
            self.logger.warning(
                "Main service recovery failed for %s: %s",
                self.session_id,
                exc,
            )
            return False

        try:
            ps_after = await self._run_docker_compose_command(["ps", "-a"], check=False)
            ps_after_text = (ps_after.stdout or ps_after.stderr or "").strip()
        except Exception as exc:
            ps_after_text = f"<failed to collect compose ps -a after recovery: {exc}>"

        self.logger.warning(
            "Main service recovery succeeded for %s. compose ps -a after recovery:\n%s",
            self.session_id,
            ps_after_text or "<empty>",
        )
        return True

    async def _find_main_service_container_ids(self) -> list[str]:
        project_name = self._compose_project_name
        result = await self._run_host_command(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=com.docker.compose.project={project_name}",
                "--filter",
                f"label=com.docker.compose.service={self._MAIN_SERVICE_NAME}",
            ],
            check=False,
            timeout_sec=15,
        )
        output = (result.stdout or result.stderr or "").strip()
        if not output:
            return []
        return [line.strip() for line in output.splitlines() if line.strip()]

    async def get_main_service_identity(self) -> str | None:
        """Return a stable identity for the current main service container set."""
        container_ids = await self._find_main_service_container_ids()
        if not container_ids:
            return None
        return ",".join(sorted(container_ids))

    def set_fail_fast_on_main_service_unavailable(self, enabled: bool) -> None:
        self._fail_fast_on_main_service_unavailable = bool(enabled)

    async def _log_main_service_diagnostics(
        self,
        trigger_command: list[str],
        *,
        result: ExecResult | None = None,
        error: RuntimeError | None = None,
    ) -> None:
        if not self._command_targets_main_service(trigger_command):
            return

        headline = (
            f"Final main-service diagnostics for {self.session_id} "
            f"after docker compose {trigger_command[0] if trigger_command else '<unknown>'} failure."
        )
        error_text = self._extract_retryable_error_text(result=result, error=error)
        self.logger.warning("%s Last error text: %s", headline, error_text or "<empty>")

        try:
            compose_ps = await self._run_docker_compose_command(["ps", "-a"], check=False)
            compose_ps_text = (compose_ps.stdout or compose_ps.stderr or "").strip()
        except Exception as exc:
            compose_ps_text = f"<failed to collect compose ps -a: {exc}>"
        self.logger.warning(
            "Compose ps -a for %s:\n%s",
            self.session_id,
            compose_ps_text or "<empty>",
        )

        try:
            container_ids = await self._find_main_service_container_ids()
        except Exception as exc:
            self.logger.warning(
                "Failed to discover main container ids for %s: %s",
                self.session_id,
                exc,
            )
            return

        if not container_ids:
            self.logger.warning(
                "No main container ids found for %s during diagnostics.",
                self.session_id,
            )
            return

        self.logger.warning(
            "Main container ids for %s: %s",
            self.session_id,
            ", ".join(container_ids),
        )

        for container_id in container_ids:
            try:
                inspect_result = await self._run_host_command(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        "{{json .State}}",
                        container_id,
                    ],
                    check=False,
                    timeout_sec=15,
                )
                inspect_text = (inspect_result.stdout or inspect_result.stderr or "").strip()
            except Exception as exc:
                inspect_text = f"<failed to inspect container {container_id}: {exc}>"

            self.logger.warning(
                "Main container state for %s (%s): %s",
                self.session_id,
                container_id,
                inspect_text or "<empty>",
            )

            try:
                logs_result = await self._run_host_command(
                    [
                        "docker",
                        "logs",
                        "--tail",
                        str(self._MAIN_SERVICE_DIAGNOSTIC_LOG_TAIL_LINES),
                        container_id,
                    ],
                    check=False,
                    timeout_sec=15,
                )
                logs_text = (logs_result.stdout or logs_result.stderr or "").strip()
            except Exception as exc:
                logs_text = f"<failed to read docker logs for {container_id}: {exc}>"

            self.logger.warning(
                "Main container logs for %s (%s), tail=%s:\n%s",
                self.session_id,
                container_id,
                self._MAIN_SERVICE_DIAGNOSTIC_LOG_TAIL_LINES,
                logs_text or "<empty>",
            )

    async def _run_main_service_command_with_retry(
        self,
        command: list[str],
        check: bool,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        for attempt in range(1, self._MAIN_SERVICE_COMMAND_RETRY_ATTEMPTS + 1):
            try:
                result = await self._run_docker_compose_command(
                    command,
                    check=check,
                    timeout_sec=timeout_sec,
                )
            except RuntimeError as error:
                if self._fail_fast_on_main_service_unavailable and (
                    self._is_nonrecoverable_main_service_unavailable(
                        command,
                        error=error,
                    )
                    or self._is_transient_main_service_unavailable(
                        command,
                        error=error,
                    )
                ):
                    await self._log_main_service_diagnostics(
                        command,
                        error=error,
                    )
                    raise
                if self._is_nonrecoverable_main_service_unavailable(
                    command,
                    error=error,
                ):
                    await self._log_main_service_diagnostics(
                        command,
                        error=error,
                    )
                    raise
                should_retry = (
                    attempt < self._MAIN_SERVICE_COMMAND_RETRY_ATTEMPTS
                    and self._is_transient_main_service_unavailable(
                        command,
                        error=error,
                    )
                )
                if not should_retry:
                    if self._is_transient_main_service_unavailable(
                        command,
                        error=error,
                    ):
                        await self._log_main_service_diagnostics(
                            command,
                            error=error,
                        )
                    raise

                recovered = await self._try_recover_main_service(command, attempt)
                if recovered:
                    continue

                self.logger.warning(
                    "Transient docker compose %s failure for %s; retrying in %.1fs "
                    "(attempt %s/%s): %s",
                    command[0],
                    self.session_id,
                    self._MAIN_SERVICE_COMMAND_RETRY_DELAY_SEC,
                    attempt + 1,
                    self._MAIN_SERVICE_COMMAND_RETRY_ATTEMPTS,
                    error,
                )
                await asyncio.sleep(self._MAIN_SERVICE_COMMAND_RETRY_DELAY_SEC)
                continue

            should_retry = (
                result.return_code != 0
                and not self._fail_fast_on_main_service_unavailable
                and not self._is_nonrecoverable_main_service_unavailable(
                    command,
                    result=result,
                )
                and attempt < self._MAIN_SERVICE_COMMAND_RETRY_ATTEMPTS
                and self._is_transient_main_service_unavailable(
                    command,
                    result=result,
                )
            )
            if not should_retry:
                if (
                    result.return_code != 0
                    and (
                        self._is_transient_main_service_unavailable(
                            command,
                            result=result,
                        )
                        or self._is_nonrecoverable_main_service_unavailable(
                            command,
                            result=result,
                        )
                    )
                ):
                    await self._log_main_service_diagnostics(
                        command,
                        result=result,
                    )
                return result

            recovered = await self._try_recover_main_service(command, attempt)
            if recovered:
                continue

            self.logger.warning(
                "Transient docker compose %s failure for %s; retrying in %.1fs "
                "(attempt %s/%s): stdout=%r stderr=%r",
                command[0],
                self.session_id,
                self._MAIN_SERVICE_COMMAND_RETRY_DELAY_SEC,
                attempt + 1,
                self._MAIN_SERVICE_COMMAND_RETRY_ATTEMPTS,
                result.stdout,
                result.stderr,
            )
            await asyncio.sleep(self._MAIN_SERVICE_COMMAND_RETRY_DELAY_SEC)

        raise RuntimeError(
            f"Failed to run docker compose command after "
            f"{self._MAIN_SERVICE_COMMAND_RETRY_ATTEMPTS} attempts: {command}"
        )

    async def _hard_cleanup_containers(self) -> None:
        if self._keep_containers:
            return

        project_name = self._compose_project_name
        filter_value = f"label=com.docker.compose.project={project_name}"
        command = (
            "docker ps -a -q --filter "
            f"{shlex.quote(filter_value)}"
            " | xargs -r docker rm -f"
        )

        try:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            self.logger.debug("Docker not found; skipping container cleanup.")
            return

        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""

        removed_ids = [line.strip() for line in stdout.splitlines() if line.strip()]
        stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        non_missing_lines = []
        for line in stderr_lines:
            lowered = line.lower()
            if "no such container" in lowered or "no such object" in lowered:
                continue
            non_missing_lines.append(line)

        if process.returncode != 0 and non_missing_lines:
            self.logger.warning(
                "Hard cleanup failed for project %s: %s",
                project_name,
                "; ".join(non_missing_lines),
            )
            return

        if removed_ids:
            self.logger.warning(
                "Hard cleanup removed %s container(s) for project %s",
                len(removed_ids),
                project_name,
            )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        # Non-interactive execs (verifier, downloads, housekeeping) should not
        # allocate a TTY; doing so can hang when Harbor itself runs without one.
        exec_command = ["exec", "-T"]

        if cwd:
            exec_command.extend(["-w", cwd])

        exec_env = pager_env_dict()
        if env:
            exec_env.update(env)
        for key, value in exec_env.items():
            exec_command.extend(["-e", f"{key}={value}"])

        exec_command.append("main")
        exec_command.extend(["bash", "-c", command])

        return await self._run_main_service_command_with_retry(
            exec_command, check=False, timeout_sec=timeout_sec
        )

    async def _cleanup_build_cache(self) -> None:
        """Clean up Docker build cache to free up disk space."""
        try:
            # Use --max-used-space to keep cache under the specified size limit
            process = await asyncio.create_subprocess_exec(
                "docker",
                "buildx",
                "prune",
                "--force",
                "--max-used-space",
                "30GB",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()
        except Exception:
            # Fallback to docker builder prune if buildx is not available
            try:
                process = await asyncio.create_subprocess_exec(
                    "docker",
                    "builder",
                    "prune",
                    "--force",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await process.wait()
            except Exception:
                # Silent fail - build cache cleanup is not critical
                pass

    async def attach(self) -> None:
        variables = " ".join(
            f"export {k}={shlex.quote(str(v))}"
            for k, v in self._env_vars.to_env_dict(include_os_env=False).items()
        )

        # Build the -f flags for docker compose
        compose_file_args = []
        for path in self._docker_compose_paths:
            compose_file_args.extend(["-f", str(path.resolve().absolute())])

        project_name = self.session_id.lower().replace(".", "-")
        compose_base = ["docker", "compose", "-p", project_name] + compose_file_args

        os.execvp(
            "bash",
            [
                "bash",
                "-c",
                f"{variables}; "
                + " ".join(compose_base + ["exec", "-it", "main", "bash"])
                + "; "
                + " ".join(compose_base + ["down"]),
            ],
        )
