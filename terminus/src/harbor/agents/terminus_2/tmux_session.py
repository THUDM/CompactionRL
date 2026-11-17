import asyncio
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from harbor.agents.terminus_2.asciinema_handler import AsciinemaHandler
from harbor.environments.base import BaseEnvironment
from harbor.utils.logger import logger


class TmuxSession:
    _ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
    _ENDS_WITH_NEWLINE_PATTERN = r"[\r\n]$"
    _NEWLINE_CHARS = "\r\n"
    # tmux silently drops commands exceeding its internal buffer (~16 KB since
    # tmux 1.9, see https://github.com/tmux/tmux/issues/254). Keep a
    # conservative margin below that hard ceiling.
    _TMUX_SEND_KEYS_MAX_COMMAND_LENGTH = 16_000
    _TMUX_SESSION_START_MAX_ATTEMPTS = 10
    _TMUX_SESSION_START_RETRY_DELAY_SEC = 0.2
    _HOST_TMUX_BOOTSTRAP_DIR = Path(tempfile.gettempdir()) / "harbor-bootstrap"
    _HOST_TMUX_BINARY_PATH = _HOST_TMUX_BOOTSTRAP_DIR / "tmux.real"
    _HOST_TMUX_WRAPPER_PATH = _HOST_TMUX_BOOTSTRAP_DIR / "tmux"
    _HOST_TMUX_LIB_DIR = _HOST_TMUX_BOOTSTRAP_DIR / "lib"
    _PIPE_PANE_SIZE_SENTINEL = "__HARBOR_PIPE_SIZE__:"
    GET_ASCIINEMA_TIMESTAMP_SCRIPT_CONTAINER_PATH = Path(
        "/tmp/get-asciinema-timestamp.sh"
    )
    _GET_ASCIINEMA_TIMESTAMP_SCRIPT_HOST_PATH = Path(__file__).parent / (
        "get-asciinema-timestamp.sh"
    )

    def __init__(
        self,
        session_name: str,
        environment: BaseEnvironment,
        logging_path: Path,
        local_asciinema_recording_path: Path | None,
        remote_asciinema_recording_path: Path | None,
        pane_width: int = 160,
        pane_height: int = 40,
    ):
        try:
            self._pane_width = int(pane_width)
            self._pane_height = int(pane_height)
        except (ValueError, TypeError):
            raise ValueError("pane_width and pane_height must be valid integers.")
        if self._pane_width <= 0 or self._pane_height <= 0:
            raise ValueError("pane_width and pane_height must be positive integers.")
        self._logging_path = logging_path
        self._local_asciinema_recording_path = local_asciinema_recording_path
        self._remote_asciinema_recording_path = remote_asciinema_recording_path
        self._session_name = session_name
        self._logger = logger
        self._previous_buffer: str | None = None
        self._pipe_pane_offset_bytes: int = 0
        self._incremental_output_initialized = False
        self._disable_recording = False
        self._tmux_executable = "tmux"
        self._allow_host_tmux_fallback = os.environ.get(
            "HARBOR_ALLOW_HOST_TMUX_FALLBACK", ""
        ).lower() in {"1", "true", "yes"}
        self.environment = environment
        self._markers: list[tuple[float, str]] = []

    # TODO: Add asciinema logging
    # @property
    # def logging_path(self) -> Path:
    #     return (
    #         Path(DockerComposeManager.CONTAINER_SESSION_LOGS_PATH)
    #         / f"{self._session_name}.log"
    #     )

    # @property
    # def _recording_path(self) -> Path | None:
    #     if self._disable_recording:
    #         return None
    #     return (
    #         Path(DockerComposeManager.CONTAINER_SESSION_LOGS_PATH)
    #         / f"{self._session_name}.cast"
    #     )

    async def _attempt_tmux_installation(self) -> None:
        """
        Install both tmux and asciinema in a single operation for efficiency.
        """
        # Check what's already installed
        tmux_result = await self.environment.exec(command="tmux -V")
        tmux_installed = tmux_result.return_code == 0

        needs_asciinema = self._remote_asciinema_recording_path is not None
        if needs_asciinema:
            asciinema_result = await self.environment.exec(
                command="asciinema --version"
            )
            asciinema_installed = asciinema_result.return_code == 0
        else:
            asciinema_installed = True

        if tmux_installed and asciinema_installed:
            self._logger.debug("Both tmux and asciinema are already installed")
            return

        if tmux_installed and needs_asciinema and not asciinema_installed:
            self._logger.debug(
                "Skipping optional asciinema installation; recording will be disabled if unavailable"
            )
            return

        tools_needed = []
        if not tmux_installed:
            tools_needed.append("tmux")
        if needs_asciinema and not asciinema_installed:
            tools_needed.append("asciinema")

        self._logger.debug(f"Installing: {', '.join(tools_needed)}")

        # Detect system and package manager
        system_info = await self._detect_system_info()

        if system_info["package_manager"]:
            install_command = self._get_combined_install_command(
                system_info, tools_needed
            )
            if install_command:
                self._logger.debug(
                    f"Installing tools using {system_info['package_manager']}: {install_command}"
                )
                result = await self.environment.exec(command=install_command)

                if result.return_code == 0:
                    # Verify installations
                    if not tmux_installed:
                        verify_tmux = await self.environment.exec(command="tmux -V")
                        if verify_tmux.return_code != 0:
                            self._logger.warning(
                                "tmux installation verification failed"
                            )
                            await self._build_tmux_from_source()

                    if needs_asciinema and not asciinema_installed:
                        verify_asciinema = await self.environment.exec(
                            command="asciinema --version"
                        )
                        if verify_asciinema.return_code != 0:
                            self._logger.warning(
                                "asciinema installation verification failed"
                            )
                            await self._install_asciinema_with_pip()

                    return

        # Fallback to individual installations
        if not tmux_installed:
            self._logger.warning("Installing tmux from source...")
            await self._build_tmux_from_source()
            tmux_installed = await self._is_tmux_available()
            if (
                not tmux_installed
                and self._allow_host_tmux_fallback
                and await self._install_tmux_from_host()
            ):
                tmux_installed = True
            elif not tmux_installed and not self._allow_host_tmux_fallback:
                self._logger.debug(
                    "Skipping host tmux fallback; set HARBOR_ALLOW_HOST_TMUX_FALLBACK=1 to enable it"
                )

        if needs_asciinema and not asciinema_installed:
            self._logger.warning("Installing asciinema via pip...")
            await self._install_asciinema_with_pip()

    async def _is_tmux_available(self, executable: str | None = None) -> bool:
        if executable is None:
            command = self._tmux_version_probe_command()
        else:
            command = f"{shlex.quote(executable)} -V"
        result = await self.environment.exec(command=command)
        return result.return_code == 0

    async def _is_asciinema_available(self) -> bool:
        result = await self.environment.exec(command="asciinema --version")
        return result.return_code == 0

    async def _detect_system_info(self) -> dict:
        """
        Detect the operating system and available package managers.
        """
        system_info: dict[str, str | None] = {
            "os": None,
            "package_manager": None,
            "update_command": None,
        }

        # Check for OS release files
        os_release_result = await self.environment.exec(
            command="cat /etc/os-release 2>/dev/null || echo 'not found'"
        )

        # Check uname for system type
        uname_result = await self.environment.exec(command="uname -s")

        # Detect package managers by checking if they exist
        package_managers = [
            "apt-get",
            "dnf",
            "yum",
            "apk",
            "pacman",
            "brew",
            "pkg",
            "zypper",
        ]

        for pm_name in package_managers:
            check_result = await self.environment.exec(
                command=f"which {pm_name} >/dev/null 2>&1"
            )
            if check_result.return_code == 0:
                system_info["package_manager"] = pm_name
                break

        # Try to determine OS from available info
        if (
            os_release_result.return_code == 0
            and os_release_result.stdout
            and "not found" not in os_release_result.stdout
        ):
            stdout_lower = os_release_result.stdout.lower()
            if "ubuntu" in stdout_lower or "debian" in stdout_lower:
                system_info["os"] = "debian-based"
            elif "fedora" in stdout_lower or "rhel" in stdout_lower:
                system_info["os"] = "rhel-based"
            elif "alpine" in stdout_lower:
                system_info["os"] = "alpine"
            elif "arch" in stdout_lower:
                system_info["os"] = "arch"
        elif uname_result.return_code == 0 and uname_result.stdout:
            stdout_lower = uname_result.stdout.lower()
            if "darwin" in stdout_lower:
                system_info["os"] = "macos"
            elif "freebsd" in stdout_lower:
                system_info["os"] = "freebsd"

        self._logger.debug(f"Detected system: {system_info}")
        return system_info

    def _get_combined_install_command(self, system_info: dict, tools: list[str]) -> str:
        """
        Get the appropriate installation command for multiple tools based on system info.
        """
        package_manager = system_info.get("package_manager")

        if not package_manager or not isinstance(package_manager, str):
            return ""

        # Build the package list
        packages = " ".join(tools)

        # Package manager commands with non-interactive flags
        install_commands = {
            "apt-get": f"DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y {packages}",
            "dnf": f"dnf install -y {packages}",
            "yum": f"yum install -y {packages}",
            "apk": f"apk add --no-cache {packages}",
            "pacman": f"pacman -S --noconfirm {packages}",
            "brew": f"brew install {packages}",
            "pkg": f"ASSUME_ALWAYS_YES=yes pkg install -y {packages}",
            "zypper": f"zypper install -y -n {packages}",
        }

        return install_commands.get(package_manager, "")

    def _tmux_probe_candidates(self) -> list[str]:
        candidates = [
            self._tmux_executable,
            "tmux",
            "/usr/local/bin/tmux",
        ]
        if self._allow_host_tmux_fallback or self._tmux_executable == str(
            self._HOST_TMUX_WRAPPER_PATH
        ):
            candidates.append(str(self._HOST_TMUX_WRAPPER_PATH))
        return list(dict.fromkeys(candidates))

    def _tmux_version_probe_command(self) -> str:
        candidates = self._tmux_probe_candidates()
        return " || ".join(f"{shlex.quote(candidate)} -V" for candidate in candidates)

    async def _probe_tmux_executable(self, executable: str):
        return await self.environment.exec(command=f"{shlex.quote(executable)} -V")

    async def _select_working_tmux_executable(self) -> bool:
        for candidate in self._tmux_probe_candidates():
            result = await self._probe_tmux_executable(candidate)
            if result.return_code == 0:
                self._tmux_executable = candidate
                return True
        return False

    def _collect_host_tmux_dependencies(self, tmux_binary: Path) -> list[tuple[str, Path]]:
        result = subprocess.run(
            ["ldd", str(tmux_binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        dependencies: list[tuple[str, Path]] = []
        skip_prefixes = (
            "libc.so.",
            "libm.so.",
            "libresolv.so.",
            "libtinfo.so.",
            "ld-linux",
        )

        for line in result.stdout.splitlines():
            match = re.search(r"\s*(\S+)\s+=>\s+(/[^ ]+)", line)
            if not match:
                continue
            soname, lib_path = match.groups()
            if soname.startswith(skip_prefixes):
                continue
            resolved_path = Path(lib_path).resolve()
            if resolved_path.exists():
                dependencies.append((soname, resolved_path))

        return dependencies

    async def _install_tmux_from_host(self) -> bool:
        host_tmux = shutil.which("tmux")
        if not host_tmux:
            return False

        host_tmux_path = Path(host_tmux)
        previous_tmux_executable = self._tmux_executable
        try:
            dependencies = self._collect_host_tmux_dependencies(host_tmux_path)
        except (OSError, subprocess.CalledProcessError) as exc:
            self._logger.warning("Failed to inspect host tmux dependencies: %s", exc)
            return False

        bootstrap_dir = str(self._HOST_TMUX_BOOTSTRAP_DIR)
        lib_dir = str(self._HOST_TMUX_LIB_DIR)
        wrapper_path = str(self._HOST_TMUX_WRAPPER_PATH)
        real_binary_path = str(self._HOST_TMUX_BINARY_PATH)

        setup_result = await self.environment.exec(
            command=(
                f"mkdir -p {shlex.quote(bootstrap_dir)} {shlex.quote(lib_dir)} "
                f"&& rm -f {shlex.quote(real_binary_path)} {shlex.quote(wrapper_path)}"
            )
        )
        if setup_result.return_code != 0:
            self._logger.warning(
                "Failed to prepare host tmux bootstrap directory: %s",
                (setup_result.stderr or "").strip(),
            )
            return False

        try:
            await self.environment.upload_file(
                source_path=host_tmux_path,
                target_path=bootstrap_dir,
            )
            move_result = await self.environment.exec(
                command=(
                    f"mv {shlex.quote(str(self._HOST_TMUX_BOOTSTRAP_DIR / host_tmux_path.name))} "
                    f"{shlex.quote(real_binary_path)} && chmod +x {shlex.quote(real_binary_path)}"
                )
            )
            if move_result.return_code != 0:
                self._logger.warning(
                    "Failed to finalize host tmux upload: %s",
                    (move_result.stderr or "").strip(),
                )
                return False

            for soname, resolved_path in dependencies:
                await self.environment.upload_file(
                    source_path=resolved_path,
                    target_path=lib_dir,
                )
                if resolved_path.name != soname:
                    link_result = await self.environment.exec(
                        command=(
                            f"ln -sf {shlex.quote(resolved_path.name)} "
                            f"{shlex.quote(str(self._HOST_TMUX_LIB_DIR / soname))}"
                        )
                    )
                    if link_result.return_code != 0:
                        self._logger.warning(
                            "Failed to link host tmux dependency %s: %s",
                            soname,
                            (link_result.stderr or "").strip(),
                        )
                        return False

            wrapper_command = f"""cat <<'EOF' > {shlex.quote(wrapper_path)}
#!/bin/sh
export LD_LIBRARY_PATH={lib_dir}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
exec {real_binary_path} "$@"
EOF
chmod +x {shlex.quote(wrapper_path)}"""
            wrapper_result = await self.environment.exec(command=wrapper_command)
            if wrapper_result.return_code != 0:
                self._logger.warning(
                    "Failed to write host tmux wrapper: %s",
                    (wrapper_result.stderr or "").strip(),
                )
                return False
        except Exception as exc:
            self._logger.warning("Failed to upload host tmux fallback: %s", exc)
            return False

        self._tmux_executable = wrapper_path
        probe_result = await self._probe_tmux_executable(self._tmux_executable)
        if probe_result.return_code == 0:
            return True

        self._logger.warning(
            "Host tmux fallback is incompatible with environment: %s",
            self._format_exec_result(probe_result),
        )
        self._tmux_executable = previous_tmux_executable
        return False

    async def _build_tmux_from_source(self) -> None:
        """
        Build tmux from source as a fallback option.
        """
        try:
            # Install build dependencies based on detected system - with non-interactive flags
            dep_commands = [
                "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libevent-dev libncurses5-dev curl",
                "yum groupinstall -y 'Development Tools' && yum install -y libevent-devel ncurses-devel curl",
                "dnf groupinstall -y 'Development Tools' && dnf install -y libevent-devel ncurses-devel curl",
                "apk add --no-cache build-base libevent-dev ncurses-dev curl",
            ]

            # Try to install build dependencies
            for cmd in dep_commands:
                result = await self.environment.exec(command=cmd)
                if result.return_code == 0:
                    break

            # Download and build tmux
            build_cmd = (
                "cd /tmp && "
                "curl -L https://github.com/tmux/tmux/releases/download/3.4/tmux-3.4.tar.gz -o tmux.tar.gz && "
                "tar -xzf tmux.tar.gz && "
                "cd tmux-3.4 && "
                "./configure --prefix=/usr/local && "
                "make && "
                "make install"
            )

            result = await self.environment.exec(command=build_cmd)

            # Verify installation
            verify_result = await self.environment.exec(
                command=self._tmux_version_probe_command()
            )
            if verify_result.return_code == 0:
                self._logger.debug("tmux successfully built and installed from source")
            else:
                self._logger.error("Failed to install tmux from source")

        except Exception as e:
            self._logger.error(f"Failed to build tmux from source: {e}")

    async def _install_asciinema_with_pip(self) -> None:
        """
        Install asciinema using pip as a fallback.
        """
        try:
            # Try to install python3-pip first - with non-interactive flags
            pip_install_commands = [
                "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip",
                "yum install -y python3-pip",
                "dnf install -y python3-pip",
                "apk add --no-cache python3 py3-pip",
            ]

            # Try to install pip
            for cmd in pip_install_commands:
                result = await self.environment.exec(command=cmd)
                if result.return_code == 0:
                    break

            # Install asciinema using pip
            pip_commands = ["pip3 install asciinema", "pip install asciinema"]

            for cmd in pip_commands:
                result = await self.environment.exec(command=cmd)
                if result.return_code == 0:
                    # Verify installation
                    verify_result = await self.environment.exec(
                        command="asciinema --version"
                    )
                    if verify_result.return_code == 0:
                        self._logger.debug("asciinema successfully installed using pip")
                        return

            self._logger.error("Failed to install asciinema using pip")

        except Exception as e:
            self._logger.error(f"Failed to install asciinema with pip: {e}")

    @property
    def _tmux_start_session(self) -> str:
        # Return a single command string instead of a list
        # Use script to create a pseudo-TTY for tmux without needing Docker's -it flags
        return (
            f"mkdir -p {shlex.quote(str(self._logging_path.parent))} && "
            f": > {shlex.quote(str(self._logging_path))} && "
            f"export SHELL=/bin/bash && "
            f"export GIT_PAGER=cat && "
            f"export PAGER=cat && "
            # Use script to allocate a PTY for tmux, and propagate tmux's exit code.
            f'script -eqc "'
            f"{shlex.quote(self._tmux_executable)} new-session -x {self._pane_width} -y {self._pane_height} -d -s {shlex.quote(self._session_name)} 'bash --login' \\; "
            f"pipe-pane -t {shlex.quote(self._session_name)} "
            f"{shlex.quote(f'cat >> {self._logging_path}')}"
            f'" /dev/null'
        )

    @staticmethod
    def _format_exec_result(result) -> str:
        parts = []
        if result.stdout:
            stdout = result.stdout.strip()
            if stdout:
                parts.append(f"stdout: {stdout}")
        if result.stderr:
            stderr = result.stderr.strip()
            if stderr:
                parts.append(f"stderr: {stderr}")
        return "; ".join(parts) if parts else "no stdout/stderr captured"

    async def _wait_for_session_startup(self):
        last_result = None
        for attempt in range(self._TMUX_SESSION_START_MAX_ATTEMPTS):
            last_result = await self.environment.exec(
                command=(
                    f"{shlex.quote(self._tmux_executable)} has-session -t "
                    f"{shlex.quote(self._session_name)}"
                )
            )
            if last_result.return_code == 0:
                return last_result

            if attempt < self._TMUX_SESSION_START_MAX_ATTEMPTS - 1:
                await asyncio.sleep(self._TMUX_SESSION_START_RETRY_DELAY_SEC)

        return last_result

    def _tmux_send_keys(self, keys: list[str]) -> list[str]:
        """Build one or more ``tmux send-keys`` commands for *keys*.

        If the shell-escaped command would exceed the tmux command-length
        limit, the keys are spread across multiple commands so that each
        individual command stays within the limit.  Oversized single keys
        are split into sub-strings whose quoted form fits.
        """
        prefix = shlex.quote(self._tmux_executable) + " send-keys -t " + shlex.quote(
            self._session_name
        )
        max_len = self._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH

        escaped_keys = [shlex.quote(key) for key in keys]
        single = prefix + " " + " ".join(escaped_keys)
        if len(single) <= max_len:
            return [single]

        commands: list[str] = []
        current_escaped: list[str] = []
        current_len = len(prefix)

        def _flush() -> None:
            nonlocal current_len
            if current_escaped:
                commands.append(prefix + " " + " ".join(current_escaped))
                current_escaped.clear()
                current_len = len(prefix)

        for key in keys:
            escaped = shlex.quote(key)
            addition = 1 + len(escaped)  # space + quoted key

            if current_len + addition <= max_len:
                current_escaped.append(escaped)
                current_len += addition
            elif len(prefix) + addition <= max_len:
                _flush()
                current_escaped.append(escaped)
                current_len = len(prefix) + addition
            else:
                _flush()
                max_escaped = max_len - len(prefix) - 1
                for chunk_escaped in self._split_key_for_tmux(key, max_escaped):
                    if current_len + 1 + len(chunk_escaped) <= max_len:
                        current_escaped.append(chunk_escaped)
                        current_len += 1 + len(chunk_escaped)
                    else:
                        _flush()
                        current_escaped.append(chunk_escaped)
                        current_len = len(prefix) + 1 + len(chunk_escaped)

        _flush()
        return commands

    @staticmethod
    def _split_key_for_tmux(key: str, max_escaped_len: int) -> list[str]:
        """Split *key* into ``shlex.quote``-d chunks each ≤ *max_escaped_len*."""
        chunks: list[str] = []
        remaining = key
        while remaining:
            lo, hi, best = 1, len(remaining), 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if len(shlex.quote(remaining[:mid])) <= max_escaped_len:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            chunks.append(shlex.quote(remaining[:best]))
            remaining = remaining[best:]
        return chunks

    def _tmux_capture_pane(self, capture_entire: bool = False) -> str:
        if capture_entire:
            extra_args = ["-S", "-"]
        else:
            extra_args = []

        return " ".join(
            [
                shlex.quote(self._tmux_executable),
                "capture-pane",
                "-p",
                *extra_args,
                "-t",
                self._session_name,
            ]
        )

    async def start(self) -> None:
        await self._attempt_tmux_installation()
        if not await self._select_working_tmux_executable():
            raise RuntimeError(
                f"{self.environment.session_id}: tmux is unavailable after installation attempts"
            )
        start_session_result = await self.environment.exec(
            command=self._tmux_start_session
        )
        if start_session_result.return_code != 0:
            raise RuntimeError(
                "Failed to start tmux session. "
                f"{self._format_exec_result(start_session_result)}"
            )
        has_session_result = await self._wait_for_session_startup()
        if has_session_result.return_code != 0:
            raise RuntimeError(
                "Failed to verify tmux session startup. "
                f"{self._format_exec_result(has_session_result)}"
            )

        history_limit = 10_000_000
        command = (
            f"{shlex.quote(self._tmux_executable)} set-option -g "
            f"history-limit {history_limit}"
        )
        set_history_result = await self.environment.exec(command=command)
        if set_history_result.return_code != 0:
            self._logger.warning(
                "Failed to increase tmux history-limit: %s",
                (set_history_result.stderr or "").strip(),
            )

        if self._remote_asciinema_recording_path:
            if not await self._is_asciinema_available():
                self._logger.warning(
                    "asciinema is unavailable after installation attempts; disabling recording"
                )
                self._disable_recording = True
                self._remote_asciinema_recording_path = None
                self._local_asciinema_recording_path = None
                return
            self._logger.debug("Starting recording.")
            await self.send_keys(
                keys=[
                    f"asciinema rec --stdin {self._remote_asciinema_recording_path}",
                    "Enter",
                ],
                min_timeout_sec=1.0,
            )
            await self.send_keys(
                keys=[
                    "clear",
                    "Enter",
                ],
            )

        if self._remote_asciinema_recording_path:
            await self.environment.upload_file(
                source_path=self._GET_ASCIINEMA_TIMESTAMP_SCRIPT_HOST_PATH,
                target_path=str(self.GET_ASCIINEMA_TIMESTAMP_SCRIPT_CONTAINER_PATH),
            )

    async def stop(self) -> None:
        if self._remote_asciinema_recording_path:
            self._logger.debug("Stopping recording.")
            await self.send_keys(
                keys=["C-d"],
                min_timeout_sec=0.1,
            )

            # Wait a moment for the recording to finish writing
            import asyncio

            await asyncio.sleep(0.5)

            if self._local_asciinema_recording_path:
                self._local_asciinema_recording_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                # Ensure recording exists locally before merging markers
                await self.environment.download_file(
                    source_path=str(self._remote_asciinema_recording_path),
                    target_path=self._local_asciinema_recording_path,
                )

            # Merge markers into the recording
            if self._markers and self._local_asciinema_recording_path:
                self._logger.debug(
                    f"Merging {len(self._markers)} markers into recording"
                )
                handler = AsciinemaHandler(
                    self._markers, self._local_asciinema_recording_path
                )
                handler.merge_markers()
                self._logger.debug(
                    f"Successfully merged markers into {self._local_asciinema_recording_path}"
                )
            else:
                self._logger.debug("No markers to merge")
        ...

    def _is_enter_key(self, key: str) -> bool:
        return key in self._ENTER_KEYS

    def _ends_with_newline(self, key: str) -> bool:
        result = re.search(self._ENDS_WITH_NEWLINE_PATTERN, key)
        return result is not None

    async def is_session_alive(self) -> bool:
        """Check if the tmux session is still alive."""
        result = await self.environment.exec(
            command="{} has-session -t {}".format(
                shlex.quote(self._tmux_executable),
                shlex.quote(self._session_name),
            )
        )
        return result.return_code == 0

    def _is_executing_command(self, key: str) -> bool:
        return self._is_enter_key(key) or self._ends_with_newline(key)

    def _prevent_execution(self, keys: list[str]) -> list[str]:
        keys = keys.copy()
        while keys and self._is_executing_command(keys[-1]):
            if self._is_enter_key(keys[-1]):
                keys.pop()
            else:
                stripped_key = keys[-1].rstrip(self._NEWLINE_CHARS)

                if stripped_key:
                    keys[-1] = stripped_key
                else:
                    keys.pop()

        return keys

    def _prepare_keys(
        self,
        keys: str | list[str],
        block: bool,
    ) -> tuple[list[str], bool]:
        """
        Prepare keys for sending to the terminal.

        Args:
            keys (str | list[str]): The keys to send to the terminal.
            block (bool): Whether to wait for the command to complete.

        Returns:
            tuple[list[str], bool]: The keys to send to the terminal and whether the
                the keys are blocking.
        """
        if isinstance(keys, str):
            keys = [keys]

        if not block or not keys or not self._is_executing_command(keys[-1]):
            return keys, False

        keys = self._prevent_execution(keys)
        keys.extend([f"; {self._tmux_executable} wait -S done", "Enter"])

        return keys, True

    async def _send_blocking_keys(
        self,
        keys: list[str],
        max_timeout_sec: float,
    ):
        start_time_sec = time.time()

        for command in self._tmux_send_keys(keys):
            result = await self.environment.exec(command=command)
            if result.return_code != 0:
                raise RuntimeError(
                    f"{self.environment.session_id}: failed to send blocking keys: {result.stderr}"
                )

        result = await self.environment.exec(
            f"timeout {max_timeout_sec}s {shlex.quote(self._tmux_executable)} wait done"
        )
        if result.return_code != 0:
            raise TimeoutError(f"Command timed out after {max_timeout_sec} seconds")

        elapsed_time_sec = time.time() - start_time_sec
        self._logger.debug(f"Blocking command completed in {elapsed_time_sec:.2f}s.")

    async def _send_non_blocking_keys(
        self,
        keys: list[str],
        min_timeout_sec: float,
    ):
        start_time_sec = time.time()

        for command in self._tmux_send_keys(keys):
            result = await self.environment.exec(command=command)
            if result.return_code != 0:
                raise RuntimeError(
                    f"{self.environment.session_id}: failed to send non-blocking keys: "
                    f"stdout={result.stdout!r} stderr={result.stderr!r}"
                )

        elapsed_time_sec = time.time() - start_time_sec

        if elapsed_time_sec < min_timeout_sec:
            await asyncio.sleep(min_timeout_sec - elapsed_time_sec)

    async def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ):
        """
        Execute a command in the tmux session.

        Args:
            keys (str): The keys to send to the tmux session.
            block (bool): Whether to wait for the command to complete.
            min_timeout_sec (float): Minimum time in seconds to wait after executing.
                Defaults to 0.
            max_timeout_sec (float): Maximum time in seconds to wait for blocking
                commands. Defaults to 3 minutes.
        """
        if block and min_timeout_sec > 0.0:
            self._logger.debug("min_timeout_sec will be ignored because block is True.")

        prepared_keys, is_blocking = self._prepare_keys(
            keys=keys,
            block=block,
        )

        self._logger.debug(
            f"Sending keys: {prepared_keys}"
            f" min_timeout_sec: {min_timeout_sec}"
            f" max_timeout_sec: {max_timeout_sec}"
        )

        if is_blocking:
            await self._send_blocking_keys(
                keys=prepared_keys,
                max_timeout_sec=max_timeout_sec,
            )
        else:
            await self._send_non_blocking_keys(
                keys=prepared_keys,
                min_timeout_sec=min_timeout_sec,
            )

    async def capture_pane(self, capture_entire: bool = False) -> str:
        result = await self.environment.exec(
            self._tmux_capture_pane(capture_entire=capture_entire)
        )
        return result.stdout or ""

    async def _get_visible_screen(self) -> str:
        return await self.capture_pane(capture_entire=False)

    async def _read_pipe_pane_output_since_offset(self) -> tuple[str | None, int | None]:
        logging_path = shlex.quote(str(self._logging_path))
        command = (
            f"LOG_PATH={logging_path}; "
            f"OFFSET={int(self._pipe_pane_offset_bytes)}; "
            'SIZE=0; '
            'if [ -f "$LOG_PATH" ]; then '
            'SIZE=$(wc -c < "$LOG_PATH" | tr -d "[:space:]"); '
            'if [ -z "$SIZE" ]; then SIZE=0; fi; '
            'if [ "$SIZE" -lt "$OFFSET" ]; then OFFSET=0; fi; '
            'if [ "$SIZE" -gt "$OFFSET" ]; then tail -c +$((OFFSET + 1)) "$LOG_PATH"; fi; '
            "fi; "
            f'printf "\\n{self._PIPE_PANE_SIZE_SENTINEL}%s\\n" "$SIZE"'
        )
        result = await self.environment.exec(command=command)
        if result.return_code != 0:
            return None, None
        stdout = result.stdout or ""
        sentinel = f"\n{self._PIPE_PANE_SIZE_SENTINEL}"
        idx = stdout.rfind(sentinel)
        if idx < 0:
            fallback_idx = stdout.rfind(self._PIPE_PANE_SIZE_SENTINEL)
            if fallback_idx != 0:
                return None, None
            idx = fallback_idx
            content = ""
            size_part = stdout[len(self._PIPE_PANE_SIZE_SENTINEL):].strip()
        else:
            content = stdout[:idx]
            size_part = stdout[idx + len(sentinel):].strip()
        try:
            size = int(size_part)
        except (TypeError, ValueError):
            return None, None
        return content, size

    async def _find_new_content(self, current_buffer: str) -> str | None:
        pb = "" if self._previous_buffer is None else self._previous_buffer.strip()
        if pb in current_buffer:
            idx = current_buffer.index(pb)
            # Find the end of the previous buffer content
            if "\n" in pb:
                idx = pb.rfind("\n")
            return current_buffer[idx:]
        return None

    async def _get_incremental_output_via_capture_pane(self) -> str:
        current_buffer = await self.capture_pane(capture_entire=True)

        if self._previous_buffer is None:
            self._previous_buffer = current_buffer
            visible_screen = await self._get_visible_screen()
            return f"Current Terminal Screen:\n{visible_screen}"

        new_content = await self._find_new_content(current_buffer)
        self._previous_buffer = current_buffer

        if new_content is not None:
            if new_content.strip():
                return f"New Terminal Output:\n{new_content}"
            return f"Current Terminal Screen:\n{await self._get_visible_screen()}"
        return f"Current Terminal Screen:\n{await self._get_visible_screen()}"

    async def get_incremental_output(self) -> str:
        """
        Get either new terminal output since last call, or current screen if
        unable to determine.

        This method tracks the previous buffer state and attempts to find new content
        by comparing against the current full buffer. This provides better handling for
        commands with large output that would overflow the visible screen.

        Returns:
            str: Formatted output with either "New Terminal Output:" or
                 "Current Terminal Screen:"
        """
        if not self._incremental_output_initialized:
            _, size = await self._read_pipe_pane_output_since_offset()
            if size is not None:
                self._pipe_pane_offset_bytes = size
            self._incremental_output_initialized = True
            visible_screen = await self._get_visible_screen()
            return f"Current Terminal Screen:\n{visible_screen}"

        new_content, size = await self._read_pipe_pane_output_since_offset()
        if size is not None:
            self._pipe_pane_offset_bytes = size
            if new_content is not None:
                if new_content.strip():
                    return f"New Terminal Output:\n{new_content}"
                return f"Current Terminal Screen:\n{await self._get_visible_screen()}"

        return await self._get_incremental_output_via_capture_pane()
