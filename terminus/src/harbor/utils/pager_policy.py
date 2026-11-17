"""Pager policy helpers for terminal agents."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass


PAGER_ENV = {"PAGER": "cat", "GIT_PAGER": "cat"}
PAGER_OUTPUT_LIMIT_BYTES = 16 * 1024
PAGER_USAGE_STATUS = "pager_usage"

_EXPLICIT_PAGER_COMMANDS = {
    "less",
    "more",
    "most",
    "view",
    "vi",
    "vim",
    "nvim",
}
_GIT_PAGER_SUBCOMMANDS = {
    "blame",
    "diff",
    "grep",
    "log",
    "show",
    "whatchanged",
}
_SAFE_PAGER_VALUES = {"", "cat", "/bin/cat", "/usr/bin/cat"}
_CONTROL_TOKENS = {"|", "||", "&", "&&", ";", "\n"}
_HEREDOC_OPERATORS = {"<<", "<<-"}
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)


@dataclass(frozen=True)
class PagerUsageDetection:
    is_pager_usage: bool
    reason: str | None = None
    command: str | None = None


def pager_env_exports() -> str:
    return "export GIT_PAGER=cat PAGER=cat"


def pager_env_dict() -> dict[str, str]:
    return dict(PAGER_ENV)


def detect_pager_usage(keystrokes: str) -> PagerUsageDetection:
    """Detect explicit pager/editor usage not neutralized by PAGER/GIT_PAGER=cat."""
    for segment in _iter_command_segments(keystrokes):
        detection = _detect_segment_pager_usage(segment)
        if detection.is_pager_usage:
            return detection
    return PagerUsageDetection(False)


def has_neutralized_pager_output(keystrokes: str) -> bool:
    """Return true for pager-capable commands made safe by PAGER/GIT_PAGER=cat."""
    if detect_pager_usage(keystrokes).is_pager_usage:
        return False
    return any(_segment_has_neutralized_pager_output(segment) for segment in _iter_command_segments(keystrokes))


def _iter_command_segments(keystrokes: str) -> list[list[str]]:
    sanitized_keystrokes = _strip_heredoc_bodies(keystrokes)
    segments: list[list[str]] = []
    try:
        lexer = shlex.shlex(
            sanitized_keystrokes,
            posix=True,
            punctuation_chars="|&;<>()",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        tokens = sanitized_keystrokes.replace("\n", " ; ").split()

    current: list[str] = []
    for token in tokens:
        if token in _CONTROL_TOKENS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _strip_heredoc_bodies(keystrokes: str) -> str:
    pending_delimiters: list[tuple[str, bool]] = []
    sanitized_lines: list[str] = []

    for line in keystrokes.splitlines(keepends=True):
        if pending_delimiters:
            delimiter, allow_tab_strip = pending_delimiters[0]
            candidate = line.rstrip("\r\n")
            if allow_tab_strip:
                candidate = candidate.lstrip("\t")
            if candidate == delimiter:
                pending_delimiters.pop(0)
            continue

        sanitized_lines.append(line)
        pending_delimiters.extend(_find_heredoc_delimiters(line))

    return "".join(sanitized_lines)


def _find_heredoc_delimiters(line: str) -> list[tuple[str, bool]]:
    try:
        lexer = shlex.shlex(line, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return []

    delimiters: list[tuple[str, bool]] = []
    index = 0
    while index < len(tokens) - 1:
        token = tokens[index]
        if token not in _HEREDOC_OPERATORS:
            index += 1
            continue
        delimiter = tokens[index + 1]
        if delimiter and delimiter not in _CONTROL_TOKENS:
            delimiters.append((delimiter, token == "<<-"))
        index += 2
    return delimiters


def _detect_segment_pager_usage(tokens: list[str]) -> PagerUsageDetection:
    env_overrides, command_tokens = _strip_leading_env(tokens)
    for key, value in env_overrides.items():
        if key in {"PAGER", "GIT_PAGER"} and not _is_safe_pager_value(value):
            return PagerUsageDetection(
                True,
                reason=f"{key} overrides pager to {value!r}",
                command=" ".join(tokens),
            )

    if not command_tokens:
        return PagerUsageDetection(False)

    command = _basename(command_tokens[0])
    if command in _EXPLICIT_PAGER_COMMANDS:
        return PagerUsageDetection(
            True,
            reason=f"explicit pager/editor command {command!r}",
            command=" ".join(tokens),
        )

    if command != "git":
        return PagerUsageDetection(False)

    if _git_invokes_explicit_pager(command_tokens[1:]):
        return PagerUsageDetection(
            True,
            reason="explicit git pager configuration",
            command=" ".join(tokens),
        )

    return PagerUsageDetection(False)


def _strip_leading_env(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    env_overrides: dict[str, str] = {}
    remaining = list(tokens)

    while remaining and _ASSIGNMENT_RE.match(remaining[0]):
        key, value = remaining.pop(0).split("=", 1)
        env_overrides[key] = value

    if remaining and _basename(remaining[0]) == "env":
        remaining.pop(0)
        while remaining and _ASSIGNMENT_RE.match(remaining[0]):
            key, value = remaining.pop(0).split("=", 1)
            env_overrides[key] = value

    return env_overrides, remaining


def _git_invokes_explicit_pager(args: list[str]) -> bool:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--no-pager":
            index += 1
            continue
        if arg in {"--paginate", "-p"}:
            return True
        if arg == "-c" and index + 1 < len(args):
            if _is_unsafe_core_pager_setting(args[index + 1]):
                return True
            index += 2
            continue
        if arg.startswith("-c"):
            setting = arg[2:]
            if setting and _is_unsafe_core_pager_setting(setting):
                return True
        if not arg.startswith("-"):
            break
        index += 1

    if args and args[0] == "config":
        return _git_config_sets_unsafe_pager(args[1:])
    return False


def _git_config_sets_unsafe_pager(args: list[str]) -> bool:
    remaining = [arg for arg in args if arg not in {"--global", "--local", "--system", "--worktree"}]
    for index, arg in enumerate(remaining):
        if arg in {"core.pager", "pager.log", "pager.show", "pager.diff"}:
            if index + 1 < len(remaining):
                return not _is_safe_pager_value(remaining[index + 1])
    return False


def _is_unsafe_core_pager_setting(setting: str) -> bool:
    key, sep, value = setting.partition("=")
    if sep != "=":
        return False
    normalized_key = key.strip().lower()
    if normalized_key not in {"core.pager", "pager.log", "pager.show", "pager.diff"}:
        return False
    return not _is_safe_pager_value(value)


def _segment_has_neutralized_pager_output(tokens: list[str]) -> bool:
    _, command_tokens = _strip_leading_env(tokens)
    if not command_tokens or _basename(command_tokens[0]) != "git":
        return False

    args = command_tokens[1:]
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-c":
            index += 2
            continue
        if arg.startswith("-c"):
            index += 1
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return arg in _GIT_PAGER_SUBCOMMANDS
    return False


def _is_safe_pager_value(value: str) -> bool:
    return value.strip().lower() in _SAFE_PAGER_VALUES


def _basename(command: str) -> str:
    return command.rsplit("/", 1)[-1].strip().lower()
