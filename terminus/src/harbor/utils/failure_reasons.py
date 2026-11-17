from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from harbor.models.trial.result import TrialResult

FAILURE_REASON_TIMEOUT = "timeout"
FAILURE_REASON_OVERLONG = "overlong"
FAILURE_REASON_OVERTURN = "overturn"
FAILURE_REASON_FORMAT_ERROR = "format_error"
FAILURE_REASON_INCORRECT = "incorrect"
FAILURE_REASON_OTHER_ERROR = "other_error"

_OVERLONG_EXCEPTION_TYPES = {
    "ContextLengthExceededError",
    "ContextManagementLimitExceededError",
    "OutputLengthExceededError",
}
_FORMAT_ERROR_EXCEPTION_TYPES = {
    "ParseError",
    "VerifierOutputParseError",
}
_TIMEOUT_EXCEPTION_TYPES = {
    "AgentTimeoutError",
    "VerifierTimeoutError",
    "AgentSetupTimeoutError",
    "EnvironmentStartTimeoutError",
}


def _primary_reward(trial_result: TrialResult) -> float | None:
    verifier_result = trial_result.verifier_result
    if verifier_result is None or verifier_result.rewards is None:
        return None

    rewards = verifier_result.rewards
    if len(rewards) == 1:
        try:
            return float(next(iter(rewards.values())))
        except (TypeError, ValueError):
            return None

    for key in ("default", "main", "reward"):
        if key in rewards:
            try:
                return float(rewards[key])
            except (TypeError, ValueError):
                return None

    try:
        return float(next(iter(rewards.values())))
    except (StopIteration, TypeError, ValueError):
        return None


def _agent_metadata(trial_result: TrialResult) -> dict[str, Any]:
    metadata = (
        trial_result.agent_result.metadata
        if trial_result.agent_result and trial_result.agent_result.metadata
        else None
    )
    return metadata if isinstance(metadata, dict) else {}


def _parse_int_metadata(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _load_trajectory(trial_dir: Path | None) -> dict[str, Any] | None:
    if trial_dir is None:
        return None
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if not trajectory_path.exists():
        return None
    try:
        return json.loads(trajectory_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _trajectory_has_task_complete(trajectory_data: dict[str, Any] | None) -> bool | None:
    if not trajectory_data:
        return None
    steps = trajectory_data.get("steps")
    if not isinstance(steps, list):
        return None

    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_calls = step.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if (
                isinstance(tool_call, dict)
                and tool_call.get("function_name") in {"mark_task_complete", "task_complete"}
            ):
                return True
    return False


def trial_dir_from_result(trial_result: TrialResult) -> Path | None:
    trial_uri = getattr(trial_result, "trial_uri", None)
    if not isinstance(trial_uri, str) or not trial_uri:
        return None

    parsed = urlparse(trial_uri)
    if parsed.scheme in {"", "file"}:
        raw_path = unquote(parsed.path if parsed.scheme == "file" else trial_uri)
        if raw_path:
            return Path(raw_path)
    return None


def classify_failure_reason(
    trial_result: TrialResult,
    trial_dir: Path | None = None,
) -> tuple[str | None, dict[str, Any]]:
    if trial_dir is None:
        trial_dir = trial_dir_from_result(trial_result)

    reward = _primary_reward(trial_result)
    metadata = _agent_metadata(trial_result)
    exception_info = trial_result.exception_info
    exc_type = exception_info.exception_type if exception_info else ""
    exc_msg = exception_info.exception_message if exception_info else ""
    lowered = f"{exc_type} {exc_msg}".lower()

    format_error_count = _parse_int_metadata(metadata, "format_error_count") or 0
    context_length_exceeded_count = (
        _parse_int_metadata(metadata, "context_length_exceeded_count") or 0
    )
    output_length_exceeded_count = (
        _parse_int_metadata(metadata, "output_length_exceeded_count") or 0
    )
    n_episodes = _parse_int_metadata(metadata, "n_episodes")

    max_turns_value = trial_result.config.agent.kwargs.get("max_turns")
    if isinstance(max_turns_value, bool):
        max_turns = int(max_turns_value)
    elif isinstance(max_turns_value, int):
        max_turns = max_turns_value
    elif isinstance(max_turns_value, float):
        max_turns = int(max_turns_value)
    elif isinstance(max_turns_value, str):
        try:
            max_turns = int(max_turns_value.strip())
        except ValueError:
            max_turns = None
    else:
        max_turns = None

    task_complete = _trajectory_has_task_complete(_load_trajectory(trial_dir))

    timeout = exc_type in _TIMEOUT_EXCEPTION_TYPES or "timed out" in lowered
    overlong = (
        exc_type in _OVERLONG_EXCEPTION_TYPES
        or context_length_exceeded_count > 0
        or output_length_exceeded_count > 0
        or "context length" in lowered
        or "output length" in lowered
        or "max_tokens" in lowered
    )
    format_error = (
        exc_type in _FORMAT_ERROR_EXCEPTION_TYPES
        or format_error_count > 0
        or "parse" in lowered
        or "format" in lowered
    )
    overturn = (
        max_turns is not None
        and n_episodes is not None
        and n_episodes >= max_turns
        and task_complete is not True
    )
    success = reward is not None and reward > 0
    incorrect = not success and reward is not None and reward <= 0
    other_error = exception_info is not None and not any(
        (timeout, overlong, format_error, overturn)
    )

    if success:
        reason = None
    elif timeout:
        reason = FAILURE_REASON_TIMEOUT
    elif overlong:
        reason = FAILURE_REASON_OVERLONG
    elif overturn:
        reason = FAILURE_REASON_OVERTURN
    elif format_error:
        reason = FAILURE_REASON_FORMAT_ERROR
    elif other_error:
        reason = FAILURE_REASON_OTHER_ERROR
    elif incorrect:
        reason = FAILURE_REASON_INCORRECT
    else:
        reason = None

    detail = {
        "reward": reward,
        "exception_type": exc_type or None,
        "exception_message": exc_msg or None,
        "timeout": timeout,
        "overlong": overlong,
        "overturn": overturn,
        "format_error": format_error,
        "incorrect": incorrect,
        "other_error": other_error,
        "n_episodes": n_episodes,
        "max_turns": max_turns,
        "task_complete": task_complete,
        "format_error_count": format_error_count,
        "context_length_exceeded_count": context_length_exceeded_count,
        "output_length_exceeded_count": output_length_exceeded_count,
    }
    return reason, detail


def apply_failure_reason(
    trial_result: TrialResult,
    trial_dir: Path | None = None,
) -> TrialResult:
    reason, detail = classify_failure_reason(trial_result, trial_dir=trial_dir)
    trial_result.failure_reason = reason
    trial_result.failure_reason_detail = detail

    if trial_result.agent_result is not None:
        if trial_result.agent_result.metadata is None:
            trial_result.agent_result.metadata = {}
        if isinstance(trial_result.agent_result.metadata, dict):
            trial_result.agent_result.metadata["failure_reason"] = reason
            trial_result.agent_result.metadata["failure_reason_detail"] = detail

    return trial_result
