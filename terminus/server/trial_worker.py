#!/usr/bin/env python3
"""
Standalone trial worker script.

This script runs a single Harbor trial and outputs the result as JSON to stdout.
It's designed to be spawned as a subprocess by harbor_server.py.

Usage:
    python trial_worker.py --instance-id <id> --dataset-path <path> --trials-dir <dir> \
        --unique-sid <sid> --max-iterations <n> [--api-base <url>] [--api-key <key>] \
        [--llm-query-type <name>] \
        [--tokenizer-path <path>] [--model-name <name>] [--rollout-max-context-length <n>] \
        [--sampling-params <json>]
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Add harbor to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harbor.trial.trial import Trial
from harbor.models.trial.config import (
    TrialConfig,
    TaskConfig,
    AgentConfig,
    EnvironmentConfig,
    VerifierConfig,
)
from harbor.models.agent.name import AgentName


DEFAULT_AGENT_NAME = AgentName.TERMINUS_2.value

dummy_reward = -100
dummy_messages = [
    {"role": "user", "content": "dummy", "num_tokens": 0},
    {
        "role": "assistant",
        "content": "dummy",
        "reasoning_content": "dummy",
        "weight_version": None,
        "num_tokens": 0,
    },
]


_OVERLONG_EXCEPTION_TYPES = {
    "ContextLengthExceededError",
    "ContextManagementLimitExceededError",
    "OutputLengthExceededError",
    "SingleTurnOverlongError",
}
_FORMAT_ERROR_EXCEPTION_TYPES = {
    "ParseError",
    "VerifierOutputParseError",
}
_TIMEOUT_EXCEPTION_TYPES = {
    "AgentTimeoutError",
}
_PAGER_USAGE_EXCEPTION_TYPES = {
    "PagerUsageError",
}


def _normalize_num_tokens(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value.strip()), 0)
        except ValueError:
            return 0
    return 0


def _compute_assistant_tokens(messages: list[dict] | None) -> int:
    if not isinstance(messages, list):
        return 0

    assistant_tokens = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        assistant_tokens += _normalize_num_tokens(message.get("num_tokens"))
    return assistant_tokens


def _load_trajectory(trial_dir: Path) -> dict | None:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if not trajectory_path.exists():
        return None
    try:
        with open(trajectory_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load trajectory from {trajectory_path}: {e}", file=sys.stderr, flush=True)
        return None


def _normalize_trajectory_tool_call(tool_call: dict) -> dict | None:
    if not isinstance(tool_call, dict):
        return None
    function_name = tool_call.get("function_name")
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}

    if function_name == "bash_command":
        function_name = "execute_commands"
        arguments = {"commands": arguments.get("keystrokes", "")}
    elif function_name == "mark_task_complete":
        function_name = "task_complete"
        arguments = {}
    elif function_name not in {"execute_commands", "task_complete", "image_read"}:
        return None

    return {
        "id": str(tool_call.get("tool_call_id") or ""),
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": arguments,
        },
    }


def _extract_trajectory_tool_calls_by_assistant_turn(
    trajectory_data: dict | None,
) -> list[list[dict] | None]:
    if not trajectory_data:
        return []
    steps = trajectory_data.get("steps")
    if not isinstance(steps, list):
        return []

    assistant_turn_tool_calls: list[list[dict] | None] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("source") != "agent":
            continue

        has_assistant_payload = any(
            step.get(key) for key in ("message", "reasoning_content", "tool_calls")
        )
        if not has_assistant_payload:
            continue

        raw_tool_calls = step.get("tool_calls")
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            normalized_tool_calls = [
                normalized
                for item in raw_tool_calls
                if (normalized := _normalize_trajectory_tool_call(item)) is not None
            ]
            assistant_turn_tool_calls.append(normalized_tool_calls or None)
        else:
            assistant_turn_tool_calls.append(None)
    return assistant_turn_tool_calls


def _extract_trajectory_tool_observations_by_assistant_turn(
    trajectory_data: dict | None,
) -> list[str | None]:
    if not trajectory_data:
        return []
    steps = trajectory_data.get("steps")
    if not isinstance(steps, list):
        return []

    assistant_turn_observations: list[str | None] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("source") != "agent":
            continue

        has_assistant_payload = any(
            step.get(key) for key in ("message", "reasoning_content", "tool_calls")
        )
        if not has_assistant_payload:
            continue

        raw_tool_calls = step.get("tool_calls")
        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            assistant_turn_observations.append(None)
            continue

        normalized_tool_calls = [
            normalized
            for item in raw_tool_calls
            if (normalized := _normalize_trajectory_tool_call(item)) is not None
        ]
        last_non_context_tool_call = None
        for tool_call in reversed(normalized_tool_calls):
            function_name = tool_call.get("function", {}).get("name")
            if function_name and function_name != "context_management":
                last_non_context_tool_call = tool_call
                break
        if last_non_context_tool_call is None:
            assistant_turn_observations.append(None)
            continue

        observation = step.get("observation") or {}
        results = observation.get("results") if isinstance(observation, dict) else None
        if not isinstance(results, list):
            assistant_turn_observations.append(None)
            continue

        matched_content: str | None = None
        target_call_id = last_non_context_tool_call.get("id")
        for result in results:
            if not isinstance(result, dict):
                continue
            source_call_id = result.get("source_call_id")
            content = result.get("content")
            if (
                isinstance(target_call_id, str)
                and target_call_id
                and source_call_id not in {None, "", target_call_id}
            ):
                continue
            if isinstance(content, str) and content:
                matched_content = content
                break
        assistant_turn_observations.append(matched_content)

    return assistant_turn_observations


def _inject_system_prompt_and_tool_calls(
    messages: list[dict] | None,
    system_prompt: str | None,
    trajectory_tool_calls_by_turn: list[list[dict] | None],
    trajectory_tool_observations_by_turn: list[str | None] | None = None,
) -> list[dict]:
    if not isinstance(messages, list):
        return []

    enriched_messages: list[dict] = []
    if isinstance(system_prompt, str) and system_prompt:
        if not messages or messages[0].get("role") != "system":
            enriched_messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                    "content_origin": system_prompt,
                    "num_tokens": 0,
                }
            )

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_copy = dict(msg)
        if "loss_mask" not in msg_copy and "_loss_mask" in msg_copy:
            msg_copy["loss_mask"] = bool(msg_copy["_loss_mask"])
        msg_copy.pop("_loss_mask", None)
        if "content_origin" not in msg_copy:
            content = msg_copy.get("content")
            msg_copy["content_origin"] = content if isinstance(content, str) else None
        msg_copy["num_tokens"] = _normalize_num_tokens(msg_copy.get("num_tokens"))
        enriched_messages.append(msg_copy)
    return enriched_messages


def _extract_export_tool_calls(message: dict | None) -> list[dict]:
    if not isinstance(message, dict):
        return []
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []

    normalized_tool_calls = []
    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_name = tool_call.get("function_name")
        if not isinstance(function_name, str) or not function_name.strip():
            function = tool_call.get("function")
            if isinstance(function, dict):
                nested_name = function.get("name")
                if isinstance(nested_name, str) and nested_name.strip():
                    function_name = nested_name
        tool_call_id = tool_call.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            nested_id = tool_call.get("id")
            if isinstance(nested_id, str) and nested_id.strip():
                tool_call_id = nested_id
        normalized_tool_calls.append(
            {
                "function_name": function_name.strip() if isinstance(function_name, str) else "",
                "tool_call_id": tool_call_id.strip() if isinstance(tool_call_id, str) else "",
            }
        )
    return normalized_tool_calls


def _get_last_export_tool_result_call_id(message: dict | None) -> str | None:
    tool_calls = _extract_export_tool_calls(message)
    for tool_call in reversed(tool_calls):
        function_name = tool_call.get("function_name")
        if function_name and function_name != "context_management":
            tool_call_id = tool_call.get("tool_call_id")
            return tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None
    return None


def _trajectory_has_task_complete(trajectory_data: dict | None) -> bool | None:
    if not trajectory_data:
        return None
    steps = trajectory_data.get("steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        tool_calls = step.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if tool_call.get("function_name") in {"mark_task_complete", "task_complete"}:
                return True
    return False


def _messages_have_task_complete(messages: list[dict] | None) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function_name = None
            function_payload = tool_call.get("function")
            if isinstance(function_payload, dict):
                function_name = function_payload.get("name")
            if function_name is None:
                function_name = tool_call.get("function_name")
            if function_name in {"mark_task_complete", "task_complete"}:
                return True
    return False


def _messages_have_any_assistant(messages: list[dict] | None) -> bool:
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict) and message.get("role") == "assistant"
        for message in messages
    )


def _message_groups_have_any_assistant(*message_groups: object) -> bool:
    for group in message_groups:
        if isinstance(group, list) and group and all(
            isinstance(item, dict) for item in group
        ):
            if _messages_have_any_assistant(group):
                return True
            continue
        if isinstance(group, list):
            for messages in group:
                if _messages_have_any_assistant(messages):
                    return True
    return False


def _metadata_has_any_assistant_message(metadata: dict | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    return _message_groups_have_any_assistant(
        metadata.get("messages_list"),
        metadata.get("all_messages"),
        metadata.get("managed_messages_list"),
    )


def _infer_initial_generate_completed(
    agent_metadata: dict | None,
    messages: list[dict] | None,
) -> bool:
    assistant_evidence = _messages_have_any_assistant(
        messages
    ) or _metadata_has_any_assistant_message(agent_metadata)
    explicit_value = (
        agent_metadata.get("initial_generate_completed")
        if isinstance(agent_metadata, dict)
        else None
    )
    explicit_completed = _coerce_bool(explicit_value, default=assistant_evidence)
    if not explicit_completed and assistant_evidence:
        # Some agents/older metadata either omit this field or leave a stale
        # false even though the exported conversation already contains model
        # responses. Do not drop otherwise trainable payloads in that case.
        return True
    return explicit_completed


def _parse_int_metadata(metadata: dict | None, key: str) -> int | None:
    if not metadata:
        return None
    value = metadata.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _parse_n_episodes(metadata: dict | None) -> int | None:
    return _parse_int_metadata(metadata, "n_episodes")


def _parse_sampling_params(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
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
            return default
    return default


def _normalize_summary_training_mode(
    value, default: str = "main_chat"
) -> str:
    if isinstance(value, bool):
        return "full" if value else "main_chat"
    if isinstance(value, (int, float)):
        return "full" if bool(value) else "main_chat"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return "full"
        if lowered in {"0", "false", "no", "n", "off", ""}:
            return "main_chat"
        if lowered in {"main_chat", "summary", "full"}:
            return lowered
    return default


def _normalize_context_management_mode(
    value, default: str = "none"
) -> str:
    if isinstance(value, bool):
        return "auto" if value else "none"
    if isinstance(value, (int, float)):
        return "auto" if bool(value) else "none"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return "auto"
        if lowered in {"0", "false", "no", "n", "off", ""}:
            return "none"
        if lowered in {"none", "auto", "tool"}:
            return lowered
    return default


def _classify_exception(
    exception_info,
) -> tuple[bool, bool, bool, bool, bool, str, str]:
    if not exception_info:
        return False, False, False, False, False, "", ""
    exc_type = exception_info.exception_type or ""
    exc_msg = exception_info.exception_message or ""
    lowered = f"{exc_type} {exc_msg}".lower()
    singleturn_overlong = exc_type == "SingleTurnOverlongError"
    overlong = (
        exc_type in _OVERLONG_EXCEPTION_TYPES
        or "context length" in lowered
        or "output length" in lowered
        or "max_tokens" in lowered
    )
    format_error = (
        exc_type in _FORMAT_ERROR_EXCEPTION_TYPES
        or "parse" in lowered
        or "format" in lowered
    )
    timeout = (
        exc_type in _TIMEOUT_EXCEPTION_TYPES
        or "agent execution timed out" in lowered
    )
    return (
        True,
        singleturn_overlong,
        overlong,
        format_error,
        timeout,
        exc_type,
        exc_msg,
    )


def _status_from_flags(
    singleturn_overlong: bool,
    overlong: bool,
    overturn: bool,
    format_error: bool,
    aborted: bool,
    timeout: bool = False,
    pager_usage: bool = False,
) -> str:
    if pager_usage:
        return "pager_usage"
    if timeout:
        return "timeout"
    if singleturn_overlong:
        return "singleturn_overlong"
    if overlong:
        return "overlong"
    if overturn:
        return "overturn"
    if format_error:
        return "format_error"
    if aborted:
        return "aborted"
    return "completed"


def _prune_incomplete_trailing_main_chat(
    managed_messages_list: list[list[dict]],
    rollout_messages_list: list[list[dict]],
    final_messages: list[dict],
    post_context_management_generate_pending: bool,
) -> tuple[list[list[dict]], list[dict]]:
    """Keep rebuilt continuations even if no post-context-management generate happened.

    After context management, the agent may rebuild the main chat and then stop
    before the first new generate call for that continuation. Keep that segment
    exportable so the buffer can save it locally as invalid, while normal filtering
    prevents it from being fetched for training.
    """
    _ = managed_messages_list
    _ = post_context_management_generate_pending
    return rollout_messages_list, final_messages

def _assemble_export_messages_list(
    *,
    rollout_messages_list: list[list[dict]],
    final_messages: list[dict],
    summary_messages_list: list[list[dict]],
    uid_prefix: str,
    agent_messages_tito_keys: list[str],
    agent_summary_tito_keys: list[str],
    summary_training_mode: str,
) -> tuple[list[list[dict]], list[str], list[str]]:
    _ = summary_messages_list
    _ = agent_summary_tito_keys
    _ = summary_training_mode

    messages_list = list(rollout_messages_list)
    messages_list_tags = ["main_chat"] * len(messages_list)
    messages_list_uids = [uid_prefix for _ in messages_list]

    if agent_messages_tito_keys:
        for i in range(len(messages_list_uids)):
            if i < len(agent_messages_tito_keys):
                messages_list_uids[i] = agent_messages_tito_keys[i]
            else:
                messages_list_uids[i] = agent_messages_tito_keys[-1]

    if not messages_list:
        messages_list = [final_messages]
        messages_list_tags = ["main_chat"]
        if agent_messages_tito_keys:
            messages_list_uids = [agent_messages_tito_keys[-1]]
        else:
            messages_list_uids = [uid_prefix]

    return messages_list, messages_list_tags, messages_list_uids


def _select_final_uid(
    *,
    uid_prefix: str,
    messages_list_tags: list[str],
    messages_list_uids: list[str],
) -> str:
    for idx in range(len(messages_list_tags) - 1, -1, -1):
        tag = str(messages_list_tags[idx] or "").strip().lower()
        if tag != "summary" and idx < len(messages_list_uids):
            candidate_uid = str(messages_list_uids[idx] or "").strip()
            if candidate_uid:
                return candidate_uid
    return uid_prefix

def _ensure_weight_version(messages: list[dict]) -> list[dict]:
    updated = []
    for msg in messages:
        msg_copy = dict(msg)
        if "loss_mask" not in msg_copy and "_loss_mask" in msg_copy:
            msg_copy["loss_mask"] = bool(msg_copy["_loss_mask"])
        msg_copy.pop("_loss_mask", None)
        if "content_origin" not in msg_copy:
            history_content = msg_copy.get("history_content")
            content = msg_copy.get("content")
            if isinstance(history_content, str):
                msg_copy["content_origin"] = history_content
            elif isinstance(content, str):
                msg_copy["content_origin"] = content
            else:
                msg_copy["content_origin"] = None
        msg_copy["num_tokens"] = _normalize_num_tokens(msg_copy.get("num_tokens"))
        if msg_copy.get("role") == "assistant":
            msg_copy.setdefault("weight_version", None)
        else:
            msg_copy.pop("weight_version", None)
        msg_copy.pop("history_content", None)
        updated.append(msg_copy)
    return updated


def _ensure_messages_start_with_system(
    messages: list[dict] | None,
    system_prompt: str | None,
) -> list[dict]:
    if not isinstance(messages, list):
        return []

    normalized_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_copy = dict(msg)
        if "loss_mask" not in msg_copy and "_loss_mask" in msg_copy:
            msg_copy["loss_mask"] = bool(msg_copy["_loss_mask"])
        msg_copy.pop("_loss_mask", None)
        if "content_origin" not in msg_copy:
            content = msg_copy.get("content")
            msg_copy["content_origin"] = content if isinstance(content, str) else None
        msg_copy["num_tokens"] = _normalize_num_tokens(msg_copy.get("num_tokens"))
        msg_copy.pop("history_content", None)
        normalized_messages.append(msg_copy)
    if not isinstance(system_prompt, str) or not system_prompt:
        return normalized_messages

    if normalized_messages and normalized_messages[0].get("role") == "system":
        normalized_messages[0]["content"] = system_prompt
        normalized_messages[0]["content_origin"] = system_prompt
        return normalized_messages

    return [
        {
            "role": "system",
            "content": system_prompt,
            "content_origin": system_prompt,
            "num_tokens": 0,
        },
        *normalized_messages,
    ]


def _extract_weight_versions(messages: list[dict]) -> list:
    return [
        msg.get("weight_version")
        for msg in messages
        if msg.get("role") == "assistant"
    ]


def _build_split_rollout_results(
    messages_list: list[list[dict]],
    messages_list_tags: list[str],
    messages_list_raw_rewards: list[float],
    messages_list_weight_versions: list[list],
    messages_list_uids: list[str] | None,
    system_prompt: str | None,
    base_uid: str | None,
    base_instance_id: str,
    context_management_count: int,
    context_management_limit_hit: bool,
    summary_context_management_indices: list[int] | None = None,
    rollout_status: str | None = None,
    post_context_management_generate_interrupted_reason: str | None = None,
    task_complete: bool | None = None,
    training_mode: bool = False,
    missing_first_generate_reason: str | None = None,
    missing_first_generate_split_index: int | None = None,
) -> list[dict]:
    split_results: list[dict] = []
    total = len(messages_list)
    uid_prefix = str(base_uid or "unknown_uid")
    data_id = uid_prefix
    cm_index = 0
    summary_index = 0
    for idx, messages in enumerate(messages_list):
        if not isinstance(messages, list) or not messages:
            continue
        tag = "main_chat"
        if idx < len(messages_list_tags):
            tag = str(messages_list_tags[idx])
        is_summary = tag.strip().lower() == "summary"
        export_system_prompt = (
            system_prompt
            if not is_summary and isinstance(system_prompt, str) and system_prompt
            else None
        )
        messages = _ensure_messages_start_with_system(messages, export_system_prompt)
        if is_summary:
            split_uid = f"{uid_prefix}_summary_{summary_index}"
            summary_index += 1
            split_kind = "summary"
        else:
            if cm_index == 0:
                split_uid = uid_prefix
            else:
                split_uid = f"{uid_prefix}_cm_{cm_index}"
            cm_index += 1
            split_kind = "cm"
        if isinstance(messages_list_uids, list) and idx < len(messages_list_uids):
            candidate_uid = str(messages_list_uids[idx] or "").strip()
            if candidate_uid:
                split_uid = candidate_uid
        raw_reward = dummy_reward
        if idx < len(messages_list_raw_rewards):
            raw_reward = messages_list_raw_rewards[idx]
        weight_versions = []
        if idx < len(messages_list_weight_versions):
            current = messages_list_weight_versions[idx]
            if isinstance(current, list):
                weight_versions = current
        if is_summary:
            current_context_management_index = summary_index
            if (
                isinstance(summary_context_management_indices, list)
                and summary_index - 1 < len(summary_context_management_indices)
            ):
                current_context_management_index = (
                    summary_context_management_indices[summary_index - 1]
                )
        else:
            if context_management_count <= 0:
                current_context_management_index = -1
            else:
                current_context_management_index = cm_index - 1
        split_metadata = {
            "data_id": data_id,
            "context_management_index": current_context_management_index,
            "context_management_limit_hit": context_management_limit_hit,
        }
        if (
            isinstance(missing_first_generate_reason, str)
            and missing_first_generate_reason
            and (
                missing_first_generate_split_index is None
                or idx == missing_first_generate_split_index
            )
        ):
            split_metadata["missing_first_generate_reason"] = (
                missing_first_generate_reason
            )
            split_metadata["buffer_filter_force_invalid"] = True
        if export_system_prompt:
            split_metadata["system_prompt"] = export_system_prompt
        split_metadata["assistant_tokens"] = _compute_assistant_tokens(messages)
        split_metadata["later_assistant_tokens"] = 0
        split_result = {
            "instance_id": f"{base_instance_id}_{uuid.uuid4()}",
            "messages": messages,
            "messages_list_tag": tag,
            "raw_reward": raw_reward,
            "reward": raw_reward,
            "uid": split_uid,
            "source_uid": uid_prefix,
            "split_kind": split_kind,
            "weight_versions": weight_versions,
            "messages_list_index": idx,
            "messages_list_total": total,
            "messages_list_source": "messages_list",
            "metadata": split_metadata,
            "meta_data": split_metadata,
            "system_prompt": export_system_prompt,
        }
        if is_summary:
            split_result.update(
                {
                    "overlong": False,
                    "format_error": False,
                    "overturn": False,
                    "aborted": False,
                    "rollout_status": "completed",
                }
            )
        split_results.append(split_result)

    main_chat_indices = [
        idx
        for idx, item in enumerate(split_results)
        if str(item.get("messages_list_tag") or "main_chat").strip().lower() != "summary"
    ]
    total_main_chat_segments = len(main_chat_indices)
    for position, split_idx in enumerate(main_chat_indices):
        split_metadata = split_results[split_idx].setdefault("metadata", {})
        split_results[split_idx]["meta_data"] = split_metadata
        current_cm_index = split_metadata.get("context_management_index")
        current_split_task_complete = _messages_have_task_complete(
            split_results[split_idx].get("messages")
        )
        next_outcome: dict[str, Any] = {
            "status": "unknown",
            "reason": "unknown",
            "message": "Next context-management outcome is unknown.",
            "next_context_management_index": (
                current_cm_index + 1 if isinstance(current_cm_index, int) else None
            ),
            "next_uid": None,
            "details": {
                "current_uid": split_results[split_idx].get("uid"),
                "current_messages_list_index": split_results[split_idx].get(
                    "messages_list_index"
                ),
                "current_messages_list_total": split_results[split_idx].get(
                    "messages_list_total"
                ),
                "current_context_management_index": current_cm_index,
                "context_management_count": context_management_count,
                "exported_main_chat_segments": total_main_chat_segments,
                "rollout_status": rollout_status,
                "current_split_task_complete": current_split_task_complete,
                "trial_task_complete": task_complete,
                "training_mode": training_mode,
                "context_management_limit_hit": context_management_limit_hit,
                "post_context_management_generate_interrupted_reason": (
                    post_context_management_generate_interrupted_reason
                ),
            },
        }
        if position + 1 < len(main_chat_indices):
            next_split = split_results[main_chat_indices[position + 1]]
            next_outcome = {
                "status": "succeeded",
                "reason": "next_context_management_segment_exported",
                "message": (
                    f"Next context-management segment was exported as "
                    f"{next_split.get('uid')}."
                ),
                "next_context_management_index": next_split.get("metadata", {}).get(
                    "context_management_index"
                ),
                "next_uid": next_split.get("uid"),
                "details": {
                    "current_uid": split_results[split_idx].get("uid"),
                    "next_exported_uid": next_split.get("uid"),
                    "current_context_management_index": current_cm_index,
                    "next_context_management_index": next_split.get("metadata", {}).get(
                        "context_management_index"
                    ),
                    "context_management_count": context_management_count,
                    "exported_main_chat_segments": total_main_chat_segments,
                    "rollout_status": rollout_status,
                    "current_split_task_complete": current_split_task_complete,
                    "trial_task_complete": task_complete,
                    "training_mode": training_mode,
                },
            }
        elif (
            isinstance(current_cm_index, int)
            and current_cm_index >= 0
            and current_cm_index < context_management_count
        ):
            status = "failed"
            reason = (
                "next_context_management_happened_but_following_continuation_segment_not_exported"
            )
            message = (
                "A later context-management action was recorded, but no following "
                "continuation segment was exported."
            )
            if current_split_task_complete is True:
                status = "not_triggered"
                reason = (
                    "task_completed_before_following_context_management_continuation_export"
                )
                message = (
                    "A context-management action happened, but the task was already "
                    "complete before any following continuation segment needed to be exported."
                )
            elif (
                post_context_management_generate_interrupted_reason
                == "context_length_before_post_context_management_generate"
            ):
                reason = "post_context_management_generate_context_length_exceeded"
                message = (
                    "The next context-management continuation could not be exported "
                    "because context length was exceeded before the first post-"
                    "context-management generate completed."
                )
            elif (
                post_context_management_generate_interrupted_reason
                == "auto_context_management_before_generate"
            ):
                reason = (
                    "automatic_context_management_retriggered_before_post_generate"
                )
                message = (
                    "The next context-management continuation could not be exported "
                    "because automatic context management would have triggered again "
                    "before the first post-context-management generate."
                )
            elif (
                post_context_management_generate_interrupted_reason
                == "session_ended_before_post_context_management_generate"
            ):
                reason = "session_ended_before_post_context_management_generate"
                message = (
                    "The next context-management continuation could not be exported "
                    "because the session ended before the first post-context-management "
                    "generate completed."
                )
            elif rollout_status == "format_error":
                reason = (
                    "run_ended_with_format_error_before_post_context_management_continuation_export"
                )
                message = (
                    "The next context-management continuation could not be exported "
                    "because the run ended with a format error before the rebuilt "
                    "continuation became exportable."
                )
            elif rollout_status == "pager_usage":
                reason = (
                    "run_ended_with_pager_usage_before_post_context_management_continuation_export"
                )
                message = (
                    "The next context-management continuation could not be exported "
                    "because the run ended after explicit pager usage before the "
                    "rebuilt continuation became exportable."
                )
            elif rollout_status in {"overlong", "singleturn_overlong"}:
                reason = (
                    "run_ended_overlong_before_post_context_management_continuation_export"
                )
                message = (
                    "The next context-management continuation could not be exported "
                    "because the run ended due to context/output limits before the "
                    "rebuilt continuation became exportable."
                )
            elif rollout_status == "overturn":
                reason = (
                    "run_ended_overturn_before_post_context_management_continuation_export"
                )
                message = (
                    "The next context-management continuation could not be exported "
                    "because the run ended after exceeding turn or timeout limits "
                    "before the rebuilt continuation became exportable."
                )
            elif rollout_status == "aborted":
                reason = (
                    "run_aborted_before_post_context_management_continuation_export"
                )
                message = (
                    "The next context-management continuation could not be exported "
                    "because the run aborted before the rebuilt continuation became exportable."
                )
            elif rollout_status == "completed":
                reason = (
                    "run_completed_before_post_context_management_continuation_export"
                )
                message = (
                    "A context-management action happened, but the run completed before "
                    "any following continuation segment was exported."
                )
            next_outcome = {
                "status": status,
                "reason": reason,
                "message": message,
                "next_context_management_index": current_cm_index + 1,
                "next_uid": None,
                "details": {
                    "failure_stage": "post_context_management_continuation_export",
                    "current_uid": split_results[split_idx].get("uid"),
                    "current_context_management_index": current_cm_index,
                    "expected_next_context_management_index": current_cm_index + 1,
                    "context_management_count": context_management_count,
                    "exported_main_chat_segments": total_main_chat_segments,
                    "current_messages_list_total": split_results[split_idx].get(
                        "messages_list_total"
                    ),
                    "current_messages_list_index": split_results[split_idx].get(
                        "messages_list_index"
                    ),
                    "rollout_status": rollout_status,
                    "current_split_task_complete": current_split_task_complete,
                    "trial_task_complete": task_complete,
                    "training_mode": training_mode,
                    "context_management_limit_hit": context_management_limit_hit,
                    "post_context_management_generate_interrupted_reason": (
                        post_context_management_generate_interrupted_reason
                    ),
                },
            }
        else:
            if context_management_limit_hit:
                status = "failed"
                reason = "context_management_limit_hit"
                message = (
                    "No later context-management segment was produced because the "
                    "context-management limit was reached."
                )
            elif current_split_task_complete is True:
                status = "not_triggered"
                reason = "task_completed_before_next_context_management"
                message = (
                    "No later context-management segment was needed because the task "
                    "was already complete."
                )
            elif rollout_status == "completed":
                status = "not_triggered"
                reason = "run_completed_before_next_context_management"
                message = (
                    "No later context-management segment was needed before the run "
                    "completed."
                )
            elif rollout_status == "format_error":
                status = "failed"
                reason = "run_ended_with_format_error_before_next_context_management"
                message = (
                    "No later context-management segment was produced because the run "
                    "ended with a format error first."
                )
            elif rollout_status == "pager_usage":
                status = "failed"
                reason = "run_ended_with_pager_usage_before_next_context_management"
                message = (
                    "No later context-management segment was produced because the run "
                    "ended after explicit pager usage first."
                )
            elif rollout_status in {"overlong", "singleturn_overlong"}:
                status = "failed"
                reason = "run_ended_overlong_before_next_context_management"
                message = (
                    "No later context-management segment was produced because the run "
                    "ended due to context/output length limits first."
                )
            elif rollout_status == "overturn":
                status = "failed"
                reason = "run_ended_overturn_before_next_context_management"
                message = (
                    "No later context-management segment was produced because the run "
                    "ended after exceeding the turn/timeout limits first."
                )
            elif rollout_status == "aborted":
                status = "failed"
                reason = "run_aborted_before_next_context_management"
                message = (
                    "No later context-management segment was produced because the run "
                    "aborted first."
                )
            else:
                status = "unknown"
                reason = "next_context_management_not_observed"
                message = (
                    "No later context-management segment was observed, and the run "
                    "termination reason was not explicit."
                )
            next_outcome = {
                "status": status,
                "reason": reason,
                "message": message,
                "next_context_management_index": (
                    current_cm_index + 1 if isinstance(current_cm_index, int) else None
                ),
                "next_uid": None,
                "details": {
                    "failure_stage": "before_next_context_management"
                    if status == "failed"
                    else "not_triggered",
                    "current_uid": split_results[split_idx].get("uid"),
                    "current_context_management_index": current_cm_index,
                    "next_context_management_index": (
                        current_cm_index + 1
                        if isinstance(current_cm_index, int)
                        else None
                    ),
                    "context_management_count": context_management_count,
                    "exported_main_chat_segments": total_main_chat_segments,
                    "current_messages_list_total": split_results[split_idx].get(
                        "messages_list_total"
                    ),
                    "current_messages_list_index": split_results[split_idx].get(
                        "messages_list_index"
                    ),
                    "rollout_status": rollout_status,
                    "current_split_task_complete": current_split_task_complete,
                    "trial_task_complete": task_complete,
                    "training_mode": training_mode,
                    "context_management_limit_hit": context_management_limit_hit,
                    "post_context_management_generate_interrupted_reason": (
                        post_context_management_generate_interrupted_reason
                    ),
                },
            }
        split_metadata["next_context_management_outcome"] = next_outcome

    later_running_sum = 0
    for split_idx in range(len(split_results) - 1, -1, -1):
        split_result = split_results[split_idx]
        split_metadata = split_result.setdefault("metadata", {})
        split_result["meta_data"] = split_metadata
        split_metadata["later_assistant_tokens"] = later_running_sum
        if str(split_result.get("messages_list_tag") or "").strip().lower() != "summary":
            later_running_sum += _normalize_num_tokens(
                split_metadata.get("assistant_tokens")
            )
    return split_results


def extract_messages_from_trial(
    trial_result,
    trial_dir: Path,
    system_prompt: str | None = None,
) -> tuple[list[list[dict]], list[list[dict]], list[dict], list[list[dict]]]:
    """Extract managed snapshots, rollout snapshots, final messages, and summary snapshots."""
    trajectory_data = _load_trajectory(trial_dir)

    if (trial_result.agent_result
        and trial_result.agent_result.metadata
    ):
        metadata = trial_result.agent_result.metadata
        managed_messages_list = metadata.get("managed_messages_list")
        normalized_managed_list = []
        if isinstance(managed_messages_list, list):
            for messages in managed_messages_list:
                if isinstance(messages, list) and messages:
                    normalized_managed_list.append(
                        _inject_system_prompt_and_tool_calls(
                            _ensure_weight_version(messages),
                            system_prompt,
                            [],
                            None,
                        )
                    )

        summary_messages_list = metadata.get("summary_messages_list")
        normalized_summary_list = []
        if isinstance(summary_messages_list, list):
            for messages in summary_messages_list:
                if isinstance(messages, list) and messages:
                    normalized_summary_list.append(_ensure_weight_version(messages))

        messages_list = metadata.get("messages_list")
        if isinstance(messages_list, list):
            messages_list_tags = metadata.get("messages_list_tags")
            if not isinstance(messages_list_tags, list):
                messages_list_tags = ["main_chat"] * len(messages_list)
            normalized_list = []
            for idx, messages in enumerate(messages_list):
                if isinstance(messages, list) and messages:
                    tag = (
                        str(messages_list_tags[idx])
                        if idx < len(messages_list_tags)
                        else "main_chat"
                    )
                    export_system_prompt = (
                        None if tag.strip().lower() == "summary" else system_prompt
                    )
                    normalized_list.append(
                        _inject_system_prompt_and_tool_calls(
                            _ensure_weight_version(messages),
                            export_system_prompt,
                            [],
                            None,
                        )
                    )
            if normalized_list:
                return (
                    normalized_managed_list,
                    normalized_list,
                    normalized_list[-1],
                    normalized_summary_list,
                )

        all_messages = metadata.get("all_messages")
        if isinstance(all_messages, list) and all_messages:
            final_messages = _inject_system_prompt_and_tool_calls(
                _ensure_weight_version(all_messages),
                system_prompt,
                [],
                None,
            )
            if normalized_managed_list:
                return (
                    normalized_managed_list,
                    [*normalized_managed_list, final_messages],
                    final_messages,
                    normalized_summary_list,
                )
            return [], [final_messages], final_messages, normalized_summary_list

    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if trajectory_path.exists():
        try:
            if trajectory_data is None:
                with open(trajectory_path, "r") as f:
                    trajectory_data = json.load(f)

            messages = []
            if "interactions" in trajectory_data:
                for interaction in trajectory_data["interactions"]:
                    if "input" in interaction:
                        messages.append({
                            "role": "user",
                            "content": interaction["input"]
                        })
                    if "output" in interaction:
                        messages.append({
                            "role": "assistant",
                            "content": interaction["output"],
                            "reasoning_content": interaction.get("reasoning_content", "")
                        })
            if not messages:
                print(f"[WARNING] [{trial_dir.name}] Trajectory parsed but no messages extracted from {trajectory_path}", file=sys.stderr, flush=True)
                return [], [dummy_messages], dummy_messages, []
            final_messages = _inject_system_prompt_and_tool_calls(
                _ensure_weight_version(messages),
                system_prompt,
                [],
                None,
            )
            return [], [final_messages], final_messages, []
        except Exception as e:
            print(f"[WARNING] [{trial_dir.name}] Failed to parse trajectory {trajectory_path}: {e}", file=sys.stderr, flush=True)

    return [], [dummy_messages], dummy_messages, []


def get_reward_from_verifier_result(verifier_result, instance_id: str = "unknown") -> float:
    """Extract reward from verifier result."""
    if verifier_result is None:
        print(f"[WARNING] [{instance_id}] verifier_result is None, returning dummy_reward={dummy_reward}", file=sys.stderr, flush=True)
        return dummy_reward

    rewards = verifier_result.rewards
    if rewards is None:
        print(f"[WARNING] [{instance_id}] verifier_result.rewards is None, returning dummy_reward={dummy_reward}", file=sys.stderr, flush=True)
        return dummy_reward

    if len(rewards) == 1:
        return list(rewards.values())[0]

    if "default" in rewards:
        return rewards["default"]
    if "main" in rewards:
        return rewards["main"]

    if not rewards:
        print(f"[WARNING] [{instance_id}] verifier_result.rewards is empty dict, returning dummy_reward={dummy_reward}", file=sys.stderr, flush=True)
    return list(rewards.values())[0] if rewards else dummy_reward


async def run_trial(args) -> dict:
    """Run a single trial and return the result."""
    instance_id = args.instance_id
    dataset_path = args.dataset_path
    trials_dir = Path(args.trials_dir)
    unique_sid = args.unique_sid

    # Create TaskConfig
    task_path = Path(dataset_path) / instance_id
    if not task_path.exists():
        raise FileNotFoundError(f"Task not found: {task_path}")

    task_config = TaskConfig(path=task_path)

    sampling_params = args.sampling_params or {}
    temperature = sampling_params.get("temperature")
    top_p = sampling_params.get("top_p")
    max_new_tokens = sampling_params.get("max_new_tokens")
    effective_max_input_tokens = args.rollout_max_context_length
    if (
        args.context_management_max_input_tokens is not None
        and args.context_management_max_input_tokens > 0
    ):
        effective_max_input_tokens = args.context_management_max_input_tokens
    effective_llm_query_type = args.llm_query_type
    if (
        not effective_llm_query_type
        and isinstance(args.model_name, str)
        and "claude" in args.model_name.lower()
    ):
        effective_llm_query_type = args.model_name

    agent_kwargs = {
        "max_turns": args.max_iterations,
        "store_all_messages": True,
        "session_id": args.uid or unique_sid,
        "api_base": args.api_base,
        "summary_api_base": args.summary_api_base or args.api_base,
        "llm_query_type": effective_llm_query_type,
        "api_key": args.api_key,
        "enable_summarize": args.enable_summarize,
        "summarization_mode": args.summarization_mode,
        "summary_training": args.summary_training,
        "summary_history_user_turns": args.summary_history_user_turns,
        "enable_tito": args.enable_tito,
        "model_info": {"max_input_tokens": effective_max_input_tokens},
        "tokenizer_path": args.tokenizer_path,
        "use_generate": args.use_generate,
        "remove_main_key": args.remove_main_key,
        "interleaved_thinking": True,
    }
    if args.chat_template_path:
        agent_kwargs["chat_template_path"] = args.chat_template_path
    if args.agent_name == AgentName.TERMINUS_2.value and args.parser_name:
        agent_kwargs["parser_name"] = args.parser_name
    if (
        args.context_management_proactive_threshold is not None
        and args.context_management_proactive_threshold >= 0
    ):
        agent_kwargs["proactive_summarization_threshold"] = (
            args.context_management_proactive_threshold
        )
    if (
        args.context_management_max_new_tokens is not None
        and args.context_management_max_new_tokens > 0
    ):
        agent_kwargs["max_new_tokens"] = args.context_management_max_new_tokens
    if args.context_management_max_count is not None:
        agent_kwargs["context_management_max_count"] = (
            args.context_management_max_count
        )
    if temperature is not None:
        try:
            agent_kwargs["temperature"] = float(temperature)
        except (TypeError, ValueError) as e:
            print(f"[WARNING] [{instance_id}] Failed to parse temperature={temperature!r}: {e}", file=sys.stderr, flush=True)
    if top_p is not None:
        try:
            agent_kwargs["top_p"] = float(top_p)
        except (TypeError, ValueError) as e:
            print(f"[WARNING] [{instance_id}] Failed to parse top_p={top_p!r}: {e}", file=sys.stderr, flush=True)
    if (
        max_new_tokens is not None
        and args.context_management_max_new_tokens is None
    ):
        try:
            agent_kwargs["max_new_tokens"] = int(max_new_tokens)
        except (TypeError, ValueError) as e:
            print(f"[WARNING] [{instance_id}] Failed to parse max_new_tokens={max_new_tokens!r}: {e}", file=sys.stderr, flush=True)

    agent_name = args.agent_name or args.agent
    if agent_name not in AgentName.values():
        raise ValueError(
            f"Invalid agent_name={agent_name!r}. Valid agent names: "
            f"{sorted(AgentName.values())}"
        )

    # Create AgentConfig
    agent_config = AgentConfig(
        name=agent_name,
        model_name=args.model_name or "default",
        kwargs=agent_kwargs,
    )

    # Create configs
    env_config = EnvironmentConfig(
        delete=False,
        kwargs={
            "override_cpus": 16,
            "override_memory_mb": 32768,
        },
    )
    verifier_config = VerifierConfig()

    trial_config = TrialConfig(
        task=task_config,
        trial_name=unique_sid,
        trials_dir=trials_dir,
        agent=agent_config,
        environment=env_config,
        verifier=verifier_config,
        tokenizer_path=args.tokenizer_path,
        timeout_multiplier=args.timeout_multiplier,
    )

    # Run trial
    trial = Trial(trial_config)
    trial_result = await trial.run()

    # Extract results
    trial_dir = trials_dir / unique_sid
    raw_reward = get_reward_from_verifier_result(
        trial_result.verifier_result,
        instance_id=instance_id,
    )

    uid_prefix = args.uid or unique_sid
    agent_messages_tito_keys = []
    agent_summary_tito_keys = []
    agent_system_prompt = None
    summary_context_management_indices = []
    post_context_management_generate_pending = False
    post_context_management_generate_interrupted_reason = None
    if (
        trial_result.agent_result
        and trial_result.agent_result.metadata
        and isinstance(trial_result.agent_result.metadata, dict)
    ):
        agent_metadata = trial_result.agent_result.metadata
        raw_main_keys = trial_result.agent_result.metadata.get("messages_list_tito_keys")
        if isinstance(raw_main_keys, list):
            agent_messages_tito_keys = [str(x) for x in raw_main_keys if x]
        raw_summary_context_management_indices = (
            trial_result.agent_result.metadata.get(
                "summary_messages_context_management_indices"
            )
        )
        raw_system_prompt = trial_result.agent_result.metadata.get("system_prompt")
        if isinstance(raw_system_prompt, str) and raw_system_prompt.strip():
            agent_system_prompt = raw_system_prompt
        post_context_management_generate_pending = _coerce_bool(
            agent_metadata.get("post_context_management_generate_pending"),
            default=False,
        )
        interrupted_reason = agent_metadata.get(
            "post_context_management_generate_interrupted_reason"
        )
        if isinstance(interrupted_reason, str) and interrupted_reason.strip():
            post_context_management_generate_interrupted_reason = interrupted_reason
        if isinstance(raw_summary_context_management_indices, list):
            summary_context_management_indices = [
                int(x)
                for x in raw_summary_context_management_indices
                if isinstance(x, int)
            ]

    managed_messages_list, rollout_messages_list, messages, summary_messages_list = (
        extract_messages_from_trial(
            trial_result,
            trial_dir,
            system_prompt=agent_system_prompt,
        )
    )
    original_rollout_messages_count = len(rollout_messages_list)
    rollout_messages_list, messages = _prune_incomplete_trailing_main_chat(
        managed_messages_list,
        rollout_messages_list,
        messages,
        post_context_management_generate_pending,
    )

    effective_summary_training_mode = (
        args.summary_training
        if args.summarization_mode == "summarize"
        else "main_chat"
    )
    messages_list, messages_list_tags, messages_list_uids = (
        _assemble_export_messages_list(
            rollout_messages_list=rollout_messages_list,
            final_messages=messages,
            summary_messages_list=summary_messages_list,
            uid_prefix=uid_prefix,
            agent_messages_tito_keys=agent_messages_tito_keys,
            agent_summary_tito_keys=agent_summary_tito_keys,
            summary_training_mode=effective_summary_training_mode,
        )
    )

    final_uid = _select_final_uid(
        uid_prefix=uid_prefix,
        messages_list_tags=messages_list_tags,
        messages_list_uids=messages_list_uids,
    )

    weight_versions = _extract_weight_versions(messages)
    messages_list_weight_versions = [
        _extract_weight_versions(item) for item in messages_list
    ]
    managed_messages_list_for_export = list(managed_messages_list)
    managed_messages_list_weight_versions = [
        _extract_weight_versions(item) for item in managed_messages_list_for_export
    ]
    context_management_count = len(managed_messages_list)

    # Determine status flags
    (
        has_exception,
        singleturn_overlong,
        overlong,
        format_error,
        timeout,
        exc_type,
        exc_msg,
    ) = _classify_exception(
        trial_result.exception_info
    )
    agent_metadata = (
        trial_result.agent_result.metadata
        if trial_result.agent_result and trial_result.agent_result.metadata
        else None
    )
    metadata_forced_status = (
        agent_metadata.get("forced_rollout_status")
        if isinstance(agent_metadata, dict)
        else None
    )
    pager_usage = (
        exc_type in _PAGER_USAGE_EXCEPTION_TYPES
        or metadata_forced_status == "pager_usage"
    )
    training_mode = os.environ.get("TRAINING", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "t",
    )
    initial_generate_completed = _infer_initial_generate_completed(
        agent_metadata,
        messages,
    )
    format_error_count = _parse_int_metadata(agent_metadata, "format_error_count")
    if training_mode and format_error_count is not None and format_error_count > 0:
        format_error = True
    n_episodes = _parse_n_episodes(agent_metadata)
    task_complete = _trajectory_has_task_complete(_load_trajectory(trial_dir))

    overturn = False
    if n_episodes is not None and n_episodes >= args.max_iterations:
        if task_complete is not True:
            overturn = True

    if (
        post_context_management_generate_interrupted_reason
        == "auto_context_management_before_generate"
        and training_mode
    ):
        overlong = True
        exc_type = "PostContextManagementGenerateOverlong"
        exc_msg = (
            "Automatic context management would trigger again before the first "
            "post-context-management generate."
        )

    aborted = has_exception and not (
        singleturn_overlong or overlong or format_error or overturn or pager_usage
    )
    context_management_limit_hit = (
        exc_type == "ContextManagementLimitExceededError"
    )

    rollout_status = _status_from_flags(
        singleturn_overlong=singleturn_overlong,
        overlong=overlong,
        overturn=overturn,
        format_error=format_error,
        aborted=aborted,
        timeout=timeout,
        pager_usage=pager_usage,
    )
    if pager_usage:
        raw_reward = 0
    if (
        post_context_management_generate_interrupted_reason
        == "auto_context_management_before_generate"
    ):
        raw_reward = dummy_reward
    post_context_management_missing_first_generate = (
        post_context_management_generate_pending
        and bool(managed_messages_list)
        and bool(rollout_messages_list)
    )
    pruned_new_context_management_segment = False
    missing_first_generate_reason = None
    missing_first_generate_split_index = None
    if not initial_generate_completed:
        missing_first_generate_reason = "initial_trajectory_missing_first_successful_generate"
        missing_first_generate_split_index = 0
        raw_reward = dummy_reward
    elif post_context_management_missing_first_generate:
        missing_first_generate_reason = (
            "post_context_management_trajectory_missing_first_successful_generate"
        )
        missing_first_generate_split_index = len(rollout_messages_list) - 1
        raw_reward = dummy_reward
    if singleturn_overlong and raw_reward == dummy_reward:
        raw_reward = 0
    if format_error and raw_reward == dummy_reward:
        raw_reward = 0
    if pager_usage:
        raw_reward = 0
    if aborted and missing_first_generate_reason is None:
        raw_reward = dummy_reward

    messages_list_raw_rewards = [raw_reward for _ in messages_list]
    split_rollout_results = _build_split_rollout_results(
        messages_list=messages_list,
        messages_list_tags=messages_list_tags,
        messages_list_raw_rewards=messages_list_raw_rewards,
        messages_list_weight_versions=messages_list_weight_versions,
        messages_list_uids=messages_list_uids,
        system_prompt=agent_system_prompt,
        base_uid=uid_prefix,
        base_instance_id=instance_id,
        context_management_count=context_management_count,
        context_management_limit_hit=context_management_limit_hit,
        summary_context_management_indices=summary_context_management_indices,
        rollout_status=rollout_status,
        post_context_management_generate_interrupted_reason=(
            post_context_management_generate_interrupted_reason
        ),
        task_complete=task_complete,
        training_mode=training_mode,
        missing_first_generate_reason=missing_first_generate_reason,
        missing_first_generate_split_index=missing_first_generate_split_index,
    )

    rollout_status_detail = {
        "exception_type": exc_type or None,
        "exception_message": exc_msg or None,
        "training_mode": training_mode,
        "n_episodes": n_episodes,
        "max_iterations": args.max_iterations,
        "task_complete": task_complete,
        "timeout": timeout,
        "pager_usage": pager_usage,
        "pager_usage_detail": (
            agent_metadata.get("pager_usage")
            if isinstance(agent_metadata, dict)
            else None
        ),
        "context_length_exceeded_count": _parse_int_metadata(
            agent_metadata, "context_length_exceeded_count"
        ),
        "output_length_exceeded_count": _parse_int_metadata(
            agent_metadata, "output_length_exceeded_count"
        ),
        "summarization_count": _parse_int_metadata(
            agent_metadata, "summarization_count"
        ),
        "context_management_max_count": _parse_int_metadata(
            agent_metadata, "context_management_max_count"
        ),
        "context_management_limit_hit": context_management_limit_hit,
        "training_mode": training_mode,
        "initial_generate_completed": initial_generate_completed,
        "missing_first_generate_reason": missing_first_generate_reason,
        "pruned_new_context_management_segment": pruned_new_context_management_segment,
        "post_context_management_missing_first_generate": (
            post_context_management_missing_first_generate
        ),
    }

    rollout_metadata = {
        "data_id": uid_prefix,
        "base_uid": uid_prefix,
        "final_main_chat_uid": final_uid,
        "agent_name": args.agent_name,
        "context_management_index": (
            context_management_count if context_management_count > 0 else -1
        ),
        "context_management_limit_hit": context_management_limit_hit,
        "messages_list_uids": list(messages_list_uids),
        "split_rollout_results_count": len(split_rollout_results),
        "drop_entire_payload": False,
        "initial_generate_completed": initial_generate_completed,
    }
    rollout_metadata["assistant_tokens"] = _compute_assistant_tokens(messages)
    rollout_metadata["later_assistant_tokens"] = 0
    debug_metadata_keys = (
        "terminus2_code_path",
        "main_key_recording_version",
    )
    if isinstance(agent_metadata, dict):
        for key in debug_metadata_keys:
            if key in agent_metadata:
                rollout_metadata[key] = agent_metadata.get(key)
        for split_result in split_rollout_results:
            split_metadata = split_result.setdefault("metadata", {})
            split_result["meta_data"] = split_metadata
            split_metadata["agent_name"] = args.agent_name
            for key in debug_metadata_keys:
                if key in agent_metadata:
                    split_metadata[key] = agent_metadata.get(key)
    if isinstance(agent_system_prompt, str) and agent_system_prompt:
        rollout_metadata["system_prompt"] = agent_system_prompt

    rollout_result = {
        "instance_id": instance_id,
        "sid": unique_sid,
        "uid": uid_prefix,
        "messages": messages,
        "managed_messages_list": managed_messages_list_for_export,
        "messages_list": messages_list,
        "messages_list_uids": messages_list_uids,
        "messages_list_raw_rewards": messages_list_raw_rewards,
        "split_rollout_results": split_rollout_results,
        "raw_reward": raw_reward,
        "singleturn_overlong": singleturn_overlong,
        "overturn": overturn,
        "overlong": overlong,
        "format_error": format_error,
        "aborted": aborted,
        "pager_usage": pager_usage,
        "rollout_status": rollout_status,
        "rollout_status_detail": rollout_status_detail,
        "weight_versions": weight_versions,
        "managed_messages_list_weight_versions": managed_messages_list_weight_versions,
        "messages_list_weight_versions": messages_list_weight_versions,
        "metadata": rollout_metadata,
        "meta_data": rollout_metadata,
    }
    if isinstance(agent_system_prompt, str) and agent_system_prompt:
        rollout_result["system_prompt"] = agent_system_prompt

    if trial_result.exception_info:
        rollout_result["error"] = f"{exc_type}: {exc_msg}"

    return rollout_result


async def run_trial_with_signals(args) -> dict:
    loop = asyncio.get_running_loop()
    task = loop.create_task(run_trial(args))
    previous_handlers = {}

    def _handle_signal(signum, frame):
        if not task.cancelled():
            task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_signal)

    try:
        return await task
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description="Run a single Harbor trial")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--trials-dir", required=True)
    parser.add_argument("--unique-sid", required=True)
    parser.add_argument("--agent-name", default=AgentName.TERMINUS_KIRA.value)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--summary-api-base", default=None)
    parser.add_argument("--llm-query-type", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--chat-template-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--rollout-max-context-length", type=int, default=131072)
    parser.add_argument("--sampling-params", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--parser-name", default="xml")
    parser.add_argument("--use-generate", default="true")
    parser.add_argument("--enable-summarize", default="none")
    parser.add_argument("--summarization-mode", default="summarize")
    parser.add_argument("--summary-training", default="false")
    parser.add_argument("--summary-history-user-turns", default="2")
    parser.add_argument("--enable-tito", default="false")
    parser.add_argument("--context-management-max-input-tokens", default=None)
    parser.add_argument("--context-management-proactive-threshold", default=None)
    parser.add_argument("--context-management-max-new-tokens", default=None)
    parser.add_argument("--context-management-max-count", default="-1")
    parser.add_argument("--remove-main-key", action="store_true")
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT_NAME,
        help="Harbor agent name to run, e.g. terminus-2 or terminus-2-cli.",
    )

    args = parser.parse_args()
    args.sampling_params = _parse_sampling_params(args.sampling_params)
    args.enable_summarize = _normalize_context_management_mode(
        args.enable_summarize, default="none"
    )
    args.summary_training = _normalize_summary_training_mode(
        args.summary_training, default="main_chat"
    )
    args.enable_tito = _coerce_bool(args.enable_tito, default=False)
    args.use_generate = _coerce_bool(args.use_generate, default=True)
    args.summary_history_user_turns = _coerce_int(
        args.summary_history_user_turns, default=2
    )
    args.summarization_mode = str(args.summarization_mode or "summarize").strip().lower()
    if args.summarization_mode not in {"summarize", "truncate"}:
        args.summarization_mode = "summarize"
    args.context_management_max_input_tokens = _coerce_int(
        args.context_management_max_input_tokens, default=None
    )
    args.context_management_proactive_threshold = _coerce_int(
        args.context_management_proactive_threshold, default=None
    )
    args.context_management_max_new_tokens = _coerce_int(
        args.context_management_max_new_tokens, default=None
    )
    args.context_management_max_count = _coerce_int(
        args.context_management_max_count, default=-1
    )

    try:
        result = asyncio.run(run_trial_with_signals(args))
        # Output result as single-line JSON to stdout
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)
    except Exception as e:
        import traceback
        error_metadata = {
            "messages_list_uids": [args.uid or args.unique_sid],
        }
        error_result = {
            "instance_id": args.instance_id,
            "sid": args.unique_sid,
            "uid": args.uid or args.unique_sid,
            "messages": dummy_messages,
            "managed_messages_list": [],
            "messages_list": [dummy_messages],
            "messages_list_uids": [args.uid or args.unique_sid],
            "messages_list_raw_rewards": [dummy_reward],
            "raw_reward": dummy_reward,
            "overturn": False,
            "overlong": False,
            "format_error": False,
            "aborted": True,
            "rollout_status": "aborted",
            "rollout_status_detail": {
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "n_episodes": None,
                "max_iterations": args.max_iterations,
                "task_complete": None,
                "timeout": None,
                "context_length_exceeded_count": None,
                "output_length_exceeded_count": None,
                "summarization_count": None,
                "context_management_max_count": args.context_management_max_count,
                "context_management_limit_hit": (
                    type(e).__name__ == "ContextManagementLimitExceededError"
                ),
            },
            "weight_versions": _extract_weight_versions(dummy_messages),
            "managed_messages_list_weight_versions": [],
            "messages_list_weight_versions": [_extract_weight_versions(dummy_messages)],
            "error": str(e),
            "metadata": {
                **error_metadata,
                "data_id": args.uid or args.unique_sid,
                "context_management_index": -1,
                "context_management_limit_hit": (
                    type(e).__name__ == "ContextManagementLimitExceededError"
                ),
            },
            "meta_data": {
                **error_metadata,
                "data_id": args.uid or args.unique_sid,
                "context_management_index": -1,
                "context_management_limit_hit": (
                    type(e).__name__ == "ContextManagementLimitExceededError"
                ),
            },
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
