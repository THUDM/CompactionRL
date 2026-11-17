"""
Harbor Server Compatible - Subprocess-based trial execution.

This server is designed for swe_generator payloads and does not support
legacy request formats.
"""

import time
import argparse
import asyncio
import json
import os
import re
import sys
import uuid
import datetime
import shutil
import logging
import subprocess
import shlex
from html import escape
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    import orjson
except ModuleNotFoundError:
    orjson = None
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from harbor.models.agent.name import AgentName
from harbor.models.trial.result import TrialResult
from harbor.utils.profiling_timeline import (
    _CATEGORY_COLORS,
    _render_svg,
    build_trial_profile_record,
)


def json_dumps(obj: object) -> str:
    if orjson is not None:
        return orjson.dumps(obj).decode("utf-8")
    return json.dumps(obj, ensure_ascii=False)

app = FastAPI()

# Thread pool for running subprocesses
_executor = ThreadPoolExecutor(max_workers=int(os.environ.get('HARBOR_MAX_WORKERS', '16')), thread_name_prefix='worker-')

_running_workers_count = 0
_running_workers_lock = None
_running_workers_details = {}
_profiling_artifacts_lock = None

_DEFAULT_EVAL_OUTPUT_DIR = str(
    (Path(__file__).resolve().parent.parent / "harbor_logs").resolve()
)

EVAL_OUTPUT_DIR = os.environ.get('EVAL_OUTPUT_DIR', _DEFAULT_EVAL_OUTPUT_DIR)
KEEP_TEMP_DIR = os.environ.get('HARBOR_KEEP_TEMP_DIR', 'true').lower() in ['true', '1', 'yes']
HARBOR_AGENT_NAME = os.environ.get('HARBOR_AGENT_NAME', 'terminus-kira')
ENABLE_PROFILING = os.environ.get("HARBOR_ENABLE_PROFILING", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TRIAL_WORKER_PATH = Path(__file__).parent / "trial_worker.py"
_DEFAULT_DATASET_PATH = ""
DATASET_PATH = os.environ.get("DATASET_PATH")
if not DATASET_PATH and Path(_DEFAULT_DATASET_PATH).exists():
    DATASET_PATH = _DEFAULT_DATASET_PATH
_current_dataset_info = None
_current_dataset_map = {}
_ROLL_OUT_JSON_DIR_ENV = "HARBOR_ROLLOUT_JSON_DIR"
_ROLL_OUT_JSON_DIR = None
_SERVER_START_TS = None

# Setup logging
LOG_FILE = os.environ.get('HARBOR_SERVER_LOG', os.path.join(EVAL_OUTPUT_DIR, 'harbor_server.log'))
LOG_LEVEL = os.environ.get('HARBOR_LOG_LEVEL', 'INFO').upper()
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

logger = logging.getLogger(__name__)


def log_info(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)
    logging.info(msg)


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
dummy_rollout_result = {
    "instance_id": "dummy",
    "uid": "dummy",
    "messages": dummy_messages,
    "managed_messages_list": [],
    "messages_list": [dummy_messages],
    "messages_list_uids": ["dummy"],
    "messages_list_raw_rewards": [dummy_reward],
    "raw_reward": dummy_reward,
    "overturn": False,
    "overlong": False,
    "format_error": False,
    "aborted": False,
    "weight_versions": [None],
    "metadata": {
        "data_id": "dummy",
        "context_management_index": -1,
        "context_management_limit_hit": False,
        "messages_list_uids": ["dummy"],
    },
    "meta_data": {
        "data_id": "dummy",
        "context_management_index": -1,
        "context_management_limit_hit": False,
        "messages_list_uids": ["dummy"],
    },
}


def _normalize_num_tokens(value) -> int:
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


def _compute_assistant_tokens(messages: object) -> int:
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


def _normalize_rollout_json_dir(value: str | None) -> Path | None:
    if not value:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return Path(normalized).expanduser().resolve()


def _set_rollout_json_dir(value: str | None) -> None:
    global _ROLL_OUT_JSON_DIR
    _ROLL_OUT_JSON_DIR = _normalize_rollout_json_dir(value)


def _sanitize_path_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or fallback


def _read_text_tail(
    path: Path, max_lines: int = 40, max_chars: int = 4000
) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if not text:
        return None
    tail = "\n".join(text.splitlines()[-max_lines:]).strip()
    if not tail:
        return None
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _safe_load_json_file(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_server_trial_profile_record(
    trials_dir: Path, unique_sid: str
) -> dict[str, object] | None:
    trial_dir = trials_dir / unique_sid
    result_blob = _safe_load_json_file(trial_dir / "result.json")
    if not isinstance(result_blob, dict):
        return None
    try:
        trial_result = TrialResult.model_validate(result_blob)
    except Exception as e:
        logger.debug(
            f"[{unique_sid}] Failed to parse trial result for profiling timeline: {e}"
        )
        return None
    try:
        record = build_trial_profile_record(trial_result)
    except Exception as e:
        logger.debug(
            f"[{unique_sid}] Failed to build profiling timeline record: {e}"
        )
        return None
    record["server_sid"] = unique_sid
    return record


def _ensure_server_profile_html_header(html_path: Path) -> None:
    if html_path.exists() and html_path.stat().st_size > 0:
        return
    legend_items = "".join(
        (
            '<span class="legend-item">'
            f'<span class="swatch" style="background:{color}"></span>{escape(category)}'
            "</span>"
        )
        for category, color in _CATEGORY_COLORS.items()
    )
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Trial Profiling Timeline</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; color: #111827; }}
    .legend {{ display: flex; gap: 16px; margin: 8px 0 16px; flex-wrap: wrap; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}
    .meta {{ color: #4b5563; font-size: 13px; margin-bottom: 12px; }}
    .trial-profile {{ border-top: 1px solid #e5e7eb; padding-top: 16px; margin-top: 16px; }}
    .trial-profile h2 {{ font-size: 16px; margin: 0 0 8px; }}
    .chart {{ border: 1px solid #e5e7eb; overflow: auto; max-width: 100%; }}
  </style>
</head>
<body>
  <h1>Trial Profiling Timeline</h1>
  <div class="meta">Append-only view. Each section is appended when one trial finishes.</div>
  <div class="legend">{legend_items}</div>
""",
        encoding="utf-8",
    )


def _append_server_profile_html_record(html_path: Path, record: dict[str, object]) -> None:
    trial_name = escape(str(record.get("trial_name") or record.get("server_sid") or "trial"))
    started_at = escape(str(record.get("started_at") or ""))
    duration_sec = escape(str(record.get("duration_sec") or ""))
    svg = _render_svg([record])
    with html_path.open("a", encoding="utf-8") as f:
        f.write(
            f"""  <section class="trial-profile">
    <h2>{trial_name}</h2>
    <div class="meta">started_at={started_at} duration_sec={duration_sec}</div>
    <div class="chart">{svg}</div>
  </section>
"""
        )


def _write_server_profile_artifacts(record: dict[str, object]) -> None:
    if not ENABLE_PROFILING:
        return
    output_dir = Path(EVAL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "profiling_timeline.jsonl"
    html_path = output_dir / "profiling_timeline.html"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")
    _ensure_server_profile_html_header(html_path)
    _append_server_profile_html_record(html_path, record)


async def _persist_server_profile_artifacts(
    trials_dir: Path, unique_sid: str
) -> None:
    if not ENABLE_PROFILING:
        return
    record = await asyncio.to_thread(
        _load_server_trial_profile_record,
        trials_dir,
        unique_sid,
    )
    if not isinstance(record, dict):
        return
    if _profiling_artifacts_lock is None:
        await asyncio.to_thread(_write_server_profile_artifacts, record)
        return
    async with _profiling_artifacts_lock:
        await asyncio.to_thread(_write_server_profile_artifacts, record)


def _build_trial_diagnostics(
    trials_dir: Path,
    unique_sid: str,
    *,
    fallback_type: str | None = None,
    error_message: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, object]:
    trial_dir = trials_dir / unique_sid
    diagnostics: dict[str, object] = {
        "diagnostics_version": 1,
        "trial_dir": str(trial_dir),
    }
    if fallback_type:
        diagnostics["fallback_type"] = fallback_type
    if error_message:
        diagnostics["fallback_error_message"] = error_message

    progress = _safe_load_json_file(trial_dir / "progress.json")
    if progress is not None:
        diagnostics["progress"] = progress
        stage = progress.get("stage")
        if isinstance(stage, str) and stage.strip():
            diagnostics["timeout_stage"] = stage.strip()
        updated_at = progress.get("updated_at")
        if isinstance(updated_at, str) and updated_at.strip():
            diagnostics["timeout_stage_updated_at"] = updated_at.strip()
        detail = progress.get("detail")
        if isinstance(detail, str) and detail.strip():
            diagnostics["timeout_stage_detail"] = detail.strip()

    result = _safe_load_json_file(trial_dir / "result.json")
    if result is not None:
        diagnostics["partial_result"] = result

    exception_tail = _read_text_tail(trial_dir / "exception.txt", max_lines=60)
    if exception_tail:
        diagnostics["timeout_exception_tail"] = exception_tail

    trial_log_tail = _read_text_tail(trial_dir / "trial.log", max_lines=80)
    if trial_log_tail:
        diagnostics["timeout_trial_log_tail"] = trial_log_tail

    verifier_stdout_tail = _read_text_tail(
        trial_dir / "verifier" / "test-stdout.txt", max_lines=60
    )
    if verifier_stdout_tail:
        diagnostics["timeout_verifier_stdout_tail"] = verifier_stdout_tail

    verifier_stderr_tail = _read_text_tail(
        trial_dir / "verifier" / "test-stderr.txt", max_lines=60
    )
    if verifier_stderr_tail:
        diagnostics["timeout_verifier_stderr_tail"] = verifier_stderr_tail

    trajectory_tail = _read_text_tail(
        trial_dir / "agent" / "trajectory.json", max_lines=40, max_chars=3000
    )
    if trajectory_tail:
        diagnostics["timeout_agent_trajectory_tail"] = trajectory_tail

    stdout_tail = None
    if isinstance(stdout, str) and stdout.strip():
        stdout_tail = "\n".join(stdout.splitlines()[-40:]).strip()
        if len(stdout_tail) > 4000:
            stdout_tail = stdout_tail[-4000:]
    if stdout_tail:
        diagnostics["fallback_stdout_tail"] = stdout_tail

    stderr_tail = None
    if isinstance(stderr, str) and stderr.strip():
        stderr_tail = "\n".join(stderr.splitlines()[-40:]).strip()
        if len(stderr_tail) > 4000:
            stderr_tail = stderr_tail[-4000:]
    if stderr_tail:
        diagnostics["fallback_stderr_tail"] = stderr_tail

    reason_parts = []
    if fallback_type:
        reason_parts.append(f"fallback={fallback_type}")
    timeout_stage = diagnostics.get("timeout_stage")
    if isinstance(timeout_stage, str) and timeout_stage:
        reason_parts.append(f"stage={timeout_stage}")
    timeout_stage_detail = diagnostics.get("timeout_stage_detail")
    if isinstance(timeout_stage_detail, str) and timeout_stage_detail:
        reason_parts.append(f"detail={timeout_stage_detail}")
    if error_message:
        reason_parts.append(f"error={error_message}")
    diagnostics["timeout_error_summary"] = "; ".join(reason_parts)

    return diagnostics


def _is_dummy_message_sequence(messages: object) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    first_message = messages[0]
    last_message = messages[-1]
    if not isinstance(first_message, dict) or not isinstance(last_message, dict):
        return False
    first_content = first_message.get("content")
    last_content = last_message.get("content")
    if not isinstance(first_content, str) or not isinstance(last_content, str):
        return False
    return (
        first_message.get("role") == "user"
        and last_message.get("role") == "assistant"
        and first_content.strip() == "dummy"
        and last_content.strip() == "dummy"
    )


def _is_dummy_rollout_payload(result: dict) -> bool:
    messages_list = result.get("messages_list")
    if isinstance(messages_list, list):
        normalized_messages_list = [
            item for item in messages_list if isinstance(item, list) and item
        ]
        if normalized_messages_list:
            return all(_is_dummy_message_sequence(item) for item in normalized_messages_list)
    return _is_dummy_message_sequence(result.get("messages"))


def _coerce_string_list(
    value: object,
    *,
    expected_len: int | None = None,
    fallback_prefix: str | None = None,
) -> list[str]:
    if isinstance(value, list):
        items = [str(item or "").strip() for item in value]
    else:
        items = []
    if expected_len is None:
        return items
    if len(items) < expected_len:
        for idx in range(len(items), expected_len):
            if fallback_prefix:
                items.append(fallback_prefix if idx == 0 else f"{fallback_prefix}_cm_{idx}")
            else:
                items.append("")
    elif len(items) > expected_len:
        items = items[:expected_len]
    return items


def _merge_result_metadata(result: dict, extra_metadata: dict[str, object]) -> None:
    metadata = _ensure_result_metadata(result)
    metadata.update(extra_metadata)


def _is_completed_non_dummy_assistant_message(message: object) -> bool:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    content = message.get("content")
    if isinstance(content, str) and content.strip() and content.strip() != "dummy":
        return True
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return True
    tool_calls = message.get("tool_calls")
    return isinstance(tool_calls, list) and len(tool_calls) > 0


def _trim_messages_to_last_completed_assistant(messages: object) -> list[dict]:
    if not isinstance(messages, list):
        return []
    last_completed_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if _is_completed_non_dummy_assistant_message(messages[idx]):
            last_completed_assistant_idx = idx
            break
    if last_completed_assistant_idx is None:
        return []
    trimmed_messages = [
        dict(message) for message in messages[: last_completed_assistant_idx + 1]
        if isinstance(message, dict)
    ]
    return trimmed_messages


def _dedupe_adjacent_message_snapshots(
    messages_list: list[list[dict]],
) -> list[list[dict]]:
    deduped: list[list[dict]] = []
    previous_snapshot: list[dict] | None = None
    for snapshot in messages_list:
        if previous_snapshot is not None and snapshot == previous_snapshot:
            continue
        deduped.append(snapshot)
        previous_snapshot = snapshot
    return deduped


def _trim_and_filter_message_snapshots(messages_list: object) -> list[list[dict]]:
    if not isinstance(messages_list, list):
        return []
    trimmed_snapshots = []
    for snapshot in messages_list:
        trimmed_snapshot = _trim_messages_to_last_completed_assistant(snapshot)
        if trimmed_snapshot:
            trimmed_snapshots.append(trimmed_snapshot)
    return _dedupe_adjacent_message_snapshots(trimmed_snapshots)


def _has_any_completed_non_dummy_assistant(*message_groups: object) -> bool:
    for group in message_groups:
        if isinstance(group, list) and group and all(isinstance(item, dict) for item in group):
            if any(_is_completed_non_dummy_assistant_message(item) for item in group):
                return True
        elif isinstance(group, list):
            for snapshot in group:
                if (
                    isinstance(snapshot, list)
                    and any(
                        _is_completed_non_dummy_assistant_message(item)
                        for item in snapshot
                        if isinstance(item, dict)
                    )
                ):
                    return True
    return False


def _recover_timeout_rollout_payload(
    trials_dir: Path,
    unique_sid: str,
    instance_id: str,
    uid: str | None,
) -> dict[str, object]:
    uid_prefix = uid or unique_sid
    recovered_metadata: dict[str, object] = {
        "data_id": uid_prefix,
        "base_uid": uid_prefix,
        "context_management_index": -1,
        "context_management_count": 0,
        "context_management_limit_hit": False,
        "messages_list_uids": [uid_prefix],
    }
    recovered: dict[str, object] = {
        "instance_id": instance_id,
        "sid": unique_sid,
        "uid": uid_prefix,
        "messages": dummy_messages,
        "managed_messages_list": [],
        "messages_list": [dummy_messages],
        "messages_list_tags": ["main_chat"],
        "messages_list_uids": [uid_prefix],
        "messages_list_raw_rewards": [dummy_reward],
        "raw_reward": dummy_reward,
        "overturn": False,
        "overlong": False,
        "format_error": False,
        "aborted": False,
        "rollout_status": "timeout",
        "weight_versions": _extract_weight_versions(dummy_messages),
        "managed_messages_list_weight_versions": [],
        "messages_list_weight_versions": [_extract_weight_versions(dummy_messages)],
        "metadata": recovered_metadata,
        "meta_data": recovered_metadata,
    }
    try:
        from trial_worker import (
            _assemble_export_messages_list,
            _prune_incomplete_trailing_main_chat,
            _select_final_uid,
            extract_messages_from_trial,
        )
        from types import SimpleNamespace

        trial_dir = trials_dir / unique_sid
        result_blob = _safe_load_json_file(trial_dir / "result.json")
        agent_metadata: dict[str, object] | None = None
        if isinstance(result_blob, dict):
            agent_result_blob = result_blob.get("agent_result")
            if isinstance(agent_result_blob, dict):
                maybe_metadata = agent_result_blob.get("metadata")
                if isinstance(maybe_metadata, dict):
                    agent_metadata = maybe_metadata
        result_system_prompt = None
        if isinstance(agent_metadata, dict):
            candidate_system_prompt = agent_metadata.get("system_prompt")
            if isinstance(candidate_system_prompt, str) and candidate_system_prompt.strip():
                result_system_prompt = candidate_system_prompt
        agent_messages_tito_keys: list[str] = []
        explicit_messages_list_uids: list[str] = []
        post_context_management_generate_pending = False
        summary_training_mode = "main_chat"
        if isinstance(agent_metadata, dict):
            raw_main_keys = agent_metadata.get("messages_list_tito_keys")
            if isinstance(raw_main_keys, list):
                agent_messages_tito_keys = [str(item) for item in raw_main_keys if item]
            raw_explicit_uids = agent_metadata.get("messages_list_uids")
            if isinstance(raw_explicit_uids, list):
                explicit_messages_list_uids = [str(item) for item in raw_explicit_uids if item]
            post_context_management_generate_pending = _coerce_bool(
                agent_metadata.get("post_context_management_generate_pending"),
                default=False,
            )
            raw_summary_training_mode = agent_metadata.get("summary_training")
            if (
                isinstance(raw_summary_training_mode, str)
                and raw_summary_training_mode.strip()
            ):
                summary_training_mode = raw_summary_training_mode.strip().lower()

        managed_messages_list, rollout_messages_list, final_messages, summary_messages_list = (
            extract_messages_from_trial(
                SimpleNamespace(
                    agent_result=SimpleNamespace(metadata=agent_metadata)
                    if isinstance(agent_metadata, dict)
                    else None
                ),
                trial_dir,
                system_prompt=result_system_prompt,
            )
        )
        managed_messages_list = _trim_and_filter_message_snapshots(managed_messages_list)
        rollout_messages_list = _trim_and_filter_message_snapshots(rollout_messages_list)
        final_messages = _trim_messages_to_last_completed_assistant(final_messages)
        rollout_messages_list, final_messages = _prune_incomplete_trailing_main_chat(
            managed_messages_list,
            rollout_messages_list,
            final_messages,
            post_context_management_generate_pending,
        )
        managed_messages_list = _trim_and_filter_message_snapshots(managed_messages_list)
        rollout_messages_list = _trim_and_filter_message_snapshots(rollout_messages_list)
        final_messages = _trim_messages_to_last_completed_assistant(final_messages)
        if not final_messages and rollout_messages_list:
            final_messages = rollout_messages_list[-1]

        if _has_any_completed_non_dummy_assistant(
            managed_messages_list, rollout_messages_list, final_messages
        ):
            messages_list, messages_list_tags, messages_list_uids = (
                _assemble_export_messages_list(
                    rollout_messages_list=rollout_messages_list,
                    final_messages=final_messages,
                    summary_messages_list=summary_messages_list,
                    uid_prefix=uid_prefix,
                    agent_messages_tito_keys=(
                        agent_messages_tito_keys or explicit_messages_list_uids
                    ),
                    agent_summary_tito_keys=[],
                    summary_training_mode=summary_training_mode,
                )
            )
            messages_list = _trim_and_filter_message_snapshots(messages_list)
            if not messages_list and final_messages:
                messages_list = [final_messages]
                messages_list_tags = ["main_chat"]
                messages_list_uids = [uid_prefix]
            if messages_list:
                final_messages = messages_list[-1]
            messages_list_tags = _coerce_string_list(
                messages_list_tags,
                expected_len=len(messages_list),
            )
            if not any(messages_list_tags):
                messages_list_tags = ["main_chat"] * len(messages_list)
            if not agent_messages_tito_keys and explicit_messages_list_uids:
                messages_list_uids = _coerce_string_list(
                    explicit_messages_list_uids,
                    expected_len=len(messages_list),
                    fallback_prefix=uid_prefix,
                )
            else:
                messages_list_uids = _coerce_string_list(
                    messages_list_uids,
                    expected_len=len(messages_list),
                    fallback_prefix=uid_prefix,
                )
            messages_list_uids = [
                item if item else (uid_prefix if idx == 0 else f"{uid_prefix}_cm_{idx}")
                for idx, item in enumerate(messages_list_uids)
            ]
            messages_list_weight_versions = [
                _extract_weight_versions(item) for item in messages_list
            ]
            managed_messages_list_weight_versions = [
                _extract_weight_versions(item) for item in managed_messages_list
            ]
            context_management_count = len(managed_messages_list)
            context_management_limit_hit = bool(
                agent_metadata.get("context_management_limit_hit")
            ) if isinstance(agent_metadata, dict) else False
            final_main_chat_uid = _select_final_uid(
                uid_prefix=uid_prefix,
                messages_list_tags=messages_list_tags,
                messages_list_uids=messages_list_uids,
            )
            recovered["messages"] = final_messages
            recovered["managed_messages_list"] = managed_messages_list
            recovered["messages_list"] = messages_list
            recovered["messages_list_tags"] = messages_list_tags
            recovered["messages_list_uids"] = messages_list_uids
            recovered["messages_list_raw_rewards"] = [0] * len(messages_list)
            recovered["raw_reward"] = 0
            recovered["weight_versions"] = _extract_weight_versions(final_messages)
            recovered["managed_messages_list_weight_versions"] = (
                managed_messages_list_weight_versions
            )
            recovered["messages_list_weight_versions"] = messages_list_weight_versions
            if isinstance(result_system_prompt, str) and result_system_prompt:
                recovered["system_prompt"] = result_system_prompt
                recovered_metadata["system_prompt"] = result_system_prompt
            if isinstance(agent_metadata, dict):
                for key in (
                    "summary_training",
                    "context_management_max_count",
                    "rollout_status_detail",
                ):
                    if key in agent_metadata:
                        recovered_metadata[key] = agent_metadata[key]
            recovered_metadata.update(
                {
                    "data_id": uid_prefix,
                    "base_uid": uid_prefix,
                    "final_main_chat_uid": final_main_chat_uid,
                    "context_management_index": (
                        context_management_count if context_management_count > 0 else -1
                    ),
                    "context_management_count": context_management_count,
                    "context_management_limit_hit": context_management_limit_hit,
                    "messages_list_uids": list(messages_list_uids),
                }
            )
            recovered["split_rollout_results"] = _build_split_rollout_results(recovered)
            recovered_metadata["split_rollout_results_count"] = len(
                recovered["split_rollout_results"]
            )
    except Exception:
        pass
    return recovered


def _format_timestamp(timestamp: datetime.datetime | None = None) -> str:
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _coerce_int(value: object, default: int | None = 0) -> int | None:
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
    value: object, default: str = "main_chat"
) -> str:
    if isinstance(value, bool):
        return "full" if value else "main_chat"
    if isinstance(value, (int, float)):
        return "full" if bool(value) else "main_chat"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return "full"
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return "main_chat"
        if normalized in {"main_chat", "summary", "full"}:
            return normalized
    return default


def _normalize_context_management_mode(
    value: object, default: str = "none"
) -> str:
    if isinstance(value, bool):
        return "auto" if value else "none"
    if isinstance(value, (int, float)):
        return "auto" if bool(value) else "none"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return "auto"
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return "none"
        if normalized in {"none", "auto", "tool"}:
            return normalized
    return default


def _ensure_server_start_ts() -> None:
    global _SERVER_START_TS
    if _SERVER_START_TS is None:
        _SERVER_START_TS = _format_timestamp()


def _persist_rollout_result(
    result: dict, exception_path: Path | None = None
) -> None:
    if _ROLL_OUT_JSON_DIR is None:
        return
    try:
        _ensure_server_start_ts()
        rollout_status = result.get("rollout_status") or _rollout_status_from_flags(result)
        instance_id = str(result.get("instance_id") or "unknown")
        sid = str(result.get("sid") or uuid.uuid4())

        status_dir = _sanitize_path_component(str(rollout_status), "unknown_status")
        instance_dir = _sanitize_path_component(instance_id, "unknown_instance")
        sid_name = _sanitize_path_component(sid, "unknown_sid")
        result_ts = _sanitize_path_component(_format_timestamp(), "unknown_time")
        sid_dir = f"{sid_name}_{result_ts}"

        output_dir = (
            _ROLL_OUT_JSON_DIR / str(_SERVER_START_TS) / status_dir / instance_dir / sid_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "result.json"
        if orjson is not None:
            with output_path.open("wb") as f:
                f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2))
        else:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        if exception_path is not None and exception_path.exists():
            exception_target = output_dir / "exception.txt"
            exception_target.write_bytes(exception_path.read_bytes())
    except Exception as e:
        log_info(f"[{result.get('instance_id', 'unknown')}] Failed to persist rollout result: {e}")
        logger.warning(f"Failed to persist rollout result: {e}")


_set_rollout_json_dir(os.environ.get(_ROLL_OUT_JSON_DIR_ENV))


def _cleanup_worker_containers(unique_sid: str) -> None:
    project_name = unique_sid.lower().replace(".", "-")
    filter_value = f"label=com.docker.compose.project={project_name}"
    command = (
        "docker ps -a -q --filter "
        f"{shlex.quote(filter_value)}"
        " | xargs -r docker rm -f"
    )
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        log_info(f"[{unique_sid}] Docker not found; skipping container cleanup.")
        return

    removed_ids = [
        line.strip() for line in (result.stdout or "").splitlines() if line.strip()
    ]
    stderr_lines = [
        line.strip()
        for line in (result.stderr or "").splitlines()
        if line.strip()
    ]
    non_missing_lines = []
    for line in stderr_lines:
        lowered = line.lower()
        if "no such container" in lowered or "no such object" in lowered:
            continue
        non_missing_lines.append(line)

    if result.returncode != 0 and non_missing_lines:
        log_info(f"[{unique_sid}] Docker cleanup failed: {'; '.join(non_missing_lines)}")
        return

    if removed_ids:
        log_info(
            f"[{unique_sid}] Removed {len(removed_ids)} container(s) for project {project_name}"
        )


def _rollout_status_from_flags(result: dict) -> str:
    for key in ("singleturn_overlong", "overlong", "overturn", "format_error"):
        if result.get(key):
            return key
    if result.get("rollout_status") == "timeout":
        return "timeout"
    detail = result.get("rollout_status_detail")
    if result.get("pager_usage"):
        return "pager_usage"
    if isinstance(detail, dict) and (
        detail.get("pager_usage")
        or detail.get("exception_type") == "PagerUsageError"
    ):
        return "pager_usage"
    if result.get("rollout_status") == "pager_usage":
        return "pager_usage"
    if result.get("aborted"):
        return "aborted"
    return "completed"


def _extract_weight_versions(messages: list[dict]) -> list:
    return [
        msg.get("weight_version")
        for msg in messages
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    ]


def _annotate_message_content_origin(message: dict) -> dict:
    msg_copy = dict(message)
    if "loss_mask" in msg_copy:
        msg_copy["loss_mask"] = bool(msg_copy["loss_mask"])
    elif "_loss_mask" in msg_copy:
        msg_copy["loss_mask"] = bool(msg_copy["_loss_mask"])
    msg_copy.pop("_loss_mask", None)
    msg_copy["num_tokens"] = _normalize_num_tokens(msg_copy.get("num_tokens"))

    if "content_origin" in msg_copy:
        msg_copy.pop("history_content", None)
        return msg_copy

    history_content = msg_copy.get("history_content")
    if isinstance(history_content, str):
        msg_copy["content_origin"] = history_content
        msg_copy.pop("history_content", None)
        return msg_copy

    content = msg_copy.get("content")
    msg_copy["content_origin"] = content if isinstance(content, str) else None
    msg_copy.pop("history_content", None)
    return msg_copy


def _normalize_messages_content_origin(messages: object) -> object:
    if not isinstance(messages, list):
        return messages

    normalized_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            normalized_messages.append(msg)
            continue
        normalized_messages.append(_annotate_message_content_origin(msg))
    return normalized_messages


def _normalize_rollout_result_content_origin(result: dict) -> dict:
    result = dict(result)

    if "messages" in result:
        result["messages"] = _normalize_messages_content_origin(result.get("messages"))

    if isinstance(result.get("managed_messages_list"), list):
        result["managed_messages_list"] = [
            _normalize_messages_content_origin(messages)
            for messages in result["managed_messages_list"]
        ]

    if isinstance(result.get("messages_list"), list):
        result["messages_list"] = [
            _normalize_messages_content_origin(messages)
            for messages in result["messages_list"]
        ]

    if isinstance(result.get("summary_messages_list"), list):
        result["summary_messages_list"] = [
            _normalize_messages_content_origin(messages)
            for messages in result["summary_messages_list"]
        ]

    if isinstance(result.get("split_rollout_results"), list):
        normalized_split_results = []
        for item in result["split_rollout_results"]:
            if not isinstance(item, dict):
                normalized_split_results.append(item)
                continue
            normalized_split_results.append(
                _normalize_rollout_result_content_origin(item)
            )
        result["split_rollout_results"] = normalized_split_results

    return result


def _attach_result_token_metadata(result: dict) -> dict:
    result = dict(result)
    messages = result.get("messages")
    if isinstance(messages, list):
        metadata = _ensure_result_metadata(result)
        metadata["assistant_tokens"] = _compute_assistant_tokens(messages)
        metadata.setdefault("later_assistant_tokens", 0)

    split_rollout_results = result.get("split_rollout_results")
    if isinstance(split_rollout_results, list):
        normalized_split_results = []
        for item in split_rollout_results:
            if isinstance(item, dict):
                normalized_split_results.append(_attach_result_token_metadata(item))
            else:
                normalized_split_results.append(item)
        result["split_rollout_results"] = normalized_split_results

    return result


def _build_split_rollout_results(result: dict) -> list[dict]:
    messages_list = result.get("messages_list")
    if not isinstance(messages_list, list):
        messages_list = []
    messages_list = [item for item in messages_list if isinstance(item, list) and item]
    if not messages_list:
        return []

    tags = result.get("messages_list_tags")
    if not isinstance(tags, list):
        tags = ["main_chat"] * len(messages_list)
    rewards = result.get("messages_list_raw_rewards")
    if not isinstance(rewards, list):
        rewards = [result.get("raw_reward", dummy_reward)] * len(messages_list)
    uids = result.get("messages_list_uids")
    if not isinstance(uids, list):
        uids = []
    weight_versions_list = result.get("messages_list_weight_versions")
    if not isinstance(weight_versions_list, list):
        weight_versions_list = [[] for _ in messages_list]

    split_results = []
    total = len(messages_list)
    uid_prefix = str(
        result.get("uid")
        or result.get("source_uid")
        or result.get("sid")
        or result.get("instance_id")
        or "unknown_uid"
    )
    data_id = str(
        result.get("metadata", {}).get("data_id")
        or result.get("meta_data", {}).get("data_id")
        or uid_prefix
    )
    context_management_count = result.get("metadata", {}).get(
        "context_management_count"
    )
    if not isinstance(context_management_count, int):
        managed_messages_list = result.get("managed_messages_list")
        if isinstance(managed_messages_list, list):
            context_management_count = len(managed_messages_list)
        else:
            context_management_count = 0
    context_management_limit_hit = bool(
        result.get("metadata", {}).get("context_management_limit_hit")
        or result.get("meta_data", {}).get("context_management_limit_hit")
        or result.get("rollout_status_detail", {}).get("context_management_limit_hit")
    )
    system_prompt = result.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt:
        metadata_payload = result.get("metadata", {})
        if isinstance(metadata_payload, dict):
            candidate = metadata_payload.get("system_prompt")
            if isinstance(candidate, str) and candidate:
                system_prompt = candidate
            else:
                system_prompt = None
        else:
            system_prompt = None
    cm_index = 0
    summary_index = 0
    for idx, messages in enumerate(messages_list):
        tag = str(tags[idx]) if idx < len(tags) else "main_chat"
        is_summary = tag.strip().lower() == "summary"
        export_system_prompt = (
            system_prompt if not is_summary and isinstance(system_prompt, str) and system_prompt else None
        )
        if export_system_prompt:
            normalized_messages = [
                dict(msg) for msg in messages if isinstance(msg, dict)
            ]
            if normalized_messages and normalized_messages[0].get("role") == "system":
                normalized_messages[0]["content"] = export_system_prompt
            else:
                normalized_messages = [
                    {"role": "system", "content": export_system_prompt, "num_tokens": 0},
                    *normalized_messages,
                ]
            messages = normalized_messages
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
        if idx < len(uids):
            uid_candidate = str(uids[idx] or "").strip()
            if uid_candidate:
                split_uid = uid_candidate
        raw_reward = result.get("raw_reward", dummy_reward)
        if idx < len(rewards):
            try:
                raw_reward = float(rewards[idx])
            except (TypeError, ValueError):
                raw_reward = result.get("raw_reward", dummy_reward)
        weight_versions = []
        if idx < len(weight_versions_list) and isinstance(weight_versions_list[idx], list):
            weight_versions = weight_versions_list[idx]
        if is_summary:
            current_context_management_index = max(1, summary_index)
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
        if export_system_prompt:
            split_metadata["system_prompt"] = export_system_prompt
        split_metadata["assistant_tokens"] = _compute_assistant_tokens(messages)
        split_metadata["later_assistant_tokens"] = 0
        split_result = {
            "instance_id": (
                f"{result.get('instance_id') or data_id}_{uuid.uuid4()}"
            ),
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


def _ensure_result_metadata(result: dict) -> dict:
    metadata = result.get("metadata")
    meta_data = result.get("meta_data")

    if isinstance(metadata, dict):
        if isinstance(meta_data, dict) and meta_data is not metadata:
            for key, value in meta_data.items():
                metadata.setdefault(key, value)
        result["meta_data"] = metadata
        return metadata

    if isinstance(meta_data, dict):
        result["metadata"] = meta_data
        return meta_data

    metadata = {}
    result["metadata"] = metadata
    result["meta_data"] = metadata
    return metadata


def _derive_metadata_error_reason(result: dict) -> str | None:
    rollout_status = result.get("rollout_status")
    detail = result.get("rollout_status_detail")
    exception_type = None
    exception_message = None
    if isinstance(detail, dict):
        raw_exception_type = detail.get("exception_type")
        raw_exception_message = detail.get("exception_message")
        if isinstance(raw_exception_type, str):
            exception_type = raw_exception_type.strip() or None
        if isinstance(raw_exception_message, str):
            exception_message = raw_exception_message.strip() or None

    top_level_error = result.get("error")
    if isinstance(top_level_error, str):
        top_level_error = top_level_error.strip() or None
    else:
        top_level_error = None

    if exception_type and exception_message:
        return f"{exception_type}: {exception_message}"
    if exception_message:
        return exception_message
    if top_level_error:
        return top_level_error
    if exception_type:
        return exception_type
    if rollout_status in {
        "aborted",
        "format_error",
        "overlong",
        "singleturn_overlong",
        "overturn",
    }:
        return f"rollout_status={rollout_status}"
    return None


def _build_forced_context_management_result(
    instance_id: str,
    sid: str,
    force_context_management_count: int,
) -> dict:
    managed_messages_list: list[list[dict]] = []
    for idx in range(force_context_management_count):
        managed_messages_list.append(
            [
                {
                    "role": "user",
                    "content": (
                        "Forced context-management test input. "
                        f"snapshot={idx + 1}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Forced context-management snapshot before manage. "
                        f"index={idx + 1}"
                    ),
                    "reasoning_content": "",
                    "weight_version": idx + 1,
                },
            ]
        )

    final_messages = [
        {
            "role": "user",
            "content": "Forced context-management test input. final",
        },
        {
            "role": "assistant",
            "content": (
                "Forced final assistant response after context management. "
                f"count={force_context_management_count}"
            ),
            "reasoning_content": "",
            "weight_version": force_context_management_count,
        },
    ]
    messages_list = [*managed_messages_list, final_messages]
    rollout_metadata = {
        "context_truncation_count": force_context_management_count,
        "summarization_count": force_context_management_count,
        "n_episodes": 1,
        "data_id": sid,
        "context_management_index": (
            force_context_management_count if force_context_management_count > 0 else -1
        ),
        "context_management_limit_hit": False,
        "messages_list_uids": [
            sid if idx == 0 else f"{sid}_cm_{idx}"
            for idx in range(len(messages_list))
        ],
    }

    return {
        "instance_id": instance_id,
        "sid": sid,
        "messages": final_messages,
        "managed_messages_list": managed_messages_list,
        "messages_list": messages_list,
        "messages_list_uids": list(rollout_metadata["messages_list_uids"]),
        "messages_list_raw_rewards": [1] * len(messages_list),
        "raw_reward": 1,
        "overturn": False,
        "overlong": False,
        "format_error": False,
        "aborted": False,
        "rollout_status": "completed",
        "rollout_status_detail": {
            "exception_type": None,
            "exception_message": None,
            "n_episodes": 1,
            "max_iterations": 1,
            "task_complete": True,
            "timeout": False,
            "context_length_exceeded_count": force_context_management_count,
            "output_length_exceeded_count": 0,
            "summarization_count": force_context_management_count,
            "context_management_limit_hit": False,
        },
        "weight_versions": _extract_weight_versions(final_messages),
        "managed_messages_list_weight_versions": [
            _extract_weight_versions(item) for item in managed_messages_list
        ],
        "messages_list_weight_versions": [
            _extract_weight_versions(item) for item in messages_list
        ],
        "metadata": rollout_metadata,
        "meta_data": rollout_metadata,
    }


def _apply_rollout_status(result: dict) -> dict:
    if "raw_reward" not in result or result["raw_reward"] is None:
        log_info(f"[{result.get('instance_id', 'unknown')}] raw_reward missing or None, setting to {dummy_reward} and marking aborted")
        result["raw_reward"] = dummy_reward
        result["aborted"] = True
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        log_info(f"[{result.get('instance_id', 'unknown')}] messages empty or invalid (type={type(messages).__name__}), setting dummy messages and raw_reward={dummy_reward}")
        result["messages"] = dummy_messages
        result["raw_reward"] = dummy_reward
        result["aborted"] = True
        messages = result["messages"]
    messages_list = result.get("messages_list")
    if not isinstance(messages_list, list):
        messages_list = [messages]
    else:
        messages_list = [
            item for item in messages_list if isinstance(item, list) and item
        ]
        if not messages_list:
            messages_list = [messages]
    result["messages_list"] = messages_list

    metadata = _ensure_result_metadata(result)
    messages_list_tags = result.get("messages_list_tags")
    if not isinstance(messages_list_tags, list):
        messages_list_tags = ["main_chat"] * len(messages_list)
    else:
        messages_list_tags = [str(tag) for tag in messages_list_tags]
        if len(messages_list_tags) < len(messages_list):
            messages_list_tags.extend(
                ["main_chat"] * (len(messages_list) - len(messages_list_tags))
            )
        elif len(messages_list_tags) > len(messages_list):
            messages_list_tags = messages_list_tags[: len(messages_list)]

    messages_list_raw_rewards = result.get("messages_list_raw_rewards")
    if not isinstance(messages_list_raw_rewards, list):
        messages_list_raw_rewards = [result["raw_reward"]] * len(messages_list)
    else:
        normalized_rewards = []
        for value in messages_list_raw_rewards:
            try:
                normalized_rewards.append(float(value))
            except (TypeError, ValueError):
                normalized_rewards.append(result["raw_reward"])
        messages_list_raw_rewards = normalized_rewards
        if len(messages_list_raw_rewards) < len(messages_list):
            messages_list_raw_rewards.extend(
                [result["raw_reward"]]
                * (len(messages_list) - len(messages_list_raw_rewards))
            )
        elif len(messages_list_raw_rewards) > len(messages_list):
            messages_list_raw_rewards = messages_list_raw_rewards[: len(messages_list)]
    result["messages_list_raw_rewards"] = messages_list_raw_rewards

    messages_list_uids = result.get("messages_list_uids")
    if not isinstance(messages_list_uids, list):
        messages_list_uids = []
    else:
        messages_list_uids = [
            str(uid) if uid is not None else "" for uid in messages_list_uids
        ]
    fallback_uid = str(result.get("uid") or "")
    if len(messages_list_uids) < len(messages_list):
        messages_list_uids.extend(
            [fallback_uid] * (len(messages_list) - len(messages_list_uids))
        )
    elif len(messages_list_uids) > len(messages_list):
        messages_list_uids = messages_list_uids[: len(messages_list)]
    messages_list_uids = [
        uid if uid else fallback_uid for uid in messages_list_uids
    ]
    result["messages_list_uids"] = messages_list_uids
    metadata["messages_list_uids"] = messages_list_uids

    managed_messages_list = result.get("managed_messages_list")
    if not isinstance(managed_messages_list, list):
        managed_messages_list = messages_list[:-1] if len(messages_list) > 1 else []
    else:
        managed_messages_list = [
            item for item in managed_messages_list if isinstance(item, list) and item
        ]
    result["managed_messages_list"] = managed_messages_list

    if "weight_versions" not in result:
        result["weight_versions"] = _extract_weight_versions(messages)
    if "managed_messages_list_weight_versions" not in result:
        result["managed_messages_list_weight_versions"] = [
            _extract_weight_versions(item) for item in managed_messages_list
        ]
    if "messages_list_weight_versions" not in result:
        result["messages_list_weight_versions"] = [
            _extract_weight_versions(item) for item in messages_list
        ]

    rollout_status = result.get("rollout_status")
    if isinstance(rollout_status, str):
        rollout_status = rollout_status.strip()
    flag_status = _rollout_status_from_flags(result)
    valid_statuses = {
        "completed",
        "format_error",
        "overturn",
        "overlong",
        "singleturn_overlong",
        "aborted",
        "timeout",
        "pager_usage",
    }
    if not isinstance(rollout_status, str) or rollout_status not in valid_statuses:
        rollout_status = flag_status
    elif flag_status != "completed" and flag_status != rollout_status:
        rollout_status = flag_status
    result["rollout_status"] = rollout_status

    for key in (
        "format_error",
        "overturn",
        "overlong",
        "singleturn_overlong",
        "aborted",
        "pager_usage",
    ):
        result[key] = rollout_status == key

    metadata["rollout_status"] = rollout_status
    metadata.setdefault("context_management_count", len(managed_messages_list))
    metadata.setdefault(
        "data_id",
        str(result.get("uid") or result.get("sid") or result.get("instance_id") or ""),
    )
    metadata.setdefault(
        "context_management_index",
        metadata["context_management_count"]
        if metadata["context_management_count"] > 0
        else -1,
    )
    metadata.setdefault(
        "context_management_limit_hit",
        bool(result.get("rollout_status_detail", {}).get("context_management_limit_hit")),
    )
    if isinstance(result.get("system_prompt"), str) and result["system_prompt"]:
        metadata.setdefault("system_prompt", result["system_prompt"])
    if "rollout_status_detail" in result:
        metadata["rollout_status_detail"] = result["rollout_status_detail"]
    error_reason = _derive_metadata_error_reason(result)
    if error_reason is not None:
        metadata["error_reason"] = error_reason

    if result.get("aborted"):
        log_info(f"[{result.get('instance_id', 'unknown')}] result is aborted, forcing raw_reward={dummy_reward}")
        result["raw_reward"] = dummy_reward
    if _is_dummy_rollout_payload(result):
        result["raw_reward"] = dummy_reward
        result["messages_list_raw_rewards"] = [dummy_reward] * len(messages_list)

    split_rollout_results = result.get("split_rollout_results")
    if not isinstance(split_rollout_results, list) or not all(
        isinstance(item, dict) for item in split_rollout_results
    ):
        split_rollout_results = _build_split_rollout_results(
            {
                **result,
                "messages_list_tags": messages_list_tags,
            }
        )
    else:
        normalized_split_results = []
        for item in split_rollout_results:
            item_copy = dict(item)
            item_system_prompt = item_copy.get("system_prompt")
            if not isinstance(item_system_prompt, str) or not item_system_prompt:
                item_metadata = item_copy.get("metadata", {})
                if isinstance(item_metadata, dict):
                    candidate = item_metadata.get("system_prompt")
                    if isinstance(candidate, str) and candidate:
                        item_system_prompt = candidate
            messages = item_copy.get("messages")
            item_tag = str(item_copy.get("messages_list_tag") or "main_chat").strip().lower()
            if (
                item_tag != "summary"
                and isinstance(messages, list)
                and isinstance(item_system_prompt, str)
                and item_system_prompt
            ):
                normalized_messages = [
                    dict(msg) for msg in messages if isinstance(msg, dict)
                ]
                if normalized_messages and normalized_messages[0].get("role") == "system":
                    normalized_messages[0]["content"] = item_system_prompt
                else:
                    normalized_messages = [
                        {"role": "system", "content": item_system_prompt},
                        *normalized_messages,
                    ]
                item_copy["messages"] = normalized_messages
            normalized_split_results.append(item_copy)
        split_rollout_results = normalized_split_results
    if _is_dummy_rollout_payload(result):
        normalized_dummy_split_results = []
        for item in split_rollout_results:
            if not isinstance(item, dict):
                normalized_dummy_split_results.append(item)
                continue
            item_copy = dict(item)
            item_copy["raw_reward"] = dummy_reward
            item_copy["reward"] = dummy_reward
            normalized_dummy_split_results.append(item_copy)
        split_rollout_results = normalized_dummy_split_results
    result["split_rollout_results"] = split_rollout_results
    metadata["split_rollout_results_count"] = len(split_rollout_results)
    result.pop("messages_list_tags", None)
    metadata.pop("messages_list_tags", None)
    result.pop("summary_messages_list", None)
    result.pop("summary_messages_list_weight_versions", None)
    return result


def _finalize_rollout_result(
    result: dict, exception_path: Path | None = None
) -> dict:
    result = _apply_rollout_status(result)
    result = _normalize_rollout_result_content_origin(result)
    result = _attach_result_token_metadata(result)
    _persist_rollout_result(result, exception_path=exception_path)
    return result


def _normalize_instance(req_data: dict) -> dict:
    instance = req_data.get("instance") or {}
    if hasattr(instance, "to_dict"):
        instance = instance.to_dict()
    if not isinstance(instance, dict):
        instance = {}
    return instance


def _apply_payload_metadata(
    result: dict, model_type: str | None, task_type: str | None
) -> dict:
    metadata = _ensure_result_metadata(result)
    if model_type is not None:
        metadata["model_type"] = model_type
    if task_type is not None:
        metadata["task_type"] = task_type
    return result


def _resolve_router_api_base(
    req_data: dict,
    router_ip_key: str = "sglang_router_ip",
    router_port_key: str = "sglang_router_port",
) -> str | None:
    router_ip = req_data.get(router_ip_key)
    router_port = req_data.get(router_port_key)
    if router_ip and router_port:
        return f"http://{router_ip}:{router_port}"
    return None


def _resolve_api_base(req_data: dict) -> str | None:
    return req_data.get("api_base") or req_data.get("base_url") or _resolve_router_api_base(req_data)


def _compose_project_name(unique_sid: str) -> str:
    return unique_sid.lower().replace(".", "-")


def _get_main_container_state(unique_sid: str) -> str:
    project_name = _compose_project_name(unique_sid)
    command = [
        "docker",
        "ps",
        "-a",
        "--filter",
        f"label=com.docker.compose.project={project_name}",
        "--filter",
        "label=com.docker.compose.service=main",
        "--format",
        "{{.State}}",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "docker_unavailable"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return f"docker_error:{stderr}" if stderr else "docker_error"

    states = [line.strip().lower() for line in (result.stdout or "").splitlines() if line.strip()]
    if not states:
        return "missing"
    return states[0]


def _terminate_subprocess(
    process: subprocess.Popen,
    *,
    unique_sid: str,
    grace_seconds: int,
    reason: str,
) -> tuple[str, str]:
    log_info(f"[{unique_sid}] Terminating subprocess: {reason}")
    process.terminate()
    try:
        return process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        log_info(f"[{unique_sid}] Killing subprocess after terminate grace period: {reason}")
        process.kill()
        return process.communicate()


def _build_main_container_unavailable_result(
    *,
    instance_id: str,
    unique_sid: str,
    uid: str | None,
    trials_dir: Path,
    exception_path: Path,
    container_state: str,
    watchdog_failures: int,
    stdout: str | None,
    stderr: str | None,
    model_type: str | None,
    task_type: str | None,
) -> dict:
    error_message = (
        "Main container became unavailable during trial "
        f"(state={container_state}, consecutive_failures={watchdog_failures})"
    )
    diagnostics = _build_trial_diagnostics(
        trials_dir,
        unique_sid,
        fallback_type="main_container_unavailable",
        error_message=error_message,
        stdout=stdout,
        stderr=stderr,
    )
    diagnostics["main_container_state"] = container_state
    diagnostics["watchdog_failures"] = watchdog_failures
    diagnostics["detected_by"] = "server_container_watchdog"

    return _finalize_rollout_result(
        _apply_payload_metadata(
            {
                "instance_id": instance_id,
                "sid": unique_sid,
                "uid": uid or unique_sid,
                "messages": dummy_messages,
                "raw_reward": dummy_reward,
                "overturn": False,
                "overlong": False,
                "format_error": False,
                "aborted": True,
                "rollout_status": "aborted",
                "error": error_message,
                "rollout_status_detail": {
                    "exception_type": "MainContainerUnavailable",
                    "exception_message": error_message,
                    "container_state": container_state,
                    "watchdog_failures": watchdog_failures,
                    "detected_by": "server_container_watchdog",
                },
                "metadata": diagnostics,
                "meta_data": diagnostics,
            },
            model_type=model_type,
            task_type=task_type,
        ),
        exception_path=exception_path,
    )


def _get_trial_progress_stage(trials_dir: Path, unique_sid: str) -> str | None:
    progress = _safe_load_json_file(trials_dir / unique_sid / "progress.json")
    if not isinstance(progress, dict):
        return None
    stage = progress.get("stage")
    if isinstance(stage, str):
        normalized = stage.strip()
        return normalized or None
    return None


def _is_active_watch_stage(progress_stage: str | None) -> bool:
    return progress_stage in {"agent_setup", "agent_execution", "verification"}


def _find_task_dirs(dataset_path: Path) -> list[Path]:
    if not dataset_path.exists():
        return []
    if (dataset_path / "task.toml").exists():
        return [dataset_path]
    task_dirs = []
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir() and (item / "task.toml").exists():
            task_dirs.append(item)
    return task_dirs


def _set_current_dataset(dataset_path: str) -> dict:
    global DATASET_PATH, _current_dataset_info, _current_dataset_map
    dataset_root = Path(dataset_path).expanduser().resolve()
    task_dirs = _find_task_dirs(dataset_root)
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {dataset_root}")

    _current_dataset_map = {task_dir.name: True for task_dir in task_dirs}
    _current_dataset_info = {
        "dataset_path": str(dataset_root),
        "task_count": len(task_dirs),
    }
    DATASET_PATH = str(dataset_root)
    return _current_dataset_info


def run_trial_subprocess(
    instance_id: str,
    dataset_path: str,
    unique_sid: str,
    trials_dir: Path,
    agent_name: str,
    max_iterations: int,
    timeout_multiplier: float,
    api_base: str | None,
    summary_api_base: str | None,
    tokenizer_path: str | None,
    chat_template_path: str | None,
    uid: str | None,
    timeout_seconds: int,
    model_name: str | None,
    llm_query_type: str | None,
    api_key: str | None,
    max_context_length: int | None,
    sampling_params: dict,
    model_type: str | None,
    task_type: str | None,
    parser_name: str | None,
    use_generate: bool,
    enable_summarize: str,
    summarization_mode: str,
    summary_training: str,
    summary_history_user_turns: int,
    enable_tito: bool,
    context_management_max_input_tokens: int | None,
    context_management_proactive_threshold: int | None,
    context_management_max_new_tokens: int | None,
    context_management_max_count: int,
    remove_main_key: bool,
) -> dict:
    """Run trial in a subprocess (blocking, runs in thread pool)."""
    cleanup_grace_seconds = int(os.environ.get('HARBOR_SUBPROCESS_CLEANUP_GRACE_SECONDS', '20'))
    watchdog_interval_seconds = float(
        os.environ.get("HARBOR_CONTAINER_WATCHDOG_INTERVAL_SECONDS", "300")
    )
    watchdog_failure_threshold = int(
        os.environ.get("HARBOR_CONTAINER_WATCHDOG_FAILURE_THRESHOLD", "3")
    )
    watchdog_kill_grace_seconds = int(
        os.environ.get("HARBOR_CONTAINER_WATCHDOG_KILL_GRACE_SECONDS", "10")
    )
    unavailable_states = {"missing", "exited", "dead", "removing"}
    exception_path = trials_dir / unique_sid / "exception.txt"
    cmd = [
        sys.executable, str(TRIAL_WORKER_PATH),
        "--instance-id", instance_id,
        "--dataset-path", dataset_path,
        "--trials-dir", str(trials_dir),
        "--unique-sid", unique_sid,
        "--agent-name", agent_name,
        "--max-iterations", str(max_iterations),
        "--timeout-multiplier", str(timeout_multiplier),
    ]
    if parser_name:
        cmd.extend(["--parser-name", parser_name])
    cmd.extend(["--use-generate", str(use_generate).lower()])
    if llm_query_type:
        cmd.extend(["--llm-query-type", llm_query_type])
    cmd.extend(["--enable-summarize", enable_summarize])
    cmd.extend(["--summarization-mode", summarization_mode])
    cmd.extend(["--summary-training", summary_training])
    cmd.extend(["--summary-history-user-turns", str(summary_history_user_turns)])
    cmd.extend(["--enable-tito", str(enable_tito).lower()])
    if context_management_max_input_tokens is not None:
        cmd.extend(
            [
                "--context-management-max-input-tokens",
                str(context_management_max_input_tokens),
            ]
        )
    if context_management_proactive_threshold is not None:
        cmd.extend(
            [
                "--context-management-proactive-threshold",
                str(context_management_proactive_threshold),
            ]
        )
    if context_management_max_new_tokens is not None:
        cmd.extend(
            [
                "--context-management-max-new-tokens",
                str(context_management_max_new_tokens),
            ]
        )
    if context_management_max_count is not None:
        cmd.extend(
            [
                "--context-management-max-count",
                str(context_management_max_count),
            ]
        )

    if api_base:
        cmd.extend(["--api-base", api_base])
    if summary_api_base:
        cmd.extend(["--summary-api-base", summary_api_base])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if tokenizer_path:
        cmd.extend(["--tokenizer-path", tokenizer_path])
    if chat_template_path:
        cmd.extend(["--chat-template-path", chat_template_path])
    if uid:
        cmd.extend(["--uid", uid])
    if model_name:
        cmd.extend(["--model-name", model_name])
    if max_context_length is not None:
        cmd.extend(["--rollout-max-context-length", str(max_context_length)])
    if sampling_params:
        cmd.extend(["--sampling-params", json.dumps(sampling_params)])
    if remove_main_key:
        cmd.append("--remove-main-key")
    cmd.extend(["--agent", agent_name])
    log_info(f"[{unique_sid}] Starting subprocess: {' '.join(cmd[:6])}...")

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        start_time = time.monotonic()
        consecutive_watchdog_failures = 0
        last_logged_container_state = "unknown"
        seen_running_once = False
        last_startup_wait_signature: tuple[str, str | None, bool] | None = None

        while True:
            remaining_timeout_seconds = timeout_seconds - (time.monotonic() - start_time)
            if remaining_timeout_seconds <= 0:
                log_info(f"[{unique_sid}] Subprocess timeout after {timeout_seconds}s")
                stdout, stderr = _terminate_subprocess(
                    process,
                    unique_sid=unique_sid,
                    grace_seconds=cleanup_grace_seconds,
                    reason=f"timeout after {timeout_seconds}s",
                )

                timeout_diagnostics = _build_trial_diagnostics(
                    trials_dir,
                    unique_sid,
                    fallback_type="subprocess_timeout",
                    error_message=f"Timeout after {timeout_seconds}s",
                    stdout=stdout,
                    stderr=stderr,
                )
                timeout_stage = timeout_diagnostics.get("timeout_stage")
                timeout_stage_suffix = (
                    f" (stage={timeout_stage})"
                    if isinstance(timeout_stage, str) and timeout_stage
                    else ""
                )
                timeout_payload = _recover_timeout_rollout_payload(
                    trials_dir=trials_dir,
                    unique_sid=unique_sid,
                    instance_id=instance_id,
                    uid=uid,
                )
                _merge_result_metadata(timeout_payload, timeout_diagnostics)
                timeout_payload.update({
                    "error": f"Timeout after {timeout_seconds}s{timeout_stage_suffix}",
                    "rollout_status_detail": {
                        "exception_type": "SubprocessTimeoutExpired",
                        "exception_message": (
                            f"Subprocess timed out after {timeout_seconds}s"
                            f"{timeout_stage_suffix}"
                        ),
                        "timeout": True,
                        "timeout_seconds": timeout_seconds,
                        "timeout_stage": timeout_stage,
                    },
                })
                return _finalize_rollout_result(
                    _apply_payload_metadata(
                        timeout_payload, model_type=model_type, task_type=task_type
                    ),
                    exception_path=exception_path,
                )

            poll_timeout_seconds = min(
                max(watchdog_interval_seconds, 0.1),
                remaining_timeout_seconds,
            )
            try:
                stdout, stderr = process.communicate(timeout=poll_timeout_seconds)
                break
            except subprocess.TimeoutExpired:
                container_state = _get_main_container_state(unique_sid)
                if container_state == "running":
                    seen_running_once = True
                progress_stage = _get_trial_progress_stage(trials_dir, unique_sid)
                strict_watch_enabled = seen_running_once and (
                    _is_active_watch_stage(progress_stage) or progress_stage is None
                )
                if container_state != last_logged_container_state:
                    logger.debug(
                        f"[{unique_sid}] Watchdog observed main container state: "
                        f"{container_state}"
                    )
                    last_logged_container_state = container_state

                if container_state in unavailable_states and strict_watch_enabled:
                    consecutive_watchdog_failures += 1
                    log_info(
                        f"[{unique_sid}] Main container unavailable "
                        f"(state={container_state}, consecutive_failures="
                        f"{consecutive_watchdog_failures}/{watchdog_failure_threshold})"
                    )
                else:
                    if container_state in unavailable_states and not strict_watch_enabled:
                        stage_label = progress_stage or "unknown"
                        startup_wait_signature = (
                            container_state,
                            progress_stage,
                            seen_running_once,
                        )
                        if startup_wait_signature != last_startup_wait_signature:
                            logger.debug(
                                f"[{unique_sid}] Watchdog observed startup-phase unavailable "
                                f"main container state={container_state}; waiting for container "
                                f"startup (stage={stage_label}, seen_running_once={seen_running_once})"
                            )
                            last_startup_wait_signature = startup_wait_signature
                    consecutive_watchdog_failures = 0

                if consecutive_watchdog_failures < watchdog_failure_threshold:
                    continue

                stdout, stderr = _terminate_subprocess(
                    process,
                    unique_sid=unique_sid,
                    grace_seconds=watchdog_kill_grace_seconds,
                    reason=(
                        "main container became unavailable "
                        f"(state={container_state}, failures={consecutive_watchdog_failures})"
                    ),
                )
                return _build_main_container_unavailable_result(
                    instance_id=instance_id,
                    unique_sid=unique_sid,
                    uid=uid,
                    trials_dir=trials_dir,
                    exception_path=exception_path,
                    container_state=container_state,
                    watchdog_failures=consecutive_watchdog_failures,
                    stdout=stdout,
                    stderr=stderr,
                    model_type=model_type,
                    task_type=task_type,
                )

        result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)

        parsed_output = None
        if result.stdout.strip():
            # Parse the last line as JSON (in case there's other output)
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                try:
                    parsed_output = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if parsed_output is not None:
            parsed_output = _apply_payload_metadata(
                parsed_output, model_type=model_type, task_type=task_type
            )
            parsed_output = _finalize_rollout_result(
                parsed_output, exception_path=exception_path
            )
            log_info(f"[{unique_sid}] Rollout status: {parsed_output.get('rollout_status')}")
            return parsed_output

        # If we couldn't parse output, check stderr
        log_info(f"[{unique_sid}] Subprocess failed. returncode={result.returncode}")
        if result.stderr:
            log_info(f"[{unique_sid}] stderr: {result.stderr[:1000]}")
        error_message = (
            f"Subprocess failed: {result.stderr[:500] if result.stderr else 'unknown error'}"
        )
        diagnostics = _build_trial_diagnostics(
            trials_dir,
            unique_sid,
            fallback_type="subprocess_failed",
            error_message=error_message,
            stdout=result.stdout,
            stderr=result.stderr,
        )

        return _finalize_rollout_result(_apply_payload_metadata({
            "instance_id": instance_id,
            "sid": unique_sid,
            "uid": uid or unique_sid,
            "messages": dummy_messages,
            "raw_reward": dummy_reward,
            "overturn": False,
            "overlong": False,
            "format_error": False,
            "aborted": True,
            "rollout_status": "aborted",
            "error": error_message,
            "rollout_status_detail": {
                "exception_type": "SubprocessFailed",
                "exception_message": error_message,
                "returncode": result.returncode,
            },
            "metadata": diagnostics,
            "meta_data": diagnostics,
        }, model_type=model_type, task_type=task_type), exception_path=exception_path)
    except Exception as e:
        log_info(f"[{unique_sid}] Subprocess error: {e}")
        error_message = str(e)
        diagnostics = _build_trial_diagnostics(
            trials_dir,
            unique_sid,
            fallback_type="subprocess_exception",
            error_message=error_message,
        )
        return _finalize_rollout_result(_apply_payload_metadata({
            "instance_id": instance_id,
            "sid": unique_sid,
            "uid": uid or unique_sid,
            "messages": dummy_messages,
            "raw_reward": dummy_reward,
            "overturn": False,
            "overlong": False,
            "format_error": False,
            "aborted": True,
            "rollout_status": "aborted",
            "error": error_message,
            "rollout_status_detail": {
                "exception_type": type(e).__name__,
                "exception_message": error_message,
            },
            "metadata": diagnostics,
            "meta_data": diagnostics,
        }, model_type=model_type, task_type=task_type), exception_path=exception_path)
    finally:
        if process is not None:
            _cleanup_worker_containers(unique_sid)


@app.post("/preload")
async def preload_dataset(req: Request):
    try:
        body = await req.body()
        payload = await req.json() if body else {}
        dataset_path = payload.get("dataset_path") or DATASET_PATH
        if not dataset_path:
            return JSONResponse(
                content={"error": "dataset_path required"}, status_code=400
            )
        info = _set_current_dataset(dataset_path)
        return {"message": "Dataset pre-loaded successfully", "dataset_info": info}
    except Exception as e:
        logger.error(f"Error pre-loading dataset: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to pre-load dataset: {str(e)}"},
            status_code=500,
        )


@app.post("/run_worker")
async def run_worker(req_data: dict):
    """Stream instance processing results."""
    global _running_workers_count, _running_workers_details

    worker_id = str(uuid.uuid4())
    instance = _normalize_instance(req_data)
    instance_id = req_data.get("instance_id") or instance.get("instance_id") or "unknown"
    unique_sid = f"{instance_id}_{str(uuid.uuid4())[:24]}"

    async def stream_generator():
        global _running_workers_count, _running_workers_details
        temp_dir = None
        model_type = None
        task_type = None

        try:
            uid = req_data.get("uid")
            if not uid:
                yield json_dumps(
                    {"type": "error", "data": {"error": "uid required"}}
                ) + "\n"
                return

            if not instance_id or instance_id == 'unknown':
                yield json_dumps({"type": "error", "data": {"error": "instance_id required"}}) + "\n"
                return

            dataset_info = _current_dataset_info
            dataset_path = None
            if dataset_info and dataset_info.get("dataset_path"):
                dataset_path = dataset_info["dataset_path"]
            elif DATASET_PATH:
                dataset_path = DATASET_PATH
            if not dataset_path:
                yield json_dumps(
                    {"type": "error", "data": {"error": "dataset_path not configured"}}
                ) + "\n"
                return
            if _current_dataset_map and instance_id not in _current_dataset_map:
                yield json_dumps(
                    {
                        "type": "error",
                        "data": {"error": f"instance_id not found: {instance_id}"},
                    }
                ) + "\n"
                return

            max_iterations = req_data.get('max_iterations', 100)
            timeout_minutes = req_data.get('timeout', 60)
            timeout_seconds = timeout_minutes * 60
            timeout_multiplier = req_data.get("timeout_multiplier", 100.0)

            tokenizer_path = req_data.get("tokenizer_path")
            api_base = _resolve_api_base(req_data)
            summary_api_base = req_data.get("summary_api_base") or req_data.get("summary_base_url") or _resolve_router_api_base(
                req_data,
                router_ip_key="summary_sglang_router_ip",
                router_port_key="summary_sglang_router_port",
            )
            if summary_api_base is None:
                summary_api_base = api_base
            model_name = req_data.get("model")
            llm_query_type = req_data.get("llm_query_type")
            if (
                not llm_query_type
                and isinstance(model_name, str)
                and "claude" in model_name.lower()
            ):
                llm_query_type = model_name
            api_key = req_data.get("api_key")
            max_context_length = req_data.get("rollout_max_context_length", 131072)
            sampling_params = req_data.get("sampling_params") or {}
            if not isinstance(sampling_params, dict):
                sampling_params = {}
            model_type = req_data.get("model_type")
            task_type = req_data.get("task_type")
            agent_name = (
                req_data.get("agent_name")
                or req_data.get("agent")
                or AgentName.TERMINUS_KIRA.value
            )
            parser_name = req_data.get("parser_name") if agent_name == AgentName.TERMINUS_2.value else None
            use_generate = _coerce_bool(req_data.get("use_generate"), default=True)
            chat_template_path = req_data.get("chat_template_path")
            enable_summarize = _normalize_context_management_mode(
                req_data.get("enable_summarize"), default="none"
            )
            summarization_mode = str(
                req_data.get("summarization_mode") or "summarize"
            ).strip().lower()
            if summarization_mode not in {"summarize", "truncate"}:
                summarization_mode = "summarize"
            summary_training = _normalize_summary_training_mode(
                req_data.get("summary_training"), default="main_chat"
            )
            summary_history_user_turns = _coerce_int(
                req_data.get("summary_history_user_turns"),
                default=2,
            )
            enable_tito = _coerce_bool(req_data.get("enable_tito"), default=False)
            context_management_max_input_tokens = _coerce_int(
                req_data.get("context_management_max_input_tokens"),
                default=None,
            )
            context_management_proactive_threshold = _coerce_int(
                req_data.get("context_management_proactive_threshold"),
                default=None,
            )
            context_management_max_new_tokens = _coerce_int(
                req_data.get("context_management_max_new_tokens"),
                default=None,
            )
            context_management_max_count = _coerce_int(
                req_data.get("context_management_max_count"),
                default=-1,
            )
            force_context_management_count = max(
                0, _coerce_int(req_data.get("force_context_management_count"), default=0)
            )
            remove_main_key = _coerce_bool(req_data.get("remove_main_key"), default=False)

            if force_context_management_count > 0:
                forced_result = _build_forced_context_management_result(
                    instance_id=instance_id,
                    sid=unique_sid,
                    force_context_management_count=force_context_management_count,
                )
                forced_result["uid"] = uid or unique_sid
                forced_result = _finalize_rollout_result(
                    _apply_payload_metadata(
                        forced_result,
                        model_type=model_type,
                        task_type=task_type,
                    )
                )
                yield json_dumps({"type": "result", "data": forced_result}) + "\n"
                return

            start_time = time.time()
            async with _running_workers_lock:
                _running_workers_count += 1
                _running_workers_details[worker_id] = {
                    "instance_id": instance_id,
                    "uid": uid,
                    "start_time": datetime.datetime.now().isoformat(),
                    "_cached_start_time": start_time,
                    "status": "running",
                    "model_type": model_type,
                    "task_type": task_type,
                    "agent": HARBOR_AGENT_NAME,
                    "sampling_params": sampling_params,
                }
            log_info(f"Worker {worker_id} started for {instance_id}. Total: {_running_workers_count}")

            temp_dir = Path(EVAL_OUTPUT_DIR) / unique_sid
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Run subprocess in thread pool
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                _executor,
                run_trial_subprocess,
                instance_id,
                dataset_path,
                unique_sid,
                temp_dir,
                agent_name,
                max_iterations,
                timeout_multiplier,
                api_base,
                summary_api_base,
                tokenizer_path,
                chat_template_path,
                uid,
                timeout_seconds,
                model_name,
                llm_query_type,
                api_key,
                max_context_length,
                sampling_params,
                model_type,
                task_type,
                parser_name,
                use_generate,
                enable_summarize,
                summarization_mode,
                summary_training,
                summary_history_user_turns,
                enable_tito,
                context_management_max_input_tokens,
                context_management_proactive_threshold,
                context_management_max_new_tokens,
                context_management_max_count,
                remove_main_key,
            )

            # Send heartbeats while waiting
            heartbeat_interval = 30
            while True:
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=heartbeat_interval
                    )
                    break  # Got result
                except asyncio.TimeoutError:
                    # Still running, send heartbeat
                    elapsed = int(time.time() - start_time)
                    heartbeat = {
                        "type": "heartbeat",
                        "worker_id": worker_id,
                        "instance_id": instance_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "elapsed_seconds": elapsed,
                    }
                    yield json_dumps(heartbeat) + "\n"

            log_info(f"[{worker_id}] Finished for {instance_id}")
            if ENABLE_PROFILING and temp_dir is not None:
                await _persist_server_profile_artifacts(temp_dir, unique_sid)
            yield json_dumps({"type": "result", "data": result}) + "\n"

        except Exception as e:
            log_info(f"[{worker_id}] Error in stream_generator for {instance_id}: {e}")
            logger.error(f"Error in stream_generator: {e}", exc_info=True)
            error_result = dummy_rollout_result.copy()
            diagnostics = {}
            if temp_dir is not None:
                diagnostics = _build_trial_diagnostics(
                    temp_dir,
                    unique_sid,
                    fallback_type="stream_generator_exception",
                    error_message=str(e),
                )
            error_result.update({
                'instance_id': instance_id,
                'sid': unique_sid,
                'uid': uid or unique_sid,
                'aborted': True,
                'rollout_status': "aborted",
                'error': str(e),
                "rollout_status_detail": {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                },
                "metadata": diagnostics,
                "meta_data": diagnostics,
            })
            exception_path = None
            if temp_dir is not None:
                exception_path = temp_dir / unique_sid / "exception.txt"
            error_result = _finalize_rollout_result(
                _apply_payload_metadata(
                    error_result, model_type=model_type, task_type=task_type
                ),
                exception_path=exception_path,
            )
            yield json_dumps({"type": "result", "data": error_result}) + "\n"

        finally:
            async with _running_workers_lock:
                if worker_id in _running_workers_details:
                    _running_workers_count -= 1
                    del _running_workers_details[worker_id]
                    log_info(f"Worker {worker_id} cleaned up. Total: {_running_workers_count}")

            # Cleanup temp dir
            if not KEEP_TEMP_DIR and temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_err:
                    log_info(f"[{worker_id}] Failed to cleanup temp dir {temp_dir}: {cleanup_err}")

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


@app.get("/worker_stats")
async def get_worker_stats():
    """Get statistics about running workers."""
    async with _running_workers_lock:
        workers_list = []
        for wid, details in _running_workers_details.items():
            elapsed = time.time() - details.get('_cached_start_time', time.time())
            workers_list.append({
                'worker_id': wid,
                'instance_id': details.get('instance_id'),
                'status': details.get('status', 'running'),
                'elapsed_seconds': int(elapsed),
            })

        return {
            'total_running': _running_workers_count,
            'workers': workers_list,
        }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    global _running_workers_lock
    global _profiling_artifacts_lock
    global DATASET_PATH
    _running_workers_lock = asyncio.Lock()
    _profiling_artifacts_lock = asyncio.Lock() if ENABLE_PROFILING else None
    Path(EVAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if ENABLE_PROFILING:
        for artifact_name in ("profiling_timeline.html", "profiling_timeline.jsonl"):
            artifact_path = Path(EVAL_OUTPUT_DIR) / artifact_name
            try:
                if artifact_path.exists():
                    artifact_path.unlink()
            except Exception as e:
                logger.warning(
                    f"Failed to remove stale profiling artifact {artifact_path}: {e}"
                )
    DATASET_PATH = os.environ.get("DATASET_PATH")
    if not DATASET_PATH and Path(_DEFAULT_DATASET_PATH).exists():
        DATASET_PATH = _DEFAULT_DATASET_PATH
    dataset_message = f"Dataset: {DATASET_PATH}"
    if DATASET_PATH:
        try:
            info = _set_current_dataset(DATASET_PATH)
            dataset_message = (
                f"Dataset: {info.get('dataset_path')} (tasks={info.get('task_count')})"
            )
        except Exception as e:
            logger.warning(f"Failed to preload dataset from {DATASET_PATH}: {e}")
    rollout_message = "Rollout JSON dir: disabled"
    if _ROLL_OUT_JSON_DIR is not None:
        _ensure_server_start_ts()
        rollout_root = _ROLL_OUT_JSON_DIR / str(_SERVER_START_TS)
        rollout_root.mkdir(parents=True, exist_ok=True)
        rollout_message = f"Rollout JSON dir: {rollout_root}"
    log_info(
        f"Harbor compatible server started. Output dir: {EVAL_OUTPUT_DIR}, "
        f"{dataset_message}, Agent: {HARBOR_AGENT_NAME}, {rollout_message}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--rollout-json-dir",
        default=None,
        help="Directory to persist rollout JSON results by status/instance.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "5001")),
        help="Port to bind the harbor server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    args, _unknown = parser.parse_known_args()
    if args.rollout_json_dir:
        _set_rollout_json_dir(args.rollout_json_dir)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=10800,
        timeout_graceful_shutdown=120,
        limit_concurrency=None,
        backlog=2048,
    )
