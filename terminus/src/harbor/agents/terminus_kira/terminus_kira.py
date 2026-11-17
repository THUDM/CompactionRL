"""
TerminusKira - A native tool-use variant of Terminus2.

This agent inherits from harbor's Terminus2 and replaces the ICL (In-Context Learning)
JSON/XML parsing approach with native tool calling via the `tools` parameter in LLM API calls.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import litellm
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
    BadRequestError,
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_2 import (
    Command,
    ContextManagementLimitExceededError,
)
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMBackend,
    LLMResponse,
    OutputLengthExceededError,
    ParseError,
)
from harbor.llms.chat import Chat
from harbor.agents.terminus_kira.anthropic_caching import add_anthropic_caching
from harbor.models.metric import UsageInfo
from harbor.models.trajectories import (
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
)
from harbor.utils.pager_policy import (
    PAGER_OUTPUT_LIMIT_BYTES,
    PAGER_USAGE_STATUS,
    detect_pager_usage,
    has_neutralized_pager_output,
)


class BlockError(Exception):
    """Raised when infrastructure API call blocks for too long."""
    pass


class ToolCallTransportError(RuntimeError):
    """Transient transport failure while performing native tool calls."""


class SingleTurnOverlongError(OutputLengthExceededError):
    """Raised when a single generation exceeds output length in training mode."""

    pass


class PagerUsageError(RuntimeError):
    """Raised when a training trace explicitly invokes a pager/editor."""


BLOCK_TIMEOUT_SEC = 600  # 10 minutes
_MARKER_PREFIX = "__CMDEND__"  # Marker prefix for command completion detection
_SHORT_COMMAND_POLLING_THRESHOLD_SEC = 1.0
_SHORT_COMMAND_PREFIXES = (
    "pwd",
    "ls",
    "echo",
    "cat ",
    "head ",
    "tail ",
    "sed ",
    "grep ",
    "rg ",
    "find ",
    "which ",
    "git status",
    "git diff",
    "tmux ",
)
_CONTINUATION_PROMPT_LINE_PATTERNS = (
    re.compile(r"^\s*>\s*$"),
    re.compile(
        r"^\s*(quote|dquote|bquote|cmdsubst|heredoc|if|for|while|until|case|select|subsh)>\s*$"
    ),
)


@dataclass
class ToolCallResponse:
    """Extended response that includes tool calls."""

    content: str | None
    tool_calls: list[dict[str, Any]]
    raw_content: str | None = None
    reasoning_content: str | None = None
    history_content: str | None = None
    usage: UsageInfo | None = None
    terminal_stop_token: str | None = None
    finish_reason: str | None = None


@dataclass
class ImageReadRequest:
    """Request to read and analyze an image file."""

    file_path: str
    image_read_instruction: str


# Tool description strings
_EXECUTE_COMMANDS_DESC = (
    "Call this to execute commands in the terminal. "
    "Provide only the commands string."
)

_COMMANDS_DESC = (
    "A single string containing the exact keystrokes to send to the terminal. "
)

_COMMAND_STRING_DESC = (
    "String containing the exact keystrokes to send to the terminal. "
    "The text will be used completely verbatim as keystrokes. "
    "Write commands exactly as you want them sent to the terminal. "
    "Shell commands MUST end with a newline (\\n) to execute. "
    "For special key sequences, use tmux-style escape sequences: C-c for Ctrl+C, C-d for Ctrl+D. "
    "Each command's keystrokes are sent exactly as written to the terminal. "
    "Do not include extra whitespace before or after the keystrokes unless it's part of the intended command."
)

_TASK_COMPLETE_DESC = "Call this when the task is complete."

_IMAGE_READ_DESC = (
    "Read and analyze an image file. "
    "Use this ONLY for image files that you need to visually analyze. "
    "Do NOT use this for text files — use shell commands (cat, head, etc.) instead. "
    "The image will be sent to the model for visual analysis "
    "and you will receive a text description in the next turn."
)

_FILE_PATH_DESC = (
    "Absolute path to the image file. "
    "Supported formats: PNG, JPG, JPEG, GIF, WEBP."
)

_IMAGE_READ_INSTRUCTION_DESC = (
    "A text instruction describing what you want to learn from the image. "
    "Be specific about what information to extract."
)

_ADDITIONAL_SYSTEM_PROMPT = (
    "Do not include separate analysis or plan fields.\n\n"
    "The environment sets GIT_PAGER=cat and PAGER=cat. Normal git "
    "diff/log/show/blame output prints directly instead of opening an "
    "interactive pager, and pager-like output may be capped at 16384 bytes. "
    "Do not invoke explicit pager or interactive editor commands such as "
    "less, more, most, view, vi, vim, or nvim."
)

# Tool definitions for native tool use
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_commands",
            "description": _EXECUTE_COMMANDS_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "commands": {
                        "type": "string",
                        "description": _COMMANDS_DESC,
                    },
                },
                "required": ["commands"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": _TASK_COMPLETE_DESC,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_read",
            "description": _IMAGE_READ_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": _FILE_PATH_DESC,
                    },
                    "image_read_instruction": {
                        "type": "string",
                        "description": _IMAGE_READ_INSTRUCTION_DESC,
                    },
                },
                "required": ["file_path", "image_read_instruction"],
            },
        },
    },
]

class TerminusKira(Terminus2):
    """
    TerminusKira extends harbor's Terminus2 with native tool calling.

    Instead of prompting the model to output JSON/XML and parsing it,
    TerminusKira uses the `tools` parameter in LLM API calls for structured outputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._marker_seq = 0
        self._total_time_saved = 0.0
        self._pager_usage_detected: dict[str, str] | None = None

    @staticmethod
    def _normalize_openai_api_base(api_base: str | None) -> str | None:
        if not api_base:
            return api_base

        parsed = urlparse(api_base)
        if parsed.path not in ("", "/"):
            return api_base.rstrip("/")

        return api_base.rstrip("/") + "/v1"

    @staticmethod
    def _format_tool_calls_for_history(tool_calls: list[dict[str, Any]]) -> str:
        if not tool_calls:
            return ""

        lines = ["Executed tool calls:"]
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name", "unknown_tool")
            arguments = tool_call.get("function", {}).get("arguments", "{}")
            lines.append(f"- {function_name}: {arguments}")
        return "\n".join(lines)

    @staticmethod
    def _tool_calls_for_trajectory(
        raw_tool_calls: list[dict[str, Any]] | None,
    ) -> list[ToolCall] | None:
        """Convert OpenAI-style tool calls into ATIF ToolCall objects."""
        if not raw_tool_calls:
            return None

        tool_calls: list[ToolCall] = []
        for idx, tc in enumerate(raw_tool_calls):
            function = tc.get("function", {}) if isinstance(tc, dict) else {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {"_raw": arguments}
            tool_calls.append(
                ToolCall(
                    tool_call_id=tc.get("id") or tc.get("tool_call_id") or f"call_{idx}",
                    function_name=function.get("name", "unknown_tool"),
                    arguments=arguments,
                )
            )
        return tool_calls or None

    def _build_assistant_history_message(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]],
        reasoning_content: str | None = None,
        history_content: str | None = None,
        num_tokens: int = 0,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": (content or "").strip(),
            "num_tokens": int(num_tokens or 0),
        }
        if tool_calls:
            message["tool_calls"] = self._normalize_tool_calls_for_storage(tool_calls)
        if isinstance(history_content, str) and history_content:
            message["history_content"] = history_content
            message["content_origin"] = history_content
        else:
            message["content_origin"] = message["content"]
        if self._interleaved_thinking and reasoning_content:
            message["reasoning_content"] = reasoning_content
        if self._enable_tito and self._session_id:
            message["main_key"] = self._session_id
        return message

    def _build_prompt_history_message(
        self,
        role: str,
        content: str,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": role,
            "content": content,
            "num_tokens": 0,
        }
        if self._enable_tito and self._session_id:
            message["main_key"] = self._session_id
        return message

    @staticmethod
    def _prepare_messages_for_llm_call(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prepared_messages = json.loads(json.dumps(messages))
        for msg in prepared_messages:
            if not isinstance(msg, dict):
                continue
            # Preserve exact raw assistant text via history_content, but do not feed
            # stored structured tool_calls back into the next model call. They are
            # saved for export only and would otherwise drift from the raw text path.
            msg.pop("tool_calls", None)
        if (
            prepared_messages
            and isinstance(prepared_messages[0], dict)
            and prepared_messages[0].get("role") == "system"
        ):
            content = prepared_messages[0].get("content")
            if isinstance(content, str):
                if "Do not include separate analysis or plan fields." not in content:
                    prepared_messages[0]["content"] = (
                        f"{content.rstrip()}\n\n{_ADDITIONAL_SYSTEM_PROMPT}"
                    )
            else:
                prepared_messages[0]["content"] = _ADDITIONAL_SYSTEM_PROMPT
        else:
            prepared_messages.insert(
                0,
                {
                    "role": "system",
                    "content": _ADDITIONAL_SYSTEM_PROMPT,
                },
            )
        return prepared_messages

    def _append_truncated_assistant_to_chat(
        self,
        chat: Chat,
        prompt: str | None,
        prompt_role: str,
        truncated_response: str | None,
    ) -> None:
        """Persist the failed assistant turn before surfacing single-turn overlong."""
        new_messages: list[dict[str, Any]] = []
        if prompt is not None:
            new_messages.append(self._build_prompt_history_message(prompt_role, prompt))

        raw_content = truncated_response if isinstance(truncated_response, str) else ""
        tool_calls = self._extract_legacy_tool_calls(raw_content)
        content = (
            self._strip_legacy_tool_call_markup(raw_content)
            if tool_calls
            else raw_content
        )
        new_messages.append(
            self._build_assistant_history_message(
                content=content,
                tool_calls=tool_calls,
                reasoning_content=None,
                history_content=raw_content,
                num_tokens=0,
            )
        )
        chat.extend_messages(
            new_messages,
            token_counter=self._count_tokens_for_messages,
        )
        chat.reset_response_chain()

    @staticmethod
    def _tool_call_arguments_to_dict(tool_call: dict[str, Any]) -> dict[str, Any]:
        arguments = {}
        function = tool_call.get("function")
        if isinstance(function, dict):
            arguments = function.get("arguments", {})
        elif isinstance(tool_call.get("arguments"), (dict, str)):
            arguments = tool_call.get("arguments", {})
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _tool_call_name(tool_call: dict[str, Any]) -> str:
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                return name
        name = tool_call.get("name")
        if isinstance(name, str):
            return name
        name = tool_call.get("function_name")
        return name if isinstance(name, str) else ""

    def _normalize_tool_calls_for_storage(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for idx, tool_call in enumerate(tool_calls, start=1):
            function_name = self._tool_call_name(tool_call)
            if not isinstance(function_name, str) or not function_name:
                continue
            tool_call_id = str(
                tool_call.get("id")
                or tool_call.get("tool_call_id")
                or f"call_{len(self._trajectory_steps) + 1}_{idx}"
            )
            normalized.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": self._tool_call_arguments_to_dict(tool_call),
                    },
                }
            )
        return normalized

    def _to_trajectory_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[ToolCall]:
        trajectory_tool_calls: list[ToolCall] = []
        for normalized in self._normalize_tool_calls_for_storage(tool_calls):
            function = normalized.get("function", {})
            trajectory_tool_calls.append(
                ToolCall(
                    tool_call_id=str(normalized.get("id") or ""),
                    function_name=str(function.get("name") or ""),
                    arguments=function.get("arguments")
                    if isinstance(function.get("arguments"), dict)
                    else {},
                )
            )
        return trajectory_tool_calls

    def _capture_rendered_generate_system_prompt(self, chat: Chat) -> None:
        rendered_system_prompt = getattr(
            self._llm,
            "_last_rendered_system_prompt",
            None,
        )
        if not (
            isinstance(rendered_system_prompt, str)
            and rendered_system_prompt.strip()
        ) and getattr(self._llm, "_use_generate", False):
            render_method = getattr(self._llm, "_render_generate_system_prompt", None)
            if callable(render_method):
                try:
                    rendered_system_prompt = render_method(tools=TOOLS)
                except Exception:
                    rendered_system_prompt = None
        if isinstance(rendered_system_prompt, str) and rendered_system_prompt.strip():
            chat.set_rendered_system_prompt(rendered_system_prompt)

    @staticmethod
    def _is_control_keystroke_only(keystrokes: str) -> bool:
        stripped = keystrokes.strip()
        if not stripped:
            return False
        parts = stripped.split()
        return all(re.fullmatch(r"[CMS]-[A-Za-z]", part) for part in parts)

    @staticmethod
    def _is_single_interactive_keystroke(keystrokes: str) -> bool:
        stripped = keystrokes.strip()
        if not stripped:
            return False
        if len(stripped) != 1:
            return False
        # Preserve single-key interactions for pagers/editors, e.g. q, j, k, /.
        # Shell commands should generally be longer and still receive a trailing
        # newline below.
        return True

    @classmethod
    def _normalize_keystrokes_for_execution(cls, keystrokes: str) -> str:
        if not keystrokes:
            return keystrokes
        if cls._is_control_keystroke_only(keystrokes):
            return keystrokes
        if cls._is_single_interactive_keystroke(keystrokes):
            return keystrokes
        if keystrokes.endswith(("\n", "\r")):
            return keystrokes
        return keystrokes + "\n"

    @classmethod
    def _should_use_completion_marker(cls, original_keystrokes: str) -> bool:
        if not original_keystrokes:
            return False
        if cls._is_control_keystroke_only(original_keystrokes):
            return False
        if cls._is_single_interactive_keystroke(original_keystrokes):
            return False
        return True

    def _init_llm(
        self,
        llm_backend: LLMBackend | str,
        model_name: str,
        temperature: float,
        top_p: float | None,
        max_new_tokens: int | None,
        collect_rollout_details: bool,
        llm_kwargs: dict | None,
        api_base: str | None,
        session_id: str | None,
        max_thinking_tokens: int | None,
        reasoning_effort: str | None,
        model_info: dict | None,
        tokenizer_path: str | None,
        chat_template_path: str | None,
        use_generate: bool,
        api_key: str | None,
        request_timeout_sec: float | None,
        use_responses_api: bool,
    ) -> BaseLLM:
        # Terminus-KIRA performs primary model calls directly via native tool calling.
        # When the inference server only supports /generate, we rely on LiteLLM's
        # generate mode to obtain full text and parse tool_calls from it.
        effective_use_generate = use_generate and tokenizer_path is not None
        return super()._init_llm(
            llm_backend=llm_backend,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            collect_rollout_details=collect_rollout_details,
            llm_kwargs=llm_kwargs,
            api_base=api_base,
            session_id=session_id,
            max_thinking_tokens=max_thinking_tokens,
            reasoning_effort=reasoning_effort,
            model_info=model_info,
            tokenizer_path=tokenizer_path,
            chat_template_path=chat_template_path,
            use_generate=effective_use_generate,
            api_key=api_key,
            request_timeout_sec=request_timeout_sec,
            use_responses_api=use_responses_api,
        )

    def _unwind_messages_to_free_tokens(self, chat: Chat, target_free_tokens: int) -> None:
        """
        Trim chat history until enough free context remains for a retry.

        Terminus2 provides `_trim_history_for_context()`. Terminus-KIRA calls this
        helper in its ContextLengthExceededError recovery path.
        """
        self._trim_history_for_context(
            chat=chat,
            extra_messages=[],
            max_new_tokens=target_free_tokens,
            reason="KIRA context unwind",
        )

    async def _with_block_timeout(self, coro, timeout_sec: int = BLOCK_TIMEOUT_SEC):
        """Wrap coroutine with block detection timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_sec)
        except asyncio.TimeoutError:
            raise BlockError(f"Infrastructure API blocked for {timeout_sec}s")

    @staticmethod
    def _should_poll_for_completion(
        keystrokes: str,
        duration_sec: float,
        use_completion_marker: bool,
    ) -> bool:
        """Skip pane polling only for clearly short shell inspection commands."""
        if not use_completion_marker:
            return False

        normalized = keystrokes.strip()
        if (
            duration_sec <= _SHORT_COMMAND_POLLING_THRESHOLD_SEC
            and any(normalized.startswith(prefix) for prefix in _SHORT_COMMAND_PREFIXES)
        ):
            return False

        return True

    @staticmethod
    def _poll_interval_for_duration(duration_sec: float) -> float:
        """Scale pane polling interval with expected command duration."""
        if duration_sec <= 2.0:
            return 1.0
        if duration_sec <= 5.0:
            return 1.25
        if duration_sec <= 15.0:
            return 1.5
        return 2.0

    @staticmethod
    def _tail_screen_for_note(screen: str, max_lines: int = 10) -> str:
        lines = [line.rstrip() for line in screen.splitlines()]
        if not lines:
            return ""
        return "\n".join(lines[-max_lines:]).strip()

    @classmethod
    def _is_continuation_prompt_line(cls, line: str) -> bool:
        stripped = line.rstrip()
        if not stripped:
            return False
        return any(pattern.fullmatch(stripped) for pattern in _CONTINUATION_PROMPT_LINE_PATTERNS)

    @classmethod
    def _looks_like_continuation_prompt(cls, screen: str) -> bool:
        lines = [line.rstrip() for line in screen.splitlines() if line.strip()]
        if not lines:
            return False

        tail = lines[-8:]
        if cls._is_continuation_prompt_line(tail[-1]):
            return True

        prefixed_lines = sum(1 for line in tail if line.lstrip().startswith("> "))
        continuation_prompt_lines = sum(
            1 for line in tail if cls._is_continuation_prompt_line(line)
        )
        return continuation_prompt_lines >= 1 and prefixed_lines >= 2

    async def _recover_continuation_prompt_if_needed(
        self,
        session: TmuxSession,
        command_keystrokes: str,
    ) -> str | None:
        screen_before_recovery = await session.capture_pane(capture_entire=False)
        if not self._looks_like_continuation_prompt(screen_before_recovery):
            return None

        await session.send_keys(
            "C-c",
            block=False,
            min_timeout_sec=0.0,
        )
        await asyncio.sleep(0.5)
        screen_after_recovery = await session.capture_pane(capture_entire=False)

        before_tail = self._tail_screen_for_note(screen_before_recovery)
        after_tail = self._tail_screen_for_note(screen_after_recovery)
        note_lines = [
            "Framework recovery note:",
            f"- The previous command got stuck in a shell continuation prompt after: {command_keystrokes!r}",
            "- The shell was waiting for more input (for example an unterminated quote, heredoc, or multiline shell fragment).",
            "- The framework sent `C-c` to recover the terminal before continuing.",
        ]
        if before_tail:
            note_lines.extend(
                [
                    "- Screen before recovery:",
                    before_tail,
                ]
            )
        if after_tail:
            note_lines.extend(
                [
                    "- Screen after recovery:",
                    after_tail,
                ]
            )
        return "\n".join(note_lines)

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        """Execute commands with marker-based polling for early completion detection.

        Sends a unique echo marker after each command. If the marker appears in
        the output before duration_sec, we move on immediately instead of waiting
        for the full duration. This reduces unnecessary wait time for fast commands.
        """
        recovery_notes: list[str] = []
        for command_index, command in enumerate(commands, start=1):
            self._marker_seq += 1
            marker = f"{_MARKER_PREFIX}{self._marker_seq}__"
            start = time.monotonic()
            span_started_at = datetime.now(timezone.utc)
            original_keystrokes = command.keystrokes
            keystrokes = self._normalize_keystrokes_for_execution(original_keystrokes)
            use_completion_marker = self._should_use_completion_marker(
                original_keystrokes
            )
            should_poll = self._should_poll_for_completion(
                original_keystrokes,
                command.duration_sec,
                use_completion_marker,
            )
            marker_seen = False
            recovered_from_continuation = False

            # Send the command
            await session.send_keys(
                keystrokes,
                block=False,
                min_timeout_sec=0.0,
            )
            if use_completion_marker:
                # Send marker: will execute when shell returns after command
                await session.send_keys(
                    f"echo '{marker}'\n",
                    block=False,
                    min_timeout_sec=0.0,
                )

            if not should_poll:
                await asyncio.sleep(command.duration_sec)
                self._append_profiling_span(
                    category="tool_call",
                    label=self._profile_preview(original_keystrokes),
                    started_at=span_started_at,
                    finished_at=datetime.now(timezone.utc),
                    metadata={
                        "episode": self._n_episodes,
                        "command_index": command_index,
                        "duration_sec": command.duration_sec,
                        "used_completion_marker": use_completion_marker,
                        "polled_for_completion": should_poll,
                        "marker_seen": False,
                        "recovered_from_continuation": False,
                    },
                )
                continue

            # Poll for marker, exit early if found before duration.
            await asyncio.sleep(min(0.5, command.duration_sec))
            poll_interval_sec = self._poll_interval_for_duration(command.duration_sec)
            while time.monotonic() - start < command.duration_sec:
                pane_content = await session.capture_pane()
                if marker in pane_content:
                    marker_seen = True
                    break
                remaining_sec = command.duration_sec - (time.monotonic() - start)
                if remaining_sec <= 0:
                    break
                await asyncio.sleep(min(poll_interval_sec, remaining_sec))

            saved = command.duration_sec - (time.monotonic() - start)
            if saved > 0.1:
                self._total_time_saved += saved
                self.logger.debug(
                    f"[polling] saved {saved:.1f}s "
                    f"(duration={command.duration_sec:.1f}s) "
                    f"cmd={command.keystrokes!r}"
                )

            if not marker_seen:
                recovery_note = await self._recover_continuation_prompt_if_needed(
                    session=session,
                    command_keystrokes=original_keystrokes,
                )
                if recovery_note:
                    recovered_from_continuation = True
                    recovery_notes.append(recovery_note)
            self._append_profiling_span(
                category="tool_call",
                label=self._profile_preview(original_keystrokes),
                started_at=span_started_at,
                finished_at=datetime.now(timezone.utc),
                metadata={
                    "episode": self._n_episodes,
                    "command_index": command_index,
                    "duration_sec": command.duration_sec,
                    "used_completion_marker": use_completion_marker,
                    "polled_for_completion": should_poll,
                    "marker_seen": marker_seen,
                    "recovered_from_continuation": recovered_from_continuation,
                },
            )

        # Filter out marker lines from output so LLM sees clean output
        output = await self._get_incremental_output_profiled(
            session,
            metadata={
                "episode": self._n_episodes,
                "command_count": len(commands),
                "trigger": "post_command_observation",
            },
        )
        markers = {
            f"{_MARKER_PREFIX}{seq}__" for seq in range(1, self._marker_seq + 1)
        }
        lines = output.split("\n")
        lines = [line for line in lines if not any(m in line for m in markers)]
        output = "\n".join(lines)
        if recovery_notes:
            output = "\n\n".join([*recovery_notes, output])
        max_bytes = (
            PAGER_OUTPUT_LIMIT_BYTES
            if any(has_neutralized_pager_output(command.keystrokes) for command in commands)
            else 30000
        )
        return False, self._limit_output_length(output, max_bytes=max_bytes)

    @staticmethod
    def name() -> str:
        return "terminus-kira"

    def version(self) -> str | None:
        return "1.0.0"

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run the agent, storing the original instruction for later use."""
        self._original_instruction = instruction
        await super().run(instruction, environment, context)

    def _get_parser(self):
        """Return None since we use native tool calling instead of parsing."""
        return None

    def _get_prompt_template_path(self) -> Path:
        """Return the path to the prompt template for native tool use."""
        return (
            Path(__file__).parent.parent
            / "prompt-templates"
            / "terminus-kira.txt"
        )

    def _get_error_response_type(self) -> str:
        """Return error response type for native tool use."""
        return "response with valid tool calls"

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Return task completion confirmation message for native tool use."""
        instruction = getattr(self, "_original_instruction", "N/A")
        return (
            f"Original task:\n{instruction}\n\n" \
            f"Current terminal state:\n{terminal_output}\n\n" \
            "Are you sure you want to mark the task as complete?\n\n" \
            "[!] Checklist\n" \
            "- Does your solution meet the requirements in the original task above? [TODO/DONE]\n" \
            "- Does your solution account for potential changes in numeric values, array sizes, file contents, or configuration parameters? [TODO/DONE]\n" \
            "- Have you verified your solution from the all perspectives of a test engineer, a QA engineer, and the user who requested this task?\n" \
            "  - test engineer [TODO/DONE]\n" \
            "  - QA engineer [TODO/DONE]\n" \
            "  - user who requested this task [TODO/DONE]\n\n" \
            "After this point, solution grading will begin and no further edits will be possible. If everything looks good, call task_complete tool again."
        )

    def _limit_output_length(self, output: str, max_bytes: int = 30000) -> str:
        return super()._limit_output_length(output, max_bytes)

    def _extract_tool_calls(self, response) -> list[dict[str, Any]]:
        """Extract tool calls from litellm response."""
        tool_calls = []
        try:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    tc_id = ""
                    tc_type = "function"
                    function_name = ""
                    function_arguments = "{}"

                    if isinstance(tc, dict):
                        tc_id = tc.get("id", "")
                        tc_type = tc.get("type", "function")
                        function = tc.get("function") or {}
                    else:
                        tc_id = getattr(tc, "id", "")
                        tc_type = getattr(tc, "type", "function")
                        function = getattr(tc, "function", {}) or {}

                    if isinstance(function, dict):
                        function_name = function.get("name", "")
                        function_arguments = function.get("arguments", "{}")
                    else:
                        function_name = getattr(function, "name", "")
                        function_arguments = getattr(function, "arguments", "{}")

                    tool_calls.append({
                        "id": tc_id,
                        "type": tc_type,
                        "function": {
                            "name": function_name,
                            "arguments": function_arguments,
                        },
                    })
        except (AttributeError, IndexError):
            pass
        return tool_calls

    @staticmethod
    def _coerce_legacy_arg_value(raw_value: str) -> Any:
        # GLM native tool-call markup carries raw argument text inside
        # <arg_value>...</arg_value>. All currently supported arguments are
        # strings, so preserve the original text instead of coercing JSON
        # scalars like `2` or `true` into int/bool.
        return raw_value.strip()

    @staticmethod
    def _iter_legacy_tool_call_bodies(content: str):
        start_token = "<tool_call>"
        end_token = "</tool_call>"
        search_from = 0
        while True:
            start_idx = content.find(start_token, search_from)
            if start_idx < 0:
                break
            body_start = start_idx + len(start_token)
            end_idx = content.find(end_token, body_start)
            if end_idx >= 0:
                yield content[body_start:end_idx]
            search_from = body_start

    @classmethod
    def _normalize_native_tool_name(cls, raw_name: str) -> str | None:
        normalized = raw_name.strip()
        if normalized in {"execute_commands", "task_complete", "image_read"}:
            return normalized
        return None

    @staticmethod
    def _is_valid_execute_commands_payload(arguments: dict[str, Any]) -> bool:
        if set(arguments.keys()) != {"commands"}:
            return False
        return isinstance(arguments.get("commands"), str)

    @staticmethod
    def _is_valid_task_complete_payload(arguments: dict[str, Any]) -> bool:
        return not arguments

    @staticmethod
    def _is_valid_image_read_payload(arguments: dict[str, Any]) -> bool:
        if set(arguments.keys()) != {"file_path", "image_read_instruction"}:
            return False
        return isinstance(arguments.get("file_path"), str) and isinstance(
            arguments.get("image_read_instruction"),
            str,
        )

    @classmethod
    def _extract_legacy_tool_calls(cls, content: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for match_idx, body in enumerate(
            cls._iter_legacy_tool_call_bodies(content),
            start=1,
        ):
            body = (body or "").strip()
            if not body:
                continue

            # GLM generate mode emits native function-call markup as:
            # <tool_call>execute_commands<arg_key>commands</arg_key><arg_value>pwd</arg_value></tool_call>
            native_match = re.match(r"([^<]+)", body)
            if not native_match:
                continue

            raw_name = native_match.group(1).strip()
            function_name = cls._normalize_native_tool_name(raw_name)
            if not function_name:
                continue

            arg_section = body[native_match.end() :]
            arg_matches = list(
                re.finditer(
                    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                    arg_section,
                    flags=re.DOTALL,
                )
            )
            if arg_section.strip() and not arg_matches:
                continue

            arguments: dict[str, Any] = {}
            for arg_match in arg_matches:
                key = arg_match.group(1).strip()
                if not key:
                    continue
                value = cls._coerce_legacy_arg_value(arg_match.group(2))
                arguments[key] = value

            if function_name == "execute_commands":
                if not cls._is_valid_execute_commands_payload(arguments):
                    continue
            elif function_name == "task_complete":
                if not cls._is_valid_task_complete_payload(arguments):
                    continue
            elif function_name == "image_read":
                if not cls._is_valid_image_read_payload(arguments):
                    continue

            arguments_str = json.dumps(arguments, ensure_ascii=False)
            signature = (function_name, arguments_str)
            if signature in seen:
                continue
            seen.add(signature)
            tool_calls.append(
                {
                    "id": f"salvaged_native_{function_name}_{match_idx}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": arguments_str,
                    },
                }
            )
        return tool_calls

    @staticmethod
    def _strip_legacy_tool_call_markup(content: str | None) -> str:
        if not content:
            return ""
        stripped = re.sub(
            r"<tool_call>\s*[A-Za-z0-9_]+.*?</tool_call>",
            "",
            content,
            flags=re.DOTALL,
        )
        return stripped.strip()

    @classmethod
    def _salvage_tool_calls_from_content(cls, content: str | None) -> list[dict[str, Any]]:
        # Legacy textual salvage is intentionally disabled. Invalid or missing
        # tool calls should surface as format errors instead of being repaired.
        return []

    @staticmethod
    def _render_legacy_tool_call_markup(tool_calls: list[dict[str, Any]]) -> str:
        rendered_blocks: list[str] = []
        for tc in tool_calls:
            function = tc.get("function") or {}
            if not isinstance(function, dict):
                continue
            function_name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(function_name, str) or not function_name:
                continue

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}

            block = [f"<tool_call>{function_name}"]
            for key, value in arguments.items():
                block.append(f"<arg_key>{key}</arg_key>")
                block.append(f"<arg_value>{value}</arg_value>")
            block.append("</tool_call>")
            rendered_blocks.append("".join(block))
        return "".join(rendered_blocks)

    @classmethod
    def _reconstruct_history_content(
        cls,
        content: str,
        reasoning_content: str | None,
        tool_calls: list[dict[str, Any]],
    ) -> str:
        content = content or ""
        reasoning = reasoning_content.strip() if isinstance(reasoning_content, str) else ""
        parts: list[str] = []
        if reasoning:
            parts.append(f"<think>{reasoning}</think>")
            parts.append(content)
        else:
            parts.append(content if content.startswith("<think>") else f"<think>{content}")
        parts.append(cls._render_legacy_tool_call_markup(tool_calls))
        return "".join(parts)

    def _extract_usage_info(self, response) -> UsageInfo | None:
        """Extract usage info from litellm response."""
        try:
            usage = response.usage
            if usage:
                cost = 0.0
                try:
                    cost = litellm.completion_cost(completion_response=response) or 0.0
                except Exception:
                    pass
                return UsageInfo(
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=usage.completion_tokens or 0,
                    cache_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                    cost_usd=cost,
                )
        except (AttributeError, TypeError):
            pass
        return None

    @staticmethod
    def _feedback_to_parse_result(feedback: str):
        errors: list[str] = []
        warnings: list[str] = []
        for raw_line in feedback.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("ERROR:"):
                errors.append(line.removeprefix("ERROR:").strip())
                continue
            if line.startswith("WARNINGS:"):
                warnings.append(line.removeprefix("WARNINGS:").strip())
                continue
            if warnings:
                warnings.append(line)
            elif errors:
                errors.append(line)
        return SimpleNamespace(
            error="\n".join(errors),
            warning="\n".join(warnings),
        )

    def _parse_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        raw_content: str | None = None,
    ) -> tuple[list[Command], bool, str, str, str, ImageReadRequest | None]:
        """Parse tool calls into commands.

        Returns:
            Tuple of (commands, is_task_complete, feedback, analysis, plan, image_read)
        """
        commands = []
        is_task_complete = False
        feedback = ""
        analysis = ""
        plan = ""
        image_read = None

        if not tool_calls:
            if isinstance(raw_content, str) and "<tool_call>" in raw_content:
                feedback = (
                    "WARNINGS: Your response contained tool-call markup, "
                    "but it did not match the required format. "
                    "Use execute_commands with exactly the commands string argument, "
                    "task_complete with no arguments, or image_read with both "
                    "file_path and image_read_instruction."
                )
            else:
                feedback = (
                    "WARNINGS: Your response contained no tool calls. "
                    "Please use execute_commands, task_complete, or image_read."
                )
            return commands, is_task_complete, feedback, analysis, plan, image_read

        def append_warning(message: str) -> None:
            nonlocal feedback
            warning = f"WARNINGS: {message}"
            if not feedback:
                feedback = warning
            elif warning not in feedback:
                feedback += f"\n{warning}"

        def normalize_commands_payload(raw_commands: Any) -> list[dict[str, Any]]:
            if raw_commands is None:
                return []

            if isinstance(raw_commands, dict):
                raw_items = [raw_commands]
            elif isinstance(raw_commands, list):
                raw_items = raw_commands
            elif isinstance(raw_commands, str):
                stripped = raw_commands.strip()
                if not stripped:
                    return []
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return [{"keystrokes": raw_commands}]
                return normalize_commands_payload(parsed)
            else:
                append_warning(
                    "execute_commands.commands must be a list, dict, or string."
                )
                return []

            normalized: list[dict[str, Any]] = []
            for item in raw_items:
                if isinstance(item, dict):
                    normalized.append(item)
                    continue
                if isinstance(item, str):
                    stripped = item.strip()
                    if not stripped:
                        continue
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        normalized.append({"keystrokes": item})
                        continue

                    if isinstance(parsed, dict):
                        normalized.append(parsed)
                    elif isinstance(parsed, list):
                        normalized.extend(normalize_commands_payload(parsed))
                    else:
                        append_warning(
                            "Ignored a command entry with unsupported JSON type."
                        )
                    continue
                if isinstance(item, list):
                    normalized.extend(normalize_commands_payload(item))
                    continue
                append_warning(
                    "Ignored a command entry with unsupported type "
                    f"{type(item).__name__}."
                )

            return normalized

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                self.logger.warning(f"Skipping malformed tool call: {tool_call!r}")
                continue
            function_name = tool_call.get("function", {}).get("name", "")
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")

            try:
                if isinstance(arguments_str, str):
                    arguments = json.loads(arguments_str)
                else:
                    arguments = arguments_str
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                continue

            if not isinstance(arguments, dict):
                self.logger.warning(
                    f"Skipping tool call with non-dict arguments: {arguments!r}"
                )
                continue

            if function_name == "execute_commands":
                analysis = arguments.get("analysis", "")
                plan = arguments.get("plan", "")

                cmds = normalize_commands_payload(arguments.get("commands", []))
                for cmd in cmds:
                    keystrokes = cmd.get("keystrokes", "")
                    if not isinstance(keystrokes, str):
                        append_warning(
                            "Ignored a command with non-string keystrokes."
                        )
                        continue
                    if not keystrokes:
                        append_warning("Ignored an empty command.")
                        continue
                    duration = cmd.get("duration", 1.0)
                    try:
                        duration_sec = float(duration)
                    except (TypeError, ValueError):
                        append_warning(
                            f"Invalid duration '{duration}', defaulting to 1.0."
                        )
                        duration_sec = 1.0

                    commands.append(
                        Command(
                            keystrokes=self._normalize_keystrokes_for_execution(
                                keystrokes
                            ),
                            duration_sec=min(max(duration_sec, 0.0), 60.0),
                        )
                    )
            elif function_name == "task_complete":
                # Mark task as complete
                is_task_complete = True
            elif function_name == "image_read":
                # Extract image read request
                file_path = arguments.get("file_path", "")
                instruction = arguments.get("image_read_instruction", "")
                if file_path and instruction:
                    image_read = ImageReadRequest(
                        file_path=file_path,
                        image_read_instruction=instruction,
                    )
                else:
                    feedback = (
                        "WARNINGS: image_read requires both file_path and "
                        "image_read_instruction arguments."
                    )
            else:
                # Unknown function name - provide feedback
                feedback = (
                    f"WARNINGS: Unknown function '{function_name}'. "
                    "Please use execute_commands, task_complete, or image_read."
                )
                self.logger.warning(f"Unknown function called: {function_name}")

        return commands, is_task_complete, feedback, analysis, plan, image_read

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(
                (
                    BadRequestError,
                    LiteLLMAuthenticationError,
                    ContextLengthExceededError,
                    OutputLengthExceededError,
                )
            )
        ),
        reraise=True,
    )
    async def _call_llm_for_image(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> object:
        """Call litellm.acompletion with retry for transient errors.

        Retries on rate limit, network, and server errors.
        Does NOT retry on BadRequestError (e.g. image too large).
        """
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": 900,  # 15 minutes timeout, retry on timeout
            "drop_params": True,
        }
        # Image analysis doesn't need high reasoning effort
        # Skip reasoning_effort to use default (faster response)
        return await litellm.acompletion(**kwargs)

    async def _execute_image_read(
        self,
        image_read: ImageReadRequest,
        chat: Chat,
        original_instruction: str = "",
    ) -> str:
        """Execute a file read command to analyze an image file.

        Reads the file from the container via base64, sends it as a multimodal
        message to the LLM, and returns the analysis result.
        """
        if self._session is None:
            raise RuntimeError("Session is not set")

        file_path = image_read.file_path

        # Read image from container as base64 via harbor environment exec
        result = await self._with_block_timeout(
            self._session.environment.exec(command=f"base64 {file_path}")
        )
        if result.return_code != 0:
            error_output = result.stderr or ""
            return f"ERROR: Failed to read file '{file_path}': {error_output}"

        b64 = (result.stdout or "").replace("\n", "")

        # Determine MIME type from file extension
        ext = Path(file_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime = mime_map.get(ext)
        if mime is None:
            return (
                f"ERROR: Unsupported image format '{ext}'. "
                f"Convert to PNG first (e.g. convert image{ext} to image.png), "
                f"then use `image_read` on the PNG file."
            )

        # Construct multimodal user message
        multimodal_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image_read.image_read_instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ]

        messages = add_anthropic_caching(multimodal_messages, self._model_name)

        # Call LLM with retry logic
        try:
            response = await self._call_llm_for_image(
                messages=messages,
                model=self._model_name,
                temperature=self._temperature,
                max_tokens=self._llm.get_model_output_limit(),
            )
        except Exception as e:
            return f"ERROR: {e}"

        response_text = response["choices"][0]["message"]["content"]

        # Manually update token counts from litellm response
        usage = response.get("usage", {})
        if usage:
            chat._cumulative_input_tokens += usage.get("prompt_tokens", 0)
            chat._cumulative_output_tokens += usage.get("completion_tokens", 0)
            prompt_details = usage.get("prompt_tokens_details")
            cached = (
                getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
            )
            chat._cumulative_cache_tokens += cached or 0

        return f"File Read Result for '{file_path}':\n{response_text}"

    async def _call_llm_with_tools(
        self,
        messages: list[dict],
    ) -> ToolCallResponse:
        """Call LLM directly with tools parameter.

        This bypasses harbor's Chat class to get access to tool_calls.
        """
        if (
            getattr(self._llm, "_use_generate", False)
            and hasattr(self._llm, "generate_with_tool_calls")
        ):
            llm_response, tool_calls = await self._llm.generate_with_tool_calls(
                messages=messages,
                tools=TOOLS,
                logging_path=None,
            )
            return ToolCallResponse(
                content=llm_response.content,
                tool_calls=tool_calls,
                raw_content=llm_response.raw_content,
                reasoning_content=llm_response.reasoning_content,
                history_content=llm_response.history_content,
                usage=llm_response.usage,
                terminal_stop_token=llm_response.terminal_stop_token,
                finish_reason=llm_response.finish_reason,
            )
        if getattr(self._llm, "_use_generate", False):
            response = await self._llm.call(
                prompt=None,
                message_history=messages,
                session_id_override=self._session_id,
                tools=TOOLS,
            )
            raw_content = response.history_content
            if not isinstance(raw_content, str):
                raw_content = response.content or ""
            tool_calls = self._extract_legacy_tool_calls(raw_content)
            reasoning_content = response.reasoning_content
            if tool_calls:
                content = response.content or ""
                content = self._strip_legacy_tool_call_markup(content)
            else:
                content = raw_content
                if "<think>" in raw_content or "</think>" in raw_content:
                    reasoning_content = None
            return ToolCallResponse(
                content=content,
                tool_calls=tool_calls,
                raw_content=response.raw_content,
                reasoning_content=reasoning_content,
                history_content=raw_content,
                usage=response.usage,
                terminal_stop_token=getattr(response, "terminal_stop_token", None),
                finish_reason=getattr(response, "finish_reason", None),
            )

        return await self._call_llm_with_tools_completion(messages)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(
                (
                    BadRequestError,
                    LiteLLMAuthenticationError,
                    ContextLengthExceededError,
                    OutputLengthExceededError,
                    ParseError,
                )
            )
        ),
        reraise=True,
    )
    async def _call_llm_with_tools_completion(
        self,
        messages: list[dict],
    ) -> ToolCallResponse:
        # Apply Anthropic caching
        messages = add_anthropic_caching(messages, self._model_name)

        # Build completion kwargs
        completion_kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "tools": TOOLS,
            "timeout": 900,  # 15 minutes timeout, retry on timeout
            "drop_params": True,
            "api_key": self._api_key or "none",
        }

        # Add api_base if available
        if hasattr(self._llm, "_api_base") and self._llm._api_base:
            completion_kwargs["api_base"] = self._normalize_openai_api_base(
                self._llm._api_base
            )
        if hasattr(self._llm, "_request_timeout_sec") and self._llm._request_timeout_sec:
            completion_kwargs["timeout"] = self._llm._request_timeout_sec
        llm_top_p = getattr(self._llm, "_top_p", None)
        if llm_top_p is not None:
            completion_kwargs["top_p"] = llm_top_p
        if hasattr(self._llm, "_max_new_tokens") and self._llm._max_new_tokens is not None:
            completion_kwargs["max_tokens"] = self._llm._max_new_tokens
        if self._session_id is not None:
            completion_kwargs["extra_body"] = {"session_id": self._session_id}

        # Add reasoning effort if available
        # When reasoning_effort is set, temperature MUST be 1 (API requirement)
        if self._reasoning_effort:
            completion_kwargs["reasoning_effort"] = self._reasoning_effort
            completion_kwargs["temperature"] = 1

        try:
            response = await litellm.acompletion(**completion_kwargs)
        except LiteLLMContextWindowExceededError:
            raise ContextLengthExceededError()
        except asyncio.CancelledError as exc:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            raise ToolCallTransportError(
                "Native tool-calling request was cancelled while waiting for the "
                "provider response; retrying as a transient transport failure."
            ) from exc

        # Extract response data
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = self._extract_tool_calls(response)
        usage_info = self._extract_usage_info(response)

        # Check for truncation
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            raise OutputLengthExceededError(
                "Response was truncated due to max tokens limit",
                truncated_response=content,
            )

        # Extract reasoning content (for models that support it)
        reasoning_content = None
        if hasattr(message, "reasoning_content"):
            reasoning_content = message.reasoning_content

        parser_fn = getattr(self._llm, "_parse_content_protocol_text", None)
        if callable(parser_fn):
            content, reasoning_content, parsed_tool_calls = parser_fn(
                content,
                reasoning_content,
            )
            if parsed_tool_calls:
                tool_calls = parsed_tool_calls
        history_content = self._reconstruct_history_content(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )

        return ToolCallResponse(
            content=content,
            tool_calls=tool_calls,
            raw_content=content,
            reasoning_content=reasoning_content,
            history_content=history_content,
            usage=usage_info,
        )

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str | None,
        prompt_role: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> tuple[
        list[Command],
        bool,
        str,
        str,
        str,
        LLMResponse,
        ImageReadRequest | None,
        list[dict[str, Any]],
    ]:
        """Handle LLM interaction using native tool calling.

        This overrides the parent's _handle_llm_interaction to use native tools
        instead of JSON/XML parsing.
        """
        _, prompt_path, response_path = logging_paths

        if prompt_path is not None:
            prompt_path.write_text(prompt or "")

        # Build messages from chat history + new prompt
        messages = self._prepare_messages_for_llm_call(chat.messages.copy())
        if prompt is not None:
            messages.append({"role": prompt_role, "content": prompt})

        try:
            llm_call_started_at = datetime.now(timezone.utc)
            start_time = time.time()
            try:
                tool_response = await self._call_llm_with_tools(messages)
            except Exception as exc:
                self._append_profiling_span(
                    category="generate",
                    label="main_generate",
                    started_at=llm_call_started_at,
                    finished_at=datetime.now(timezone.utc),
                    metadata={
                        "episode": self._n_episodes,
                        "prompt_role": prompt_role,
                        "raised": type(exc).__name__,
                    },
                )
                raise
            end_time = time.time()
            request_time_ms = (end_time - start_time) * 1000
            self._api_request_times.append(request_time_ms)
            self._append_profiling_span(
                category="generate",
                label="main_generate",
                started_at=llm_call_started_at,
                finished_at=datetime.now(timezone.utc),
                    metadata={
                        "episode": self._n_episodes,
                        "prompt_role": prompt_role,
                        "tool_call_count": len(tool_response.tool_calls),
                        "finish_reason": getattr(tool_response, "finish_reason", None),
                        "terminal_stop_token": getattr(
                            tool_response, "terminal_stop_token", None
                        ),
                    },
                )

            self._capture_rendered_generate_system_prompt(chat)

            new_messages = []
            if prompt is not None:
                new_messages.append(
                    self._build_prompt_history_message(prompt_role, prompt)
                )
            new_messages.append(
                self._build_assistant_history_message(
                    tool_response.content,
                    tool_response.tool_calls,
                    tool_response.reasoning_content,
                    tool_response.history_content,
                    tool_response.usage.completion_tokens if tool_response.usage else 0,
                )
            )
            chat.extend_messages(
                new_messages,
                token_counter=self._count_tokens_for_messages,
            )
            chat.reset_response_chain()

            # Update cumulative metrics
            if tool_response.usage:
                chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
                chat._cumulative_output_tokens += tool_response.usage.completion_tokens
                chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
                chat._cumulative_cost += tool_response.usage.cost_usd

        except ContextLengthExceededError:
            if not self._enable_summarize:
                self.logger.debug("Context length exceeded and summarization is OFF.")
                raise

            self.logger.debug("Context length exceeded. Using fallback summarization.")

            if session is None:
                raise RuntimeError("Cannot handle context length error without session")

            summary_prompt = None
            try:
                summary_prompt, subagent_refs = await self._with_block_timeout(
                    self._summarize(chat, original_instruction, session)
                )
                self._pending_subagent_refs = subagent_refs
                self._pending_handoff_prompt = summary_prompt
            except ContextManagementLimitExceededError:
                raise
            except Exception as e:
                self.logger.debug(f"SUMMARIZATION failed: {e}")

            if summary_prompt is None:
                current_screen = await self._with_block_timeout(
                    session.capture_pane(capture_entire=False)
                )
                limited_screen = current_screen[-1000:] if current_screen else ""
                summary_prompt = (
                    f"{original_instruction}\n\nCurrent state: {limited_screen}"
                )

            # Retry with summarized context
            messages = self._prepare_messages_for_llm_call(chat.messages.copy())
            if summary_prompt is not None:
                messages.append({"role": "user", "content": summary_prompt})

            while True:
                llm_call_started_at = datetime.now(timezone.utc)
                start_time = time.time()
                try:
                    tool_response = await self._call_llm_with_tools(messages)
                    break
                except ContextLengthExceededError as exc:
                    self._append_profiling_span(
                        category="generate",
                        label="main_generate_after_summary",
                        started_at=llm_call_started_at,
                        finished_at=datetime.now(timezone.utc),
                        metadata={
                            "episode": self._n_episodes,
                            "prompt_role": "user",
                            "raised": type(exc).__name__,
                        },
                    )
                    self._ensure_context_management_budget(
                        trigger="context_length_after_summary"
                    )
                    try:
                        summary_prompt, subagent_refs = await self._with_block_timeout(
                            self._summarize(chat, original_instruction, session)
                        )
                        self._pending_subagent_refs = subagent_refs
                        self._pending_handoff_prompt = summary_prompt
                    except ContextManagementLimitExceededError:
                        raise
                    except Exception as summary_exc:
                        self.logger.debug(
                            f"SUMMARIZATION failed after context length retry: {summary_exc}"
                        )
                        raise exc

                    if summary_prompt is None:
                        raise exc

                    messages = self._prepare_messages_for_llm_call(
                        chat.messages.copy()
                    )
                    messages.append({"role": "user", "content": summary_prompt})
                    continue
                except Exception as exc:
                    self._append_profiling_span(
                        category="generate",
                        label="main_generate_after_summary",
                        started_at=llm_call_started_at,
                        finished_at=datetime.now(timezone.utc),
                        metadata={
                            "episode": self._n_episodes,
                            "prompt_role": "user",
                            "raised": type(exc).__name__,
                        },
                    )
                    raise
            end_time = time.time()
            request_time_ms = (end_time - start_time) * 1000
            self._api_request_times.append(request_time_ms)
            self._append_profiling_span(
                category="generate",
                label="main_generate_after_summary",
                started_at=llm_call_started_at,
                finished_at=datetime.now(timezone.utc),
                metadata={
                    "episode": self._n_episodes,
                    "prompt_role": "user",
                    "tool_call_count": len(tool_response.tool_calls),
                    "finish_reason": getattr(tool_response, "finish_reason", None),
                    "terminal_stop_token": getattr(
                        tool_response, "terminal_stop_token", None
                    ),
                },
            )
            self._capture_rendered_generate_system_prompt(chat)

            new_messages = []
            if summary_prompt is not None:
                new_messages.append(
                    self._build_prompt_history_message("user", summary_prompt)
                )
            new_messages.append(
                self._build_assistant_history_message(
                    tool_response.content,
                    tool_response.tool_calls,
                    tool_response.reasoning_content,
                    tool_response.history_content,
                    tool_response.usage.completion_tokens if tool_response.usage else 0,
                )
            )
            chat.extend_messages(
                new_messages,
                token_counter=self._count_tokens_for_messages,
            )
            chat.reset_response_chain()

            # Update cumulative metrics
            if tool_response.usage:
                chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
                chat._cumulative_output_tokens += tool_response.usage.completion_tokens
                chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
                chat._cumulative_cost += tool_response.usage.cost_usd

        except OutputLengthExceededError as e:
            self.logger.debug(f"Output length exceeded: {e}")

            if self._training_mode:
                self._append_truncated_assistant_to_chat(
                    chat=chat,
                    prompt=prompt,
                    prompt_role=prompt_role,
                    truncated_response=getattr(e, "truncated_response", None),
                )
                raise SingleTurnOverlongError(str(e)) from e

            error_msg = (
                "ERROR!! Your response was truncated. "
                "Please provide a shorter response with fewer commands."
            )

            retry_messages = []
            if prompt is not None:
                retry_messages.append(
                    {"role": prompt_role, "content": prompt, "num_tokens": 0}
                )
            retry_messages.extend(
                [
                    {"role": "assistant", "content": "[truncated]", "num_tokens": 0},
                    {"role": "tool", "content": error_msg, "num_tokens": 0},
                ]
            )
            chat.extend_messages(
                retry_messages,
                token_counter=self._count_tokens_for_messages,
            )
            chat.reset_response_chain()

            # Retry
            messages = self._prepare_messages_for_llm_call(chat.messages.copy())
            llm_call_started_at = datetime.now(timezone.utc)
            start_time = time.time()
            try:
                tool_response = await self._call_llm_with_tools(messages)
            except Exception as exc:
                self._append_profiling_span(
                    category="generate",
                    label="main_generate_after_truncation",
                    started_at=llm_call_started_at,
                    finished_at=datetime.now(timezone.utc),
                    metadata={
                        "episode": self._n_episodes,
                        "prompt_role": prompt_role,
                        "raised": type(exc).__name__,
                    },
                )
                raise
            end_time = time.time()
            self._api_request_times.append((end_time - start_time) * 1000)
            self._append_profiling_span(
                category="generate",
                label="main_generate_after_truncation",
                started_at=llm_call_started_at,
                finished_at=datetime.now(timezone.utc),
                metadata={
                    "episode": self._n_episodes,
                    "prompt_role": prompt_role,
                    "tool_call_count": len(tool_response.tool_calls),
                    "finish_reason": getattr(tool_response, "finish_reason", None),
                    "terminal_stop_token": getattr(
                        tool_response, "terminal_stop_token", None
                    ),
                },
            )

            self._capture_rendered_generate_system_prompt(chat)

            assistant_message = self._build_assistant_history_message(
                tool_response.content,
                tool_response.tool_calls,
                tool_response.reasoning_content,
                tool_response.history_content,
                tool_response.usage.completion_tokens if tool_response.usage else 0,
            )
            chat.append_message(
                assistant_message,
                token_counter=self._count_tokens_for_messages,
            )
            chat.reset_response_chain()

            # Update cumulative metrics
            if tool_response.usage:
                chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
                chat._cumulative_output_tokens += tool_response.usage.completion_tokens
                chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
                chat._cumulative_cost += tool_response.usage.cost_usd

        self._resume_generation_from_existing_user = False
        self._mark_post_context_management_generate_complete()

        # Log response
        if response_path is not None:
            response_text = (
                f"Raw Content: {tool_response.history_content or ''}\n\n"
                f"Content: {tool_response.content or ''}\n\n"
                f"Tool Calls: {json.dumps(tool_response.tool_calls, indent=2)}"
            )
            response_path.write_text(response_text)

        # Parse tool calls into commands
        commands, is_task_complete, feedback, analysis, plan, image_read = self._parse_tool_calls(
            tool_response.tool_calls,
            raw_content=tool_response.history_content or tool_response.content,
        )
        if (
            self._training_mode
            and tool_response.terminal_stop_token
            and tool_response.terminal_stop_token != "<|observation|>"
            and tool_response.finish_reason not in {"length", "max_tokens", "max_new_tokens"}
        ):
            stop_token_feedback = (
                "WARNINGS: Generate stopped on an unexpected stop token "
                f"{tool_response.terminal_stop_token!r} instead of '<|observation|>'."
            )
            feedback = (
                f"{feedback}\n{stop_token_feedback}".strip()
                if feedback
                else stop_token_feedback
            )
        format_result = self._feedback_to_parse_result(feedback)
        self._record_format_issue(format_result)
        self._maybe_raise_format_error(format_result)

        if self._training_mode:
            for command in commands:
                detection = detect_pager_usage(command.keystrokes)
                if detection.is_pager_usage:
                    self._pager_usage_detected = {
                        "status": PAGER_USAGE_STATUS,
                        "reason": detection.reason or "explicit pager usage",
                        "command": detection.command or command.keystrokes,
                    }
                    raise PagerUsageError(
                        f"{self._pager_usage_detected['reason']}: "
                        f"{self._pager_usage_detected['command']}"
                    )

        # Create LLMResponse for compatibility with parent class
        llm_response = LLMResponse(
            content=tool_response.content or "",
            raw_content=tool_response.raw_content,
            tool_calls=tool_response.tool_calls,
            reasoning_content=tool_response.reasoning_content,
            history_content=tool_response.history_content or tool_response.content or "",
            usage=tool_response.usage,
            terminal_stop_token=tool_response.terminal_stop_token,
            finish_reason=tool_response.finish_reason,
        )

        return (
            commands,
            is_task_complete,
            feedback,
            analysis,
            plan,
            llm_response,
            image_read,
            tool_response.tool_calls,
        )

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: Chat,
        logging_dir: Path | None = None,
        original_instruction: str = "",
    ) -> int:
        """Run the agent loop with support for image_read tool."""
        if self._context is None:
            raise RuntimeError("Agent context is not set. This should never happen.")

        if self._session is None:
            raise RuntimeError("Session is not set. This should never happen.")

        prompt = initial_prompt
        prompt_role = "user"

        self._context.n_input_tokens = 0
        self._context.n_output_tokens = 0
        self._context.n_cache_tokens = 0
        self._context.cost_usd = None

        for episode in range(self._max_episodes):
            self._n_episodes = episode + 1
            if not await self._with_block_timeout(self._session.is_session_alive()):
                self.logger.debug("Session has ended, breaking out of agent loop")
                return episode + 1

            if original_instruction and self._enable_summarize:
                proactive_summary_result = await self._with_block_timeout(
                    self._check_proactive_summarization(
                        chat,
                        original_instruction,
                        self._session,
                        prompt,
                    )
                )
                if proactive_summary_result:
                    prompt, subagent_refs = proactive_summary_result
                    prompt_role = "user"
                    self._pending_subagent_refs = subagent_refs
                    self._pending_handoff_prompt = prompt

            logging_paths = self._setup_episode_logging(logging_dir, episode)

            # Track token counts and cost before this step
            tokens_before_input = chat.total_input_tokens
            tokens_before_output = chat.total_output_tokens
            tokens_before_cache = chat.total_cache_tokens
            cost_before = chat.total_cost

            (
                commands,
                is_task_complete,
                feedback,
                analysis,
                plan,
                llm_response,
                image_read,
                raw_tool_calls,
            ) = await self._handle_llm_interaction(
                chat,
                prompt,
                prompt_role,
                logging_paths,
                original_instruction,
                self._session,
            )

            # If we have pending subagent refs, add a system step
            if self._pending_subagent_refs:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="Performed context summarization and handoff to continue task.",
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    subagent_trajectory_ref=self._pending_subagent_refs
                                )
                            ]
                        ),
                    )
                )
                self._pending_subagent_refs = None

            if self._pending_handoff_prompt:
                if self._linear_history:
                    self._split_trajectory_on_summarization(
                        self._pending_handoff_prompt
                    )
                else:
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="user",
                            message=self._pending_handoff_prompt,
                        )
                    )
                self._pending_handoff_prompt = None

            # Create message content
            if self._save_raw_content_in_trajectory:
                message_content = llm_response.raw_content or llm_response.content
            else:
                message_parts = []
                if analysis:
                    message_parts.append(f"Analysis: {analysis}")
                if plan:
                    message_parts.append(f"Plan: {plan}")
                if message_parts:
                    message_content = "\n".join(message_parts)
                else:
                    message_content = llm_response.raw_content or llm_response.content

            self._context.n_input_tokens = chat.total_input_tokens
            self._context.n_output_tokens = chat.total_output_tokens
            self._context.n_cache_tokens = chat.total_cache_tokens
            self._context.cost_usd = chat.total_cost if chat.total_cost > 0 else None

            self._record_asciinema_marker(
                f"Episode {episode}: {len(commands)} commands"
                + (" (image_read)" if image_read else ""),
            )

            if feedback and "ERROR:" in feedback:
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    f"Please fix these issues and provide a proper "
                    f"{self._get_error_response_type()}."
                )
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=llm_response.raw_content or llm_response.content,
                        reasoning_content=llm_response.reasoning_content,
                        observation=Observation(
                            results=[ObservationResult(content=prompt)]
                        ),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens - tokens_before_output,
                            cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                continue

            if image_read is not None:
                # File read path
                image_read_started_at = datetime.now(timezone.utc)
                image_read_result = await self._execute_image_read(
                    image_read, chat, original_instruction
                )
                self._append_profiling_span(
                    category="tool_call",
                    label=self._profile_preview(
                        f"image_read {image_read.file_path} :: {image_read.image_read_instruction}"
                    ),
                    started_at=image_read_started_at,
                    finished_at=datetime.now(timezone.utc),
                    metadata={
                        "episode": self._n_episodes,
                        "tool_name": "image_read",
                        "file_path": image_read.file_path,
                    },
                )

                # Capture pending state before modifying
                was_pending_completion = self._pending_completion

                # Handle task completion with double confirmation
                if is_task_complete:
                    if self._pending_completion:
                        observation = image_read_result
                    else:
                        self._pending_completion = True
                        observation = self._get_completion_confirmation_message(
                            image_read_result
                        )
                else:
                    self._pending_completion = False
                    if feedback and "WARNINGS:" in feedback:
                        observation = (
                            f"Previous response had warnings:\n{feedback}\n\n"
                            f"{image_read_result}"
                        )
                    else:
                        observation = image_read_result

                # Build tool_calls for image_read
                tool_calls_list = self._to_trajectory_tool_calls(raw_tool_calls)
                observation_results: list[ObservationResult] = []

                if not tool_calls_list:
                    tool_calls_list.append(
                        ToolCall(
                            tool_call_id=f"call_{episode}_image_read",
                            function_name="image_read",
                            arguments={
                                "file_path": image_read.file_path,
                                "image_read_instruction": image_read.image_read_instruction,
                            },
                        )
                    )
                    if is_task_complete:
                        tool_calls_list.append(
                            ToolCall(
                                tool_call_id=f"call_{episode}_task_complete",
                                function_name="mark_task_complete",
                                arguments={},
                            )
                        )

                if not self._save_raw_content_in_trajectory:
                    observation_results.append(ObservationResult(content=observation))
                else:
                    observation_results.append(ObservationResult(content=observation))

                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=message_content,
                        reasoning_content=llm_response.reasoning_content,
                        tool_calls=tool_calls_list or None,
                        observation=Observation(results=observation_results),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens - tokens_before_output,
                            cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                appended_tool_observation = self._append_tool_observation_to_chat(
                    chat,
                    observation,
                    tool_calls_list,
                    token_counter=self._count_tokens_for_messages,
                )
                self._dump_trajectory()

                if is_task_complete and was_pending_completion:
                    return episode + 1

                prompt = None if appended_tool_observation else observation
                prompt_role = "tool"
            else:
                # Commands path (existing behavior)
                timeout_occurred, terminal_output = await self._with_block_timeout(
                    self._execute_commands(
                        commands,
                        self._session,
                    )
                )

                was_pending_completion = self._pending_completion

                if is_task_complete:
                    if self._pending_completion:
                        observation = terminal_output
                    else:
                        self._pending_completion = True
                        observation = self._get_completion_confirmation_message(
                            terminal_output
                        )
                else:
                    self._pending_completion = False
                    if feedback and "WARNINGS:" in feedback:
                        observation = (
                            f"Previous response had warnings:\n{feedback}\n\n"
                            f"{self._limit_output_length(terminal_output)}"
                        )
                    else:
                        observation = self._limit_output_length(terminal_output)

                # Record trajectory step
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before

                tool_calls_list = self._to_trajectory_tool_calls(raw_tool_calls)
                if not tool_calls_list:
                    if commands:
                        for i, cmd in enumerate(commands):
                            tool_calls_list.append(
                                ToolCall(
                                    tool_call_id=f"call_{episode}_{i + 1}",
                                    function_name="bash_command",
                                    arguments={
                                        "keystrokes": cmd.keystrokes,
                                        "duration": cmd.duration_sec,
                                    },
                                )
                            )
                    if is_task_complete:
                        tool_calls_list.append(
                            ToolCall(
                                tool_call_id=f"call_{episode}_task_complete",
                                function_name="mark_task_complete",
                                arguments={},
                            )
                        )

                tool_calls: list[ToolCall] | None = None
                observation_results: list[ObservationResult] = []

                if not self._save_raw_content_in_trajectory:
                    if commands:
                        observation_results.append(ObservationResult(content=observation))
                    if is_task_complete and not commands:
                        observation_results.append(ObservationResult(content=observation))
                    elif not is_task_complete and not commands:
                        observation_results.append(ObservationResult(content=observation))
                    tool_calls = tool_calls_list or None
                else:
                    observation_results.append(ObservationResult(content=observation))
                    tool_calls = tool_calls_list or None

                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=message_content,
                        reasoning_content=llm_response.reasoning_content,
                        tool_calls=tool_calls,
                        observation=Observation(results=observation_results),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens - tokens_before_output,
                            cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                appended_tool_observation = self._append_tool_observation_to_chat(
                    chat,
                    observation,
                    tool_calls_list,
                    token_counter=self._count_tokens_for_messages,
                )
                self._dump_trajectory()

                if is_task_complete and was_pending_completion:
                    return episode + 1

                prompt = None if appended_tool_observation else observation
                prompt_role = "tool"

        return self._n_episodes
