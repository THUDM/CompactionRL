import copy
import json
import logging
import os
import re
import shlex
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlparse

from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)

from harbor.agents.base import BaseAgent
from harbor.agents.terminus_2.terminus_json_plain_parser import (
    TerminusJSONPlainParser,
)
from harbor.agents.terminus_2.terminus_xml_plain_parser import (
    TerminusXMLPlainParser,
)
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMBackend,
    LLMResponse,
    OutputLengthExceededError,
    ParseError,
)
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
from harbor.llms.utils import prepare_messages_for_generate
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.agent.rollout_detail import RolloutDetail
from harbor.models.agent.trajectory_config import TrajectoryConfig
from harbor.models.metric import UsageInfo
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.logger import logger
from harbor.utils.trajectory_utils import format_trajectory_json


@dataclass
class Command:
    keystrokes: str
    duration_sec: float


@dataclass
class SubagentMetrics:
    """Metrics for subagent operations (summarization, etc.)."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0


class ContextManagementLimitExceededError(ContextLengthExceededError):
    """Raised when another context-management action is required but disallowed."""


class SummaryGenerationFailedError(RuntimeError):
    """Raised when full summary generation repeatedly fails."""


def _normalize_summary_training_mode(value: Any, default: str = "main_chat") -> str:
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


def _normalize_context_management_mode(value: Any, default: str = "auto") -> str:
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


class Terminus2(BaseAgent):
    SUPPORTS_ATIF: bool = True
    _TOLERATED_WARNING_SUBSTRINGS = (
        "Extra text detected before JSON object",
        "Extra text detected after JSON object",
        "Missing duration field, using default 1.0",
        "Missing duration attribute, using default 1.0",
        "Fields appear in wrong order.",
        "Sections appear in wrong order.",
        "should end with newline when followed by another command",
    )
    _CONTEXT_WINDOW_STATUS_HEADER = "Context window status:"
    _CLAUDE_QUERY_BASE_URL = os.environ.get("SUMMARY_API_BASE", "http://127.0.0.1:31200")
    _CLAUDE_QUERY_API_KEY = os.environ.get("SUMMARY_API_KEY", "none")
    _CLAUDE_QUERY_THINKING_TYPE = "adaptive"
    _CLAUDE_QUERY_MAX_TOKENS = 16384

    def _init_llm(
        self,
        llm_backend: LLMBackend | str,
        model_name: str,
        temperature: float,
        top_p: float | None,
        max_new_tokens: int | None,
        collect_rollout_details: bool,
        llm_kwargs: dict | None,
        # LiteLLM-specific args
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
        """Initialize the LLM backend based on llm_backend parameter.

        Args:
            llm_backend: The LLM backend to use.
            model_name: Name of the model.
            temperature: Sampling temperature.
            top_p: Top-p sampling value.
            max_new_tokens: Maximum new tokens to generate.
            collect_rollout_details: Whether to collect token IDs and logprobs.
            llm_kwargs: Additional kwargs passed to the LLM constructor.
            api_base: Base URL for LiteLLM API endpoint.
            session_id: Session ID for LiteLLM.
            max_thinking_tokens: Max thinking tokens for LiteLLM extended thinking.
            reasoning_effort: Reasoning effort level for LiteLLM.
            model_info: Model info dict for LiteLLM custom models.
            tokenizer_path: Optional tokenizer path for generate mode and counting.
            chat_template_path: Optional path to a chat template file used when the
                tokenizer does not ship one or when you want to override it.
            use_generate: Whether to use the generate endpoint path.
            api_key: API key for model requests.
            request_timeout_sec: Per-request timeout for LLM calls.
            use_responses_api: Whether to use the Responses API.

        Returns:
            An initialized LLM instance.

        Raises:
            ValueError: If llm_backend is not a recognized backend.
        """
        # Normalize enum to string value for matching
        backend_value = (
            llm_backend.value if isinstance(llm_backend, LLMBackend) else llm_backend
        )

        match backend_value:
            case LLMBackend.LITELLM.value:
                return LiteLLM(
                    model_name=model_name,
                    api_base=api_base,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    collect_rollout_details=collect_rollout_details,
                    session_id=session_id,
                    max_thinking_tokens=max_thinking_tokens,
                    reasoning_effort=reasoning_effort,
                    model_info=model_info,
                    tokenizer_path=tokenizer_path,
                    chat_template_path=chat_template_path,
                    use_generate=use_generate,
                    api_key=api_key,
                    request_timeout_sec=request_timeout_sec,
                    use_responses_api=use_responses_api,
                    **(llm_kwargs or {}),
                )
            case _:
                raise ValueError(
                    f"Unknown llm_backend: {llm_backend!r}. "
                    f"Supported backends: {[b.value for b in LLMBackend]}"
                )

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_turns: int | None = None,
        parser_name: str = "json",
        api_base: str | None = None,
        summary_api_base: str | None = None,
        temperature: float = 0.7,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "default"]
        | None = None,
        collect_rollout_details: bool = False,
        session_id: str | None = None,
        enable_summarize: Literal["none", "auto", "tool"] | bool = "auto",
        summarization_mode: Literal["summarize", "truncate"] = "summarize",
        summary_training: Literal["main_chat", "summary", "full"] | bool = "main_chat",
        summary_history_user_turns: int = 2,
        enable_tito: bool = False,
        proactive_summarization_threshold: int = 8000,
        context_management_max_count: int = -1,
        max_thinking_tokens: int | None = None,
        model_info: dict | None = None,
        tokenizer_path: str | None = None,
        chat_template_path: str | None = None,
        use_generate: bool = False,
        api_key: str | None = None,
        llm_request_timeout_sec: float | None = 600.0,
        trajectory_config: TrajectoryConfig | None = None,
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        store_all_messages: bool = False,
        record_terminal_session: bool = True,
        interleaved_thinking: bool = False,
        suppress_max_turns_warning: bool = False,
        use_responses_api: bool = False,
        llm_backend: LLMBackend | str = LLMBackend.LITELLM,
        llm_kwargs: dict | None = None,
        llm_call_kwargs: dict[str, Any] | None = None,
        llm_query_type: str | None = None,
        episode_log_mode: Literal["all", "last"] = "last",
        llm: BaseLLM | None = None,
        *args,
        **kwargs,
    ):
        """Initialize Terminus 2 agent.

        Args:
            logs_dir: Directory to store logs
            model_name: Name of the model to use
            max_episodes: Maximum number of episodes (default: 1000000)
            parser_name: Parser to use - "json" or "xml" (default: "json")
            api_base: Base URL for the API endpoint
            summary_api_base: Optional base URL for summary-only LLM calls. When
                omitted, summary requests use the same API base as the main chat.
            temperature: Sampling temperature (default: 0.7)
            top_p: Top-p sampling value (default: None)
            max_new_tokens: Maximum new tokens to generate (default: None)
            reasoning_effort: Qualitative or quantitative measure of effort (default: None)
            collect_rollout_details: Whether to collect detailed rollout data including token IDs.
                NOTE: Rollout details will be incomplete if context summarization occurs.
                See class docstring for details. (default: False)
            session_id: Session ID for the agent (default: None)
            enable_summarize: Context-management mode.
                "none" disables context management, "auto" restores the legacy
                automatic behavior, and "tool" uses the explicit
                context-management tool flow. Boolean values are accepted for
                backward compatibility: true -> "auto", false -> "none".
                (default: "auto")
            summarization_mode: Context management mode when the model requests it.
                "summarize" uses multi-step subagent summarization, while "truncate"
                drops older conversation turns to fit the context window. (default: "summarize")
            summary_training: Controls which data is exported for training.
                "main_chat" keeps only the main conversation, "summary" exports
                only summary subagent pairs, and "full" exports both. Boolean
                values are accepted for backward compatibility. (default: "main_chat")
            summary_history_user_turns: Number of recent `(assistant tool call,
                tool result)` pairs from the pre-summary main chat to keep after
                summary compaction. The final assistant that requested context
                management is excluded because it is not part of the rebuilt
                history tail. (default: 2)
            enable_tito: Whether TiTO middleware is enabled and requires rolling
                session/main keys on context management boundaries. (default: False)
            proactive_summarization_threshold: Number of free tokens below which to trigger
                proactive summarization. Set to 0 to disable proactive summarization. (default: 8000)
            context_management_max_count: Maximum number of context-management actions
                (summaries + truncations). Use -1 for no limit. (default: -1)
            max_thinking_tokens: Maximum thinking tokens for Anthropic extended thinking mode.
                Minimum value is 1024. (default: None)
            model_info: Optional dict containing model information for custom models.
                Used to register the model with litellm. Common fields include:
                - max_input_tokens: Maximum input tokens (context length)
                - max_output_tokens: Maximum output tokens
                - input_cost_per_token: Cost per input token (optional)
                - output_cost_per_token: Cost per output token (optional)
                (default: None)
            tokenizer_path: Optional tokenizer path for tokenizer-based token counting
                and generate-mode prompt formatting. (default: None)
            chat_template_path: Optional file path for a Jinja chat template used to
                override or supply tokenizer.chat_template. This is useful when
                reusing a tokenizer that does not bundle the right template.
            use_generate: Whether to route requests through the generate endpoint.
                (default: False)
            api_key: API key to use for model calls. (default: None)
            llm_request_timeout_sec: Per-request LLM timeout in seconds. Set to None
                or <=0 to disable. (default: 600.0)
            trajectory_config: Optional TrajectoryConfig containing trajectory-related configurations.
                Available options:
                - raw_content (bool): If True, dump raw LLM responses into trajectory without
                  parsing into tool_calls. Useful for SFT data export. (default: False)
                - linear_history (bool): If True, split trajectory into separate files when context
                  summarization occurs, ensuring each trajectory represents a continuous linear
                  history sent to the LLM. When False, keep all steps from the main agent
                  in a single trajectory file despite chat history resets. (default: False)
                (default: None)
            tmux_pane_width: Starting tmux pane width (maps to `tmux -x`, default: 160)
            tmux_pane_height: Starting tmux pane height (maps to `tmux -y`, default: 40)
            record_terminal_session: Whether to capture terminal recordings via asciinema.
                (default: True)
            interleaved_thinking: Whether to include reasoning content in chat history
                and send to litellm in next round's conversation (default: False)
            suppress_max_turns_warning: Whether to suppress the warning about artificially
                limiting max_turns (default: False)
            llm_backend: LLM backend to use. Use LLMBackend.LITELLM or "litellm".
                (default: LLMBackend.LITELLM)
            llm_kwargs: Additional kwargs to pass to the LLM constructor.
                (default: None)
            llm_call_kwargs: Extra kwargs to forward to LLM calls (e.g., extra_body).
            episode_log_mode: Which episode logs to keep ("all" or "last"). When set to
                "last", logs are written to a fixed episode directory and files are
                overwritten each episode. (default: "last")
            **kwargs: Additional arguments
        """
        super().__init__(logs_dir, model_name, *args, **kwargs)

        if model_name is None:
            raise ValueError("model_name is required for Terminus 2")

        self._model_name = model_name
        self._parser_name = parser_name
        self._collect_rollout_details = collect_rollout_details
        self._reasoning_effort = reasoning_effort
        self._main_api_base = api_base
        self._summary_api_base = summary_api_base or api_base
        self._tokenizer_path = tokenizer_path
        self._chat_template_path = chat_template_path
        resolved_model_info = self._resolve_model_info(model_name, model_info)
        self._llm = llm or self._init_llm(
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
            model_info=resolved_model_info,
            tokenizer_path=tokenizer_path,
            chat_template_path=chat_template_path,
            use_generate=use_generate,
            api_key=api_key,
            request_timeout_sec=llm_request_timeout_sec,
            use_responses_api=use_responses_api,
        )
        self._parser = self._get_parser()
        self._context_management_mode = _normalize_context_management_mode(
            enable_summarize
        )
        self._prompt_template = self._get_prompt_template_path().read_text()
        self._timeout_template = self._get_timeout_template_path().read_text()
        self._logger = logger.getChild(__name__)
        self._logger.setLevel(logging.DEBUG)
        self._temperature = temperature

        # Handle deprecated 'episodes' kwarg
        episodes_from_kwargs = kwargs.get("episodes")
        if episodes_from_kwargs is not None:
            self.logger.warning(
                "The 'episodes' parameter is deprecated and will be removed in a future version. "
                "Please use 'max_turns' instead."
            )

        # Determine the final max episodes value with proper precedence:
        # 1. max_turns (new parameter)
        # 2. max_episodes (deprecated but still supported)
        # 3. episodes from kwargs (deprecated)
        # 4. Default value of 1000000
        final_max_episodes = None
        if max_turns is not None:
            final_max_episodes = max_turns
        elif kwargs.get("max_episodes") is not None:
            final_max_episodes = kwargs.get("max_episodes")
        elif episodes_from_kwargs is not None:
            final_max_episodes = episodes_from_kwargs

        if final_max_episodes is not None:
            if not suppress_max_turns_warning:
                self.logger.warning(
                    f"max_turns (f.k.a. max_episodes) artificially limited to {final_max_episodes}. "
                    "Consider removing this limit for better task completion."
                )
            self._max_episodes = final_max_episodes
        else:
            self._max_episodes = 1000000
        self._chat: Chat | None = None
        self._context: AgentContext | None = None
        self._profiling_spans: list[dict[str, Any]] = []
        self._timestamped_markers: list[tuple[float, str]] = []
        self._pending_completion = False
        self._session: TmuxSession | None = None
        self._api_request_times: list[float] = []
        self._n_episodes: int = 0
        self._initial_generate_completed = False
        self._session_id = session_id if session_id else str(uuid.uuid4())
        self._initial_session_id = self._session_id
        self._enable_tito = bool(enable_tito)
        self._tito_key_rotation_count = 0
        self._messages_list_tito_keys: list[str] = (
            [self._session_id] if self._enable_tito else []
        )
        self._sync_llm_session_id(self._session_id)
        self._trajectory_steps: list[Step] = []
        self._record_terminal_session = record_terminal_session
        self._llm_call_kwargs = dict(llm_call_kwargs) if llm_call_kwargs else {}
        self._llm_query_type = llm_query_type

        self._summarization_count: int = (
            0  # Track number of summarization subagents created
        )
        self._full_summary_success_count: int = 0
        self._short_summary_success_count: int = 0
        self._ultimate_summary_success_count: int = 0
        self._short_summary_log_index: int = 0
        self._pending_subagent_refs: list[SubagentTrajectoryRef] | None = (
            None  # Track subagent refs to include in next step
        )
        self._pending_handoff_prompt: str | None = (
            None  # Track handoff prompt to include as user step
        )
        self._managed_messages: list[list[dict]] = []
        self._rollout_messages: list[list[dict]] = []
        self._pending_context_truncations: list[dict] = []
        self._context_truncation_count = 0
        self._subagent_metrics = SubagentMetrics()  # Track subagent metrics separately
        self._subagent_rollout_details: list[
            RolloutDetail
        ] = []  # Track rollout details for each subagent
        self._format_error_count = 0
        self._context_length_exceeded_count = 0
        self._output_length_exceeded_count = 0
        self._summary_training = _normalize_summary_training_mode(summary_training)
        self._summary_messages: list[list[dict]] = []
        self._summary_messages_context_management_indices: list[int] = []
        self._summary_history_user_turns = max(0, int(summary_history_user_turns))
        self._resume_generation_from_existing_user = False
        self._awaiting_post_context_management_generate = False
        self._post_context_management_generate_interrupted_reason: str | None = None
        self._enable_summarize = self._context_management_mode != "none"
        normalized_mode = (
            "summarize"
            if summarization_mode is None
            else str(summarization_mode).strip().lower()
        )
        if normalized_mode not in ("summarize", "truncate"):
            raise ValueError(
                "summarization_mode must be one of: summarize, truncate"
            )
        self._summarization_mode = normalized_mode
        self._proactive_summarization_threshold = proactive_summarization_threshold
        self._context_management_max_count = (
            int(context_management_max_count)
            if context_management_max_count is not None
            else -1
        )
        self._tmux_pane_width = tmux_pane_width
        self._tmux_pane_height = tmux_pane_height
        self._tokenizer_path = tokenizer_path
        self._tokenizer_for_count = None
        self._api_key = api_key

        # Trajectory configuration
        self._trajectory_config = trajectory_config or {}
        self._save_raw_content_in_trajectory = self._trajectory_config.get(
            "raw_content", False
        )
        self._linear_history = self._trajectory_config.get("linear_history", False)
        # Optional: include full chat messages in TrialResult metadata (can be large)
        self._store_all_messages = store_all_messages
        self._episode_log_mode = self._normalize_episode_log_mode(episode_log_mode)
        self._training_mode = self._is_training_mode()
        self._interleaved_thinking = interleaved_thinking or self._training_mode
        self._llm_kwargs = llm_kwargs

    def _is_auto_context_management_enabled(self) -> bool:
        return self._context_management_mode == "auto"

    def _is_tool_context_management_enabled(self) -> bool:
        return self._context_management_mode == "tool"

    def _should_automatically_manage_context_length(self) -> bool:
        return self._context_management_mode in {"auto", "tool"}

    def _sync_llm_session_id(self, session_id: str | None) -> None:
        """Synchronize runtime session/main_key to the LLM backend."""
        set_session_id = getattr(self._llm, "set_session_id", None)
        if callable(set_session_id):
            set_session_id(session_id)
            return
        # Fallback for older LLM implementations.
        if hasattr(self._llm, "_session_id"):
            setattr(self._llm, "_session_id", session_id)

    def _get_llm_api_base(self) -> str | None:
        return getattr(self._llm, "_api_base", None)

    def _set_llm_api_base(self, api_base: str | None) -> None:
        if hasattr(self._llm, "_api_base"):
            setattr(self._llm, "_api_base", api_base)

    def _append_router_usage_log(
        self,
        mode: Literal["main_chat", "summary"],
        api_base: str | None,
        session_id: str | None = None,
    ) -> None:
        filename = (
            "main_chat_router_usage.jsonl"
            if mode == "main_chat"
            else "summary_router_usage.jsonl"
        )
        # log_path = self.logs_dir / filename

        parsed = urlparse(api_base) if api_base else None
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "api_base": api_base,
            "router_host": parsed.hostname if parsed else None,
            "router_port": parsed.port if parsed else None,
            "model_name": self._model_name,
            "session_id": session_id,
            "llm_query_type": self._resolve_llm_query_type(),
        }

    @contextmanager
    def _summary_api_base_scope(self):
        original_api_base = self._get_llm_api_base()
        target_api_base = self._summary_api_base or original_api_base
        if target_api_base == original_api_base:
            yield
            return
        self._set_llm_api_base(target_api_base)
        try:
            yield
        finally:
            self._set_llm_api_base(original_api_base)

    def _preview_next_tito_main_key(self) -> str:
        return f"{self._initial_session_id}_cm_{self._tito_key_rotation_count + 1}"

    def _preview_summary_tito_main_key(self) -> str:
        return f"{self._initial_session_id}_summary_{self._summarization_count}"

    def _rotate_tito_main_key(self, reason: str, *, new_key: str | None = None) -> None:
        """Rotate TiTO key after context-management boundaries."""
        if not self._enable_tito:
            return
        self._tito_key_rotation_count += 1
        if new_key is None:
            new_key = f"{self._initial_session_id}_cm_{self._tito_key_rotation_count}"
        self._session_id = new_key
        self._sync_llm_session_id(new_key)
        self._logger.debug(
            "Rotated TiTO main_key after context management "
            f"(reason={reason}, new_key={new_key})"
        )

    def _mark_context_management_boundary(self) -> None:
        """Require one successful main-chat generate before exporting continuation."""
        self._awaiting_post_context_management_generate = True
        self._post_context_management_generate_interrupted_reason = None

    def _mark_post_context_management_generate_complete(self) -> None:
        self._awaiting_post_context_management_generate = False
        self._post_context_management_generate_interrupted_reason = None
        if (
            self._enable_tito
            and self._session_id
            and (
                not self._messages_list_tito_keys
                or self._messages_list_tito_keys[-1] != self._session_id
            )
        ):
            self._messages_list_tito_keys.append(self._session_id)

    def _append_rollout_messages_snapshot(
        self,
        messages: list[dict[str, Any]] | None,
    ) -> None:
        if not isinstance(messages, list) or not messages:
            return
        normalized_messages = [dict(message) for message in messages if isinstance(message, dict)]
        if not normalized_messages:
            return
        self._rollout_messages.append(normalized_messages)

    def _record_main_key_on_latest_generate_turn(
        self,
        chat: Chat,
        *,
        prompt: str | None,
        response: LLMResponse,
    ) -> None:
        def append_debug_record(record: dict[str, Any]) -> None:
            try:
                debug_path = self.logs_dir / "main_key_turns.jsonl"
                with open(debug_path, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                self._logger.debug("Failed to write main_key_turns debug log", exc_info=True)

        base_record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "enable_tito": self._enable_tito,
            "session_id": self._session_id,
            "initial_session_id": self._initial_session_id,
            "response_main_key": response.main_key,
            "prompt_present": prompt is not None,
            "chat_message_count": len(chat.messages),
        }
        if not self._enable_tito:
            append_debug_record({**base_record, "status": "skip_tito_disabled"})
            return
        main_key = response.main_key or self._session_id
        if not main_key or len(chat.messages) == 0:
            append_debug_record(
                {
                    **base_record,
                    "status": "skip_missing_main_key_or_messages",
                    "resolved_main_key": main_key,
                }
            )
            return

        last_message = chat.messages[-1]
        last_role = last_message.get("role") if isinstance(last_message, dict) else None
        assistant_recorded = False
        if isinstance(last_message, dict) and last_message.get("role") == "assistant":
            last_message["main_key"] = main_key
            assistant_recorded = True

        if prompt is None or len(chat.messages) < 2:
            append_debug_record(
                {
                    **base_record,
                    "status": "recorded",
                    "resolved_main_key": main_key,
                    "last_role": last_role,
                    "assistant_recorded": assistant_recorded,
                    "prompt_user_recorded": False,
                }
            )
            return
        previous_message = chat.messages[-2]
        previous_role = (
            previous_message.get("role") if isinstance(previous_message, dict) else None
        )
        prompt_matches_previous = (
            isinstance(previous_message, dict)
            and previous_message.get("role") == "user"
            and previous_message.get("content") == prompt
        )
        prompt_user_recorded = False
        if (
            isinstance(previous_message, dict)
            and previous_message.get("role") == "user"
            and previous_message.get("content") == prompt
        ):
            previous_message["main_key"] = main_key
            prompt_user_recorded = True
        append_debug_record(
            {
                **base_record,
                "status": "recorded",
                "resolved_main_key": main_key,
                "last_role": last_role,
                "previous_role": previous_role,
                "assistant_recorded": assistant_recorded,
                "prompt_matches_previous": prompt_matches_previous,
                "prompt_user_recorded": prompt_user_recorded,
            }
        )

    def _resolve_model_info(
        self, model_name: str | None, provided_model_info: dict | None
    ) -> dict | None:
        if provided_model_info:
            return provided_model_info
        if model_name and "hosted_vllm" in model_name:
            self.logger.warning(
                "Model info is required when using hosted_vllm models. "
                "Please set `model_info` in your Terminus 2 configuration with "
                "`max_input_tokens`, `max_output_tokens`, and cost fields. "
                "Falling back to LiteLLM defaults, which may cause context or pricing issues."
            )
        return None

    @staticmethod
    def name() -> str:
        return AgentName.TERMINUS_2.value

    def version(self) -> str | None:
        return "2.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if self._record_terminal_session:
            local_recording_path = environment.trial_paths.agent_dir / "recording.cast"
            remote_recording_path = EnvironmentPaths.agent_dir / "recording.cast"
        else:
            local_recording_path = None
            remote_recording_path = None

        self._session = TmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
            local_asciinema_recording_path=local_recording_path,
            remote_asciinema_recording_path=remote_recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
        )
        await self._session.start()

    def _get_parser(self):
        """Return the appropriate parser instance for this format."""
        if self._parser_name == "json":
            return TerminusJSONPlainParser()
        elif self._parser_name == "xml":
            return TerminusXMLPlainParser()
        else:
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    def _get_prompt_template_path(self) -> Path:
        """Return the path to the prompt template for this format."""
        if self._parser_name == "json":
            filename = (
                "terminus-json-plain-tool.txt"
                if self._is_tool_context_management_enabled()
                else "terminus-json-plain.txt"
            )
            return Path(__file__).parent / "templates" / filename
        elif self._parser_name == "xml":
            filename = (
                "terminus-xml-plain-tool.txt"
                if self._is_tool_context_management_enabled()
                else "terminus-xml-plain.txt"
            )
            return Path(__file__).parent / "templates" / filename
        else:
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    def _get_initial_system_prompt(
        self,
        instruction: str,
        terminal_state: str,
    ) -> str | None:
        _ = instruction
        _ = terminal_state
        return None

    def _get_initial_user_prompt(
        self,
        instruction: str,
        terminal_state: str,
    ) -> str:
        return self._prompt_template.format(
            instruction=instruction,
            terminal_state=terminal_state,
        )

    def _get_timeout_template_path(self) -> Path:
        """Return the path to the timeout template for this format."""
        return Path(__file__).parent / "templates" / "timeout.txt"

    @staticmethod
    def _parse_skill_frontmatter(content: str) -> dict[str, str] | None:
        """Parse YAML frontmatter from SKILL.md content, returning name and description."""
        import re

        import yaml

        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return None
        try:
            fm = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
        if not isinstance(fm, dict) or "name" not in fm or "description" not in fm:
            return None
        return {"name": fm["name"], "description": fm["description"]}

    async def _build_skills_section(self, environment: BaseEnvironment) -> str | None:
        """Discover Agent Skills in skills_dir and return an <available_skills> XML block.

        Follows the Agent Skills spec: scans for subdirectories containing SKILL.md
        inside the remote environment, parses YAML frontmatter for name/description,
        and provides the absolute path so the model can ``cat`` the file to activate
        a skill.
        """
        if not self.skills_dir:
            return None

        if not await environment.is_dir(self.skills_dir):
            return None

        # List subdirectories containing SKILL.md in the remote environment
        result = await environment.exec(
            f"find {shlex.quote(self.skills_dir)} -mindepth 2 -maxdepth 2"
            " -name SKILL.md -type f | sort",
            timeout_sec=10,
        )

        if result.return_code != 0 or not result.stdout or not result.stdout.strip():
            return None

        skill_md_paths = result.stdout.strip().splitlines()
        entries: list[tuple[str, str, str]] = []  # (name, description, location)

        for skill_md_path in skill_md_paths:
            cat_result = await environment.exec(
                f"cat {shlex.quote(skill_md_path)}", timeout_sec=10
            )
            if cat_result.return_code != 0 or not cat_result.stdout:
                continue
            fm = self._parse_skill_frontmatter(cat_result.stdout)
            if fm is None:
                continue
            entries.append((fm["name"], fm["description"], skill_md_path))

        if not entries:
            return None

        from xml.etree.ElementTree import Element, SubElement, indent, tostring

        root = Element("available_skills")
        for name, description, location in entries:
            skill = SubElement(root, "skill")
            SubElement(skill, "name").text = name
            SubElement(skill, "description").text = description
            SubElement(skill, "location").text = location

        indent(root, space="  ")
        return "\n" + tostring(root, encoding="unicode")

    def _get_error_response_type(self) -> str:
        """Return the response type name for error messages.

        Examples: 'JSON response', 'response'
        """
        if self._parser_name == "json":
            return "JSON response"
        elif self._parser_name == "xml":
            return "response"
        else:
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Return the format-specific task completion confirmation message."""
        if self._parser_name == "json":
            return (
                f"Current terminal state:\n{terminal_output}\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger your solution to be graded and you won't be able to "
                'make any further corrections. If so, include "task_complete": true '
                "in your JSON response again."
            )
        elif self._parser_name == "xml":
            return (
                f"Current terminal state:\n{terminal_output}\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger your solution to be graded and you won't be able to "
                "make any further corrections. If so, include "
                "<task_complete>true</task_complete> again."
            )
        else:
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    def _get_no_action_observation_message(self) -> str:
        """Return a tool-style observation when the assistant produced no action."""
        if self._parser_name == "json":
            return (
                "No action was taken because your response did not include any tool "
                "call. Please follow the required format and call a function on your "
                "next response. Use execute_commands to run terminal commands, or "
                "task_complete if the task is complete."
            )
        elif self._parser_name == "xml":
            return (
                "No action was taken because your response did not include any tool "
                "call. Please follow the required format and call a function on your "
                "next response. Use execute_commands to run terminal commands, or "
                "task_complete if the task is complete."
            )
        else:
            raise ValueError(
                f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
            )

    def _setup_episode_logging(
        self, logging_dir: Path | None, episode: int
    ) -> tuple[Path | None, Path | None, Path | None]:
        if logging_dir is None:
            return None, None, None

        if self._episode_log_mode == "last":
            episode_dir_name = "episode-last"
        else:
            episode_dir_name = f"episode-{episode}"

        episode_logging_dir = logging_dir / episode_dir_name
        episode_logging_dir.mkdir(parents=True, exist_ok=True)

        return (
            episode_logging_dir / "debug.json",
            episode_logging_dir / "prompt.txt",
            episode_logging_dir / "response.txt",
        )

    @staticmethod
    def _normalize_episode_log_mode(
        mode: str | None,
    ) -> Literal["all", "last"]:
        if mode is None:
            return "last"
        normalized = str(mode).strip().lower()
        if normalized in ("all", "last"):
            return normalized
        raise ValueError("episode_log_mode must be one of: all, last")


    def _count_total_tokens(self, chat: Chat, prompt: str | None = None) -> int:
        """Count total tokens, optionally including a pending user prompt."""
        history_tokens = chat.get_cached_history_token_count(
            self._count_tokens_for_messages
        )
        if prompt is None:
            return history_tokens
        return history_tokens + self._count_tokens_for_messages(
            [{"role": "user", "content": prompt}]
        )

    def _append_context_window_status(
        self, chat: Chat, prompt: str | None
    ) -> str | None:
        """Append current context-window usage to the user prompt."""
        if prompt is None:
            return None
        if not self._is_tool_context_management_enabled():
            return prompt

        context_limit = self._llm.get_model_context_limit()
        if not context_limit or context_limit <= 0:
            return prompt

        marker = f"\n\n{self._CONTEXT_WINDOW_STATUS_HEADER}\n"
        base_prompt = prompt
        if marker in base_prompt:
            base_prompt = base_prompt.split(marker, 1)[0].rstrip()

        # There is no persisted per-turn remaining_tokens field in Terminus2 today.
        # Reuse a single pre-append token estimate for this prompt instead of
        # repeatedly recounting after the status block is appended.
        used_tokens = self._count_total_tokens(chat, base_prompt)
        remaining_tokens = max(context_limit - used_tokens, 0)
        status_block = (
            f"{self._CONTEXT_WINDOW_STATUS_HEADER}\n"
            f"- Used context length: {used_tokens} tokens\n"
            f"- Remaining context length available: {remaining_tokens} tokens\n"
            f"- Total context length: {context_limit} tokens"
        )
        return f"{base_prompt.rstrip()}\n\n{status_block}"

    def _use_claude_query_path(self) -> bool:
        resolved_query_type = self._resolve_llm_query_type()
        self.logger.info(f"self._llm_query_type {self._llm_query_type}")
        #     f.write(f"{resolved_query_type}" + "\n")
        return bool(resolved_query_type) and "claude" in resolved_query_type.lower()

    def _resolve_llm_query_type(self) -> str | None:
        if isinstance(self._llm_query_type, str) and self._llm_query_type.strip():
            return self._llm_query_type
        if isinstance(self._model_name, str) and "claude" in self._model_name.lower():
            return self._model_name
        return None

    @staticmethod
    def _convert_messages_to_anthropic(
        prompt: str | None,
        message_history: list[dict] | None,
    ) -> tuple[str | None, list[dict]]:
        """Convert generic chat history into Anthropic text-only messages."""
        system_parts: list[str] = []
        anthropic_messages: list[dict] = []

        if prompt:
            system_parts.append(str(prompt))

        for msg in message_history or []:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                if content:
                    system_parts.append(str(content))
                continue

            if role not in {"user", "assistant"}:
                continue

            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                text = "\n".join(part for part in parts if part)
            else:
                text = str(content)

            anthropic_messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                }
            )

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, anthropic_messages

    @staticmethod
    def _parse_claude_response_blocks(response: Any) -> tuple[str, str | None]:
        """Extract visible text and thinking blocks from a Claude response."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []

        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "thinking":
                thinking_text = getattr(block, "thinking", None)
                if thinking_text:
                    thinking_parts.append(thinking_text)

        final_text = "".join(text_parts)
        final_thinking = "".join(thinking_parts) if thinking_parts else None
        return final_text, final_thinking

    async def _call_claude_query(
        self,
        *,
        prompt: str | None,
        message_history: list[dict],
        logging_path: Path | None,
    ) -> LLMResponse:
        """Call the hard-coded Claude gateway path for llm_query_type='*claude*'."""
        resolved_query_type = self._resolve_llm_query_type()
        if resolved_query_type is None:
            raise ValueError("Claude query path requested without a Claude model selector")
        self.logger.info("In claude query")
        #     f.write("In claude query" + "\n")
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError(
                "Please install anthropic first: pip install anthropic"
            ) from exc

        system_prompt, anthropic_messages = self._convert_messages_to_anthropic(
            prompt=prompt,
            message_history=message_history,
        )

        client = AsyncAnthropic(
            api_key=self._CLAUDE_QUERY_API_KEY,
            base_url=self._CLAUDE_QUERY_BASE_URL,
            timeout=None,
            max_retries=3,
        )

        request_kwargs: dict[str, Any] = {
            "model": resolved_query_type,
            "messages": anthropic_messages,
            "max_tokens": self._CLAUDE_QUERY_MAX_TOKENS,
            "betas": ["interleaved-thinking-2025-05-14"],
            "thinking": {"type": self._CLAUDE_QUERY_THINKING_TYPE},
        }
        if system_prompt:
            request_kwargs["system"] = system_prompt
        request_kwargs.update(self._llm_call_kwargs)

        streamed_text_chunks: list[str] = []
        async with client.beta.messages.stream(**request_kwargs) as stream_obj:
            async for text in stream_obj.text_stream:
                if text:
                    streamed_text_chunks.append(text)

            response = await stream_obj.get_final_message()

        final_text, final_thinking = self._parse_claude_response_blocks(response)
        if not final_text:
            final_text = "".join(streamed_text_chunks)

        usage_payload: dict[str, Any] = {}
        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            if hasattr(usage_obj, "to_dict"):
                usage_payload = usage_obj.to_dict()
            elif hasattr(usage_obj, "model_dump"):
                usage_payload = usage_obj.model_dump()
            elif isinstance(usage_obj, dict):
                usage_payload = usage_obj

        usage_info = UsageInfo(
            prompt_tokens=int(usage_payload.get("input_tokens", 0) or 0),
            completion_tokens=int(usage_payload.get("output_tokens", 0) or 0),
            cache_tokens=int(usage_payload.get("cache_read_input_tokens", 0) or 0),
            cost_usd=0.0,
        )

        if logging_path is not None:
            log_payload = {
                "request": {
                    "prompt": prompt,
                    "message_history": message_history,
                    "base_url": self._CLAUDE_QUERY_BASE_URL,
                    "model_name": resolved_query_type,
                    "stream": True,
                    "thinking_type": self._CLAUDE_QUERY_THINKING_TYPE,
                    "max_tokens": self._CLAUDE_QUERY_MAX_TOKENS,
                },
                "response": {
                    "id": getattr(response, "id", None),
                    "stop_reason": getattr(response, "stop_reason", None),
                    "usage": usage_payload,
                    "content": final_text,
                    "thinking": final_thinking,
                },
            }
            logging_path.write_text(
                json.dumps(log_payload, ensure_ascii=False, indent=2)
            )

        return LLMResponse(
            content=final_text,
            reasoning_content=final_thinking,
            history_content=final_text,
            response_id=getattr(response, "id", None),
            usage=usage_info,
        )

    def _count_total_tokens_with_tokenizer(self, messages: list[dict]) -> int:
        """Count tokens using a tokenizer chat template (currently for Qwen3)."""
        if not messages:
            return 0
        tokenizer = self._get_tokenizer_for_counting()
        rendered_messages = prepare_messages_for_generate(messages)
        try:
            prompt_text = tokenizer.apply_chat_template(
                rendered_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
                clear_thinking=False,
            )
        except TypeError:
            # Older tokenizer templates may not accept clear_thinking.
            prompt_text = tokenizer.apply_chat_template(
                rendered_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        return len(tokenizer.encode(prompt_text))

    def _get_tokenizer_for_counting(self):
        tokenizer_path = self._tokenizer_path
        if not tokenizer_path:
            raise ValueError(
                "Tokenizer-based token counting requires tokenizer_path."
            )

        if self._tokenizer_for_count is not None:
            return self._tokenizer_for_count

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self._chat_template_path:
            tokenizer.chat_template = Path(self._chat_template_path).read_text(
                encoding="utf-8"
            )
        if not tokenizer.chat_template:
            raise ValueError(
                "Tokenizer has no chat_template. Provide a tokenizer with chat_template."
            )

        self._tokenizer_for_count = tokenizer
        return tokenizer

    def _limit_output_length(self, output: str, max_bytes: int = 10000) -> str:
        """
        Limit output to specified byte length, keeping first and last portions.

        Args:
            output: The terminal output to potentially truncate
            max_bytes: Maximum allowed bytes (default 10000)

        Returns:
            str: Original output if under limit, or truncated with middle omitted
        """
        if len(output.encode("utf-8")) <= max_bytes:
            return output

        # Calculate portions (half each for first and last)
        portion_size = max_bytes // 2

        # Convert to bytes for accurate splitting
        output_bytes = output.encode("utf-8")

        # Get first portion
        first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")

        # Get last portion
        last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")

        # Calculate omitted bytes
        omitted_bytes = (
            len(output_bytes)
            - len(first_portion.encode("utf-8"))
            - len(last_portion.encode("utf-8"))
        )

        return (
            f"{first_portion}\n[... output limited to {max_bytes} bytes; "
            f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
        )

    def _build_truncate_rehydration_user_message(
        self,
        original_user_content: str,
        recent_messages_preserved: bool,
    ) -> str:
        parts: list[str] = []
        if original_user_content and original_user_content.strip():
            parts.append(original_user_content.strip())

        parts.append(
            "This session is being continued from a previous conversation that "
            "ran out of context. Earlier chat history was truncated to fit the "
            "model window."
        )
        if recent_messages_preserved:
            parts.append(
                "Recent messages are preserved verbatim below and should be treated "
                "as the source of truth for the latest state."
            )
        parts.append(
            "Continue the conversation from where it left off without asking any "
            "further questions. Resume directly from the preserved recent messages."
        )
        return "\n\n".join(parts)

    def _truncate_message_history_keep_last_half(self, chat: Chat) -> dict | None:
        """Rebuild history for truncate mode using the first user and recent assistant/tool pairs."""
        messages = chat.messages
        if len(messages) <= 2:
            return None

        first_user_index = next(
            (i for i, msg in enumerate(messages) if msg.get("role") == "user"),
            None,
        )
        if first_user_index is None:
            return None

        spans = self._build_tool_result_pairs(messages, first_user_index + 1)
        if not spans:
            return None

        if self._summary_history_user_turns <= 0:
            retained_pairs: list[tuple[int, int]] = []
        else:
            retained_pairs = spans[-self._summary_history_user_turns :]
        num_pairs_to_keep = len(retained_pairs)

        rebuilt_messages: list[dict[str, Any]] = [
            dict(msg) for msg in messages[:first_user_index]
        ]
        first_user = dict(messages[first_user_index])
        first_user_content = first_user.get("content")
        first_user["content"] = self._build_truncate_rehydration_user_message(
            first_user_content if isinstance(first_user_content, str) else "",
            recent_messages_preserved=bool(retained_pairs),
        )
        rebuilt_messages.append(first_user)

        for start_i, end_i in retained_pairs:
            rebuilt_messages.extend(dict(msg) for msg in messages[start_i : end_i + 1])

        if len(rebuilt_messages) == len(messages):
            return None

        self._append_rollout_messages_snapshot(chat.messages_with_weight_version())
        self._set_loss_mask_prefix_false(
            rebuilt_messages,
            prefix_count=len(rebuilt_messages),
        )
        chat.set_messages(rebuilt_messages)
        chat.reset_response_chain()
        self._resume_generation_from_existing_user = True
        self._rotate_tito_main_key(reason="truncate")
        self._mark_context_management_boundary()
        self._managed_messages.append(chat.messages_with_weight_version())
        self._logger.debug(
            "Truncated chat history (mode=truncate). "
            f"Original messages: {len(messages)}, kept: {len(rebuilt_messages)}, "
            f"assistant/tool-result pairs kept: {num_pairs_to_keep} of {len(spans)}."
        )
        return {
            "original_messages": len(messages),
            "kept_messages": len(rebuilt_messages),
            "pairs_kept": num_pairs_to_keep,
            "pairs_total": len(spans),
        }

    def _queue_context_truncation_event(self, trigger: str, info: dict) -> None:
        if not info:
            return
        event_info = {"trigger": trigger}
        event_info.update(info)
        self._pending_context_truncations.append(event_info)
        self._context_truncation_count += 1
        self._logger.info(
            "Context truncation triggered "
            f"(session_id={self._session_id}, trigger={trigger}, "
            f"kept_messages={info.get('kept_messages')}, "
            f"original_messages={info.get('original_messages')}, "
            f"pairs_kept={info.get('pairs_kept')}/{info.get('pairs_total')})"
        )

    def _current_context_management_count(self) -> int:
        return self._summarization_count + self._context_truncation_count

    def _ensure_context_management_budget(self, trigger: str) -> None:
        if self._context_management_max_count < 0:
            return

        used_count = self._current_context_management_count()
        if used_count < self._context_management_max_count:
            return

        raise ContextManagementLimitExceededError(
            "Context management limit exceeded. "
            f"trigger={trigger}, used={used_count}, "
            f"max={self._context_management_max_count}."
        )

    def _count_tokens_for_messages(
        self, messages: list[dict[str, str]]
    ) -> int:
        """Count tokens for an explicit message list."""
        if self._tokenizer_path:
            return self._count_total_tokens_with_tokenizer(messages)
        from litellm.utils import token_counter
        return token_counter(model=self._model_name, messages=messages)

    def _trim_history_for_context(
        self,
        chat: Chat,
        extra_messages: list[dict[str, str]],
        max_new_tokens: int,
        reason: str,
    ) -> None:
        """Trim chat history until messages + extra_messages fit within context."""
        context_limit = self._llm.get_model_context_limit()
        if not context_limit:
            return

        removed_pairs = 0
        while len(chat.messages) > 1:
            total_tokens = self._count_tokens_for_messages(
                chat.messages + extra_messages
            )
            if total_tokens + max_new_tokens <= context_limit:
                break
            chat.set_messages(chat.messages[:-2])
            removed_pairs += 1

        free_tokens = context_limit - self._count_tokens_for_messages(
            chat.messages + extra_messages
        )
        if removed_pairs > 0:
            chat.reset_response_chain()
        self._logger.debug(
            f"{reason}: Remaining messages={len(chat.messages)}, "
            f"pairs_removed={removed_pairs}, free_tokens≈{free_tokens}"
        )

    def _collect_subagent_rollout_detail(self, response: LLMResponse) -> None:
        """Collect rollout details from a subagent LLM response.

        Args:
            response: The LLM response containing token IDs and logprobs
        """
        if not self._collect_rollout_details:
            return

        rollout_detail: RolloutDetail = {}
        if response.prompt_token_ids is not None:
            rollout_detail["prompt_token_ids"] = [response.prompt_token_ids]
        if response.completion_token_ids is not None:
            rollout_detail["completion_token_ids"] = [response.completion_token_ids]
        if response.logprobs is not None:
            rollout_detail["logprobs"] = [response.logprobs]

        if rollout_detail:
            self._subagent_rollout_details.append(rollout_detail)

    def _update_subagent_metrics(self, usage_info) -> None:
        """Update subagent metrics with usage information from an LLM response.

        Args:
            usage_info: Usage information from LLM response containing token counts and cost
        """
        if not usage_info:
            return

        self._subagent_metrics.total_prompt_tokens += usage_info.prompt_tokens
        self._subagent_metrics.total_completion_tokens += usage_info.completion_tokens
        self._subagent_metrics.total_cached_tokens += usage_info.cache_tokens
        self._subagent_metrics.total_cost_usd += usage_info.cost_usd

    def _extract_usage_metrics(self, usage_info) -> tuple[int, int, int, float]:
        """Extract and normalize metrics from usage info.

        Args:
            usage_info: Usage information from LLM response

        Returns:
            Tuple of (prompt_tokens, completion_tokens, cached_tokens, cost_usd)
        """
        if not usage_info:
            return 0, 0, 0, 0

        return (
            usage_info.prompt_tokens,
            usage_info.completion_tokens,
            usage_info.cache_tokens,
            usage_info.cost_usd if usage_info.cost_usd > 0 else 0,
        )

    def _track_api_request_time(self, start_time: float) -> None:
        """Track API request time from start timestamp.

        Args:
            start_time: Start time from time.time()
        """
        end_time = time.time()
        request_time_ms = (end_time - start_time) * 1000
        self._api_request_times.append(request_time_ms)

    def _build_subagent_training_messages(
        self,
        prompt: str,
        message_history: list[dict[str, Any]],
        response: LLMResponse,
        main_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the full subagent context used for generation with loss flags."""
        full_context = copy.deepcopy(message_history)
        prompt_message: dict[str, Any] = {"role": "user", "content": prompt}
        if main_key:
            prompt_message["main_key"] = main_key
        full_context.append(prompt_message)

        training_messages = []
        for message in full_context:
            message_copy = dict(message)
            message_copy["loss_mask"] = False
            message_copy["num_tokens"] = int(message_copy.get("num_tokens") or 0)
            training_messages.append(message_copy)

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": response.content,
            "reasoning_content": response.reasoning_content or "",
            "weight_version": response.weight_version,
            "loss_mask": True,
            "num_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        if main_key:
            assistant_message["main_key"] = main_key
        if response.history_content:
            assistant_message["history_content"] = response.history_content
        training_messages.append(assistant_message)
        return training_messages

    @staticmethod
    def _stringify_training_message_content(content: Any) -> str:
        if isinstance(content, str):
            normalized = content.strip()
        else:
            try:
                normalized = json.dumps(content, ensure_ascii=False).strip()
            except TypeError:
                normalized = str(content).strip()
        user_prefix = "<|user|>"
        while normalized.startswith(user_prefix):
            normalized = normalized[len(user_prefix) :].lstrip()
        try:
            return normalized.strip()
        except AttributeError:
            return str(normalized).strip()

    def _build_summary_single_turn_user_content(
        self,
        prompt: str,
        message_history: list[dict[str, Any]],
    ) -> str:
        rendered_history = copy.deepcopy(message_history)
        if getattr(self._llm, "_use_generate", False):
            rendered_history = prepare_messages_for_generate(rendered_history)

        transcript_blocks: list[str] = []
        for message in rendered_history:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role == "system" or not role:
                continue
            content = self._stringify_training_message_content(
                message.get("content", "")
            )
            if not content:
                continue
            transcript_blocks.append(f"{role.capitalize()}:\n{content}")

        prompt_content = self._stringify_training_message_content(prompt)
        if prompt_content:
            transcript_blocks.append(f"User:\n{prompt_content}")

        return "\n\n".join(transcript_blocks)

    def _build_summary_training_messages_single_turn(
        self,
        user_content: str,
        response: LLMResponse,
        main_key: str | None = None,
    ) -> list[dict[str, Any]]:
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": response.content,
            "reasoning_content": response.reasoning_content or "",
            "weight_version": response.weight_version,
            "loss_mask": True,
            "num_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        if main_key:
            assistant_message["main_key"] = main_key
        if response.history_content:
            assistant_message["history_content"] = response.history_content

        return [
            {
                "role": "user",
                "content": user_content,
                "loss_mask": False,
                "num_tokens": 0,
                **({"main_key": main_key} if main_key else {}),
            },
            assistant_message,
        ]

    async def _run_subagent(
        self,
        prompt: str,
        message_history: list[dict],
        steps: list[Step],
        session_id: str,
        agent_name: str,
        filename_suffix: str,
        summary_text: str,
        subagent_name_for_logging: str = "subagent",
        llm_session_id_override: str | None = None,
        include_rendered_system_prompt: bool = False,
    ) -> tuple[LLMResponse, SubagentTrajectoryRef, list[dict[str, Any]]]:
        """Run a subagent and return its response and trajectory reference.

        This helper encapsulates the common pattern of:
        1. Appending the prompt step to the trajectory
        2. Calling the LLM with the prompt and message history
        3. Tracking API request time
        4. Updating subagent metrics
        5. Appending the response step to the trajectory
        6. Collecting rollout details
        7. Saving the subagent trajectory

        Args:
            prompt: The prompt to send to the LLM
            message_history: The message history to provide context
            steps: The trajectory steps (will be modified to include prompt and response steps)
            session_id: Session ID for the subagent
            agent_name: Name of the subagent (e.g., "terminus-2-summarization-summary")
            filename_suffix: Suffix for trajectory filename (e.g., "summary", "questions", "answers")
            summary_text: Human-readable summary for the trajectory ref
            subagent_name_for_logging: Name used in logging messages

        Returns:
            tuple: (LLM response, SubagentTrajectoryRef, full generation context)
        """
        # Append the prompt step
        prompt_step_id = len(steps) + 1
        steps.append(
            Step(
                step_id=prompt_step_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=prompt,
            )
        )
        response_step_id = prompt_step_id + 1

        start_time = time.time()
        self._logger.debug("directly calling subagent LLM...")
        subagent_logging_dir = self.logs_dir / "subagent-logs"
        subagent_logging_dir.mkdir(parents=True, exist_ok=True)
        subagent_logging_path = subagent_logging_dir / f"{session_id}.debug.json"
        subagent_call_kwargs = dict(self._llm_call_kwargs)
        subagent_call_kwargs.pop("logging_path", None)
        if (
            include_rendered_system_prompt
            and self._history_starts_with_rendered_tools_system_prompt(message_history)
        ):
            subagent_call_kwargs.pop("tools", None)
        response: LLMResponse | None = None
        active_llm_main_key = (
            llm_session_id_override
            if self._enable_tito and llm_session_id_override is not None
            else (session_id if self._enable_tito else None)
        )
        for attempt in range(3):
            try:
                with self._summary_api_base_scope():
                    self._append_router_usage_log(
                        mode="summary",
                        api_base=self._get_llm_api_base() or self._summary_api_base,
                        session_id=session_id,
                    )
                    span_started_at = datetime.now(timezone.utc)
                    response = await self._llm.call(
                        prompt=prompt,
                        message_history=message_history,
                        logging_path=subagent_logging_path,
                        session_id_override=active_llm_main_key,
                        include_rendered_system_prompt=include_rendered_system_prompt,
                        disable_length_limit=True,
                        **subagent_call_kwargs,
                    )
                    self._append_profiling_span(
                        category="generate",
                        label=f"summary:{filename_suffix}",
                        started_at=span_started_at,
                        finished_at=datetime.now(timezone.utc),
                        metadata={
                            "session_id": session_id,
                            "agent_name": agent_name,
                            "subagent_name": subagent_name_for_logging,
                            "attempt": attempt + 1,
                        },
                    )
                break
            except ContextLengthExceededError:
                raise
            except (OutputLengthExceededError, ParseError):
                raise
            except Exception as exc:
                self._logger.warning(
                    "Subagent LLM call failed for %s (attempt %d/3): %s",
                    subagent_name_for_logging,
                    attempt + 1,
                    exc,
                )
                if attempt == 2:
                    raise
        if response is None:
            raise SummaryGenerationFailedError(
                f"Subagent LLM call failed for {subagent_name_for_logging}"
            )
        self._track_api_request_time(start_time)

        usage_info = response.usage
        self._update_subagent_metrics(usage_info)

        self._append_subagent_response_step(
            steps,
            response_step_id,
            response,
            usage_info,
            subagent_name_for_logging,
        )

        self._collect_subagent_rollout_detail(response)

        trajectory_ref = self._save_subagent_trajectory(
            session_id=session_id,
            agent_name=agent_name,
            steps=steps,
            usage_info=usage_info,
            filename_suffix=filename_suffix,
            summary_text=summary_text,
        )

        training_messages = self._build_subagent_training_messages(
            prompt=prompt,
            message_history=message_history,
            response=response,
            main_key=response.main_key or active_llm_main_key,
        )

        return response, trajectory_ref, training_messages

    @staticmethod
    def _profile_preview(text: str | None, max_len: int = 120) -> str:
        if not text:
            return ""
        normalized = " ".join(text.split())
        if len(normalized) <= max_len:
            return normalized
        return normalized[: max_len - 3] + "..."

    def _append_profiling_span(
        self,
        category: str,
        label: str,
        started_at: datetime,
        finished_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if finished_at < started_at:
            return
        self._profiling_spans.append(
            {
                "category": category,
                "label": label,
                "started_at": started_at,
                "finished_at": finished_at,
                "metadata": metadata,
            }
        )

    def _append_observation_processing_span(
        self,
        label: str,
        started_at: datetime,
        finished_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._append_profiling_span(
            category="observation_processing",
            label=label,
            started_at=started_at,
            finished_at=finished_at,
            metadata=metadata,
        )

    def _chat_profiling_callback(
        self,
        label: str,
        started_at: datetime,
        finished_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._append_observation_processing_span(
            label=label,
            started_at=started_at,
            finished_at=finished_at,
            metadata=metadata,
        )

    async def _get_incremental_output_profiled(
        self,
        session: TmuxSession,
        *,
        label: str = "tmux_get_incremental_output",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        started_at = datetime.now(timezone.utc)
        try:
            return await session.get_incremental_output()
        finally:
            self._append_observation_processing_span(
                label=label,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                metadata=metadata,
            )

    def _build_summary_training_candidate(
        self,
        training_messages: list[dict[str, Any]],
        tito_key: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, int] | None:
        if self._summary_training not in {"summary", "full"}:
            return None
        return (
            training_messages,
            tito_key,
            self._summarization_count,
        )

    @staticmethod
    def _clear_loss_masks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated: list[dict[str, Any]] = []
        for message in messages:
            message_copy = dict(message)
            message_copy.pop("loss_mask", None)
            message_copy.pop("_loss_mask", None)
            updated.append(message_copy)
        return updated

    @staticmethod
    def _set_all_loss_masks(
        messages: list[dict[str, Any]],
        *,
        value: bool,
    ) -> list[dict[str, Any]]:
        updated: list[dict[str, Any]] = []
        for message in messages:
            message_copy = dict(message)
            message_copy.pop("_loss_mask", None)
            message_copy["loss_mask"] = value
            updated.append(message_copy)
        return updated

    @staticmethod
    def _set_last_assistant_loss_mask(
        messages: list[dict[str, Any]],
        *,
        value: bool,
    ) -> list[dict[str, Any]]:
        updated = [dict(message) for message in messages]
        for idx in range(len(updated) - 1, -1, -1):
            if updated[idx].get("role") != "assistant":
                continue
            updated[idx].pop("_loss_mask", None)
            updated[idx]["loss_mask"] = value
            break
        return updated

    def _build_main_chat_summary_generation_messages(
        self,
        training_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        updated = [dict(message) for message in training_messages]
        summary_assistant_idx = -1
        for idx in range(len(updated) - 1, -1, -1):
            if updated[idx].get("role") == "assistant":
                summary_assistant_idx = idx
                break

        if summary_assistant_idx < 0:
            return updated

        if self._summary_training == "full":
            return updated

        if self._summary_training == "summary":
            for message in updated:
                message.pop("_loss_mask", None)
                message["loss_mask"] = False
            updated[summary_assistant_idx]["loss_mask"] = True
            return updated

        updated[summary_assistant_idx].pop("_loss_mask", None)
        updated[summary_assistant_idx]["loss_mask"] = False
        return updated

    def _build_summary_generation_rollout_messages(
        self,
        chat: Chat,
        summary_prompt: str,
        summary_response: LLMResponse,
        main_key: str | None = None,
    ) -> list[dict[str, Any]]:
        recorded_main_key = summary_response.main_key or main_key
        summary_messages = chat.messages_with_weight_version()
        summary_messages.append(
            {
                "role": "user",
                "content": summary_prompt,
                "content_origin": summary_prompt,
                "num_tokens": 0,
                **({"main_key": recorded_main_key} if recorded_main_key else {}),
            }
        )
        summary_messages.append(
            {
                "role": "assistant",
                "content": summary_response.content,
                "content_origin": summary_response.history_content
                or summary_response.content,
                "reasoning_content": summary_response.reasoning_content or "",
                "weight_version": summary_response.weight_version,
                "num_tokens": (
                    summary_response.usage.completion_tokens
                    if summary_response.usage
                    else 0
                ),
            }
        )
        if recorded_main_key:
            summary_messages[-1]["main_key"] = recorded_main_key
        return self._build_main_chat_summary_generation_messages(summary_messages)

    def _append_summary_generation_turns_to_chat(
        self,
        chat: Chat,
        summary_prompt: str,
        summary_response: LLMResponse,
        main_key: str | None = None,
    ) -> None:
        recorded_main_key = summary_response.main_key or main_key
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": summary_response.content,
            "history_content": (
                summary_response.history_content or summary_response.content
            ),
            "reasoning_content": summary_response.reasoning_content or "",
            "weight_version": summary_response.weight_version,
            "num_tokens": (
                summary_response.usage.completion_tokens
                if summary_response.usage
                else 0
            ),
        }
        if recorded_main_key:
            assistant_message["main_key"] = recorded_main_key
        chat.extend_messages(
            [
                {
                    "role": "user",
                    "content": summary_prompt,
                    "num_tokens": 0,
                    **({"main_key": recorded_main_key} if recorded_main_key else {}),
                },
                assistant_message,
            ],
            token_counter=self._count_tokens_for_messages,
        )

    def _build_summary_training_candidate_from_response(
        self,
        *,
        user_content: str,
        response: LLMResponse,
        tito_key: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, int] | None:
        training_messages = self._build_summary_training_messages_single_turn(
            user_content=user_content,
            response=response,
            main_key=tito_key,
        )
        return self._build_summary_training_candidate(
            training_messages,
            tito_key=tito_key,
        )

    def _append_summary_training_candidate(
        self,
        candidate: tuple[list[dict[str, Any]], str | None, int] | None,
    ) -> None:
        if candidate is None:
            return
        summary_messages, tito_key, context_management_index = candidate
        self._summary_messages.append(summary_messages)
        self._summary_messages_context_management_indices.append(
            context_management_index
        )

    def _build_summary_rehydration_user_message(
        self,
        summary_text: str,
        recent_messages_preserved: bool,
    ) -> str:
        summary_body = self._format_summary_rehydration_summary(summary_text)
        if not summary_body:
            summary_body = "Summary:\nNo prior summary content was produced."

        parts = [
            (
                "This session is being continued from a previous conversation that "
                "ran out of context. The summary below covers the earlier portion "
                "of the conversation."
            ),
            summary_body,
        ]
        if recent_messages_preserved:
            parts.append(
                "Recent messages are preserved verbatim below and should be treated "
                "as the source of truth for the latest state."
            )
        parts.append(
            "Continue the conversation from where it left off without asking any "
            "further questions. Resume directly from this summary and the "
            "preserved recent messages."
        )
        return "\n\n".join(parts)

    @staticmethod
    def _format_summary_rehydration_summary(summary_text: str) -> str:
        summary_body = (summary_text or "").strip()
        if not summary_body:
            return ""

        summary_body = re.sub(
            r"<analysis>.*?</analysis>",
            "",
            summary_body,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        summary_match = re.search(
            r"<summary>(.*?)</summary>",
            summary_body,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if summary_match:
            summary_body = f"Summary:\n{summary_match.group(1).strip()}"

        return re.sub(r"\n{3,}", "\n\n", summary_body).strip()

    _SUMMARY_REBUILD_ESTIMATED_TOKENS = 1000

    def _build_claude_style_summary_prompt(
        self,
        original_instruction: str,
        retained_pair_count: int | None = None,
    ) -> str:
        if retained_pair_count is None:
            retained_pair_count = self._summary_history_user_turns

        if retained_pair_count <= 0:
            recent_message_note = "No recent assistant/tool-result pairs"
        elif retained_pair_count == 1:
            recent_message_note = "The most recent assistant/tool-result pair"
        else:
            recent_message_note = (
                f"The most recent {retained_pair_count} assistant/tool-result pairs"
            )
        return (
            "Your job is to create a detailed summary of the conversation so far, "
            "paying close attention to the user's explicit requests and the work "
            "already completed.\n\n"
            f"Original task:\n{(original_instruction or '').strip()}\n\n"
            "You are summarizing the earlier portion of the conversation only. "
            f"{recent_message_note} will remain in the conversation verbatim, so do "
            "not repeat them in detail unless they are required for continuity.\n\n"
            "Do not assume you can read external files or use tools. Work only from "
            "the conversation provided in this context.\n\n"
            "Before providing the summary, think through what matters most to "
            "preserve. Write the response using exactly these tags:\n"
            "<analysis>\n"
            "Brief internal reasoning about what to preserve.\n"
            "</analysis>\n"
            "<summary>\n"
            "Your summary here.\n"
            "</summary>\n\n"
            "Inside <summary>, organize the content so the next model can continue "
            "without reading any external files. Include:\n"
            "1. Primary Request and Intent\n"
            "2. Key Technical Concepts\n"
            "3. Files and Code Sections examined or changed\n"
            "4. Errors encountered and how they were addressed\n"
            "5. Problem Solving and important decisions\n"
            "6. All user messages and explicit asks from the summarized portion\n"
            "7. Pending Tasks\n"
            "8. Current Work\n"
            "9. Optional Next Step\n\n"
            "Be concrete. Preserve exact commands, outputs, code behavior, and error "
            "text when those details matter for the work to continue. Do not output JSON."
        )

    @staticmethod
    def _set_loss_mask_prefix_false(
        messages: list[dict[str, Any]],
        prefix_count: int,
    ) -> None:
        for idx, message in enumerate(messages):
            if idx >= prefix_count:
                break
            message.pop("_loss_mask", None)
            message["loss_mask"] = False

    @staticmethod
    def _extract_tool_calls_from_message(
        message: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(message, dict):
            return []
        raw_tool_calls = message.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            return []

        normalized_tool_calls: list[dict[str, Any]] = []
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
                    "function_name": (
                        function_name.strip()
                        if isinstance(function_name, str)
                        else ""
                    ),
                    "tool_call_id": (
                        tool_call_id.strip()
                        if isinstance(tool_call_id, str)
                        else ""
                    ),
                }
            )
        return normalized_tool_calls

    @classmethod
    def _message_is_context_management_call(cls, message: dict[str, Any] | None) -> bool:
        tool_calls = cls._extract_tool_calls_from_message(message)
        return bool(tool_calls) and all(
            tool_call.get("function_name") == "context_management"
            for tool_call in tool_calls
        )

    @classmethod
    def _get_last_non_context_tool_call_id(
        cls,
        message: dict[str, Any] | None,
    ) -> str | None:
        tool_calls = cls._extract_tool_calls_from_message(message)
        for tool_call in reversed(tool_calls):
            function_name = tool_call.get("function_name")
            if function_name and function_name != "context_management":
                tool_call_id = tool_call.get("tool_call_id")
                return tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None
        return None

    @staticmethod
    def _message_main_key(message: dict[str, Any] | None) -> str | None:
        if not isinstance(message, dict):
            return None
        main_key = message.get("main_key")
        if not isinstance(main_key, str):
            return None
        main_key = main_key.strip()
        return main_key or None

    @staticmethod
    def _history_starts_with_rendered_tools_system_prompt(
        message_history: list[dict[str, Any]] | None,
    ) -> bool:
        if not message_history:
            return False
        first_message = message_history[0]
        if not isinstance(first_message, dict) or first_message.get("role") != "system":
            return False
        content = first_message.get("content")
        if not isinstance(content, str):
            return False
        return "# Tools" in content and "<tools>" in content and "<tool_call>" in content

    @staticmethod
    def _summary_generation_message_history(
        chat: Chat,
        *,
        include_rendered_system_prompt: bool,
    ) -> list[dict[str, Any]]:
        if include_rendered_system_prompt:
            return chat.messages_with_rendered_system_prompt()
        return chat.messages

    def _build_compaction_candidate_messages(
        self,
        chat_messages: list[dict[str, Any]],
        pending_observation: str | None = None,
    ) -> list[dict[str, Any]]:
        candidate_messages = [dict(msg) for msg in chat_messages[1:]]
        if not candidate_messages:
            return []

        while (
            candidate_messages
            and candidate_messages[-1].get("role") == "assistant"
            and self._message_is_context_management_call(candidate_messages[-1])
        ):
            candidate_messages.pop()

        if not candidate_messages:
            return []

        pending_text = pending_observation.strip() if isinstance(pending_observation, str) else ""
        if not pending_text:
            return candidate_messages

        last_message = candidate_messages[-1]
        if last_message.get("role") != "assistant":
            return candidate_messages

        last_tool_call_id = self._get_last_non_context_tool_call_id(last_message)
        if not last_tool_call_id:
            return candidate_messages

        candidate_messages.append(
            {
                "role": "tool",
                "content": pending_text,
                "content_origin": pending_text,
                "tool_call_id": last_tool_call_id,
                **(
                    {"main_key": self._message_main_key(last_message)}
                    if self._message_main_key(last_message)
                    else {}
                ),
            }
        )
        return candidate_messages

    def _build_summary_generation_history(
        self,
        chat_messages: list[dict[str, Any]],
        pending_observation: str | None = None,
    ) -> list[dict[str, Any]]:
        summary_history: list[dict[str, Any]] = []
        if chat_messages and chat_messages[0].get("role") == "user":
            summary_history.append(dict(chat_messages[0]))
        summary_history.extend(
            self._build_compaction_candidate_messages(
                chat_messages,
                pending_observation=pending_observation,
            )
        )
        return summary_history

    @classmethod
    def _build_tool_result_pairs(
        cls,
        messages: list[dict[str, Any]],
        start_index: int = 0,
    ) -> list[tuple[int, int]]:
        """Group history into assistant/tool pairs without splitting the pair."""
        pairs: list[tuple[int, int]] = []
        i = max(0, start_index)
        while i < len(messages):
            current = messages[i]
            if current.get("role") != "assistant":
                i += 1
                continue
            if cls._message_is_context_management_call(current):
                i += 1
                continue

            start_i = i
            end_i = i
            last_tool_call_id = cls._get_last_non_context_tool_call_id(current)
            next_index = i + 1
            if next_index < len(messages) and messages[next_index].get("role") == "tool":
                next_tool_call_id = messages[next_index].get("tool_call_id")
                if (
                    not last_tool_call_id
                    or not next_tool_call_id
                    or next_tool_call_id == last_tool_call_id
                ):
                    end_i = next_index
                    i = next_index + 1
                else:
                    i += 1
            else:
                i += 1
            pairs.append((start_i, end_i))
        return pairs

    def _split_summary_history(
        self,
        chat_messages: list[dict[str, Any]],
        pending_observation: str | None = None,
        retained_pair_count: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self._summary_history_user_turns < 0:
            return [], []

        # Skip the original task prompt because it is replaced by the summary prompt.
        candidate_messages = self._build_compaction_candidate_messages(
            chat_messages,
            pending_observation=pending_observation,
        )
        if not candidate_messages:
            return [], []

        tool_result_pairs = self._build_tool_result_pairs(candidate_messages)
        if not tool_result_pairs:
            return [], []

        if retained_pair_count is None:
            retained_pair_count = self._summary_history_user_turns
        retained_pair_count = max(
            0,
            min(int(retained_pair_count), len(tool_result_pairs)),
        )

        if retained_pair_count == 0:
            retained_pairs: list[tuple[int, int]] = []
            summarized_pairs = tool_result_pairs
        else:
            retained_pairs = tool_result_pairs[-retained_pair_count:]
            summarized_pairs = tool_result_pairs[:-retained_pair_count]

        summarized_messages: list[dict[str, Any]] = []
        for start_i, end_i in summarized_pairs:
            summarized_messages.extend(candidate_messages[start_i : end_i + 1])

        retained_messages: list[dict[str, Any]] = []
        for start_i, end_i in retained_pairs:
            retained_messages.extend(candidate_messages[start_i : end_i + 1])
        return summarized_messages, retained_messages

    def _select_summary_history_tail_pair_count(
        self,
        chat_messages: list[dict[str, Any]],
        pending_observation: str | None = None,
    ) -> int:
        requested_pair_count = max(0, int(self._summary_history_user_turns))
        if requested_pair_count == 0:
            return 0

        context_limit = self._llm.get_model_context_limit()
        if not context_limit:
            return requested_pair_count

        reserved_tokens = getattr(self._llm, "_max_new_tokens", None) or 8192
        min_free_tokens = max(self._proactive_summarization_threshold, reserved_tokens)

        tool_result_pairs = self._build_tool_result_pairs(
            self._build_compaction_candidate_messages(
                chat_messages,
                pending_observation=pending_observation,
            )
        )
        if not tool_result_pairs:
            return 0

        retained_pair_count = min(requested_pair_count, len(tool_result_pairs))
        while retained_pair_count >= 0:
            _, retained_messages = self._split_summary_history(
                chat_messages,
                pending_observation=pending_observation,
                retained_pair_count=retained_pair_count,
            )
            rebuilt_messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": self._build_summary_rehydration_user_message(
                        "",
                        recent_messages_preserved=bool(retained_messages),
                    ),
                },
                *retained_messages,
            ]
            rebuilt_tokens = (
                self._count_tokens_for_messages(rebuilt_messages)
                + self._SUMMARY_REBUILD_ESTIMATED_TOKENS
            )
            free_tokens = context_limit - rebuilt_tokens
            if free_tokens >= min_free_tokens:
                if retained_pair_count != requested_pair_count:
                    self._logger.debug(
                        "Reduced summary history tail from "
                        f"{requested_pair_count} to {retained_pair_count} "
                        "assistant/tool-result pairs to avoid immediate "
                        "post-compaction context management. "
                        f"estimated_summary_tokens={self._SUMMARY_REBUILD_ESTIMATED_TOKENS}, "
                        f"free_tokens≈{free_tokens}, min_free_tokens={min_free_tokens}."
                    )
                return retained_pair_count
            retained_pair_count -= 1

        return 0

    def _build_summary_subagent_history(
        self,
        chat_messages: list[dict[str, Any]],
        pending_observation: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        summarized_messages, retained_messages = self._split_summary_history(
            chat_messages,
            pending_observation=pending_observation,
        )

        summary_history: list[dict[str, Any]] = []
        if chat_messages and chat_messages[0].get("role") == "user":
            summary_history.append(dict(chat_messages[0]))
        summary_history.extend(dict(msg) for msg in summarized_messages)
        return summary_history, retained_messages

    def _trim_message_history_keep_latest_pairs(
        self,
        messages: list[dict[str, Any]],
        extra_messages: list[dict[str, str]],
        max_new_tokens: int,
        reason: str,
    ) -> list[dict[str, Any]]:
        trimmed_messages = [dict(msg) for msg in messages]
        context_limit = self._llm.get_model_context_limit()
        if not context_limit:
            return trimmed_messages

        removed_chunks = 0
        while len(trimmed_messages) > 1:
            total_tokens = self._count_tokens_for_messages(
                trimmed_messages + extra_messages
            )
            if total_tokens + max_new_tokens <= context_limit:
                break
            trimmed_messages = trimmed_messages[2:]
            removed_chunks += 1

        free_tokens = context_limit - self._count_tokens_for_messages(
            trimmed_messages + extra_messages
        )
        self._logger.debug(
            f"{reason}: Remaining messages={len(trimmed_messages)}, "
            f"chunks_removed={removed_chunks}, free_tokens≈{free_tokens}"
        )
        return trimmed_messages

    async def _materialize_pending_tool_result_before_context_management(
        self,
        chat: Chat,
        session: TmuxSession,
        pending_observation: str | None = None,
    ) -> bool:
        """Append the missing tool result for a trailing assistant tool call.

        Context management should not split between an assistant tool call and its
        corresponding tool result. If the current chat still ends with such an
        assistant message, persist the pending tool observation as the matching
        tool message before any truncation/summarization runs.
        """
        if not chat.messages:
            return False

        last_message = chat.messages[-1]
        if not isinstance(last_message, dict) or last_message.get("role") != "assistant":
            return False

        last_tool_call_id = self._get_last_non_context_tool_call_id(last_message)
        if not last_tool_call_id:
            return False

        if isinstance(pending_observation, str) and pending_observation.strip():
            terminal_output = self._limit_output_length(pending_observation)
        else:
            terminal_output = self._limit_output_length(
                await self._get_incremental_output_profiled(
                    session,
                    metadata={
                        "episode": self._n_episodes,
                        "trigger": "pre_context_management_pending_tool_result",
                    },
                )
            )
        if not isinstance(terminal_output, str) or not terminal_output.strip():
            self._logger.debug(
                "Trailing assistant tool call detected before context management, "
                "but no terminal output was available to materialize as a tool result."
            )
            return False

        chat.append_message(
            {
                "role": "tool",
                "content": terminal_output,
                "content_origin": terminal_output,
                "tool_call_id": last_tool_call_id,
                **(
                    {"main_key": self._message_main_key(last_message)}
                    if self._message_main_key(last_message)
                    else {}
                ),
            },
            token_counter=self._count_tokens_for_messages,
        )
        self._logger.debug(
            "Materialized trailing tool result before context management for "
            "tool_call_id=%s.",
            last_tool_call_id,
        )
        return True

    def _rebuild_main_chat_after_summary(
        self,
        chat: Chat,
        summary_response: LLMResponse,
        retained_pair_count: int,
        pending_observation: str | None = None,
    ) -> None:
        source_messages = chat.messages[:-2] if len(chat.messages) >= 2 else []
        _, history_tail = self._split_summary_history(
            source_messages,
            pending_observation=pending_observation,
            retained_pair_count=retained_pair_count,
        )
        rebuilt_messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._build_summary_rehydration_user_message(
                    summary_response.content,
                    recent_messages_preserved=bool(history_tail),
                ),
            },
            *history_tail,
        ]
        self._set_loss_mask_prefix_false(
            rebuilt_messages,
            prefix_count=len(rebuilt_messages),
        )
        chat.set_messages(rebuilt_messages)
        chat.reset_response_chain()
        self._resume_generation_from_existing_user = True
        self._logger.debug(
            "Rebuilt main chat after summary. "
            f"Retained assistant/tool-result pairs={retained_pair_count}, "
            f"messages={len(rebuilt_messages)}."
        )

    async def _check_proactive_summarization(
        self,
        chat: Chat,
        original_instruction: str,
        session: TmuxSession,
        prompt: str | None,
    ) -> tuple[str | None, list[SubagentTrajectoryRef] | None] | None:
        """Automatic context management before the next LLM call when nearing overflow."""
        if not self._should_automatically_manage_context_length():
            return None

        if self._pending_completion:
            return None

        context_limit = self._llm.get_model_context_limit()
        current_tokens = self._count_total_tokens(chat, prompt)
        free_tokens = context_limit - current_tokens
        reserved_tokens = getattr(self._llm, "_max_new_tokens", None) or 8192
        min_free_tokens = max(self._proactive_summarization_threshold, reserved_tokens)

        if free_tokens >= min_free_tokens:
            return None

        action_label = (
            "summarizing"
            if self._summarization_mode == "summarize"
            else "truncating"
        )
        self._logger.debug(
            f"Proactively {action_label}. "
            f"Free tokens: approximately {free_tokens}. "
            f"Reserved tokens: {reserved_tokens}. "
            f"Threshold: {self._proactive_summarization_threshold}."
        )
        await self._materialize_pending_tool_result_before_context_management(
            chat,
            session,
            pending_observation=prompt,
        )
        if self._summarization_mode == "truncate":
            self._ensure_context_management_budget(trigger="proactive_truncate")
            truncation_info = self._truncate_message_history_keep_last_half(chat)
            if truncation_info:
                self._queue_context_truncation_event("proactive", truncation_info)
                self._logger.debug(
                    "Proactively truncated chat history to free context."
                )
                return None, None
            return None

        self._ensure_context_management_budget(trigger="proactive_summarize")
        try:
            return await self._summarize(
                chat,
                original_instruction,
                session,
                continuation_prompt=prompt,
            )
        except ContextManagementLimitExceededError:
            raise
        except Exception as e:
            self.logger.error(f"Error in proactively summarizing: {e}")
            return None

    async def _execute_llm_request(
        self,
        chat: Chat,
        prompt: str | None,
        logging_path: Path | None,
        response_path: Path | None,
    ) -> LLMResponse:
        start_time = time.time()
        self._logger.debug("calling chat")
        main_api_base = (
            self._CLAUDE_QUERY_BASE_URL
            if self._use_claude_query_path()
            else (self._get_llm_api_base() or self._main_api_base)
        )
        self._append_router_usage_log(
            mode="main_chat",
            api_base=main_api_base,
            session_id=self._session_id,
        )
        if self._use_claude_query_path():
            llm_response = await self._call_claude_query(
                prompt=prompt,
                message_history=chat.messages,
                logging_path=logging_path,
            )
            chat.record_response(prompt, llm_response)
            self._record_main_key_on_latest_generate_turn(
                chat,
                prompt=prompt,
                response=llm_response,
            )
        else:
            llm_response = await chat.chat(
                prompt,
                logging_path=logging_path,
                **self._llm_call_kwargs,
            )
            self._record_main_key_on_latest_generate_turn(
                chat,
                prompt=prompt,
                response=llm_response,
            )
        end_time = time.time()
        request_time_ms = (end_time - start_time) * 1000
        self._api_request_times.append(request_time_ms)

        if response_path is not None:
            response_path.write_text(llm_response.content)
        return llm_response

    def _flush_pending_context_management_steps(self, chat: Chat) -> None:
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

        if self._pending_context_truncations:
            for truncation_info in self._pending_context_truncations:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="Performed context truncation to fit the model window.",
                        extra={"context_truncation": truncation_info},
                    )
                )
            self._pending_context_truncations = []

        if self._pending_handoff_prompt:
            if self._linear_history:
                self._split_trajectory_on_summarization(self._pending_handoff_prompt)
            else:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="user",
                        message=self._append_context_window_status(
                            chat, self._pending_handoff_prompt
                        ),
                    )
                )
            self._pending_handoff_prompt = None

    async def _summarize(
        self,
        chat: Chat,
        original_instruction: str,
        session: TmuxSession,
        continuation_prompt: str | None = None,
        managed_messages_snapshot: list[dict[str, Any]] | None = None,
    ) -> tuple[str | None, list[SubagentTrajectoryRef] | None]:
        """Create a single-stage handoff summary for context management.

        Args:
            chat: Chat object containing conversation history (may be unwound)
            original_instruction: The original task instruction from user
            session: TmuxSession for capturing current terminal state

        Returns:
            tuple: (handoff_prompt, subagent_trajectory_refs)
                - handoff_prompt: The prompt to continue with after summary handoff
                - subagent_trajectory_refs: List containing the summary subagent trajectory ref
        """
        if len(chat.messages) == 0:
            return original_instruction, None

        await self._materialize_pending_tool_result_before_context_management(
            chat,
            session,
            pending_observation=continuation_prompt,
        )

        self._ensure_context_management_budget(trigger="summarize")
        self._summarization_count += 1
        subagent_trajectory_refs = []
        _ = managed_messages_snapshot

        summary_session_id = (
            f"{self._session_id}-summarization-{self._summarization_count}-summary"
        )
        next_main_chat_key = (
            self._preview_next_tito_main_key() if self._enable_tito else self._session_id
        )
        summary_generation_key = self._session_id
        if self._enable_tito and self._summary_training == "main_chat":
            summary_generation_key = self._preview_summary_tito_main_key()

        retained_pair_count = self._select_summary_history_tail_pair_count(
            chat.messages,
        )
        summary_prompt = self._build_claude_style_summary_prompt(
            original_instruction,
            retained_pair_count=retained_pair_count,
        )
        include_summary_rendered_system_prompt = (
            self._enable_tito and self._summary_training in {"full", "main_chat"}
        )
        summary_message_history = self._summary_generation_message_history(
            chat,
            include_rendered_system_prompt=include_summary_rendered_system_prompt,
        )

        summary_steps: list[Step] = []

        (
            summary_response,
            summary_trajectory_ref,
            summary_training_messages,
        ) = await self._run_subagent(
            prompt=summary_prompt,
            message_history=summary_message_history,
            steps=summary_steps,
            session_id=summary_session_id,
            agent_name="terminus-2-summarization-summary",
            filename_suffix="summary",
            summary_text=f"Context summarization {self._summarization_count}: Step 1 - Summary generation",
            subagent_name_for_logging="summary generation LLM call",
            llm_session_id_override=summary_generation_key,
            include_rendered_system_prompt=include_summary_rendered_system_prompt,
        )
        subagent_trajectory_refs.append(summary_trajectory_ref)
        _ = summary_training_messages
        self._rotate_tito_main_key(reason="summarize", new_key=next_main_chat_key)
        self._mark_context_management_boundary()
        self._append_summary_generation_turns_to_chat(
            chat,
            summary_prompt,
            summary_response,
            main_key=summary_generation_key if self._enable_tito else None,
        )
        self._append_rollout_messages_snapshot(
            self._build_main_chat_summary_generation_messages(
                chat.messages_with_weight_version()
            )
        )

        self._rebuild_main_chat_after_summary(
            chat=chat,
            summary_response=summary_response,
            retained_pair_count=retained_pair_count,
        )
        self._managed_messages.append(chat.messages_with_weight_version())

        return None, subagent_trajectory_refs

    @retry(
        stop=stop_after_attempt(3),
        # Do not retry since it is only raised when summarization is OFF, and we will
        # likely hit the context length limit again despite retrying.
        retry=(
            retry_if_not_exception_type(ContextLengthExceededError)
            & retry_if_not_exception_type(OutputLengthExceededError)
            & retry_if_not_exception_type(ParseError)
            & retry_if_not_exception_type(SummaryGenerationFailedError)
            # To avoid asyncio.CancelledError retries which inherits from BaseException
            # rather than Exception
            & retry_if_exception_type(Exception)
        ),
        reraise=True,
    )
    async def _query_llm(
        self,
        chat: Chat,
        prompt: str | None,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> LLMResponse:
        self._logger.debug("Querying LLM...")
        logging_path, prompt_path, response_path = logging_paths
        raw_prompt = prompt
        prompt = self._append_context_window_status(chat, raw_prompt)

        if prompt_path is not None:
            prompt_path.write_text(prompt or "")

        try:
            response = await self._execute_llm_request(
                chat=chat,
                prompt=prompt,
                logging_path=logging_path,
                response_path=response_path,
            )
            self._initial_generate_completed = True
            self._resume_generation_from_existing_user = False
            self._mark_post_context_management_generate_complete()
            return response

        except ContextLengthExceededError:
            self._context_length_exceeded_count += 1
            if self._awaiting_post_context_management_generate:
                if not self._should_automatically_manage_context_length():
                    self._post_context_management_generate_interrupted_reason = (
                        "context_length_before_post_context_management_generate"
                    )
                    self._logger.debug(
                        "Context length exceeded before the first post-context-management "
                        "generate could be sent, and automatic context management is OFF."
                    )
                    raise
                self._ensure_context_management_budget(
                    trigger="context_length_post_context_management"
                )
                self._post_context_management_generate_interrupted_reason = None
                self._logger.debug(
                    "Context length exceeded before the first post-context-management "
                    "generate could be sent. Continuing with another context "
                    "management pass."
                )
            if not self._should_automatically_manage_context_length():
                self.logger.debug(
                    "Context length exceeded and automatic context management is OFF."
                )
                raise

            if self._summarization_mode == "truncate":
                self._logger.debug(
                    "Context length exceeded. Using automatic truncation mode."
                )
                await self._materialize_pending_tool_result_before_context_management(
                    chat,
                    session,
                    pending_observation=raw_prompt,
                )
                max_attempts = 3
                for attempt in range(max_attempts):
                    self._ensure_context_management_budget(
                        trigger="context_length_truncate"
                    )
                    truncation_info = self._truncate_message_history_keep_last_half(chat)
                    if truncation_info:
                        self._queue_context_truncation_event(
                            "context_length", truncation_info
                        )
                    else:
                        self._logger.debug(
                            "Truncation skipped (insufficient history to drop)."
                        )
                    retry_prompt = None
                    if prompt_path is not None:
                        prompt_path.write_text(retry_prompt or "")
                    try:
                        response = await self._execute_llm_request(
                            chat=chat,
                            prompt=retry_prompt,
                            logging_path=logging_path,
                            response_path=response_path,
                        )
                        self._mark_post_context_management_generate_complete()
                        return response
                    except ContextLengthExceededError:
                        self._context_length_exceeded_count += 1
                        if attempt == max_attempts - 1:
                            raise
                raise RuntimeError(
                    "Context length exceeded after truncation attempts."
                )

            self._logger.debug(
                "Context length exceeded. Using automatic fallback summarization."
            )

            if session is None:
                raise RuntimeError("Cannot handle context length error without session")

            await self._materialize_pending_tool_result_before_context_management(
                chat,
                session,
                pending_observation=raw_prompt,
            )

            self._ensure_context_management_budget(
                trigger="context_length_fallback_summarize"
            )
            managed_messages_snapshot = chat.messages_with_weight_version()

            summary_prompt = None
            summary_failure: Exception | None = None
            try:
                self.logger.debug("SUMMARIZATION: Attempting full summary")
                summary_prompt, subagent_trajectory_refs = await self._summarize(
                    chat,
                    original_instruction,
                    session,
                    continuation_prompt=raw_prompt,
                    managed_messages_snapshot=managed_messages_snapshot,
                )
                self._full_summary_success_count += 1
                self._pending_subagent_refs = subagent_trajectory_refs
                self._pending_handoff_prompt = None
                self.logger.debug("SUMMARIZATION: Full summary succeeded")
            except Exception as e:
                summary_failure = e
                self.logger.debug(f"SUMMARIZATION: Full summary failed: {e}")

            if summary_prompt is None:
                if self._training_mode:
                    raise SummaryGenerationFailedError(
                        "Full summary generation failed after retries."
                    ) from summary_failure
                self.logger.error(
                    "SUMMARIZATION: Full summary failed; using synthetic assistant fallback."
                )
                llm_response = LLMResponse(
                    content="Technical difficulties. Please continue with the task.",
                    history_content="Technical difficulties. Please continue with the task.",
                )
                if response_path is not None:
                    response_path.write_text(llm_response.content)
                return llm_response

            next_prompt = (
                None if self._resume_generation_from_existing_user else summary_prompt
            )
            next_prompt = self._append_context_window_status(chat, next_prompt)
            if prompt_path is not None:
                prompt_path.write_text(next_prompt or "")

            try:
                response = await self._execute_llm_request(
                    chat=chat,
                    prompt=next_prompt,
                    logging_path=logging_path,
                    response_path=response_path,
                )
                self._resume_generation_from_existing_user = False
                self._mark_post_context_management_generate_complete()
                return response
            except Exception as e:
                self.logger.error(f"Even fallback chat failed: {e}")
                llm_response = LLMResponse(
                    content="Technical difficulties. Please continue with the task.",
                    history_content="Technical difficulties. Please continue with the task.",
                )
                if response_path is not None:
                    response_path.write_text(llm_response.content)
                return llm_response

        except OutputLengthExceededError as e:
            self._output_length_exceeded_count += 1
            self._logger.debug(f"Output length exceeded: {e}")

            raw_truncated_response = getattr(e, "truncated_response", None)
            truncated_response = (
                raw_truncated_response
                if isinstance(raw_truncated_response, str)
                else "[TRUNCATED RESPONSE NOT AVAILABLE]"
            )

            # Try to salvage a valid response from the truncated output
            # Only available for XML parser
            salvaged_response = None
            has_multiple_blocks = False

            if hasattr(self._parser, "salvage_truncated_response"):
                salvaged_response, has_multiple_blocks = (
                    self._parser.salvage_truncated_response(truncated_response)
                )

            if salvaged_response:
                self.logger.debug(
                    "Output exceeded length but found valid response. "
                    "Using truncated version."
                )

                if response_path is not None:
                    response_path.write_text(salvaged_response)

                return salvaged_response

            # If we get here, we couldn't salvage a valid response
            # Try to parse the truncated response to get warnings
            warnings_text = ""
            try:
                parse_result = self._parser.parse_response(truncated_response)
                if parse_result.warning:
                    warnings_text = (
                        f"\n\nParser warnings from your truncated response:\n"
                        f"{parse_result.warning}"
                    )
            except Exception as parse_error:
                self.logger.debug(f"Failed to parse truncated response: {parse_error}")

            # Get the actual output limit for the model
            output_limit = self._llm.get_model_output_limit()
            if output_limit is not None:
                limit_str = f"{output_limit} tokens"
            else:
                limit_str = "the maximum output length"

            error_msg = (
                "ERROR!! NONE of the actions you just requested were performed "
                f"because you exceeded {limit_str}. "
                f"Your outputs must be less than {limit_str}. Re-issue this request, "
                f"breaking it into chunks each of which is less than {limit_str}."
            )

            if warnings_text:
                error_msg += warnings_text

            self._append_output_length_retry_messages(
                chat,
                prompt=prompt,
                raw_truncated_response=raw_truncated_response
                if isinstance(raw_truncated_response, str)
                else None,
                retry_prompt=error_msg,
            )

            if response_path is not None:
                response_path.write_text(error_msg)

            return await self._query_llm(
                chat=chat,
                prompt=error_msg,
                logging_paths=logging_paths,
                original_instruction=original_instruction,
                session=session,
            )

        except ParseError as e:
            self._format_error_count += 1
            if self._training_mode:
                self._logger.error("Format error detected; aborting: %s", e)
            else:
                self._logger.error("Format error detected: %s", e)
            raise

        except Exception as e:
            self.logger.error(f"Unknown Error in LLM interaction: {e}")
            raise e

    @staticmethod
    def _split_think_content(text: str) -> tuple[str, str]:
        """Split <think> reasoning from content and avoid leaking think tags."""
        if not text:
            return "", ""
        think_close_tag = "</think>"
        if think_close_tag not in text:
            if "<think>" in text:
                _, tail = text.split("<think>", 1)
                return "", tail.strip()
            return text.strip(), ""

        reasoning, content = text.split(think_close_tag, 1)
        reasoning_content = reasoning.replace("<think>", "").strip()
        sanitized_content = (
            content.replace("<think>", "").replace("</think>", "").strip()
        )
        return sanitized_content, reasoning_content

    def _append_output_length_retry_messages(
        self,
        chat: Chat,
        *,
        prompt: str | None,
        raw_truncated_response: str | None,
        retry_prompt: str,
    ) -> None:
        content, reasoning_content = self._split_think_content(
            raw_truncated_response or ""
        )
        main_key = self._session_id if self._enable_tito else None
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
            "weight_version": None,
            **({"main_key": main_key} if main_key else {}),
        }
        if isinstance(raw_truncated_response, str):
            assistant_message["history_content"] = raw_truncated_response
            assistant_message["content_origin"] = raw_truncated_response

        chat.messages.append(
            {
                "role": "user",
                "content": prompt,
                **({"main_key": main_key} if main_key else {}),
            }
        )
        chat.messages.append(assistant_message)
        chat.messages.append(
            {
                "role": "user",
                "content": retry_prompt,
                **({"main_key": main_key} if main_key else {}),
            }
        )
        chat.reset_response_chain()

    async def _perform_requested_context_management(
        self,
        chat: Chat,
        original_instruction: str,
        session: TmuxSession,
    ) -> str:
        """Execute one model-requested context-management action and return tool output."""
        if self._context_management_mode == "none":
            raise ParseError(
                "Model requested context management but enable_summarize is false."
            )

        if self._summarization_mode == "truncate":
            self._ensure_context_management_budget(trigger="model_requested_truncate")
            truncation_info = self._truncate_message_history_keep_last_half(chat)
            if truncation_info:
                self._queue_context_truncation_event("model_requested", truncation_info)
            else:
                self._logger.debug(
                    "Context management request skipped (insufficient history to truncate)."
                )
            if truncation_info:
                return ""
            terminal_state = self._limit_output_length(
                await session.capture_pane(capture_entire=False)
            )
            return (
                "Context management request was accepted, but there was not enough chat "
                "history to truncate.\n\nContinue working on this task.\n\nCurrent terminal state:\n"
                f"{terminal_state}"
            )

        self._ensure_context_management_budget(trigger="model_requested_summarize")
        handoff_prompt, subagent_trajectory_refs = await self._summarize(
            chat, original_instruction, session
        )
        self._full_summary_success_count += 1
        self._pending_subagent_refs = subagent_trajectory_refs
        self._pending_handoff_prompt = None
        return handoff_prompt

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str | None,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> tuple[list[Command], bool, bool, str, str, str, LLMResponse]:
        llm_response = await self._query_llm(
            chat, prompt, logging_paths, original_instruction, session
        )

        result = self._parser.parse_response(llm_response.content)
        self._record_format_issue(result)
        self._maybe_raise_format_error(result)

        feedback = ""
        if result.error:
            feedback += f"ERROR: {result.error}"
            if result.warning:
                feedback += f"\nWARNINGS: {result.warning}"
        elif result.warning:
            feedback += f"WARNINGS: {result.warning}"

        if (
            self._training_mode
            and llm_response.terminal_stop_token
            and llm_response.terminal_stop_token != "<|observation|>"
            and llm_response.finish_reason not in {"length", "max_tokens", "max_new_tokens"}
        ):
            stop_token_feedback = (
                "WARNINGS: Generate stopped on an unexpected stop token "
                f"{llm_response.terminal_stop_token!r} instead of '<|observation|>'."
            )
            feedback = (
                f"{feedback}\n{stop_token_feedback}".strip()
                if feedback
                else stop_token_feedback
            )

        if result.warning:
            self.logger.debug(f"Parser warnings: {result.warning}")

        commands = []
        for parsed_cmd in result.commands:
            commands.append(
                Command(
                    keystrokes=parsed_cmd.keystrokes,
                    duration_sec=min(parsed_cmd.duration, 60),
                )
            )

        return (
            commands,
            result.request_context_management,
            result.is_task_complete,
            feedback,
            result.analysis,
            result.plan,
            llm_response,
        )

    def _maybe_raise_format_error(self, parse_result) -> None:
        if not self._training_mode:
            return

        issues = []
        if parse_result.error:
            issues.append(f"ERROR: {parse_result.error}")
        if parse_result.warning:
            filtered_warning = self._filter_tolerated_warnings(parse_result.warning)
            if filtered_warning:
                issues.append(f"WARNINGS: {filtered_warning}")

        if not issues:
            return

        issue_text = "\n".join(issues)
        self._logger.error("Format error detected; aborting: %s", issue_text)
        raise ParseError(f"Format error detected: {issue_text}")

    def _record_format_issue(self, parse_result) -> None:
        if parse_result.error:
            self._format_error_count += 1
            return
        if parse_result.warning:
            filtered_warning = self._filter_tolerated_warnings(parse_result.warning)
            if filtered_warning:
                self._format_error_count += 1

    @classmethod
    def _filter_tolerated_warnings(cls, warning: str) -> str:
        if not warning:
            return ""
        lines = [line.strip() for line in warning.splitlines() if line.strip()]
        filtered = []
        for line in lines:
            if line.startswith("- "):
                line = line[2:].strip()
            if any(substr in line for substr in cls._TOLERATED_WARNING_SUBSTRINGS):
                continue
            filtered.append(line)
        if not filtered:
            return ""
        return "- " + "\n- ".join(filtered)

    @staticmethod
    def _is_training_mode() -> bool:
        training_flag = os.environ.get("TRAINING", "")
        return training_flag.strip().lower() in ("1", "true", "yes", "y", "t")

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        """Execute a batch of commands in the terminal.

        Args:
            commands: List of commands to execute
            session: TmuxSession instance

        Returns:
            Tuple of (timeout_occurred, terminal_output)
        """
        for command in commands:
            try:
                await session.send_keys(
                    command.keystrokes,
                    block=False,
                    min_timeout_sec=command.duration_sec,
                )
            except TimeoutError:
                return True, self._timeout_template.format(
                    timeout_sec=command.duration_sec,
                    command=command.keystrokes,
                    terminal_state=self._limit_output_length(
                        await self._get_incremental_output_profiled(
                            session,
                            metadata={
                                "episode": self._n_episodes,
                                "trigger": "command_timeout",
                            },
                        )
                    ),
                )

        return False, self._limit_output_length(
            await self._get_incremental_output_profiled(
                session,
                metadata={
                    "episode": self._n_episodes,
                    "command_count": len(commands),
                    "trigger": "post_command_observation",
                },
            )
        )

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: Chat,
        logging_dir: Path | None = None,
        original_instruction: str = "",
    ) -> None:
        if self._context is None:
            raise RuntimeError("Agent context is not set. This should never happen.")

        prompt: str | None = initial_prompt

        self._context.n_input_tokens = 0
        self._context.n_output_tokens = 0
        self._context.n_cache_tokens = 0
        self._context.cost_usd = None

        if self._session is None:
            raise RuntimeError("Session is not set. This should never happen.")

        # Step ID offset accounts for initial steps in the trajectory:
        # Step 1: system message (includes task instruction)
        # Steps 2+: agent episodes (starting from episode 0)

        for episode in range(self._max_episodes):
            self._n_episodes = episode + 1
            if not await self._session.is_session_alive():
                if self._awaiting_post_context_management_generate:
                    self._post_context_management_generate_interrupted_reason = (
                        "session_ended_before_post_context_management_generate"
                    )
                    self.logger.debug(
                        "Session ended before the first post-context-management generate; "
                        "stopping without exporting the rebuilt continuation."
                    )
                self.logger.debug("Session has ended, breaking out of agent loop")
                return

            if (
                original_instruction
                and self._should_automatically_manage_context_length()
            ):
                if self._awaiting_post_context_management_generate:
                    self._post_context_management_generate_interrupted_reason = (
                        "auto_context_management_before_generate"
                    )
                    self.logger.debug(
                        "Another automatic context-management check would run before the "
                        "first post-context-management generate. Stopping early."
                    )
                    return
                proactive_summary_result = await self._check_proactive_summarization(
                    chat,
                    original_instruction,
                    self._session,
                    prompt,
                )
                if proactive_summary_result:
                    prompt, subagent_refs = proactive_summary_result
                    self._pending_subagent_refs = subagent_refs
                    self._pending_handoff_prompt = prompt
                    if self._resume_generation_from_existing_user:
                        prompt = None

            logging_paths = self._setup_episode_logging(logging_dir, episode)

            # Track token counts and cost before this step
            tokens_before_input = chat.total_input_tokens
            tokens_before_output = chat.total_output_tokens
            tokens_before_cache = chat.total_cache_tokens
            cost_before = chat.total_cost

            (
                commands,
                request_context_management,
                is_task_complete,
                feedback,
                analysis,
                plan,
                llm_response,
            ) = await self._handle_llm_interaction(
                chat, prompt, logging_paths, original_instruction, self._session
            )

            if (
                self._should_automatically_manage_context_length()
                and not request_context_management
            ):
                self._flush_pending_context_management_steps(chat)

            # Create message content from analysis and plan, or use raw response if raw_content is enabled
            if self._save_raw_content_in_trajectory:
                # Use the raw LLM response content for SFT data export
                message_content = llm_response.content
            else:
                # Parse into structured format (analysis + plan)
                message_parts = []
                if analysis:
                    message_parts.append(f"Analysis: {analysis}")
                if plan:
                    message_parts.append(f"Plan: {plan}")
                message_content = "\n".join(message_parts) if message_parts else ""

            self._context.n_input_tokens = chat.total_input_tokens
            self._context.n_output_tokens = chat.total_output_tokens
            self._context.n_cache_tokens = chat.total_cache_tokens
            self._context.cost_usd = chat.total_cost if chat.total_cost > 0 else None

            if feedback and "ERROR:" in feedback:
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    f"Please fix these issues and provide a proper "
                    f"{self._get_error_response_type()}."
                )
                # For error cases, we still want to record the step
                # Use the raw response as the message since parsing failed
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before

                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=llm_response.content,
                        reasoning_content=llm_response.reasoning_content,
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    content=prompt,
                                )
                            ]
                        ),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens
                            - tokens_before_output,
                            cached_tokens=cache_tokens_used
                            if cache_tokens_used > 0
                            else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                continue

            if request_context_management:
                self._record_asciinema_marker(
                    f"Episode {episode}: context management",
                )
                observation = await self._perform_requested_context_management(
                    chat,
                    original_instruction,
                    self._session,
                )
                self._pending_completion = False
                if feedback and "WARNINGS:" in feedback:
                    observation = (
                        f"Previous response had warnings:\n{feedback}\n\n{observation}"
                    )

                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before

                tool_calls: list[ToolCall] | None = None
                observation_results = [ObservationResult(content=observation)]
                if not self._save_raw_content_in_trajectory:
                    tool_calls = [
                        ToolCall(
                            tool_call_id=f"call_{episode}_context_management",
                            function_name="context_management",
                            arguments={"mode": self._summarization_mode},
                        )
                    ]

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
                            completion_tokens=chat.total_output_tokens
                            - tokens_before_output,
                            cached_tokens=cache_tokens_used
                            if cache_tokens_used > 0
                            else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                self._attach_tool_calls_to_last_assistant_message(chat, tool_calls)

                self._flush_pending_context_management_steps(chat)
                prompt = None if self._resume_generation_from_existing_user else observation
                self._dump_trajectory()
                continue

            self._record_asciinema_marker(
                f"Episode {episode}: {len(commands)} commands",
            )

            if not commands and not is_task_complete:
                no_action_observation = self._get_no_action_observation_message()
                if self._training_mode:
                    self._format_error_count += 1
                    self._logger.error(
                        "Format error detected; assistant response contained no "
                        "commands, context-management request, or task-complete "
                        "signal."
                    )
                    raise ParseError(
                        "Format error detected: Response did not include any "
                        "commands, context_management request, or task_complete "
                        "signal."
                    )

                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before
                observation_results = [
                    ObservationResult(
                        content=no_action_observation,
                    )
                ]
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=message_content,
                        reasoning_content=llm_response.reasoning_content,
                        observation=Observation(results=observation_results),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens
                            - tokens_before_output,
                            cached_tokens=cache_tokens_used
                            if cache_tokens_used > 0
                            else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                chat.append_message(
                    {
                        "role": "tool",
                        "content": no_action_observation,
                        "content_origin": no_action_observation,
                        **(
                            {"main_key": self._message_main_key(chat.messages[-1])}
                            if self._message_main_key(chat.messages[-1])
                            else {}
                        ),
                    },
                    token_counter=self._count_tokens_for_messages,
                )
                self._dump_trajectory()
                prompt = None
                continue

            timeout_occurred, terminal_output = await self._execute_commands(
                commands,
                self._session,
            )

            # Capture the pending completion state before potentially modifying it
            was_pending_completion = self._pending_completion

            # Construct the observation (what gets sent back to the LLM)
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

            # Record the step in trajectory
            cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
            step_cost = chat.total_cost - cost_before

            # Create tool_calls array from commands and task completion
            # Note: Although Terminus 2 doesn't offer native tool calling (it uses text-based command parsing),
            # we represent parsed commands as tool calls for better trajectory analysis and compatibility with tooling.
            # However, when raw_content mode is enabled, we skip tool_calls generation to preserve the raw LLM response.
            tool_calls: list[ToolCall] | None = None
            observation_results: list[ObservationResult] = []

            if not self._save_raw_content_in_trajectory:
                # Only create tool_calls when NOT in raw_content mode
                tool_calls_list: list[ToolCall] = []

                if commands:
                    for i, cmd in enumerate(commands):
                        tool_call_id = f"call_{episode}_{i + 1}"
                        tool_calls_list.append(
                            ToolCall(
                                tool_call_id=tool_call_id,
                                function_name="bash_command",
                                arguments={
                                    "keystrokes": cmd.keystrokes,
                                    "duration": cmd.duration_sec,
                                },
                            )
                        )

                    # Add observation result after all tool calls are created
                    # Note: All commands share the same terminal output in this architecture,
                    # so we omit source_call_id to indicate the observation applies to the entire step.
                    observation_results.append(
                        ObservationResult(
                            content=observation,
                        )
                    )

                # Add task_complete as a tool call if the agent marked the task complete
                if is_task_complete:
                    task_complete_call_id = f"call_{episode}_task_complete"
                    tool_calls_list.append(
                        ToolCall(
                            tool_call_id=task_complete_call_id,
                            function_name="mark_task_complete",
                            arguments={},
                        )
                    )
                    # If there are no commands, we still need to add an observation result
                    if not commands:
                        observation_results.append(
                            ObservationResult(
                                content=observation,
                            )
                        )
                elif not commands:
                    # No commands and no task completion, just the observation
                    observation_results.append(
                        ObservationResult(
                            content=observation,
                        )
                    )

                tool_calls = tool_calls_list or None
            else:
                # In raw_content mode, just add observation without tool_calls
                observation_results.append(
                    ObservationResult(
                        content=observation,
                    )
                )

            # Build the step object using Pydantic models
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
                        completion_tokens=chat.total_output_tokens
                        - tokens_before_output,
                        cached_tokens=cache_tokens_used
                        if cache_tokens_used > 0
                        else None,
                        cost_usd=step_cost if step_cost > 0 else None,
                        prompt_token_ids=llm_response.prompt_token_ids,
                        completion_token_ids=llm_response.completion_token_ids,
                        logprobs=llm_response.logprobs,
                    ),
                )
            )
            self._attach_tool_calls_to_last_assistant_message(chat, tool_calls)
            appended_tool_observation = self._append_tool_observation_to_chat(
                chat,
                observation,
                tool_calls,
                token_counter=self._count_tokens_for_messages,
            )

            # Dump trajectory after every episode to ensure we persist even if program terminates
            self._dump_trajectory()

            if is_task_complete:
                if was_pending_completion:
                    # Task is confirmed complete (this is the second time task_complete was True), return
                    return
                else:
                    # First completion attempt - ask for confirmation and continue
                    prompt = None if appended_tool_observation else observation
                    continue

            prompt = None if appended_tool_observation else observation

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        self._chat = Chat(
            self._llm,
            interleaved_thinking=self._interleaved_thinking,
            history_token_counter=self._count_tokens_for_messages,
            profiling_callback=self._chat_profiling_callback,
        )
        self._context = context

        if self._session is None:
            raise RuntimeError("Session is not set")

        # Get the terminal state for the initial prompt
        terminal_state = self._limit_output_length(
            await self._session.get_incremental_output()
        )

        augmented_instruction = instruction
        if self.mcp_servers:
            mcp_info = "\n\nMCP Servers:\nThe following MCP servers are available for this task.\n"
            for s in self.mcp_servers:
                if s.transport == "stdio":
                    args_str = " ".join(s.args)
                    mcp_info += f"- {s.name}: stdio transport, command: {s.command} {args_str}\n"
                else:
                    mcp_info += f"- {s.name}: {s.transport} transport, url: {s.url}\n"
            augmented_instruction = instruction + mcp_info

        skills_section = await self._build_skills_section(environment)
        if skills_section:
            augmented_instruction += skills_section

        initial_system_prompt = self._get_initial_system_prompt(
            augmented_instruction,
            terminal_state,
        )
        if isinstance(initial_system_prompt, str) and initial_system_prompt.strip():
            self._chat.set_rendered_system_prompt(initial_system_prompt)
        initial_prompt = self._get_initial_user_prompt(
            augmented_instruction,
            terminal_state,
        )
        initial_prompt = self._append_context_window_status(self._chat, initial_prompt)

        self._trajectory_steps.append(
            Step(
                step_id=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=initial_prompt,
            )
        )

        actual_episodes = self._n_episodes
        try:
            await self._run_agent_loop(
                initial_prompt=initial_prompt,
                chat=self._chat,
                logging_dir=self.logs_dir,
                original_instruction=instruction,
            )
        finally:
            # Preserve in-flight episode count even if the agent is cancelled or times out.
            actual_episodes = max(actual_episodes, self._n_episodes)
            # Always capture metrics and rollout details even if the agent times out or errors
            # Combine main chat rollout details with subagent rollout details
            context.rollout_details = (
                self._chat.rollout_details + self._subagent_rollout_details
            )

            # Include subagent metrics in context totals
            context.n_input_tokens = (
                self._chat.total_input_tokens
                + self._subagent_metrics.total_prompt_tokens
            )
            context.n_output_tokens = (
                self._chat.total_output_tokens
                + self._subagent_metrics.total_completion_tokens
            )
            context.n_cache_tokens = (
                self._chat.total_cache_tokens
                + self._subagent_metrics.total_cached_tokens
            )
            total_cost = self._chat.total_cost + self._subagent_metrics.total_cost_usd
            context.cost_usd = total_cost if total_cost > 0 else None
            context.metadata = {
                "n_episodes": actual_episodes,
                "api_request_times_msec": self._api_request_times,
                "summarization_count": (
                    self._summarization_count + self._context_truncation_count
                ),
                "full_summary_success_count": self._full_summary_success_count,
                "short_summary_success_count": self._short_summary_success_count,
                "ultimate_summary_success_count": self._ultimate_summary_success_count,
                "format_error_count": self._format_error_count,
                "context_length_exceeded_count": self._context_length_exceeded_count,
                "output_length_exceeded_count": self._output_length_exceeded_count,
                "context_truncation_count": self._context_truncation_count,
                "context_management_max_count": self._context_management_max_count,
                "post_context_management_generate_pending": (
                    self._awaiting_post_context_management_generate
                ),
                "initial_generate_completed": self._initial_generate_completed,
                "post_context_management_generate_interrupted_reason": (
                    self._post_context_management_generate_interrupted_reason
                ),
                "terminus2_code_path": __file__,
                "main_key_recording_version": 2,
                "profiling_spans": [*self._profiling_spans],
            }
            rendered_system_prompt = self._chat.rendered_system_prompt
            if (
                isinstance(rendered_system_prompt, str)
                and rendered_system_prompt.strip()
            ):
                context.metadata["system_prompt"] = rendered_system_prompt
            pager_usage_detected = getattr(self, "_pager_usage_detected", None)
            if isinstance(pager_usage_detected, dict):
                context.metadata["pager_usage"] = pager_usage_detected
                context.metadata["forced_rollout_status"] = "pager_usage"
            if self._store_all_messages:
                all_messages = self._chat.messages_with_weight_version()
                context.metadata["all_messages"] = all_messages
                context.metadata["managed_messages_list"] = [
                    *self._managed_messages
                ]
                context.metadata["messages_list"] = [
                    *self._rollout_messages,
                    all_messages,
                ]
                context.metadata["summary_messages_list"] = [
                    *self._summary_messages
                ]
                context.metadata["summary_messages_context_management_indices"] = [
                    *self._summary_messages_context_management_indices
                ]
                context.metadata["summary_training"] = self._summary_training
                if self._enable_tito:
                    context.metadata["messages_list_tito_keys"] = [
                        *self._messages_list_tito_keys
                    ]

            # Dump trajectory to JSON
            self._dump_trajectory()

    # ===== Trajectory-related helper functions =====

    @staticmethod
    def _remove_metrics_from_copied_steps(steps: list[Step]) -> None:
        """Remove metrics from copied trajectory steps and mark as copied context.

        Args:
            steps: List of trajectory steps to modify in-place
        """
        for step in steps:
            # Mark all copied steps with is_copied_context=True
            step.is_copied_context = True

            # Remove metrics to avoid duplication
            if step.metrics:
                step.metrics = None
                if step.extra is None:
                    step.extra = {}
                step.extra["note"] = (
                    "Metrics omitted to avoid duplication - already recorded in parent trajectory"
                )

    @staticmethod
    def _attach_tool_calls_to_last_assistant_message(
        chat: Chat,
        tool_calls: list[ToolCall] | None,
    ) -> None:
        """Persist parsed tool calls into chat history for subsequent turns."""
        if not tool_calls or not chat.messages:
            return
        last_message = chat.messages[-1]
        if not isinstance(last_message, dict) or last_message.get("role") != "assistant":
            return
        last_message["tool_calls"] = [
            {
                "tool_call_id": tool_call.tool_call_id,
                "function_name": tool_call.function_name,
                "arguments": copy.deepcopy(tool_call.arguments),
            }
            for tool_call in tool_calls
        ]

    @classmethod
    def _append_tool_observation_to_chat(
        cls,
        chat: Chat,
        observation: str | None,
        tool_calls: list[ToolCall] | None,
        token_counter: Callable[[list[dict]], int] | None = None,
    ) -> bool:
        """Persist a tool result into chat history so export can use runtime messages directly."""
        if not isinstance(observation, str) or not observation.strip() or not tool_calls:
            return False

        tool_call_id: str | None = None
        for tool_call in reversed(tool_calls):
            if tool_call.function_name != "context_management":
                tool_call_id = tool_call.tool_call_id
                break
        if tool_call_id is None:
            return False

        chat.append_message(
            {
                "role": "tool",
                "content": observation,
                "content_origin": observation,
                "tool_call_id": tool_call_id,
                **(
                    {"main_key": cls._message_main_key(chat.messages[-1])}
                    if cls._message_main_key(chat.messages[-1])
                    else {}
                ),
            },
            token_counter=token_counter,
        )
        return True

    def _prepare_copied_trajectory_steps(
        self, steps_to_include: int
    ) -> tuple[list[Step], int]:
        """Prepare trajectory steps for subagent by copying and removing metrics.

        Args:
            steps_to_include: Number of steps to include from main trajectory

        Returns:
            Tuple of (copied_steps, next_step_id)
        """
        copied_steps = copy.deepcopy(self._trajectory_steps[:steps_to_include])
        self._remove_metrics_from_copied_steps(copied_steps)
        next_step_id = len(copied_steps) + 1
        return copied_steps, next_step_id

    def _append_subagent_response_step(
        self,
        steps: list[Step],
        step_id: int,
        response: LLMResponse,
        usage_info,
        subagent_name: str,
    ) -> None:
        """Append a response step with conditional metrics to trajectory steps.

        Args:
            steps: List of steps to append to
            step_id: ID for the new step
            response: LLM response
            usage_info: Usage info (may be None)
            subagent_name: Name of subagent for warning message
        """
        if usage_info:
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=self._model_name,
                    message=response.content,
                    reasoning_content=response.reasoning_content,
                    metrics=Metrics(
                        prompt_tokens=usage_info.prompt_tokens,
                        completion_tokens=usage_info.completion_tokens,
                        cached_tokens=usage_info.cache_tokens,
                        cost_usd=usage_info.cost_usd
                        if usage_info.cost_usd > 0
                        else None,
                        prompt_token_ids=response.prompt_token_ids,
                        completion_token_ids=response.completion_token_ids,
                        logprobs=response.logprobs,
                    ),
                )
            )
        else:
            self.logger.warning(f"Failed to get token usage for {subagent_name}")
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=self._model_name,
                    message=response.content,
                    reasoning_content=response.reasoning_content,
                )
            )

    def _save_subagent_trajectory(
        self,
        session_id: str,
        agent_name: str,
        steps: list[Step],
        usage_info,
        filename_suffix: str,
        summary_text: str,
    ) -> SubagentTrajectoryRef:
        """Save a subagent trajectory to disk and return its reference.

        Args:
            session_id: Session ID for the subagent
            agent_name: Name of the subagent (e.g., "terminus-2-summarization-summary")
            steps: List of trajectory steps
            usage_info: Usage information from LLM response
            filename_suffix: Suffix for trajectory filename (e.g., "summary", "questions", "answers")
            summary_text: Human-readable summary for the trajectory ref extra field

        Returns:
            SubagentTrajectoryRef for inclusion in parent trajectory
        """
        total_prompt, total_completion, total_cached, total_cost = (
            self._extract_usage_metrics(usage_info)
        )

        trajectory = Trajectory(
            session_id=session_id,
            agent=Agent(
                name=agent_name,
                version=self.version() or "unknown",
                model_name=self._model_name,
                extra={
                    "parent_session_id": self._session_id,
                    "summarization_index": self._summarization_count,
                },
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                total_cached_tokens=total_cached,
                total_cost_usd=total_cost if total_cost > 0 else None,
            ),
        )

        trajectory_path = (
            self.logs_dir
            / f"trajectory.summarization-{self._summarization_count}-{filename_suffix}.json"
        )
        try:
            with open(trajectory_path, "w") as f:
                f.write(format_trajectory_json(trajectory.to_json_dict()))
            self.logger.debug(
                f"{filename_suffix.capitalize()} subagent trajectory saved to {trajectory_path}"
            )
        except Exception as save_error:
            self.logger.error(
                f"Failed to save {filename_suffix} subagent trajectory: {save_error}"
            )

        return SubagentTrajectoryRef(
            session_id=session_id,
            trajectory_path=str(trajectory_path.name),
            extra={"summary": summary_text},
        )

    def _convert_chat_messages_to_steps(
        self,
        chat_messages: list[dict],
        additional_user_message: str | None = None,
        mark_as_copied: bool = False,
    ) -> list[Step]:
        """Convert chat messages to trajectory steps.

        Args:
            chat_messages: List of chat messages with 'role' and 'content' fields
            additional_user_message: Optional additional user message to append as final step
            mark_as_copied: If True, mark all steps with is_copied_context=True (for continuation trajectories)

        Returns:
            List of Step objects representing the chat history
        """
        steps = []
        step_id = 1

        for msg in chat_messages:
            role = msg["role"]
            content = msg["content"]

            # Map chat role to trajectory source
            if role == "assistant":
                source = "agent"
                step = Step(
                    step_id=step_id,
                    source=source,
                    message=content,
                    model_name=self._model_name,
                    reasoning_content=msg.get("reasoning_content"),
                )
            else:
                source = role
                step = Step(
                    step_id=step_id,
                    source=source,
                    message=content,
                )

            # Mark as copied context if this is for a continuation trajectory
            if mark_as_copied:
                step.is_copied_context = True

            steps.append(step)
            step_id += 1

        # Add the additional user message if provided
        # Note: The additional user message is NOT marked as copied since it's the new handoff prompt
        if additional_user_message:
            steps.append(
                Step(
                    step_id=step_id,
                    source="user",
                    message=additional_user_message,
                )
            )

        return steps

    def _split_trajectory_on_summarization(self, handoff_prompt: str) -> None:
        """Split trajectory on summarization when linear_history is enabled.

        Saves current trajectory segment and resets for continuation with full linear history.

        Args:
            handoff_prompt: The handoff prompt containing answers
        """
        # Save current trajectory segment before creating a continuation
        # When _summarization_count is 1, dump to trajectory.json (continuation_index = 0)
        # When _summarization_count is 2, dump to trajectory.cont-1.json (continuation_index = 1)
        self._dump_trajectory_with_continuation_index(self._summarization_count - 1)

        # Create new session_id for continuation
        self._session_id = (
            f"{self._session_id.split('-cont-')[0]}-cont-{self._summarization_count}"
        )

        # After dumping the trajectory till the summarization stage, reset trajectory by
        # converting from chat messages (excluding the last response which will be added
        # by the normal agent loop flow). Mark all these steps as copied context since they
        # were already present in the previous trajectory segment.
        if self._chat:
            self._trajectory_steps = self._convert_chat_messages_to_steps(
                self._chat.messages[:-1], mark_as_copied=True
            )

    def _dump_trajectory_with_continuation_index(self, continuation_index: int) -> None:
        """Dump trajectory data to JSON file with specified continuation index.

        Args:
            continuation_index: The continuation index to use for filename and metadata.
                               For the initial trajectory, use 0.
                               For the first continuation, use 1, etc.
        """
        if not self._context:
            self.logger.warning("No context available, skipping trajectory dump")
            return

        dump_started_at = datetime.now(timezone.utc)
        trajectory_path: Path | None = None

        # Construct the trajectory using Pydantic models for validation
        final_metrics = FinalMetrics(
            total_prompt_tokens=self._context.n_input_tokens,
            total_completion_tokens=self._context.n_output_tokens,
            total_cached_tokens=self._context.n_cache_tokens or 0,
            total_cost_usd=self._context.cost_usd if self._context.cost_usd else None,
        )

        agent_extra = {
            "parser": self._parser_name,
            "temperature": self._temperature,
        }
        if self._llm_kwargs:
            agent_extra["llm_kwargs"] = self._llm_kwargs
        if self._linear_history and continuation_index > 0:
            agent_extra["continuation_index"] = continuation_index

        # Determine if this trajectory will be continued
        # In linear_history mode, when saving during summarization (i.e., continuation_index < _summarization_count),
        # this trajectory will have a continuation
        continued_trajectory_ref = None
        if self._linear_history and continuation_index < self._summarization_count:
            # This trajectory segment will be continued in the next file
            next_continuation_index = continuation_index + 1
            continued_trajectory_ref = f"trajectory.cont-{next_continuation_index}.json"

        trajectory = Trajectory(
            session_id=self._session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self._model_name,
                extra=agent_extra,
            ),
            steps=self._trajectory_steps,
            final_metrics=final_metrics,
            continued_trajectory_ref=continued_trajectory_ref,
        )

        # Determine trajectory filename based on continuation index
        if self._linear_history and continuation_index > 0:
            trajectory_path = (
                self.logs_dir / f"trajectory.cont-{continuation_index}.json"
            )
        else:
            trajectory_path = self.logs_dir / "trajectory.json"

        try:
            with open(trajectory_path, "w") as f:
                json_str = format_trajectory_json(trajectory.to_json_dict())
                f.write(json_str)
            self.logger.debug(f"Trajectory dumped to {trajectory_path}")
        except Exception as e:
            self.logger.error(f"Failed to dump trajectory: {e}")
        finally:
            self._append_observation_processing_span(
                label="trajectory_dump",
                started_at=dump_started_at,
                finished_at=datetime.now(timezone.utc),
                metadata={
                    "continuation_index": continuation_index,
                    "trajectory_path": (
                        str(trajectory_path) if trajectory_path is not None else None
                    ),
                    "step_count": len(self._trajectory_steps),
                },
            )

    def _dump_trajectory(self) -> None:
        """Dump trajectory data to JSON file following ATIF format."""
        self._dump_trajectory_with_continuation_index(self._summarization_count)

    # TODO: Add asciinema logging
    def _record_asciinema_marker(self, marker_text: str) -> None:
        return
        # re
        # current_timestamp = self._session.get_asciinema_timestamp()
        # self._timestamped_markers.append((current_timestamp, marker_text))
