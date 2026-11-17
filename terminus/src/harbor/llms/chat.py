from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from harbor.llms.base import BaseLLM, LLMResponse
from harbor.llms.utils import derive_content_origin_from_message
from harbor.models.agent.rollout_detail import RolloutDetail


def _derive_content_origin(message: dict) -> str | None:
    return derive_content_origin_from_message(message)


def _normalize_main_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


class Chat:
    def __init__(
        self,
        model: BaseLLM,
        interleaved_thinking: bool = False,
        history_token_counter: Callable[[list[dict]], int] | None = None,
        profiling_callback: (
            Callable[[str, datetime, datetime, dict | None], None] | None
        ) = None,
    ):
        self._model = model
        self._messages = []
        self._cached_history_token_count: int | None = 0
        self._history_token_counter = history_token_counter
        self._profiling_callback = profiling_callback
        self._rendered_system_prompt: str | None = None
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._cumulative_cache_tokens = 0
        self._cumulative_cost = 0.0
        self._prompt_token_ids_list: list[list[int]] = []
        self._completion_token_ids_list: list[list[int]] = []
        self._logprobs_list: list[list[float]] = []
        self._interleaved_thinking = interleaved_thinking
        self._last_response_id: str | None = None

    @property
    def total_input_tokens(self) -> int:
        return self._cumulative_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._cumulative_output_tokens

    @property
    def total_cache_tokens(self) -> int:
        return self._cumulative_cache_tokens

    @property
    def total_cost(self) -> float:
        return self._cumulative_cost

    @property
    def messages(self) -> list:
        return self._messages

    def invalidate_history_token_count(self) -> None:
        self._cached_history_token_count = None

    def set_messages(self, messages: list[dict]) -> None:
        self._messages = list(messages)
        self.invalidate_history_token_count()

    def append_message(
        self,
        message: dict,
        token_counter: Callable[[list[dict]], int] | None = None,
    ) -> None:
        self._messages.append(message)
        if token_counter is not None and self._cached_history_token_count is not None:
            started_at = datetime.now(timezone.utc)
            counted_tokens = token_counter([message])
            finished_at = datetime.now(timezone.utc)
            self._cached_history_token_count += counted_tokens
            self._emit_profiling_callback(
                "token_counting_chat_append_message",
                started_at,
                finished_at,
                {
                    "message_role": message.get("role"),
                    "message_count": 1,
                    "counted_tokens": counted_tokens,
                },
            )
        else:
            self.invalidate_history_token_count()

    def extend_messages(
        self,
        messages: list[dict],
        token_counter: Callable[[list[dict]], int] | None = None,
    ) -> None:
        if not messages:
            return
        self._messages.extend(messages)
        if token_counter is not None and self._cached_history_token_count is not None:
            started_at = datetime.now(timezone.utc)
            counted_tokens = token_counter(messages)
            finished_at = datetime.now(timezone.utc)
            self._cached_history_token_count += counted_tokens
            self._emit_profiling_callback(
                "token_counting_chat_extend_messages",
                started_at,
                finished_at,
                {
                    "message_count": len(messages),
                    "roles": [msg.get("role") for msg in messages if isinstance(msg, dict)],
                    "counted_tokens": counted_tokens,
                },
            )
        else:
            self.invalidate_history_token_count()

    def get_cached_history_token_count(
        self,
        token_counter: Callable[[list[dict]], int],
    ) -> int:
        if self._cached_history_token_count is None:
            started_at = datetime.now(timezone.utc)
            self._cached_history_token_count = token_counter(self._messages)
            finished_at = datetime.now(timezone.utc)
            self._emit_profiling_callback(
                "token_counting_history_recount",
                started_at,
                finished_at,
                {
                    "message_count": len(self._messages),
                    "counted_tokens": self._cached_history_token_count,
                },
            )
        return self._cached_history_token_count

    def _emit_profiling_callback(
        self,
        label: str,
        started_at: datetime,
        finished_at: datetime,
        metadata: dict | None = None,
    ) -> None:
        if self._profiling_callback is None:
            return
        self._profiling_callback(label, started_at, finished_at, metadata)

    @property
    def rendered_system_prompt(self) -> str | None:
        return self._rendered_system_prompt

    def set_rendered_system_prompt(self, system_prompt: str | None) -> None:
        if self._rendered_system_prompt is None and system_prompt is not None:
            self._rendered_system_prompt = system_prompt

    def _resolve_response_main_key(self, call_kwargs: dict) -> str | None:
        return _normalize_main_key(
            call_kwargs.get("session_id_override")
            if call_kwargs.get("session_id_override") is not None
            else getattr(self._model, "_session_id", None)
        )

    def _resolve_recorded_response_main_key(
        self,
        llm_response: LLMResponse,
        call_kwargs: dict,
    ) -> str | None:
        response_main_key = _normalize_main_key(llm_response.main_key)
        if response_main_key is not None:
            return response_main_key
        return self._resolve_response_main_key(call_kwargs)

    def messages_with_weight_version(self) -> list[dict]:
        """Return a copy of messages annotated with weight_version fields."""
        annotated = []
        if (
            isinstance(self._rendered_system_prompt, str)
            and self._rendered_system_prompt.strip()
        ):
            annotated.append(
                {
                    "role": "system",
                    "content": self._rendered_system_prompt,
                    "content_origin": self._rendered_system_prompt,
                    "num_tokens": 0,
                }
            )
        for msg in self._messages:
            msg_copy = dict(msg)
            msg_copy["content_origin"] = _derive_content_origin(msg_copy)
            msg_copy.pop("history_content", None)
            msg_copy["num_tokens"] = int(msg_copy.get("num_tokens") or 0)
            if msg_copy.get("role") == "assistant":
                msg_copy.setdefault("weight_version", None)
            else:
                msg_copy.pop("weight_version", None)
            annotated.append(msg_copy)
        return annotated

    def messages_with_rendered_system_prompt(self) -> list[dict]:
        """Return model history with the captured rendered system prompt restored.

        Runtime chat history intentionally stores user/assistant/tool turns without
        the tokenizer-rendered system/tools block. TiTO prefix replay needs that
        block when a side call, such as summary generation, reuses the main chat
        main_key.
        """
        messages = [dict(msg) for msg in self._messages]
        if messages and messages[0].get("role") == "system":
            return messages
        if (
            not isinstance(self._rendered_system_prompt, str)
            or not self._rendered_system_prompt.strip()
        ):
            return messages
        return [
            {
                "role": "system",
                "content": self._rendered_system_prompt,
                "content_origin": self._rendered_system_prompt,
                "num_tokens": 0,
            },
            *messages,
        ]

    @property
    def rollout_details(self) -> list[RolloutDetail]:
        """Get detailed rollout information from all chat interactions.

        Returns:
            List containing a single RolloutDetail representing the complete
            linear chat history, or an empty list if no rollout data was collected.

            TODO: consider multiple rollout details for non-linear chat histories, e.g.
            subagents, summarization, etc.
        """
        if (
            not self._prompt_token_ids_list
            and not self._completion_token_ids_list
            and not self._logprobs_list
        ):
            return []

        rollout_detail: RolloutDetail = {}

        if self._prompt_token_ids_list:
            rollout_detail["prompt_token_ids"] = self._prompt_token_ids_list

        if self._completion_token_ids_list:
            rollout_detail["completion_token_ids"] = self._completion_token_ids_list

        if self._logprobs_list:
            rollout_detail["logprobs"] = self._logprobs_list

        return [rollout_detail]

    async def chat(
        self,
        prompt: str | None,
        logging_path: Path | None = None,
        **kwargs,
    ) -> LLMResponse:
        llm_response: LLMResponse = await self._model.call(
            prompt=prompt,
            message_history=self._messages,
            logging_path=logging_path,
            previous_response_id=self._last_response_id,
            **kwargs,
        )
        rendered_system_prompt = getattr(
            self._model,
            "_last_rendered_system_prompt",
            None,
        )
        if isinstance(rendered_system_prompt, str) and rendered_system_prompt.strip():
            self.set_rendered_system_prompt(rendered_system_prompt)

        # Track response chain for Responses API
        if llm_response.response_id is not None:
            self._last_response_id = llm_response.response_id

        # Get token usage and cost from the LLM response
        usage = llm_response.usage
        if usage is not None:
            self._cumulative_input_tokens += usage.prompt_tokens
            self._cumulative_output_tokens += usage.completion_tokens
            self._cumulative_cache_tokens += usage.cache_tokens
            self._cumulative_cost += usage.cost_usd

        # Accumulate rollout details from the response
        self._accumulate_rollout_details(llm_response)

        # Build assistant message with optional reasoning content
        assistant_message = {"role": "assistant", "content": llm_response.content}
        main_key = self._resolve_recorded_response_main_key(llm_response, kwargs)
        if main_key is not None:
            assistant_message["main_key"] = main_key
        if llm_response.history_content is not None:
            assistant_message["history_content"] = llm_response.history_content
            assistant_message["content_origin"] = llm_response.history_content
        else:
            assistant_message["content_origin"] = llm_response.content
        assistant_message["num_tokens"] = (
            llm_response.usage.completion_tokens if llm_response.usage is not None else 0
        )
        if self._interleaved_thinking and llm_response.reasoning_content:
            assistant_message["reasoning_content"] = llm_response.reasoning_content
        assistant_message["weight_version"] = llm_response.weight_version
        new_messages = [assistant_message]
        if prompt is not None:
            user_message = {
                "role": "user",
                "content": prompt,
                "content_origin": prompt,
                "num_tokens": 0,
            }
            if main_key is not None:
                user_message["main_key"] = main_key
            new_messages.insert(0, user_message)

        self.extend_messages(
            new_messages,
            token_counter=self._history_token_counter,
        )
        return llm_response

    def reset_response_chain(self) -> None:
        """Reset the response chain so the next call sends full message history.

        Call this whenever chat._messages is directly modified (e.g., after
        summarization or unwinding) to ensure the next Responses API call
        doesn't use a stale previous_response_id.
        """
        self._last_response_id = None

    def _accumulate_rollout_details(self, llm_response: LLMResponse) -> None:
        """Accumulate rollout details from an LLM response.

        Args:
            llm_response: The LLM response containing token IDs and logprobs
        """
        # Accumulate prompt token IDs per turn
        if llm_response.prompt_token_ids:
            self._prompt_token_ids_list.append(llm_response.prompt_token_ids)

        # Accumulate completion token IDs per turn
        if llm_response.completion_token_ids:
            # Store completion token IDs for this turn
            self._completion_token_ids_list.append(llm_response.completion_token_ids)

            # Store logprobs for this turn (if available)
            if llm_response.logprobs:
                self._logprobs_list.append(llm_response.logprobs)
