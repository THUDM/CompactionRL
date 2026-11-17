import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import litellm
from litellm import CustomStreamWrapper, Message
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
)
from litellm.exceptions import (
    BadRequestError as LiteLLMBadRequestError,
)
from litellm.exceptions import (
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from litellm.utils import get_model_info
from pydantic import BaseModel
from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
    ParseError,
)
from harbor.llms.glm_generate_parser import (
    ParsedGLMGenerateOutput,
    parse_glm_generate_output,
)
from harbor.llms.utils import (
    add_anthropic_caching,
    prepare_messages_for_generate,
    split_provider_model_name,
    validate_hosted_vllm_model_config,
)
from harbor.models.metric import UsageInfo
from harbor.utils.logger import logger
import aiohttp

# This is used as a fallback for when the model does not support response_format

STRUCTURED_RESPONSE_PROMPT_TEMPLATE = """
You must respond in the following JSON format.

Here is the json schema:

```json
{schema}
```

Here is the prompt:

{prompt}
"""


def get_url_base(url: str) -> str:
    """
    Extract the base (scheme + host + optional port) from a full URL.
    """
    parsed = urlparse(url)

    # Combine scheme and netloc.
    base = f"{parsed.scheme}://{parsed.netloc}"
    return base


class LiteLLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        temperature: float = 1,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        api_base: str | None = None,
        session_id: str | None = None,
        collect_rollout_details: bool = False,
        max_thinking_tokens: int | None = None,
        reasoning_effort: str | None = None,
        model_info: dict[str, Any] | None = None,
        tokenizer_path: str | None = None,
        chat_template_path: str | None = None,
        use_generate: bool = False,
        api_key: str | None = None,
        request_timeout_sec: float | None = 600.0,
        use_responses_api: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._model_name = model_name
        self._llm_kwargs = kwargs
        self._temperature = temperature
        self._top_p = top_p
        self._max_new_tokens = max_new_tokens
        self._model_info = model_info
        self._logger = logger.getChild(__name__)
        self._logger.setLevel(0)

        hosted_vllm_validation: tuple[str, dict[str, Any]] | None = None
        if "hosted_vllm" in model_name.lower():
            hosted_vllm_validation = validate_hosted_vllm_model_config(
                model_name, self._model_info
            )

        (
            self._provider_prefix,
            self._canonical_model_name,
        ) = split_provider_model_name(model_name)

        if hosted_vllm_validation is not None:
            self._provider_prefix = "hosted_vllm"
            self._canonical_model_name, self._model_info = hosted_vllm_validation

        # Only use canonical model name for hosted_vllm; other providers like
        # openrouter need the full prefixed name for LiteLLM lookups
        self._litellm_model_name = (
            self._canonical_model_name
            if self._provider_prefix == "hosted_vllm"
            else model_name
        )

        # Register custom model if model_info is provided
        if self._model_info is not None:
            try:
                litellm.register_model({self._litellm_model_name: self._model_info})
                self._logger.debug(
                    f"Registered custom model '{model_name}' with info: {self._model_info}"
                )
            except Exception as e:
                self._logger.warning(
                    f"Failed to register custom model '{model_name}': {e}"
                )

        self._supported_params = get_supported_openai_params(self._litellm_model_name)
        self._api_base = api_base
        self._session_id = session_id
        self._collect_rollout_details = collect_rollout_details
        self._max_thinking_tokens = max_thinking_tokens
        self._reasoning_effort = reasoning_effort
        self._tokenizer_path = tokenizer_path
        self._chat_template_path = chat_template_path
        self._use_generate = use_generate
        self._api_key = api_key
        self._request_timeout_sec = request_timeout_sec
        self._parser_tokenizer = None
        self._last_rendered_system_prompt: str | None = None

        if self._use_generate:
            if not self._tokenizer_path:
                raise ValueError(
                    "use_generate requires tokenizer_path to be provided."
                )
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path,
                trust_remote_code=True,
            )
            if self._chat_template_path:
                self.tokenizer.chat_template = Path(self._chat_template_path).read_text(
                    encoding="utf-8"
                )

        if self._supported_params is not None:
            self._supports_response_format = "response_format" in self._supported_params
            self._supports_temperature = "temperature" in self._supported_params
        else:
            self._supports_response_format = False
            self._supports_temperature = False

        self._use_responses_api = use_responses_api
        self._structured_response_prompt_template = STRUCTURED_RESPONSE_PROMPT_TEMPLATE

    def _build_generate_template_kwargs(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> dict[str, Any]:
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": True,
            "clear_thinking": False,
        }
        if tools is not None:
            template_kwargs["tools"] = tools
        return template_kwargs

    def _apply_generate_chat_template(
        self,
        rendered_messages: list[dict[str, Any] | Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        template_kwargs = self._build_generate_template_kwargs(
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )
        try:
            return self.tokenizer.apply_chat_template(
                rendered_messages,
                **template_kwargs,
            )
        except TypeError as exc:
            if "clear_thinking" not in str(exc):
                raise
            template_kwargs.pop("clear_thinking", None)
            return self.tokenizer.apply_chat_template(
                rendered_messages,
                **template_kwargs,
            )

    def _get_parser_tokenizer(self):
        if self._parser_tokenizer is not None:
            return self._parser_tokenizer
        if not self._tokenizer_path:
            return None
        try:
            self._parser_tokenizer = self._load_tokenizer(self._tokenizer_path)
        except Exception as exc:
            self._logger.debug(f"Failed to load parser tokenizer: {exc}")
            self._parser_tokenizer = None
        return self._parser_tokenizer

    def _render_generate_prompt(self, messages, tools: Any = None) -> str:
        attempts = (
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
                "clear_thinking": False,
            },
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            },
            {
                "tokenize": False,
                "add_generation_prompt": True,
            },
        )

        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                if tools is None:
                    return self.tokenizer.apply_chat_template(messages, **kwargs)
                return self.tokenizer.apply_chat_template(messages, tools=tools, **kwargs)
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                break

        raise ValueError(
            "Failed to render messages with tokenizer chat_template. "
            "Provide a compatible tokenizer or set chat_template_path."
        ) from last_error

    def _render_generate_system_prompt(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        try:
            return self._apply_generate_chat_template(
                [],
                tools=tools,
                add_generation_prompt=True,
            )
        except Exception:
            return None

    @staticmethod
    def _extract_rendered_system_prompt_from_text(text: str) -> str | None:
        system_marker = "<|system|>"
        system_start = text.find(system_marker)
        if system_start < 0:
            return None

        next_markers = ("<|user|>", "<|assistant|>", "<|observation|>")
        next_positions = [
            text.find(marker, system_start + len(system_marker))
            for marker in next_markers
        ]
        next_positions = [pos for pos in next_positions if pos >= 0]
        if not next_positions:
            return None

        rendered_system_prompt = text[system_start + len(system_marker): min(next_positions)]
        return rendered_system_prompt or None

    def set_session_id(self, session_id: str | None) -> None:
        """Update session_id/main_key used for subsequent calls."""
        self._session_id = session_id

    @property
    def _lookup_model_name(self) -> str:
        """Get the model name to use for lookups in LiteLLM's model database.

        Returns the canonical model name (without provider prefix) if available,
        otherwise falls back to the original model name.
        """
        return self._litellm_model_name or self._model_name

    @property
    def _display_name(self) -> str:
        """Get a display name for logging that shows both canonical and original names if different."""
        lookup_name = self._lookup_model_name
        if lookup_name != self._model_name:
            return f"{lookup_name} (from '{self._model_name}')"
        return lookup_name

    def get_model_context_limit(self) -> int:
        """Get the context limit (max input tokens) for the current model.

        Returns:
            int: The maximum input tokens the model can accept, or a fallback value if unavailable.
        """
        fallback_context_limit = 1000000

        try:
            model_info = get_model_info(self._lookup_model_name)
            max_input_tokens = model_info.get("max_input_tokens")

            # Fallback to max_tokens if max_input_tokens not available
            if max_input_tokens is None:
                max_input_tokens = model_info.get("max_tokens")

            if max_input_tokens is not None:
                return max_input_tokens

            # Model info exists but doesn't have context limit info
            self._logger.warning(
                f"Model '{self._display_name}' info found but missing context limit fields. "
                f"Using fallback context limit: {fallback_context_limit}"
            )
        except Exception as e:
            self._logger.warning(
                f"Failed to retrieve model info for '{self._display_name}': {e}. "
                f"Using fallback context limit: {fallback_context_limit}"
            )

        return fallback_context_limit

    def get_model_output_limit(self) -> int | None:
        """Get the output limit (max output tokens) for the current model.

        Returns:
            int | None: The maximum output tokens the model can generate, or None if unavailable.
        """
        try:
            model_info = get_model_info(self._lookup_model_name)
            max_output_tokens = model_info.get("max_output_tokens")

            if max_output_tokens is None:
                # Model info exists but doesn't have max_output_tokens
                self._logger.debug(
                    f"Model '{self._display_name}' info found but missing max_output_tokens field."
                )

            return max_output_tokens
        except Exception as e:
            self._logger.debug(
                f"Failed to retrieve model info for '{self._display_name}': {e}."
            )
            return None

    def _clean_value(self, value):
        match value:
            case _ if callable(value):
                return None
            case dict():
                return {
                    k: v
                    for k, v in {
                        k: self._clean_value(v) for k, v in value.items()
                    }.items()
                    if v is not None
                }
            case list():
                return [
                    self._clean_value(v)
                    for v in value
                    if self._clean_value(v) is not None
                ]
            case str() | int() | float() | bool():
                return value
            case _:
                return str(value)

    def _init_logger_fn(self, logging_path: Path):
        def logger_fn(model_call_dict: dict):
            clean_dict = self._clean_value(model_call_dict)
            if isinstance(clean_dict, dict) and "api_key" in clean_dict:
                hash_key = hashlib.sha256(clean_dict["api_key"].encode()).hexdigest()
                clean_dict["api_key_sha256"] = hash_key
                del clean_dict["api_key"]

            if isinstance(clean_dict, dict) and "x-api-key" in clean_dict:
                hash_key = hashlib.sha256(clean_dict["x-api-key"].encode()).hexdigest()
                clean_dict["x-api-key_sha256"] = hash_key
                del clean_dict["x-api-key"]

            # Only save post_api_call (response) to preserve logprobs
            # logger_fn is called multiple times with different event types, and
            # write_text() overwrites the file each time. By filtering for
            # post_api_call only, we ensure the response with logprobs is saved.
            log_event_type = clean_dict.get("log_event_type", "unknown")
            if log_event_type == "post_api_call":
                logging_path.write_text(
                    json.dumps(
                        clean_dict,
                        indent=4,
                    )
                )

        return logger_fn

    def _write_debug_log(
        self,
        logging_path: Path | None,
        messages: list[dict[str, Any] | Message],
        query_text: str,
        payload: dict | None,
        raw_response: Any,
        response_data: Any,
        content: str,
        reasoning_content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        if logging_path is None:
            return
        logging_path.write_text(
            json.dumps(
                {
                    "messages": messages,
                    "query_text": query_text,
                    "payload": payload,
                    "raw_response": raw_response,
                    "response_data": response_data,
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls,
                },
                indent=4,
            )
        )

    @staticmethod
    def _strip_stop_tokens(text: str, stop_tokens: list[str]) -> str:
        if not text:
            return text
        trimmed = text.rstrip()
        for token in stop_tokens:
            if not token:
                continue
            while trimmed.endswith(token):
                trimmed = trimmed[: -len(token)].rstrip()
        return trimmed

    def _estimate_token_count(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception as exc:
            self._logger.debug(f"Token count estimate failed: {exc}")
            return 0

    def _parse_content_protocol_text(
        self,
        content: str,
        reasoning_content: str | None,
    ) -> tuple[str, str | None, list[dict[str, Any]] | None]:
        """Recover reasoning/tool calls when providers stuff protocol text into content."""
        if not content:
            return content, reasoning_content, None

        protocol_markers = (
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            "<invoke name=",
            "<parameter name=",
            "Executed tool calls:",
            "Executed tool_commands:",
            "Executed tool_calls:",
        )
        if not any(marker in content for marker in protocol_markers):
            return content, reasoning_content, None

        parsed = parse_glm_generate_output(
            tokenizer=self._get_parser_tokenizer(),
            raw_text=content,
        )

        parsed_reasoning = reasoning_content
        if parsed.reasoning_content:
            parsed_reasoning = parsed.reasoning_content

        parsed_tool_calls = parsed.tool_calls or None
        parsed_content = parsed.content
        if parsed_tool_calls is None:
            match = re.search(
                r"Executed tool(?: calls|_commands|_calls):\s*-\s*([A-Za-z0-9_]+)\s*:\s*(\{.*\}|\[.*\])",
                content,
                flags=re.DOTALL,
            )
            if match:
                fn_name = match.group(1)
                arguments = match.group(2).strip()
                parsed_tool_calls = [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": arguments,
                        },
                    }
                ]
                parsed_content = ""
        if not parsed_content and parsed_reasoning and not parsed.tool_calls:
            parsed_content = ""

        return parsed_content, parsed_reasoning, parsed_tool_calls

    @staticmethod
    def _fallback_generate_content(
        raw_text: str,
        parsed: ParsedGLMGenerateOutput,
    ) -> str:
        """Preserve the raw /generate payload when structured parsing yields nothing."""
        if parsed.content or parsed.reasoning_content or parsed.tool_calls:
            return parsed.content
        return (raw_text or "").strip()

    @staticmethod
    def _merge_generate_text(accumulated: str, new_text: str) -> str:
        if not accumulated:
            return new_text
        if not new_text:
            return accumulated
        if new_text.startswith(accumulated):
            return new_text

        max_overlap = min(len(accumulated), len(new_text))
        for overlap in range(max_overlap, 0, -1):
            if accumulated.endswith(new_text[:overlap]):
                return accumulated + new_text[overlap:]
        return accumulated + new_text

    @staticmethod
    def _has_dangling_generate_markup(text: str) -> bool:
        stripped = (text or "").rstrip()
        if not stripped:
            return False

        tag_pairs = (
            ("<think>", "</think>"),
            ("<tool_call>", "</tool_call>"),
            ("<arg_key>", "</arg_key>"),
            ("<arg_value>", "</arg_value>"),
            ("<arguments>", "</arguments>"),
        )
        for open_tag, close_tag in tag_pairs:
            if stripped.count(open_tag) > stripped.count(close_tag):
                return True

        partial_suffixes = (
            "<",
            "</",
            "<tool",
            "</tool",
            "<arg",
            "</arg",
            "<think",
            "</think",
            "<arguments",
            "</arguments",
        )
        return any(stripped.endswith(suffix) for suffix in partial_suffixes)

    @classmethod
    def _is_complete_generate_turn(
        cls,
        raw_text: str,
        parsed: ParsedGLMGenerateOutput,
    ) -> bool:
        stripped = (raw_text or "").strip()
        if not stripped:
            logger.warning("[is_complete] INCOMPLETE: empty raw_text")
            return False

        # XML-format Terminus2 outputs (parser_name="xml") use the structural
        # `</response>` close tag — its presence means the response is complete
        # regardless of what comes after (residual thinking text, partial new
        # tags, etc.) or the bareword markers below. Has to come BEFORE the
        # dangling-markup check, because the model sometimes keeps emitting
        # `<think>...` after `</response>` (no closing `</think>`), which
        # otherwise trips the open-vs-close tag-count balance and forces a
        # bogus continuation.
        if "</response>" in stripped:
            return True

        if cls._has_dangling_generate_markup(stripped):
            tail = stripped[-200:].replace('\n', '\\n')
            logger.warning("[is_complete] INCOMPLETE: dangling. tail=%s", tail)
            return False

        plain_tool_call_pattern = r"\b(?:bash_command|mark_task_complete)\s*\("
        if re.search(plain_tool_call_pattern, stripped):
            return True

        # Distinctive tool-call XML tags only. Bareword markers like
        # `task_complete`, `execute_commands`, `image_read`, and the
        # `Executed tool ...` strings false-positive on substring matches
        # in chain-of-thought reasoning (e.g. "I'll mark the task_complete
        # now"), forcing unnecessary continuations and eventually tripping
        # ParseError. The XML tags below are unambiguous.
        protocol_markers = (
            "<tool_call>",
            "</tool_call>",
            "<arg_key>",
            "<arg_value>",
            "<arguments>",
        )
        if any(marker in stripped for marker in protocol_markers) and not parsed.tool_calls:
            tail = stripped[-200:].replace('\n', '\\n')
            logger.warning("[is_complete] INCOMPLETE: tool-call markers no parse. tail=%s", tail)
            return False

        return True

    async def _post_generate(
        self,
        *,
        session: aiohttp.ClientSession,
        base_url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        logging_path: Path | None,
        messages: list[dict[str, Any] | Message],
        query_text: str,
        include_tool_calls_in_log: bool,
    ) -> dict[str, Any]:
        async with session.post(
            f"{base_url}/generate",
            headers=headers,
            json=payload,
        ) as response:
            if response.status != 200:
                content_text = await response.text()
                log_kwargs: dict[str, Any] = {
                    "logging_path": logging_path,
                    "messages": messages,
                    "query_text": query_text,
                    "payload": payload,
                    "raw_response": content_text,
                    "response_data": None,
                    "content": "",
                    "reasoning_content": "",
                }
                if include_tool_calls_in_log:
                    log_kwargs["tool_calls"] = None
                self._write_debug_log(**log_kwargs)
                error_message = f"ERROR RESPONSE: {response.status=} {content_text=}"
                self._logger.error(error_message)
                if self._is_context_length_error_text(content_text):
                    raise ContextLengthExceededError(error_message)
                raise ValueError(error_message)
            response.raise_for_status()
            return await response.json(content_type=None)

    async def _generate_until_complete(
        self,
        *,
        prompt_text: str,
        stop_tokens: list[str],
        max_new_tokens: int,
        logging_path: Path | None,
        messages: list[dict[str, Any] | Message],
        include_tool_calls_in_log: bool,
        session_id_override: str | None = None,
    ) -> tuple[str, ParsedGLMGenerateOutput, dict[str, Any], dict[str, Any]]:
        sampling_params: dict[str, Any] = {
            "temperature": self._temperature,
            "max_new_tokens": max_new_tokens,
            "stop": stop_tokens,
            "no_stop_trim": False,
        }
        if self._top_p is not None:
            sampling_params["top_p"] = self._top_p

        payload: dict[str, Any] = {
            "text": prompt_text,
            "sampling_params": sampling_params,
        }
        active_session_id = (
            session_id_override
            if session_id_override is not None
            else self._session_id
        )
        if active_session_id is not None:
            payload["main_key"] = active_session_id

        api_key = self._api_key or "none"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        api_base_url = self._api_base
        if not api_base_url:
            raise ValueError("api_base not set: pass api_base to LiteLLM constructor")
        base_url = get_url_base(api_base_url)

        session_kwargs: dict[str, Any] = {}
        if self._request_timeout_sec is not None and self._request_timeout_sec > 0:
            session_kwargs["timeout"] = aiohttp.ClientTimeout(
                total=self._request_timeout_sec
            )

        accumulated_raw_text = ""
        parsed = parse_glm_generate_output(tokenizer=self.tokenizer, raw_text="")
        response_data: dict[str, Any] = {}
        continuation_limit = 4
        remaining_new_tokens = max_new_tokens

        async with aiohttp.ClientSession(**session_kwargs) as session:
            for attempt in range(continuation_limit):
                if remaining_new_tokens <= 0:
                    raise OutputLengthExceededError(
                        f"Model {self._model_name} hit max_tokens limit. Response was truncated.",
                        truncated_response=accumulated_raw_text,
                    )
                payload["text"] = prompt_text + accumulated_raw_text
                sampling_params["max_new_tokens"] = remaining_new_tokens

                response_data = await self._post_generate(
                    session=session,
                    base_url=base_url,
                    headers=headers,
                    payload=payload,
                    logging_path=logging_path,
                    messages=messages,
                    query_text=payload["text"],
                    include_tool_calls_in_log=include_tool_calls_in_log,
                )

                new_raw_text = response_data.get("text", "")
                accumulated_raw_text = self._merge_generate_text(
                    accumulated_raw_text, new_raw_text
                )
                parsed = parse_glm_generate_output(
                    tokenizer=self.tokenizer,
                    raw_text=accumulated_raw_text,
                )
                if self._is_complete_generate_turn(accumulated_raw_text, parsed):
                    break

                remaining_new_tokens -= max(self._estimate_token_count(new_raw_text), 1)
                self._logger.debug(
                    "Incomplete /generate turn detected; requesting continuation. "
                    f"attempt={attempt + 1} remaining_new_tokens={remaining_new_tokens}"
                )
            else:
                raise ParseError(
                    "Failed to obtain a complete /generate reply after continuation attempts.\n"
                    f"raw_response:\n{accumulated_raw_text}"
                )

        return accumulated_raw_text, parsed, response_data, payload

    async def generate(
        self,
        messages,
        logging_path,
        session_id_override: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        include_rendered_system_prompt: bool = True,
        disable_length_limit: bool = False,
    ):
        active_session_id = (
            session_id_override if session_id_override is not None else self._session_id
        )
        rendered_messages = prepare_messages_for_generate(messages)
        text = self._render_generate_prompt(rendered_messages, tools=tools)
        rendered_system_prompt = self._extract_rendered_system_prompt_from_text(text)
        if include_rendered_system_prompt:
            self._last_rendered_system_prompt = rendered_system_prompt
        else:
            self._last_rendered_system_prompt = None
            if (
                isinstance(rendered_system_prompt, str)
                and rendered_system_prompt
                and text.startswith(rendered_system_prompt)
            ):
                text = text[len(rendered_system_prompt):]
            elif isinstance(rendered_system_prompt, str) and rendered_system_prompt:
                self._logger.warning(
                    "Failed to strip rendered system prompt from generate input."
                )
        stop_token = getattr(self.tokenizer, "eos_token", None) or "<|im_end|>"
        extra_stop_tokens = ["<|user|>", "<|assistant|>", "<|system|>", "<|observation|>"]
        stop_tokens = [stop_token, "<|im_end|>"] + extra_stop_tokens
        stop_tokens = [token for token in stop_tokens if token]
        max_new_tokens_val = (
            self._max_new_tokens if self._max_new_tokens is not None else 8192
        )

        max_context_length = self.get_model_context_limit()

        try:
            max_new_tokens_int = int(max_new_tokens_val)
        except (TypeError, ValueError):
            max_new_tokens_int = 0

        try:
            prompt_token_count = self._estimate_token_count(text)
        except Exception:
            prompt_token_count = 0

        if max_context_length and not disable_length_limit:
            if prompt_token_count is not None:
                total_tokens = prompt_token_count + max_new_tokens_int
                if total_tokens > max_context_length:
                    raise ContextLengthExceededError(
                        "Requested token count exceeds the model's maximum context length "
                        f"of {max_context_length} tokens. You requested a total of {total_tokens} tokens: "
                        f"{prompt_token_count} tokens from the input messages and {max_new_tokens_int} tokens "
                        "for the completion."
                    )

        raw_text, parsed, response_data, payload = await self._generate_until_complete(
            prompt_text=text,
            stop_tokens=stop_tokens,
            max_new_tokens=max_new_tokens_int,
            logging_path=logging_path,
            messages=rendered_messages,
            include_tool_calls_in_log=False,
            session_id_override=session_id_override,
        )
        raw_text_before_trim = raw_text
        completion_token_ids = self._extract_generate_completion_token_ids(
            response_data
        )
        if isinstance(raw_text, str) and "<think>" not in raw_text[:20]:
            raw_text = f"<think>{raw_text}"
            if isinstance(raw_text_before_trim, str):
                raw_text_before_trim = raw_text

        raw_text, completion_token_ids, terminal_stop_token = self._trim_trailing_stop_token_id(
            raw_text, completion_token_ids, stop_tokens
        )
        if terminal_stop_token is None:
            terminal_stop_token = self._detect_terminal_stop_token(
                raw_text_before_trim, stop_tokens
            )
        raw_text = self._strip_stop_tokens(raw_text, stop_tokens)
        response_data["text"] = raw_text
        self._logger.debug(f"{raw_text=}")

        training_flag = os.environ.get("TRAINING", "")
        is_training = training_flag.strip().lower() in ("1", "true", "yes", "y", "t")
        finish_reason = self._extract_generate_finish_reason(response_data)
        open_count = raw_text.count("<think>")
        close_count = raw_text.count("</think>")
        if is_training:
            valid_think_tag_counts = (
                (0, 0),  # no think tags
                (1, 1),  # one balanced think block
                (0, 1),  # only closing tag (generation starts after auto-added <think>)
            )
            if (open_count, close_count) not in valid_think_tag_counts:
                error_message = (
                    "Mismatched <think> tags in model output. "
                    f"{open_count=} {close_count=} {len(raw_text)=}\n"
                    f"raw_response:\n{raw_text}"
                )
                self._logger.error(error_message)
                raise ParseError(error_message)

        weight_version = None
        meta_info = response_data.get("meta_info")
        if isinstance(meta_info, dict):
            weight_version = meta_info.get("weight_version")
            if weight_version is None:
                weight_version = meta_info.get("weight_info")
        if weight_version is None:
            weight_version = response_data.get("weight_version")
        if weight_version is None:
            weight_version = response_data.get("weight_info")

        completion_token_ids = self._extract_generate_completion_token_ids(
            response_data
        )
        content = self._strip_stop_tokens(
            self._fallback_generate_content(raw_text, parsed),
            stop_tokens,
        )
        reasoning_content = self._strip_stop_tokens(parsed.reasoning_content, stop_tokens)
        content, completion_token_ids, _ = self._trim_trailing_stop_token_id(
            content, completion_token_ids, stop_tokens
        )

        completion_token_count: int
        if completion_token_ids is not None:
            completion_token_count = len(completion_token_ids)
        else:
            try:
                completion_token_count = len(
                    self.tokenizer.encode(raw_text, add_special_tokens=False)
                )
            except Exception as exc:
                self._logger.debug(
                    f"Completion token count estimate failed: {exc}"
                )
                completion_token_count = 0

        usage_info = UsageInfo(
            prompt_tokens=prompt_token_count or 0,
            completion_tokens=completion_token_count,
            cache_tokens=0,
            cost_usd=0.0,
        )

        self._logger.debug(f"{content=}")
        self._logger.debug(f"{reasoning_content=}")

        self._write_debug_log(
            logging_path=logging_path,
            messages=rendered_messages,
            query_text=text,
            payload=payload,
            raw_response=raw_text,
            response_data=response_data,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=parsed.tool_calls,
        )

        return LLMResponse(
            content=content,
            raw_content=raw_text,
            reasoning_content=reasoning_content,
            history_content=raw_text,
            usage=usage_info,
            completion_token_ids=completion_token_ids,
            weight_version=weight_version,
            terminal_stop_token=terminal_stop_token,
            finish_reason=finish_reason,
            main_key=active_session_id,
        )

    async def generate_with_tool_calls(
        self,
        messages: list[dict[str, Any] | Message],
        tools: Any,
        logging_path: Path | None,
    ) -> tuple[LLMResponse, list[dict[str, Any]]]:
        """
        /generate variant that embeds tool schema into the prompt and parses tool calls
        back out of the returned text (reasoning/content/tool_calls).
        """
        response = await self.generate(
            messages=messages, logging_path=logging_path, tools=tools,
        )
        parsed = parse_glm_generate_output(
            tokenizer=self.tokenizer, raw_text=response.history_content or response.content or "",
        )
        return response, parsed.tool_calls

    def _extract_weight_version(self, response, message) -> str | int | None:
        weight_version = None
        if isinstance(message, dict):
            weight_version = message.get("weight_version")
            if weight_version is None:
                weight_version = message.get("weight_info")

        if weight_version is not None:
            return weight_version

        meta_info = None
        if isinstance(response, dict):
            meta_info = response.get("meta_info")
        else:
            getter = getattr(response, "get", None)
            if callable(getter):
                meta_info = getter("meta_info")

        if isinstance(meta_info, dict):
            weight_version = meta_info.get("weight_version")
            if weight_version is None:
                weight_version = meta_info.get("weight_info")

        if weight_version is not None:
            return weight_version

        if isinstance(response, dict):
            weight_version = response.get("weight_version")
            if weight_version is None:
                weight_version = response.get("weight_info")
        else:
            getter = getattr(response, "get", None)
            if callable(getter):
                weight_version = getter("weight_version")
                if weight_version is None:
                    weight_version = getter("weight_info")

        return weight_version

    async def _sleep_before_retry(self, attempt: int) -> None:
        await asyncio.sleep(min(15, 4 * (2 ** max(0, attempt - 1))))

    @staticmethod
    def _should_retry_transient_exception(exc: Exception) -> bool:
        non_retryable = (
            ContextLengthExceededError,
            OutputLengthExceededError,
            ParseError,
            LiteLLMAuthenticationError,
            LiteLLMBadRequestError,
        )
        return not isinstance(exc, non_retryable)

    async def call(
        self,
        prompt: str | None,
        message_history: list[dict[str, Any] | Message] = [],
        response_format: dict | type[BaseModel] | None = None,
        logging_path: Path | None = None,
        session_id_override: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        if self._use_responses_api:
            return await self._call_responses(
                prompt,
                message_history,
                response_format,
                logging_path,
                session_id_override=session_id_override,
                **kwargs,
            )

        if response_format is not None and not self._supports_response_format:
            if isinstance(response_format, dict):
                schema = json.dumps(response_format, indent=2)
            else:
                schema = json.dumps(response_format.model_json_schema(), indent=2)
            prompt = self._structured_response_prompt_template.format(
                schema=schema, prompt=prompt
            )
            response_format = None

        include_rendered_system_prompt = kwargs.pop(
            "include_rendered_system_prompt",
            True,
        )
        disable_length_limit = kwargs.pop("disable_length_limit", False)

        # Prepare messages with caching for Anthropic models
        messages = list(message_history)
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})

        # self._logger.debug = print
        self._logger.debug(f"{self._use_generate=}")
        if self._use_generate:
            generate_tools = kwargs.pop("tools", None)
            # Generate mode is used with TiTO main_key tracking. Do not wrap the
            # full call in a broad retry: once /generate returns text, local
            # post-processing errors must be handled by the rollout instead of
            # sending the same prefix again under the same main_key.
            return await self.generate(
                messages,
                logging_path,
                session_id_override=session_id_override,
                tools=generate_tools,
                include_rendered_system_prompt=include_rendered_system_prompt,
                disable_length_limit=disable_length_limit,
            )

        messages = add_anthropic_caching(messages, self._model_name)

        try:
            # Build completion_kwargs with all parameters
            completion_kwargs = {
                **self._build_base_kwargs(logging_path),
                "messages": messages,
                "temperature": self._temperature,
                "response_format": response_format,
                "reasoning_effort": self._reasoning_effort,
            }
            if (
                self._request_timeout_sec is not None
                and self._request_timeout_sec > 0
            ):
                completion_kwargs["timeout"] = self._request_timeout_sec
            if self._top_p is not None:
                completion_kwargs["top_p"] = self._top_p
            if self._max_new_tokens is not None:
                completion_kwargs["max_tokens"] = self._max_new_tokens
            completion_kwargs["api_key"] = self._api_key or "none"

            # Add logprobs and return_token_ids if rollout details collection is enabled
            if self._collect_rollout_details:
                completion_kwargs["logprobs"] = True
                # Request token IDs from provider (supported by vLLM)
                # Note: Some providers (e.g., OpenAI) will reject this parameter, but we'll catch and retry without it
                if "extra_body" not in completion_kwargs:
                    completion_kwargs["extra_body"] = {}
                extra_body: dict[str, Any] = completion_kwargs["extra_body"]  # type: ignore[assignment]
                extra_body["return_token_ids"] = True

            # Add any additional kwargs, deep-merging extra_body to preserve
            # internally-set fields (e.g., return_token_ids) when caller also
            # passes extra_body via llm_call_kwargs.
            if "extra_body" in completion_kwargs and "extra_body" in kwargs:
                existing_extra_body: dict[str, Any] = completion_kwargs["extra_body"]  # type: ignore[assignment]
                new_extra_body: dict[str, Any] = kwargs.pop("extra_body")
                completion_kwargs["extra_body"] = {
                    **existing_extra_body,
                    **new_extra_body,
                }
            elif "extra_body" in kwargs:
                kwargs["extra_body"] = {**kwargs["extra_body"]}
            completion_kwargs.update(kwargs)

            # Add thinking parameter for Anthropic models if max_thinking_tokens is set
            if self._max_thinking_tokens is not None and (
                "anthropic" in self._model_name.lower()
                or "claude" in self._model_name.lower()
            ):
                budget = self._max_thinking_tokens
                if budget < 1024:
                    self._logger.warning(
                        f"max_thinking_tokens={budget} is below minimum of 1024. "
                        "Using minimum value of 1024."
                    )
                    budget = 1024
                completion_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }

            active_session_id = (
                session_id_override
                if session_id_override is not None
                else self._session_id
            )
            # Add session_id to extra_body if available
            if active_session_id is not None:
                if "extra_body" not in completion_kwargs:
                    completion_kwargs["extra_body"] = {}
                extra_body: dict[str, Any] = completion_kwargs["extra_body"]  # type: ignore[assignment]
                extra_body["session_id"] = active_session_id

            try:
                response = await litellm.acompletion(**completion_kwargs)
            except LiteLLMBadRequestError as e:
                # If provider (e.g., OpenAI) rejects extra_body parameters, retry without them
                # Some providers reject custom parameters like: return_token_ids, session_id, etc.
                error_msg = str(e)
                # Match both OpenAI's standard wording ("Unrecognized request
                # argument") and gateway-specific phrasings (e.g. GLM gateway
                # returns "Unknown parameter: 'session_id'"). Without the
                # broader matcher, the same retry path that already strips
                # `return_token_ids` simply doesn't fire on these gateways.
                gateway_rejection = (
                    "Unrecognized request argument" in error_msg
                    or "Unknown parameter" in error_msg
                    or "Unsupported parameter" in error_msg
                )
                if gateway_rejection and "extra_body" in completion_kwargs:
                    rejected_params = []
                    if "return_token_ids" in error_msg:
                        rejected_params.append("return_token_ids")
                    if "session_id" in error_msg:
                        rejected_params.append("session_id")

                    if rejected_params:
                        self._logger.warning(
                            f"Provider {self._model_name} rejected extra_body parameters: {', '.join(rejected_params)}. "
                            f"Retrying without them. Token IDs will not be available."
                        )
                        extra_body_val = completion_kwargs.get("extra_body")
                        if extra_body_val and isinstance(extra_body_val, dict):
                            extra_body: dict[str, Any] = extra_body_val
                            for param in rejected_params:
                                if param in extra_body:
                                    del extra_body[param]

                            if not extra_body:
                                del completion_kwargs["extra_body"]
                        response = await litellm.acompletion(**completion_kwargs)
                    else:
                        raise e
                else:
                    raise e
        except Exception as e:
            self._handle_litellm_error(e)

        if isinstance(response, CustomStreamWrapper):
            raise NotImplementedError("Streaming is not supported for T bench yet")

        # Extract usage information
        usage_info = self._extract_usage_info(response)

        # Extract and process token IDs if rollout details collection is enabled
        prompt_token_ids = None
        completion_token_ids = None
        logprobs = None

        if self._collect_rollout_details:
            prompt_token_ids, completion_token_ids = self._extract_token_ids(response)
            logprobs = self._extract_logprobs(response)

        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content")
        weight_version = self._extract_weight_version(response, message)

        # Sometimes the LLM returns a response with a finish reason of "length"
        # This typically means we hit the max_tokens limit, not the context window
        if choice.get("finish_reason") == "length":
            # Create exception with truncated response attached
            exc = OutputLengthExceededError(
                f"Model {self._model_name} hit max_tokens limit. "
                f"Response was truncated. Consider increasing max_tokens if possible.",
                truncated_response=content,
            )
            raise exc

        # glm sometimes only outputs reasoning_content and no content
        if not content and reasoning_content:
            content = reasoning_content
            reasoning_content = None

        # Some providers/models emit <think>...</think> directly inside `content`
        # instead of returning a dedicated `reasoning_content` field. Extract it
        # so agents can record and (optionally) interleave it into chat history.
        if reasoning_content is None and content:
            think_close_tag = "</think>"
            if think_close_tag in content:
                raw_reasoning, raw_content = content.split(think_close_tag, 1)
                extracted_reasoning = raw_reasoning.replace("<think>", "").strip()
                sanitized_content = (
                    raw_content.replace("<think>", "")
                    .replace("</think>", "")
                    .lstrip("\n")
                    .strip()
                )
                if extracted_reasoning:
                    reasoning_content = extracted_reasoning
                content = sanitized_content
            elif "<think>" in content:
                # No close tag; treat the tail as reasoning-only output.
                _, tail = content.split("<think>", 1)
                extracted_reasoning = tail.strip()
                if extracted_reasoning:
                    reasoning_content = extracted_reasoning
                content = ""

        content, reasoning_content, parsed_tool_calls = self._parse_content_protocol_text(
            content,
            reasoning_content,
        )

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            reasoning_content=reasoning_content,
            history_content=content,
            usage=usage_info,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            logprobs=logprobs,
            weight_version=weight_version,
            main_key=active_session_id,
        )

    def _extract_token_ids(self, response) -> tuple[list[int] | None, list[int] | None]:
        """Extract token IDs from a response.

        Returns:
            Tuple of (prompt_token_ids, completion_token_ids) where:
            - prompt_token_ids: Full prompt token IDs including conversation history
            - completion_token_ids: Token IDs for the generated completion as provided by the provider
        """
        try:
            # Extract completion token IDs
            completion_token_ids = None
            choices = response.get("choices", [])
            if choices:
                choice = choices[0]
                if (
                    hasattr(choice, "provider_specific_fields")
                    and choice.provider_specific_fields
                ):
                    provider_token_ids = choice.provider_specific_fields.get(
                        "token_ids"
                    )
                    if provider_token_ids and isinstance(provider_token_ids, list):
                        completion_token_ids = provider_token_ids

            # Extract full prompt token IDs (including conversation history)
            prompt_token_ids = getattr(response, "prompt_token_ids", None)

            return prompt_token_ids, completion_token_ids

        except Exception as e:
            self._logger.debug(f"Error extracting token IDs: {e}")
            return None, None

    def _extract_logprobs(self, response) -> list[float] | None:
        """Extract logprobs from a response.

        Args:
            response: The LLM response object

        Returns:
            List of log probabilities for each token, or None if not available.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return None

            logprobs_data = choices[0].get("logprobs")
            if not logprobs_data:
                return None

            content = logprobs_data.get("content", [])
            return [
                token_data["logprob"]
                for token_data in content
                if "logprob" in token_data
            ]
        except (KeyError, TypeError, IndexError):
            return None

    def _extract_generate_completion_token_ids(
        self, response_data: dict[str, Any]
    ) -> list[int] | None:
        """Extract completion token IDs from a /generate response when available."""
        candidate_paths = (
            response_data.get("token_ids"),
            response_data.get("output_token_ids"),
            response_data.get("completion_token_ids"),
            response_data.get("meta_info", {}).get("token_ids")
            if isinstance(response_data.get("meta_info"), dict)
            else None,
            response_data.get("meta_info", {}).get("output_token_ids")
            if isinstance(response_data.get("meta_info"), dict)
            else None,
            response_data.get("meta_info", {}).get("completion_token_ids")
            if isinstance(response_data.get("meta_info"), dict)
            else None,
        )
        for token_ids in candidate_paths:
            if (
                isinstance(token_ids, list)
                and token_ids
                and all(isinstance(token_id, int) for token_id in token_ids)
            ):
                return token_ids
        return None

    def _trim_trailing_stop_token_id(
        self,
        text: str | None,
        completion_token_ids: list[int] | None,
        stop_tokens: list[str],
    ) -> tuple[str | None, list[int] | None, str | None]:
        """Drop a terminal stop token from both text and token IDs when present."""
        if not completion_token_ids or self.tokenizer is None:
            return text, completion_token_ids, None

        try:
            stop_token_ids: dict[int, str] = {}
            for token in stop_tokens:
                if not token:
                    continue
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
                if len(encoded) == 1:
                    stop_token_ids[encoded[0]] = token
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if isinstance(eos_token_id, int):
                stop_token_ids.setdefault(eos_token_id, self._get_eos_token_text())
        except Exception as exc:
            self._logger.debug(f"Failed to encode stop tokens for trimming: {exc}")
            return text, completion_token_ids, None

        if not stop_token_ids or completion_token_ids[-1] not in stop_token_ids:
            return text, completion_token_ids, None

        trimmed_token_ids = completion_token_ids[:-1]
        trimmed_text = text
        trimmed_stop_token = stop_token_ids.get(completion_token_ids[-1])
        if text:
            try:
                stop_text = self.tokenizer.decode(
                    [completion_token_ids[-1]], skip_special_tokens=False
                )
            except Exception as exc:
                self._logger.debug(
                    f"Failed to decode trailing stop token for trimming: {exc}"
                )
                stop_text = None
            if stop_text and text.endswith(stop_text):
                trimmed_text = text[: -len(stop_text)].rstrip()
            else:
                trimmed_text = text.rstrip()

        return trimmed_text, trimmed_token_ids, trimmed_stop_token

    def _extract_generate_finish_reason(self, response_data: dict[str, Any]) -> str | None:
        candidate_paths = (
            response_data.get("finish_reason"),
            response_data.get("stop_reason"),
            response_data.get("meta_info", {}).get("finish_reason")
            if isinstance(response_data.get("meta_info"), dict)
            else None,
            response_data.get("meta_info", {}).get("stop_reason")
            if isinstance(response_data.get("meta_info"), dict)
            else None,
        )
        for value in candidate_paths:
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return None

    def _detect_terminal_stop_token(
        self, text: str | None, stop_tokens: list[str]
    ) -> str | None:
        if not isinstance(text, str) or not text:
            return None
        trimmed = text.rstrip()
        for token in stop_tokens:
            if token and trimmed.endswith(token):
                return token
        return None

    def _get_eos_token_text(self) -> str:
        if self.tokenizer is None:
            return ""
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if isinstance(eos_token, str):
            return eos_token
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            try:
                return self.tokenizer.decode([eos_token_id], skip_special_tokens=False)
            except Exception:
                return ""
        return ""

    def _extract_cost(self, response) -> float:
        """Extract cost from a response's _hidden_params or compute via litellm.

        Args:
            response: The LLM response object

        Returns:
            The cost in USD, or 0.0 if unavailable.
        """
        cost = 0.0
        if hasattr(response, "_hidden_params"):
            hidden_params = response._hidden_params
            if isinstance(hidden_params, dict):
                cost = hidden_params.get("response_cost", 0.0) or 0.0

        if cost == 0.0:
            try:
                cost = litellm.completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0

        return float(cost)

    def _extract_usage_info(self, response) -> UsageInfo | None:
        """Extract token usage and cost from a completion API response.

        Args:
            response: The LLM response object

        Returns:
            UsageInfo with token counts and cost, or None if not available.
        """
        try:
            # LiteLLM returns a ModelResponse object with a usage attribute
            # See: https://docs.litellm.ai/docs/completion/prompt_caching
            usage = response.usage

            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)

            # Get cache tokens from prompt_tokens_details.cached_tokens
            cache_tokens = 0
            if hasattr(usage, "prompt_tokens_details"):
                prompt_tokens_details = usage.prompt_tokens_details
                if prompt_tokens_details is not None:
                    cache_tokens = (
                        getattr(prompt_tokens_details, "cached_tokens", 0) or 0
                    )

            cost = self._extract_cost(response)

            return UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_tokens=cache_tokens,
                cost_usd=cost,
            )
        except (AttributeError, TypeError):
            return None

    def _is_context_length_error_text(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        phrases = (
            "context length exceeded",
            "context_length_exceeded",
            "maximum context length",
            "`inputs` tokens + `max_new_tokens`",
            "requested token count exceeds",
            "model's context length",
            "longer than the model's context length",
        )
        return any(phrase in lowered for phrase in phrases)

    def _extract_responses_usage_info(self, response) -> UsageInfo | None:
        """Extract token usage and cost from a responses API response.

        Args:
            response: The responses API response object

        Returns:
            UsageInfo with token counts and cost, or None if not available.
        """
        if not hasattr(response, "usage") or response.usage is None:
            return None

        usage = response.usage
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0
        cost = self._extract_cost(response)

        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_tokens=0,
            cost_usd=cost,
        )

    def _handle_litellm_error(self, e: Exception) -> None:
        """Translate litellm exceptions into harbor exceptions.

        Always re-raises; never returns normally.
        """
        if isinstance(e, LiteLLMContextWindowExceededError):
            raise ContextLengthExceededError
        if isinstance(e, LiteLLMAuthenticationError):
            raise e
        if isinstance(e, LiteLLMBadRequestError):
            if self._is_context_length_error(e):
                raise ContextLengthExceededError from e
        raise e

    def _build_base_kwargs(self, logging_path: Path | None = None) -> dict[str, Any]:
        """Build the base kwargs shared by both completion and responses API calls."""
        logger_fn = (
            self._init_logger_fn(logging_path) if logging_path is not None else None
        )
        base_kwargs: dict[str, Any] = {
            **self._llm_kwargs,
            "model": self._model_name,
            "drop_params": True,
            "logger_fn": logger_fn,
            "api_base": self._api_base,
            "api_key": self._api_key or "none",
        }
        if self._request_timeout_sec is not None and self._request_timeout_sec > 0:
            base_kwargs["timeout"] = self._request_timeout_sec
        return base_kwargs

    def _is_context_length_error(self, error: LiteLLMBadRequestError) -> bool:
        """Check provider error payloads for context-length overflow signals."""

        parts = [
            str(error),
            str(getattr(error, "body", "")),
            str(getattr(error, "message", "")),
            str(getattr(error, "error", "")),
        ]

        combined = " ".join(part for part in parts if part)
        return self._is_context_length_error_text(combined)

    async def _call_responses(
        self,
        prompt: str | None,
        message_history: list[dict[str, Any] | Message] = [],
        response_format: dict | type[BaseModel] | None = None,
        logging_path: Path | None = None,
        session_id_override: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Call the OpenAI Responses API via litellm.aresponses().

        When previous_response_id is provided (via kwargs), uses server-side
        state chaining — only the new user prompt is sent. Otherwise, builds
        the full input from message_history + prompt.
        """
        previous_response_id = kwargs.pop("previous_response_id", None)
        active_session_id = (
            session_id_override
            if session_id_override is not None
            else self._session_id
        )

        try:
            # Build responses_kwargs
            responses_kwargs: dict[str, Any] = self._build_base_kwargs(logging_path)

            if self._reasoning_effort is not None:
                responses_kwargs["reasoning"] = {
                    "effort": self._reasoning_effort,
                }
            elif self._supports_temperature:
                # Reasoning models don't support temperature, only set it
                # when reasoning_effort is not configured
                responses_kwargs["temperature"] = self._temperature

            # Get max_output_tokens from model info
            max_output_tokens = self.get_model_output_limit()
            if max_output_tokens is not None:
                responses_kwargs["max_output_tokens"] = max_output_tokens

            if response_format is not None:
                responses_kwargs["response_format"] = response_format

            if previous_response_id is not None:
                # Server-side state chaining: only send the new prompt
                responses_kwargs["previous_response_id"] = previous_response_id
                responses_kwargs["input"] = prompt or ""
            else:
                # Build full input from message history + new prompt
                input_items = []
                for msg in message_history:
                    role = (
                        msg.get("role", "user")
                        if isinstance(msg, dict)
                        else getattr(msg, "role", "user")
                    )
                    content = (
                        msg.get("content", "")
                        if isinstance(msg, dict)
                        else getattr(msg, "content", "")
                    )
                    input_items.append({"role": role, "content": content})
                if prompt is not None:
                    input_items.append({"role": "user", "content": prompt})
                responses_kwargs["input"] = input_items
            responses_kwargs.update(kwargs)

            response = await litellm.aresponses(**responses_kwargs)

        except Exception as e:
            self._handle_litellm_error(e)

        # Extract text content from response.output
        content = ""
        reasoning_content = None
        for output_item in response.output:
            if getattr(output_item, "type", None) == "message":
                for content_part in getattr(output_item, "content", []):
                    if getattr(content_part, "type", None) == "output_text":
                        content += getattr(content_part, "text", "")

        # Extract usage information
        usage_info = self._extract_responses_usage_info(response)

        # Check for truncation via response status
        response_status = getattr(response, "status", None)
        if response_status == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            reason = (
                getattr(incomplete_details, "reason", "unknown")
                if incomplete_details
                else "unknown"
            )
            if reason == "max_output_tokens":
                raise OutputLengthExceededError(
                    f"Model {self._model_name} hit max_tokens limit. "
                    f"Response was truncated.",
                    truncated_response=content,
                )

        response_id = getattr(response, "id", None)

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            history_content=content,
            usage=usage_info,
            response_id=response_id,
            main_key=active_session_id,
        )
