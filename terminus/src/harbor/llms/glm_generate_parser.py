import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedGLMGenerateOutput:
    reasoning_content: str
    content: str
    tool_calls: list[dict[str, Any]]


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_MINIMAX_TOOL_OPEN = "<minimax:tool_call>"
_MINIMAX_TOOL_CLOSE = "</minimax:tool_call>"


def _normalize_with_tokenizer(tokenizer: Any, text: str) -> str:
    """
    Best-effort normalization using the HF tokenizer.

    The /generate server returns raw text. We still run text through
    encode->decode to normalize any special tokenization edge cases and to
    satisfy the "tokenizer-based parsing" requirement.
    """
    if tokenizer is None:
        return text
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return text


def _coerce_tool_value(value: str) -> Any:
    candidate = value.strip()
    if not candidate:
        return ""
    if candidate.startswith("{") or candidate.startswith("["):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return candidate
    lowered = candidate.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        if re.fullmatch(r"-?\d+", candidate):
            return int(candidate)
        if re.fullmatch(r"-?\d+\.\d+", candidate):
            return float(candidate)
    except Exception:
        pass
    return candidate


def _extract_first_json_array(text: str) -> list[Any] | None:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, list) else None
    return None


def _parse_tool_block(block_text: str) -> tuple[str, dict[str, Any]]:
    text = block_text.strip()
    if not text:
        return "", {}

    # Format 1: <tool_call>name ... <arg_key>k</arg_key><arg_value>v</arg_value> ...</tool_call>
    # name is everything up to first tag-ish marker on a line.
    name = ""
    args: dict[str, Any] = {}

    # Try to extract <arguments> JSON blob if present
    arguments_match = re.search(r"<arguments>(.*?)</arguments>", text, flags=re.DOTALL)
    if arguments_match:
        name_part = text[: arguments_match.start()].strip()
        if name_part:
            name = name_part.splitlines()[0].strip()
        raw_args = arguments_match.group(1).strip()
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                args = parsed
            else:
                args = {"_": parsed}
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        return name, args

    # Key/value pairs
    pairs = re.findall(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
        text,
        flags=re.DOTALL,
    )
    if pairs:
        # Name is first non-empty line before first <arg_key>
        before_first = text.split("<arg_key>", 1)[0].strip()
        if before_first:
            name = before_first.splitlines()[0].strip()
        for k, v in pairs:
            key = k.strip()
            if not key:
                continue
            args[key] = _coerce_tool_value(v)
        return name, args

    # Fallback: treat first line as name and no args
    name = text.splitlines()[0].strip()
    return name, {}


def _parse_minimax_tool_block(block_text: str) -> list[tuple[str, dict[str, Any]]]:
    text = block_text.strip()
    if not text:
        return []

    tool_calls: list[tuple[str, dict[str, Any]]] = []
    for invoke_match in re.finditer(
        r'<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>',
        text,
        flags=re.DOTALL,
    ):
        fn_name = invoke_match.group(1).strip()
        invoke_body = invoke_match.group(2)
        if not fn_name:
            continue

        args: dict[str, Any] = {}
        for param_match in re.finditer(
            r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>',
            invoke_body,
            flags=re.DOTALL,
        ):
            key = param_match.group(1).strip()
            if not key:
                continue
            args[key] = _coerce_tool_value(param_match.group(2))
        tool_calls.append((fn_name, args))
    return tool_calls


def _extract_malformed_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    """
    Best-effort recovery for malformed generate outputs inside a <tool_call> block.

    Common broken patterns observed from GLM include:
      - execute_commands": {"analysis":"...<arg_key>plan":"...<arg_key>commands":[...]
      - partial JSON/object-like payloads mixed with XML-ish fragments
    """
    raw = text.strip()
    if not raw:
        return "", {}

    fn_name = ""
    name_match = re.search(
        r"(execute_commands|task_complete|image_read)",
        raw,
        flags=re.IGNORECASE,
    )
    if name_match:
        fn_name = name_match.group(1)

    args: dict[str, Any] = {}

    json_args = _extract_first_json_object(raw)
    if json_args:
        args = json_args

    if fn_name == "execute_commands" and (
        not args or "commands" not in args or "analysis" not in args
    ):
        tag_analysis_match = re.search(
            r"<arg_key>\s*analysis\s*</arg_key>\s*<arg_value>(.*?)(?:</arg_value>|\"\s*,\s*\"plan\"|<arg_key>|$)",
            raw,
            flags=re.DOTALL,
        )
        tag_plan_match = re.search(
            r"<arg_key>\s*plan\s*</arg_key>\s*<arg_value>(.*?)(?:</arg_value>|\"\s*,\s*\"commands\"|<arg_key>|$)",
            raw,
            flags=re.DOTALL,
        )
        tag_commands_match = re.search(
            r"<arg_key>\s*commands\s*</arg_key>\s*<arg_value>(.*?)(?:</arg_value>|<arg_key>|$)",
            raw,
            flags=re.DOTALL,
        )
        analysis_match = re.search(
            r'analysis"?\s*:\s*"(.*?)(?:</arg_value>|<arg_key>|"\s*,\s*"plan"|$)',
            raw,
            flags=re.DOTALL,
        )
        plan_match = re.search(
            r'plan"?\s*:\s*"(.*?)(?:</arg_value>|<arg_key>|"\s*,\s*"commands"|$)',
            raw,
            flags=re.DOTALL,
        )
        commands_match = re.search(
            r'commands"?\s*:\s*(\[[\s\S]*?\])(?:</arg_value>|<arg_key>|$)',
            raw,
            flags=re.DOTALL,
        )

        analysis_value = None
        if tag_analysis_match:
            analysis_value = tag_analysis_match.group(1).strip().rstrip('",')
        elif analysis_match:
            analysis_value = analysis_match.group(1).strip()
        if analysis_value:
            args["analysis"] = analysis_value

        plan_value = None
        if tag_plan_match:
            plan_value = tag_plan_match.group(1).strip().rstrip('",')
        elif plan_match:
            plan_value = plan_match.group(1).strip()
        if plan_value:
            args["plan"] = plan_value

        commands_value: list[Any] | None = None
        if tag_commands_match:
            commands_value = _extract_first_json_array(tag_commands_match.group(1))
        if commands_value is None and commands_match:
            commands_value = _extract_first_json_array(commands_match.group(1))
        if commands_value is None:
            commands_value = _extract_first_json_array(raw)
        if commands_value is None:
            command_pairs = re.findall(
                r'"keystrokes"\s*:\s*"(.*?)"(?:\s*,\s*"duration"\s*:\s*([0-9]+(?:\.[0-9]+)?))?',
                raw,
                flags=re.DOTALL,
            )
            if command_pairs:
                commands_value = []
                for keystrokes, duration in command_pairs:
                    command: dict[str, Any] = {"keystrokes": keystrokes}
                    if duration:
                        try:
                            command["duration"] = float(duration)
                        except ValueError:
                            pass
                    commands_value.append(command)

        if commands_value is not None:
            args["commands"] = commands_value
        elif "keystrokes" in args:
            command = {"keystrokes": args.get("keystrokes", "")}
            if "duration" in args:
                command["duration"] = args["duration"]
            args["commands"] = [command]

        if "commands" not in args and (
            "keystrokes" in args or "duration" in args or "analysis" in args or "plan" in args
        ):
            args["commands"] = []

    if not fn_name:
        if "commands" in args:
            fn_name = "execute_commands"
        elif "file_path" in args or "image_read_instruction" in args:
            fn_name = "image_read"

    return fn_name, args


def _extract_malformed_minimax_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    raw = text.strip()
    if not raw:
        return "", {}

    name_match = re.search(r'<invoke\s+name="([^"]+)"', raw)
    fn_name = name_match.group(1).strip() if name_match else ""
    args: dict[str, Any] = {}

    for param_match in re.finditer(
        r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>',
        raw,
        flags=re.DOTALL,
    ):
        key = param_match.group(1).strip()
        if not key:
            continue
        args[key] = _coerce_tool_value(param_match.group(2))

    if not args:
        json_args = _extract_first_json_object(raw)
        if json_args:
            args = json_args

    if fn_name == "execute_commands" and (
        not args or "commands" not in args or "analysis" not in args
    ):
        analysis_match = re.search(
            r'<parameter\s+name="analysis">\s*(.*?)(?:",\s*"plan"|</parameter>|$)',
            raw,
            flags=re.DOTALL,
        )
        plan_match = re.search(
            r'"plan"\s*:\s*"(.*?)(?:"\s*,\s*"commands"|</parameter>|$)',
            raw,
            flags=re.DOTALL,
        )
        commands_match = re.search(
            r'"commands"\s*:\s*(\[[\s\S]*?\])(?:\s*}|</parameter>|$)',
            raw,
            flags=re.DOTALL,
        )

        if analysis_match:
            args["analysis"] = analysis_match.group(1).strip().rstrip('",')
        if plan_match:
            args["plan"] = plan_match.group(1).strip()
        if commands_match:
            commands_raw = commands_match.group(1).strip()
            try:
                parsed_commands = json.loads(commands_raw)
                if isinstance(parsed_commands, list):
                    args["commands"] = parsed_commands
            except json.JSONDecodeError:
                pass

    if not fn_name:
        if "commands" in args:
            fn_name = "execute_commands"
        elif "file_path" in args or "image_read_instruction" in args:
            fn_name = "image_read"

    return fn_name, args

def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """
    Extract and parse the first JSON object found in text.

    This is a best-effort recovery for model outputs like:
      -execute_commands: {"analysis": "...", "commands": [...]}
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def parse_glm_generate_output(
    *,
    tokenizer: Any,
    raw_text: str,
) -> ParsedGLMGenerateOutput:
    """
    Parse a GLM /generate response into (reasoning_content, content, tool_calls).

    Expected model output patterns:
    - reasoning block: <think> ... </think>
    - tool blocks: <tool_call> ... </tool_call>
    """
    normalized = _normalize_with_tokenizer(tokenizer, raw_text or "").strip()

    reasoning_content = ""
    remaining = normalized

    # /generate returns completion text only, so the opening "<think>" tag may be
    # present in the prompt (via chat_template) rather than in the returned text.
    # In that common case we only see "</think>" in `normalized`.
    if _THINK_CLOSE in normalized:
        # Split on the first closing tag to keep behavior stable if multiple blocks appear.
        before_close, after_close = normalized.split(_THINK_CLOSE, 1)
        if _THINK_OPEN in before_close:
            reasoning_content = before_close.split(_THINK_OPEN, 1)[-1].strip()
        else:
            reasoning_content = before_close.strip()
        remaining = after_close.strip()
    elif _THINK_OPEN in normalized and _THINK_CLOSE not in normalized:
        # Unterminated think block: keep reasoning only, avoid contaminating content.
        reasoning_content = normalized.split(_THINK_OPEN, 1)[-1].strip()
        remaining = ""

    tool_calls: list[dict[str, Any]] = []
    content = remaining

    # Extract tool blocks from remaining content.
    blocks: list[tuple[int, int, str, str]] = []
    tool_block_markers = (
        (_TOOL_OPEN, _TOOL_CLOSE),
        (_MINIMAX_TOOL_OPEN, _MINIMAX_TOOL_CLOSE),
    )
    for open_marker, close_marker in tool_block_markers:
        for match in re.finditer(
            re.escape(open_marker) + r"(.*?)" + re.escape(close_marker),
            remaining,
            flags=re.DOTALL,
        ):
            blocks.append((match.start(), match.end(), open_marker, match.group(1)))
    blocks.sort(key=lambda item: item[0])

    if blocks:
        # Remove blocks from content.
        parts: list[str] = []
        last_end = 0
        for start, end, open_marker, block_inner in blocks:
            parts.append(remaining[last_end:start])
            last_end = end

            if open_marker == _MINIMAX_TOOL_OPEN:
                parsed_minimax_calls = _parse_minimax_tool_block(block_inner)
                for fn_name, fn_args in parsed_minimax_calls:
                    tool_calls.append(
                        {
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": json.dumps(
                                    fn_args, ensure_ascii=False, separators=(",", ":")
                                ),
                            },
                        }
                    )
                continue

            fn_name, fn_args = _parse_tool_block(block_inner)
            if not fn_name or (fn_name and not fn_args and any(
                marker in block_inner for marker in ("analysis", "commands", "image_read")
            )):
                recovered_name, recovered_args = _extract_malformed_tool_call(block_inner)
                if recovered_name:
                    fn_name = recovered_name
                    fn_args = recovered_args
            if fn_name:
                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": json.dumps(
                                fn_args, ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    }
                )
        parts.append(remaining[last_end:])
        content = "".join(parts).strip()
    else:
        if _TOOL_OPEN in remaining:
            fn_name, args = _extract_malformed_tool_call(remaining)
            if fn_name:
                tool_calls.append(
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": json.dumps(
                                args, ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    }
                )
                content = ""
        elif "<invoke name=" in remaining:
            fn_name, args = _extract_malformed_minimax_tool_call(remaining)
            if fn_name:
                tool_calls.append(
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": json.dumps(
                                args, ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    }
                )
                content = ""
        elif re.search(
            r"Executed tool(?: calls|_commands|_calls):\s*-\s*[A-Za-z0-9_]+\s*:",
            remaining,
            flags=re.DOTALL,
        ):
            name_match = re.search(
                r"Executed tool(?: calls|_commands|_calls):\s*-\s*([A-Za-z0-9_]+)\s*:",
                remaining,
                flags=re.DOTALL,
            )
            fn_name = name_match.group(1).strip() if name_match else ""
            args = _extract_first_json_object(remaining) or {}
            if not fn_name:
                if "commands" in args:
                    fn_name = "execute_commands"
                elif "file_path" in args or "image_read_instruction" in args:
                    fn_name = "image_read"

            if fn_name:
                tool_calls.append(
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": json.dumps(
                                args, ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    }
                )
                content = ""
        # Fallback: sometimes model outputs a malformed tool block with a closing tag
        # but no opening "<tool_call>" marker, e.g.
        #   Executed tool calls:
        #   - execute_commands
        #   <arg_key>...</arg_key><arg_value>...</arg_value>
        #   </tool_call>
        # This tries to recover a single tool call from that pattern.
        elif _TOOL_CLOSE in remaining and _TOOL_OPEN not in remaining:
            close_idx = remaining.find(_TOOL_CLOSE)
            prefix = remaining[:close_idx]

            # Try to get function name from "- name" line
            fn_name: str | None = None
            name_match = re.search(
                r"^\s*-\s*([A-Za-z0-9_]+)\s*$", prefix, flags=re.MULTILINE
            )
            if name_match:
                fn_name = name_match.group(1).strip()
            if fn_name is None:
                # Try "-name:" or "- name:" patterns
                name_match = re.search(r"-\s*([A-Za-z0-9_]+)\s*:", prefix)
                if name_match:
                    fn_name = name_match.group(1).strip()

            # Parse args from arg_key/arg_value tags
            pairs = re.findall(
                r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                prefix,
                flags=re.DOTALL,
            )
            args: dict[str, Any] = {}
            for k, v in pairs:
                key = k.strip()
                if not key:
                    continue
                args[key] = _coerce_tool_value(v)

            if not args:
                json_args = _extract_first_json_object(prefix)
                if json_args:
                    args = json_args

            # If name is still missing, infer from args shape.
            if fn_name is None:
                if "commands" in args:
                    fn_name = "execute_commands"
                elif "file_path" in args or "image_read_instruction" in args:
                    fn_name = "image_read"

            if fn_name is not None:
                tool_calls.append(
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": json.dumps(
                                args, ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    }
                )
                suffix = remaining[close_idx + len(_TOOL_CLOSE) :]
                content = suffix.strip()

    return ParsedGLMGenerateOutput(
        reasoning_content=reasoning_content.strip(),
        content=content.strip(),
        tool_calls=tool_calls,
    )
