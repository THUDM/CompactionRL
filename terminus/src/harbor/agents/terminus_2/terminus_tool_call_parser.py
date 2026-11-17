from __future__ import annotations

import ast
import json
import re

from harbor.agents.terminus_2.terminus_json_plain_parser import (
    ParseResult,
    ParsedCommand,
)


class TerminusToolCallParser:
    """Parse native or inline tool calls into Terminus command batches."""

    def parse_response(
        self, response: str, tool_calls: list[dict] | None = None
    ) -> ParseResult:
        warnings: list[str] = []
        commands: list[ParsedCommand] = []
        is_task_complete = False
        raw_calls: list[tuple[str, dict]] = []

        if tool_calls:
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                name = function.get("name") or ""
                raw_arguments = function.get("arguments")
                if isinstance(raw_arguments, str):
                    try:
                        arguments = json.loads(raw_arguments) if raw_arguments else {}
                    except json.JSONDecodeError:
                        arguments = {}
                        warnings.append(
                            f"tool_call '{name}' has unparseable JSON arguments"
                        )
                elif isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    arguments = {}
                raw_calls.append((name, arguments))

        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL):
            body = match.group(1).strip()
            name_match = re.match(r"\s*([A-Za-z_][\w\-]*)", body)
            if not name_match:
                warnings.append("malformed <tool_call>: missing function name")
                continue
            name = name_match.group(1)
            raw_calls.append((name, self._parse_inline_args(body)))

        if not raw_calls:
            raw_calls.extend(self._parse_plain_function_calls(response, warnings))

        if not raw_calls:
            return ParseResult(
                commands=[],
                is_task_complete=False,
                error="No tool calls found",
                warning=self._format_warnings(warnings),
            )

        for name, arguments in raw_calls:
            if name == "bash_command":
                keystrokes = arguments.get("keystrokes", "")
                if not isinstance(keystrokes, str):
                    keystrokes = str(keystrokes)
                duration_raw = arguments.get("duration", 1.0)
                try:
                    duration = (
                        float(duration_raw) if duration_raw not in ("", None) else 1.0
                    )
                except (TypeError, ValueError):
                    duration = 1.0
                commands.append(
                    ParsedCommand(
                        keystrokes=keystrokes,
                        duration=min(max(duration, 0.0), 60.0),
                    )
                )
            elif name == "mark_task_complete":
                is_task_complete = True
            else:
                warnings.append(f"unknown tool '{name}'")

        if not commands and not is_task_complete:
            return ParseResult(
                commands=[],
                is_task_complete=False,
                error="No actionable tool calls found",
                warning=self._format_warnings(warnings),
            )

        return ParseResult(
            commands=commands,
            is_task_complete=is_task_complete,
            error="",
            warning=self._format_warnings(warnings),
        )

    @staticmethod
    def _parse_inline_args(body: str) -> dict[str, str]:
        args: dict[str, str] = {}
        keys = re.findall(r"<arg_key>(.*?)</arg_key>", body, re.DOTALL)
        values = re.findall(r"<arg_value>(.*?)</arg_value>", body, re.DOTALL)
        for key, value in zip(keys, values):
            args[key.strip()] = value
        return args

    @staticmethod
    def _parse_plain_function_calls(
        response: str, warnings: list[str]
    ) -> list[tuple[str, dict]]:
        text = response.strip()
        if not text:
            return []

        try:
            tree = ast.parse(text, mode="exec")
        except SyntaxError:
            return []

        parsed_calls: list[tuple[str, dict]] = []
        for node in tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not isinstance(call.func, ast.Name):
                continue

            name = call.func.id
            arguments: dict = {}

            try:
                if name == "bash_command":
                    if call.args:
                        arguments["keystrokes"] = ast.literal_eval(call.args[0])
                    if len(call.args) > 1:
                        arguments["duration"] = ast.literal_eval(call.args[1])
                    for kw in call.keywords:
                        if kw.arg in {"keystrokes", "duration"}:
                            arguments[kw.arg] = ast.literal_eval(kw.value)
                elif name == "mark_task_complete":
                    for kw in call.keywords:
                        if kw.arg is not None:
                            arguments[kw.arg] = ast.literal_eval(kw.value)
                else:
                    continue
            except (ValueError, SyntaxError):
                warnings.append(f"tool_call '{name}' has unparseable Python arguments")
                continue

            parsed_calls.append((name, arguments))

        return parsed_calls

    @staticmethod
    def _format_warnings(warnings: list[str]) -> str:
        if not warnings:
            return ""
        return "- " + "\n- ".join(warnings)
