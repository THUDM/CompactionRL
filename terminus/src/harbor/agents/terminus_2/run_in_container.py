"""In-container entrypoint for terminus-2 when shipped as an installed agent.

This script is invoked by the host-side `installed/terminus_2.py` wrapper
once per trial, *inside* the task container. It constructs a real
`Terminus2` instance, gives it a `LocalEnvironment` that talks to the
container directly (no docker daemon hop), runs the agent loop to
completion, and serialises the resulting `AgentContext` for the host to
pick up via `populate_context_post_run`.

Inputs (CLI flags):
    --config PATH        JSON file with kwargs for `Terminus2.__init__`.
                         Strings, ints, floats, bools, lists, and nested
                         dicts (e.g. `model_info`) are passed through.
    --instruction PATH   File containing the natural-language instruction
                         the agent should solve.
    --logs-dir PATH      Where the agent should write trajectory.json,
                         episode logs, and the asciinema recording.
                         Defaults to /logs/agent (the bind-mounted dir
                         that the harbor host reads from).
    --context-out PATH   Where to write the post-run AgentContext as JSON.
                         Defaults to <logs-dir>/context.json.

The trajectory file (`trajectory.json`) is written by `Terminus2._dump_trajectory`
into `--logs-dir`; the host reads it via the mount, so it appears under
`trial_dir/agent/` automatically.

This module imports the full `harbor.agents.terminus_2.Terminus2` so the
container must have harbor + its runtime dependencies installed (the
install-terminus-2.sh.j2 template handles that bootstrap).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from harbor.agents.terminus_2.local_environment import LocalEnvironment
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.models.agent.context import AgentContext


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="harbor-terminus-2",
        description="In-container entrypoint for the terminus-2 agent.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--instruction", required=True, type=Path)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("/logs/agent"),
        help="Directory the agent writes trajectory.json/recording/etc. to. "
        "Defaults to /logs/agent (the harbor bind-mount).",
    )
    parser.add_argument(
        "--context-out",
        type=Path,
        default=None,
        help="Where to dump the post-run AgentContext as JSON. "
        "Defaults to <logs-dir>/context.json.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="installed-terminus-2",
        help="session_id used in error messages and tmux tagging.",
    )
    return parser.parse_args(argv)


def _build_environment(session_id: str, logs_dir: Path) -> LocalEnvironment:
    env = LocalEnvironment(session_id=session_id)
    # `Terminus2.setup` reads `environment.trial_paths.agent_dir` to derive the
    # asciinema recording path. Inside the container both the "host" and
    # "remote" agent dirs are the same bind-mounted directory, so we just
    # mirror logs_dir on a SimpleNamespace stand-in.
    env.trial_paths = SimpleNamespace(agent_dir=logs_dir)  # type: ignore[attr-defined]
    return env


async def _run(args: argparse.Namespace) -> int:
    config = json.loads(args.config.read_text())
    instruction = args.instruction.read_text()

    logs_dir: Path = args.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    context_out: Path = args.context_out or (logs_dir / "context.json")

    # `logs_dir` and `session_id` are constructor-positional; allow the JSON
    # to override them but provide sensible defaults from CLI args.
    config.setdefault("logs_dir", str(logs_dir))
    config.setdefault("session_id", args.session_id)
    # Convert known Path-shaped strings — Terminus2 takes plain strings for
    # most of these, so no conversion needed beyond logs_dir.
    config["logs_dir"] = Path(config["logs_dir"])

    agent = Terminus2(**config)
    environment = _build_environment(
        session_id=str(config["session_id"]), logs_dir=logs_dir
    )
    context = AgentContext()

    await agent.setup(environment)
    try:
        await agent.run(instruction, environment, context)
    finally:
        # Always write the context, even if `run` raised — the host wrapper
        # uses this to surface partial token counts on failure.
        context_out.write_text(context.model_dump_json(indent=2))

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return asyncio.run(_run(args))
    except SystemExit:
        raise
    except BaseException as exc:
        # Surface the full traceback to the host log; the wrapper reads
        # /logs/agent/exception.txt to enrich the trial result.
        err_path = args.logs_dir / "exception.txt"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
