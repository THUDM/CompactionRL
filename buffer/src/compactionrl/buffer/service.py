"""One-rollout-per-prompt collector; no GRPO group barrier on the PPO path."""

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException

from .schema import finalize_rollout
from .store import Buffer
from .collector import collect, assign_group_advantages


def create_app(config, store):
    workers = []
    state = {"started": False, "completed": 0, "failed": 0, "last_error": None}
    task_ids = config["task_ids"]
    if isinstance(task_ids, str):
        task_ids = json.loads(Path(task_ids).read_text())
    if not task_ids:
        raise ValueError("task_ids must not be empty")
    counter = 0

    async def worker(args):
        nonlocal counter
        timeout = float(config.get("timeout_seconds", 1200))
        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                stats = store.stats()
                if stats.get("ready", 0) + stats.get("leased", 0) >= config.get(
                    "prefetch_segments", 16
                ):
                    await asyncio.sleep(0.5)
                    continue
                task_id = task_ids[counter % len(task_ids)]
                counter += 1
                try:
                    model = args.get("hf_checkpoint", config.get("model_path"))
                    size = int(config.get("group_size", 1))
                    if size < 1:
                        raise ValueError("group_size must be positive")
                    rollouts = await asyncio.gather(
                        *(collect(client, config, model, task_id) for _ in range(size))
                    )
                    records = (
                        rollouts[0] if size == 1 else assign_group_advantages(rollouts)
                    )
                    if len(records) > store.capacity:
                        raise ValueError(
                            "One rollout group exceeds the buffer capacity"
                        )
                    while True:
                        try:
                            store.put(records)
                            break
                        except OverflowError:
                            await asyncio.sleep(0.5)
                    state["completed"] += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    state["failed"] += 1
                    state["last_error"] = f"{type(exc).__name__}: {exc}"
                    await asyncio.sleep(1)

    @asynccontextmanager
    async def lifespan(app):
        yield
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    @app.get("/buffer/stats")
    async def health():
        return {**state, "queue": store.stats()}

    @app.post("/start_rollout")
    async def start(args: dict):
        if not state["started"]:
            state["started"] = True
            workers.extend(
                asyncio.create_task(worker(args))
                for _ in range(config.get("concurrency", 1))
            )
        return {"success": True}

    @app.post("/buffer/write")
    async def write(payload: dict):
        try:
            count = store.put(
                finalize_rollout(payload["segments"], payload["rollout_id"])
            )
            return {"success": True, "inserted": count}
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/get_rollout_data")
    async def read(payload: dict):
        try:
            lease, data = store.take(int(payload.get("batch_size", 1)))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"success": bool(data), "lease": lease, "data": data, "collector": state}

    @app.post("/buffer/ack")
    async def ack(payload: dict):
        return {"success": True, "acknowledged": store.ack(payload["lease"])}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--database", default="runs/buffer.sqlite3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=32000)
    args = parser.parse_args()
    config = yaml.safe_load(os.path.expandvars(Path(args.config).read_text()))
    uvicorn.run(
        create_app(config, Buffer(args.database, config.get("capacity", 1024))),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
