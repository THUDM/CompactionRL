#!/usr/bin/env python3
"""Evaluate a fixed local Harbor task set through /run_worker; retain each run."""

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import uuid

import httpx
import yaml


async def evaluate(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    ids = json.loads(Path(args.tasks).read_text())
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation task IDs")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for run in range(args.runs):
            semaphore = asyncio.Semaphore(args.concurrency)

            async def one(task):
                async with semaphore:
                    req = dict(
                        cfg["agent"],
                        instance_id=task,
                        uid=str(uuid.uuid4()),
                        model=args.model,
                        tokenizer_path=args.model,
                        base_url=cfg["generation_url"],
                        use_generate=True,
                        enable_tito=True,
                        timeout=args.timeout / 60,
                    )
                    result = None
                    try:
                        async with client.stream(
                            "POST", cfg["harbor_url"] + "/run_worker", json=req
                        ) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if line.strip():
                                    event = json.loads(line)
                                    if event.get("type") == "result":
                                        result = event["data"]
                                    elif event.get("type") == "error":
                                        raise RuntimeError(str(event["data"]))
                        if result is None:
                            raise ValueError("No final result")
                        reward = result.get("reward", result.get("raw_reward"))
                        if reward is None or not 0 <= float(reward) <= 1:
                            raise ValueError("Infrastructure failure / invalid reward")
                        return {
                            "instance_id": task,
                            "reward": float(reward),
                            "result": result,
                        }
                    except Exception as exc:
                        return {"instance_id": task, "reward": None, "error": str(exc)}

            rows = await asyncio.gather(*(one(task) for task in ids))
            (out / f"run_{run + 1}.json").write_text(json.dumps(rows, indent=2))
            failures = sum(r["reward"] is None for r in rows)
            score = 100 * sum(r["reward"] == 1 for r in rows) / len(ids)
            scores.append(score)
            print(
                f"Run {run + 1}: pass@1={score:.2f}%, infrastructure failures={failures}/{len(ids)}"
            )
        summary = {
            "tasks": len(ids),
            "runs": len(scores),
            "pass_at_1_percent": scores,
            "mean": statistics.mean(scores),
            "std_population": statistics.pstdev(scores),
            "std_sample": statistics.stdev(scores) if len(scores) > 1 else None,
            "failure_policy": "Infrastructure failures count in denominator and are separately recorded; inspect before paper comparison.",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1800)
    asyncio.run(evaluate(p.parse_args()))
