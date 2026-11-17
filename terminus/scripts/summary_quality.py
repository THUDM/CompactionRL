#!/usr/bin/env python3
"""Repeat summary judging and compute paired bootstrap intervals.

Input JSONL: instance_id, method, summary_id, segment_1, segment_2.
Aggregate summaries within each task/method before paired comparison. No API is
called when --scores is supplied. Requests contain the supplied trace excerpts.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import statistics

import httpx

DIMENSIONS = [
    "F1_requirement_fidelity",
    "F2_current_state_fidelity",
    "F3_failure_history_fidelity",
    "F4_next_step_validity",
]


def scores(obj):
    values = []
    for key in DIMENSIONS:
        value = obj["faithfulness"][key]["score"]
        if value in ("N/A", None):
            continue
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 <= value <= 5
        ):
            raise ValueError("Invalid judge score")
        values.append(float(value))
    value = obj["usefulness"]["U_continuation_efficiency_and_progress"]["score"]
    if value not in ("N/A", None):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 <= value <= 5
        ):
            raise ValueError("Invalid usefulness score")
        values.append(float(value))
    return values


def aggregate(rows, left, right, seed=0, repeats=10000):
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    by_summary = defaultdict(list)
    for row in rows:
        values = scores(row["judge"])
        if values:
            by_summary[(row["instance_id"], row["method"], row["summary_id"])].append(
                statistics.mean(values)
            )
    by_task = defaultdict(list)
    for (task, method, _), values in by_summary.items():
        by_task[(task, method)].append(statistics.mean(values))
    paired = []
    for task in sorted({task for task, method in by_task}):
        if (task, left) in by_task and (task, right) in by_task:
            paired.append(
                statistics.mean(by_task[(task, right)])
                - statistics.mean(by_task[(task, left)])
            )
    if not paired:
        raise ValueError("No paired tasks with usable scores")
    rng = random.Random(seed)
    boots = sorted(
        statistics.mean(rng.choices(paired, k=len(paired))) for _ in range(repeats)
    )
    return {
        "left": left,
        "right": right,
        "paired_tasks": len(paired),
        "mean_difference": statistics.mean(paired),
        "bootstrap_95_percent_interval": [
            boots[int(0.025 * repeats)],
            boots[min(repeats - 1, int(0.975 * repeats))],
        ],
        "seed": seed,
        "bootstrap_repeats": repeats,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--scores")
    p.add_argument("--output", required=True)
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--left", default="rl")
    p.add_argument("--right", default="compactionrl")
    a = p.parse_args()
    if a.repeats < 1:
        p.error("--repeats must be positive")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if a.scores:
        rows = [
            json.loads(l) for l in Path(a.scores).read_text().splitlines() if l.strip()
        ]
    else:
        if not a.input or not a.base_url or not a.model:
            p.error("Judging requires --input --base-url --model")
        template = (
            (Path(__file__).resolve().parents[1] / "prompts/summary_judge.txt")
            .read_text()
            .split("Trace.")[0]
        )
        rows = []
        with httpx.Client(
            timeout=180,
            headers={
                "Authorization": "Bearer " + os.environ.get("JUDGE_API_KEY", "none")
            },
        ) as client:
            with (out / "scores.jsonl").open("w") as file:
                for line in Path(a.input).read_text().splitlines():
                    if not line.strip():
                        continue
                    source = json.loads(line)
                    prompt = (
                        template
                        + "\nTrace.\n<segment_1>\n"
                        + source["segment_1"]
                        + "\n</segment_1>\n<segment_2>\n"
                        + source["segment_2"]
                        + "\n</segment_2>"
                    )
                    for run in range(a.repeats):
                        response = client.post(
                            a.base_url.rstrip("/") + "/chat/completions",
                            json={
                                "model": a.model,
                                "messages": [{"role": "user", "content": prompt}],
                                "reasoning_effort": "medium",
                            },
                        )
                        response.raise_for_status()
                        judge = json.loads(
                            response.json()["choices"][0]["message"]["content"]
                        )
                        scores(judge)
                        row = {
                            k: source[k]
                            for k in ["instance_id", "method", "summary_id"]
                        }
                        row.update(run=run, judge=judge, model=a.model)
                        file.write(json.dumps(row) + "\n")
                        file.flush()
                        rows.append(row)
    (out / "comparison.json").write_text(
        json.dumps(aggregate(rows, a.left, a.right), indent=2)
    )


if __name__ == "__main__":
    main()
