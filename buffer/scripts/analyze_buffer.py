#!/usr/bin/env python3
"""Export rollout-level compaction statistics from the persistent buffer."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

p = argparse.ArgumentParser()
p.add_argument("--database", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
groups = defaultdict(list)
with sqlite3.connect(
    "file:" + str(Path(a.database).resolve()) + "?mode=ro", uri=True
) as db:
    for (payload,) in db.execute("SELECT payload FROM samples"):
        r = json.loads(payload)
        groups[r["metadata"]["rollout_id"]].append(r)
rows = []
for uid, records in groups.items():
    records.sort(key=lambda r: r["metadata"]["segment_index"])
    rows.append(
        {
            "rollout_id": uid,
            "reward": records[0]["reward"],
            "segments": len(records),
            "compactions": records[0]["metadata"]["segment_count"] - 1,
            "optimized_tokens": sum(r["metadata"]["optimized_tokens"] for r in records),
        }
    )
result = {
    "rollouts": rows,
    "count": len(rows),
    "mean_compactions": sum(r["compactions"] for r in rows) / max(len(rows), 1),
    "compacted_task_accuracy": sum(r["reward"] == 1 for r in rows if r["compactions"])
    / max(sum(r["compactions"] > 0 for r in rows), 1),
}
Path(a.output).write_text(json.dumps(result, indent=2))
