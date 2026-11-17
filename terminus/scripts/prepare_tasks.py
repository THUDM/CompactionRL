#!/usr/bin/env python3
"""Validate a manifest against an already prepared Harbor dataset; never copy tests into prompts."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--manifest")
p.add_argument("--output", required=True)
a = p.parse_args()
root = Path(a.dataset)
ids = (
    json.loads(Path(a.manifest).read_text())
    if a.manifest
    else sorted(d.name for d in root.iterdir() if (d / "task.toml").is_file())
)
if not ids or len(ids) != len(set(ids)):
    raise SystemExit("Empty manifest or duplicate task IDs")
missing = [
    i
    for i in ids
    if not (root / i / "task.toml").is_file()
    or not (root / i / "instruction.md").is_file()
]
if missing:
    raise SystemExit(f"Missing Harbor tasks: {missing[:20]}")
Path(a.output).write_text(json.dumps(ids, indent=2) + "\n")
print(f"Validated {len(ids)} tasks")
