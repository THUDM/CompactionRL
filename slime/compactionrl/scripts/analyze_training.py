#!/usr/bin/env python3
"""Export scalar training curves from TensorBoard events for plotting Figure 4.

Tag names are preserved; the script does not guess absent measurements. No
external tracking account is needed. TensorBoard is an existing slime dependency.
"""

import argparse
import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser()
p.add_argument("--events", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
events = EventAccumulator(a.events, size_guidance={"scalars": 0})
events.Reload()
with Path(a.output).open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tag", "step", "wall_time", "value"])
    for tag in sorted(events.Tags().get("scalars", [])):
        if any(
            word in tag.lower()
            for word in [
                "entropy",
                "summary",
                "reasoning",
                "compaction",
                "reward",
                "loss",
                "grad",
            ]
        ):
            for point in events.Scalars(tag):
                writer.writerow([tag, point.step, point.wall_time, point.value])
