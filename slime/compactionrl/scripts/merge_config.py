#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml


def merge(a, b):
    result = dict(a)
    for key, value in b.items():
        result[key] = (
            merge(result.get(key, {}), value) if isinstance(value, dict) else value
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("overlay")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = merge(
        yaml.safe_load(Path(a.base).read_text()),
        yaml.safe_load(Path(a.overlay).read_text()),
    )
    Path(a.output).write_text(yaml.safe_dump(cfg, sort_keys=False))
