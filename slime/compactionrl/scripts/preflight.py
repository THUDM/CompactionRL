#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--model")
p.add_argument("--mode", choices=["raw", "bridge"], default="raw")
a = p.parse_args()
errors = []
for name in [
    "torch",
    "ray",
    "sglang",
    "megatron.training",
    "mbridge" if a.mode == "raw" else "megatron.bridge",
    "transformer_engine",
    "transformers",
    "httpx",
    "fastapi",
    "yaml",
    "aiohttp",
    "dirhash",
    "tensorboard",
]:
    try:
        found = importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        found = False
    print(name, "OK" if found else "MISSING")
    if not found:
        errors.append(name)

# Container limits can be smaller than the RAM exposed in /proc/meminfo.
# Report both so CPU-offload allocations can be sized against the real runtime.
meminfo = Path("/proc/meminfo")
if meminfo.is_file():
    values = dict(line.split(":", 1) for line in meminfo.read_text().splitlines())
    for name in ("MemTotal", "MemAvailable"):
        print(name + " GiB", round(int(values[name].split()[0]) / 1024**2, 2))
    for name in ("memory.max", "memory.current"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            value = path.read_text().strip()
            print(
                "cgroup " + name + " GiB",
                value if value == "max" else round(int(value) / 1024**3, 2),
            )
if not shutil.which("docker"):
    errors.append("docker")
else:
    for command in (
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        ["docker", "compose", "version"],
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            print(
                " ".join(command[:2]),
                result.stdout.strip() if result.returncode == 0 else "FAILED",
            )
            if result.returncode:
                errors.append(" ".join(command[:2]))
        except subprocess.TimeoutExpired:
            errors.append("Docker check timed out")
if a.model:
    path = Path(a.model)
    try:
        cfg = json.loads((path / "config.json").read_text())
        if cfg.get("model_type") != "glm4_moe_lite":
            errors.append("Expected GLM-4.7-Flash model_type=glm4_moe_lite")
        index = path / "model.safetensors.index.json"
        if index.exists():
            for shard in set(json.loads(index.read_text())["weight_map"].values()):
                if not (path / shard).is_file():
                    errors.append("Missing model shard: " + shard)
        elif not (path / "model.safetensors").exists():
            errors.append("No model weights")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(str(exc))
if shutil.which("nvidia-smi"):
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv",
        ],
        check=False,
    )
else:
    errors.append("nvidia-smi")
if errors:
    raise SystemExit("Preflight failed: " + ", ".join(errors))
