#!/usr/bin/env python3
"""Launch CompactionRL from this checkout; external assets are explicit arguments."""

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "buffer/src"))

import httpx
import yaml

from compactionrl.runtime.storage import (
    prepare_storage,
    require_data_filesystem,
    storage_environment,
    validate_ray_socket_path,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local GLM-4.7-Flash HF directory")
    p.add_argument("--config", default=str(ROOT / "slime/compactionrl/configs/glm47_flash/smoke.yaml"))
    p.add_argument(
        "--dataset",
        help="Directory of prepared Harbor tasks; smoke defaults to terminus/examples/",
    )
    p.add_argument(
        "--tasks", help="JSON array of task IDs; overrides the configuration manifest"
    )
    p.add_argument(
        "--steps",
        type=int,
        help="Target total rollout updates, including resumed updates",
    )
    p.add_argument("--output", default=str(ROOT / "runs/glm47-flash"))
    p.add_argument(
        "--work-dir",
        default=os.environ.get("COMPACTIONRL_WORK_DIR"),
        help="Runtime/cache directory on a data mount; defaults to OUTPUT/runtime",
    )
    p.add_argument(
        "--resume",
        help="Run directory containing actor/ and critic/, or its actor/ directory",
    )
    p.add_argument(
        "--actor-init", help="Optional Megatron actor checkpoint for a fresh run"
    )
    p.add_argument(
        "--critic-init", help="Optional Megatron critic checkpoint for a fresh run"
    )
    p.add_argument(
        "--megatron-path",
        default=os.environ.get("MEGATRON_PATH")
        or (
            str(ROOT / ".deps/Megatron-LM")
            if (ROOT / ".deps/Megatron-LM").is_dir()
            else None
        ),
        help="Optional upstream Megatron-LM checkout; omit if installed",
    )
    p.add_argument(
        "--ray-address",
        help="Existing Ray cluster address; omitted for a private local cluster",
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="Head-node address reachable by Ray workers"
    )
    p.add_argument(
        "--bind-host",
        default="127.0.0.1",
        help="Service bind address; use 0.0.0.0 for a cluster",
    )
    p.add_argument("--harbor-port", type=int, default=18080)
    p.add_argument("--tito-port", type=int, default=31200)
    p.add_argument(
        "--tito-binary",
        default=str(ROOT / "terminus/bin/tito-linux-x86_64"),
        help="TiTO executable with the documented CLI and token protocol",
    )
    p.add_argument("--buffer-port", type=int, default=32000)
    p.add_argument("--router-port", type=int, default=31000)
    p.add_argument("--router-metrics-port", type=int, default=31001)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all service/training commands without starting them",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate assets/training arguments with an 8 GiB address-space cap; no model creation",
    )
    return p


def model_args():
    script = ROOT / "slime/scripts/models/glm4.7-30B-A3B.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"',
            "bash",
            str(script),
        ],
        check=True,
        capture_output=True,
    )
    return [x.decode() for x in result.stdout.split(b"\0") if x]


def cli_options(options):
    result = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            result.append(flag)
        elif value is not False and value is not None:
            result.append(flag)
            result.extend(str(x) for x in value) if isinstance(
                value, list
            ) else result.append(str(value))
    return result


def build_plan(args):
    config_path = Path(args.config).resolve()
    cfg = copy.deepcopy(yaml.safe_load(config_path.read_text()))
    out = Path(args.output).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else out / "runtime"
    model = str(Path(args.model).resolve())
    train = cfg["training"]
    if args.resume and (args.actor_init or args.critic_init):
        raise ValueError(
            "--resume cannot be combined with --actor-init or --critic-init"
        )
    steps = args.steps if args.steps is not None else train.get("num_rollout", 5)
    if steps < 1:
        raise ValueError("--steps must be positive")
    multi = (
        train["actor_num_nodes"] > 1
        or train["actor_num_nodes"] * train["actor_num_gpus_per_node"]
        + train["rollout_num_gpus"]
        > 8
    )
    if multi and not args.ray_address:
        raise ValueError(
            "This configuration needs an existing cluster: provide --ray-address, --host and --bind-host"
        )
    if args.ray_address and (
        args.host in {"127.0.0.1", "localhost", "0.0.0.0"}
        or args.bind_host == "127.0.0.1"
    ):
        raise ValueError(
            "Cluster workers need a reachable --host and --bind-host 0.0.0.0"
        )
    dataset = Path(args.dataset).resolve() if args.dataset else ROOT / "terminus/examples"
    task_ids = str(Path(args.tasks).resolve()) if args.tasks else cfg["task_ids"]
    if isinstance(task_ids, str):
        manifest = Path(task_ids)
        if not manifest.is_absolute():
            manifest = ROOT / manifest
        task_ids = json.loads(manifest.read_text())
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(t, str) for t in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError(
            "Task manifest must be a nonempty JSON array of unique task IDs"
        )
    if not args.dataset and set(task_ids) - {"file-smoke", "compaction-smoke"}:
        raise ValueError("Provide --dataset for non-smoke tasks")
    ports = [
        args.harbor_port,
        args.tito_port,
        args.buffer_port,
        args.router_port,
        args.router_metrics_port,
    ]
    if len(set(ports)) != 5 or any(not 1 <= port <= 65535 for port in ports):
        raise ValueError("Service ports must be distinct and valid")
    cfg.update(
        model_path=model,
        task_ids=task_ids,
        harbor_url=f"http://{args.host}:{args.harbor_port}",
        generation_url=f"http://{args.host}:{args.tito_port}",
    )
    actor_load = str(Path(args.actor_init).resolve()) if args.actor_init else model
    critic_load = str(Path(args.critic_init).resolve()) if args.critic_init else model
    if args.resume:
        resume = Path(args.resume).resolve()
        if resume.name == "actor":
            resume = resume.parent
        actor_load, critic_load = str(resume / "actor"), str(resume / "critic")
        # Continuation needs full optimizer and RNG state, unlike historical warm starts.
        for key in ("no_load_optim", "no_load_rng", "finetune"):
            train.pop(key, None)
    elif args.actor_init or args.critic_init:
        train.update(no_load_optim=True, no_load_rng=True, finetune=True)
    train.update(
        hf_checkpoint=model,
        load=actor_load,
        critic_load=critic_load,
        save=str(out / "actor"),
        critic_save=str(out / "critic"),
        num_rollout=steps,
        rollout_buffer_url=f"http://{args.host}:{args.buffer_port}",
        sglang_context_length=train["rollout_max_context_len"],
        sglang_router_ip=args.host,
        sglang_router_port=args.router_port,
        rollout_function_path="slime.rollout.compaction_buffer.generate_rollout",
        use_tensorboard=True,
        tb_project_name="compactionrl",
        tb_experiment_name=out.name,
    )
    # A zero-coefficient KL term does not require loading an extra reference model.
    if train.get("kl_loss_coef", 0) == 0 and train.get("kl_coef", 0) == 0:
        train["use_kl_loss"] = False
    cfg["training"] = train
    command = [
        sys.executable,
        str(ROOT / "slime/train_compactionrl.py"),
        *model_args(),
        *cli_options(train),
    ]
    runtime_config = out / "runtime.yaml"
    services = {
        # Supplying --sglang-router-ip tells slime to use an existing router.
        # Own that process here so fixed ports and multi-node URLs stay explicit.
        "router": [
            sys.executable,
            "-m",
            "sglang_router.launch_router",
            "--host",
            args.bind_host,
            "--port",
            str(args.router_port),
            "--prometheus-port",
            str(args.router_metrics_port),
            "--prometheus-host",
            args.bind_host,
            "--log-level",
            "warn",
            "--disable-health-check",
            "--disable-retries",
            "--request-timeout-secs",
            str(cfg.get("timeout_seconds", 600)),
            "--shutdown-grace-period-secs",
            "3",
        ],
        "harbor": [
            sys.executable,
            str(ROOT / "terminus/server/harbor_server.py"),
            "--host",
            args.bind_host,
            "--port",
            str(args.harbor_port),
        ],
        "tito": [
            str(Path(args.tito_binary).resolve()),
            "--model",
            model,
            "--router-url",
            f"http://{args.host}:{args.router_port}",
            "--host",
            args.bind_host,
            "--port",
            str(args.tito_port),
        ],
        "buffer": [
            sys.executable,
            "-m",
            "compactionrl.buffer.service",
            "--config",
            str(runtime_config),
            "--database",
            str(out / "buffer.sqlite3"),
            "--host",
            args.bind_host,
            "--port",
            str(args.buffer_port),
        ],
    }
    services["tito"].extend(
        [
            "--ttl",
            str(cfg.get("tito_ttl_seconds", 3600)),
            "--request-timeout",
            str(cfg.get("timeout_seconds", 600)),
        ]
    )
    paths = [
        str(ROOT / "buffer/src"),
        str(ROOT / "slime"),
        str(ROOT / "terminus/src"),
    ]
    if args.megatron_path:
        paths.append(str(Path(args.megatron_path).resolve()))
    env = dict(
        PYTHONPATH=os.pathsep.join(paths),
        DATASET_PATH=str(dataset),
        EVAL_OUTPUT_DIR=str(out / "trials"),
        HARBOR_AGENT_NAME="terminus-kira",
        HARBOR_MAX_WORKERS=str(
            cfg.get("harbor_max_workers", max(16, cfg.get("concurrency", 1)))
        ),
        COMPACTIONRL_BATCH_TIMEOUT=str(cfg.get("timeout_seconds", 600) + 300),
        TRAINING="1",
        MODEL_PATH=model,
        WANDB_MODE="disabled",
        TENSORBOARD_DIR=str(out / "tensorboard"),
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        COMPACTIONRL_RAY_ADDRESS=args.ray_address or "local",
        COMPACTIONRL_VALIDATE_ONLY="1" if args.validate_only else "0",
        COMPACTIONRL_TITO_PYTHON=sys.executable,
    )
    env.update(storage_environment(work_dir))
    return {
        "config": cfg,
        "services": services,
        "train": command,
        "env": env,
        "output": str(out),
        "work_dir": str(work_dir),
        "ports": ports,
    }


def validate_assets(plan, args):
    tito = Path(args.tito_binary)
    if not tito.is_file() or not os.access(tito, os.X_OK):
        raise ValueError("--tito-binary must be an executable TiTO binary")
    model = Path(plan["config"]["model_path"])
    if (
        json.loads((model / "config.json").read_text()).get("model_type")
        != "glm4_moe_lite"
    ):
        raise ValueError("Expected GLM-4.7-Flash model_type=glm4_moe_lite")
    index = model / "model.safetensors.index.json"
    shards = (
        set(json.loads(index.read_text())["weight_map"].values())
        if index.exists()
        else {"model.safetensors"}
    )
    if any(not (model / name).is_file() for name in shards):
        raise ValueError("Missing model weight shards")
    for key in ("load", "critic_load"):
        path = Path(plan["config"]["training"][key])
        if (
            str(path) != str(model)
            and not (path / "latest_checkpointed_iteration.txt").is_file()
        ):
            raise ValueError(f"{key} must be a Megatron checkpoint directory: {path}")
    dataset = Path(plan["env"]["DATASET_PATH"])
    missing = [
        task
        for task in plan["config"]["task_ids"]
        if not (dataset / task / "task.toml").is_file()
    ]
    if missing:
        raise ValueError(
            f"Missing Harbor tasks in --dataset: {missing[:5]} ({len(missing)} total)"
        )
    if (
        args.megatron_path
        and not (Path(args.megatron_path) / "megatron/training").is_dir()
    ):
        raise ValueError("--megatron-path must contain megatron/training")


def check_docker(env):
    info = subprocess.run(
        ["docker", "info", "--format", "{{.DockerRootDir}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )
    docker_root = info.stdout.strip()
    if not docker_root or not Path(docker_root).is_dir():
        raise ValueError(
            "Cannot verify Docker storage on this node; configure a local data-volume Docker daemon"
        )
    require_data_filesystem(docker_root)
    name = "compactionrl-preflight-" + uuid.uuid4().hex[:12]
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "python:3.12-slim",
                "python",
                "-c",
                "print('container ready')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        if result.returncode:
            raise RuntimeError(
                "Docker cannot start task containers: " + result.stderr[-2000:]
            )
    finally:
        # Only our uniquely named probe; never stop unrelated containers.
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            timeout=15,
            check=False,
            env=env,
        )


def launch(plan, args):
    validate_assets(plan, args)
    if not args.ray_address and not args.validate_only:
        validate_ray_socket_path(plan["work_dir"])
    prepare_storage(plan["work_dir"], plan["output"])
    env = os.environ.copy()
    env.update(plan["env"])
    if not env.get("TORCH_CUDA_ARCH_LIST"):
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        architectures = sorted(
            set(x.strip() for x in gpu_info.stdout.splitlines() if x.strip())
        )
        if not architectures:
            raise ValueError(
                "No CUDA architecture found; set TORCH_CUDA_ARCH_LIST explicitly"
            )
        env["TORCH_CUDA_ARCH_LIST"] = ";".join(architectures)
    print("CUDA JIT architectures:", env["TORCH_CUDA_ARCH_LIST"], flush=True)

    if args.validate_only:
        subprocess.run(plan["train"], env=env, cwd=ROOT, check=True, timeout=300)
        return
    out = Path(plan["output"])
    if out.exists() and any(
        (out / name).exists()
        for name in ("command.json", "buffer.sqlite3", "actor", "critic")
    ):
        raise ValueError(
            "Use a new --output directory; --resume reads checkpoints separately"
        )
    for port in plan["ports"]:
        with socket.socket() as sock:
            try:
                sock.bind((args.bind_host, port))
            except OSError as exc:
                raise ValueError(f"Port {port} is unavailable") from exc
    # Validate the bounded training parser and container runtime before GPU allocation.
    subprocess.run(
        plan["train"],
        env={**env, "COMPACTIONRL_VALIDATE_ONLY": "1"},
        cwd=ROOT,
        check=True,
        timeout=300,
    )
    check_docker(env)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime.yaml").write_text(yaml.safe_dump(plan["config"], sort_keys=False))
    (out / "command.json").write_text(json.dumps(plan["train"], indent=2))
    processes, logs = [], []

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)

    def start(name, command):
        log = (out / (name + ".log")).open("a")
        logs.append(log)
        p = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ROOT,
            start_new_session=True,
        )
        processes.append(p)
        return p

    try:
        for name, command in plan["services"].items():
            start(name, command)
        for url in (
            f"http://{args.host}:{args.router_port}/workers",
            plan["config"]["harbor_url"] + "/health",
            plan["config"]["generation_url"] + "/health",
            plan["config"]["training"]["rollout_buffer_url"] + "/health",
        ):
            for _ in range(120):
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError("Service exited; inspect output logs")
                try:
                    if httpx.get(url, timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"Service did not become ready: {url}")
        trainer = start("train", plan["train"])
        while trainer.poll() is None:
            limit_file = Path("/sys/fs/cgroup/memory.max")
            usage_file = Path("/sys/fs/cgroup/memory.current")
            if limit_file.is_file() and usage_file.is_file():
                limit = limit_file.read_text().strip()
                if limit != "max" and int(usage_file.read_text()) > int(limit) * 0.93:
                    raise RuntimeError(
                        "Container RAM usage exceeded 93% of its limit; stopped this launch "
                        "before OOM. Inspect memory.max/current and reduce offload allocations."
                    )
            time.sleep(2)
        code = trainer.returncode
        if code:
            raise SystemExit(code)
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=5)
        for log in logs:
            log.close()


def main():
    p = parser()
    args = p.parse_args()
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        launch(plan, args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        p.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
