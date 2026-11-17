"""Parse the retained slime recipe and own the Ray driver's lifecycle."""

import json
import os
import signal


def configure_validation_limits():
    """Bound parser checks before importing any training dependencies."""
    import atexit
    import resource
    import sys

    if sys.platform != "linux":
        raise RuntimeError("Training argument validation requires Linux")
    inherited = resource.getrlimit(resource.RLIMIT_AS)
    limit = min([8 * 1024**3, *(v for v in inherited if v != resource.RLIM_INFINITY)])
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"
    ):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def report_memory():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Validation peak process RSS: {peak / 1024:.1f} MiB", flush=True)

    atexit.register(report_memory)
    print(
        f"Validation address-space hard limit: {limit / 1024**3:.1f} GiB; GPUs hidden",
        flush=True,
    )


def main():
    validation_only = os.environ.get("COMPACTIONRL_VALIDATE_ONLY") == "1"
    if validation_only:
        configure_validation_limits()
    from compactionrl.runtime.storage import prepare_storage, storage_environment

    work_dir = os.environ.get("COMPACTIONRL_WORK_DIR")
    if not work_dir:
        raise ValueError(
            "Use the project launcher to select a data-volume runtime directory"
        )
    os.environ.update(prepare_storage(work_dir, work_dir))
    cuda_architectures = os.environ.get("TORCH_CUDA_ARCH_LIST")
    from slime.utils.arguments import parse_args

    args = parse_args()
    if validation_only:
        # Validate configuration only; full trainer/Bridge imports belong to the
        # training process. A parser check must not instantiate a model provider.
        from megatron.training.arguments import core_transformer_config_from_args

        core_transformer_config_from_args(args)
        keys = [
            "hf_checkpoint",
            "load",
            "critic_load",
            "num_rollout",
            "advantage_estimator",
            "use_critic",
            "clip_type",
            "eps_clip",
            "eps_clip_high",
            "actor_num_nodes",
            "actor_num_gpus_per_node",
            "rollout_num_gpus",
            "rollout_buffer_url",
            "sglang_router_ip",
            "sglang_router_port",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "custom_advantage_function_path",
            "use_compaction_gae",
            "use_step_level_advantage",
            "stepwise_advantage_type",
        ]
        print("CompactionRL argument validation passed")
        print(json.dumps({key: getattr(args, key, None) for key in keys}, indent=2))
        return
    import ray
    from train_async import train

    # Megatron's legacy fused-kernel import clears this variable globally.
    # Restore the launcher's value before Ray starts workers without visible GPUs.
    if cuda_architectures:
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_architectures

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    # A private local cluster is the default. Remote mode uses an explicitly
    # supplied cluster and requires this checkout/assets at identical paths.
    keys = [
        "PYTHONPATH",
        "COMPACTIONRL_BATCH_TIMEOUT",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "TORCH_CUDA_ARCH_LIST",
        "LITELLM_LOCAL_MODEL_COST_MAP",
        "TOKENIZERS_PARALLELISM",
        "WANDB_MODE",
        "TENSORBOARD_DIR",
    ]
    prepare_storage(work_dir, args.save)
    keys.extend(storage_environment(work_dir))
    runtime_env = {
        "env_vars": {key: os.environ[key] for key in keys if key in os.environ}
    }
    address = os.environ.get("COMPACTIONRL_RAY_ADDRESS", "local")
    options = (
        {
            "object_store_memory": 2 * 1024**3,
            "include_dashboard": False,
            "_temp_dir": os.environ["RAY_TMPDIR"],
            "_system_config": {
                "object_spilling_config": json.dumps(
                    {
                        "type": "filesystem",
                        "params": {
                            "directory_path": os.environ["COMPACTIONRL_RAY_SPILL_DIR"]
                        },
                    }
                ),
            },
        }
        if address == "local"
        else {}
    )
    ray.init(address=address, runtime_env=runtime_env, **options)
    try:
        train(args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
