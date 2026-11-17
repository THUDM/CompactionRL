# GLM-4.7-Flash launch guide

The launcher starts the SGLang router, Harbor, TiTO and buffer, followed by the
slime training driver. Model weights and prepared Harbor task environments are
external assets. Install dependencies using the [environment guide](environment.md).

## Small run

The smoke profile targets one 8 × 80 GB GPU node: four GPUs shared by actor/critic
and four for rollout. It uses two bundled tasks, 8k context and five update batches.
The compaction task uses a 4,096-token trigger and zero retained interaction pairs
so a summary fits inside the small budget. Paper configurations retain two pairs.

```bash
export COMPACTIONRL_WORK_DIR=/data/cr
source scripts/storage_env.sh
export MODEL_PATH=/absolute/path/to/GLM-4.7-Flash
bash slime/compactionrl/scripts/install_megatron.sh
# The launcher finds .deps/Megatron-LM automatically.
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --dry-run
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --validate-only
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" \
  --steps 5 --output "$PWD/runs/flash-smoke"
```

`--dry-run` prints commands and resolved configuration. `--validate-only` checks
model/task assets and training arguments without starting services or allocating
training GPUs. The parser process hides GPUs and applies an 8 GiB address-space
limit, uses single-threaded library settings and reports peak process RSS.

Normal launch repeats the asset and argument checks, runs a Docker probe, then
starts the services and training driver. For the single-node smoke profile, it
also starts a private local Ray cluster. A failed Docker probe stops the launch
before model loading.

The smoke memory settings include TP/EP/ETP = 4/4/1, sequence parallelism,
BF16 gradient reduction, 70% optimizer CPU offload and full activation
recomputation. The smoke profile disables the additional policy weight backup,
since it uses no reference or old-policy model. Training allocations use CPU
backing with memory-saver disk backup disabled. Use the paper configuration for
the full training settings.

## Flash paper training

`scripts/train_glm47_flash.sh` defaults to
[`glm47_flash/paper.yaml`](../configs/glm47_flash/paper.yaml). It specifies
64k context, PPO, 50 critic-only batches and 180 subsequent PPO batches. Set the
actor/critic and rollout GPU allocations and parallelism to match your cluster.

Prepare an explicit JSON array of the paper training task IDs and the corresponding
Harbor task directories/images. The launcher currently does not expand the
`${TRAIN_TASK_MANIFEST}` placeholder in the YAML, so pass `--tasks` explicitly.

Install the checkout and dependencies on a pre-existing Ray cluster, with model,
repository, task and output paths accessible at the same absolute paths. Set
`HEAD_IP` to the head-node address reachable by workers. On each node, configure
Ray's temporary and spill directories on the data volume before starting Ray.
`--ray-address` is the GCS address (normally port 6379), not the dashboard URL.
Source `scripts/storage_env.sh` on every node, and place `ray start --temp-dir`
and `--object-spilling-directory` on its data volume. Check each node's Docker
data root as well. The launcher connects to the cluster without managing SSH
access or stopping existing Ray processes.

```bash
bash scripts/train_glm47_flash.sh \
  --config slime/compactionrl/configs/glm47_flash/paper.yaml \
  --model "$MODEL_PATH" \
  --tasks /absolute/path/to/paper-training-task-ids.json \
  --dataset /absolute/path/to/prepared-swedev-tasks \
  --ray-address "$HEAD_IP:6379" --host "$HEAD_IP" --bind-host 0.0.0.0 \
  --steps 230 --output "$PWD/runs/flash-paper" --dry-run
```

Replace `--dry-run` with `--validate-only` for local asset/argument checks, then
remove that flag to start services and training. The current implementation
counts the critic-only batches in `--steps`: 230 total batches leave 180 PPO
batches after warmup. See [training semantics](reproduction.md#training-semantics).

The [Air paper configuration](../configs/glm45_air/paper.yaml) specifies 80k context
and 150 total batches. The Flash launcher loads a Flash-specific model definition
and validates `model_type=glm4_moe_lite`. For Air training, use the Air-SFT starting
checkpoint and integrate the Air model definition and parallelism with the
training driver; the Flash wrapper's model definition is specific to Flash.

## Checkpoints and outputs

Each invocation requires a new output directory. It contains `runtime.yaml`,
`command.json`, service logs, TensorBoard events, the buffer database, task traces
and actor/critic checkpoints. Override ports through launcher arguments; all
services and the trainer receive the same resolved endpoints.
The service flags are `--harbor-port`, `--tito-port`, `--buffer-port` and
`--router-port`; `--router-metrics-port` controls the separate metrics listener.

To continue a successfully saved smoke run for one more batch:

```bash
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" \
  --resume "$PWD/runs/flash-smoke" --steps 6 \
  --output "$PWD/runs/flash-resume"
```

`--resume` accepts the previous run directory or its `actor/` subdirectory and
uses the sibling critic checkpoint. `--steps` remains the total target, including
batches completed before resuming. Use the same configuration for continuation;
for paper training, supply the paper config and task/cluster arguments again.

For a fresh run initialized from Megatron weights without optimizer/RNG
continuation, use `--actor-init` and/or `--critic-init`. Otherwise actor and critic
are initialized from the HF model through mbridge; the critic has a scalar value
head. Raw GLM conversion exports updated weights to SGLang.

## Deployment requirements

- Use Linux x86_64 and Python 3.12. TiTO uses the current Python environment via
  `COMPACTIONRL_TITO_PYTHON`; see its [runtime requirements](../../../terminus/bin/README.md).
- Install the pinned public Megatron checkout using `install_megatron.sh`. The
  launcher discovers `.deps/Megatron-LM`, or accepts `--megatron-path` or
  `MEGATRON_PATH`. An installed Megatron-LM with `megatron.training` is also
  supported. The launcher builds `PYTHONPATH` from this checkout and the selected
  Megatron installation.
- The current storage checks require the checkout, output/runtime directories
  and Docker data root on a filesystem distinct from `/`. Symlinks are resolved
  before checking. Configure Docker's data root before launching.
- Runtime storage uses `--work-dir` or `COMPACTIONRL_WORK_DIR`, defaulting to
  `OUTPUT/runtime`. Temporary files, HF, Torch extension, Triton, CUDA, FlashInfer
  and Ray caches, and local Ray spill files are placed there. Python bytecode
  writes are disabled.
- Source `scripts/storage_env.sh` before installing dependencies to place package,
  model and compilation caches on the chosen data volume. For a pre-existing Ray
  cluster, configure each node's storage before starting Ray; a driver cannot
  relocate an already running cluster's files.
- Use a short runtime path: the derived `ray/` directory should be at most 36
  bytes to accommodate Ray's Unix socket names.
- CPU offload and actor/critic swapping require substantial host RAM. Check both
  host and container limits. The launcher stops its own run when cgroup memory
  usage exceeds 93%; this does not replace a container memory quota.
- Service cleanup targets this launch's processes and Docker probe. Runtime
  caches and buffer records remain for inspection.

See [checks](reproduction.md#checks) for preflight and training inspection.
