# GLM-4.7-Flash launch recipes

The launcher uses the Harbor and slime code, SQLite buffer and TiTO executable
from this checkout. Buffer-to-sample conversion and GAE are built into slime. The old
private helper scripts, reward/checker services, tmux
sessions, experiment directories and W&B gateway are not dependencies.

External assets are the model checkpoint and prepared Harbor task directories.
The Python/CUDA training libraries and Docker remain software dependencies.
TiTO is the executable under `terminus/bin/`, using this project’s Python 3.12 environment;
see [binary requirements and protocol](../../../terminus/bin/README.md). An
installed Megatron-LM with `megatron.training` is supported; `--megatron-path` (or
`MEGATRON_PATH`) optionally adds an upstream checkout. Inherited `PYTHONPATH` is
not reused, so an old private checkout cannot silently provide project modules.

## Small run

From the repository root after installation:

```bash
export COMPACTIONRL_WORK_DIR=/short/path/on/data/volume/cr
source scripts/storage_env.sh
export MODEL_PATH=/path/to/GLM-4.7-Flash
bash slime/compactionrl/scripts/install_megatron.sh
# The launcher finds .deps/Megatron-LM automatically.
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --dry-run
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --validate-only
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" \
  --steps 5 --output "$PWD/runs/flash-smoke"
```

The smoke profile uses one 8×80GB node, with 4 GPUs shared by actor/critic and 4
for rollout, short contexts and two repository-owned tasks. It retains standard
clipped PPO for a small integration check. The reference profile below uses the
historical VAPO/DPPO settings. The smoke task uses 300 ledger rows, a 4,096-token
remaining-budget trigger and zero retained interaction pairs so a summary fits
inside the 8k budget. The reference profile retains two pairs. A resource target
is not a measured minimum.

The smoke profile disables the additional policy weight backup because it uses
no reference or old-policy model. Actor/critic model swapping and CPU optimizer
offload still use substantial host RAM; inspect both host and container memory
limits before running.

Its memory configuration is TP/EP/ETP = 4/4/1, sequence parallelism, BF16 gradient
reduction, 70% optimizer CPU offload and full activation recomputation. These are
smoke-test resource choices; the reference recipe is unchanged. BF16 gradient
reduction can change numerical results and is not a paper reproduction setting.

Training allocations use CPU backing and disable memory-saver disk backup. Size
host/container RAM for model swapping and optimizer offload before running. The
launcher stops its own launch when cgroup RAM usage exceeds 93% of its limit.

`--validate-only` runs the actual argument parser and configuration checks in a
process with an 8 GiB address-space hard limit, hidden GPUs and single-threaded
library settings. It reports peak process RSS and avoids importing the full
trainer/Bridge or constructing a model provider. The bounded check still needs
Linux verification. Address space differs from RSS, and the training RAM watchdog
does not constitute a kernel-enforced container quota.

On Linux, both `--output` and `--work-dir` must resolve to a filesystem distinct
from `/`; paths through symlinks are checked before creating directories. Runtime
storage defaults to `OUTPUT/runtime`. The launcher overrides inherited temporary,
HF, Torch extension, Triton, CUDA, FlashInfer and Ray cache locations. Ray's local
object spill files use that same data volume. Python bytecode writes are disabled.
`scripts/storage_env.sh` also redirects pip/uv build and download caches during
installation. Docker is checked separately: its existing data root must be on a
data volume before any probe or task container starts. The launcher does not
reconfigure the daemon.

Choose a short `--work-dir` for a local Ray cluster: its `ray/` path should be at
most 36 bytes because Ray appends a session directory and Unix socket filename.
The launcher checks this before loading GPUs. Local Ray cluster startup with
these settings remains unverified.

`--dry-run` prints every service command, the training command, resolved settings
and project environment without starting processes. `--validate-only` checks
model shards, task files and the actual slime/Megatron/SGLang argument parsers;
it does not allocate training GPUs or start Ray/services. Normal launch repeats
these checks, starts a uniquely named Docker probe, then starts the SGLang router, Harbor/TiTO/buffer
and a private local Ray cluster. Failed container startup stops before model loading.

The output contains `runtime.yaml`, `command.json`, individual service logs,
TensorBoard events, the buffer database and actor/critic checkpoints. A new output
directory is required for every invocation. Port overrides update the services,
collector, trainer and router together (`--router-metrics-port` controls its
separate metrics listener). Cleanup targets only this launch's processes
and Docker probe.

```bash
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" \
  --resume "$PWD/runs/flash-smoke" --steps 6 \
  --output "$PWD/runs/flash-resume"
```

`--resume` accepts the previous run directory or its `actor/` subdirectory and
uses the sibling critic checkpoint. `--steps` is the target total number of rollout
updates. To initialize a new run from Megatron weights without optimizer/RNG
continuation, use `--actor-init` and/or `--critic-init` instead. HF actor and critic
initialization through mbridge is the default when neither is supplied. The GLM
`raw` path handles weight export to SGLang; actor and critic models are built from
the repository model definition and runtime arguments.

## Historical reference script

`slime/compactionrl/configs/glm47_flash/reference.yaml` adapts the supplied script
`0502_glm47flash_vapo_swe_64k_tito_summaryauto_mixtrain_swedev1.2k_summaryasaturn_gae_0.5timeout_disablepager_fixmainkey.sh`.
Its internal experiment tag starts with `0504`; the filename alone is not proof of
which checkout or checkpoint generated a paper result.

| Setting | Reference profile |
|---|---|
| Actor/critic allocation | 2 nodes × 8 GPUs, shared |
| Rollout allocation | 64 GPUs, 4 per engine |
| Total allocation | 80 GPUs; typically 10 nodes with 8 GPUs each |
| TP / PP / CP / EP / ETP | 4 / 2 / 2 / 8 / 1 |
| Estimator and clip | VAPO, DPPO, epsilon low 0.2 / high 3.0 |
| Batch / samples per prompt | 128 / 1 |
| Actor / critic LR | 2e-6 / 3e-6 |
| Critic epochs / warmup in this script | 2 / 0 |
| Model context / generation cap | 81,920 / 65,536 |
| Summary input / output caps | 65,536 / 10,240 |
| Compaction threshold / retained pairs / maximum count | 10,240 / 2 / 3 |
| Weight update interval | 8 rollout batches |
| Collector concurrency / timeout | 500 / 7,200 seconds |

The script's training manifest contains **1,105 unique IDs**, now included as
`buffer/data/manifests/swedev_reference_1105.json`. It is not an inferred 1,200-task list.
Only IDs are included; prepare the benchmark environments/images separately.

Use a pre-existing Ray cluster. On its head and worker nodes, install this checkout
and dependencies, and expose the same absolute model, repository and output paths
(e.g. shared storage). Start Ray using its normal CLI; the launcher neither SSHes
to machines nor stops existing clusters. The `--ray-address` is the Ray GCS
address, normally `HEAD_IP:6379`, not the dashboard URL.

For an existing cluster, source `scripts/storage_env.sh` on every node before
starting Ray and explicitly place `ray start --temp-dir` and
`--object-spilling-directory` on its data volume. A driver cannot relocate files
of an already running cluster. Verify each node's Docker data root as well.

```bash
# Run on the head node of an already started Ray cluster.
bash scripts/train_glm47_flash.sh --model "$MODEL_PATH" \
  --dataset /path/to/prepared-swedev-tasks \
  --ray-address "$HEAD_IP:6379" --host "$HEAD_IP" --bind-host 0.0.0.0 \
  --steps 180 --output "$PWD/runs/flash-reference" --validate-only
# Remove --validate-only to launch once the environment checks pass.
```

The reference config retains the original script's large 300,000-update cap;
pass an explicit `--steps` for your experiment. The example 180-update target
is a bounded run, not a claim to reproduce the paper. `--tasks /path/to/ids.json`
can select a different JSON array of task IDs.

## Deliberate differences and unresolved semantics

- The GLM profiles use `megatron_to_hf_mode: raw` and mbridge HF loading, which
  uses the pinned public Megatron Core 0.15.3 checkout. The optional NVIDIA Bridge
  mode requires a matching Megatron Core/Bridge combination; the host's initial
  private Megatron checkout was incompatible with Bridge 0.3.0rc0.
- The paper specification remains in `paper.yaml`. Its 64k budget and standard
  PPO settings differ from the historical script's 80k model budget and DPPO.
- slime's strict buffer conversion and built-in approximate GAE are used in both profiles;
  the release conventions are documented in [reproduction notes](reproduction.md#release-implementation).
- Optimizer/RNG checkpoints are saved for true continuation. The old script
  disabled optimizer save/load, so its `--load` behavior was not full resume.
- A zero-coefficient KL loss is disabled to avoid loading an unused reference
  policy. Dummy math rewards, TIR checker endpoints and unused AIME eval flags
  are removed. Harbor verifier rewards drive training.
- The historical `update_critic_expert_weights_only` flag is retained in the
  reference config. In the selected slime fork it eagerly initializes chained
  optimizer states but does **not** implement a parameter freeze. No unverified
  expert/head freezing rule has been invented here.
- Task timeout multipliers are explicit (smoke 1.0, reference 0.5). The supplied
  script's filename suggests 0.5 but its referenced YAML omitted the multiplier;
  this is an explicit release choice, not proven historical runtime behavior.
- Service cleanup targets the launcher's own processes. TiTO's TTL and request
  timeout follow the chosen run profile. Its supplied executable and runtime
  requirements are documented in `terminus/bin/README.md`.

See [validation status](reproduction.md#validation-status) for completed checks. Parser checks are not
proof of model loading, distributed weight updates, or successful training.
