<h1 align="center">CompactionRL</h1>

<p align="center">
  <b>Reinforcement Learning with Context Compaction for Long-Horizon Agents</b>
</p>

<p align="center">
  Yujiang Li · Zhenyu Hou · Yi Jing · Jie Tang · Yuxiao Dong
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.05378"><img src="https://img.shields.io/badge/arXiv-2607.05378-b31b1b.svg" alt="arXiv paper"></a>
  <a href="https://arxiv.org/pdf/2607.05378"><img src="https://img.shields.io/badge/Paper-PDF-blue.svg" alt="Paper PDF"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#results">Results</a> ·
  <a href="#repository-layout">Code</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#training">Training</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#citation">Citation</a>
</p>

**CompactionRL** trains agents to solve tasks and summarize their own interaction
history under a fixed context budget. When the context approaches its limit, the
policy generates a summary and continues from the compressed state. Task actions
and summary tokens learn together from the final task reward.

This repository brings together the agent framework, asynchronous RL training,
and rollout buffer, with configurations for GLM-4.7-Flash and GLM-4.5-Air-SFT.

## Overview

![Context compaction and benchmark results](assets/overview.png)

*Context compaction extends the interaction horizon while bounding the working
context.*

Long agent trajectories accumulate tool outputs, reasoning and intermediate
state. A useful summary must preserve the information needed by later actions,
so compaction quality directly affects task success. CompactionRL makes that
summary part of the trainable policy and includes it in the RL objective.

The method combines three elements:

- **Compaction-aware rollouts.** The agent summarizes earlier history and resumes
  with the summary and recent tool-interaction pairs, allowing execution to
  continue within the context budget.
- **Joint execution and summary training.** Single-rollout, critic-based PPO
  optimizes both kinds of generated tokens using the final task reward. Token
  loss normalization accounts for different execution and summary lengths.
- **Cross-segment credit assignment.** A position correction to local GAE uses
  the number of trainable tokens in later segments to account for each segment's
  distance to the final outcome.

### Method

![CompactionRL rollout and training framework](assets/method.png)

*Execution and summary segments share the final reward. Training retains the
original generated tokens, log probabilities and loss masks across compaction.*

For segment $s$, let $N_{>s}$ be the number of optimized tokens generated in later
segments of the same rollout. The cross-segment correction is:

$$
\hat{A}_{s,i} = (\gamma\lambda_s)^{N_{>s}} A^{\mathrm{local}}_{s,i}.
$$

This approximates trajectory-level credit assignment while keeping segments
independently batchable. Summary tokens participate in the loss; observations
and repeated context prefixes are masked. See the
[paper-to-code map](slime/compactionrl/docs/implementation.md) and
[training semantics](slime/compactionrl/docs/reproduction.md#training-semantics)
for segment construction, token counts and actor/critic updates.

## Results

Pass@1 accuracy (%), mean ± standard deviation over three evaluation runs
([Table 2](https://arxiv.org/abs/2607.05378v2)):

| Backbone | Peak context | Method | SWE-bench Verified (200 tasks) | Terminal-Bench 2.0 |
| --- | --- | --- | ---: | ---: |
| GLM-4.7-Flash (30B-A3B) | 64k | Base + inference compaction | 50.5 ± 1.2 | 13.5 ± 0.9 |
| GLM-4.7-Flash (30B-A3B) | 64k | **CompactionRL** | **56.0 ± 0.4** | **20.2 ± 0.9** |
| GLM-4.5-Air-SFT (106B-A12B) | 80k | Base + inference compaction | 59.8 ± 1.9 | 21.3 ± 1.6 |
| GLM-4.5-Air-SFT (106B-A12B) | 80k | **CompactionRL** | **66.4 ± 0.7** | **26.2 ± 2.1** |

CompactionRL improves the two benchmarks by **5.5 / 6.7 points** on
GLM-4.7-Flash and **6.6 / 4.9 points** on GLM-4.5-Air-SFT compared with their
inference-compaction baselines. Compacted evaluations allow up to three
compactions while retaining the stated peak context length.

<details>
<summary>Additional controlled comparisons from Table 2</summary>

| Backbone | Method | Training compaction | Inference compaction | SWE-bench Verified (200 tasks) | Terminal-Bench 2.0 |
| --- | --- | :---: | :---: | ---: | ---: |
| GLM-4.7-Flash | Base | — | No | 47.3 ± 2.7 | 14.6 ± 1.6 |
| GLM-4.7-Flash | RL | No | No | 52.5 ± 2.5 | 16.9 ± 2.4 |
| GLM-4.7-Flash | RL + inference compaction | No | Yes | 48.2 ± 0.5 | 12.4 ± 0.9 |
| GLM-4.7-Flash | CompactionRL without summary loss | Yes | Yes | 54.5 ± 1.2 | 12.4 ± 0.0 |
| GLM-4.5-Air-SFT | Base | — | No | 57.8 ± 2.1 | 18.0 ± 2.8 |
| GLM-4.5-Air-SFT | RL | No | No | 58.3 ± 0.6 | 20.2 ± 3.7 |
| GLM-4.5-Air-SFT | RL + inference compaction | No | Yes | 62.5 ± 1.1 | 23.6 ± 0.9 |
| GLM-4.5-Air-SFT | CompactionRL without summary loss | Yes | Yes | 64.5 ± 2.4 | 21.7 ± 2.8 |

These comparisons use the same model-specific peak context lengths as the main
table. The two RL rows for each backbone use the same checkpoint. The
no-summary-loss variant retains compaction during training and inference.

</details>

SWE-bench uses the fixed 200-instance subset from Appendix A.1, and
Terminal-Bench uses the full 2.0 set.

## Repository Layout

```text
CompactionRL/
├── slime/                  # slime training framework with CompactionRL
│   ├── slime/              # PPO, GAE, model backends and buffer conversion
│   └── compactionrl/       # Training configs, scripts, patch, docs and tests
├── terminus/               # Harbor / Terminus-KIRA agent framework
│   ├── bin/                # TiTO Linux executable
│   ├── prompts/            # Summary, resume and offline judge prompts
│   ├── examples/           # File and compaction smoke tasks
│   └── scripts/            # Task preparation, evaluation and analysis
├── buffer/                 # SQLite buffer and rollout collector
│   ├── src/compactionrl/   # Buffer schema, service and storage helpers
│   └── data/manifests/     # Task-ID manifests
├── scripts/                # Unified training wrappers and storage setup
├── assets/                 # Paper overview and method figures
└── SOURCE_MANIFEST.json    # Source revisions and overlay scope
```

| Component | Guide | Purpose |
| --- | --- | --- |
| `slime/` | [Training guide](slime/compactionrl/README.md) | PPO, critic, cross-segment GAE, token-normalized loss and training configurations |
| `terminus/` | [Agent guide](terminus/README.md) | Harbor / Terminus-KIRA, compaction, Docker task execution, evaluation and TiTO |
| `buffer/` | [Buffer guide](buffer/README.md) | Rollout collection, token metadata and training-batch delivery |

Source revisions and framework changes are recorded in
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) and the
[slime overlay](slime/compactionrl/docs/slime_overlay.md).

## Getting Started

### Requirements

- Linux x86_64, Python **3.12**, and glibc ≥ 2.34 for the bundled TiTO executable.
- A compatible NVIDIA training environment with PyTorch, CUDA, Transformer
  Engine, SGLang, Megatron-LM and mbridge. See the
  [environment guide](slime/compactionrl/docs/environment.md) for dependency
  versions and compatibility requirements.
- Docker with the Compose plugin for task execution.
- Local model weights and prepared benchmark task environments.

### Installation

Run from the repository root in an existing compatible GPU environment. The
current launcher requires the checkout, output/cache directories and Docker data
root on a mounted data filesystem; see the [deployment guide](slime/compactionrl/docs/glm47_launch.md#deployment-requirements).

```bash
# Choose a short runtime path on a mounted data volume.
export COMPACTIONRL_WORK_DIR=/data/cr
source scripts/storage_env.sh

python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e 'buffer[test]'
pip install -e terminus
pip install --no-deps -e slime

# Install the pinned public Megatron-LM checkout used by the Flash recipe.
bash slime/compactionrl/scripts/install_megatron.sh
export MEGATRON_PATH="$PWD/.deps/Megatron-LM"
export PYTHONPATH="$PWD/buffer/src:$PWD/slime:$PWD/terminus/src:$MEGATRON_PATH"
```

Download the official [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
HF checkpoint to a local directory before running the examples. The experiments
use GLM-4.7-Flash (30B-A3B) and GLM-4.5-Air-SFT (106B-A12B); see
[models and data](slime/compactionrl/docs/reproduction.md#models-and-data).

## Training

### Quick start: Flash smoke

Use the Flash checkpoint with `model_type=glm4_moe_lite`. The smoke recipe
targets one **8 × 80 GB GPU** node: four GPUs shared by
actor/critic and four for rollout. It uses an **8k context**, two bundled tasks
and **five update batches** to check integration. CPU offload requires host RAM;
check both the host and container memory limits before running.

Inspect the environment and generated commands first:

```bash
export MODEL_PATH=/absolute/path/to/GLM-4.7-Flash
python slime/compactionrl/scripts/preflight.py --model "$MODEL_PATH"
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --dry-run
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --validate-only
```

`--dry-run` prints the service and training commands. `--validate-only` checks
model/task assets and training arguments. Once these checks pass, launch:

```bash
bash scripts/smoke_train_glm47_flash.sh \
  --model "$MODEL_PATH" --steps 5 --output "$PWD/runs/glm47-flash-smoke"
```

The launcher starts the SGLang router, Harbor, TiTO and buffer, then the slime
training driver. The output directory contains service logs, TensorBoard events,
the buffer database and actor/critic checkpoints. Use a new output directory for
each invocation.

To continue a saved run for one further update batch:

```bash
bash scripts/smoke_train_glm47_flash.sh \
  --model "$MODEL_PATH" --resume "$PWD/runs/glm47-flash-smoke" \
  --steps 6 --output "$PWD/runs/glm47-flash-resume"
```

`--steps` specifies the total target, including batches completed before resuming.
See the [launch guide](slime/compactionrl/docs/glm47_launch.md) for ports,
initialization, checkpoint continuation and resource settings.

### Paper configurations

| Configuration | Purpose |
| --- | --- |
| [Flash smoke](slime/compactionrl/configs/glm47_flash/smoke.yaml) | Five-batch integration check, 8k context and one 8-GPU node |
| [Flash paper](slime/compactionrl/configs/glm47_flash/paper.yaml) | 30B, 64k context, PPO and critic pretraining |
| [Air paper](slime/compactionrl/configs/glm45_air/paper.yaml) | 106B Air-SFT, 80k context, PPO and critic pretraining |
| [Paper ablations](slime/compactionrl/configs/ablations/README.md) | Summary loss, cross-segment GAE, loss normalization, SUPO and 160k context comparisons |

For Flash, prepare the paper training task list and Harbor-format environments,
and use a compatible Ray cluster with this checkout, model and task directories
available at the same absolute paths on all nodes. Inspect the plan first:

```bash
bash scripts/train_glm47_flash.sh \
  --config slime/compactionrl/configs/glm47_flash/paper.yaml \
  --model "$MODEL_PATH" \
  --tasks /absolute/path/to/paper-training-task-ids.json \
  --dataset /absolute/path/to/prepared-swedev-tasks \
  --ray-address "$HEAD_IP:6379" --host "$HEAD_IP" --bind-host 0.0.0.0 \
  --steps 230 --output "$PWD/runs/glm47-flash-paper" --dry-run
```

Use `--validate-only` in place of `--dry-run` to check local assets and training
arguments. Remove that flag to launch once environment checks pass. Configure
cluster parallelism for the available hardware as described in the
[launch guide](slime/compactionrl/docs/glm47_launch.md#flash-paper-training).

The training wrapper defaults to the Flash paper configuration. Pass `--tasks`
explicitly: the current launcher does not expand the YAML's
`${TRAIN_TASK_MANIFEST}` placeholder. In this implementation, **230 batches =
50 critic-only batches + 180 PPO batches**. Air's configuration uses **150 =
50 + 100**. The two critic passes per batch also apply during warmup; see
[training semantics](slime/compactionrl/docs/reproduction.md#training-semantics)
for the distinction between batches and optimizer updates.

## Evaluation

Use the fixed [SWE-bench Verified manifest](terminus/data/manifests/swe_verified_200.json)
with locally prepared Harbor tasks and Docker images:

```bash
python terminus/scripts/prepare_tasks.py \
  --dataset /absolute/path/to/swe-verified-harbor \
  --manifest terminus/data/manifests/swe_verified_200.json \
  --output "$COMPACTIONRL_WORK_DIR/swe200.json"
```

This command validates existing tasks; it does not download or convert benchmark
content. For Terminal-Bench 2.0, omit `--manifest` to select the full prepared
local dataset and record its version.

Configure the inference, Harbor and TiTO services for the checkpoint being
evaluated, then run:

```bash
export EVAL_MODEL_PATH=/absolute/path/to/evaluation-hf-checkpoint
python terminus/scripts/evaluate.py \
  --config slime/compactionrl/configs/glm47_flash/paper.yaml \
  --tasks "$COMPACTIONRL_WORK_DIR/swe200.json" --model "$EVAL_MODEL_PATH" \
  --runs 3 --output "$PWD/runs/evaluation"
```

The evaluation script calls existing services; it does not start them. Use the
Air configuration for its 80k evaluation setting and the checkpoint matching
the desired result row.

Every run is retained and aggregate Pass@1 statistics are written to
`summary.json`. Infrastructure failures remain in the denominator and are
recorded separately. See the [evaluation protocol](slime/compactionrl/docs/reproduction.md#evaluation-protocol)
for timeout settings and result aggregation.

## Documentation and Checks

| Guide | Contents |
| --- | --- |
| [Training](slime/compactionrl/docs/glm47_launch.md) | Smoke and paper launch, services, storage, resume and resource requirements |
| [Paper-to-code map](slime/compactionrl/docs/implementation.md) | Algorithms, prompts and token metadata |
| [slime overlay](slime/compactionrl/docs/slime_overlay.md) | Source revisions, framework changes and patch reconstruction |
| [Buffer protocol](buffer/docs/buffer_protocol.md) | Segment construction, masks and batch delivery |
| [Reproduction](slime/compactionrl/docs/reproduction.md) | Training settings, data and evaluation protocol |
| [Paper ablations](slime/compactionrl/configs/ablations/README.md) | Configuration overlays and inference settings |

After installation, run the project CPU checks:

```bash
pytest -c pytest.ini -q
python -B scripts/check_release.py
```

## Acknowledgments and License

CompactionRL builds on [slime](https://github.com/THUDM/slime) and
[Harbor](https://github.com/laude-institute/harbor), with the Terminus-KIRA agent
scaffold. We thank their contributors for the training and execution frameworks.

Project code is licensed under [Apache-2.0](LICENSE). Retained upstream notices
and the TiTO distribution scope are described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model checkpoints, benchmark
content and container images retain their respective licenses.

## Citation

If you use CompactionRL in your research, please cite our
[paper](https://arxiv.org/abs/2607.05378)
([PDF](https://arxiv.org/pdf/2607.05378)):

```bibtex
@misc{li2026compactionrl,
  title         = {CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents},
  author        = {Yujiang Li and Zhenyu Hou and Yi Jing and Jie Tang and Yuxiao Dong},
  year          = {2026},
  eprint        = {2607.05378},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.05378}
}
```
