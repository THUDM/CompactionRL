# Paper experiments and reproduction

This guide describes the training settings, data preparation and evaluation
protocols for [CompactionRL](https://arxiv.org/abs/2607.05378v2).

## Paper settings

| Setting | Paper specification |
| --- | --- |
| Models | GLM-4.7-Flash (30B-A3B); GLM-4.5-Air-SFT (106B-A12B) |
| Peak context | Flash 64k; Air-SFT 80k |
| Global batch / independent rollouts per prompt | 128 segments / 1 rollout |
| Actor / critic learning rate | 2e-6 / 3e-6 |
| Critic pretraining / critic passes per subsequent PPO batch | 50 steps / 2 |
| Discount / length-adaptive GAE | gamma 1; alpha 1.5 |
| Response cap / compaction trigger / summary output cap | 10,240 / 10,240 / 10,240 tokens |
| Maximum compactions / retained interaction pairs | 3 / 2 |
| Evaluation | Temperature and top-p 1; up to 250 turns; three runs |

The [Flash](../configs/glm47_flash/paper.yaml) and
[Air](../configs/glm45_air/paper.yaml) configurations encode these settings. The
[launch guide](glm47_launch.md#flash-paper-training) provides the Flash entry point
and model-specific launch requirements. The separate smoke profile uses
8k context and two bundled tasks to check integration.

## Training semantics

Training starts with critic pretraining, followed by joint actor/critic PPO
batches. The configuration fields count these batches as follows:

| Field or operation | Current behavior |
| --- | --- |
| `num_rollout` / launcher `--steps` | Total training-batch target, including critic-only batches and any resumed batches |
| Flash `num_rollout=230` | 50 critic-only batches followed by 180 PPO batches |
| Air `num_rollout=150` | 50 critic-only batches followed by 100 PPO batches |
| `critic_train_epoch=2` | Two critic training passes per batch, including the critic-only phase |
| `global_batch_size=128` | 128 trainable segments supplied by the buffer, not 128 complete rollouts |
| Actor GAE | Length-adaptive lambda computed from optimized-token count, with the cross-segment position correction |
| Critic targets | Use the configured `lambd_critic=1.0`; this is separate from actor length-adaptive lambda |

During critic-only batches, the actor does not update. During subsequent PPO
batches, the critic trains first and provides values for the actor update. Set
`--steps 230` for Flash to include 50 critic-pretraining batches and 180 PPO
batches. Each batch includes two critic training passes; individual optimizer
updates depend on the batch and microbatch configuration.

## Release implementation

The [paper-to-code map](implementation.md) locates these operations:

- Execution and summary tokens share the final task reward. No separate
  summary-quality or summary-length reward is used in training.
- Observation tokens and repeated context prefixes are excluded from the loss.
  A full context snapshot may contain execution tokens followed by summary
  tokens. The collector excludes a duplicate standalone summary export while
  retaining its generated tokens in the main snapshot.
- Subsequent-token counts are computed from original generation masks before
  segments are split into batches. Cross-segment GAE applies a position discount
  to each segment's local GAE; it approximates full-trajectory credit assignment.
- Summary generation and context reconstruction preserve recent interaction pairs.
  The [summary](../../../terminus/prompts/summary.txt) and
  [resume](../../../terminus/prompts/resume.txt) files record the paper templates.
  Runtime text is built by `Terminus2._build_claude_style_summary_prompt` and
  `Terminus2._build_summary_rehydration_user_message`; editing the text files alone
  does not change those runtime functions.
- Token/logprob transport and segment metadata are documented in the
  [buffer protocol](../../../buffer/docs/buffer_protocol.md). TiTO is supplied as
  a [Linux executable](../../../terminus/bin/README.md).

Source revisions and retained framework changes are documented in
[SOURCE_MANIFEST.json](../../../SOURCE_MANIFEST.json) and the
[overlay](slime_overlay.md).

## Models and data

The official [Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) and
[Air](https://huggingface.co/zai-org/GLM-4.5-Air) weights are public. The paper's
Air-SFT starting point is Air after supervised fine-tuning on GLM-4.7-generated
trajectories. Select the starting checkpoint for the corresponding experiment.
The Flash launcher initializes actor and critic from the supplied HF model;
`--actor-init` and `--critic-init` select Megatron-format initialization checkpoints.
See [checkpoint handling](glm47_launch.md#checkpoints-and-outputs) for initialization
and continuation.

The [evaluation manifest](../../../terminus/data/manifests/swe_verified_200.json)
contains the fixed 200 SWE-bench Verified instances from Appendix A.1.
Terminal-Bench 2.0 uses the full prepared dataset. Task contents, Harbor-format
environments and container images must be obtained/prepared separately;
`prepare_tasks.py` only validates an existing task directory.

For training, pass an explicit JSON array of task IDs through `--tasks` and the
prepared Harbor task directory through `--dataset`. Keep the task selection and
dataset version with the run configuration.

## Evaluation protocol

`terminus/scripts/evaluate.py` calls existing inference, Harbor and TiTO services;
it does not start them. Supply an HF checkpoint and use the model-specific agent
configuration. For each result, retain the checkpoint identifier, repository
revision, dataset version/manifest, complete resolved configuration and run files.

Evaluation settings and outputs:

- Paper YAMLs specify generation/context/compaction settings. The evaluation
  script takes its HTTP/task timeout from `--timeout` (default 1800 seconds),
  independently of the YAML's top-level `timeout_seconds`.
- The paper YAMLs do not set `agent.timeout_multiplier`; Harbor currently defaults
  it to 100.0. Set this field explicitly in the agent configuration to control
  task-environment timeouts.
- Infrastructure failures remain in the denominator and are recorded separately.
  Report their count and the retry policy alongside the scores.
- `summary.json` contains the aggregate mean and both `std_population` and
  `std_sample` across runs.
- Evaluate the checkpoint corresponding to each row. Enabling inference compaction
  on a no-compaction RL checkpoint is a separate evaluation from training with
  compaction. Select training and inference configurations separately for these
  comparisons.

## Paper experiment coverage

| Paper item | Configuration or tool | Usage |
| --- | --- | --- |
| Table 1: different summary models | Harbor accepts a separate summary service endpoint | Set the summary model and service independently of the acting model |
| Table 2: main results | Flash/Air paper configs and no-compaction/no-summary-loss overlays | Evaluate the checkpoint and inference configuration corresponding to each row |
| Table 3: PPO and optimization ablations | SUPO-style, sequence-level-loss and no-cross-segment-GAE overlays | Keep the model, data and training budget consistent across comparisons |
| Table 4: 160k comparison | Long-context no-compaction overlay | Specify training and evaluation configurations separately for all table cells |
| Table 8: sensitivity | `rollout_max_context_length`, `context_management_max_count`, `summary_history_user_turns` | Vary inference settings while keeping the trained checkpoint fixed |
| Figures 3/4: behavior and training curves | `buffer/scripts/analyze_buffer.py`, `slime/compactionrl/scripts/analyze_training.py` | Analyze buffer records and export TensorBoard scalars |
| Appendix C / Table 9: summary quality | `terminus/prompts/summary_judge.txt`, `terminus/scripts/summary_quality.py` | Score summaries repeatedly and compare models with paired bootstrap |

See [paper ablations](../configs/ablations/README.md) for overlay usage. Summary
judging is an offline analysis, not a reward used by the training pipeline.

<a id="validation-status"></a>

## Checks

From the repository root, inspect dependencies and model assets, run the CPU
tests, and check the resolved launch configuration:

```bash
python slime/compactionrl/scripts/preflight.py --model "$MODEL_PATH"
pytest -c pytest.ini -q
python -B scripts/check_release.py
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --dry-run
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --validate-only
```

During a training run, inspect actor/critic losses, summary tokens in the loss
mask, policy parameter updates and rollout weight versions. Use the
[resume example](glm47_launch.md#checkpoints-and-outputs) to check checkpoint
continuation. Service logs, TensorBoard events and the buffer database are
retained in the output directory for inspection.
