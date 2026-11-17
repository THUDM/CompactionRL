# Paper ablation recipes

These overlays correspond to experiments in the paper. Deep-merge an overlay
with the relevant model's **paper configuration** to produce a standalone YAML;
there is no implicit configuration inheritance at runtime.

| Overlay | Paper comparison | Change |
| --- | --- | --- |
| `no_compaction.yaml` | Table 2: ordinary RL | Disable compaction in the agent |
| `no_summary_loss.yaml` | Table 2: no summary loss | Keep compaction, exclude summary generation from training |
| `no_cross_segment_gae.yaml` | Table 3 | Use local GAE without the position discount |
| `sequence_level_loss.yaml` | Table 3 | Use sequence-normalized loss |
| `supo.yaml` | Table 3 | Group four independent rollouts per prompt and broadcast each rollout advantage to its segments |
| `long_context_160k.yaml` | Table 4 | Use 160k context with compaction disabled |

For example, build the Flash no-summary-loss configuration:

```bash
python slime/compactionrl/scripts/merge_config.py \
  slime/compactionrl/configs/glm47_flash/paper.yaml \
  slime/compactionrl/configs/ablations/no_summary_loss.yaml \
  --output "$COMPACTIONRL_WORK_DIR/flash-no-summary-loss.yaml"
```

Pass the output through `--config` to the Flash training wrapper together with
explicit `--tasks`, model, dataset and cluster options from the
[launch guide](../../docs/glm47_launch.md#flash-paper-training).

Training and inference settings must be chosen separately. To evaluate ordinary
RL with inference compaction, train with `no_compaction`, then evaluate that same
checkpoint with the base paper agent configuration. Table 4 similarly combines
long-context and compacted inference settings across training checkpoints.

For Table 8, vary the inference context budget, maximum number of compactions,
or retained interaction pairs while keeping the trained checkpoint fixed. See
[experiment coverage](../../docs/reproduction.md#paper-experiment-coverage) for
the corresponding configuration fields and analysis tools.
