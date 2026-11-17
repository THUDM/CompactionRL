# Ablation overlays

These YAML files contain deep-merge overrides to the model's base configuration.
Use `slime/compactionrl/scripts/merge_config.py` to produce an explicit standalone config, then pass
that config to the corresponding launcher/evaluation command. There is no implicit
configuration inheritance at runtime.

| Overlay | Change |
|---|---|
| `no_compaction.yaml` | No training/evaluation compaction |
| `no_summary_loss.yaml` | Summary generation has its own key; train main snapshots only |
| `no_cross_segment_gae.yaml` | Local GAE without trajectory-position discount |
| `sequence_level_loss.yaml` | Use the inherited sequence-normalized loss reducer |
| `long_context_160k.yaml` | Extended-context no-compaction reference |
| `supo.yaml` | Four independent rollouts per prompt, normalized before segment batching |
| `historical_dppo.yaml` | Historical clipping variant; not the paper's standard PPO equation |
| `stepwise_advantage.yaml` | Optional stepwise GAE with end-of-step critic supervision; see [details](../../docs/stepwise_advantage.md) |

Inference-only compaction uses a checkpoint trained with `no_compaction`, then
enables the base agent compaction settings during evaluation. Sensitivity studies
vary `rollout_max_context_length`, `context_management_max_count`, or
`summary_history_user_turns`. These variants need their own test/compute budget;
the small smoke run does not validate all of them.
