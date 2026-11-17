# Experiment recipes and reproduction

The paper uses GLM-4.7-Flash (30B-A3B) and GLM-4.5-Air-SFT (106B-A12B).
The Air starting point is GLM-4.5-Air after SFT on GLM-4.7 trajectories.
Model weights and prepared benchmark task environments are supplied separately.

## Paper settings

| Setting | Paper recipe |
| --- | --- |
| Context budget | Flash 64k; Air-SFT 80k |
| Global batch / rollouts per prompt | 128 / 1 |
| Actor / critic learning rate | 2e-6 / 3e-6 |
| Critic warmup / updates per policy update | 50 steps / 2 |
| Discount / length-adaptive GAE | gamma 1; alpha 1.5 |
| Compaction trigger / summary output limit | 10,240 tokens / 10,240 tokens |
| Maximum compactions / retained interaction pairs | 3 / 2 |
| Evaluation | Temperature and top-p 1; up to 250 turns; three runs |

The [Flash](../configs/glm47_flash/paper.yaml) and
[Air](../configs/glm45_air/paper.yaml) YAML files record these settings.
Their cluster parallelism requires hardware calibration. The
[Flash smoke](../configs/glm47_flash/smoke.yaml) recipe targets five updates on
one 8-GPU node with 8k context. The
[reference recipe](../configs/glm47_flash/reference.yaml) adapts the supplied
historical VAPO/DPPO script with 80k context and 80 GPUs. See
[launch instructions](glm47_launch.md) before selecting a configuration.

## Release implementation

The complete official slime baseline and source identifiers are recorded in
[SOURCE_MANIFEST.json](../../../SOURCE_MANIFEST.json). The
[overlay](slime_overlay.md) describes its CompactionRL changes, and the
[implementation map](implementation.md) locates the paper's code paths.

- Generated execution and summary tokens share the final task reward.
  Observation tokens and repeated context prefixes are excluded from the loss.
- Subsequent-token counts come from the original generation loss masks before
  batching. Length-adaptive lambda uses the effective optimized-token count.
- Cross-segment GAE applies a position discount to each segment's local GAE;
  it approximates full-trajectory credit assignment and keeps segments independently
  batchable. Scalar task rewards remain unchanged across repeated critic updates.
- [Stepwise advantage](stepwise_advantage.md) retains its separate chunk-count
  convention. The optional SUPO-style baseline normalizes rewards over independent
  rollouts and assigns each rollout's result to all of its segments.
- The Flash launcher uses the public Megatron checkout, mbridge HF loading and
  raw GLM weight export. Optimizer and RNG checkpoints support continuation.
  The reference script's expert-only flag does not implement a parameter freeze.
- TiTO is the supplied [Linux executable](../../../terminus/bin/README.md),
  with its documented CPython 3.12 runtime requirements.

## Evaluation and assets

The [training manifest](../../../buffer/data/manifests/swedev_reference_1105.json)
contains the reference script's 1,105 unique SWE-Dev IDs. The
[evaluation manifest](../../../terminus/data/manifests/swe_verified_200.json)
contains the fixed 200 SWE-bench Verified instances from Appendix A.1.
Terminal-Bench 2.0 uses the full prepared dataset. Keep dataset versions,
selection procedures, seeds and run records when comparing results.

Exact paper reproduction still requires the historical source/checkpoint lineage,
training-data filtering, the Air-SFT and critic starting checkpoints, and benchmark
assets. The paper's PPO settings and the historical DPPO script are recorded
separately. Historical analysis inputs and summary-judge selection conventions
also require reconciliation. No summary judge is used as an extra training reward.

Config overlays include summary-loss masking, cross-segment discount, loss
normalization and context/history sweeps; see [ablations](../configs/ablations/README.md).
Buffer, training-curve and offline summary-quality analysis tools are included
in their corresponding components.

## Validation status

| Check | Status |
| --- | --- |
| CPU contracts and native slime role configuration | 44 project tests and 6 native tests passed before component regrouping |
| Current component layout, packaging and launch commands | Local path/storage/runtime checks, component wheel checks and launcher dry-runs passed; the eight launcher tests were rerun after the task-path fix |
| Task execution and compaction transport | A real Harbor Oracle task and a real Flash compacted collection returned reward 1.0 |
| Bounded parser after the memory-limit change | Local isolation checks passed; Linux enforcement and peak measurements remain pending |
| Real GLM optimizer updates, summary gradients and save/resume | Pending |
| Paper benchmark scores and minimum GPU/RAM requirements | Pending |

For a GPU run, check finite actor/critic losses, optimized summary tokens in a
compacted rollout, changing policy parameters and rollout weight versions,
checkpoint save, and one further resumed update. Loop iterations alone do not
establish these properties. CPU tests and argument parsing do not establish
end-to-end model training.
