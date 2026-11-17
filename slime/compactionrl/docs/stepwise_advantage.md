# Optional stepwise advantage

The development branch's stepwise implementation is retained as an optional
CompactionRL extension in slime's `slime/utils/ppo_utils.py`. The paper and both
default GLM profiles use token GAE. Native slime algorithms and custom advantage
callbacks keep their upstream behavior.

Enable the optional variant by merging its overlay into a GLM configuration:

```bash
python slime/compactionrl/scripts/merge_config.py slime/compactionrl/configs/glm47_flash/smoke.yaml \
  slime/compactionrl/configs/ablations/stepwise_advantage.yaml \
  --output "$COMPACTIONRL_WORK_DIR/stepwise.yaml"
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" \
  --config "$COMPACTIONRL_WORK_DIR/stepwise.yaml" --validate-only
```

Consecutive generated tokens form a step. Observation runs do not advance the
GAE recurrence. `stepwise_advantage_type: last_token` uses the last predicted
value in each generated run and supervises the critic only at that run's end.
`step_average` uses its mean predicted value and supervises all generated tokens.
The actor receives the step's advantage at every generated token. Optional
`critic_stepwise_max_chunk_size` splits long runs; values at or below zero mean
no splitting. Summary tokens remain generated tokens.

The historical adaptive lambda counts all response mask chunks, including
observation chunks. Token GAE instead counts optimized tokens. The stepwise
extension preserves the historical chunk-count convention and applies
`(gamma * lambda) ** N_after` using the buffer's subsequent optimized-token count.
These two time scales make it an experimental variant, separate from the paper's
default approximation. Invalid discounts are rejected explicitly.

Unlike the old duplicated framework paths, full-response gathering and original
mask validation are shared with the strict CompactionRL adapter. Observation
advantages/returns are zeroed because those positions have no loss. End-of-step
critic masks are explicitly transported into slime's microbatches and used in
its loss normalization; the development code did not request that field in the
trainer's batch key list.

CPU tests cover both value reductions, subsequent-token discounting and critic
mask/chunk boundaries. Distributed GPU training and numerical equivalence to
historical runs have not been verified.
