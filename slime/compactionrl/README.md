# CompactionRL training in slime

The parent `slime/` directory retains the complete official upstream framework
at `41dc3b6d21d3c75b212965077a1cc4117932f06d`. Its README, model support, tools,
backends and examples remain intact.

CompactionRL's GAE and optional stepwise advantage are implemented directly in
[ppo_utils.py](../slime/utils/ppo_utils.py); buffer conversion is in
[compaction_buffer.py](../slime/rollout/compaction_buffer.py). The integrated
driver is [train_compactionrl.py](../train_compactionrl.py).

This directory keeps the project-specific training materials together:

| Directory | Contents |
|---|---|
| `configs/` | GLM-4.7-Flash smoke/reference/paper, GLM-4.5-Air and ablations |
| `scripts/` | Integrated launcher, preflight, config merging, Megatron installer and training analysis |
| `patches/` | The reviewable 17-file slime code overlay |
| `docs/` | [Launch recipes](docs/glm47_launch.md), [overlay](docs/slime_overlay.md), [stepwise advantage](docs/stepwise_advantage.md) and environment notes |
| `tests/` | Training and cross-component CPU integration contracts |

From the repository root, install the components and compatible GPU stack as
described in the [main README](../../README.md). The unified wrappers remain:

```bash
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --dry-run
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --validate-only
bash scripts/train_glm47_flash.sh --help
```

The launcher resolves the buffer, Terminus server and TiTO executable from this
checkout. It accepts model weights and benchmark task environments as external
assets. Token GAE is selected with `--use-compaction-gae`; native slime defaults
and its two-argument custom advantage callbacks retain their upstream behavior.

The single-machine smoke target is 8 × 80 GB GPUs; minimum resources and five
optimizer updates remain unverified. See [validation status](docs/reproduction.md#validation-status) before running.
