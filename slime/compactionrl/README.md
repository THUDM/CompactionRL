# CompactionRL training in slime

CompactionRL implements critic-based PPO with trainable context compaction in
[slime](https://github.com/THUDM/slime). The [paper-to-code map](docs/implementation.md)
locates token-normalized loss, cross-segment GAE, rollout conversion and model
updates. Source revisions and changes are recorded in the [overlay guide](docs/slime_overlay.md).

| Directory | Contents |
| --- | --- |
| `configs/` | Flash smoke, Flash/Air paper configurations and paper ablations |
| `scripts/` | Launcher, preflight, configuration merging, Megatron installer and training analysis |
| `docs/` | [Launch guide](docs/glm47_launch.md), [environment](docs/environment.md) and [reproduction](docs/reproduction.md) |
| `tests/` | Training and cross-component CPU checks |
| `patches/` | Changes to the pinned upstream framework |

After following the [installation instructions](../../README.md#getting-started),
run these commands from the repository root to inspect the small-run plan:

```bash
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --dry-run
bash scripts/smoke_train_glm47_flash.sh --model "$MODEL_PATH" --validate-only
bash scripts/train_glm47_flash.sh --help
```

The smoke target is one 8 × 80 GB node, 8k context and five update batches.
See the [launch guide](docs/glm47_launch.md) for training and checkpoint
continuation commands.

For paper training, `scripts/train_glm47_flash.sh` defaults to the Flash paper
configuration. Supply the model, explicit task-ID list, prepared task dataset and
Ray cluster options shown in the [launch guide](docs/glm47_launch.md#flash-paper-training).
The Air configuration specifies the 106B Air-SFT experiment. Model-specific
launch requirements are described in the same guide.
