# Paper-to-code map

This table locates the implementation of the paper methods and experiments. Source
revisions are documented in the [slime overlay](slime_overlay.md).

| Paper item | Implementation |
|---|---|
| Context budget / compaction trigger | `terminus/src/harbor/agents/terminus_2/terminus_2.py` |
| Terminus-KIRA tool actions | `terminus/src/harbor/agents/terminus_kira/terminus_kira.py` |
| Prompt rendering and original history | `terminus/src/harbor/llms/chat.py`, `lite_llm.py` |
| Rollout snapshots | `terminus/server/trial_worker.py` |
| Token preservation | `terminus/bin/tito-linux-x86_64` |
| Shared reward / N_after | `buffer/src/compactionrl/buffer/schema.py` |
| Batch delivery | `buffer/src/compactionrl/buffer/store.py`, `service.py` |
| Training metadata / loss-mask transport | `slime/slime/rollout/compaction_buffer.py` |
| Eq. 9 approximate cross-segment GAE | `slime/slime/utils/ppo_utils.py` |
| Token-normalized PPO | `slime/slime/backends/megatron_utils/loss.py`, `slime/slime/backends/megatron_utils/cp_utils.py`, `slime/slime/backends/megatron_utils/model.py` |
| Actor and critic updates | `slime/train_compactionrl.py`, `slime/slime/backends/megatron_utils/actor.py` |
| Value head | `slime/slime/backends/megatron_utils/model_provider.py` |
| GLM-4.7-Flash bridge | retained raw GLM conversion, plus mbridge HF loading registered under `slime/slime_plugins/mbridge` |
| Evaluation subset | `terminus/data/manifests/swe_verified_200.json` |
| Evaluation runs / result aggregation | `terminus/scripts/evaluate.py` |
| SUPO-style grouped baseline | `buffer/src/compactionrl/buffer/collector.py`, `slime/slime/utils/ppo_utils.py` |
| Compaction statistics / training curves | `buffer/scripts/analyze_buffer.py`, `slime/compactionrl/scripts/analyze_training.py` |
| Summary judge and paired bootstrap | `terminus/prompts/summary_judge.txt`, `terminus/scripts/summary_quality.py` |

See [reproduction](reproduction.md) for training settings and evaluation protocols.
The paper prompt text files are reference copies; runtime summary/resume text is
built in `terminus_2.py`, as described in the [implementation notes](reproduction.md#release-implementation).
