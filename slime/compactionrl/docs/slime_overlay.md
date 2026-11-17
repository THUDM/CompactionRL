# Official slime baseline and CompactionRL overlay

`slime` contains the complete official THUDM/slime tree at
`41dc3b6d21d3c75b212965077a1cc4117932f06d`, plus the project overlay. This was the
last official `main` parent merged into the supplied CompactionRL development
branch. Its Git tree was matched against the public GitHub source archive.

All 469 upstream file/link entries are retained. Native model bridges, reward
functions, examples, FSDP support, tools, tests, documentation and installation
metadata are included. The upstream README and dependency list are unchanged.
CompactionRL GAE is implemented in `slime/utils/ppo_utils.py`, buffer rollout in
`slime/rollout/compaction_buffer.py`, and its managed training entry point in
`train_compactionrl.py`. The minimal buffer is packaged from `buffer/` as
`compactionrl`; TiTO is supplied under `terminus/bin/`.

## Overlay scope

The exact 17-file list is in [SOURCE_MANIFEST.json](../../../SOURCE_MANIFEST.json). The reviewable patch is
[`slime/compactionrl/patches/slime_compactionrl.patch`](../patches/slime_compactionrl.patch).

| Change | Purpose |
|---|---|
| Sample fields, DP partitioning and packed data | Preserve segment identity, original masks and subsequent-token counts |
| PPO/VAPO arguments and built-in CompactionRL GAE | Support the supplied recipe with `--use-compaction-gae`, preserving native algorithms and callbacks |
| Optional stepwise critic masks | Transport end-of-step supervision into the existing packed loss reducer |
| Critic initialization and HF/checkpoint import | Preserve the scalar value head and chained optimizer state |
| Model provider and Bridge recomputation | Honor explicit activation recomputation settings |
| Actor memory-saver startup | Resolve the installed public preload library and explicitly disable model disk backup |
| Metrics and disabled profiler handling | Retain compaction diagnostics and tolerate optional absent parser fields |

The original CompactionRL development snapshot is
`bc092af4572c2db5c8459e9e9e890169e2ca1120`, with the recorded working changes.
Its old private buffer rollout and duplicate advantage plugin are replaced by the
project adapters. The patch also includes fixes made during integration.

Useful stepwise GAE is retained in slime with an explicit optional
configuration; see [stepwise advantage](stepwise_advantage.md). Duplicate GAE
paths, unused checker/browser flags and automatic script uploads were removed
from the overlay. The official default rollout, algorithm choices and native
logging remain intact.

## Reconstruct and check

Use a short working directory on a mounted data volume. For example, export
`COMPACTIONRL_WORK_DIR` and source `scripts/storage_env.sh` before downloads.

```bash
SLIME_BASE="$COMPACTIONRL_WORK_DIR/slime-base"
git clone https://github.com/THUDM/slime.git "$SLIME_BASE"
git -c safe.directory="$SLIME_BASE" -C "$SLIME_BASE" checkout 41dc3b6d21d3c75b212965077a1cc4117932f06d
git -c safe.directory="$SLIME_BASE" -C "$SLIME_BASE" apply --check "$PWD/slime/compactionrl/patches/slime_compactionrl.patch"
git -c safe.directory="$SLIME_BASE" -C "$SLIME_BASE" apply "$PWD/slime/compactionrl/patches/slime_compactionrl.patch"
python -B scripts/check_release.py
```

To verify preservation against an unmodified checkout or extracted archive:

```bash
python -B scripts/check_release.py --slime-upstream /path/to/unmodified-slime
```

The patch was applied to a fresh copy of the pinned official tree and all 472
resulting framework file/link entries matched the vendored tree exactly.
The separate `slime/compactionrl/` directory holds project configs, documentation,
scripts, tests and the patch; it is excluded from framework reconstruction checks.
See [checks](reproduction.md#checks) for preflight and training inspection.
