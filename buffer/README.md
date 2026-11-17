# CompactionRL buffer

This component contains the minimal persistent rollout buffer and Harbor
collector. It keeps one rollout per prompt on the PPO path, computes subsequent
optimized-token counts before batching, and preserves original generation IDs,
log probabilities and masks.

| Directory | Contents |
|---|---|
| `src/compactionrl/buffer/` | Schema validation, SQLite FIFO, delivery leases, collector and HTTP service |
| `src/compactionrl/runtime/` | Shared data-volume/cache configuration used by the integrated launcher |
| `docs/` | [Buffer and token protocol](docs/buffer_protocol.md) |
| `data/manifests/` | The reference script's 1,105 training task IDs |
| `scripts/` | Buffer database analysis |
| `tests/` | Storage-path tests; queue/collector integration tests are in slime's CompactionRL suite |

Install from the repository root:

```bash
source scripts/storage_env.sh
pip install -e buffer
```

The Python package and console command remain `compactionrl` and
`compactionrl-buffer`. This directory is the installable project; there is no
separate root Python package to install.

The [integrated training launcher](../slime/compactionrl/README.md) supplies the
resolved config, model and service endpoints. For a manual service start:

```bash
compactionrl-buffer --config /path/to/resolved-runtime.yaml \
  --database "$COMPACTIONRL_WORK_DIR/buffer.sqlite3" --port 32000
python buffer/scripts/analyze_buffer.py --help
```

Completed database rows remain for audit and deduplication; rotate the database
between runs. Delivery acknowledgement is not an optimizer transaction.
See [validation status](../slime/compactionrl/docs/reproduction.md#validation-status) and the project [license](LICENSE).
