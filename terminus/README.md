# Harbor / Terminus-KIRA for CompactionRL

This directory contains the trimmed Harbor fork used by CompactionRL. It retains
Terminus-KIRA, Terminus2, local Docker tasks, verification, and rollout export.
Only the Docker environment backend is included. The top-level
[README](../README.md) describes installation and the supported training path.
See [source manifest](../SOURCE_MANIFEST.json) for the selected commit and
release changes.

All execution-side materials are kept in this component:

| Directory | Contents |
|---|---|
| `src/harbor/`, `server/` | Terminus-KIRA agent, compaction, Docker execution, verification and HTTP worker service |
| `bin/` | [TiTO executable and requirements](bin/README.md); the launcher preserves its independent process |
| `prompts/` | Summary, resumed-context and summary-quality judge prompts |
| `examples/` | Small file and compaction tasks |
| `data/manifests/` | Fixed 200-instance SWE-bench Verified evaluation subset |
| `scripts/` | Task-list preparation, evaluation and summary-quality analysis |
| `tests/` | Harbor release, TiTO and analysis contracts |

Install this copy from the repository root:

```bash
pip install -e terminus
python terminus/scripts/prepare_tasks.py --help
python terminus/scripts/evaluate.py --help
```

Upstream: [laude-institute/harbor](https://github.com/laude-institute/harbor).
The upstream Apache-2.0 license is retained in [LICENSE](LICENSE).

## Upstream citation

```bibtex
@software{Harbor_Framework_Team_Harbor_2026,
  author = {{Harbor Framework Team}},
  title = {{Harbor: A framework for evaluating and optimizing agents and models in container environments}},
  url = {https://github.com/laude-institute/harbor},
  year = {2026}
}
```
