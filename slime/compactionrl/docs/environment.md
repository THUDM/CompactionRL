# Runtime environment

Use Linux, Python 3.12, NVIDIA GPUs and Docker with the Compose plugin.
The Flash smoke recipe targets one 8×80GB GPU node. CPU offload also needs
substantial host RAM; inspect the container memory limit as well as host memory.

These versions were used during integration checks:

| Dependency | Version / revision |
| --- | --- |
| PyTorch | 2.9.1+cu129 |
| SGLang | 0.5.10 |
| Transformers | 5.3.0 |
| Ray | 2.55.1 |
| Transformer Engine | 2.16.0+f41a44f4 |
| mbridge | 0.15.1 |
| torch-memory-saver | 0.0.9.post1+slime |
| uv-build | 0.8.24 |
| Megatron-LM / Core | public core_v0.15.3, 309ffca6a40553362d44fd17efc56b772fa1aa44 |

The Flash profiles use mbridge for HF loading and raw GLM conversion for rollout
weight updates. The installer places the pinned public Megatron checkout under
`.deps/Megatron-LM`; the launcher discovers it or accepts `--megatron-path`.
Optional NVIDIA Bridge mode requires a matching Bridge/Core API combination.
The profiles disable ordinary and piecewise CUDA Graphs, trading potential
inference speed for a simpler integration configuration.

The launcher detects CUDA compute capabilities through `nvidia-smi` and forwards
`TORCH_CUDA_ARCH_LIST` to Ray workers, including CPU management workers that
import Megatron CUDA extensions. Set this explicitly for a cluster that needs
specific compilation targets.

Place the checkout, runtime/cache paths, outputs and Docker data root on a mounted
data volume. The launcher disables model disk backup, redirects runtime caches
and Ray spill files, and checks storage destinations before normal launch.
Source `scripts/storage_env.sh` before package installation to redirect build
caches and provide command-scoped Git trust settings on FUSE mounts. See
[launch requirements](glm47_launch.md) for the short Ray socket path and Docker
checks. Existing system/home Git profiles are preserved.

The buffer and CPU tests can run without CUDA. Numerical tests require a matching
PyTorch installation. The [preflight script](../scripts/preflight.py) reports
libraries, model shards, GPUs, Docker and host/container memory. See
[checks](reproduction.md#checks) for preflight and training inspection.
