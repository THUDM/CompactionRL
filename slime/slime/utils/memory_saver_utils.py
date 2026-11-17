"""Resolve the installed memory-saver preload without assuming a wheel suffix."""

import os
from pathlib import Path


def memory_saver_preload():
    import torch_memory_saver

    if hasattr(torch_memory_saver, "configure_subprocess"):
        with torch_memory_saver.configure_subprocess():
            return os.environ["LD_PRELOAD"]
    # Compatibility with older wheels that predate the public helper.
    path = Path(torch_memory_saver.__file__).parent.parent / "torch_memory_saver_hook_mode_preload.abi3.so"
    if not path.is_file():
        raise FileNotFoundError(f"Memory-saver preload library is missing: {path}")
    existing = os.environ.get("LD_PRELOAD", "")
    return f"{path}:{existing}" if existing else str(path)
