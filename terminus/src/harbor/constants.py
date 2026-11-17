import os
from pathlib import Path

CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "harbor"
)
TASK_CACHE_DIR = CACHE_DIR / "tasks"
NOTIFICATIONS_PATH = CACHE_DIR / "notifications.json"
DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"
)
