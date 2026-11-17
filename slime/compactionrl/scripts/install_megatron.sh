#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
MEGATRON_DIR="$ROOT/.deps/Megatron-LM"
MEGATRON_COMMIT=309ffca6a40553362d44fd17efc56b772fa1aa44
MEGATRON_TAG=core_v0.15.3
PYTHON_BIN="${PYTHON:-python3}"
source "$ROOT/scripts/storage_env.sh"

"$PYTHON_BIN" -B -c 'import sys; from compactionrl.runtime.storage import require_data_filesystem; assert sys.prefix != sys.base_prefix, "Activate the project virtual environment first"; require_data_filesystem(sys.prefix)'
mkdir -p "$ROOT/.deps"
if [[ ! -d "$MEGATRON_DIR" ]]; then
  git clone --depth 1 --branch "$MEGATRON_TAG" \
    https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_DIR"
fi
if [[ -d "$MEGATRON_DIR/.git" ]]; then
  ACTUAL_COMMIT="$(git -c safe.directory="$MEGATRON_DIR" -C "$MEGATRON_DIR" rev-parse HEAD)"
  [[ "$ACTUAL_COMMIT" == "$MEGATRON_COMMIT" ]] || {
    echo "Existing Megatron checkout has a different revision: $ACTUAL_COMMIT" >&2
    exit 1
  }
else
  "$PYTHON_BIN" - "$MEGATRON_DIR" "$MEGATRON_COMMIT" <<'PY'
import json, sys
from pathlib import Path
source = Path(sys.argv[1]) / '.compactionrl-source.json'
assert source.is_file(), 'Use the pinned Git checkout, or a verified source archive with its provenance marker'
assert json.loads(source.read_text())['commit'] == sys.argv[2]
PY
fi
"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$MEGATRON_DIR"
echo "Megatron installed from $MEGATRON_DIR ($MEGATRON_COMMIT)"
