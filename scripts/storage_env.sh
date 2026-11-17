#!/usr/bin/env bash
# Source before installing dependencies so build and package caches use a data volume.
COMPACTIONRL_SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export COMPACTIONRL_WORK_DIR="${COMPACTIONRL_WORK_DIR:-$COMPACTIONRL_SOURCE_ROOT/.runtime}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$COMPACTIONRL_SOURCE_ROOT/buffer/src${PYTHONPATH:+:$PYTHONPATH}"
COMPACTIONRL_STORAGE_PYTHON="${PYTHON:-python3}"
"$COMPACTIONRL_STORAGE_PYTHON" -B -c '
import sys
from compactionrl.runtime.storage import prepare_storage
prepare_storage(sys.argv[1], sys.argv[2])
' "$COMPACTIONRL_WORK_DIR" "$COMPACTIONRL_SOURCE_ROOT" || return 1
while IFS= read -r -d '' COMPACTIONRL_STORAGE_ASSIGNMENT; do
  export "$COMPACTIONRL_STORAGE_ASSIGNMENT"
done < <("$COMPACTIONRL_STORAGE_PYTHON" -B -c '
import sys
from compactionrl.runtime.storage import storage_environment
for key, value in storage_environment(sys.argv[1]).items():
    sys.stdout.buffer.write((key + "=" + value + "\0").encode())
' "$COMPACTIONRL_WORK_DIR")
# pip build plugins may invoke Git themselves. Trust only these project-created
# checkouts through subprocess environment; do not write a global Git config.
COMPACTIONRL_GIT_CONFIG_COUNT="${GIT_CONFIG_COUNT:-0}"
if [[ ! "$COMPACTIONRL_GIT_CONFIG_COUNT" =~ ^[0-9]+$ ]]; then
  echo "GIT_CONFIG_COUNT must be a nonnegative integer" >&2
  return 1
fi
for COMPACTIONRL_GIT_DIRECTORY in "$COMPACTIONRL_SOURCE_ROOT" "$COMPACTIONRL_SOURCE_ROOT/.deps/Megatron-LM"; do
  export "GIT_CONFIG_KEY_${COMPACTIONRL_GIT_CONFIG_COUNT}=safe.directory"
  export "GIT_CONFIG_VALUE_${COMPACTIONRL_GIT_CONFIG_COUNT}=$COMPACTIONRL_GIT_DIRECTORY"
  COMPACTIONRL_GIT_CONFIG_COUNT=$((COMPACTIONRL_GIT_CONFIG_COUNT + 1))
done
export GIT_CONFIG_COUNT="$COMPACTIONRL_GIT_CONFIG_COUNT"
# VCS file-finder plugins deliberately strip GIT_* variables. Git still honors
# XDG_CONFIG_HOME, so give those subprocesses a profile on the selected data volume.
"$COMPACTIONRL_STORAGE_PYTHON" -B -c '
import sys
from compactionrl.runtime.storage import prepare_git_profile
prepare_git_profile(sys.argv[1], sys.argv[2])
' "$COMPACTIONRL_SOURCE_ROOT" "$COMPACTIONRL_WORK_DIR" || return 1
unset COMPACTIONRL_STORAGE_ASSIGNMENT COMPACTIONRL_STORAGE_PYTHON COMPACTIONRL_SOURCE_ROOT
unset COMPACTIONRL_GIT_CONFIG_COUNT COMPACTIONRL_GIT_DIRECTORY
