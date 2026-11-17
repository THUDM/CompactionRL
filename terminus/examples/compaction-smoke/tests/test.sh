#!/bin/sh
mkdir -p /logs/verifier
python - <<'PY'
import json
from pathlib import Path
try:
    data=json.loads(Path('/app/result.json').read_text())
    ok=data=={'count':300,'sum':299*300//2}
except Exception:
    ok=False
Path('/logs/verifier/reward.txt').write_text('1' if ok else '0')
PY
