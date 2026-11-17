#!/bin/sh
set -eu
mkdir -p /logs/verifier
python - <<'PYCODE'
import json
from pathlib import Path
try:
    ok = json.loads(Path('/app/result.json').read_text()) == {'status':'ready','total':55}
except Exception:
    ok = False
Path('/logs/verifier/reward.txt').write_text('1' if ok else '0')
PYCODE
