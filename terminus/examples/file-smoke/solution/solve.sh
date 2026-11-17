#!/bin/sh
set -eu
python - <<'PYCODE'
import json
from pathlib import Path
Path('/app/result.json').write_text(json.dumps({'status':'ready','total':sum(range(1,11))}))
PYCODE
