"""Exercise the same TiTO payload that is embedded in the distributed executable."""

import hashlib
import json
import marshal
from pathlib import Path
import struct
import sys
import types

ROOT = Path(__file__).parent
path = ROOT / "terminus/bin/tito-linux-x86_64"
data = path.read_bytes()
metadata = json.loads(path.with_suffix(".json").read_text())
assert hashlib.sha256(data).hexdigest() == metadata["binary_sha256"]
marker = b"COMPACTIONRL_TITO_PY312\0"
offset = data.rfind(marker) + len(marker)
assert offset >= len(marker)
length = struct.unpack(">I", data[offset : offset + 4])[0]
payload = data[offset + 4 :]
assert len(payload) == length
assert hashlib.sha256(payload).hexdigest() == metadata["payload_sha256"]
module = types.ModuleType("tito_runtime_test")
module.__file__ = "<distributed-tito-payload>"
sys.modules[module.__name__] = module
exec(marshal.loads(payload), module.__dict__)
