# TiTO executable

`tito-linux-x86_64` runs the token-in/token-out process used by Harbor. The runtime
implementation is distributed in the executable rather than as a separate source
package. Its SHA256 and payload provenance are in `tito-linux-x86_64.json`.

This release is an ELF launcher with an embedded CPython bytecode payload. It
requires **Linux x86_64, glibc 2.34 or newer, and the project's CPython 3.12
environment** with FastAPI, Uvicorn, HTTPX, Transformers and Tokenizers installed.
It is not a standalone Rust build. The historical launcher's Rust executable was
unavailable on the preparation host, so this packages the verified release TiTO
implementation instead. No model weights or third-party libraries are bundled.

The GLM launcher starts it automatically and selects its current interpreter via
`COMPACTIONRL_TITO_PYTHON`. For a manual start, source the storage environment first:

```bash
source scripts/storage_env.sh
export COMPACTIONRL_TITO_PYTHON="$(command -v python)"
terminus/bin/tito-linux-x86_64 --model "$MODEL_PATH" \
  --router-url http://127.0.0.1:31000 --host 127.0.0.1 --port 31200
```

Use `--tito-binary /path/to/compatible-executable` with the GLM launcher to override
the binary. A replacement must support the same CLI and protocol. `/generate`
preserves original generated IDs/logprobs; `/tokens` exports aligned IDs, masks,
weight versions and generation spans; `/release` releases completed keys.
Prefix divergence or uncertain generation fails without retrying that key.

The executable's CPU integration test checks actual process startup, multiple
turns, summary-token metadata, weight versions, prefix divergence, uncertain HTTP
failure and key release. This test uses a tiny tokenizer and mocked generation
backend; it does not verify a GPU optimizer update.
