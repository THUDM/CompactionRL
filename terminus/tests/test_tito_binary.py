import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() != "x86_64",
    reason="Linux x86_64 executable",
)
def test_executable_multiturn_contract_and_uncertain_failure(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "prompt": 1, "answer": 2, "summary": 3}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    model = tmp_path / "tokenizer"
    model.mkdir()
    tokenizer.save(str(model / "tokenizer.json"))
    (model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>"})
    )
    calls = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            fail = len(calls) == 3
            self.send_response(502 if fail else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "text": "unused",
                        "meta_info": {
                            "finish_reason": {"type": "length"},
                            "output_token_logprobs": [
                                [-0.2, 2 if len(calls) == 1 else 3, None]
                            ],
                            "weight_version": f"v{len(calls)}",
                        },
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    env = {
        **os.environ,
        "COMPACTIONRL_TITO_PYTHON": sys.executable,
        "HF_HUB_OFFLINE": "1",
    }
    log = (tmp_path / "tito.log").open("w")
    process = subprocess.Popen(
        [
            str(ROOT / "terminus/bin/tito-linux-x86_64"),
            "--model",
            str(model),
            "--router-url",
            f"http://127.0.0.1:{upstream.server_port}",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2) as client:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail((tmp_path / "tito.log").read_text())
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
            else:
                pytest.fail("TiTO executable did not become ready")
            first = client.post("/generate", json={"main_key": "k", "text": "prompt"})
            first.raise_for_status()
            history = "prompt" + first.json()["text"]
            second = client.post(
                "/generate",
                json={
                    "main_key": "k",
                    "text": history + " Your job is to create a detailed summary",
                },
            )
            second.raise_for_status()
            info = client.post("/tokens", json={"main_key": "k"}).json()
            assert info["weight_versions"] == ["v1", "v2"]
            assert info["input_ids"][:2] == [1, 2]
            assert info["loss_mask"][0] == 0 and sum(info["loss_mask"]) == 2
            assert info["generated_spans"][1]["summary"] is True
            assert (
                client.post(
                    "/generate", json={"main_key": "k", "text": "divergent"}
                ).status_code
                == 409
            )
            assert len(calls) == 2
            assert (
                client.post(
                    "/generate", json={"main_key": "bad", "text": "prompt"}
                ).status_code
                == 502
            )
            assert (
                client.post(
                    "/generate", json={"main_key": "bad", "text": "prompt"}
                ).status_code
                == 409
            )
            assert len(calls) == 3
            assert client.post("/release", json={"main_key": "k"}).status_code == 200
            assert client.post("/tokens", json={"main_key": "k"}).status_code == 404
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()
        upstream.shutdown()
        upstream.server_close()
