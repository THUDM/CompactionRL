import asyncio
import importlib.util
from pathlib import Path

import httpx
import pytest

from tito_runtime_test import create_app


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_tito_http_multiturn_summary_export_and_release():
    requests = []

    def generate(request):
        import json

        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "text": "A",
                "meta_info": {
                    "output_token_logprobs": [[-0.1, 65]],
                    "weight_version": "v1",
                },
            },
        )

    async def run():
        app = create_app(
            Tokenizer(), "http://router", transport=httpx.MockTransport(generate)
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://tito"
            ) as client:
                first = await client.post(
                    "/generate", json={"main_key": "r", "text": "P"}
                )
                assert first.status_code == 200
                prompt = "PA\nYour job is to create a detailed summary"
                second = await client.post(
                    "/generate", json={"main_key": "r", "text": prompt}
                )
                assert second.status_code == 200
                exported = (await client.post("/tokens", json={"main_key": "r"})).json()
                assert exported["input_ids"] == [ord(c) for c in prompt + "A"]
                assert sum(exported["loss_mask"]) == 2
                assert [s["summary"] for s in exported["generated_spans"]] == [
                    False,
                    True,
                ]
                assert exported["weight_versions"] == ["v1", "v1"]
                assert (
                    await client.post(
                        "/generate", json={"main_key": "r", "text": "diverged"}
                    )
                ).status_code == 409
                assert len(requests) == 2
                assert requests[1]["input_ids"][:2] == [80, 65]
                await client.post("/release", json={"main_key": "r"})
                assert (
                    await client.post("/tokens", json={"main_key": "r"})
                ).status_code == 404
                state = (await client.get("/health")).json()
                assert state["traces"] == state["key_locks"] == 0

    asyncio.run(run())


def test_tito_uncertain_http_failure_is_not_retried():
    calls = []

    def fail(request):
        calls.append(request)
        return httpx.Response(503)

    async def run():
        app = create_app(
            Tokenizer(), "http://router", transport=httpx.MockTransport(fail)
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://tito"
            ) as client:
                payload = {"main_key": "r", "text": "P"}
                assert (await client.post("/generate", json=payload)).status_code == 502
                assert (await client.post("/generate", json=payload)).status_code == 409
                assert len(calls) == 1

    asyncio.run(run())


def load_analysis():
    path = Path(__file__).parents[2] / "terminus/scripts/summary_quality.py"
    spec = importlib.util.spec_from_file_location("summary_quality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_judge_averages_repeats_then_summaries_then_paired_tasks():
    module = load_analysis()

    def row(task, method, summary, value):
        return {
            "instance_id": task,
            "method": method,
            "summary_id": summary,
            "judge": {
                "faithfulness": {key: {"score": value} for key in module.DIMENSIONS},
                "usefulness": {
                    "U_continuation_efficiency_and_progress": {"score": value}
                },
            },
        }

    rows = [
        row("a", "left", "a", 0),
        row("a", "right", "a", 4),
        row("a", "right", "a", 4),
        row("a", "right", "b", 0),
        row("b", "left", "a", 1),
        row("b", "right", "a", 2),
    ]
    result = module.aggregate(rows, "left", "right", repeats=500)
    assert result["paired_tasks"] == 2
    assert result["mean_difference"] == pytest.approx(1.5)
    assert result["bootstrap_95_percent_interval"] == [1, 2]
    rows[0]["judge"]["faithfulness"][module.DIMENSIONS[0]]["score"] = 6
    with pytest.raises(ValueError):
        module.aggregate(rows, "left", "right")
