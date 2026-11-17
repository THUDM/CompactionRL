import concurrent.futures

import pytest
import torch

from compactionrl.buffer.schema import finalize_rollout, validate_tokens
from compactionrl.buffer.store import Buffer
from tito_runtime_test import Trace
from slime.utils.ppo_utils import masked_segment_gae
from slime.rollout.compaction_buffer import record_to_sample


def record(uid, mask=(0, 1, 0, 1)):
    return dict(
        uid=uid,
        instance_id="task",
        reward=1.0,
        messages=[{"role": "assistant", "content": "done"}],
        metadata={},
        token_info={
            "input_ids": list(range(len(mask))),
            "loss_mask": list(mask),
            "logprobs": [-0.2] * len(mask),
            "weight_versions": ["0"],
        },
    )


def test_masked_counts_and_training_transport():
    records = finalize_rollout([record("a"), record("b", (0, 0, 1))], "rollout")
    assert [r["metadata"]["later_assistant_tokens"] for r in records] == [1, 0]
    sample = record_to_sample(records[0], 0)
    assert sample.train_metadata["later_assistant_tokens"] == 1
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 0, 1]
    assert sample.rollout_log_probs == [-0.2] * 3


@pytest.mark.parametrize(
    "field,value",
    [("loss_mask", [0, 0, 0, 0]), ("logprobs", [float("nan")] * 4), ("input_ids", [1])],
)
def test_invalid_tokens_fail(field, value):
    r = record("bad")
    r["token_info"][field] = value
    with pytest.raises(ValueError):
        validate_tokens(r["token_info"])


def test_shared_reward_and_duplicate_guard():
    a, b = record("a"), record("b")
    b["reward"] = 0
    with pytest.raises(ValueError):
        finalize_rollout([a, b], "r")
    with pytest.raises(ValueError):
        finalize_rollout([a, a], "r")


def test_gae_matches_hand_computed_terminal_distance():
    values = torch.zeros(4)
    mask = torch.tensor([1, 0, 0, 1])
    adv, ret = masked_segment_gae(values, mask, 1.0, 3, alpha=1.5)
    lam = 2 / 3
    torch.testing.assert_close(adv, torch.tensor([lam**4, 0, 0, lam**3]))
    torch.testing.assert_close(adv, ret)
    local, _ = masked_segment_gae(values, mask, 1.0, 3, alpha=1.5, enabled=False)
    torch.testing.assert_close(local, torch.tensor([lam, 0, 0, 1.0]))


def test_gae_nonzero_values_and_terminal_boundary():
    adv, ret = masked_segment_gae(
        torch.tensor([0.2, 999.0, 0.4]), torch.tensor([1, 0, 1]), 1, 0, lambd=1
    )
    torch.testing.assert_close(adv, torch.tensor([0.8, 0, 0.6]))
    torch.testing.assert_close(ret, torch.tensor([1.0, 0, 1.0]))


def test_queue_idempotence_and_atomic_capacity(tmp_path):
    b = Buffer(tmp_path / "q.sqlite", capacity=2)
    assert b.put([record("a"), record("b")]) == 2
    assert b.put([record("a")]) == 0
    with pytest.raises(OverflowError):
        b.put([record("c")])
    bad = record("a")
    bad["reward"] = 0
    with pytest.raises(ValueError):
        b.put([bad])
    lease, batch = b.take(2)
    assert [r["uid"] for r in batch] == ["a", "b"]
    assert b.take(1) == (None, [])
    assert b.ack(lease) == 2
    assert b.ack(lease) == 0
    assert b.put([record("c")]) == 1


def test_concurrent_readers_never_share_a_live_lease(tmp_path):
    b = Buffer(tmp_path / "q.sqlite", capacity=8)
    b.put([record(str(i)) for i in range(8)])
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        batches = list(pool.map(lambda _: b.take(2), range(4)))
    ids = [r["uid"] for _, rows in batches for r in rows]
    assert len(ids) == len(set(ids)) == 8


def test_lease_recovery(tmp_path, monkeypatch):
    import compactionrl.buffer.store as store

    b = Buffer(tmp_path / "q.sqlite", lease_seconds=1)
    b.put([record("a")])
    old, _ = b.take(1)
    now = store.time.time()
    monkeypatch.setattr(store.time, "time", lambda: now + 2)
    new, rows = b.take(1)
    assert new != old and rows[0]["uid"] == "a"
    assert b.ack(old) == 0


def test_tito_preserves_generated_ids_without_reencoding():
    t = Trace()
    prompt, n = t.prepare("prompt", lambda x: [11, 12])
    t.complete(prompt, n, [900, 901], [-0.1, -0.2], "promptanswer", "v1")
    seen = []

    def encode(x):
        seen.append(x)
        return [30]

    prompt, n = t.prepare("promptanswer observation", encode)
    assert seen == [" observation"]
    assert prompt == [11, 12, 900, 901, 30]
    assert t.export()["loss_mask"] == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        t.prepare("changed prefix", encode)


def test_uncertain_generation_cannot_be_retried():
    t = Trace(failed=True)
    with pytest.raises(ValueError):
        t.prepare("x", lambda x: [1])


def test_critic_repeated_epochs_do_not_replace_scalar_rewards(monkeypatch):
    import sys
    import types
    from types import SimpleNamespace
    from slime.utils.ppo_utils import compute_compaction_advantages as estimate

    core = types.ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_context_parallel_world_size=lambda: 1)
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    cp = types.ModuleType("slime.backends.megatron_utils.cp_utils")
    cp.all_gather_with_cp = lambda x, *_: x
    cp.slice_log_prob_with_cp = lambda x, *_: x
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.cp_utils", cp)
    args = SimpleNamespace(
        kl_coef=0,
        gamma=1,
        lambd=1,
        lambd_critic=1,
        use_length_adaptive_gae=True,
        length_adaptive_alpha=1.5,
        compaction_advantage_mode="gae",
    )
    data = dict(
        values=[torch.tensor([0.2, 0.3])],
        loss_masks=[torch.tensor([1, 1])],
        total_lengths=[3],
        response_lengths=[2],
        rewards=[1.0],
        metadata=[{"optimized_tokens": 2, "later_assistant_tokens": 0}],
    )
    estimate(args, data, False)
    first = data["returns"][0].clone()
    estimate(args, data, False)
    assert data["rewards"] == [1.0]
    torch.testing.assert_close(data["returns"][0], first)
    assert data["token_rewards"][0].tolist() == [0, 1]


def test_group_baseline_normalizes_rollouts_not_segments():
    from compactionrl.buffer.collector import assign_group_advantages

    a = finalize_rollout([record("a"), record("a2")], "ra")
    r = record("b")
    r["reward"] = 0
    b = finalize_rollout([r], "rb")
    result = assign_group_advantages([a, b])
    advantages = [r["metadata"]["group_advantage"] for r in result]
    assert advantages[0] == advantages[1]
    assert advantages[0] == pytest.approx(-advantages[2])
    assert len({r["metadata"]["group_id"] for r in result}) == 1


def test_harbor_stream_to_finalized_segments():
    import asyncio
    import httpx
    from compactionrl.buffer.collector import collect

    def handler(request):
        if request.url.path == "/run_worker":
            data = {"reward": 1, "split_rollout_results": [record("a"), record("b")]}
            return httpx.Response(
                200,
                text='{"type":"heartbeat"}\n'
                + __import__("json").dumps({"type": "result", "data": data})
                + "\n",
            )
        if request.url.path == "/tokens":
            return httpx.Response(200, json=record("a")["token_info"])
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await collect(
                client,
                {
                    "harbor_url": "http://harbor",
                    "generation_url": "http://tito",
                    "agent": {},
                },
                "model",
                "task",
            )

    records = asyncio.run(run())
    assert [r["metadata"]["later_assistant_tokens"] for r in records] == [2, 0]


def test_real_slime_dp_partition_preserves_segment_metadata():
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    # Execute the actual pure partition method without starting Ray/GPU actors.
    source = Path(__file__).parents[3] / "slime/slime/ray/rollout.py"
    tree = ast.parse(source.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_split_train_data_by_dp"
    )
    namespace = {"Box": lambda x: x, "ray": SimpleNamespace(put=lambda x: x)}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    data = {
        "tokens": [[0, 1]] * 4,
        "metadata": [{"later_assistant_tokens": n} for n in [7, 3, 0, 0]],
    }
    result = namespace[method.name](
        SimpleNamespace(args=SimpleNamespace(balance_data=False)), data, 2
    )
    assert [m["later_assistant_tokens"] for m in result[0]["metadata"]] == [7, 0]
    assert [m["later_assistant_tokens"] for m in result[1]["metadata"]] == [3, 0]


def test_summary_tokens_receive_nonzero_gradient_without_prefix_gradient():
    from tito_runtime_test import Trace

    t = Trace()
    t.complete([10, 11], 2, [20, 21], [-0.1, -0.2], "prompt-summary", "v0", True)
    r = record("s")
    r["token_info"] = t.export()
    r = finalize_rollout([r], "r")[0]
    assert r["metadata"]["summary_optimized_tokens"] == 2
    logp = torch.tensor([-0.1] * 4, requires_grad=True)
    mask = torch.tensor(t.mask)
    adv, _ = masked_segment_gae(torch.zeros(4), mask, 1, 0)
    loss = -(logp * adv * mask).sum() / mask.sum()
    loss.backward()
    assert logp.grad[:2].tolist() == [0, 0]
    assert (logp.grad[2:].abs() > 0).all()
