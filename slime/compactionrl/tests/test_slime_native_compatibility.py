"""Check native slime behavior at the CompactionRL integration boundaries."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).parents[3]


def test_native_ppo_and_dual_clip_keep_their_numerical_behavior():
    from slime.utils.ppo_utils import compute_policy_loss

    ratio = torch.tensor([1.5, 1.5, 0.5, 4.0])
    advantage = torch.tensor([1.0, -1.0, 1.0, -1.0])
    # Hand-evaluated PPO bounds, including the negative-advantage dual clip.
    expected = torch.tensor([-1.2, 1.5, -0.5, 4.0])
    loss, clip = compute_policy_loss(-ratio.log(), advantage, 0.2, 0.2)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(clip, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    dual, _ = compute_policy_loss(-ratio.log(), advantage, 0.2, 0.2, 2.0)
    torch.testing.assert_close(dual, torch.tensor([-1.2, 1.5, -0.5, 2.0]))


def test_native_custom_advantage_callback_still_takes_two_arguments():
    path = ROOT / "slime/slime/backends/megatron_utils/loss.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_advantages_and_returns"
    )
    calls = []

    def native_callback(args, data):
        calls.append(args.custom_advantage_function_path)
        data.update(advantages=[torch.tensor([0.75])], returns=[torch.tensor([1.0])])

    namespace = {
        "Namespace": SimpleNamespace,
        "RolloutBatch": dict,
        "torch": torch,
        "mpu": SimpleNamespace(is_pipeline_last_stage=lambda: True),
        "load_function": lambda _: native_callback,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    args = SimpleNamespace(
        custom_advantage_function_path="native.plugin",
        use_rollout_logprobs=True,
        kl_coef=0,
        use_opd=False,
        normalize_advantages=False,
    )
    data = dict(
        rollout_log_probs=[torch.tensor([-0.2])],
        rewards=[1.0],
        values=[torch.tensor([0.25])],
        loss_masks=[torch.tensor([1])],
        response_lengths=[1],
        total_lengths=[2],
    )
    namespace[node.name](args, data, is_policy=False)
    assert calls == ["native.plugin"]
    assert data["rewards"] == [1.0]
    torch.testing.assert_close(data["advantages"][0], torch.tensor([0.75]))


def test_slime_compaction_dispatch_uses_actor_and_critic_gae_without_custom_plugin(
    monkeypatch,
):
    import sys
    import types
    from slime.utils.ppo_utils import compute_compaction_advantages

    core = types.ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_context_parallel_world_size=lambda: 1)
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    cp = types.ModuleType("slime.backends.megatron_utils.cp_utils")
    cp.all_gather_with_cp = lambda x, *_: x
    cp.slice_log_prob_with_cp = lambda x, *_: x
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.cp_utils", cp)
    path = ROOT / "slime/slime/backends/megatron_utils/loss.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_advantages_and_returns"
    )
    namespace = {
        "Namespace": SimpleNamespace,
        "RolloutBatch": dict,
        "torch": torch,
        "mpu": SimpleNamespace(is_pipeline_last_stage=lambda: True),
        "compute_compaction_advantages": compute_compaction_advantages,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    args = SimpleNamespace(
        use_compaction_gae=True,
        custom_advantage_function_path=None,
        use_rollout_logprobs=True,
        kl_coef=0,
        use_opd=False,
        normalize_advantages=False,
        gamma=1,
        lambd=1,
        lambd_critic=1,
        use_length_adaptive_gae=True,
        length_adaptive_alpha=1.5,
        compaction_advantage_mode="gae",
    )
    data = dict(
        rollout_log_probs=[torch.zeros(3)],
        rewards=[1.0],
        values=[torch.zeros(3)],
        loss_masks=[torch.tensor([1, 0, 1])],
        total_lengths=[4],
        response_lengths=[3],
        metadata=[{"optimized_tokens": 2, "later_assistant_tokens": 1}],
    )
    namespace[node.name](args, data, is_policy=True)
    torch.testing.assert_close(data["advantages"][0], torch.tensor([4 / 9, 0, 2 / 3]))
    namespace[node.name](args, data, is_policy=False)
    torch.testing.assert_close(data["returns"][0], torch.tensor([1.0, 0.0, 1.0]))
    assert data["rewards"] == [1.0]
