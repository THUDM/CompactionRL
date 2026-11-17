import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
spec = importlib.util.spec_from_file_location(
    "launcher", ROOT / "slime/compactionrl/scripts/train_glm47_flash.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def make_args(*extras):
    return launcher.parser().parse_args(["--model", "/models/flash", *extras])


def test_smoke_uses_only_checkout_services_and_consistent_ports(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/obsolete-private-checkout")
    plan = launcher.build_plan(
        make_args(
            "--output",
            str(tmp_path / "run"),
            "--tito-port",
            "32101",
            "--buffer-port",
            "32102",
            "--router-port",
            "32103",
            "--harbor-port",
            "32104",
        )
    )
    assert plan["config"]["generation_url"].endswith(":32101")
    assert plan["config"]["harbor_url"].endswith(":32104")
    assert plan["config"]["training"]["rollout_buffer_url"].endswith(":32102")
    assert "http://127.0.0.1:32103" in plan["services"]["tito"]
    router = plan["services"]["router"]
    assert router[1:3] == ["-m", "sglang_router.launch_router"]
    assert router[router.index("--port") + 1] == "32103"
    assert "--disable-retries" in router
    assert plan["train"][1] == str(ROOT / "slime/train_compactionrl.py")
    dataset = Path(plan["env"]["DATASET_PATH"])
    assert dataset == ROOT / "terminus/examples"
    assert all((dataset / task / "task.toml").is_file() for task in plan["config"]["task_ids"])
    assert plan["config"]["training"]["use_compaction_gae"] is True
    assert "--custom-advantage-function-path" not in plan["train"]
    assert "/obsolete-private-checkout" not in plan["env"]["PYTHONPATH"]
    assert all(str(ROOT) in path for path in plan["env"]["PYTHONPATH"].split(":"))
    assert plan["config"]["training"]["num_rollout"] == 5
    assert plan["config"]["training"]["clip_type"] == "default"
    assert not (tmp_path / "run").exists()


def test_reference_preserves_recipe_and_expands_list_arguments(tmp_path):
    args = make_args(
        "--config",
        str(ROOT / "slime/compactionrl/configs/glm47_flash/reference.yaml"),
        "--dataset",
        str(tmp_path),
        "--ray-address",
        "10.0.0.1:6379",
        "--host",
        "10.0.0.1",
        "--bind-host",
        "0.0.0.0",
        "--steps",
        "10",
    )
    plan = launcher.build_plan(args)
    cfg = plan["config"]["training"]
    assert (cfg["advantage_estimator"], cfg["clip_type"], cfg["eps_clip_high"]) == (
        "vapo",
        "dppo",
        3.0,
    )
    assert (
        cfg["actor_num_nodes"] * cfg["actor_num_gpus_per_node"]
        + cfg["rollout_num_gpus"]
        == 80
    )
    assert (
        cfg["tensor_model_parallel_size"],
        cfg["pipeline_model_parallel_size"],
        cfg["context_parallel_size"],
        cfg["expert_model_parallel_size"],
    ) == (4, 2, 2, 8)
    i = plan["train"].index("--sglang-cuda-graph-bs")
    assert plan["train"][i + 1 : i + 5] == ["1", "2", "4", "8"]
    assert len(plan["config"]["task_ids"]) == 1105
    assert cfg["num_rollout"] == 10
    assert not cfg["use_kl_loss"]
    assert "--no-save-optim" not in plan["train"]


def test_resume_uses_actor_and_critic_from_one_run(tmp_path):
    plan = launcher.build_plan(
        make_args("--resume", str(tmp_path / "previous/actor"), "--steps", "6")
    )
    cfg = plan["config"]["training"]
    assert cfg["load"] == str(tmp_path / "previous/actor")
    assert cfg["critic_load"] == str(tmp_path / "previous/critic")
    assert cfg["num_rollout"] == 6
    with pytest.raises(ValueError, match="cannot be combined"):
        launcher.build_plan(
            make_args("--resume", str(tmp_path), "--actor-init", str(tmp_path))
        )


def test_reference_rejects_implicit_local_cluster():
    with pytest.raises(ValueError, match="existing cluster"):
        launcher.build_plan(
            make_args("--config", str(ROOT / "slime/compactionrl/configs/glm47_flash/reference.yaml"))
        )


def test_manifest_override_and_missing_assets_fail_before_launch(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "glm4_moe_lite"}))
    (model / "model.safetensors").touch()
    args = make_args("--model", str(model), "--dataset", str(tmp_path / "tasks"))
    plan = launcher.build_plan(args)
    with pytest.raises(ValueError, match="Missing Harbor tasks"):
        launcher.validate_assets(plan, args)


def test_critic_initialization_keeps_chained_optimizer_container():
    import ast
    from types import SimpleNamespace

    path = ROOT / "slime/slime/backends/megatron_utils/model.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "initialize_chained_optimizer_states"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    calls = []
    children = [
        SimpleNamespace(
            optimizer=name,
            config="cfg",
            init_state_fn=lambda opt, cfg: calls.append((opt, cfg)),
        )
        for name in ("dense", "expert")
    ]
    chain = SimpleNamespace(chained_optimizers=children)
    namespace[node.name](chain)
    assert chain.chained_optimizers is children
    assert calls == [("dense", "cfg"), ("expert", "cfg")]


def test_hf_backbone_import_preserves_value_head_even_on_error():
    import ast
    from contextlib import contextmanager
    from types import SimpleNamespace

    path = ROOT / "slime/slime/utils/megatron_bridge_utils.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "preserve_critic_value_head"
    )
    namespace = {
        "contextmanager": contextmanager,
        "unwrap_model": lambda models: models,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    head = object()
    critic = SimpleNamespace(role="critic", output_layer=head)
    with pytest.raises(RuntimeError):
        with namespace[node.name]([critic]):
            assert critic.output_layer is None
            raise RuntimeError("simulated backbone import failure")
    assert critic.output_layer is head
    actor = SimpleNamespace(role="actor", output_layer=head)
    with namespace[node.name]([actor]):
        assert actor.output_layer is head


def test_bridge_honors_explicit_activation_recomputation():
    import ast
    from types import SimpleNamespace

    path = ROOT / "slime/slime/utils/megatron_bridge_utils.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "configure_bridge_recomputation"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    provider = SimpleNamespace(recompute_granularity=None, recompute_num_layers=99)
    namespace[node.name](
        provider,
        SimpleNamespace(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        ),
    )
    assert provider.recompute_granularity == "full"
    assert provider.recompute_method == "uniform"
    assert provider.recompute_num_layers == 1
    namespace[node.name](provider, SimpleNamespace(recompute_granularity=None))
    assert provider.recompute_granularity == "full"
