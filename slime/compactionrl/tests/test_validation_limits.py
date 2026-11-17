"""Validate the driver's bounded parser path without loading GPU libraries."""

import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch


DRIVER = Path(__file__).parents[3] / "slime/train_compactionrl.py"


class ValidationIsolationTest(unittest.TestCase):
    def run_driver(self, inherited):
        limits = []
        resource = types.ModuleType("resource")
        resource.RLIMIT_AS, resource.RLIM_INFINITY, resource.RUSAGE_SELF = 9, -1, 0
        resource.getrlimit = lambda key: inherited
        resource.setrlimit = lambda key, value: limits.append(value)
        resource.getrusage = lambda who: types.SimpleNamespace(ru_maxrss=2048)
        expected = min([8 * 1024**3, *(v for v in inherited if v != -1)])

        def parse_args():
            self.assertEqual(limits, [(expected, expected)])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            return types.SimpleNamespace()

        arguments = types.ModuleType("slime.utils.arguments")
        arguments.parse_args = parse_args
        storage = types.ModuleType("compactionrl.runtime.storage")
        storage.prepare_storage = lambda *args: {}
        storage.storage_environment = lambda *args: {}
        megatron_args = types.ModuleType("megatron.training.arguments")
        config = Mock()
        megatron_args.core_transformer_config_from_args = config

        class ForbidTrainingImports:
            def find_spec(self, fullname, path=None, target=None):
                forbidden = ("ray", "mbridge", "slime.backends.megatron_utils")
                if any(fullname == n or fullname.startswith(n + ".") for n in forbidden):
                    raise AssertionError(f"Validation imported training runtime: {fullname}")

        modules = {
            "resource": resource,
            arguments.__name__: arguments,
            storage.__name__: storage,
            megatron_args.__name__: megatron_args,
        }
        env = {"COMPACTIONRL_VALIDATE_ONLY": "1", "COMPACTIONRL_WORK_DIR": "/unused"}
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, env),
            patch("sys.platform", "linux"),
            patch("sys.meta_path", [ForbidTrainingImports(), *sys.meta_path]),
            patch("atexit.register") as register,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            runpy.run_path(str(DRIVER), run_name="__main__")
            config.assert_called_once()
            register.call_args.args[0]()
            self.assertIn("Validation peak process RSS: 2.0 MiB", output.getvalue())

    def test_validation_is_bounded_before_parser_and_skips_training_imports(self):
        self.run_driver((-1, -1))

    def test_validation_preserves_a_lower_inherited_limit(self):
        self.run_driver((4 * 1024**3, 6 * 1024**3))


if __name__ == "__main__":
    unittest.main()
