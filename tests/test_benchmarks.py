# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark runners."""

import pytest

from srtctl.benchmarks import get_runner, list_benchmarks
from srtctl.benchmarks.base import SCRIPTS_DIR


class TestBenchmarkRegistry:
    """Test benchmark runner registry."""

    def test_list_benchmarks(self):
        """All expected benchmarks are registered."""
        benchmarks = list_benchmarks()
        assert "sa-bench" in benchmarks
        assert "sglang-bench" in benchmarks
        assert "mmlu" in benchmarks
        assert "gpqa" in benchmarks
        assert "longbenchv2" in benchmarks
        assert "router" in benchmarks

    def test_get_runner_valid(self):
        """Can get runner for valid benchmark type."""
        runner = get_runner("sa-bench")
        assert runner.name == "SA-Bench"
        assert "sa-bench" in runner.script_path

    def test_get_runner_invalid(self):
        """Raises ValueError for unknown benchmark type."""
        with pytest.raises(ValueError, match="Unknown benchmark type"):
            get_runner("nonexistent-benchmark")


class TestSABenchRunner:
    """Test SA-Bench runner."""

    def test_validate_config_missing_isl(self):
        """Validates that isl is required."""
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench", osl=1024, concurrencies="4x8"),
        )
        errors = runner.validate_config(config)
        assert any("isl" in e for e in errors)

    def test_validate_config_valid(self):
        """Valid config passes validation."""
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench", isl=1024, osl=1024, concurrencies="4x8"),
        )
        errors = runner.validate_config(config)
        assert errors == []

    def test_validate_config_rejects_non_positive_num_requests(self):
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench",
                isl=1024,
                osl=1024,
                concurrencies="4x8",
                num_requests=0,
            ),
        )
        errors = runner.validate_config(config)
        assert any("num_requests" in e for e in errors)

    def test_build_command_num_requests_defaults_to_empty(self):
        """Unset num_requests is passed as empty string so bench.sh uses 10× concurrency."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench", isl=1024, osl=128, concurrencies="1x2", req_rate="inf"),
        )
        cmd = runner.build_command(config, runtime)
        # Positional layout (see SABenchRunner.build_command):
        #   ..., random_range_ratio[13], num_requests[14], dataset_name[15],
        #   dataset_path[16], skip_warmup[17]
        # Empty num_requests means bench.sh falls back to 10 × concurrency.
        assert cmd[14] == ""
        assert cmd[15] == "random"
        assert cmd[16] == ""
        # skip_warmup defaults to "false" — warmup runs.
        assert cmd[17] == "false"

    def test_build_command_num_requests_explicit(self):
        from unittest.mock import MagicMock

        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench",
                isl=1024,
                osl=128,
                concurrencies="1x2",
                req_rate="inf",
                num_requests=500,
            ),
        )
        cmd = runner.build_command(config, runtime)
        assert cmd[14] == "500"

    def test_build_command_preformatted_dataset(self):
        """dataset_name=preformatted threads the path through to bench.sh."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench",
                isl=131072,
                osl=8192,
                concurrencies="1x2",
                dataset_name="preformatted",
                dataset_path="/datasets/deepseek-r1.jsonl",
            ),
        )
        cmd = runner.build_command(config, runtime)
        assert cmd[15] == "preformatted"
        assert cmd[16] == "/datasets/deepseek-r1.jsonl"

    def test_validate_preformatted_requires_dataset_path(self):
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench",
                isl=1024,
                osl=1024,
                concurrencies="4x8",
                dataset_name="preformatted",
            ),
        )
        errors = runner.validate_config(config)
        assert any("dataset_path" in e for e in errors)

    def test_validate_dataset_path_without_preformatted(self):
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench",
                isl=1024,
                osl=1024,
                concurrencies="4x8",
                dataset_name="random",
                dataset_path="/datasets/foo.jsonl",
            ),
        )
        errors = runner.validate_config(config)
        assert any("dataset_path is only used" in e for e in errors)

    def test_build_command_skip_warmup_true(self):
        """skip_warmup=True is passed through to bench.sh as 'true' at position 17."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench",
                isl=1024,
                osl=128,
                concurrencies="1x2",
                req_rate="inf",
                skip_warmup=True,
            ),
        )
        cmd = runner.build_command(config, runtime)
        assert cmd[17] == "true"

    def test_build_command_skip_warmup_default_false(self):
        """skip_warmup defaults to False (warmup runs) when not set in the config."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench", isl=1024, osl=128, concurrencies="1x2"),
        )
        cmd = runner.build_command(config, runtime)
        assert cmd[17] == "false"

    def test_skip_warmup_loaded_from_yaml(self):
        """benchmark.skip_warmup: true in YAML is parsed onto BenchmarkConfig."""
        from srtctl.core.schema import SrtConfig

        yaml_dict = {
            "name": "test",
            "model": {"path": "/model", "container": "/image", "precision": "fp4"},
            "resources": {"gpu_type": "h100"},
            "benchmark": {
                "type": "sa-bench",
                "isl": 1024,
                "osl": 128,
                "concurrencies": "1x2",
                "skip_warmup": True,
            },
        }
        config = SrtConfig.Schema().load(yaml_dict)
        assert config.benchmark.skip_warmup is True

    def test_validate_unknown_dataset_name(self):
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench",
                isl=1024,
                osl=1024,
                concurrencies="4x8",
                dataset_name="sharegpt",
            ),
        )
        errors = runner.validate_config(config)
        assert any("dataset_name" in e for e in errors)


class TestSGLangBenchRunner:
    """Test SGLang-Bench runner."""

    def test_validate_config_valid(self):
        from srtctl.benchmarks.sglang_bench import SGLangBenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SGLangBenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sglang-bench", isl=1024, osl=1024, concurrencies="4x8", req_rate="inf"),
        )
        errors = runner.validate_config(config)
        assert errors == []

    def test_validate_config_missing_fields(self):
        from srtctl.benchmarks.sglang_bench import SGLangBenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SGLangBenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sglang-bench"),
        )
        errors = runner.validate_config(config)
        assert any("benchmark.isl is required" in e for e in errors)
        assert any("benchmark.osl is required" in e for e in errors)
        assert any("benchmark.concurrencies is required" in e for e in errors)

    def test_validate_config_rejects_zero_values(self):
        from srtctl.benchmarks.sglang_bench import SGLangBenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SGLangBenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sglang-bench", isl=0, osl=1, concurrencies=[0], req_rate=0),
        )
        errors = runner.validate_config(config)
        assert any("benchmark.isl must be a positive integer" in e for e in errors)
        assert any("benchmark.concurrencies values must be positive integers" in e for e in errors)
        assert any(
            "benchmark.req_rate must be a positive integer" in e or "benchmark.req_rate must be a positive number" in e
            for e in errors
        )

    def test_build_command(self):
        from unittest.mock import MagicMock

        from srtctl.benchmarks.sglang_bench import SGLangBenchRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = SGLangBenchRunner()
        runtime = MagicMock()
        runtime.frontend_port = 8000

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sglang-bench", isl=1024, osl=128, concurrencies=[1, 2]),
        )

        cmd = runner.build_command(config, runtime)
        assert cmd == [
            "bash",
            "/srtctl-benchmarks/sglang-bench/bench.sh",
            "http://localhost:8000",
            "1024",
            "128",
            "1x2",
            "inf",
        ]


class TestMooncakeRouterRunner:
    """Test Mooncake Router benchmark runner."""

    def test_build_command_includes_tokenizer_path(self):
        """Build command passes tokenizer path to aiperf.

        This fixes a bug where aiperf couldn't load the tokenizer because it was
        using the served model name (e.g., "Qwen/Qwen3-32B") to find the tokenizer,
        but that's not a valid HuggingFace ID or local path. The fix passes
        --tokenizer /model explicitly since the model is mounted there.
        """
        from unittest.mock import MagicMock

        from srtctl.benchmarks.mooncake_router import MooncakeRouterRunner

        runner = MooncakeRouterRunner()

        config = MagicMock()
        config.benchmark = MagicMock()
        config.benchmark.mooncake_workload = "conversation"
        config.benchmark.ttft_threshold_ms = 2000
        config.benchmark.itl_threshold_ms = 25
        config.served_model_name = "Qwen/Qwen3-32B"

        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False  # Local model mounted at /model

        cmd = runner.build_command(config, runtime)

        # Command: bash, script, endpoint, model_name, workload, ttft, itl, tokenizer_path
        assert cmd[7] == "/model"  # tokenizer path


class TestScriptsExist:
    """Test that benchmark scripts exist."""

    def test_scripts_dir_exists(self):
        """Scripts directory exists."""
        assert SCRIPTS_DIR.exists()

    def test_sa_bench_script_exists(self):
        """SA-Bench script exists."""
        script = SCRIPTS_DIR / "sa-bench" / "bench.sh"
        assert script.exists()

    def test_mmlu_script_exists(self):
        """MMLU script exists."""
        script = SCRIPTS_DIR / "mmlu" / "bench.sh"
        assert script.exists()
