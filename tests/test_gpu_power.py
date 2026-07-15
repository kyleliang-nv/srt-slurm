# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-GPU TGP configuration and host-side nvidia-smi setup."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.core.gpu_power import (
    apply_gpu_power_limits,
    build_node_gpu_power_map,
    build_power_setup_bash,
)
from srtctl.core.schema import GpuPowerConfig, ModelConfig, ResourceConfig, SrtConfig
from srtctl.core.topology import Process


def _process(
    node: str,
    mode: str,
    gpu_indices: frozenset[int],
) -> Process:
    return Process(
        node=node,
        gpu_indices=gpu_indices,
        sys_port=8081,
        http_port=30000,
        endpoint_mode=mode,  # type: ignore[arg-type]
        endpoint_index=0,
    )


class TestBuildNodeGpuPowerMap:
    def test_separate_nodes_get_role_limits(self):
        processes = [
            _process("gpu-01", "prefill", frozenset({0, 1, 2, 3})),
            _process("gpu-02", "decode", frozenset({0, 1, 2, 3})),
        ]
        gpu_power = GpuPowerConfig(prefill_tgp=700, decode_tgp=500)

        node_map = build_node_gpu_power_map(processes, gpu_power)

        assert node_map == {
            "gpu-01": {0: 700, 1: 700, 2: 700, 3: 700},
            "gpu-02": {0: 500, 1: 500, 2: 500, 3: 500},
        }

    def test_shared_node_gets_per_gpu_limits(self):
        processes = [
            _process("gpu-01", "prefill", frozenset({0, 1, 2, 3})),
            _process("gpu-01", "decode", frozenset({4, 5, 6, 7})),
        ]
        gpu_power = GpuPowerConfig(prefill_tgp=700, decode_tgp=500)

        node_map = build_node_gpu_power_map(processes, gpu_power)

        assert node_map == {
            "gpu-01": {
                0: 700,
                1: 700,
                2: 700,
                3: 700,
                4: 500,
                5: 500,
                6: 500,
                7: 500,
            }
        }

    def test_unconfigured_role_is_skipped(self):
        processes = [
            _process("gpu-01", "prefill", frozenset({0, 1})),
            _process("gpu-02", "decode", frozenset({0, 1})),
        ]
        gpu_power = GpuPowerConfig(prefill_tgp=600)

        node_map = build_node_gpu_power_map(processes, gpu_power)

        assert node_map == {"gpu-01": {0: 600, 1: 600}}

    def test_conflicting_gpu_assignment_raises(self):
        processes = [
            _process("gpu-01", "prefill", frozenset({0, 1})),
            _process("gpu-01", "decode", frozenset({1, 2})),
        ]
        gpu_power = GpuPowerConfig(prefill_tgp=700, decode_tgp=500)

        with pytest.raises(ValueError, match="conflicting TGP"):
            build_node_gpu_power_map(processes, gpu_power)


class TestBuildPowerSetupBash:
    def test_groups_gpus_by_tgp(self):
        bash = build_power_setup_bash({0: 700, 1: 700, 4: 500, 5: 500})

        assert bash == (
            "nvidia-smi -pm 1 -i 0,1,4,5 || sudo -n nvidia-smi -pm 1 -i 0,1,4,5 && "
            "nvidia-smi -pl 500 -i 4,5 || sudo -n nvidia-smi -pl 500 -i 4,5 && "
            "nvidia-smi -pl 700 -i 0,1 || sudo -n nvidia-smi -pl 700 -i 0,1"
        )

    def test_empty_map_returns_empty_string(self):
        assert build_power_setup_bash({}) == ""


class TestApplyGpuPowerLimits:
    @patch("srtctl.core.gpu_power.start_srun_process")
    def test_runs_host_side_srun_per_node(self, mock_srun):
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_srun.return_value = mock_proc

        processes = [
            _process("gpu-01", "prefill", frozenset({0, 1})),
            _process("gpu-02", "decode", frozenset({2, 3})),
        ]
        gpu_power = GpuPowerConfig(prefill_tgp=700, decode_tgp=500)

        apply_gpu_power_limits(
            processes,
            gpu_power,
            log_dir=Path("/tmp/logs"),
            srun_options={"export": "ALL"},
        )

        assert mock_srun.call_count == 2

        first_call = mock_srun.call_args_list[0].kwargs
        assert first_call["nodelist"] == ["gpu-01"]
        assert first_call["container_image"] is None
        assert first_call["srun_options"] == {"export": "ALL"}
        assert first_call["use_bash_wrapper"] is False
        assert "sudo -n nvidia-smi -pl 700 -i 0,1" in first_call["command"][2]

        second_call = mock_srun.call_args_list[1].kwargs
        assert second_call["nodelist"] == ["gpu-02"]
        assert "sudo -n nvidia-smi -pl 500 -i 2,3" in second_call["command"][2]

    @patch("srtctl.core.gpu_power.start_srun_process")
    def test_nonzero_exit_raises(self, mock_srun):
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1
        mock_srun.return_value = mock_proc

        processes = [_process("gpu-01", "prefill", frozenset({0}))]
        gpu_power = GpuPowerConfig(prefill_tgp=700)

        with pytest.raises(RuntimeError, match="Failed to set GPU power limits"):
            apply_gpu_power_limits(processes, gpu_power, log_dir=Path("/tmp/logs"))


class TestGpuPowerConfigValidation:
    def test_positive_tgp_required(self):
        with pytest.raises(Exception, match="positive integer"):
            GpuPowerConfig(prefill_tgp=0)

    def test_disaggregated_rejects_agg_tgp(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        container = tmp_path / "container.sqsh"
        container.touch()

        with pytest.raises(Exception, match="agg_tgp"):
            SrtConfig(
                name="test",
                model=ModelConfig(path=str(model_dir), container=str(container), precision="fp8"),
                resources=ResourceConfig(
                    gpu_type="h100",
                    gpus_per_node=8,
                    prefill_nodes=1,
                    decode_nodes=1,
                    prefill_workers=1,
                    decode_workers=1,
                ),
                gpu_power=GpuPowerConfig(prefill_tgp=700, decode_tgp=500, agg_tgp=600),
            )

    def test_aggregated_rejects_prefill_decode_tgp(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        container = tmp_path / "container.sqsh"
        container.touch()

        with pytest.raises(Exception, match="prefill_tgp/decode_tgp"):
            SrtConfig(
                name="test",
                model=ModelConfig(path=str(model_dir), container=str(container), precision="fp8"),
                resources=ResourceConfig(
                    gpu_type="h100",
                    gpus_per_node=8,
                    agg_nodes=1,
                    agg_workers=1,
                ),
                gpu_power=GpuPowerConfig(prefill_tgp=700),
            )
