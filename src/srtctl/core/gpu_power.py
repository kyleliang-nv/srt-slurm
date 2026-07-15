# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side GPU power limit setup via nvidia-smi."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.slurm import start_srun_process
from srtctl.core.topology import Process, WorkerMode

if TYPE_CHECKING:
    from srtctl.core.schema import GpuPowerConfig

logger = logging.getLogger(__name__)


def _mode_tgp_watts(gpu_power: GpuPowerConfig, mode: WorkerMode) -> int | None:
    if mode == "prefill":
        return gpu_power.prefill_tgp
    if mode == "decode":
        return gpu_power.decode_tgp
    return gpu_power.agg_tgp


def build_node_gpu_power_map(
    processes: list[Process],
    gpu_power: GpuPowerConfig,
) -> dict[str, dict[int, int]]:
    """Map each worker node GPU index to its target TGP in watts.

    Uses per-process ``gpu_indices`` so prefill and decode workers on the same
    node can receive different power limits.
    """
    node_gpus: dict[str, dict[int, int]] = defaultdict(dict)

    for process in processes:
        tgp = _mode_tgp_watts(gpu_power, process.endpoint_mode)
        if tgp is None:
            continue

        for gpu_idx in process.gpu_indices:
            existing = node_gpus[process.node].get(gpu_idx)
            if existing is not None and existing != tgp:
                raise ValueError(
                    f"GPU {gpu_idx} on {process.node} has conflicting TGP assignments: "
                    f"{existing}W vs {tgp}W"
                )
            node_gpus[process.node][gpu_idx] = tgp

    return {node: dict(gpu_map) for node, gpu_map in node_gpus.items()}


def build_power_setup_bash(gpu_to_watts: dict[int, int]) -> str:
    """Build bash that enables persistence mode and sets per-GPU power limits."""
    if not gpu_to_watts:
        return ""

    all_gpus = sorted(gpu_to_watts)
    gpu_list = ",".join(str(gpu_idx) for gpu_idx in all_gpus)

    by_tgp: dict[int, list[int]] = defaultdict(list)
    for gpu_idx, watts in gpu_to_watts.items():
        by_tgp[watts].append(gpu_idx)

    parts = [f"sudo nvidia-smi -pm 1 -i {gpu_list}"]
    for watts in sorted(by_tgp):
        gpu_ids = ",".join(str(gpu_idx) for gpu_idx in sorted(by_tgp[watts]))
        parts.append(f"sudo nvidia-smi -pl {watts} -i {gpu_ids}")

    return " && ".join(parts)


def apply_gpu_power_limits(
    processes: list[Process],
    gpu_power: GpuPowerConfig,
    *,
    log_dir: Path,
    srun_options: dict[str, str] | None = None,
) -> None:
    """Apply configured TGP limits on worker nodes before backend startup.

    Runs host-side ``srun`` steps (no container) so ``nvidia-smi`` talks to the
    physical GPUs on each node.
    """
    node_map = build_node_gpu_power_map(processes, gpu_power)
    if not node_map:
        logger.info("GPU power limits configured but no worker GPUs matched; skipping")
        return

    for node in sorted(node_map):
        gpu_map = node_map[node]
        bash_cmd = build_power_setup_bash(gpu_map)
        power_log = log_dir / f"gpu_power_{node}.out"
        logger.info("Setting GPU power on %s: %s", node, bash_cmd)

        proc = start_srun_process(
            command=["bash", "-c", bash_cmd],
            nodelist=[node],
            output=str(power_log),
            container_image=None,
            srun_options=srun_options,
            use_bash_wrapper=False,
        )
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Failed to set GPU power limits on {node} (exit {return_code}); see {power_log}"
            )

        logger.info("GPU power limits applied on %s", node)
