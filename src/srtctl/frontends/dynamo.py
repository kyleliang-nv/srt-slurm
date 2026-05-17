# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo frontend implementation.

Uses NATS/etcd for communication between frontend and backend workers.
"""

import logging
from typing import TYPE_CHECKING, Any

from srtctl.core.health import WorkerHealthResult, check_dynamo_health
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

DYNAMO_SYSTEM_PORT_BASE = 8081


class DynamoFrontend:
    """Dynamo frontend implementation.

    Uses dynamo.frontend module with NATS/etcd for worker discovery.
    Health checks via /health endpoint.
    """

    @property
    def type(self) -> str:
        return "dynamo"

    @property
    def health_endpoint(self) -> str:
        return "/health"

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        """Parse dynamo /health endpoint response."""
        return check_dynamo_health(response_json, expected_prefill, expected_decode)

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert frontend args dict to CLI arguments."""
        if not args:
            return []
        result = []
        for key, value in args.items():
            if value is True:
                result.append(f"--{key}")
            elif value is not False and value is not None:
                result.extend([f"--{key}", str(value)])
        return result

    def start_frontends(
        self,
        topology: Any,  # FrontendTopology
        runtime: "RuntimeContext",
        config: Any,  # SrtConfig
        backend: Any,  # BackendProtocol
        backend_processes: list["Process"],
    ) -> list["ManagedProcess"]:
        """Start dynamo frontends on designated nodes."""
        from srtctl.core.processes import ManagedProcess

        processes: list[ManagedProcess] = []

        # When TRTLLM workers opt into PMIx v5, keep the single-process
        # frontend out of Slurm's PMI/PMIx direct-launch path. dynamo.frontend
        # imports an OpenMPI stack from /opt/dynamo/venv that fails MPI_Init
        # against this rack's Slurm PMIx plugins (errors mention
        # pmix3x_client.c). It does not need a multi-rank MPI world, so force
        # OpenMPI singleton mode and omit --mpi for the frontend step.
        use_openmpi_singleton = getattr(backend, "enable_pmix_v5", False)
        mpi_plugin = None if use_openmpi_singleton else "pmix"

        for idx, node in enumerate(topology.frontend_nodes):
            logger.info("Starting dynamo frontend %d on %s", idx, node)

            frontend_log = runtime.log_dir / f"{node}_frontend_{idx}.out"
            cmd = ["python3", "-m", "dynamo.frontend", f"--http-port={topology.frontend_port}"]
            cmd.extend(self.get_frontend_args_list(config.frontend.args))

            env_to_set = {
                "ETCD_ENDPOINTS": f"http://{runtime.nodes.infra}:2379",
                "NATS_SERVER": f"nats://{runtime.nodes.infra}:4222",
                "DYN_REQUEST_PLANE": config.dynamo.request_plane,
                "DYN_SYSTEM_PORT": str(DYNAMO_SYSTEM_PORT_BASE + idx),
                "DYN_TCP_POOL_SIZE": "2048",
            }
            if use_openmpi_singleton:
                env_to_set["OMPI_MCA_ess"] = "singleton"

            # Add frontend env from config
            if config.frontend.env:
                env_to_set.update(config.frontend.env)

            # Build bash preamble (setup script + dynamo install)
            bash_preamble = self._build_preamble(config)
            if use_openmpi_singleton:
                # OpenMPI still detects Slurm direct launch through inherited
                # SLURM_/PMI_/PMIX_ env vars even when this frontend srun omits
                # --mpi. Scrub those before importing dynamo.frontend; the
                # frontend only needs NATS/etcd and its HTTP/system ports.
                slurm_env_scrub = "unset ${!SLURM_@} ${!PMI_@} ${!PMIX_@}"
                bash_preamble = (
                    f"{slurm_env_scrub} && {bash_preamble}"
                    if bash_preamble
                    else slurm_env_scrub
                )

            proc = start_srun_process(
                command=cmd,
                nodelist=[node],
                output=str(frontend_log),
                container_image=str(runtime.container_image),
                container_mounts=runtime.container_mounts,
                env_to_set=env_to_set,
                bash_preamble=bash_preamble,
                # TODO(jthomson): I don't have the faintest clue of
                # why this is needed in later versions of Dynamo, but it is.
                mpi=mpi_plugin,
            )

            processes.append(
                ManagedProcess(
                    name=f"frontend_{idx}",
                    popen=proc,
                    log_file=frontend_log,
                    node=node,
                    critical=True,
                )
            )

        return processes

    def _build_preamble(self, config: Any) -> str | None:
        """Build bash preamble for dynamo frontend processes."""
        parts = []

        # Custom setup script
        if config.setup_script:
            script_path = f"/configs/{config.setup_script}"
            parts.append(
                f"echo 'Running setup script: {script_path}' && "
                f"if [ -f '{script_path}' ]; then bash '{script_path}'; else echo 'WARNING: {script_path} not found'; fi"
            )

        # Dynamo installation (required for dynamo frontend)
        # Skip if dynamo.install is False (container already has dynamo installed)
        if config.dynamo.install:
            parts.append(config.dynamo.get_install_commands())

        if not parts:
            return None

        return " && ".join(parts)
