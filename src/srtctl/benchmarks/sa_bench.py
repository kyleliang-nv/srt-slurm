# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SA-Bench throughput/latency benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("sa-bench")
class SABenchRunner(BenchmarkRunner):
    """SA-Bench throughput and latency benchmark.

    Tests serving throughput at various concurrency levels.

    Required config fields:
        - benchmark.isl: Input sequence length
        - benchmark.osl: Output sequence length
        - benchmark.concurrencies: Concurrency levels (e.g., "4x8x16x32")

    Optional:
        - benchmark.req_rate: Request rate (default: "inf")
        - benchmark.num_requests: Main benchmark --num-prompts; if unset, uses 10 × each concurrency level
        - benchmark.dataset_name: "random" (default) or "preformatted"
        - benchmark.dataset_path: in-container path to a JSONL dataset file when dataset_name="preformatted"
          (mount it yourself via container_mounts / extra_mount). Each line:
          {"input": {"messages": [...]}, "num_tokens": N, "max_tokens": M}.
          Skips all tokenization, so isl/osl become nominal labels and the file's lengths are used.
        - benchmark.skip_warmup: if true, skip the per-concurrency warmup run (default: false).
    """

    @property
    def name(self) -> str:
        return "SA-Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/sa-bench/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "sa-bench")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if b.isl is None:
            errors.append("benchmark.isl is required for sa-bench")
        if b.osl is None:
            errors.append("benchmark.osl is required for sa-bench")
        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for sa-bench")
        if b.num_requests is not None and b.num_requests <= 0:
            errors.append("benchmark.num_requests must be a positive integer when set")

        if b.dataset_name is not None and b.dataset_name not in ("random", "preformatted"):
            errors.append(f"benchmark.dataset_name must be 'random' or 'preformatted' (got {b.dataset_name!r})")
        if b.dataset_name == "preformatted" and not b.dataset_path:
            errors.append(
                "benchmark.dataset_path is required when benchmark.dataset_name=preformatted "
                "(in-container path to the JSONL file; mount it via container_mounts/extra_mount)"
            )
        if b.dataset_path and b.dataset_name not in (None, "preformatted"):
            errors.append("benchmark.dataset_path is only used when benchmark.dataset_name=preformatted")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        r = config.resources
        endpoint = f"http://localhost:{runtime.frontend_port}"

        # Format concurrencies as x-separated string if it's a list
        concurrencies = b.concurrencies
        if isinstance(concurrencies, list):
            concurrencies = "x".join(str(c) for c in concurrencies)

        # Compute GPU info for result filename
        is_disaggregated = r.is_disaggregated
        if is_disaggregated:
            prefill_gpus = r.prefill_gpus
            decode_gpus = r.decode_gpus
            total_gpus = prefill_gpus + decode_gpus
        else:
            total_gpus = (r.agg_nodes or 1) * r.gpus_per_node
            prefill_gpus = 0
            decode_gpus = 0

        # Tokenizer path: HF model ID or container source path. The benchmark
        # process does not need the node-local server copy.
        tokenizer_path = str(runtime.model_path) if runtime.is_hf_model else str(runtime.benchmark_model_path)

        cmd = [
            "bash",
            self.script_path,
            endpoint,
            str(b.isl),
            str(b.osl),
            str(concurrencies) if concurrencies else "",
            str(b.req_rate) if b.req_rate else "inf",
            tokenizer_path,
            config.served_model_name,
            str(is_disaggregated).lower(),
            str(total_gpus),
            str(prefill_gpus),
            str(decode_gpus),
            str(b.random_range_ratio) if b.random_range_ratio is not None else "0.8",
            str(b.num_requests) if b.num_requests is not None else "",
            b.dataset_name or "random",
            b.dataset_path or "",
            "true" if b.skip_warmup else "false",
        ]
        return cmd
