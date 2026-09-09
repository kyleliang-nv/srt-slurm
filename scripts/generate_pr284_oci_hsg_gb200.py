#!/usr/bin/env python3
"""Generate the OCI-HSG GB200 duplicate of InferenceMAX PR #284's sweep.

The source recipes are read from an immutable InferenceMAX checkout.  The
generator changes only platform-specific paths, NICs, Slurm settings, and the
hardware label; all runtime treatment knobs remain those of PR #284.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import yaml

INFERENCEMAX_COMMIT = "7ae649d194874b4570533ea2c809a4c0811b1dea"
SRT_SLURM_COMMIT = "d50ee7280c33d469df8708e363e23be2456e94fb"
OCI_HSG_SRT_SLURM_COMMIT = "06b7cc8306cfcbf362121bc5d972c41c1c81ed30"

REMOTE_BASE = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_infbench/"
    "users/kylliang/srtslurm_studies/20260908_pr284_gb200_oci_hsg"
)
MODEL_PATH = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/"
    "nvidia_minimax-m3-nvfp4/hf/hf-9014640_orig"
)
DRAFT_MODEL_PATH = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/"
    "inferact_minimax-m3-eagle3-gqa/hf/hf-9669248_orig"
)
HF_CACHE = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_inferencex/"
    "common/cache"
)
AGENTX_CACHE = HF_CACHE / "datasets-agentx-aiperfb7b16"
CONTAINER_DIR = HF_CACHE / "containers"
RECIPE_DIR = Path(
    "benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3/gb200-fp4/agentic"
)
OUTPUT_DIR = Path("configs/studies/pr284-oci-hsg-gb200-20260908")

ARMS = {
    "fix1-v028-flashinfer": {
        "suffix": "",
        "image": "vllm/vllm-openai:v0.28.0",
        "dynamo": "1.5.0.dev20260906",
        "draft_attention": "FLASHINFER",
    },
    "fix2-v0271-flashattn": {
        "suffix": "-ab-v0271-flashattn",
        "image": "vllm/vllm-openai:v0.27.1",
        "dynamo": "1.3.1",
        "draft_attention": "FLASH_ATTN",
    },
    "nightly-native": {
        "suffix": "-nightly-native",
        "image": "vllm/vllm-openai:nightly-9ea8f3ffc354901b740f0b31988900897b7221d7",
        "dynamo": "1.5.0.dev20260908",
        "draft_attention": "FLASHINFER",
    },
}

POINTS = (
    ("tp4-resident-c1", "agg-tp4-agentic", 1, 4, False),
    ("tp4-resident-c10", "agg-tp4-agentic", 10, 4, False),
    ("tp4-simplecpu-c15", "agg-tp4-vllm-simple-agentic", 15, 4, True),
    ("tp4-simplecpu-c25", "agg-tp4-vllm-simple-agentic", 25, 4, True),
    ("tp4-simplecpu-c30", "agg-tp4-vllm-simple-agentic", 30, 4, True),
    ("tp8-resident-c1", "agg-tp8-agentic", 1, 8, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-max-root",
        type=Path,
        required=True,
        help="InferenceMAX checkout pinned to the PR #284 commit",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="srt-slurm checkout receiving the generated configs",
    )
    return parser.parse_args()


def load_source(root: Path, stem: str, suffix: str) -> dict:
    path = root / RECIPE_DIR / f"{stem}{suffix}.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_speculative_config(config: dict, arm: dict) -> None:
    raw = config["backend"]["vllm_config"]["aggregated"]["speculative-config"]
    spec = json.loads(raw)
    spec["model"] = str(DRAFT_MODEL_PATH)
    spec["rejection_sample_method"] = "synthetic"
    spec["synthetic_acceptance_length"] = 2.78
    config["backend"]["vllm_config"]["aggregated"]["speculative-config"] = json.dumps(
        spec, separators=(",", ":")
    )


def transform(source: dict, arm_name: str, arm: dict, point: tuple) -> dict:
    point_name, _stem, concurrency, tp_size, simple_cpu = point
    config = copy.deepcopy(source)
    run_name = f"pr284-oci-hsg-gb200-{arm_name}-{point_name}"
    container_name = arm["image"].replace("/", "_").replace(":", "_")
    container_path = CONTAINER_DIR / f"{container_name}_arm64.sqsh"

    config["name"] = run_name
    config["model"]["path"] = str(MODEL_PATH)
    config["model"]["container"] = str(container_path)
    config["resources"]["gpu_type"] = "gb200"
    config["slurm"] = {
        "account": "coreai_comparch_inferencex",
        "partition": "batch",
        "time_limit": "04:00:00",
    }
    config["sbatch_directives"].update({"qos": "normal", "gres": "gpu:4"})

    environment = config.setdefault("environment", {})
    environment.update(
        {
            "HF_HOME": str(HF_CACHE),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "SRTCTL_CONTAINER_STAGE_SOURCE": f"docker://{arm['image']}",
            "SRTCTL_CONTAINER_STAGE_ARCH": "aarch64",
            "SRTCTL_CONTAINER_STAGE_OUTPUT": str(container_path),
            "MINIMAX_M3_EAGLE3_DRAFT_MODEL": str(DRAFT_MODEL_PATH),
            "MINIMAX_M3_EAGLE3_DRAFT_LOCAL_DIR": str(DRAFT_MODEL_PATH),
            "PR284_INFERENCEMAX_COMMIT": INFERENCEMAX_COMMIT,
            "PR284_SOURCE_SRT_SLURM_COMMIT": SRT_SLURM_COMMIT,
            "OCI_HSG_SRT_SLURM_COMMIT": OCI_HSG_SRT_SLURM_COMMIT,
            "PR284_SOURCE_HARDWARE": "GB200",
            "PR284_EXECUTION_HARDWARE": "GB200-OCI-HSG",
            "PR284_ROUTING_CONTROL": "kv-router-exact-port",
        }
    )

    backend_env = config["backend"]["aggregated_environment"]
    backend_env["UCX_NET_DEVICES"] = "mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1"
    backend_env["NCCL_IB_HCA"] = "mlx5_0,mlx5_1,mlx5_3,mlx5_4"
    backend_env["HF_HOME"] = str(HF_CACHE)
    backend_env["HF_HUB_OFFLINE"] = "1"

    set_speculative_config(config, arm)

    benchmark_env = config["benchmark"]["env"]
    benchmark_env.update(
        {
            "MODEL": "nvidia/MiniMax-M3-NVFP4",
            "SERVED_MODEL_NAME": "nvidia/MiniMax-M3-NVFP4",
            "MODEL_PREFIX": "minimaxm3",
            "FRAMEWORK": "dynamo-vllm",
            "PRECISION": "fp4",
            "CONC": str(concurrency),
            "RESULT_FILENAME": run_name,
            "DURATION": "3600",
            "FULL_E2E_DURATION_SECONDS": "3600",
            "RUNNER_TYPE": "gb200-oci-hsg",
            "IMAGE": arm["image"],
            "SPEC_DECODING": "eagle3-gqa-k3-synthetic-al2.78",
            "TP": str(tp_size),
            "GPU_COUNT": str(tp_size),
            "AIPERF_WARMUP_REQUESTS_PER_LANE": "10",
            "AIPERF_TRACE_IDLE_GAP_CAP_SECONDS": "300",
            "AIPERF_SERVER_METRICS_COLLECTION_INTERVAL": "1.0",
            "AGENTIC_WARMUP_GRACE_PERIOD": "1800",
            "SOURCE_CLUSTER": "OCI-HSG",
            "PORT_ID": run_name,
            "STUDY_CHANGE": "pr284-exact-runtime-duplicate-on-oci-hsg-gb200",
            "ROUTING_CONTROL": "pr284-kv-router-exact-port",
            "KV_OFFLOADING": "simplecpu" if simple_cpu else "disabled",
        }
    )

    config["extra_mount"] = [
        f"{REMOTE_BASE / 'InferenceMAX'}:/infmax-workspace",
        f"{AGENTX_CACHE}:/aiperf_mmap_cache",
        f"{HF_CACHE}:/hf_hub_cache",
        f"{DRAFT_MODEL_PATH}:{DRAFT_MODEL_PATH}",
    ]
    return config


def validate(config: dict, arm_name: str, arm: dict, point: tuple) -> None:
    point_name, _stem, concurrency, tp_size, simple_cpu = point
    assert config["resources"]["gpu_type"] == "gb200"
    assert config["resources"]["gpus_per_agg"] == tp_size
    assert config["resources"]["agg_nodes"] == tp_size // 4
    assert config["slurm"]["time_limit"] == "04:00:00"
    assert config["identity"]["container"]["image"] == arm["image"]
    assert config["model"]["container"].startswith(str(CONTAINER_DIR))
    assert config["dynamo"]["version"] == arm["dynamo"]
    assert config["frontend"]["args"]["router-mode"] == "kv"

    vllm = config["backend"]["vllm_config"]["aggregated"]
    assert json.loads(vllm["compilation-config"])["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    spec = json.loads(vllm["speculative-config"])
    assert spec["method"] == "eagle3"
    assert spec["num_speculative_tokens"] == 3
    assert spec["attention_backend"] == arm["draft_attention"]
    assert spec["rejection_sample_method"] == "synthetic"
    assert spec["synthetic_acceptance_length"] == 2.78

    env = config["benchmark"]["env"]
    assert env["CONC"] == str(concurrency)
    assert env["DURATION"] == "3600"
    assert env["TP"] == str(tp_size)
    assert env["ROUTING_CONTROL"] == "pr284-kv-router-exact-port"
    assert config["environment"]["OCI_HSG_SRT_SLURM_COMMIT"] == (
        OCI_HSG_SRT_SLURM_COMMIT
    )
    assert config["environment"]["SRTCTL_CONTAINER_STAGE_SOURCE"] == (
        f"docker://{arm['image']}"
    )
    assert ("kv-transfer-config" in vllm) is simple_cpu
    assert point_name in config["name"]
    assert arm_name in config["name"]


def main() -> None:
    args = parse_args()
    git_dir = args.inference_max_root / ".git"
    if not git_dir.exists():
        raise SystemExit(f"not an InferenceMAX checkout: {args.inference_max_root}")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=args.inference_max_root,
        text=True,
    ).strip()
    if source_commit != INFERENCEMAX_COMMIT:
        raise SystemExit(
            f"InferenceMAX checkout is {source_commit}, expected {INFERENCEMAX_COMMIT}"
        )

    output_dir = args.repo_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    expected: set[Path] = set()

    for arm_name, arm in ARMS.items():
        for point in POINTS:
            point_name, stem, _concurrency, _tp_size, _simple_cpu = point
            source = load_source(args.inference_max_root, stem, arm["suffix"])
            config = transform(source, arm_name, arm, point)
            validate(config, arm_name, arm, point)
            path = output_dir / f"{arm_name}--{point_name}.yaml"
            path.write_text(
                yaml.safe_dump(config, sort_keys=False, width=120),
                encoding="utf-8",
            )
            expected.add(path)

    for stale in output_dir.glob("*.yaml"):
        if stale not in expected and not stale.name.startswith("srtslurm."):
            stale.unlink()

    print(f"generated and validated {len(expected)} configs in {output_dir}")


if __name__ == "__main__":
    main()
