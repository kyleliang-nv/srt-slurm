#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SA-Bench: Throughput/latency benchmark
# Expects: endpoint isl osl concurrencies req_rate tokenizer_path model_name is_disaggregated total_gpus prefill_gpus decode_gpus random_range_ratio [num_requests] [dataset_name] [dataset_path]
# When dataset_name=preformatted, dataset_path is the in-container path to a JSONL file
# (one record per line: {"input": {"messages": [...]}, "num_tokens": N, "max_tokens": M}).
# In that mode bench.sh skips --random-* flags and --use-chat-template (template is already baked in).

set -e

# Ensure benchmark dependencies are available.
# Creates an isolated venv with --system-site-packages so container packages are
# reused and only missing deps get installed — without touching system Python.
SA_BENCH_VENV="/tmp/sa-bench-venv"
SA_BENCH_DEPS=(aiohttp numpy pandas datasets Pillow tqdm transformers huggingface_hub)

ensure_sa_bench_deps() {
    # Quick check: if all deps import fine in current Python, skip venv entirely
    if python3 -c "import aiohttp, numpy, pandas, datasets, PIL, tqdm, transformers, huggingface_hub" 2>/dev/null; then
        echo "All sa-bench deps already available — skipping venv setup"
        return
    fi

    echo "Missing sa-bench deps — installing into venv at $SA_BENCH_VENV ..."
    if [ ! -d "$SA_BENCH_VENV" ]; then
        python3 -m venv --system-site-packages "$SA_BENCH_VENV"
    fi
    source "$SA_BENCH_VENV/bin/activate"
    pip install "${SA_BENCH_DEPS[@]}"
    echo "sa-bench deps ready"
}

ensure_sa_bench_deps

#
# Optional profiling (via worker profiling endpoints):
#   PROFILE_TYPE: "nsys" or "torch" to enable profiling (or "none" to disable)
#   PROFILE_OUTPUT_DIR: Directory inside the container to save profiler output (e.g., /logs/profiles)
#   WORKER_PORT: Default port to use when an endpoint is provided as IP only (defaults to 9090)
#
# Worker targets (prefer *_ENDPOINTS; *_IPS is supported for backward-compat):
#   PROFILE_PREFILL_ENDPOINTS: Comma-separated list of prefill worker endpoints (ip:port or ip)
#   PROFILE_DECODE_ENDPOINTS: Comma-separated list of decode worker endpoints (ip:port or ip)
#   PROFILE_AGG_ENDPOINTS: Comma-separated list of aggregated worker endpoints (ip:port or ip)
#   PROFILE_PREFILL_IPS / PROFILE_DECODE_IPS / PROFILE_AGG_IPS: Comma-separated IPs (uses WORKER_PORT)
#
# Step ranges (stop_step is exclusive; num_steps = stop_step - start_step):
#   PROFILE_PREFILL_START_STEP / PROFILE_PREFILL_STOP_STEP
#   PROFILE_DECODE_START_STEP / PROFILE_DECODE_STOP_STEP
#   PROFILE_AGG_START_STEP / PROFILE_AGG_STOP_STEP

ENDPOINT=$1
ISL=$2
OSL=$3
CONCURRENCIES=$4
REQ_RATE=${5:-inf}
MODEL_PATH=${6:-/model/}
MODEL_NAME=${7:-"model"}
IS_DISAGGREGATED=${8:-false}
TOTAL_GPUS=${9:-0}
PREFILL_GPUS=${10:-0}
DECODE_GPUS=${11:-0}
RANDOM_RANGE_RATIO=${12:-0.8}
# Optional: fixed --num-prompts for the main benchmark; if empty, uses 10 × concurrency per level
NUM_REQUESTS_CONFIG=${13:-}
# Optional: dataset selection. "random" (default) or "preformatted" (JSONL at DATASET_PATH).
DATASET_NAME=${14:-random}
DATASET_PATH=${15:-}

if [ "$DATASET_NAME" = "preformatted" ] && [ -z "$DATASET_PATH" ]; then
    echo "ERROR: dataset_name=preformatted requires dataset_path (in-container JSONL path)" >&2
    exit 2
fi
if [ "$DATASET_NAME" = "preformatted" ] && [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: preformatted dataset not found at $DATASET_PATH (mount it via container_mounts/extra_mount)" >&2
    exit 2
fi

# Parse endpoint into host:port
HOST=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f1)
PORT=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f2 | cut -d/ -f1)

WORK_DIR="$(dirname "$0")"

echo "SA-Bench Config: endpoint=${ENDPOINT}; isl=${ISL}; osl=${OSL}; concurrencies=${CONCURRENCIES}; req_rate=${REQ_RATE}; model=${MODEL_NAME}; num_requests=${NUM_REQUESTS_CONFIG:-<10x concurrency>}; dataset=${DATASET_NAME}${DATASET_PATH:+ (${DATASET_PATH})}"

# Build dataset args once. For "random" we synthesize prompts with --random-*;
# for "preformatted" we point at a pre-baked JSONL and skip tokenization entirely.
DATASET_ARGS=()
EXTRA_MAIN_ARGS=()
if [ "$DATASET_NAME" = "preformatted" ]; then
    DATASET_ARGS=(--dataset-name preformatted --dataset-path "$DATASET_PATH")
    # Chat template is already rendered into the prompts; do NOT re-apply.
else
    DATASET_ARGS=(
        --dataset-name random
        --random-input-len "$ISL"
        --random-output-len "$OSL"
        --random-range-ratio "${RANDOM_RANGE_RATIO}"
    )
    EXTRA_MAIN_ARGS=(--use-chat-template)
fi

# Profiling shared helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/profiling.sh
source "${SCRIPT_DIR}/../lib/profiling.sh"
profiling_init_from_env

cleanup() { stop_all_profiling; }
trap cleanup EXIT

# Parse concurrency list
IFS='x' read -r -a CONCURRENCY_LIST <<< "$CONCURRENCIES"

# Quick curl to verify endpoint is working
echo "Verifying endpoint..."
curl -s "${ENDPOINT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
        \"stream\": false,
        \"max_tokens\": 10
    }" | head -c 200
echo ""

ulimit -n 65536 2>/dev/null || true  # May fail in containers without CAP_SYS_RESOURCE

# Benchmark
result_dir="/logs/sa-bench_isl_${ISL}_osl_${OSL}"
mkdir -p "$result_dir"

# Start profiling before benchmark
start_all_profiling

for concurrency in "${CONCURRENCY_LIST[@]}"; do

    num_warmup_prompts=$((concurrency * 2))
    python3 -u "${WORK_DIR}/benchmark_serving.py" \
        --model "${MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
        --host "$HOST" --port "$PORT" \
        --backend "dynamo" --endpoint /v1/completions \
        --disable-tqdm \
        "${DATASET_ARGS[@]}" \
        --num-prompts "$num_warmup_prompts" \
        --ignore-eos \
        --request-rate 250 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency" \
        --trust-remote-code

    if [ -n "$NUM_REQUESTS_CONFIG" ]; then
        num_prompts="$NUM_REQUESTS_CONFIG"
    else
        num_prompts=$((concurrency * 10))
    fi
    
    # Generate result filename based on mode
    if [ "$IS_DISAGGREGATED" = "true" ]; then
        result_filename="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}.json"
    else
        result_filename="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}.json"
    fi
    
    echo "Running benchmark with concurrency: $concurrency"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"

    set -x
    python3 -u "${WORK_DIR}/benchmark_serving.py" \
        --model "${MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
        --host "$HOST" --port "$PORT" \
        --backend "dynamo" --endpoint /v1/completions \
        --disable-tqdm \
        "${DATASET_ARGS[@]}" \
        --num-prompts "$num_prompts" \
        --ignore-eos \
        --request-rate "${REQ_RATE}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency" \
        --trust-remote-code \
        "${EXTRA_MAIN_ARGS[@]}" \
        --save-result --result-dir "$result_dir" --result-filename "$result_filename"
    set +x

    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done

stop_all_profiling

echo "SA-Bench complete. Results in $result_dir"

