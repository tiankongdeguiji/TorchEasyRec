#!/bin/bash
# Copyright (c) 2025 Alibaba, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

BASE=/mnt/data/hongsheng.jhs
RECSYS=$BASE/recsys-examples
SIDGR=$RECSYS/examples/sid-gr-inference
SGLANG_REPO=$BASE/sglang
MODEL_DIR=$BASE/models/Qwen3-0.6B
GR_DECODE_ATTEN_ROOT=$RECSYS/corelib/gr_decode_atten
OUT_DIR=${OUT_DIR:?OUT_DIR must name the writable result directory}
PYTHONPATH=$SIDGR/src:$SGLANG_REPO/python

export CUDA_VISIBLE_DEVICES=0
export GR_DECODE_ATTEN_ROOT
export HF_ENDPOINT=https://hf-mirror.com
export MODEL_DIR
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH
export SGLANG_REPO
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

mkdir -p "$OUT_DIR"/{accuracy,ablations,logs,nsys,offline,online,workloads}
cd "$SIDGR"

server_pid=
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=
}
trap stop_server EXIT

echo "===== PREFLIGHT ====="
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
  --format=csv,noheader | tee "$OUT_DIR/gpu.txt"
python - <<'PY' | tee "$OUT_DIR/runtime.txt"
import importlib.metadata
import torch

print("python", __import__("sys").version.replace("\n", " "))
print("torch", torch.__version__, "cuda", torch.version.cuda)
for package in (
    "flashinfer-python",
    "nvidia-cutlass-dsl",
    "quack-kernels",
    "transformers",
    "sglang-kernel",
):
    print(package, importlib.metadata.version(package))
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
git -C "$RECSYS" rev-parse HEAD | tee "$OUT_DIR/recsys_commit.txt"
git -C "$SGLANG_REPO" rev-parse HEAD | tee "$OUT_DIR/sglang_commit.txt"
PYTHONPATH=src python tools/check_kernel_backends.py | tee "$OUT_DIR/kernel_backends.txt"
python -m pytest -q \
  tests/test_qwen3_fusion_policy.py \
  tests/test_qwen3_single_layer_prefill.py \
  tests/test_real_weight_tiny_tool.py | tee "$OUT_DIR/fusion_tests.txt"

make_workload() {
  local requests=$1
  PYTHONPATH=src python tools/make_qwen3_beam_workload.py \
    --model-dir "$MODEL_DIR" \
    --context-len 1000 \
    --requests "$requests" \
    --no-tokenizer \
    --output-jsonl "$OUT_DIR/workloads/ctx1000_req${requests}.jsonl"
}

for requests in 1 2 4 8; do
  make_workload "$requests"
done

run_offline() {
  local label=$1
  local disabled=$2
  local next_norm=$3
  local requests=$4
  local output=$5
  echo "OFFLINE label=$label disabled=${disabled:-<empty>} next_norm=$next_norm batch=$requests"
  env \
    GR_INFERENCE_QWEN3_DISABLE_FUSIONS="$disabled" \
    GR_INFERENCE_DECODE_NEXT_INPUT_NORM_FUSION="$next_norm" \
    PYTHONPATH="$PYTHONPATH" \
    python tools/run_qwen3_real_weight_serving.py \
      --model-dir "$MODEL_DIR" \
      --workload-jsonl "$OUT_DIR/workloads/ctx1000_req${requests}.jsonl" \
      --context-len 1000 \
      --decode-steps 2 \
      --beam-width 256 \
      --requests "$requests" \
      --max-batch-size "$requests" \
      --batched-decode \
      --continuous \
      --beam-kv-pool-capacity "$requests" \
      --context-kv-pool-capacity "$requests" \
      --decode-backend real \
      --device cuda \
      --warmup-runs 2 \
      --repeat 10 \
      --output-json "$output" \
      >"$OUT_DIR/logs/${label}_b${requests}.log" 2>&1
}

echo "===== OFFLINE BINARY A/B/A ====="
for requests in 1 2 4 8; do
  run_offline fused_a "" 0 "$requests" "$OUT_DIR/offline/fused_a_b${requests}.json"
  run_offline unfused all 0 "$requests" "$OUT_DIR/offline/unfused_b${requests}.json"
  run_offline fused_b "" 0 "$requests" "$OUT_DIR/offline/fused_b_b${requests}.json"
done

echo "===== BATCH-4 ABLATIONS ====="
for family in qk_norm rope add_rmsnorm mlp; do
  run_offline "$family" "$family" 0 4 "$OUT_DIR/ablations/no_${family}.json"
done
run_offline max_fused "" 1 4 "$OUT_DIR/ablations/max_fused.json"

echo "===== CORRECTNESS ====="
run_accuracy() {
  local label=$1
  local disabled=$2
  env \
    GR_INFERENCE_QWEN3_DISABLE_FUSIONS="$disabled" \
    GR_INFERENCE_DECODE_NEXT_INPUT_NORM_FUSION=0 \
    PYTHONPATH="$PYTHONPATH" \
    python tools/run_qwen3_real_weight_serving.py \
      --model-dir "$MODEL_DIR" \
      --workload-jsonl "$OUT_DIR/workloads/ctx1000_req4.jsonl" \
      --context-len 1000 \
      --decode-steps 2 \
      --beam-width 256 \
      --requests 4 \
      --max-batch-size 4 \
      --batched-decode \
      --continuous \
      --beam-kv-pool-capacity 4 \
      --context-kv-pool-capacity 4 \
      --decode-backend real \
      --device cuda \
      --record-outputs \
      --warmup-runs 1 \
      --repeat 1 \
      --output-json "$OUT_DIR/accuracy/${label}.json" \
      >"$OUT_DIR/logs/accuracy_${label}.log" 2>&1
}
run_accuracy fused ""
run_accuracy unfused all

echo "===== NSIGHT ====="
run_nsys() {
  local label=$1
  local disabled=$2
  env \
    GR_INFERENCE_QWEN3_DISABLE_FUSIONS="$disabled" \
    GR_INFERENCE_DECODE_NEXT_INPUT_NORM_FUSION=0 \
    GR_INFERENCE_NVTX=1 \
    PYTHONPATH="$PYTHONPATH" \
    nsys profile \
      --force-overwrite=true \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --cuda-graph-trace=node \
      --output "$OUT_DIR/nsys/$label" \
      python tools/run_qwen3_real_weight_serving.py \
        --model-dir "$MODEL_DIR" \
        --workload-jsonl "$OUT_DIR/workloads/ctx1000_req4.jsonl" \
        --context-len 1000 \
        --decode-steps 2 \
        --beam-width 256 \
        --requests 4 \
        --max-batch-size 4 \
        --batched-decode \
        --continuous \
        --beam-kv-pool-capacity 4 \
        --context-kv-pool-capacity 4 \
        --decode-backend real \
        --device cuda \
        --warmup-runs 1 \
        --repeat 1 \
        --cuda-profiler-range \
        --output-json "$OUT_DIR/nsys/$label.json" \
        >"$OUT_DIR/logs/nsys_$label.log" 2>&1
  nsys export \
    --type sqlite \
    --force-overwrite=true \
    --output "$OUT_DIR/nsys/$label.sqlite" \
    "$OUT_DIR/nsys/$label.nsys-rep"
}
run_nsys fused ""
run_nsys unfused all

echo "===== ONLINE ====="
run_online() {
  local label=$1
  local disabled=$2
  local port=$3
  echo "ONLINE label=$label disabled=${disabled:-<empty>} port=$port"
  env \
    GR_INFERENCE_QWEN3_DISABLE_FUSIONS="$disabled" \
    GR_INFERENCE_DECODE_NEXT_INPUT_NORM_FUSION=0 \
    GR_MODEL_DIR="$MODEL_DIR" \
    GR_CONTEXT_LEN=1000 \
    GR_DECODE_STEPS=3 \
    GR_BEAM_WIDTH=256 \
    GR_MAX_BATCH_SIZE=4 \
    GR_BEAM_KV_POOL_CAPACITY=4 \
    GR_CONTEXT_KV_POOL_CAPACITY=4 \
    GR_HTTP_HOST=127.0.0.1 \
    GR_HTTP_PORT="$port" \
    GR_DECODE_BACKEND=real \
    GR_DEVICE=cuda \
    PYTHONPATH="$PYTHONPATH" \
    bash scripts/serve_qwen3_gr_http.sh \
    >"$OUT_DIR/logs/server_$label.log" 2>&1 &
  server_pid=$!
  local ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/ready" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    tail -n 100 "$OUT_DIR/logs/server_$label.log" >&2
    return 1
  fi
  for round in 1 2 3; do
    local round_dir="$OUT_DIR/online/$label/round$round"
    mkdir -p "$round_dir"
    GR_BENCH_HOST=127.0.0.1 \
    GR_BENCH_PORT="$port" \
    MODEL_DIR="$MODEL_DIR" \
    OUT_DIR="$round_dir" \
    REQUESTS=64 \
    CONTEXT_LEN=1000 \
    DECODE_STEPS=3 \
    BEAM_WIDTH=256 \
    REQUEST_RATE=inf \
    MAX_CONCURRENCY=4 \
    WARMUP_REQUESTS=8 \
    SGLANG_PYTHON=python \
    bash scripts/run_gr_sglang_bench_serving_beam_benchmark.sh \
      >"$OUT_DIR/logs/online_${label}_round${round}.log" 2>&1
  done
  stop_server
  sleep 5
}
run_online fused "" 8000
run_online unfused all 8000

python "$BASE/TorchEasyRec/rfc/llm4rec/inference_bench/fusion/summarize.py" \
  "$OUT_DIR" | tee "$OUT_DIR/summary.txt"
echo "FUSION_BENCHMARK_DONE"
