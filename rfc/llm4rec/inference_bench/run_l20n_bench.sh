#!/bin/bash
# sid-gr-inference vs SGLang beam-search benchmark on the 72GB SM120 host.
# Faithful (H100-style) config: NO memory-deviation envs; prefill CUDA graphs ON.
# Run ON the remote host: nohup bash run_l20n_bench.sh > bench.log 2>&1 &
set -exo pipefail

BASE=/mnt/data/hongsheng.jhs
RECSYS=$BASE/recsys-examples
SIDGR=$RECSYS/examples/sid-gr-inference
export SGLANG_REPO=$BASE/sglang
export GR_DECODE_ATTEN_ROOT=$RECSYS/corelib/gr_decode_atten
export MODEL_DIR=$BASE/models/Qwen3-1.7B
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# huggingface.co is blocked on this pod; bench_serving's "random" dataset samples
# from ShareGPT and downloads it from the hub, so route ALL hub traffic through
# the mirror (offline mode is not enough - the dataset must be fetchable once).
export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
BENCH_DIR=$(cd "$(dirname "$0")" && pwd)
RESULTS=$BASE/bench_results/l20n_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS"

echo "===== ENV ====="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
pip list --format=freeze 2>/dev/null | grep -iE "cutlass|quack|flashinfer|cuda-tile|tvm|dlpack|torch==|sglang|sgl|flash-attn|transformers==" | tee "$RESULTS/env_manifest.txt"
git -C $BASE/sglang log -1 --oneline | tee -a "$RESULTS/env_manifest.txt"
git -C $RECSYS log -1 --oneline | tee -a "$RESULTS/env_manifest.txt"

cd "$SIDGR"
echo "===== GATES (first SM120/Blackwell validation) ====="
PYTHONPATH=src python tools/check_kernel_backends.py
python -m pytest tests/test_real_decode_attention_smoke.py -x -q
python -m pytest tests/test_real_decode_attention_correctness.py -x -q
PYTHONPATH=$SGLANG_REPO/python python "$BENCH_DIR/fork_beam_sanity.py"

if [ "${SKIP_OFFLINE:-0}" != "1" ]; then
echo "===== OFFLINE PERF (faithful grid) ====="
CONTEXT_LENS="1000 5000" BEAM_WIDTHS="256" BATCH_SIZES="1 2 4 8" REPEAT=3 \
  bash scripts/run_offline_perf_benchmark.sh

echo "===== OFFLINE ACCURACY ====="
CONTEXT_LENS="1000 5000" BEAM_WIDTHS="256" BATCH_SIZES="1 2 4 8" CORRECTNESS_REPEAT=1 \
  bash scripts/run_offline_accuracy_benchmark.sh

fi

echo "===== ONLINE: GR server ====="
( GR_MODEL_DIR=$MODEL_DIR GR_CONTEXT_LEN=5000 GR_DECODE_STEPS=3 GR_BEAM_WIDTH=256 \
  GR_MAX_BATCH_SIZE=4 GR_BEAM_KV_POOL_CAPACITY=4 GR_CONTEXT_KV_POOL_CAPACITY=4 \
  GR_HTTP_HOST=127.0.0.1 GR_HTTP_PORT=8000 GR_DECODE_BACKEND=real GR_DEVICE=cuda \
  bash scripts/serve_qwen3_gr_http.sh > "$RESULTS/gr_server.log" 2>&1 ) &
for i in $(seq 1 120); do curl -fsS http://127.0.0.1:8000/ready >/dev/null 2>&1 && break; sleep 5; done
curl -fsS http://127.0.0.1:8000/ready
for round in 1 2 3; do
  HOST=127.0.0.1 PORT=8000 MODEL_DIR=$MODEL_DIR REQUESTS=64 CONTEXT_LEN=5000 DECODE_STEPS=3 \
  BEAM_WIDTH=256 REQUEST_RATE=inf MAX_CONCURRENCY=4 WARMUP_REQUESTS=0 \
  bash scripts/run_gr_sglang_bench_serving_beam_benchmark.sh || true
done
pkill -f serve_qwen3_gr_http || true; pkill -f tools/serve_qwen3_gr_http.py || true; sleep 15

echo "===== ONLINE: SGLang fork server ====="
( PYTHONPATH=$SGLANG_REPO/python python -m sglang.launch_server \
    --model-path $MODEL_DIR --host 127.0.0.1 --port 30000 \
    --enable-beam-search --disable-radix-cache > "$RESULTS/sgl_server.log" 2>&1 ) &
# probe /generate, NOT /health_generate (fork reports a false 503 under beam mode)
for i in $(seq 1 150); do
  curl -sS -m 5 http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
    -d '{"input_ids":[1,2,3],"sampling_params":{"max_new_tokens":1,"n":2,"temperature":0.0}}' 2>/dev/null | grep -q '^{' && break
  sleep 5
done
for round in 1 2 3; do
  HOST=127.0.0.1 PORT=30000 MODEL_DIR=$MODEL_DIR REQUESTS=64 CONTEXT_LEN=5000 DECODE_STEPS=3 \
  BEAM_WIDTH=256 REQUEST_RATE=inf MAX_CONCURRENCY=4 WARMUP_REQUESTS=16 \
  bash scripts/run_sglang_serving_beam_benchmark.sh || true
done
pkill -f sglang.launch_server || true

echo "===== COLLECT RESULTS ====="
cp -r benchmark_artifacts/sglang_compare/offline_perf_* "$RESULTS/" 2>/dev/null || true
cp -r benchmark_artifacts/sglang_compare/offline_accuracy_* "$RESULTS/" 2>/dev/null || true
cp benchmark_artifacts/sglang_compare/*serving*.jsonl "$RESULTS/" 2>/dev/null || true
echo "===== RESULTS: OFFLINE PERF ====="; cat "$RESULTS"/offline_perf_*/summary.md || true
echo "===== RESULTS: PER-CELL GR SUCCESS ====="
python3 - <<PY
import json, glob
for f in sorted(glob.glob("$RESULTS/offline_perf_*/gr_ctx*req*.json")):
    d = json.load(open(f)); sm = d.get("scheduler_metrics") or {}
    print(f.split("/")[-1], "succeeded=", sm.get("succeeded_requests"), "failed=", sm.get("failed_requests"))
PY
echo "===== RESULTS: OFFLINE ACCURACY ====="; cat "$RESULTS"/offline_accuracy_*/summary.md || true
echo "===== RESULTS: ONLINE ====="
for f in "$RESULTS"/*serving*.jsonl; do
  echo "--- $f"
  tail -1 "$f" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('completed', d.get('completed'), '| req/s', round(d.get('request_throughput',0),3), '| median_e2e_ms', round(d.get('median_e2e_latency_ms',0),1), '| p99_e2e_ms', round(d.get('p99_e2e_latency_ms',0),1))"
done
echo "RESULTS_DIR=$RESULTS"
echo "BENCHMARK_DONE"
