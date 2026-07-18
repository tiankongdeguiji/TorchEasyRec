#!/usr/bin/env bash
# Constrained-decoding cost eval for sid-gr-inference at the online operating
# point: ctx1000, beam 256, client concurrency 4, 64 requests, decode_steps 3.
# A/B: unconstrained vs --catalog-jsonl at several catalog sizes, plus a
# profiled round per config and a trie-mask microbench.
set -uxo pipefail

BASE=${BASE:-/mnt/data/hongsheng.jhs}
SIDGR=${SIDGR:-$BASE/recsys-examples/examples/sid-gr-inference}
MODEL_DIR=${MODEL_DIR:-$BASE/models/Qwen3-0.6B}
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS=${RESULTS:-$BASE/bench_results/constrained_ctx1000_$(date +%Y%m%d_%H%M%S)}
CATALOG_SIZES=${CATALOG_SIZES:-"10000 100000 1000000"}
ROUNDS=${ROUNDS:-3}
CTX=1000
mkdir -p "$RESULTS/catalogs"

run_client() {  # $1 = num requests
  ( cd "$SIDGR" && \
    GR_BENCH_HOST=127.0.0.1 GR_BENCH_PORT=8000 SGLANG_REPO=$BASE/sglang \
    MODEL_DIR=$MODEL_DIR CONTEXT_LEN=$CTX DECODE_STEPS=3 \
    BEAM_WIDTH=256 REQUEST_RATE=inf MAX_CONCURRENCY=4 WARMUP_REQUESTS=0 REQUESTS=$1 \
    bash scripts/run_gr_sglang_bench_serving_beam_benchmark.sh )
}

start_server() {  # $1 = extra env as "K=V K=V", $2 = log file
  ( cd "$SIDGR" && env $1 \
      GR_MODEL_DIR=$MODEL_DIR GR_CONTEXT_LEN=$CTX GR_DECODE_STEPS=3 GR_BEAM_WIDTH=256 \
      GR_MAX_BATCH_SIZE=4 GR_BEAM_KV_POOL_CAPACITY=4 GR_CONTEXT_KV_POOL_CAPACITY=4 \
      GR_HTTP_HOST=127.0.0.1 GR_HTTP_PORT=8000 GR_DECODE_BACKEND=real GR_DEVICE=cuda \
      bash scripts/serve_qwen3_gr_http.sh > "$2" 2>&1 ) &
  for i in $(seq 1 150); do
    curl -fsS http://127.0.0.1:8000/ready >/dev/null 2>&1 && return 0; sleep 5
  done
  echo "SERVER_READY_TIMEOUT"; tail -50 "$2"; return 1
}

stop_server() {
  pkill -f tools/serve_qwen3_gr_http.py || true; sleep 15
}

run_rounds() {  # $1 = phase dir name, $2 = num timed rounds
  mkdir -p "$RESULTS/$1"
  run_client 8 || true  # warmup
  rm -f "$SIDGR"/benchmark_artifacts/sglang_compare/*serving*.jsonl
  for round in $(seq 1 "$2"); do
    run_client 64 || true
  done
  mv "$SIDGR"/benchmark_artifacts/sglang_compare/*serving*.jsonl "$RESULTS/$1/" 2>/dev/null || true
}

echo "===== catalogs ====="
for n in $CATALOG_SIZES; do
  python "$BENCH_DIR/gen_sid_catalog.py" --items "$n" \
    --out "$RESULTS/catalogs/sid_${n}.jsonl" | tee "$RESULTS/catalogs/sid_${n}.stats.json"
done

echo "===== A: unconstrained baseline (same-day parity) ====="
stop_server
start_server "" "$RESULTS/uncon_server.log"
run_rounds uncon "$ROUNDS"
stop_server
start_server "GR_PROFILE_CONTINUOUS_DECODE=1" "$RESULTS/uncon_server_prof.log"
run_rounds uncon_prof 1
stop_server

for n in $CATALOG_SIZES; do
  echo "===== B: constrained, catalog $n items ====="
  CAT="$RESULTS/catalogs/sid_${n}.jsonl"
  start_server "GR_CATALOG_JSONL=$CAT GR_CATALOG_EOS_TOKEN_ID=151645" \
    "$RESULTS/cat${n}_server.log" || { stop_server; continue; }
  curl -fsS http://127.0.0.1:8000/catalog/status | tee "$RESULTS/cat${n}_status.json" || true
  if ! python "$BENCH_DIR/validate_catalog_paths.py" --catalog "$CAT" --context-len $CTX; then
    echo "GATE_FAIL_CONSTRAINED_$n"; tail -80 "$RESULTS/cat${n}_server.log"; stop_server; continue
  fi
  run_rounds "cat${n}" "$ROUNDS"
  stop_server
  start_server "GR_CATALOG_JSONL=$CAT GR_CATALOG_EOS_TOKEN_ID=151645 GR_PROFILE_CONTINUOUS_DECODE=1" \
    "$RESULTS/cat${n}_server_prof.log" || { stop_server; continue; }
  run_rounds "cat${n}_prof" 1
  stop_server
done

echo "===== C: trie mask microbench ====="
for n in $CATALOG_SIZES; do
  ( cd "$SIDGR" && python "$BENCH_DIR/bench_trie_mask.py" \
      --catalog "$RESULTS/catalogs/sid_${n}.jsonl" --device cuda ) \
    | tee -a "$RESULTS/trie_mask_micro.jsonl"
done

echo "===== SUMMARY ====="
for f in "$RESULTS"/uncon/*.jsonl "$RESULTS"/cat*/*.jsonl; do
  [ -f "$f" ] || continue
  echo "--- $f"
  tail -1 "$f" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('completed', d.get('completed'), '| req/s', round(d.get('request_throughput',0),3), '| median_e2e_ms', round(d.get('median_e2e_latency_ms',0),1), '| p99_e2e_ms', round(d.get('p99_e2e_latency_ms',0),1))"
done
echo "RESULTS_DIR=$RESULTS"
echo "CONSTRAINED_EVAL_DONE"
