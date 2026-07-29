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
DOCKER=/etc/dsw/runtime/export_bin/docker
IMAGE=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-test@sha256:c1cf5a01e53113707f6fc67f4cdc8e223c9d3fe3cf2b5a9c657084c45e213353
RUN_ID=qwen06b_fusion_$(date +%Y%m%d_%H%M%S)
RESULTS=$BASE/bench_results/$RUN_ID
CONTAINER=llm4rec-qwen-fusion-$RUN_ID

mkdir -p "$RESULTS"
"$DOCKER" pull "$IMAGE" | tee "$RESULTS/image_pull.txt"
"$DOCKER" image inspect "$IMAGE" >"$RESULTS/image_inspect.json"

if [[ $(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | wc -l) -ne 0 ]]; then
  echo "A GPU process is already running; refusing to benchmark" >&2
  exit 1
fi

cleanup() {
  "$DOCKER" rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$DOCKER" run --rm \
  --name "$CONTAINER" \
  --gpus device=0 \
  --network host \
  --shm-size 32g \
  -e OUT_DIR="$RESULTS" \
  -v "$BASE/recsys-examples:$BASE/recsys-examples:ro" \
  -v "$BASE/sglang:$BASE/sglang:ro" \
  -v "$BASE/models/Qwen3-0.6B:$BASE/models/Qwen3-0.6B:ro" \
  -v "$BASE/TorchEasyRec:$BASE/TorchEasyRec:ro" \
  -v "$RESULTS:$RESULTS:rw" \
  "$IMAGE" \
  bash "$BASE/TorchEasyRec/rfc/llm4rec/inference_bench/fusion/run_inside_container.sh" \
  2>&1 | tee "$RESULTS/container.log"

echo "RESULTS_DIR=$RESULTS"
