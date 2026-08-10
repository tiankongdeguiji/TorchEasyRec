# Qwen3 fusion benchmark runbook

This runbook reproduces the `sid-gr-inference` Qwen3-0.6B fused versus
strict-unfused benchmark on `ty_l20n_dev`. It covers source synchronization,
DSW Docker-in-Docker execution, monitoring, validation, failure handling, and
publication. The benchmark compares `Qwen3GRModel` with itself; SGLang is a
kernel/runtime dependency, not a competing server.

## Canonical benchmark

| Parameter             | Value                                                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model                 | Qwen3-0.6B bf16                                                                                                                                  |
| GPU                   | One RTX PRO 5000 72 GB Blackwell, SM120                                                                                                          |
| Offline workload      | context 1000, beam 256, batch 1/2/4/8, two warmups, ten measured repetitions                                                                     |
| Online workload       | context 1000, beam 256, 64 requests, maximum concurrency 4, three output tokens, eight warmup requests, three rounds                             |
| Fused policy          | `GR_INFERENCE_QWEN3_DISABLE_FUSIONS=`                                                                                                            |
| Strict-unfused policy | `GR_INFERENCE_QWEN3_DISABLE_FUSIONS=all`                                                                                                         |
| Shared policy         | `GR_INFERENCE_DECODE_NEXT_INPUT_NORM_FUSION=0`                                                                                                   |
| Image                 | `mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-test@sha256:c1cf5a01e53113707f6fc67f4cdc8e223c9d3fe3cf2b5a9c657084c45e213353` |

The harness also runs single-family `qk_norm`, `rope`, `add_rmsnorm`, and
`mlp` ablations at batch 4, an exploratory max-fused row, fused/unfused
correctness, and one Nsight Systems capture per headline configuration.

Latency speedup is `unfused_ms / fused_ms`. Throughput speedup is
`fused_req_s / unfused_req_s`.

## Source and data topology

Do not edit any repository directly on the DSW machine. Make changes in a
local worktree, commit, push to `ty_git`, and fast-forward the corresponding
remote checkout.

| Purpose                    | Local path                                                                                 | DSW/container path                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| recsys-examples source     | `/mnt/disk/hongsheng.jhs/Workspace/recsys-examples-sidgr-fusion`                           | `/mnt/data/hongsheng.jhs/recsys-examples`                           |
| sid-gr-inference           | `/mnt/disk/hongsheng.jhs/Workspace/recsys-examples-sidgr-fusion/examples/sid-gr-inference` | `/mnt/data/hongsheng.jhs/recsys-examples/examples/sid-gr-inference` |
| GR decode-attention kernel | `/mnt/disk/hongsheng.jhs/Workspace/recsys-examples-sidgr-fusion/corelib/gr_decode_atten`   | `/mnt/data/hongsheng.jhs/recsys-examples/corelib/gr_decode_atten`   |
| SGLang source              | `/mnt/disk/hongsheng.jhs/Workspace/sglang`                                                 | `/mnt/data/hongsheng.jhs/sglang`                                    |
| TorchEasyRec benchmark     | `/mnt/disk/hongsheng.jhs/Workspace/tzrec-rfc-llm4rec-bench`                                | `/mnt/data/hongsheng.jhs/TorchEasyRec`                              |
| Model checkpoint           | n/a                                                                                        | `/mnt/data/hongsheng.jhs/models/Qwen3-0.6B`                         |
| Full raw results           | n/a                                                                                        | `/mnt/data/hongsheng.jhs/bench_results/qwen06b_fusion_<timestamp>`  |
| Curated results            | `rfc/llm4rec/inference_bench/results/`                                                     | same relative path under remote TorchEasyRec                        |

Expected source revisions for the published reference run:

- recsys-examples branch `llm4rec/qwen3-fusion-bench`, commit `f8b71ff58dbb6eb4bc685712d7c4540444d1c7a4`;
- SGLang branch `llm4rec/beam-search-bench`, commit `2aac32adcf839bfd3f8d02d6cc4a840d9d82b6a9`;
- TorchEasyRec branch `rfc_llm4rec_inference_bench`.

If newer commits are intentionally tested, record their complete hashes and
do not compare results as if they came from the reference source revisions.

## 1. Synchronize source revisions

Verify and push local worktrees first:

```bash
git -C /mnt/disk/hongsheng.jhs/Workspace/recsys-examples-sidgr-fusion \
  status --short --branch
git -C /mnt/disk/hongsheng.jhs/Workspace/recsys-examples-sidgr-fusion \
  push ty_git llm4rec/qwen3-fusion-bench

git -C /mnt/disk/hongsheng.jhs/Workspace/tzrec-rfc-llm4rec-bench \
  status --short --branch
git -C /mnt/disk/hongsheng.jhs/Workspace/tzrec-rfc-llm4rec-bench \
  push ty_git rfc_llm4rec_inference_bench
```

Fast-forward the DSW checkouts:

```bash
ssh ty_l20n_dev 'bash -lc "$(cat)"' <<'REMOTE'
set -euo pipefail

git -C /mnt/data/hongsheng.jhs/recsys-examples config http.version HTTP/1.1
git -C /mnt/data/hongsheng.jhs/recsys-examples fetch ty_git \
  llm4rec/qwen3-fusion-bench
git -C /mnt/data/hongsheng.jhs/recsys-examples checkout \
  llm4rec/qwen3-fusion-bench
git -C /mnt/data/hongsheng.jhs/recsys-examples merge --ff-only \
  ty_git/llm4rec/qwen3-fusion-bench

git -C /mnt/data/hongsheng.jhs/sglang config http.version HTTP/1.1
git -C /mnt/data/hongsheng.jhs/sglang fetch ty_git \
  llm4rec/beam-search-bench
git -C /mnt/data/hongsheng.jhs/sglang checkout llm4rec/beam-search-bench
git -C /mnt/data/hongsheng.jhs/sglang merge --ff-only \
  ty_git/llm4rec/beam-search-bench

git -C /mnt/data/hongsheng.jhs/TorchEasyRec config http.version HTTP/1.1
git -C /mnt/data/hongsheng.jhs/TorchEasyRec fetch ty_git \
  rfc_llm4rec_inference_bench
git -C /mnt/data/hongsheng.jhs/TorchEasyRec checkout \
  rfc_llm4rec_inference_bench
git -C /mnt/data/hongsheng.jhs/TorchEasyRec merge --ff-only \
  ty_git/rfc_llm4rec_inference_bench
REMOTE
```

The three remote worktrees must be clean before launching. Never use reset,
checkout-discard, or cleanup commands to remove an unexplained remote change.

## 2. DSW preflight

The DSW pod already provides Docker-in-Docker. Do not install Docker, CUDA,
Python packages, or a virtual environment on the host or in the benchmark
image.

- Docker client: `/etc/dsw/runtime/export_bin/docker`
- DSW proxy socket: `/var/docker/proxy/docker-proxy.sock`
- Invoke remote commands through `bash -lc` so the DSW runtime environment is
  active.

Run these read-only checks:

```bash
ssh ty_l20n_dev 'bash -lc "$(cat)"' <<'REMOTE'
set -euo pipefail
DOCKER=/etc/dsw/runtime/export_bin/docker
IMAGE=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-test@sha256:c1cf5a01e53113707f6fc67f4cdc8e223c9d3fe3cf2b5a9c657084c45e213353

test -S /var/docker/proxy/docker-proxy.sock
"$DOCKER" version
"$DOCKER" image inspect "$IMAGE" --format "{{json .RepoDigests}}"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,utilization.gpu,driver_version \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
test -s /mnt/data/hongsheng.jhs/models/Qwen3-0.6B/config.json
test -d /mnt/data/hongsheng.jhs/models/Qwen3-0.6B
test -d /mnt/data/hongsheng.jhs/recsys-examples/corelib/gr_decode_atten
test -f /mnt/data/hongsheng.jhs/TorchEasyRec/rfc/llm4rec/inference_bench/fusion/run_ty_l20n_dev.sh
ss -ltn | grep -E ":(8000|30000) " && exit 1 || true
df -h /mnt/data /mnt/systemDisk
REMOTE
```

Launch only when there are no GPU compute processes. The launcher currently
refuses to run if any visible GPU has an active compute process, even though
the nested container receives only GPU 0.

If a stale benchmark container exists, inspect its logs and ownership first.
Remove only a confirmed stale container whose name begins with
`llm4rec-qwen-fusion-`; never remove unrelated containers.

## 3. Launch

The launcher pulls the pinned digest, records image inspection, checks GPU
idleness, mounts source/model paths read-only, mounts only the timestamped
result directory read-write, and removes its named container on exit.

Run detached and retain the launcher PID:

```bash
ssh ty_l20n_dev 'bash -lc "$(cat)"' <<'REMOTE'
set -euo pipefail
BASE=/mnt/data/hongsheng.jhs
STAMP=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG=$BASE/bench_results/qwen06b_fusion_launcher_${STAMP}.log
cd $BASE/TorchEasyRec
nohup bash rfc/llm4rec/inference_bench/fusion/run_ty_l20n_dev.sh \
  >"$LAUNCH_LOG" 2>&1 &
echo "PID=$!"
echo "LAUNCH_LOG=$LAUNCH_LOG"
REMOTE
```

The result directory is created as
`/mnt/data/hongsheng.jhs/bench_results/qwen06b_fusion_<timestamp>`. The
launcher prints its exact `RESULTS_DIR` when complete. During execution, find
the newest active directory with:

```bash
ssh ty_l20n_dev 'ls -dt \
  /mnt/data/hongsheng.jhs/bench_results/qwen06b_fusion_20* | head -n 1'
```

Do not launch a second benchmark while the first container, server, or GPU
process is active.

## 4. Monitor continuously

Poll every 30-60 seconds. Record the launcher PID, launch log, and result
directory in the experiment notes.

```bash
ssh ty_l20n_dev 'bash -lc "$(cat)"' <<'REMOTE'
PID=<launcher-pid>
LOG=<launcher-log>
OUT=<result-directory>

ps -p "$PID" -o pid=,etime=,stat=,cmd= || true
/etc/dsw/runtime/export_bin/docker ps --format "{{.Names}} {{.Status}}" \
  | grep llm4rec-qwen-fusion || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm \
  --format=csv,noheader -i 0
grep -E "^(=====|OFFLINE|ONLINE|FUSION_BENCHMARK_DONE|RESULTS_DIR)" \
  "$LOG" | tail -n 40
find "$OUT/offline" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l
find "$OUT/ablations" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l
find "$OUT/online" -name "*.jsonl" 2>/dev/null | wc -l
tail -n 30 "$LOG"
REMOTE
```

Expected phase order and completion counts:

1. `PREFLIGHT`: runtime manifest, backend selection, 68 fusion/model tests;
1. `OFFLINE BINARY A/B/A`: 12 JSON files;
1. `BATCH-4 ABLATIONS`: five JSON files;
1. `CORRECTNESS`: two JSON files;
1. `NSIGHT`: two reports, two SQLite exports, and two benchmark JSON files;
1. `ONLINE`: six JSONL files, three per configuration;
1. `FUSION_BENCHMARK_DONE` and `RESULTS_DIR=...`.

GPU utilization can read 0% between short benchmark processes. Determine
progress from phase markers and artifact counts, not from one utilization
sample.

## 5. Validate results

A valid run has a zero launcher exit status, the completion marker, no live
benchmark container, and an idle GPU after cleanup.

```bash
ssh ty_l20n_dev 'bash -lc "$(cat)"' <<'REMOTE'
set -euo pipefail
OUT=<result-directory>
LOG=<launcher-log>

grep -F FUSION_BENCHMARK_DONE "$LOG"
test -s "$OUT/summary.txt"
python /mnt/data/hongsheng.jhs/TorchEasyRec/rfc/llm4rec/inference_bench/fusion/summarize.py \
  "$OUT"
python /mnt/data/hongsheng.jhs/TorchEasyRec/rfc/llm4rec/inference_bench/fusion/analyze_nsys.py \
  --fused "$OUT/nsys/fused.sqlite" \
  --unfused "$OUT/nsys/unfused.sqlite" \
  --output-json "$OUT/nsys_breakdown.json" \
  --output-markdown "$OUT/nsys_breakdown.md"

/etc/dsw/runtime/export_bin/docker ps --format "{{.Names}}" \
  | grep llm4rec-qwen-fusion && exit 1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader -i 0
REMOTE
```

Acceptance gates:

- fused confirmation median is within 5% of the first fused run at every
  batch size;
- every offline row has the expected success count and zero failures;
- fused/unfused top-1 token-sequence match is 100%;
- mean top-256 set overlap is at least 0.95;
- all six online files contain 64 completions and no errors;
- the fused Nsight trace contains Q/K norm, RoPE, add-RMSNorm, and MLP fusion
  kernels;
- the strict-unfused trace contains none of those fused kernels.

Reference values from the first accepted run:

| Batch | Fused wall | Strict-unfused wall | Latency speedup |
| ----: | ---------: | ------------------: | --------------: |
|     1 |  19.479 ms |           24.412 ms |          1.253x |
|     2 |  34.132 ms |           40.889 ms |          1.198x |
|     4 |  61.687 ms |           73.434 ms |          1.190x |
|     8 | 114.797 ms |          145.136 ms |          1.264x |

Reference online medians are 49.776 req/s and 79.752 ms p50 fused versus
43.323 req/s and 91.419 ms p50 strict-unfused. Investigate any successful
headline row that differs from its reference by more than 20%. Do not discard
or replace an outlier silently.

## 6. Interpret online tail events

The accepted reference run observed a repeatable strict-unfused round-3 p99
stall. A diagnostic GC callback tied it to a 201.4 ms generation-2 Python
collection, while the GPU/operator medians remained stable. Consequently:

- use median-of-three for headline online comparisons;
- report every round, including p50/p90/p99, req/s, and failures;
- preserve the original outlier and server logs;
- keep any GC-disabled or otherwise modified diagnostic separate from the
  headline run;
- do not attribute an uncovered Nsight GPU gap entirely to Python without
  CPU-stack evidence.

If another tail event occurs, inspect the server log, client JSONL, request
latencies, graph-capture counters, GPU errors, and host contention before
rerunning. A clean rerun is supporting evidence, not permission to delete the
first run.

## 7. Failure recovery

| Symptom                                          | Action                                                                                                |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Image digest cannot be pulled                    | Stop. Verify ACR connectivity and digest; do not fall back to a mutable tag.                          |
| Missing/incompatible Python or CUDA package      | Mark image validation failed. Do not install or upgrade inside the image.                             |
| GPU-busy guard fails                             | Identify the process owner and wait. Do not kill an unrelated process.                                |
| Source commit mismatch                           | Stop, update the local worktree, push `ty_git`, then fast-forward the remote checkout.                |
| Kernel backend or fusion test fails              | Stop before measurement and retain `kernel_backends.txt`, `fusion_tests.txt`, and container logs.     |
| Server does not become ready                     | Inspect `logs/server_<label>.log`, port 8000, model path, and GPU memory.                             |
| Partial JSON exists after failure                | Treat the phase as incomplete regardless of recorded wall time; verify scheduler success counts.      |
| Nsight export fails                              | Preserve `.nsys-rep` and profiler logs, then rerun only export if the report is complete.             |
| Fused kernel missing or strict trace contains it | Mark that configuration invalid; requested environment policy alone is not dispatch proof.            |
| Fused confirmation drift exceeds 5%              | Check GPU contention, clocks, thermals, and background processes, then rerun the full A/B/A sequence. |
| GitHub fetch reports transient TLS/HTTP2 errors  | Keep the worktree unchanged, force Git HTTP/1.1, and retry the fetch.                                 |

The launcher cleanup trap removes its named container. After an interrupted
SSH session, confirm whether the nested container is still active before
starting anything else.

## 8. Preserve and publish artifacts

Keep the complete raw result directory on DSW. For local analysis, copy it to
the gitignored TorchEasyRec `experiments/` directory without requiring remote
`rsync`:

```bash
LOCAL=/mnt/disk/hongsheng.jhs/Workspace/tzrec-rfc-llm4rec-bench
REMOTE=/mnt/data/hongsheng.jhs/bench_results/<result-directory-name>
mkdir -p "$LOCAL/experiments/<result-directory-name>"
ssh ty_l20n_dev "tar -C '$REMOTE' -cf - ." \
  | tar -C "$LOCAL/experiments/<result-directory-name>" -xf -
```

Commit a curated report plus selected machine-readable artifacts under:

```text
rfc/llm4rec/inference_bench/results/
```

Preserve at least:

- runtime, GPU, image digest, and source commit manifests;
- compact summary and acceptance results;
- offline, ablation, correctness, and online JSON/JSONL;
- generated Nsight breakdown JSON/Markdown;
- logs needed to substantiate a diagnosed anomaly.

Keep complete `.nsys-rep`, SQLite traces, image inspection, and verbose logs
in the remote raw directory unless they are specifically needed for review.

Before publishing:

```bash
cd /mnt/disk/hongsheng.jhs/Workspace/tzrec-rfc-llm4rec-bench
python rfc/llm4rec/inference_bench/fusion/summarize.py \
  experiments/<result-directory-name>
python rfc/llm4rec/inference_bench/fusion/analyze_nsys.py \
  --fused experiments/<result-directory-name>/nsys/fused.sqlite \
  --unfused experiments/<result-directory-name>/nsys/unfused.sqlite \
  --output-json experiments/<result-directory-name>/nsys_breakdown.json \
  --output-markdown experiments/<result-directory-name>/nsys_breakdown.md
pre-commit run -a
git status --short
```

Use a `[perf]` commit subject, push `rfc_llm4rec_inference_bench` to
`ty_git`, and fast-forward `/mnt/data/hongsheng.jhs/TorchEasyRec`. Do not
create a PR unless explicitly requested.

## Final checklist

- [ ] Local changes flowed through worktrees, commits, `ty_git`, and remote
  fast-forwards.
- [ ] The container image is pinned and its digest recorded.
- [ ] Source commits and runtime versions are recorded.
- [ ] No package was installed on the host or in the benchmark image.
- [ ] Exactly one GPU was visible inside the nested benchmark container.
- [ ] Offline A/B/A, ablations, correctness, Nsight, and online phases all
  completed.
- [ ] Success counts, parity, overlap, dispatch, and drift gates passed.
- [ ] Per-round online variance and any tail anomaly are reported.
- [ ] Full raw artifacts remain on DSW; curated artifacts are committed.
- [ ] No benchmark container, server, or GPU process remains.
- [ ] Both local and remote worktrees are clean after publication.
