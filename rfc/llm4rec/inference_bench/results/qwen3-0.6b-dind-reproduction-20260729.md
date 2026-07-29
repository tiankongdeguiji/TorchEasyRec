# Qwen3-0.6B SM120 Docker-in-Docker reproduction

## Outcome

The Qwen3-0.6B bf16, context-1000, beam-256 benchmark reproduced on
`ty_l20n_dev`. The DSW pod ran the TorchEasyRec 1.3.0 CUDA 13 image and used
its built-in Docker proxy to execute the benchmark image:

`mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-test:sglang-dev-cu13`

The inner image digest was
`sha256:c1cf5a01e53113707f6fc67f4cdc8e223c9d3fe3cf2b5a9c657084c45e213353`.
It provided Torch 2.11.0+cu130, FlashInfer 0.6.12, Cutlass DSL 4.5.2,
Quack 0.5.0, Transformers 5.12.1, SGLang kernel 0.4.4, and FlashAttention
4.0.0b15.

The nested container saw one RTX PRO 5000 72GB Blackwell GPU at compute
capability 12.0. `gr_decode_atten` backend selection, its SM120 smoke and
correctness tests, and the SGLang scored-beam gate all passed before timing.

## Source and mount contract

| Purpose                    | Absolute path                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------- |
| recsys-examples            | `/mnt/data/hongsheng.jhs/recsys-examples`                                           |
| sid-gr-inference           | `/mnt/data/hongsheng.jhs/recsys-examples/examples/sid-gr-inference`                 |
| decode-attention kernel    | `/mnt/data/hongsheng.jhs/recsys-examples/corelib/gr_decode_atten`                   |
| SGLang beam fork           | `/mnt/data/hongsheng.jhs/sglang`                                                    |
| model                      | `/mnt/data/hongsheng.jhs/models/Qwen3-0.6B`                                         |
| complete remote result set | `/mnt/data/hongsheng.jhs/bench_results/acr_sglang_dev_cu13_qwen06b_20260729_092142` |

The run used recsys-examples commit
`f7d61a245a7c045c2d617dbd95d6b3f4f8a44226` and SGLang commit
`2aac32adcf839bfd3f8d02d6cc4a840d9d82b6a9`. Repositories and the model
were mounted read-only; only the timestamped result directory was writable.

DSW exposes Docker at `/etc/dsw/runtime/export_bin/docker` through
`/var/docker/proxy/docker-proxy.sock`. Its authorization plugin rejects
`--ipc=host`, while GPU requests, host networking, NAS bind mounts, and
`--shm-size=32g` are allowed. The benchmark therefore used a private 32GB
IPC namespace. No Python packages were installed on the DSW host or inside
the benchmark image.

## Offline performance

Method: deterministic synthetic token IDs, context 1000, beam 256, three
output tokens, prefix/radix caches disabled, one warmup, and three measured
runs. `batch` is the number of requests submitted together inside each
embedded engine, not HTTP client concurrency.

| Batch | GR wall ms | GR req/s | GR prefill ms | GR decode ms | SGLang wall ms | SGLang req/s | SGLang/GR |
| ----: | ---------: | -------: | ------------: | -----------: | -------------: | -----------: | --------: |
|     1 |     19.521 |   51.226 |         8.623 |       10.527 |         36.607 |       27.317 |    1.875x |
|     2 |     34.174 |   58.525 |        14.715 |       18.834 |         60.798 |       32.896 |    1.779x |
|     4 |     61.898 |   64.623 |        25.731 |       35.057 |        111.166 |       35.982 |    1.796x |
|     8 |    114.843 |   69.660 |        49.279 |       63.630 |        215.809 |       37.070 |    1.879x |

Every GR cell reported `succeeded_requests == requests` and zero failures.
Each SGLang cell contains three successful measured runs. Compared with the
earlier native-SGLang-image SM120 run, absolute wall-time deviations were:

| Batch | GR deviation | SGLang deviation |
| ----: | -----------: | ---------------: |
|     1 |       +0.34% |           +2.16% |
|     2 |       +0.76% |           -0.64% |
|     4 |       +0.89% |           +0.52% |
|     8 |       +0.72% |           -0.71% |

All values are well inside the 20% reproduction threshold. The maximum
absolute deviation was 2.16%.

## Online steady state

Method: context 1000, beam 256, three output tokens, 64 requests, unlimited
arrival rate, and maximum client concurrency 4. The GR server used no client
warmup requests because its startup captures its serving graphs. Each
SGLang round used eight warmup requests. All measured rounds completed 64/64
requests and contained no non-empty errors.

| Engine |          Round |  Req/s | p50 ms | p90 ms | p99 ms |
| ------ | -------------: | -----: | -----: | -----: | -----: |
| GR     |              1 | 49.145 |  80.35 |  84.55 |  90.80 |
| GR     |              2 | 49.929 |  79.27 |  82.40 |  85.17 |
| GR     |   3, straggler | 40.562 |  79.47 |  83.75 | 372.13 |
| GR     | 4, replacement | 48.318 |  80.54 |  88.32 | 101.82 |
| SGLang |              1 | 33.827 | 116.16 | 118.57 | 119.58 |
| SGLang |              2 | 33.693 | 117.12 | 119.71 | 121.67 |
| SGLang |              3 | 33.847 | 116.19 | 118.11 | 121.02 |

GR round 3 contained a single host/scheduler straggler: p50 remained stable
at 79.47ms while p99 increased to 372.13ms. The server remained ready with
zero worker errors and zero failed requests after the round. The replacement
round is retained alongside the outlier rather than overwriting it.

Using GR rounds 1, 2, and 4 and all three SGLang rounds:

| Engine | Median req/s | Median p50 ms | Median p90 ms | Median p99 ms |
| ------ | -----------: | ------------: | ------------: | ------------: |
| GR     |       49.145 |         80.35 |         84.55 |         90.80 |
| SGLang |       33.827 |        116.19 |        118.57 |        121.02 |

SGLang/GR p50 latency is 1.446x; GR/SGLang throughput is 1.453x. Relative to
the previous SM120 online result, GR throughput changed by about -2.1% and
GR p50 by +2.1%; SGLang throughput changed by about -0.5% and p50 by +0.3%.

## Conclusion

The nested-Docker path reproduces the prior SM120 result. Docker-in-Docker
does not introduce a material performance shift in the measured inference
windows: all offline cells remain within 2.2%, online medians remain within
2.1%, and GR retains a roughly 1.45x advantage over the SGLang beam fork at
the online operating point.

The complete logs and per-cell JSON files remain in the remote result
directory above. The committed subset under `sm120-qwen3-0.6b-dind/`
contains the runtime manifest, preflight outputs, offline summaries, online
JSONL files, and computed online summary.
