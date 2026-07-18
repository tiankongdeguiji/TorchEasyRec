# Performance breakdown: sid-gr-inference vs SGLang beam search

**Workload:** Qwen3-0.6B bf16, context 5000 tokens, batch 4 (= online max_concurrency 4),
beam width 256, 3 SID output tokens, cross-request caches disabled on both engines.
**Hardware:** 1x RTX PRO 5000 72GB Blackwell (SM120), driver 580.126.09, CUDA 13.
**Method:** the harness's own nsys flow (`scripts/run_short_context_nsys_compare.sh`,
`--capture-range=cudaProfilerApi`, `--cuda-graph-trace=node`, NVTX stages) + the GR
module profiler (`--profile-continuous-decode --profile-detail fine`) + sqlite kernel
timeline analysis. Raw traces and the auto-generated report:
`sm120-qwen3-0.6b/nsys_breakdown_ctx5000_b4.md`.

**Online reference points (bench_serving, steady state):**
sid-gr p50 **248.5 ms** / 16.11 req/s; SGLang fork p50 **472.9 ms** / 8.43 req/s (1.91x).

## 1. Executive summary

Both engines are **operator-bound, not Python-bound**: >95% of each engine's wall is
GPU kernel execution, and beam decode runs under CUDA graphs on both sides. GEMM time,
prefill-attention time, and scheduling efficiency are near-identical across the two
engines. **The entire ~2x gap is one kernel class**: SGLang's per-beam-row decode
attention (`BatchDecodeWithPagedKVCacheKernel`, 257.7 ms) vs sid-gr's beams-in-query
decode attention (`gr_decode_atten`, 18.8 ms) - a **13.7x** difference caused by
re-reading the shared context KV once per beam row instead of once per user.

## 2. End-to-end decomposition

### sid-gr-inference: 248.5 ms online p50

| component                               |    time | share | nature                                                                                                        |
| --------------------------------------- | ------: | ----: | ------------------------------------------------------------------------------------------------------------- |
| client tokenize + HTTP + admission      |  ~17 ms |  6.8% | Python/network (off-GPU)                                                                                      |
| engine prefill (4x5000 tok)             | ~181 ms | 72.8% | GPU operators (95.4% kernel-busy; CPU overhead 8.6 ms)                                                        |
| engine decode (3 beam steps, 1024 rows) |  ~49 ms | 19.8% | CUDA-graph replays; Python bookkeeping \<1 ms measured (topk-indices 0.25, batch-build 0.20, kv-scatter 0.16) |
| scheduler tick                          | ~1.5 ms |  0.6% | Python                                                                                                        |

### SGLang fork: 472.9 ms online p50

| component                         |    time | share | timeline window |
| --------------------------------- | ------: | ----: | --------------- |
| client tokenize + HTTP + queue    |   ~9 ms |  1.9% | off-GPU         |
| true prefill (4x5000 tok)         | ~120 ms | 25.4% | 0-120 ms        |
| beam-expansion step (1 -> 256)    |  ~46 ms |  9.7% | 120-166 ms      |
| decode steps 2-3 (1024 beam rows) | ~285 ms | 60.3% | 166-450 ms      |
| final log_softmax + top-k tail    |  ~10 ms |  2.1% | 450-462 ms      |

GPU-busy fraction of the engine window: sid-gr **86.8%** (226.6/261 ms, and the real
graphed decode is tighter than the profiled eager one), SGLang **96.4%** (445.8/462 ms).
CPU gaps >50 us: 7.4 ms vs 7.9 ms. Kernel launches: 1261 vs 1925 (+29 graph launches).

## 3. GPU operator distribution

### sid-gr-inference (226.6 ms kernel total)

| bucket                  |   ms |     % | main kernels                                         |
| ----------------------- | ---: | ----: | ---------------------------------------------------- |
| GEMM / linear           | 90.5 | 39.9% | cutlass bf16 s16816gemm (MLP 128x256: 52.9/58 calls) |
| prefill flash attention | 61.5 | 27.1% | pytorch_flash flash_fwd, 28 calls                    |
| decode beam attention   | 18.8 |  8.3% | gr_decode_atten Sm120 CuTe, 56 calls                 |
| QKV pack + qk_norm_rope | 13.8 |  6.1% | flashinfer fused                                     |
| memcpy / fill           |  9.7 |  4.3% |                                                      |
| SiLU MLP activation     |  8.5 |  3.7% |                                                      |
| elementwise / copies    |  8.0 |  3.5% |                                                      |
| KV-cache writes         |  7.3 |  3.2% | write_packed_qkv_prefill_kv                          |
| top-K / beam selection  |  6.4 |  2.8% |                                                      |
| RMSNorm                 |  4.6 |  2.0% |                                                      |
| logits log_softmax      | 0.03 |   ~0% | (lm_head GEMM counted above)                         |

### SGLang fork (445.8 ms kernel total)

| bucket                                     |    ms |     % | main kernels                                                      |
| ------------------------------------------ | ----: | ----: | ----------------------------------------------------------------- |
| decode attention                           | 257.7 | 57.8% | BatchDecodeWithPagedKVCacheKernel, 56 calls (28 layers x 2 steps) |
| GEMM / linear                              |  91.5 | 20.5% | cutlass bf16 s16816gemm                                           |
| prefill attention                          |  59.5 | 13.3% | BatchPrefillWithRaggedKV 41.8 + paged step-1 17.7                 |
| norm / rope / act / kv-store / elementwise |   ~18 |  4.0% |                                                                   |
| sampling (radix top-k + LogSoftMax)        |   8.4 |  1.9% |                                                                   |
| memcpy / other                             |   ~10 |  2.2% |                                                                   |

## 4. Head-to-head on identical work

| kernel class          |       sid-gr |       SGLang |     ratio |
| --------------------- | -----------: | -----------: | --------: |
| decode beam attention |  **18.8 ms** | **257.7 ms** | **13.7x** |
| prefill attention     |      61.5 ms |      59.5 ms |       ~1x |
| GEMMs                 |      90.5 ms |      91.5 ms |       ~1x |
| everything else       |       ~56 ms |       ~37 ms |      0.7x |
| **kernel total**      | **226.6 ms** | **445.8 ms** | **1.97x** |

**Root cause.** The fork's beam search materializes each of the 1024 beams as an
independent sequence; its decode attention therefore re-reads the shared ~5000-token
context KV once per beam row, per layer, per step (~12 MB x 1024 rows per layer).
`gr_decode_atten` places the 256 beams of a request in the query dimension of one
attention call, so the context KV is read once per user per layer per step and
amortized across all beams (the "bandwidth sharing" mechanism of RFC section 10,
measured earlier at 135x at the isolated-kernel level).

## 5. Scaling implications

- The gap grows with context length and beam width (decode-attention KV traffic scales
  with ctx x beams for SGLang, ctx for sid-gr): measured offline ratios rise from
  1.80x at ctx1000 to 2.00-2.11x at ctx5000 (0.6B).
- The gap shrinks as the model grows (GEMMs - identical on both engines - dilute the
  attention delta): 1.7B ratios are 1.45-1.61x vs 0.6B's 1.80-2.11x.
- Neither engine leaves meaningful Python overhead on the table at bs4; optimizing
  hosts/schedulers would not close SGLang's gap - only a beams-in-query decode
  attention would.

## 6. Caveats

- Profiled runs inflate absolute walls (graph-node tracing; the GR module profiler
  forces eager decode + per-module sync: profiled 262.7/74.2 ms vs benchmark
  231.5/49.3 ms). Use profiles for distribution, benchmarks for totals.
- Buckets from the auto-report classify pytorch_flash prefill under "other"; the
  tables above re-classify by kernel identity.
- Single cell profiled (ctx5000/bs4 = the online operating point). Other cells scale
  per section 5; re-run `run_short_context_nsys_compare.sh` with CONTEXT_LEN/REQUESTS
  overrides to reproduce any cell.
