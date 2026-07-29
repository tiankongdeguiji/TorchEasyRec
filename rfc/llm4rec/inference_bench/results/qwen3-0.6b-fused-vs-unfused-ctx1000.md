# Qwen3-0.6B fused versus strict-unfused SID-GR inference

Date: 2026-07-29

## Outcome

On one RTX PRO 5000 72 GB Blackwell (SM120), Qwen3-0.6B bf16, context
1000, beam 256, and three generated tokens, the current Qwen3 fusion policy
improves offline wall latency by **1.190x-1.264x** across batch sizes 1-8.
At the production-shaped online point (64 requests, maximum concurrency 4),
the median-of-three result is:

| Configuration      |      req/s |   p50 (ms) |   p90 (ms) |   p99 (ms) |
| ------------------ | ---------: | ---------: | ---------: | ---------: |
| Current auto-fused | **49.776** | **79.752** | **83.423** | **85.838** |
| Strict unfused     |     43.323 |     91.419 |     95.870 |     97.892 |
| Fusion benefit     | **1.149x** | **1.146x** | **1.149x** | **1.140x** |

Latency speedup is `unfused_ms / fused_ms`; throughput speedup is
`fused_req_s / unfused_req_s`. All 384 headline online requests and all
offline requests completed without errors or timeouts.

RoPE is the dominant individual fusion: disabling it adds 8.421 ms at batch
4 and explains 71.7% of the full 11.747 ms strict-unfused gap. Packed
SiLU-multiply explains another 15.4%. Q/K norm and residual-add-plus-RMSNorm
are individually small at this model and shape.

## Scope and controls

This compares `Qwen3GRModel` with itself. SGLang supplies runtime kernels but
is not a competing server in this experiment. Both sides keep the following
identical:

- Qwen3-0.6B bf16 weights, packed QKV and gate/up projections;
- FlashAttention prefill and `gr_decode_atten` beam decode attention;
- beam top-k, ContextKV/BeamKV layouts, scheduling, and CUDA graphs;
- synthetic deterministic prompts, context 1000, beam 256, and disabled
  prefix/radix caches;
- `GR_INFERENCE_DECODE_NEXT_INPUT_NORM_FUSION=0`.

`GR_INFERENCE_QWEN3_DISABLE_FUSIONS=all` is the strict-unfused side. It
disables fused in-place Q/K normalization, fused Q/K RoPE, residual-add plus
RMSNorm, and packed SiLU-multiply. It retains standalone RMSNorm and the same
GEMMs and attention kernels. The empty setting preserves current automatic
dispatch.

The exploratory `max-fused` row changes only
`GR_INFERENCE_DECODE_NEXT_INPUT_NORM_FUSION=1`; it is not a headline
baseline.

## Reproducibility

| Item                             | Value                                                                                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Host                             | `ty_l20n_dev`, DSW Docker-in-Docker                                                                                                              |
| GPU                              | NVIDIA RTX PRO 5000 72 GB Blackwell, SM120                                                                                                       |
| Driver / CUDA                    | 580.126.09 / CUDA 13.0                                                                                                                           |
| Image                            | `mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-test@sha256:c1cf5a01e53113707f6fc67f4cdc8e223c9d3fe3cf2b5a9c657084c45e213353` |
| Image ID                         | `sha256:8c77349eb73ff953a9256e84ce84a243396bb1c7506cf401e34d9d1a0e33e469`                                                                        |
| Python / PyTorch                 | 3.12.3 / 2.11.0+cu130                                                                                                                            |
| FlashInfer / Cutlass DSL / Quack | 0.6.12 / 4.5.2 / 0.5.0                                                                                                                           |
| Transformers / SGLang kernel     | 5.12.1 / 0.4.4                                                                                                                                   |
| recsys-examples                  | `f8b71ff58dbb6eb4bc685712d7c4540444d1c7a4` (`llm4rec/qwen3-fusion-bench`)                                                                        |
| SGLang source                    | `2aac32adcf839bfd3f8d02d6cc4a840d9d82b6a9`                                                                                                       |
| Full remote artifacts            | `/mnt/data/hongsheng.jhs/bench_results/qwen06b_fusion_20260729_112520`                                                                           |

The pinned container saw exactly one idle GPU. Kernel selection, fusion-policy
tests, single-layer prefill tests, and the tiny real-weight tool all passed
before measurement (68 tests). No package was installed or upgraded in the
benchmark image.

## Correctness and dispatch gates

Fused and strict-unfused executions used identical deterministic input IDs:

| Gate                            |       Result |
| ------------------------------- | -----------: |
| Top-1 full token-sequence match | **1.000000** |
| Mean top-256 set overlap        | **0.964844** |
| Failed/timed-out requests       |        **0** |

Nsight kernel names independently verify effective, rather than merely
requested, policy:

| Fusion family                | Fused launches | Strict-unfused launches | Gate |
| ---------------------------- | -------------: | ----------------------: | ---- |
| SGLang fused Q/K norm        |             84 |                       0 | PASS |
| Fused RoPE                   |             84 |                       0 | PASS |
| FlashInfer fused add-RMSNorm |            111 |                       0 | PASS |
| Packed SiLU-multiply         |             84 |                       0 | PASS |

## Offline binary A/B/A

Each process performs two warmups and ten measured repetitions. The fused
measurement combines the first and confirmation sample sets (20 samples);
the strict-unfused measurement is its ten-sample median.

| Batch | Fused prefill / decode / wall (ms) | Unfused prefill / decode / wall (ms) | Fused / unfused req/s | Latency speedup | Fused confirmation drift |
| ----: | ---------------------------------: | -----------------------------------: | --------------------: | --------------: | -----------------------: |
|     1 |        8.601 / 10.516 / **19.479** |             11.032 / 13.017 / 24.412 |       51.337 / 40.963 |      **1.253x** |                   0.035% |
|     2 |       14.690 / 18.801 / **34.132** |             18.267 / 22.017 / 40.889 |       58.596 / 48.913 |      **1.198x** |                   0.257% |
|     4 |       25.694 / 35.023 / **61.687** |             32.708 / 39.705 / 73.434 |       64.844 / 54.471 |      **1.190x** |                   0.161% |
|     8 |      49.362 / 63.664 / **114.797** |            72.074 / 71.285 / 145.136 |       69.688 / 55.121 |      **1.264x** |                   0.029% |

All confirmation drifts are far below the 5% acceptance threshold. These
values also reproduce the prior SM120 fused reference
(19.5/33.9/61.4/114.0 ms).

## Batch-4 ablations

The causal baseline is the 61.687 ms combined fused A/B median. “Share of
strict gap” divides the isolated delta by the 11.747 ms difference between
strict-unfused and fused. Isolated contributions need not sum exactly because
the paths interact.

| Disabled family           | Prefill (ms) | Decode (ms) |  Wall (ms) |                Delta | Share of strict gap |
| ------------------------- | -----------: | ----------: | ---------: | -------------------: | ------------------: |
| None (current auto-fused) |       25.694 |      35.023 | **61.687** |                    - |                   - |
| Q/K norm                  |       25.481 |      35.653 |     62.052 |       +0.366 (+0.6%) |                3.1% |
| RoPE                      |       30.667 |      38.448 |     70.108 |  **+8.421 (+13.7%)** |           **71.7%** |
| Add + RMSNorm             |       25.897 |      35.105 |     61.947 |       +0.260 (+0.4%) |                2.2% |
| MLP activation            |       26.796 |      35.683 |     63.494 |   **+1.807 (+2.9%)** |           **15.4%** |
| All four (strict unfused) |       32.708 |      39.705 |     73.434 | **+11.747 (+19.0%)** |                100% |
| Max-fused next-input norm |       25.812 |      35.050 |     61.753 |       +0.067 (+0.1%) |                 n/a |

The remaining 7.6% of the strict gap is interaction and measurement residual.
The decode-next-input norm optimization is within noise at this point.

## Nsight Systems operator distribution

The traces cover one batch-4 measured execution. Nsight tracing changes
absolute wall time slightly, so the ten-repeat offline runs are authoritative
for latency; the trace is authoritative for kernel identity, launch count,
and time distribution.

| Metric               |     Fused | Strict unfused |              Delta |
| -------------------- | --------: | -------------: | -----------------: |
| GPU trace span       | 61.837 ms |      73.064 ms |         +11.227 ms |
| GPU busy union       | 58.190 ms |      68.376 ms |         +10.186 ms |
| CUDA kernel time sum | 58.128 ms |      68.286 ms |         +10.158 ms |
| Uncovered GPU gap    |  3.647 ms |       4.688 ms |          +1.041 ms |
| Kernel launches      | **1,261** |      **3,472** | **+2,211 (2.75x)** |
| Explicit memcpy      |  0.014 ms |       0.014 ms |                 ~0 |
| Explicit memset      |  0.048 ms |       0.075 ms |          +0.027 ms |

About 90.7% of the trace-span increase is additional GPU-busy time; 9.3% is
additional uncovered gap between GPU activities. The latter includes Python
launch/orchestration and synchronization, but the trace did not sample CPU
stacks, so it should not be labeled pure Python time.

| Kernel family                        | Fused ms / launches | Strict unfused ms / launches |      Delta |
| ------------------------------------ | ------------------: | ---------------------------: | ---------: |
| GEMM                                 |        29.460 / 339 |                 29.358 / 339 |     -0.102 |
| Decode attention (`gr_decode_atten`) |          4.613 / 56 |                   4.651 / 56 |     +0.038 |
| Prefill attention                    |          3.846 / 28 |                   3.911 / 28 |     +0.065 |
| Beam top-k kernels                   |         6.334 / 101 |                  6.323 / 101 |     -0.011 |
| MLP activation                       |      **0.686 / 84** |              **2.532 / 168** | **+1.847** |
| Generic elementwise fallback         |          6.546 / 86 |               12.617 / 1,541 | **+6.071** |
| Layout/copy kernels                  |         3.411 / 200 |                  7.421 / 872 | **+4.009** |
| Norm kernels                         |         1.393 / 255 |                  1.155 / 339 |     -0.238 |
| Fused RoPE kernel                    |          0.510 / 84 |                        0 / 0 |     -0.510 |

The GEMM, attention, and top-k rows are effectively invariant, as intended.
The strict path’s extra cost is the 2,211 additional launches and roughly
10.1 ms spent mainly in compositional elementwise and layout kernels.
Generic Torch kernel names cannot reliably separate RoPE from normalization,
so the single-family ablations—not heuristic Nsight naming—are the causal
per-family attribution.

## Online rounds and variance

Each row is a separate 64-request client process against a persistent server,
with maximum concurrency 4, unlimited request rate, eight warmup requests,
and three output tokens.

| Configuration / round |  req/s | p50 (ms) | p90 (ms) |    p99 (ms) | Within-round latency stddev (ms) |
| --------------------- | -----: | -------: | -------: | ----------: | -------------------------------: |
| Fused 1               | 49.776 |   79.752 |   82.423 |      85.651 |                            2.449 |
| Fused 2               | 49.830 |   79.622 |   83.423 |      85.838 |                            2.829 |
| Fused 3               | 48.874 |   80.915 |   85.775 |      87.084 |                            3.093 |
| Strict unfused 1      | 43.323 |   91.267 |   95.870 |      97.306 |                            2.776 |
| Strict unfused 2      | 43.350 |   91.419 |   95.599 |      97.892 |                            2.860 |
| Strict unfused 3      | 37.445 |   92.013 |   98.861 | **319.719** |                       **55.030** |

Fused round-to-round throughput coefficient of variation is 1.09%; p50 CV is
0.89%. Strict-unfused p50 remains stable (0.43% CV), but its third-round
throughput drops 13.6% and p99 rises by 227% relative to the median.
Median-of-three protects the headline result, but this repeatable tail event
required investigation.

### Strict-path Python GC tail event

A fresh three-round rerun reproduced the shape: rounds 1-2 were
43.19-43.50 req/s with 97.1-97.8 ms p99; round 3 fell to 37.18 req/s with
332.58 ms p99. Server logs contained no failure, timeout, graph capture, or
GPU error.

A third run added Python GC callbacks without changing inference. During the
round-3 four-request stall, generation-2 cyclic GC paused the process for
**201.433 ms** and collected 309 objects. The measured round had 291.924 ms
p99. A diagnostic run with cyclic GC disabled after process start removed the
periodic stall:

| GC-disabled diagnostic round |  req/s | p50 (ms) | p90 (ms) | p99 (ms) |
| ---------------------------: | -----: | -------: | -------: | -------: |
|                            1 | 43.405 |   91.920 |   94.829 |   96.874 |
|                            2 | 43.773 |   90.207 |   93.633 |   96.605 |
|                            3 | 43.781 |   90.941 |   94.451 |   97.143 |

These diagnostic values are not substituted into the headline result because
disabling GC changes server runtime policy. They show that the long tail is
host-side cyclic collection triggered by the allocation-heavy strict path,
not a fusion-kernel or GPU-clock regression. Production serving should avoid
unbounded automatic generation-2 collection on the request path—for example,
collect at startup/drain boundaries after verifying object-lifetime behavior.

## Conclusions

1. Keep the current automatic Qwen3 fusions enabled. They produce a
   repeatable 19-26% offline latency reduction and about 15% online
   throughput/median-latency benefit at this SID-GR operating point.
1. Prioritize RoPE fusion portability and regression coverage. It contributes
   roughly 72% of the strict gap; MLP activation is the second meaningful
   family at roughly 15%.
1. Fusing Q/K norm and add-RMSNorm is still correct and reduces launches, but
   each contributes less than 1% wall latency for Qwen3-0.6B at context 1000.
1. The attention-bandwidth design is unaffected: fused and unfused
   `gr_decode_atten` times differ by only 0.038 ms, and GEMM/top-k time is
   unchanged.
1. Fix the HTTP server’s automatic-GC tail exposure separately from operator
   fusion. It does not change the robust A/B conclusion, but it can turn a
   ~92 ms request batch into a ~320 ms p99 event.

Selected raw artifacts are under
`results/sm120-qwen3-0.6b-fusion/`. The complete Nsight reports, SQLite
exports, logs, image inspection, and diagnostic runs remain in the full remote
artifact directory.
