# Three-way comparison: sid-gr-inference vs SGLang beam search

Qwen3-1.7B bf16, beam 256, 3 SID output tokens. Same harness, three environments:

|                        | A10 (local docker)                                           | SM120 (this run)                                                    | H100 (upstream README)   |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------- | ------------------------ |
| GPU                    | 1x A10 23GB GDDR6, SM86                                      | 1x RTX PRO 5000 72GB Blackwell, SM120                               | H100 80GB HBM3, SM90     |
| Env                    | `sidgr-bench:cu13` image + cuda-compat-13 on r535            | native host (pod IS lmsysorg/sglang:dev-cu13), driver 580.126.09    | lmsysorg/sglang:dev-cu13 |
| Kernel pins            | cutlass-dsl 4.5.1/cu13, quack 0.4.1, flashinfer per image    | cutlass-dsl 4.5.2, quack 0.5.0, flashinfer 0.6.12, torch 2.11+cu130 | upstream pins of the day |
| GR prefill CUDA graphs | OFF (10.9GB private pools did not fit)                       | ON (faithful)                                                       | ON                       |
| SGLang engine knobs    | mem_fraction 0.60, slots 2100, ctx-len 8192 (forced by 23GB) | defaults                                                            | defaults                 |

## Offline performance (REPEAT=3 medians; ratio = SGLang/GR, >1 means GR faster)

|  ctx | batch | A10 ratio | SM120 ratio | H100 ratio | SM120 GR ms | H100 GR ms |
| ---: | ----: | --------: | ----------: | ---------: | ----------: | ---------: |
| 1000 |     1 |     1.428 |       1.491 |      1.903 |        35.2 |       17.6 |
| 1000 |     2 |     1.484 |       1.455 |      2.086 |        62.4 |       27.8 |
| 1000 |     4 |     1.464 |       1.451 |      2.143 |       118.3 |       47.7 |
| 1000 |     8 |     1.460 |       1.453 |      2.138 |       230.9 |       93.2 |
| 5000 |     1 |     1.535 |       1.607 |      2.238 |       108.6 |       42.3 |
| 5000 |     2 |     1.566 |       1.563 |      2.217 |       219.3 |       80.9 |
| 5000 |     4 |     1.263 |       1.532 |      2.269 |       450.1 |      154.2 |
| 5000 |     8 |     1.501 |       1.525 |      2.226 |       896.4 |      307.9 |

- GR wins ALL cells in all three environments. SM120 ratios 1.451-1.607 (median 1.49);
  every cell verified `succeeded_requests == requests` on the GR side.
- SM120 absolute walls ~2.9x H100 (bandwidth-bound: GDDR7 vs HBM3) and ~3x faster than A10.
- The ratio gap vs H100 persists even with prefill graphs ON here, so the A10 run's
  graph-off handicap explains only part of it; the remainder is arch/tuning
  (Sm120 kernel maturity vs Sm90) and the newer SGLang-side wheels.

## Offline accuracy (top1 = rank-1 beam token-tuple identity vs SGLang)

|  ctx | batch | A10 top1 | SM120 top1 | SM120 topK overlap |
| ---: | ----: | -------: | ---------: | -----------------: |
| 1000 |     1 |    1.000 |      1.000 |              0.965 |
| 1000 |     2 |    1.000 |      0.500 |              0.945 |
| 1000 |     4 |    0.750 |      1.000 |              0.962 |
| 1000 |     8 |    0.875 |      0.875 |              0.963 |
| 5000 |     1 |    1.000 |      1.000 |              0.961 |
| 5000 |     2 |    1.000 |      1.000 |              0.975 |
| 5000 |     4 |    1.000 |      1.000 |              0.960 |
| 5000 |     8 |    1.000 |      1.000 |              0.959 |

28/30 (A10) and 28/30 (SM120) requests match exactly; every mismatch is a near-tie
rank-1/rank-2 flip with ~0.95+ topK overlap (bf16 numerics differ per arch/kernels;
upstream's own H100 online smoke shows the same effect at 58/64).

## Online serving (bench_serving; ctx 5000, beam 256, 64 requests, max_concurrency 4)

|               | GR req/s | GR median/p99 ms | SGLang req/s | SGLang median/p99 ms | ratio |
| ------------- | -------: | ---------------: | -----------: | -------------------: | ----: |
| A10           |    2.959 |      1346 / 1365 |        1.927 |          2074 / 2089 | 1.54x |
| SM120         |    8.534 |        468 / 475 |        5.692 |            701 / 704 | 1.50x |
| H100 (README) |        - |                - |            - |                    - | 1.85x |

Both sides 64/64 successful on SM120; steady-state (3rd round) reported.

## SM120/Blackwell validation

First known validation of `gr_decode_atten` on compute capability 12.0: backend
selection picks the fused "dsl" path (Sm120 kernel class), and both the smoke and
correctness pytest suites pass before any timing was taken.

## Operational fixes required (committed on the bench branches)

1. `HF_ENDPOINT=https://hf-mirror.com` for the online client - bench_serving's
   "random" dataset secretly downloads ShareGPT from huggingface.co (blocked on the
   pod); plain offline mode cannot work.
1. Probe `/generate`, never `/health_generate` - the beam fork reports a false 503
   forever in server mode (detokenizer heartbeat never starts) while serving fine.
1. `WARMUP_REQUESTS=8` (vs README's 16): N concurrent warmups need N\*(beam+1)
   request slots; 16 -> 4112 > the fork's 4096-slot clamp, and slot exhaustion is a
   FATAL scheduler error. 16 fits H100 only by scheduling race.
1. Per-cell `succeeded_requests` verification - the harness emits wall times even for
   cells whose requests all failed (reporting bug found on the A10 run).

Raw artifacts: `sm120/offline/` (perf + accuracy summaries, env manifest),
`sm120/online/` (bench_serving jsonl, steady state = last line).
A10 artifacts and full narrative: `experiments/llm4rec/rfc_research/`
(gitignored, local box) and the RFC section 10.
