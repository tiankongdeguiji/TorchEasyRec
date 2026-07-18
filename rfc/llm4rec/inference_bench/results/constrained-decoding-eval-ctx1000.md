# sid-gr-inference constrained decoding cost — ctx1000 / beam 256 / mc=4 / 64 requests

Evaluation of item-constrained decoding (`--catalog-jsonl`) at the established
online operating point: Qwen3-0.6B bf16, SM120 (RTX PRO 5000 72GB), context
1000, beam width 256, decode_steps 3, client concurrency 4 (`mc=4`), 64
requests, 3 rounds. Two parts: **Part 1** measures the stock Python trie-walk
implementation (`TrieItemMaskProvider`); **Part 2** measures the STATIC (CSR)
tensorized backend we implemented from
[youtube/static-constraint-decoding](https://github.com/youtube/static-constraint-decoding)
(`StaticTrieItemMaskProvider`, `--catalog-mask-backend static`, recsys-examples
`llm4rec/sidgr-bench` @f7d61a2).

Harness: `../constrained/eval_constrained_ctx1000.sh` (branch
`rfc_llm4rec_inference_bench`). Raw artifacts:
`sm120-qwen3-0.6b/constrained_ctx1000/` (Part 1 run, @794adce) and
`sm120-qwen3-0.6b/constrained_ctx1000_backends/` (Part 2 full matrix rerun).
Run dates 2026-07-18.

## TL;DR

**The stock constrained decoding is a server-side Python bottleneck that
dominates E2E at realistic catalog sizes; the STATIC CSR backend removes it.**
At 1M items, the Python walk takes online p50 from **78.7 ms → 1450 ms
(18.4×)** and throughput from **50.3 → 2.8 req/s**; the cost is *not* GPU
masking but one CUDA element-write per allowed token per beam (~8.7 µs each,
~31k per request per step). The STATIC backend — dense level-0 head + flat CSR
transition tail, ~15 tensor ops per step mask — is **catalog-size-independent
at p50 ≈ 94 ms (1.19× unconstrained, 42.3 req/s)**: 15.4× faster than the
Python walk at 1M items, with bit-identical masks (equivalence-tested) and all
beams on-catalog.

## Setup

- Synthetic SID catalogs: 3 levels × 8192-way codebooks, disjoint token ranges
  inside the Qwen3 vocab (L1 ∈ \[1000,9192), L2 ∈ \[10000,18192),
  L3 ∈ \[20000,28192)), uniform-random items, seeded
  (`../constrained/gen_sid_catalog.py`).

  | items | L1 fan-out | avg L2 children | avg L3 children | trie build (server start) |
  | ----- | ---------- | --------------- | --------------- | ------------------------- |
  | 10k   | 5788       | 1.73            | 1.00            | 0.09 s                    |
  | 100k  | 8192       | 12.2            | 1.00            | 1.06 s                    |
  | 1M    | 8192       | 121.2           | 1.01            | 9.7 s                     |

- With `--catalog-jsonl` configured, the server attaches a provider snapshot to
  **every** request (`gr_serving/api.py`), so the A/B needs no client changes.

- Correctness gate passed at every size: all 256 returned beams are exact
  catalog token paths (`GATE_CONSTRAINED_PATHS_PASS`,
  `../constrained/validate_catalog_paths.py`).

- Unconstrained baseline re-run same-day: 50.3 req/s / p50 78.6 ms — matches
  the prior report (50.2 / 78.7).

## E2E A/B (online, steady-state rounds)

| config        | p50 e2e (ms)       | p99 e2e (ms)        | req/s                | p50 vs unconstrained |
| ------------- | ------------------ | ------------------- | -------------------- | -------------------- |
| unconstrained | 78.6 / 79.7 / 79.3 | 85.3 / 85.7 / 354\* | 50.3 / 49.8 / 41.2\* | 1×                   |
| catalog 10k   | 312 / 312 / 318    | 319 / 319 / 325     | 12.8 / 12.8 / 12.6   | **4.0×**             |
| catalog 100k  | 488 / 491 / 491    | 498 / 510 / 498     | 8.2 / 8.1 / 8.2      | **6.2×**             |
| catalog 1M    | 1451 / 1454 / 1445 | 1471 / 1463 / 1470  | 2.77 / 2.75 / 2.80   | **18.4×**            |

\* unconstrained round 3 hit the known single-straggler flake (one ~350 ms
request); rounds 1–2 are steady state. Constrained rounds are tight across all
3 rounds — the added cost is deterministic Python work, not jitter.

## Where the time goes

`step_mask` (`gr_runtime/item_constraints.py`) builds a `[256, vocab]` bool
mask per request per decode step: a Python loop over 256 beams, each doing a
host-side parent-chain walk (`BeamPath.token_trace`, no GPU sync) + a trie dict
lookup (`allowed_next`) + **one `mask[beam, token] = True` per allowed token on
a CUDA tensor** — each a full Python→dispatcher→kernel round-trip.
`initial_mask` does the same over the level-1 fan-out at prefill. The batched
continuous path (`gr_serving/continuous.py::_continuous_step_item_mask`) calls
each request's provider **serially** (the provider API asserts batch_size=1),
so at batch 4 every request also waits for the other three requests' walks.
All of this runs outside the decode CUDA graphs (graphs stay intact) — the GPU
simply idles while Python builds masks.

Microbench of the real provider (per request per call, SM120, beam 256, vocab
151936; `../constrained/bench_trie_mask.py`):

| catalog | initial_mask | step-1 mask  | step-2 mask | step-1 allowed tokens (Σ 256 beams) |
| ------- | ------------ | ------------ | ----------- | ----------------------------------- |
| 10k     | 50.1 ms      | 4.1 ms       | 2.7 ms      | 433                                 |
| 100k    | 68.5 ms      | 27.1 ms      | 2.5 ms      | 3003                                |
| 1M      | 68.8 ms      | **270.4 ms** | 2.6 ms      | 30 887                              |

- `initial_mask` saturates near ~68 ms once L1 fan-out caps at the codebook
  size (8192 element-writes).
- step-1 cost is linear in allowed-token count: 270 ms / 30 887 ≈ **8.7 µs per
  element-write** — pure dispatch overhead, not compute.
- step-2 is cheap because avg L3 fan-out ≈ 1.

**The model closes the E2E delta.** With batch 4 serializing per-request walks,
predicted delta ≈ 4 × (initial + step1 + step2):

| catalog | predicted Δp50 | observed Δp50 |
| ------- | -------------- | ------------- |
| 10k     | +228 ms        | +233 ms       |
| 100k    | +392 ms        | +410 ms       |
| 1M      | +1367 ms       | +1372 ms      |

Corroboration that the wall is CPU-Python-bound: the profiled rounds (per-
section CUDA syncs enabled) inflate unconstrained p50 by +29% (78.6 → 101.5 ms)
but constrained-1M p50 by only +1.4% (1451 → 1471 ms) — extra syncs cost
nothing when the GPU is already waiting on Python.

## What a fix buys

Reference variants in the microbench keep the identical trie walk and change
only the tensor writes (step-1 mask, 1M catalog):

| variant                                           | ms    | vs current |
| ------------------------------------------------- | ----- | ---------- |
| current (per-token CUDA element writes)           | 270.4 | 1×         |
| one `index_put` per beam (`mask[beam, idx]=True`) | 9.7   | 28×        |
| build bool mask on host, one H2D copy             | 9.2   | 29×        |

So a ~30-line change (batch the writes per beam) already collapses the step
cost to ~10 ms; the pure Python trie walk itself is under 9 ms of the 270 ms.
The remaining structural costs — per-request provider serialization at batch 4
and the per-beam Python walk — are what the STATIC CSR backend (Part 2)
eliminates. The 9.7 s Python trie build at 1M items (server startup and
`/catalog/reload`) also argues for a precompiled CSR artifact.

## Part 2: STATIC (CSR) tensorized backend

We implemented the RFC's STATIC-style constraint decoding (D-decision, §10) as
`gr_runtime/static_item_constraints.py` in sid-gr-inference, adapting the
reference at github.com/youtube/static-constraint-decoding: a dense level-0
head (`start_mask`; level-0 state of token `t` is `t + 1`), a flat CSR
transition tail (`edge_keys = parent * V + token`, globally sorted, walked
with one `searchsorted` per trace level), and a single `index_put` to
materialize the `[W, vocab]` mask. The reference's V×V dense second layer does
not transfer (it masks a 2048-codebook vocab; the LLM vocab is 151936), so
only level 0 is dense. Selectable per server via `--catalog-mask-backend static` / `GR_CATALOG_MASK_BACKEND`; hot reload keeps the backend;
`/catalog/status` reports it. Masks are bit-identical to the Python walk
(`tests/test_static_item_constraints.py`, equivalence on CPU and CUDA over
depths 2–4, EOS/terminal/off-trie rows), and the on-catalog beam gate passed
in every phase below.

Same-day A/B, all rounds 64/64 completed (steady-state medians; stragglers
disclosed below):

| config          | p50 e2e (ms) | p99 e2e (ms) | req/s | p50 vs uncon | vs python backend |
| --------------- | ------------ | ------------ | ----- | ------------ | ----------------- |
| unconstrained   | 78.9         | 84.5         | 50.3  | 1×           | —                 |
| 10k python      | 314          | 321          | 12.7  | 3.98×        | 1×                |
| 10k **static**  | **93.0**     | 100          | 42.6  | 1.18×        | **3.4×**          |
| 100k python     | 488          | 495          | 8.2   | 6.19×        | 1×                |
| 100k **static** | **94.5**     | 101          | 41.9  | 1.20×        | **5.2×**          |
| 1M python       | 1450         | 1474         | 2.8   | 18.4×        | 1×                |
| 1M **static**   | **94.1**     | 100          | 42.3  | 1.19×        | **15.4×**         |

The STATIC p50 is **flat in catalog size** (93.0 / 94.5 / 94.1 ms) — the mask
cost no longer scales with fan-out. Straggler flake: one ~350–500 ms request
appeared in unconstrained round 3 and 100k-static round 3 (the known
single-request flake, not backend-related).

Microbench on the same box (per request per call, beam 256):

| catalog | step-1 python | step-1 static | speedup  | initial python | initial static |
| ------- | ------------- | ------------- | -------- | -------------- | -------------- |
| 10k     | 6.2 ms        | 0.57 ms       | 11×      | 50.5 ms        | 0.01 ms        |
| 100k    | 25.6 ms       | 0.59 ms       | 43×      | 65.9 ms        | 0.01 ms        |
| 1M      | 266.2 ms      | **0.65 ms**   | **410×** | 68.3 ms        | 0.01 ms        |

Static step-2 masks are 0.38–0.40 ms. Per request the mask work is now
~1.1 ms (vs ~340 ms python at 1M); ×4 batch-serialized requests ≈ 4 ms of the
+15 ms E2E delta over unconstrained. The residual ~11 ms is size-independent
constrained-path engine work outside the provider: stacking per-request masks
to `[B, W, vocab]`, the `batched_item_mask_limited_beam_width` count+`.item()`
sync per step, `masked_fill` over scores, and per-beam host-side completion
checks and item resolution at response time. Tightening that residual would
need engine-level changes (e.g. STATIC's candidate-gather top-k over W×K
instead of masking the full vocab) — not required at this operating point.

Remaining startup cost: the CSR build adds only ~0.1–0.6 s, but the inherited
dict-trie build (used for response-time item resolution) still costs ~9 s at
1M items, so server start and `/catalog/reload` latency are unchanged — the
precompiled-artifact argument above stands.

## Caveats

- Catalogs are uniform-random; real RQ-VAE SIDs are skewed. Cost scales with
  *average* step-1 fan-out (items ÷ 8192 here), so skew shifts per-beam
  variance, not the order of magnitude.
- Prompts are the benchmark's synthetic random-token contexts; constraint cost
  is prompt-independent (mask work depends only on trie shape and beam width).
- `mc=4` means 4 concurrent client requests; the server decodes them in one
  batch-4 continuous step, which is what serializes the four Python walks.
