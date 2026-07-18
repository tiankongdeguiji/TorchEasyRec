# sid-gr-inference constrained decoding cost — ctx1000 / beam 256 / mc=4 / 64 requests

Evaluation of the **current trie-based item-constraint implementation**
(`TrieItemMaskProvider`, `--catalog-jsonl`) at the established online operating
point: Qwen3-0.6B bf16, SM120 (RTX PRO 5000 72GB), context 1000, beam width 256,
decode_steps 3, client concurrency 4 (`mc=4`), 64 requests, 3 rounds. All prior
benchmarks in this directory ran **unconstrained** decode; this is the first
measurement with the catalog engaged.

Harness: `../constrained/eval_constrained_ctx1000.sh` (branch
`rfc_llm4rec_inference_bench`). Raw artifacts:
`sm120-qwen3-0.6b/constrained_ctx1000/`. Run date 2026-07-18, recsys-examples
`llm4rec/sidgr-bench` @794adce.

## TL;DR

**Constrained decoding is a server-side Python bottleneck that dominates E2E at
realistic catalog sizes.** At 1M items, online p50 goes **78.7 ms → 1451 ms
(18.4×)** and throughput **50.3 → 2.8 req/s**. The cost is *not* GPU masking —
it is the per-beam Python trie walk that sets each allowed token with an
individual `mask[beam, token] = True` write on a CUDA tensor (~8.7 µs per
dispatcher round-trip, ~31k writes per request per step at 1M items). The same
trie walk with batched writes costs 9.7 ms instead of 270 ms (28×), and the
RFC's STATIC-style CSR trie removes the Python walk entirely.

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
and the per-beam Python walk — are what the RFC's STATIC-style CSR tensorized
trie (D-decision, §10) eliminates: allowed-token lookup becomes a batched GPU
gather over CSR arrays, with no per-beam Python and no batch_size=1 provider
limit. The 9.7 s Python trie build at 1M items (server startup and
`/catalog/reload`) also argues for a precompiled CSR artifact.

## Caveats

- Catalogs are uniform-random; real RQ-VAE SIDs are skewed. Cost scales with
  *average* step-1 fan-out (items ÷ 8192 here), so skew shifts per-beam
  variance, not the order of magnitude.
- Prompts are the benchmark's synthetic random-token contexts; constraint cost
  is prompt-independent (mask work depends only on trie shape and beam width).
- `mc=4` means 4 concurrent client requests; the server decodes them in one
  batch-4 continuous step, which is what serializes the four Python walks.
