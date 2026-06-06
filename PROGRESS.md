# HSTU predict batch-size misalignment — debug progress

## Symptom

`tzrec.predict` (DlrmHSTU, checkpoint) gives different `probs_is_*` for the same
request depending on `--batch_size`. bz=1 matches the serving reference exactly;
large batch corrupts a sparse subset of rows.

## Established facts (empirical, on 8.216.130.229)

Ground-truth = reserved `scores` column = per-row server (RTP) probs, travels with each row.
Test = within each output row, compare predicted `probs_is_play` vs that row's own server score.

- bz=1 : 0/2899 rows wrong (0.00%) — perfect, matches server.
- bz=512, nw=8 : 41/2897 wrong (1.42%).
- bz=512, nw=1 : 415/2897 wrong (14.33%).
- bz=512 rerun (nw=8): identical 41 rows wrong → **fully deterministic** (not a race).

Corruption is a **request-level permutation**: a wrong row's prediction equals ANOTHER
request's correct prediction (clean swap, L1~1e-6). Aggregate distribution preserved.

## Ruled out

- Data pipeline: parsed `__features__` identical across batch sizes for a given request.
- Attention divisor `scaling_seqlen`: pinned to config max_seq_len=2048 (fixed); Triton + pytorch both use it. Batch-invariant.
- Positional encoder: per-sample, batch-invariant.
- Thread-safety / race: deterministic (same 41 rows on rerun); fewer threads → MORE corruption.
- `sort_by_length` attention path: transparent (off_z remapped for both read & write).
- split_2D_jagged / concat_2D_jagged: per-sample correct (grid (max_seq_len, B), own offsets).

## Current hypothesis

Deterministic, batch-composition-dependent, within-batch request-level reorder between the
model's per-candidate predictions and the reserve (input) order. num_workers changes batch
composition (parquet sharding) → changes which/how many rows trigger it.

## Instrumentation (this branch, env TZREC_DBG=1)

Per-request candidate-portion signatures at each stage in `DlrmHSTU.predict` /
`HSTUTransducer.forward`:
NUMTGT, CANDIN (input, order-preserving tag), ITEM, PRE_CAND (post-preprocess),
STU_CAND (post-STU stack), USER (post compose/split), PRED:<task>.
Compare bz=1 (ground-truth per-request map keyed by CANDIN sig) vs bz=512: the first stage
where position p's sig stops matching its CANDIN identity localizes the reorder.
