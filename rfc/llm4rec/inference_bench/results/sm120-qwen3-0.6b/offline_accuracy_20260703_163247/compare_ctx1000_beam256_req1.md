# SID-GR Inference vs SGLang Beam Search 对比

## Workload

- model: `/mnt/data/hongsheng.jhs/models/Qwen3-0.6B`
- context_len: `1000`
- GR decode_steps: `2`
- SGLang decode_steps: `3`
- GR output_token_budget: `3`
- SGLang output_token_budget: `3`
- output_token_budget_match: `True`
- beam_width: `256`
- GR serving_mode: `continuous`
- SGLang arrival_mode: `batch`
- GR arrival_stagger_ticks: `0`
- SGLang arrival_stagger_ms: `0.0`
- SGLang arrival_burst_size: `1`
- matched_requests: `1`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `29.18169499935175`
- GR qps: `34.26805742511579`
- SGLang wall_ms_median: `36.38680100084457`
- SGLang qps_median: `27.482492895618638`
- SGLang request_latency_ms_p50_median: `36.38680100084457`
- SGLang request_latency_ms_p95_median: `36.38680100084457`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.9375`
- Ordered prefix match mean: `0.08203125`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.9375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.0122617085774746, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=21, common_prefix_count=8

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 240, 'same_rank_count': 21, 'within_1_count': 48, 'within_5_count': 98, 'within_10_count': 141, 'mean': 1.6291666666666667, 'median': 0.0, 'max_abs': 85}

## Top1 Cross Scores

- beamcmp-0: GR top1=(17, 15, 16) score=-9.011795997619629 SGLang-score=-2.999534289042155 scaled_delta=-0.013193130493164062; SGLang top1=(17, 15, 16) score=-2.999534289042155 GR-score=-9.011795997619629 scaled_delta=-0.013193130493164062

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 240, 'mean': -0.02226501703262329, 'median': -0.013193130493164062, 'p05': -0.14908504486083984, 'p95': 0.10154342651367188, 'min': -0.376190185546875, 'max': 0.14345932006835938}, corr=0.9938085764154865

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
