# SID-GR Inference vs SGLang Beam Search 对比

## Workload

- model: `/mnt/data/hongsheng.jhs/models/Qwen3-1.7B`
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

- GR wall_ms_median: `45.06366199984768`
- GR qps: `22.190828610497302`
- SGLang wall_ms_median: `52.425337999920885`
- SGLang qps_median: `19.07474587958802`
- SGLang request_latency_ms_p50_median: `52.425337999920885`
- SGLang request_latency_ms_p95_median: `52.425337999920885`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.96484375`
- Ordered prefix match mean: `0.09765625`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.4288571675618496, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=25, common_prefix_count=2

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 247, 'same_rank_count': 25, 'within_1_count': 51, 'within_5_count': 121, 'within_10_count': 176, 'mean': 1.7611336032388665, 'median': 0.0, 'max_abs': 55}

## Top1 Cross Scores

- beamcmp-0: GR top1=(17, 15, 17) score=-8.173013687133789 SGLang-score=-2.74415651957194 scaled_delta=0.05945587158203125; SGLang top1=(17, 15, 17) score=-2.74415651957194 GR-score=-8.173013687133789 scaled_delta=0.05945587158203125

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 247, 'mean': -0.023906209690850756, 'median': -0.017795562744140625, 'p05': -0.13208484649658203, 'p95': 0.07465744018554688, 'min': -0.23511314392089844, 'max': 0.13537883758544922}, corr=0.996735997998546

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
