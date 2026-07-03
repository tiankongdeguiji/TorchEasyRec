# SID-GR Inference vs SGLang Beam Search 对比

## Workload

- model: `/mnt/data/hongsheng.jhs/models/Qwen3-1.7B`
- context_len: `5000`
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

- GR wall_ms_median: `115.70479599959071`
- GR qps: `8.642684094128105`
- SGLang wall_ms_median: `171.93944900009228`
- SGLang qps_median: `5.816000957403692`
- SGLang request_latency_ms_p50_median: `171.93944900009228`
- SGLang request_latency_ms_p95_median: `171.93944900009228`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.9609375`
- Ordered prefix match mean: `0.09375`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.9609375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-4.184875011444092, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=24, common_prefix_count=3

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 246, 'same_rank_count': 24, 'within_1_count': 47, 'within_5_count': 123, 'within_10_count': 166, 'mean': 0.3780487804878049, 'median': 0.0, 'max_abs': 39}

## Top1 Cross Scores

- beamcmp-0: GR top1=(7, 15, 13) score=-6.3676605224609375 SGLang-score=-2.1827855110168457 scaled_delta=0.1806960105895996; SGLang top1=(7, 15, 13) score=-2.1827855110168457 GR-score=-6.3676605224609375 scaled_delta=0.1806960105895996

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 246, 'mean': 0.05150100855323357, 'median': 0.058193206787109375, 'p05': -0.06361007690429688, 'p95': 0.16723918914794922, 'min': -0.17829322814941406, 'max': 0.2671394348144531}, corr=0.9953024623881288

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
