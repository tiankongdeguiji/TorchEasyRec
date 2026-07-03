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
- matched_requests: `2`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `232.25584399961008`
- GR qps: `8.611193438918841`
- SGLang wall_ms_median: `339.0462130000742`
- SGLang qps_median: `5.898900867533247`
- SGLang request_latency_ms_p50_median: `339.0462130000742`
- SGLang request_latency_ms_p95_median: `339.0462130000742`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.974609375`
- Ordered prefix match mean: `0.0703125`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-4.208762327829996, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=17, common_prefix_count=3
- beamcmp-1 vs beamcmp-1: top1=True, topk_overlap=0.984375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.872501691182455, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=19, common_prefix_count=5

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 247, 'same_rank_count': 17, 'within_1_count': 44, 'within_5_count': 120, 'within_10_count': 177, 'mean': 0.3117408906882591, 'median': 0.0, 'max_abs': 41}
- beamcmp-1: {'overlap_count': 252, 'same_rank_count': 19, 'within_1_count': 43, 'within_5_count': 107, 'within_10_count': 170, 'mean': -0.03571428571428571, 'median': 0.0, 'max_abs': 41}

## Top1 Cross Scores

- beamcmp-0: GR top1=(7, 15, 13) score=-6.366395950317383 SGLang-score=-2.157633622487386 scaled_delta=0.10650491714477539; SGLang top1=(7, 15, 13) score=-2.157633622487386 GR-score=-6.366395950317383 scaled_delta=0.10650491714477539
- beamcmp-1: GR top1=(195, 195, 195) score=-8.732593536376953 SGLang-score=-2.8600918451944985 scaled_delta=-0.15231800079345703; SGLang top1=(195, 195, 195) score=-2.8600918451944985 GR-score=-8.732593536376953 scaled_delta=-0.15231800079345703

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 247, 'mean': -0.010634671338656653, 'median': -0.0116424560546875, 'p05': -0.12424278259277344, 'p95': 0.11298942565917969, 'min': -0.2245349884033203, 'max': 0.26489925384521484}, corr=0.9958633976542707
- beamcmp-1: GR - SGLang\*token_len={'count': 252, 'mean': -0.016997666586013066, 'median': -0.030079364776611328, 'p05': -0.124908447265625, 'p95': 0.10208702087402344, 'min': -0.22547531127929688, 'max': 0.19758129119873047}, corr=0.9949121370390851

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
