# SID-GR Inference vs SGLang Beam Search 对比

## Workload

- model: `/mnt/data/hongsheng.jhs/models/Qwen3-0.6B`
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

- GR wall_ms_median: `128.38906000069983`
- GR qps: `15.577651242162675`
- SGLang wall_ms_median: `228.33440599970345`
- SGLang qps_median: `8.759082939093277`
- SGLang request_latency_ms_p50_median: `228.33440599970345`
- SGLang request_latency_ms_p95_median: `228.33440599970345`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.953125`
- Ordered prefix match mean: `0.078125`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.94921875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.508430163065592, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=25, common_prefix_count=8
- beamcmp-1 vs beamcmp-1: top1=True, topk_overlap=0.95703125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-7.002551396687826, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=15, common_prefix_count=3

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 243, 'same_rank_count': 25, 'within_1_count': 47, 'within_5_count': 124, 'within_10_count': 186, 'mean': 1.8641975308641976, 'median': 1.0, 'max_abs': 47}
- beamcmp-1: {'overlap_count': 245, 'same_rank_count': 15, 'within_1_count': 59, 'within_5_count': 159, 'within_10_count': 201, 'mean': -1.6285714285714286, 'median': -1.0, 'max_abs': 34}

## Top1 Cross Scores

- beamcmp-0: GR top1=(7, 16, 15) score=-9.784372329711914 SGLang-score=-3.2759421666463218 scaled_delta=0.04345417022705078; SGLang top1=(7, 16, 15) score=-3.2759421666463218 GR-score=-9.784372329711914 scaled_delta=0.04345417022705078
- beamcmp-1: GR top1=(46452, 298, 197) score=-10.484429359436035 SGLang-score=-3.4818779627482095 scaled_delta=-0.03879547119140625; SGLang top1=(46452, 298, 197) score=-3.4818779627482095 GR-score=-10.484429359436035 scaled_delta=-0.03879547119140625

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 243, 'mean': 0.013415866427951388, 'median': 0.011568069458007812, 'p05': -0.08673572540283203, 'p95': 0.1178741455078125, 'min': -0.17163658142089844, 'max': 0.17099285125732422}, corr=0.9966282532736068
- beamcmp-1: GR - SGLang\*token_len={'count': 245, 'mean': -0.01750473100311902, 'median': -0.0107879638671875, 'p05': -0.1092061996459961, 'p95': 0.058165550231933594, 'min': -0.2243671417236328, 'max': 0.17126750946044922}, corr=0.9980477113400609

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
