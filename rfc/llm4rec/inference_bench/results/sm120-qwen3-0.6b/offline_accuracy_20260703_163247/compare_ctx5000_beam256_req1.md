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
- matched_requests: `1`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `66.20572700012417`
- GR qps: `15.104433488029283`
- SGLang wall_ms_median: `117.14527100048144`
- SGLang qps_median: `8.536409463732344`
- SGLang request_latency_ms_p50_median: `117.14527100048144`
- SGLang request_latency_ms_p95_median: `117.14527100048144`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.94921875`
- Ordered prefix match mean: `0.10546875`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.94921875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.533501625061035, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=27, common_prefix_count=4

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 243, 'same_rank_count': 27, 'within_1_count': 62, 'within_5_count': 141, 'within_10_count': 203, 'mean': 1.7037037037037037, 'median': 1.0, 'max_abs': 32}

## Top1 Cross Scores

- beamcmp-0: GR top1=(7, 16, 15) score=-9.787919998168945 SGLang-score=-3.25441837310791 scaled_delta=-0.024664878845214844; SGLang top1=(7, 16, 15) score=-3.25441837310791 GR-score=-9.787919998168945 scaled_delta=-0.024664878845214844

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 243, 'mean': 0.0018460623030799898, 'median': 0.0048847198486328125, 'p05': -0.07808494567871094, 'p95': 0.08518028259277344, 'min': -0.12654781341552734, 'max': 0.14816665649414062}, corr=0.9977990851917955

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
