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
- matched_requests: `2`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `81.84939600005237`
- GR qps: `24.435122282377264`
- SGLang wall_ms_median: `90.65477399963129`
- SGLang qps_median: `22.06171734549947`
- SGLang request_latency_ms_p50_median: `90.65477399963129`
- SGLang request_latency_ms_p95_median: `90.65477399963129`

## Correctness Against SGLang

- Top1 exact match rate: `0.5`
- TopK set overlap mean: `0.9453125`
- Ordered prefix match mean: `0.0703125`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.9453125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.403966267903646, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=10, common_prefix_count=2
- beamcmp-1 vs beamcmp-1: top1=False, topk_overlap=0.9453125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.175289630889893, gr_top1_rank_in_sglang=1, sglang_top1_rank_in_gr=1, same_position_count=26, common_prefix_count=0

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 242, 'same_rank_count': 10, 'within_1_count': 49, 'within_5_count': 104, 'within_10_count': 172, 'mean': -0.1652892561983471, 'median': -1.0, 'max_abs': 54}
- beamcmp-1: {'overlap_count': 242, 'same_rank_count': 26, 'within_1_count': 46, 'within_5_count': 106, 'within_10_count': 153, 'mean': -0.5495867768595041, 'median': 0.0, 'max_abs': 50}

## Top1 Cross Scores

- beamcmp-0: GR top1=(17, 15, 17) score=-8.11324691772461 SGLang-score=-2.7092806498209634 scaled_delta=0.01459503173828125; SGLang top1=(17, 15, 17) score=-2.7092806498209634 GR-score=-8.11324691772461 scaled_delta=0.01459503173828125
- beamcmp-1: GR top1=(220, 17, 15) score=-7.796180725097656 SGLang-score=-2.625893751780192 scaled_delta=0.08150053024291992; SGLang top1=(195, 195, 195) score=-2.6208910942077637 GR-score=-7.824390411376953 scaled_delta=0.03828287124633789

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 242, 'mean': 0.007117898011010541, 'median': 0.01656341552734375, 'p05': -0.11432647705078125, 'p95': 0.13328266143798828, 'min': -0.19536209106445312, 'max': 0.1770801544189453}, corr=0.9958392941584385
- beamcmp-1: GR - SGLang\*token_len={'count': 242, 'mean': 0.04819776204006731, 'median': 0.06210136413574219, 'p05': -0.06167125701904297, 'p95': 0.14475631713867188, 'min': -0.11975479125976562, 'max': 0.1978473663330078}, corr=0.9965155950415956

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
