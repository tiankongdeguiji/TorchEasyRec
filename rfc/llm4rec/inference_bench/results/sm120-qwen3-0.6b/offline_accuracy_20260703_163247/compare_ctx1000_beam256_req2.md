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
- matched_requests: `2`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `53.84325600061857`
- GR qps: `37.14485617246147`
- SGLang wall_ms_median: `62.25562300096499`
- SGLang qps_median: `32.12561217111263`
- SGLang request_latency_ms_p50_median: `62.25562300096499`
- SGLang request_latency_ms_p95_median: `62.25562300096499`

## Correctness Against SGLang

- Top1 exact match rate: `0.5`
- TopK set overlap mean: `0.939453125`
- Ordered prefix match mean: `0.080078125`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.9296875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.009470621744791, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=21, common_prefix_count=2
- beamcmp-1 vs beamcmp-1: top1=False, topk_overlap=0.94921875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.858140309651693, gr_top1_rank_in_sglang=1, sglang_top1_rank_in_gr=1, same_position_count=20, common_prefix_count=0

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 238, 'same_rank_count': 21, 'within_1_count': 43, 'within_5_count': 118, 'within_10_count': 161, 'mean': 0.029411764705882353, 'median': 0.0, 'max_abs': 84}
- beamcmp-1: {'overlap_count': 243, 'same_rank_count': 20, 'within_1_count': 64, 'within_5_count': 137, 'within_10_count': 191, 'mean': -3.753086419753086, 'median': -2.0, 'max_abs': 37}

## Top1 Cross Scores

- beamcmp-0: GR top1=(17, 15, 16) score=-9.008975982666016 SGLang-score=-2.999505360921224 scaled_delta=-0.01045989990234375; SGLang top1=(17, 15, 16) score=-2.999505360921224 GR-score=-9.008975982666016 scaled_delta=-0.01045989990234375
- beamcmp-1: GR top1=(220, 17, 15) score=-10.301958084106445 SGLang-score=-3.453126907348633 scaled_delta=0.057422637939453125; SGLang top1=(5140, 223, 251) score=-3.4438177744547525 GR-score=-10.35663890838623 scaled_delta=-0.025185585021972656

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 238, 'mean': 0.003205591890992237, 'median': 0.012468338012695312, 'p05': -0.12384605407714844, 'p95': 0.09161376953125, 'min': -0.32047367095947266, 'max': 0.1960773468017578}, corr=0.9954919416710589
- beamcmp-1: GR - SGLang\*token_len={'count': 243, 'mean': -0.009778215070810828, 'median': -0.011068344116210938, 'p05': -0.09781646728515625, 'p95': 0.08249759674072266, 'min': -0.1521444320678711, 'max': 0.1412811279296875}, corr=0.9981280811505125

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
