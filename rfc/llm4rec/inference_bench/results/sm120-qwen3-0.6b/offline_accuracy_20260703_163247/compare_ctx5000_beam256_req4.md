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
- matched_requests: `4`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `266.6922380012693`
- GR qps: `14.998561750345964`
- SGLang wall_ms_median: `461.0861350010964`
- SGLang qps_median: `8.675168686194583`
- SGLang request_latency_ms_p50_median: `461.0861350010964`
- SGLang request_latency_ms_p95_median: `461.0861350010964`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.951171875`
- Ordered prefix match mean: `0.0771484375`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.953125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.505588213602701, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=25, common_prefix_count=4
- beamcmp-1 vs beamcmp-1: top1=True, topk_overlap=0.94921875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-7.002460479736328, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=17, common_prefix_count=6
- beamcmp-2 vs beamcmp-2: top1=True, topk_overlap=0.95703125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.817390124003092, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=16, common_prefix_count=1
- beamcmp-3 vs beamcmp-3: top1=True, topk_overlap=0.9453125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.524337450663248, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=21, common_prefix_count=4

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 244, 'same_rank_count': 25, 'within_1_count': 54, 'within_5_count': 133, 'within_10_count': 185, 'mean': -1.2418032786885247, 'median': -1.0, 'max_abs': 34}
- beamcmp-1: {'overlap_count': 243, 'same_rank_count': 17, 'within_1_count': 47, 'within_5_count': 146, 'within_10_count': 203, 'mean': -3.7901234567901234, 'median': -3.0, 'max_abs': 43}
- beamcmp-2: {'overlap_count': 245, 'same_rank_count': 16, 'within_1_count': 39, 'within_5_count': 124, 'within_10_count': 177, 'mean': -2.1387755102040815, 'median': -2.0, 'max_abs': 33}
- beamcmp-3: {'overlap_count': 242, 'same_rank_count': 21, 'within_1_count': 50, 'within_5_count': 126, 'within_10_count': 171, 'mean': 2.347107438016529, 'median': 1.0, 'max_abs': 43}

## Top1 Cross Scores

- beamcmp-0: GR top1=(7, 16, 15) score=-9.784387588500977 SGLang-score=-3.278799374898275 scaled_delta=0.052010536193847656; SGLang top1=(7, 16, 15) score=-3.278799374898275 GR-score=-9.784387588500977 scaled_delta=0.052010536193847656
- beamcmp-1: GR top1=(3315, 241, 108) score=-10.474632263183594 SGLang-score=-3.4721717834472656 scaled_delta=-0.058116912841796875; SGLang top1=(3315, 241, 108) score=-3.4721717834472656 GR-score=-10.474632263183594 scaled_delta=-0.058116912841796875
- beamcmp-2: GR top1=(3315, 118, 238) score=-10.185811996459961 SGLang-score=-3.3684218724568686 scaled_delta=-0.08054637908935547; SGLang top1=(3315, 118, 238) score=-3.3684218724568686 GR-score=-10.185811996459961 scaled_delta=-0.08054637908935547
- beamcmp-3: GR top1=(880, 299, 339) score=-9.778005599975586 SGLang-score=-3.2536681493123374 scaled_delta=-0.01700115203857422; SGLang top1=(880, 299, 339) score=-3.2536681493123374 GR-score=-9.778005599975586 scaled_delta=-0.01700115203857422

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 244, 'mean': 0.01087189893253514, 'median': 0.007952690124511719, 'p05': -0.09291458129882812, 'p95': 0.10397529602050781, 'min': -0.13578510284423828, 'max': 0.2221841812133789}, corr=0.9964404238062324
- beamcmp-1: GR - SGLang\*token_len={'count': 243, 'mean': -0.03730324560722696, 'median': -0.03893470764160156, 'p05': -0.11671638488769531, 'p95': 0.04971504211425781, 'min': -0.18311691284179688, 'max': 0.1060037612915039}, corr=0.9983292597372081
- beamcmp-2: GR - SGLang\*token_len={'count': 245, 'mean': 0.004669376295440051, 'median': 0.00815582275390625, 'p05': -0.09224224090576172, 'p95': 0.09489250183105469, 'min': -0.1432476043701172, 'max': 0.1682281494140625}, corr=0.997257712442767
- beamcmp-3: GR - SGLang\*token_len={'count': 242, 'mean': 0.014672444871634492, 'median': 0.015608787536621094, 'p05': -0.10268688201904297, 'p95': 0.10500335693359375, 'min': -0.1512308120727539, 'max': 0.21114063262939453}, corr=0.9954954304324068

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
