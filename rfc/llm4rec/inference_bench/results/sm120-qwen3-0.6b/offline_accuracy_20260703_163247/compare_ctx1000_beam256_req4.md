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
- matched_requests: `4`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `99.95984599845542`
- GR qps: `40.01606805258391`
- SGLang wall_ms_median: `113.75113699978101`
- SGLang qps_median: `35.16448367463528`
- SGLang request_latency_ms_p50_median: `113.75113699978101`
- SGLang request_latency_ms_p95_median: `113.75113699978101`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.9541015625`
- Ordered prefix match mean: `0.0810546875`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.9609375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.08524227142334, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=18, common_prefix_count=2
- beamcmp-1 vs beamcmp-1: top1=True, topk_overlap=0.9375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.895942052205404, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=19, common_prefix_count=2
- beamcmp-2 vs beamcmp-2: top1=True, topk_overlap=0.94921875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.844433466593424, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=21, common_prefix_count=4
- beamcmp-3 vs beamcmp-3: top1=True, topk_overlap=0.96875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.08055559794108, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=25, common_prefix_count=3

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 246, 'same_rank_count': 18, 'within_1_count': 54, 'within_5_count': 137, 'within_10_count': 177, 'mean': -1.1300813008130082, 'median': 0.0, 'max_abs': 77}
- beamcmp-1: {'overlap_count': 240, 'same_rank_count': 19, 'within_1_count': 52, 'within_5_count': 130, 'within_10_count': 205, 'mean': -3.45, 'median': -3.0, 'max_abs': 37}
- beamcmp-2: {'overlap_count': 243, 'same_rank_count': 21, 'within_1_count': 51, 'within_5_count': 141, 'within_10_count': 199, 'mean': -0.22633744855967078, 'median': 0.0, 'max_abs': 33}
- beamcmp-3: {'overlap_count': 248, 'same_rank_count': 25, 'within_1_count': 58, 'within_5_count': 153, 'within_10_count': 204, 'mean': 2.588709677419355, 'median': 1.0, 'max_abs': 46}

## Top1 Cross Scores

- beamcmp-0: GR top1=(17, 15, 16) score=-9.085773468017578 SGLang-score=-3.0005311965942383 scaled_delta=-0.08417987823486328; SGLang top1=(17, 15, 16) score=-3.0005311965942383 GR-score=-9.085773468017578 scaled_delta=-0.08417987823486328
- beamcmp-1: GR top1=(5140, 223, 251) score=-10.3352689743042 SGLang-score=-3.4393269220987954 scaled_delta=-0.0172882080078125; SGLang top1=(5140, 223, 251) score=-3.4393269220987954 GR-score=-10.3352689743042 scaled_delta=-0.0172882080078125
- beamcmp-2: GR top1=(6794, 258, 1211) score=-8.745458602905273 SGLang-score=-2.901025136311849 scaled_delta=-0.04238319396972656; SGLang top1=(6794, 258, 1211) score=-2.901025136311849 GR-score=-8.745458602905273 scaled_delta=-0.04238319396972656
- beamcmp-3: GR top1=(220, 17, 15) score=-9.151665687561035 SGLang-score=-3.0711100896199546 scaled_delta=0.061664581298828125; SGLang top1=(220, 17, 15) score=-3.0711100896199546 GR-score=-9.151665687561035 scaled_delta=0.061664581298828125

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 246, 'mean': -0.02892236011784251, 'median': -0.03333568572998047, 'p05': -0.10125732421875, 'p95': 0.06856822967529297, 'min': -0.19100666046142578, 'max': 0.25606822967529297}, corr=0.996151873008421
- beamcmp-1: GR - SGLang\*token_len={'count': 240, 'mean': 0.014256099859873453, 'median': 0.017032623291015625, 'p05': -0.0848684310913086, 'p95': 0.09905433654785156, 'min': -0.1546773910522461, 'max': 0.2327117919921875}, corr=0.9978690537165888
- beamcmp-2: GR - SGLang\*token_len={'count': 243, 'mean': 0.00014619866516364454, 'median': 0.0045623779296875, 'p05': -0.10488319396972656, 'p95': 0.10301876068115234, 'min': -0.2120809555053711, 'max': 0.2144327163696289}, corr=0.9974214935267771
- beamcmp-3: GR - SGLang\*token_len={'count': 248, 'mean': 0.01818402736417709, 'median': 0.014646530151367188, 'p05': -0.06732177734375, 'p95': 0.09893608093261719, 'min': -0.1480093002319336, 'max': 0.1265735626220703}, corr=0.9977408824621496

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
