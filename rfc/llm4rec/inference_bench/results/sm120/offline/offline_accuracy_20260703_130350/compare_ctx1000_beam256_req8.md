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
- matched_requests: `8`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `300.3597169999921`
- GR qps: `26.634730115956962`
- SGLang wall_ms_median: `335.61913899984575`
- SGLang qps_median: `23.836542885606047`
- SGLang request_latency_ms_p50_median: `335.61913899984575`
- SGLang request_latency_ms_p95_median: `335.61913899984575`

## Correctness Against SGLang

- Top1 exact match rate: `0.875`
- TopK set overlap mean: `0.962890625`
- Ordered prefix match mean: `0.0830078125`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.94140625, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.5104786554972325, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=20, common_prefix_count=3
- beamcmp-1 vs beamcmp-1: top1=False, topk_overlap=0.9609375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.184144496917725, gr_top1_rank_in_sglang=1, sglang_top1_rank_in_gr=1, same_position_count=26, common_prefix_count=0
- beamcmp-2 vs beamcmp-2: top1=True, topk_overlap=0.98046875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.591471989949545, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=21, common_prefix_count=6
- beamcmp-3 vs beamcmp-3: top1=True, topk_overlap=0.9765625, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.502619425455729, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=23, common_prefix_count=6
- beamcmp-4 vs beamcmp-4: top1=True, topk_overlap=0.953125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.208349386850992, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=18, common_prefix_count=1
- beamcmp-5 vs beamcmp-5: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-4.347076733907064, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=21, common_prefix_count=4
- beamcmp-6 vs beamcmp-6: top1=True, topk_overlap=0.95703125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.313915888468424, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=24, common_prefix_count=5
- beamcmp-7 vs beamcmp-7: top1=True, topk_overlap=0.96875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-4.874546686808268, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=17, common_prefix_count=4

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 241, 'same_rank_count': 20, 'within_1_count': 57, 'within_5_count': 114, 'within_10_count': 159, 'mean': -0.24066390041493776, 'median': 0.0, 'max_abs': 62}
- beamcmp-1: {'overlap_count': 246, 'same_rank_count': 26, 'within_1_count': 55, 'within_5_count': 136, 'within_10_count': 181, 'mean': 0.8821138211382114, 'median': 1.0, 'max_abs': 36}
- beamcmp-2: {'overlap_count': 251, 'same_rank_count': 21, 'within_1_count': 51, 'within_5_count': 133, 'within_10_count': 182, 'mean': 0.49800796812749004, 'median': 0.0, 'max_abs': 37}
- beamcmp-3: {'overlap_count': 250, 'same_rank_count': 23, 'within_1_count': 45, 'within_5_count': 125, 'within_10_count': 193, 'mean': 0.1, 'median': 1.5, 'max_abs': 48}
- beamcmp-4: {'overlap_count': 244, 'same_rank_count': 18, 'within_1_count': 51, 'within_5_count': 114, 'within_10_count': 162, 'mean': -0.24180327868852458, 'median': 0.0, 'max_abs': 61}
- beamcmp-5: {'overlap_count': 247, 'same_rank_count': 21, 'within_1_count': 48, 'within_5_count': 111, 'within_10_count': 161, 'mean': 0.4979757085020243, 'median': 0.0, 'max_abs': 52}
- beamcmp-6: {'overlap_count': 245, 'same_rank_count': 24, 'within_1_count': 54, 'within_5_count': 125, 'within_10_count': 175, 'mean': 0.061224489795918366, 'median': 0.0, 'max_abs': 59}
- beamcmp-7: {'overlap_count': 248, 'same_rank_count': 17, 'within_1_count': 47, 'within_5_count': 113, 'within_10_count': 179, 'mean': -0.3024193548387097, 'median': 0.0, 'max_abs': 69}

## Top1 Cross Scores

- beamcmp-0: GR top1=(17, 15, 17) score=-8.21316909790039 SGLang-score=-2.7026904424031577 scaled_delta=-0.10509777069091797; SGLang top1=(17, 15, 17) score=-2.7026904424031577 GR-score=-8.21316909790039 scaled_delta=-0.10509777069091797
- beamcmp-1: GR top1=(220, 17, 15) score=-7.761784553527832 SGLang-score=-2.5889336268107095 scaled_delta=0.005016326904296875; SGLang top1=(195, 195, 195) score=-2.5776400566101074 GR-score=-7.802634239196777 scaled_delta=-0.06971406936645508
- beamcmp-2: GR top1=(220, 17, 15) score=-8.353044509887695 SGLang-score=-2.761572519938151 scaled_delta=-0.06832695007324219; SGLang top1=(220, 17, 15) score=-2.761572519938151 GR-score=-8.353044509887695 scaled_delta=-0.06832695007324219
- beamcmp-3: GR top1=(220, 17, 15) score=-8.242925643920898 SGLang-score=-2.7403062184651694 scaled_delta=-0.022006988525390625; SGLang top1=(220, 17, 15) score=-2.7403062184651694 GR-score=-8.242925643920898 scaled_delta=-0.022006988525390625
- beamcmp-4: GR top1=(17, 20, 21) score=-7.833656311035156 SGLang-score=-2.6253069241841636 scaled_delta=0.042264461517333984; SGLang top1=(17, 20, 21) score=-2.6253069241841636 GR-score=-7.833656311035156 scaled_delta=0.042264461517333984
- beamcmp-5: GR top1=(198, 13874, 19324) score=-6.548443794250488 SGLang-score=-2.2013670603434243 scaled_delta=0.055657386779785156; SGLang top1=(198, 13874, 19324) score=-2.2013670603434243 GR-score=-6.548443794250488 scaled_delta=0.055657386779785156
- beamcmp-6: GR top1=(220, 17, 15) score=-7.972606182098389 SGLang-score=-2.6586902936299643 scaled_delta=0.0034646987915039062; SGLang top1=(220, 17, 15) score=-2.6586902936299643 GR-score=-7.972606182098389 scaled_delta=0.0034646987915039062
- beamcmp-7: GR top1=(220, 17, 15) score=-7.3226141929626465 SGLang-score=-2.4480675061543784 scaled_delta=0.02158832550048828; SGLang top1=(220, 17, 15) score=-2.4480675061543784 GR-score=-7.3226141929626465 scaled_delta=0.02158832550048828

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 241, 'mean': 0.003272412723525431, 'median': -0.000858306884765625, 'p05': -0.1325531005859375, 'p95': 0.1296825408935547, 'min': -0.2326335906982422, 'max': 0.19044113159179688}, corr=0.9953332939182178
- beamcmp-1: GR - SGLang\*token_len={'count': 246, 'mean': 0.01516464473755379, 'median': 0.01632404327392578, 'p05': -0.0858907699584961, 'p95': 0.13143444061279297, 'min': -0.14327335357666016, 'max': 0.21035194396972656}, corr=0.9969123471955119
- beamcmp-2: GR - SGLang\*token_len={'count': 251, 'mean': 0.0033474994370661882, 'median': 0.008652687072753906, 'p05': -0.119903564453125, 'p95': 0.10758304595947266, 'min': -0.20517921447753906, 'max': 0.1640310287475586}, corr=0.9957002363412413
- beamcmp-3: GR - SGLang\*token_len={'count': 250, 'mean': 0.01842481231689453, 'median': 0.0001621246337890625, 'p05': -0.09494209289550781, 'p95': 0.1544361114501953, 'min': -0.1258373260498047, 'max': 0.2794361114501953}, corr=0.995531409605333
- beamcmp-4: GR - SGLang\*token_len={'count': 244, 'mean': -0.013160938122233406, 'median': -0.011220932006835938, 'p05': -0.14650535583496094, 'p95': 0.096954345703125, 'min': -0.24607372283935547, 'max': 0.23386192321777344}, corr=0.9965767382213733
- beamcmp-5: GR - SGLang\*token_len={'count': 247, 'mean': 0.02854557269015293, 'median': 0.024941444396972656, 'p05': -0.1075754165649414, 'p95': 0.164947509765625, 'min': -0.18094539642333984, 'max': 0.24208545684814453}, corr=0.9948009801233112
- beamcmp-6: GR - SGLang\*token_len={'count': 245, 'mean': -0.0344822358111946, 'median': -0.031076431274414062, 'p05': -0.15116119384765625, 'p95': 0.06404876708984375, 'min': -0.22809982299804688, 'max': 0.1811981201171875}, corr=0.9955685245689907
- beamcmp-7: GR - SGLang\*token_len={'count': 248, 'mean': 0.02929229505600468, 'median': 0.03678703308105469, 'p05': -0.10566139221191406, 'p95': 0.14658737182617188, 'min': -0.15514755249023438, 'max': 0.29785823822021484}, corr=0.9957419743294184

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
