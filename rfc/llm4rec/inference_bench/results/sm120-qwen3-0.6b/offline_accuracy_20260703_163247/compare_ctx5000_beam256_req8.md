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
- matched_requests: `8`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `530.3829740005312`
- GR qps: `15.083440442399999`
- SGLang wall_ms_median: `920.1994280010695`
- SGLang qps_median: `8.693767629673752`
- SGLang request_latency_ms_p50_median: `920.1994280010695`
- SGLang request_latency_ms_p95_median: `920.1994280010695`

## Correctness Against SGLang

- Top1 exact match rate: `0.875`
- TopK set overlap mean: `0.947265625`
- Ordered prefix match mean: `0.064453125`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.97265625, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.458278656005859, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=24, common_prefix_count=4
- beamcmp-1 vs beamcmp-1: top1=True, topk_overlap=0.94921875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-7.044219652811686, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=20, common_prefix_count=7
- beamcmp-2 vs beamcmp-2: top1=True, topk_overlap=0.96875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.748745600382486, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=19, common_prefix_count=1
- beamcmp-3 vs beamcmp-3: top1=True, topk_overlap=0.94921875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.51692803700765, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=18, common_prefix_count=4
- beamcmp-4 vs beamcmp-4: top1=True, topk_overlap=0.91796875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.863736152648926, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=8, common_prefix_count=3
- beamcmp-5 vs beamcmp-5: top1=False, topk_overlap=0.93359375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.449932734171549, gr_top1_rank_in_sglang=None, sglang_top1_rank_in_gr=1, same_position_count=11, common_prefix_count=0
- beamcmp-6 vs beamcmp-6: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.8363447189331055, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=15, common_prefix_count=4
- beamcmp-7 vs beamcmp-7: top1=True, topk_overlap=0.921875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.949521700541178, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=17, common_prefix_count=1

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 249, 'same_rank_count': 24, 'within_1_count': 65, 'within_5_count': 143, 'within_10_count': 194, 'mean': 0.8232931726907631, 'median': 0.0, 'max_abs': 53}
- beamcmp-1: {'overlap_count': 243, 'same_rank_count': 20, 'within_1_count': 56, 'within_5_count': 145, 'within_10_count': 194, 'mean': -2.6790123456790123, 'median': -2.0, 'max_abs': 30}
- beamcmp-2: {'overlap_count': 248, 'same_rank_count': 19, 'within_1_count': 54, 'within_5_count': 131, 'within_10_count': 183, 'mean': 1.0120967741935485, 'median': 0.0, 'max_abs': 38}
- beamcmp-3: {'overlap_count': 243, 'same_rank_count': 18, 'within_1_count': 35, 'within_5_count': 112, 'within_10_count': 163, 'mean': 0.5967078189300411, 'median': 1.0, 'max_abs': 56}
- beamcmp-4: {'overlap_count': 235, 'same_rank_count': 8, 'within_1_count': 11, 'within_5_count': 48, 'within_10_count': 116, 'mean': 11.838297872340426, 'median': 11.0, 'max_abs': 43}
- beamcmp-5: {'overlap_count': 239, 'same_rank_count': 11, 'within_1_count': 34, 'within_5_count': 120, 'within_10_count': 177, 'mean': 4.167364016736402, 'median': 3.0, 'max_abs': 43}
- beamcmp-6: {'overlap_count': 247, 'same_rank_count': 15, 'within_1_count': 52, 'within_5_count': 130, 'within_10_count': 185, 'mean': -1.1983805668016194, 'median': -1.0, 'max_abs': 39}
- beamcmp-7: {'overlap_count': 236, 'same_rank_count': 17, 'within_1_count': 46, 'within_5_count': 128, 'within_10_count': 174, 'mean': 1.4322033898305084, 'median': 1.0, 'max_abs': 45}

## Top1 Cross Scores

- beamcmp-0: GR top1=(7, 16, 15) score=-9.746917724609375 SGLang-score=-3.2886390686035156 scaled_delta=0.11899948120117188; SGLang top1=(7, 16, 15) score=-3.2886390686035156 GR-score=-9.746917724609375 scaled_delta=0.11899948120117188
- beamcmp-1: GR top1=(46452, 298, 197) score=-10.534134864807129 SGLang-score=-3.489915211995443 scaled_delta=-0.06438922882080078; SGLang top1=(46452, 298, 197) score=-3.489915211995443 GR-score=-10.534134864807129 scaled_delta=-0.06438922882080078
- beamcmp-2: GR top1=(3315, 118, 238) score=-10.128852844238281 SGLang-score=-3.3801072438557944 scaled_delta=0.011468887329101562; SGLang top1=(3315, 118, 238) score=-3.3801072438557944 GR-score=-10.128852844238281 scaled_delta=0.011468887329101562
- beamcmp-3: GR top1=(880, 299, 339) score=-9.768877029418945 SGLang-score=-3.2519489924112954 scaled_delta=-0.013030052185058594; SGLang top1=(880, 299, 339) score=-3.2519489924112954 GR-score=-9.768877029418945 scaled_delta=-0.013030052185058594
- beamcmp-4: GR top1=(10764, 44104, 53189) score=-10.273401260375977 SGLang-score=-3.409665107727051 scaled_delta=-0.04440593719482422; SGLang top1=(10764, 44104, 53189) score=-3.409665107727051 GR-score=-10.273401260375977 scaled_delta=-0.04440593719482422
- beamcmp-5: GR top1=(151645, 151645, 198) score=-9.888897895812988 SGLang-score=None scaled_delta=None; SGLang top1=(12281, 26991, 8178) score=-3.438965161641439 GR-score=-10.34880542755127 scaled_delta=-0.031909942626953125
- beamcmp-6: GR top1=(16, 17, 18) score=-10.28396224975586 SGLang-score=-3.447617530822754 scaled_delta=0.058890342712402344; SGLang top1=(16, 17, 18) score=-3.447617530822754 GR-score=-10.28396224975586 scaled_delta=0.058890342712402344
- beamcmp-7: GR top1=(3315, 112, 251) score=-10.425851821899414 SGLang-score=-3.476330121358236 scaled_delta=0.0031385421752929688; SGLang top1=(3315, 112, 251) score=-3.476330121358236 GR-score=-10.425851821899414 scaled_delta=0.0031385421752929688

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 249, 'mean': 0.007202527609216161, 'median': 0.0050487518310546875, 'p05': -0.08494758605957031, 'p95': 0.11899948120117188, 'min': -0.17308807373046875, 'max': 0.1798257827758789}, corr=0.9966819209459457
- beamcmp-1: GR - SGLang\*token_len={'count': 243, 'mean': -0.025670130066420317, 'median': -0.02888011932373047, 'p05': -0.11848640441894531, 'p95': 0.0668497085571289, 'min': -0.18297386169433594, 'max': 0.15951251983642578}, corr=0.9981332130938941
- beamcmp-2: GR - SGLang\*token_len={'count': 248, 'mean': 0.0020428049948907666, 'median': 0.004182338714599609, 'p05': -0.09311294555664062, 'p95': 0.10260963439941406, 'min': -0.15404415130615234, 'max': 0.2312030792236328}, corr=0.9968906243466239
- beamcmp-3: GR - SGLang\*token_len={'count': 243, 'mean': -0.013627982433931327, 'median': -0.02121734619140625, 'p05': -0.13431358337402344, 'p95': 0.10768890380859375, 'min': -0.2398662567138672, 'max': 0.1836538314819336}, corr=0.9941673871882089
- beamcmp-4: GR - SGLang\*token_len={'count': 235, 'mean': -0.02725462000420753, 'median': -0.024931907653808594, 'p05': -0.1324453353881836, 'p95': 0.08071422576904297, 'min': -0.19441604614257812, 'max': 0.1882171630859375}, corr=0.9968052129150766
- beamcmp-5: GR - SGLang\*token_len={'count': 239, 'mean': -0.009688038207497057, 'median': -0.014127731323242188, 'p05': -0.12581348419189453, 'p95': 0.09316444396972656, 'min': -0.21886062622070312, 'max': 0.1332225799560547}, corr=0.9966999556162864
- beamcmp-6: GR - SGLang\*token_len={'count': 247, 'mean': -0.008052609710075594, 'median': -0.012824058532714844, 'p05': -0.1066884994506836, 'p95': 0.10610389709472656, 'min': -0.20075321197509766, 'max': 0.1783580780029297}, corr=0.9953941419613812
- beamcmp-7: GR - SGLang\*token_len={'count': 236, 'mean': -0.01657606383501473, 'median': -0.008119583129882812, 'p05': -0.14905643463134766, 'p95': 0.09167003631591797, 'min': -0.21343040466308594, 'max': 0.16402530670166016}, corr=0.9963159442879035

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
