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
- matched_requests: `8`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `958.299413000077`
- GR qps: `8.348121569807699`
- SGLang wall_ms_median: `1364.2634479997469`
- SGLang qps_median: `5.8639700504542756`
- SGLang request_latency_ms_p50_median: `1364.2634479997469`
- SGLang request_latency_ms_p95_median: `1364.2634479997469`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.958984375`
- Ordered prefix match mean: `0.09326171875`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-4.2154765129089355, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=31, common_prefix_count=3
- beamcmp-1 vs beamcmp-1: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.841022491455078, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=27, common_prefix_count=6
- beamcmp-2 vs beamcmp-2: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.579702377319336, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=27, common_prefix_count=13
- beamcmp-3 vs beamcmp-3: top1=True, topk_overlap=0.9296875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.362219969431559, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=18, common_prefix_count=4
- beamcmp-4 vs beamcmp-4: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.046229044596354, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=21, common_prefix_count=6
- beamcmp-5 vs beamcmp-5: top1=True, topk_overlap=0.9609375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.720319112141928, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=21, common_prefix_count=1
- beamcmp-6 vs beamcmp-6: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.4399925867716465, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=32, common_prefix_count=4
- beamcmp-7 vs beamcmp-7: top1=True, topk_overlap=0.95703125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.693438529968262, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=14, common_prefix_count=1

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 247, 'same_rank_count': 31, 'within_1_count': 51, 'within_5_count': 150, 'within_10_count': 195, 'mean': 0.1862348178137652, 'median': 0.0, 'max_abs': 33}
- beamcmp-1: {'overlap_count': 247, 'same_rank_count': 27, 'within_1_count': 56, 'within_5_count': 132, 'within_10_count': 181, 'mean': 1.4453441295546559, 'median': 0.0, 'max_abs': 67}
- beamcmp-2: {'overlap_count': 247, 'same_rank_count': 27, 'within_1_count': 58, 'within_5_count': 144, 'within_10_count': 197, 'mean': -0.048582995951417005, 'median': 0.0, 'max_abs': 38}
- beamcmp-3: {'overlap_count': 238, 'same_rank_count': 18, 'within_1_count': 38, 'within_5_count': 94, 'within_10_count': 139, 'mean': 3.407563025210084, 'median': 2.0, 'max_abs': 48}
- beamcmp-4: {'overlap_count': 247, 'same_rank_count': 21, 'within_1_count': 50, 'within_5_count': 138, 'within_10_count': 191, 'mean': -1.3319838056680162, 'median': -1.0, 'max_abs': 30}
- beamcmp-5: {'overlap_count': 246, 'same_rank_count': 21, 'within_1_count': 53, 'within_5_count': 139, 'within_10_count': 182, 'mean': -2.459349593495935, 'median': -2.0, 'max_abs': 51}
- beamcmp-6: {'overlap_count': 247, 'same_rank_count': 32, 'within_1_count': 75, 'within_5_count': 131, 'within_10_count': 178, 'mean': 0.8218623481781376, 'median': 0.0, 'max_abs': 50}
- beamcmp-7: {'overlap_count': 245, 'same_rank_count': 14, 'within_1_count': 44, 'within_5_count': 139, 'within_10_count': 191, 'mean': 2.8448979591836734, 'median': 2.0, 'max_abs': 57}

## Top1 Cross Scores

- beamcmp-0: GR top1=(7, 15, 13) score=-6.377893447875977 SGLang-score=-2.162416934967041 scaled_delta=0.10935735702514648; SGLang top1=(7, 15, 13) score=-2.162416934967041 GR-score=-6.377893447875977 scaled_delta=0.10935735702514648
- beamcmp-1: GR top1=(195, 195, 195) score=-8.732065200805664 SGLang-score=-2.891042709350586 scaled_delta=-0.05893707275390625; SGLang top1=(195, 195, 195) score=-2.891042709350586 GR-score=-8.732065200805664 scaled_delta=-0.05893707275390625
- beamcmp-2: GR top1=(16, 17, 18) score=-8.374031066894531 SGLang-score=-2.7943286895751953 scaled_delta=0.008955001831054688; SGLang top1=(16, 17, 18) score=-2.7943286895751953 GR-score=-8.374031066894531 scaled_delta=0.008955001831054688
- beamcmp-3: GR top1=(17, 15, 16) score=-8.021432876586914 SGLang-score=-2.659212907155355 scaled_delta=-0.04379415512084961; SGLang top1=(17, 15, 16) score=-2.659212907155355 GR-score=-8.021432876586914 scaled_delta=-0.04379415512084961
- beamcmp-4: GR top1=(220, 17, 15) score=-9.069635391235352 SGLang-score=-3.0234063466389975 scaled_delta=0.000583648681640625; SGLang top1=(220, 17, 15) score=-3.0234063466389975 GR-score=-9.069635391235352 scaled_delta=0.000583648681640625
- beamcmp-5: GR top1=(17, 15, 16) score=-8.568857192993164 SGLang-score=-2.848538080851237 scaled_delta=-0.023242950439453125; SGLang top1=(17, 15, 16) score=-2.848538080851237 GR-score=-8.568857192993164 scaled_delta=-0.023242950439453125
- beamcmp-6: GR top1=(220, 17, 15) score=-8.16981315612793 SGLang-score=-2.7298205693562827 scaled_delta=0.01964855194091797; SGLang top1=(220, 17, 15) score=-2.7298205693562827 GR-score=-8.16981315612793 scaled_delta=0.01964855194091797
- beamcmp-7: GR top1=(220, 17, 15) score=-8.573383331298828 SGLang-score=-2.8799448013305664 scaled_delta=0.0664510726928711; SGLang top1=(220, 17, 15) score=-2.8799448013305664 GR-score=-8.573383331298828 scaled_delta=0.0664510726928711

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 247, 'mean': -0.00834247457836321, 'median': -0.011676788330078125, 'p05': -0.10930919647216797, 'p95': 0.1121673583984375, 'min': -0.20920181274414062, 'max': 0.23435688018798828}, corr=0.9969179449464858
- beamcmp-1: GR - SGLang\*token_len={'count': 247, 'mean': -0.022989960334561615, 'median': -0.018713951110839844, 'p05': -0.17565536499023438, 'p95': 0.09561538696289062, 'min': -0.3006553649902344, 'max': 0.1818227767944336}, corr=0.9945781846126912
- beamcmp-2: GR - SGLang\*token_len={'count': 247, 'mean': 0.01908882619880954, 'median': 0.015398025512695312, 'p05': -0.07604503631591797, 'p95': 0.11922264099121094, 'min': -0.15081024169921875, 'max': 0.2371978759765625}, corr=0.997002473226905
- beamcmp-3: GR - SGLang\*token_len={'count': 238, 'mean': -0.004172407278493673, 'median': -0.0030431747436523438, 'p05': -0.12072086334228516, 'p95': 0.1417102813720703, 'min': -0.1695852279663086, 'max': 0.29030609130859375}, corr=0.9950766508522749
- beamcmp-4: GR - SGLang\*token_len={'count': 247, 'mean': -0.0090998251911117, 'median': 0.0020503997802734375, 'p05': -0.11437034606933594, 'p95': 0.08742618560791016, 'min': -0.21649742126464844, 'max': 0.18601036071777344}, corr=0.9968717765174975
- beamcmp-5: GR - SGLang\*token_len={'count': 246, 'mean': -0.020217806343140642, 'median': -0.017389297485351562, 'p05': -0.12915802001953125, 'p95': 0.06897926330566406, 'min': -0.26049327850341797, 'max': 0.13271617889404297}, corr=0.9973965703299021
- beamcmp-6: GR - SGLang\*token_len={'count': 247, 'mean': -0.0015910492252241746, 'median': 0.0027456283569335938, 'p05': -0.09857463836669922, 'p95': 0.09827041625976562, 'min': -0.2053699493408203, 'max': 0.2127981185913086}, corr=0.9965127700744142
- beamcmp-7: GR - SGLang\*token_len={'count': 245, 'mean': 0.027122571517010124, 'median': 0.038826942443847656, 'p05': -0.08487701416015625, 'p95': 0.13069725036621094, 'min': -0.2234344482421875, 'max': 0.1914510726928711}, corr=0.9961545800677392

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
