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
- matched_requests: `8`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `189.62644200109935`
- GR qps: `42.188209173663765`
- SGLang wall_ms_median: `214.94036500007496`
- SGLang qps_median: `37.21962601113667`
- SGLang request_latency_ms_p50_median: `214.94036500007496`
- SGLang request_latency_ms_p95_median: `214.94036500007496`

## Correctness Against SGLang

- Top1 exact match rate: `0.875`
- TopK set overlap mean: `0.95458984375`
- Ordered prefix match mean: `0.08203125`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.95703125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.107646942138672, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=13, common_prefix_count=1
- beamcmp-1 vs beamcmp-1: top1=False, topk_overlap=0.93359375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.9218902587890625, gr_top1_rank_in_sglang=1, sglang_top1_rank_in_gr=1, same_position_count=14, common_prefix_count=0
- beamcmp-2 vs beamcmp-2: top1=True, topk_overlap=0.953125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.870645523071289, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=25, common_prefix_count=4
- beamcmp-3 vs beamcmp-3: top1=True, topk_overlap=0.95703125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-6.113503138224283, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=26, common_prefix_count=3
- beamcmp-4 vs beamcmp-4: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-4.70784076054891, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=21, common_prefix_count=4
- beamcmp-5 vs beamcmp-5: top1=True, topk_overlap=0.94140625, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.684834798177084, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=16, common_prefix_count=5
- beamcmp-6 vs beamcmp-6: top1=True, topk_overlap=0.95703125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.803023338317871, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=30, common_prefix_count=11
- beamcmp-7 vs beamcmp-7: top1=True, topk_overlap=0.97265625, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.85685920715332, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=23, common_prefix_count=3

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 245, 'same_rank_count': 13, 'within_1_count': 36, 'within_5_count': 103, 'within_10_count': 155, 'mean': 0.9877551020408163, 'median': 3.0, 'max_abs': 66}
- beamcmp-1: {'overlap_count': 239, 'same_rank_count': 14, 'within_1_count': 45, 'within_5_count': 113, 'within_10_count': 169, 'mean': -6.2845188284518825, 'median': -5.0, 'max_abs': 42}
- beamcmp-2: {'overlap_count': 244, 'same_rank_count': 25, 'within_1_count': 50, 'within_5_count': 137, 'within_10_count': 198, 'mean': 0.9262295081967213, 'median': 0.0, 'max_abs': 40}
- beamcmp-3: {'overlap_count': 245, 'same_rank_count': 26, 'within_1_count': 60, 'within_5_count': 152, 'within_10_count': 199, 'mean': 1.2938775510204081, 'median': 1.0, 'max_abs': 75}
- beamcmp-4: {'overlap_count': 247, 'same_rank_count': 21, 'within_1_count': 59, 'within_5_count': 142, 'within_10_count': 191, 'mean': -1.582995951417004, 'median': -1.0, 'max_abs': 34}
- beamcmp-5: {'overlap_count': 241, 'same_rank_count': 16, 'within_1_count': 36, 'within_5_count': 102, 'within_10_count': 153, 'mean': 0.9253112033195021, 'median': 0.0, 'max_abs': 56}
- beamcmp-6: {'overlap_count': 245, 'same_rank_count': 30, 'within_1_count': 67, 'within_5_count': 130, 'within_10_count': 192, 'mean': -0.23265306122448978, 'median': 0.0, 'max_abs': 39}
- beamcmp-7: {'overlap_count': 249, 'same_rank_count': 23, 'within_1_count': 55, 'within_5_count': 144, 'within_10_count': 205, 'mean': -0.26506024096385544, 'median': 0.0, 'max_abs': 32}

## Top1 Cross Scores

- beamcmp-0: GR top1=(17, 15, 16) score=-9.097177505493164 SGLang-score=-2.989530563354492 scaled_delta=-0.1285858154296875; SGLang top1=(17, 15, 16) score=-2.989530563354492 GR-score=-9.097177505493164 scaled_delta=-0.1285858154296875
- beamcmp-1: GR top1=(220, 17, 15) score=-10.345819473266602 SGLang-score=-3.4442596435546875 scaled_delta=-0.013040542602539062; SGLang top1=(5140, 223, 251) score=-3.423929214477539 GR-score=-10.357977867126465 scaled_delta=-0.08619022369384766
- beamcmp-2: GR top1=(6794, 258, 1211) score=-8.780275344848633 SGLang-score=-2.9096298217773438 scaled_delta=-0.05138587951660156; SGLang top1=(6794, 258, 1211) score=-2.9096298217773438 GR-score=-8.780275344848633 scaled_delta=-0.05138587951660156
- beamcmp-3: GR top1=(220, 17, 15) score=-9.205422401428223 SGLang-score=-3.091919263203939 scaled_delta=0.07033538818359375; SGLang top1=(220, 17, 15) score=-3.091919263203939 GR-score=-9.205422401428223 scaled_delta=0.07033538818359375
- beamcmp-4: GR top1=(17, 20, 21) score=-7.061875343322754 SGLang-score=-2.3540345827738443 scaled_delta=0.00022840499877929688; SGLang top1=(17, 20, 21) score=-2.3540345827738443 GR-score=-7.061875343322754 scaled_delta=0.00022840499877929688
- beamcmp-5: GR top1=(17, 15, 16) score=-8.536327362060547 SGLang-score=-2.8514925638834634 scaled_delta=0.01815032958984375; SGLang top1=(17, 15, 16) score=-2.8514925638834634 GR-score=-8.536327362060547 scaled_delta=0.01815032958984375
- beamcmp-6: GR top1=(17642, 197, 197) score=-8.729055404663086 SGLang-score=-2.926032066345215 scaled_delta=0.049040794372558594; SGLang top1=(17642, 197, 197) score=-2.926032066345215 GR-score=-8.729055404663086 scaled_delta=0.049040794372558594
- beamcmp-7: GR top1=(220, 17, 15) score=-8.790496826171875 SGLang-score=-2.9336376190185547 scaled_delta=0.010416030883789062; SGLang top1=(220, 17, 15) score=-2.9336376190185547 GR-score=-8.790496826171875 scaled_delta=0.010416030883789062

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 245, 'mean': -0.04334877948371731, 'median': -0.06180572509765625, 'p05': -0.15064525604248047, 'p95': 0.12651348114013672, 'min': -0.19736194610595703, 'max': 0.31505680084228516}, corr=0.9936232424385797
- beamcmp-1: GR - SGLang\*token_len={'count': 239, 'mean': 0.008662439290449709, 'median': -0.0009279251098632812, 'p05': -0.10059165954589844, 'p95': 0.11829280853271484, 'min': -0.16013813018798828, 'max': 0.18842697143554688}, corr=0.9977639315783764
- beamcmp-2: GR - SGLang\*token_len={'count': 244, 'mean': 0.007158150438402519, 'median': 0.006848335266113281, 'p05': -0.11388587951660156, 'p95': 0.11426162719726562, 'min': -0.21867752075195312, 'max': 0.14250659942626953}, corr=0.9971313367044998
- beamcmp-3: GR - SGLang\*token_len={'count': 245, 'mean': 0.025700448483836895, 'median': 0.025620460510253906, 'p05': -0.0399017333984375, 'p95': 0.11391067504882812, 'min': -0.2592153549194336, 'max': 0.2349996566772461}, corr=0.9976985054569057
- beamcmp-4: GR - SGLang\*token_len={'count': 247, 'mean': -0.014038724937902288, 'median': -0.019250869750976562, 'p05': -0.10553264617919922, 'p95': 0.08353900909423828, 'min': -0.2349538803100586, 'max': 0.28945350646972656}, corr=0.9978679198821317
- beamcmp-5: GR - SGLang\*token_len={'count': 241, 'mean': -0.024253338699024247, 'median': -0.024690628051757812, 'p05': -0.1452493667602539, 'p95': 0.10737037658691406, 'min': -0.18397903442382812, 'max': 0.28272533416748047}, corr=0.9948131949840252
- beamcmp-6: GR - SGLang\*token_len={'count': 245, 'mean': 0.04695181165422712, 'median': 0.049040794372558594, 'p05': -0.06821060180664062, 'p95': 0.13019466400146484, 'min': -0.1126699447631836, 'max': 0.20794105529785156}, corr=0.9976346492651653
- beamcmp-7: GR - SGLang\*token_len={'count': 249, 'mean': 0.02716990934318328, 'median': 0.02921772003173828, 'p05': -0.05208396911621094, 'p95': 0.1068124771118164, 'min': -0.12598323822021484, 'max': 0.1227121353149414}, corr=0.9979806121007199

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
