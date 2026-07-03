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
- matched_requests: `4`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `388.58638800002154`
- GR qps: `10.293721353923953`
- SGLang wall_ms_median: `169.94779699962237`
- SGLang qps_median: `23.536639312887875`
- SGLang request_latency_ms_p50_median: `169.94779699962237`
- SGLang request_latency_ms_p95_median: `169.94779699962237`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.9619140625`
- Ordered prefix match mean: `0.0869140625`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.94921875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.428997993469238, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=24, common_prefix_count=2
- beamcmp-1 vs beamcmp-1: top1=True, topk_overlap=0.953125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.134064515431723, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=21, common_prefix_count=4
- beamcmp-2 vs beamcmp-2: top1=True, topk_overlap=0.98046875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.5181420644124355, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=27, common_prefix_count=3
- beamcmp-3 vs beamcmp-3: top1=True, topk_overlap=0.96484375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.5096893310546875, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=17, common_prefix_count=2

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 243, 'same_rank_count': 24, 'within_1_count': 58, 'within_5_count': 103, 'within_10_count': 155, 'mean': -0.1646090534979424, 'median': 0.0, 'max_abs': 56}
- beamcmp-1: {'overlap_count': 244, 'same_rank_count': 21, 'within_1_count': 52, 'within_5_count': 117, 'within_10_count': 168, 'mean': 1.2008196721311475, 'median': 0.0, 'max_abs': 46}
- beamcmp-2: {'overlap_count': 251, 'same_rank_count': 27, 'within_1_count': 53, 'within_5_count': 124, 'within_10_count': 185, 'mean': 1.2231075697211156, 'median': 1.0, 'max_abs': 32}
- beamcmp-3: {'overlap_count': 247, 'same_rank_count': 17, 'within_1_count': 37, 'within_5_count': 111, 'within_10_count': 165, 'mean': -0.05668016194331984, 'median': 0.0, 'max_abs': 68}

## Top1 Cross Scores

- beamcmp-0: GR top1=(17, 15, 17) score=-8.155973434448242 SGLang-score=-2.726975440979004 scaled_delta=0.02495288848876953; SGLang top1=(17, 15, 17) score=-2.726975440979004 GR-score=-8.155973434448242 scaled_delta=0.02495288848876953
- beamcmp-1: GR top1=(220, 17, 15) score=-7.739432334899902 SGLang-score=-2.60536781946818 scaled_delta=0.07667112350463867; SGLang top1=(220, 17, 15) score=-2.60536781946818 GR-score=-7.739432334899902 scaled_delta=0.07667112350463867
- beamcmp-2: GR top1=(220, 17, 15) score=-8.30154800415039 SGLang-score=-2.7834059397379556 scaled_delta=0.04866981506347656; SGLang top1=(220, 17, 15) score=-2.7834059397379556 GR-score=-8.30154800415039 scaled_delta=0.04866981506347656
- beamcmp-3: GR top1=(220, 17, 15) score=-8.267538070678711 SGLang-score=-2.7578487396240234 scaled_delta=0.006008148193359375; SGLang top1=(220, 17, 15) score=-2.7578487396240234 GR-score=-8.267538070678711 scaled_delta=0.006008148193359375

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 243, 'mean': -1.4960520552019032e-05, 'median': 0.015621185302734375, 'p05': -0.14551162719726562, 'p95': 0.13361072540283203, 'min': -0.19594669342041016, 'max': 0.20210552215576172}, corr=0.9953132199459376
- beamcmp-1: GR - SGLang\*token_len={'count': 244, 'mean': 0.029592611750618357, 'median': 0.02386188507080078, 'p05': -0.0844125747680664, 'p95': 0.12960529327392578, 'min': -0.19332504272460938, 'max': 0.22015762329101562}, corr=0.9965261018763828
- beamcmp-2: GR - SGLang\*token_len={'count': 251, 'mean': 0.0004938095214357414, 'median': -0.005549430847167969, 'p05': -0.10884666442871094, 'p95': 0.11041259765625, 'min': -0.1946430206298828, 'max': 0.19096660614013672}, corr=0.9963565724776344
- beamcmp-3: GR - SGLang\*token_len={'count': 247, 'mean': 0.028011213912654986, 'median': 0.0126800537109375, 'p05': -0.14317035675048828, 'p95': 0.19490814208984375, 'min': -0.2311115264892578, 'max': 0.3331727981567383}, corr=0.9914333077842491

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
