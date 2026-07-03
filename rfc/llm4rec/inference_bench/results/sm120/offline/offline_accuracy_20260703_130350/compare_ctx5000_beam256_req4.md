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
- matched_requests: `4`
- match_strategy: `request_id`

## Performance

- GR wall_ms_median: `478.3843299996988`
- GR qps: `8.361477893731424`
- SGLang wall_ms_median: `685.9459029997197`
- SGLang qps_median: `5.831363643266245`
- SGLang request_latency_ms_p50_median: `685.9459029997197`
- SGLang request_latency_ms_p95_median: `685.9459029997197`

## Correctness Against SGLang

- Top1 exact match rate: `1.0`
- TopK set overlap mean: `0.9599609375`
- Ordered prefix match mean: `0.1103515625`
- Token length match rate: `1.0`
- beamcmp-0 vs beamcmp-0: top1=True, topk_overlap=0.9609375, gr_token_len=3, sglang_token_len=3, top1_score_delta=-4.251989364624023, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=26, common_prefix_count=4
- beamcmp-1 vs beamcmp-1: top1=True, topk_overlap=0.97265625, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.763333320617676, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=23, common_prefix_count=10
- beamcmp-2 vs beamcmp-2: top1=True, topk_overlap=0.95703125, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.569195111592611, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=46, common_prefix_count=17
- beamcmp-3 vs beamcmp-3: top1=True, topk_overlap=0.94921875, gr_token_len=3, sglang_token_len=3, top1_score_delta=-5.366792043050131, gr_top1_rank_in_sglang=0, sglang_top1_rank_in_gr=0, same_position_count=18, common_prefix_count=7

## Rank Movement Summary

- beamcmp-0: {'overlap_count': 246, 'same_rank_count': 26, 'within_1_count': 59, 'within_5_count': 128, 'within_10_count': 190, 'mean': 0.43902439024390244, 'median': 0.0, 'max_abs': 41}
- beamcmp-1: {'overlap_count': 249, 'same_rank_count': 23, 'within_1_count': 43, 'within_5_count': 111, 'within_10_count': 178, 'mean': -0.5140562248995983, 'median': 0.0, 'max_abs': 39}
- beamcmp-2: {'overlap_count': 245, 'same_rank_count': 46, 'within_1_count': 80, 'within_5_count': 153, 'within_10_count': 194, 'mean': 0.14285714285714285, 'median': 0.0, 'max_abs': 43}
- beamcmp-3: {'overlap_count': 243, 'same_rank_count': 18, 'within_1_count': 50, 'within_5_count': 108, 'within_10_count': 161, 'mean': -3.271604938271605, 'median': -3.0, 'max_abs': 59}

## Top1 Cross Scores

- beamcmp-0: GR top1=(7, 15, 13) score=-6.379634857177734 SGLang-score=-2.127645492553711 scaled_delta=0.0033016204833984375; SGLang top1=(7, 15, 13) score=-2.127645492553711 GR-score=-6.379634857177734 scaled_delta=0.0033016204833984375
- beamcmp-1: GR top1=(195, 195, 195) score=-8.6263427734375 SGLang-score=-2.863009452819824 scaled_delta=-0.037314414978027344; SGLang top1=(195, 195, 195) score=-2.863009452819824 GR-score=-8.6263427734375 scaled_delta=-0.037314414978027344
- beamcmp-2: GR top1=(16, 17, 18) score=-8.359439849853516 SGLang-score=-2.790244738260905 scaled_delta=0.011294364929199219; SGLang top1=(16, 17, 18) score=-2.790244738260905 GR-score=-8.359439849853516 scaled_delta=0.011294364929199219
- beamcmp-3: GR top1=(17, 15, 16) score=-8.052204132080078 SGLang-score=-2.6854120890299478 scaled_delta=0.004032135009765625; SGLang top1=(17, 15, 16) score=-2.6854120890299478 GR-score=-8.052204132080078 scaled_delta=0.004032135009765625

## Scaled Score Delta Summary

- beamcmp-0: GR - SGLang\*token_len={'count': 246, 'mean': 0.00761564572652181, 'median': -0.005687713623046875, 'p05': -0.09845924377441406, 'p95': 0.13709735870361328, 'min': -0.15079689025878906, 'max': 0.27507877349853516}, corr=0.9966804307054224
- beamcmp-1: GR - SGLang\*token_len={'count': 249, 'mean': -0.025327268853245013, 'median': -0.035683631896972656, 'p05': -0.12413406372070312, 'p95': 0.07886791229248047, 'min': -0.23561668395996094, 'max': 0.21237754821777344}, corr=0.9953110328561947
- beamcmp-2: GR - SGLang\*token_len={'count': 245, 'mean': 0.01042193977200255, 'median': 0.008680343627929688, 'p05': -0.08390426635742188, 'p95': 0.114654541015625, 'min': -0.20890426635742188, 'max': 0.1578807830810547}, corr=0.9974815795153597
- beamcmp-3: GR - SGLang\*token_len={'count': 243, 'mean': -0.004635834399564767, 'median': 0.0012426376342773438, 'p05': -0.1425933837890625, 'p95': 0.10661602020263672, 'min': -0.2762870788574219, 'max': 0.21995258331298828}, corr=0.994631602546875

## Caveats

- GR max_decode_steps counts decode iterations after the initial prefill beam selection; fixed-length GR outputs normally contain GR decode_steps + 1 token ids.
- SGLang score semantics may differ from GR cumulative logprob/logit score.
- For fixed-length outputs, SGLang sequence_score appears to be length-normalized; the report also includes SGLang score scaled by token length for diagnostics.
- If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.
- Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.
