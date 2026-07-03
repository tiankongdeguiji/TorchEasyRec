# GR vs SGLang Offline Accuracy

|  ctx | beam | batch | top1 exact | topK overlap | ordered prefix | token len match | output budget match | report                            |
| ---: | ---: | ----: | ---------: | -----------: | -------------: | --------------: | ------------------- | --------------------------------- |
| 1000 |  256 |     1 |      1.000 |        0.965 |          0.098 |           1.000 | True                | `compare_ctx1000_beam256_req1.md` |
| 1000 |  256 |     2 |      0.500 |        0.945 |          0.070 |           1.000 | True                | `compare_ctx1000_beam256_req2.md` |
| 1000 |  256 |     4 |      1.000 |        0.962 |          0.087 |           1.000 | True                | `compare_ctx1000_beam256_req4.md` |
| 1000 |  256 |     8 |      0.875 |        0.963 |          0.083 |           1.000 | True                | `compare_ctx1000_beam256_req8.md` |
| 5000 |  256 |     1 |      1.000 |        0.961 |          0.094 |           1.000 | True                | `compare_ctx5000_beam256_req1.md` |
| 5000 |  256 |     2 |      1.000 |        0.975 |          0.070 |           1.000 | True                | `compare_ctx5000_beam256_req2.md` |
| 5000 |  256 |     4 |      1.000 |        0.960 |          0.110 |           1.000 | True                | `compare_ctx5000_beam256_req4.md` |
| 5000 |  256 |     8 |      1.000 |        0.959 |          0.093 |           1.000 | True                | `compare_ctx5000_beam256_req8.md` |

Metric notes:

- top1 exact: whether GR and SGLang rank-1 token IDs match.
- topK overlap: set overlap between GR and SGLang beam candidates.
- output budget match should be true for the fixed-length offline benchmark.
