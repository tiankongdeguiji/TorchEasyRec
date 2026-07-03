# GR vs SGLang Offline Accuracy

|  ctx | beam | batch | top1 exact | topK overlap | ordered prefix | token len match | output budget match | report                            |
| ---: | ---: | ----: | ---------: | -----------: | -------------: | --------------: | ------------------- | --------------------------------- |
| 1000 |  256 |     1 |      1.000 |        0.938 |          0.082 |           1.000 | True                | `compare_ctx1000_beam256_req1.md` |
| 1000 |  256 |     2 |      0.500 |        0.939 |          0.080 |           1.000 | True                | `compare_ctx1000_beam256_req2.md` |
| 1000 |  256 |     4 |      1.000 |        0.954 |          0.081 |           1.000 | True                | `compare_ctx1000_beam256_req4.md` |
| 1000 |  256 |     8 |      0.875 |        0.955 |          0.082 |           1.000 | True                | `compare_ctx1000_beam256_req8.md` |
| 5000 |  256 |     1 |      1.000 |        0.949 |          0.105 |           1.000 | True                | `compare_ctx5000_beam256_req1.md` |
| 5000 |  256 |     2 |      1.000 |        0.953 |          0.078 |           1.000 | True                | `compare_ctx5000_beam256_req2.md` |
| 5000 |  256 |     4 |      1.000 |        0.951 |          0.077 |           1.000 | True                | `compare_ctx5000_beam256_req4.md` |
| 5000 |  256 |     8 |      0.875 |        0.947 |          0.064 |           1.000 | True                | `compare_ctx5000_beam256_req8.md` |

Metric notes:

- top1 exact: whether GR and SGLang rank-1 token IDs match.
- topK overlap: set overlap between GR and SGLang beam candidates.
- output budget match should be true for the fixed-length offline benchmark.
