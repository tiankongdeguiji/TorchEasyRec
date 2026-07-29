# Nsight Systems fusion breakdown

GPU time is the union and sum of CUDA activities inside the profiled
measurement. `uncovered gap` is time between GPU activities; it includes
host orchestration and synchronization but is not a CPU-sampling profile.

| Metric               |    Fused | Strict unfused | Unfused - fused |
| -------------------- | -------: | -------------: | --------------: |
| GPU trace span (ms)  |   61.837 |         73.064 |         +11.227 |
| GPU busy union (ms)  |   58.190 |         68.376 |         +10.186 |
| Kernel time sum (ms) |   58.128 |         68.286 |         +10.158 |
| Uncovered gap (ms)   |    3.647 |          4.688 |          +1.041 |
| Kernel launches      | 1261.000 |       3472.000 |       +2211.000 |

| Kernel family     | Fused ms / launches | Unfused ms / launches | Delta ms |
| ----------------- | ------------------: | --------------------: | -------: |
| attention_decode  |          4.613 / 56 |            4.651 / 56 |   +0.038 |
| attention_prefill |          3.846 / 28 |            3.911 / 28 |   +0.065 |
| beam_topk         |         6.334 / 101 |           6.323 / 101 |   -0.011 |
| elementwise_other |          6.546 / 86 |         12.617 / 1541 |   +6.071 |
| gemm              |        29.460 / 339 |          29.358 / 339 |   -0.102 |
| kv_layout         |          1.330 / 28 |            0.318 / 28 |   -1.011 |
| layout_copy       |         3.411 / 200 |           7.421 / 872 |   +4.009 |
| mlp_activation    |          0.686 / 84 |           2.532 / 168 |   +1.847 |
| norm              |         1.393 / 255 |           1.155 / 339 |   -0.238 |
| rope              |          0.510 / 84 |             0.000 / 0 |   -0.510 |

| Fusion family | Fused launches | Strict-unfused launches | Gate |
| ------------- | -------------: | ----------------------: | ---- |
| qk_norm       |             84 |                       0 | PASS |
| rope          |             84 |                       0 | PASS |
| add_rmsnorm   |            111 |                       0 | PASS |
| mlp           |             84 |                       0 | PASS |

Generic compositional Torch kernels cannot be assigned perfectly to
RoPE versus normalization from Nsight kernel names alone. The paired
single-family offline ablations are the causal attribution source;
the `elementwise_other` and `layout_copy` rows show their aggregate
GPU cost.
