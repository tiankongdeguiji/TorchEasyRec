# GR vs SGLang nsys Breakdown

Primary rows use active CUDA/stage windows and pure kernel-name buckets. This avoids mixing CUDA Graph visibility differences with SGLang module NVTX buckets.

## Overall

| Metric / stage                           |      GR |  SGLang | Notes                                                               |
| ---------------------------------------- | ------: | ------: | ------------------------------------------------------------------- |
| Active CUDA window                       | 261.109 | 462.344 | first-to-last captured kernel; avoids CUDA Graph precapture runtime |
| CUDA runtime API total                   | 221.757 | 431.567 | runtime API duration clipped to the active CUDA window              |
| CUDA graph API total                     |       0 |   0.404 | CUDA graph runtime API duration clipped to the active CUDA window   |
| CPU runtime gaps >50us                   |   7.413 |   7.927 | gaps between CUDA runtime calls inside the active window            |
| Kernel total                             | 226.627 | 445.809 | sum of captured CUDA kernel durations                               |
| Decode attention kernels                 |  18.787 |     n/a | attention bucket inside the decode stage                            |
| Other kernels excluding decode attention | 207.840 | 445.809 | overall kernel total minus decode-stage attention kernels           |
| CUDA graph launches                      |       0 |      29 | cudaGraphLaunch runtime calls in the active window                  |
| Kernel launches                          |    1261 |    1925 | kernel rows overlapping the active window                           |

## Prefill Stage

| Metric / stage        |      GR | SGLang | Notes                                                                        |
| --------------------- | ------: | -----: | ---------------------------------------------------------------------------- |
| Stage total           | 187.420 |    n/a | prefill stage window from NVTX boundary; includes host gaps inside the stage |
| Attention kernels     |   0.000 |    n/a | prefill attention for prefill; decode attention for decode                   |
| Non-attention kernels | 178.842 |    n/a | stage kernel total minus stage attention kernels                             |
| CPU overhead          |   8.579 |    n/a | stage total minus stage kernel total; rough host/runtime/bubble component    |

## Decode Stage

| Metric / stage        |     GR | SGLang | Notes                                                                       |
| --------------------- | -----: | -----: | --------------------------------------------------------------------------- |
| Stage total           | 74.187 |    n/a | decode stage window from NVTX boundary; includes host gaps inside the stage |
| Attention kernels     | 18.787 |    n/a | prefill attention for prefill; decode attention for decode                  |
| Non-attention kernels | 28.998 |    n/a | stage kernel total minus stage attention kernels                            |
| CPU overhead          | 26.402 |    n/a | stage total minus stage kernel total; rough host/runtime/bubble component   |

## Legacy Flat Details

| Metric / stage             |       GR |    SGLang | Notes                                                                   |
| -------------------------- | -------: | --------: | ----------------------------------------------------------------------- |
| CUDA capture window        |  267.091 |   492.157 | raw kernel/runtime event window; can include CUDA Graph precapture      |
| Kernel total               |  226.627 |   445.809 | sum of all captured CUDA kernel durations                               |
| Non-attention kernel total |  207.840 |   128.543 | legacy mixed bucket; prefer the stage tables above                      |
| CUDA window - kernel total |   40.464 |    46.348 | rough host/runtime/scheduling gap; kernels may overlap                  |
| CUDA runtime API total     |  221.891 |   431.746 | raw sum of captured CUDA runtime API durations                          |
| NVTX span                  | 3087.313 | 14365.093 | diagnostic only; may include long-lived ranges                          |
| prefill NVTX total         |  856.147 |       n/a | legacy broad NVTX-name sum; can double count nested ranges              |
| decode step 1              |   36.519 |       n/a | GR has initial token selection in prefill path                          |
| decode step 2              |   37.615 |       n/a | GR has initial token selection in prefill path                          |
| decode step 3              |      n/a |       n/a | GR has initial token selection in prefill path                          |
| CUDA graph launches        |        0 |        29 | raw CUDA runtime API calls                                              |
| kernel launch count        |     1261 |      1925 | captured kernel rows                                                    |
| CPU gaps                   |   13.223 |    37.426 | raw gaps between CUDA runtime calls >50us                               |
| topK / beam selection      |    6.362 |     6.847 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| logits / log_softmax       |    0.034 |     3.972 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| QKV / qk_norm_rope         |   40.776 |     3.659 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| attention                  |   18.787 |   317.266 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| GEMM / linear              |   90.516 |    92.027 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| MLP activation             |    8.497 |     6.672 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| RMSNorm / layernorm        |    4.643 |     3.180 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| memory copy / fill         |    9.683 |     5.138 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| NVTX qkv_proj              |   25.286 |       n/a | fine-grained module/stage NVTX bucket                                   |
| NVTX qk_norm_rope          |   15.490 |       n/a | fine-grained module/stage NVTX bucket                                   |
| NVTX logits                |    0.319 |       n/a | fine-grained module/stage NVTX bucket                                   |
| NVTX rmsnorm               |    4.080 |       n/a | fine-grained module/stage NVTX bucket                                   |

## Overall Kernel Buckets

| bucket                  |  GR ms | SGLang ms |
| ----------------------- | -----: | --------: |
| `GEMM / linear`         | 90.516 |    92.027 |
| `MLP activation`        |  8.497 |     6.672 |
| `QKV / qk_norm_rope`    | 13.788 |     3.659 |
| `RMSNorm / layernorm`   |  4.643 |     3.180 |
| `attention`             | 18.787 |   317.266 |
| `logits / log_softmax`  |  0.034 |     3.972 |
| `memory copy / fill`    |  9.683 |     5.138 |
| `other`                 | 74.317 |     7.046 |
| `topK / beam selection` |  6.362 |     6.847 |

## Top GR Kernels

- `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (int)4, (bool)0, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)128, (int)128, (int)64, (int)4, cutlass::bfloat16_t>>, (bool)0, (bool)1, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0, (bool)0>(pytorch_flash::Flash_fwd_params)`: 61.534 ms, count=28
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)`: 52.941 ms, count=58
- `kernel_cutlass_kernel_srcsm120flash_fwdFlashAttentionForwardSm120_object_at__tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16g_0`: 18.787 ms, count=56
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8>(T1::Params)`: 16.539 ms, count=28
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8>(T1::Params)`: 12.692 ms, count=84
- `void silu_and_mul_packed_vec_kernel<c10::BFloat16, (int)8>(const T1 *, T1 *, long, long)`: 8.497 ms, count=84
- `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 12)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`: 7.955 ms, count=168
- `void write_packed_qkv_prefill_kv_kernel<c10::BFloat16>(const T1 *, const T1 *, T1 *, T1 *, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, long)`: 7.313 ms, count=28
- `void <unnamed>::fused_rope_kernel<(bool)1, (long)128, (bool)1, __nv_bfloat16, int, (unsigned int)16>(<unnamed>::FusedRopeParams)`: 6.476 ms, count=84
- `void <unnamed>::fused_qknorm_warp<(long)128, (bool)1, __nv_bfloat16>(<unnamed>::QKNormParams)`: 6.177 ms, count=84
- `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign128o102410241_tensorptrbf16gmemalign_0`: 4.464 ms, count=111
- `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)`: 4.445 ms, count=20

## Top GR CUDA Runtime Calls

- `cudaDeviceSynchronize_v3020`: 217.085 ms, count=1722
- `cudaLaunchKernel_v7000`: 1.654 ms, count=527
- `cudaLaunchKernelExC_v11060`: 1.296 ms, count=395
- `cuLaunchKernel`: 1.021 ms, count=339
- `cudaMemsetAsync_v3020`: 0.339 ms, count=122
- `cudaMemcpyAsync_v3020`: 0.137 ms, count=21
- `cudaStreamSynchronize_v3020`: 0.116 ms, count=21
- `cuKernelGetFunction`: 0.114 ms, count=339
- `cuKernelGetName`: 0.108 ms, count=754
- `cuProfilerStart`: 0.012 ms, count=1
- `cudaStreamIsCapturing_v10000`: 0.009 ms, count=21

## Top GR NVTX Ranges

- `continuous.prefill`: 187.384 ms, count=1
- `model.forward_prefill`: 186.301 ms, count=1
- `continuous.decode_microbatch_total`: 74.134 ms, count=2
- `model.forward_decode_step`: 58.124 ms, count=2
- `continuous.beam_selection`: 14.650 ms, count=2
- `prefill.layer0.total`: 6.776 ms, count=1
- `prefill.layer26.total`: 6.761 ms, count=1
- `prefill.layer27.total`: 6.752 ms, count=1
- `prefill.layer23.total`: 6.752 ms, count=1
- `prefill.layer24.total`: 6.745 ms, count=1
- `prefill.layer25.total`: 6.744 ms, count=1
- `prefill.layer20.total`: 6.733 ms, count=1
- `prefill.layer21.total`: 6.719 ms, count=1
- `prefill.layer22.total`: 6.719 ms, count=1
- `prefill.layer18.total`: 6.684 ms, count=1
- `prefill.layer19.total`: 6.682 ms, count=1
- `prefill.layer17.total`: 6.672 ms, count=1
- `prefill.layer1.total`: 6.569 ms, count=1
- `prefill.layer16.total`: 6.551 ms, count=1
- `prefill.layer3.total`: 6.548 ms, count=1

## Top SGLang Kernels

- `void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned int)1, (unsigned int)8, (unsigned int)16, (unsigned int)2, (unsigned int)4, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>, Params>(T9)`: 257.719 ms, count=56
- `void flashinfer::BatchPrefillWithRaggedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)1, (unsigned int)128, (unsigned int)2, (unsigned int)2, (unsigned int)8, (unsigned int)8, (unsigned int)4, (unsigned int)1, (flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, int, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>>, RaggedParams>(T2)`: 41.827 ms, count=28
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)`: 32.617 ms, count=140
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)`: 25.328 ms, count=30
- `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)1, (unsigned int)128, (unsigned int)2, (unsigned int)2, (unsigned int)8, (unsigned int)8, (unsigned int)4, (unsigned int)1, (flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, int, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>>, PagedParams>(T2)`: 17.721 ms, count=28
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8>(T1::Params)`: 16.219 ms, count=56
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8>(T1::Params)`: 12.779 ms, count=112
- `void <unnamed>::act_and_mul_kernel<__nv_bfloat16, (<unnamed>::ActivationKind)0, (bool)1, (bool)0>(<unnamed>::ActivationParams)`: 6.672 ms, count=112
- `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)`: 4.396 ms, count=8
- `void <unnamed>::fused_qknorm_warp<(long)128, (bool)1, __nv_bfloat16>(<unnamed>::QKNormParams)`: 4.146 ms, count=112
- `void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, at::native::<unnamed>::LogSoftMaxForwardEpilogue>(T4 *, const T2 *, int)`: 3.972 ms, count=4
- `void <unnamed>::fused_rope_kernel<(bool)1, (long)128, (bool)1, __nv_bfloat16, long, (unsigned int)16>(<unnamed>::FusedRopeParams)`: 3.659 ms, count=112

## Top SGLang CUDA Runtime Calls

- `cudaMemcpyAsync_v3020`: 423.965 ms, count=192
- `cudaLaunchKernel_v7000`: 2.675 ms, count=675
- `cudaLaunchKernelExC_v11060`: 2.259 ms, count=649
- `cuLaunchKernel`: 1.030 ms, count=339
- `cudaStreamSynchronize_v3020`: 0.600 ms, count=126
- `cudaGraphLaunch_v10000`: 0.404 ms, count=29
- `cudaMemsetAsync_v3020`: 0.378 ms, count=140
- `cuKernelGetName`: 0.134 ms, count=846
- `cuKernelGetFunction`: 0.100 ms, count=339
- `cuLaunchKernelEx`: 0.074 ms, count=8
- `cudaStreamIsCapturing_v10000`: 0.062 ms, count=152
- `cudaEventQuery_v3020`: 0.033 ms, count=7

## Top SGLang NVTX Ranges

- `cub::DeviceScan::InclusiveSumByKey`: 0.350 ms, count=28
- `cub::DeviceScan::InclusiveScan`: 0.164 ms, count=9
- `cub::DeviceSelect::Unique`: 0.078 ms, count=6
- `cub::DeviceRadixSort`: 0.054 ms, count=7
- `cub::DeviceSelect::Flagged`: 0.050 ms, count=3
- `cub::DeviceReduce::Sum`: 0.030 ms, count=3
- `cub::DeviceRunLengthEncode::Encode`: 0.020 ms, count=1
- `CCCL`: 0.000 ms, count=1

## NVTX Bucket Totals

| bucket               |  GR ms | SGLang ms |
| -------------------- | -----: | --------: |
| `MLP`                | 77.828 |       n/a |
| `QKV / qk_norm_rope` | 40.776 |       n/a |
| `logits`             |  0.319 |       n/a |
| `qk_norm_rope`       | 15.490 |       n/a |
| `qkv_proj`           | 25.286 |       n/a |
| `rmsnorm`            |  4.080 |       n/a |

## Diagnostic Notes

- SGLang module NVTX buckets are empty. For per-layer/per-op SGLang breakdown, rerun with SGLANG_PROFILE_MODULES=1; use the non-profile run for fair latency.
