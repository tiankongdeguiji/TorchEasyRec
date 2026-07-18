# GR vs SGLang nsys Breakdown

Primary rows use active CUDA/stage windows and pure kernel-name buckets. This avoids mixing CUDA Graph visibility differences with SGLang module NVTX buckets.

## Overall

| Metric / stage                           |     GR |  SGLang | Notes                                                               |
| ---------------------------------------- | -----: | ------: | ------------------------------------------------------------------- |
| Active CUDA window                       | 93.554 | 109.327 | first-to-last captured kernel; avoids CUDA Graph precapture runtime |
| CUDA runtime API total                   | 54.659 |  86.543 | runtime API duration clipped to the active CUDA window              |
| CUDA graph API total                     |      0 |   0.451 | CUDA graph runtime API duration clipped to the active CUDA window   |
| CPU runtime gaps >50us                   |  7.502 |   6.042 | gaps between CUDA runtime calls inside the active window            |
| Kernel total                             | 58.579 |  96.281 | sum of captured CUDA kernel durations                               |
| Decode attention kernels                 |  4.705 |     n/a | attention bucket inside the decode stage                            |
| Other kernels excluding decode attention | 53.875 |  96.281 | overall kernel total minus decode-stage attention kernels           |
| CUDA graph launches                      |      0 |      29 | cudaGraphLaunch runtime calls in the active window                  |
| Kernel launches                          |   1261 |    1558 | kernel rows overlapping the active window                           |

## Prefill Stage

| Metric / stage        |     GR | SGLang | Notes                                                                        |
| --------------------- | -----: | -----: | ---------------------------------------------------------------------------- |
| Stage total           | 33.845 |    n/a | prefill stage window from NVTX boundary; includes host gaps inside the stage |
| Attention kernels     |  0.000 |    n/a | prefill attention for prefill; decode attention for decode                   |
| Non-attention kernels | 25.075 |    n/a | stage kernel total minus stage attention kernels                             |
| CPU overhead          |  8.771 |    n/a | stage total minus stage kernel total; rough host/runtime/bubble component    |

## Decode Stage

| Metric / stage        |     GR | SGLang | Notes                                                                       |
| --------------------- | -----: | -----: | --------------------------------------------------------------------------- |
| Stage total           | 60.262 |    n/a | decode stage window from NVTX boundary; includes host gaps inside the stage |
| Attention kernels     |  4.705 |    n/a | prefill attention for prefill; decode attention for decode                  |
| Non-attention kernels | 28.800 |    n/a | stage kernel total minus stage attention kernels                            |
| CPU overhead          | 26.757 |    n/a | stage total minus stage kernel total; rough host/runtime/bubble component   |

## Legacy Flat Details

| Metric / stage             |       GR |    SGLang | Notes                                                                   |
| -------------------------- | -------: | --------: | ----------------------------------------------------------------------- |
| CUDA capture window        |   97.362 |   136.919 | raw kernel/runtime event window; can include CUDA Graph precapture      |
| Kernel total               |   58.579 |    96.281 | sum of all captured CUDA kernel durations                               |
| Non-attention kernel total |   53.875 |    45.960 | legacy mixed bucket; prefer the stage tables above                      |
| CUDA window - kernel total |   38.783 |    40.638 | rough host/runtime/scheduling gap; kernels may overlap                  |
| CUDA runtime API total     |   54.793 |    86.662 | raw sum of captured CUDA runtime API durations                          |
| NVTX span                  | 2574.556 | 12763.909 | diagnostic only; may include long-lived ranges                          |
| prefill NVTX total         |  152.525 |       n/a | legacy broad NVTX-name sum; can double count nested ranges              |
| decode step 1              |   30.143 |       n/a | GR has initial token selection in prefill path                          |
| decode step 2              |   30.065 |       n/a | GR has initial token selection in prefill path                          |
| decode step 3              |      n/a |       n/a | GR has initial token selection in prefill path                          |
| CUDA graph launches        |        0 |        29 | raw CUDA runtime API calls                                              |
| kernel launch count        |     1261 |      1558 | captured kernel rows                                                    |
| CPU gaps                   |   11.140 |    33.403 | raw gaps between CUDA runtime calls >50us                               |
| topK / beam selection      |    6.295 |     6.788 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| logits / log_softmax       |    0.034 |     3.960 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| QKV / qk_norm_rope         |   13.925 |     0.518 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| attention                  |    4.705 |    50.321 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| GEMM / linear              |   29.673 |    29.457 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| MLP activation             |    0.712 |     0.648 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| RMSNorm / layernorm        |    0.902 |     0.869 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| memory copy / fill         |    3.407 |     2.451 | legacy NVTX module bucket if present, else heuristic kernel-name bucket |
| NVTX qkv_proj              |    9.547 |       n/a | fine-grained module/stage NVTX bucket                                   |
| NVTX qk_norm_rope          |    4.378 |       n/a | fine-grained module/stage NVTX bucket                                   |
| NVTX logits                |    0.315 |       n/a | fine-grained module/stage NVTX bucket                                   |
| NVTX rmsnorm               |    4.042 |       n/a | fine-grained module/stage NVTX bucket                                   |

## Overall Kernel Buckets

| bucket                  |  GR ms | SGLang ms |
| ----------------------- | -----: | --------: |
| `GEMM / linear`         | 29.673 |    29.457 |
| `MLP activation`        |  0.712 |     0.648 |
| `QKV / qk_norm_rope`    |  1.859 |     0.518 |
| `RMSNorm / layernorm`   |  0.902 |     0.869 |
| `attention`             |  4.705 |    50.321 |
| `logits / log_softmax`  |  0.034 |     3.960 |
| `memory copy / fill`    |  3.407 |     2.451 |
| `other`                 | 10.993 |     1.269 |
| `topK / beam selection` |  6.295 |     6.788 |

## Top GR Kernels

- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)`: 13.974 ms, count=112
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params)`: 8.789 ms, count=112
- `kernel_cutlass_kernel_srcsm120flash_fwdFlashAttentionForwardSm120_object_at__tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16g_0`: 4.705 ms, count=56
- `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)`: 4.433 ms, count=20
- `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (int)4, (bool)0, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)128, (int)128, (int)64, (int)4, cutlass::bfloat16_t>>, (bool)0, (bool)1, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0, (bool)0>(pytorch_flash::Flash_fwd_params)`: 3.888 ms, count=28
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)`: 2.733 ms, count=2
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)`: 2.244 ms, count=2
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8>(T1::Params)`: 2.148 ms, count=56
- `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`: 2.113 ms, count=6
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)`: 1.764 ms, count=56
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7)`: 1.732 ms, count=3
- `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 12)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`: 1.669 ms, count=168

## Top GR CUDA Runtime Calls

- `cudaDeviceSynchronize_v3020`: 49.475 ms, count=1722
- `cudaLaunchKernel_v7000`: 1.837 ms, count=527
- `cudaLaunchKernelExC_v11060`: 1.467 ms, count=395
- `cuLaunchKernel`: 1.131 ms, count=339
- `cudaMemsetAsync_v3020`: 0.360 ms, count=122
- `cudaMemcpyAsync_v3020`: 0.140 ms, count=21
- `cuKernelGetFunction`: 0.124 ms, count=339
- `cuKernelGetName`: 0.124 ms, count=754
- `cudaStreamSynchronize_v3020`: 0.110 ms, count=21
- `cuProfilerStart`: 0.014 ms, count=1
- `cudaStreamIsCapturing_v10000`: 0.010 ms, count=21

## Top GR NVTX Ranges

- `continuous.decode_microbatch_total`: 60.208 ms, count=2
- `model.forward_decode_step`: 44.126 ms, count=2
- `continuous.prefill`: 33.800 ms, count=1
- `model.forward_prefill`: 32.702 ms, count=1
- `continuous.beam_selection`: 14.624 ms, count=2
- `layer1.decode_total`: 2.020 ms, count=2
- `layer0.decode_total`: 1.605 ms, count=2
- `layer2.decode_total`: 1.474 ms, count=2
- `layer3.decode_total`: 1.457 ms, count=2
- `layer8.decode_total`: 1.452 ms, count=2
- `layer4.decode_total`: 1.447 ms, count=2
- `layer15.decode_total`: 1.443 ms, count=2
- `layer10.decode_total`: 1.441 ms, count=2
- `prefill.layer0.total`: 1.435 ms, count=1
- `layer5.decode_total`: 1.434 ms, count=2
- `layer16.decode_total`: 1.431 ms, count=2
- `layer12.decode_total`: 1.429 ms, count=2
- `layer7.decode_total`: 1.429 ms, count=2
- `layer14.decode_total`: 1.429 ms, count=2
- `layer9.decode_total`: 1.427 ms, count=2

## Top SGLang Kernels

- `void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned int)1, (unsigned int)8, (unsigned int)16, (unsigned int)2, (unsigned int)4, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>, Params>(T9)`: 47.000 ms, count=56
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)`: 13.815 ms, count=112
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(T1::Params)`: 8.763 ms, count=112
- `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)`: 4.403 ms, count=8
- `void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, at::native::<unnamed>::LogSoftMaxForwardEpilogue>(T4 *, const T2 *, int)`: 3.960 ms, count=3
- `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)1, (unsigned int)128, (unsigned int)2, (unsigned int)2, (unsigned int)8, (unsigned int)8, (unsigned int)4, (unsigned int)1, (flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, int, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>>, PagedParams>(T2)`: 3.320 ms, count=28
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)`: 2.736 ms, count=2
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8>(T1::Params)`: 2.074 ms, count=56
- `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)`: 1.803 ms, count=56
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7)`: 1.717 ms, count=3
- `void at::native::mbtopk::gatherTopK<float, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, unsigned int, T2, at::cuda::detail::TensorInfo<T1, T2>, T2, at::cuda::detail::TensorInfo<long, T2>, T2, unsigned int, unsigned int, T1 *, unsigned int *, unsigned int *, unsigned int)`: 1.379 ms, count=2
- `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 12)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`: 0.684 ms, count=84

## Top SGLang CUDA Runtime Calls

- `cudaMemcpyAsync_v3020`: 80.621 ms, count=170
- `cudaLaunchKernel_v7000`: 2.287 ms, count=620
- `cudaLaunchKernelExC_v11060`: 1.459 ms, count=452
- `cuLaunchKernel`: 0.657 ms, count=227
- `cudaStreamSynchronize_v3020`: 0.537 ms, count=115
- `cudaGraphLaunch_v10000`: 0.451 ms, count=29
- `cudaMemsetAsync_v3020`: 0.348 ms, count=140
- `cuKernelGetName`: 0.100 ms, count=734
- `cuKernelGetFunction`: 0.066 ms, count=227
- `cuLaunchKernelEx`: 0.052 ms, count=5
- `cudaStreamIsCapturing_v10000`: 0.044 ms, count=130
- `cudaEventQuery_v3020`: 0.016 ms, count=4

## Top SGLang NVTX Ranges

- `cub::DeviceScan::InclusiveSumByKey`: 0.321 ms, count=28
- `cub::DeviceScan::InclusiveScan`: 0.104 ms, count=6
- `cub::DeviceSelect::Unique`: 0.070 ms, count=6
- `cub::DeviceRadixSort`: 0.047 ms, count=7
- `cub::DeviceSelect::Flagged`: 0.047 ms, count=3
- `cub::DeviceReduce::Sum`: 0.021 ms, count=3
- `cub::DeviceRunLengthEncode::Encode`: 0.017 ms, count=1
- `CCCL`: 0.000 ms, count=1

## NVTX Bucket Totals

| bucket               |  GR ms | SGLang ms |
| -------------------- | -----: | --------: |
| `MLP`                | 21.351 |       n/a |
| `QKV / qk_norm_rope` | 13.925 |       n/a |
| `logits`             |  0.315 |       n/a |
| `qk_norm_rope`       |  4.378 |       n/a |
| `qkv_proj`           |  9.547 |       n/a |
| `rmsnorm`            |  4.042 |       n/a |

## Diagnostic Notes

- SGLang module NVTX buckets are empty. For per-layer/per-op SGLang breakdown, rerun with SGLANG_PROFILE_MODULES=1; use the non-profile run for fair latency.
