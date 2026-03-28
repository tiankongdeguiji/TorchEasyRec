# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Monkey-patch Triton autotuner to catch RuntimeError during benchmarking.
# Triton's autotuner only catches OutOfResources/PTXASError, but certain
# configs can trigger CUDA illegal memory access (RuntimeError) due to
# codegen bugs (e.g., Triton 3.6.0 on Hopper with large BLOCK_N * BLOCK_D).
# Without this patch, one bad config crashes the entire process.
import warnings as _warnings

import triton.runtime.autotuner as _triton_autotuner

_orig_bench = _triton_autotuner.Autotuner._bench


def _safe_bench(self, *args, config, **kwargs):
    try:
        return _orig_bench(self, *args, config=config, **kwargs)
    except RuntimeError as e:
        _warnings.warn(
            f"Triton autotuning: config {config} caused RuntimeError: {e}. "
            f"Skipping this config."
        )
        return [float("inf"), float("inf"), float("inf")]


_triton_autotuner.Autotuner._bench = _safe_bench
