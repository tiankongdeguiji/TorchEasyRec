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

"""Analyze fused and unfused Nsight Systems SQLite exports."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

_DISPATCH_PATTERNS = {
    "qk_norm": "fused_qknorm_warp",
    "rope": "fused_rope_kernel",
    "add_rmsnorm": "fused_add_rmsnorm",
    "mlp": "silu_and_mul_packed_vec_kernel",
}


def _category(short_name: str, demangled_name: str) -> str:
    name = f"{short_name} {demangled_name}".lower()
    if short_name == "Kernel2":
        return "gemm"
    if any(
        pattern in name
        for pattern in (
            "computeblock",
            "devicescanbykey",
            "gathertopk",
            "radixsort",
            "softmaxforward",
            "mbtopk",
        )
    ):
        return "beam_topk"
    if "flash_fwd_kernel" in name:
        return "attention_prefill"
    if "flashattentionforwardsm" in name or "kernel_cutlass_kernel_srcsm" in name:
        return "attention_decode"
    if "fused_rope" in name:
        return "rope"
    if "qknorm" in name or "rmsnorm" in name:
        return "norm"
    if "silu_and_mul" in name or "silu_kernel" in name:
        return "mlp_activation"
    if "binaryfunctor<c10::bfloat16" in name and "mulfunctor" in name:
        return "mlp_activation"
    if "write_packed_qkv" in name:
        return "kv_layout"
    if any(
        pattern in name
        for pattern in (
            "catarraybatchedcopy",
            "direct_copy",
            "scatter_gather",
            "vectorized_gather",
        )
    ):
        return "layout_copy"
    if any(
        pattern in name
        for pattern in (
            "elementwise_kernel",
            "reduce_kernel",
            "fill",
        )
    ):
        return "elementwise_other"
    return "other"


def _union_duration_ns(intervals: list[tuple[int, int]]) -> int:
    merged_ns = 0
    current_start = None
    current_end = None
    for start, end in sorted(intervals):
        if current_start is None:
            current_start, current_end = start, end
        elif start > current_end:
            merged_ns += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        merged_ns += current_end - current_start
    return merged_ns


def _read_trace(path: Path) -> dict:
    connection = sqlite3.connect(path)
    rows = connection.execute(
        """
        SELECT short.value, demangled.value, kernel.start, kernel.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
        JOIN StringIds AS short ON short.id = kernel.shortName
        JOIN StringIds AS demangled ON demangled.id = kernel.demangledName
        """
    ).fetchall()
    categories = defaultdict(lambda: {"launches": 0, "time_ms": 0.0})
    dispatch = {family: 0 for family in _DISPATCH_PATTERNS}
    intervals = []
    kernel_time_ns = 0
    for short_name, demangled_name, start, end in rows:
        duration_ns = end - start
        category = _category(short_name, demangled_name)
        categories[category]["launches"] += 1
        categories[category]["time_ms"] += duration_ns / 1e6
        kernel_time_ns += duration_ns
        intervals.append((start, end))
        full_name = f"{short_name} {demangled_name}".lower()
        for family, pattern in _DISPATCH_PATTERNS.items():
            if pattern in full_name:
                dispatch[family] += 1

    explicit_memory = {}
    for table, label in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
    ):
        memory_rows = connection.execute(f"SELECT start, end FROM {table}").fetchall()
        intervals.extend(memory_rows)
        explicit_memory[label] = {
            "operations": len(memory_rows),
            "time_ms": sum(end - start for start, end in memory_rows) / 1e6,
        }
    connection.close()

    trace_start = min(start for start, _ in intervals)
    trace_end = max(end for _, end in intervals)
    trace_span_ms = (trace_end - trace_start) / 1e6
    gpu_busy_ms = _union_duration_ns(intervals) / 1e6
    for values in categories.values():
        values["percent_kernel_time"] = values["time_ms"] / (kernel_time_ns / 1e6) * 100
    return {
        "source": path.name,
        "kernel_launches": len(rows),
        "kernel_time_ms": kernel_time_ns / 1e6,
        "trace_span_ms": trace_span_ms,
        "gpu_busy_union_ms": gpu_busy_ms,
        "uncovered_gap_ms": trace_span_ms - gpu_busy_ms,
        "categories": dict(sorted(categories.items())),
        "explicit_memory": explicit_memory,
        "dispatch_evidence": dispatch,
    }


def _validate_dispatch(fused: dict, unfused: dict) -> None:
    for family in _DISPATCH_PATTERNS:
        if fused["dispatch_evidence"][family] <= 0:
            raise ValueError(f"fused trace has no dispatch evidence for {family}")
        if unfused["dispatch_evidence"][family] != 0:
            raise ValueError(f"unfused trace still dispatched the {family} fusion")


def _write_markdown(result: dict, path: Path) -> None:
    fused = result["fused"]
    unfused = result["unfused"]
    categories = sorted(set(fused["categories"]) | set(unfused["categories"]))
    lines = [
        "# Nsight Systems fusion breakdown",
        "",
        "GPU time is the union and sum of CUDA activities inside the profiled",
        "measurement. `uncovered gap` is time between GPU activities; it includes",
        "host orchestration and synchronization but is not a CPU-sampling profile.",
        "",
        "| Metric | Fused | Strict unfused | Unfused - fused |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("trace_span_ms", "GPU trace span (ms)"),
        ("gpu_busy_union_ms", "GPU busy union (ms)"),
        ("kernel_time_ms", "Kernel time sum (ms)"),
        ("uncovered_gap_ms", "Uncovered gap (ms)"),
        ("kernel_launches", "Kernel launches"),
    ):
        fused_value = fused[key]
        unfused_value = unfused[key]
        lines.append(
            f"| {label} | {fused_value:.3f} | {unfused_value:.3f} | "
            f"{unfused_value - fused_value:+.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "| Kernel family | Fused ms / launches | "
                "Unfused ms / launches | Delta ms |"
            ),
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for category in categories:
        fused_row = fused["categories"].get(category, {"time_ms": 0, "launches": 0})
        unfused_row = unfused["categories"].get(category, {"time_ms": 0, "launches": 0})
        lines.append(
            f"| {category} | {fused_row['time_ms']:.3f} / "
            f"{fused_row['launches']} | {unfused_row['time_ms']:.3f} / "
            f"{unfused_row['launches']} | "
            f"{unfused_row['time_ms'] - fused_row['time_ms']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "| Fusion family | Fused launches | Strict-unfused launches | Gate |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for family in _DISPATCH_PATTERNS:
        fused_count = fused["dispatch_evidence"][family]
        unfused_count = unfused["dispatch_evidence"][family]
        lines.append(
            f"| {family} | {fused_count} | {unfused_count} | "
            f"{'PASS' if fused_count > 0 and unfused_count == 0 else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Generic compositional Torch kernels cannot be assigned perfectly to",
            "RoPE versus normalization from Nsight kernel names alone. The paired",
            "single-family offline ablations are the causal attribution source;",
            "the `elementwise_other` and `layout_copy` rows show their aggregate",
            "GPU cost.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    """Parse traces, enforce dispatch gates, and write machine/human summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--unfused", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "fused": _read_trace(args.fused),
        "unfused": _read_trace(args.unfused),
    }
    _validate_dispatch(result["fused"], result["unfused"])
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_markdown(result, args.output_markdown)


if __name__ == "__main__":
    main()
