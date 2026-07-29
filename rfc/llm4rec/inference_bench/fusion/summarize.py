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

"""Validate and summarize the Qwen3 fusion benchmark artifacts."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _median(values) -> float:
    return float(statistics.median(float(value) for value in values))


def _online_record(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one benchmark record")
    row = rows[0]
    if row["completed"] != 64 or any(row["errors"]):
        raise ValueError(f"{path} did not complete 64 error-free requests")
    return row


def _beam_tokens(output: dict) -> list[tuple[int, ...]]:
    return [
        tuple(int(token) for token in beam["output_ids"])
        for beam in output["beam_results"]
    ]


def main() -> None:
    """Validate artifacts and print a compact result summary."""
    root = Path(sys.argv[1])
    print("OFFLINE")
    for batch in (1, 2, 4, 8):
        fused_a = _read_json(root / f"offline/fused_a_b{batch}.json")
        fused_b = _read_json(root / f"offline/fused_b_b{batch}.json")
        unfused = _read_json(root / f"offline/unfused_b{batch}.json")
        fused_samples = fused_a["wall_ms_samples"] + fused_b["wall_ms_samples"]
        fused_ms = _median(fused_samples)
        unfused_ms = float(unfused["wall_ms_median"])
        confirmation_delta = abs(
            float(fused_b["wall_ms_median"]) / float(fused_a["wall_ms_median"]) - 1
        )
        if confirmation_delta > 0.05:
            raise ValueError(f"batch {batch} fused confirmation drift exceeded 5%")
        for result in (fused_a, fused_b, unfused):
            metrics = result["scheduler_metrics"]
            if (
                metrics["succeeded_requests"] != batch
                or metrics["failed_requests"] != 0
            ):
                raise ValueError(f"batch {batch} contains a failed offline run")
        print(
            f"b={batch} fused_ms={fused_ms:.3f} unfused_ms={unfused_ms:.3f} "
            f"latency_speedup={unfused_ms / fused_ms:.4f} "
            f"fused_qps={batch / fused_ms * 1000:.3f} "
            f"unfused_qps={batch / unfused_ms * 1000:.3f}"
        )

    fused = _read_json(root / "accuracy/fused.json")
    unfused = _read_json(root / "accuracy/unfused.json")
    fused_by_workload = {row["workload_id"]: row for row in fused["outputs"]}
    unfused_by_workload = {row["workload_id"]: row for row in unfused["outputs"]}
    exact = []
    overlaps = []
    for workload_id in sorted(fused_by_workload):
        fused_tokens = _beam_tokens(fused_by_workload[workload_id])
        unfused_tokens = _beam_tokens(unfused_by_workload[workload_id])
        exact.append(fused_tokens[0] == unfused_tokens[0])
        overlaps.append(
            len(set(fused_tokens) & set(unfused_tokens))
            / max(len(set(unfused_tokens)), 1)
        )
    exact_rate = sum(exact) / len(exact)
    overlap = statistics.mean(overlaps)
    print(f"CORRECTNESS top1_exact={exact_rate:.6f} top256_overlap={overlap:.6f}")
    if exact_rate != 1.0 or overlap < 0.95:
        raise ValueError("fusion correctness gate failed")

    print("ONLINE")
    medians = {}
    for label in ("fused", "unfused"):
        rows = []
        for round_idx in (1, 2, 3):
            path = next((root / f"online/{label}/round{round_idx}").glob("*.jsonl"))
            rows.append(_online_record(path))
        medians[label] = {
            "req_s": _median(row["request_throughput"] for row in rows),
            "p50": _median(row["median_e2e_latency_ms"] for row in rows),
            "p90": _median(row["p90_e2e_latency_ms"] for row in rows),
            "p99": _median(row["p99_e2e_latency_ms"] for row in rows),
        }
        print(label, json.dumps(medians[label], sort_keys=True))
    throughput_speedup = medians["fused"]["req_s"] / medians["unfused"]["req_s"]
    print(
        "ONLINE_RATIOS",
        f"latency_speedup={medians['unfused']['p50'] / medians['fused']['p50']:.4f}",
        f"throughput_speedup={throughput_speedup:.4f}",
    )

    print("ABLATIONS")
    for family in ("qk_norm", "rope", "add_rmsnorm", "mlp"):
        result = _read_json(root / f"ablations/no_{family}.json")
        print(f"no_{family} wall_ms={float(result['wall_ms_median']):.3f}")
    max_fused = _read_json(root / "ablations/max_fused.json")
    print(f"max_fused wall_ms={float(max_fused['wall_ms_median']):.3f}")


if __name__ == "__main__":
    main()
