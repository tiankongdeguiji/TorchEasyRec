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

"""Gate: SGLang beam-search fork sanity - n beams with per-beam sequence_score.

Spawn-safe (sglang Engine forks subprocesses): keep the main guard.
"""

import os

import sglang as sgl


def main() -> None:
    """Run one beam-search generate and assert per-beam results are returned."""
    model_dir = os.environ.get("MODEL_DIR", "/mnt/data/hongsheng.jhs/models/Qwen3-1.7B")
    print("sglang from:", sgl.__file__)
    eng = sgl.Engine(
        model_path=model_dir,
        enable_beam_search=True,
        disable_radix_cache=True,
    )
    out = eng.generate(
        input_ids=[[1024 + i for i in range(128)]],
        sampling_params={"max_new_tokens": 3, "n": 8, "temperature": 0.0},
    )
    items = out if isinstance(out, list) else [out]
    first = items[0]
    meta = first.get("meta_info", {})
    beams = meta.get("beam_results") or []
    print(
        "num beam_results:", len(beams), "| sequence_score:", meta.get("sequence_score")
    )
    assert len(beams) == 8, f"expected 8 beam_results, got {len(beams)}"
    assert meta.get("sequence_score") is not None, "missing sequence_score"
    print("GATE_FORK_BEAM_PASS")
    eng.shutdown()


if __name__ == "__main__":
    main()
