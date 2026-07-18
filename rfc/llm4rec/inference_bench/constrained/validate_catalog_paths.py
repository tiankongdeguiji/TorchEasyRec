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

"""Gate: constrained decode actually engaged - all beams must be catalog paths.

Sends one /generate to the GR server and asserts every returned beam's
token_ids (EOS-stripped) is an exact token path in the catalog JSONL.
"""

import argparse
import json
import urllib.request


def main() -> None:
    """Validate one constrained generation against the catalog."""
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--context-len", type=int, default=1000)
    p.add_argument("--beam-width", type=int, default=256)
    p.add_argument("--decode-steps", type=int, default=3)
    p.add_argument("--eos-token-id", type=int, default=151645)
    args = p.parse_args()

    paths = set()
    with open(args.catalog, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                paths.add(tuple(json.loads(line)["token_ids"]))

    body = json.dumps(
        {
            "input_ids": [1024 + (i % 32000) for i in range(args.context_len)],
            "sampling_params": {
                "max_new_tokens": args.decode_steps,
                "n": args.beam_width,
                "temperature": 0.0,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{args.host}:{args.port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read())

    first = out[0] if isinstance(out, list) else out
    beams = first.get("meta_info", {}).get("beam_results") or []
    assert beams, f"no beam_results in response: {json.dumps(first)[:500]}"
    bad = 0
    for beam in beams:
        toks = [t for t in beam["token_ids"] if t != args.eos_token_id]
        if tuple(toks) not in paths:
            bad += 1
    print(f"beams={len(beams)} off_catalog={bad}")
    assert bad == 0, f"{bad}/{len(beams)} beams decoded off-catalog paths"
    print("GATE_CONSTRAINED_PATHS_PASS")


if __name__ == "__main__":
    main()
