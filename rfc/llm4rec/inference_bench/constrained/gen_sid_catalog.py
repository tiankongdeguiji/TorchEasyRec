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

"""Generate a synthetic SID item catalog JSONL for sid-gr-inference.

Items are 3-level semantic IDs over per-level 8192-way codebooks mapped to
disjoint token-id ranges inside the Qwen3 vocab (mirrors an RQ-VAE itemic
token layout). Output schema matches SemanticItemCatalog.from_jsonl defaults:
{"item_id": ..., "token_ids": [t1, t2, t3]}.
"""

import argparse
import json
import random
from collections import defaultdict


def main() -> None:
    """Generate the catalog and print trie fan-out stats."""
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--codebook-size", type=int, default=8192)
    # disjoint per-level token ranges, all < Qwen3 vocab 151936
    p.add_argument("--level-bases", type=int, nargs=3, default=[1000, 10000, 20000])
    args = p.parse_args()

    rng = random.Random(args.seed)
    k = args.codebook_size
    b1, b2, b3 = args.level_bases
    assert b3 + k <= 151936, "token ids must stay inside the Qwen3 vocab"

    paths = set()
    while len(paths) < args.items:
        paths.add(
            (
                b1 + rng.randrange(k),
                b2 + rng.randrange(k),
                b3 + rng.randrange(k),
            )
        )

    l2_children = defaultdict(set)
    l3_children = defaultdict(set)
    for t1, t2, t3 in paths:
        l2_children[t1].add(t2)
        l3_children[(t1, t2)].add(t3)

    with open(args.out, "w", encoding="utf-8") as f:
        for i, path in enumerate(sorted(paths)):
            f.write(
                json.dumps({"item_id": f"item_{i}", "token_ids": list(path)}) + "\n"
            )

    l1_fanout = len(l2_children)
    avg_l2 = sum(len(v) for v in l2_children.values()) / max(l1_fanout, 1)
    avg_l3 = sum(len(v) for v in l3_children.values()) / max(len(l3_children), 1)
    stats = {
        "items": len(paths),
        "l1_fanout": l1_fanout,
        "avg_l2_children": round(avg_l2, 2),
        "avg_l3_children": round(avg_l3, 3),
        # per request: initial_mask writes = l1_fanout;
        # step-1 mask writes ~= beam_width * avg_l2_children
        "est_step1_mask_writes_beam256": int(256 * avg_l2),
    }
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
