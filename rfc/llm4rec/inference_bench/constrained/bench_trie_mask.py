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

"""Microbench the CURRENT sid-gr-inference trie mask construction per decode step.

Times the real TrieItemMaskProvider.initial_mask / step_mask calls against a
fabricated trie-consistent BeamPath (what the continuous scheduler passes per
request per step), decomposed by step, plus two vectorized reference variants
that keep the same trie walk but batch the tensor writes — the attainable
floor without changing the algorithm.

Requires gr_inference importable (pip install -e examples/sid-gr-inference).
"""

import argparse
import json
import random
import time
from types import SimpleNamespace

import torch
from gr_inference.gr_kv.beam_path import BeamPath
from gr_inference.gr_runtime.item_constraints import SemanticItemCatalog


def _time(fn, iters: int, sync: bool) -> float:
    fn()  # warmup
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    """Run the microbench and print one JSON result line per timing."""
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--backend", choices=("python", "static"), default="python")
    p.add_argument("--beam-width", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=151936)
    p.add_argument("--eos-token-id", type=int, default=151645)
    p.add_argument("--device", default="cuda")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    dev = torch.device(args.device)
    sync = dev.type == "cuda"
    width = args.beam_width

    t0 = time.perf_counter()
    catalog = SemanticItemCatalog.from_jsonl(args.catalog)
    provider = catalog.provider(
        vocab_size=args.vocab_size,
        eos_token_id=args.eos_token_id,
        mask_backend=args.backend,
    )
    load_ms = (time.perf_counter() - t0) * 1000.0

    trie = provider.trie
    l1_tokens = sorted(trie.allowed_next(()))
    # trie-consistent 2-step beam path: step0 picks L1 tokens, step1 valid L2
    beams_l1 = [rng.choice(l1_tokens) for _ in range(width)]
    beams_l2 = [rng.choice(sorted(trie.allowed_next((t,)))) for t in beams_l1]
    path = BeamPath(max_decode_steps=3, max_beam_width=width)
    path.append([0] * width, beams_l1, [0.0] * width)
    gen_step1 = SimpleNamespace(beam_path=path)

    path2 = BeamPath(max_decode_steps=3, max_beam_width=width)
    path2.append([0] * width, beams_l1, [0.0] * width)
    path2.append(list(range(width)), beams_l2, [0.0] * width)
    gen_step2 = SimpleNamespace(beam_path=path2)

    logits = torch.zeros((1, width, args.vocab_size), device=dev)

    def current_step_mask(gen):
        return provider.step_mask(gen, logits)

    # reference A: same walk, one index_put per beam instead of per token
    def rowbatched_step_mask(gen):
        mask = torch.zeros((width, args.vocab_size), dtype=torch.bool, device=dev)
        for beam in range(width):
            allowed = [
                t
                for t in provider.allowed_next(gen.beam_path.token_trace(beam))
                if 0 <= t < args.vocab_size
            ]
            if allowed:
                mask[beam, torch.tensor(allowed, device=dev)] = True
        return mask

    # reference B: same walk, build on host then one H2D copy
    def hostbuilt_step_mask(gen):
        mask = torch.zeros((width, args.vocab_size), dtype=torch.bool)
        for beam in range(width):
            allowed = [
                t
                for t in provider.allowed_next(gen.beam_path.token_trace(beam))
                if 0 <= t < args.vocab_size
            ]
            if allowed:
                mask[beam, torch.tensor(allowed)] = True
        return mask.to(dev, non_blocking=False)

    results = {
        "catalog": args.catalog,
        "backend": args.backend,
        "items": catalog.item_count,
        "device": args.device,
        "beam_width": width,
        "catalog_load_and_trie_build_ms": round(load_ms, 1),
        "initial_mask_ms": round(
            _time(lambda: provider.initial_mask(logits), args.iters, sync), 2
        ),
        "step1_mask_current_ms": round(
            _time(lambda: current_step_mask(gen_step1), max(args.iters // 2, 2), sync),
            2,
        ),
        "step2_mask_current_ms": round(
            _time(lambda: current_step_mask(gen_step2), args.iters, sync), 2
        ),
        "step1_allowed_tokens_total": sum(
            len(provider.allowed_next((t,))) for t in beams_l1
        ),
    }
    if args.backend == "python":
        results["step1_mask_rowbatched_ms"] = round(
            _time(lambda: rowbatched_step_mask(gen_step1), args.iters, sync), 2
        )
        results["step1_mask_hostbuilt_ms"] = round(
            _time(lambda: hostbuilt_step_mask(gen_step1), args.iters, sync), 2
        )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
