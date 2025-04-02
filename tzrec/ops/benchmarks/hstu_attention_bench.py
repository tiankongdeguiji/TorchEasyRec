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

import os
from typing import List, Optional, Tuple

import click
import pandas as pd
import torch

# @manual=//triton:triton
import triton

from tzrec.ops import Kernel
from tzrec.ops.hstu_attention import delta_hstu_mha, hstu_mha
from tzrec.utils.test_util import generate_sparse_seq_len


def _apply_sampling(
    lengths: torch.Tensor,
    alpha: float,
    max_seq_len: int,
) -> torch.Tensor:
    threshold = int(max_seq_len ** (alpha / 2))
    no_sample_prob = (max_seq_len**alpha) / torch.pow(lengths, 2)
    users_to_sample = torch.logical_and(
        lengths > threshold,
        torch.rand_like(no_sample_prob) < 1 - no_sample_prob,
    )
    lengths = torch.where(users_to_sample, threshold, lengths)
    return lengths


def _get_kernel(provider: str) -> Kernel:
    if provider == "triton":
        return Kernel.TRITON
    elif provider == "pytorch":
        return Kernel.PYTORCH
    else:
        raise ValueError(f"Unknown provider {provider}")


def _flops(
    batch_size: int,
    max_seqlen: int,
    attn_dim: int,
    hidden_dim: int,
    nheads: int,
    seq_offsets: torch.Tensor,
    mode: str = "fwd",
) -> float:
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    ratio = 2.0  # triangular masking
    f1 = 0.0
    f2 = 0.0
    for i in range(batch_size):
        seq_len = int((seq_offsets[i + 1] - seq_offsets[i]).item())
        # (QK^T), dQ = d(QK^T)K, dK^T = Q^Td(QK^T)
        f1 += 2 * nheads * attn_dim * seq_len**2 // ratio
        # (QK^T)V, d(QK^T) = dOV^T, dV = (QK^T)^TdO,
        f2 += 2 * nheads * hidden_dim * seq_len**2 // ratio
    if mode == "fwd":
        return f1 + f2  # computes (QK^T) and (QK^T)V
    elif mode == "bwd":
        return 3 * f1 + 2 * f2  # computes (QK^T), dQ, dK, dV, d(QK^T)
    else:
        return 4 * f1 + 3 * f2


@click.command()
@click.option(
    "--batch-size",
    type=int,
    default=512,
)
@click.option("--heads", type=int, default=4)
@click.option("--attn-dim", type=int, default=128)
@click.option("--hidden-dim", type=int, default=128)
@click.option("--max-seq-len-log2", type=int, default=13)
@click.option("--data-type", type=str, default="bf16")
@click.option("--seq-sparsity", type=float, default=0.95)
@click.option("--has-delta-q", type=bool, default=False)
@click.option("--delta-size", type=int, default=256)
@click.option("--target-size", type=int, default=20)
@click.option("--bench-backward", type=bool, default=True)
@click.option("--bench-forward", type=bool, default=True)
@click.option("--bench-pytorch", type=bool, default=False)
@click.option("--report-flops", type=bool, default=False)
@click.option("--return-result", type=bool, default=False)
@click.option("--max-attn-len", type=int, default=0)
@click.option("--contextual-seq-len", type=int, default=0)
@click.option("--sampling-alpha", type=float, default=2.0)
def main(  # noqa: C901
    batch_size: int,
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    max_seq_len_log2: int,
    data_type: str,
    seq_sparsity: float,
    has_delta_q: bool,
    delta_size: int,
    target_size: int,
    bench_backward: bool,
    bench_forward: bool,
    bench_pytorch: bool,
    report_flops: bool,
    return_result: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    sampling_alpha: float,
) -> Optional[Tuple[List[triton.testing.Benchmark], List[pd.DataFrame]]]:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if data_type == "fp32":
        dtype = torch.float32
    elif data_type == "fp16":
        dtype = torch.float16
    elif data_type == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported data type: {data_type}.")

    line_vals = ["triton"]
    line_names = ["Triton"]
    styles = [("red", "-")]
    if bench_pytorch:
        line_vals.append("pytorch")
        line_names.append("PyTorch")
        styles.append(("green", "-"))

    bench_backward = False if has_delta_q else bench_backward
    modes = []
    if bench_forward:
        modes.append("fwd")
    if bench_backward:
        modes.append("bwd")
    assert len(modes) > 0

    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(8, max_seq_len_log2)],
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            styles=styles,
            ylabel="ms",
            plot_name=f"hstu-attn-b{batch_size}-h{heads}-d{attn_dim}-v{hidden_dim}--sparsity{seq_sparsity}-{mode}-{dtype}-target{target_size}-mattn{max_attn_len}-c{contextual_seq_len}-sl_alpha{sampling_alpha}",
            args={
                "batch_size": batch_size,
                "heads": heads,
                "attn_dim": attn_dim,
                "hidden_dim": hidden_dim,
                "dtype": dtype,
                "mode": mode,
                "seq_sparsity": seq_sparsity,
                "has_delta_q": has_delta_q,
                "delta_size": delta_size,
                "target_size": target_size,
                "bench_backward": bench_backward,
                "report_flops": report_flops,
                "max_attn_len": max_attn_len,
                "contextual_seq_len": contextual_seq_len,
                "sampling_alpha": sampling_alpha,
            },
        )
        for mode in modes
    ]

    @triton.testing.perf_report(configs)
    def _bench_hstu_attention(
        batch_size: int,
        heads: int,
        seq_len: int,
        attn_dim: int,
        hidden_dim: int,
        mode: str,
        provider: str,
        dtype: torch.dtype,
        seq_sparsity: float,
        has_delta_q: bool,
        delta_size: int,
        target_size: int,
        bench_backward: bool,
        report_flops: bool,
        max_attn_len: int,
        contextual_seq_len: int,
        sampling_alpha: float,
    ) -> float:
        assert mode in ["fwd", "bwd"]
        warmup = 25
        rep = 1000
        torch.manual_seed(1001)  # for reproducibility
        alpha = 1.0 / attn_dim
        causal = True
        lengths = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=seq_len,
            sparsity=seq_sparsity,
            device=torch.device("cuda"),
        )
        lengths = _apply_sampling(lengths, sampling_alpha, max_seq_len=seq_len)
        if has_delta_q:
            lengths = lengths + delta_size
            num_targets = torch.ones_like(lengths) * delta_size
            seq_len = seq_len + delta_size
        else:
            delta_size = 0
            num_targets = None
            if target_size != 0:
                num_targets = torch.randint(
                    1,
                    target_size + 1,
                    (batch_size,),
                    device=lengths.device,
                    dtype=lengths.dtype,
                )
                num_targets = torch.where(num_targets > lengths, lengths, num_targets)
        max_attn_len = max_attn_len if max_attn_len < seq_len else seq_len
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)
        L = int(seq_offsets[-1].item())
        x = torch.empty(
            (L, heads, attn_dim * 2 + hidden_dim),
            dtype=dtype,
            device=torch.device("cuda"),
        ).uniform_(-0.01, 0.01)
        q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)
        delta_q = torch.empty(
            (batch_size * delta_size, heads, attn_dim),
            dtype=dtype,
            device=torch.device("cuda"),
        ).uniform_(-0.1, 0.1)
        delta_x_offsets = torch.arange(0, delta_size, device=torch.device("cuda"))
        delta_x_offsets = (seq_offsets[1:] - delta_size).view(
            batch_size, 1
        ) + delta_x_offsets.view(1, delta_size)
        delta_x_offsets = delta_x_offsets.view(-1)

        if bench_backward:
            q = q.requires_grad_(True)
            k = k.requires_grad_(True)
            v = v.requires_grad_(True)
        assert provider in ["triton", "pytorch"]
        if has_delta_q:
            fn = lambda: delta_hstu_mha(  # noqa E731
                max_seq_len=seq_len,
                alpha=alpha,
                delta_q=delta_q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                kernel=_get_kernel(provider),
            )
        else:
            fn = lambda: hstu_mha(  # noqa E73
                max_seq_len=seq_len,
                alpha=alpha,
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                causal=causal,
                dropout_pr=0.0,
                training=True,
                num_targets=num_targets,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                sort_by_length=True,
                kernel=_get_kernel(provider),
            )
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)  # noqa E731
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        all_flops = _flops(
            batch_size, seq_len, attn_dim, hidden_dim, heads, seq_offsets, mode
        )
        if has_delta_q:
            all_flops = all_flops / seq_len * delta_size
        if report_flops:
            return all_flops / ms / 1e9
        else:
            return ms

    df = _bench_hstu_attention.run(
        print_data=True,
        show_plots=False,
        save_path="/tmp/" + os.environ["USER"],
        return_df=return_result,
    )

    if return_result:
        return configs, df


if __name__ == "__main__":
    main()
