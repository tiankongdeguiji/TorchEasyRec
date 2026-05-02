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

# The hstu_attn wheel exposes a unified, SM-dispatched
# ``hstu_attn_varlen_func`` (BF16/FP16 on Ampere/Ada, BF16/FP16 + FP8 on
# Hopper) backed by its own ``autograd.Function`` that handles
# saved-tensor management, FP8 quantization (when requested), and the
# kernel call. We delegate to that rather than maintain a parallel
# ``torch.library`` op + ``autograd.Function`` here.
#
# The wheel's ``Function`` is a plain ``autograd.Function``, not a
# ``@torch.library.custom_op``, so the multi-thread AOTI deadlock that
# previously motivated the in-house low-level ``torch.library`` wrapping
# does not apply.

import functools
from typing import Optional

import torch

from tzrec.utils.logging_util import logger

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

_FP8_HEAD_DIMS = (64, 128, 256)


@functools.lru_cache(maxsize=1)
def _assert_fp8_capable() -> None:
    """Raise if the current CUDA device cannot run the FP8 attention path."""
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 hstu requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(
            f"FP8 hstu requires SM>=90 (Hopper); current device is "
            f"sm{major}{minor}. Disable fp8_quant_mode (set to -1) "
            "or run on Hopper."
        )


def _assert_fp8_headdim(head_dim: int) -> None:
    """Reject head dims unsupported by the wheel's FP8 kernels."""
    if head_dim not in _FP8_HEAD_DIMS:
        raise ValueError(
            f"FP8 hstu requires attention_dim in {_FP8_HEAD_DIMS}, got {head_dim}."
        )


@functools.lru_cache(maxsize=1)
def _warn_fp8_mode_zero_unsafe() -> None:
    logger.warning(
        "fp8_quant_mode=0 uses descale=1.0 -- no overflow protection. "
        "Use mode 4 (per-batch) or 5 (per-tensor) for production."
    )


class _ContiguousGradPassthrough(torch.autograd.Function):
    """Identity in forward; ``grad.contiguous()`` in backward.

    The wheel's ``_hstu_attn_varlen_backward`` requires ``dout`` to have a
    contiguous last dimension, but standard PyTorch reductions (e.g.
    ``out.sum().backward()``) propagate a broadcast / expanded grad that
    fails that check.  The pre-refactor in-house autograd Function used
    to call ``grad_output.contiguous()`` explicitly before the C op.  We
    re-establish that contract here without re-implementing the wheel's
    Function: this thin Function sits *after* the wheel's Function in
    the autograd graph, so when grad flows back, ours runs first and
    hands a contiguous grad to the wheel's backward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return grad.contiguous()


@torch.fx.wrap
def cutlass_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    attn_func: Optional[torch.Tensor] = None,
    scaling_seqlen: int = -1,
    quant_mode: int = -1,
) -> torch.Tensor:
    """CUTLASS-based HSTU multi-head attention.

    Thin wrapper around ``hstu_attn_varlen_func`` from the ``hstu_attn``
    wheel. The wheel auto-dispatches to the SM 8.x or SM 9.x variant.

    Supports two mask modes:

    - Standard fixed-mask path (``attn_func=None``): causal/local/
      context/target masking driven by ``causal``, ``max_attn_len``,
      ``contextual_seq_len`` and ``num_targets``.
    - Arbitrary-mask NFUNC path (``attn_func`` provided): the kernel
      interprets ``attn_func`` as a jagged ``(nheads, 3, total_q)`` int32
      tensor encoding two disjoint column-intervals per query row. The
      caller is responsible for constructing it (see
      ``build_sla_func_tensor`` for the SLA case). In this mode the
      kernel forces ``window_size = (-1, 0)`` so ``causal`` and
      ``max_attn_len`` are redundant and must be left at their defaults
      (``causal=True, max_attn_len=0``).

    The kernel uses int32 cu_seqlens internally, so the cumulative sum
    ``seq_offsets[-1]`` (total token count in the batch) must fit in
    int32 (< 2**31 ≈ 2.1B tokens).

    Args:
        max_seq_len: maximum sequence length in the batch.
        alpha: scaling factor for attention scores.
        q: query tensor of shape (total, nheads, attn_dim).
        k: key tensor of shape (total, nheads, attn_dim).
        v: value tensor of shape (total, nheads, hidden_dim).
        seq_offsets: cumulative sequence offsets of shape (batch_size + 1,).
        causal: whether to apply causal masking (fixed-mask path only).
        num_targets: number of target tokens per batch element.
        max_attn_len: maximum attention window length (fixed-mask path;
            0 means unlimited).
        contextual_seq_len: number of contextual tokens per sequence.
        attn_func: pre-built arbitrary-mask func tensor of shape
            ``(nheads, 3, total_q)``, int32. When provided, selects the
            NFUNC mask path; ``causal`` and ``max_attn_len`` must be at
            defaults.
        scaling_seqlen: divisor used to scale the attention output inside
            the kernel. ``-1`` (default) falls back to ``max_seq_len`` so
            the behavior matches the legacy code path.
        quant_mode: FP8 quantization mode forwarded to the wheel's Hopper
            ``hstu_attn_varlen_func``. ``-1`` (default) keeps the BF16/FP16
            path; ``0..5`` enable FP8 with the granularity defined in the
            ``STU.fp8_quant_mode`` proto field. ``quant_mode != -1``
            requires SM>=90 and ``q.shape[2] in (64, 128, 256)``.

    Returns:
        output tensor of shape (total, nheads, hidden_dim).
    """
    if quant_mode != -1:
        if quant_mode < 0 or quant_mode > 5:
            raise ValueError(
                f"fp8 quant_mode must be -1 or in [0, 5]; got {quant_mode}."
            )
        if quant_mode == 0:
            _warn_fp8_mode_zero_unsafe()
        _assert_fp8_capable()
        _assert_fp8_headdim(q.shape[2])
    if q.shape[2] != v.shape[2]:
        raise ValueError(
            f"CUTLASS hstu_attn requires attention_dim == hidden_dim, "
            f"got q.shape[2]={q.shape[2]} != v.shape[2]={v.shape[2]}"
        )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"CUTLASS hstu_attn supports fp16 and bf16, got {q.dtype}. "
            f"Set train_config.mixed_precision to 'BF16' or 'FP16'."
        )
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative; got {contextual_seq_len}"
        )
    if attn_func is not None:
        if not causal:
            raise ValueError(
                "attn_func requires causal=True; the NFUNC mask path "
                "forces window_size_right=0 so causal=False has no effect."
            )
        if max_attn_len > 0:
            raise ValueError(
                f"attn_func is mutually exclusive with max_attn_len "
                f"(got max_attn_len={max_attn_len}); any local window "
                "must be encoded by the func tensor itself."
            )
        # The CUDA kernel addresses ``attn_func`` via pointer arithmetic
        # (``func_ptr + cu_seqlens[b]``) assuming a contiguous int32
        # ``(nheads, 3, total_q)`` jagged layout on the same device as q.
        # A caller that bypasses ``build_sla_func_tensor`` (mismatched
        # total_q, int64 dtype, a permuted view, or a CPU tensor) would
        # otherwise produce OOB reads or silent NaNs with no upstream
        # signal -- catch it here.
        torch._assert(
            attn_func.dtype == torch.int32,
            "attn_func must be int32",
        )
        torch._assert(attn_func.dim() == 3, "attn_func must be 3-D")
        torch._assert(
            attn_func.shape[0] == q.shape[1]
            and attn_func.shape[1] == 3
            and attn_func.shape[2] == q.shape[0],
            "attn_func must have shape (nheads, 3, total_q)",
        )
        torch._assert(
            attn_func.device == q.device,
            "attn_func must be on the same device as q",
        )
        attn_func = attn_func.contiguous()

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens = seq_offsets.to(torch.int32)

    num_targets_int32: Optional[torch.Tensor] = None
    if num_targets is not None:
        num_targets_int32 = num_targets.to(torch.int32)

    num_contexts_tensor: Optional[torch.Tensor] = None
    if contextual_seq_len > 0:
        batch_size = seq_offsets.size(0) - 1
        num_contexts_tensor = torch.full(
            (batch_size,),
            contextual_seq_len,
            dtype=torch.int32,
            device=q.device,
        )

    if attn_func is not None:
        # NFUNC mask path: the func tensor fully describes the mask.
        window_size = (-1, 0)
    else:
        if causal:
            window_size = (max_attn_len, 0) if max_attn_len > 0 else (-1, 0)
        else:
            window_size = (-1, -1)

    from hstu_attn import hstu_attn_varlen_func  # lazy

    call_kwargs = dict(
        num_contexts=num_contexts_tensor,
        num_targets=num_targets_int32,
        target_group_size=1,
        window_size=window_size,
        alpha=alpha,
        rab=None,
        has_drab=False,
        func=attn_func,
        scaling_seqlen=scaling_seqlen,
    )
    if quant_mode != -1:
        # The Hopper variant of hstu_attn_varlen_func accepts quant_mode;
        # the Ampere/Ada variant does not. We've already gated to SM>=90
        # above, so adding the kwarg only on the FP8 path is safe.
        call_kwargs["quant_mode"] = quant_mode

    out = hstu_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seq_len,
        max_seq_len,
        **call_kwargs,
    )
    # Wheel's backward requires contiguous dout; intercept grad here.
    return _ContiguousGradPassthrough.apply(out)
