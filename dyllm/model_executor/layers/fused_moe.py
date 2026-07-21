"""Fused Mixture-of-Experts (MoE) layer.

This module implements a vLLM-style *fused* MoE using Triton kernels.  The core
optimization is exactly the one used by FlashAttention / vLLM: instead of looping
over the experts (which launches many tiny, memory-bound GEMMs and wastes the
GPU), we

  1. route every token to ``top_k`` experts,
  2. *sort* the (token, expert) pairs by expert id and pad each expert's bucket
     to a multiple of ``BLOCK_M`` (``moe_align_block_size``), and
  3. run a single **grouped GEMM** where every ``BLOCK_M x BLOCK_N`` output tile
     picks up the weight matrix of the expert that owns its token-block.

The activation (``gelu_and_mul`` for DiffusionGemma, ``silu_and_mul`` for
Llama/Mixtral) is fused into its own kernel, and the routing weight is folded
into the second (down-projection) GEMM.  This does **not** change the numerical
result relative to a naive per-expert implementation -- it is purely a fusion /
scheduling optimization -- which is verified in ``tests/test_fused_moe.py``.

Expert weight layout (matches the DiffusionGemma checkpoint, ``nn.Linear`` /
``[out, in]`` convention):
    w1 (gate_up_proj): ``[E, 2 * I, H]``   gate = w1[:, :I], up = w1[:, I:]
    w2 (down_proj):    ``[E, H, I]``
where ``E`` = num_experts, ``H`` = hidden_size, ``I`` = moe_intermediate_size.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Token -> expert alignment (mirrors vLLM's moe_align_block_size)
# ---------------------------------------------------------------------------
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort ``(token, expert)`` pairs by expert and pad each expert bucket to a
    multiple of ``block_size``.

    Returns
    -------
    sorted_token_ids : int32 ``[num_padded]``
        Flattened ``token * top_k + k`` indices, grouped by expert.  Padding
        slots are filled with ``num_tokens * top_k`` (an out-of-range sentinel
        that the GEMM kernel masks out).
    expert_ids : int32 ``[num_padded // block_size]``
        Expert id owning each ``block_size`` block of ``sorted_token_ids``.
    num_tokens_post_pad : int32 ``[1]``
        Total number of (padded) token slots.
    """
    device = topk_ids.device
    num_tokens, top_k = topk_ids.shape
    total_tokens = num_tokens * top_k
    flat = topk_ids.reshape(-1)  # [T]

    # Stable sort of the (token, expert) pairs by expert id.
    order = torch.argsort(flat, stable=True)  # [T] -> original flat position
    sorted_experts = flat[order]

    counts = torch.bincount(flat, minlength=num_experts)  # [E]
    padded = ((counts + block_size - 1) // block_size) * block_size  # [E]
    pad_starts = torch.cumsum(padded, 0) - padded  # [E] start in padded layout
    comp_starts = torch.cumsum(counts, 0) - counts  # [E] start in compact layout

    total_padded = int(padded.sum().item())
    sorted_token_ids = torch.full(
        (total_padded,), total_tokens, dtype=torch.int32, device=device
    )

    # Position of each compact-sorted element inside the padded layout.
    arange_t = torch.arange(total_tokens, device=device)
    within = arange_t - comp_starts[sorted_experts]  # rank within its expert
    dest = pad_starts[sorted_experts] + within
    sorted_token_ids[dest] = order.to(torch.int32)

    num_blocks = total_padded // block_size
    block_starts = torch.arange(num_blocks, device=device) * block_size
    pad_ends = torch.cumsum(padded, 0)  # [E]
    expert_ids = torch.searchsorted(pad_ends, block_starts, right=True).to(torch.int32)

    num_tokens_post_pad = torch.tensor([total_padded], dtype=torch.int32, device=device)
    return sorted_token_ids, expert_ids, num_tokens_post_pad


# ---------------------------------------------------------------------------
# Grouped GEMM kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_moe_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C[sorted_token] = A[sorted_token // top_k] @ B[expert]^T (optionally * weight).

    ``B`` has ``[E, N, K]`` ("out, in") layout, so this computes a standard
    ``x @ W.T`` per expert.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        if IEEE:
            accumulator += tl.dot(a, b, input_precision="ieee")
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    block_m: int,
    ieee: bool = False,
    block_n: int = 64,
    block_k: int = 32,
    group_m: int = 8,
) -> None:
    N = b.shape[1]
    K = a.shape[1]
    EM = sorted_token_ids.shape[0]
    num_valid_tokens = topk_weights.numel() if topk_weights is not None else a.shape[0] * top_k

    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)
    _fused_moe_gemm_kernel[grid](
        a,
        b,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        top_k=top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        IEEE=ieee,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
    )


# ---------------------------------------------------------------------------
# Fused gated activation kernels
# ---------------------------------------------------------------------------
@triton.jit
def _gated_act_kernel(
    out_ptr,
    x_ptr,
    d,
    stride_xm,
    stride_om,
    ACT: tl.constexpr,  # 0 = gelu_tanh, 1 = silu
    BLOCK: tl.constexpr,
):
    """out[m, :] = act(x[m, :d]) * x[m, d:2d]   (gate first, up second)."""
    pid_m = tl.program_id(0)
    for start in range(0, d, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < d
        gate = tl.load(x_ptr + pid_m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + pid_m * stride_xm + d + offs, mask=mask, other=0.0).to(tl.float32)
        if ACT == 0:
            # gelu, tanh approximation (== torch gelu_pytorch_tanh)
            inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
            # tanh(z) = 1 - 2 / (exp(2z) + 1), version-robust across triton releases
            tanh = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
            act = 0.5 * gate * (1.0 + tanh)
        else:
            act = gate / (1.0 + tl.exp(-gate))
        res = act * up
        tl.store(out_ptr + pid_m * stride_om + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def _gated_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    """x: [M, 2d] -> [M, d]."""
    M, two_d = x.shape
    d = two_d // 2
    out = torch.empty((M, d), dtype=x.dtype, device=x.device)
    act_code = 0 if activation == "gelu" else 1
    block = min(triton.next_power_of_2(d), 1024)
    _gated_act_kernel[(M,)](out, x, d, x.stride(0), out.stride(0), ACT=act_code, BLOCK=block)
    return out


# ---------------------------------------------------------------------------
# Public fused-experts entry point
# ---------------------------------------------------------------------------
def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "gelu",
    block_m: int = 64,
) -> torch.Tensor:
    """Fused MoE expert computation.

    Parameters
    ----------
    hidden_states : ``[num_tokens, H]``
    w1 : ``[E, 2 * I, H]``  gate_up projection ([out, in]).
    w2 : ``[E, H, I]``      down projection ([out, in]).
    topk_weights, topk_ids : ``[num_tokens, top_k]`` routing weights & expert ids.
    activation : ``"gelu"`` (DiffusionGemma) or ``"silu"``.
    """
    assert hidden_states.is_cuda and w1.is_cuda and w2.is_cuda
    num_tokens, H = hidden_states.shape
    E, two_i, _ = w1.shape
    I = two_i // 2
    top_k = topk_ids.shape[1]
    T = num_tokens * top_k
    # fp32 inputs use IEEE matmul (no TF32) so results match a reference exactly.
    ieee = hidden_states.dtype == torch.float32

    sorted_token_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(topk_ids, block_m, E)

    # GEMM 1: gate_up projection -> [T, 2I]
    inter1 = torch.empty((T, two_i), dtype=hidden_states.dtype, device=hidden_states.device)
    _invoke_gemm(
        hidden_states, w1, inter1,
        topk_weights=None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_pad,
        mul_routed_weight=False,
        top_k=top_k,
        block_m=block_m,
        ieee=ieee,
    )

    # Fused gated activation -> [T, I]
    inter2 = _gated_activation(inter1, activation)

    # GEMM 2: down projection (fold in routing weight) -> [T, H]
    out = torch.empty((T, H), dtype=hidden_states.dtype, device=hidden_states.device)
    _invoke_gemm(
        inter2, w2, out,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_pad,
        mul_routed_weight=True,
        top_k=1,  # inter2 is already in flattened (token*top_k) space
        block_m=block_m,
        ieee=ieee,
    )

    return out.view(num_tokens, top_k, H).sum(dim=1)


# ---------------------------------------------------------------------------
# Naive reference (for correctness verification)
# ---------------------------------------------------------------------------
def naive_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "gelu",
) -> torch.Tensor:
    """Reference MoE that loops over experts.  Same math as ``fused_experts``."""
    num_tokens, H = hidden_states.shape
    E = w1.shape[0]
    out = torch.zeros_like(hidden_states)
    if activation == "gelu":
        act_fn = lambda t: torch.nn.functional.gelu(t, approximate="tanh")
    else:
        act_fn = torch.nn.functional.silu
    for e in range(E):
        mask = topk_ids == e  # [num_tokens, top_k]
        tok_idx, slot = mask.nonzero(as_tuple=True)
        if tok_idx.numel() == 0:
            continue
        x = hidden_states[tok_idx]  # [n, H]
        gate_up = x @ w1[e].t()  # [n, 2I]
        g, u = gate_up.chunk(2, dim=-1)
        h = act_fn(g) * u  # [n, I]
        y = h @ w2[e].t()  # [n, H]
        y = y * topk_weights[tok_idx, slot].unsqueeze(-1)
        out.index_add_(0, tok_idx, y.to(out.dtype))
    return out


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def moe_gate(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
    router_scale: torch.Tensor | None = None,
    per_expert_scale: torch.Tensor | None = None,
    renormalize: bool = True,
    rms_norm_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k routing weights / expert ids.

    Faithful to DiffusionGemma's ``Gemma4TextRouter``:
      1. optional RMSNorm (no learnable scale) of the router input,
      2. multiply by ``router_scale`` (per hidden dim) and ``hidden**-0.5``,
      3. linear projection -> softmax (fp32) -> top-k -> renormalize,
      4. multiply the *gathered* top-k weights by ``per_expert_scale[idx]``.

    ``router_scale`` / ``per_expert_scale`` / ``rms_norm_eps`` are optional so the
    function also serves as a plain softmax-top-k router when they are omitted.
    """
    x = hidden_states.float()
    if rms_norm_eps is not None:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_norm_eps)
    if router_scale is not None:
        x = x * router_scale.float() * (x.shape[-1] ** -0.5)
    logits = torch.nn.functional.linear(x, router_weight.float())  # [T, E]
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if per_expert_scale is not None:
        topk_weights = topk_weights * per_expert_scale.float()[topk_ids]
    return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)
