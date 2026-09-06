"""Small process-group registry for DyLLM model parallelism.

DyLLM currently has one model-parallel world and no data/pipeline parallel
dimension.  Following vLLM's layout, enabling EP reuses that world for MoE
layers while attention continues to use tensor parallelism.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


_EP_GROUP: dist.ProcessGroup | None = None
_EP_SIZE = 1
_EP_RANK = 0


def initialize_parallel_state(expert_parallel_size: int) -> None:
    global _EP_GROUP, _EP_SIZE, _EP_RANK

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    world_size = dist.get_world_size()
    if expert_parallel_size not in (1, world_size):
        raise ValueError(
            "expert_parallel_size must be 1 (expert tensor parallelism) or "
            f"tensor_parallel_size ({world_size}); got {expert_parallel_size}"
        )

    _EP_SIZE = expert_parallel_size
    if expert_parallel_size == 1:
        _EP_GROUP = None
        _EP_RANK = 0
    else:
        _EP_GROUP = dist.group.WORLD
        _EP_RANK = dist.get_rank()


def get_expert_parallel_group() -> dist.ProcessGroup | None:
    return _EP_GROUP


def get_expert_parallel_world_size() -> int:
    return _EP_SIZE


def get_expert_parallel_rank() -> int:
    return _EP_RANK


def tensor_parallel_cosine_mask(stats: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return the global TP cosine decision from additive local statistics.

    ``stats[..., 0:3]`` are respectively the local dot product and the two
    local squared L2 norms. Only these three FP32 values per row are reduced;
    no hidden/context vector is gathered.
    """

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    dot = stats[..., 0]
    norm_old = stats[..., 1].clamp_min(1e-8)
    norm_new = stats[..., 2].clamp_min(1e-8)
    cosine = dot * torch.rsqrt(norm_old) * torch.rsqrt(norm_new)
    return cosine < threshold
