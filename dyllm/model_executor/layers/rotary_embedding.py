from functools import lru_cache
import torch
from torch import nn

from dyllm.utils.context import get_context


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Clamp positions to valid cache range to prevent out-of-bounds access
        # This is needed when sequence length exceeds max_position_embeddings
        max_pos = self.cos_sin_cache.size(0)
        positions = torch.clamp(positions, 0, max_pos - 1)

        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


class DiffusionGemmaRotary(nn.Module):
    """Rotary embedding with cos/sin computed per call (a 262K position table
    would cost ~0.5 GB per geometry). ``partial_rotary_factor`` = proportional rope."""

    def __init__(self, head_dim: int, base: float, partial_rotary_factor: float = 1.0):
        super().__init__()
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(dtype=torch.float32) / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat((inv_freq, torch.zeros(nope_angles, dtype=torch.float32)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # per-step cache shared across layers
        cache = get_context().step_cache
        key_ = ("rope", self.inv_freq.shape[0])
        if cache is not None and key_ in cache:
            cos, sin = cache[key_]
        else:
            freqs = torch.outer(positions.float(), self.inv_freq)
            cos = freqs.cos().unsqueeze(1)
            sin = freqs.sin().unsqueeze(1)
            if cache is not None:
                cache[key_] = (cos, sin)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)
