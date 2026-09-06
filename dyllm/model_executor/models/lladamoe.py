"""LLaDA-MoE for DyLLM.

``inclusionAI/LLaDA-MoE-7B-A1B-Instruct`` is a decoder-only, *bidirectional*
masked-diffusion Mixture-of-Experts LM (OLMoE-style).  Unlike DiffusionGemma it fits
DyLLM's engine directly: same masked-diffusion, saliency / partial-attention path as
the dense :mod:`llada` model -- the only differences are

  * ``qk_layernorm``: per-head RMSNorm on Q and K (over ``head_dim``) before RoPE,
  * the dense MLP is replaced by a 64-expert top-8 sparse MoE, computed with the
    fused Triton kernel :func:`fused_experts` + :func:`moe_gate` router
    (softmax over all experts, top-k, **no** renormalization -- ``norm_topk_prob``
    is false for this checkpoint), silu-gated experts.

Everything else (attention with the sparse KV cache, the MLP output cache, the
sandwich RMSNorm residual pattern) mirrors :mod:`dyllm.model_executor.models.llada`.
"""

import torch
from torch import nn
import torch.distributed as dist
from collections.abc import Iterable

from dyllm.utils.context import get_context
from dyllm.utils.weight_loader import AutoWeightsLoader
from dyllm.model_executor.layers.attention import Attention
from dyllm.model_executor.layers.layernorm import RMSNorm
from dyllm.model_executor.layers.linear import (
    KVParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
    ReplicatedLinear,
)
from dyllm.model_executor.layers.rotary_embedding import get_rope
from dyllm.model_executor.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from dyllm.model_executor.layers.mlp_cache_manage import MLPcache
from dyllm.model_executor.layers.fused_moe import fused_experts, moe_gate
from dyllm.engine.cache_manager import CacheManager
from dyllm.utils.metadata import get_metadata
from dyllm.utils.util import gather_rows_2D
from dyllm.distributed import (
    get_expert_parallel_group,
    get_expert_parallel_rank,
    get_expert_parallel_world_size,
)


class LLaDAMoESparseBlock(nn.Module):
    """Sparse MoE feed-forward: softmax top-k router + fused silu experts.

    Expert weights are stored fused so a single grouped-GEMM kernel serves the
    rank-local expert shards. The per-expert checkpoint tensors are partitioned
    and stacked into these parameters at load time (see
    ``LLaDAMoEForDLM.load_weights``).
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.renormalize = bool(getattr(config, "norm_topk_prob", False))
        H = config.hidden_size
        I = config.expert_intermediate_size
        self.hidden_size = H
        self.intermediate_size = I
        self.ep_size = get_expert_parallel_world_size()
        self.ep_rank = get_expert_parallel_rank()
        self.expert_parallel = self.ep_size > 1

        if self.expert_parallel:
            if self.num_experts % self.ep_size != 0:
                raise ValueError(f"num_experts ({self.num_experts}) must be divisible by EP size ({self.ep_size})")
            self.num_local_experts = self.num_experts // self.ep_size
            self.expert_start = self.ep_rank * self.num_local_experts
            self.local_intermediate_size = I
        else:
            tp_size = dist.get_world_size()
            if I % tp_size != 0:
                raise ValueError(f"expert_intermediate_size ({I}) must be divisible by TP size ({tp_size})")
            # vLLM's non-EP mode tensor-parallelizes every expert instead of
            # replicating its complete matrices on every rank.
            self.num_local_experts = self.num_experts
            self.expert_start = 0
            self.local_intermediate_size = I // tp_size

        self.gate = ReplicatedLinear(H, self.num_experts, bias=False)
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_local_experts, 2 * self.local_intermediate_size, H))
        self.down_proj = nn.Parameter(torch.empty(self.num_local_experts, H, self.local_intermediate_size))
        self.cache_update = MLPcache(H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_w, topk_i = moe_gate(x, self.gate.weight, self.top_k, renormalize=self.renormalize)
        if self.expert_parallel:
            output = self._expert_parallel_forward(x, topk_w, topk_i)
        else:
            output = fused_experts(
                x,
                self.gate_up_proj,
                self.down_proj,
                topk_w,
                topk_i,
                activation="silu",
            )
            if dist.get_world_size() > 1:
                dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return self.cache_update(output)

    def _expert_parallel_forward(
        self,
        x: torch.Tensor,
        topk_w: torch.Tensor,
        topk_i: torch.Tensor,
    ) -> torch.Tensor:
        """Run only this rank's experts and reduce their token contributions.

        Attention row-parallelism leaves ``x`` replicated on every TP rank, so
        there is no need to all-gather or transmit input vectors first.  Each
        EP rank selects the routed assignments owned by its contiguous expert
        range, computes them once, and sums the partial token outputs.  This is
        the all-reduce baseline corresponding to vLLM's linear expert placement;
        an all-to-all backend can replace it later when sequence-parallel input
        shards are introduced.
        """

        expert_end = self.expert_start + self.num_local_experts
        owned = (topk_i >= self.expert_start) & (topk_i < expert_end)
        flat_owned = owned.reshape(-1)
        output = torch.zeros_like(x)
        token_ids = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)[flat_owned]
        local_expert_ids = (topk_i.reshape(-1)[flat_owned] - self.expert_start).unsqueeze(1)
        local_weights = topk_w.reshape(-1)[flat_owned].unsqueeze(1)
        contributions = fused_experts(
            x.index_select(0, token_ids),
            self.gate_up_proj,
            self.down_proj,
            local_weights,
            local_expert_ids,
            activation="silu",
        )
        output.index_add_(0, token_ids, contributions)

        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=get_expert_parallel_group())
        return output


class LLaDAMoEAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int,
        rms_norm_eps: float = 1e-05,
        qkv_bias: bool = False,
        rope_theta: float = 50000.0,
        rope_scaling: tuple | None = None,
        qk_layernorm: bool = True,
        threshold: float = 0.99,
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(hidden_size, self.total_num_heads * self.head_dim, bias=qkv_bias)
        self.kv_proj = KVParallelLinear(hidden_size, self.head_dim, self.total_num_kv_heads, bias=qkv_bias)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps) if qk_layernorm else None
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps) if qk_layernorm else None

        self.k_cache = CacheManager(self.num_kv_heads * self.head_dim)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads, threshold)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        metadata = get_metadata()

        q = self.q_proj(hidden_states)
        if self.q_norm is not None:
            q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)

        if ctx.is_full:
            kv = self.kv_proj(hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            if self.k_norm is not None:
                k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
        else:
            kv = self.kv_proj(hidden_states[ctx.idx_salient_row])
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            if self.k_norm is not None:
                k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
            if ctx.idx_salient_row_k is not None:
                k_temp = torch.zeros(ctx.total_seqlen, self.kv_size, dtype=k.dtype, device=k.device)
            else:
                k_temp = torch.zeros(ctx.total_seqlen_k, self.kv_size, dtype=k.dtype, device=k.device)
            k_temp[ctx.idx_salient_row] = k
            k = k_temp

        def split_last(x, H, D):
            *prefix, _ = x.shape
            return x.view(*prefix, H, D)

        q = split_last(q, self.num_heads, self.head_dim)
        k = split_last(k, self.num_kv_heads, self.head_dim)
        v = split_last(v, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        if ctx.is_full:
            self.k_cache.reset_full(k.flatten(-2, -1), metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs)
            o = self.attn(q, k, v)
        else:
            if ctx.idx_salient_row_k is not None:
                self.k_cache.scatter_update(
                    metadata.running_seqs_tensor, ctx.idx_salient_row_k, k[ctx.idx_salient_row].flatten(-2, -1)
                )
            else:
                self.k_cache.scatter_update(
                    metadata.running_seqs_tensor, ctx.idx_salient_row, k[ctx.idx_salient_row].flatten(-2, -1)
                )
            o = self.attn(
                q, self.k_cache.get_seqs(metadata.running_seqs_tensor).view(-1, self.num_kv_heads, self.head_dim), v
            )
        output = self.o_proj(o.flatten(-2, -1))
        self.k_cache.finish(metadata.finished_seqs)
        return output


class LLaDAMoEDecoderLayer(nn.Module):
    def __init__(self, config, threshold: float) -> None:
        super().__init__()
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.self_attn = LLaDAMoEAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=getattr(config, "rope_theta", 50000),
            rope_scaling=getattr(config, "rope_scaling", None),
            qk_layernorm=bool(getattr(config, "qk_layernorm", False)),
            threshold=threshold,
        )
        self.mlp = LLaDAMoESparseBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        ctx = get_context()
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if not ctx.is_full:
            hidden_states = gather_rows_2D(hidden_states, ctx.idx_salient_row)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LLaDAMoEModel(nn.Module):
    def __init__(self, config, threshold: float):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LLaDAMoEDecoderLayer(config, threshold) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LLaDAMoEForDLM(nn.Module):
    packed_modules_mapping = {
        "kv_proj": ["k_proj", "v_proj"],
    }

    def __init__(self, config, threshold: float):
        super().__init__()
        self.config = config
        self.model = LLaDAMoEModel(config, threshold)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def normalize_weight_name(self, name: str) -> str:
        # Checkpoint already uses HF-standard names matching the module tree.
        return name

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        block = self.model.layers[0].mlp
        local_intermediate = block.local_intermediate_size
        non_expert: list[tuple[str, torch.Tensor]] = []

        # Stack per-expert checkpoint tensors into the fused MoE parameters.
        for name, w in weights:
            if ".mlp.experts." in name:
                head, tail = name.split(".mlp.experts.")
                _eid, proj = tail.split(".", 1)
                eid = int(_eid)
                if block.expert_parallel:
                    if not block.expert_start <= eid < block.expert_start + block.num_local_experts:
                        continue
                    local_eid = eid - block.expert_start
                else:
                    local_eid = eid

                if proj == "gate_proj.weight":
                    if not block.expert_parallel:
                        w = w.chunk(dist.get_world_size(), dim=0)[dist.get_rank()]
                    params[f"{head}.mlp.gate_up_proj"].data[local_eid, :local_intermediate].copy_(w)
                elif proj == "up_proj.weight":
                    if not block.expert_parallel:
                        w = w.chunk(dist.get_world_size(), dim=0)[dist.get_rank()]
                    params[f"{head}.mlp.gate_up_proj"].data[local_eid, local_intermediate:].copy_(w)
                elif proj == "down_proj.weight":
                    if not block.expert_parallel:
                        w = w.chunk(dist.get_world_size(), dim=1)[dist.get_rank()]
                    params[f"{head}.mlp.down_proj"].data[local_eid].copy_(w)
                else:
                    raise KeyError(f"Unexpected expert weight: {name}")
            else:
                non_expert.append((name, w))

        # Everything else (embeddings, attention w/ packed kv, norms, router, lm_head).
        loader = AutoWeightsLoader(self, skip_prefixes=None)
        loader.load_weights(iter(non_expert))
        return set(params.keys())

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
