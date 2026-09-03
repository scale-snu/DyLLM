"""DiffusionGemma for DyLLM -- ``google/diffusiongemma-26B-A4B-it``.

One 30-layer MoE stack shared between causal (prefill/commit) and bidirectional
(canvas denoise) attention, 5:1 sliding:global layers, self-conditioning between steps."""

from collections.abc import Iterable

import torch
from torch import nn
import torch.distributed as dist

from dyllm.model_executor.layers.attention_triton import attend_triton, sparse_prep

try:
    from dyllm.attention_ops import attention_sparse_varlen
except ImportError:
    attention_sparse_varlen = None

from dyllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from dyllm.model_executor.layers.layernorm import DiffusionGemmaRMSNorm
from dyllm.model_executor.layers.rotary_embedding import DiffusionGemmaRotary
from dyllm.model_executor.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from dyllm.model_executor.layers.fused_moe import fused_experts, moe_gate
from dyllm.utils.context import get_context
from dyllm.utils.metadata import get_metadata
from dyllm.configs import DiffusionGemmaConfig


class DiffusionGemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class DiffusionGemmaSelfConditioning(nn.Module):
    """Gated MLP over the previous denoise step's soft embeddings, added to the
    canvas embeddings and followed by an RMSNorm with no learnable scale."""

    def __init__(self, config):
        super().__init__()
        self.pre_norm = DiffusionGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm = DiffusionGemmaRMSNorm(eps=config.rms_norm_eps, with_scale=False)
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, inputs_embeds: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        normed = self.pre_norm(z)
        sc = self.down_proj(nn.functional.gelu(self.gate_proj(normed), approximate="tanh") * self.up_proj(normed))
        return self.post_norm(inputs_embeds + sc)


class DiffusionGemmaAttention(nn.Module):
    # packed dV for the sparse kernel, shared by all layers of a type
    _vdelta_shared: dict = {}
    """Dual-mode attention selected via ``ctx.attn_mode``: causal (prefill/commit)
    vs bidirectional (denoise). Global layers have no ``v_proj`` -- V reuses ``k_proj``
    through ``v_norm``; softmax scale is 1.0 since the q/k RMSNorms take that role."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
        rope_theta: float,
        partial_rotary_factor: float = 1.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        self.is_sliding = sliding_window is not None
        self.head_dim = head_dim
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = max(1, num_kv_heads // tp_size)
        self.sliding_window = sliding_window
        self.scaling = 1.0
        # flat KV cache, per-seq region [start, start+cap) = [prefix rows | canvas slot]
        self.key_cache: torch.Tensor | None = None
        self.value_cache: torch.Tensor | None = None
        self.cache_slots: dict[int, list[int]] = {}  # sid -> [start, cap, prefix_len]
        self.cache_free_rows = 0
        self.page: int | None = None  # sliding slot stride (row-page size)
        self.capture_context = False
        self._context: torch.Tensor | None = None
        self.diffusiongemma_sparse = False
        self.diffusiongemma_sparse_threshold: float | None = None
        self._sparse_cache: dict | None = None
        self._prep_cache: dict | None = None  # mixed-path metadata

        self.q_proj = ColumnParallelLinear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = (
            ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=False)
            if self.is_sliding
            else None
        )
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = DiffusionGemmaRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(head_dim, eps=rms_norm_eps)
        self.v_norm = DiffusionGemmaRMSNorm(eps=rms_norm_eps, with_scale=False)

        self.rotary = DiffusionGemmaRotary(head_dim, rope_theta, partial_rotary_factor)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        mode = ctx.attn_mode

        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k_raw = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v_in = (
            self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
            if self.v_proj is not None
            else k_raw
        )

        q = self.q_norm(q)
        k = self.k_norm(k_raw)
        q, k = self.rotary(positions, q, k)
        v = self.v_norm(v_in)

        cu = ctx.cu_seqlens_q_cpu or [0, q.size(0)]
        outs = self._attend_with_cache(q, k, v, mode, cu)
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        o = o.flatten(-2, -1)
        if self.capture_context:
            self._context = o
        out = self.o_proj(o)
        return out

    def _attend_with_cache(self, q, k, v, mode, cu):
        """causal: attend [cache | new] and append; denoise: write the canvas slot
        and attend on the cache. One batched call over the KV cache."""
        metadata = get_metadata()
        seq_ids = metadata.running_seqs
        ctx = get_context()
        assert q.is_cuda and q.dtype in (torch.float16, torch.bfloat16), "DG runs on cuda bf16/fp16"
        modes = ctx.modes  # per-seq causal flags (list[bool]); None = uniform `mode`
        if modes is None:
            modes = [mode == "causal"] * (len(cu) - 1)
        any_causal, any_denoise = any(modes), not all(modes)
        sparse = (
            any_denoise
            and ctx.ctx_cache is not None
            and ctx.salient_rows is not None
            and self.diffusiongemma_sparse
            and attention_sparse_varlen is not None
        )
        if any_causal:
            self._cache_admit(seq_ids, k)
        outs = [self._attend_mixed(q, k, v, cu, seq_ids, modes, sparse)]
        if any_causal:
            self._sparse_cache = None
        for sid in getattr(metadata, "finished_seqs", None) or []:
            freed = self.cache_slots.pop(sid, None)
            if freed is not None:
                self.cache_free_rows += freed[1]
                self._sparse_cache = None
        return outs

    def _attend_mixed(self, q, k, v, cu, seq_ids, modes, sparse):
        """One dense attention call over the KV cache with per-seq causal flags.
        causal seqs: new rows appended (+sliding trim); denoise seqs: canvas slot
        written. sparse: denoise segments then take the salient 3-stage path."""
        ctx = get_context()
        prep = self._mixed_prep(seq_ids, cu, modes)
        trims = []
        for i, sid in enumerate(seq_ids):
            start, cap, plen = self.cache_slots[sid]
            n = cu[i + 1] - cu[i]
            assert plen + n <= cap
            if modes[i]:
                keep = plen + n
                if self.sliding_window is not None:
                    keep = min(keep, self.sliding_window - 1)
                trims.append((sid, keep, plen + n))
        if sparse and not all(modes):
            out = self._attend_mixed_sparse(q, k, v, prep)
        else:
            # new rows sit right after the prefix
            self.key_cache.index_copy_(0, prep["kv_rows"], k)
            self.value_cache.index_copy_(0, prep["kv_rows"], v)
            out = self._attend_dense_paged(q, prep)
        for sid, keep, total in trims:
            entry = self.cache_slots[sid]
            start = entry[0]
            if keep < total:
                self.key_cache[start : start + keep] = self.key_cache[start + total - keep : start + total].clone()
                self.value_cache[start : start + keep] = self.value_cache[start + total - keep : start + total].clone()
            entry[2] = keep
            if self.sliding_window is not None and entry[1] > self.page:
                self.cache_free_rows += entry[1] - self.page
                entry[1] = self.page
        return out

    def _mixed_prep(self, seq_ids, cu, modes):
        """Per-batch metadata for the paged kernel and the sparse packing; rebuilt
        only when seqs / segment lengths / cache regions / modes change. Built on
        the CPU and moved with pinned non-blocking copies (no host sync)."""
        key = (
            tuple(seq_ids), tuple(cu), tuple(modes),
            tuple(self.cache_slots[s][0] for s in seq_ids),
            tuple(self.cache_slots[s][2] for s in seq_ids),
        )
        prep = self._prep_cache
        if prep is not None and prep["key"] == key:
            return prep
        dev, page = self.key_cache.device, self.page
        B = len(seq_ids)
        starts = [self.cache_slots[s][0] for s in seq_ids]
        plens = [self.cache_slots[s][2] for s in seq_ids]
        kv_lens = [plens[i] + cu[i + 1] - cu[i] for i in range(B)]
        n_pages = [-(-L // page) for L in kv_lens]
        pk = [0]
        for L in kv_lens:
            pk.append(pk[-1] + L)
        with torch.device("cpu"):
            bt = torch.zeros(B, max(n_pages), dtype=torch.int32)
            for i in range(B):
                first = starts[i] // page
                bt[i, : n_pages[i]] = torch.arange(first, first + n_pages[i], dtype=torch.int32)
            gather = torch.cat([torch.arange(starts[i], starts[i] + kv_lens[i]) for i in range(B)])
            enc = None
            if any(modes):
                enc = torch.zeros(cu[-1], dtype=torch.bool)
                for i, m in enumerate(modes):
                    if m:
                        enc[cu[i] : cu[i + 1]] = True
            seg = torch.tensor([cu[i + 1] - cu[i] for i in range(B)])
            row_seq = torch.repeat_interleave(torch.arange(B), seg)
            local = torch.arange(cu[-1]) - torch.tensor(cu[:-1])[row_seq]
            kv_rows = torch.tensor(starts)[row_seq] + torch.tensor(plens)[row_seq] + local
            packed_rows = torch.tensor(pk[:-1])[row_seq] + torch.tensor(plens)[row_seq] + local
            den_rows = torch.tensor([not m for m in modes], dtype=torch.int8)[row_seq]
            seg_end = torch.tensor(cu[1:]) - 1
            seqused = torch.tensor(kv_lens, dtype=torch.int32)
            causal = torch.tensor(modes, dtype=torch.bool)
            cu_k = torch.tensor(pk, dtype=torch.int32)

        def h2d(t):
            return t.pin_memory().to(dev, non_blocking=True)

        prep = {
            "key": key, "kv_lens": kv_lens, "max_kv": max(kv_lens), "packed_total": pk[-1],
            "row_seq": h2d(row_seq),
            "seg_len": h2d(seg.float()),
            "seg_end": h2d(seg_end),
            # per flat row: cache row, packed row (kernel layout), denoise flag
            "kv_rows": h2d(kv_rows), "packed_rows": h2d(packed_rows), "den_rows": h2d(den_rows),
            "vd_key": (self.is_sliding, self.num_kv_heads, self.head_dim, tuple(kv_lens), tuple(cu)),
            "bt": h2d(bt), "seqused": h2d(seqused),
            "causal": h2d(causal),
            "gather_idx": h2d(gather),
            "cu_k": h2d(cu_k), "enc_rows": h2d(enc) if enc is not None else None,
        }
        self._prep_cache = prep
        return prep

    def _cache_pages(self):
        shape = (self.key_cache.size(0) // self.page, self.page, self.num_kv_heads, self.head_dim)
        return self.key_cache.view(shape), self.value_cache.view(shape)

    def _attend_dense_paged(self, q, prep):
        ctx = get_context()
        kp, vp = self._cache_pages()
        out = torch.empty_like(q)
        window = (self.sliding_window - 1, 0) if self.sliding_window is not None else (-1, -1)
        attend_triton(
            q, kp, vp, out, ctx.cu_seqlens_q, ctx.max_seqlen_q, prep["seqused"], prep["max_kv"],
            prep["bt"], prep["causal"], self.scaling, window,
        )
        return out

    def _attend_mixed_sparse(self, q, k, v, prep):
        """Salient 3-stage path over a mixed batch: encode rows count as all-salient,
        o_sal = one per-seq-causal call over the salient Q rows, csrc kernel over the
        whole batch."""
        ctx = get_context()
        n = q.size(0)
        dev = q.device
        sal = ctx.salient_rows
        if prep["enc_rows"] is not None:
            sal = sal | prep["enc_rows"]  # encode rows: always exact
        # salient rows compacted in order: row i -> slot pos[i]; per-seq counts at segment ends
        sal_i = sal.to(torch.int32)
        incl = torch.cumsum(sal_i, 0, dtype=torch.int32)
        cu_sal = nn.functional.pad(incl.index_select(0, prep["seg_end"]), (1, 0))
        pin, ev = self._cnt_slot(dev)
        pin.copy_(cu_sal[-1:], non_blocking=True)
        ev.record()
        q_sal, idx_q, idx_k, stats = self._scratch(n, dev, q.dtype)
        vd = self._vdelta_buffer(prep, q)
        # one launch: salient K/V -> cache, packed dV, compacted q / indices
        sparse_prep(sal_i, prep["den_rows"], prep["kv_rows"], prep["packed_rows"], incl,
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    self.key_cache, self.value_cache, vd, q_sal, idx_q, idx_k)
        kp, vp = self._cache_pages()
        window = (self.sliding_window - 1, 0) if self.sliding_window is not None else (-1, -1)
        o_sal = torch.empty_like(q_sal)
        attend_triton(
            q_sal, kp, vp, o_sal, cu_sal, ctx.max_seqlen_q, prep["seqused"], prep["max_kv"],
            prep["bt"], prep["causal"], self.scaling, window,
        )
        kgp = self.key_cache.index_select(0, prep["gather_idx"])
        c_full = ctx.ctx_cache.view(n, self.num_heads, self.head_dim)
        # host sync: the kernel takes exact-size index tensors
        ev.synchronize()
        cnt = int(pin[0])
        if cnt == n:
            out = o_sal
            cos = nn.functional.cosine_similarity(out.view(n, -1).float(), c_full.view(n, -1).float(), dim=-1)
            verdict = cos < self.diffusiongemma_sparse_threshold
            if prep["enc_rows"] is not None:
                # encode rows carry no cached context; their verdict is meaningless
                verdict = verdict | prep["enc_rows"]
            ctx.salient_rows = verdict
            return out
        if cnt == 0:
            ctx.salient_rows = torch.zeros_like(sal)
            return c_full.to(q.dtype)
        mask = torch.empty(n, dtype=torch.bool, device=dev)
        out = attention_sparse_varlen(
            (q * self.head_dim**0.5).to(torch.bfloat16).contiguous(),
            kgp.to(torch.bfloat16).contiguous(),
            vd.to(torch.bfloat16).contiguous(),
            c_full.to(torch.bfloat16).contiguous(),
            o_sal[:cnt].to(torch.bfloat16).contiguous(),
            ctx.cu_seqlens_q, prep["cu_k"],
            ctx.max_seqlen_q, prep["max_kv"], n,
            cu_sal, idx_q[:cnt], stats, mask,
            float(self.diffusiongemma_sparse_threshold), True, idx_k[:cnt],
        )
        if prep["enc_rows"] is not None:
            mask |= prep["enc_rows"]
        ctx.salient_rows = mask
        return out.to(q.dtype)

    _cnt_shared: dict = {}
    _scratch_shared: dict = {}

    def _cnt_slot(self, dev):
        slot = DiffusionGemmaAttention._cnt_shared.get(str(dev))
        if slot is None:
            slot = (torch.empty(1, dtype=torch.int32, device="cpu", pin_memory=True), torch.cuda.Event())
            DiffusionGemmaAttention._cnt_shared[str(dev)] = slot
        return slot

    def _scratch(self, n, dev, dtype):
        # launch scratch, grown to the largest n seen
        key = (self.num_heads, self.head_dim, dtype, str(dev))
        buf = DiffusionGemmaAttention._scratch_shared.get(key)
        if buf is None or buf[0].size(0) < n:
            buf = (torch.empty(n, self.num_heads, self.head_dim, dtype=dtype, device=dev),
                   torch.empty(n, dtype=torch.int32, device=dev),
                   torch.empty(n, dtype=torch.int32, device=dev),
                   torch.zeros(n, 3, dtype=torch.float32, device=dev))
            DiffusionGemmaAttention._scratch_shared[key] = buf
        stats = buf[3][:n]
        stats.zero_()
        return buf[0][:n], buf[1][:n], buf[2][:n], stats

    def _vdelta_buffer(self, prep, q):
        key = prep["vd_key"]
        buf = DiffusionGemmaAttention._vdelta_shared.get(key[0])
        if buf is None or buf[0] != key or buf[1].device != q.device:
            t = torch.zeros(prep["packed_total"], self.num_kv_heads, self.head_dim, dtype=q.dtype, device=q.device)
            DiffusionGemmaAttention._vdelta_shared[key[0]] = (key, t)
            return t
        return buf[1]

    def _cache_admit(self, seq_ids, k):
        ctx = get_context()
        new_sids = [(i, sid) for i, sid in enumerate(seq_ids) if sid not in self.cache_slots]
        if not new_sids:
            return
        cu = ctx.cu_seqlens_q_cpu or [0, k.size(0)]
        new_regions = []
        for i, sid in new_sids:
            cap = ctx.kv_cache_caps[i]
            if self.sliding_window is not None:
                # one page per seq: trimmed prefix + canvas
                if self.page is None:
                    need = self.sliding_window - 1 + ctx.canvas_len
                    self.page = -(-need // 256) * 256
                cap = -(-max(self.page, cu[i + 1] - cu[i]) // self.page) * self.page
            else:
                if self.page is None:
                    self.page = 256
                cap = -(-cap // self.page) * self.page
            new_regions.append((sid, cap))
        grow = sum(cap for _, cap in new_regions)
        shape = (grow, self.num_kv_heads, self.head_dim)
        if self.key_cache is None:
            self.key_cache, self.value_cache = k.new_empty(shape), k.new_empty(shape)
            cursor = 0
        elif self.cache_free_rows:
            live = sorted(self.cache_slots.items(), key=lambda item: item[1][0])
            total = sum(entry[1] for _, entry in live) + grow
            new_k = k.new_empty((total, self.num_kv_heads, self.head_dim))
            new_v = k.new_empty((total, self.num_kv_heads, self.head_dim))
            cursor = 0
            for sid, (st, cap, plen) in live:
                # the whole region moves: mid-canvas seqs keep their stale canvas K/V
                new_k[cursor : cursor + cap] = self.key_cache[st : st + cap]
                new_v[cursor : cursor + cap] = self.value_cache[st : st + cap]
                self.cache_slots[sid] = [cursor, cap, plen]
                cursor += cap
            self.key_cache, self.value_cache = new_k, new_v
            self.cache_free_rows = 0
        else:
            cursor = self.key_cache.size(0)
            self.key_cache = torch.cat([self.key_cache, k.new_empty(shape)])
            self.value_cache = torch.cat([self.value_cache, k.new_empty(shape)])
        for sid, cap in new_regions:
            self.cache_slots[sid] = [cursor, cap, 0]
            cursor += cap


class DiffusionGemmaRouter(nn.Module):
    """Router parameters; the input RMSNorm (no learnable scale) runs inside ``moe_gate``."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k_experts
        self.rms_norm_eps = config.rms_norm_eps
        self.proj = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return moe_gate(
            x,
            self.proj.weight,
            self.top_k,
            router_scale=self.scale,
            per_expert_scale=self.per_expert_scale,
            rms_norm_eps=self.rms_norm_eps,
        )


class DiffusionGemmaExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        intermediate = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_experts, 2 * intermediate, config.hidden_size)
        )
        self.down_proj = nn.Parameter(torch.empty(config.num_experts, config.hidden_size, intermediate))

    def forward(self, x: torch.Tensor, topk_w: torch.Tensor, topk_i: torch.Tensor) -> torch.Tensor:
        return fused_experts(x, self.gate_up_proj, self.down_proj, topk_w, topk_i, activation="gelu")


class DiffusionGemmaSalientCache(nn.Module):
    """Per-layer salient state: context/FFN caches, row selection, sparse handoff."""

    def __init__(self, threshold: float | None, hidden_size: int):
        super().__init__()
        self.threshold = threshold
        self.hidden_size = hidden_size
        self.key = None
        self.ctx_cache = None
        self.ffn_cache = None
        self.valid = None
        self.active = False

    def publish(self, attn):
        ctx = get_context()
        self.active = False
        if (
            self.threshold is None
            or not attn.diffusiongemma_sparse
            or ctx.attn_mode not in ("denoise", "mixed")
            or ctx.denoise_rows is None
            or self.ctx_cache is None
        ):
            ctx.ctx_cache = None
            ctx.salient_rows = None
            return
        key = (tuple(get_metadata().running_seqs), tuple(ctx.cu_seqlens_q_cpu))
        if self.key != key:
            self._remap(key, self.ctx_cache)
        ctx.ctx_cache = self.ctx_cache
        # layer 0 starts from every denoise row (self-conditioning moves the whole
        # input between steps), deeper layers from the previous verdict
        incoming = ctx.salient_rows if ctx.salient_rows is not None else ctx.denoise_rows
        salient = incoming | ~self.valid
        if ctx.full_rows is not None:
            salient = salient | ctx.full_rows  # first steps of a canvas: dense
        ctx.salient_rows = salient
        self.active = True

    def rows(self, attn):
        ctx = get_context()
        if (
            self.threshold is None
            or ctx.attn_mode not in ("denoise", "mixed")
            or ctx.denoise_rows is None
        ):
            return None
        context = attn._context
        key = (tuple(get_metadata().running_seqs), tuple(ctx.cu_seqlens_q_cpu))
        if self.key != key:
            self._remap(key, context)
        if self.active and ctx.salient_rows is not None:
            salient = ctx.salient_rows | ~self.valid
        else:
            # before the first publish of a canvas there is no verdict: every row salient
            salient = torch.ones_like(self.valid)
        if ctx.denoise_rows is not None:
            salient = salient | ~ctx.denoise_rows  # encode rows: always dense
        if ctx.full_rows is not None:
            salient = salient | ctx.full_rows
        if not self.active:
            ctx.salient_rows = salient  # next layer's incoming
        self.ctx_cache = context
        self.valid = self.valid | salient
        return salient

    def merge_ffn(self, ffn, idx):
        if idx is None:
            self.ffn_cache = ffn
            return ffn
        self.ffn_cache.index_copy_(0, idx, ffn)
        return self.ffn_cache

    def _remap(self, key, template):
        seq_ids, cu = key
        N = cu[-1]
        new_ctx = template.new_zeros(N, template.size(-1))
        new_ffn = template.new_zeros(N, self.hidden_size)
        new_valid = torch.zeros(N, dtype=torch.bool, device=template.device)
        if self.key is not None and self.ffn_cache is not None:
            old_ids, old_cu = self.key
            old_pos = {sid: j for j, sid in enumerate(old_ids)}
            for i, sid in enumerate(seq_ids):
                j = old_pos.get(sid)
                if j is None or cu[i + 1] - cu[i] != old_cu[j + 1] - old_cu[j]:
                    continue
                new_ctx[cu[i] : cu[i + 1]] = self.ctx_cache[old_cu[j] : old_cu[j + 1]]
                new_ffn[cu[i] : cu[i + 1]] = self.ffn_cache[old_cu[j] : old_cu[j + 1]]
                new_valid[cu[i] : cu[i + 1]] = self.valid[old_cu[j] : old_cu[j + 1]]
        self.ctx_cache, self.ffn_cache, self.valid = new_ctx, new_ffn, new_valid
        self.key = key


class DiffusionGemmaDecoderLayer(nn.Module):
    """Dense MLP and MoE experts run in parallel (shared-expert design); the router
    reads the hidden states before the feed-forward norms."""

    def __init__(self, config, layer_idx: int, threshold: float | None = None):
        super().__init__()
        hidden_size = config.hidden_size
        eps = config.rms_norm_eps
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        rope = config.rope_parameters["sliding_attention" if is_sliding else "full_attention"]
        self.self_attn = DiffusionGemmaAttention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads if is_sliding else config.num_global_key_value_heads,
            head_dim=config.head_dim if is_sliding else config.global_head_dim,
            sliding_window=config.sliding_window if is_sliding else None,
            rope_theta=rope["rope_theta"],
            partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
            rms_norm_eps=eps,
        )
        self.mlp = DiffusionGemmaMLP(hidden_size, config.intermediate_size)
        self.router = DiffusionGemmaRouter(config)
        self.experts = DiffusionGemmaExperts(config)
        self.input_layernorm = DiffusionGemmaRMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm_1 = DiffusionGemmaRMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(hidden_size, eps=eps)
        self.register_buffer("layer_scalar", torch.ones(1))
        self.salient = DiffusionGemmaSalientCache(threshold, hidden_size)
        if threshold is not None:
            self.self_attn.capture_context = True
            self.self_attn.diffusiongemma_sparse_threshold = threshold

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        self.salient.publish(self.self_attn)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        salient = self.salient.rows(self.self_attn)
        idx = None
        if salient is not None:
            idx = salient.nonzero().squeeze(1)
            n = idx.numel()
            if n == 0:
                return (residual + self.salient.ffn_cache) * self.layer_scalar
            if n == salient.numel():
                idx = None
        x = residual if idx is None else residual.index_select(0, idx)
        branch_mlp = self.post_feedforward_layernorm_1(
            self.mlp(self.pre_feedforward_layernorm(x))
        )
        topk_w, topk_i = self.router(x)
        branch_moe = self.experts(self.pre_feedforward_layernorm_2(x), topk_w, topk_i)
        branch_moe = self.post_feedforward_layernorm_2(branch_moe)

        ffn = self.post_feedforward_layernorm(branch_mlp + branch_moe)
        if salient is not None:
            ffn = self.salient.merge_ffn(ffn, idx)
        hidden_states = residual + ffn
        return hidden_states * self.layer_scalar


class DiffusionGemmaModel(nn.Module):
    def __init__(self, config, threshold: float | None = None):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = nn.ModuleList(
            [DiffusionGemmaDecoderLayer(config, i, threshold) for i in range(config.num_hidden_layers)]
        )
        self.norm = DiffusionGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_conditioning = DiffusionGemmaSelfConditioning(config)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        if ctx.attn_mode == "mixed":
            # self-conditioning on the denoise rows only (encode rows carry none)
            rows = ctx.denoise_rows
            z = ctx.self_conditioning
            sig = z[rows] if z is not None else torch.zeros_like(hidden_states[rows])
            hidden_states = hidden_states.clone()
            hidden_states[rows] = self.self_conditioning(hidden_states[rows], sig)
        elif ctx.attn_mode == "denoise":
            # canvas rows only; zero signal is not a skip (weightless post-norm)
            z = ctx.self_conditioning
            sig = z if z is not None else torch.zeros_like(hidden_states)
            hidden_states = self.self_conditioning(hidden_states, sig)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class DiffusionGemmaForDLM(nn.Module):
    def __init__(self, config: DiffusionGemmaConfig, threshold: float | None = None):
        super().__init__()
        self.config = config
        text_config = getattr(config, "text_config", None) or config
        self.text_config = text_config
        self.model = DiffusionGemmaModel(text_config, threshold)
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # raw logits; the sampler applies the softcap
        return self.lm_head(hidden_states)

    def soft_embed(self, probs: torch.Tensor) -> torch.Tensor:
        # sampler probs projected through the embedding, scaled like token embeddings
        w = self.model.embed_tokens.weight
        return probs.to(w.dtype).view(-1, probs.shape[-1]) @ w * self.model.embed_scale

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    def normalize_weight_name(self, name: str) -> str | None:
        if "self_conditioning." in name:
            return "model.self_conditioning." + name.split("self_conditioning.", 1)[1]
        if name.startswith(("model.encoder.vision_tower.", "model.encoder.embed_vision.")):
            return None
        if name.startswith("model.encoder.language_model."):
            return "model." + name[len("model.encoder.language_model.") :]
        if name.startswith("model.decoder."):
            return "model." + name[len("model.decoder.") :]
        if name.startswith("lm_head."):
            return None
        return name

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # the checkpoint ties both attention modes to one backbone; vision tower not ported
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        expected = set(self.state_dict().keys()) - {"lm_head.weight"}
        loaded: set[str] = set()
        unmatched: list[str] = []
        for name, w in weights:
            name = self.normalize_weight_name(name)
            if name is None or name in loaded:
                continue
            target = params.get(name)
            if target is not None:
                loader = getattr(target, "weight_loader", None)
                if loader is not None:
                    loader(target, w)
                else:
                    target.data.copy_(w)
            elif name in buffers:
                buffers[name].copy_(w)
            else:
                unmatched.append(name)
                continue
            loaded.add(name)
        assert not unmatched, f"unmatched checkpoint weights: {unmatched[:8]}"
        missing = expected - loaded
        assert not missing, f"weights missing from checkpoint: {sorted(missing)[:8]}"
        return loaded
