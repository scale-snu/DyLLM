"""Triton ops for the DiffusionGemma attention step.

Dense backend over a paged KV cache: one unified-attention call per layer for a
mixed causal/bidirectional batch, the mode carried as a per-seq causal flag
(both head sizes).

Sparse prep: fused per-row prep for the sparse step -- salient rows write K/V into
the cache, their dV into the packed buffer and compact q / row indices; the rest
only zero their dV slot and keep their stale cache K/V (the approximation).
"""

try:
    import triton
    import triton.language as tl
    from dyllm.model_executor.layers.triton_attn import unified_attention
except Exception:  # triton missing (cpu envs)
    triton = None
    unified_attention = None


def attend_triton(q, k_pages, v_pages, out, cu_q, max_q, seqused_k, max_k,
                  block_table, causal, sm_scale, window=(-1, -1)):
    # k/v_pages: [num_pages, page, H_kv, D]; causal: bool or [B] bool tensor.
    # window applies to causal rows only (patched kernel; bidi rows see all KV).
    unified_attention(
        q, k_pages, v_pages, out, cu_q, max_q, seqused_k, max_k,
        sm_scale, causal, window, block_table, 0.0, None, None, None,
    )
    return out


if triton is not None:

    @triton.jit
    def _sparse_prep_kernel(
        sal_ptr, den_ptr, kv_row_ptr, pk_row_ptr, incl_ptr,
        q_ptr, k_ptr, v_ptr,
        key_cache_ptr, value_cache_ptr, vd_ptr, q_sal_ptr, idx_q_ptr, idx_k_ptr,
        KV_ROW: tl.constexpr, Q_ROW: tl.constexpr, BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0)
        sal = tl.load(sal_ptr + i)
        den = tl.load(den_ptr + i)
        kv_row = tl.load(kv_row_ptr + i).to(tl.int64)
        pk_row = tl.load(pk_row_ptr + i).to(tl.int64)
        pos = tl.load(incl_ptr + i).to(tl.int64) - 1  # inclusive cumsum of sal -> compact slot
        write_kv = sal != 0
        take_dv = write_kv & (den != 0)
        for off in tl.static_range(0, KV_ROW, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = cols < KV_ROW
            v_new = tl.load(v_ptr + i * KV_ROW + cols, mask=m & write_kv, other=0.0)
            v_old = tl.load(value_cache_ptr + kv_row * KV_ROW + cols, mask=m & write_kv, other=0.0)
            k_new = tl.load(k_ptr + i * KV_ROW + cols, mask=m & write_kv, other=0.0)
            tl.store(key_cache_ptr + kv_row * KV_ROW + cols, k_new, mask=m & write_kv)
            tl.store(value_cache_ptr + kv_row * KV_ROW + cols, v_new, mask=m & write_kv)
            dv = tl.where(take_dv, v_new - v_old, 0.0).to(v_new.dtype)
            tl.store(vd_ptr + pk_row * KV_ROW + cols, dv, mask=m)
        for off in tl.static_range(0, Q_ROW, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = (cols < Q_ROW) & write_kv
            q_row = tl.load(q_ptr + i * Q_ROW + cols, mask=m, other=0.0)
            tl.store(q_sal_ptr + pos * Q_ROW + cols, q_row, mask=m)
        if write_kv:
            tl.store(idx_q_ptr + pos, i.to(tl.int32))
            tl.store(idx_k_ptr + pos, pk_row.to(tl.int32))


    def sparse_prep(sal, den, kv_row, pk_row, incl, q, k, v, key_cache, value_cache, vd, q_sal, idx_q, idx_k):
        """All row tensors are [n]; q [n,H,D]; k/v [n,Hkv,D]; caches [*,Hkv,D]; vd [T,Hkv,D];
        q_sal [n_sal_cap,H,D]; idx_* int32. Contiguous inputs assumed."""
        n = q.shape[0]
        if n == 0:
            return
        kv_row_len = k.shape[1] * k.shape[2]
        q_row_len = q.shape[1] * q.shape[2]
        _sparse_prep_kernel[(n,)](
            sal, den, kv_row, pk_row, incl, q, k, v, key_cache, value_cache, vd, q_sal, idx_q, idx_k,
            KV_ROW=kv_row_len, Q_ROW=q_row_len, BLOCK=1024, num_warps=4,
        )

else:
    sparse_prep = None
