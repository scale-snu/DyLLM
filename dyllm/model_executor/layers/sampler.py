import math
from typing import Optional, Tuple

import torch
import torch.distributions as dists
import torch.nn.functional as F
from torch import nn

from dyllm.utils.context import Context

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_indices_kernel(output_ptr, cu_counts_ptr, n_groups, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_groups:
        return

    start = tl.load(cu_counts_ptr + pid)
    end = tl.load(cu_counts_ptr + pid + 1)
    count = end - start

    if count == 0:
        return

    output_offset = output_ptr + start
    for i in range(0, count, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < count
        tl.store(output_offset + offsets, pid, mask=mask)


def triton_repeat_interleave(cu_filtered: torch.Tensor, total_tokens: int, out_tensor: torch.Tensor):
    B = cu_filtered.shape[0] - 1
    grid = (B,)
    _expand_indices_kernel[grid](out_tensor, cu_filtered, B, BLOCK_SIZE=256)
    return out_tensor


@triton.jit
def filter_and_count_kernel(
    scores_ptr,
    tokens_ptr,
    pos_ptr,
    num_transfer_ptr,
    conf_thr_ptr,
    out_tokens_ptr,
    out_pos_ptr,
    out_counts_ptr,
    stride_b,
    stride_l,
    B,
    Max_Len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    num_transfer_val = tl.load(num_transfer_ptr + pid)
    conf_thr_val = tl.load(conf_thr_ptr + pid)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < Max_Len

    row_start = pid * stride_b
    s_val = tl.load(scores_ptr + row_start + offs, mask=mask, other=float("-inf"))
    t_val = tl.load(tokens_ptr + row_start + offs, mask=mask, other=0)
    p_val = tl.load(pos_ptr + row_start + offs, mask=mask, other=0)

    is_top_k = offs < num_transfer_val
    is_above_conf = s_val >= conf_thr_val
    is_data_valid = s_val > float("-inf")

    keep = (is_top_k | is_above_conf) & is_data_valid

    t_out = tl.where(keep, t_val, -1)
    p_out = tl.where(keep, p_val, 0)

    count = tl.sum(tl.where(keep, 1, 0))

    tl.store(out_tokens_ptr + row_start + offs, t_out, mask=mask)
    tl.store(out_pos_ptr + row_start + offs, p_out, mask=mask)
    tl.store(out_counts_ptr + pid, count)


def launch_filter_and_count(dense_scores, dense_tokens, dense_pos, num_transfer, conf_thr, B, Max_Len):
    out_tokens = torch.empty_like(dense_tokens)
    out_pos = torch.empty_like(dense_pos)
    out_counts = torch.empty((B,), dtype=torch.int32, device=dense_scores.device)

    grid = (B,)
    BLOCK_SIZE = triton.next_power_of_2(Max_Len)

    filter_and_count_kernel[grid](
        dense_scores,
        dense_tokens,
        dense_pos,
        num_transfer,
        conf_thr,
        out_tokens,
        out_pos,
        out_counts,
        dense_scores.stride(0),
        dense_scores.stride(1),
        B,
        Max_Len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_pos, out_tokens, out_counts


@triton.jit
def ragged_to_dense_kernel(
    scores_ptr,
    tokens_ptr,
    pos_ptr,
    cu_seqlens_ptr,
    dense_scores_ptr,
    dense_tokens_ptr,
    dense_pos_ptr,
    max_len,
    n_total_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    start_idx = tl.load(cu_seqlens_ptr + pid)
    end_idx = tl.load(cu_seqlens_ptr + pid + 1)
    actual_len = end_idx - start_idx

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < actual_len

    ragged_offs = start_idx + offs
    ragged_mask = mask & (ragged_offs < n_total_tokens)

    s_val = tl.load(scores_ptr + ragged_offs, mask=ragged_mask, other=float("-inf"))
    t_val = tl.load(tokens_ptr + ragged_offs, mask=ragged_mask, other=0)
    p_val = tl.load(pos_ptr + ragged_offs, mask=ragged_mask, other=0)

    dense_offs = pid * max_len + offs

    s_out = tl.where(mask, s_val, float("-inf"))
    t_out = tl.where(mask, t_val, 0)
    p_out = tl.where(mask, p_val, 0)

    tl.store(dense_scores_ptr + dense_offs, s_out)
    tl.store(dense_tokens_ptr + dense_offs, t_out)
    tl.store(dense_pos_ptr + dense_offs, p_out)


def launch_ragged_to_dense(scores, tokens, pos, cu_seqlens, B, max_len=32):
    device = scores.device
    dense_scores = torch.empty((B, max_len), dtype=scores.dtype, device=device)
    dense_tokens = torch.empty((B, max_len), dtype=tokens.dtype, device=device)
    dense_pos = torch.empty((B, max_len), dtype=pos.dtype, device=device)

    BLOCK_SIZE = triton.next_power_of_2(max_len)

    grid = (B,)
    ragged_to_dense_kernel[grid](
        scores,
        tokens,
        pos,
        cu_seqlens,
        dense_scores,
        dense_tokens,
        dense_pos,
        max_len,
        scores.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dense_scores, dense_tokens, dense_pos


import torch
import torch.nn.functional as F
from typing import Optional, Union


def top_p_logits(logits: torch.Tensor, top_p: Union[float, torch.Tensor, None] = None) -> torch.Tensor:
    if top_p is None:
        return logits

    p_val = top_p.view(-1, 1)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p_val
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)

    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def top_k_logits(logits: torch.Tensor, top_k: Union[int, torch.Tensor, None] = None) -> torch.Tensor:
    if top_k is None:
        return logits

    max_k = int(top_k.max().item())
    max_k = min(max_k, logits.size(-1))
    top_k_vals, _ = torch.topk(logits, max_k, dim=-1)
    k_indices = (top_k - 1).clamp(min=0).long().unsqueeze(-1)
    thresholds = top_k_vals.gather(1, k_indices)
    return logits.masked_fill(logits < thresholds, torch.finfo(logits.dtype).min)


class BaseSampler(nn.Module):
    def __init__(self, algorithm: str = "confidence"):
        super().__init__()
        self.algorithm = algorithm

    def adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def sample_token(
        self,
        probs: torch.Tensor,
        top_tokens: torch.Tensor,
        temperatures: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return top_tokens

    def compute_scores(
        self,
        probs: torch.Tensor,
        top_probs: torch.Tensor,
    ) -> torch.Tensor:
        if self.algorithm == "confidence":
            scores = top_probs
        elif self.algorithm == "margin_confidence":
            top2_probs, _ = probs.topk(k=2, dim=-1)
            scores = top2_probs[:, 0] - top2_probs[:, 1]
        elif self.algorithm == "random":
            scores = torch.rand_like(top_probs)
        else:
            raise ValueError(f"Unsupported algorithm for LLaDA: {self.algorithm}")
        return scores

    def forward(
        self,
        input_logits: torch.Tensor,
        ctx: Context,
        input_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        temperatures: Optional[torch.Tensor] = None,
        num_transfer: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        block_size: int = 32,
        confidence_thresholds: Optional[torch.Tensor] = None,
    ):
        logits = self.adjust_logits(input_logits)
        device = logits.device

        relative_idx, batch_offsets, cu_filtered = input_indices
        local_idx = relative_idx
        B = cu_filtered.shape[0] - 1

        seq_offsets = ctx.cu_seqlens_q
        base_offsets = seq_offsets[:-1]

        total_tokens = relative_idx.shape[0]
        group_ids = torch.empty(total_tokens, device=device, dtype=torch.long)
        triton_repeat_interleave(cu_filtered, total_tokens, out_tensor=group_ids)

        if num_transfer is not None:
            num_transfer_per_seq = num_transfer.flatten().clamp_min_(0).to(device)
        else:
            num_transfer_per_seq = torch.zeros(B, device=device, dtype=torch.long)

        base_offsets_flat = base_offsets.index_select(0, group_ids)
        global_rows = base_offsets_flat + local_idx
        cand_logits = logits.index_select(0, global_rows)

        if temperatures is not None:
            scaled_logits = cand_logits / temperatures.unsqueeze(-1)
        else:
            scaled_logits = cand_logits

        if top_k is not None:
            scaled_logits = top_k_logits(scaled_logits, top_k)
        if top_p is not None:
            scaled_logits = top_p_logits(scaled_logits, top_p)

        probs = F.softmax(scaled_logits, dim=-1)
        top_probs, top_tokens = probs.max(dim=-1)
        scores = self.compute_scores(probs, top_probs)

        sampled_all = self.sample_token(probs, top_tokens, temperatures)

        tokens_offset = batch_offsets.gather(0, group_ids)
        abs_idx = local_idx + tokens_offset

        dense_scores, dense_tokens, dense_pos = launch_ragged_to_dense(
            scores, sampled_all, abs_idx, cu_filtered, B, block_size
        )

        sorted_scores, sorted_indices = torch.sort(dense_scores, dim=1, descending=True)
        sorted_tokens = torch.gather(dense_tokens, 1, sorted_indices)
        sorted_pos = torch.gather(dense_pos, 1, sorted_indices)

        if confidence_thresholds is None:
            conf_thr = torch.full((B,), float("inf"), device=device, dtype=sorted_scores.dtype)
        else:
            conf_thr = confidence_thresholds.to(sorted_scores.dtype)

        out_pos, out_tokens, out_counts = launch_filter_and_count(
            sorted_scores, sorted_tokens, sorted_pos, num_transfer_per_seq, conf_thr, B, block_size
        )

        return out_pos, out_tokens, out_counts


class LLaDASampler(BaseSampler):
    pass


class DreamSampler(BaseSampler):
    def adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.size(0) <= 1:
            return logits
        return torch.cat([logits[:1], logits[:-1]], dim=0)

    def sample_token(
        self,
        probs: torch.Tensor,
        top_tokens: torch.Tensor,
        temperatures: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if temperatures is not None and torch.any(temperatures > 0):
            try:
                sampled_all = dists.Categorical(probs=probs).sample()
            except Exception:
                sampled_all = top_tokens
        else:
            sampled_all = top_tokens
        return sampled_all

    def compute_scores(
        self,
        probs: torch.Tensor,
        top_probs: torch.Tensor,
    ) -> torch.Tensor:
        if self.algorithm == "entropy":
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            scores = torch.sum(probs * log_probs, dim=-1)
        elif self.algorithm == "origin":
            scores = torch.rand_like(top_probs)
        else:
            raise ValueError(f"Unsupported algorithm for Dream: {self.algorithm}")
        return scores


def _diffusiongemma_row_stats(logits: torch.Tensor, temp: torch.Tensor, softcap: float):
    """Per-row statistics of processed = softcap(logits) / temp, in fp32
    (logits.to(float32) -> softcap -> temperature -> entropy).
    logits [n, V] (any dtype), temp [n] fp32 -> argmax [n], entropy [n] fp32,
    probs [n, V] in the logits dtype (self-conditioning input)."""
    x = logits.float()
    if softcap > 0:
        x = torch.tanh(x / softcap) * softcap
    x = x / temp[:, None]
    normalized = x - torch.logsumexp(x, dim=-1, keepdim=True)
    probs = torch.softmax(normalized, dim=-1)
    entropy = -(normalized * probs).sum(-1)
    return torch.argmax(x, dim=-1), entropy, probs.to(logits.dtype)


_diffusiongemma_row_stats_compiled = None


def _diffusiongemma_row_stats_fn(device):
    """torch.compile'd on cuda; eager otherwise."""
    global _diffusiongemma_row_stats_compiled
    if device.type != "cuda":
        return _diffusiongemma_row_stats
    if _diffusiongemma_row_stats_compiled is None:
        _diffusiongemma_row_stats_compiled = torch.compile(_diffusiongemma_row_stats, dynamic=True)
    return _diffusiongemma_row_stats_compiled


class DiffusionGemmaSampler(nn.Module):
    """Entropy-bound canvas sampler for DiffusionGemma

    ``cur_steps`` counts denoising steps remaining (``max_denoising_steps`` down to 1).
    Per-request values (entropy bound, temperature schedule, stopping thresholds)
    are passed per call -- the engine batches them as ``[B]`` tensors, matching the
    other samplers. The acceptance mask and the argmax history are passed in and
    returned rather than stored, so per-sequence state lives with the sequence.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def temperature(self, cur_steps, t_min, t_max, max_denoising_steps):
        return t_min + (t_max - t_min) * (cur_steps / max_denoising_steps)

    def sample(self, logits: torch.Tensor, cur_steps, t_min, t_max, max_denoising_steps,
               softcap=None, chunk: int = 1024):
        """One stochastic step over processed = softcap(logits) / temp, in fp32
        (logits.to(float32) -> softcap -> temperature).
        Returns (denoiser, argmax, entropy, probs); probs stays fp32 -- the
        multinomial draws from it. Row-chunked except the draw itself, which runs
        on the whole canvas so the RNG stream matches a single-shot sampler."""
        temp = self.temperature(cur_steps, t_min, t_max, max_denoising_steps)
        temp = torch.as_tensor(temp, device=logits.device, dtype=torch.float32).reshape(-1)
        lead = logits.shape[:-1]
        flat = logits.reshape(-1, logits.shape[-1])
        temp_rows = temp.repeat_interleave(flat.shape[0] // temp.numel())
        probs = torch.empty(flat.shape, dtype=torch.float32, device=flat.device)
        argmax = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
        entropy = torch.empty(flat.shape[0], dtype=torch.float32, device=flat.device)
        cap = float(softcap) if softcap else 0.0
        for i in range(0, flat.shape[0], chunk):
            x = flat[i : i + chunk].float()
            if cap > 0:
                x = torch.tanh(x / cap) * cap
            x = x / temp_rows[i : i + chunk, None]
            p = torch.softmax(x, dim=-1)
            probs[i : i + chunk] = p
            entropy[i : i + chunk] = -(p * (x - torch.logsumexp(x, dim=-1, keepdim=True))).sum(-1)
            argmax[i : i + chunk] = x.argmax(-1)
        denoiser = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return denoiser.view(lead), argmax.view(lead), entropy.view(lead), probs.view(*lead, -1)

    @staticmethod
    def _row_entropy(logits: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        # row-wise, so chunking is exact; keeps the [rows, vocab] temporaries small
        flat = logits.reshape(-1, logits.shape[-1])
        ent = torch.cat([dists.Categorical(logits=flat[i : i + chunk]).entropy() for i in range(0, flat.shape[0], chunk)])
        return ent.view(logits.shape[:-1])

    def stats(self, logits: torch.Tensor, cur_steps, t_min, t_max, max_denoising_steps, softcap=None, chunk: int = 1024):
        """Deterministic step statistics over processed = softcap(logits) / temp:
        argmax [.., L], entropy [.., L] fp32, probs [.., L, V] (SC input).
        ``cur_steps``/``t_*`` index dim 0 of ``logits`` (a seq, or a row when the
        caller passes ``[rows, 1, V]``). Row-chunked so the fp32 temporaries stay
        small; the row function is torch.compile'd on cuda. Chunks are written into
        preallocated outputs -- collecting them for a final ``cat`` would hold two
        full [rows, vocab] copies at once."""
        temp = self.temperature(cur_steps, t_min, t_max, max_denoising_steps)
        temp = torch.as_tensor(temp, device=logits.device, dtype=torch.float32).reshape(-1)
        lead = logits.shape[:-1]
        flat = logits.reshape(-1, logits.shape[-1])
        temp_rows = temp.repeat_interleave(flat.shape[0] // temp.numel())
        fn = _diffusiongemma_row_stats_fn(logits.device)
        cap = float(softcap) if softcap else 0.0
        rows, dev = flat.shape[0], flat.device
        argmax = torch.empty(rows, dtype=torch.long, device=dev)
        entropy = torch.empty(rows, dtype=torch.float32, device=dev)
        probs = torch.empty_like(flat)
        for i in range(0, rows, chunk):
            a, e, p = fn(flat[i : i + chunk], temp_rows[i : i + chunk], cap)
            argmax[i : i + chunk] = a
            entropy[i : i + chunk] = e
            probs[i : i + chunk] = p
        return argmax.view(lead), entropy.view(lead), probs.view(*lead, -1)

    def accept(
        self,
        current_canvas: torch.Tensor,
        denoiser_canvas: torch.Tensor,
        processed_logits: torch.Tensor,
        entropy_bound,
    ):
        return self.accept_from_entropy(current_canvas, denoiser_canvas, self._row_entropy(processed_logits), entropy_bound)

    def accept_from_entropy(self, current_canvas, denoiser_canvas, token_entropy, entropy_bound):
        sorted_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
        bound = entropy_bound.unsqueeze(-1) if isinstance(entropy_bound, torch.Tensor) else entropy_bound
        # accept in entropy order while the entropy mass accepted before each row stays under the bound
        sorted_mask = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy <= bound
        accepted_mask = torch.scatter(torch.zeros_like(sorted_mask), -1, sorted_indices, sorted_mask)
        return torch.where(accepted_mask, denoiser_canvas, current_canvas), accepted_mask

    def renoise(self, accepted_canvas: torch.Tensor, accepted_mask: torch.Tensor) -> torch.Tensor:
        random_canvas = torch.randint(
            0, self.vocab_size, accepted_canvas.shape, device=accepted_canvas.device
        )
        return torch.where(accepted_mask, accepted_canvas, random_canvas)

    def check_stop(
        self,
        argmax_canvas: torch.Tensor,
        processed_logits: torch.Tensor,
        history: Optional[torch.Tensor],
        convergence_threshold,
        stability_threshold: int,
    ):
        return self.check_stop_from_entropy(
            argmax_canvas, self._row_entropy(processed_logits), history, convergence_threshold, stability_threshold
        )

    def check_stop_from_entropy(self, argmax_canvas, token_entropy, history, convergence_threshold, stability_threshold):
        if stability_threshold == 0:
            stable = torch.ones(argmax_canvas.shape[0], dtype=torch.bool, device=argmax_canvas.device)
        else:
            if history is None:
                history = torch.full(
                    (stability_threshold, *argmax_canvas.shape),
                    -1,
                    dtype=argmax_canvas.dtype,
                    device=argmax_canvas.device,
                )
            stable = (history == argmax_canvas[None]).all(dim=-1).all(dim=0)
            history = torch.roll(history, shifts=-1, dims=0)
            history[-1] = argmax_canvas
        mean_entropy = token_entropy.mean(dim=-1)
        confident = mean_entropy < convergence_threshold
        # per-row: already below the stop threshold (salient path may freeze these rows)
        thr = convergence_threshold.unsqueeze(-1) if isinstance(convergence_threshold, torch.Tensor) else convergence_threshold
        converged = token_entropy < thr
        return stable & confident, history, converged

    def forward(
        self,
        logits: torch.Tensor,
        canvas: torch.Tensor,
        history: Optional[torch.Tensor],
        cur_steps,
        *,
        entropy_bound,
        t_min,
        t_max,
        max_denoising_steps,
        convergence_threshold,
        stability_threshold: int,
        deterministic: bool = False,
        softcap=None,
    ):
        """One denoising step; freezing finished rows stays with the caller.
        Returns (new_canvas, argmax, probs, stop, history, converged); ``probs`` =
        softmax(softcap(logits) / temp), the next step's self-conditioning input.
        ``logits`` are the raw LM-head logits when ``softcap`` is given; both
        paths apply it in fp32.

        ``deterministic``: argmax input tokens and no re-noise -- rejected positions
        keep their current token so contexts can stabilize across steps.
        """
        if deterministic:
            argmax_canvas, token_entropy, processed = self.stats(
                logits, cur_steps, t_min, t_max, max_denoising_steps, softcap
            )
            new_canvas, _ = self.accept_from_entropy(canvas, argmax_canvas, token_entropy, entropy_bound)
        else:
            denoiser_canvas, argmax_canvas, token_entropy, probs = self.sample(
                logits, cur_steps, t_min, t_max, max_denoising_steps, softcap
            )
            accepted, accepted_mask = self.accept_from_entropy(canvas, denoiser_canvas, token_entropy, entropy_bound)
            new_canvas = self.renoise(accepted, accepted_mask)
            processed = probs.to(logits.dtype)
        stop, history, converged = self.check_stop_from_entropy(
            argmax_canvas, token_entropy, history, convergence_threshold, stability_threshold
        )
        return new_canvas, argmax_canvas, processed, stop, history, converged
