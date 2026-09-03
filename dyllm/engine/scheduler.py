from collections import deque
from typing import List, Optional
import torch
import numpy as np

from dyllm.config import Config
from dyllm.engine.sequence import Sequence, SequenceStatus
from dyllm.utils.metadata import get_metadata


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask = config.mask_id
        self.full: deque[Sequence] = deque()
        self.sparse: deque[Sequence] = deque()
        self.prune: deque[Sequence] = deque()
        self.finished: List[int] = []
        self.canvas_length = config.canvas_length
        self.is_diffusiongemma = config.hf_config.model_type == "diffusion_gemma"
        if self.is_diffusiongemma:
            self.vocab_size = config.vocab_size
            self.eos_ids = config.eos_ids
            self.pad_id = config.pad_id if config.pad_id is not None else 0
            self.num_rounds = 0
            self.num_canvas_steps = 0

    def is_finished(self):
        return not self.full and not self.sparse

    def add(self, seq: Sequence):
        self.full.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool | list[bool]]:
        if self.is_diffusiongemma:
            return self.schedule_diffusiongemma()
        # full: stays in full list until it goes through enough numbers of full steps
        scheduled_seqs = []
        seen = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.full and num_seqs < self.max_num_seqs:
            if num_batched_tokens + len(self.full[0]) > self.max_num_batched_tokens:
                break
            seq = self.full.popleft()
            if seq.seq_id in seen:
                self.full.appendleft(seq)
                break

            seq.processed_steps += 1
            num_batched_tokens += len(seq)
            num_seqs += 1
            scheduled_seqs.append(seq)
            seen.append(seq.seq_id)

            if seq.processed_steps < seq.num_full_steps:
                seq.status = SequenceStatus.FULL
                self.full.append(seq)
            else:
                seq.status = SequenceStatus.SPARSE
                self.sparse.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # sparse: need at least 1 full step before running a sparse step
        while self.sparse and num_seqs < self.max_num_seqs:
            if num_batched_tokens > self.max_num_batched_tokens:
                break
            seq = self.sparse.popleft()
            if seq.seq_id in seen:
                self.sparse.appendleft(seq)
                break

            num_seqs += 1
            num_batched_tokens += len(seq)
            scheduled_seqs.append(seq)
            seq.processed_steps += 1
            seen.append(seq.seq_id)

        assert scheduled_seqs
        self.sparse.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def schedule_diffusiongemma(self) -> tuple[list[Sequence], list[bool]]:
        # mixed batch: seqs awaiting their encode pass ride as causal segments, the rest as bidirectional
        scheduled_seqs, modes = [], []
        num_tokens = 0
        for seq in self.full:
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            if seq.needs_encode:
                rows = len(seq) - seq.encoded_len
                causal = True
            else:
                rows = self.canvas_length
                causal = False
            if num_tokens + rows > self.max_num_batched_tokens:
                break
            if not causal:
                seq.processed_steps += 1  # denoise steps only -- cur_step depends on it
            scheduled_seqs.append(seq)
            modes.append(causal)
            num_tokens += rows
        assert scheduled_seqs
        self.num_rounds += 1
        self.num_canvas_steps += sum(1 for m in modes if not m)
        return scheduled_seqs, modes

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.FULL
        self.full.appendleft(seq)

    def eos_and_done(self, seq: Sequence, pos: int):
        for i in range(1, pos + 1):
            if seq[pos - i] == self.mask:
                return False
        return True

    def postprocess(
        self,
        seqs: list[Sequence],
        selected_positions: torch.Tensor,
        selected_tokens: torch.Tensor,
        selected_counts: torch.Tensor,
    ):
        finished = []

        B = selected_counts.size(0)
        L = selected_tokens.size(1)
        packed_gpu = torch.cat([selected_counts.long(), selected_tokens.view(-1), selected_positions.view(-1)])

        packed_cpu = packed_gpu.cpu()
        packed_list = packed_cpu.tolist()
        num_unmasked_per_seq = packed_list[:B]

        tokens_start = B
        tokens_end = B + (B * L)
        flat_tokens = packed_list[tokens_start:tokens_end]
        flat_pos = packed_list[tokens_end:]

        for b, seq in enumerate(seqs):
            num_unmasked = num_unmasked_per_seq[b]
            if num_unmasked > 0:
                start_idx = b * L
                end_idx = start_idx + num_unmasked
                tok_slice = flat_tokens[start_idx:end_idx]
                pos_slice = flat_pos[start_idx:end_idx]
                seq.update_token(pos_slice, tok_slice)
                seq.update_block_idx()

                if self.eos in tok_slice:
                    eos_idx = tok_slice.index(self.eos)
                    eos_pos = pos_slice[eos_idx]
                    if self.eos_and_done(seq, eos_pos) and not seq.ignore_eos:
                        seq.status = SequenceStatus.FINISHED
                        finished.append(seq.seq_id)
                        if seq.processed_steps < seq.num_full_steps:
                            if seq in self.full:
                                self.full.remove(seq)
                        else:
                            if seq in self.sparse:
                                self.sparse.remove(seq)
                        continue

            steps_done = seq.confidence_threshold is None and seq.processed_steps == seq.num_steps
            if steps_done or seq.num_completion_tokens >= seq.max_new_tokens:
                seq.status = SequenceStatus.FINISHED
                finished.append(seq.seq_id)
                if seq.processed_steps < seq.num_full_steps:
                    if seq in self.full:
                        self.full.remove(seq)
                else:
                    if seq in self.sparse:
                        self.sparse.remove(seq)

        metadata = get_metadata()
        metadata.finished_seqs = finished

    def postprocess_diffusiongemma(self, seqs: list[Sequence], result: dict):
        for sid in result.get("encoded", []):
            for seq in seqs:
                if seq.seq_id == sid:
                    seq.mark_encoded()
        den = result.get("denoise_seqs")
        if not den:
            get_metadata().finished_seqs = []
            return
        seqs = den
        finished = []
        canvas, argmax = result["canvas"], result["argmax"]
        stop, history, z = result["stop"].tolist(), result["history"], result["self_conditioning"]
        for b, seq in enumerate(seqs):
            last_step = seq.cur_step == 1
            seq.apply_step(canvas[b], argmax[b], history[:, b], z[b])
            if stop[b] or last_step:
                done = seq.commit_canvas(self.eos_ids, self.pad_id) and not seq.ignore_eos
                if done or seq.num_completion_tokens >= seq.max_new_tokens:
                    seq.status = SequenceStatus.FINISHED
                    finished.append(seq.seq_id)
                    if seq in self.full:
                        self.full.remove(seq)
                else:
                    seq.open_canvas(self.vocab_size, self.canvas_length, device=canvas.device)
        metadata = get_metadata()
        metadata.finished_seqs = finished
