import torch
from copy import copy
from enum import Enum, auto
from itertools import count

from dyllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    FULL = auto()
    SPARSE = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.FULL
        self.token_ids = copy(token_ids)
        self.mask_id = sampling_params.mask_id
        self.last_tokens = [token_ids[-1]]
        self.last_token_pos = [0]
        self.salient_ids = []
        self.output_length = sampling_params.max_new_tokens
        self.num_full_steps = sampling_params.num_full_steps
        self.num_steps = sampling_params.steps
        self.processed_steps = 0
        self.num_prompt_tokens = sum(1 for t in token_ids if t != self.mask_id)
        self.num_tokens = self.num_prompt_tokens
        self.temperature = sampling_params.temperature
        self.max_new_tokens = sampling_params.max_new_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k
        self.confidence_threshold = sampling_params.confidence_threshold

        self.block_size = sampling_params.block_size
        self.block_idx = 0

        # DiffusionGemma per-request sampling params (None for mask-based models)
        self.entropy_bound = sampling_params.entropy_bound
        self.t_min = sampling_params.t_min
        self.t_max = sampling_params.t_max
        self.max_denoising_steps = sampling_params.max_denoising_steps
        self.convergence_threshold = sampling_params.convergence_threshold
        self.stability_threshold = sampling_params.stability_threshold
        # DiffusionGemma canvas state
        self.canvas = None
        self.canvas_argmax = None
        self.canvas_history = None
        self.self_conditioning = None
        self.canvas_start_step = 0
        self.encoded_len = 0  # rows already written to the prefix KV cache

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def idx_updated_rows(self):
        return self.idx_updated_rows

    @property
    def num_transfer_tokens(self):
        assert self.output_length % self.num_steps == 0
        return self.output_length // self.num_steps

    def update_token(self, pos: list[int], token_id: list[int]):
        for p, t in zip(pos, token_id):
            if self.token_ids[p] == self.mask_id and t != self.mask_id:
                self.num_tokens += 1
            self.token_ids[p] = t
        self.last_tokens = copy(token_id)
        self.last_token_pos = copy(pos)

    def update_block_idx(self):
        if self.num_completion_tokens > 0 and self.block_size > 0:
            self.block_idx = (self.num_completion_tokens) // self.block_size

    @property
    def cur_step(self):
        # remaining denoising steps incl. current; valid only after schedule() increments
        return self.max_denoising_steps - (self.processed_steps - self.canvas_start_step) + 1

    @property
    def canvas_step(self):
        # 1-based denoise step index within the current canvas
        return self.processed_steps - self.canvas_start_step

    @property
    def needs_encode(self):
        # prompt or freshly committed canvas awaits its encode pass
        return self.encoded_len < len(self.token_ids)

    def mark_encoded(self):
        self.encoded_len = len(self.token_ids)

    def open_canvas(self, vocab_size: int, canvas_length: int, device="cpu"):
        # cpu randint keeps the standalone-loop RNG contract; the engine passes cuda
        self.canvas = torch.randint(0, vocab_size, (1, canvas_length))[0].to(device)
        self.canvas_argmax = self.canvas.clone()
        self.canvas_history = torch.full((self.stability_threshold, canvas_length), -1, dtype=torch.long, device=device)
        self.self_conditioning = None
        self.canvas_start_step = self.processed_steps

    def apply_step(self, canvas, argmax, history: torch.Tensor, self_conditioning: torch.Tensor):
        self.canvas = canvas
        self.canvas_argmax = argmax
        self.canvas_history = history
        self.self_conditioning = self_conditioning

    def commit_canvas(self, eos_ids, pad_id: int) -> bool:
        # first eos kept, the rest becomes pad
        argmax = self.canvas_argmax
        tokens = argmax.tolist() if torch.is_tensor(argmax) else list(argmax)
        finished = False
        for i, t in enumerate(tokens):
            if t in eos_ids:
                tokens[i + 1 :] = [pad_id] * (len(tokens) - i - 1)
                finished = True
                break
        self.token_ids.extend(tokens)
        self.num_tokens += len(tokens)
        self.canvas = None
        self.canvas_argmax = None
        self.canvas_history = None
        self.self_conditioning = None
        return finished

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.token_ids if self.num_completion_tokens == 0 else self.last_tokens,
        )

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_tokens = state[-1]
