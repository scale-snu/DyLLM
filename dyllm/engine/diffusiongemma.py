import torch

from dyllm.config import Config
from dyllm.configs.diffusion_gemma_runtime import DiffusionGemmaRuntimeConfig
from dyllm.engine.scheduler import Scheduler
from dyllm.engine.sequence import Sequence, SequenceStatus
from dyllm.sampling_params import SamplingParams
from dyllm.utils.metadata import get_metadata


class DiffusionGemmaSequence(Sequence):
    """Per-request state for DiffusionGemma canvas denoising."""

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams,
        runtime_config: DiffusionGemmaRuntimeConfig,
        device: str = "cuda",
    ):
        super().__init__(token_ids, sampling_params)
        self.entropy_bound = self._setting(sampling_params.entropy_bound, runtime_config.entropy_bound)
        self.t_min = self._setting(sampling_params.t_min, runtime_config.t_min)
        self.t_max = self._setting(sampling_params.t_max, runtime_config.t_max)
        self.max_denoising_steps = self._setting(
            sampling_params.max_denoising_steps,
            runtime_config.max_denoising_steps,
        )
        self.convergence_threshold = self._setting(
            sampling_params.convergence_threshold,
            runtime_config.convergence_threshold,
        )
        self.stability_threshold = self._setting(
            sampling_params.stability_threshold,
            runtime_config.stability_threshold,
        )
        self.canvas = None
        self.canvas_argmax = None
        self.canvas_history = None
        self.self_conditioning = None
        self.canvas_start_step = 0
        self.encoded_len = 0
        self.open_canvas(
            runtime_config.vocab_size,
            runtime_config.canvas_length,
            device=device,
        )

    @staticmethod
    def _setting(request_value, checkpoint_value):
        return checkpoint_value if request_value is None else request_value

    @property
    def cur_step(self):
        return self.max_denoising_steps - self.canvas_step + 1

    @property
    def canvas_step(self):
        return self.processed_steps - self.canvas_start_step

    @property
    def needs_encode(self):
        return self.encoded_len < len(self.token_ids)

    def mark_encoded(self):
        self.encoded_len = len(self.token_ids)

    def open_canvas(self, vocab_size: int, canvas_length: int, device="cpu"):
        # Keep canvas initialization on the CPU so seeded runs preserve the
        # reference implementation's RNG stream.
        self.canvas = torch.randint(0, vocab_size, (canvas_length,)).to(device)
        self.canvas_argmax = self.canvas.clone()
        self.canvas_history = torch.full(
            (self.stability_threshold, canvas_length),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.self_conditioning = None
        self.canvas_start_step = self.processed_steps

    def apply_step(
        self,
        canvas: torch.Tensor,
        argmax: torch.Tensor,
        history: torch.Tensor,
        self_conditioning: torch.Tensor,
    ):
        self.canvas = canvas
        self.canvas_argmax = argmax
        self.canvas_history = history
        self.self_conditioning = self_conditioning

    def commit_canvas(self, eos_ids: frozenset[int], pad_id: int) -> bool:
        tokens = self.canvas_argmax.tolist()
        remaining = self.max_new_tokens - self.num_completion_tokens
        tokens = tokens[:remaining]
        finished = False
        for index, token in enumerate(tokens):
            if token in eos_ids:
                tokens[index + 1 :] = [pad_id] * (len(tokens) - index - 1)
                finished = True
                break
        self.token_ids.extend(tokens)
        self.num_tokens += len(tokens)
        self.canvas = None
        self.canvas_argmax = None
        self.canvas_history = None
        self.self_conditioning = None
        return finished


class DiffusionGemmaScheduler(Scheduler):
    """Scheduler policy for causal prefix encoding and canvas denoising."""

    def __init__(self, config: Config):
        super().__init__(config)
        runtime_config = config.diffusiongemma
        if runtime_config is None:
            raise ValueError("DiffusionGemma scheduler requires runtime settings")
        self.runtime_config = runtime_config
        self.num_rounds = 0
        self.num_canvas_steps = 0

    def schedule(self) -> tuple[list[DiffusionGemmaSequence], list[bool]]:
        scheduled_seqs = []
        modes = []
        num_tokens = 0
        for seq in self.full:
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            if seq.needs_encode:
                rows = len(seq) - seq.encoded_len
                causal = True
            else:
                rows = self.runtime_config.canvas_length
                causal = False
            if num_tokens + rows > self.max_num_batched_tokens:
                break
            if not causal:
                seq.processed_steps += 1
            scheduled_seqs.append(seq)
            modes.append(causal)
            num_tokens += rows
        if not scheduled_seqs:
            raise RuntimeError(
                "No DiffusionGemma request fits max_num_batched_tokens; increase it "
                "to at least the prompt or canvas length"
            )
        self.num_rounds += 1
        self.num_canvas_steps += sum(not mode for mode in modes)
        return scheduled_seqs, modes

    def postprocess(self, seqs: list[DiffusionGemmaSequence], result: dict):
        encoded_ids = set(result.get("encoded", []))
        for seq in seqs:
            if seq.seq_id in encoded_ids:
                seq.mark_encoded()

        denoise_seqs = result.get("denoise_seqs")
        if not denoise_seqs:
            get_metadata().finished_seqs = []
            return

        finished = []
        canvas = result["canvas"]
        argmax = result["argmax"]
        stop = result["stop"].tolist()
        history = result["history"]
        self_conditioning = result["self_conditioning"]
        for index, seq in enumerate(denoise_seqs):
            last_step = seq.cur_step == 1
            seq.apply_step(
                canvas[index],
                argmax[index],
                history[:, index],
                self_conditioning[index],
            )
            if stop[index] or last_step:
                hit_eos = seq.commit_canvas(
                    self.runtime_config.eos_ids,
                    self.runtime_config.pad_id,
                )
                done = hit_eos and not seq.ignore_eos
                if done or seq.num_completion_tokens >= seq.max_new_tokens:
                    seq.status = SequenceStatus.FINISHED
                    finished.append(seq.seq_id)
                    self.full.remove(seq)
                else:
                    seq.open_canvas(
                        self.runtime_config.vocab_size,
                        self.runtime_config.canvas_length,
                        device=canvas.device,
                    )
        get_metadata().finished_seqs = finished
