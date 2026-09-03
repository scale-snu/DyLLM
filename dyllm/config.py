import os
import json
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs: int = 16
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    mask_id: int = -1
    num_full_steps: int = 8
    threshold: float = 0.99
    dist_port: int | None = None
    # --- everything below is DiffusionGemma-only (the mask-based models ignore it) ---
    canvas_length: int | None = None
    eos_ids: set | None = None
    entropy_bound: float | None = None
    t_max: float | None = None
    t_min: float | None = None
    max_denoising_steps: int | None = None
    convergence_threshold: float | None = None
    stability_threshold: int | None = None
    pad_id: int | None = None
    vocab_size: int | None = None
    hidden_size: int | None = None
    dtype: str = "bfloat16"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.mask_id = getattr(self.hf_config, "mask_token_id", None)
        if hasattr(self.hf_config, "max_position_embeddings"):
            max_context_length = self.hf_config.max_position_embeddings
        elif hasattr(self.hf_config, "max_sequence_length"):
            max_context_length = self.hf_config.max_sequence_length
        else:
            text_config = getattr(self.hf_config, "text_config", None)
            max_context_length = getattr(text_config, "max_position_embeddings", self.max_model_len)
        self.max_model_len = min(self.max_model_len, max_context_length)
        assert self.max_num_batched_tokens >= self.max_model_len
        self.canvas_length = getattr(self.hf_config, "canvas_length", None)
        text_config = getattr(self.hf_config, "text_config", None) or self.hf_config
        self.vocab_size = getattr(text_config, "vocab_size", None)
        self.hidden_size = getattr(text_config, "hidden_size", None)
        gen_config_path = os.path.join(self.model, "generation_config.json")
        if os.path.isfile(gen_config_path):
            with open(gen_config_path) as f:
                gen_config = json.load(f)
            eos = gen_config.get("eos_token_id")
            if isinstance(eos, int):
                eos = [eos]
            if eos:
                self.eos_ids = set(eos)
            self.entropy_bound = (gen_config.get("sampler_config") or {}).get("entropy_bound")
            self.t_max = gen_config.get("t_max")
            self.t_min = gen_config.get("t_min")
            self.max_denoising_steps = gen_config.get("max_denoising_steps")
            self.convergence_threshold = gen_config.get("confidence_threshold")
            self.stability_threshold = gen_config.get("stability_threshold")
            self.pad_id = gen_config.get("pad_token_id")
