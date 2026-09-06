import os
from dataclasses import dataclass, field

from transformers import AutoConfig

from dyllm.configs import DiffusionGemmaConfig  # noqa: F401
from dyllm.configs.diffusion_gemma_runtime import DiffusionGemmaRuntimeConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs: int = 16
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    mask_id: int = -1
    num_full_steps: int = 8
    threshold: float = 0.99
    dist_port: int | None = None
    dtype: str = "bfloat16"
    shm_name: str | None = field(default=None, init=False, repr=False)
    diffusiongemma: DiffusionGemmaRuntimeConfig | None = field(default=None, init=False)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert 1 <= self.tensor_parallel_size <= 8
        if self.expert_parallel_size < 1:
            raise ValueError("expert_parallel_size must be at least 1")
        if self.expert_parallel_size not in (1, self.tensor_parallel_size):
            raise ValueError(
                "expert_parallel_size must be 1 or equal tensor_parallel_size; "
                "DyLLM currently has no separate data-parallel dimension"
            )
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        num_experts = int(getattr(self.hf_config, "num_experts", 0) or 0)
        if self.expert_parallel_size > 1:
            if num_experts == 0:
                raise ValueError("expert_parallel_size > 1 is only valid for MoE models")
            if num_experts % self.expert_parallel_size != 0:
                raise ValueError(
                    f"num_experts ({num_experts}) must be divisible by "
                    f"expert_parallel_size ({self.expert_parallel_size})"
                )
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
        if self.hf_config.model_type == "diffusion_gemma":
            if self.tensor_parallel_size != 1:
                raise ValueError("DiffusionGemma currently requires tensor_parallel_size == 1")
            self.diffusiongemma = DiffusionGemmaRuntimeConfig.from_pretrained(self.model, self.hf_config)
