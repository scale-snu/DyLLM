import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionGemmaRuntimeConfig:
    """Generation settings used by the DiffusionGemma engine."""

    canvas_length: int
    vocab_size: int
    hidden_size: int
    eos_ids: frozenset[int]
    entropy_bound: float
    t_max: float
    t_min: float
    max_denoising_steps: int
    convergence_threshold: float
    stability_threshold: int
    pad_id: int

    @classmethod
    def from_pretrained(cls, model_path: str, hf_config):
        generation_config_path = os.path.join(model_path, "generation_config.json")
        if not os.path.isfile(generation_config_path):
            raise ValueError("DiffusionGemma requires generation_config.json in the model directory")

        with open(generation_config_path, encoding="utf-8") as file:
            generation_config = json.load(file)

        text_config = getattr(hf_config, "text_config", None) or hf_config
        eos_ids = generation_config.get("eos_token_id")
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]

        values = {
            "canvas_length": getattr(hf_config, "canvas_length", None),
            "vocab_size": getattr(text_config, "vocab_size", None),
            "hidden_size": getattr(text_config, "hidden_size", None),
            "eos_ids": eos_ids,
            "entropy_bound": (generation_config.get("sampler_config") or {}).get("entropy_bound"),
            "t_max": generation_config.get("t_max"),
            "t_min": generation_config.get("t_min"),
            "max_denoising_steps": generation_config.get("max_denoising_steps"),
            "convergence_threshold": generation_config.get("confidence_threshold"),
            "stability_threshold": generation_config.get("stability_threshold"),
            "pad_id": generation_config.get("pad_token_id", 0),
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError("DiffusionGemma checkpoint is missing required generation settings: " + ", ".join(missing))

        values["eos_ids"] = frozenset(values["eos_ids"])
        return cls(**values)
