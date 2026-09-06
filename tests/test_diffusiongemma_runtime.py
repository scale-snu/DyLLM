import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dyllm.configs.diffusion_gemma_runtime import DiffusionGemmaRuntimeConfig
from dyllm.engine.diffusiongemma import (
    DiffusionGemmaScheduler,
    DiffusionGemmaSequence,
)
from dyllm.engine.llm_engine import DLLMEngine
from dyllm.sampling_params import SamplingParams
from dyllm.utils.metadata import reset_metadata
from dyllm.utils.transformers_compat import load_tokenizer


def runtime_config(**overrides):
    values = {
        "canvas_length": 4,
        "vocab_size": 16,
        "hidden_size": 8,
        "eos_ids": frozenset({9}),
        "entropy_bound": 0.1,
        "t_max": 0.8,
        "t_min": 0.4,
        "max_denoising_steps": 2,
        "convergence_threshold": 0.005,
        "stability_threshold": 1,
        "pad_id": 0,
    }
    values.update(overrides)
    return DiffusionGemmaRuntimeConfig(**values)


class DiffusionGemmaRuntimeTest(unittest.TestCase):
    def setUp(self):
        reset_metadata()

    def test_sequence_uses_request_overrides_and_caps_final_canvas(self):
        params = SamplingParams(max_new_tokens=3, entropy_bound=0.2)
        seq = DiffusionGemmaSequence([1, 2], params, runtime_config(), device="cpu")

        self.assertEqual(seq.entropy_bound, 0.2)
        self.assertEqual(seq.t_min, 0.4)
        seq.canvas_argmax = torch.tensor([3, 9, 7, 8])

        self.assertTrue(seq.commit_canvas(frozenset({9}), pad_id=0))
        self.assertEqual(seq.token_ids, [1, 2, 3, 9, 0])
        self.assertEqual(seq.num_completion_tokens, 3)

    def test_scheduler_keeps_diffusion_policy_out_of_base_scheduler(self):
        runtime = runtime_config()
        config = SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=8,
            eos=-1,
            mask_id=None,
            diffusiongemma=runtime,
        )
        scheduler = DiffusionGemmaScheduler(config)
        seq = DiffusionGemmaSequence([1, 2], SamplingParams(max_new_tokens=3), runtime, device="cpu")
        scheduler.add(seq)

        scheduled, modes = scheduler.schedule()
        self.assertEqual(scheduled, [seq])
        self.assertEqual(modes, [True])
        scheduler.postprocess(scheduled, {"encoded": [seq.seq_id]})

        scheduled, modes = scheduler.schedule()
        self.assertEqual(modes, [False])
        self.assertEqual(seq.processed_steps, 1)
        result = {
            "denoise_seqs": [seq],
            "canvas": torch.tensor([[3, 9, 7, 8]]),
            "argmax": torch.tensor([[3, 9, 7, 8]]),
            "stop": torch.tensor([True]),
            "history": torch.tensor([[[3, 9, 7, 8]]]),
            "self_conditioning": torch.zeros(1, 4, runtime.hidden_size),
        }
        scheduler.postprocess(scheduled, result)

        self.assertTrue(seq.is_finished)
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(seq.token_ids[-3:], [3, 9, 0])

    def test_engine_sequence_factory_does_not_mutate_caller_tokens(self):
        engine = object.__new__(DLLMEngine)
        engine.mask = None
        engine.is_diffusiongemma = True
        engine._sequence_device = "cpu"
        engine.config = SimpleNamespace(diffusiongemma=runtime_config())
        prompt = [1, 2]

        seq = engine._create_sequence(prompt, SamplingParams(max_new_tokens=3))

        self.assertEqual(prompt, [1, 2])
        self.assertIsInstance(seq, DiffusionGemmaSequence)
        self.assertEqual(seq.token_ids, prompt)

    def test_tokenizer_loader_normalizes_transformers_4_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "tokenizer_config.json"
            config_path.write_text(
                json.dumps({"extra_special_tokens": ["<|video|>"]}),
                encoding="utf-8",
            )
            with patch("dyllm.utils.transformers_compat.AutoTokenizer.from_pretrained") as from_pretrained:
                load_tokenizer(directory, use_fast=True)

        kwargs = from_pretrained.call_args.kwargs
        self.assertEqual(
            kwargs["extra_special_tokens"],
            {"extra_special_token_0": "<|video|>"},
        )


if __name__ == "__main__":
    unittest.main()
