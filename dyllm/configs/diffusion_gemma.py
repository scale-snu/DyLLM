# coding=utf-8
# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DiffusionGemma model configuration"""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class DiffusionGemmaTextConfig(PretrainedConfig):
    model_type = "diffusion_gemma_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=262_144,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=30,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=131_072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=512,
        layer_types=None,
        final_logit_softcapping=30.0,
        use_bidirectional_attention=None,
        num_experts=None,
        top_k_experts=None,
        moe_intermediate_size=None,
        global_head_dim=512,
        num_global_key_value_heads=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.final_logit_softcapping = final_logit_softcapping
        self.use_bidirectional_attention = use_bidirectional_attention
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.global_head_dim = global_head_dim
        self.num_global_key_value_heads = num_global_key_value_heads

        if self.use_bidirectional_attention == "all":
            self.is_causal = False
            self.sliding_window = (self.sliding_window // 2) + 1

        if self.layer_types is None:
            sliding_window_pattern = 6
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.layer_types and self.layer_types[-1] != "full_attention":
            logger.warning(
                f"Last layer must use `full_attention`, but got `{self.layer_types[-1]}`. "
                "Forcing last layer to `full_attention`."
            )
            self.layer_types[-1] = "full_attention"

        if rope_parameters is None:
            rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
            }
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class DiffusionGemmaConfig(PretrainedConfig):
    model_type = "diffusion_gemma"
    sub_configs = {"text_config": DiffusionGemmaTextConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        boi_token_id=255_999,
        eoi_token_id=258_882,
        image_token_id=258_880,
        initializer_range=0.02,
        tie_word_embeddings=True,
        canvas_length=256,
        **kwargs,
    ):
        if text_config is None:
            text_config = DiffusionGemmaTextConfig()
            logger.info("text_config is None. Using default DiffusionGemmaTextConfig.")
        elif isinstance(text_config, dict):
            text_config = DiffusionGemmaTextConfig(**text_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.initializer_range = initializer_range
        self.canvas_length = canvas_length
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


try:
    AutoConfig.register("diffusion_gemma", DiffusionGemmaConfig)
except ValueError:
    pass
