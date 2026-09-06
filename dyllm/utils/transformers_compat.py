import json
import os

from transformers import AutoTokenizer


def load_tokenizer(model_path: str, **kwargs):
    """Load tokenizer metadata written by either Transformers 4 or 5.

    DiffusionGemma checkpoints use the Transformers 5 mapping form for model
    specific special tokens, while early checkpoints represented
    ``extra_special_tokens`` as a list. Transformers 4.57 assumes that field is
    a mapping and otherwise fails while constructing the fast tokenizer.
    """

    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.isfile(tokenizer_config_path):
        with open(tokenizer_config_path, encoding="utf-8") as file:
            tokenizer_config = json.load(file)
        extra_special_tokens = tokenizer_config.get("extra_special_tokens")
        if isinstance(extra_special_tokens, list):
            kwargs.setdefault(
                "extra_special_tokens",
                {f"extra_special_token_{index}": token for index, token in enumerate(extra_special_tokens)},
            )
    return AutoTokenizer.from_pretrained(model_path, **kwargs)
