from __future__ import annotations
from typing import List
import time

import torch
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from dyllm.dllm import dLLM
from dyllm.sampling_params import SamplingParams
from dyllm.utils.transformers_compat import load_tokenizer


def _cut_on_first_stop(text: str, stops: list[str]) -> str:
    if not stops:
        return text
    cut = min([text.find(s) for s in stops if s in text] + [len(text)])
    return text[:cut]


@register_model("dyllm")
class DyLLMAdapter(LM):
    """
    Minimal lm-eval-harness adapter that implements generate_until using dLLM.
    """

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        max_new_toks: int = 256,
        tensor_parallel_size: int = 1,
        expert_parallel_size: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        ignore_eos: bool = False,
        trust_remote_code: bool = True,
        num_steps: int = 256,
        num_full_steps: int = 16,
        block_size: int = 32,
        threshold: float = 0.99,
        **kwargs,
    ):
        super().__init__()
        self._batch_size = int(batch_size)
        self._max_new_toks = int(max_new_toks)

        def to_float(x, default):
            if x is None or str(x) == "None":
                return default
            return float(x)

        self.temperature = to_float(temperature, None)
        self.top_p = to_float(top_p, 1.0)
        self.ignore_eos = ignore_eos
        self.num_steps = int(num_steps)
        self.num_full_steps = int(num_full_steps)
        self.block_size = int(block_size)
        self.threshold = float(threshold)
        trust_remote_code = trust_remote_code
        self.model_path = model_path

        # Tokenizer (CPU)
        self.tokenizer = load_tokenizer(
            model_path, trust_remote_code=trust_remote_code, local_files_only=True, use_fast=True
        )

        # Engine (GPU)
        self.engine = dLLM(
            model_path,
            threshold=threshold,
            enforce_eager=True,
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
        )
        self.is_instruct = "instruct" in model_path.lower()

    # ---- LM required properties ----
    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def eot_token_id(self) -> int:
        eid = self.tokenizer.eos_token_id
        return int(eid) if eid is not None else -1

    @property
    def max_gen_toks(self) -> int:
        return self._max_new_toks

    @property
    def max_length(self) -> int:
        mlen = getattr(self.tokenizer, "model_max_length", 4096)
        return int(mlen if mlen and mlen != int(1e30) else 4096)

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

    # ---- Token helpers (lm-eval uses these in some paths) ----
    def apply_chat_template(
        self,
        conversation,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ):
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    def tok_encode(self, s: str) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def tok_decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    # ---- Not needed for GSM8K tasks; leave unimplemented ----
    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not implemented for DyLLMAdapter")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling not implemented for DyLLMAdapter")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []

        total_time = 0.0
        for i in range(0, len(requests), self._batch_size):
            batch = requests[i : i + self._batch_size]

            # prompts = [inst.args[0] for inst in batch]
            prompts = []
            for inst in batch:
                raw_prompt = inst.args[0]

                # Check if this is a HumanEval task
                doc = getattr(inst, "doc", {})
                task_id = str(doc.get("task_id", "")).lower() if doc else ""
                is_humaneval = task_id.startswith("humaneval")

                # Apply chat template only for instruct models on non-HumanEval tasks
                if self.is_instruct and not is_humaneval:
                    messages = [{"role": "user", "content": raw_prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(formatted_prompt)
                else:
                    prompts.append(raw_prompt)

            batch_max_toks = self._max_new_toks

            sp = SamplingParams(
                max_new_tokens=batch_max_toks,
                temperature=self.temperature,
                top_p=self.top_p,
                steps=self.num_steps,
                num_full_steps=self.num_full_steps,
                block_size=self.block_size,
                ignore_eos=self.ignore_eos,
            )

            start_time = time.perf_counter_ns()
            outs = self.engine.generate(prompts, sp)  # [{"text": ..., "token_ids": ...}, ...]
            end_time = time.perf_counter_ns()
            total_time += (end_time - start_time) / 1e9

            for inst, o in zip(batch, outs):
                stops = inst.args[1].get("until", [])
                doc = getattr(inst, "doc", {})
                task_id = str(doc.get("task_id", "")).lower() if doc else ""

                if self.is_instruct and task_id.startswith("humaneval"):
                    stops = []
                all_stops = stops + ["<|eot_id|>", "<|endoftext|>", "</s>"]

                trimmed = _cut_on_first_stop(o["text"], all_stops).strip()
                results.append(trimmed)
                # print(f"Prompt: {inst.args[0]}\nGenerated: {trimmed}\n")
            print(f"time for batch {i // self._batch_size + 1}: {(end_time - start_time) / 1e9:.2f} seconds")
        print(f"Total generation time for all batches in generate_until: {total_time:.2f} seconds")
        return results


# --- prompt construction and thought-channel handling for the adapter below ---

THINK_MARKER = "<|think|>"
# thought channel renders as: <|channel>thought\n ... <channel|> (type is plain text)
THOUGHT_OPEN_CANDIDATES = ("<|channel>",)
THOUGHT_CLOSE_CANDIDATES = ("<channel|>",)


def build_prompt(tokenizer, user_prompt: str, thinking: bool = False) -> str:
    # enable_thinking is handled by the chat template itself (injects <|think|>)
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def resolve_thought_ids(tokenizer):
    """Return (open_id, close_id), or None if the vocab has no thought tokens."""
    unk = getattr(tokenizer, "unk_token_id", None)
    for open_tok, close_tok in zip(THOUGHT_OPEN_CANDIDATES, THOUGHT_CLOSE_CANDIDATES):
        open_id = tokenizer.convert_tokens_to_ids(open_tok)
        close_id = tokenizer.convert_tokens_to_ids(close_tok)
        if open_id is not None and close_id is not None and open_id != unk and close_id != unk:
            return open_id, close_id
    return None


def strip_thought_ids(ids, open_id, close_id):
    """Drop [open .. close] spans; an unclosed span drops through the end."""
    out = []
    in_thought = False
    for t in ids:
        if in_thought:
            if t == close_id:
                in_thought = False
        elif t == open_id:
            in_thought = True
        else:
            out.append(t)
    return out


@register_model("dyllm_diffusiongemma")
class DiffusionGemmaAdapter(LM):
    """lm-eval adapter for DiffusionGemma: chat-format ``generate_until``,
    optional thinking mode, thought channel stripped before scoring."""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        max_new_toks: int = 2048,
        thinking: bool = False,
        seed: int = 1234,
        trust_remote_code: bool = True,
        threshold=None,
        **kwargs,
    ):
        super().__init__()
        self._batch_size = int(batch_size)
        self._max_new_toks = int(max_new_toks)
        self.thinking = str(thinking).lower() in ("1", "true", "yes")
        self.model_path = model_path

        self.tokenizer = load_tokenizer(
            model_path, trust_remote_code=trust_remote_code, local_files_only=True, use_fast=True
        )
        self.engine = dLLM(
            model_path,
            enforce_eager=True,
            tensor_parallel_size=1,
            threshold=None if threshold in (None, "", "None") else float(threshold),
        )
        self.thought_ids = resolve_thought_ids(self.tokenizer)
        if self.thought_ids is None:
            print("WARNING: thought-channel tokens not found in vocab -- outputs will not be stripped")
        torch.manual_seed(int(seed))
        self._gen_tokens = 0
        self._gen_time = 0.0
        self._pad_id = int(self.tokenizer.pad_token_id or 0)
        # per-request committed tokens: thought channel included, pads excluded
        self.gen_token_counts: List[int] = []
        # steps/canvas and TPF (tech report Eq.10) read from the scheduler's counters
        self._gen_lens: List[int] = []  # per-request committed length incl. pads (canvas multiples)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def eot_token_id(self) -> int:
        eid = self.tokenizer.eos_token_id
        return int(eid) if eid is not None else -1

    @property
    def max_gen_toks(self) -> int:
        return self._max_new_toks

    @property
    def max_length(self) -> int:
        mlen = getattr(self.tokenizer, "model_max_length", 4096)
        return int(mlen if mlen and mlen != int(1e30) else 4096)

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

    def tok_encode(self, s: str) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def tok_decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def apply_chat_template(self, conversation, tokenize: bool = True, add_generation_prompt: bool = True):
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not implemented for DiffusionGemmaAdapter")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling not implemented for DiffusionGemmaAdapter")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        for i in range(0, len(requests), self._batch_size):
            batch = requests[i : i + self._batch_size]
            prompt_ids = [
                self.tokenizer.encode(
                    build_prompt(self.tokenizer, inst.args[0], thinking=self.thinking),
                    add_special_tokens=False,  # template already carries bos
                )
                for inst in batch
            ]
            sp = SamplingParams(max_new_tokens=self._max_new_toks)
            start = time.perf_counter()
            outs = self.engine.generate([list(p) for p in prompt_ids], sp)
            self._gen_time += time.perf_counter() - start

            for inst, p_ids, o in zip(batch, prompt_ids, outs):
                completion = o["token_ids"][len(p_ids) :]
                self._gen_tokens += len(completion)
                self._gen_lens.append(len(completion))
                self.gen_token_counts.append(sum(1 for t in completion if t != self._pad_id))
                # the model emits a thought channel even in non-thinking mode (empty)
                if self.thought_ids is not None:
                    completion = strip_thought_ids(completion, *self.thought_ids)
                text = self.tokenizer.decode(completion, skip_special_tokens=True)
                stops = inst.args[1].get("until", []) if len(inst.args) > 1 else []
                results.append(_cut_on_first_stop(text, stops).strip())
        if self._gen_time > 0:
            # non-pad count excludes the canvas tail after EOS (the paper's "tokens generated")
            nonpad = sum(self.gen_token_counts)
            print(
                f"[dyllm_diffusiongemma] {self._gen_tokens} tokens ({nonpad} non-pad) in {self._gen_time:.1f}s "
                f"-> {self._gen_tokens / self._gen_time:.1f} TPS ({nonpad / self._gen_time:.1f} non-pad TPS)"
            )
            canvases = max(1, self._gen_tokens / 256)
            k_minus_1 = sum(max(0, -(-t // 256) - 1) for t in self._gen_lens)
            steps = self.engine.scheduler.num_canvas_steps
            print(
                f"[dyllm_diffusiongemma] steps/canvas {steps / canvases:.1f} | forwards {steps + k_minus_1} "
                f"| TPF {self._gen_tokens / max(1, steps + k_minus_1):.1f} | rounds {self.engine.scheduler.num_rounds}"
            )
        return results
