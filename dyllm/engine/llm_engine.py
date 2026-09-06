import atexit
import torch.multiprocessing as mp
import os
import threading
import queue
import time
import socket

from dyllm.config import Config
from dyllm.engine.model_runner import ModelRunner
from dyllm.engine.sequence import Sequence
from dyllm.engine.scheduler import Scheduler
from dyllm.engine.diffusiongemma import (
    DiffusionGemmaScheduler,
    DiffusionGemmaSequence,
)
from dyllm.sampling_params import SamplingParams
from dyllm.utils.metadata import get_metadata
from dyllm.utils.transformers_compat import load_tokenizer


class DLLMEngine:
    def __init__(self, model, threshold, **kwargs):
        config = Config(model, **kwargs)
        config.threshold = threshold
        self.config = config
        self.is_diffusiongemma = config.hf_config.model_type == "diffusion_gemma"
        self._sequence_device = "cuda"
        self._closed = False
        env_dist_port = os.environ.get("DYLLM_DIST_PORT")
        if env_dist_port is not None:
            config.dist_port = int(env_dist_port)
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                config.dist_port = s.getsockname()[1]
        if config.tensor_parallel_size > 1:
            config.shm_name = f"dyllm-{os.getpid()}-{config.dist_port}"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = load_tokenizer(
            config.model,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        config.eos = self.tokenizer.eos_token_id
        self.mask = config.mask_id

        ctx = mp.get_context("spawn")
        self.ps = []
        self.events = []
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        try:
            self.model_runner = ModelRunner(config, 0, self.events)
        except BaseException:
            for process in self.ps:
                if process.is_alive():
                    process.terminate()
                process.join()
            raise
        scheduler_cls = DiffusionGemmaScheduler if self.is_diffusiongemma else Scheduler
        self.scheduler = scheduler_cls(config)

        atexit.register(self.exit)

        self._in_q: queue.Queue | None = None
        self._out_q: queue.Queue | None = None
        self._stop_event: threading.Event | None = None
        self._worker: threading.Thread | None = None

    def exit(self):
        if self._closed:
            return
        self._closed = True
        self.stop_async()
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            model_runner.call("exit")
            del self.model_runner
        for process in getattr(self, "ps", []):
            process.join()

    def start_async(self):
        """Start background worker; safe to call multiple times."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._in_q = queue.Queue()
        self._out_q = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._async_worker, name="dllm-engine-worker", daemon=True)
        self._worker.start()

    def stop_async(self, timeout: float = 5.0):
        if self._worker is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=timeout)
        self._worker = None
        self._in_q = None
        self._out_q = None
        self._stop_event = None

    def add_request_async(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Enqueue a request and return immediately with seq_id."""
        if self._worker is None:
            raise RuntimeError("Async worker not started")

        seq = self._create_sequence(prompt, sampling_params)

        self._in_q.put(seq)
        return seq.seq_id

    asnync_add_request = add_request_async

    def poll_finished(self, max_items: int | None = None) -> list[dict]:
        if self._out_q is None:
            return []
        items = []
        n = 0
        while True:
            if max_items is not None and n >= max_items:
                break
            try:
                seq_id, token_ids = self._out_q.get_nowait()
            except queue.Empty:
                break
            items.append(
                {
                    "seq_id": seq_id,
                    "token_ids": token_ids,
                    "text": self.tokenizer.decode(token_ids),
                }
            )
            n += 1
        return items

    def wait_finished(self, seq_ids: list[int], timeout: float | None = None) -> list[dict]:
        if self._out_q is None:
            return []
        want = set(seq_ids)
        got: dict[int, list[int]] = {}
        start_t = time.monotonic()
        while want:
            remaining = None
            if timeout is not None:
                elapsed = time.monotonic() - start_t
                remaining = max(0.0, timeout - elapsed)
                if remaining == 0.0:
                    break
            try:
                sid, token_ids = self._out_q.get(timeout=remaining)
            except queue.Empty:
                break
            if sid in want:
                got[sid] = token_ids
                want.remove(sid)

        return [
            {
                "seq_id": sid,
                "token_ids": got[sid],
                "text": self.tokenizer.decode(got[sid]),
            }
            for sid in sorted(got.keys())
        ]

    def _drain_requests(self) -> int:
        if self._in_q is None:
            return 0
        metadata = get_metadata()
        drained = 0
        while True:
            try:
                seq: Sequence = self._in_q.get_nowait()
            except queue.Empty:
                break
            if seq.seq_id not in metadata.all_seqs:
                metadata.all_seqs.append(seq.seq_id)
            self.scheduler.add(seq)
            drained += 1
        return drained

    def _async_worker(self):
        idle_backoff = 0
        # simpler stop condition
        while self._stop_event is not None and not self._stop_event.is_set():
            added = self._drain_requests()
            did_step = False
            if not self.scheduler.is_finished():
                output, _ = self.step()
                for seq_id, token_ids in output:
                    if self._out_q is not None:
                        self._out_q.put((seq_id, token_ids))
                did_step = True

            if not did_step and added == 0:
                idle_backoff = min(idle_backoff + 1, 6)
                time.sleep(0.001 * (2**idle_backoff))
            else:
                idle_backoff = 0

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        metadata = get_metadata()
        seq = self._create_sequence(prompt, sampling_params)
        if seq.seq_id not in metadata.all_seqs:
            metadata.all_seqs.append(seq.seq_id)
        self.scheduler.add(seq)

    def _create_sequence(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = list(prompt)
        if sampling_params.input_len is not None:
            if not token_ids:
                raise ValueError("input_len cannot be applied to an empty prompt")
            if len(token_ids) > sampling_params.input_len:
                token_ids = token_ids[: sampling_params.input_len]
            else:
                token_ids += [token_ids[-1]] * (sampling_params.input_len - len(token_ids))
        if self.mask is not None:
            token_ids += [self.mask] * sampling_params.max_new_tokens

        sampling_params.mask_id = self.mask
        if self.is_diffusiongemma:
            return DiffusionGemmaSequence(
                token_ids,
                sampling_params,
                self.config.diffusiongemma,
                device=self._sequence_device,
            )
        return Sequence(token_ids, sampling_params)

    def is_finished(self):
        return self.scheduler.is_finished()

    @property
    def execution_stats(self) -> dict[str, int]:
        return {
            "full_forwards": self.model_runner.num_full_forwards,
            "sparse_forwards": self.model_runner.num_sparse_forwards,
        }

    def step(self):
        seqs, is_full = self.scheduler.schedule()
        finished_seqs = list(get_metadata().finished_seqs)
        if self.is_diffusiongemma:
            result = self.model_runner.call("run", seqs, is_full, finished_seqs)
            self.scheduler.postprocess(seqs, result)
        else:
            selected_positions, selected_tokens, selected_counts = self.model_runner.call(
                "run", seqs, is_full, finished_seqs
            )
            self.scheduler.postprocess(seqs, selected_positions, selected_tokens, selected_counts)
        outputs = [(seq.seq_id, seq.token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if (is_full is True) else -len(seqs)
        return outputs, num_tokens

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        while not self.is_finished():
            output, _ = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        if len(token_ids[-sampling_params[0].max_new_tokens :]) == 0:
            return [{"text": "", "token_ids": []} for _ in outputs]
        outputs = [
            {
                "text": self.tokenizer.decode(
                    token_ids[-sampling_params[0].max_new_tokens :], skip_special_tokens=True
                ),
                "token_ids": token_ids,
            }
            for token_ids in outputs
        ]

        return outputs

    def generate_async(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        timeout: float | None = None,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        seq_ids = []
        for prompt, sp in zip(prompts, sampling_params):
            seq_ids.append(self.asnync_add_request(prompt, sp))

        return self.wait_finished(seq_ids, timeout=timeout)
