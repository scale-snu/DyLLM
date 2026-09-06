"""Minimal stand-ins for the vLLM internals the unified attention kernel imports.
Probe-only; the kernel files themselves are unmodified vLLM sources (Apache-2.0)."""
import enum
import torch
import triton
import triton.language as tl


class _Envs:
    VLLM_BATCH_INVARIANT = False


class _Platform:
    @staticmethod
    def fp8_dtype():
        return torch.float8_e4m3fn

    @staticmethod
    def is_device_capability_family(major):
        return torch.cuda.get_device_capability()[0] == major // 10


class KVQuantMode(enum.IntEnum):
    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    INT4_PER_TOKEN_HEAD = 4


envs = _Envs()
current_platform = _Platform()


def init_logger(name):
    import logging
    return logging.getLogger(name)
