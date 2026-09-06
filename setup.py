import os
import subprocess
from datetime import datetime
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
)


def get_arch_flags():
    # Supported deployment targets: A100, H100, and the generic fallback for
    # Blackwell until a dedicated SM100 kernel is added.
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0 9.0 10.0")
    flags = []
    for arch in arch_list.split():
        num = arch.replace(".", "") + ("a" if arch == "9.0" else "")
        flags += [f"-gencode=arch=compute_{num},code=sm_{num}"]
    return flags


def get_cutlass_include_dirs():
    """Get CUTLASS include directories from pip installation."""
    import sys

    site_packages = None
    for path in sys.path:
        if "site-packages" in path:
            site_packages = path
            break
    if site_packages:
        cutlass_include = os.path.join(site_packages, "cutlass_library", "source", "include")
        if os.path.exists(cutlass_include):
            return [cutlass_include]
    return []


def get_features_args():
    return []


def get_nvcc_thread_args():
    n = os.environ.get("NVCC_THREADS")
    return [f"-t{n}"] if n else []


cxx_args = ["-O3", "-std=c++17"]

nvcc_args = (
    [
        "-O3",
        "-std=c++17",
        "-DNDEBUG",
        "-D_USE_MATH_DEFINES",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]
    + get_features_args()
    + get_arch_flags()
    + get_nvcc_thread_args()
)

ext_modules = []

attn_src = "dyllm/csrc/attention"
cutlass_includes = get_cutlass_include_dirs()

attn_defines = ["-DDYLLM_D512_SECONDARY"]
attn_sources = [
    os.path.join(attn_src, "attention.cpp"),
    os.path.join(attn_src, "attention_aux_kernels.cu"),
    os.path.join(attn_src, "attention_ops_kernels_sm80.cu"),
    os.path.join(attn_src, "attention_ops_kernels_sm80_d512.cu"),
]

if any("compute_90a" in f for f in get_arch_flags()):
    attn_sources += [
        os.path.join(attn_src, "attention_ops_kernels_sm90.cu"),
        os.path.join(attn_src, "attention_ops_kernels_sm90_d512.cu"),
    ]
else:
    attn_defines.append("-DDYLLM_NO_H100")

ext_modules.append(
    CUDAExtension(
        name="dyllm.attention_ops",
        sources=attn_sources,
        include_dirs=[attn_src] + cutlass_includes,
        extra_compile_args={"cxx": cxx_args + attn_defines, "nvcc": nvcc_args + attn_defines},
    )
)

cache_src = "dyllm/csrc/cache"
ext_modules.append(
    CUDAExtension(
        name="dyllm.cache",
        sources=[
            os.path.join(cache_src, "cache_kernels.cu"),
            os.path.join(cache_src, "cache.cpp"),
        ],
        include_dirs=[cache_src],
        extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
    )
)

ops_src = "dyllm/csrc/custom_ops"
ext_modules.append(
    CUDAExtension(
        name="dyllm.custom_ops",
        sources=[
            os.path.join(ops_src, "pos_encoding_kernel.cu"),
            os.path.join(ops_src, "layernorm_kernel.cu"),
            os.path.join(ops_src, "custom_ops.cpp"),
        ],
        include_dirs=[ops_src],
        extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
    )
)

try:
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    rev = "+" + subprocess.check_output(cmd).decode("ascii").rstrip()
except Exception:
    now = datetime.now()
    rev = "+" + now.strftime("%Y-%m-%d-%H-%M-%S")

setup(
    name="dyllm",
    version="0.1.0" + rev,
    description="DyLLM CUDA extensions",
    packages=find_packages(include=["dyllm*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
