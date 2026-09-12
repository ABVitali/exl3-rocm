"""ROCm build of the vllm-exl3 native plugin (`vllm_exl3_c`): p2b fused MoE (FMA tile),
batched GEMV (K=2 m==1), CPU reference dequant. No fat GEMM on ROCm.

    cd rocm && python setup_vllm_exl3_c.py develop
"""
import os
from pathlib import Path
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent
CSRC = HERE / "vllm_exl3_csrc"
EXT = HERE / "exllamav3_ext"
assert torch.version.hip, "this setup is for ROCm PyTorch (torch.version.hip)"
gpu_arch = os.environ.get("EXL3_ROCM_ARCH", "gfx942")
# e.g. EXL3_ROCM_EXTRA_FLAGS="-mwavefrontsize64" to force the 64-lane layout on RDNA3 test boxes
extra_flags = os.environ.get("EXL3_ROCM_EXTRA_FLAGS", "").split()

setup(
    name="vllm-exl3-c-rocm",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="vllm_exl3_c",
            sources=[str(CSRC / "bindings.cpp"), str(CSRC / "p2b_moe.cu"), str(CSRC / "p2b_batched.cu")],
            include_dirs=[str(CSRC), str(EXT), str(EXT / "quant")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DEXL3_ROCM=1"],
                "nvcc": ["-O3", "-std=c++20", "-DEXL3_ROCM=1", f"--offload-arch={gpu_arch}",
                         "-Wno-unused-result", "-Wno-unused-value"] + extra_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
