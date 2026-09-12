"""ROCm build of the exllamav3_ext subset (see bindings_rocm.cpp).

Installs a top-level module named `exllamav3_ext`, which exllamav3/ext.py picks up via
is_precompiled_extension_available() instead of JIT-compiling the full CUDA tree.

    cd rocm && pip install -e . --no-build-isolation   (or: python setup_exllamav3_ext.py develop)
"""
import os
from pathlib import Path
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent / "exllamav3_ext"
assert torch.version.hip, "this setup is for ROCm PyTorch (torch.version.hip)"

sources = [
    "bindings_rocm.cpp",
    "hgemm.cu",
    "quant/reconstruct.cu",
    "quant/hadamard.cu",
    "quant/exl3_devctx.cu",
    "quant/exl3_moe_rocm.cu",
    "libtorch/linear_rocm.cpp",
]
gpu_arch = os.environ.get("EXL3_ROCM_ARCH", "gfx942")
# e.g. EXL3_ROCM_EXTRA_FLAGS="-mwavefrontsize64" to force the 64-lane layout on RDNA3 test boxes
extra_flags = os.environ.get("EXL3_ROCM_EXTRA_FLAGS", "").split()   # MI300X dev box; gfx950 for MI355X

setup(
    name="exllamav3-ext-rocm",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="exllamav3_ext",
            sources=[str(ROOT / s) for s in sources],
            include_dirs=[str(ROOT), str(ROOT / "quant"), str(ROOT / "libtorch")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DEXL3_ROCM=1"],
                "nvcc": ["-O3", "-std=c++20", "-DEXL3_ROCM=1", "-DHIPBLAS_USE_HIP_HALF", f"--offload-arch={gpu_arch}",
                         "-Wno-unused-result", "-Wno-unused-value"] + extra_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
