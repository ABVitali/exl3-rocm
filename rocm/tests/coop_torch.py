# torch process + ONE trivial cooperative launch (no EXL3 code). Does the process segfault at exit?
import torch, sys
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void coop_k(int* out) { auto g = cg::this_grid(); out[blockIdx.x] = 1; g.sync(); if (threadIdx.x == 0) out[blockIdx.x] += 1; }
void coop_launch(at::Tensor t) {
    int* p = t.data_ptr<int>(); void* args[] = {&p};
    auto e = hipLaunchCooperativeKernel((const void*)coop_k, dim3(8), dim3(256), args, 0, at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(e == hipSuccess, hipGetErrorString(e));
}
'''
m = load_inline(name="coop_torch", cpp_sources="#include <torch/extension.h>\nvoid coop_launch(at::Tensor t);", cuda_sources=src, functions=["coop_launch"], with_cuda=True,
                extra_cuda_cflags=["-std=c++20"], extra_include_paths=[], verbose=False)
t = torch.zeros(64, dtype=torch.int32, device="cuda")
mode = sys.argv[1] if len(sys.argv) > 1 else "coop"
if mode == "coop":
    m.coop_launch(t); torch.cuda.synchronize(); print("coop launch from torch ok:", t[:2].tolist())
else:
    print("no cooperative launch, torch only")
print("exiting")
