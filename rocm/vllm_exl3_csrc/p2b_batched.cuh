#pragma once
#include <ATen/Tensor.h>

at::Tensor p2b_gemv_batched_cuda(const at::Tensor& x,
                                 const at::Tensor& trellis_ptrs,
                                 const at::Tensor& suh_ptrs,
                                 const at::Tensor& svh_ptrs,
                                 const at::Tensor& expert_indices,
                                 int64_t bits, bool mcg, int64_t mmode = 1);
