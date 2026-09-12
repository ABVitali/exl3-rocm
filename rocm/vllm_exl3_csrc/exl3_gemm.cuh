#pragma once

#include <ATen/Tensor.h>

at::Tensor exl3_gemm_cuda(const at::Tensor& x, const at::Tensor& trellis,
                          const at::Tensor& suh, const at::Tensor& svh,
                          int64_t bits, bool mcg);
