#pragma once

#include <ATen/Tensor.h>

// Small-m native EXL3 GEMV entry point.  The implementation keeps the exact
// trellis decoder shared with dequant_trellis and launches a bounded CUDA
// dot-product grid for m <= 8.
at::Tensor exl3_gemv_cuda(const at::Tensor& x, const at::Tensor& trellis,
                          const at::Tensor& suh, const at::Tensor& svh,
                          int64_t bits, bool mcg, int64_t mmode = 1);
