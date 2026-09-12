// ROCm implementation of the BC_Linear* runners declared in linear.h.
//
// The CUDA build routes BC_LinearEXL3 through the cooperative tensor-core GEMM
// (exl3_gemm). That kernel is inline PTX and is not ported. On ROCm the same
// math runs as: input Hadamard with the sign vector, reconstruct the trellis to
// fp16 (W_hat), hipBLAS GEMM, output Hadamard with the sign vector, bias. This
// is exactly LinearEXL3.reconstruct_hgemm's non-fused path in Python, so the
// result is the original-basis product x @ W. Slow for tiny m (it materialises
// the whole weight), which is fine: decode goes through the plugin's p2b path.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "linear.h"
#include "../hgemm.cuh"
#include "../quant/reconstruct.cuh"
#include "../quant/hadamard.cuh"

void BC_LinearFP16::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    TORCH_CHECK(!graph, "ROCm port: BC_LinearFP16 graph capture not supported");
    hgemm(x, weight, y);
    if (bias) y.add_(bias.value());
}

void BC_LinearFP16::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}

void BC_LinearEXL3::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    TORCH_CHECK(!graph, "ROCm port: BC_LinearEXL3 graph capture not supported");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "BC_LinearEXL3: x and y must be 2-D");
    const int64_t k = trellis.size(0) * 16;
    const int64_t n = trellis.size(1) * 16;
    TORCH_CHECK(x.size(1) == k, "BC_LinearEXL3: in_features mismatch");
    TORCH_CHECK(y.size(1) == n, "BC_LinearEXL3: out_features mismatch");

    at::Tensor xh_ = at::empty_like(x);
    had_r_128(x, xh_, suh, c10::nullopt, 1.0f);

    at::Tensor w = at::empty({k, n}, x.options().dtype(at::kHalf));
    reconstruct(w, trellis, K, mcg, mul1);

    hgemm(xh_, w, y);
    had_r_128(y, y, c10::nullopt, svh, 1.0f);

    if (bias) y.add_(bias.value());
}

void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}

at::Tensor BC_LinearEXL3::run_alloc(const at::Tensor& x, int64_t out_features, bool output_fp32)
{
    std::vector<int64_t> out_shape = x.sizes().vec();
    out_shape.back() = out_features;
    at::Tensor y = at::empty(out_shape, x.options().dtype(output_fp32 ? at::kFloat : at::kHalf));
    if (out_features == 0) return y;
    at::Tensor x_flat = x.contiguous().view({-1, x.size(-1)});
    at::Tensor y_flat = y.view({-1, out_features});
    run(x_flat, y_flat);
    return y;
}
