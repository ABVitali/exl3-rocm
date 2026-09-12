// Copied from Mia's AI Lab, GLM-5.3-Flash-EXL3-2x-DGX-Sparks,
// overlay/exl3_fat_gemm.cuh at commit 4b8d3c7 ("perf(exl3): accelerate fat-expert prefill").
// https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
//
// Copyright (c) 2026 Mia's AI Lab. Licensed under the MIT License.
// Full licence text: see THIRD_PARTY_NOTICES.md in the root of this repository.
//
// This file is unmodified from the original.

#pragma once

#include <torch/extension.h>

void exl3_fat_gemm(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_gemm_scatter(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    int64_t K,
    bool mcg,
    bool mul1);
