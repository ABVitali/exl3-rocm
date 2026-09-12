// ROCm implementation of exllamav3's batched MoE entry point `exl3_moe` (rows sorted by expert).
//
// The CUDA original is one cooperative kernel with tensor-core GEMMs and grid-wide barriers.
// Here the same contract is executed as six plain launches per call, with the M-row FMA tile
// (decode each trellis word once, apply it to up to MROWS activation rows). No cooperative
// launch: graph-capturable, and no CG hardware queue (the exit-time SIGSEGV path).
//
// Temp buffers are the plugin's [concurrency, TEMP_ROWS, dim] tensors, used FLAT: temp row =
// sorted row index. Capacity = concurrency * TEMP_ROWS >= token_sorted.numel() is required
// (exl3_moe_max_concurrency returns 8 -> 16384 rows, above the plugin's 2048-token chunking).
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../hip_compat.cuh"
#include "exl3_moe.cuh"
#include "exl3_moe_common.cuh"
#include "exl3_devctx.cuh"
#include "exl3_gemv_kernel.cuh"
#include "hadamard_inner.cuh"

namespace exl3_moe_rocm {

#ifndef EXL3_MOE_MROWS
#define EXL3_MOE_MROWS 8
#endif
constexpr int MROWS = EXL3_MOE_MROWS;    // activation rows per decoded tile (4 for RDNA3 forced wave64)
constexpr int WK = 16, WNT = 2, PF = 4;  // k-split warps, n-tiles per warp, prefetch depth
constexpr int THREADS = WK * 32, COLS = WNT * 16;
constexpr int MAX_CONCURRENCY = 8;

// 0) expert_start[E+1] = prefix sums of expert_count; row_expert[row] = expert (or -1 if the
//    expert exceeds max_tokens_per_expert and is left to the caller's fallback path)
__global__ void rows_kernel(const int64_t* __restrict__ expert_count, int num_experts, int total_rows,
                            int max_tokens, int* __restrict__ expert_start, int* __restrict__ row_expert)
{
    __shared__ int s_start[1025];
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < num_experts; ++e) { s_start[e] = acc; expert_start[e] = acc; acc += (int) expert_count[e]; }
        s_start[num_experts] = acc; expert_start[num_experts] = acc;
    }
    __syncthreads();
    for (int row = threadIdx.x; row < total_rows; row += blockDim.x) {
        if (row >= s_start[num_experts]) { row_expert[row] = -1; continue; }   // sentinel expert n_exp (fat / non-local routes, weight 0)
        int lo = 0, hi = num_experts;             // find e with s_start[e] <= row < s_start[e+1]
        while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (s_start[mid] <= row) lo = mid; else hi = mid; }
        int cnt = s_start[lo + 1] - s_start[lo];
        row_expert[row] = (cnt > 0 && cnt <= max_tokens) ? lo : -1;
    }
}

// 1) gather x[token] + input Hadamard (x suh) into temp_state_g / temp_state_u, one warp per 128 slice
__global__ __launch_bounds__(256) void gather_kernel(
    const half* __restrict__ hidden_state, half* __restrict__ temp_g, half* __restrict__ temp_u,
    const half** __restrict__ gate_suh, const half** __restrict__ up_suh,
    const int64_t* __restrict__ token_sorted, const int* __restrict__ row_expert,
    int hidden_dim, int total_rows, int gated)
{
    const int wpt = hidden_dim / 128;
    const int w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (w >= total_rows * wpt) return;
    const int row = w / wpt, off = w % wpt;
    const int e = row_expert[row];
    if (e < 0) return;
    const half* in = hidden_state + (size_t) token_sorted[row] * hidden_dim + off * 128;
    if (gated) had_hf_r_128_inner<true, false>(in, temp_g + (size_t) row * hidden_dim + off * 128, gate_suh[e] + off * 128, 0.088388347648f);
    had_hf_r_128_inner<true, false>(in, temp_u + (size_t) row * hidden_dim + off * 128, up_suh[e] + off * 128, 0.088388347648f);
}

// 2/4) batched GEMV: C[rows of expert e, group*32 .. +32) = A[rows] @ W_e, trellis decoded once per
//      MROWS rows. Block = 16 k-split warps x 32 lanes; grid = (experts, n-groups, matrices).
template <int bits, int cb>
__global__ __launch_bounds__(THREADS) void gemv_kernel(
    const half* __restrict__ A0, const half* __restrict__ A1, half* __restrict__ C0, half* __restrict__ C1,
    const uint16_t** __restrict__ trellis0, const uint16_t** __restrict__ trellis1,
    const int* __restrict__ expert_start, const int64_t* __restrict__ expert_count,
    int size_k, int size_n, int max_tokens)
{
    const int e = blockIdx.x, group = blockIdx.y, mat = blockIdx.z;
    const int start = expert_start[e];
    const int cnt = (int) expert_count[e];
    if (cnt == 0 || cnt > max_tokens) return;
    const half* A = mat ? A1 : A0;
    half* C = mat ? C1 : C0;
    const uint32_t* B32 = reinterpret_cast<const uint32_t*>((mat ? trellis1 : trellis0)[e]);

    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int kslices = size_k / 16, ntiles = size_n / 16;
    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) { int i1 = lane >> 1; x_src_b = i1; x_src_a = (i1 + 15) & 15; }
    else if constexpr (bits == 3) {
        int t_offset = lane << 3; int b1 = (t_offset + 257) * 3; int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32; int i2 = (b2 - 1) / 32; x_s2 = (i2 + 1) * 32 - b2; x_src_a = i0 % 24; x_src_b = i2 % 24;
    }
    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;
    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3) return lane < 24 ? EXL3_LDCS(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else return EXL3_LDCS(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };
    auto decode_tile = [&] (const uint32_t* bw, int t, FragB& f0, FragB& f1) {
        if constexpr (bits == 4) { uint32_t aw = EXL3_SHFL(bw[t], (lane + 31) & 31); exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1); }
        else if constexpr (bits == 2) { const uint32_t w = bw[t >> 1]; const int base = (t & 1) << 4;
            uint32_t bwv = EXL3_SHFL(w, base + x_src_b); uint32_t awv = EXL3_SHFL(w, base + x_src_a); exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1); }
        else { uint32_t awv = EXL3_SHFL(bw[t], x_src_a); uint32_t bwv = EXL3_SHFL(bw[t], x_src_b); exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1); }
    };

    __shared__ float sh_red[WK][MROWS][COLS];

    for (int r0 = 0; r0 < cnt; r0 += MROWS) {
        const int nrows = min(MROWS, cnt - r0);
        const half* Abase = A + (size_t) (start + r0) * size_k;

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d) if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);

        float acc[MROWS][WNT][2] = {};
        for (int ib = 0; ib < myn; ib += PF) {
            #pragma unroll
            for (int d = 0; d < PF; ++d) {
                const int i = ib + d;
                if (i >= myn) break;
                uint32_t bw[LOADS];
                #pragma unroll
                for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];
                if (i + PF < myn) {
                    #pragma unroll
                    for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
                }
                const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
                float2 bb[WNT][4];
                #pragma unroll
                for (int t = 0; t < WNT; ++t) {
                    FragB f0, f1;
                    decode_tile(bw, t, f0, f1);
                    bb[t][0] = __half22float2(f0[0]); bb[t][1] = __half22float2(f0[1]);
                    bb[t][2] = __half22float2(f1[0]); bb[t][3] = __half22float2(f1[1]);
                }
                #pragma unroll
                for (int r = 0; r < MROWS; ++r) {
                    if (r >= nrows) break;
                    const half2* Ar = reinterpret_cast<const half2*>(Abase + (size_t) r * size_k);
                    const float2 a0 = __half22float2(Ar[a_col]), a2 = __half22float2(Ar[a_col + 4]);
                    #pragma unroll
                    for (int t = 0; t < WNT; ++t) {
                        acc[r][t][0] += a0.x * bb[t][0].x + a0.y * bb[t][0].y + a2.x * bb[t][1].x + a2.y * bb[t][1].y;
                        acc[r][t][1] += a0.x * bb[t][2].x + a0.y * bb[t][2].y + a2.x * bb[t][3].x + a2.y * bb[t][3].y;
                    }
                }
            }
        }
        // reduce the 4 lanes sharing column n = lane / 4, then per-warp partials to shared memory
        #pragma unroll
        for (int r = 0; r < MROWS; ++r) {
            if (r >= nrows) break;
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
                #pragma unroll
                for (int f = 0; f < 2; ++f) { float v = acc[r][t][f]; v += EXL3_SHFL_XOR(v, 1); v += EXL3_SHFL_XOR(v, 2); acc[r][t][f] = v; }
        }
        if ((lane & 3) == 0) {
            const int n = lane >> 2;
            #pragma unroll
            for (int r = 0; r < nrows; ++r)
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) sh_red[warp][r][t * 16 + f * 8 + n] = acc[r][t][f];
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < nrows * COLS; idx += THREADS) {
            const int r = idx / COLS, col = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j) sum += sh_red[j][r][col];
            C[(size_t) (start + r0 + r) * size_n + group * COLS + col] = __float2half_rn(sum);
        }
        __syncthreads();
    }
}

// 3) output Hadamard for g,u (x svh) + activation + gate + input Hadamard for d (x down_suh); in place in temp_int_g
__global__ __launch_bounds__(256) void guad_kernel(
    half* __restrict__ temp_int_g, half* __restrict__ temp_int_u,
    const half** __restrict__ gate_svh, const half** __restrict__ up_svh, const half** __restrict__ down_suh,
    const int* __restrict__ row_expert, int inter_dim, int total_rows, float act_limit, int act_function)
{
    const int wpt = inter_dim / 128;
    const int w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (w >= total_rows * wpt) return;
    const int row = w / wpt, off = w % wpt;
    const int e = row_expert[row];
    if (e < 0) return;
    const size_t p = (size_t) row * inter_dim + off * 128;
    had_hf_r_128_guad_inner(temp_int_g + p, temp_int_u + p, temp_int_g + p,
                            gate_svh[e] + off * 128, up_svh[e] + off * 128, down_suh[e] + off * 128,
                            0.088388347648f, act_limit, act_function);
}

// 5) output Hadamard for d (x down_svh) scaled by the routing weight, atomically added into output_state[token]
__global__ __launch_bounds__(256) void d_out_kernel(
    const half* __restrict__ temp_g, float* __restrict__ output_state, const half** __restrict__ down_svh,
    const int64_t* __restrict__ token_sorted, const half* __restrict__ weight_sorted, const int* __restrict__ row_expert,
    int hidden_dim, int total_rows)
{
    const int wpt = hidden_dim / 128;
    const int w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (w >= total_rows * wpt) return;
    const int row = w / wpt, off = w % wpt;
    const int e = row_expert[row];
    if (e < 0) return;
    had_hf_r_128_d_inner(temp_g + (size_t) row * hidden_dim + off * 128,
                         output_state + (size_t) token_sorted[row] * hidden_dim + off * 128,
                         down_svh[e] + off * 128, 0.088388347648f * __half2float(weight_sorted[row]));
}

template <int bits, int cb>
static void launch_gemv(dim3 grid, cudaStream_t stream, const half* A0, const half* A1, half* C0, half* C1,
                        const uint16_t** t0, const uint16_t** t1, const int* expert_start, const int64_t* expert_count,
                        int size_k, int size_n, int max_tokens)
{
    gemv_kernel<bits, cb><<<grid, THREADS, 0, stream>>>(A0, A1, C0, C1, t0, t1, expert_start, expert_count, size_k, size_n, max_tokens);
}

using gemv_fn = void (*)(dim3, cudaStream_t, const half*, const half*, half*, half*, const uint16_t**, const uint16_t**,
                         const int*, const int64_t*, int, int, int);
static gemv_fn select_gemv(int K, bool mul1)
{
    if (K == 2) return mul1 ? launch_gemv<2, 2> : launch_gemv<2, 1>;
    if (K == 3) return mul1 ? launch_gemv<3, 2> : launch_gemv<3, 1>;
    if (K == 4) return mul1 ? launch_gemv<4, 2> : launch_gemv<4, 1>;
    return nullptr;
}

}  // namespace exl3_moe_rocm

int exl3_moe_max_concurrency(int device) { (void) device; return exl3_moe_rocm::MAX_CONCURRENCY; }

void exl3_moe
(
    const at::Tensor& hidden_state, const at::Tensor& output_state, const at::Tensor& expert_count,
    const at::Tensor& token_sorted, const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g, const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g, const at::Tensor& temp_intermediate_u,
    const int act_function, const int K_gate, const int K_up, const int K_down,
    const at::Tensor& gate_ptrs_trellis, const at::Tensor& gate_ptrs_suh, const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis, const at::Tensor& up_ptrs_suh, const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis, const at::Tensor& down_ptrs_suh, const at::Tensor& down_ptrs_svh,
    const bool gate_mcg, const bool gate_mul1, const bool up_mcg, const bool up_mul1,
    const bool down_mcg, const bool down_mul1, const float act_limit, const int num_active
)
{
    using namespace exl3_moe_rocm;
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (num_active == 0) return;
    TORCH_CHECK_DTYPE(hidden_state, kHalf);
    TORCH_CHECK_DIM(hidden_state, 2);
    const int bsz = hidden_state.size(0);
    const int hidden_dim = hidden_state.size(1);
    TORCH_CHECK_DTYPE(output_state, kFloat);
    TORCH_CHECK_SHAPES_FULL(output_state, hidden_state);
    TORCH_CHECK_DTYPE(expert_count, kLong);
    TORCH_CHECK_DIM(expert_count, 1);
    const int num_experts = expert_count.size(0) - 1;
    TORCH_CHECK(num_experts >= 1 && num_experts <= 1024, "exl3_moe (ROCm): 1..1024 experts");
    TORCH_CHECK_DTYPE(token_sorted, kLong);
    TORCH_CHECK_DIM(token_sorted, 1);
    TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);
    TORCH_CHECK_DTYPE(weight_sorted, kHalf);
    const int total_rows = token_sorted.size(0);
    TORCH_CHECK_DTYPE(temp_state_g, kHalf); TORCH_CHECK_DTYPE(temp_state_u, kHalf);
    TORCH_CHECK_DIM(temp_state_g, 3);
    TORCH_CHECK_SHAPES(temp_state_g, 2, hidden_state, 1, 1);
    TORCH_CHECK_SHAPES_FULL(temp_state_g, temp_state_u);
    const int max_tokens = temp_state_g.size(1);
    const int64_t capacity = (int64_t) temp_state_g.size(0) * max_tokens;
    TORCH_CHECK(total_rows <= capacity, "exl3_moe (ROCm): temp capacity ", capacity, " rows < ", total_rows,
                " sorted rows; raise concurrency (exl3_moe_max_concurrency) or chunk the batch");
    TORCH_CHECK_DTYPE(temp_intermediate_g, kHalf); TORCH_CHECK_DTYPE(temp_intermediate_u, kHalf);
    TORCH_CHECK_DIM(temp_intermediate_g, 3);
    TORCH_CHECK_SHAPES_FULL(temp_intermediate_g, temp_intermediate_u);
    TORCH_CHECK_SHAPES(temp_intermediate_g, 1, temp_state_g, 1, 1);
    const int inter_dim = temp_intermediate_g.size(2);
    TORCH_CHECK(hidden_dim % 128 == 0 && inter_dim % 128 == 0, "exl3_moe (ROCm): dims must be multiples of 128");
    TORCH_CHECK(gate_mcg == up_mcg && up_mcg == down_mcg && gate_mul1 == up_mul1 && up_mul1 == down_mul1,
                "exl3_moe: gate/up/down must share the same codebook");
    TORCH_CHECK(gate_mcg != gate_mul1, "exl3_moe: only mcg and mul1 codebooks are supported");
    TORCH_CHECK(K_gate == K_up, "exl3_moe (ROCm): gate/up must share K");
    gemv_fn gemv_gu = select_gemv(K_gate, gate_mul1), gemv_d = select_gemv(K_down, gate_mul1);
    TORCH_CHECK(gemv_gu != nullptr && gemv_d != nullptr, "exl3_moe (ROCm): K must be 2, 3 or 4 (got gu ", K_gate, ", down ", K_down, ")");
    for (const at::Tensor* p : {&gate_ptrs_trellis, &gate_ptrs_suh, &gate_ptrs_svh, &up_ptrs_trellis, &up_ptrs_suh,
                                &up_ptrs_svh, &down_ptrs_trellis, &down_ptrs_suh, &down_ptrs_svh})
        TORCH_CHECK(p->dim() == 1 && p->size(0) == num_experts && p->scalar_type() == at::kLong, "exl3_moe: bad pointer table");
    if (total_rows == 0) return;

    int device; cudaGetDevice(&device);
    int* scratch = DevCtx::instance().get_locks(device);       // unused by the ROCm kernels otherwise
    int* expert_start = scratch;                               // [num_experts + 1]
    int* row_expert = scratch + 1024 + 1;                      // [total_rows] (<= capacity <= MAX_TILES_C - 1025)
    TORCH_CHECK(total_rows + 1025 < MAX_TILES_C, "exl3_moe (ROCm): too many rows for scratch");

    const half* x = (const half*) hidden_state.data_ptr();
    float* out = (float*) output_state.data_ptr();
    half* tg = (half*) temp_state_g.data_ptr(); half* tu = (half*) temp_state_u.data_ptr();
    half* ig = (half*) temp_intermediate_g.data_ptr(); half* iu = (half*) temp_intermediate_u.data_ptr();
    const int64_t* ec = (const int64_t*) expert_count.data_ptr();
    const int64_t* ts = (const int64_t*) token_sorted.data_ptr();
    const half* ws = (const half*) weight_sorted.data_ptr();
    auto P16 = [] (const at::Tensor& t) { return (const uint16_t**) t.data_ptr(); };
    auto PH  = [] (const at::Tensor& t) { return (const half**) t.data_ptr(); };
    const int gated = act_function != MOE_ACT_RELU2_NOGATE;

    rows_kernel<<<1, 256, 0, stream>>>(ec, num_experts, total_rows, max_tokens, expert_start, row_expert);
    const int warps_h = total_rows * (hidden_dim / 128), warps_i = total_rows * (inter_dim / 128);
    gather_kernel<<<CEIL_DIVIDE(warps_h, 8), 256, 0, stream>>>(x, tg, tu, PH(gate_ptrs_suh), PH(up_ptrs_suh), ts, row_expert, hidden_dim, total_rows, gated);
    gemv_gu(dim3(num_experts, inter_dim / COLS, gated ? 2 : 1), stream, gated ? tg : tu, tu, gated ? ig : iu, iu,
         gated ? P16(gate_ptrs_trellis) : P16(up_ptrs_trellis), P16(up_ptrs_trellis), expert_start, ec, hidden_dim, inter_dim, max_tokens);
    guad_kernel<<<CEIL_DIVIDE(warps_i, 8), 256, 0, stream>>>(ig, iu, PH(gate_ptrs_svh), PH(up_ptrs_svh), PH(down_ptrs_suh), row_expert, inter_dim, total_rows, act_limit, act_function);
    gemv_d(dim3(num_experts, hidden_dim / COLS, 1), stream, ig, ig, tg, tg, P16(down_ptrs_trellis), P16(down_ptrs_trellis), expert_start, ec, inter_dim, hidden_dim, max_tokens);
    d_out_kernel<<<CEIL_DIVIDE(warps_h, 8), 256, 8 * 128 * sizeof(float), stream>>>(tg, out, PH(down_ptrs_svh), ts, ws, row_expert, hidden_dim, total_rows);
    cuda_check(cudaPeekAtLastError());
}
