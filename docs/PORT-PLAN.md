# EXL3 -> ROCm port (CDNA3 gfx942 / CDNA4 gfx950)

Goal: run DeepSeek-V4.1-Flash EXL3 3.5bpw (bot-lab-21) on ONE AMD MI355X (288 GB).
Sizing (measured 2026-09-11): non-Engram on-GPU = 257.0 GB = 239.4 GiB vs 268 GiB. Fits.
Engram (203 GB FP8) -> host/NVMe via vLLM-side staging (tonyd2wild recipe) or HLWQ Q4.

## Where the work actually is (after reading every kernel on the path)

The plugin's DeepSeek-V4.1 integration (vllm_exl3/deepseek_v41.py) replaces ONLY the routed
experts with EXL3 and delegates dense/shared/DSpark/Engram to vLLM's own deepseek_v41 model.
So the kernel port surface is the routed-expert path, and it splits in two:

  DECODE  (m == 1)  -> vllm_exl3_c.p2b_fused_moe   (csrc/p2b_moe.cu, run_gemv_tile)
  PREFILL (m  > 1)  -> exllamav3 LinearEXL3.forward -> reconstruct (+had) + hgemm (cuBLAS)
                       [fat experts > 256 rows -> exl3_fat_gemm, tensor-core: NOT ported, gated
                        by hasattr(native_c, "exl3_fat_gemm") so it falls back automatically]

KEY FINDING 1: the decode kernel uses mma.m16n8k16 with only row 0 of A populated (M == 1).
  The 16x8 tensor-core product is one dot product per output column. Replaced the mma with
  4 FMAs per lane + two xor-shuffles (lanes 4n..4n+3 share column n). Lane->(k,n) mapping
  derived from the PTX fragment spec, cross-checked against reconstruct_kernel's
  fragment->row-major swizzle AND the plugin's dequant_cpu; tools/mma_layout_check.cpp
  reproduces the mma result exactly (0 mismatches / 2000 trials). => NO MFMA WORK NEEDED
  for a functional port. fp32 accumulation, slightly more accurate than the fp16-acc original.

KEY FINDING 2: the trellis decode (exl3_dq.cuh, codebook.cuh) is pure integer arithmetic.
  Only two constructs were NVIDIA-specific: lop3.b32 0x6a == (a & b) ^ c, and __dp4a (byte sum).

KEY FINDING 3: exllamav3's ext.py uses a PRECOMPILED `exllamav3_ext` module if importable
  (is_precompiled_extension_available), so we ship a subset extension with the same name and
  never touch the 193-file CUDA tree. LinearEXL3 needs exactly: reconstruct,
  reconstruct_slice, reconstruct_had_slice, had_r_128, hgemm, BC_LinearEXL3.

## What was changed (rocm/ overlay, patched copies of upstream files)

  exllamav3_ext/hip_compat.cuh      NEW  EXL3_SHFL/_XOR/_DOWN (width 32), EXL3_LDCS, EXL3_DP4A,
                                         EXL3_LOP3_6A. On CUDA they expand to the originals.
  exllamav3_ext/ptx.cuh                  all inline PTX under #ifndef EXL3_ROCM; portable
                                         bfe64 / FSHF_IMM / BFE16_IMM / mul_lo|hi on ROCm
  quant/codebook.cuh                     6x lop3 -> macro, __dp4a -> EXL3_DP4A
  quant/hadamard_inner.cuh               5 shuffles -> width-32 macros (u64 / float / half2)
  quant/reconstruct.cu                   8 shfl_down -> macro; drop `register` (C++17)
  quant/exl3_gemv_kernel.cuh             helpers kept; mma_ab_h + mma kernel body CUDA-only
  compat.cuh                             tanh_opt: ROCm branch also on __HIP_PLATFORM_AMD__
  libtorch/linear_rocm.cpp          NEW  BC_LinearEXL3 = had_r_128 -> reconstruct -> hgemm -> had_r_128
  bindings_rocm.cpp                 NEW  subset exports + exl3_moe_max_concurrency stub (=1)
  vllm_exl3_csrc/p2b_moe.cu              run_gemv_tile: FMA tile under EXL3_ROCM, mma under #else
  vllm_exl3_csrc/p2b_batched.cu          same for p2b_run_gemv_tile_2; m>1 worklist CUDA-only;
                                         include trick replaced on ROCm
  vllm_exl3_csrc/bindings.cpp            exl3_gemv/exl3_gemm -> CPU-decoded reference on ROCm;
                                         no exl3_fat_gemm export; EXL3_ROCM_PORT attr
  setup_exllamav3_ext.py / setup_vllm_exl3_c.py / bootstrap_box.sh / tests/test_rocm_smoke.py

Everything else (cudaLaunchCooperativeKernel, cooperative_groups grid.sync, cublasGemmEx,
CUDAGuard, fp16 intrinsics) is left to torch's hipify; all were verified present in the ROCm
headers (amd_warp_functions.h, amd_hip_fp16.h, amd_device_functions.h in ref/hip/).

Wave64: kernels keep 32-lane virtual warps (warp = tid/32, lane = tid%32). HIP __shfl masks
the source lane by width and __shfl_xor/_down clamp inside the segment, so width 32 gives
CUDA semantics per half-wave. No __syncwarp anywhere. __launch_bounds__(512) = 8 waves.

## Runtime settings for the ROCm build
  EXL3_MOE_BACKEND=native     decode via p2b_fused_moe (exllamav3 exl3_moe is not exported)
  fat GEMM absent             prefill falls back to LinearEXL3 -> reconstruct+hgemm (SLOW, correct)
  Engram                      FP8 rows via vLLM's own path; plugin ngram_dequant not needed

## Phases
  0 (done, local)   overlay + shim + FMA tiles + oracle tests + CPU layout proof
  1a (DONE 2026-09-11, Framework 16 / 780M gfx1103 as gfx1102, rootful podman rocm/pytorch:latest)
                    ./rocm/framework.sh {up|build|test}. 22/22 oracle tests pass in wave32 AND with
                    -mwavefrontsize64 (EXL3_ROCM_EXTRA_FLAGS). p2b_fused_moe vs CPU decoder: cosine
                    1.000000, rel L2 8e-4 for K=2/3/4. Open: SIGSEGV in libhsa-runtime64 at process
                    teardown after cooperative launches (p2b path only); results correct; recheck on MI300X.
  1b (DONE 2026-09-11 18:17-18:22, MI300X VF devcloud atl1, docker rocm/pytorch:latest + rocm:latest 7.2)
                    22/22 oracle tests natively on gfx942 (with and without -mwavefrontsize64), p2b cosine
                    1.000000 rel L2 8e-4 K=2/3/4; plugin tests 42 pass / 4 non-bug fails (2x no vllm, GB10
                    latency target 687us>300us, 1-ulp fp16 tolerance). Same results rebuilt under ROCm 7.2
                    (torch 2.14+rocm7.2 needs -std=c++20). Exit-time SIGSEGV in libhsa-runtime64 at
                    "Deleting CG enabled hardware queue" reproduces on MI300X under BOTH 7.14 and 7.2, only
                    after our cooperative launches from a torch process; a trivial cooperative kernel exits 0.
  2 (DONE 2026-09-11 19:16, MI300X)  END-TO-END SERVING WORKS. vllm/vllm-openai-rocm:nightly (vLLM 0.28.1rc1.dev681,
                    torch 2.12, HIP 7.2) + exllamav3 python (NOCOMPILE) + vllm-exl3 plugin (VLLM_EXL3_NO_CUDA=1) +
                    our two extensions rebuilt in that image (22/22). `rocm/serve_glm.sh` serves
                    vcruz305/GLM-5.3-Flash-EXL3-K2 (320B MoE, 2.05 bpw, 91 GiB) with --quantization exl3,
                    VLLM_ROCM_USE_AITER=1 (required by the sparse indexer on ROCm), EXL3_FUSED_MOE=1
                    VLLM_EXL3_MOE_KERNEL=native, --enforce-eager. Correct answers (17*23=391, d/dx x^3, primes...).
                    12.2 tok/s single-stream eager = overhead-bound (MoE step ~690 us/layer vs ~10 us of bandwidth).
                    QUALITY GATE PASSED: native vs exact path on identical context (549 decode tokens):
                    |dlogprob| 0.012 mean / 0.25 max, KL(top5) 0.0018; divergences only at near-ties.
                    Speed ladder (1x MI300X, single stream): eager 12.2 | graphs 16.5 | MTP k=2 24 | MTP+graphs 22-23
                    (graphs need VLLM_USE_BREAKABLE_CUDAGRAPH=1). Exact path 4.0. Kernel is the next lever.
                    V4.1 EXL3 pack (minus engram shards) is on the node at /root/models; needs a 288 GB card.
  2b (measured) concurrency: aggregate flat ~24-29 tok/s at 1-8 streams (one p2b launch per row; plugin cap
                    8 rows -> collapse to reconstruct path beyond, raise with VLLM_EXL3_NATIVE_MOE_MAX_ROWS).
                    => next kernel work, in order: batched-rows tile (M<=16), plain launches instead of
                    cooperative (also fixes exit SIGSEGV + graph capture), pk_fma/MFMA.
  3 ($4.50/hr MI355X spot mem1)  V4.1: EXL3 3.5bpw on GPU (257 GB) + Engram via upstream
                    --engram-config.cpu_offload (pinned host, UVA); vLLM has models/deepseek_v4_1/amd/
  later             port exl3_fat_gemm (mma+ldmatrix+cp.async) to MFMA/rocWMMA for prefill speed

## Non-negotiable rule
  Never trust "it ran". Every kernel gets diffed against the CPU reference (dequant_cpu is the
  plugin's own host decoder; had/reconstruct vs fp32 Sylvester) before it is believed.

## Phase 3: batched MoE (`exl3_moe`) on ROCm (started 2026-09-12)

Problem: the plugin's native ROCm path decodes each expert's trellis once per row (one
`p2b_fused_moe` launch per row, cap `VLLM_EXL3_NATIVE_MOE_MAX_ROWS`), so throughput stays flat
with concurrency (MI300X, GLM-5.3-Flash K2: 23.8/19.8/26.4/28.8 tok/s aggregate at 1/2/4/8 streams).

Fix: implement exllamav3's `exl3_moe` contract (rows sorted by expert, one call per layer) in
`rocm/exllamav3_ext/quant/exl3_moe_rocm.cu`. The plugin's `exllamav3` MoE backend then batches with
no Python changes (`VLLM_EXL3_MOE_KERNEL=exllamav3`, or `auto` above the native row cap).

Design: six plain launches per call instead of one cooperative kernel (no grid barriers: graph
capturable, no CG hardware queue, so the ROCm exit-time SIGSEGV path is not exercised):
0. `rows_kernel`: expert_start prefix sums + row->expert map (scratch = DevCtx locks buffer).
1. `gather_kernel`: gather x[token] + input Hadamard (suh) into temp_state_g/u (flat rows).
2. `gemv_kernel<K,cb>`: batched GEMV, block = 16 k-split warps, grid (experts, N/32, g|u);
   trellis tile decoded once per MROWS=8 rows (FMA path, fp32 accumulate).
3. `guad_kernel`: output Hadamard g/u + activation + gate + input Hadamard d (in place).
4. `gemv_kernel` for down.
5. `d_out_kernel`: output Hadamard d, routing weight, fp32 atomic add into output_state.
Temp buffers are the plugin's [concurrency, 2048, dim] tensors used flat; `exl3_moe_max_concurrency`
returns 8 (16384 sorted rows >= 2048 tokens x 8 experts). Experts with more than 2048 rows are
skipped as upstream does (caller's fallback). K in {2,3,4}, mcg or mul1, gate/up/down share K.

Validation: `rocm/tests/test_exl3_moe_rocm.py` (CPU dequant_trellis reference at K=2/3/4, bsz up
to 37, top-k up to 8, act_limit; p2b single-row cross-check). Then MI300X `bench_conc.py` with
`EXL3_FUSED_MOE=1 VLLM_EXL3_MOE_KERNEL=exllamav3`.

Status 2026-09-12 01:50: validated on the Framework (gfx1102 via HSA override, torch 2.13+rocm7.14):
13/13 exl3_moe tests pass (cosine 1.000000, rel L2 8e-4, all K, batches to 37 rows, top-k 8, act_limit;
1.000000 vs p2b single row), smoke suite still 22/22. First-build bug: the scatter helper
`had_hf_r_128_d_inner` stages through `extern __shared__ temp_shared` (128 floats per warp); launching
without dynamic LDS silently added zeros (AMD drops out-of-range LDS writes, reads return 0). Fixed by
launching d_out_kernel with 8*128*4 bytes.

02:00: forced wave64 build also passes 35/35 (needs `-DEXL3_MOE_MROWS=4` on RDNA3: a 512-thread wave64
workgroup there only fits ~128 VGPRs; gfx942 has 2x the budget so MROWS=8 stays). Kernel resources
(gfx1102, -Rpass-analysis=kernel-resource-usage): wave32 138-141 VGPRs / occupancy 7, wave64 (MROWS 4)
106-110 / 4, no spills. Row loops guarded by nrows (a 1-row expert no longer pays 8 rows of FMAs).

iGPU timing per layer, GLM shape (H 4096, I 2048, K2, top-8), batched exl3_moe vs per-row p2b loop
(rocm/tools/bench_moe_kernel.py):
  E=32:  rows 1/2/4/8/16/32 -> 0.45x/0.69x/0.84x/1.35x/2.03x/2.63x
  E=8:   rows 1/2/4/8/16/32 -> 0.46x/1.05x/2.08x/3.52x/3.69x/3.65x (all experts active)
  E=128: rows 1/8/32 (8/53/107 active) -> 0.40x/0.69x/1.17x
Reading: scaling has the right shape, empty blocks for inactive experts cost <0.5 ms, but the batched
path costs ~0.6 ms per active expert vs ~0.28 ms for p2b (1 row): ~2x per-expert gap to close before
the MI300X run (rocprof per-kernel breakdown next). The plugin's `auto` backend keeps native p2b for
<= cap rows, so the crossover (~6-8 rows on the iGPU) is where batching starts paying.
Profile (torch.profiler, iGPU): 99% of batched GPU time is gemv_kernel; per active expert it is
~1.2x p2b's per-row decode (6.3 vs 5.3 ms profiled at 1 row); the rest of the wall-clock gap at small
batches is idle time between the 6 dependent launches, which HIP-graph capture removes in vLLM.
Sentinel expert (n_exp: fat / non-local routes, weight 0) now excluded explicitly in rows_kernel.
Plugin-level wiring test rocm/tests/test_plugin_moe_wiring.py drives vllm_exl3.exl3.apply_exl3_fused_moe
with VLLM_EXL3_MOE_KERNEL=exllamav3 (no vLLM needed): 8/8 pass incl. expert-parallel expert_map
(int64, pinned) and a 300-token batch (fat-expert path). Total on Framework: 43/43 wave32, 35/35 wave64.
Droplet-side chain ready: rocm/tools/batch_chain.sh (native cap32 vs exllamav3 vs auto, bench_conc
1/2/4/8, logprob fidelity vs native, gfx942 kernel microbench). MI300X run pending user decision.

### MI355X spot results (2026-09-12 03:10-03:25, droplet 599786077, gfx950, vllm nightly dev681, HIP 7.2)
All 43 oracle/plugin tests pass on gfx950 first try (native wave64, MROWS 8). GLM-5.3-Flash EXL3 K2,
eager, MTP k=2, 8 seqs, bench_conc 1/2/4/8 streams (aggregate tok/s):
  native per-row (cap 32):  28.6 / 29.8 / 38.3 / 37.9
  batched exllamav3:        32.2 / 50.8 / 73.1 / 112.3   (1.13x / 1.70x / 1.91x / 2.96x)
  auto (cap 32 = native):   29.0 / 31.4 / 38.3 / 39.0
gfx950 kernel microbench (E=288, top-8): crossover at 2 rows; 8 rows 1.52x, 16 2.73x, 32 3.99x, 64 6.73x.
Position-wise fidelity vs native dump: batched 73% top-1 / KL 0.026; native rerun 82% / KL 0.019 (same band).
Results: rocm/results/mi355x_599786077/. Next: DeepSeek V4.1-Flash on this card via the dedicated image
vllm/vllm-openai-rocm:deepseekv41-flash-0909 (Docker Hub 2026-09-11) + --engram-config.cpu_offload.

### DeepSeek V4.1-Flash on the MI355X: bring-up log (2026-09-12)
- Runtime: image vllm/vllm-openai-rocm:deepseekv41-flash-0909 (vllm 0.28.1rc1.dev398+g79a7108d9, HIP 7.2,
  aiter present). V4.1 lives in vllm/models/deepseek_v4_1/{common,amd,nvidia}; ROCm resolves to amd/vl_model.py.
  Our extensions + plugin build and pass 43/43 (plus 8 mul1/mixed-K cases) inside this image on gfx950.
- Pack bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard: 48 shards, 460 GB incl. 2x101.5 GB Engram shards;
  routed experts EXL3 (codebook mul1, per-layer bits (4,4)x19 (3,3)x18 (3,4)x3), everything else original
  MXFP8 block [32,32] ue8m0. Engram config lives under text_config (engram_layer_ids [1, 14]).
- Attempt 1: --engram-config '{"cpu_offload": true}' rejected: vllm/config/engram.py gates EngramConfig on
  current_platform.is_cuda(). The implementation is common (ParallelEngramEmbedding, pinned host + UVA) and
  Engram() defaults to cpu_offload=True when no EngramConfig is given; is_uva_available() is True on ROCm.
- Attempt 2: without --engram-config: GPU had only 40 GB free (leftover GLM server in exl3vllm). Killed it.
- Attempt 3: hipErrorOutOfMemory inside ParallelEngramEmbedding.__init__ (pinned host alloc, engram.py:667).
  Root cause: amdgpu GTT = 128868 MB (gttsize=-1 -> half of the 251 GB RAM) caps GPU-accessible pinned
  host memory; Engram needs 2 x 101.4 GB = 203 GB pinned. Fix: /etc/modprobe.d/amdgpu-gtt.conf
  (amdgpu gttsize=235000, ttm pages_limit=page_pool_size=60160000) + same on the kernel cmdline,
  update-initramfs, update-grub, reboot. Serve script now passes --swap-space 0.
- Attempts 4-6: `amdgpu.gttsize` alone -> "GTT 246 GB but TTM 135 GB, unusual" and the FIRST table failed;
  `ttm.pages_limit` on the cmdline ignored because this box runs the DKMS amdgpu stack whose TTM is the
  separate module `amdttm` (updates/dkms/amdttm.ko). Correct knob: `amdttm.pages_limit=60160000
  amdttm.page_pool_size=60160000` (kernel cmdline via GRUB + /etc/modprobe.d, update-initramfs, reboot).
  Verified after reboot: /sys/module/amdttm/parameters/pages_limit=60160000 (229 GB), GTT 235000M.
- Attempt 7 (03:07 UTC): still hipErrorOutOfMemory on the FIRST table with host RAM at 7 GB. HIP log
  (AMD_LOG_LEVEL=2): "Allocation failed : Pinned Memory, size :137438953472" = 128 GiB: torch's caching host
  allocator rounds pinned requests to the next power of two, so each 91.6 GiB table is a 128 GiB
  hipHostMalloc (2 x 128 GiB > 251 GB RAM anyway). Fresh-process probes: 100-110 GB pinned OK, exact table
  shape OK; pageable alloc + cudaHostRegister(Portable|Mapped) pins exactly and UVA views accept it.
- Attempt 9 (03:17 UTC): rocm/patches/engram_pinned_exact.py applied in the container (ParallelEngramEmbedding
  allocates with _pinned_exact when cpu_offload; original kept as engram.py.orig). Result: both tables pinned
  ("Engram table offloaded to pinned host memory: 384006168 rows x 256, 94.42 GiB per rank" x2). Then
  KeyError layers.0.attn.fused_wqa_wkv.weight_scale_inv: dense layers built unquantized because the plugin's
  V4.1 delegate keys off quantization_config.non_routed_quantization (deepseek_v4_fp8, block [32,32]) +
  scope + mtp_experts, while this pack (bot-lab-21, made for their cuda-exl3 stack) uses
  original_quantization_config (fp8) and expert_bits {layer: {gu, down}}.
- Attempt 10/11: local pack config.json translated (backup config.json.orig): non_routed_quantization,
  scope deepseek_v41_routed_experts, mtp_experts source (+start_layer 40), layer_bits {layer: gate/up K}.
  Mixed layers 20/22/24 (gu K3, down K4): rocm/patches/plugin_mixed_k.py makes the plugin pass K_down from
  the w2 trellis shape (both ROCm kernels accept per-matrix K). Pack tensor names/marker scalars (.mul1)
  match the plugin's loader.
- Attempts 11-12: model constructs, both tables pinned, 48 shards streamed, 231 GiB on the GPU, then the
  host runs out of RAM (engine anon RSS 254 GB = 189 GiB pinned + ~50-60 GB loader working set > 251 GB):
  OOM kill, then thrash with a 64 GB swapfile (needed the GLM pack deleted for disk). RAM-resident
  Engram is not viable on this 251 GB host. User: stop, develop the disk-resident path instead.
- Disk-resident Engram (rocm/patches/engram_disk.py, env VLLM_ENGRAM_DISK=1): EngramDiskTable reads rows
  by pread from the safetensors shard (offsets from the header; scales are `embed.scale` F8_E8M0 [N,8]),
  48-thread gather with dedupe, then the UNCHANGED Triton lookup kernel runs on the gathered slice with
  identity ids (rows outside the vocab window / padded heads stay zero). Standalone test
  rocm/tests/dsv41_engram_disk_test.py vs the original kernel on a device-resident window: bit-exact at
  T=1/8/256; cold NVMe timing 4.1 ms (24 rows), 9.5 ms (192), 97 ms (1536), 217 ms (12288 rows).
- Attempt 13 (04:08 UTC): VLLM_ENGRAM_DISK=1: host RAM fine (22 GB). Failed at the mixed-K layer:
  w2_trellis dest (144, 320, 48) != loaded (144, 320, 64): the plugin sizes the stacked down trellis with
  the layer's single K. Fix: rocm/patches/plugin_down_bits.py adds `layer_bits_down` to Exl3Config,
  down_bits_for_prefix(), Exl3MoEMethod(bits_down=...), w2_trellis sized with bits_down; pack config.json
  carries layer_bits_down from expert_bits[layer]["down"].
- Attempts 14-15 (04:12/04:17 UTC): disk Engram active (no pinned tables), yet host anonymous memory grew
  ~6.5 GB per body shard walked (228 GB, thousands of sub-GB allocations, engine CPU-bound) -> would OOM.
  Experiment: vLLM's safetensors iterator is zero-copy (6.5 GiB yielded, RssAnon flat). Root cause: H2D
  copies straight from the private file mapping make the ROCm runtime pin the pages with write intent,
  breaking copy-on-write; touched shard pages become private anonymous memory for the mapping's lifetime.
  (Also the true reason the pinned attempts died, on top of the 189 GiB tables.)
  Fix: rocm/patches/plugin_clone_on_load.py: `loaded_weight.detach().clone()` before the device copy
  (3 loader sites), so file pages stay clean page cache and the transient is freed per tensor.
- Attempt 16 (04:30 UTC): + clone-on-load: host RAM flat (21-32 GB used, rest page cache); "Loading weights
  took 393.01 seconds"; "Available KV cache memory: 37.15 GiB" (2,636,363 tokens); first forward failed in
  my lazy _disk_table(): get_current_vllm_config() outside the construction context. Fix: resolve
  model_config.model in __init__ (engram_disk.py updated).
- Attempt 17 (04:38 UTC): SERVING. "Application startup complete" 04:44 UTC. Weights 393 s, KV 37.15 GiB
  (2.6M tokens), both Engram tables "on disk" from shards 47/48, host RAM 26 GB used. First prompt coherent
  (Rayleigh scattering answer, reasoning stream in content since no reasoning parser). bench_tps single
  stream: TTFT 0.69 s, 7.5 tok/s decode (eager, no DSpark/MTP, no graphs, 1 seq). Config: rocm/serve_dsv41.sh
  + patch set (engram_pinned_exact, engram_disk, plugin_mixed_k, plugin_down_bits, plugin_clone_on_load)
  + translated pack config.json. Engram disk lookups do a host gather per forward: incompatible with HIP
  graph capture as written (v2: prefetch outside the graph).

Follow-ups after the V4.1 milestone:
- exl3_moe temp capacity: flat rows = concurrency (8) x 2048 = 16384 sorted rows; V4.1 profiling with
  4096 batched tokens x top-6 = 24576 rows tripped the guard. Options: raise exl3_moe_max_concurrency
  (16 -> ~1 GB temps at H=5120) or chunk expert ranges in the launcher. Sweep runs with 2048 batched tokens.
- Engram disk lookup v2: prefetch the row gather outside the graph (or async on a side stream) so HIP graph
  capture and DSpark/MTP speculative decoding can be enabled on top; hot-row cache (bot-lab-21 style).
- Loader: the clone-on-load copy costs ~1 extra memcpy of 257 GB (load 393 s); a pinned staging ring
  would be faster than per-tensor clones.

V4.1 concurrency on 1x MI355X (eager, no speculative decoding, Engram on NVMe, 8 seqs, 2048 batched tokens,
bench_conc 128 tokens/stream), batched exllamav3 backend: 1/2/4/8 streams -> 8.0 / 15.4 / 24.9 / 56.0 tok/s
aggregate (per-stream 8.0 / 7.7 / 6.2 / 7.0). Native per-row (cap 32) A/B: 8.2 / 15.4 / 29.8 / 57.8:
no difference up to 8 streams. On V4.1 the eager per-step time (~125 ms) is dominated by non-MoE work
(40 layers of sparse MLA/CED launches, host Engram gather ~8 ms) and 8 x 6 rows is a small MoE load.
Next: 8/16/32 streams with MAX_NUM_SEQS=32 for both backends to find where batching starts to pay.
- 05:28 UTC: native 32-seq sweep crashed at 16 streams: prefill of 16 requests pushed experts over the fat
  threshold; apply_exl3_batched_fat reconstructed w2 with the layer's gate/up K on mixed layers ("packed
  dimension 2 is incorrect size"). Fix: rocm/patches/plugin_fat_down_k.py (down K from the trellis words).
  Patch set is now 6: engram_pinned_exact, engram_disk, plugin_mixed_k, plugin_down_bits, plugin_clone_on_load,
  plugin_fat_down_k. 8-stream native (32-seq server): 54.4 tok/s aggregate.
- Native per-row, 32-seq server, 8/16/32 streams: 57.8 / 91.9 / 122.0 tok/s aggregate (per-stream
  7.2 / 5.7 / 3.8). Batched exllamav3, 32-seq server: 57.5 / 91.6 / 130.3 (per-stream 7.2 / 5.7 / 4.1):
  +7% at 32 streams, parity below. On V4.1 (eager, no spec decode) the step time is dominated by non-MoE
  work; the batched kernel's headroom shows only when rows per step are large. Graphs/DSpark are the next
  levers for this model, not the MoE kernel.
