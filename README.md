# exl3-rocm

Run EXL3-quantized MoE models (ExLlamaV3 trellis format) on AMD Instinct GPUs through vLLM.

EXL3 packs are the cheapest way to fit very large mixture-of-experts checkpoints on a single card
(3 to 4 bits per weight for the routed experts, everything else untouched), but the format only shipped
with CUDA kernels. This repository is a ROCm port of the inference side: the trellis decode, the
reconstruct and Hadamard kernels, the fused per-row MoE kernel of the `vllm-exl3` plugin, and a new
batched MoE kernel, validated on CDNA3 (MI300X, gfx942) and CDNA4 (MI355X, gfx950).

What it has been used for:

- GLM-5.3-Flash EXL3 K2 on one MI300X and one MI355X.
- DeepSeek V4.1-Flash EXL3 3.5 bpw (552B backbone, 384 routed experts per layer, Engram tables) on one
  MI355X, with the Engram tables served from NVMe instead of RAM.

## Results

GLM-5.3-Flash EXL3 K2 (`vcruz305/GLM-5.3-Flash-EXL3-K2`), vLLM, eager mode, MTP k=2, 8 sequences,
aggregate decode throughput in tokens/s at 1 / 2 / 4 / 8 concurrent streams (`rocm/tools/bench_conc.py`):

| card | MoE backend | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| MI300X | native per-row (cap 32) | 23.8 | 19.8 | 26.4 | 28.8 |
| MI355X | native per-row (cap 32) | 28.6 | 29.8 | 38.3 | 37.9 |
| MI355X | batched `exl3_moe` (this port) | 32.2 | 50.8 | 73.1 | 112.3 |

Fidelity of the batched kernel against the per-row path on the same prompts (common greedy prefix
analysis, `rocm/tools/logprob_compare.py`): mean |delta logprob| 0.024, KL(top-5) 0.0040, which is inside
the per-row path's own run-to-run variation (0.023 / 0.0049). Kernel oracle tests compare every kernel
against a CPU reference decoder (cosine 1.000000, relative L2 below 1e-3).

DeepSeek V4.1-Flash EXL3 3.5 bpw (`bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard`) on one MI355X
(288 GB): 257 GB of non-Engram weights on the card, both 94 GiB Engram tables read on demand from the
NVMe shards, 37 GiB left for KV. Eager mode, no speculative decoding: 8 tok/s single stream, 56 tok/s
aggregate at 8 streams, 130 tok/s at 32 streams. Weight load 200 to 400 s.

Kernel microbenchmark on gfx950 (`rocm/tools/bench_moe_kernel.py`, 288 experts, top-8): the batched
kernel overtakes the per-row loop from 2 rows per step and reaches 1.5x at 8 rows, 4.0x at 32, 6.7x at 64.

## How it works

The port lives in `rocm/` as an overlay on two upstream trees that are fetched, not vendored:

- `rocm/exllamav3_ext/`: a subset of ExLlamaV3's extension with the PTX-only parts (tensor-core `mma`,
  `cp.async`, `ldmatrix`, CUDA graphs) confined behind `#ifndef EXL3_ROCM`, warp shuffles mapped to HIP
  32-lane semantics (`hip_compat.cuh`), and the CUDA GEMV replaced by exact fp32 FMA on the same
  fragment layout. `quant/exl3_moe_rocm.cu` implements ExLlamaV3's batched `exl3_moe` contract as six
  plain launches per layer (row map, gather plus input Hadamard, batched gate/up GEMV, activation,
  batched down GEMV, weighted scatter-add): each trellis tile is decoded once per 8 activation rows,
  there are no grid barriers and no cooperative launch, and the vLLM plugin picks it up through its
  existing `exllamav3` MoE backend with no Python changes.
- `rocm/vllm_exl3_csrc/`: the `vllm-exl3` native extension (`p2b_fused_moe`, the per-row fused MoE
  kernel) on the same FMA path.
- `rocm/patches/`: small patches applied inside the serving container. `plugin_*.py` teach the plugin
  about packs whose gate/up and down projections use different K in some layers and about the ROCm
  loader behaviour described below. `engram_*.py` change vLLM's DeepSeek V4.1 Engram embedding so the
  tables are pinned at their exact size, or read from disk instead of RAM.

## Requirements

- ROCm 7.2 or newer on the host, a container with PyTorch for ROCm and hipcc (`rocm/pytorch:latest` for
  the kernel tests, `vllm/vllm-openai-rocm:nightly` for serving, `vllm/vllm-openai-rocm:deepseekv41-flash-0909`
  for DeepSeek V4.1).
- Build arch through `EXL3_ROCM_ARCH` (`gfx942`, `gfx950`; `gfx1102` with `HSA_OVERRIDE_GFX_VERSION=11.0.2`
  for RDNA3 laptops, where forced wave64 also needs `-DEXL3_MOE_MROWS=4`).

## Quick start

```bash
./fetch-upstream.sh                                  # exllamav3, vllm-exl3, cuda-exl3 at pinned commits

# kernel tests on a laptop iGPU (rootful podman, RDNA3) or any ROCm box (docker):
./rocm/framework.sh up && ./rocm/framework.sh build && ./rocm/framework.sh test
EXL3_CTR=docker ./rocm/box.sh up && ./rocm/box.sh build && ./rocm/box.sh test

# GLM-5.3-Flash on a CDNA droplet: pulls images, builds, runs the oracle tests, downloads the pack and
# benchmarks native vs batched vs auto MoE backends
./rocm/droplet_phase3.sh                             # then rocm/tools/batch_chain.sh inside the vLLM container

# serve
VLLM_ROCM_USE_AITER=1 EXL3_FUSED_MOE=1 VLLM_EXL3_MOE_KERNEL=exllamav3 ./rocm/serve_glm.sh
```

Inside the vLLM container the extensions are built in place (`python setup_*.py build_ext --inplace`) and
imported with `PYTHONPATH=/work/rocm` after `import torch`. Serve knobs: `VLLM_EXL3_MOE_KERNEL`
(`native` = per-row, `exllamav3` = batched, `auto` = per-row up to `VLLM_EXL3_NATIVE_MOE_MAX_ROWS` rows,
batched above), `VLLM_ROCM_USE_AITER=1` (needed by the sparse indexer), `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
for HIP graphs.

## DeepSeek V4.1-Flash on one MI355X

1. Image `vllm/vllm-openai-rocm:deepseekv41-flash-0909` (it registers `DeepseekV41ForCausalLM` and the
   `deepseek_v41` reasoning and tool-call parsers; upstream vLLM main does not carry the model).
2. Build the extensions with `EXL3_ROCM_ARCH=gfx950`, apply the patches in this order:
   `engram_pinned_exact.py`, `engram_disk.py`, `plugin_mixed_k.py`, `plugin_down_bits.py`,
   `plugin_clone_on_load.py`, `plugin_fat_down_k.py`.
3. `rocm/tools/translate_v41_pack_config.py MODEL_DIR` rewrites the pack's `config.json` to the plugin's
   metadata contract (`non_routed_quantization`, `scope`, `mtp_experts`, `layer_bits`, `layer_bits_down`).
4. `MODEL_DIR=... ./rocm/serve_dsv41.sh` (eager, Engram on disk by default; `HOST`, `REASONING_PARSER`,
   `TOOL_PARSER`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS` knobs for agent harnesses).

Things learned on the way, each of which cost an attempt:

- vLLM's `--engram-config` CPU offload is gated to CUDA, but the AMD model path offloads by default when no
  `EngramConfig` is passed.
- Pinned host memory for the GPU is capped at half of RAM by the DKMS driver's TTM (`amdttm`, not
  `ttm`): raise `amdttm.pages_limit` on the kernel command line if you want the tables in RAM.
- PyTorch's caching host allocator rounds pinned allocations to the next power of two, so a 91.6 GiB table
  becomes a 128 GiB request; `engram_pinned_exact.py` pins the exact size.
- A host-to-device copy straight from a safetensors private mapping makes the ROCm runtime pin the pages
  writable, which breaks copy-on-write and leaves every touched shard page as anonymous memory:
  `plugin_clone_on_load.py` clones each tensor first. Without it the loader needs more RAM than the model.
- With 251 GB of host RAM the tables do not fit in RAM next to the loader anyway; the disk-resident
  lookup (`engram_disk.py`) reads 24 rows of 256 bytes per token and layer through a thread pool,
  bit-identical to the pinned path, about 4 ms per token per layer cold.
- Killing a vLLM server on ROCm: the engine child is a `multiprocessing` `spawn_main` process; if it
  survives it keeps the whole card allocated.

## Layout

```
rocm/exllamav3_ext/    patched ExLlamaV3 extension subset + ROCm bindings + batched exl3_moe
rocm/vllm_exl3_csrc/   patched vllm-exl3 native extension (per-row fused MoE)
rocm/patches/          patches applied in the serving container (plugin, vLLM Engram)
rocm/tests/            kernel oracle tests, batched MoE tests, plugin wiring test, Engram disk test
rocm/tools/            benchmarks, logprob fidelity tools, download helper, run chains
rocm/*.sh              container drivers (framework.sh, box.sh), droplet chains, serve scripts
rocm/results/          logs and measurements from the MI300X and MI355X runs
docs/PORT-PLAN.md      the working log of the port, phase by phase
tools/                 CPU check of the mma fragment layout used by the FMA replacement
```

## Status and limitations

- Validated: kernel oracles on gfx1102 (wave32 and forced wave64), gfx942 and gfx950; end-to-end serving of
  GLM-5.3-Flash and DeepSeek V4.1-Flash.
- Eager mode only for V4.1 with Engram on disk: the lookup does a host gather inside the forward pass,
  which HIP graph capture cannot record. Prefetching it outside the graph is the next step, together with
  DSpark speculative decoding.
- The batched kernel's temp buffers cover 16384 sorted rows per call; raise `exl3_moe_max_concurrency`
  or chunk the launcher for larger prefill batches with high top-k.
- The fat-expert and non-batched fallbacks use ExLlamaV3's reconstruct path (dequantize plus hgemm).
- Not ported: the CUDA `exl3_gemm` tensor-core path, `exl3_fat_gemm`, ExLlamaV3's cooperative kernels,
  and the quantizer (packs must be produced on NVIDIA hardware).

## Licensing

See `LICENSES.md`: original work is MIT, files derived from `vllm-exl3` are AGPL-3.0, files derived
from ExLlamaV3 are MIT.
