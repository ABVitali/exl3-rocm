# Licensing

This repository is an overlay on three upstream projects. The licence of each part follows the code it
derives from; original work is MIT.

| path | derived from | licence |
|---|---|---|
| `rocm/exllamav3_ext/**` | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) (patched copies of its CUDA extension, plus new files `hip_compat.cuh`, `bindings_rocm.cpp`, `libtorch/linear_rocm.cpp`, `quant/exl3_moe_rocm.cu`) | MIT (upstream copyright turboderp) |
| `rocm/vllm_exl3_csrc/**`, `rocm/setup_vllm_exl3_c.py`, `rocm/patches/plugin_*.py` | [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) (patched copies of its native extension, and patches to its Python plugin) | AGPL-3.0, see `LICENSE.AGPL-3.0` |
| `rocm/patches/engram_*.py` | patches applied to [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) model code inside the `vllm/vllm-openai-rocm` V4.1 image | Apache-2.0 for the modified code; the patch scripts themselves MIT |
| everything else (`rocm/tests`, `rocm/tools`, `rocm/*.sh`, `rocm/setup_exllamav3_ext.py`, `tools/`, `docs/`) | original | MIT, see `LICENSE` |

The upstream trees are not vendored; `fetch-upstream.sh` clones them at the commits this overlay was
developed against. [Zeuss5/cuda-exl3](https://github.com/Zeuss5/cuda-exl3) (MIT) is fetched for reference
only; nothing here derives from it.
