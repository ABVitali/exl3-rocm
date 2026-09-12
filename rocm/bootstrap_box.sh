#!/usr/bin/env bash
# Phase 1 bring-up on a 1x MI300X (gfx942) devcloud droplet. Run as root/marss on the box.
# Kernel work needs NO model weights: this builds the two extensions and runs the oracle tests.
set -euo pipefail
cd "$(dirname "$0")/.."     # repo root (~/exl3-rocm on the box)

# 1. ROCm PyTorch via the official container (host has ROCm 7.2 + docker on the devcloud image)
IMG=${EXL3_ROCM_IMAGE:-rocm/pytorch:latest}
docker pull "$IMG"
docker run --rm -it --name exl3rocm \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size 16g --security-opt seccomp=unconfined \
  -v "$PWD":/work -w /work "$IMG" bash -lc '
set -euxo pipefail
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
rocminfo | grep -m1 gfx
pip install -q pytest ninja

# 2. exllamav3 Python package WITHOUT building its CUDA extension
pip install -q tokenizers numpy rich typing_extensions safetensors pillow pyyaml marisa_trie pydantic llguidance
EXLLAMA_NOCOMPILE= pip install -q -e upstream-exllamav3 --no-build-isolation --no-deps

# 3. our two ROCm extensions (subset exllamav3_ext, then the plugin)
export EXL3_ROCM_ARCH=${EXL3_ROCM_ARCH:-gfx942}
( cd rocm && python setup_exllamav3_ext.py develop 2>&1 | tail -20 )
( cd rocm && python setup_vllm_exl3_c.py develop 2>&1 | tail -20 )
python -c "import exllamav3_ext as e, vllm_exl3_c as c; print(\"exllamav3_ext ROCm:\", e.EXL3_ROCM_PORT, \"| vllm_exl3_c ROCm:\", c.EXL3_ROCM_PORT)"

# 4. numerical oracles: nothing is believed until it matches the CPU references
python -m pytest -x -q rocm/tests/test_rocm_smoke.py -s
'
