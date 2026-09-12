#!/usr/bin/env bash
# EXL3 ROCm port: generic box driver (droplet or laptop). Runs everything inside the rocm/pytorch
# container. Parametrized by env:
#   EXL3_CTR          container runtime command      (default: docker;  laptop uses "sudo -n podman")
#   EXL3_ROCM_ARCH    kernel target                  (default: gfx942;  laptop: gfx1102)
#   EXL3_HSA_OVERRIDE HSA_OVERRIDE_GFX_VERSION value (default: none;    laptop: 11.0.2)
#   EXL3_ROCM_EXTRA_FLAGS  e.g. -mwavefrontsize64
#   ./rocm/box.sh up|build|test|plugin-tests|shell|down
set -euo pipefail
IMG=${EXL3_ROCM_IMAGE:-docker.io/rocm/pytorch:latest}
NAME=exl3rocm
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
P=${EXL3_CTR:-docker}
ARCH=${EXL3_ROCM_ARCH:-gfx942}
pexec() { $P exec -e EXL3_ROCM_ARCH="$ARCH" -e EXL3_ROCM_EXTRA_FLAGS="${EXL3_ROCM_EXTRA_FLAGS:-}" "$NAME" bash -lc "$*"; }
case "${1:-}" in
  up)
    if $P container inspect "$NAME" >/dev/null 2>&1; then $P start "$NAME" >/dev/null; else
      grp=""; for g in video render; do gid=$(getent group $g 2>/dev/null | cut -d: -f3); [ -n "$gid" ] && grp="$grp --group-add $gid"; done
      $P run -d --name "$NAME" --device /dev/kfd --device /dev/dri $grp \
        --ipc=host --security-opt seccomp=unconfined \
        ${EXL3_HSA_OVERRIDE:+-e HSA_OVERRIDE_GFX_VERSION=$EXL3_HSA_OVERRIDE} \
        -v "$ROOT":/work -w /work "$IMG" sleep infinity >/dev/null
    fi
    pexec 'python -c "import torch;print(\"torch\",torch.__version__,\"hip\",torch.version.hip,\"|\",torch.cuda.get_device_name(0),\"| archs\",torch.cuda.get_arch_list())"; rocminfo | grep -m1 -E "gfx[0-9]+"; hipcc --version | head -1'
    ;;
  build)
    pexec 'pip install -q pytest ninja 2>&1 | grep -v notice | tail -1;
      cd rocm && rm -rf build && find . \( -name "*_hip.*" -o -name "*.hip" \) -delete;
      python setup_exllamav3_ext.py build_ext --inplace > build_ext1.log 2>&1; echo "exllamav3_ext build exit: $?";
      grep -nE "error:|fatal error" build_ext1.log | head -20 | cut -c1-240;
      python setup_vllm_exl3_c.py build_ext --inplace > build_ext2.log 2>&1; echo "vllm_exl3_c build exit: $?";
      grep -nE "error:|fatal error" build_ext2.log | head -20 | cut -c1-240;
      ls -la *.so 2>/dev/null | cut -c25-;
      PYTHONPATH=/work/rocm python -c "import torch, exllamav3_ext as e, vllm_exl3_c as c; print(\"exllamav3_ext ROCm:\", e.EXL3_ROCM_PORT, \"| vllm_exl3_c ROCm:\", c.EXL3_ROCM_PORT)"'
    ;;
  test)  pexec 'cd /work && PYTHONPATH=/work/rocm python -m pytest -q -s rocm/tests/test_rocm_smoke.py rocm/tests/test_exl3_moe_rocm.py rocm/tests/test_plugin_moe_wiring.py 2>&1 | grep -E "passed|failed|cosine|Error|error" | tail -12; echo "pytest exit: ${PIPESTATUS[0]}"' ;;
  plugin-tests)
    pexec 'cd /work; pip install -q tokenizers numpy rich typing_extensions safetensors pillow pyyaml marisa_trie pydantic llguidance 2>&1 | grep -v notice | tail -1;
      EXLLAMA_NOCOMPILE= pip install -q -e upstream-exllamav3 --no-build-isolation --no-deps 2>&1 | grep -v notice | tail -1;
      VLLM_EXL3_NO_CUDA=1 pip install -q -e upstream-vllm-exl3 --no-build-isolation --no-deps 2>&1 | grep -v notice | tail -1;
      cd upstream-vllm-exl3 && PYTHONPATH=/work/rocm python -m pytest -q tests/test_native_p2b_moe.py tests/test_native_dequant.py tests/test_dequant_parity.py tests/test_native_p2b_batched.py tests/test_exl3_linear.py 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed|Cosine|latency" | tail -12' ;;
  shell) $P exec -it "$NAME" bash ;;
  down)  $P rm -f "$NAME" ;;
  *) echo "usage: $0 {up|build|test|plugin-tests|shell|down}"; exit 2 ;;
esac
