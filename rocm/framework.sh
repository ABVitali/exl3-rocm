#!/usr/bin/env bash
# EXL3 ROCm port: Framework 16 (Ryzen 7840HS, Radeon 780M iGPU = gfx1103) test box driver.
# Runs everything inside the rocm/pytorch container via rootful podman; no host packages touched.
# The 780M is not a stock ROCm target: HSA_OVERRIDE_GFX_VERSION=11.0.2 makes the runtime present it
# as gfx1102 and our kernels are built for gfx1102 to match. RDNA3 is wave32 by default; pass
# EXL3_ROCM_EXTRA_FLAGS="-mwavefrontsize64" to `build` to exercise the 64-lane layout.
#
#   ./rocm/framework.sh up      # create/start the container
#   ./rocm/framework.sh build   # build exllamav3_ext (subset) + vllm_exl3_c
#   ./rocm/framework.sh test    # oracle tests
#   ./rocm/framework.sh shell   # interactive
#   ./rocm/framework.sh down
set -euo pipefail
IMG=${EXL3_ROCM_IMAGE:-docker.io/rocm/pytorch:latest}
NAME=exl3rocm
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
P="sudo -n podman"

pexec() { $P exec -e EXL3_ROCM_ARCH="${EXL3_ROCM_ARCH:-gfx1102}" -e EXL3_ROCM_EXTRA_FLAGS="${EXL3_ROCM_EXTRA_FLAGS:-}" "$NAME" bash -lc "$*"; }

case "${1:-}" in
  up)
    if $P container exists "$NAME"; then $P start "$NAME" >/dev/null; else
      $P run -d --name "$NAME" --device /dev/kfd --device /dev/dri \
        --ipc=host --security-opt seccomp=unconfined \
        -e HSA_OVERRIDE_GFX_VERSION=11.0.2 -e PYTORCH_ROCM_ARCH=gfx1102 \
        -v "$ROOT":/work -w /work "$IMG" sleep infinity >/dev/null
    fi
    pexec 'python -c "import torch;print(\"torch\",torch.__version__,\"hip\",torch.version.hip,\"|\",torch.cuda.get_device_name(0),\"| archs\",torch.cuda.get_arch_list())"; rocminfo | grep -m1 -E "gfx[0-9]+"; hipcc --version | head -2'
    ;;
  build)
    pexec 'pip install -q pytest ninja 2>&1 | tail -1;
      cd rocm && rm -rf build && find . \( -name "*_hip.*" -o -name "*.hip" \) -delete;
      python setup_exllamav3_ext.py build_ext --inplace > build_ext1.log 2>&1; echo "exllamav3_ext build exit: $?";
      grep -nE "error:|Error:|fatal error" build_ext1.log | head -30 | cut -c1-240;
      python setup_vllm_exl3_c.py build_ext --inplace > build_ext2.log 2>&1; echo "vllm_exl3_c build exit: $?";
      grep -nE "error:|Error:|fatal error" build_ext2.log | head -30 | cut -c1-240;
      ls -la *.so 2>/dev/null;
      PYTHONPATH=/work/rocm python -c "import torch, exllamav3_ext as e, vllm_exl3_c as c; print(\"exllamav3_ext ROCm:\", e.EXL3_ROCM_PORT, \"| vllm_exl3_c ROCm:\", c.EXL3_ROCM_PORT)"'
    ;;
  test)  pexec 'PYTHONPATH=/work/rocm python -m pytest -x -q rocm/tests/test_rocm_smoke.py -s 2>&1 | tail -40' ;;
  shell) $P exec -it "$NAME" bash ;;
  down)  $P rm -f "$NAME" ;;
  *) echo "usage: $0 {up|build|test|shell|down}"; exit 2 ;;
esac
