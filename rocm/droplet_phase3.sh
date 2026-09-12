#!/usr/bin/env bash
# Phase 3 on a fresh 1x MI300X droplet (root, repo at /root/exl3-rocm): GLM download overlapped with
# pulls/builds, gfx942 oracle tests (incl. batched exl3_moe + plugin wiring), vLLM container with the
# plugin and our extensions, then rocm/tools/batch_chain.sh (native vs batched vs auto). Log markers "###".
set -uo pipefail; cd /root/exl3-rocm
log() { echo "### $(date +%H:%M:%S) $*"; }
GLM=vcruz305/GLM-5.3-Flash-EXL3-K2; MDIR=/models/GLM-5.3-Flash-EXL3-K2; mkdir -p /root/models
log "STEP start GLM download (python:3.12-slim) + vllm image pull in background"
docker pull -q python:3.12-slim >/dev/null 2>&1
docker rm -f dl-glm >/dev/null 2>&1
docker run -d --name dl-glm -v /root/models:/models -v /root/exl3-rocm:/work python:3.12-slim bash -c "pip install -q huggingface_hub hf_transfer && HF_HUB_ENABLE_HF_TRANSFER=1 python /work/rocm/tools/hf_dl.py $GLM $MDIR > /models/dl-glm.log 2>&1" >/dev/null && echo "dl-glm started"
(docker pull -q vllm/vllm-openai-rocm:nightly > /root/pull-vllm.log 2>&1; echo "pull-vllm exit $?" >> /root/pull-vllm.log) &
ARCH=${EXL3_ROCM_ARCH:-gfx942}
if [ "${SKIP_ROCM_PYTORCH:-0}" = 1 ]; then log "SKIP run_droplet.sh (rocm/pytorch torch is gfx942-only; arch=$ARCH), tests run in the vLLM image"
else
  log "STEP run_droplet.sh (rocm/pytorch: build + oracle tests, arch=$ARCH)"
  EXL3_ROCM_ARCH=$ARCH ./rocm/run_droplet.sh > /root/droplet.log 2>&1
  grep -E "passed|failed|build exit|exit code" /root/droplet.log | tail -8
fi
log "STEP exl3vllm container (plugin + extensions inside vllm/vllm-openai-rocm:nightly)"
wait; tail -1 /root/pull-vllm.log
docker rm -f exl3vllm >/dev/null 2>&1
docker run -d --name exl3vllm --entrypoint sleep --device /dev/kfd --device /dev/dri --ipc=host --security-opt seccomp=unconfined -v /root/exl3-rocm:/work -v /root/models:/models -w /work vllm/vllm-openai-rocm:nightly infinity >/dev/null && echo "exl3vllm up"
docker exec exl3vllm bash -lc '
python -c "import torch, vllm; print(\"torch\", torch.__version__, \"hip\", torch.version.hip, \"| vllm\", vllm.__version__, \"|\", torch.cuda.get_device_name(0), \"| archs\", torch.cuda.get_arch_list())" 2>&1 | tail -1; rocminfo 2>/dev/null | grep -m1 -oE "gfx[0-9a-f]+"
pip install -q tokenizers numpy rich typing_extensions safetensors pillow pyyaml marisa_trie pydantic llguidance pytest ninja 2>&1 | grep -viE "notice|already" | tail -2
EXLLAMA_NOCOMPILE= pip install -q -e upstream-exllamav3 --no-build-isolation --no-deps 2>&1 | grep -viE notice | tail -1
VLLM_EXL3_NO_CUDA=1 pip install -q -e upstream-vllm-exl3 --no-build-isolation --no-deps 2>&1 | grep -viE notice | tail -1
export EXL3_ROCM_ARCH='"$ARCH"'; cd rocm
python setup_exllamav3_ext.py build_ext --inplace > build_ext1.log 2>&1; echo "ext1 exit $?"; grep -m3 -E "error:|fatal" build_ext1.log
python setup_vllm_exl3_c.py build_ext --inplace > build_ext2.log 2>&1; echo "ext2 exit $?"; grep -m3 -E "error:|fatal" build_ext2.log
cd /work; PYTHONPATH=/work/rocm python -m pytest -q rocm/tests/test_exl3_moe_rocm.py rocm/tests/test_plugin_moe_wiring.py rocm/tests/test_rocm_smoke.py 2>&1 | tail -1'
log "STEP wait for GLM download"
while [ "$(docker inspect -f '{{.State.Running}}' dl-glm 2>/dev/null)" = true ]; do sleep 30; done
tail -2 /models/dl-glm.log; du -sh $MDIR 2>/dev/null; ls $MDIR | grep -c safetensors
log "STEP batch_chain.sh (native cap32 vs exllamav3 vs auto)"
docker exec exl3vllm bash -lc './rocm/tools/batch_chain.sh' 2>&1 | tee /root/batch_chain.log | grep -E "^###|tok/s|x$|cosine|KL|passed|FAILED|ENGINE|TIMEOUT" 
log "DONE"
