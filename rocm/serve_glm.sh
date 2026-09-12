#!/usr/bin/env bash
# Phase 2 end-to-end: GLM-5.3-Flash EXL3 K2 through vllm-exl3 + our ROCm kernels. Run INSIDE exl3vllm.
# Derived from vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe scripts/serve_one_spark.sh minus NVIDIA-only flags.
set -uo pipefail
MODEL_DIR=${MODEL_DIR:-/models/GLM-5.3-Flash-EXL3-K2}
export PYTHONPATH=/work/rocm
export EXL3_FUSED_MOE=${EXL3_FUSED_MOE:-1} VLLM_EXL3_MOE_KERNEL=${VLLM_EXL3_MOE_KERNEL:-native}
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}
ARGS=(serve "$MODEL_DIR" --served-model-name GLM-5.3-Flash-EXL3 --host 127.0.0.1 --port 8888
  --tensor-parallel-size 1 --quantization exl3 --load-format auto
  --max-model-len ${MAX_MODEL_LEN:-8192} --max-num-seqs ${MAX_NUM_SEQS:-1} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-2048}
  --kv-cache-dtype ${KV_DTYPE:-auto} --skip-mm-profiling --limit-mm-per-prompt '{"image":4,"video":1}'
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
  --chat-template "$MODEL_DIR/chat_template.jinja" --no-enable-prefix-caching
  --gpu-memory-utilization ${GPU_MEM_UTIL:-0.87})
[ "${ENFORCE_EAGER:-1}" = 1 ] && ARGS+=(--enforce-eager)
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)
echo "### $(date +%H:%M:%S) vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
