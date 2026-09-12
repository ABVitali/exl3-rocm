#!/usr/bin/env bash
# DeepSeek V4.1-Flash EXL3 3.5bpw (bot-lab-21 Pollard pack) on one MI355X, inside the exl3dsv41 container
# (vllm/vllm-openai-rocm:deepseekv41-flash-0909). Routed experts = EXL3 via the plugin; everything else
# = original MXFP8; Engram tables offloaded to pinned host RAM (--engram-config cpu_offload).
set -uo pipefail
MODEL_DIR=${MODEL_DIR:-/models/DeepSeek-V4.1-Flash-EXL3-3.5bpw}
export PYTHONPATH=/work/rocm EXL3_FUSED_MOE=1 VLLM_EXL3_MOE_KERNEL=${VLLM_EXL3_MOE_KERNEL:-exllamav3}
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1} VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# Engram tables read on demand from the NVMe shards (rocm/patches/engram_disk.py) instead of 189 GiB pinned RAM
export VLLM_ENGRAM_DISK=${VLLM_ENGRAM_DISK:-1} VLLM_ENGRAM_DISK_THREADS=${VLLM_ENGRAM_DISK_THREADS:-48}
ARGS=(vllm serve "$MODEL_DIR" --served-model-name DeepSeek-V4.1-Flash-EXL3 --host "${HOST:-127.0.0.1}" --port 8889
  --quantization exl3 --tokenizer-mode deepseek_v41 --tensor-parallel-size 1
  --max-model-len "${MAX_MODEL_LEN:-8192}" --max-num-seqs "${MAX_NUM_SEQS:-1}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.96}" --no-enable-prefix-caching --skip-mm-profiling --limit-mm-per-prompt '{"image":0}'
)
# --engram-config is CUDA-gated in the image's validator; the AMD model path offloads Engram to pinned host
# memory by default (common/engram.py: cpu_offload=True when no EngramConfig is given).
[ -n "${ENGRAM_CONFIG:-}" ] && ARGS+=(--engram-config "$ENGRAM_CONFIG")
[ "${ENFORCE_EAGER:-1}" = 1 ] && ARGS+=(--enforce-eager)
# Harness use (dsh): reasoning_content + tool calls need the V4.1 parsers; HOST=0.0.0.0 exposes the port on the
# container's bridge IP only (reach it through an SSH tunnel, the server has no auth).
[ -n "${REASONING_PARSER:-}" ] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
[ -n "${TOOL_PARSER:-}" ] && ARGS+=(--tool-call-parser "$TOOL_PARSER" --enable-auto-tool-choice)
echo "+ ${ARGS[*]} ${EXTRA_ARGS:-}"
exec "${ARGS[@]}" ${EXTRA_ARGS:-}
