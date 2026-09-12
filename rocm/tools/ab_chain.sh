#!/usr/bin/env bash
# Inside exl3vllm. Chain: exact-path A/B (quality gate), then HIP-graph mode, then MTP. Logs with ### markers.
set -uo pipefail; cd /work
log() { echo "### $(date +%H:%M:%S) $*"; }
wait_ready() { for i in $(seq 1 60); do grep -qE "Application startup complete" "$1" 2>/dev/null && return 0; grep -qE "Engine core initialization failed|RuntimeError: Engine" "$1" 2>/dev/null && { log "ENGINE FAILED ($1)"; grep -nE "EngineCore pid=[0-9]+\) ERROR" "$1" | sed -E "s/^[0-9]+:.*core.py:[0-9]+\] //" | grep -vE "^\s*\^|^  File|^    return|^\s*$" | tail -4 | cut -c1-200; return 1; }; sleep 10; done; log "TIMEOUT waiting for $1"; return 1; }
restart() { pkill -f "vllm serve" 2>/dev/null; sleep 5; pkill -9 -f "vllm serve" 2>/dev/null; sleep 3; }

log "STEP exact path (VLLM_EXL3_MOE_KERNEL=exllamav3 EXL3_FUSED_MOE=0, eager)"
restart; (VLLM_ROCM_USE_AITER=1 VLLM_EXL3_MOE_KERNEL=exllamav3 EXL3_FUSED_MOE=0 ./rocm/serve_glm.sh > /work/vllm-serve-exact.log 2>&1 &)
if wait_ready /work/vllm-serve-exact.log; then
  grep -iE "fall|python loop|native" /work/vllm-serve-exact.log | grep -viE "deprecat|native_|IrOp|route" | sed -E "s/^\((EngineCore|APIServer) pid=[0-9]+\) //" | sort -u | head -4 | cut -c1-160
  python rocm/tools/logprob_dump.py /work/rocm/results/lp_exact.json 2>&1 | tail -1
  python rocm/tools/bench_tps.py --tokens 128
  log "COMPARE native vs exact"; python rocm/tools/logprob_compare.py /work/rocm/results/lp_native.json /work/rocm/results/lp_exact.json
fi

log "STEP native + HIP graphs (ENFORCE_EAGER=0)"
restart; (VLLM_ROCM_USE_AITER=1 ENFORCE_EAGER=0 ./rocm/serve_glm.sh > /work/vllm-serve-graphs.log 2>&1 &)
if wait_ready /work/vllm-serve-graphs.log; then python rocm/tools/bench_tps.py --tokens 256; python rocm/tools/bench_tps.py --tokens 256; fi

log "STEP native + MTP k=2 (eager)"
restart; (VLLM_ROCM_USE_AITER=1 EXTRA_ARGS='--speculative-config {"method":"mtp","num_speculative_tokens":2}' ./rocm/serve_glm.sh > /work/vllm-serve-mtp.log 2>&1 &)
if wait_ready /work/vllm-serve-mtp.log; then python rocm/tools/bench_tps.py --tokens 256; grep -oE "Draft acceptance rate: [0-9.]+|acceptance[^,]*" /work/vllm-serve-mtp.log | tail -2; fi
log "DONE"
