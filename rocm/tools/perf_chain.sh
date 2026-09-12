#!/usr/bin/env bash
# Perf knobs on the native path: MTP k=2 (eager), then MTP + HIP graphs (VLLM_USE_BREAKABLE_CUDAGRAPH=1). Inside exl3vllm.
set -uo pipefail; cd /work
log() { echo "### $(date +%H:%M:%S) $*"; }
wait_ready() { for i in $(seq 1 60); do grep -qE "Application startup complete" "$1" 2>/dev/null && return 0; grep -qE "Engine core initialization failed" "$1" 2>/dev/null && { log "ENGINE FAILED ($1)"; grep -nE "EngineCore pid=[0-9]+\) ERROR" "$1" | sed -E "s/^[0-9]+:.*core.py:[0-9]+\] //" | grep -E "Error" | tail -2 | cut -c1-200; return 1; }; sleep 10; done; log "TIMEOUT $1"; return 1; }
restart() { pkill -f "vllm serve" 2>/dev/null; sleep 5; pkill -9 -f "vllm serve" 2>/dev/null; sleep 3; }
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":2}'
log "STEP native eager + MTP k=2 (token-counted bench)"
restart; (VLLM_ROCM_USE_AITER=1 EXTRA_ARGS="$MTP" ./rocm/serve_glm.sh > /work/vllm-serve-mtp2.log 2>&1 &)
wait_ready /work/vllm-serve-mtp2.log && { python rocm/tools/bench_tps.py --tokens 256; python rocm/tools/bench_tps.py --tokens 256; }
log "STEP native + HIP graphs + MTP k=2"
restart; (VLLM_ROCM_USE_AITER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1 ENFORCE_EAGER=0 EXTRA_ARGS="$MTP" ./rocm/serve_glm.sh > /work/vllm-serve-gm.log 2>&1 &)
if wait_ready /work/vllm-serve-gm.log; then python rocm/tools/bench_tps.py --tokens 256; python rocm/tools/bench_tps.py --tokens 256; log "leaving graphs+MTP server running"; else log "graphs+MTP failed; restarting eager+MTP"; restart; (VLLM_ROCM_USE_AITER=1 EXTRA_ARGS="$MTP" ./rocm/serve_glm.sh > /work/vllm-serve.log 2>&1 &); wait_ready /work/vllm-serve.log; fi
log "DONE"
