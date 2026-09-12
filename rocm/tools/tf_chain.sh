#!/usr/bin/env bash
# Teacher-forced A/B (identical context for both paths): score 5 fixed texts on the native server, then on the
# exact reconstruct-path server, and compare per-token logprobs. Inside exl3vllm; run after ab_chain.sh.
set -uo pipefail; cd /work
log() { echo "### $(date +%H:%M:%S) $*"; }
wait_ready() { for i in $(seq 1 60); do grep -qE "Application startup complete" "$1" 2>/dev/null && return 0; grep -qE "Engine core initialization failed" "$1" 2>/dev/null && { log "ENGINE FAILED ($1)"; return 1; }; sleep 10; done; log "TIMEOUT $1"; return 1; }
restart() { pkill -f "vllm serve" 2>/dev/null; sleep 5; pkill -9 -f "vllm serve" 2>/dev/null; sleep 3; }
log "STEP native (eager) teacher-forced scoring"
restart; (VLLM_ROCM_USE_AITER=1 ./rocm/serve_glm.sh > /work/vllm-serve-tf-native.log 2>&1 &)
wait_ready /work/vllm-serve-tf-native.log && python rocm/tools/logprob_dump.py --echo /work/rocm/results/lp_native_tf.json 2>&1 | tail -6
log "STEP exact path teacher-forced scoring"
restart; (VLLM_ROCM_USE_AITER=1 VLLM_EXL3_MOE_KERNEL=exllamav3 EXL3_FUSED_MOE=0 ./rocm/serve_glm.sh > /work/vllm-serve-tf-exact.log 2>&1 &)
wait_ready /work/vllm-serve-tf-exact.log && python rocm/tools/logprob_dump.py --echo /work/rocm/results/lp_exact_tf.json 2>&1 | tail -6
log "COMPARE teacher-forced native vs exact"; python rocm/tools/logprob_compare.py /work/rocm/results/lp_native_tf.json /work/rocm/results/lp_exact_tf.json
log "STEP native + HIP graphs retry (VLLM_USE_BREAKABLE_CUDAGRAPH=1 ENFORCE_EAGER=0)"
restart; (VLLM_ROCM_USE_AITER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1 ENFORCE_EAGER=0 ./rocm/serve_glm.sh > /work/vllm-serve-graphs2.log 2>&1 &)
if wait_ready /work/vllm-serve-graphs2.log; then python rocm/tools/bench_tps.py --tokens 256; python rocm/tools/bench_tps.py --tokens 256; grep -iE "cudagraph|capture" /work/vllm-serve-graphs2.log | grep -viE deprecat | tail -3 | sed -E "s/^\((EngineCore|APIServer) pid=[0-9]+\) //" | cut -c1-160; else grep -nE "EngineCore pid=[0-9]+\) ERROR" /work/vllm-serve-graphs2.log | sed -E "s/^[0-9]+:.*core.py:[0-9]+\] //" | grep -E "Error|error" | tail -3 | cut -c1-200; fi
log "STEP leave native server running"; restart; (VLLM_ROCM_USE_AITER=1 ./rocm/serve_glm.sh > /work/vllm-serve.log 2>&1 &); wait_ready /work/vllm-serve.log
log "DONE"
