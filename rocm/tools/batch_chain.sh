#!/usr/bin/env bash
# MI300X phase-3 measurement, run inside the vLLM container (cd /work). Same server flags for every
# backend: eager, MTP k=2, MAX_NUM_SEQS=8. Backends: native (per-row p2b, cap 32) = previous baseline,
# exllamav3 (batched exl3_moe for every batch), auto (native <= 8 rows, batched above).
set -uo pipefail; cd /work
OUT=/work/results; mkdir -p $OUT
log() { echo "### $(date +%H:%M:%S) $*"; }
wait_ready() { for i in $(seq 1 90); do grep -qE "Application startup complete" "$1" 2>/dev/null && return 0; grep -qE "Engine core initialization failed" "$1" 2>/dev/null && { log "ENGINE FAILED ($1)"; grep -nE "ERROR" "$1" | tail -3 | cut -c1-220; return 1; }; sleep 10; done; log "TIMEOUT $1"; return 1; }
restart() { pkill -f "vllm serv[e]" 2>/dev/null; sleep 5; pkill -9 -f "vllm serv[e]" 2>/dev/null; pkill -9 -f "spawn_mai[n]" 2>/dev/null; sleep 3; }
MTP='--speculative-config {"method":"mtp","num_speculative_tokens":2}'

log "STEP kernel microbench gfx942 (E=288, GLM shape)"
PYTHONPATH=/work/rocm python rocm/tools/bench_moe_kernel.py 288 1 2 4 8 16 32 64 2>&1 | tail -10 | tee $OUT/moe_kernel_bench.txt

for BE in native exllamav3 auto; do
  log "STEP serve backend=$BE"
  restart
  (VLLM_ROCM_USE_AITER=1 VLLM_EXL3_MOE_KERNEL=$BE VLLM_EXL3_NATIVE_MOE_MAX_ROWS=32 MAX_NUM_SEQS=8 EXTRA_ARGS="$MTP" ./rocm/serve_glm.sh > /work/vllm-serve-$BE.log 2>&1 &)
  wait_ready /work/vllm-serve-$BE.log || continue
  log "bench_conc backend=$BE"
  python rocm/tools/bench_conc.py --streams 1,2,4,8 --tokens 256 2>&1 | tee $OUT/conc_$BE.txt
  if [ "$BE" != native ]; then
    log "fidelity backend=$BE vs native dump"
    python rocm/tools/logprob_dump.py $OUT/lp_$BE.json >/dev/null 2>&1 && python rocm/tools/logprob_compare.py $OUT/lp_native.json $OUT/lp_$BE.json 2>&1 | tail -6 | tee $OUT/fidelity_$BE.txt
  else
    python rocm/tools/logprob_dump.py $OUT/lp_native.json >/dev/null 2>&1
  fi
done
log "DONE"
